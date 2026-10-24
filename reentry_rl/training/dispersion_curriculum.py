"""Automatic dispersion curriculum (ADR-style; OpenAI et al. 2019, "Solving Rubik's Cube
with a Robot Hand", automatic domain randomization).

The training sampler draws the dispersion scale itself per episode (ScenarioSampler mode
"adr"): with probability p_edge at the current boundary s_k, otherwise uniform in
[0, s_k], so near-nominal cases stay in the training mix while the boundary expands. The
boundary advances through `levels` when the policy is good enough AT the boundary,
measured on the training episodes drawn there (stochastic policy, no extra rollouts):
over the last `window` boundary episodes, at least `reach_min` reach the handover and the
median landing error is at most `d_med_km`, after at least `min_steps` at the level. It
advances unconditionally after `max_steps` (logged as forced): at the larger scales part
of the tail is physically infeasible (e.g. -2 sigma lift or +2 sigma density), and the
gate must not stall the run on it.

New scales are pushed with env_method("set_dispersion_scale") and apply from each env's
next reset. History: <run>/eval/dispersion.csv (one row per rollout); state for --resume:
<run>/dispersion_state.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


class DispersionCurriculumCallback(BaseCallback):
    def __init__(self, levels, outdir, min_steps=1_500_000, max_steps=4_000_000, window=300,
                 reach_min=0.97, d_med_km=5.0, verbose=1):
        super().__init__(verbose)
        self.levels = [float(x) for x in levels]
        self.outdir = Path(outdir)
        self.min_steps, self.max_steps = int(min_steps), int(max_steps)
        self.window = int(window)
        self.reach_min, self.d_med_km = float(reach_min), float(d_med_km)
        self.level = 0
        self.level_start = 0
        self._eps = []                       # (d_km, reached, peak_ratio) at the boundary
        self._n_all = 0
        self.history_path = self.outdir / "eval" / "dispersion.csv"
        self.state_path = self.outdir / "dispersion_state.json"

    def scale(self):
        return self.levels[self.level]

    def describe(self):
        return {"levels": self.levels, "min_steps": self.min_steps, "max_steps": self.max_steps,
                "window": self.window, "reach_min": self.reach_min, "d_med_km": self.d_med_km}

    # -- state -----------------------------------------------------------------
    def save(self, step):
        self.state_path.write_text(json.dumps({"level": self.level, "level_start": self.level_start,
                                               "scale": self.scale(), "step": int(step),
                                               **self.describe()}, indent=1))

    def restore(self):
        if self.state_path.exists():
            st = json.loads(self.state_path.read_text())
            self.level, self.level_start = int(st["level"]), int(st["level_start"])
        return self.scale()

    # -- SB3 hooks ---------------------------------------------------------------
    def _on_step(self):
        s = self.scale()
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done or "outcome" not in info:
                continue
            self._n_all += 1
            ds = info.get("disp_scale")
            if ds is not None and abs(float(ds) - s) < 1e-9:
                self._eps.append((float(info.get("d_km", np.nan)), info["outcome"] == "reached",
                                  float(info.get("peak_ratio", np.nan))))
        return True

    def _on_rollout_end(self):
        step = int(self.num_timesteps)
        at = step - self.level_start
        last = self._eps[-self.window:]
        n = len(last)
        if n:
            d = np.array([e[0] for e in last])
            reach = float(np.mean([e[1] for e in last]))
            d_med = float(np.nanmedian(d))
            d_p90 = float(np.nanpercentile(d, 90))
            pk = np.array([e[2] for e in last])
            excess = float(np.nanmean(np.maximum(pk - 1.0, 0.0)))
        else:
            reach = d_med = d_p90 = excess = float("nan")
        rec = {"step": step, "level": self.level, "scale": self.scale(), "steps_at_level": at,
               "boundary_episodes": n, "reach_rate": reach, "d_km_median": d_med, "d_km_p90": d_p90,
               "peak_excess_mean": excess, "advanced": "", "forced": False}
        self.logger.record("dispersion/scale", self.scale())
        if n:
            self.logger.record("dispersion/d_km_median", d_med)
            self.logger.record("dispersion/reach_rate", reach)
        if self.level < len(self.levels) - 1 and at >= self.min_steps:
            passed = n >= self.window and reach >= self.reach_min and d_med <= self.d_med_km
            forced = (not passed) and at >= self.max_steps
            if passed or forced:
                old = self.scale()
                self.level += 1
                self.level_start = step
                self._eps = []
                self.training_env.env_method("set_dispersion_scale", self.scale())
                rec["advanced"], rec["forced"] = f"{old:g}->{self.scale():g}", bool(forced)
                if self.verbose:
                    print(f"[dispersion] step {step:,}: scale {old:g} -> {self.scale():g} "
                          f"({'forced' if forced else 'passed'}: reach {reach:.2f}, "
                          f"median d {d_med:.2f} km, p90 {d_p90:.1f} km over {n} boundary episodes)")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([rec]).to_csv(self.history_path, mode="a", index=False,
                                   header=not self.history_path.exists())
        self.save(step)
