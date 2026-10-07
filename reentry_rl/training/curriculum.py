"""Terminal-tolerance curriculum (epsilon-annealing, Zavoli & Federici 2021).

The warm-start policy hits the target position but arrives at gamma ~ -25 deg
and psi ~ 20 deg; asking for the 0.1 % handover tolerances at once would
replace a 500-unit landing bonus by nothing. Instead the angle channels of
the reward (rewards.py, tier 1b) start loose and light and are tightened level
by level:

  level  w_ang_mult  fpa_ramp  psi_ramp     advance when, on the validation set,
    0      0.25        20 deg    90 deg      >= `pass_frac` of the cases reach the
    1      0.50        10        45          handover feasibly with d <= d_pass_km,
    2      1.00         5        20          |dfpa| <= fpa_ramp/2, |dpsi| <= psi_ramp/2,
    3      1.00         2         8          on `patience` consecutive evaluations
    4      1.00         0.5       2          and after >= `min_steps` at the level
    5      1.00         0.1       0.5

Weights are pushed to every training env with env_method("set_reward_params")
and take effect at each env's next reset. The level history is written to
<run>/eval/curriculum.csv and the current level to <run>/curriculum_state.json
(restored on --resume).
"""
import json
from pathlib import Path

import pandas as pd

PRESETS = {
    "v20": dict(
        base={"w_succ_fpa": 250.0, "w_succ_psi": 250.0},
        levels=[
            {"w_ang_mult": 0.25, "fpa_ramp_deg": 20.0, "psi_ramp_deg": 90.0},
            {"w_ang_mult": 0.50, "fpa_ramp_deg": 10.0, "psi_ramp_deg": 45.0},
            {"w_ang_mult": 1.00, "fpa_ramp_deg": 5.0, "psi_ramp_deg": 20.0},
            {"w_ang_mult": 1.00, "fpa_ramp_deg": 2.0, "psi_ramp_deg": 8.0},
            {"w_ang_mult": 1.00, "fpa_ramp_deg": 0.5, "psi_ramp_deg": 2.0},
            {"w_ang_mult": 1.00, "fpa_ramp_deg": 0.1, "psi_ramp_deg": 0.5},
        ]),
}


class TerminalCurriculum:
    def __init__(self, preset="v20", outdir=None, pass_frac=0.75, patience=2,
                 min_steps=300_000, d_pass_km=2.0, level=0):
        spec = PRESETS[preset]
        self.preset = preset
        self.base = dict(spec["base"])
        self.levels = [dict(l) for l in spec["levels"]]
        self.pass_frac = float(pass_frac)
        self.patience = int(patience)
        self.min_steps = int(min_steps)
        self.d_pass_km = float(d_pass_km)
        self.level = int(level)
        self.level_start = 0
        self.streak = 0
        self.outdir = Path(outdir) if outdir else None

    # -- state ---------------------------------------------------------------
    def params(self, level=None):
        lv = self.level if level is None else level
        return {**self.base, **self.levels[lv]}

    def apply_to(self, weights):
        for k, v in self.params().items():
            setattr(weights, k, v)
        return weights

    def _state_path(self):
        return self.outdir / "curriculum_state.json" if self.outdir else None

    def save(self, step):
        if self.outdir is None:
            return
        self._state_path().write_text(json.dumps(
            {"preset": self.preset, "level": self.level, "level_start": self.level_start,
             "step": step, "params": self.params()}, indent=1))

    def restore(self):
        p = self._state_path()
        if p is not None and p.exists():
            st = json.loads(p.read_text())
            self.level, self.level_start = int(st["level"]), int(st["level_start"])
        return self.level

    # -- decision --------------------------------------------------------------
    def passed(self, df):
        p = self.levels[self.level]
        ok = ((df["outcome"] == "reached") & (df["max_ratio"] <= 1.02)
              & (df["d_km"] <= self.d_pass_km)
              & (df["dfpa_deg"].abs() <= p["fpa_ramp_deg"] / 2.0)
              & (df["dpsi_deg"].abs() <= p["psi_ramp_deg"] / 2.0))
        return float(ok.mean())

    def on_validation(self, callback, df, rec):
        """ValidationCallback listener: maybe advance, then push weights."""
        step = int(callback.num_timesteps)
        frac = self.passed(df)
        self.streak = self.streak + 1 if frac >= self.pass_frac else 0
        advanced = False
        if (self.streak >= self.patience and step - self.level_start >= self.min_steps
                and self.level < len(self.levels) - 1):
            self.level += 1
            self.level_start = step
            self.streak = 0
            advanced = True
            callback.training_env.env_method("set_reward_params", **self.params())
            if callback._runner is not None:
                callback._runner.env.set_reward_params(**self.params())
            self.apply_to(callback.weights)
            print(f"[curriculum] step {step:,}: -> level {self.level} {self.params()}")
        callback.logger.record("curriculum/level", self.level)
        callback.logger.record("curriculum/pass_frac", frac)
        if self.outdir is not None:
            path = self.outdir / "eval" / "curriculum.csv"
            pd.DataFrame([{"step": step, "level": self.level, "pass_frac": frac,
                           "streak": self.streak, "advanced": advanced}]).to_csv(
                path, mode="a", index=False, header=not path.exists())
            self.save(step)
