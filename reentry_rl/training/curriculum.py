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

Preset "v20r" prepends a position re-acquisition level (angle channels off,
advance once d <= 5 km): under the corrected dynamics the warm start is ~300 km
from the target, where the log position potential is nearly flat (~0.15/km)
while the level-0 heading bonus pays ~4.3/deg, so v20b traded position for
heading (308 -> 349 km while dpsi 77 -> 65 deg). Level keys starting with "_"
are curriculum metadata (pass criteria), not reward weights.

Preset "v23" (point B) switches the angle channels on early: the position-only
level hands over once >= 50 % of the validation cases land within 30 km, and
every angle level carries its own position criterion (30 -> 2 km) so the
curriculum can tighten the angles while the landing error is still tens of km
(the v20 levels required d <= 2 km at every level). Built for a dispersed
validation set (S2_val_x0.1), hence the 50 % pass fraction (key "_pass_frac").

Preset "v24" loosens the angle pass criteria (v23 sat on its first angle level for
18M steps at heading 52-61 deg / FPA 15-21 deg against 45 / 10 deg): each level
carries explicit pass thresholds ("_fpa_pass_deg", "_psi_pass_deg") instead of
half the bonus ramp, starting at 20 deg / 65 deg and tightening level by level.

Preset "v24b" widens the FPA bonus ramps of v24 to ~2.5x the pass threshold (levels 2-6:
10/5/2/0.5/0.1 -> 30/15/8/2.5/0.6 deg). The concave bonus w*(1 - (dgamma/ramp)^2)^+ is
zero beyond the ramp, so with v24's ramp equal to (or below) the pass threshold a policy
that arrives 17-29 deg off (v24b at level 2: ramp 10 deg) gets no FPA bonus gradient at
all and trades FPA for position (steeper terminal dive). Heading ramps unchanged.

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
_V23 = [  # (w_ang_mult, fpa_ramp, psi_ramp, d_pass_km)
    (0.25, 20.0, 90.0, 30.0), (0.50, 10.0, 45.0, 25.0), (1.00, 5.0, 20.0, 20.0),
    (1.00, 2.0, 8.0, 10.0), (1.00, 0.5, 2.0, 5.0), (1.00, 0.1, 0.5, 2.0)]
PRESETS["v23"] = dict(
    base=dict(PRESETS["v20"]["base"]),
    levels=[{"w_ang_mult": 0.0, "w_succ_fpa": 0.0, "w_succ_psi": 0.0,
             "_d_pass_km": 30.0, "_check_angles": False, "_pass_frac": 0.5}]
           + [{"w_ang_mult": c, "fpa_ramp_deg": f, "psi_ramp_deg": q, "w_succ_fpa": 250.0,
               "w_succ_psi": 250.0, "_d_pass_km": d, "_pass_frac": 0.5} for c, f, q, d in _V23])
_V24 = [  # (w_ang_mult, fpa_ramp, psi_ramp, d_pass_km, fpa_pass_deg, psi_pass_deg)
    (0.25, 20.0, 90.0, 30.0, 20.0, 65.0), (0.50, 10.0, 45.0, 25.0, 12.0, 45.0),
    (1.00, 5.0, 20.0, 20.0, 6.0, 25.0), (1.00, 2.0, 8.0, 10.0, 3.0, 10.0),
    (1.00, 0.5, 2.0, 5.0, 1.0, 3.0), (1.00, 0.1, 0.5, 2.0, 0.25, 1.0)]
PRESETS["v24"] = dict(
    base=dict(PRESETS["v20"]["base"]),
    levels=[{"w_ang_mult": 0.0, "w_succ_fpa": 0.0, "w_succ_psi": 0.0,
             "_d_pass_km": 30.0, "_check_angles": False, "_pass_frac": 0.5}]
           + [{"w_ang_mult": c, "fpa_ramp_deg": f, "psi_ramp_deg": q, "w_succ_fpa": 250.0,
               "w_succ_psi": 250.0, "_d_pass_km": d, "_fpa_pass_deg": fp, "_psi_pass_deg": qp,
               "_pass_frac": 0.5} for c, f, q, d, fp, qp in _V24])
_FPA_RAMP_V24B = [20.0, 30.0, 15.0, 8.0, 2.5, 0.6]       # angle levels 1-6
PRESETS["v24b"] = dict(
    base=dict(PRESETS["v24"]["base"]),
    levels=[dict(PRESETS["v24"]["levels"][0])]
           + [{**lv, "fpa_ramp_deg": r} for lv, r in zip(PRESETS["v24"]["levels"][1:], _FPA_RAMP_V24B)])
PRESETS["v20r"] = dict(
    base=dict(PRESETS["v20"]["base"]),
    levels=[{"w_ang_mult": 0.0, "w_succ_fpa": 0.0, "w_succ_psi": 0.0,
             "_d_pass_km": 5.0, "_check_angles": False}]
           + [{**lv, "w_succ_fpa": 250.0, "w_succ_psi": 250.0} for lv in PRESETS["v20"]["levels"]])


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
        """Reward weights of a level (metadata keys starting with '_' removed)."""
        lv = self.level if level is None else level
        return {k: v for k, v in {**self.base, **self.levels[lv]}.items() if not k.startswith("_")}

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
        lv = self.levels[self.level]
        p = {**self.base, **lv}
        ok = ((df["outcome"] == "reached") & (df["max_ratio"] <= 1.02)
              & (df["d_km"] <= lv.get("_d_pass_km", self.d_pass_km)))
        if lv.get("_check_angles", True):
            ok &= ((df["dfpa_deg"].abs() <= lv.get("_fpa_pass_deg", p["fpa_ramp_deg"] / 2.0))
                   & (df["dpsi_deg"].abs() <= lv.get("_psi_pass_deg", p["psi_ramp_deg"] / 2.0)))
        return float(ok.mean())

    def on_validation(self, callback, df, rec):
        """ValidationCallback listener: maybe advance, then push weights."""
        step = int(callback.num_timesteps)
        frac = self.passed(df)
        need = self.levels[self.level].get("_pass_frac", self.pass_frac)
        self.streak = self.streak + 1 if frac >= need else 0
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
