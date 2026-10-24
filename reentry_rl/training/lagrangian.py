"""PID-Lagrangian path constraints (Stooke, Achiam & Abbeel 2020).

The path constraints become episodic constraints of the training distribution,

    J_k = E_episodes[ max(0, peak_k / limit_k - 1) ] <= budget_k,   k in {Qdot, qbar, n},

and the reward carries one Lagrange multiplier per constraint on the episode's
peak excess (rewards.py: -lam_k * max(0, peak_k - 1) on every ending; this
replaces the fixed w_peak). After every PPO rollout the multipliers are
updated by a PID controller on the error e_k = J_k - budget_k measured on the
training episodes that ended in that rollout:

    I_k   <- max(0, I_k + e_k)
    lam_k  = clip(lam0 + Kp e_k + Ki I_k + Kd max(0, e_k - e_k_prev), 0, lam_max)

The integral term is plain dual ascent; the proportional and derivative terms
damp the oscillation of pure Lagrangian PPO. Anti-windup: I_k is clamped to
(lam_max - lam0) / Ki, the most the cap can use (in v23 the load-factor integral
had wound up to 9.0 against 1.8 needed for the cap, which would have kept lam_n
pinned for ~1,800 rollouts after the constraint was met). lam_max bounds the return scale
(v20d: a 1000-weight peak charge on 12x violations broke the critic). rate_max bounds
the INCREASE of each multiplier per rollout (v24: at the end of the warm-up the cold-start
policy violated the heat rate by 73 %, Kp*e alone took lam_Qdot from 200 to ~1,700 in one
rollout and to the 4,000 cap in three — a -2,900 step in the return), with conditional
integration (I is frozen while the limiter is active, so the ramp does not wind up).
lam_max may be a dict per constraint (v25: under 0.5x VISTA dispersions the heat-rate
excess is partly unavoidable for any guidance, so lam_Qdot sat at the 4,000 cap, while the
load-factor excess is a choice of the terminal pull-up — at the shared cap it was cheaper
than the terminal-angle rewards and the nominal flight pulled 1.10x the limit). New
multipliers are pushed with env_method("set_reward_params") and take effect at
each env's next reset. History: <run>/eval/lagrange.csv; state for --resume:
<run>/lagrange_state.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

CONSTRAINTS = ("Qdot", "qbar", "n")


class PIDLagrangianCallback(BaseCallback):
    def __init__(self, outdir, budget=0.005, kp=2000.0, ki=1000.0, kd=0.0, lam_init=200.0,
                 lam_max=2000.0, min_episodes=4, weights=None, val_callback=None, start_step=0,
                 rate_max=None, verbose=1):
        super().__init__(verbose)
        self.outdir = Path(outdir)
        self.budget = float(budget)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.lam0 = float(lam_init)
        caps = lam_max if isinstance(lam_max, dict) else {}
        default = float(lam_max) if not isinstance(lam_max, dict) else float(caps.get("default", 2000.0))
        self.lam_max = {k: float(caps.get(k, default)) for k in CONSTRAINTS}
        self.min_episodes = int(min_episodes)
        # warm-up: multipliers held at lam_init until this many steps (a cold-start policy
        # violates every constraint by ~2x; at the cap the penalties would swamp the return)
        self.start_step = int(start_step)
        self.rate_max = None if rate_max is None else float(rate_max)
        self.weights = weights
        self.val_callback = val_callback
        self.lam = {k: self.lam0 for k in CONSTRAINTS}
        self.I = {k: 0.0 for k in CONSTRAINTS}
        self.e_prev = {k: 0.0 for k in CONSTRAINTS}
        self._excess = {k: [] for k in CONSTRAINTS}
        self.history_path = self.outdir / "eval" / "lagrange.csv"
        self.state_path = self.outdir / "lagrange_state.json"

    def _i_max(self, k):
        return (self.lam_max[k] - self.lam0) / self.ki if self.ki > 0 else float("inf")

    # -- state -----------------------------------------------------------------
    def params(self):
        return {f"lam_{k}": float(v) for k, v in self.lam.items()}

    def save(self):
        self.state_path.write_text(json.dumps({"lam": self.lam, "I": self.I, "e_prev": self.e_prev,
                                               "budget": self.budget, "kp": self.kp, "ki": self.ki,
                                               "kd": self.kd, "lam_init": self.lam0,
                                               "lam_max": self.lam_max, "rate_max": self.rate_max},
                                              indent=1))

    def restore(self, path=None):
        """Load multipliers and integrals from this run's state (resume) or from another
        run's lagrange_state.json (warm start: path)."""
        src = Path(path) if path is not None else self.state_path
        if src.exists():
            st = json.loads(src.read_text())
            self.lam, self.I, self.e_prev = st["lam"], st["I"], st["e_prev"]
            self.I = {k: min(v, self._i_max(k)) for k, v in self.I.items()}   # anti-windup
        return self.lam

    def apply_to(self, weights):
        for k, v in self.params().items():
            setattr(weights, k, v)
        return weights

    # -- SB3 hooks ---------------------------------------------------------------
    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "peak_Qdot" in info:
                for k in CONSTRAINTS:
                    # heat rate relative to the scenario's attainable floor when the env
                    # reports one (qdot_floor), as in the reward
                    floor = max(1.0, float(info.get("qdot_lb", 0.0))) if k == "Qdot" else 1.0
                    self._excess[k].append(max(0.0, float(info[f"peak_{k}"]) - floor))
        return True

    def _on_rollout_end(self):
        n_ep = len(self._excess["Qdot"])
        if n_ep < self.min_episodes:
            return
        rec = {"step": int(self.num_timesteps), "episodes": n_ep}
        warm = self.num_timesteps < self.start_step
        for k in CONSTRAINTS:
            x = np.asarray(self._excess[k])
            J = float(x.mean())
            e = J - self.budget
            if not warm:
                I_new = min(max(0.0, self.I[k] + e), self._i_max(k))
                d = max(0.0, e - self.e_prev[k])
                lam = float(np.clip(self.lam0 + self.kp * e + self.ki * I_new + self.kd * d,
                                    0.0, self.lam_max[k]))
                if self.rate_max is not None and lam > self.lam[k] + self.rate_max:
                    lam = self.lam[k] + self.rate_max
                    I_new = min(I_new, self.I[k])      # conditional integration (no ramp windup)
                self.I[k], self.lam[k] = I_new, lam
                self.e_prev[k] = e
            rec[f"J_{k}"] = J
            rec[f"viol_rate_{k}"] = float((x > 0.0).mean())
            rec[f"lam_{k}"] = self.lam[k]
            self.logger.record(f"lagrange/J_{k}", J)
            self.logger.record(f"lagrange/lam_{k}", self.lam[k])
            self.logger.record(f"lagrange/viol_rate_{k}", rec[f"viol_rate_{k}"])
            self._excess[k] = []
        params = self.params()
        self.training_env.env_method("set_reward_params", **params)
        if self.weights is not None:
            self.apply_to(self.weights)
        if self.val_callback is not None and self.val_callback._runner is not None:
            self.val_callback._runner.env.set_reward_params(**params)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([rec]).to_csv(self.history_path, mode="a", index=False,
                                   header=not self.history_path.exists())
        self.save()
        if self.verbose:
            print("[lagrange] step {:,}: ".format(rec["step"]) + "  ".join(
                f"{k}: J={rec[f'J_{k}']:.4f} viol={rec[f'viol_rate_{k}']:.2f} lam={self.lam[k]:.0f}"
                for k in CONSTRAINTS))
