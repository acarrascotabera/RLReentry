"""Shared training/eval helpers — Windows-safe vec-env construction + eval-to-CSV.

SubprocVecEnv on Windows uses 'spawn', so worker processes must be able to
`import reentry_rl`. `ensure_child_importable` puts the repo root on PYTHONPATH
so that works even when the script is run as a file without `pip install -e .`.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from ..envs.stage1_bank import Stage1BankEnv
from ..envs.stage2_bank_aoa import Stage2BankAoaEnv
from ..physics import constants as C

# Stage registry (extended in later milestones).
ENV_CLASSES = {"stage1": Stage1BankEnv, "stage2": Stage2BankAoaEnv}
J_REF = {"stage1": C.J_REF_BANK_ONLY, "stage2": C.J_REF_BANK_AOA}

CSV_FIELDS = ["t_s", "alt_km", "lon_deg", "lat_deg", "V_mps", "gamma_deg", "psi_deg",
              "sigma_deg", "alpha_deg", "Qdot_Wm2", "qbar_Pa", "n_g", "J_heat"]


def ensure_child_importable(repo_root):
    """Make `reentry_rl` importable in spawned SubprocVecEnv workers."""
    rr = str(repo_root)
    cur = os.environ.get("PYTHONPATH", "")
    if rr not in cur.split(os.pathsep):
        os.environ["PYTHONPATH"] = rr + (os.pathsep + cur if cur else "")


def build_vecenv(env_cls, n_envs, weights, n_substeps, seed, subproc, env_kwargs=None):
    """VecNormalize-wrapped (obs only) vectorized env. Picklable via make_vec_env.
    `env_kwargs` are extra EntryEnvBase options (obs_version, exact_terminal, ...)."""
    vec_cls = SubprocVecEnv if subproc else DummyVecEnv
    venv = make_vec_env(
        env_cls, n_envs=n_envs, seed=seed, vec_env_cls=vec_cls,
        env_kwargs=dict(weights=weights, n_substeps=n_substeps, **(env_kwargs or {})),
    )
    return VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)


def evaluate_to_csv(model, obs_rms, env_cls, weights, n_substeps, j_ref, outdir,
                    stage="stage1", seed=987654, env_kwargs=None):
    """Deterministic rollout -> eval_trajectory.csv + eval_summary.txt. Returns (summary, last_info)."""
    venv = build_vecenv(env_cls, 1, weights, n_substeps, seed, subproc=False, env_kwargs=env_kwargs)
    venv.obs_rms = obs_rms          # reuse the trained observation statistics
    venv.training = False
    venv.norm_reward = False

    obs = venv.reset()
    rows, total_r, last = [], 0.0, {}
    for _ in range(6000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = venv.step(action)
        info = infos[0]
        rows.append({k: info[k] for k in CSV_FIELDS})
        total_r += float(reward[0])
        last = info
        if done[0]:
            break

    pd.DataFrame(rows).to_csv(Path(outdir) / "eval_trajectory.csv", index=False)
    summary = _format_summary(last, total_r, j_ref, stage)
    (Path(outdir) / "eval_summary.txt").write_text(summary + "\n")
    return summary, last


def _format_summary(info, total_r, j_ref, stage):
    jh = info.get("J_heat", float("nan"))
    return "\n".join([
        f"{stage} eval (deterministic)",
        f"  outcome        : {info.get('outcome')}",
        f"  is_success     : {info.get('is_success')}",
        f"  t_final [s]    : {info.get('t_s', float('nan')):.1f}",
        f"  alt [km]       : {info.get('alt_km', float('nan')):.3f}",
        f"  landing err km : {info.get('d_km', float('nan')):.2f}",
        f"  fpa err [deg]  : {info.get('fpa_err_deg', float('nan')):.2f}",
        f"  psi err [deg]  : {info.get('psi_err_deg', float('nan')):.2f}",
        f"  J_heat [J/m^2] : {jh:.4e}",
        f"  J_ref  [J/m^2] : {j_ref:.4e}",
        f"  J_heat / J_ref : {jh / j_ref:.4f}",
        f"  episode reward : {total_r:+.2f}",
    ])




class ValidationCallback(BaseCallback):
    """Model selection on a fixed scenario set (replaces EvalCallback, which
    kept the best of single nominal rollouts by episode reward).

    Every `eval_freq` calls the current policy is rolled out deterministically
    on every scenario of the validation set (in-process, with the training
    env's observation statistics). The feasibility-gated selection score
    (evaluate_policy.selection_score) decides `best/best_model.zip`; any
    evaluation scoring below `save_below` is also kept in snapshots/ (the best
    policies were repeatedly found between checkpoints). The history is
    appended to eval/validation_history.csv."""

    def __init__(self, stage, scenarios, eval_freq, outdir, weights, env_kwargs=None,
                 score_mode="position", save_below=None, set_name="validation", verbose=0):
        super().__init__(verbose)
        self.stage = stage
        self.scenarios = list(scenarios)
        self.eval_freq = max(int(eval_freq), 1)
        self.outdir = Path(outdir)
        self.weights = weights
        self.env_kwargs = dict(env_kwargs or {})
        self.score_mode = score_mode
        self.save_below = save_below
        self.set_name = set_name
        self.best = float("inf")
        self._runner = None
        self.history_path = self.outdir / "eval" / "validation_history.csv"
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(self):
        from ..validation.evaluate_policy import PolicyRunner, selection_score
        if self._runner is None:
            self._runner = PolicyRunner(self.model, self.training_env.obs_rms, self.stage,
                                        self.weights, self.env_kwargs)
        r = self._runner
        rms = self.training_env.obs_rms               # stats keep moving during training
        r.mean = np.asarray(rms.mean, dtype=np.float64)
        r.std = np.sqrt(np.asarray(rms.var, dtype=np.float64) + 1e-8)
        df = r.run_many(self.scenarios)
        return df, selection_score(df, self.score_mode)

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True
        df, score = self.evaluate()
        step = int(self.num_timesteps)
        rec = {"step": step, "score": score, "n": len(df),
               "reached_rate": float((df["outcome"] == "reached").mean()),
               "feasible_rate": float(df["feasible"].mean()),
               "success_pos_1km": float(df["success_pos_1km"].mean()),
               "success_vista_mc": float(df["success_vista_mc"].mean()),
               "success_rel_1pct": float(df["success_rel_1pct"].mean()),
               "success_rel_0p1pct": float(df["success_rel_0p1pct"].mean()),
               "d_km_median": float(df["d_km"].median()),
               "abs_dfpa_median": float(df["dfpa_deg"].abs().median()),
               "abs_dpsi_median": float(df["dpsi_deg"].abs().median()),
               "max_ratio_max": float(df["max_ratio"].max()),
               "alpha_std_mean": float(df["alpha_std_deg"].mean()),
               "nominal_d_km": float(df["d_km"].iloc[0]),
               "nominal_dfpa_deg": float(df["dfpa_deg"].iloc[0]),
               "nominal_dpsi_deg": float(df["dpsi_deg"].iloc[0])}
        for k, v in rec.items():
            if k != "step":
                self.logger.record(f"val/{k}", v)
        pd.DataFrame([rec]).to_csv(self.history_path, mode="a", index=False,
                                   header=not self.history_path.exists())
        if score < self.best:
            self.best = score
            (self.outdir / "best").mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.outdir / "best" / "best_model.zip"))
            self.training_env.save(str(self.outdir / "best" / "best_vecnormalize.pkl"))
            (self.outdir / "best" / "best.json").write_text(
                pd.Series({**rec, "set": self.set_name, "score_mode": self.score_mode}).to_json(indent=1))
        if self.save_below is not None and score < self.save_below:
            snap = self.outdir / "snapshots"
            snap.mkdir(parents=True, exist_ok=True)
            tag = f"step{step // 1000:06d}k_score{score:.3f}"
            self.model.save(str(snap / f"{tag}.zip"))
            self.training_env.save(str(snap / f"{tag}_vecnormalize.pkl"))
        if self.verbose:
            print(f"[val] step={step:,} score={score:.3f} best={self.best:.3f} "
                  f"d_med={rec['d_km_median']:.2f} km |dfpa|={rec['abs_dfpa_median']:.2f} "
                  f"|dpsi|={rec['abs_dpsi_median']:.2f} feasible={rec['feasible_rate']:.2f}")
        return True


def load_validation_set(name_or_path):
    """-> (set name, [Scenario]) from results/eval_sets/ (or a path)."""
    from ..envs.scenario import load_set
    from ..validation.evaluate_policy import resolve_set
    name, scen, _labels = load_set(resolve_set(name_or_path))
    return name, scen


def set_action_std(model, std):
    """Set the policy's Gaussian std: one value for all actions, or one per
    action dimension (e.g. a larger AoA-rate std to re-open AoA exploration)."""
    import math
    import torch
    vals = [float(v) for v in (std if isinstance(std, (list, tuple)) else [std])]
    n = model.policy.log_std.shape[0]
    if len(vals) == 1:
        vals = vals * n
    if len(vals) != n:
        raise ValueError(f"action std needs 1 or {n} values, got {len(vals)}")
    with torch.no_grad():
        model.policy.log_std.copy_(torch.tensor([math.log(v) for v in vals]))
    return vals
