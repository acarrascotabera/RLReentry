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


def build_vecenv(env_cls, n_envs, weights, n_substeps, seed, subproc):
    """VecNormalize-wrapped (obs only) vectorized env. Picklable via make_vec_env."""
    vec_cls = SubprocVecEnv if subproc else DummyVecEnv
    venv = make_vec_env(
        env_cls, n_envs=n_envs, seed=seed, vec_env_cls=vec_cls,
        env_kwargs=dict(weights=weights, n_substeps=n_substeps),
    )
    return VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)


def evaluate_to_csv(model, obs_rms, env_cls, weights, n_substeps, j_ref, outdir,
                    stage="stage1", seed=987654):
    """Deterministic rollout -> eval_trajectory.csv + eval_summary.txt. Returns (summary, last_info)."""
    venv = build_vecenv(env_cls, 1, weights, n_substeps, seed, subproc=False)
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


def rollout_metrics(model, obs_rms, env_cls, weights, n_substeps, j_ref, seed=987654):
    """Deterministic rollout -> metrics dict: terminal accuracy (d_km, psi_err,
    fpa_err), heat ratio, peak constraint ratios, and bank smoothness (sign
    reversals). Shared by HPO and final evaluation. Failed/timeout episodes get
    large default terminal errors so they score poorly."""
    venv = build_vecenv(env_cls, 1, weights, n_substeps, seed, subproc=False)
    venv.obs_rms = obs_rms
    venv.training = False
    venv.norm_reward = False
    obs = venv.reset()
    sigma, last, done = [], {}, [False]
    mn = mq = mQ = 0.0
    for _ in range(6000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, done, infos = venv.step(action)
        info = infos[0]
        last = info
        sigma.append(info["sigma_deg"])
        mn = max(mn, info["n_g"] / C.N_MAX)
        mq = max(mq, info["qbar_Pa"] / C.QBAR_MAX)
        mQ = max(mQ, info["Qdot_Wm2"] / C.QDOT_MAX)
        if done[0]:
            break
    venv.close()
    sig = np.asarray(sigma)
    rev = int((np.diff(np.sign(sig)) != 0).sum()) if sig.size > 1 else 0
    jh = float(last.get("J_heat", float("nan")))
    return dict(d_km=float(last.get("d_km", 1e4)),
                psi_err=float(last.get("psi_err_deg", 180.0)),
                fpa_err=float(last.get("fpa_err_deg", 90.0)),
                J_ratio=(jh / j_ref if j_ref else float("nan")),
                max_n=mn, max_qbar=mq, max_Qdot=mQ, max_ratio=max(mn, mq, mQ),
                reversals=rev, is_success=bool(last.get("is_success", False)),
                outcome=str(last.get("outcome", "")))


class StageMetricsCallback(BaseCallback):
    """Every `eval_freq` calls, run one deterministic episode and log the domain
    metrics (landing error, heat ratio, FPA/heading error, peak path-constraint
    ratios, success) to TensorBoard, so a long run is interpretable beyond
    ep_rew_mean. Cheap: one ~1500-step rollout per eval."""

    def __init__(self, env_cls, weights, n_substeps, j_ref, eval_freq, train_venv,
                 seed=24680, verbose=0):
        super().__init__(verbose)
        self.env_cls = env_cls
        self.weights = weights
        self.n_substeps = n_substeps
        self.j_ref = j_ref
        self.eval_freq = max(int(eval_freq), 1)
        self.train_venv = train_venv
        self.seed = seed

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True
        venv = build_vecenv(self.env_cls, 1, self.weights, self.n_substeps, self.seed, subproc=False)
        venv.obs_rms = self.train_venv.obs_rms
        venv.training = False
        venv.norm_reward = False
        obs = venv.reset()
        last, done = {}, [False]
        mQ = mn = mq = 0.0
        for _ in range(6000):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _r, done, infos = venv.step(action)
            info = infos[0]
            last = info
            mQ = max(mQ, info["Qdot_Wm2"] / C.QDOT_MAX)
            mn = max(mn, info["n_g"] / C.N_MAX)
            mq = max(mq, info["qbar_Pa"] / C.QBAR_MAX)
            if done[0]:
                break
        venv.close()
        rec = self.logger.record
        rec("stage/d_km", float(last.get("d_km", float("nan"))))
        rec("stage/J_ratio", float(last.get("J_heat", float("nan"))) / self.j_ref)
        rec("stage/fpa_err_deg", float(last.get("fpa_err_deg", float("nan"))))
        rec("stage/psi_err_deg", float(last.get("psi_err_deg", float("nan"))))
        rec("stage/max_Qdot_ratio", mQ)
        rec("stage/max_n_ratio", mn)
        rec("stage/max_qbar_ratio", mq)
        rec("stage/is_success", float(bool(last.get("is_success", False))))
        return True
