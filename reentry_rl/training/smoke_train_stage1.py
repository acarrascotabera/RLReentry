"""Stage-1 PPO smoke-train.

Purpose: verify the full pipeline end-to-end (vectorized env -> VecNormalize ->
PPO -> save model + normalizer -> deterministic eval -> trajectory CSV). It is a
SMOKE test, not a solve: the default budget is small, so do not expect the policy
to hit the footprint yet. Real training needs millions of steps + Optuna HPO
(see DESIGN.md §7, §10).

Run (from E:\\reentry_RL):
    python -m reentry_rl.training.smoke_train_stage1 --timesteps 50000
    python -m reentry_rl.training.smoke_train_stage1 --timesteps 2000000 --n-envs 8 --subproc

Artifacts land in results/stage1_smoke_<timestamp>/:
    model.zip, vecnormalize.pkl, config.json, eval_trajectory.csv, eval_summary.txt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Runnable both as a module (python -m reentry_rl.training.smoke_train_stage1)
# and as a plain file (python reentry_rl/training/smoke_train_stage1.py / IDE Run).
if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from reentry_rl.envs.stage1_bank import Stage1BankEnv
from reentry_rl.envs.rewards import RewardWeights
from reentry_rl.physics import constants as C

CSV_FIELDS = ["t_s", "alt_km", "lon_deg", "lat_deg", "V_mps", "gamma_deg",
              "psi_deg", "sigma_deg", "alpha_deg", "Qdot_Wm2", "qbar_Pa", "n_g", "J_heat"]

DEFAULT_PPO = dict(
    learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=10,
    gamma=0.999, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
    policy_kwargs=dict(net_arch=[128, 128]),
)


def _make_env(seed, weights):
    def _thunk():
        return Monitor(Stage1BankEnv(weights=weights, seed=seed))
    return _thunk


def _build_venv(n_envs, seed, weights, subproc):
    cls = SubprocVecEnv if subproc else DummyVecEnv
    venv = cls([_make_env(seed + i, weights) for i in range(n_envs)])
    return VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)


def _evaluate(model, vecnorm_stats, weights, outdir, seed=12345):
    """Roll out one deterministic episode; write trajectory CSV + summary."""
    eval_venv = DummyVecEnv([_make_env(seed, weights)])
    eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_venv.obs_rms = vecnorm_stats.obs_rms          # reuse training obs statistics
    eval_venv.training = False
    eval_venv.norm_reward = False

    obs = eval_venv.reset()
    rows = []
    total_r = 0.0
    last_info = {}
    for _ in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = eval_venv.step(action)
        info = infos[0]
        rows.append({k: info[k] for k in CSV_FIELDS})
        total_r += float(reward[0])
        last_info = info
        if done[0]:
            break

    df = pd.DataFrame(rows)
    df.to_csv(Path(outdir) / "eval_trajectory.csv", index=False)

    lines = [
        "Stage-1 smoke eval (deterministic)",
        f"  outcome        : {last_info.get('outcome')}",
        f"  is_success     : {last_info.get('is_success')}",
        f"  t_final [s]    : {last_info.get('t_s'):.1f}",
        f"  alt [km]       : {last_info.get('alt_km'):.3f}",
        f"  landing err km : {last_info.get('d_km', float('nan')):.2f}",
        f"  fpa err [deg]  : {last_info.get('fpa_err_deg', float('nan')):.2f}",
        f"  psi err [deg]  : {last_info.get('psi_err_deg', float('nan')):.2f}",
        f"  J_heat [J/m^2] : {last_info.get('J_heat'):.4e}",
        f"  J_ref  [J/m^2] : {C.J_REF_BANK_ONLY:.4e}",
        f"  J_heat / J_ref : {last_info.get('J_heat', float('nan')) / C.J_REF_BANK_ONLY:.4f}",
        f"  episode reward : {total_r:+.2f}",
    ]
    summary = "\n".join(lines)
    (Path(outdir) / "eval_summary.txt").write_text(summary + "\n")
    return summary, last_info


def main():
    ap = argparse.ArgumentParser(description="Stage-1 PPO smoke-train")
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subproc", action="store_true", help="use SubprocVecEnv (parallel processes)")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    weights = RewardWeights()
    # Anchor default artifacts under the repo's results/, independent of cwd.
    repo_root = Path(__file__).resolve().parents[2]
    default_out = repo_root / "results" / f"stage1_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir = Path(args.outdir) if args.outdir else default_out
    outdir.mkdir(parents=True, exist_ok=True)

    venv = _build_venv(args.n_envs, args.seed, weights, args.subproc)
    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=1, **DEFAULT_PPO)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    train_s = time.time() - t0

    model.save(str(outdir / "model.zip"))
    venv.save(str(outdir / "vecnormalize.pkl"))
    json.dump(
        {"timesteps": args.timesteps, "n_envs": args.n_envs, "seed": args.seed,
         "train_seconds": round(train_s, 1), "ppo": {k: v for k, v in DEFAULT_PPO.items()
                                                     if k != "policy_kwargs"},
         "net_arch": DEFAULT_PPO["policy_kwargs"]["net_arch"],
         "reward_weights": weights.__dict__},
        open(outdir / "config.json", "w"), indent=2)

    summary, _ = _evaluate(model, venv, weights, outdir, seed=args.seed + 999)

    print("\n" + "=" * 64)
    print(f"Stage-1 smoke-train done in {train_s:.1f}s -> {outdir}")
    print("=" * 64)
    print(summary)


if __name__ == "__main__":
    main()
