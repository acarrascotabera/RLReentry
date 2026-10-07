"""Expand a trained model to a grown observation space (warm-start surgery).

When the env gains a new observation feature (appended LAST, e.g. the
log-range feature added after the v14 post-mortem), previously trained models
have first-layer weights of the old width and cannot be warm-started directly.
This script rebuilds the model against the new observation space, copies every
parameter, and zero-initializes the first-layer column for the new feature —
so the expanded model's behavior is bit-identical to the original until
training grows the new weights. The VecNormalize statistics are carried over
the same way (new dim gets a neutral init; it self-corrects during training).

    python -m reentry_rl.training.expand_obs --stage stage2 \
        --old-model results/stage2_v13/best/best_model.zip \
        --old-vn results/stage2_v13/ckpt/ppo_vecnormalize_2750000_steps.pkl \
        --outdir results/stage2_v13/expanded
"""
import argparse
import pickle
from pathlib import Path

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import torch
from stable_baselines3 import PPO

from reentry_rl.envs.rewards import RewardWeights
from reentry_rl.training.common import ENV_CLASSES, build_vecenv


def expand_model(old_model_path, env_cls, outpath):
    old = PPO.load(str(old_model_path), device="cpu")
    old_dim = old.observation_space.shape[0]

    venv = build_vecenv(env_cls, 1, RewardWeights(), 3, 0, subproc=False)
    new_dim = venv.observation_space.shape[0]
    if new_dim <= old_dim:
        raise SystemExit(f"new obs dim {new_dim} is not larger than old {old_dim}")

    new = PPO("MlpPolicy", venv, device="cpu",
              n_steps=old.n_steps, batch_size=old.batch_size, n_epochs=old.n_epochs,
              gamma=old.gamma, gae_lambda=old.gae_lambda, ent_coef=old.ent_coef,
              vf_coef=old.vf_coef, max_grad_norm=old.max_grad_norm,
              policy_kwargs=old.policy_kwargs)

    old_sd = old.policy.state_dict()
    new_sd = new.policy.state_dict()
    grown = []
    for k, v_new in new_sd.items():
        v_old = old_sd[k]
        if v_old.shape == v_new.shape:
            new_sd[k] = v_old
        else:                                   # first-layer weight: zero-pad new columns
            assert v_new.shape[0] == v_old.shape[0] and v_new.shape[1] > v_old.shape[1], \
                f"unexpected shape change {k}: {v_old.shape} -> {v_new.shape}"
            padded = torch.zeros_like(v_new)
            padded[:, :v_old.shape[1]] = v_old
            new_sd[k] = padded
            grown.append(k)
    new.policy.load_state_dict(new_sd)
    new.save(str(outpath))
    venv.close()
    print(f"[expand] {old_dim} -> {new_dim} dims; zero-padded: {grown}")
    return old_dim, new_dim


def expand_vecnormalize(old_vn_path, env_cls, old_dim, new_dim, outpath):
    with open(old_vn_path, "rb") as f:
        old_vn = pickle.load(f)                 # VecNormalize object (venv=None after pickling)
    venv = build_vecenv(env_cls, 1, RewardWeights(), 3, 0, subproc=False)
    venv.obs_rms.mean[:old_dim] = old_vn.obs_rms.mean
    venv.obs_rms.var[:old_dim] = old_vn.obs_rms.var
    venv.obs_rms.count = old_vn.obs_rms.count
    # new dims keep the fresh (0, 1) init — harmless: their first-layer weights are
    # zero, and the running stats self-correct once training starts
    venv.save(str(outpath))
    venv.close()
    print(f"[expand] vecnormalize {old_dim} -> {new_dim} dims (count={old_vn.obs_rms.count:.0f})")


def main():
    ap = argparse.ArgumentParser(description="Expand a model + VecNormalize to a grown obs space")
    ap.add_argument("--stage", default="stage2", choices=list(ENV_CLASSES))
    ap.add_argument("--old-model", required=True)
    ap.add_argument("--old-vn", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env_cls = ENV_CLASSES[args.stage]
    old_dim, new_dim = expand_model(args.old_model, env_cls, outdir / "model_expanded.zip")
    expand_vecnormalize(args.old_vn, env_cls, old_dim, new_dim, outdir / "vecnormalize_expanded.pkl")
    print(f"[expand] -> {outdir/'model_expanded.zip'} + {outdir/'vecnormalize_expanded.pkl'}")


if __name__ == "__main__":
    main()
