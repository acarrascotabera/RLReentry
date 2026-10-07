"""Expand a trained model to a grown observation space (warm-start surgery).

When the env gains observation features (appended LAST: the log-range feature
of v15, the eight terminal-state features of v20), previously trained models
have first-layer weights of the old width and cannot be warm-started
directly. This script rebuilds the model against the new observation space,
copies every parameter and zero-initializes the first-layer columns of the
new features, so the expanded model flies exactly the original trajectory
until training grows the new weights.

Observation statistics: the old dims keep their VecNormalize mean/var/count;
the NEW dims get the mean/var of the new features measured on rollouts of the
(behaviour-identical) expanded policy. A fresh (0, 1) init with the inherited
count of ~2.5M samples would otherwise barely move for millions of steps.

--aoa-unlock (Stage 2): on the v10->v16->HPO lineage the AoA-rate output is
always positive (+0.22..+0.64) while alpha sits on its 40 deg state limit, so
the command has no effect and the AoA channel receives no gradient. The AoA
output bias is lowered by (min output - margin): every output stays positive,
alpha stays on the limit, the trajectory is unchanged, and exploration noise
on the AoA rate (train_sb3 --action-std bank,aoa) now reaches negative rates.

    python -m reentry_rl.training.expand_obs --stage stage2 --obs-version v20 --aoa-unlock \
        --old-model results/results/stage2_polish_hpo/trial023_0.27km_300k.zip \
        --old-vn results/results/stage2_polish_hpo/trial023_0.27km_300k_vecnormalize.pkl \
        --outdir results/stage2_v20_init
"""
import argparse
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from stable_baselines3 import PPO

from reentry_rl.envs.rewards import RewardWeights
from reentry_rl.envs.scenario import NOMINAL
from reentry_rl.training.common import ENV_CLASSES, build_vecenv, load_validation_set
from reentry_rl.validation.evaluate_policy import PolicyRunner, load_policy


def _rollout(runner, scenario, raw_aoa=False):
    """-> (observations [T, n], raw AoA-rate means [T] or None, terminal info)."""
    env = runner.env
    obs, _ = env.reset(options={"scenario": scenario})
    O, A = [obs], []
    while True:
        if raw_aoa:
            on = np.clip((obs - runner.mean) / runner.std, -10.0, 10.0)
            with torch.no_grad():
                mu = runner.model.policy.get_distribution(
                    torch.as_tensor(on[None], dtype=torch.float32)).distribution.mean.numpy()[0]
            A.append(float(mu[1]))
        obs, _r, term, trunc, info = env.step(runner.act(obs))
        O.append(obs)
        if term or trunc:
            return np.asarray(O), (np.asarray(A) if raw_aoa else None), info


def expand(old_model_path, old_vn_path, stage, obs_version, outdir, aoa_unlock=False,
           aoa_margin=0.01, stats_set="S2_val_x0.1"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env_cls = ENV_CLASSES[stage]
    old, old_rms = load_policy(old_model_path, old_vn_path)
    old_dim = old.observation_space.shape[0]

    venv = build_vecenv(env_cls, 1, RewardWeights(), 3, 0, subproc=False,
                        env_kwargs={"obs_version": obs_version})
    new_dim = venv.observation_space.shape[0]
    if new_dim <= old_dim:
        raise SystemExit(f"new obs dim {new_dim} is not larger than old {old_dim}")

    new = PPO("MlpPolicy", venv, device="cpu",
              n_steps=old.n_steps, batch_size=old.batch_size, n_epochs=old.n_epochs,
              gamma=old.gamma, gae_lambda=old.gae_lambda, ent_coef=old.ent_coef,
              vf_coef=old.vf_coef, max_grad_norm=old.max_grad_norm,
              policy_kwargs=old.policy_kwargs)
    old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
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

    rms = venv.obs_rms
    rms.mean[:old_dim] = old_rms.mean
    rms.var[:old_dim] = old_rms.var
    rms.count = old_rms.count

    # statistics of the new features on rollouts of the behaviour-identical policy
    runner = PolicyRunner(new, rms, stage, env_kwargs={"obs_version": obs_version})
    _name, scen = load_validation_set(stats_set)
    O = np.concatenate([_rollout(runner, s)[0] for s in scen])
    rms.mean[old_dim:] = O[:, old_dim:].mean(axis=0)
    rms.var[old_dim:] = np.maximum(O[:, old_dim:].var(axis=0), 1e-6)
    runner = PolicyRunner(new, rms, stage, env_kwargs={"obs_version": obs_version})

    report = {"old_dim": old_dim, "new_dim": new_dim, "grown": grown,
              "new_feature_mean": rms.mean[old_dim:].round(4).tolist(),
              "new_feature_std": np.sqrt(rms.var[old_dim:]).round(4).tolist(),
              "stats_set": stats_set, "stats_samples": int(O.shape[0])}

    if aoa_unlock:
        if stage != "stage2":
            raise SystemExit("--aoa-unlock needs the stage2 (bank + AoA) policy")
        _O, A, _i = _rollout(runner, NOMINAL, raw_aoa=True)
        shift = float(A.min() - aoa_margin)
        if shift > 0:
            with torch.no_grad():
                new.policy.action_net.bias[1] -= shift
        report["aoa_unlock"] = {"raw_aoa_rate_min": float(A.min()), "raw_aoa_rate_max": float(A.max()),
                                "bias_shift": max(shift, 0.0)}

    # verification: the expanded policy must fly the original trajectory
    ref = PolicyRunner(old, old_rms, stage).run(NOMINAL)
    got = runner.run(NOMINAL)
    report["verify_nominal"] = {k: [ref[k], got[k]] for k in ("d_km", "dfpa_deg", "dpsi_deg", "tf_s")}
    same = all(abs(ref[k] - got[k]) <= 1e-9 * max(1.0, abs(ref[k])) for k in ("d_km", "dfpa_deg", "dpsi_deg"))
    report["verify_identical"] = bool(same)

    new.save(str(outdir / "model_expanded.zip"))
    venv.save(str(outdir / "vecnormalize_expanded.pkl"))
    (outdir / "expand_report.json").write_text(json.dumps(report, indent=1))
    venv.close()
    print(json.dumps(report, indent=1))
    if not same:
        raise SystemExit("[expand] expanded policy does NOT reproduce the original trajectory")
    return report


def main():
    ap = argparse.ArgumentParser(description="Expand a model + VecNormalize to a grown obs space")
    ap.add_argument("--stage", default="stage2", choices=list(ENV_CLASSES))
    ap.add_argument("--obs-version", default="v20", help="observation layout of the expanded model")
    ap.add_argument("--old-model", required=True)
    ap.add_argument("--old-vn", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--aoa-unlock", action="store_true",
                    help="lower the AoA-rate output bias so exploration can reach negative rates")
    ap.add_argument("--stats-set", default="S2_val_x0.1",
                    help="scenarios used to measure the new features' statistics")
    args = ap.parse_args()
    expand(args.old_model, args.old_vn, args.stage, args.obs_version, args.outdir,
           aoa_unlock=args.aoa_unlock, stats_set=args.stats_set)


if __name__ == "__main__":
    main()
