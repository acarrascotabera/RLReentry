"""Stage-1 full training (PPO).

Real training run: SubprocVecEnv parallelism, TensorBoard + Monitor logging,
periodic deterministic evaluation on a fixed validation set with best-model
checkpointing (ValidationCallback), and a final eval + trajectory CSV vs the
SCP reference. Intended for multi-million-step runs.

Run (from anywhere; runnable as a file or via -m):
    python e:/reentry_RL/reentry_rl/training/train_sb3.py --timesteps 3000000 --n-envs 8
    python -m reentry_rl.training.train_sb3 --timesteps 5000000 --n-envs 12 --w-progress 25

Inspect learning curves:
    tensorboard --logdir results/stage1_train_<ts>/tb

Artifacts in results/stage1_train_<ts>/:
    model.zip, vecnormalize.pkl, config.json, best/ (best_model.zip,
    best_vecnormalize.pkl, best.json), ckpt/, snapshots/, eval/validation_history.csv,
    eval_trajectory.csv, eval_summary.txt, tb/
"""
import argparse
import json
import re
import time
from pathlib import Path

# Runnable as a file or as a module.
if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from reentry_rl.training.common import (ENV_CLASSES, J_REF, ensure_child_importable,
                                        build_vecenv, evaluate_to_csv, ValidationCallback,
                                        load_validation_set, set_action_std)
from reentry_rl.envs.rewards import RewardWeights
from reentry_rl.training.curriculum import PRESETS, TerminalCurriculum

# Manual defaults — these become the Optuna HPO search seed (milestone 2b).
DEFAULT_PPO = dict(
    learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
    gamma=0.999, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
    vf_coef=0.5, max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
)


def linear_schedule(lr0):
    """Linear LR decay lr0 -> 0 over training (SB3 passes progress_remaining: 1 -> 0).
    Lets the policy SETTLE into a converged trajectory instead of oscillating."""
    return lambda progress_remaining: lr0 * progress_remaining


def parse_args():
    ap = argparse.ArgumentParser(description="Stage-1 full PPO training")
    ap.add_argument("--stage", default="stage1", choices=list(ENV_CLASSES))
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-substeps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-subproc", action="store_true", help="use DummyVecEnv (single process)")
    ap.add_argument("--eval-freq", type=int, default=50_000, help="total env steps between evals")
    ap.add_argument("--val-set", default="S0_nominal",
                    help="validation set for model selection (results/eval_sets/<name>.json or path)")
    ap.add_argument("--score-mode", default="position", choices=["position", "full"],
                    help="selection score: landing error only, or ||(d/1km, dfpa/1deg, dpsi/5deg)||")
    ap.add_argument("--save-below", type=float, default=None,
                    help="also snapshot every model whose validation score is below this")
    ap.add_argument("--obs-version", default=None, help="observation layout (default: env default)")
    ap.add_argument("--set-weight", action="append", default=[], metavar="KEY=VALUE",
                    help="override any RewardWeights field (repeatable), e.g. w_ang_mult=0.25")
    ap.add_argument("--curriculum", default=None, choices=sorted(PRESETS),
                    help="terminal-tolerance curriculum on the FPA/heading reward channels")
    ap.add_argument("--curriculum-min-steps", type=int, default=300_000)
    ap.add_argument("--curriculum-pass-frac", type=float, default=0.75)
    ap.add_argument("--legacy-terminal", action="store_true",
                    help="terminal state at the first step below 25 km instead of the exact crossing")
    ap.add_argument("--legacy-integrator", action="store_true",
                    help="attitude bounds only after each RK4 substep (pre-2026-10 dynamics)")
    ap.add_argument("--ckpt-freq", type=int, default=250_000, help="total env steps between checkpoints")
    ap.add_argument("--outdir", default=None)
    # reward-weight overrides (the main tuning knobs)
    ap.add_argument("--w-pos", type=float, default=None)
    ap.add_argument("--w-near", type=float, default=None)
    ap.add_argument("--d-near-km", type=float, default=None)
    ap.add_argument("--w-lin", type=float, default=None)
    ap.add_argument("--d-lin-km", type=float, default=None)
    ap.add_argument("--near-convexity", type=float, default=None)
    ap.add_argument("--w-endsmooth", type=float, default=None)
    ap.add_argument("--w-succ", type=float, default=None)
    ap.add_argument("--succ-ramp-km", type=float, default=None)
    ap.add_argument("--w-fail", type=float, default=None)
    ap.add_argument("--w-smooth", type=float, default=None)
    ap.add_argument("--w-path", type=float, default=None)
    ap.add_argument("--lr-decay", action="store_true", help="linearly decay LR to 0 (aids convergence)")
    ap.add_argument("--hpo-best", default=None, help="path to an HPO best.json: load its PPO hyperparameters + reward weights")
    ap.add_argument("--ent-coef", type=float, default=None, help="override PPO entropy coef")
    ap.add_argument("--gamma", type=float, default=None,
                    help="override PPO discount (terminal-anchor weight ~ gamma^ep_len)")
    ap.add_argument("--lr", type=float, default=None, help="override PPO learning rate")
    ap.add_argument("--clip-range", type=float, default=None, help="override PPO clip range")
    ap.add_argument("--batch-size", type=int, default=None, help="override PPO minibatch size")
    ap.add_argument("--n-steps", type=int, default=None, help="override PPO rollout length per env")
    ap.add_argument("--n-epochs", type=int, default=None, help="override PPO epochs per update")
    ap.add_argument("--target-kl", type=float, default=None,
                    help="stop an update's epochs early once approx KL > 1.5 x this (PPO target_kl)")
    ap.add_argument("--init-model", default=None,
                    help="warm-start: initialize policy/value weights from this model .zip "
                         "(fresh run dir + timesteps; stored hyperparams apply unless overridden)")
    ap.add_argument("--init-vn", default=None,
                    help="VecNormalize .pkl to initialize obs stats (use with --init-model)")
    ap.add_argument("--action-std", default=None,
                    help="set the policy's initial action std, one value or one per action "
                         "(e.g. 0.02,0.15 to re-open AoA exploration on a bank-converged policy)")
    ap.add_argument("--tb", action="store_true", help="enable TensorBoard logging (off by default; flaky on Windows)")
    ap.add_argument("--resume", default=None, help="resume from the latest checkpoint in this run dir (continues in place)")
    return ap.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    subproc = not args.no_subproc
    if subproc:
        ensure_child_importable(repo_root)

    env_cls = ENV_CLASSES[args.stage]
    j_ref = J_REF[args.stage]
    env_kwargs = {"exact_terminal": not args.legacy_terminal,
                  "rate_saturation": not args.legacy_integrator}
    if args.obs_version:
        env_kwargs["obs_version"] = args.obs_version

    weights = RewardWeights()
    ppo_kwargs = dict(DEFAULT_PPO)
    resume_ckpt = resume_vn = None
    lr_base = DEFAULT_PPO["learning_rate"]

    if args.resume:                         # continue an existing run from its latest checkpoint
        outdir = Path(args.resume)
        rc = json.load(open(outdir / "config.json"))
        for k, v in rc.get("reward_weights", {}).items():
            if k in RewardWeights.__dataclass_fields__:
                setattr(weights, k, v)
        lr_base = rc.get("lr_base", DEFAULT_PPO["learning_rate"])
        cks = sorted((outdir / "ckpt").glob("ppo_*_steps.zip"),
                     key=lambda p: int(re.search(r"_(\d+)_steps", p.name).group(1)))
        if not cks:
            raise SystemExit(f"--resume: no checkpoints in {outdir/'ckpt'}")
        resume_ckpt = cks[-1]
        resume_vn = outdir / "ckpt" / resume_ckpt.name.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")
    else:
        if args.hpo_best:                   # load tuned reward weights + PPO hyperparameters
            bp = json.load(open(args.hpo_best))["best_params"]
            for k, v in bp.items():         # only weights that still exist in the minimal reward
                if k in RewardWeights.__dataclass_fields__:
                    setattr(weights, k, v)
            for k in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef"):
                if k in bp:
                    ppo_kwargs[k] = bp[k]
            if "net_width" in bp:
                ppo_kwargs["policy_kwargs"] = dict(net_arch=[bp["net_width"], bp["net_width"]])
            if "lr" in bp:
                ppo_kwargs["learning_rate"] = bp["lr"]
        for name in ("w_pos", "w_near", "d_near_km", "w_lin", "d_lin_km", "near_convexity",
                     "w_endsmooth", "w_succ", "succ_ramp_km", "w_fail", "w_smooth", "w_path"):
            val = getattr(args, name)       # individual CLI overrides win over --hpo-best
            if val is not None:
                setattr(weights, name, val)
        for kv in args.set_weight:
            key, val = kv.split("=", 1)
            if key not in RewardWeights.__dataclass_fields__:
                raise SystemExit(f"--set-weight: unknown RewardWeights field {key!r}")
            setattr(weights, key, float(val))
        if args.ent_coef is not None:
            ppo_kwargs["ent_coef"] = args.ent_coef
        if args.gamma is not None:
            ppo_kwargs["gamma"] = args.gamma
        if args.lr is not None:
            ppo_kwargs["learning_rate"] = args.lr
        if args.clip_range is not None:
            ppo_kwargs["clip_range"] = args.clip_range
        if args.batch_size is not None:
            ppo_kwargs["batch_size"] = args.batch_size
        if args.n_steps is not None:
            ppo_kwargs["n_steps"] = args.n_steps
        if args.n_epochs is not None:
            ppo_kwargs["n_epochs"] = args.n_epochs
        if args.target_kl is not None:
            ppo_kwargs["target_kl"] = args.target_kl
        lr_base = ppo_kwargs["learning_rate"] if isinstance(ppo_kwargs["learning_rate"], (int, float)) else DEFAULT_PPO["learning_rate"]
        if args.lr_decay:
            ppo_kwargs["learning_rate"] = linear_schedule(lr_base)
        ts = time.strftime("%Y%m%d_%H%M%S")
        outdir = Path(args.outdir) if args.outdir else repo_root / "results" / f"{args.stage}_train_{ts}"
    for sub in ("", "tb", "ckpt", "best", "eval"):
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    curriculum = None
    if args.curriculum:
        curriculum = TerminalCurriculum(args.curriculum, outdir, pass_frac=args.curriculum_pass_frac,
                                        min_steps=args.curriculum_min_steps)
        if args.resume:
            curriculum.restore()
        curriculum.apply_to(weights)          # envs start at the current level
        print(f"[curriculum] {args.curriculum} level {curriculum.level}: {curriculum.params()}")

    # Vectorized training env + a single-env eval env (EvalCallback syncs VecNormalize stats).
    venv = build_vecenv(env_cls, args.n_envs, weights, args.n_substeps, args.seed, subproc,
                        env_kwargs)
    if args.resume:                         # restore observation/return normalization stats
        from stable_baselines3.common.vec_env import VecNormalize
        venv = VecNormalize.load(str(resume_vn), venv.venv)
        venv.training = True
        venv.norm_reward = False
    elif args.init_vn:                      # warm start: adopt the source run's obs stats
        from stable_baselines3.common.vec_env import VecNormalize
        venv = VecNormalize.load(args.init_vn, venv.venv)
        venv.training = True
        venv.norm_reward = False
    # eval_freq / ckpt_freq are given in TOTAL env steps -> convert to per-env-call counts.
    eval_freq = max(args.eval_freq // args.n_envs, 1)
    ckpt_freq = max(args.ckpt_freq // args.n_envs, 1)
    val_name, val_scen = load_validation_set(args.val_set)
    val_cb = ValidationCallback(args.stage, val_scen, eval_freq, outdir, weights, env_kwargs,
                                score_mode=args.score_mode, save_below=args.save_below,
                                set_name=val_name, verbose=1)
    if curriculum is not None:
        val_cb.listeners.append(curriculum.on_validation)
    if args.resume and (outdir / "best" / "best.json").exists():
        # keep the pre-resume best (EvalCallback used to forget it and overwrite best/)
        val_cb.best = float(json.load(open(outdir / "best" / "best.json"))["score"])
    ckpt_cb = CheckpointCallback(save_freq=ckpt_freq, save_path=str(outdir / "ckpt"),
                                 name_prefix="ppo", save_vecnormalize=True)

    tb_log = str(outdir / "tb") if args.tb else None
    if args.resume:
        model = PPO.load(str(resume_ckpt), env=venv, device="cpu", tensorboard_log=tb_log)
        if args.lr_decay:
            model.lr_schedule = linear_schedule(lr_base)
        if args.ent_coef is not None:
            model.ent_coef = args.ent_coef
        print(f"[resume] loaded {resume_ckpt.name} at num_timesteps={model.num_timesteps:,} -> target {args.timesteps:,}")
    elif args.init_model:
        # Warm start: keep the source model's hyperparameters except the explicit
        # CLI overrides (the point is usually a gentler polish: lower lr/clip).
        overrides = {}
        if args.lr is not None:
            overrides["learning_rate"] = linear_schedule(args.lr) if args.lr_decay else args.lr
        if args.clip_range is not None:
            overrides["clip_range"] = args.clip_range
        if args.ent_coef is not None:
            overrides["ent_coef"] = args.ent_coef
        if args.gamma is not None:
            overrides["gamma"] = args.gamma
        if args.target_kl is not None:
            overrides["target_kl"] = args.target_kl
        model = PPO.load(args.init_model, env=venv, seed=args.seed, device="cpu",
                         tensorboard_log=tb_log, verbose=1, **overrides)
        print(f"[init] warm-started from {args.init_model} (overrides: {sorted(overrides)})")
    else:
        model = PPO("MlpPolicy", venv, seed=args.seed, verbose=1,
                    tensorboard_log=tb_log, **ppo_kwargs)

    if args.action_std is not None:
        stds = set_action_std(model, [float(v) for v in str(args.action_std).split(",")])
        print(f"[init] policy action std set to {stds}")

    if not args.resume:
        json.dump({
            "stage": args.stage, "timesteps": args.timesteps, "n_envs": args.n_envs,
            "n_substeps": args.n_substeps, "subproc": subproc, "seed": args.seed,
            "j_ref": j_ref, "lr_decay": args.lr_decay, "hpo_best": args.hpo_best, "lr_base": lr_base,
            "init_model": args.init_model, "init_vn": args.init_vn,
            "ppo": {k: (v if isinstance(v, (int, float, str)) else "schedule")
                    for k, v in ppo_kwargs.items() if k != "policy_kwargs"},
            "ppo_effective": {"gamma": model.gamma, "gae_lambda": model.gae_lambda,
                              "n_steps": model.n_steps, "batch_size": model.batch_size,
                              "n_epochs": model.n_epochs, "ent_coef": model.ent_coef,
                              "target_kl": model.target_kl},
            "net_arch": model.policy.net_arch,       # the model's real architecture
            "action_std": args.action_std,
            "env_kwargs": env_kwargs, "val_set": val_name, "score_mode": args.score_mode,
            "curriculum": (None if curriculum is None else
                           {"preset": args.curriculum, "min_steps": args.curriculum_min_steps,
                            "pass_frac": args.curriculum_pass_frac, "levels": curriculum.levels,
                            "base": curriculum.base}),
            "reward_weights": weights.__dict__,
        }, open(outdir / "config.json", "w"), indent=2)

    print(f"[train_sb3] stage={args.stage} timesteps={args.timesteps} n_envs={args.n_envs} "
          f"subproc={subproc} substeps={args.n_substeps} -> {outdir}")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not bool(args.resume),
                callback=[val_cb, ckpt_cb], progress_bar=False)
    train_s = time.time() - t0

    model.save(str(outdir / "model.zip"))
    venv.save(str(outdir / "vecnormalize.pkl"))
    summary, _ = evaluate_to_csv(model, venv.obs_rms, env_cls, weights, args.n_substeps,
                                 j_ref, outdir, stage=args.stage, seed=args.seed + 999,
                                 env_kwargs=env_kwargs)

    print("\n" + "=" * 64)
    print(f"{args.stage} training done in {train_s/60:.1f} min -> {outdir}")
    print("=" * 64)
    print(summary)
    print(f"\nbest model: {outdir/'best'/'best_model.zip'}")
    print(f"tensorboard --logdir {outdir/'tb'}")


if __name__ == "__main__":
    main()
