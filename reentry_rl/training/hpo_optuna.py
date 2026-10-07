"""Optuna HPO of the PPO *algorithm* for the warm-started precision polish.

Post-campaign context (see results/RUNS.md): reward shaping is frozen at the
v19 configuration (concave precision well + strong terminal magnet); the
remaining spread between 5.4 km and the 1 km goal was repeatedly bounded by
OPTIMIZATION settings (initial action std, lr/clip scale, batch size), so this
study searches exactly those. Every trial warm-starts from the campaign-best
model (v16-best, 5.39 km) and its VecNormalize statistics, keeping the
512x512 architecture fixed (required by the warm start).

Objective (MINIMIZE): the best validation score seen during the trial
(evaluate_policy.selection_score over the --val-set scenarios) —
    per case, feasible (max path-constraint ratio <= 1.02):  e = terminal error
              infeasible:                                    e = 1000 + error
              no handover reached:                           e = 10000 + error
    aggregate: 90th percentile over cases (one case = its value)
With --val-set S0_nominal --score-mode position --legacy-terminal this is
exactly the objective of the July 2026 study (the 0.273 km trial 23). The
running MINIMUM is reported to the MedianPruner. Any evaluation below
`save-below` saves the model + VecNormalize immediately (lesson from v19: the
best policies lived between checkpoints and were lost).

Note: the July study ran with the v19 reward (frozen_weights below), warm
started from v16-best which had been trained with the v16 linear well.

TPE sampler + MedianPruner, SQLite storage -> fully resumable:

    python -m reentry_rl.training.hpo_optuna --trials 40 \
        --init-model results/stage2_v16/best/best_model.zip \
        --init-vn results/stage2_v16/ckpt/ppo_vecnormalize_2500000_steps.pkl
"""
import argparse
import time
from pathlib import Path

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from reentry_rl.envs.rewards import RewardWeights
from reentry_rl.training.common import ENV_CLASSES, J_REF, ensure_child_importable, \
    build_vecenv, load_validation_set, set_action_std
from reentry_rl.validation.evaluate_policy import PolicyRunner, selection_score


def frozen_weights():
    """v19 reward configuration — FIXED for this study (algorithm HPO only)."""
    w = RewardWeights()
    w.w_succ = 500.0
    w.succ_ramp_km = 6.0
    return w


def linear_schedule(lr0):
    return lambda progress_remaining: lr0 * progress_remaining


def sample_ppo(trial):
    """Polish-phase PPO search space. net_arch is fixed by the warm start."""
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 5e-5, 2e-3, log=True)
    return dict(
        learning_rate=linear_schedule(trial.suggest_float("lr", 1e-5, 3e-4, log=True)),
        n_steps=trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
        n_epochs=trial.suggest_int("n_epochs", 3, 15),
        gamma=1.0 - one_minus_gamma,
        gae_lambda=trial.suggest_float("gae_lambda", 0.90, 0.99),
        clip_range=trial.suggest_float("clip_range", 0.05, 0.3, log=True),
        ent_coef=trial.suggest_float("ent_coef", 1e-8, 5e-3, log=True),
    )


class PolishScoreCallback(BaseCallback):
    """Every eval_freq calls: deterministic rollouts on the validation set ->
    selection score; track the trial minimum, report to the pruner, and save
    any sub-threshold model immediately."""

    def __init__(self, trial, stage, scenarios, weights, env_kwargs, score_mode, eval_freq,
                 outdir, save_below, verbose=0):
        super().__init__(verbose)
        self.trial = trial
        self.stage = stage
        self.scenarios = scenarios
        self.weights = weights
        self.env_kwargs = env_kwargs
        self.score_mode = score_mode
        self.eval_freq = max(int(eval_freq), 1)
        self.outdir = Path(outdir)
        self.save_below = float(save_below)
        self.best = float("inf")
        self.best_d = float("nan")
        self._runner = None

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True
        rms = self.training_env.obs_rms
        if self._runner is None:
            self._runner = PolicyRunner(self.model, rms, self.stage, self.weights, self.env_kwargs)
        self._runner.mean = rms.mean.astype("float64")
        self._runner.std = (rms.var.astype("float64") + 1e-8) ** 0.5
        df = self._runner.run_many(self.scenarios)
        score = selection_score(df, self.score_mode)
        d = float(df["d_km"].median())
        if score < self.best:
            self.best, self.best_d = score, d
            if score < self.save_below:         # feasible AND close -> keep the model
                tag = f"trial{self.trial.number:03d}_{score:.2f}_{self.num_timesteps//1000}k"
                self.model.save(str(self.outdir / f"{tag}.zip"))
                self.training_env.save(str(self.outdir / f"{tag}_vecnormalize.pkl"))
                print(f"[hpo] trial {self.trial.number}: SAVED {tag} "
                      f"(median d={d:.2f} km, max ratio={df['max_ratio'].max():.2f})")
        self.trial.report(self.best, self.num_timesteps)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return True


def objective(trial, args, env_cls, j_ref, weights, env_kwargs, val_scen):
    ppo_kwargs = sample_ppo(trial)
    action_std = trial.suggest_float("action_std", 0.02, 0.3, log=True)

    venv = build_vecenv(env_cls, args.n_envs, weights, args.n_substeps,
                        seed=args.seed, subproc=not args.no_subproc, env_kwargs=env_kwargs)
    venv = VecNormalize.load(args.init_vn, venv.venv)
    venv.training = True
    venv.norm_reward = False

    model = PPO.load(args.init_model, env=venv, device="cpu",
                     custom_objects=ppo_kwargs)
    set_action_std(model, action_std)

    cb = PolishScoreCallback(trial, args.stage, val_scen, weights, env_kwargs, args.score_mode,
                             eval_freq=max(args.eval_freq // args.n_envs, 1),
                             outdir=args.outdir, save_below=args.save_below)
    try:
        model.learn(total_timesteps=args.steps_per_trial, callback=cb,
                    progress_bar=False)
    finally:
        venv.close()
    trial.set_user_attr("best_d_km", cb.best_d)
    return cb.best


def main():
    ap = argparse.ArgumentParser(description="Optuna HPO: warm-started PPO precision polish")
    ap.add_argument("--stage", default="stage2", choices=list(ENV_CLASSES))
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--steps-per-trial", type=int, default=1_500_000)
    ap.add_argument("--eval-freq", type=int, default=100_000, help="total env steps between scored rollouts")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-substeps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-subproc", action="store_true")
    ap.add_argument("--init-model", required=True, help="warm-start model .zip (fixes net_arch)")
    ap.add_argument("--init-vn", required=True, help="warm-start VecNormalize .pkl")
    ap.add_argument("--save-below", type=float, default=6.0,
                    help="save model+stats whenever the validation score is below this")
    ap.add_argument("--val-set", default="S0_nominal",
                    help="validation set (results/eval_sets/<name>.json or path)")
    ap.add_argument("--score-mode", default="position", choices=["position", "full"])
    ap.add_argument("--legacy-terminal", action="store_true",
                    help="measure at the first step below 25 km (July 2026 study behaviour)")
    ap.add_argument("--weights-from", default=None,
                    help="config.json whose reward_weights replace the frozen v19 reward")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--study-name", default="stage2_polish_hpo")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if not args.no_subproc:
        ensure_child_importable(repo_root)
    outdir = Path(args.outdir) if args.outdir else repo_root / "results" / args.study_name
    outdir.mkdir(parents=True, exist_ok=True)
    args.outdir = str(outdir)

    env_cls, j_ref = ENV_CLASSES[args.stage], J_REF[args.stage]
    weights = frozen_weights()
    if args.weights_from:
        import json as _json
        for k, v in _json.load(open(args.weights_from))["reward_weights"].items():
            if k in RewardWeights.__dataclass_fields__:
                setattr(weights, k, v)
    env_kwargs = {"exact_terminal": not args.legacy_terminal}
    _val_name, val_scen = load_validation_set(args.val_set)

    storage = f"sqlite:///{(outdir / 'study.db').as_posix()}"
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=400_000))
    print(f"[hpo] study '{args.study_name}' -> {storage}")
    print(f"[hpo] warm start: {args.init_model}")
    print(f"[hpo] frozen reward: {weights}")
    t0 = time.time()
    study.optimize(lambda tr: objective(tr, args, env_cls, j_ref, weights, env_kwargs, val_scen),
                   n_trials=args.trials, gc_after_trial=True)

    print(f"\n[hpo] done in {(time.time()-t0)/3600:.1f} h — best trial:")
    bt = study.best_trial
    print(f"  value (gated d_km) : {bt.value:.2f}")
    print(f"  best_d_km          : {bt.user_attrs.get('best_d_km')}")
    for k, v in bt.params.items():
        print(f"  {k:18s}: {v}")
    import json
    json.dump({"best_params": bt.params, "best_value": bt.value,
               "best_d_km": bt.user_attrs.get("best_d_km"),
               "n_trials": len(study.trials)},
              open(outdir / "best.json", "w"), indent=2)
    print(f"[hpo] best.json -> {outdir/'best.json'}")


if __name__ == "__main__":
    main()
