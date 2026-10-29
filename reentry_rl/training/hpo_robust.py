"""Optuna HPO of the PPO algorithm around the robust policy (v26 6M, October 2026).

Every trial is a train_sb3 run with the exact setup of a reference run (its config.json:
warm start, environment, dispersion sampler, PID-Lagrangian constraints, reward weights,
validation set); only the PPO hyperparameters below vary. The reference default is
results/stage2_v28/config.json: v26 6M retrained at the 0.7x envelope under the capped
aerodynamic model, which is also the study's baseline (trial 0 is enqueued with its PPO
settings).

Search space: lr (linear decay over the trial), n_steps, batch_size, n_epochs, gamma,
gae_lambda, clip_range, ent_coef, target_kl, and a multiplier on the warm-start policy std
(per action: bank, AoA). The 512x512 net is fixed by the warm start.

Objective (MINIMIZE): evaluate_policy.robust_score on --sel-set (default S2_sel_a5_M,
151 cases: nominal + 50 each at 0.1x, 0.5x, 1x, capped aero; seeds disjoint from the
training validation set and from the S4/S5 reporting sets), the better of the trial's
final model and its best checkpoint by the in-training validation score. The in-training
validation score (64-case set, every --eval-freq) is reported as a running minimum to a
MedianPruner; a pruned trial's training process is killed.

SQLite storage, resumable:

    python -m reentry_rl.training.hpo_robust --trials 20 --steps-per-trial 1000000
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import optuna
import pandas as pd

from reentry_rl.validation.evaluate_policy import robust_score

ROOT = Path(__file__).resolve().parents[2]
LAG_WEIGHTS = ("lam_Qdot", "lam_qbar", "lam_n", "w_peak")     # set by the PID-Lagrangian


def flags_from_config(cfg):
    """train_sb3 flags that rebuild a (non-resumed) run from its config.json."""
    f = ["--stage", cfg["stage"], "--n-envs", str(cfg["n_envs"]), "--n-substeps", str(cfg["n_substeps"]),
         "--seed", str(cfg["seed"]), "--init-model", cfg["init_model"], "--init-vn", cfg["init_vn"]]
    if not cfg["subproc"]:
        f.append("--no-subproc")
    if cfg["lr_decay"]:
        f.append("--lr-decay")
    ek = cfg["env_kwargs"]
    if not ek.get("exact_terminal", True):
        f.append("--legacy-terminal")
    if not ek.get("rate_saturation", True):
        f.append("--legacy-integrator")
    for key, flag in (("obs_version", "--obs-version"), ("action_mode", "--action-mode"),
                      ("step_mode", "--step-mode"), ("peak_charge", "--peak-charge"),
                      ("qdot_margin", "--qdot-margin"), ("n_margin", "--n-margin"),
                      ("energy_steps", "--energy-steps")):
        if key in ek:
            f += [flag, str(ek[key])]
    if ek.get("qdot_floor"):
        f.append("--qdot-floor")
    td = cfg.get("train_dispersion")
    if td:
        f += ["--train-dispersion", ",".join(td["groups"])]
        if td.get("aero_cap") is not None:
            f += ["--aero-cap", str(td["aero_cap"])]
        dc = cfg.get("dispersion_curriculum")
        if dc:
            f += ["--disp-levels", ",".join(f"{v:g}" for v in dc["levels"]), "--disp-p-edge", str(td["p_edge"]),
                  "--disp-min-steps", str(dc["min_steps"]), "--disp-max-steps", str(dc["max_steps"]),
                  "--disp-window", str(dc["window"]), "--disp-reach", str(dc["reach_min"]),
                  "--disp-d-med-km", str(dc["d_med_km"])]
        else:
            f += ["--train-sigma-scale", str(td["sigma_scale"])]
    lg = cfg.get("lagrangian")
    if lg:
        f += ["--lagrangian", "--lag-budget", str(lg["budget"]), "--lag-kp", str(lg["kp"]),
              "--lag-ki", str(lg["ki"]), "--lag-kd", str(lg["kd"]), "--lag-init", str(lg["lam_init"]),
              "--lag-max", str(lg["lam_max"]), "--lag-start", str(lg["start_step"])]
        if lg.get("rate_max") is not None:
            f += ["--lag-rate", str(lg["rate_max"])]
        if cfg.get("lag_restore_from"):
            f += ["--lag-restore-from", cfg["lag_restore_from"]]
    f += ["--val-set", cfg["val_set"], "--score-mode", cfg["score_mode"]]
    for k, v in cfg["reward_weights"].items():
        if k not in LAG_WEIGHTS:
            f += ["--set-weight", f"{k}={v!r}"]
    return f


def warm_start_std(cfg):
    import io
    import zipfile
    import torch
    with zipfile.ZipFile(ROOT / cfg["init_model"]) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
    return [float(v) for v in sd["log_std"].exp()]


def baseline_params(cfg):
    p = cfg["ppo_effective"]
    return {"lr": cfg["lr_base"], "n_steps": p["n_steps"], "batch_size": p["batch_size"],
            "n_epochs": p["n_epochs"], "one_minus_gamma": 1.0 - p["gamma"],
            "gae_lambda": p["gae_lambda"], "clip_range": 0.2, "ent_coef": max(p["ent_coef"], 1e-7),
            "target_kl": p["target_kl"], "std_mult": 1.0}


def sample(trial):
    return {"lr": trial.suggest_float("lr", 1e-5, 3e-4, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
            "n_epochs": trial.suggest_int("n_epochs", 3, 12),
            "one_minus_gamma": trial.suggest_float("one_minus_gamma", 5e-5, 2e-3, log=True),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
            "clip_range": trial.suggest_float("clip_range", 0.05, 0.3, log=True),
            "ent_coef": trial.suggest_float("ent_coef", 1e-7, 1e-3, log=True),
            "target_kl": trial.suggest_categorical("target_kl", [0.01, 0.02, 0.05, 0.1]),
            "std_mult": trial.suggest_float("std_mult", 0.3, 2.0, log=True)}


def kill_tree(proc):
    if proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        proc.wait()


def read_history(path):
    try:
        return pd.read_csv(path) if path.exists() else None
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None                                   # being written


def select_eval(model, vn, sel_set, outdir, workers):
    """robust_score of one model on the selection set, plus summary statistics."""
    cmd = [sys.executable, "-m", "reentry_rl.validation.evaluate_policy", "--model", str(model),
           "--vn", str(vn), "--sets", sel_set, "--outdir", str(outdir), "--workers", str(workers)]
    with open(Path(outdir).with_suffix(".log"), "w") as log:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    df = pd.read_csv(Path(outdir) / f"{Path(sel_set).stem}.csv")
    return {"score": robust_score(df), "d_med": float(df["d_km"].median()),
            "d_p90": float(df["d_km"].quantile(0.9)), "feasible": float(df["feasible"].mean()),
            "fpa_med": float(df["dfpa_deg"].abs().median()), "psi_med": float(df["dpsi_deg"].abs().median())}


def objective(trial, args, base_flags, std0, base):
    # trial 0 is the enqueued baseline: its fixed params come through suggest_* as well
    p = sample(trial)
    tdir = Path(args.outdir) / f"trial_{trial.number:03d}"
    std = ",".join(f"{s * p['std_mult']:.6g}" for s in std0)
    cmd = [sys.executable, "-m", "reentry_rl.training.train_sb3", *base_flags,
           "--timesteps", str(args.steps_per_trial), "--outdir", str(tdir),
           "--eval-freq", str(args.eval_freq), "--ckpt-freq", str(args.steps_per_trial),
           "--val-workers", str(args.workers),
           "--lr", f"{p['lr']:.6g}", "--n-steps", str(p["n_steps"]), "--batch-size", str(p["batch_size"]),
           "--n-epochs", str(p["n_epochs"]), "--gamma", f"{1.0 - p['one_minus_gamma']:.8g}",
           "--gae-lambda", f"{p['gae_lambda']:.6g}", "--clip-range", f"{p['clip_range']:.6g}",
           "--ent-coef", f"{p['ent_coef']:.6g}", "--target-kl", str(p["target_kl"]), "--action-std", std]
    if args.val_set:
        cmd[cmd.index("--val-set") + 1] = args.val_set
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "hpo_cmd.json").write_text(json.dumps({"params": p, "cmd": cmd}, indent=1))
    print(f"[hpo] trial {trial.number}: {p}", flush=True)

    hist_path, log_path = tdir / "eval" / "validation_history.csv", tdir / "train_log.txt"
    seen, best, t_last = 0, math.inf, time.time()
    with open(log_path, "w") as out, open(tdir / "train_stderr.txt", "w") as err:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=err)
        try:
            while True:
                done = proc.poll() is not None
                h = read_history(hist_path)
                if h is not None and len(h) > seen:
                    for _, r in h.iloc[seen:].iterrows():
                        best = min(best, float(r["score"]))
                        trial.report(best, int(r["step"]))
                        print(f"[hpo] trial {trial.number}: step {int(r['step']):,} val {r['score']:.3f} "
                              f"(running min {best:.3f})", flush=True)
                    seen, t_last = len(h), time.time()
                    if trial.should_prune():
                        kill_tree(proc)
                        raise optuna.TrialPruned()
                if done:
                    break
                if time.time() - t_last > args.stall_min * 60:
                    kill_tree(proc)
                    raise RuntimeError(f"trial {trial.number}: no validation for {args.stall_min} min")
                time.sleep(30)
        finally:
            kill_tree(proc)
    if proc.returncode != 0:
        raise RuntimeError(f"trial {trial.number}: train_sb3 exit {proc.returncode} (see {tdir})")

    res = {}
    for tag, model, vn in (("final", tdir / "model.zip", tdir / "vecnormalize.pkl"),
                           ("best", tdir / "best" / "best_model.zip", tdir / "best" / "best_vecnormalize.pkl")):
        if model.exists():
            res[tag] = select_eval(model, vn, args.sel_set, tdir / f"sel_{tag}", args.workers)
            print(f"[hpo] trial {trial.number}: {tag} on {args.sel_set}: {res[tag]}", flush=True)
    tag = min(res, key=lambda k: res[k]["score"])
    for k, v in res.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("chosen", tag)
    trial.set_user_attr("val_min", best)
    return res[tag]["score"]


def main():
    ap = argparse.ArgumentParser(description="Optuna HPO of PPO around the robust policy")
    ap.add_argument("--reference", default="results/stage2_v28/config.json",
                    help="config.json of the run whose setup every trial reproduces")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--steps-per-trial", type=int, default=1_000_000)
    ap.add_argument("--eval-freq", type=int, default=250_000, help="total env steps between validations")
    ap.add_argument("--workers", type=int, default=8, help="parallel episodes in validation / selection")
    ap.add_argument("--sel-set", default="S2_sel_a5_M", help="objective set (results/eval_sets/<name>.json)")
    ap.add_argument("--val-set", default=None, help="override the reference's in-training validation set")
    ap.add_argument("--stall-min", type=float, default=120.0, help="kill a trial with no validation for this long")
    ap.add_argument("--no-baseline-first", dest="baseline_first", action="store_false",
                    help="do not enqueue the reference PPO settings as trial 0")
    ap.add_argument("--seed", type=int, default=0, help="sampler seed")
    ap.add_argument("--study-name", default="hpo_v26")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else ROOT / "results" / args.study_name
    outdir.mkdir(parents=True, exist_ok=True)
    args.outdir = str(outdir)
    cfg = json.load(open(ROOT / args.reference))
    base_flags, std0, base = flags_from_config(cfg), warm_start_std(cfg), baseline_params(cfg)

    storage = f"sqlite:///{(outdir / 'study.db').as_posix()}"
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True, n_startup_trials=6),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=args.eval_freq * 2 - 1))
    if args.baseline_first and not study.trials:
        study.enqueue_trial(base, skip_if_exists=True)
    print(f"[hpo] study '{args.study_name}' -> {storage}", flush=True)
    print(f"[hpo] reference {args.reference}; warm start {cfg['init_model']} (std {std0})", flush=True)
    t0 = time.time()
    study.optimize(lambda tr: objective(tr, args, base_flags, std0, base), n_trials=args.trials,
                   gc_after_trial=True, catch=(RuntimeError, subprocess.CalledProcessError))

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n[hpo] done in {(time.time() - t0) / 3600:.1f} h: {len(done)} complete / {len(study.trials)}")
    if done:
        bt = study.best_trial
        print(f"  best trial {bt.number}: score {bt.value:.4f} ({bt.user_attrs.get('chosen')})")
        for k, v in bt.params.items():
            print(f"  {k:16s}: {v}")
        json.dump({"best_trial": bt.number, "best_value": bt.value, "best_params": bt.params,
                   "best_attrs": bt.user_attrs, "n_trials": len(study.trials), "reference": args.reference,
                   "sel_set": args.sel_set},
                  open(outdir / "best.json", "w"), indent=2)
    rows = [{"trial": t.number, "state": t.state.name, "value": t.value, **t.params,
             **{f"{k}_{m}": v.get(m) for k, v in t.user_attrs.items() if isinstance(v, dict)
                for m in ("score", "d_med", "feasible", "fpa_med", "psi_med")}}
            for t in study.trials]
    pd.DataFrame(rows).to_csv(outdir / "trials.csv", index=False)
    print(f"[hpo] best.json, trials.csv -> {outdir}")


if __name__ == "__main__":
    main()
