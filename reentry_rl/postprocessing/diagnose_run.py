"""Diagnosis for a training run: checkpoint rollouts + learning curves.

Rolls out selected checkpoints deterministically (default: the best model and
the latest checkpoint), writing eval CSV/summary pairs under <run>/diag/ and
rendering the trajectory profile figure (plots.plot_trajectory) for each. Also
builds a learning-curve figure from evaluations.npz plus, when the captured
training stdout is available (--train-log), the dense rollout/stage metrics
(ep_rew_mean, d_km, constraint ratios) parsed from its logged tables.

Safe to run while training is still going — rollouts are read-only.

    python -m reentry_rl.postprocessing.diagnose_run --run results/stage2_v9 \
        --train-log path/to/captured_stdout.txt

--watch keeps the diag/ figures live while training runs: last_traj + learning
curves refresh at every new checkpoint (250k steps), best_traj whenever
EvalCallback saves a new best model. Exits when the train log reports
"training done" (or the final model.zip appears), after one last refresh.
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from reentry_rl.envs.rewards import RewardWeights
from reentry_rl.training.common import ENV_CLASSES, J_REF, build_vecenv, evaluate_to_csv
from reentry_rl.postprocessing.plots import plot_trajectory


def load_run(run_dir):
    cfg = json.load(open(run_dir / "config.json"))
    weights = RewardWeights()
    for k, v in cfg.get("reward_weights", {}).items():
        if k in RewardWeights.__dataclass_fields__:
            setattr(weights, k, v)
    return cfg, weights


def checkpoints(run_dir):
    """Sorted [(steps, model_zip, vecnormalize_pkl)] from <run>/ckpt/."""
    out = []
    for p in (run_dir / "ckpt").glob("ppo_*_steps.zip"):
        steps = int(re.search(r"_(\d+)_steps", p.name).group(1))
        pkl = p.with_name(p.name.replace("ppo_", "ppo_vecnormalize_")).with_suffix(".pkl")
        if pkl.exists():
            out.append((steps, p, pkl))
    return sorted(out)


def best_model_steps(run_dir):
    """Timestep at which EvalCallback saved best_model.zip (argmax eval reward)."""
    ev = np.load(run_dir / "evaluations.npz")
    mean_r = ev["results"].mean(axis=1)
    i = int(np.argmax(mean_r))
    return int(ev["timesteps"][i]), float(mean_r[i])


def obs_rms_from_pkl(pkl, env_cls, weights, n_substeps):
    venv = build_vecenv(env_cls, 1, weights, n_substeps, 0, subproc=False)
    vn = VecNormalize.load(str(pkl), venv.venv)
    venv.close()
    return vn.obs_rms


def parse_train_log(log_path):
    """Parse SB3 stdout tables -> list of {key: float} records, one per table
    block (delimited by its total_timesteps row). nan-safe."""
    recs, cur = [], {}
    pat = re.compile(r"\|\s+([\w/]+)\s+\|\s+([-\w.+e]+)\s+\|")
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        try:
            cur[key] = float(val)
        except ValueError:
            continue
        if key == "total_timesteps":
            recs.append(cur)
            cur = {}
    return recs


def series(recs, key):
    """(timesteps, values) for records that contain `key`."""
    pts = [(r["total_timesteps"], r[key]) for r in recs if key in r and "total_timesteps" in r]
    if not pts:
        return np.array([]), np.array([])
    t, v = zip(*sorted(pts))
    return np.asarray(t), np.asarray(v)


def plot_learning_curves(run_dir, recs, out_prefix, title):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=13)

    # 1 — reward curves
    a = ax[0, 0]
    t, v = series(recs, "ep_rew_mean")
    if t.size:
        a.plot(t / 1e6, v, color="C0", lw=0.9, label="train ep_rew_mean")
    ev = np.load(run_dir / "evaluations.npz")
    a.plot(ev["timesteps"] / 1e6, ev["results"].mean(axis=1), color="C1", lw=1.0,
           alpha=0.8, label="eval (deterministic)")
    a.set_xlabel("steps (M)"); a.set_ylabel("episode reward")
    a.set_title("Reward"); a.grid(True, alpha=.3); a.legend(fontsize=8)

    # 2 — landing error (log), non-landing evals marked
    a = ax[0, 1]
    t, d = series(recs, "d_km")
    if t.size:
        ok = np.isfinite(d)
        a.semilogy(t[ok] / 1e6, d[ok], ".-", color="C0", lw=0.8, ms=3, label="d_km (landed)")
        if (~ok).any():
            ytop = np.nanmax(d) if np.isfinite(d).any() else 1e4
            a.semilogy(t[~ok] / 1e6, np.full((~ok).sum(), ytop * 1.5), "x", color="red",
                       ms=5, label="no landing (fail/timeout)")
    a.axhline(1.0, ls="--", color="green", alpha=.6, label="1 km success")
    a.set_xlabel("steps (M)"); a.set_ylabel("landing error (km)")
    a.set_title("Terminal position error"); a.grid(True, alpha=.3, which="both")
    a.legend(fontsize=8)

    # 3 — episode length (train) + eval episode length
    a = ax[1, 0]
    t, v = series(recs, "ep_len_mean")
    if t.size:
        a.plot(t / 1e6, v, color="C0", lw=0.9, label="train ep_len_mean")
    a.plot(ev["timesteps"] / 1e6, ev["ep_lengths"].mean(axis=1), color="C1", lw=1.0,
           alpha=0.8, label="eval")
    a.set_xlabel("steps (M)"); a.set_ylabel("episode length (steps)")
    a.set_title("Episode length"); a.grid(True, alpha=.3); a.legend(fontsize=8)

    # 4 — peak path-constraint ratios (monitoring only; not in the reward)
    a = ax[1, 1]
    for key, c in (("max_Qdot_ratio", "C0"), ("max_n_ratio", "C1"), ("max_qbar_ratio", "C2")):
        t, v = series(recs, key)
        if t.size:
            a.plot(t / 1e6, v, color=c, lw=0.9, label=key)
    a.axhline(1.0, ls="--", color="red", alpha=.6, label="limit")
    a.set_xlabel("steps (M)"); a.set_ylabel("peak ratio (episode)")
    a.set_title("Path-constraint peaks (unrewarded)"); a.grid(True, alpha=.3)
    a.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def rollout_and_plot(tag, model_zip, pkl, steps, run_dir, cfg, weights):
    stage = cfg["stage"]
    env_cls, j_ref = ENV_CLASSES[stage], J_REF[stage]
    n_substeps = cfg.get("n_substeps", 3)
    outdir = run_dir / "diag" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(model_zip), device="cpu")
    obs_rms = obs_rms_from_pkl(pkl, env_cls, weights, n_substeps)
    summary, last = evaluate_to_csv(model, obs_rms, env_cls, weights, n_substeps,
                                    j_ref, outdir, stage=stage)
    title = f"{run_dir.name} — {tag} ({steps/1e6:.2f}M steps)"
    miss = plot_trajectory(outdir / "eval_trajectory.csv", run_dir / "diag" / f"{tag}_traj",
                           title=title)
    print(f"\n[{tag}] {model_zip.name} @ {steps:,} steps  ->  miss {miss:.1f} km")
    print(summary)
    return last


def refresh_best(run_dir, cfg, weights, cks):
    """Rollout+plot the current best_model.zip. Returns its saved-at timestep."""
    best_zip = run_dir / "best" / "best_model.zip"
    if not (best_zip.exists() and cks):
        return None
    bsteps, brew = best_model_steps(run_dir)
    _, _, near_pkl = min(cks, key=lambda c: abs(c[0] - bsteps))
    print(f"best model saved @ {bsteps:,} steps (eval reward {brew:+.1f}); "
          f"obs stats from {near_pkl.name}")
    rollout_and_plot("best", best_zip, near_pkl, bsteps, run_dir, cfg, weights)
    return bsteps


def refresh_last_and_curves(run_dir, cfg, weights, cks, train_log):
    lsteps, lzip, lpkl = cks[-1]
    rollout_and_plot("last", lzip, lpkl, lsteps, run_dir, cfg, weights)
    recs = parse_train_log(train_log) if train_log else []
    plot_learning_curves(run_dir, recs, run_dir / "diag" / "learning_curves",
                         f"{run_dir.name} — learning curves ({lsteps/1e6:.2f}M steps)")
    if train_log and Path(train_log).resolve() != (run_dir / "train_log.txt").resolve():
        shutil.copyfile(train_log, run_dir / "train_log.txt")   # keep a copy with the run
    return lsteps


def training_finished(run_dir, train_log):
    if (run_dir / "model.zip").exists():
        return True
    if train_log and Path(train_log).exists():
        try:
            return "training done" in Path(train_log).read_text(encoding="utf-8",
                                                                 errors="replace")
        except OSError:
            pass
    return False


def watch(run_dir, cfg, weights, train_log, poll_s):
    """Refresh last+curves on every new checkpoint and best on every new
    best_model.zip until training finishes (then one final refresh)."""
    done_steps, best_mtime = 0, 0.0
    while True:
        finished = training_finished(run_dir, train_log)
        try:
            cks = checkpoints(run_dir)
            if cks and cks[-1][0] > done_steps:
                done_steps = refresh_last_and_curves(run_dir, cfg, weights, cks, train_log)
            bz = run_dir / "best" / "best_model.zip"
            if bz.exists() and cks and bz.stat().st_mtime > best_mtime:
                time.sleep(3.0)                      # let EvalCallback finish writing
                best_mtime = bz.stat().st_mtime
                refresh_best(run_dir, cfg, weights, cks)
        except Exception as e:                       # file mid-write etc. -> retry next poll
            print(f"[watch] refresh failed ({e!r}); retrying next poll")
        if finished:
            print("[watch] training finished — final refresh done")
            return
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser(description="Run diagnosis: checkpoint rollouts + learning curves")
    ap.add_argument("--run", required=True, help="run directory (with config.json, ckpt/, best/)")
    ap.add_argument("--train-log", default=None, help="captured training stdout (for dense metrics)")
    ap.add_argument("--watch", action="store_true",
                    help="keep refreshing plots until training finishes")
    ap.add_argument("--poll", type=int, default=60, help="watch poll interval [s]")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg, weights = load_run(run_dir)

    if args.watch:
        watch(run_dir, cfg, weights, args.train_log, args.poll)
        return

    cks = checkpoints(run_dir)
    if not cks:
        raise SystemExit(f"no checkpoints in {run_dir/'ckpt'}")
    refresh_best(run_dir, cfg, weights, cks)
    refresh_last_and_curves(run_dir, cfg, weights, cks, args.train_log)
    print(f"\nplots -> {run_dir/'diag'}")


if __name__ == "__main__":
    main()
