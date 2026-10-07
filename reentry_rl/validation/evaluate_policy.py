"""Monte Carlo evaluation of a trained policy over fixed scenario sets.

Every policy, at every stage of the campaign, is scored on the same files in
results/eval_sets/ (see eval_sets.py), deterministically and at the exact
target-altitude crossing. Selection (training callbacks, HPO) uses only the
validation set; reported numbers come from the test sets.

    python -m reentry_rl.validation.evaluate_policy \
        --model results/results/stage2_polish_hpo/trial023_0.27km_300k.zip \
        --vn    results/results/stage2_polish_hpo/trial023_0.27km_300k_vecnormalize.pkl \
        --sets S0_nominal S1_sweep S3_test_ic_models --outdir results/baseline_eval/hpo_t23

Outputs per set: <outdir>/<set>.csv (one row per scenario) and
<outdir>/<set>_summary.json; plus <outdir>/S0_nominal_trajectory.csv.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from reentry_rl.physics import constants as C
from reentry_rl.envs.rewards import RewardWeights
from reentry_rl.envs.scenario import Scenario, load_set
from reentry_rl.envs.terminal import TOLERANCE_SETS, meets

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SET_DIR = REPO_ROOT / "results" / "eval_sets"
VECNORM_EPS = 1e-8          # VecNormalize defaults used in training
VECNORM_CLIP = 10.0

TRAJ_FIELDS = ["t_s", "alt_km", "lon_deg", "lat_deg", "V_mps", "gamma_deg", "psi_deg",
               "sigma_deg", "alpha_deg", "Qdot_Wm2", "qbar_Pa", "n_g", "J_heat"]


def _env_classes():
    from reentry_rl.envs.stage1_bank import Stage1BankEnv
    from reentry_rl.envs.stage2_bank_aoa import Stage2BankAoaEnv
    return {"stage1": Stage1BankEnv, "stage2": Stage2BankAoaEnv}


def obs_version_for(stage, obs_dim):
    """Infer the observation layout a model was trained with from its input width."""
    env_cls = _env_classes()[stage]
    for ver in ("v13", "v15", "v20"):
        if env_cls(obs_version=ver).observation_space.shape[0] == obs_dim:
            return ver
    raise ValueError(f"no {stage} observation layout has {obs_dim} features")


def load_policy(model_path, vn_path):
    """-> (PPO model on cpu, obs RunningMeanStd)."""
    import pickle
    from stable_baselines3 import PPO
    model = PPO.load(str(model_path), device="cpu")
    with open(vn_path, "rb") as f:
        vn = pickle.load(f)
    return model, vn.obs_rms


def resolve_set(name_or_path):
    p = Path(name_or_path)
    if p.suffix != ".json":
        p = EVAL_SET_DIR / f"{name_or_path}.json"
    return p


class PolicyRunner:
    """Deterministic rollouts of one policy, one scenario at a time.

    Observations are normalized exactly like VecNormalize (eval mode) and the
    action is the Gaussian mean, clipped to the action box (= SB3
    predict(deterministic=True)), without the VecEnv overhead.
    """

    def __init__(self, model, obs_rms, stage="stage2", weights=None, env_kwargs=None):
        import torch as th
        self._th = th
        self.model = model
        self.mean = np.asarray(obs_rms.mean, dtype=np.float64)
        self.std = np.sqrt(np.asarray(obs_rms.var, dtype=np.float64) + VECNORM_EPS)
        kw = dict(env_kwargs or {})
        kw.setdefault("obs_version", obs_version_for(stage, model.observation_space.shape[0]))
        self.env = _env_classes()[stage](weights=weights or RewardWeights(), **kw)
        self.j_ref = self.env.J_ref

    def act(self, obs):
        on = np.clip((obs - self.mean) / self.std, -VECNORM_CLIP, VECNORM_CLIP)
        with self._th.no_grad():
            a = self.model.policy._predict(
                self._th.as_tensor(on[None], dtype=self._th.float32), deterministic=True)
        return np.clip(a.numpy()[0], -1.0, 1.0)

    def run(self, scenario: Scenario, record=False):
        env = self.env
        obs, info = env.reset(options={"scenario": scenario})
        ret, traj = 0.0, []
        mQ = mq = mn = 0.0
        sig, alp, sdot = [], [], []
        while True:
            a = self.act(obs)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            mQ = max(mQ, info["Qdot_Wm2"] / C.QDOT_MAX)
            mq = max(mq, info["qbar_Pa"] / C.QBAR_MAX)
            mn = max(mn, info["n_g"] / C.N_MAX)
            sig.append(info["sigma_deg"])
            alp.append(info["alpha_deg"])
            sdot.append(float(a[0]) * np.rad2deg(C.SIGMA_DOT_MAX))
            if record:
                traj.append({k: info[k] for k in TRAJ_FIELDS})
            if term or trunc:
                break
        sig, alp = np.asarray(sig), np.asarray(alp)
        max_ratio = max(mQ, mq, mn)
        m = {"outcome": info["outcome"], "steps": env.steps, "tf_s": info["t_s"],
             "episode_return": ret,
             "d_km": info["d_km"], "dh_m": info["dh_m"], "dlat_deg": info["dlat_deg"],
             "dlon_deg": info["dlon_deg"], "dfpa_deg": info["dfpa_deg"],
             "dpsi_deg": info["dpsi_deg"], "V_f_mps": info["V_f_mps"],
             "J_ratio": info["J_heat"] / self.j_ref,
             "max_Qdot": mQ, "max_qbar": mq, "max_n": mn, "max_ratio": max_ratio,
             "feasible": bool(max_ratio <= 1.0), "feasible_marg": bool(max_ratio <= 1.02),
             "reversals": int((np.diff(np.sign(sig)) != 0).sum()) if sig.size > 1 else 0,
             "bank_rate_rms_dps": float(np.sqrt(np.mean(np.square(sdot)))),
             "alpha_mean_deg": float(alp.mean()), "alpha_std_deg": float(alp.std()),
             "alpha_min_deg": float(alp.min()), "alpha_max_deg": float(alp.max()),
             "alpha_at_max_frac": float(np.mean(alp >= C.ALPHA_MAX - 1e-6))}
        ok = (m["outcome"] == "reached") and m["feasible"]
        for tol in TOLERANCE_SETS:
            m[f"success_{tol}"] = bool(ok and meets(m, tol))
        return (m, traj) if record else m

    def run_many(self, scenarios, labels=None):
        rows = []
        for i, s in enumerate(scenarios):
            row = {"id": i, **(labels[i] if labels else {}), **self.run(s), **s.to_dict()}
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parallel evaluation (one PolicyRunner per worker process)
# ---------------------------------------------------------------------------
_W = {}


def _init_worker(model_path, vn_path, stage, weights_dict, env_kwargs):
    import torch
    torch.set_num_threads(1)
    model, rms = load_policy(model_path, vn_path)
    w = RewardWeights(**weights_dict) if weights_dict else RewardWeights()
    _W["runner"] = PolicyRunner(model, rms, stage, w, env_kwargs)


def _work(args):
    i, sdict, label = args
    s = Scenario.from_dict(sdict)
    return {"id": i, **label, **_W["runner"].run(s), **s.to_dict()}


def evaluate_scenarios(model_path, vn_path, scenarios, stage="stage2", labels=None,
                       workers=None, weights=None, env_kwargs=None):
    labels = labels or [{} for _ in scenarios]
    wd = weights.__dict__ if weights is not None else None
    jobs = [(i, s.to_dict(), labels[i]) for i, s in enumerate(scenarios)]
    workers = min(workers or os.cpu_count() or 1, len(jobs))
    if workers <= 1:
        _init_worker(str(model_path), str(vn_path), stage, wd, env_kwargs)
        rows = [_work(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker,
                      initargs=(str(model_path), str(vn_path), stage, wd, env_kwargs)) as pool:
            rows = pool.map(_work, jobs, chunksize=1)
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def iqm(x):
    """Interquartile mean (mean of the values between the 25th and 75th percentiles)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:max(hi, lo + 1)].mean())


def bootstrap_ci(x, stat=iqm, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    b = [stat(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)]
    return [float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))]


def _stats(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)),
            "p95": float(np.quantile(x, 0.95)), "p99": float(np.quantile(x, 0.99)),
            "max": float(np.max(x)), "iqm": iqm(x), "iqm_ci95": bootstrap_ci(x)}


def summarize(df):
    out = {"n": int(len(df)),
           "outcomes": {k: int(v) for k, v in df["outcome"].value_counts().items()},
           "feasible_rate": float(df["feasible"].mean()),
           "success_rate": {tol: float(df[f"success_{tol}"].mean()) for tol in TOLERANCE_SETS},
           "stats": {}}
    cols = {"d_km": df["d_km"], "abs_dfpa_deg": df["dfpa_deg"].abs(),
            "abs_dpsi_deg": df["dpsi_deg"].abs(), "abs_dlat_deg": df["dlat_deg"].abs(),
            "abs_dlon_deg": df["dlon_deg"].abs(), "J_ratio": df["J_ratio"],
            "max_ratio": df["max_ratio"], "alpha_std_deg": df["alpha_std_deg"]}
    for k, v in cols.items():
        out["stats"][k] = _stats(v)
    return out


# Selection metric scales: 1 km, 1 deg, 5 deg (the VISTA MC tolerance scale).
_FULL_SCALES = {"d_km": 1.0, "dfpa_deg": 1.0, "dpsi_deg": 5.0}


def case_error(row, mode="position"):
    if mode == "position":
        return float(row["d_km"])
    return float(np.sqrt(sum((row[k] / s) ** 2 for k, s in _FULL_SCALES.items())))


def selection_score(df, mode="position", q=0.9):
    """Feasibility-gated terminal error used for model selection (lower is better).

    Per case: the terminal error (position: d_km; full: ||(d/1 km, dgamma/1 deg,
    dpsi/5 deg)||), +1000 if a path-constraint ratio exceeds 1.02, +10000 if the
    episode did not reach the handover. Aggregate: the q-quantile over cases
    (a single case returns its value, so the nominal-only set reproduces the
    legacy HPO objective)."""
    v = []
    for _, row in df.iterrows():
        e = case_error(row, mode)
        if row["outcome"] != "reached":
            e += 1e4
        elif row["max_ratio"] > 1.02:
            e += 1e3
        v.append(e)
    return float(np.quantile(v, q)) if len(v) > 1 else float(v[0])


# ---------------------------------------------------------------------------
def plot_sweep(df, outpath):
    """Small multiples of the one-at-a-time sensitivity sweep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    params = list(dict.fromkeys(df["param"]))
    ncol = 5
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow), sharey=True, squeeze=False)
    for ax, p in zip(axes.flat, params):
        sub = df[df["param"] == p].sort_values("multiplier")
        x = np.concatenate([sub["multiplier"].to_numpy(), [0.0]])
        y = np.concatenate([sub["d_km"].to_numpy(), [np.nan]])
        order = np.argsort(x)
        ax.semilogy(x[order], y[order], "o-", ms=3, color="C0")
        bad = sub[(sub["outcome"] != "reached") | (~sub["feasible"])]
        ax.semilogy(bad["multiplier"], bad["d_km"], "x", color="C3", ms=6)
        ax.set_title(p, fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    for ax in axes.flat[len(params):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("landing error [km]")
    for ax in axes[-1, :]:
        ax.set_xlabel("dispersion [sigma]")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Monte Carlo evaluation of a trained policy")
    ap.add_argument("--model", required=True)
    ap.add_argument("--vn", required=True, help="VecNormalize .pkl saved with the model")
    ap.add_argument("--stage", default="stage2", choices=["stage1", "stage2"])
    ap.add_argument("--sets", nargs="+", default=["S0_nominal"],
                    help="eval-set names (results/eval_sets/<name>.json) or paths")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--legacy-terminal", action="store_true",
                    help="measure at the first step below 25 km (pre-2026-10 behaviour)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env_kwargs = {"exact_terminal": not args.legacy_terminal}
    for set_ref in args.sets:
        path = resolve_set(set_ref)
        name, scen, labels = load_set(path)
        t0 = time.time()
        df = evaluate_scenarios(args.model, args.vn, scen, args.stage, labels,
                                args.workers, env_kwargs=env_kwargs)
        df.to_csv(outdir / f"{name}.csv", index=False)
        summ = {"set": name, "set_file": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
                "model": str(args.model), "vn": str(args.vn), "stage": args.stage,
                "exact_terminal": not args.legacy_terminal,
                "wall_s": round(time.time() - t0, 1), **summarize(df)}
        (outdir / f"{name}_summary.json").write_text(json.dumps(summ, indent=1))
        if "param" in df.columns:
            plot_sweep(df, outdir / f"{name}.pdf")
        s = summ["stats"]
        print(f"[{name}] n={summ['n']} {summ['outcomes']} feasible={summ['feasible_rate']:.2f} "
              f"success(1km)={summ['success_rate']['pos_1km']:.2f} "
              f"success(0.1%)={summ['success_rate']['rel_0p1pct']:.2f} | "
              f"d_km median={s['d_km']['median']:.3f} p95={s['d_km']['p95']:.3f} | "
              f"|dfpa| median={s['abs_dfpa_deg']['median']:.2f} | |dpsi| median={s['abs_dpsi_deg']['median']:.2f} "
              f"({summ['wall_s']} s)")
        if name == "S0_nominal":
            model, rms = load_policy(args.model, args.vn)
            runner = PolicyRunner(model, rms, args.stage, env_kwargs=env_kwargs)
            _m, traj = runner.run(scen[0], record=True)
            pd.DataFrame(traj).to_csv(outdir / "S0_nominal_trajectory.csv", index=False)


if __name__ == "__main__":
    main()
