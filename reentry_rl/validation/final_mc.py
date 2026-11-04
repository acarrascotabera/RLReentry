"""Final Monte Carlo campaign (paper): RL policy vs FNPEG baseline on one large dispersion
set, under the as-designed physics and under the higher-fidelity disturbance stack.

Dispersion set S6_mc_x0.1_a5 (2000 cases, results/eval_sets/, written once):
  entry state, target and mass / density bias  : VISTA magnitudes x 0.1 (the realistic,
                                                 Apollo-class level; envs/scenario.py)
  C_L, C_D                                     : independent, 1-sigma 5 % (clip +-15 %), the
                                                 project's aerodynamic uncertainty cap
Conditions (validation/stability.py Perturbation):
  ideal   the physics both guidance laws were designed in (no navigation / actuator error)
  hifi    HIFI: navigation drift at spec, slower attitude loop (delay, lag, 20 deg/s,
          5 deg/s^2), GRAM-like correlated density, winds
Both guidance laws fly the same env with common random numbers (case i has the same
scenario and disturbance draws for both).

Statistics per run: success / feasibility rates with exact (Clopper-Pearson) 95 % intervals;
terminal-error and path-constraint percentiles (50, 90, 95, 99, 99.73) with bootstrap 95 %
intervals; heat-rate excess attributable to the guidance (peak above max(1, floor), the
floor = lift-up peak under the same perturbation).

    python -m reentry_rl.validation.final_mc --guidance rl fnpeg --conditions ideal hifi
"""
import argparse
import json
import time
from pathlib import Path

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from dataclasses import replace

from reentry_rl.envs.scenario import sample_scenarios, save_set, load_set
from reentry_rl.envs.terminal import TOLERANCE_SETS
from reentry_rl.validation import evaluate_policy as ep
from reentry_rl.validation import stability as st

REPO_ROOT = Path(__file__).resolve().parents[2]
SET_NAME = "S6_mc_x0.1_a5"
OUT = REPO_ROOT / "results" / "final_mc"
CONDITIONS = {"ideal": st.P0, "hifi": st.HIFI}
PCTS = (50, 90, 95, 99, 99.73)


def build_set(n=2000, seed=2040, aero_sigma=0.05, aero_clip=0.15):
    path = ep.EVAL_SET_DIR / f"{SET_NAME}.json"
    if path.exists():
        return path
    scen = sample_scenarios(n, ("ic", "tc", "models"), 0.1, seed)
    rng = np.random.default_rng(seed + 1)
    a = np.clip(rng.normal(1.0, aero_sigma, size=(n, 2)), 1.0 - aero_clip, 1.0 + aero_clip)
    scen = [replace(s, k_CL=float(a[i, 0]), k_CD=float(a[i, 1])) for i, s in enumerate(scen)]
    save_set(path, SET_NAME, scen, labels=[{"case": i} for i in range(n)],
             meta={"purpose": "final Monte Carlo (reporting only)", "groups": ["ic", "tc", "models"],
                   "sigma_scale": 0.1, "seed": seed, "aero_sigma": aero_sigma, "aero_clip": aero_clip,
                   "note": "ic/tc/mass/k_rho at 0.1x VISTA; C_L, C_D 1-sigma 5 %"})
    return path


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def _pct_ci(x, q, n_boot=1000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    b = [np.percentile(rng.choice(x, x.size), q) for _ in range(n_boot)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def summarize(df):
    n = len(df)
    reached = df["outcome"] == "reached"
    floor = np.maximum(1.0, df["qdot_lb"].fillna(0.0)) + 1e-3
    attributable_ok = (df["max_Qdot"] <= floor) & (df["max_qbar"] <= 1.0) & (df["max_n"] <= 1.0)
    rates = {"reached": reached, "feasible": df["feasible"].astype(bool),
             "feasible_vs_floor": reached & attributable_ok}
    for tol in TOLERANCE_SETS:
        rates[f"success_{tol}"] = df[f"success_{tol}"].astype(bool)
    out = {"n": n, "rates": {}, "percentiles": {}}
    for k, v in rates.items():
        c = int(v.sum())
        out["rates"][k] = {"rate": c / n, "ci95": clopper_pearson(c, n)}
    cols = {"d_km": df["d_km"], "abs_dfpa_deg": df["dfpa_deg"].abs(), "abs_dpsi_deg": df["dpsi_deg"].abs(),
            "abs_dh_m": df["dh_m"].abs(), "max_Qdot": df["max_Qdot"], "max_qbar": df["max_qbar"],
            "max_n": df["max_n"], "J_ratio": df["J_ratio"],
            "qdot_excess_vs_floor": (df["max_Qdot"] - np.maximum(1.0, df["qdot_lb"].fillna(0.0))).clip(lower=0.0),
            "reversals": df["reversals"], "bank_rate_rms_dps": df["bank_rate_rms_dps"]}
    for k, v in cols.items():
        v = v.to_numpy(dtype=float)
        out["percentiles"][k] = {f"p{q:g}": {"value": float(np.percentile(v, q)), "ci95": _pct_ci(v, q)}
                                 for q in PCTS}
        out["percentiles"][k]["max"] = float(v.max())
        out["percentiles"][k]["mean"] = float(v.mean())
    return out


def run(guidance, condition, n, workers, model=None, vn=None):
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{guidance}_{condition}"
    f = OUT / f"{name}.csv"
    if f.exists():
        return f
    _n, scen, _l = load_set(build_set())
    scen = scen[:n]
    labels = [{"case": i} for i in range(len(scen))]
    model = model or st.MODEL
    vn = vn or st.VN
    env_kwargs = {**ep.env_kwargs_for(st.MODEL), "exact_terminal": True, "rate_saturation": True}
    pert = CONDITIONS[condition]
    t0 = time.time()
    pool = st.Pool(workers, env_kwargs, model="fnpeg" if guidance == "fnpeg" else model, vn=vn)
    try:
        rows = [r for r, _ in pool.map(st._jobs(scen, labels, pert))]
    finally:
        pool.close()
    df = pd.DataFrame(rows).sort_values("id")
    df.to_csv(f, index=False)
    s = summarize(df)
    s.update({"guidance": guidance, "condition": condition, "perturbation": pert.to_dict(), "set": SET_NAME,
              "model": "fnpeg" if guidance == "fnpeg" else str(model), "env_kwargs": env_kwargs,
              "wall_s": time.time() - t0})
    (OUT / f"{name}_summary.json").write_text(json.dumps(s, indent=1))
    st._log(f"final_mc/{name}: d p50 {s['percentiles']['d_km']['p50']['value']:.3f} km, "
            f"p99.73 {s['percentiles']['d_km']['p99.73']['value']:.2f} km, feasible "
            f"{s['rates']['feasible']['rate']:.3f}, reached {s['rates']['reached']['rate']:.3f} "
            f"({time.time() - t0:.0f} s)")
    return f


SCALE_SETS = tuple(f"S5_scale_x{s:g}" for s in (0.1, 0.25, 0.5, 0.75, 1.0, 1.25))


def run_scale(guidance, set_name, workers):
    """One S5 robustness-vs-scale set (300 cases, as-designed physics) through the same
    harness for both guidance laws -> results/final_mc/scale/<guidance>_<set>.csv."""
    d = OUT / "scale"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{guidance}_{set_name}.csv"
    if f.exists():
        return f
    _n, scen, _l = load_set(ep.resolve_set(set_name))
    labels = [{"case": i} for i in range(len(scen))]
    env_kwargs = {**ep.env_kwargs_for(st.MODEL), "exact_terminal": True, "rate_saturation": True}
    t0 = time.time()
    pool = st.Pool(workers, env_kwargs, model="fnpeg" if guidance == "fnpeg" else st.MODEL, vn=st.VN)
    try:
        rows = [r for r, _ in pool.map(st._jobs(scen, labels, st.P0))]
    finally:
        pool.close()
    df = pd.DataFrame(rows).sort_values("id")
    df.to_csv(f, index=False)
    s = summarize(df)
    s.update({"guidance": guidance, "set": set_name, "wall_s": time.time() - t0})
    (d / f"{guidance}_{set_name}_summary.json").write_text(json.dumps(s, indent=1))
    st._log(f"final_mc/scale/{guidance}_{set_name}: d p50 {s['percentiles']['d_km']['p50']['value']:.3f} km, "
            f"p95 {s['percentiles']['d_km']['p95']['value']:.2f}, feasible {s['rates']['feasible']['rate']:.3f}")
    return f


def tables():
    """Markdown comparison of every finished run (results/final_mc/final_mc_tables.md)."""
    runs = sorted(OUT.glob("*_summary.json"))
    if not runs:
        return None
    S = {p.name[:-len("_summary.json")]: json.loads(p.read_text()) for p in runs}
    lines = ["# Final Monte Carlo", "", f"Set {SET_NAME}; 95 % intervals: Clopper-Pearson (rates), "
             "bootstrap (percentiles).", ""]
    names = list(S)
    lines.append("| | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))

    def rate(k):
        return [f"{S[n]['rates'][k]['rate'] * 100:.1f} % [{S[n]['rates'][k]['ci95'][0] * 100:.1f}, "
                f"{S[n]['rates'][k]['ci95'][1] * 100:.1f}]" for n in names]

    def pct(k, q, fmt):
        return [fmt.format(S[n]["percentiles"][k][q]["value"]) for n in names]

    rows = [("n", [str(S[n]["n"]) for n in names]),
            ("reached handover", rate("reached")),
            ("strictly feasible", rate("feasible")),
            ("feasible vs heat-rate floor", rate("feasible_vs_floor"))]
    rows += [(f"success {t}", rate(f"success_{t}")) for t in TOLERANCE_SETS]
    for k, lab, fmt in (("d_km", "miss [km]", "{:.3f}"), ("abs_dfpa_deg", "abs FPA error [deg]", "{:.3f}"),
                        ("abs_dpsi_deg", "abs heading error [deg]", "{:.3f}"),
                        ("max_Qdot", "peak heat rate / limit", "{:.3f}"),
                        ("qdot_excess_vs_floor", "heat-rate excess over floor", "{:.4f}"),
                        ("max_n", "peak load factor / limit", "{:.3f}"),
                        ("max_qbar", "peak dyn. pressure / limit", "{:.3f}"),
                        ("J_ratio", "heat load / SCvx optimum", "{:.3f}")):
        for q in ("p50", "p95", "p99.73"):
            rows.append((f"{lab} {q}", pct(k, q, fmt)))
        rows.append((f"{lab} max", [fmt.format(S[n]["percentiles"][k]["max"]) for n in names]))
    for lab, vals in rows:
        lines.append(f"| {lab} | " + " | ".join(vals) + " |")
    p = OUT / "final_mc_tables.md"
    p.write_text("\n".join(lines) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser(description="Final Monte Carlo: RL vs FNPEG")
    ap.add_argument("--guidance", nargs="+", default=["rl", "fnpeg"], choices=("rl", "fnpeg"))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scale-sets", action="store_true",
                    help="run the S5 robustness-vs-scale sets instead of the final MC set")
    args = ap.parse_args()
    if args.scale_sets:
        for s in SCALE_SETS:
            for g in args.guidance:
                run_scale(g, s, args.workers)
        st._log("SCALE SETS DONE")
        return
    build_set()
    for cond in args.conditions:
        for g in args.guidance:
            run(g, cond, args.n, args.workers)
            tables()
    st._log("FINAL MC DONE")


if __name__ == "__main__":
    main()
