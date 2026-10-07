"""Import SCvx (VISTA) Monte Carlo campaigns as RL evaluation sets + benchmark tables.

For each closed-loop campaign (montecarlo/<case>/<id>/postproc/) this writes

  results/eval_sets/SCVX_<case>.json   the campaign's exact sampled scenarios
                                       (sample_table.csv; row 0 = baseline/nominal)
  results/scvx_mc/<case>.csv           SCvx closed-loop results recomputed with the
                                       RL evaluator's metrics (metrics_table.csv)
  results/scvx_mc/<case>_summary.json  evaluate_policy.summarize() of that table

so RL policies (evaluate_policy --sets SCVX_<case>) and SCvx are scored on
identical scenarios with identical definitions: landing error as the
great-circle distance to the (dispersed) target, signed FPA/heading/altitude
errors, path-constraint peaks as ratios to the WB001 limits, and success at
each tolerance set of envs/terminal.py.

Caveat for comparisons: SCvx numbers are full closed-loop G&C (6-DOF attitude
loop and actuators in the loop); the RL policy is evaluated as 3-DOF guidance
with an ideal attitude loop.

    python -m reentry_rl.validation.scvx_benchmark
"""
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from reentry_rl.physics import constants as C
from reentry_rl.physics.geodesy import great_circle_distance_m
from reentry_rl.envs.scenario import NOMINAL, Scenario, save_set
from reentry_rl.envs.terminal import TOLERANCE_SETS, meets, wrap_deg
from reentry_rl.validation.evaluate_policy import summarize

ROOT = Path(__file__).resolve().parents[2]
MC_ROOT = Path.home() / "Documents" / "MSc thesis" / "Code" / "reentry_simulator" / "results" / "montecarlo" / "server"
CAMPAIGNS = {"gc_ic_tc": "gc_ic_tc/cb432d4cf184",      # 1000 runs + baseline
             "gc_models": "gc_models/4c4b106ad06f"}    # 500 runs + baseline

# VISTA sample column -> Scenario field
_MAP = {"ic_altitude_m": "ic_alt_m", "ic_lat_deg": "ic_lat_deg", "ic_lon_deg": "ic_lon_deg",
        "ic_velocity_mps": "ic_V_mps", "ic_fpa_deg": "ic_fpa_deg", "ic_heading_deg": "ic_psi_deg",
        "tc_altitude_m": "tc_alt_m", "tc_lat_deg": "tc_lat_deg", "tc_lon_deg": "tc_lon_deg",
        "tc_fpa_deg": "tc_fpa_deg", "tc_heading_deg": "tc_psi_deg",
        "mass_scale": "k_mass", "CD_scale": "k_CD", "CL_scale": "k_CL", "atm_rho_scale": "k_rho"}


def scenarios_from_samples(samples):
    out = []
    for _, row in samples.iterrows():
        vals = {f: float(row[c]) for c, f in _MAP.items() if c in row and np.isfinite(row[c])}
        out.append(Scenario(**{**NOMINAL.to_dict(), **vals}))
    return out


def scvx_metrics(metrics):
    """SCvx closed-loop rows -> the RL evaluator's per-case columns."""
    rows = []
    d2r = np.pi / 180.0
    for _, m in metrics.iterrows():
        tlat, tlon = m["tc_decl_latitude_deg"], m["tc_decl_longitude_deg"]
        d_km = great_circle_distance_m(m["terminal_latitude_deg"] * d2r, m["terminal_longitude_deg"] * d2r,
                                       tlat * d2r, tlon * d2r) / 1e3
        r = {"id": int(m["index"]), "outcome": "reached" if m["status"] == "completed" else "fail",
             "d_km": float(d_km), "dh_m": float(m["terminal_altitude_m"] - m["tc_decl_altitude_m"]),
             "dlat_deg": float(m["terminal_latitude_deg"] - tlat),
             "dlon_deg": float(wrap_deg(m["terminal_longitude_deg"] - tlon)),
             "dfpa_deg": float(m["terminal_fpa_deg"] - m["tc_decl_fpa_deg"]),
             "dpsi_deg": float(wrap_deg(m["terminal_heading_deg"] - m["tc_decl_heading_deg"])),
             "V_f_mps": float(m["terminal_velocity_mps"]),
             "J_ratio": float(m["heat_load_MJm2"] * 1e6 / C.J_REF_BANK_AOA),
             "max_Qdot": float(m["max_Qdot_Wm2"] / C.QDOT_MAX),
             "max_qbar": float(m["max_qbar_Pa"] / C.QBAR_MAX),
             "max_n": float(m["max_nLoad_g"] / C.N_MAX),
             "tf_s": float(m["time_of_flight_s"]), "alpha_std_deg": float("nan"),
             "terminal_aoa_deg": float(m["terminal_aoa_deg"]),
             "vista_pathOK": bool(m["pathOK"]), "vista_terminalOK": bool(m["terminalOK"])}
        r["max_ratio"] = max(r["max_Qdot"], r["max_qbar"], r["max_n"])
        r["feasible"] = bool(r["max_ratio"] <= 1.0)
        r["feasible_marg"] = bool(r["max_ratio"] <= 1.02)
        ok = r["outcome"] == "reached" and r["feasible"]
        for tol in TOLERANCE_SETS:
            r[f"success_{tol}"] = bool(ok and meets(r, tol))
        rows.append(r)
    return pd.DataFrame(rows)


def import_campaign(case, rel):
    post = MC_ROOT / rel / "postproc"
    samples = pd.read_csv(post / "sample_table.csv")
    metrics = pd.read_csv(post / "metrics_table.csv")
    scen = scenarios_from_samples(samples)
    labels = [{"scvx_index": int(i)} for i in samples["index"]]
    set_path = save_set(ROOT / "results" / "eval_sets" / f"SCVX_{case}.json", f"SCVX_{case}", scen,
                        meta={"purpose": "SCvx benchmark scenarios (exact VISTA draws)",
                              "source": f"reentry_simulator/results/montecarlo/server/{rel}",
                              "first_is_nominal": True}, labels=labels)
    df = scvx_metrics(metrics)
    out = ROOT / "results" / "scvx_mc"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{case}.csv", index=False)
    summ = {"set": f"SCVX_{case}", "method": "SCvx closed-loop G&C (VISTA)",
            "source": f"reentry_simulator/results/montecarlo/server/{rel}",
            "vista_success_rate": float(df["vista_terminalOK"].mean()),
            "vista_pathOK_rate": float(df["vista_pathOK"].mean()),
            "feasible_marg_rate": float(df["feasible_marg"].mean()), **summarize(df)}
    (out / f"{case}_summary.json").write_text(json.dumps(summ, indent=1))
    return set_path, summ


def main():
    for case, rel in CAMPAIGNS.items():
        p, s = import_campaign(case, rel)
        st = s["stats"]
        print(f"[{case}] n={s['n']} -> {p.name}")
        print(f"   SCvx: feasible {s['feasible_rate']:.3f} (<=1.02: {s['feasible_marg_rate']:.3f}) "
              f"success {json.dumps({k: round(v, 3) for k, v in s['success_rate'].items()})}")
        print(f"   d_km median {st['d_km']['median']:.3f} p95 {st['d_km']['p95']:.2f} | "
              f"|dfpa| median {st['abs_dfpa_deg']['median']:.3f} | |dpsi| median {st['abs_dpsi_deg']['median']:.3f} | "
              f"peak ratio median {st['max_ratio']['median']:.3f}")


if __name__ == "__main__":
    main()
