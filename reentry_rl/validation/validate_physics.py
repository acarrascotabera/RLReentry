"""Physics validation gate.

Prescribe the benchmark's optimal bank/AoA command history through the Python
3-DOF EOM, integrate from the benchmark IC, and compare the resulting trajectory
+ integrated heat load against the MATLAB benchmark node-for-node.

Run:  python -m reentry_rl.validation.validate_physics
"""
import numpy as np
from scipy.interpolate import interp1d

# Runnable both as a module and as a plain file (see smoke_train_stage1.py).
if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from reentry_rl.physics.constants import V_SCALE
from reentry_rl.physics.geodesy import r_nd_from_alt, altitude_m
from reentry_rl.physics.aero_wb001 import lift_drag_nd, heat_rate
from reentry_rl.physics.eom_3dof import rk4_step
from reentry_rl.validation.benchmark_scp import BENCHMARKS, load_trajectory, load_terminal_summary


def replay(folder, dt=0.2):
    """Integrate the prescribed commands; return (results dict, benchmark df, summary)."""
    df = load_trajectory(folder)
    summ = load_terminal_summary(folder)

    t = df["tAbs_s"].to_numpy()
    t0, tf = float(t[0]), float(t[-1])

    sigma_fn = interp1d(t, np.deg2rad(df["sigma_deg"].to_numpy()),
                        kind="linear", fill_value="extrapolate")
    alpha_fn = interp1d(t, df["alpha_deg"].to_numpy(),
                        kind="linear", fill_value="extrapolate")

    lat0 = np.deg2rad(df["lat_deg"].to_numpy()[0])
    state = np.array([
        r_nd_from_alt(df["alt_km"].to_numpy()[0] * 1e3, lat0),
        np.deg2rad(df["lon_deg"].to_numpy()[0]),
        lat0,
        df["V_mps"].to_numpy()[0] / V_SCALE,
        np.deg2rad(df["gamma_deg"].to_numpy()[0]),
        np.deg2rad(df["psi_deg"].to_numpy()[0]),
    ], dtype=float)

    # March with fixed-step RK4, logging every node for interpolation + heat load.
    ts = [t0]
    states = [state.copy()]
    rho0 = lift_drag_nd(state[0], state[2], state[3], float(alpha_fn(t0)))[2]
    qdots = [heat_rate(rho0, state[3] * V_SCALE)]

    tc = t0
    while tc < tf - 1e-9:
        h = min(dt, tf - tc)
        state = rk4_step(state, tc, h, lambda tt: float(sigma_fn(tt)),
                         lambda tt: float(alpha_fn(tt)))
        tc += h
        rho = lift_drag_nd(state[0], state[2], state[3], float(alpha_fn(tc)))[2]
        ts.append(tc)
        states.append(state.copy())
        qdots.append(heat_rate(rho, state[3] * V_SCALE))

    ts = np.array(ts)
    states = np.array(states)            # (N, 6)
    _trapz = getattr(np, "trapezoid", np.trapz)  # np.trapz deprecated in NumPy 2.0
    J_heat = float(_trapz(np.array(qdots), ts))

    # Convert integrated states to physical units and resample at benchmark times.
    r_nd, lon, lat, V_nd, gam, psi = states.T
    sim = {
        "alt_km": altitude_m(r_nd, lat) / 1e3,
        "lon_deg": np.rad2deg(lon),
        "lat_deg": np.rad2deg(lat),
        "V_mps": V_nd * V_SCALE,
        "gamma_deg": np.rad2deg(gam),
        "psi_deg": np.rad2deg(psi),
    }
    errs = {}
    for key in sim:
        sim_at_bench = interp1d(ts, sim[key], kind="linear",
                                fill_value="extrapolate")(t)
        errs[key] = np.abs(sim_at_bench - df[key].to_numpy())

    J_bench = summ.get("Integrated heat load [SI]", np.nan)
    results = {
        "errs": errs,
        "J_sim": J_heat,
        "J_bench": J_bench,
        "J_relpct": 100.0 * (J_heat - J_bench) / J_bench,
        "tf_sim": tf,
        "n_nodes": len(t),
    }
    return results, df, summ


def _fmt_row(name, e, unit):
    return f"  {name:10s} max={e.max():10.4g}   mean={e.mean():10.4g}   [{unit}]"


def main():
    for label, folder in BENCHMARKS.items():
        print("=" * 74)
        print(f"BENCHMARK: {label}   ({folder.name})")
        print("=" * 74)
        results, df, summ = replay(folder)
        units = {"alt_km": "km", "lon_deg": "deg", "lat_deg": "deg",
                 "V_mps": "m/s", "gamma_deg": "deg", "psi_deg": "deg"}
        print(f"  nodes={results['n_nodes']}  t_f={results['tf_sim']:.2f} s")
        print("  --- node-for-node state error (sim vs MATLAB) ---")
        for key, e in results["errs"].items():
            print(_fmt_row(key, e, units[key]))
        print("  --- integrated heat load ---")
        print(f"  J_sim   = {results['J_sim']:.6e} J/m^2")
        print(f"  J_bench = {results['J_bench']:.6e} J/m^2")
        print(f"  delta   = {results['J_relpct']:+.4f} %")
        print()


if __name__ == "__main__":
    main()
