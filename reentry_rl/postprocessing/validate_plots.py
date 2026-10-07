"""Environment-validation overlay plots.

Drives OUR 3-DOF environment/physics core with the SCvx-generated command
history (bank + AoA from the exported MATLAB benchmark) and overlays the
resulting trajectory on the SCvx-generated trajectory. This is the *visual*
counterpart of the numerical physics gate (`validation/validate_physics.py`):
if the two curves lie on top of each other, our simulator reproduces the
reference dynamics under identical controls.

Run:
    python -m reentry_rl.postprocessing.validate_plots
Artifacts: results/validation_plots/env_validation_<mode>.{png,svg}
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from reentry_rl.physics import constants as C
from reentry_rl.physics.geodesy import r_nd_from_alt, altitude_m, great_circle_distance_m
from reentry_rl.physics.aero_wb001 import lift_drag_nd, path_quantities
from reentry_rl.physics.eom_3dof import rk4_step
from reentry_rl.validation.benchmark_scp import BENCHMARKS, load_trajectory, load_terminal_summary

_DEG = np.pi / 180.0


def replay_full(folder, dt=0.2):
    """Integrate the SCvx commands through OUR core; return dense sim arrays + the
    SCvx benchmark DataFrame + the SCvx terminal summary."""
    df = load_trajectory(folder)
    summ = load_terminal_summary(folder)
    t = df["tAbs_s"].to_numpy()
    t0, tf = float(t[0]), float(t[-1])

    sigma_fn = interp1d(t, np.deg2rad(df["sigma_deg"].to_numpy()), kind="linear", fill_value="extrapolate")
    alpha_fn = interp1d(t, df["alpha_deg"].to_numpy(), kind="linear", fill_value="extrapolate")

    lat0 = np.deg2rad(df["lat_deg"].to_numpy()[0])
    state = np.array([
        r_nd_from_alt(df["alt_km"].to_numpy()[0] * 1e3, lat0),
        np.deg2rad(df["lon_deg"].to_numpy()[0]),
        lat0,
        df["V_mps"].to_numpy()[0] / C.V_SCALE,
        np.deg2rad(df["gamma_deg"].to_numpy()[0]),
        np.deg2rad(df["psi_deg"].to_numpy()[0]),
    ], dtype=float)

    ts, states, Qdots, qbars, ns, Js = [t0], [state.copy()], [], [], [], [0.0]

    def _path(st, a_deg):
        L, D, rho, _CL, _CD = lift_drag_nd(st[0], st[2], st[3], a_deg)
        return path_quantities(rho, st[3] * C.V_SCALE, L, D)   # Qdot, qbar, n

    Q0, qb0, n0 = _path(state, float(alpha_fn(t0)))
    Qdots, qbars, ns = [Q0], [qb0], [n0]
    tc = t0
    while tc < tf - 1e-9:
        h = min(dt, tf - tc)
        prevQ = Qdots[-1]
        state = rk4_step(state, tc, h, lambda tt: float(sigma_fn(tt)), lambda tt: float(alpha_fn(tt)))
        tc += h
        Q, qb, n = _path(state, float(alpha_fn(tc)))
        ts.append(tc); states.append(state.copy())
        Qdots.append(Q); qbars.append(qb); ns.append(n)
        Js.append(Js[-1] + 0.5 * (prevQ + Q) * h)        # trapezoid heat load

    ts = np.array(ts); states = np.array(states)
    r_nd, lon, lat, V_nd, gam, psi = states.T
    sim = dict(t=ts, alt_km=altitude_m(r_nd, lat) / 1e3, lon_deg=np.rad2deg(lon),
               lat_deg=np.rad2deg(lat), V_mps=V_nd * C.V_SCALE, gamma_deg=np.rad2deg(gam),
               psi_deg=np.rad2deg(psi), Qdot=np.array(Qdots), qbar=np.array(qbars),
               n=np.array(ns), J=np.array(Js))
    return sim, df, summ


def plot_validation(folder, out_prefix, title):
    sim, df, summ = replay_full(folder)
    t_b = df["tAbs_s"].to_numpy()

    # node-for-node residuals (our sim resampled to benchmark nodes)
    def at_nodes(key):
        return interp1d(sim["t"], sim[key], kind="linear", fill_value="extrapolate")(t_b)
    d_alt = np.abs(at_nodes("alt_km") - df["alt_km"].to_numpy()) * 1e3        # m
    d_V = np.abs(at_nodes("V_mps") - df["V_mps"].to_numpy())                  # m/s
    d_lon = np.abs(at_nodes("lon_deg") - df["lon_deg"].to_numpy())
    d_lat = np.abs(at_nodes("lat_deg") - df["lat_deg"].to_numpy())
    J_bench = summ.get("Integrated heat load [SI]", float("nan"))
    dJ_pct = 100.0 * (sim["J"][-1] - J_bench) / J_bench if np.isfinite(J_bench) else float("nan")

    OUR = dict(color="C0", lw=2.0, label="our env (SCvx controls)")
    SCV = dict(color="k", ls="--", lw=1.3, label="SCvx trajectory")

    fig, ax = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle(f"{title}\nmax |Δalt|={d_alt.max():.1f} m   max |ΔV|={d_V.max():.2f} m/s   "
                 f"max |Δlon|={d_lon.max():.2e}°   max |Δlat|={d_lat.max():.2e}°   "
                 f"ΔJ_heat={dJ_pct:+.4f}%", fontsize=12)

    # 1 — altitude vs velocity
    a = ax[0, 0]
    a.plot(sim["V_mps"], sim["alt_km"], **OUR)
    a.plot(df["V_mps"], df["alt_km"], **SCV)
    a.set_xlabel("Relative velocity (m/s)"); a.set_ylabel("Altitude (km)")
    a.set_title("Altitude vs velocity"); a.grid(True, alpha=.3); a.invert_xaxis(); a.legend(fontsize=8)

    # 2 — ground track
    a = ax[0, 1]
    a.plot(sim["lon_deg"], sim["lat_deg"], **OUR)
    a.plot(df["lon_deg"], df["lat_deg"], **SCV)
    a.plot(C.TARGET["lon_deg"], C.TARGET["lat_deg"], "*", ms=16, color="red", label="target")
    a.set_xlabel("Longitude (deg)"); a.set_ylabel("Latitude (deg)")
    a.set_title("Ground track"); a.grid(True, alpha=.3); a.legend(fontsize=8)

    # 3 — FPA & heading vs time
    a = ax[1, 0]
    a.plot(sim["t"], sim["gamma_deg"], color="C0", lw=2.0, label="γ (our)")
    a.plot(t_b, df["gamma_deg"], color="k", ls="--", lw=1.2, label="γ (SCvx)")
    a.plot(sim["t"], sim["psi_deg"], color="C3", lw=2.0, label="ψ (our)")
    a.plot(t_b, df["psi_deg"], color="0.4", ls="--", lw=1.2, label="ψ (SCvx)")
    a.set_xlabel("time (s)"); a.set_ylabel("Angle (deg)")
    a.set_title("Flight-path & heading angle"); a.grid(True, alpha=.3); a.legend(fontsize=8, ncol=2)

    # 4 — the SCvx controls that drove the replay
    a = ax[1, 1]
    a.plot(t_b, df["sigma_deg"], color="C1", lw=1.6, label="bank σ (SCvx cmd)")
    a.set_xlabel("time (s)"); a.set_ylabel("Bank σ (deg)", color="C1")
    a.tick_params(axis="y", labelcolor="C1"); a.grid(True, alpha=.3)
    a.set_title("SCvx control history (input to both)")
    a2 = a.twinx()
    a2.plot(t_b, df["alpha_deg"], color="C2", lw=1.6, label="AoA α (SCvx cmd)")
    a2.set_ylabel("AoA α (deg)", color="C2"); a2.tick_params(axis="y", labelcolor="C2")

    # 5 — heat flux + cumulative heat load
    a = ax[2, 0]
    a.plot(sim["t"], sim["Qdot"] / 1e6, **OUR)
    a.axhline(C.QDOT_MAX / 1e6, ls=":", color="C3", alpha=.7, label="Q̇ limit")
    a.set_xlabel("time (s)"); a.set_ylabel("Heat flux (MW/m²)", color="C0")
    a.tick_params(axis="y", labelcolor="C0"); a.grid(True, alpha=.3)
    a.set_title("Heat flux & cumulative heat load"); a.legend(fontsize=8, loc="upper right")
    a2 = a.twinx()
    a2.plot(sim["t"], sim["J"] / 1e6, color="C1", lw=2.0)
    a2.axhline(J_bench / 1e6, ls="--", color="k", lw=1.0)
    a2.set_ylabel("Heat load J (MJ/m²)  [dashed=SCvx final]", color="C1"); a2.tick_params(axis="y", labelcolor="C1")

    # 6 — node-for-node residuals (validation evidence)
    a = ax[2, 1]
    a.semilogy(t_b, np.maximum(d_alt, 1e-6), color="C0", label="|Δ altitude| [m]")
    a.semilogy(t_b, np.maximum(d_V, 1e-6), color="C1", label="|Δ velocity| [m/s]")
    a.semilogy(t_b, np.maximum(d_lat * 111.0, 1e-6), color="C2", label="|Δ lat| [km]")
    a.set_xlabel("time (s)"); a.set_ylabel("|our − SCvx|")
    a.set_title("Node-for-node residual (our env vs SCvx)"); a.grid(True, alpha=.3, which="both"); a.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(out_prefix); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return dict(d_alt_m=d_alt.max(), d_V=d_V.max(), dJ_pct=dJ_pct)


def main():
    outdir = Path(__file__).resolve().parents[2] / "results" / "validation_plots"
    titles = {
        "bank_only": "Environment validation — Stage 1 (bank-only) — our 3-DOF core vs SCvx",
        "bank_aoa":  "Environment validation — Stage 2 (bank + AoA) — our 3-DOF core vs SCvx",
    }
    for label, folder in BENCHMARKS.items():
        m = plot_validation(folder, outdir / f"env_validation_{label}", titles[label])
        print(f"{label:10s}: max|d_alt|={m['d_alt_m']:.1f} m  max|d_V|={m['d_V']:.2f} m/s  "
              f"d_J={m['dJ_pct']:+.4f}%  -> {outdir / ('env_validation_'+label+'.png')}")


if __name__ == "__main__":
    main()
