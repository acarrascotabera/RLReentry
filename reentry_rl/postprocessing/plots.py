"""Trajectory profile plots for an RL eval rollout.

Mirrors the physically-meaningful panels of the reference MATLAB figure
(reentry_simulator/.../guidance_profiles.png): altitude-vs-velocity, ground
track, dynamic pressure + load factor, heat flux + heat load, AoA + bank, and
bank-angle rate. The two SCP-solver-only panels (remaining-horizon, wall time)
are replaced with RL-relevant ones (range-to-go, FPA + heading).

Exports PNG (200 dpi) + SVG. Run:
    python -m reentry_rl.postprocessing.plots --csv <eval_trajectory.csv> --out <prefix> --title "..."
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from reentry_rl.physics import constants as C
from reentry_rl.physics.geodesy import great_circle_distance_m

_DEG = np.pi / 180.0


def _save(fig, out_prefix):
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(csv_path, out_prefix, title="Stage-1 RL guidance"):
    """Render the 3x2 profile figure from an eval_trajectory.csv. Returns miss [km]."""
    df = pd.read_csv(csv_path)
    t = df["t_s"].to_numpy()

    lat_f, lon_f = float(df["lat_deg"].iloc[-1]), float(df["lon_deg"].iloc[-1])
    miss_km = float(great_circle_distance_m(lat_f * _DEG, lon_f * _DEG,
                                            C.TARGET["lat_deg"] * _DEG,
                                            C.TARGET["lon_deg"] * _DEG) / 1e3)
    Vf, hf = float(df["V_mps"].iloc[-1]), float(df["alt_km"].iloc[-1])
    gam_f, psi_f = float(df["gamma_deg"].iloc[-1]), float(df["psi_deg"].iloc[-1])
    Qload = float(df["J_heat"].iloc[-1]) / 1e6                    # MJ/m^2
    bank_rate = np.gradient(df["sigma_deg"].to_numpy(), t)       # deg/s
    rng_km = great_circle_distance_m(df["lat_deg"].to_numpy() * _DEG,
                                     df["lon_deg"].to_numpy() * _DEG,
                                     C.TARGET["lat_deg"] * _DEG,
                                     C.TARGET["lon_deg"] * _DEG) / 1e3

    fig, ax = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(f"{title}\n$V_f$={Vf:.0f} m/s   $h_f$={hf:.2f} km   "
                 f"Heat load={Qload:.1f} MJ/m$^2$   miss={miss_km:.1f} km   "
                 f"($\\gamma_f$={gam_f:.1f}°, $\\psi_f$={psi_f:.1f}°)", fontsize=12)

    # 1 — altitude vs velocity
    a = ax[0, 0]
    a.plot(df["V_mps"], df["alt_km"], color="C0")
    a.set_xlabel("Relative velocity (m/s)"); a.set_ylabel("Altitude (km)")
    a.set_title("Altitude vs velocity"); a.grid(True, alpha=.3); a.invert_xaxis()

    # 2 — ground track + target + miss
    a = ax[0, 1]
    a.plot(df["lon_deg"], df["lat_deg"], color="C0", lw=1.3, label="trajectory")
    a.plot(C.TARGET["lon_deg"], C.TARGET["lat_deg"], "*", ms=16, color="red", label="target")
    a.plot(df["lon_deg"].iloc[0], df["lat_deg"].iloc[0], "o", ms=7, color="green", label="start")
    a.plot(lon_f, lat_f, "s", ms=7, color="black", label="final")
    a.annotate(f"miss = {miss_km:.1f} km", xy=(lon_f, lat_f),
               xytext=(0.30, 0.10), textcoords="axes fraction", fontsize=10,
               arrowprops=dict(arrowstyle="->", color="0.3"))
    a.set_xlabel("Longitude (deg)"); a.set_ylabel("Latitude (deg)")
    a.set_title("Ground track"); a.grid(True, alpha=.3); a.legend(fontsize=8, loc="best")

    # 3 — dynamic pressure + load factor (with limits)
    a = ax[1, 0]
    a.plot(t, df["qbar_Pa"] / 1e3, color="C0", label="q̄")
    a.axhline(C.QBAR_MAX / 1e3, ls="--", color="C0", alpha=.5)
    a.set_xlabel("time (s)"); a.set_ylabel("Dyn. pressure (kPa)", color="C0")
    a.tick_params(axis="y", labelcolor="C0"); a.grid(True, alpha=.3)
    a.set_title("Dynamic pressure & load factor")
    a2 = a.twinx()
    a2.plot(t, df["n_g"], color="C1", label="n")
    a2.axhline(C.N_MAX, ls="--", color="C1", alpha=.5)
    a2.set_ylabel("Load factor (g)", color="C1"); a2.tick_params(axis="y", labelcolor="C1")

    # 4 — heat flux + heat load (with limit)
    a = ax[1, 1]
    a.plot(t, df["Qdot_Wm2"] / 1e6, color="C0")
    a.axhline(C.QDOT_MAX / 1e6, ls="--", color="C0", alpha=.5)
    a.set_xlabel("time (s)"); a.set_ylabel("Heat flux (MW/m$^2$)", color="C0")
    a.tick_params(axis="y", labelcolor="C0"); a.grid(True, alpha=.3)
    a.set_title("Heat flux & heat load")
    a2 = a.twinx()
    a2.plot(t, df["J_heat"] / 1e6, color="C1")
    a2.set_ylabel("Heat load (MJ/m$^2$)", color="C1"); a2.tick_params(axis="y", labelcolor="C1")

    # 5 — AoA + bank angle
    a = ax[2, 0]
    a.plot(t, df["alpha_deg"], color="C0")
    a.set_xlabel("time (s)"); a.set_ylabel("Angle of attack (deg)", color="C0")
    a.tick_params(axis="y", labelcolor="C0"); a.grid(True, alpha=.3)
    a.set_title("AoA & bank angle")
    a2 = a.twinx()
    a2.plot(t, df["sigma_deg"], color="C1")
    a2.set_ylabel("Bank angle (deg)", color="C1"); a2.tick_params(axis="y", labelcolor="C1")

    # 6 — bank rate (smoothness) + range-to-go on twin axis
    a = ax[2, 1]
    a.plot(t, bank_rate, color="C1", lw=0.9)
    a.set_xlabel("time (s)"); a.set_ylabel("Bank rate (deg/s)", color="C1")
    a.tick_params(axis="y", labelcolor="C1"); a.grid(True, alpha=.3)
    a.set_title("Bank-angle rate  &  range-to-go")
    a2 = a.twinx()
    a2.plot(t, rng_km, color="C0")
    a2.set_ylabel("Range-to-go (km)", color="C0"); a2.tick_params(axis="y", labelcolor="C0")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_prefix)
    return miss_km


def main():
    ap = argparse.ArgumentParser(description="Trajectory profile plots from an eval CSV")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="output path prefix (no extension)")
    ap.add_argument("--title", default="Stage-1 RL guidance")
    args = ap.parse_args()
    miss = plot_trajectory(args.csv, args.out, args.title)
    print(f"miss = {miss:.1f} km  ->  {args.out}.png / .svg")


if __name__ == "__main__":
    main()
