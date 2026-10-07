"""Altitude-velocity comparison figure: closed-loop SCvx G&C vs RL guidance.

Reproduces the style of the thesis presentation's traj_alt_vel.pdf (MATLAB,
make_paper_figures.m): altitude [km] vs Earth-relative speed [m/s], reversed
x-axis, and the three path-constraint corridor boundary lines (heat rate
dashed, dynamic pressure dash-dot, load factor dotted; computed by altitude
bisection at the nominal WB001 AoA schedule) — then overlays the flown
closed-loop SCvx trajectory (from the paper's _simcache_minq.mat) and the
best RL trajectory (eval_trajectory.csv from a diagnose_run rollout).

    python -m reentry_rl.postprocessing.fig_scvx_vs_rl \
        --scvx-mat <_simcache_minq.mat> --rl-csv <eval_trajectory.csv> \
        --out <path/prefix>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from reentry_rl.physics import constants as C
from reentry_rl.physics.atmosphere import atmosphere
from reentry_rl.physics.aero_wb001 import nominal_aoa_deg, cl_cd, path_quantities


def _pc_eval(alt_m, V_mps, alpha_deg, which):
    """Path-constraint value at (alt, V, alpha) — mirrors the env/MATLAB physics."""
    rho, _T, _a, _d = atmosphere(float(alt_m))
    CL, CD = cl_cd(float(alpha_deg))
    v_nd = V_mps / C.V_SCALE
    Lnd = C.KFORCE * rho * v_nd ** 2 * CL
    Dnd = C.KFORCE * rho * v_nd ** 2 * CD
    Qdot, qbar, n = path_quantities(rho, V_mps, Lnd, Dnd)
    return {"Qdot": Qdot, "q": qbar, "n": n}[which]


def boundary_alt_km(Vgrid, which, limit):
    """Bisection on altitude for the corridor boundary h_c(V), as in
    make_paper_figures.m local_boundary_alt (42 iterations, 0..140 km)."""
    out = np.empty_like(Vgrid, dtype=float)
    for i, V in enumerate(Vgrid):
        alpha = float(nominal_aoa_deg(V))
        lo, hi = 0.0, 1.4e5
        for _ in range(42):
            mid = 0.5 * (lo + hi)
            if _pc_eval(mid, V, alpha, which) > limit:
                lo = mid          # exceeded below the boundary -> go up
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi) / 1e3
    return out


def main():
    ap = argparse.ArgumentParser(description="SCvx-vs-RL altitude-velocity comparison figure")
    ap.add_argument("--scvx-mat", required=True, help="_simcache_minq.mat from the paper pipeline")
    ap.add_argument("--rl-csv", required=True, help="RL eval_trajectory.csv")
    ap.add_argument("--out", required=True, help="output path prefix (no extension)")
    args = ap.parse_args()

    L = sio.loadmat(args.scvx_mat, squeeze_me=True, struct_as_record=False)["C"]
    rl = pd.read_csv(args.rl_csv)

    # thin the 670k-point MATLAB trajectory for a lean PDF
    idx = np.unique(np.linspace(0, len(L.V_mps) - 1, 4000).astype(int))
    Vs, hs = np.asarray(L.V_mps)[idx], np.asarray(L.h_km)[idx]

    Vgrid = np.linspace(350.0, 7460.0, 160)
    hQ = boundary_alt_km(Vgrid, "Qdot", C.QDOT_MAX)
    hq = boundary_alt_km(Vgrid, "q", C.QBAR_MAX)
    hn = boundary_alt_km(Vgrid, "n", C.N_MAX)

    plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm",
                         "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    gry = (0.35, 0.35, 0.35)
    hbQ, = ax.plot(Vgrid, hQ, "--", color=gry, lw=0.9)
    hbq, = ax.plot(Vgrid, hq, "-.", color=gry, lw=0.9)
    hbn, = ax.plot(Vgrid, hn, ":", color=gry, lw=1.1)
    p1, = ax.plot(Vs, hs, "-", color=(0.0, 0.447, 0.741), lw=1.6)
    p2, = ax.plot(rl["V_mps"], rl["alt_km"], "-", color=(0.85, 0.325, 0.098), lw=1.6)

    ax.set_xlabel(r"Earth-relative speed $V$ [m/s]")
    ax.set_ylabel(r"Altitude $h$ [km]")
    ax.set_xlim(7600, 250)                     # reversed: entry runs high -> low speed
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend([p1, p2, hbQ, hbq, hbn],
              ["SCvx G&C (closed loop)", "RL guidance (PPO)",
               r"$\dot Q = \dot Q_{\max}$", r"$\bar q = \bar q_{\max}$",
               r"$n = n_{\max}$"],
              loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"figure -> {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
