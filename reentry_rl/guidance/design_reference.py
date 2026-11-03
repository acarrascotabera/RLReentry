"""Offline design of the FNPEG baseline's reference trajectory (nominal mission).

The SCvx-optimal ground track turns the vehicle east early, back north (an S-bend at
3-4 km/s that needs ~1.5 g of lift) and then sharply east in the last 100 km. A sign-only
lateral logic cannot hold that S-bend under dispersions. The baseline therefore uses a
ground track designed for its own control structure: the nominal vehicle flown with

    phase 0   bank 0 until the pull-out (gamma >= 0)
    then      |sigma|(e) linear in energy from sigma0 (at the pull-out) to sigma_f (e = 0),
              ONE bank reversal at energy e_r: left (sigma < 0) before, right after
    AoA       the SCvx-nominal schedule alpha(V)

with (sigma0, sigma_f, e_r) solved by Newton (finite differences) so that the 25 km
crossing is at the target point with the target heading. The flight-path angle is left
to the terminal phase. The result is stored as results/baselines/fnpeg_reference.csv
(same columns as the SCvx reference) and read by FnpegGuidance.

Control margin: flown at the SCvx AoA schedule (alpha ~ 40 deg, the minimum-heat-load
choice) the nominal needs only ~19 deg of bank, i.e. it sits at the vehicle's range
limit: a few percent of L/D loss is unrecoverable and there is no lateral authority. The
reference therefore flies the schedule reduced by dalpha_ref below the heat pulse (ramped
between 6800 and 6000 m/s, like the guidance's own AoA modulation), which raises L/D, so
the nominal needs more bank (range and lateral margin) at some cost in heat load.

    python -m reentry_rl.guidance.design_reference --dalpha 5
"""
import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import json
import numpy as np
import pandas as pd

from reentry_rl.physics import constants as C
from reentry_rl.envs.rewards import RewardWeights
from reentry_rl.envs.stage2_bank_aoa import Stage2BankAoaEnv
from reentry_rl.envs.scenario import NOMINAL
from reentry_rl.guidance.reference import ReferenceTrack, REF_CSV

OUT = REF_CSV.parent / "fnpeg_reference.csv"
_DEG = np.pi / 180.0


def _action(sig, alp):
    lo, hi = C.ALPHA_MIN, C.ALPHA_MAX
    u = np.array([sig / C.BANK_MAX, (alp - 0.5 * (lo + hi)) / (0.5 * (hi - lo))])
    return np.arctanh(np.clip(u, -0.995, 0.995)).astype(np.float32)


def fly(p, env, alpha_of_V, record=False):
    """Nominal flight with p = (sigma0 [deg], sigma_f [deg], e_r). -> (terminal info, rows)."""
    s0, sf, er = p
    env.reset(options={"scenario": NOMINAL})
    phase, e_pull, rows = 0, None, []
    while True:
        z = env._z
        e = env.energy_togo(z)
        V = z[3] * C.V_SCALE
        if phase == 0 and z[4] >= 0.0:
            phase, e_pull = 1, e
        if phase == 0:
            sig = 0.0
        else:
            f = min(max((e_pull - e) / max(e_pull, 1e-6), 0.0), 1.0) if e > 0 else 1.0
            mag = (s0 + (sf - s0) * f) * _DEG
            sig = -mag if e > er else mag
        alp = alpha_of_V(V)
        _, _, term, trunc, info = env.step(_action(sig, alp))
        if record:
            rows.append({k: info[k] for k in ("t_s", "alt_km", "V_mps", "lat_deg", "lon_deg",
                                              "sigma_deg", "alpha_deg", "gamma_deg", "psi_deg")})
        if term or trunc:
            return info, rows


def residual(info):
    lat = info["lat_deg"] * _DEG
    return np.array([info["dlat_deg"] * 111.2,                       # km north
                     info["dlon_deg"] * 111.2 * np.cos(lat),           # km east
                     info["dpsi_deg"]])                                # deg


def design(p0=(20.0, 50.0, 0.5), iters=25, dalpha=0.0, v_hi=6800.0, v_lo=6000.0):
    env = Stage2BankAoaEnv(weights=RewardWeights(), obs_version="v25", action_mode="target",
                           step_mode="energy", exact_terminal=True, rate_saturation=True)
    scvx = ReferenceTrack()

    class _Ref:
        @staticmethod
        def alpha_deg(V):
            g = min(max((v_hi - V) / (v_hi - v_lo), 0.0), 1.0)
            return scvx.alpha_deg(V) - dalpha * g

    ref = _Ref()
    p = np.array(p0, dtype=float)
    step = np.array([0.5, 1.0, 0.005])
    lo, hi = np.array([0.0, 0.0, 0.02]), np.array([85.0, 85.0, 0.95])
    for k in range(iters):
        info, _ = fly(p, env, ref.alpha_deg)
        r = residual(info)
        print(f"it {k:2d} p = {np.round(p, 4)}  outcome {info['outcome']}  d {info['d_km']:8.2f} km  "
              f"dpsi {info['dpsi_deg']:7.2f}  dfpa {info['dfpa_deg']:6.2f}", flush=True)
        if info["d_km"] < 0.2 and abs(info["dpsi_deg"]) < 0.2:
            break
        J = np.zeros((3, 3))
        for j in range(3):
            q = p.copy()
            q[j] += step[j]
            J[:, j] = (residual(fly(q, env, ref.alpha_deg)[0]) - r) / step[j]
        dp = np.linalg.lstsq(J, -r, rcond=None)[0]
        lim = np.array([10.0, 15.0, 0.1])
        dp = np.clip(dp, -lim, lim)
        p = np.clip(p + dp, lo, hi)
    info, rows = fly(p, env, ref.alpha_deg, record=True)
    df = pd.DataFrame(rows).rename(columns={"alt_km": "h_km"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[["t_s", "h_km", "V_mps", "lat_deg", "lon_deg", "sigma_deg", "alpha_deg"]].to_csv(OUT, index=False,
                                                                                       float_format="%.6f")
    meta = {"dalpha_ref_deg": dalpha, "v_hi_mps": v_hi, "v_lo_mps": v_lo,
            "sigma0_deg": p[0], "sigma_f_deg": p[1], "e_rev": p[2], "d_km": info["d_km"],
            "dpsi_deg": info["dpsi_deg"], "dfpa_deg": info["dfpa_deg"], "V_f_mps": info["V_f_mps"],
            "J_heat": info["J_heat"], "peak_Qdot": info["peak_Qdot"], "peak_n": info["peak_n"],
            "peak_qbar": info["peak_qbar"]}
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Design the FNPEG reference trajectory")
    ap.add_argument("--dalpha", type=float, default=5.0, help="AoA reduction below the heat pulse [deg]")
    args = ap.parse_args()
    design(p0=(35.0, 35.0, 0.6), dalpha=args.dalpha)
