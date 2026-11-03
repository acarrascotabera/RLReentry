"""Reference ground track and AoA profile for the FNPEG baseline.

The track is the SCvx-optimal (minimum heat load) closed-loop nominal of the VISTA paper
pipeline (_simcache_minq.mat), decimated to 1 s and stored in
results/baselines/scvx_nominal_reference.csv (extracted once with `extract`). The
baseline uses it only for its lateral logic (which side of the track to bank to) and its
AoA schedule alpha(V); the bank magnitude comes from the FNPEG predictor-corrector.

For a dispersed target the track is rotated rigidly on the sphere so that its end point
and final heading map onto the episode's target point and target heading.

    python -m reentry_rl.guidance.reference --mat <_simcache_minq.mat>
"""
import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from reentry_rl.physics import constants as C
from reentry_rl.physics.geodesy import great_circle_distance_m, bearing_rad

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_CSV = REPO_ROOT / "results" / "baselines" / "scvx_nominal_reference.csv"
_DEG = np.pi / 180.0


def extract(mat_path, out=REF_CSV, dt=1.0):
    import scipy.io as sio
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["C"]
    t = np.asarray(m.t, dtype=float)
    tq = np.append(np.arange(0.0, t[-1], dt), t[-1])
    cols = {"t_s": tq}
    for k, src in (("h_km", "h_km"), ("V_mps", "V_mps"), ("lat_deg", "lat_deg"), ("lon_deg", "lon_deg"),
                   ("sigma_deg", "sigma_deg"), ("alpha_deg", "alpha_deg")):
        cols[k] = np.interp(tq, t, np.asarray(getattr(m, src), dtype=float))
    df = pd.DataFrame(cols)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6f")
    return out


def _unit(lat, lon):
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def _latlon(u):
    return np.arcsin(np.clip(u[..., 2], -1.0, 1.0)), np.arctan2(u[..., 1], u[..., 0])


def _tangent(lat, lon, psi):
    """Unit horizontal direction at (lat, lon) with heading psi (0 = north, + east)."""
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    return np.sin(psi) * east + np.cos(psi) * north


class ReferenceTrack:
    """Ground track as unit vectors with along-track arc angle s (rad, sphere of radius R0)."""

    def __init__(self, df=None, path=None):
        if df is None:
            df = pd.read_csv(path or REF_CSV)
        lat, lon = df["lat_deg"].to_numpy() * _DEG, df["lon_deg"].to_numpy() * _DEG
        self.alpha_V = df["V_mps"].to_numpy(), df["alpha_deg"].to_numpy()
        self._u0 = _unit(lat, lon)
        self._h0 = df["h_km"].to_numpy() * 1e3
        # final heading of the reference: bearing over the last ~5 km of track
        k = len(lat) - 1
        j = k
        while j > 0 and great_circle_distance_m(lat[j], lon[j], lat[k], lon[k]) < 5e3:
            j -= 1
        lat_f, lon_f = lat[k], lon[k]
        psi_f = bearing_rad(lat[j], lon[j], lat_f, lon_f)
        self.lat_f0, self.lon_f0, self.psi_f0 = float(lat_f), float(lon_f), float(psi_f)
        self.set_target(lat_f, lon_f, psi_f)

    def set_target(self, lat_t, lon_t, psi_t=None):
        """Rotate the nominal track so its end point maps onto (lat_t, lon_t): the minimal
        rotation (about p0 x p1), so the start of the track moves by no more than the target
        offset. The final heading is left to the terminal guidance phase."""
        p0 = _unit(np.asarray(self.lat_f0), np.asarray(self.lon_f0))
        p1 = _unit(np.asarray(lat_t), np.asarray(lon_t))
        ax = np.cross(p0, p1)
        sa, ca = np.linalg.norm(ax), float(p0 @ p1)
        if sa < 1e-15:
            Rm = np.eye(3)
        else:
            k = ax / sa
            K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
            Rm = np.eye(3) + sa * K + (1.0 - ca) * (K @ K)
        u = self._u0 @ Rm.T
        self.u = u / np.linalg.norm(u, axis=1, keepdims=True)
        self.lat, self.lon = _latlon(self.u)
        dots = np.clip(np.sum(self.u[1:] * self.u[:-1], axis=1), -1.0, 1.0)
        ds = np.arccos(dots)
        keep = np.concatenate([[True], ds > 1e-7])
        self.u, self.lat, self.lon = self.u[keep], self.lat[keep], self.lon[keep]
        self.h = self._h0[keep]
        dots = np.clip(np.sum(self.u[1:] * self.u[:-1], axis=1), -1.0, 1.0)
        self.s = np.concatenate([[0.0], np.cumsum(np.arccos(dots))])     # arc angle [rad]
        self.S = float(self.s[-1])
        # vertical reference: altitude and path slope dh/dxi against ground distance
        xi = self.s * C.R0
        self.slope = np.gradient(self.h, xi)
        # segment poles (left of travel) and headings
        n = np.cross(self.u[:-1], self.u[1:])
        self.n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        psi = bearing_rad(self.lat[:-1], self.lon[:-1], self.lat[1:], self.lon[1:])
        self.psi = np.append(psi, psi[-1])
        # continuous heading: segment headings at segment midpoints, unwrapped
        self._s_mid = np.append(0.5 * (self.s[:-1] + self.s[1:]), self.S)
        self._psi_u = np.unwrap(self.psi)
        self._k = 0
        return self

    def reset_search(self):
        self._k = 0

    def project(self, lat, lon):
        """-> (s along track [rad], cross-track [rad, + = right of travel], track heading [rad])."""
        p = _unit(np.asarray(lat), np.asarray(lon))
        # nearest vertex, searched forward from the last hit (monotone progress)
        lo = max(self._k - 50, 0)
        d = self.u[lo:] @ p
        k = lo + int(np.argmax(d))
        self._k = k
        best = None
        for j in (k - 1, k):
            if j < 0 or j >= len(self.s) - 1:
                continue
            nj = self.n[j]
            xt = float(np.arcsin(np.clip(p @ nj, -1.0, 1.0)))
            q = p - np.sin(xt) * nj                       # foot on the segment's great circle
            q = q / np.linalg.norm(q)
            a = float(np.arctan2(np.linalg.norm(np.cross(self.u[j], q)), self.u[j] @ q))
            seg = self.s[j + 1] - self.s[j]
            if np.cross(self.u[j], q) @ nj < 0:
                a = -a
            frac = a / max(seg, 1e-300)
            if -1e-6 <= frac <= 1.0 + 1e-6 or best is None:
                cand = (self.s[j] + np.clip(frac, 0.0, 1.0) * seg, -xt, self.psi[j], abs(frac - 0.5))
                if best is None or cand[3] < best[3]:
                    best = cand
        if best is None:
            j = min(k, len(self.s) - 2)
            return float(self.s[j]), 0.0, float(self.psi[j])
        return float(best[0]), float(best[1]), float(best[2])

    def at(self, s):
        """(lat, lon, heading) on the track at arc angle s (clamped)."""
        s = min(max(s, 0.0), self.S)
        j = int(np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.s) - 2))
        f = (s - self.s[j]) / max(self.s[j + 1] - self.s[j], 1e-300)
        u = (1 - f) * self.u[j] + f * self.u[j + 1]
        u = u / np.linalg.norm(u)
        la, lo = _latlon(u)
        return float(la), float(lo), float(self.psi[j])

    def vertical(self, s):
        """(altitude [m], slope dh/dxi, d slope / dxi [1/m]) of the reference at arc angle s."""
        s = min(max(s, 0.0), self.S)
        h = float(np.interp(s, self.s, self.h))
        m = float(np.interp(s, self.s, self.slope))
        d = 2e3 / C.R0
        m1 = float(np.interp(max(s - d, 0.0), self.s, self.slope))
        m2 = float(np.interp(min(s + d, self.S), self.s, self.slope))
        dm = (m2 - m1) / max((min(s + d, self.S) - max(s - d, 0.0)) * C.R0, 1.0)
        return h, m, dm

    def heading(self, s):
        """Continuous track heading [rad, unwrapped] at arc angle s."""
        return float(np.interp(s, self._s_mid, self._psi_u))

    def curvature(self, s, d_m=2e3):
        """Ground-track curvature d psi / d xi [rad/m] (central difference over +-d_m)."""
        d = d_m / C.R0
        a, b = max(s - d, 0.0), min(s + d, self.S)
        return float((self.heading(b) - self.heading(a)) / max((b - a) * C.R0, 1.0))

    def alpha_deg(self, V_mps):
        """Reference AoA schedule alpha(V): the SCvx nominal AoA against speed."""
        V, a = self.alpha_V
        order = np.argsort(V)
        return float(np.interp(V_mps, V[order], a[order]))


def main():
    ap = argparse.ArgumentParser(description="Extract the SCvx nominal reference track")
    ap.add_argument("--mat", required=True)
    args = ap.parse_args()
    p = extract(args.mat)
    tr = ReferenceTrack()
    print(f"{p}: {len(tr.s)} points, length {tr.S * C.R0 / 1e3:.0f} km, "
          f"final heading {np.rad2deg(tr.psi_f0):.2f} deg at ({np.rad2deg(tr.lat_f0):.3f}, {np.rad2deg(tr.lon_f0):.3f})")


if __name__ == "__main__":
    main()
