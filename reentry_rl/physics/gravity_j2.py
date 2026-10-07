"""Rotating oblate-spheroid J2 gravity — port of
reentry_simulator/guidance/physics/entry_gravity_j2.m.

Returns the radial (gR) and transverse (gDelta) gravity components,
NORMALIZED BY g0, as functions of non-dim radius and geocentric latitude.
gR is negative (points inward); the EOM consumes the signed values directly.
"""
import numpy as np

from .constants import MU, G0, R0, J2, J2_REF_RADIUS, USE_J2

_MUBAR = MU / (G0 * R0 ** 2)
_REBAR = J2_REF_RADIUS / R0


def gravity_j2(r_nd, lat_rad):
    """Return (gR_nd, gDelta_nd), gravity components normalized by g0."""
    K = (J2 if USE_J2 else 0.0) * _REBAR ** 2
    s = np.sin(lat_rad)
    c = np.cos(lat_rad)
    shape = 3.0 * s * s - 1.0
    r2 = r_nd * r_nd
    r4 = r2 * r2
    gR_nd = -_MUBAR / r2 * (1.0 - 1.5 * K * shape / r2)
    gDelta_nd = -3.0 * _MUBAR * K * (s * c) / r4
    return gR_nd, gDelta_nd
