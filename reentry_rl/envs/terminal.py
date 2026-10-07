"""Terminal (handover) conditions and success tolerances.

The SCvx reference enforces the terminal state as equality constraints on
altitude, latitude, longitude, flight path angle and heading (terminal speed
free). The RL policy is scored against the same quantities, with the target
position expressed as a great-circle radius around the target point (a 0.1 %
box on lat/lon would be 7.8 km north-south but 0.46 km east-west).

Tolerance sets (|error| <= tol on every component):

  rel_0p1pct     the "fully met" criterion: altitude, FPA and heading within
                 0.1 % of the nominal target magnitude (h 25 m, gamma 0.01 deg,
                 psi 0.09 deg) and position within a 1 km radius of the target
  rel_1pct       10x looser: 1 % (h 250 m, gamma 0.1 deg, psi 0.9 deg), 10 km radius
  vista_mc       the VISTA Monte Carlo defaults (mc_check_constraints.m):
                 h 2 km, gamma 1 deg, psi 5 deg, range 25 km
  pos_1km        the legacy position-only criterion of v1-v19 (range 1 km)

Tolerances are computed from the NOMINAL target so they stay fixed when the
target itself is dispersed.
"""
import numpy as np

from ..physics import constants as C

_T = C.TARGET


def _rel(frac, radius_km):
    return {"h_m": frac * abs(_T["alt_m"]),
            "range_km": radius_km,
            "fpa_deg": frac * abs(_T["fpa_deg"]),
            "psi_deg": frac * abs(_T["heading_deg"])}


TOLERANCE_SETS = {
    "rel_0p1pct": _rel(1e-3, 1.0),
    "rel_1pct": _rel(1e-2, 10.0),
    "vista_mc": {"h_m": 2.0e3, "fpa_deg": 1.0, "psi_deg": 5.0, "range_km": 25.0},
    "pos_1km": {"range_km": 1.0},
}

# Error keys in the env's terminal info dict, by tolerance component.
_ERR_KEYS = {"h_m": "dh_m", "lat_deg": "dlat_deg", "lon_deg": "dlon_deg",
             "fpa_deg": "dfpa_deg", "psi_deg": "dpsi_deg", "range_km": "d_km"}


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def meets(errors, tol_name):
    """True when every component named in the tolerance set is within bounds.
    `errors` is a dict with the env's terminal error keys (dh_m, dlat_deg, ...)."""
    tol = TOLERANCE_SETS[tol_name]
    for comp, bound in tol.items():
        v = errors.get(_ERR_KEYS[comp])
        if v is None or not np.isfinite(v) or abs(v) > bound:
            return False
    return True
