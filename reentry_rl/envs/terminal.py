"""Terminal (handover) conditions and success tolerances.

The SCvx reference enforces the terminal state as equality constraints on
altitude, latitude, longitude, flight path angle and heading (terminal speed
free). The RL policy is scored against the same five quantities.

Tolerance sets (|error| <= tol on every component):

  rel_0p1pct     0.1 % of the nominal target magnitude (the "fully met"
                 criterion): h 25 m, lat 0.07 deg, lon 0.012 deg,
                 gamma 0.01 deg, psi 0.09 deg
  rel_1pct       10x looser (1 %)
  vista_mc       the VISTA Monte Carlo defaults (mc_check_constraints.m):
                 h 2 km, gamma 1 deg, psi 5 deg, range 25 km
  pos_1km        the legacy position-only criterion of v1-v19 (range 1 km)

Tolerances are computed from the NOMINAL target so they stay fixed when the
target itself is dispersed.
"""
import numpy as np

from ..physics import constants as C

_T = C.TARGET


def _rel(frac):
    return {"h_m": frac * abs(_T["alt_m"]),
            "lat_deg": frac * abs(_T["lat_deg"]),
            "lon_deg": frac * abs(_T["lon_deg"]),
            "fpa_deg": frac * abs(_T["fpa_deg"]),
            "psi_deg": frac * abs(_T["heading_deg"])}


TOLERANCE_SETS = {
    "rel_0p1pct": _rel(1e-3),
    "rel_1pct": _rel(1e-2),
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
