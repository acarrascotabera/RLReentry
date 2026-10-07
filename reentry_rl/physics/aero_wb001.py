"""WB001 aerodynamics, AoA schedule and path quantities.

Ports:
  - polynomial CL/CD  : vehicles/wb001/wb001_polynomial_aero.m  (Mach-independent)
  - bank-only AoA      : vehicles/wb001/wb001_nominal_aoa.m
  - non-dim forces     : entry_aero_quantities_and_derivatives.m  (L_nd = K*rho*V_nd^2*CL)
  - path quantities    : guidance/physics/path_constraints_model.m
"""
from typing import NamedTuple

import numpy as np

from .constants import KFORCE, KQ, HEAT_RATE_VEL_EXP, V_SCALE
from .atmosphere import atmosphere
from .geodesy import altitude_m


class ModelFactors(NamedTuple):
    """Multiplicative model dispersions (VISTA montecarlo/cases/g_models.m).

    k_rho scales the atmospheric density, k_CL / k_CD the aerodynamic
    coefficients and k_mass the vehicle mass (so the non-dim force factor
    KFORCE ~ 1/m scales by 1/k_mass). `None` everywhere means nominal and
    keeps the original code path bit-identical.
    """
    k_rho: float = 1.0
    k_CL: float = 1.0
    k_CD: float = 1.0
    k_mass: float = 1.0

# WB001 nominal-AoA schedule parameters (wb001_nominal_aoa.m)
_AOA_SWITCH_MPS = 4570.0
_AOA_HIGH_DEG = 40.0
_AOA_COEF = 0.20705
_AOA_VREF_MPS = 340.0


def nominal_aoa_deg(V_mps):
    """Legacy WB001 bank-only AoA schedule [deg] as a function of speed [m/s].

    alpha = 40                              for V > 4570
    alpha = 40 - 0.20705*(V-4570)^2/340^2   for V <= 4570
    (No clamping — reproduces the reference exactly.)
    """
    V = np.asarray(V_mps, dtype=float)
    dv = V - _AOA_SWITCH_MPS
    alpha = _AOA_HIGH_DEG - _AOA_COEF * (dv ** 2) / (_AOA_VREF_MPS ** 2)
    alpha = np.where(V > _AOA_SWITCH_MPS, _AOA_HIGH_DEG, alpha)
    return alpha


def cl_cd(alpha_deg):
    """WB001 polynomial lift/drag coefficients (alpha in degrees)."""
    a = np.asarray(alpha_deg, dtype=float)
    CL = -0.041065 + 0.016292 * a + 0.0002602 * a ** 2
    CD = 0.080505 - 0.03026 * CL + 0.86495 * CL ** 2
    return CL, CD


def lift_drag_nd(r_nd, lat_rad, V_nd, alpha_deg, model: ModelFactors = None):
    """Non-dim lift/drag accelerations (in g0 units) and the local density.

    Returns (L_nd, D_nd, rho, CL, CD); with `model`, rho/CL/CD are the
    dispersed values.
    """
    h = altitude_m(r_nd, lat_rad)
    rho, _T, _a, _drho = atmosphere(h)
    CL, CD = cl_cd(alpha_deg)
    kforce = KFORCE
    if model is not None:
        rho = rho * model.k_rho
        CL = CL * model.k_CL
        CD = CD * model.k_CD
        kforce = KFORCE / model.k_mass
    L_nd = kforce * rho * V_nd ** 2 * CL
    D_nd = kforce * rho * V_nd ** 2 * CD
    return L_nd, D_nd, rho, CL, CD


def heat_rate(rho, V_mps):
    """Sutton-Graves convective heat rate [W/m^2]."""
    return KQ * np.sqrt(np.maximum(rho, 1e-300)) * V_mps ** HEAT_RATE_VEL_EXP


def path_quantities(rho, V_mps, L_nd, D_nd):
    """Return (Qdot [W/m^2], qbar [Pa], n [g]) — the three path constraints."""
    Qdot = heat_rate(rho, V_mps)
    qbar = 0.5 * rho * V_mps ** 2
    n = np.hypot(L_nd, D_nd)
    return Qdot, qbar, n
