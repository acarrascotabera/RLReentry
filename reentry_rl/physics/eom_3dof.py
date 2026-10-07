"""3-DOF translational equations of motion (rotating, oblate-J2 Earth).

Faithful port of reentry_simulator/guidance/physics/entry_dynamics_bank_aoa_no_control.m.
State (non-dimensional, except alpha which is passed in degrees):
    r    non-dim geocentric radius (r/R0)
    lon  longitude [rad]
    lat  geocentric latitude [rad]
    V    non-dim relative speed (V/VSCALE)
    gamma flight-path angle [rad]
    psi  heading [rad]
Controls passed in: sigma (bank) [rad], alpha [deg].
Returned derivatives are d(state)/d(PHYSICAL TIME in seconds) (the rateScale
factor converts non-dim state rates to physical-time rates).
"""
import numpy as np

from .constants import RATE_SCALE, OMEGA
from .gravity_j2 import gravity_j2
from .aero_wb001 import lift_drag_nd


def _safe_cos(x, floor=1e-6):
    """cos(x) with a sign-preserving magnitude floor (mirrors safe_cos_floor.m).

    Only active very near +/-90 deg, which this mission never reaches; keeps the
    lon/heading denominators finite for robustness.
    """
    c = np.cos(x)
    sign = np.where(c < 0.0, -1.0, 1.0)
    return np.where(np.abs(c) < floor, floor * sign, c)


def translational_rhs(r, lon, lat, V, gamma, psi, sigma, alpha_deg):
    """Return d/dt of [r, lon, lat, V, gamma, psi] (per physical second)."""
    L, D, _rho, _CL, _CD = lift_drag_nd(r, lat, V, alpha_deg)
    gR, gD = gravity_j2(r, lat)

    s = RATE_SCALE
    Om = OMEGA

    sinLat = np.sin(lat)
    cosLat = _safe_cos(lat)
    sinGam = np.sin(gamma)
    cosGam = _safe_cos(gamma)
    sinPsi = np.sin(psi)
    cosPsi = np.cos(psi)

    dr = s * V * sinGam
    dlon = s * V * cosGam * sinPsi / (r * cosLat)
    dlat = s * V * cosGam * cosPsi / r
    dV = s * (-D
              + gR * sinGam
              + gD * sinPsi * cosGam
              + Om ** 2 * r * cosLat * (sinGam * cosLat - cosGam * sinLat * cosPsi))
    dgam = s * (L * np.cos(sigma) / V
                + V * cosGam / r
                + gR * cosGam / V
                - gD * sinPsi * sinGam / V
                + 2.0 * Om * cosLat * sinPsi
                + Om ** 2 * r * cosLat * (cosGam * cosLat + sinGam * sinLat * cosPsi) / V)
    dpsi = s * (L * np.sin(sigma) / (V * cosGam)
                + V * cosGam * sinPsi * np.tan(lat) / r
                - gD * sinPsi / (V * cosLat)
                - 2.0 * Om * (np.tan(gamma) * cosPsi * cosLat - sinLat)
                + Om ** 2 * r * sinLat * cosLat * sinPsi / (V * cosGam))

    return np.array([dr, dlon, dlat, dV, dgam, dpsi])


def rk4_step(state, t, h, sigma_fn, alpha_fn):
    """One classic RK4 step of the 6 translational states.

    sigma_fn(t) -> bank [rad]; alpha_fn(t) -> AoA [deg]. Controls are sampled at
    the RK4 sub-stage times so a prescribed (open-loop) command history is
    integrated consistently.
    """
    def f(tt, st):
        return translational_rhs(st[0], st[1], st[2], st[3], st[4], st[5],
                                 sigma_fn(tt), alpha_fn(tt))
    k1 = f(t, state)
    k2 = f(t + 0.5 * h, state + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, state + 0.5 * h * k2)
    k4 = f(t + h, state + h * k3)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
