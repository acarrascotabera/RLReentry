"""Numba ports of the onboard physics used by the FNPEG predictor.

Same models as reentry_rl.physics (US-1976 atmosphere, WGS-84 ellipsoidal altitude,
J2 gravity, WB001 polynomial aero, rotating-Earth 3-DOF terms); checked against the
python versions in tests (`python -m reentry_rl.guidance.fastphys`).
"""
import numpy as np
from numba import njit

from ..physics import constants as C
from ..physics import atmosphere as _atm

_HB = _atm._HB.copy()
_LB = _atm._LB.copy()
_TB = _atm._TB.copy()
_PB = _atm._PB.copy()
_G0 = _atm._G0
_R = _atm._R
_RE = _atm._RE
_A, _B = C.A_ELLIPSOID, C.B_ELLIPSOID
_R0 = C.R0
_MUBAR = C.MU / (C.G0 * C.R0 ** 2)
_K2 = C.J2 * (C.J2_REF_RADIUS / C.R0) ** 2
_RS = C.RATE_SCALE
_OM = C.OMEGA
_KF = C.KFORCE
_VS = C.V_SCALE


@njit(cache=True)
def rho_us76(h):
    """Density [kg/m^3] at geometric altitude h [m] (US-1976, isothermal above the table)."""
    if h < 0.0:
        h = 0.0
    hg = _RE * h / (_RE + h)
    if hg > _HB[-1]:
        T = _TB[-1]
        H = _R * T / _G0
        return _PB[-1] / (_R * T) * np.exp(-(hg - _HB[-1]) / H)
    b = 0
    for i in range(1, _HB.size - 1):
        if hg >= _HB[i]:
            b = i
    Tb, pb, L, h0 = _TB[b], _PB[b], _LB[b], _HB[b]
    if abs(L) < 1e-12:
        return pb * np.exp(-_G0 * (hg - h0) / (_R * Tb)) / (_R * Tb)
    T = Tb + L * (hg - h0)
    return pb * (Tb / T) ** (_G0 / (_R * L)) / (_R * T)


@njit(cache=True)
def surf_radius(lat):
    s, c = np.sin(lat), np.cos(lat)
    return (_A * _B) / np.sqrt((_B * c) ** 2 + (_A * s) ** 2)


@njit(cache=True)
def alt_m(r, lat):
    return r * _R0 - surf_radius(lat)


@njit(cache=True)
def cl_cd(a):
    CL = -0.041065 + 0.016292 * a + 0.0002602 * a * a
    CD = 0.080505 - 0.03026 * CL + 0.86495 * CL * CL
    return CL, CD


@njit(cache=True)
def alpha_for_cl(CL):
    """Inverse of the CL polynomial on its increasing branch (alpha in degrees)."""
    a2, a1, a0 = 0.0002602, 0.016292, -0.041065 - CL
    disc = a1 * a1 - 4.0 * a2 * a0
    if disc < 0.0:
        return -a1 / (2.0 * a2)
    return (-a1 + np.sqrt(disc)) / (2.0 * a2)


@njit(cache=True)
def gravity(r, lat):
    s, c = np.sin(lat), np.cos(lat)
    r2 = r * r
    gR = -_MUBAR / r2 * (1.0 - 1.5 * _K2 * (3.0 * s * s - 1.0) / r2)
    gD = -3.0 * _MUBAR * _K2 * (s * c) / (r2 * r2)
    return gR, gD


@njit(cache=True)
def free_rates(r, lat, V, gam, psi):
    """Non-lift parts of dgamma/dt and dpsi/dt [rad/s] (gravity, curvature, Earth rate)."""
    gR, gD = gravity(r, lat)
    sL, cL = np.sin(lat), np.cos(lat)
    sG, cG = np.sin(gam), np.cos(gam)
    sP, cP = np.sin(psi), np.cos(psi)
    dg = _RS * (V * cG / r + gR * cG / V - gD * sP * sG / V + 2.0 * _OM * cL * sP
                + _OM ** 2 * r * cL * (cG * cL + sG * sL * cP) / V)
    dp = _RS * (V * cG * sP * np.tan(lat) / r - gD * sP / (V * cL)
                - 2.0 * _OM * (np.tan(gam) * cP * cL - sL)
                + _OM ** 2 * r * sL * cL * sP / (V * cG))
    return dg, dp


@njit(cache=True)
def _lin(x, xs, ys):
    n = xs.size
    if x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        m = (lo + hi) // 2
        if xs[m] <= x:
            lo = m
        else:
            hi = m
    f = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + f * (ys[hi] - ys[lo])


@njit(cache=True)
def alpha_cmd(V_mps, dalpha, a_V, a_A, v_hi, v_lo, a_lo):
    """AoA schedule: reference alpha(V) reduced by dalpha below the heat pulse (ramped in
    between v_hi and v_lo), floored at a_lo [deg]."""
    a = _lin(V_mps, a_V, a_A)
    if dalpha > 0.0:
        g = (v_hi - V_mps) / (v_hi - v_lo)
        g = 0.0 if g < 0.0 else (1.0 if g > 1.0 else g)
        a = max(a - dalpha * g, min(a, a_lo))
    return a


@njit(cache=True)
def eom6(x, sig, a_deg, kL, kD):
    """Full 3-DOF translational EOM (port of physics.eom_3dof.translational_rhs) with the
    onboard model scaled by the filtered force ratios: x = [r, lon, lat, V, gamma, psi]."""
    r, lat, V, g, p = x[0], x[2], x[3], x[4], x[5]
    h = alt_m(r, lat)
    rho = rho_us76(h)
    CL, CD = cl_cd(a_deg)
    q = _KF * rho * V * V
    L = kL * q * CL
    D = kD * q * CD
    gR, gD = gravity(r, lat)
    sL = np.sin(lat)
    cL = np.cos(lat)
    if abs(cL) < 1e-6:
        cL = 1e-6
    sG, cG = np.sin(g), np.cos(g)
    sP, cP = np.sin(p), np.cos(p)
    out = np.empty(6)
    out[0] = _RS * V * sG
    out[1] = _RS * V * cG * sP / (r * cL)
    out[2] = _RS * V * cG * cP / r
    out[3] = _RS * (-D + gR * sG + gD * sP * cG + _OM ** 2 * r * cL * (sG * cL - cG * sL * cP))
    out[4] = _RS * (L * np.cos(sig) / V + V * cG / r + gR * cG / V - gD * sP * sG / V
                    + 2.0 * _OM * cL * sP + _OM ** 2 * r * cL * (cG * cL + sG * sL * cP) / V)
    out[5] = _RS * (L * np.sin(sig) / (V * cG) + V * cG * sP * np.tan(lat) / r - gD * sP / (V * cL)
                    - 2.0 * _OM * (np.tan(g) * cP * cL - sL) + _OM ** 2 * r * sL * cL * sP / (V * cG))
    return out


@njit(cache=True)
def predict3d(x0, e_now, sig0, sigf, e_rev, kL, kD, h_tgt, E0, Ef, a_V, a_A,
              dalpha, v_hi, v_lo, a_lo, rec, e_wa=-1.0, e_wb=-1.0, t_max=4000.0):
    """Full 3-DOF prediction from x0 = [r, lon, lat, V, gamma, psi] with the FNPEG bank
    profile |sigma|(e) linear from sig0 (e_now) to sigf (e = 0); sign -1 (left) for
    e > e_rev, +1 (right) after, except -1 again in the corrective window e_wb < e <= e_wa;
    AoA schedule alpha_cmd(V, dalpha); to the altitude h_tgt. Returns (status, x_f) with
    status 0 = reached, 1 = skip-out, 2 = slow / timeout; with rec > 0 the samples
    (t, r, lon, lat, V, gamma, psi, h) every rec steps are returned as the third value."""
    x = x0.copy()
    t = 0.0
    h = alt_m(x[0], x[2])
    de = max(e_now, 1e-6)
    nmax = 1 + int(t_max) // max(rec, 1) + 2 if rec > 0 else 1
    out = np.zeros((nmax, 8))
    k = 0
    if rec > 0:
        out[0, 0] = 0.0
        out[0, 1:7] = x
        out[0, 7] = h
        k = 1
    it = 0
    while t < t_max:
        dt = 2.0 if h > 50e3 else 1.0
        E = 0.5 * x[3] * x[3] - _MUBAR / x[0]
        e = (E - Ef) / (E0 - Ef)
        mag = sigf if e <= 0.0 else sig0 + (sigf - sig0) * (e_now - e) / de
        if e > e_now:
            mag = sig0
        sg = -mag if (e > e_rev or (e_wb < e <= e_wa)) else mag
        a = alpha_cmd(x[3] * _VS, dalpha, a_V, a_A, v_hi, v_lo, a_lo)
        k1 = eom6(x, sg, a, kL, kD)
        k2 = eom6(x + 0.5 * dt * k1, sg, a, kL, kD)
        k3 = eom6(x + 0.5 * dt * k2, sg, a, kL, kD)
        k4 = eom6(x + dt * k3, sg, a, kL, kD)
        xn = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        hn = alt_m(xn[0], xn[2])
        it += 1
        if hn <= h_tgt:
            f = (h - h_tgt) / max(h - hn, 1e-9)
            xf = x + f * (xn - x)
            if rec > 0 and k < nmax:
                out[k, 0] = t + f * dt
                out[k, 1:7] = xf
                out[k, 7] = h_tgt
                k += 1
            return 0, xf, out[:k]
        if hn > 130e3:
            return 1, xn, out[:k]
        if xn[3] * _VS < 60.0:
            return 2, xn, out[:k]
        x, h, t = xn, hn, t + dt
        if rec > 0 and it % rec == 0 and k < nmax:
            out[k, 0] = t
            out[k, 1:7] = x
            out[k, 7] = h
            k += 1
    return 2, x, out[:k]


def _selftest():
    from ..physics.atmosphere import atmosphere
    from ..physics.geodesy import altitude_m
    from ..physics.gravity_j2 import gravity_j2
    from ..physics.aero_wb001 import cl_cd as cl_cd_py
    err = 0.0
    for h in np.linspace(0, 130e3, 2601):
        err = max(err, abs(rho_us76(h) / atmosphere(float(h))[0] - 1.0))
    print(f"atmosphere max rel err {err:.2e}")
    for lat in np.linspace(-1.2, 1.2, 7):
        r = 1.01
        assert abs(alt_m(r, lat) - altitude_m(r, lat)) < 1e-6
        g1, g2 = gravity(r, lat)
        h1, h2 = gravity_j2(r, lat)
        assert abs(g1 - h1) < 1e-14 and abs(g2 - h2) < 1e-14
    for a in np.linspace(5, 40, 8):
        c1, c2 = cl_cd(a)
        d1, d2 = cl_cd_py(a)
        assert abs(c1 - d1) < 1e-14 and abs(c2 - d2) < 1e-14
        assert abs(alpha_for_cl(c1) - a) < 1e-9
    from ..physics.eom_3dof import translational_rhs
    x = np.array([1.009, 0.1, 1.1, 0.6, -0.03, 0.4])
    for sig in (-0.7, 0.0, 0.9):
        d1 = eom6(x, sig, 33.0, 1.0, 1.0)
        d2 = translational_rhs(*x, sig, 33.0)
        assert np.max(np.abs(d1 - d2) / (np.abs(d2) + 1e-12)) < 1e-9, (d1, d2)
    print("fastphys OK")


if __name__ == "__main__":
    _selftest()
