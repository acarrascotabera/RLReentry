"""US-1976 Standard Atmosphere — faithful port of
reentry_simulator/guidance/physics/atmosphere_properties.m.

Piecewise layer model on geopotential altitude, extended to 120 km geometric;
isothermal exponential decay above. Returns density, temperature, speed of
sound and the density altitude-derivative. Vectorized over altitude.
"""
import numpy as np

_G0 = 9.80665
_R = 287.05287          # specific gas constant for air [J/(kg K)]
_GAMMA = 1.4
_RE = 6356766.0         # radius used for geopotential conversion [m]

# Geopotential layer base altitudes [m] and lapse rates [K/m]
_HB = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0,
                84852.0, 89716.0, 98451.0, 108128.0, 117778.0], dtype=float)
_LB = np.array([-6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3,
                0.0, 0.9311e-3, 4.642e-3, 12.44e-3], dtype=float)

# Precompute base temperature / pressure at each layer boundary.
_TB = np.zeros(_HB.size)
_PB = np.zeros(_HB.size)
_TB[0] = 288.15
_PB[0] = 101325.0
for _k in range(_LB.size):
    if abs(_LB[_k]) < 1e-12:
        _TB[_k + 1] = _TB[_k]
        _PB[_k + 1] = _PB[_k] * np.exp(-_G0 * (_HB[_k + 1] - _HB[_k]) / (_R * _TB[_k]))
    else:
        _TB[_k + 1] = _TB[_k] + _LB[_k] * (_HB[_k + 1] - _HB[_k])
        _PB[_k + 1] = _PB[_k] * (_TB[_k] / _TB[_k + 1]) ** (_G0 / (_R * _LB[_k]))

_INTERNAL_EDGES = _HB[1:-1]   # 10 internal boundaries -> bins 0..10


def atmosphere(h_m):
    """Return (rho [kg/m^3], T [K], a [m/s], drho_dh [kg/m^4]) at geometric altitude h_m."""
    h = np.asarray(h_m, dtype=float)
    scalar = (h.ndim == 0)
    h = np.atleast_1d(h)
    h = np.maximum(h, 0.0)

    hgeo = _RE * h / (_RE + h)               # geometric -> geopotential
    dhgeo_dh = _RE ** 2 / (_RE + h) ** 2

    rho = np.zeros_like(hgeo)
    T = np.zeros_like(hgeo)
    drho_dhgeo = np.zeros_like(hgeo)

    top = hgeo > _HB[-1]
    std = ~top

    if np.any(std):
        hs = hgeo[std]
        base = np.searchsorted(_INTERNAL_EDGES, hs, side="right")   # 0..10
        Tbase = _TB[base]
        pbase = _PB[base]
        L = _LB[base]
        h0 = _HB[base]
        Tloc = Tbase + L * (hs - h0)

        iso = np.abs(L) < 1e-12
        lap = ~iso
        rhos = np.zeros_like(hs)
        dstd = np.zeros_like(hs)

        if np.any(iso):
            Ti = Tbase[iso]
            pI = pbase[iso] * np.exp(-_G0 * (hs[iso] - h0[iso]) / (_R * Ti))
            rhos[iso] = pI / (_R * Ti)
            dstd[iso] = -rhos[iso] * _G0 / (_R * Ti)
        if np.any(lap):
            Tl = Tloc[lap]
            pL = pbase[lap] * (Tbase[lap] / Tl) ** (_G0 / (_R * L[lap]))
            rhos[lap] = pL / (_R * Tl)
            dstd[lap] = -rhos[lap] * (_G0 / _R + L[lap]) / Tl

        rho[std] = rhos
        T[std] = Tloc
        drho_dhgeo[std] = dstd

    if np.any(top):
        Ttop = _TB[-1]
        ptop = _PB[-1]
        rhotop = ptop / (_R * Ttop)
        H = _R * Ttop / _G0
        rho[top] = rhotop * np.exp(-(hgeo[top] - _HB[-1]) / H)
        drho_dhgeo[top] = -rho[top] / H
        T[top] = Ttop

    drho_dh = drho_dhgeo * dhgeo_dh
    a = np.sqrt(_GAMMA * _R * np.maximum(T, 1.0))

    if scalar:
        return float(rho[0]), float(T[0]), float(a[0]), float(drho_dh[0])
    return rho, T, a, drho_dh
