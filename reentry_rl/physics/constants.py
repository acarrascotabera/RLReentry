"""Earth, non-dimensionalization, vehicle (WB001) and mission constants.

All values are copied VERBATIM from the reference framework so the Python
simulator is bit-comparable with MATLAB:
  - Earth block : reentry_simulator/config/default_config.m  (lines ~165-189)
  - Vehicle     : reentry_simulator/vehicles/wb001/wb001_setup.m
  - Force scale : reentry_simulator/guidance/physics/entry_aero_quantities_and_derivatives.m

Non-dimensionalization (matching the reference):
    length   r_nd  = r / R0
    velocity V_nd  = V / VSCALE,         VSCALE   = sqrt(R0 * g0)
    time     via RATE_SCALE = sqrt(g0 / R0); the EOM derivatives produced by
             eom_3dof are d(state_nd)/d(physical time in seconds).
    gravity / aero accelerations are expressed in units of g0 (so the load
    factor n = hypot(L_nd, D_nd) comes out directly in g's).
"""
import numpy as np

# --------------------------------------------------------------------------
# Earth / environment (default_config.m)
# --------------------------------------------------------------------------
R0 = 6378e3                 # reference radius for non-dim [m]
RAVR = 6371e3               # mean radius [m]
G0 = 9.80665                # reference gravity [m/s^2]
MU = 3.986004418e14         # gravitational parameter [m^3/s^2]
J2 = 1.082626e-3
USE_J2 = True
USE_ELLIPSOID = True
OMEGA_E = 7.2921159e-5      # Earth rotation rate [rad/s]
FLATTENING = 1.0 / 298.257223563
A_ELLIPSOID = 6378137.0     # WGS-84 semi-major axis [m]
B_ELLIPSOID = A_ELLIPSOID * (1.0 - FLATTENING)   # semi-minor [m]
J2_REF_RADIUS = A_ELLIPSOID

# Derived scales
TIME_SCALE = np.sqrt(R0 / G0)        # [s]      ~ 806.4
V_SCALE = np.sqrt(R0 * G0)           # [m/s]    ~ 7908.49
RATE_SCALE = np.sqrt(G0 / R0)        # [1/s]    ~ 1.24008e-3
OMEGA = OMEGA_E / RATE_SCALE         # non-dim Earth rate ~ 0.058803

# --------------------------------------------------------------------------
# WB001 vehicle (wb001_setup.m ; mapped in default_config.m)
# --------------------------------------------------------------------------
MASS = 104305.0             # [kg]
SREF = 391.22               # reference area Aref [m^2]
LREF = 29.2                 # reference length / span [m]
KQ = 1.65e-4                # Sutton-Graves heat-rate coefficient
HEAT_RATE_VEL_EXP = 3.15    # n in Qdot = kQ*sqrt(rho)*V^n

# Force non-dim factor: L_nd = KFORCE * rho * V_nd^2 * CL  (acceleration in g0)
#   K = R0 * Aref / (2 * mass)     [entry_aero_quantities_and_derivatives.m:173]
KFORCE = R0 * SREF / (2.0 * MASS)

# --------------------------------------------------------------------------
# Path-constraint envelope (wb001_setup.m)
# --------------------------------------------------------------------------
QDOT_MAX = 1.5e6            # heat rate [W/m^2]
N_MAX = 2.5                 # load factor [g]
QBAR_MAX = 18e3             # dynamic pressure [Pa]

# --------------------------------------------------------------------------
# State bounds (wb001_setup.m)
# --------------------------------------------------------------------------
ALT_MIN, ALT_MAX = 5e3, 120e3
V_MIN, V_MAX = 50.0, 8500.0
FPA_MIN, FPA_MAX = np.deg2rad(-89.0), np.deg2rad(89.0)
BANK_MIN, BANK_MAX = np.deg2rad(-85.0), np.deg2rad(85.0)
ALPHA_MIN, ALPHA_MAX = 5.0, 40.0            # [deg]

# Control-rate limits (SCP control bounds, apply_guidance_mode.m)
SIGMA_DOT_MAX = np.deg2rad(45.0)            # [rad/s]
ALPHA_DOT_MAX = np.deg2rad(15.0)            # [rad/s]

# --------------------------------------------------------------------------
# WB001 heat-load mission (wb001_setup.m)
# --------------------------------------------------------------------------
IC = dict(alt_m=100e3, lon_deg=0.0, lat_deg=0.0, V_mps=7450.0, fpa_deg=-0.5, heading_deg=0.0)
TARGET = dict(alt_m=25e3, lon_deg=12.0, lat_deg=70.0, fpa_deg=-10.0, heading_deg=90.0)

# SCP-optimal heat loads (J/m^2) extracted from the benchmark terminal summaries.
# Used for reward normalization and the RL-vs-SCP comparison (mode-matched).
J_REF_BANK_ONLY = 1.0831714903e9     # bank_free_time   (Stage 1)
J_REF_BANK_AOA = 1.0719907300e9      # bank_aoa_free_time (Stage 2)
