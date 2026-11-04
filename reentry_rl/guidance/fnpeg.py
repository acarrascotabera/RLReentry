"""FNPEG-class entry guidance baseline for the WB001 mission (handover at 25 km with
terminal position, flight-path angle and heading constraints).

All phases run closed loop on the navigated state:

  phase 0  initial descent: bank 0 (full lift up) until the pull-out (gamma >= 0). This keeps
           the vehicle highest through the first heat pulse (the mission's heat-rate floor is
           the peak of this lift-up flight). Phase 0 is flown only while the onboard model
           predicts that pull-out above v_pull (every corrector cycle): a shallow entry can
           settle into equilibrium glide without one (the first versions then flew lift-up to
           the handover: 0.8-3.1 Mm overshoot in 4 of 300 cases at 1.25x VISTA).
  phase 1  FNPEG (Lu, JGCD 2014) with a single bank reversal. The bank magnitude is linear
           in energy from sigma0 (now) to sigma_f (handover energy); the sign is left before
           the reversal energy e_r and right after it. Each corrector cycle, a numerical
           predictor integrates the full 3-DOF dynamics (onboard model) to the 25 km
           handover altitude, and a damped Gauss-Newton corrector updates (sigma0, sigma_f,
           e_r) to null the predicted north/east position error and heading error at the
           handover. After the reversal only the two magnitudes are solved, in a weighted
           least-squares sense. Corrector steps are under-relaxed (x0.5), which averages the
           cycle-to-cycle effect of navigation noise on the reversal timing.
           Range extension: a fourth parameter reduces the AoA from the reference schedule
           below the heat pulse (higher L/D). It engages only once the bank magnitude is at its
           floor; a small preference residual then returns it to zero when the bank alone can
           null the predicted errors again. The LM damping is Marquardt-scaled (the parameters
           have different units) and a parameter held at a bound leaves the step.
           Constraint enforcement: the bank magnitude is bounded by quasi-equilibrium glide
           at the heat-rate / load-factor / dynamic-pressure boundary; the AoA by the
           load-factor limit.
  phase 2  terminal (last x_term km of the predicted path): full lift-vector control. The last
           phase-1 prediction (energy-consistent by construction) is the reference path.
           It is rotated onto the target point and blended at the end into the target heading,
           flight-path angle and altitude. Both channels follow it with curvature
           feedforward; their lengths shrink to x_go / sqrt 6, so each becomes the
           impact-angle (double-integrator) law near the target. Bank comes from the lift
           direction; AoA from the lift magnitude. The gains saturate at x_go = 8 km and the
           last command is held over the final 2 km (the 1/x^2 gains would otherwise turn
           navigation noise into bank/AoA thrashing and load-factor spikes).

Known limitations (measured, 2026-11-04): one reversal leaves no lateral degree of freedom
after it except the bank magnitude, so lateral errors that build up late (e.g. from a 1 %
accelerometer scale error drifting over minutes) cannot be removed; a corrective S-turn
window (corr_window) chattered under navigation noise and is off. The terminal phase needs
a fast attitude loop (it degrades under a 5 deg/s^2 bank acceleration limit).

Aerodynamic and density uncertainty is handled as in FNPEG: the ratios of the sensed lift
and drag accelerations to the onboard-model values are low-pass filtered (fading memory)
and scale the predictor's forces. The AoA schedule alpha(V) and the initial corrector
guess come from the offline reference design (guidance/design_reference.py).
"""
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from ..physics import constants as C
from ..envs.entry_env_base import _E0_NOM, _EF_NOM, _MUBAR
from ..physics.geodesy import great_circle_distance_m
from . import fastphys as fp
from .reference import ReferenceTrack, REF_CSV

_DEG = np.pi / 180.0
FNPEG_REF = REF_CSV.parent / "fnpeg_reference.csv"     # guidance/design_reference.py


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class FnpegConfig:
    v_pull_mps: float = 6800.0      # phase 0 needs a (predicted) pull-out above this speed
    sigma_min_deg: float = 15.0     # least bank magnitude in phase 1 (lateral authority)
    dalpha_max_deg: float = 25.0    # AoA reduction from the reference schedule (range extension,
    alpha_lo_deg: float = 12.0      #   a 4th corrector parameter, not below alpha_lo), ramped in
    v_hi_mps: float = 6800.0        #   between v_hi and v_lo so the heat pulse is flown at the
    v_lo_mps: float = 6000.0        #   reference AoA
    w_dalpha_km: float = 0.1        # preference residual on the AoA reduction [km per deg]
    corr_window: bool = False       # allow a corrective left-bank window after the reversal
    w_win_km: float = 50.0          #   (S-turn); preference residual on its width
                                    #   [km per unit energy-to-go]. Off: it chattered under
                                    #   navigation noise and cost nominal accuracy (2026-11-04)
    solve_dt_s: float = 2.0         # corrector cycle (the command is updated every decision)
    w_psi_km: float = 3.0           # heading residual weight [km per deg]
    lm_mu: float = 1e-3             # Levenberg-Marquardt damping (relative)
    step_relax: float = 0.5         # under-relaxation of the corrector step (averages the
                                    #   cycle-to-cycle effect of navigation noise)
    filt_tau_s: float = 10.0        # fading-memory filter of sensed/model L and D
    filt_min_D_g: float = 0.02      # filter update only above this modeled drag
    qdot_margin: float = 0.0        # QEGC bound margins (fraction of the limit)
    n_margin: float = 0.03
    qbar_margin: float = 0.03
    x_term_km: float = 150.0        # terminal phase: predicted path length to go
    l_lat_km: float = 15.0          # terminal path-following lengths (-> x_go / sqrt 6 near
    l_ver_km: float = 15.0          #   the end, where the law becomes the impact-angle law)
    zeta: float = 0.816
    blend_km: float = 20.0          # final heading / FPA / altitude blend length
    x_min_km: float = 8.0           # terminal law distance floor (gain saturation: the 1/x^2
                                    #   gains of the impact-angle law amplify navigation noise)
    x_freeze_km: float = 2.0        # hold the last terminal command over this final distance
    cmd_tau_s: float = 0.0          # first-order smoothing of the terminal commands
    meas_tau_s: float = 0.0         # first-order filter of the terminal tracking errors
    n_margin_term: float = 0.06     # load-factor margin of the terminal AoA limit
    alpha_min_term_deg: float = 8.0

    def to_dict(self):
        return asdict(self)


class FnpegGuidance:
    def __init__(self, cfg=None, track=None):
        self.cfg = cfg or FnpegConfig()
        ref = track or ReferenceTrack(path=FNPEG_REF)
        V, a = ref.alpha_V
        o = np.argsort(V)
        self._aV, self._aA = np.ascontiguousarray(V[o]), np.ascontiguousarray(a[o])
        meta_f = FNPEG_REF.with_suffix(".json")
        m = json.loads(meta_f.read_text()) if meta_f.exists() else {}
        # (sigma0 [rad], sigma_f [rad], e_r, dalpha [deg], window top e_wa, window bottom e_wb)
        self._p_ref = np.array([m.get("sigma0_deg", 30.0) * _DEG, m.get("sigma_f_deg", 20.0) * _DEG,
                                m.get("e_rev", 0.6), 0.0, -1.0, -1.0])
        self.log = False

    # -- episode setup -----------------------------------------------------------
    def reset(self, env):
        self.phase = 0
        self.p = self._p_ref.copy()
        self.reversed = False
        self.kL = self.kD = 1.0
        self._t_prev = None
        self._t_solve = -1e9
        self._t_pull, self._pull_ahead = -1e9, True
        self.x_path = np.inf                      # predicted ground path length to go [m]
        self.res = np.full(3, np.nan)
        self.term = None
        self.n_pred = 0
        self.history = []

    # -- helpers -------------------------------------------------------------------
    def _filter(self, z, c, t):
        rho = fp.rho_us76(c["alt"])
        CL, CD = fp.cl_cd(np.rad2deg(z[7]))
        q = C.KFORCE * rho * z[3] ** 2
        Lm, Dm = q * CL, q * CD
        dt = 0.0 if self._t_prev is None else max(t - self._t_prev, 0.0)
        self._t_prev = t
        if Dm > self.cfg.filt_min_D_g and dt > 0.0:
            b = 1.0 - np.exp(-dt / self.cfg.filt_tau_s)
            self.kL += b * (c["L"] / Lm - self.kL)
            self.kD += b * (c["D"] / Dm - self.kD)

    def _alpha_n_limit(self, c, alpha_now_deg, n_lim):
        """Largest AoA whose load factor at the current dynamic pressure stays <= n_lim."""
        CL, CD = fp.cl_cd(alpha_now_deg)
        if c["n"] <= 1e-6:
            return C.ALPHA_MAX
        per_cr = c["n"] / np.hypot(CL, CD)
        CLm, CDm = fp.cl_cd(C.ALPHA_MAX)
        if per_cr * np.hypot(CLm, CDm) <= n_lim:
            return C.ALPHA_MAX
        lo, hi = C.ALPHA_MIN, C.ALPHA_MAX
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            a, d = fp.cl_cd(mid)
            if per_cr * np.hypot(a, d) > n_lim:
                hi = mid
            else:
                lo = mid
        return lo

    def _sigma_qegc(self, z, alpha_deg):
        """Bank-magnitude bound of quasi-equilibrium glide at the path-constraint boundary."""
        cfg = self.cfg
        V = z[3] * C.V_SCALE
        CL, CD = fp.cl_cd(alpha_deg)
        k = max(self.kD, 1e-3)                    # sensed density ratio ~ kD
        rho_Q = ((1.0 - cfg.qdot_margin) * C.QDOT_MAX / (C.KQ * V ** C.HEAT_RATE_VEL_EXP)) ** 2
        rho_q = 2.0 * (1.0 - cfg.qbar_margin) * C.QBAR_MAX / V ** 2
        rho_n = (1.0 - cfg.n_margin) * C.N_MAX / (C.KFORCE * z[3] ** 2 * np.hypot(CL, CD) * k)
        rho_lim = min(min(rho_Q, rho_q) / k, rho_n)
        L_lim = self.kL * C.KFORCE * rho_lim * z[3] ** 2 * CL
        need = _MUBAR / z[0] ** 2 - z[3] ** 2 / z[0]
        if L_lim <= max(need, 0.0):
            return 0.0
        return float(np.arccos(np.clip(need / L_lim, -1.0, 1.0)))

    def _alpha(self, V, dalpha):
        cfg = self.cfg
        return fp.alpha_cmd(V, dalpha, self._aV, self._aA, cfg.v_hi_mps, cfg.v_lo_mps, cfg.alpha_lo_deg)

    def _predict(self, x0, e_now, p, h_tgt, rec=0):
        cfg = self.cfg
        self.n_pred += 1
        if self.reversed:                             # right bank, but for the corrective window
            e_r, wa, wb = 2.0, p[4], p[5]
        else:
            e_r, wa, wb = p[2], -1.0, -1.0
        st, xf, tr = fp.predict3d(x0, e_now, p[0], p[1], e_r, self.kL, self.kD, h_tgt, _E0_NOM, _EF_NOM,
                                  self._aV, self._aA, p[3], cfg.v_hi_mps, cfg.v_lo_mps, cfg.alpha_lo_deg, rec,
                                  wa, wb)
        return st, xf, tr

    def _residual(self, env, st, xf, p):
        lat = xf[2]
        north = np.rad2deg(xf[2] - env.tgt_lat) * 111.2
        east = np.rad2deg(_wrap(xf[1] - env.tgt_lon)) * 111.2 * np.cos(lat)
        dpsi = np.rad2deg(_wrap(xf[5] - env.tgt_psi))
        win = max(p[4] - p[5], 0.0) if self.reversed else 0.0
        r = np.array([north, east, self.cfg.w_psi_km * dpsi, self.cfg.w_dalpha_km * p[3],
                      self.cfg.w_win_km * win])
        if st == 1:
            r[0] += 5000.0                         # skip-out: far overshoot
        return r

    def _solve(self, env, x0, e_now):
        """One damped, under-relaxed Gauss-Newton step from finite differences. Before the
        reversal: (sigma0, sigma_f, e_r, dalpha). After it: (sigma0, sigma_f, dalpha) and a
        corrective left-bank window (e_wa >= e > e_wb), the S-turn that restores the lateral
        degree of freedom lost at the reversal. The AoA reduction and the window width carry
        small preference residuals, so they are used only where the bank magnitude alone
        cannot null the predicted errors."""
        cfg = self.cfg
        if self.reversed:
            free = [0, 1, 3, 4, 5] if cfg.corr_window else [0, 1, 3]
        else:
            free = [0, 1, 2, 3]
        lo = np.array([cfg.sigma_min_deg * _DEG, 0.0, 0.0, 0.0, 0.0, 0.0])
        hi = np.array([C.BANK_MAX, C.BANK_MAX, e_now, cfg.dalpha_max_deg, e_now, e_now])
        p = np.clip(self.p, lo, hi)
        p[5] = min(p[5], p[4])
        if not self.reversed and p[3] <= 0.0 and p[0] > lo[0]:
            # Before the reversal (4 parameters, 3 residuals) the AoA reduction engages only
            # with the bank magnitude at its floor (range extension): free, it drifted along the
            # weakly determined bank/AoA trade, and a level preference strong enough to stop that
            # pulled it back to zero late in low-L/D flights. After the reversal it is the third
            # degree of freedom beside the two bank magnitudes, and always free.
            free = [j for j in free if j != 3]
        st, xf, tr = self._predict(x0, e_now, p, env.tgt_alt, rec=10)
        r = self._residual(env, st, xf, p)
        if len(tr) > 1:                           # predicted ground path length to go
            d = great_circle_distance_m(tr[:-1, 3], tr[:-1, 2], tr[1:, 3], tr[1:, 2])
            self.x_path = float(np.sum(d))
        steps = np.array([0.5 * _DEG, 0.5 * _DEG, 0.003, 0.5, 0.003, -0.003])
        J = np.zeros((5, len(free)))
        for k, j in enumerate(free):
            q = p.copy()
            v = q[j] + steps[j]
            q[j] = v if lo[j] <= v <= hi[j] else q[j] - steps[j]
            if j == 5:
                q[5] = min(q[5], q[4])
            if q[j] == p[j]:
                continue
            sq, xq, _ = self._predict(x0, e_now, q, env.tgt_alt)
            J[:, k] = (self._residual(env, sq, xq, q) - r) / (q[j] - p[j])
        # Parameters at a bound that the descent direction pushes further out are held, and the
        # LM damping is Marquardt-scaled (diag of J'J): the columns have different units (rad,
        # energy, deg), and an identity damping scaled by the trace (the bank columns) crushed
        # the AoA-reduction step, so low-L/D cases ran out of range (2026-11-04).
        g = J.T @ r
        act = [k for k, j in enumerate(free)
               if np.any(J[:, k]) and not ((p[j] <= lo[j] and g[k] > 0.0) or (p[j] >= hi[j] and g[k] < 0.0))]
        dp = np.zeros(len(free))
        if act:
            Ja = J[:, act]
            A = Ja.T @ Ja
            A += cfg.lm_mu * np.diag(np.diag(A))
            dp[act] = np.linalg.solve(A, -Ja.T @ r)
        lim = np.array([10.0 * _DEG, 10.0 * _DEG, 0.05, 5.0, 0.05, 0.05])[free]
        dp = np.clip(cfg.step_relax * dp, -lim, lim)
        p[free] = p[free] + dp
        p = np.clip(p, lo, hi)
        p[5] = min(p[5], p[4])
        self.p = p
        self.res = r
        return r

    def _pullout_ahead(self, z, t):
        """Whether the lift-up descent is predicted to pull out (gamma >= 0) above v_pull;
        re-evaluated every corrector cycle."""
        if t - self._t_pull >= self.cfg.solve_dt_s:
            self._t_pull = t
            self._pull_ahead = fp.pullout3d(np.array(z[:6], dtype=float), self.kL, self.kD, self._aV,
                                            self._aA, self.cfg.v_pull_mps) > 0.0
        return self._pull_ahead

    def _check_reversal(self, e_now):
        if not self.reversed and e_now <= self.p[2]:
            self.reversed = True
            self.p[4] = self.p[5] = 0.5 * e_now        # closed corrective window (no effect)

    # -- guidance law -------------------------------------------------------------
    def command(self, env):
        """-> (bank [rad], AoA [deg]) from the navigated state of the env."""
        cfg = self.cfg
        z, c = env._z_nav, env._c_nav
        t = env.t
        self._filter(z, c, t)
        alpha_now = float(np.rad2deg(z[7]))
        V = z[3] * C.V_SCALE
        n_lim = (1.0 - cfg.n_margin) * C.N_MAX
        if self.phase == 0 and (z[4] >= 0.0 or V < cfg.v_pull_mps or not self._pullout_ahead(z, t)):
            # a shallow entry can settle into equilibrium glide without a pull-out (gamma stays
            # a few tenths of a degree negative): phase 0 is flown only while the onboard model
            # predicts a pull-out within the heat-pulse segment of the AoA schedule
            self.phase = 1
        if self.phase == 1 and self.x_path < cfg.x_term_km * 1e3:
            self._start_terminal(env, z)
        if self.phase == 2:
            return self._terminal(env, z, c, alpha_now, n_lim)

        a_nl = self._alpha_n_limit(c, alpha_now, n_lim)
        if self.phase == 0:
            return 0.0, float(np.clip(min(self._alpha(V, 0.0), a_nl), C.ALPHA_MIN, C.ALPHA_MAX))
        e_now = env.energy_togo(z)
        x0 = np.array(z[:6], dtype=float)
        self._check_reversal(e_now)
        if t - self._t_solve >= cfg.solve_dt_s:
            self._t_solve = t
            self._solve(env, x0, e_now)
            self._check_reversal(e_now)
        dalpha = float(self.p[3])
        alp = min(self._alpha(V, dalpha), a_nl)
        mag = min(self.p[0], self._sigma_qegc(z, alp))
        left = (not self.reversed) or (self.p[5] < e_now <= self.p[4])
        sig = -mag if left else mag
        if self.log:
            self.history.append({"t": t, "phase": 1, "e": e_now, "x_path_km": self.x_path / 1e3,
                                 "sig0_deg": np.rad2deg(self.p[0]), "sigf_deg": np.rad2deg(self.p[1]),
                                 "e_r": self.p[2], "rev": self.reversed, "dalpha": dalpha,
                                 "win": f"{self.p[4]:.3f}/{self.p[5]:.3f}" if self.reversed else "",
                                 "mag_deg": np.rad2deg(mag), "r_n": self.res[0], "r_e": self.res[1],
                                 "r_psi": self.res[2], "kL": self.kL, "kD": self.kD})
        return float(np.clip(sig, C.BANK_MIN, C.BANK_MAX)), float(np.clip(alp, C.ALPHA_MIN, C.ALPHA_MAX))

    # -- terminal phase -------------------------------------------------------------
    def _start_terminal(self, env, z):
        """Freeze the last prediction (1 s samples) as the terminal reference path."""
        cfg = self.cfg
        e_now = env.energy_togo(z)
        _, _, tr = self._predict(np.array(z[:6], dtype=float), e_now, self.p, env.tgt_alt, rec=1)
        df = pd.DataFrame({"t_s": tr[:, 0], "h_km": tr[:, 7] / 1e3, "V_mps": tr[:, 4] * C.V_SCALE,
                           "lat_deg": np.rad2deg(tr[:, 3]), "lon_deg": np.rad2deg(tr[:, 2]),
                           "sigma_deg": 0.0, "alpha_deg": 0.0})
        T = ReferenceTrack(df=df)
        T.set_target(env.tgt_lat, env.tgt_lon)
        T.reset_search()
        Lb = cfg.blend_km * 1e3
        self._dpsi_f = float(_wrap(env.tgt_psi - T.heading(T.S)))
        self._dm_f = float(np.tan(env.tgt_fpa) - T.slope[-1])
        self._ch = float(env.tgt_alt - T.h[-1] - self._dm_f * Lb / 2.0)
        self._cy = -self._dpsi_f * Lb / 2.0
        self.term = T
        self._term_init = False
        self._cmd, self._t_cmd = None, env.t
        self._ef, self._t_ef = None, env.t
        self.phase = 2

    def _terminal(self, env, z, c, alpha_now, n_lim):
        """3-D path following of the terminal reference with curvature feedforward; the lengths
        shrink to x_go / sqrt(6) near the end, where each channel becomes the impact-angle
        (double-integrator) law on the reference blended into the target altitude, flight-
        path angle and heading."""
        cfg, tr = self.cfg, self.term
        s, y, _ = tr.project(z[2], z[1])
        x_go = (tr.S - s) * C.R0
        V = z[3] * C.V_SCALE
        gam, psi = z[4], z[5]
        Vh = V * np.cos(gam)
        x = max(x_go, cfg.x_min_km * 1e3)
        Lb = cfg.blend_km * 1e3
        w = max(0.0, 1.0 - x_go / Lb)
        q2 = (Lb - x_go) ** 2 / (2.0 * Lb) if x_go < Lb else 0.0
        z6 = np.sqrt(6.0)
        h_r, m_r, dm_r = tr.vertical(s)
        h_mod = h_r + self._ch + self._dm_f * q2
        m_mod = m_r + self._dm_f * w
        dm_mod = dm_r + (self._dm_f / Lb if x_go < Lb else 0.0)
        y_mod = self._cy + self._dpsi_f * q2
        psi_mod = tr.heading(s) + self._dpsi_f * w
        kappa = tr.curvature(s, 2e3) + (self._dpsi_f / Lb if x_go < Lb else 0.0)
        if not self._term_init:
            # reshape once at the switch: cubic offsets (in x_go) from the vehicle's altitude /
            # slope and cross-track / heading to zero (with zero slope) at the target
            self._x0 = x
            e0, d0 = c["alt"] - h_mod, np.tan(gam) - m_mod
            self._ch3 = (-x * d0 - 2.0 * e0, 3.0 * e0 + x * d0)
            e0, d0 = y * C.R0 - y_mod, np.tan(_wrap(psi - psi_mod))
            self._cy3 = (-x * d0 - 2.0 * e0, 3.0 * e0 + x * d0)
            self._term_init = True

        def shape(cf):
            a, b = cf
            u = min(x / self._x0, 1.0)
            return (a * u ** 3 + b * u ** 2, -(3.0 * a * u ** 2 + 2.0 * b * u) / self._x0,
                    (6.0 * a * u + 2.0 * b) / self._x0 ** 2)

        oh, om, odm = shape(self._ch3)
        oy, oth, ok = shape(self._cy3)
        lv = min(cfg.l_ver_km * 1e3, x / z6)
        e_h, e_m = c["alt"] - (h_mod + oh), np.tan(gam) - (m_mod + om)
        th = _wrap(psi - psi_mod - np.arctan(oth))
        e_y = y * C.R0 - (y_mod + oy)
        if cfg.meas_tau_s > 0.0:
            # first-order filter of the measured tracking errors (navigation noise)
            e = np.array([e_h, e_m, e_y, th])
            if self._ef is None:
                self._ef = e
            else:
                b = 1.0 - np.exp(-max(env.t - self._t_ef, 0.0) / cfg.meas_tau_s)
                self._ef = self._ef + b * (e - self._ef)
            self._t_ef = env.t
            e_h, e_m, e_y, th = self._ef
        dd = dm_mod + odm - 2.0 * cfg.zeta * e_m / lv - e_h / lv ** 2
        gdot = V * np.cos(gam) ** 3 * dd
        ll = min(cfg.l_lat_km * 1e3, x / z6)
        thdot = Vh * np.cos(th) ** 2 * (-2.0 * cfg.zeta * np.tan(th) / ll - e_y / ll ** 2)
        pdot = (kappa + ok) * Vh + thdot
        g_free, p_free = fp.free_rates(z[0], z[2], z[3], gam, psi)
        Av = (gdot - g_free) * V / C.G0                      # vertical lift [g]
        Ay = (pdot - p_free) * V * np.cos(gam) / C.G0        # lateral lift [g]
        sig = float(np.clip(np.arctan2(Ay, Av), C.BANK_MIN, C.BANK_MAX))
        Lreq = float(np.hypot(Av, Ay))
        CLn, _ = fp.cl_cd(alpha_now)
        per_cl = c["L"] / max(CLn, 1e-6)                      # sensed lift per unit CL
        a_req = fp.alpha_for_cl(Lreq / max(per_cl, 1e-9))
        a_lim = self._alpha_n_limit(c, alpha_now, (1.0 - cfg.n_margin_term) * C.N_MAX)
        alp = float(np.clip(a_req, cfg.alpha_min_term_deg, min(C.ALPHA_MAX, a_lim)))
        # command conditioning against navigation noise: first-order smoothing, and a hold of
        # the last command over the final x_freeze (where the 1/x^2 gains would amplify noise)
        if x_go < cfg.x_freeze_km * 1e3 and self._cmd is not None:
            sig, alp = self._cmd
        elif cfg.cmd_tau_s > 0.0 and self._cmd is not None:
            b = 1.0 - np.exp(-max(env.t - self._t_cmd, 0.0) / cfg.cmd_tau_s)
            sig = self._cmd[0] + b * (sig - self._cmd[0])
            alp = self._cmd[1] + b * (alp - self._cmd[1])
        self._cmd, self._t_cmd = (sig, alp), env.t
        if self.log:
            self.history.append({"t": env.t, "phase": 2, "x_go_km": x_go / 1e3, "y_km": e_y / 1e3,
                                 "th_deg": np.rad2deg(th), "e_h_m": e_h, "e_m": e_m, "Av": Av, "Ay": Ay,
                                 "Lreq": Lreq, "Lnow": c["L"], "sig_deg": np.rad2deg(sig), "a_req": a_req})
        return sig, float(np.clip(alp, C.ALPHA_MIN, C.ALPHA_MAX))
