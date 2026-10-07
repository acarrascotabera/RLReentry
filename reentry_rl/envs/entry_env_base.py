"""Shared 3-DOF re-entry environment (Gymnasium).

Wraps the validated physics core. The agent commands attitude *rates* (DESIGN
decision 2), which are integrated in-env over a fixed Δt with RK4 sub-stepping.
The augmented integration state is

    z = [r_nd, lon, lat, V_nd, gamma, psi, sigma, (alpha)]

with sigma always a state (driven by the bank-rate action) and alpha a state
only when `optimize_aoa` (Stage 2+); otherwise alpha follows the WB001 schedule.

Subclasses set `optimize_aoa` and `n_actions` and may extend the observation.

Each episode flies one `Scenario` (entry-state / target offsets and model
factors, see scenario.py): passed as reset(options={"scenario": s}), drawn
from `scenario_sampler(rng)`, or nominal by default.

With `exact_terminal` (default) the episode ends exactly at the target-
altitude crossing: the crossing is located inside the RK4 substep and the
terminal state is re-integrated to it. Without it, the terminal state is the
first 1 s step below the target altitude — at the ~300 m/s handover speed
that is up to ~0.3 km of horizontal travel, the same order as the best
landing errors, so legacy numbers carry that measurement quantization.

With `rate_saturation` (default) bank and AoA behave as saturated integrators
inside every RK4 stage. Without it (legacy) the bounds were applied only after
each substep, so the stages evaluated the aero at up to h*15 deg/s = 5 deg
beyond ALPHA_MAX: the v10-HPO policies commanded +3..+10 deg/s into the 40 deg
limit and flew an effective AoA of ~40.5-41.6 deg (HPO-best: 0.42 km legacy,
259 km and q-bar/n ratio 2.0 with the bound enforced).
"""
from dataclasses import replace

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..physics import constants as C
from ..physics.atmosphere import atmosphere
from ..physics.geodesy import (altitude_m, r_nd_from_alt,
                               great_circle_distance_m, bearing_rad)
from ..physics.aero_wb001 import nominal_aoa_deg, cl_cd, path_quantities
from ..physics.eom_3dof import translational_rhs
from .rewards import (RewardWeights, path_barrier, position_penalty, fpa_penalty,
                      psi_penalty, smooth_gate, concave_bonus)
from .scenario import Scenario, NOMINAL

_DEG = np.pi / 180.0

# Observation layouts. Older policies keep their own layout so they can still
# be rolled out; new features are only ever APPENDED (see expand_obs.py).
#   v13 : 12 base features (+ AoA in Stage 2)
#   v15 : + log10 range-to-go (added at campaign v15)
#   v20 : + 8 terminal-state features: FPA and heading errors to the handover
#         target (linear and signed-log, resolution down to the 0.1 % tolerance),
#         log altitude-to-go (the episode ends on altitude), and the bearing to
#         the target relative to the required final heading (approach geometry)
OBS_VERSIONS = ("v13", "v15", "v20")


def _slog(x, scale):
    """Signed log magnitude: per-decade resolution of x down to `scale`."""
    return float(np.sign(x) * np.log10(1.0 + abs(x) / scale))


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class EntryEnvBase(gym.Env):
    metadata = {"render_modes": []}

    optimize_aoa = False          # overridden by Stage 2
    n_actions = 1                 # overridden per stage

    def __init__(self, weights: RewardWeights = None, dt=1.0, n_substeps=3,
                 max_steps=2500, j_ref=None, seed=None, log_range_obs=True,
                 obs_version=None, exact_terminal=True, rate_saturation=True,
                 scenario_sampler=None):
        super().__init__()
        # private copy: DummyVecEnv hands every env the same weights object, and the
        # curriculum changes weights per env (set_reward_params)
        self.w = replace(weights) if weights is not None else RewardWeights()
        self._pending_w = {}
        self.dt = float(dt)
        self.n_substeps = int(n_substeps)
        self.max_steps = int(max_steps)
        # log_range_obs is the legacy switch (False = pre-v15 13-feature layout)
        self.obs_version = obs_version or ("v15" if log_range_obs else "v13")
        if self.obs_version not in OBS_VERSIONS:
            raise ValueError(f"obs_version must be one of {OBS_VERSIONS}")
        self.exact_terminal = bool(exact_terminal)
        self.rate_saturation = bool(rate_saturation)
        self.scenario_sampler = scenario_sampler
        self.J_ref = float(j_ref) if j_ref is not None else C.J_REF_BANK_ONLY

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self._obs_dim(),),
                                            dtype=np.float32)

        self._set_target(NOMINAL)
        # observation normalizer: the NOMINAL mission range, kept fixed under
        # dispersions so the range-fraction feature means the same for all scenarios
        self.range0 = float(great_circle_distance_m(
            C.IC["lat_deg"] * _DEG, C.IC["lon_deg"] * _DEG, self.tgt_lat, self.tgt_lon))

        self._rng = np.random.default_rng(seed)
        self.scenario = NOMINAL
        self._model = None
        self._z = None
        self._c = None

    # -- to be specialized -------------------------------------------------
    def _obs_dim(self):
        n = 12 + (1 if self.optimize_aoa else 0)
        if self.obs_version != "v13":
            n += 1
        if self.obs_version == "v20":
            n += 8
        return n

    # -- curriculum hook ------------------------------------------------------
    def set_reward_params(self, **params):
        """Queue reward-weight changes; applied at the next reset so a potential
        never changes inside an episode (that would break the telescoping)."""
        for k in params:
            if not hasattr(self.w, k):
                raise AttributeError(f"RewardWeights has no field {k!r}")
        self._pending_w.update(params)
        return True

    def _n_state(self):
        return 7 + (1 if self.optimize_aoa else 0)

    # -- alpha resolution --------------------------------------------------
    def _alpha_deg(self, z=None):
        z = self._z if z is None else z
        if self.optimize_aoa:
            return float(np.rad2deg(z[7]))
        return float(nominal_aoa_deg(z[3] * C.V_SCALE))

    # -- dynamics ----------------------------------------------------------
    def _deriv(self, z, sdot, adot):
        sigma = z[6]
        alpha_rad = z[7] if self.optimize_aoa else None
        if self.rate_saturation:
            # saturated integrator: forces use the bounded attitude and an outward
            # rate at a bound is zero. Without this the RK4 stages evaluate the aero
            # at z + c*h*rate, i.e. up to h*15 deg/s = 5 deg beyond ALPHA_MAX, and
            # a rate command "into" the limit silently buys extra AoA.
            lo, hi = C.BANK_MIN, C.BANK_MAX
            if (sigma >= hi and sdot > 0.0) or (sigma <= lo and sdot < 0.0):
                sdot = 0.0
            sigma = min(max(sigma, lo), hi)
            if self.optimize_aoa:
                lo, hi = C.ALPHA_MIN * _DEG, C.ALPHA_MAX * _DEG
                if (alpha_rad >= hi and adot > 0.0) or (alpha_rad <= lo and adot < 0.0):
                    adot = 0.0
                alpha_rad = min(max(alpha_rad, lo), hi)
        alpha_deg = np.rad2deg(alpha_rad) if self.optimize_aoa else nominal_aoa_deg(z[3] * C.V_SCALE)
        d6 = translational_rhs(z[0], z[1], z[2], z[3], z[4], z[5], sigma, alpha_deg, self._model)
        dz = np.zeros_like(z)
        dz[0:6] = d6
        dz[6] = sdot
        if self.optimize_aoa:
            dz[7] = adot
        return dz

    def _rk4(self, z, h, sdot, adot):
        k1 = self._deriv(z, sdot, adot)
        k2 = self._deriv(z + 0.5 * h * k1, sdot, adot)
        k3 = self._deriv(z + 0.5 * h * k2, sdot, adot)
        k4 = self._deriv(z + h * k3, sdot, adot)
        z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z[6] = np.clip(z[6], C.BANK_MIN, C.BANK_MAX)
        if self.optimize_aoa:
            z[7] = np.clip(z[7], C.ALPHA_MIN * _DEG, C.ALPHA_MAX * _DEG)
        return z

    def _integrate(self, sdot, adot):
        """Advance one guidance interval. Returns the elapsed time, which is
        shorter than dt when the target-altitude crossing ends the step."""
        z = self._z
        h = self.dt / self.n_substeps
        for k in range(self.n_substeps):
            z_prev = z
            z = self._rk4(z_prev, h, sdot, adot)
            if self.exact_terminal and altitude_m(z[0], z[2]) <= self.tgt_alt:
                tau = self._crossing_fraction(z_prev, z, h, sdot, adot)
                self._z = self._rk4(z_prev, tau * h, sdot, adot)
                self._crossed = True
                return (k + tau) * h
        self._z = z
        return self.dt

    def _crossing_fraction(self, z0, z1, h, sdot, adot, iters=3):
        """Fraction tau in (0, 1] of the substep at which altitude reaches the
        target: secant iterations on alt(rk4(z0, tau*h)) - h_tgt. The altitude
        is smooth and nearly linear over a 1/3 s substep, so 3 iterations
        reach sub-millimetre accuracy."""
        f0 = altitude_m(z0[0], z0[2]) - self.tgt_alt          # > 0
        f1 = altitude_m(z1[0], z1[2]) - self.tgt_alt          # <= 0
        t0, t1 = 0.0, 1.0
        for _ in range(iters):
            if f1 == f0:
                break
            t = float(np.clip(t1 - f1 * (t1 - t0) / (f1 - f0), 1e-9, 1.0))
            zt = self._rk4(z0, t * h, sdot, adot)
            ft = altitude_m(zt[0], zt[2]) - self.tgt_alt
            t0, f0, t1, f1 = t1, f1, t, ft
            if abs(ft) < 1e-4:
                break
        return t1

    # -- derived quantities cache -----------------------------------------
    def _update_cache(self):
        z = self._z
        alt = altitude_m(z[0], z[2])
        Vmps = z[3] * C.V_SCALE
        rho, _T, a_snd, _d = atmosphere(alt)
        CL, CD = cl_cd(self._alpha_deg(z))
        kforce = C.KFORCE
        if self._model is not None:
            m = self._model
            rho, CL, CD, kforce = rho * m.k_rho, CL * m.k_CL, CD * m.k_CD, C.KFORCE / m.k_mass
        Lnd = kforce * rho * z[3] ** 2 * CL
        Dnd = kforce * rho * z[3] ** 2 * CD
        Qdot, qbar, n = path_quantities(rho, Vmps, Lnd, Dnd)
        self._c = dict(alt=alt, Vmps=Vmps, a=a_snd, Qdot=Qdot, qbar=qbar, n=n)

    def _range_to_go(self):
        z = self._z
        return float(great_circle_distance_m(z[2], z[1], self.tgt_lat, self.tgt_lon))

    def _heading_error(self):
        z = self._z
        brg = bearing_rad(z[2], z[1], self.tgt_lat, self.tgt_lon)
        return float(_wrap_pi(brg - z[5]))

    def _landing_error_km(self):
        return self._range_to_go() / 1000.0

    # -- scenario ----------------------------------------------------------
    def _set_target(self, sc: Scenario):
        tg = C.TARGET
        self.tgt_lat = (tg["lat_deg"] + sc.tc_lat_deg) * _DEG
        self.tgt_lon = (tg["lon_deg"] + sc.tc_lon_deg) * _DEG
        self.tgt_alt = tg["alt_m"] + sc.tc_alt_m
        self.tgt_fpa = (tg["fpa_deg"] + sc.tc_fpa_deg) * _DEG
        self._tgt_psi = (tg["heading_deg"] + sc.tc_psi_deg) * _DEG

    # -- Gymnasium API -----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        for k, v in self._pending_w.items():
            setattr(self.w, k, v)
        self._pending_w = {}
        sc = (options or {}).get("scenario")
        if sc is None and self.scenario_sampler is not None:
            sc = self.scenario_sampler(self._rng)
        if isinstance(sc, dict):
            sc = Scenario.from_dict(sc)
        self.scenario = sc if sc is not None else NOMINAL
        self._model = self.scenario.model_factors()
        self._set_target(self.scenario)

        ic, s = C.IC, self.scenario
        lat0 = (ic["lat_deg"] + s.ic_lat_deg) * _DEG
        V0 = ic["V_mps"] + s.ic_V_mps
        z = np.zeros(self._n_state(), dtype=float)
        z[0] = r_nd_from_alt(ic["alt_m"] + s.ic_alt_m, lat0)
        z[1] = (ic["lon_deg"] + s.ic_lon_deg) * _DEG
        z[2] = lat0
        z[3] = V0 / C.V_SCALE
        z[4] = (ic["fpa_deg"] + s.ic_fpa_deg) * _DEG
        z[5] = (ic["heading_deg"] + s.ic_psi_deg) * _DEG
        z[6] = 0.0                                   # initial bank
        if self.optimize_aoa:
            z[7] = nominal_aoa_deg(V0) * _DEG
        self._z = z
        self._crossed = False
        self.t = 0.0
        self.steps = 0
        self.J = 0.0
        self._update_cache()
        self._prev_qdot = self._c["Qdot"]
        self._prev_d_km = self._range_to_go() / 1000.0
        self._prev_action = np.zeros(self.n_actions)
        self._chatter = 0.0
        self._prev_G = self._angle_potential()
        return self._obs(), self._info()

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        sdot = float(np.clip(a[0], -1.0, 1.0)) * C.SIGMA_DOT_MAX
        adot = (float(np.clip(a[1], -1.0, 1.0)) * C.ALPHA_DOT_MAX) if self.optimize_aoa else 0.0

        dt = self._integrate(sdot, adot)
        self.t += dt
        self.steps += 1
        self._update_cache()

        # heat-load increment over the step (trapezoid on Qdot)
        dJ = 0.5 * (self._prev_qdot + self._c["Qdot"]) * dt
        self.J += dJ
        self._prev_qdot = self._c["Qdot"]

        reward, terminated, truncated, tinfo = self._reward(dJ, a)
        info = self._info()
        info.update(tinfo)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    # -- reward / termination ---------------------------------------------
    def _reward(self, dJ, action):
        w, z, c = self.w, self._z, self._c

        # dense potential-based shaping on the LOG position penalty: distributes the
        # terminal position objective over the episode for credit assignment (telescopes
        # to -log_pen(d_final), so only the END position matters). This is what drives
        # the policy to CLOSE — the no-dense 2M run stalled at 400-700 km without it.
        cur_d_km = self._range_to_go() / 1000.0
        r = position_penalty(self._prev_d_km, w) - position_penalty(cur_d_km, w)
        self._prev_d_km = cur_d_km

        # gated potential shaping of the terminal FPA/heading (tier 1b, 0 when off)
        if w.w_ang_mult > 0.0:
            G = self._angle_potential()
            r += self._prev_G - G
            self._prev_G = G

        # chattering: ACCUMULATE action change; charged once at the end as the
        # episode mean (a per-step sum taxes long glides by episode length x
        # exploration noise and made short dives out-score them — see rewards.py)
        da2 = float(np.sum(np.square(action - self._prev_action)))
        self._chatter += da2
        self._prev_action = np.asarray(action, dtype=float).copy()

        # end-game burst guard: per-step, only at low altitude (bounded phase, so
        # no episode-length tax; altitude-gated so trajectory shape can't dodge
        # it) — prices the terminal bang-bang trim directly
        if c["alt"] < w.endsmooth_alt_m:
            r -= w.w_endsmooth * da2

        # path constraints: capped soft barrier (zero when feasible)
        bar = float(path_barrier(c["Qdot"] / C.QDOT_MAX, w.path_soft)
                    + path_barrier(c["qbar"] / C.QBAR_MAX, w.path_soft)
                    + path_barrier(c["n"] / C.N_MAX, w.path_soft))
        r -= w.w_path * min(bar, w.path_cap)

        terminated = truncated = False
        tinfo = {}
        alt = c["alt"]
        bad = (not np.isfinite(alt)) or (alt > C.ALT_MAX) or (c["Vmps"] < C.V_MIN) \
            or (abs(z[4]) > 89.0 * _DEG)

        # Terminal anchor -w_pos*log10(d_end) is charged on EVERY ending below.
        # v7 charged it only on landing -> skip-out dodged it (learned to bail out);
        # v8 dropped it entirely -> dive-and-land-far (discounting favors ending
        # the episode early). Uniform anchor kills both attractors: the return is
        # monotone in final distance AND far endings are expensive for all outcomes.
        reached = self._crossed if self.exact_terminal else (alt <= self.tgt_alt)
        if bad:
            terminated = True
            r -= w.w_fail + position_penalty(cur_d_km, w)
            tinfo = {"outcome": "fail", "is_success": False}
        elif reached:
            terminated = True
            d_km = self._landing_error_km()
            success = (d_km <= 1.0)
            # anchor always applies; the landing bonus is CONCAVE (1-(d/ramp)^2)^+
            # — dense gradient from succ_ramp_km in AND anti-dispersion (Jensen),
            # reinforcing the concave near-field well that drives sigma down (v18)
            ramp = max(0.0, 1.0 - (d_km / w.succ_ramp_km) ** 2)
            r += -position_penalty(d_km, w) + w.w_succ * ramp
            tinfo = {"outcome": "reached", "is_success": bool(success),
                     "J_heat": self.J, "tf_s": self.t}
        elif self.steps >= self.max_steps:
            truncated = True
            r -= w.w_fail + position_penalty(cur_d_km, w)
            tinfo = {"outcome": "timeout", "is_success": False}

        if terminated or truncated:            # chatter charged uniformly on every ending
            chatter_mean = self._chatter / max(self.steps, 1)
            r -= w.w_smooth * chatter_mean
            tinfo["chatter_mean"] = chatter_mean
            terr = self._terminal_errors()
            tinfo.update(terr)
            # tier 1b anchors: uniform on every ending, like the position anchor
            pf, pp = fpa_penalty(terr["dfpa_deg"], w), psi_penalty(terr["dpsi_deg"], w)
            r -= w.w_ang_mult * (pf + pp)
            if tinfo["outcome"] == "reached":
                r += (w.w_succ_fpa * concave_bonus(terr["dfpa_deg"], w.fpa_ramp_deg)
                      + w.w_succ_psi * concave_bonus(terr["dpsi_deg"], w.psi_ramp_deg))
            tinfo["pen_fpa"], tinfo["pen_psi"] = pf, pp

        return r, terminated, truncated, tinfo

    def _terminal_errors(self):
        """Signed terminal errors against the episode's target (the SCvx
        equality-constrained set: h, lat, lon, gamma, psi). The legacy
        absolute keys fpa_err_deg / psi_err_deg are kept for old scripts."""
        z, c = self._z, self._c
        dfpa = float(np.rad2deg(z[4] - self.tgt_fpa))
        dpsi = float(np.rad2deg(_wrap_pi(z[5] - self.tgt_psi)))
        return {"d_km": self._landing_error_km(),
                "dh_m": float(c["alt"] - self.tgt_alt),
                "dlat_deg": float(np.rad2deg(z[2] - self.tgt_lat)),
                "dlon_deg": float(np.rad2deg(_wrap_pi(z[1] - self.tgt_lon))),
                "dfpa_deg": dfpa, "dpsi_deg": dpsi,
                "fpa_err_deg": abs(dfpa), "psi_err_deg": abs(dpsi),
                "V_f_mps": float(c["Vmps"])}

    def _angle_potential(self):
        """G(s) = c * [g_psi(range) * pen_psi + g_fpa(alt) * pen_fpa] (0 when tier 1b is off)."""
        w = self.w
        if w.w_ang_mult <= 0.0:
            return 0.0
        z, c = self._z, self._c
        g_psi = smooth_gate(self._range_to_go() / 1000.0, w.psi_gate_hi_km, w.psi_gate_lo_km)
        g_fpa = smooth_gate(c["alt"] / 1000.0, w.fpa_gate_hi_km, w.fpa_gate_lo_km)
        dpsi = float(np.rad2deg(_wrap_pi(z[5] - self.tgt_psi)))
        dfpa = float(np.rad2deg(z[4] - self.tgt_fpa))
        return w.w_ang_mult * (g_psi * psi_penalty(dpsi, w) + g_fpa * fpa_penalty(dfpa, w))

    @property
    def tgt_psi(self):
        return self._tgt_psi

    # -- observation -------------------------------------------------------
    def _obs(self):
        z, c = self._z, self._c
        feats = [
            c["alt"] / 100e3,                       # altitude (~1 -> 0.25)
            c["Vmps"] / 7450.0,                     # velocity  (phase variable)
            z[4] / (20.0 * _DEG),                   # FPA
            np.sin(z[5]), np.cos(z[5]),             # heading (wrap-safe)
            z[6] / C.BANK_MAX,                      # current bank state [-1,1]
            c["qbar"] / C.QBAR_MAX,
            (c["Vmps"] / max(c["a"], 1e-6)) / 25.0,  # Mach
            c["Qdot"] / C.QDOT_MAX,
            c["n"] / C.N_MAX,
            self._range_to_go() / self.range0,      # range-to-go (1 -> 0)
            self._heading_error() / np.pi,          # heading error [-1,1]
        ]
        if self.optimize_aoa:
            feats.append(np.rad2deg(z[7]) / 40.0)
        # log-range: per-decade resolution down to ~100 m, matching the log reward.
        # The linear range fraction above is ~1e-4 at single-km scale — invisible to
        # the nets — which capped the reward-only campaign at ~5 km (v14 post-mortem).
        # Kept LAST so older policies map onto a zero-padded input column.
        if self.obs_version != "v13":
            feats.append(np.log10(max(self._range_to_go() / 1000.0, 0.1)) / 4.0)
        if self.obs_version == "v20":
            dfpa = np.rad2deg(z[4] - self.tgt_fpa)
            dpsi_rad = _wrap_pi(z[5] - self.tgt_psi)
            dpsi = np.rad2deg(dpsi_rad)
            brg = bearing_rad(z[2], z[1], self.tgt_lat, self.tgt_lon) - self.tgt_psi
            feats += [
                dfpa / 10.0,                                  # FPA error to the handover
                _slog(dfpa, 0.01) / 3.0,                      #   resolution to 0.01 deg (0.1 %)
                np.sin(dpsi_rad), np.cos(dpsi_rad),           # heading error (wrap-safe)
                _slog(dpsi, 0.09) / 3.3,                      #   resolution to 0.09 deg (0.1 %)
                np.log10(max(c["alt"] - self.tgt_alt, 1.0)) / 5.0,   # altitude-to-go [m]
                np.sin(brg), np.cos(brg),                     # target bearing vs final heading
            ]
        return np.asarray(feats, dtype=np.float32)

    # -- info (cheap; full physical state for logging / CSV export) --------
    def _info(self):
        z, c = self._z, self._c
        return {
            "t_s": self.t,
            "alt_km": c["alt"] / 1e3,
            "lon_deg": float(np.rad2deg(z[1])),
            "lat_deg": float(np.rad2deg(z[2])),
            "V_mps": c["Vmps"],
            "gamma_deg": float(np.rad2deg(z[4])),
            "psi_deg": float(np.rad2deg(z[5])),
            "sigma_deg": float(np.rad2deg(z[6])),
            "alpha_deg": self._alpha_deg(z),
            "Qdot_Wm2": c["Qdot"],
            "qbar_Pa": c["qbar"],
            "n_g": c["n"],
            "J_heat": self.J,
        }
