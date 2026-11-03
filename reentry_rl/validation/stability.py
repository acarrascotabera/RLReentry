"""Stability and robustness analyses of a trained guidance policy (paper suite).

Every analysis runs the policy closed loop on top of a realistic dispersion level
(base set: the nominal mission + the first 100 cases of S4_scale_x0.1) and adds ONE
perturbation class at a time, swept in magnitude, so each curve isolates one margin:

  nav      navigation errors fed to the policy only (truth propagates unchanged):
           per-episode bias + per-decision white noise on position, speed, FPA and
           heading, and accelerometer scale errors on the sensed lift and drag
           (the drag / L/D / load-factor features). Level k scales a reference
           1-sigma spec NAV_SPEC (each of bias and noise).
  delay    pure time delay between the policy command and the attitude loop
           (time-delay margin)
  lag      extra first-order lag (time constant tau) in series with the attitude
           tracking loop
  rate     reduced bank-rate limit (AoA-rate limit scaled in proportion)
  gain     multiplicative error on the commanded bank angle (gain-margin analogue)
  bias     additive bank-command bias
  impulse  one impulsive state kick (FPA, heading or speed) when the energy-to-go
           crosses e (early, mid, terminal phase)
  density  altitude-varying density perturbations on top of the scenario's k_rho:
           sinusoidal waves (amplitude A, wavelength 10 km, random phase per case)
           and a Gaussian bump around 55 km

plus deterministic studies on the nominal mission:

  sensitivity  closed- vs open-loop terminal sensitivity to small initial-state and
               model errors (open loop = the nominal closed-loop command history
               replayed in time): the error-attenuation ratio of the guidance
  recovery     state histories after an impulsive kick, closed vs open loop
  corridor     entry FPA x entry speed grid (entry corridor)
  footprint    target latitude x longitude offset grid (reachable footprint)
  jacobian     spectral norm of d(command)/d(observation) along the nominal and
               dispersed trajectories (local Lipschitz constant of the policy)
  combined     every realistic perturbation stacked, on the full 300-case 0.1x and
               0.25x sets

All stochastic draws use common random numbers (seed = case id), so conditions differ
only in the perturbation. Outputs go to <outdir>/<study>/; finished files are skipped,
so an interrupted run resumes.

The classical baseline (reentry_rl/guidance/fnpeg.py) flies the same env, perturbations
and draws with --guidance fnpeg (it reads the navigated state the policy would observe).

    python -m reentry_rl.validation.stability --outdir results/stability_v26
    python -m reentry_rl.validation.stability --guidance fnpeg --outdir results/stability_fnpeg
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from collections import deque
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from reentry_rl.physics import constants as C
from reentry_rl.physics.aero_wb001 import ModelFactors
from reentry_rl.physics.eom_3dof import translational_rhs
from reentry_rl.physics.geodesy import altitude_m
from reentry_rl.envs.rewards import RewardWeights
from reentry_rl.envs.scenario import Scenario, NOMINAL, load_set
from reentry_rl.envs.stage2_bank_aoa import Stage2BankAoaEnv
from reentry_rl.validation import evaluate_policy as ep

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = REPO_ROOT / "results/stage2_v26/snap_6M/model.zip"
VN = REPO_ROOT / "results/stage2_v26/snap_6M/vecnormalize.pkl"
_DEG = np.pi / 180.0

# reference 1-sigma navigation spec (INS/GNSS-class entry navigation); applied both as a
# per-episode bias and as per-decision white noise
NAV_SPEC = dict(pos_m=300.0, vel_mps=1.0, fpa_deg=0.05, psi_deg=0.05, acc_frac=0.01)


@dataclass(frozen=True)
class Perturbation:
    nav_level: float = 0.0          # multiple of NAV_SPEC
    delay_s: float = 0.0
    lag_s: float = 0.0
    rate_frac: float = 1.0          # bank / AoA rate limits as a fraction of nominal
    bank_gain: float = 1.0
    bank_bias_deg: float = 0.0
    impulses: tuple = ()            # ((e_togo, d_fpa_deg, d_psi_deg, d_V_mps), ...)
    rho_wave_amp: float = 0.0       # sinusoidal density waves: amplitude (fraction)
    rho_wave_len_m: float = 10e3
    rho_bump_amp: float = 0.0       # Gaussian density bump: amplitude (fraction)
    rho_bump_alt_m: float = 55e3
    rho_bump_width_m: float = 5e3

    def to_dict(self):
        d = asdict(self)
        d["impulses"] = [list(i) for i in self.impulses]
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["impulses"] = tuple(tuple(i) for i in d.get("impulses", ()))
        return cls(**d)


class PerturbedStage2Env(Stage2BankAoaEnv):
    """Stage-2 env with navigation errors, actuator perturbations and disturbances.

    The observation is built from the perceived state; the dynamics, path constraints
    and terminal errors use the true state. Commands pass through gain / bias, a time
    delay and a first-order lag before the (rate-limited) attitude tracking law. Only
    the energy-stepped target-mode configuration of the v24+ policies is supported."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.step_mode != "energy" or self.action_mode != "target":
            raise ValueError("PerturbedStage2Env needs step_mode='energy', action_mode='target'")
        self.pert = Perturbation()
        self._pert_seed = 0
        self._act_on = False

    def set_perturbation(self, pert, seed=0):
        self.pert = pert
        self._pert_seed = int(seed)

    # -- episode setup -------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        p = self.pert
        rng = np.random.default_rng(self._pert_seed)
        self._nav_rng = np.random.default_rng(self._pert_seed + 7919)
        # per-episode navigation biases (standard normal, scaled at use: CRN across levels)
        self._nav_bias = rng.standard_normal(8)
        self._rho_phase = rng.uniform(0.0, 2.0 * np.pi)
        self._act_on = False                    # heat-rate floor uses the ideal actuator
        obs, info = super().reset(seed=seed, options=options)
        self._base_model = self._model if self._model is not None else ModelFactors()
        z = self._z
        self._sig_d = self._sig_e = z[6]
        self._alp_d = self._alp_e = z[7]
        self._queue = deque()
        self._imp_done = [False] * len(p.impulses)
        self._act_on = True
        return obs, info

    # -- navigation errors (observation only) -------------------------------------
    def _obs(self):
        k = self.pert.nav_level
        if k <= 0.0 or self._z is None:
            self._z_nav, self._c_nav = self._z, self._c       # navigated = true state
            return super()._obs()
        s = NAV_SPEC
        b = self._nav_bias
        w = self._nav_rng.standard_normal(8)
        e = k * (b + w)                          # bias + white noise, each 1-sigma = spec
        z_true, c_true = self._z, self._c
        zn = z_true.copy()
        zn[0] += e[0] * s["pos_m"] / C.R0                         # altitude
        zn[2] += e[1] * s["pos_m"] / C.R0                         # north
        zn[1] += e[2] * s["pos_m"] / (C.R0 * max(np.cos(zn[2]), 1e-3))   # east
        zn[3] += e[3] * s["vel_mps"] / C.V_SCALE
        zn[4] += e[4] * s["fpa_deg"] * _DEG
        zn[5] += e[5] * s["psi_deg"] * _DEG
        try:
            self._z = zn
            self._update_cache()
            c = self._c
            c["L"] *= 1.0 + e[6] * s["acc_frac"]
            c["D"] *= 1.0 + e[7] * s["acc_frac"]
            c["n"] = float(np.hypot(c["L"], c["D"]))
            obs = super()._obs()
            self._z_nav, self._c_nav = zn, c                  # what a guidance law can see
        finally:
            self._z, self._c = z_true, c_true
        return obs

    # -- actuator path -------------------------------------------------------------
    def _deriv(self, z, sdot, adot):
        if not self._act_on:
            return super()._deriv(z, sdot, adot)
        rf = self.pert.rate_frac
        sig_max, alp_max = C.SIGMA_DOT_MAX * rf, C.ALPHA_DOT_MAX * rf
        sigma, alpha_rad = z[6], z[7]
        sdot = float(np.clip(self.track_gain * (self._sig_e - sigma), -sig_max, sig_max))
        adot = float(np.clip(self.track_gain * (self._alp_e - alpha_rad), -alp_max, alp_max))
        lo, hi = C.BANK_MIN, C.BANK_MAX
        if (sigma >= hi and sdot > 0.0) or (sigma <= lo and sdot < 0.0):
            sdot = 0.0
        sigma = min(max(sigma, lo), hi)
        lo, hi = C.ALPHA_MIN * _DEG, C.ALPHA_MAX * _DEG
        if (alpha_rad >= hi and adot > 0.0) or (alpha_rad <= lo and adot < 0.0):
            adot = 0.0
        alpha_rad = min(max(alpha_rad, lo), hi)
        d6 = translational_rhs(z[0], z[1], z[2], z[3], z[4], z[5], sigma,
                               np.rad2deg(alpha_rad), self._model)
        dz = np.zeros_like(z)
        dz[0:6] = d6
        dz[6] = sdot
        dz[7] = adot
        return dz

    def _substep(self, z, t_now, h):
        """Per-substep hooks: release delayed commands, advance the lag, apply the
        altitude-dependent density factor and any due impulse. Returns the state."""
        p = self.pert
        q = self._queue
        while q and q[0][0] <= t_now + 1e-9:
            _, self._sig_d, self._alp_d = q.popleft()
        if p.lag_s > 0.0:
            f = 1.0 - np.exp(-h / p.lag_s)
            self._sig_e += f * (self._sig_d - self._sig_e)
            self._alp_e += f * (self._alp_d - self._alp_e)
        else:
            self._sig_e, self._alp_e = self._sig_d, self._alp_d
        if p.rho_wave_amp or p.rho_bump_amp:
            alt = altitude_m(z[0], z[2])
            f = 1.0
            if p.rho_wave_amp:
                f += p.rho_wave_amp * np.sin(2.0 * np.pi * alt / p.rho_wave_len_m + self._rho_phase)
            if p.rho_bump_amp:
                f += p.rho_bump_amp * np.exp(-((alt - p.rho_bump_alt_m) / p.rho_bump_width_m) ** 2)
            bm = self._base_model
            self._model = bm._replace(k_rho=bm.k_rho * max(f, 0.05))
        if p.impulses:
            e = self.energy_togo(z)
            for i, (e_imp, dg, dpsi, dv) in enumerate(p.impulses):
                if not self._imp_done[i] and e <= e_imp:
                    z = z.copy()
                    z[3] += dv / C.V_SCALE
                    z[4] += dg * _DEG
                    z[5] += dpsi * _DEG
                    self._imp_done[i] = True
        return z

    def _integrate_energy(self, sdot, adot):
        if not self._act_on:
            return super()._integrate_energy(sdot, adot)
        p = self.pert
        sig_c = float(np.clip(p.bank_gain * self._sig_t + p.bank_bias_deg * _DEG, C.BANK_MIN, C.BANK_MAX))
        self._queue.append((self.t + p.delay_s, sig_c, self._alp_t))
        # same loop as EntryEnvBase._integrate_energy, with the per-substep hooks
        z = self._z
        h = self.dt / self.n_substeps
        level = self._next_energy_level(self.energy_togo(z))
        cap = self.dt_max if level is not None else self.dt_late
        self._step_barrier = 0.0
        elapsed = 0.0
        while True:
            z = self._substep(z, self.t + elapsed, h)
            z_prev = z
            z = self._rk4(z_prev, h, sdot, adot)
            if self.exact_terminal and altitude_m(z[0], z[2]) <= self.tgt_alt:
                tau = self._crossing_fraction(z_prev, z, h, sdot, adot)
                z = self._rk4(z_prev, tau * h, sdot, adot)
                self._accumulate(z, tau * h)
                self._crossed = True
                elapsed += tau * h
                break
            if level is not None and self.energy_togo(z) <= level:
                self._accumulate(z, h)
                elapsed += h
                break
            self._accumulate(z, h)
            elapsed += h
            if elapsed >= cap - 1e-9:
                break
        self._z = z
        return elapsed



# ---------------------------------------------------------------------------
# Runners (one per worker): closed loop, or open-loop replay of the nominal commands
# ---------------------------------------------------------------------------
class StabilityRunner(ep.PolicyRunner):
    def __init__(self, model, obs_rms, env_kwargs):
        super().__init__(model, obs_rms, "stage2", RewardWeights(), env_kwargs)
        kw = dict(env_kwargs)
        kw.setdefault("obs_version", ep.obs_version_for("stage2", model.observation_space.shape[0]))
        self.env = PerturbedStage2Env(weights=RewardWeights(), **kw)
        self.mode = "closed"
        self._sched_t = self._sched_a = None
        self._rec = None

    def act(self, obs):
        if self.mode == "open":
            i = max(int(np.searchsorted(self._sched_t, self.env.t, side="right")) - 1, 0)
            return self._sched_a[i]
        a = self._closed_loop_act(obs)
        if self._rec is not None:
            self._rec.append((self.env.t, a.copy()))
        return a

    def _closed_loop_act(self, obs):
        return ep.PolicyRunner.act(self, obs)

    def nominal_schedule(self):
        """Closed-loop nominal command history (time, raw action) for open-loop replay."""
        if self._sched_t is None:
            self.env.set_perturbation(Perturbation(), 0)
            self.mode, self._rec = "closed", []
            self.run(NOMINAL)
            self._sched_t = np.array([t for t, _ in self._rec])
            self._sched_a = np.array([a for _, a in self._rec])
            self._rec = None
        return self._sched_t, self._sched_a

    def run_case(self, scenario, pert, seed, mode="closed", record=False):
        if mode == "open":
            self.nominal_schedule()
        self.env.set_perturbation(pert, seed)
        self.mode = mode
        try:
            return self.run(scenario, record=record)
        finally:
            self.mode = "closed"


class FnpegRunner(StabilityRunner):
    """Same rollouts with the FNPEG baseline in place of the policy: the guidance reads the
    navigated state (env._z_nav / _c_nav) and its (bank, AoA) command is mapped onto the
    env's target-mode action, so actuators, perturbations and metrics are identical."""

    def __init__(self, env_kwargs, cfg=None):
        from reentry_rl.guidance.fnpeg import FnpegGuidance
        kw = dict(env_kwargs)
        kw.setdefault("obs_version", "v25")
        self.env = PerturbedStage2Env(weights=RewardWeights(), **kw)
        self.j_ref = self.env.J_ref
        self.guid = FnpegGuidance(cfg)
        self.mode = "closed"
        self._sched_t = self._sched_a = None
        self._rec = None

    def _closed_loop_act(self, obs):
        env = self.env
        if env.steps == 0:
            self.guid.reset(env)
        sig, alp = self.guid.command(env)
        lo, hi = C.ALPHA_MIN, C.ALPHA_MAX
        u = np.array([sig / C.BANK_MAX, (alp - 0.5 * (lo + hi)) / (0.5 * (hi - lo))])
        return np.arctanh(np.clip(u, -0.995, 0.995)).astype(np.float32)


_W = {}


def _init_worker(model_path, vn_path, env_kwargs):
    import torch
    torch.set_num_threads(1)
    ep.TRAJ_FIELDS = ep.TRAJ_FIELDS + ["energy_togo"]
    if str(model_path) == "fnpeg":
        _W["runner"] = FnpegRunner(env_kwargs)
        return
    model, rms = ep.load_policy(model_path, vn_path)
    _W["runner"] = StabilityRunner(model, rms, env_kwargs)


def _work(job):
    i, sdict, pdict, seed, mode, label, record = job
    r = _W["runner"]
    out = r.run_case(Scenario.from_dict(sdict), Perturbation.from_dict(pdict), seed, mode, record)
    if record:
        m, traj = out
        return {"id": i, **label, **m, **Scenario.from_dict(sdict).to_dict()}, traj
    return {"id": i, **label, **out, **Scenario.from_dict(sdict).to_dict()}, None


# ---------------------------------------------------------------------------
# Study definitions
# ---------------------------------------------------------------------------
P0 = Perturbation()
NAV_LEVELS = (0.5, 1, 2, 4, 8)
DELAYS = (0.5, 1, 2, 4, 8)
LAGS = (0.5, 1, 2, 4, 8)
RATE_FRACS = (2 / 3, 4 / 9, 2 / 9, 1 / 9)             # 30, 20, 10, 5 deg/s bank rate
GAINS = (0.5, 0.7, 0.85, 0.95, 1.05, 1.15, 1.3, 1.5)
BIASES = (-10, -5, -2, -1, 1, 2, 5, 10)
E_IMP = {"early": 0.7, "mid": 0.35, "late": 0.08}
KICKS = {"fpa-1": (-1.0, 0, 0), "fpa+1": (1.0, 0, 0), "psi+5": (0, 5.0, 0), "V-100": (0, 0, -100.0)}
WAVES = (0.1, 0.2, 0.3)
BUMPS = (-0.3, 0.3)
# combined "realistic" stack: nav at spec, 1 s delay, 1 s lag, 30 deg/s, +-10 % density waves
COMBINED = Perturbation(nav_level=1.0, delay_s=1.0, lag_s=1.0, rate_frac=2 / 3, rho_wave_amp=0.1)


def margin_conditions():
    """[(name, group, magnitude, Perturbation)] for the margin sweeps on the base set."""
    c = [("baseline", "baseline", 0.0, P0)]
    c += [(f"nav_x{k:g}", "nav", k, replace(P0, nav_level=k)) for k in NAV_LEVELS]
    c += [(f"delay_{d:g}s", "delay", d, replace(P0, delay_s=d)) for d in DELAYS]
    c += [(f"lag_{t:g}s", "lag", t, replace(P0, lag_s=t)) for t in LAGS]
    c += [(f"rate_{45 * f:.0f}dps", "rate", 45 * f, replace(P0, rate_frac=f)) for f in RATE_FRACS]
    c += [(f"gain_{g:g}", "gain", g, replace(P0, bank_gain=g)) for g in GAINS]
    c += [(f"bias_{b:+g}deg", "bias", b, replace(P0, bank_bias_deg=b)) for b in BIASES]
    for ph, e in E_IMP.items():
        for kn, (dg, dpsi, dv) in KICKS.items():
            c.append((f"imp_{kn}_{ph}", f"impulse_{kn}", e, replace(P0, impulses=((e, dg, dpsi, dv),))))
    c += [(f"rhowave_{a:g}", "rho_wave", a, replace(P0, rho_wave_amp=a)) for a in WAVES]
    c += [(f"rhobump_{a:+g}", "rho_bump", a, replace(P0, rho_bump_amp=a)) for a in BUMPS]
    return c


SENS = {"ic_fpa_deg": 0.05, "ic_V_mps": 10.0, "ic_alt_m": 1000.0, "ic_psi_deg": 0.2,
        "ic_lat_deg": 0.1, "k_rho": 0.02, "k_CL": 0.02, "k_CD": 0.02, "k_mass": 0.02}


def base_set(n=100):
    _name, scen, _lab = load_set(ep.resolve_set("S4_scale_x0.1"))
    return [NOMINAL] + scen[:n], [{"case": "nominal"}] + [{"case": f"x0.1_{i}"} for i in range(n)]


# ---------------------------------------------------------------------------
class Pool:
    """One persistent worker pool for the whole suite (pool respawns between sets
    have hung on Windows with WinError 5)."""

    def __init__(self, workers, env_kwargs, model=None, vn=None):
        self.workers = workers
        args = (str(model or MODEL), str(vn or VN), env_kwargs)
        if workers <= 1:
            _init_worker(*args)
            self.pool = None
        else:
            self.pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=args)

    def map(self, jobs):
        if self.pool is None:
            return [_work(j) for j in jobs]
        return self.pool.map(_work, jobs, chunksize=1)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def _jobs(scen, labels, pert, mode="closed", record=False):
    return [(i, s.to_dict(), pert.to_dict(), i, mode, labels[i], record) for i, s in enumerate(scen)]


def _write(df, path, meta):
    df.to_csv(path.parent / f"{path.name}.csv", index=False)     # not with_suffix: names hold "0.5"
    summ = {**meta, **ep.summarize(df), "robust_score": ep.robust_score(df),
            "bank_rate_rms_dps_mean": float(df["bank_rate_rms_dps"].mean()),
            "reversals_mean": float(df["reversals"].mean())}
    path.with_name(path.name + "_summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def study_margins(pool, out, n_base):
    d = out / "margins"
    d.mkdir(parents=True, exist_ok=True)
    scen, labels = base_set(n_base)
    for name, group, mag, pert in margin_conditions():
        if (d / f"{name}.csv").exists():
            continue
        t0 = time.time()
        rows = [r for r, _ in pool.map(_jobs(scen, labels, pert))]
        df = pd.DataFrame(rows).sort_values("id")
        s = _write(df, d / name, {"condition": name, "group": group, "magnitude": mag,
                                  "perturbation": pert.to_dict(), "n_base": n_base})
        _log(f"margins/{name}: d_med {s['stats']['d_km']['median']:.2f} km, "
             f"p95 {s['stats']['d_km']['p95']:.1f}, feasible {s['feasible_rate']:.2f} ({time.time() - t0:.0f} s)")


def study_sensitivity(pool, out):
    d = out / "sensitivity"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "sensitivity.csv"
    if f.exists():
        return
    scen, labels = [NOMINAL], [{"param": "none", "mult": 0.0}]
    for k, dv in SENS.items():
        for m in (-5, -1, 1, 5):
            scen.append(replace(NOMINAL, **{k: getattr(NOMINAL, k) + m * dv}))
            labels.append({"param": k, "mult": float(m), "delta": m * dv})
    rows = []
    for mode in ("closed", "open"):
        for r, _ in pool.map(_jobs(scen, labels, P0, mode)):
            rows.append({**r, "mode": mode})
    pd.DataFrame(rows).to_csv(f, index=False)
    _log("sensitivity done")


def study_recovery(pool, out):
    d = out / "recovery"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "recovery_summary.csv").exists():
        return
    jobs, tags = [], []
    for mode in ("closed", "open"):
        tags.append((mode, "none", "none"))
        for ph, e in E_IMP.items():
            for kn, (dg, dpsi, dv) in KICKS.items():
                tags.append((mode, kn, ph))
    for j, (mode, kn, ph) in enumerate(tags):
        pert = P0 if kn == "none" else replace(P0, impulses=((E_IMP[ph],) + KICKS[kn],))
        jobs.append((j, NOMINAL.to_dict(), pert.to_dict(), 0, mode, {"mode": mode, "kick": kn, "phase": ph}, True))
    rows = []
    for (r, traj), (mode, kn, ph) in zip(pool.map(jobs), tags):
        pd.DataFrame(traj).to_csv(d / f"traj_{mode}_{kn}_{ph}.csv", index=False)
        rows.append(r)
    pd.DataFrame(rows).to_csv(d / "recovery_summary.csv", index=False)
    _log("recovery done")


def study_grid(pool, out, name, axes):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{name}.csv"
    if f.exists():
        return
    (k1, v1), (k2, v2) = axes
    scen, labels = [], []
    for a in v1:
        for b in v2:
            scen.append(replace(NOMINAL, **{k1: float(a), k2: float(b)}))
            labels.append({k1 + "_grid": float(a), k2 + "_grid": float(b)})
    rows = [r for r, _ in pool.map(_jobs(scen, labels, P0))]
    pd.DataFrame(rows).sort_values("id").to_csv(f, index=False)
    _log(f"{name} done ({len(scen)} cases)")


def study_combined(pool, out):
    d = out / "combined"
    d.mkdir(parents=True, exist_ok=True)
    for set_name in ("S4_scale_x0.1", "S4_scale_x0.25"):
        _n, scen, _l = load_set(ep.resolve_set(set_name))
        labels = [{"case": i} for i in range(len(scen))]
        for tag, pert in (("baseline", P0), ("combined", COMBINED)):
            name = f"{set_name}_{tag}"
            if (d / f"{name}.csv").exists():
                continue
            t0 = time.time()
            df = pd.DataFrame([r for r, _ in pool.map(_jobs(scen, labels, pert))]).sort_values("id")
            s = _write(df, d / name, {"set": set_name, "condition": tag, "perturbation": pert.to_dict()})
            _log(f"combined/{name}: d_med {s['stats']['d_km']['median']:.2f} km, "
                 f"feasible {s['feasible_rate']:.2f} ({time.time() - t0:.0f} s)")


def study_jacobian(out, env_kwargs, n_cases=20):
    """Spectral norm of d[bank_cmd, aoa_cmd] (deg) / d(normalized obs) along trajectories."""
    import torch as th
    d = out / "jacobian"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "jacobian.csv"
    if f.exists():
        return
    model, rms = ep.load_policy(MODEL, VN)
    runner = StabilityRunner(model, rms, env_kwargs)
    env, pol = runner.env, model.policy
    scen, labels = base_set(n_cases)
    lo, hi = C.ALPHA_MIN, C.ALPHA_MAX
    rows = []
    for (s, lab) in zip(scen, labels):
        env.set_perturbation(P0, 0)
        obs, _ = env.reset(options={"scenario": s})
        while True:
            on = np.clip((obs - runner.mean) / runner.std, -ep.VECNORM_CLIP, ep.VECNORM_CLIP)
            x = th.tensor(on[None], dtype=th.float32)

            def cmd(xx):
                mu = pol.get_distribution(xx).distribution.mean[0]
                u = th.tanh(th.clamp(mu, -3.0, 3.0))
                return th.stack([u[0] * np.rad2deg(C.BANK_MAX), 0.5 * (lo + hi) + 0.5 * (hi - lo) * u[1]])

            J = th.autograd.functional.jacobian(cmd, x)[:, 0, :].numpy()
            sv = np.linalg.svd(J, compute_uv=False)
            rows.append({**lab, "t_s": env.t, "energy_togo": env.energy_togo(), "alt_km": env._c["alt"] / 1e3,
                         "jac_norm": float(sv[0]), "jac_norm_bank": float(np.linalg.norm(J[0])),
                         "jac_norm_aoa": float(np.linalg.norm(J[1]))})
            a = runner.act(obs)
            obs, _r, term, trunc, _info = env.step(a)
            if term or trunc:
                break
    pd.DataFrame(rows).to_csv(f, index=False)
    _log(f"jacobian done ({len(rows)} decisions)")


STUDIES = ("margins", "sensitivity", "recovery", "corridor", "footprint", "jacobian", "combined")


def main():
    ap = argparse.ArgumentParser(description="Stability / robustness analyses of the v26 6M policy")
    ap.add_argument("--outdir", default="results/stability_v26")
    ap.add_argument("--studies", nargs="+", default=list(STUDIES), choices=STUDIES)
    ap.add_argument("--n-base", type=int, default=100, help="0.1x cases in the margin base set")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--guidance", choices=("rl", "fnpeg"), default="rl",
                    help="rl: the policy at --model/--vn; fnpeg: the classical baseline (same env)")
    ap.add_argument("--model", default=None, help="policy zip (default: v26 6M)")
    ap.add_argument("--vn", default=None, help="VecNormalize pickle of --model")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="margin conditions to run (names, e.g. baseline gain_0.95); default all")
    args = ap.parse_args()
    global MODEL, VN
    if args.model:
        MODEL, VN = Path(args.model).resolve(), Path(args.vn).resolve()
    if args.conditions:
        keep, allc = set(args.conditions), margin_conditions
        globals()["margin_conditions"] = lambda: [c for c in allc() if c[0] in keep]
    out = Path(args.outdir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    # the env (dynamics, actuators, path constraints) is the one the policy was trained in;
    # the FNPEG baseline flies the very same env (its reward terms are unused)
    env_kwargs = {**ep.env_kwargs_for(MODEL), "exact_terminal": True, "rate_saturation": True}
    guid = "fnpeg" if args.guidance == "fnpeg" else str(MODEL)
    meta = {"guidance": args.guidance, "model": guid, "vn": str(VN), "env_kwargs": env_kwargs,
            "nav_spec": NAV_SPEC, "combined": COMBINED.to_dict(), "n_base": args.n_base}
    if args.guidance == "fnpeg":
        from reentry_rl.guidance.fnpeg import FnpegConfig
        meta["fnpeg_config"] = FnpegConfig().to_dict()
    (out / "suite.json").write_text(json.dumps(meta, indent=1))
    t0 = time.time()
    _log(f"suite start ({args.guidance}): {args.studies}")
    if "jacobian" in args.studies and args.guidance == "rl":
        study_jacobian(out, env_kwargs)
    pool = Pool(args.workers, env_kwargs, model=guid)
    try:
        if "sensitivity" in args.studies:
            study_sensitivity(pool, out)
        if "recovery" in args.studies:
            study_recovery(pool, out)
        if "margins" in args.studies:
            study_margins(pool, out, args.n_base)
        if "corridor" in args.studies:
            study_grid(pool, out, "corridor", (("ic_fpa_deg", np.round(np.arange(-1.5, 0.51, 0.2), 3)),
                                               ("ic_V_mps", np.arange(-400.0, 401.0, 80.0))))
        if "footprint" in args.studies:
            study_grid(pool, out, "footprint", (("tc_lat_deg", np.arange(-6.0, 6.01, 1.0)),
                                                ("tc_lon_deg", np.arange(-6.0, 6.01, 1.0))))
        if "combined" in args.studies:
            study_combined(pool, out)
    finally:
        pool.close()
    _log(f"SUITE DONE in {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
