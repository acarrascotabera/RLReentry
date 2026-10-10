"""Mission scenarios: dispersed entry state, target and model for one episode.

A `Scenario` holds additive offsets on the entry-interface state and on the
terminal target, plus multiplicative model factors. The default instance is
the nominal WB001 mission, so `reset()` without a scenario is unchanged.

The dispersion distributions mirror the VISTA SCvx Monte Carlo cases, so RL
and SCvx are evaluated against the same uncertainty model:

  ic     : VISTA montecarlo/cases/g_ic_tc.m   (entry-interface part)
  tc     : VISTA montecarlo/cases/g_ic_tc.m   (terminal-target part)
  models : VISTA montecarlo/cases/g_models.m  (mass, C_L, C_D, density)

All are independent Gaussians clipped to the VISTA bounds. VISTA states that
its IC magnitudes are about an order of magnitude wider than Apollo-class
delivery errors (stress test); `sigma_scale` rescales sigma and clip bounds
together, so sigma_scale=0.1 gives the "realistic" level.
"""
import json
from dataclasses import dataclass, asdict, fields, replace
from pathlib import Path

import numpy as np

from ..physics.aero_wb001 import ModelFactors


@dataclass(frozen=True)
class Scenario:
    # entry-interface offsets (added to constants.IC)
    ic_alt_m: float = 0.0
    ic_lat_deg: float = 0.0
    ic_lon_deg: float = 0.0
    ic_V_mps: float = 0.0
    ic_fpa_deg: float = 0.0
    ic_psi_deg: float = 0.0
    # terminal-target offsets (added to constants.TARGET)
    tc_alt_m: float = 0.0
    tc_lat_deg: float = 0.0
    tc_lon_deg: float = 0.0
    tc_fpa_deg: float = 0.0
    tc_psi_deg: float = 0.0
    # multiplicative model factors
    k_mass: float = 1.0
    k_CL: float = 1.0
    k_CD: float = 1.0
    k_rho: float = 1.0

    def model_factors(self):
        """ModelFactors, or None when the model is nominal (bit-identical path)."""
        mf = ModelFactors(k_rho=self.k_rho, k_CL=self.k_CL, k_CD=self.k_CD, k_mass=self.k_mass)
        return None if mf == ModelFactors() else mf

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in names})


NOMINAL = Scenario()

# name -> (group, mean, sigma, clip_lo, clip_hi); copied from the VISTA cases.
# Model factors have mean 1 (VISTA op 'scale'); offsets have mean 0 ('add').
DISPERSIONS = {
    "ic_alt_m":   ("ic", 0.0, 5.0e3, -15e3, 15e3),
    "ic_lat_deg": ("ic", 0.0, 1.0, -3.0, 3.0),
    "ic_lon_deg": ("ic", 0.0, 1.0, -3.0, 3.0),
    "ic_V_mps":   ("ic", 0.0, 60.0, -200.0, 200.0),
    "ic_fpa_deg": ("ic", 0.0, 0.25, -0.75, 0.75),
    "ic_psi_deg": ("ic", 0.0, 1.0, -3.0, 3.0),
    "tc_alt_m":   ("tc", 0.0, 2.0e3, -5e3, 5e3),
    "tc_lat_deg": ("tc", 0.0, 1.0, -3.0, 3.0),
    "tc_lon_deg": ("tc", 0.0, 1.0, -3.0, 3.0),
    "tc_fpa_deg": ("tc", 0.0, 0.5, -1.5, 1.5),
    "tc_psi_deg": ("tc", 0.0, 0.5, -1.5, 1.5),
    "k_mass":     ("models", 1.0, 0.05, 0.85, 1.15),
    "k_CD":       ("models", 1.0, 0.20, 0.50, 1.50),
    "k_CL":       ("models", 1.0, 0.20, 0.50, 1.50),
    "k_rho":      ("models", 1.0, 0.25, 0.40, 1.60),
}


def _params(groups):
    groups = tuple(groups)
    unknown = set(groups) - {g for g, *_ in DISPERSIONS.values()}
    if unknown:
        raise ValueError(f"unknown dispersion groups {sorted(unknown)}")
    return [k for k, (g, *_) in DISPERSIONS.items() if g in groups]


def _scaled(name, sigma_scale):
    _g, mean, sigma, lo, hi = DISPERSIONS[name]
    return mean, sigma * sigma_scale, mean + (lo - mean) * sigma_scale, mean + (hi - mean) * sigma_scale


def sample_scenarios(n, groups=("ic", "models"), sigma_scale=1.0, seed=0):
    """n independent clipped-Gaussian draws over the selected groups."""
    rng = np.random.default_rng(seed)
    names = _params(groups)
    out = []
    for _ in range(int(n)):
        vals = {}
        for k in names:
            mean, sigma, lo, hi = _scaled(k, sigma_scale)
            vals[k] = float(np.clip(rng.normal(mean, sigma), lo, hi))
        out.append(replace(NOMINAL, **vals))
    return out


def sweep_scenarios(groups=("ic", "tc", "models"), multipliers=(-3, -2, -1, 1, 2, 3),
                    sigma_scale=1.0):
    """One-parameter-at-a-time sweep at mean + m*sigma (clipped). Returns
    [(param, multiplier, Scenario)] for sensitivity plots."""
    out = []
    for k in _params(groups):
        mean, sigma, lo, hi = _scaled(k, sigma_scale)
        for m in multipliers:
            out.append((k, float(m), replace(NOMINAL, **{k: float(np.clip(mean + m * sigma, lo, hi))})))
    return out


class ScenarioSampler:
    """Callable rng -> Scenario for training-time domain randomization (picklable,
    so it can be handed to SubprocVecEnv workers through env_kwargs). Each env draws
    a new scenario at every reset from its own seeded rng."""

    def __init__(self, groups=("ic", "models"), sigma_scale=1.0):
        self.groups = tuple(groups)
        self.sigma_scale = float(sigma_scale)
        self.names = _params(self.groups)

    def __call__(self, rng):
        vals = {}
        for k in self.names:
            mean, sigma, lo, hi = _scaled(k, self.sigma_scale)
            vals[k] = float(np.clip(rng.normal(mean, sigma), lo, hi))
        return replace(NOMINAL, **vals)

    def describe(self):
        return {"groups": list(self.groups), "sigma_scale": self.sigma_scale}


def sampler(groups=("ic", "models"), sigma_scale=1.0):
    """Callable rng -> Scenario, for training-time domain randomization."""
    return ScenarioSampler(groups, sigma_scale)


# ---------------------------------------------------------------------------
# Evaluation-set files (JSON, version controlled under results/eval_sets/)
# ---------------------------------------------------------------------------
def save_set(path, name, scenarios, meta=None, labels=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, s in enumerate(scenarios):
        row = {"id": i, **s.to_dict()}
        if labels is not None:
            row.update(labels[i])
        rows.append(row)
    doc = {"name": name, "meta": meta or {}, "n": len(rows),
           "dispersion_spec": {k: dict(zip(("group", "mean", "sigma", "clip_lo", "clip_hi"), v))
                               for k, v in DISPERSIONS.items()},
           "scenarios": rows}
    path.write_text(json.dumps(doc, indent=1))
    return path


def load_set(path):
    """-> (name, [Scenario], [label dict]) where labels hold any extra row keys."""
    doc = json.loads(Path(path).read_text())
    names = {f.name for f in fields(Scenario)}
    scen, labels = [], []
    for row in doc["scenarios"]:
        scen.append(Scenario.from_dict(row))
        labels.append({k: v for k, v in row.items() if k not in names})
    return doc["name"], scen, labels
