"""Stage 1 — bank-rate-only guidance (DESIGN §6).

Action: 1-D, normalized bank rate sigma_dot in [-1, 1] -> [-45, +45] deg/s.
AoA follows the legacy WB001 schedule (Stage 1 == reference `bank_free_time`).
Reference SCP optimum for comparison: J_heat = 1.0832e9 J/m^2.
"""
import numpy as np

from ..physics import constants as C
from .entry_env_base import EntryEnvBase
from .rewards import RewardWeights


class Stage1BankEnv(EntryEnvBase):
    optimize_aoa = False
    n_actions = 1

    def __init__(self, weights: RewardWeights = None, **kwargs):
        kwargs.setdefault("j_ref", C.J_REF_BANK_ONLY)
        super().__init__(weights=weights, **kwargs)


def make_stage1_env(weights: RewardWeights = None, **kwargs):
    """Factory used by the training/eval scripts."""
    return Stage1BankEnv(weights=weights, **kwargs)


def _selftest(seed=0, deterministic_bank=None):
    """Run one episode (random or fixed-bank) and report — a no-SB3 env smoke check."""
    env = Stage1BankEnv(seed=seed)
    obs, info = env.reset(seed=seed)
    assert env.observation_space.shape == obs.shape, "obs shape mismatch"
    assert np.all(np.isfinite(obs)), "non-finite obs at reset"
    rng = np.random.default_rng(seed)
    total_r = 0.0
    term = trunc = False
    while not (term or trunc):
        if deterministic_bank is None:
            act = rng.uniform(-1, 1, size=(1,)).astype(np.float32)
        else:
            act = np.array([deterministic_bank], dtype=np.float32)
        obs, r, term, trunc, info = env.step(act)
        total_r += r
        assert np.all(np.isfinite(obs)), "non-finite obs during rollout"
        assert np.isfinite(r), "non-finite reward"
    return total_r, info


if __name__ == "__main__":
    print("Stage1BankEnv self-test (no SB3) — random and fixed-bank rollouts\n")
    for label, db in [("random", None), ("bank=0 (full lift-up)", 0.0)]:
        tr, info = _selftest(seed=1, deterministic_bank=db)
        print(f"[{label}]")
        print(f"  outcome={info.get('outcome'):8s}  steps->t_f={info.get('t_s'):.0f}s  "
              f"total_reward={tr:+.2f}")
        print(f"  terminal: alt={info['alt_km']:.2f} km  V={info['V_mps']:.0f} m/s  "
              f"lat={info['lat_deg']:.2f}  lon={info['lon_deg']:.2f}")
        print(f"            d_km={info.get('d_km', float('nan')):.1f}  "
              f"fpa_err={info.get('fpa_err_deg', float('nan')):.2f}  "
              f"psi_err={info.get('psi_err_deg', float('nan')):.2f}  "
              f"J_heat={info['J_heat']:.3e}  (J_ref={C.J_REF_BANK_ONLY:.3e})\n")
    print("OK — env runs, observations and rewards finite.")
