"""Stage 2 — bank-rate + AoA-rate guidance (DESIGN §6).

Action: 2-D, normalized [bank_rate, aoa_rate] in [-1,1]^2 -> [+/-45 deg/s, +/-15 deg/s].
AoA is now a controlled state (it followed the legacy WB001 schedule in Stage 1).
The reward is unchanged from Stage 1 (it is action-dimension-agnostic by design), so
this run ALSO tests the reward-transfer hypothesis. The extra L/D authority from AoA
is the lever the bank-only feasible-position wall (~130 km) has been pointing to.
Reference SCP optimum for comparison: J_heat = 1.072e9 J/m^2 (bank_aoa_free_time).
"""
import numpy as np

from ..physics import constants as C
from .entry_env_base import EntryEnvBase
from .rewards import RewardWeights


class Stage2BankAoaEnv(EntryEnvBase):
    optimize_aoa = True
    n_actions = 2

    def __init__(self, weights: RewardWeights = None, **kwargs):
        kwargs.setdefault("j_ref", C.J_REF_BANK_AOA)
        super().__init__(weights=weights, **kwargs)


def make_stage2_env(weights: RewardWeights = None, **kwargs):
    """Factory used by the training/eval scripts."""
    return Stage2BankAoaEnv(weights=weights, **kwargs)


def _selftest(seed=0):
    env = Stage2BankAoaEnv(seed=seed)
    obs, info = env.reset(seed=seed)
    assert env.observation_space.shape == obs.shape, "obs shape mismatch"
    assert np.all(np.isfinite(obs)), "non-finite obs at reset"
    rng = np.random.default_rng(seed)
    total_r, term, trunc = 0.0, False, False
    while not (term or trunc):
        act = rng.uniform(-1, 1, size=(2,)).astype(np.float32)
        obs, r, term, trunc, info = env.step(act)
        total_r += r
        assert np.all(np.isfinite(obs)), "non-finite obs during rollout"
        assert np.isfinite(r), "non-finite reward"
    return total_r, info


if __name__ == "__main__":
    print("Stage2BankAoaEnv self-test (no SB3) — random 2-D action rollout\n")
    tr, info = _selftest(seed=1)
    print(f"  outcome={info.get('outcome')}  t_f={info.get('t_s'):.0f}s  total_reward={tr:+.2f}")
    print(f"  d_km={info.get('d_km', float('nan')):.1f}  alpha_f={info['alpha_deg']:.1f} deg  "
          f"J_heat={info['J_heat']:.3e}  (J_ref={C.J_REF_BANK_AOA:.3e})")
    print("OK — Stage 2 env runs, observations and rewards finite.")
