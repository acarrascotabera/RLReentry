"""Gymnasium environments wrapping the validated physics core.

  entry_env_base.EntryEnvBase — shared 3-DOF env (obs / reward / termination / RK4)
  stage1_bank.Stage1BankEnv   — bank-rate only (1-D action), alpha from schedule
"""
from .rewards import RewardWeights
from .entry_env_base import EntryEnvBase
from .stage1_bank import Stage1BankEnv, make_stage1_env

__all__ = ["RewardWeights", "EntryEnvBase", "Stage1BankEnv", "make_stage1_env"]
