"""Load exported MATLAB SCP benchmark runs (guidance_executed_trajectory.txt +
guidance_terminal_summary.txt).

Handles BOTH CSV layouts:
  - bank_aoa_free_time : 11 columns incl. alpha_deg, uAlpha_degps
  - bank_free_time     :  9 columns (no alpha) -> alpha reconstructed from the
                          WB001 nominal schedule, uAlpha set to 0
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..physics.aero_wb001 import nominal_aoa_deg

# The two WB001 min-heatload benchmarks (relative to the reference repo root).
REF_ROOT = Path(r"E:\Code\reentry_simulator")
BENCHMARKS = {
    "bank_only": REF_ROOT / "results" / "guidance_unit_test" / "guidance_run_wb001_20260617_191855",
    "bank_aoa": REF_ROOT / "results" / "validation" / "guidance_run_wb001_min_heatload_bank_aoa_free_time",
}


def load_trajectory(folder):
    """Return a DataFrame with at least: tAbs_s, alt_km, lon_deg, lat_deg, V_mps,
    gamma_deg, psi_deg, sigma_deg, alpha_deg, uSigma_degps, uAlpha_degps."""
    folder = Path(folder)
    df = pd.read_csv(folder / "guidance_executed_trajectory.txt")
    if "alpha_deg" not in df.columns:
        df["alpha_deg"] = nominal_aoa_deg(df["V_mps"].to_numpy())
    if "uAlpha_degps" not in df.columns:
        df["uAlpha_degps"] = 0.0
    return df


def load_terminal_summary(folder):
    """Parse guidance_terminal_summary.txt into a {label: float} dict."""
    txt = (Path(folder) / "guidance_terminal_summary.txt").read_text()
    out = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(.+?)\s*:\s*([-+0-9.eE]+)\s*$", line)
        if m:
            try:
                out[m.group(1).strip()] = float(m.group(2))
            except ValueError:
                pass
    return out
