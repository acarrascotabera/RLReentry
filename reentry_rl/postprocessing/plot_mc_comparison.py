"""Empirical CDFs of the landing error for several policies on the same Monte Carlo sets.

    python -m reentry_rl.postprocessing.plot_mc_comparison

Writes results/curves/mc_comparison.png: (left) the realistic 500-case test set
S3_test_ic_models_x0.1, (right) the SCvx gc_ic_tc campaign scenarios, where the
SCvx closed-loop G&C result is drawn from results/scvx_mc/. Feasible cases
(peak path-constraint ratio <= 1) are counted separately in the legend.
"""
from pathlib import Path

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

POLICIES = [("v21 best (nominal training, 29.2M)", RES / "stage2_v21/eval_mc_best_29p2M"),
            ("v22 best (dispersion training)", RES / "stage2_v22/eval_mc_best"),
            ("v22 final", RES / "stage2_v22/eval_mc_final")]


def _cdf(ax, d, color, label, ls="-"):
    x = np.sort(np.asarray(d, float))
    ax.semilogx(x, np.arange(1, x.size + 1) / x.size, ls, color=color, lw=1.8, label=label)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
    for ax, set_name, title in ((axes[0], "S3_test_ic_models_x0.1", "realistic dispersions (500 cases, 0.1x VISTA)"),
                                (axes[1], "SCVX_gc_ic_tc", "SCvx gc_ic_tc campaign scenarios (1001, VISTA)")):
        _style(ax)
        if set_name == "SCVX_gc_ic_tc":
            s = pd.read_csv(RES / "scvx_mc" / "gc_ic_tc.csv")
            _cdf(ax, s["d_km"], INK, f"SCvx closed-loop G&C (feasible {s['feasible'].mean():.0%})", ls="--")
        for c, (label, base) in zip(C, POLICIES):
            f = base / f"{set_name}.csv"
            if f.exists():
                d = pd.read_csv(f)
                _cdf(ax, d["d_km"], c, f"{label} (feasible {d['feasible'].mean():.0%})")
        for x in (1, 10, 100):
            ax.axvline(x, color=GRID, lw=1.0, zorder=0)
        ax.set_xlabel("landing error at the 25 km handover [km]")
        ax.set_ylabel("fraction of cases")
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left", labelcolor=INK)
    fig.suptitle("Landing-error distributions on identical scenarios", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = RES / "curves" / "mc_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
