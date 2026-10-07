"""Training curves along warm-start chains, on a cumulative-steps axis.

A chain is a sequence of runs where each run was warm-started from the best
model of the previous one. Each run is placed at the cumulative step count of
its warm start; the part of a run after the model that seeded the next run is
a discarded branch and is drawn faded.

Figures (results/curves/):
  chain_comparison.png  nominal landing error, old campaign chain (legacy
                        dynamics, position-only reward) vs the v20 chain
                        (corrected dynamics)
  v20_chain_terminal.png  v20 chain: landing error, FPA and heading errors,
                        peak path-constraint ratio, AoA spread, training return
  v20_chain_ppo.png     v20 chain: approx KL, explained variance, policy std,
                        value loss

A stagnation summary per run is printed and written to
results/curves/stagnation.json.

    python -m reentry_rl.postprocessing.plot_chain_curves
"""
import json
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

from reentry_rl.postprocessing.diagnose_run import parse_train_log, series

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "curves"

# Reference palette (dataviz skill, categorical slots 1-3, light surface)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# (label, run dir, steps of the model that seeded the next run or None = live/last)
OLD_CHAIN = [("v10", RES / "results/stage2_v10", 200_000),
             ("v12", RES / "results/stage2_v12", 350_000),
             ("v13", RES / "results/stage2_v13", 2_850_000),
             ("v16", RES / "results/stage2_v16", 2_550_000)]
OLD_HPO = (300_000, 0.273)          # trial 23: +0.3M polish steps, legacy measurement
V20_CHAIN = [("v20b", RES / "stage2_v20b", 600_000),
             ("v20c", RES / "stage2_v20c", 900_000),
             ("v20d", RES / "stage2_v20d", None)]


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
    ax.title.set_color(INK)


def _rolling_median(t, v, window=15):
    if v.size < 3:
        return t, v
    s = pd.Series(v).rolling(window, center=True, min_periods=max(3, window // 3)).median()
    return t, s.to_numpy()


def _segments(chain):
    """-> list of dicts with offset, cutoff and data per run."""
    out, offset = [], 0
    for label, run, cut in chain:
        log = run / "train_log.txt"
        recs = parse_train_log(log) if log.exists() else []
        vh = run / "eval" / "validation_history.csv"
        out.append({"label": label, "run": run, "offset": offset, "cut": cut, "recs": recs,
                    "val": pd.read_csv(vh) if vh.exists() else None})
        if cut is not None:
            offset += cut
    return out


def _split(t, v, cut):
    """(kept part, discarded part) of a run's series at its warm-start cutoff."""
    if cut is None:
        return (t, v), (t[:0], v[:0])
    k = t <= cut
    j = np.r_[np.where(~k)[0][:1] - 1, np.where(~k)[0]] if (~k).any() else np.array([], int)
    j = j[j >= 0]
    return (t[k], v[k]), (t[j], v[j])


def _plot_run(ax, t, v, cut, offset, color, label, logy=False, ms=0, lw=1.4):
    (tk, vk), (td, vd) = _split(np.asarray(t, float), np.asarray(v, float), cut)
    plot = ax.semilogy if logy else ax.plot
    plot((tk + offset) / 1e6, vk, "-", color=color, lw=lw, marker="o" if ms else None, ms=ms, label=label)
    if td.size:
        plot((td + offset) / 1e6, vd, "-", color=color, lw=lw * 0.8, alpha=0.3,
             marker="o" if ms else None, ms=ms * 0.8)


def chain_comparison(old, new, path):
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    _style(ax)
    for i, seg in enumerate(old):
        t, d = series(seg["recs"], "d_km")
        _plot_run(ax, t, d, seg["cut"], seg["offset"], C1,
                  "old chain v10-v16 + HPO (legacy dynamics, position only)" if i == 0 else None,
                  logy=True, ms=2, lw=1.1)
    hpo_x = (old[-1]["offset"] + old[-1]["cut"] + OLD_HPO[0]) / 1e6
    ax.semilogy([hpo_x], [OLD_HPO[1]], "D", color=C1, ms=7)
    ax.annotate("HPO trial 23\n0.27 km", (hpo_x, OLD_HPO[1]), xytext=(8, -2),
                textcoords="offset points", fontsize=8, color=INK2, va="center")
    for i, seg in enumerate(new):
        v = seg["val"]
        if v is None:
            continue
        _plot_run(ax, v["step"].to_numpy(), v["nominal_d_km"].to_numpy(), seg["cut"], seg["offset"], C2,
                  "v20 chain v20b-v20d (corrected dynamics)" if i == 0 else None, logy=True, ms=3)
    for segs, ytxt, col in ((old[1:], 0.97, C1), (new[1:], 0.62, C2)):
        for seg in segs:
            ax.axvline(seg["offset"] / 1e6, color=GRID, lw=1.0, ls=":", zorder=0)
            ax.text(seg["offset"] / 1e6 + 0.05, ytxt, seg["label"], transform=ax.get_xaxis_transform(),
                    fontsize=7, color=col, ha="left", va="top", rotation=90)
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
    ax.text(ax.get_xlim()[1], 1.0, " 1 km", fontsize=8, color=INK2, va="center")
    ax.set_xlabel("cumulative training steps along the warm-start chain [M]")
    ax.set_ylabel("nominal landing error [km]")
    ax.set_title("Landing error along the warm-start chains (faded: discarded branch after the "
                 "model that seeded the next run)", fontsize=10, loc="left", pad=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def v20_terminal(new, path):
    cols = [C1, C2, C3]
    panels = [("nominal landing error [km]", "nominal_d_km", True, None),
              ("|FPA error| at handover [deg]", "nominal_dfpa_deg", False, None),
              ("|heading error| at handover [deg]", "nominal_dpsi_deg", False, None),
              ("peak path-constraint ratio", "max_ratio_max", False, 1.0),
              ("AoA spread along the rollout [deg]", "alpha_std_mean", False, None),
              ("training return (ep_rew_mean)", None, False, None)]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.5), facecolor=SURFACE)
    for ax, (title, key, logy, ref) in zip(axes.flat, panels):
        _style(ax)
        for c, seg in zip(cols, new):
            if key is None:
                t, v = series(seg["recs"], "ep_rew_mean")
                if not t.size:
                    continue
                _plot_run(ax, t, v, seg["cut"], seg["offset"], c, None, lw=0.6)
                ts, vs = _rolling_median(t, v)
                _plot_run(ax, ts, vs, seg["cut"], seg["offset"], c, seg["label"], lw=1.8)
                continue
            if seg["val"] is None:
                continue
            v = seg["val"][key].to_numpy()
            v = np.abs(v) if "deg" in key and "std" not in key else v
            _plot_run(ax, seg["val"]["step"].to_numpy(), v, seg["cut"], seg["offset"], c,
                      seg["label"], logy=logy, ms=3)
        if ref is not None:
            ax.axhline(ref, color=INK2, lw=0.8, ls="--")
        for seg in new[1:]:
            ax.axvline(seg["offset"] / 1e6, color=GRID, lw=1.0, ls=":", zorder=0)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("cumulative steps [M]", fontsize=8)
    axes[0, 0].legend(fontsize=8, frameon=False, labelcolor=INK)
    axes[2, 1].text(0.01, 0.02, "reward definition changes between runs: returns are comparable\n"
                    "within a run only (v20b: angle channels on; v20c: off; v20d: off + peak charge)",
                    transform=axes[2, 1].transAxes, fontsize=7, color=INK2)
    fig.suptitle("v20 chain under the corrected dynamics (deterministic nominal validation every 100k steps)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def v20_ppo(new, path):
    cols = [C1, C2, C3]
    panels = [("approx KL per update (log)", "approx_kl", True, 0.05),
              ("explained variance of the value function", "explained_variance", False, None),
              ("policy std (mean over actions)", "std", False, None),
              ("value loss (log)", "value_loss", True, None)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), facecolor=SURFACE)
    for ax, (title, key, logy, ref) in zip(axes.flat, panels):
        _style(ax)
        for c, seg in zip(cols, new):
            t, v = series(seg["recs"], key)
            if t.size:
                _plot_run(ax, t, v, seg["cut"], seg["offset"], c, seg["label"], logy=logy, lw=0.9)
        if ref is not None:
            ax.axhline(ref, color=INK2, lw=0.8, ls="--")
            ax.text(ax.get_xlim()[0], ref, " target_kl", fontsize=7, color=INK2, va="bottom")
        for seg in new[1:]:
            ax.axvline(seg["offset"] / 1e6, color=GRID, lw=1.0, ls=":", zorder=0)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("cumulative steps [M]", fontsize=8)
    axes[0, 0].legend(fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle("v20 chain: PPO optimisation diagnostics", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def stagnation(seg, window=1_000_000):
    """Trend of a run over its last `window` steps: return slope, log-d slope,
    and steps since the best validation."""
    out = {"run": seg["label"]}
    t, r = series(seg["recs"], "ep_rew_mean")
    if t.size > 4:
        k = t >= t[-1] - window
        out["steps"] = int(t[-1])
        out["return_slope_per_M"] = float(np.polyfit(t[k] / 1e6, r[k], 1)[0])
        h = max(int(k.sum() // 2), 1)
        out["return_last_half_mean"] = float(r[k][-h:].mean())
        out["return_prev_half_mean"] = float(r[k][:-h].mean()) if k.sum() > h else float("nan")
    v = seg["val"]
    if v is not None and len(v) > 2:
        k = v["step"] >= v["step"].iloc[-1] - window
        out["log10_d_slope_per_M"] = float(np.polyfit(v["step"][k] / 1e6, np.log10(v["nominal_d_km"][k]), 1)[0])
        i = int(v["score"].idxmin())
        out["best_score"] = float(v["score"].iloc[i])
        out["best_step"] = int(v["step"].iloc[i])
        out["steps_since_best"] = int(v["step"].iloc[-1] - v["step"].iloc[i])
        out["last_d_km"] = float(v["nominal_d_km"].iloc[-1])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old, new = _segments(OLD_CHAIN), _segments(V20_CHAIN)
    chain_comparison(old, new, OUT / "chain_comparison.png")
    v20_terminal(new, OUT / "v20_chain_terminal.png")
    v20_ppo(new, OUT / "v20_chain_ppo.png")
    stag = [stagnation(s) for s in new]
    (OUT / "stagnation.json").write_text(json.dumps(stag, indent=1))
    for s in stag:
        print(json.dumps(s))
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
