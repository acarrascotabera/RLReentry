"""Performance versus uncertainty level on the common-random-number sets S4_scale_x<s>.

    python -m reentry_rl.postprocessing.robustness_curves

Case i of every S4 set is the same standard-normal draw shrunk by the scale s, so the
curves compare like with like. Per policy and scale:
  - landing error (median, p90), median |dgamma| and |dpsi| at the handover;
  - strict feasibility (every path ratio <= 1) against the physical ceiling, the fraction of
    cases whose heat-rate floor is <= 1 (results/eval_sets/floors/, heat_rate_floor.py);
  - guidance-attributable feasibility: heat rate <= max(1, floor), qbar <= 1, n <= 1 — a
    violation that no guidance could avoid is not counted against the policy;
  - bank-rate RMS (control activity).
Writes results/curves/robustness_curves.png and results/curves/robustness_table.md.
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
SCALES = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25)
# categorical slots 1-4 of the reference palette (adjacent forms; direct labels required)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
# every evaluated policy goes in the table; the figure draws FIGURE (<= 4 series)
POLICIES = [("v24b 30M (trained at 0.1x)", RES / "stage2_v24b/eval_S4_final_30M"),
            ("v25 4M (boundary 0.5x)", RES / "stage2_v25/eval_S4_ck4M"),
            ("v25 8M (boundary 0.7x)", RES / "stage2_v25/eval_S4_ck8M"),
            ("v25 11.5M (0.7x, heat floor)", RES / "stage2_v25/eval_S4_ck11p5M"),
            ("v26 6M (0.7x, margins, 8 deg bonuses)", RES / "stage2_v26/eval_S4_ck6M"),
            ("v26 8M (0.7x, margins, 8 deg bonuses)", RES / "stage2_v26/eval_S4_ck8M")]
FIGURE = ["v24b 30M (trained at 0.1x)", "v25 8M (boundary 0.7x)",
          "v26 6M (0.7x, margins, 8 deg bonuses)", "v26 8M (0.7x, margins, 8 deg bonuses)"]
TOL = 1e-3          # numerical tolerance on the attributable heat-rate check


SET_PREFIX = "S4"          # S4: VISTA aero dispersions; S5: C_L / C_D sigma capped at 5 %


def _set(s):
    return f"{SET_PREFIX}_scale_x{s:g}"


def load(base, s):
    f = base / f"{_set(s)}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    fl = RES / "eval_sets" / "floors" / f"{_set(s)}.csv"
    if fl.exists():
        d = d.merge(pd.read_csv(fl), on="id", how="left")
    return d


def metrics(d):
    floor = np.maximum(1.0, d["qdot_lb"]) if "qdot_lb" in d else 1.0
    attributable = (d["max_Qdot"] <= floor + TOL) & (d["max_qbar"] <= 1.0) & (d["max_n"] <= 1.0)
    excess = np.maximum(d["max_Qdot"] - floor, 0) + np.maximum(d["max_qbar"] - 1, 0) + np.maximum(d["max_n"] - 1, 0)
    return {"reach": float((d["outcome"] == "reached").mean()),
            "d_med": float(d["d_km"].median()), "d_p90": float(d["d_km"].quantile(0.9)),
            "fpa_med": float(d["dfpa_deg"].abs().median()), "psi_med": float(d["dpsi_deg"].abs().median()),
            "feasible": float(d["feasible"].mean()),
            "ceiling": float((d["qdot_lb"] <= 1.0).mean()) if "qdot_lb" in d else float("nan"),
            "attr_feasible": float(attributable.mean()),
            "attr_excess_mean": float(excess.mean()),
            "bank_rate_rms": float(d["bank_rate_rms_dps"].median()),
            "reversals": float(d["reversals"].median())}


def _style(ax, title, ylabel, logy=False):
    ax.set_facecolor(SURFACE)
    if logy:
        from matplotlib.ticker import FuncFormatter
        ax.set_yscale("log")
        fmt = FuncFormatter(lambda v, _pos: f"{v:g}")
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _pos: ""))
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    ax.set_xlabel("dispersion scale (x VISTA; 0.1 = realistic, 1 = stress test)", fontsize=8, color=INK2)
    ax.set_xticks(SCALES)
    ax.set_xticklabels([f"{s:g}" for s in SCALES])


def main():
    import sys
    global SET_PREFIX, POLICIES
    if len(sys.argv) > 1:
        SET_PREFIX = sys.argv[1]
    if SET_PREFIX != "S4":           # same policies, evaluation directories named after the set
        POLICIES = [(lab, base.parent / base.name.replace("eval_S4", f"eval_{SET_PREFIX}"))
                    for lab, base in POLICIES]
    rows = []
    for label, base in POLICIES:
        for s in SCALES:
            d = load(base, s)
            if d is not None:
                rows.append({"policy": label, "scale": s, "n": len(d), **metrics(d)})
    if not rows:
        raise SystemExit("no S4 results found")
    tab = pd.DataFrame(rows)
    out = RES / "curves"
    out.mkdir(parents=True, exist_ok=True)

    panels = [("d_med", "median landing error", "km", True),
              ("d_p90", "90th-percentile landing error", "km", True),
              ("feasible", "strictly feasible (all path ratios <= 1)", "fraction of cases", False),
              ("attr_feasible", "guidance-attributable feasibility", "fraction of cases", False),
              ("fpa_med", "median |flight-path-angle error|", "deg", True),
              ("psi_med", "median |heading error|", "deg", True)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), facecolor=SURFACE)
    for ax, (key, title, unit, logy) in zip(axes.flat, panels):
        _style(ax, title, unit, logy)
        if key == "feasible":
            ceil = tab.groupby("scale")["ceiling"].first()
            ax.plot(ceil.index, ceil.values, "--", color=MUTED, lw=1.5)
            ax.annotate("physical ceiling (heat-rate floor <= 1)", (ceil.index[0], ceil.values[0]),
                        textcoords="offset points", xytext=(4, 5), ha="left", fontsize=7, color=INK2)
        for c, label in zip(C, FIGURE):
            sub = tab[tab["policy"] == label].sort_values("scale")
            if sub.empty:
                continue
            ax.plot(sub["scale"], sub[key], "-o", color=c, lw=2, ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        if not logy:
            ax.set_ylim(0, 1.02)
    # one legend for the figure, plus direct labels on the first panel's line ends, pushed
    # apart vertically (in display space) so close end values do not collide
    ax0 = axes.flat[0]
    fig.canvas.draw()
    ends = []
    for label in FIGURE:
        sub = tab[tab["policy"] == label].sort_values("scale")
        if not sub.empty:
            x, y = sub["scale"].iloc[-1], sub["d_med"].iloc[-1]
            ends.append([ax0.transData.transform((x, y))[1], x, y, label.split(" (")[0]])
    ends.sort()
    gap = 11.0 * fig.dpi / 72.0
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for ypix, x, y, text in ends:
        dy_pt = (ypix - ax0.transData.transform((x, y))[1]) * 72.0 / fig.dpi
        ax0.annotate(text, (x, y), textcoords="offset points", xytext=(6, dy_pt), va="center",
                     fontsize=7, color=INK2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK)
    model = ("VISTA dispersions" if SET_PREFIX == "S4"
             else "VISTA dispersions, C_L / C_D sigma capped at 5 %")
    fig.suptitle(f"Robustness versus uncertainty level ({model}; 300 cases per scale, common random numbers)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    suffix = "" if SET_PREFIX == "S4" else f"_{SET_PREFIX}"
    fig.savefig(out / f"robustness_curves{suffix}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    cols = ["policy", "scale", "n", "reach", "d_med", "d_p90", "fpa_med", "psi_med", "feasible", "ceiling",
            "attr_feasible", "attr_excess_mean", "bank_rate_rms", "reversals"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in tab[cols].iterrows():
        lines.append("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in r) + " |")
    (out / f"robustness_table{suffix}.md").write_text("\n".join(lines) + "\n")
    print(out / f"robustness_curves{suffix}.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
