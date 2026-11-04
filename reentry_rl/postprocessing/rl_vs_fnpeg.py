"""RL policy versus the FNPEG baseline: final Monte Carlo and robustness versus dispersion scale.

    python -m reentry_rl.postprocessing.rl_vs_fnpeg

Inputs are the outputs of validation/final_mc.py (same cases and seeds for both laws):
  results/final_mc/<law>_<condition>.csv        S6 final MC set, ideal and HIFI
  results/final_mc/scale/<law>_S5_scale_x*.csv  S5 robustness-vs-scale sets
Writes results/final_mc/figures/fig_final_mc_tails.png (exceedance curves of the terminal
errors and path-constraint peaks, handover speed distribution), fig_scale.png (median / p95
landing error, strict and attributable feasibility, median FPA / heading error versus scale)
and results/final_mc/scale_tables.md.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator

from reentry_rl.postprocessing.robustness_curves import (C, INK, INK2, MUTED, GRID, SURFACE, SCALES,
                                                         TOL)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "final_mc"
LAWS = [("rl", "RL policy (v26 6M)", C[0]), ("fnpeg", "FNPEG baseline", C[1])]
CONDS = [("ideal", "as designed", "-"), ("hifi", "HIFI models", "--")]
_PLAIN = FuncFormatter(lambda v, _pos: f"{v:g}")
_NONE = FuncFormatter(lambda v, _pos: "")


def _ax_style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)


def _log(ax, axis, lim=None):
    """Log scale with plain tick labels; 1-2-5 ticks when the axis spans fewer than 3 decades."""
    getattr(ax, f"set_{axis}scale")("log")
    if lim is not None:
        getattr(ax, f"set_{axis}lim")(*lim)
    lo, hi = getattr(ax, f"get_{axis}lim")()
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0) if np.log10(hi / lo) < 3 else (1.0,)))
    a.set_major_formatter(_PLAIN)
    a.set_minor_formatter(_NONE)


# -- final MC: exceedance curves ---------------------------------------------------
TAIL_PANELS = [("d_km", "landing error", "miss distance at the handover [km]", 1.0, "1 km"),
               ("dfpa_deg", "flight-path-angle error", "|FPA error| [deg]", None, None),
               ("dpsi_deg", "heading error", "|heading error| [deg]", None, None),
               ("max_n", "peak load factor", "peak load factor / limit", 1.0, "limit"),
               ("max_qbar", "peak dynamic pressure", "peak dynamic pressure / limit", 1.0, "limit")]


def load_final():
    runs = {}
    for law, _, _ in LAWS:
        for cond, _, _ in CONDS:
            f = OUT / f"{law}_{cond}.csv"
            if f.exists():
                runs[(law, cond)] = pd.read_csv(f)
    return runs


def fig_final_tails(runs, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), facecolor=SURFACE)
    n_max = max(len(d) for d in runs.values())
    for ax, (key, title, xlabel, ref, ref_lab) in zip(axes.flat, TAIL_PANELS):
        _ax_style(ax, f"{title}: fraction of cases exceeding x", xlabel, "fraction of cases > x")
        lo = np.inf
        for law, _, c in LAWS:
            for cond, _, ls in CONDS:
                d = runs.get((law, cond))
                if d is None:
                    continue
                x = np.sort(np.abs(d[key].to_numpy()))
                n = len(x)
                y = (n - np.arange(n)) / n
                ax.step(x, y, where="post", color=c, ls=ls, lw=2)
                lo = min(lo, np.quantile(x, 0.005))
        if key in ("d_km", "dfpa_deg", "dpsi_deg"):
            _log(ax, "x", (max(lo, 1e-3), ax.get_xlim()[1]))
        _log(ax, "y", (0.5 / n_max, 1.05))
        ax.axhline(0.0027, color=MUTED, lw=1, ls=":")
        if key == "d_km":
            ax.annotate("p99.73 level", (0, 0.0027), xycoords=("axes fraction", "data"), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, color=INK2)
        if ref is not None:
            ax.axvline(ref, color=MUTED, lw=1.2, ls="--")
            ax.annotate(ref_lab, (ref, 0), xycoords=("data", "axes fraction"), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, color=INK2)

    # handover speed: distribution, not a tail
    ax = axes.flat[5]
    _ax_style(ax, "handover speed (cumulative distribution)", "speed at the 25 km handover [m/s]",
              "fraction of cases <= x")
    for law, _, c in LAWS:
        for cond, _, ls in CONDS:
            d = runs.get((law, cond))
            if d is None:
                continue
            x = np.sort(d["V_f_mps"].to_numpy())
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=c, ls=ls, lw=2)
    ax.set_ylim(0, 1.02)

    # direct labels on the landing-error panel, each at an exceedance level and on a side where
    # its curve is clear of the others (the three non-ideal curves run close together)
    ax0 = axes.flat[0]
    spots = {("fnpeg", "ideal"): (0.1, 1), ("fnpeg", "hifi"): (0.006, -1), ("rl", "ideal"): (0.03, 1),
             ("rl", "hifi"): (0.1, 1)}
    for (law, cond), (lev, side) in spots.items():
        d = runs.get((law, cond))
        if d is None:
            continue
        x = float(np.quantile(d["d_km"], 1 - lev))
        ax0.annotate({"rl": "RL", "fnpeg": "FNPEG"}[law] + (" HIFI" if cond == "hifi" else ""), (x, lev),
                     xytext=(4 * side, 0), textcoords="offset points", ha="left" if side > 0 else "right",
                     va="center", fontsize=7, color=INK2)
    handles = [Line2D([], [], color=c, ls=ls, lw=2, label=f"{name}, {cname}")
               for law, name, c in LAWS for cond, cname, ls in CONDS if (law, cond) in runs]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK)
    n_txt = ", ".join(sorted({str(len(d)) for d in runs.values()}))
    fig.suptitle(f"Final Monte Carlo, RL versus FNPEG (set S6, {n_txt} cases per run, common random numbers)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    p = out / "fig_final_mc_tails.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return p


# -- robustness versus dispersion scale ----------------------------------------------
def scale_metrics(d, summ):
    floor = np.maximum(1.0, d["qdot_lb"])
    attributable = (d["max_Qdot"] <= floor + TOL) & (d["max_qbar"] <= 1.0) & (d["max_n"] <= 1.0)
    ci = summ["rates"]["feasible"]["ci95"] if summ else [np.nan, np.nan]
    return {"n": len(d), "reach": float((d["outcome"] == "reached").mean()),
            "d_med": float(d["d_km"].median()), "d_p95": float(d["d_km"].quantile(0.95)),
            "d_max": float(d["d_km"].max()),
            "fpa_med": float(d["dfpa_deg"].abs().median()), "psi_med": float(d["dpsi_deg"].abs().median()),
            "feasible": float(d["feasible"].mean()), "feas_lo": ci[0], "feas_hi": ci[1],
            "ceiling": float((d["qdot_lb"] <= 1.0).mean()),
            "attr_feasible": float(attributable.mean()),
            "n_max": float(d["max_n"].max()), "qbar_max": float(d["max_qbar"].max()),
            "J_med": float(d["J_ratio"].median())}


def load_scale():
    rows = []
    for law, _, _ in LAWS:
        for s in SCALES:
            f = OUT / "scale" / f"{law}_S5_scale_x{s:g}.csv"
            if not f.exists():
                continue
            sj = f.with_name(f.stem + "_summary.json")
            summ = json.loads(sj.read_text()) if sj.exists() else None
            rows.append({"law": law, "scale": s, **scale_metrics(pd.read_csv(f), summ)})
    return pd.DataFrame(rows)


def fig_scale(tab, out):
    panels = [("d_med", "median landing error", "km", True),
              ("d_p95", "95th-percentile landing error", "km", True),
              ("feasible", "strictly feasible (all path ratios <= 1), 95 % CI", "fraction of cases", False),
              ("attr_feasible", "guidance-attributable feasibility", "fraction of cases", False),
              ("fpa_med", "median |flight-path-angle error|", "deg", True),
              ("psi_med", "median |heading error|", "deg", True)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), facecolor=SURFACE)
    for ax, (key, title, unit, logy) in zip(axes.flat, panels):
        _ax_style(ax, title, "dispersion scale (x VISTA; 0.1 = realistic, 1 = stress test)", unit)
        ax.set_xticks(SCALES)
        ax.set_xticklabels([f"{s:g}" for s in SCALES])
        if key == "feasible":
            ceil = tab.groupby("scale")["ceiling"].first()
            ax.plot(ceil.index, ceil.values, "--", color=MUTED, lw=1.5)
            ax.annotate("physical ceiling (heat-rate floor <= 1)", (ceil.index[0], ceil.values[0]),
                        textcoords="offset points", xytext=(4, 5), ha="left", fontsize=7, color=INK2)
        for law, name, c in LAWS:
            sub = tab[tab["law"] == law].sort_values("scale")
            if sub.empty:
                continue
            if key == "feasible":
                ax.fill_between(sub["scale"], sub["feas_lo"], sub["feas_hi"], color=c, alpha=0.15, lw=0)
            ax.plot(sub["scale"], sub[key], "-o", color=c, lw=2, ms=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, label=name)
        if logy:
            _log(ax, "y")
        else:
            ax.set_ylim(0, 1.02)
    ax0 = axes.flat[0]
    for law, name, _ in LAWS:
        sub = tab[tab["law"] == law].sort_values("scale")
        if not sub.empty:
            ax0.annotate(name.split(" (")[0], (sub["scale"].iloc[-1], sub["d_med"].iloc[-1]),
                         textcoords="offset points", xytext=(-4, 7), ha="right", fontsize=7, color=INK2)
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Robustness versus uncertainty level, RL versus FNPEG (S5: VISTA dispersions, C_L / C_D sigma "
                 "capped at 5 %; 300 cases per scale, common random numbers)", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    p = out / "fig_scale.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return p


def scale_table(tab):
    cols = [("scale", "{:g}"), ("n", "{:d}"), ("reach", "{:.3f}"), ("d_med", "{:.3g}"), ("d_p95", "{:.3g}"),
            ("d_max", "{:.3g}"), ("fpa_med", "{:.3g}"), ("psi_med", "{:.3g}"), ("feasible", "{:.3f}"),
            ("ceiling", "{:.3f}"), ("attr_feasible", "{:.3f}"), ("n_max", "{:.2f}"), ("qbar_max", "{:.2f}"),
            ("J_med", "{:.3f}")]
    lines = ["# Robustness versus dispersion scale: RL versus FNPEG", "",
             "S5 sets (VISTA dispersions, C_L / C_D 1-sigma capped at 5 %), 300 cases per scale, the same "
             "cases and seeds for both laws. Miss distances in km, angles in deg; feasible = every path "
             "ratio <= 1 (95 % Clopper-Pearson interval in brackets); ceiling = heat-rate floor <= 1; "
             "attributable = heat rate <= max(1, floor), qbar and n <= 1.", ""]
    for law, name, _ in LAWS:
        sub = tab[tab["law"] == law].sort_values("scale")
        if sub.empty:
            continue
        lines += [f"## {name}", "", "| " + " | ".join(k for k, _ in cols) + " |", "|" + "---|" * len(cols)]
        for _, r in sub.iterrows():
            vals = []
            for k, fmt in cols:
                v = fmt.format(int(r[k]) if fmt == "{:d}" else r[k])
                if k == "feasible":
                    v += f" [{r['feas_lo']:.3f}, {r['feas_hi']:.3f}]"
                vals.append(v)
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    p = OUT / "scale_tables.md"
    p.write_text("\n".join(lines))
    return p


# -- stability margins (validation/stability.py, same 101-case base set) ---------------------
SUITES = {"rl": ROOT / "results" / "stability_v26", "fnpeg": ROOT / "results" / "stability_fnpeg"}
# group, title, x label, log x, nominal magnitude (drawn from the baseline run)
MARGIN_GROUPS = [("nav", "navigation errors (white)", "x reference spec", True, None),
                 ("nav_gm", "navigation drift (Gauss-Markov)", "x spec (1 % accelerometer scale)", True, None),
                 ("delay", "command time delay", "s", True, None),
                 ("lag", "extra first-order lag", "time constant [s]", True, None),
                 ("rate", "bank-rate limit", "deg/s (nominal 45)", False, 45.0),
                 ("acc", "attitude acceleration limit", "deg/s^2 (bank and AoA)", True, None),
                 ("gain", "bank-command gain", "gain (nominal 1)", False, 1.0),
                 ("bias", "bank-command bias", "deg", False, 0.0),
                 ("rho_gm", "density perturbation (Gauss-Markov)", "x reference level", True, None),
                 ("wind", "winds", "x reference level", True, None)]


def load_suites():
    from reentry_rl.postprocessing.stability_report import load_margins
    tabs = {}
    for law, d in SUITES.items():
        if (d / "margins").is_dir():
            t = load_margins(d)
            if not t.empty and (t["group"] == "baseline").any():
                tabs[law] = t
    return tabs


def _with_nominal(t, group, nominal):
    sub = t[t["group"] == group]
    if nominal is not None:
        b = t[t["group"] == "baseline"].iloc[0]
        sub = pd.concat([sub, pd.DataFrame([dict(magnitude=nominal, d_med=b["d_med"], d_p95=b["d_p95"],
                                                 feasible=b["feasible"])])])
    return sub.sort_values("magnitude")


def fig_margins_compare(tabs, out):
    fig, axes = plt.subplots(4, 5, figsize=(18, 12.5), facecolor=SURFACE)
    for k, (g, title, xl, logx, nominal) in enumerate(MARGIN_GROUPS):
        r, c = 2 * (k // 5), k % 5
        top, bot = axes[r, c], axes[r + 1, c]
        _ax_style(top, title, "", "landing error [km]" if c == 0 else "")
        _ax_style(bot, "", xl, "strictly feasible" if c == 0 else "")
        for law, name, col in LAWS:
            t = tabs.get(law)
            if t is None:
                continue
            sub = _with_nominal(t, g, nominal)
            if sub.empty:
                continue
            base = t[t["group"] == "baseline"].iloc[0]
            top.fill_between(sub["magnitude"], sub["d_med"], sub["d_p95"], color=col, alpha=0.12, lw=0)
            top.plot(sub["magnitude"], sub["d_med"], "-o", color=col, lw=2, ms=5, markeredgecolor=SURFACE,
                     markeredgewidth=1.2, label=f"{name}: median (band to p95)")
            top.axhline(base["d_med"], color=col, lw=1, ls=":")
            bot.plot(sub["magnitude"], sub["feasible"], "-o", color=col, lw=2, ms=5, markeredgecolor=SURFACE,
                     markeredgewidth=1.2)
            bot.axhline(base["feasible"], color=col, lw=1, ls=":")
        _log(top, "y")
        bot.set_ylim(0, 1.02)
        if logx:
            for a in (top, bot):
                a.set_xscale("log")
                a.xaxis.set_major_locator(FixedLocator(sorted(set(_with_nominal(
                    next(iter(tabs.values())), g, nominal)["magnitude"]))))
                a.xaxis.set_major_formatter(_PLAIN)
                a.xaxis.set_minor_formatter(_NONE)
        top.set_xlim(bot.get_xlim())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, lw=1, ls=":"))
    labels.append("unperturbed baseline (each law)")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Stability margins, RL versus FNPEG: one perturbation class at a time on the nominal mission "
                 "+ 100 cases at 0.1x VISTA dispersions (same cases and seeds)", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    p = out / "fig_margins_compare.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return p


def margins_table(tabs):
    """Every margin condition side by side: median / p95 miss [km], strictly feasible."""
    keys = sorted(set().union(*[set(zip(t["group"], t["magnitude"], t["condition"])) for t in tabs.values()]))
    laws = [(law, name) for law, name, _ in LAWS if law in tabs]
    lines = ["# Stability margins: RL versus FNPEG", "",
             "Base set: nominal mission + 100 cases at 0.1x VISTA dispersions, the same cases and seeds for "
             "both laws; one perturbation class per condition. Cells: median / p95 miss [km], strictly "
             "feasible fraction.", "",
             "| group | magnitude | condition | " + " | ".join(n for _, n in laws) + " |",
             "|---|---|---|" + "---|" * len(laws)]
    for g, m, cond in keys:
        cells = []
        for law, _ in laws:
            r = tabs[law][tabs[law]["condition"] == cond]
            cells.append("" if r.empty else f"{r['d_med'].iloc[0]:.3g} / {r['d_p95'].iloc[0]:.3g}, "
                                            f"{r['feasible'].iloc[0]:.2f}")
        lines.append(f"| {g} | {m:g} | {cond} | " + " | ".join(cells) + " |")
    p = OUT / "margins_tables.md"
    p.write_text("\n".join(lines) + "\n")
    return p


def main():
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    runs = load_final()
    if runs:
        print(fig_final_tails(runs, figs))
    tab = load_scale()
    if not tab.empty:
        print(fig_scale(tab, figs))
        print(scale_table(tab))
    tabs = load_suites()
    if tabs:
        print(fig_margins_compare(tabs, figs))
        print(margins_table(tabs))


if __name__ == "__main__":
    main()
