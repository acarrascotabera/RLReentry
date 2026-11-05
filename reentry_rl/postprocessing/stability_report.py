"""Figures and tables of the stability suite (validation/stability.py).

    python -m reentry_rl.postprocessing.stability_report [results/stability_v26]

Writes <dir>/figures/*.png and <dir>/stability_tables.md:
  fig_margins       landing error and feasibility versus each perturbation magnitude
                    (navigation errors, time delay, lag, rate limit, bank gain and bias)
  fig_disturbances  impulsive kicks by flight phase, density waves and bump
  fig_sensitivity   closed- vs open-loop terminal displacement per initial / model error
  fig_recovery      range-to-go deviation after a kick, closed vs open loop (nominal)
  fig_corridor      entry corridor (FPA x speed) and target footprint (lat x lon offsets)
  fig_jacobian      policy Jacobian norm along the trajectories
Margin of a perturbation class: the largest swept magnitude whose median landing error
stays within 2x the baseline and whose feasible fraction drops by at most 10 points.
"""
import json
import sys
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
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

from reentry_rl.physics import constants as Cn
from reentry_rl.physics.geodesy import great_circle_distance_m

ROOT = Path(__file__).resolve().parents[2]
# same palette and ink as robustness_curves.py
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
_DEG = np.pi / 180.0
LAW = "v26 6M"            # set from <dir>/suite.json in main()
_PLAIN = FuncFormatter(lambda v, _pos: f"{v:g}")       # 0.5, 1, 10, 100 rather than 10^x

MARGIN_PANELS = [("nav", "navigation error level", "x reference spec", True),
                 ("delay", "command time delay", "s", True),
                 ("lag", "extra first-order lag", "time constant [s]", True),
                 ("rate", "bank-rate limit", "deg/s (nominal 45)", False),
                 ("gain", "bank-command gain", "gain (nominal 1)", False),
                 ("bias", "bank-command bias", "deg", False)]
PHASES = ["early", "mid", "late"]
KICKS = ["fpa-1", "fpa+1", "psi+5", "V-100"]
KICK_LABEL = {"fpa-1": "FPA -1 deg", "fpa+1": "FPA +1 deg", "psi+5": "heading +5 deg", "V-100": "speed -100 m/s"}


def _style(ax, title, xlabel, ylabel, logy=False, logx=False):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(_PLAIN)
        ax.yaxis.set_minor_formatter(NullFormatter())
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(_PLAIN)
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _line(ax, x, y, c, label=None, ls="-"):
    ax.plot(x, y, ls, color=c, lw=2, marker="o", ms=5, markeredgecolor=SURFACE,
            markeredgewidth=1.2, label=label)


def load_margins(d):
    rows = []
    for f in sorted((d / "margins").glob("*_summary.json")):
        s = json.loads(f.read_text())
        st = s["stats"]
        rows.append({"condition": s["condition"], "group": s["group"], "magnitude": s["magnitude"],
                     "d_med": st["d_km"]["median"], "d_p95": st["d_km"]["p95"],
                     "fpa_med": st["abs_dfpa_deg"]["median"], "psi_med": st["abs_dpsi_deg"]["median"],
                     "feasible": s["feasible_rate"], "reached": s["outcomes"].get("reached", 0) / s["n"],
                     "robust_score": s["robust_score"], "bank_rate_rms": s["bank_rate_rms_dps_mean"],
                     "reversals": s["reversals_mean"], "n": s["n"]})
    for f in sorted((d / "margins").glob("*.csv")):
        df = pd.read_csv(f)
        for r in rows:
            if r["condition"] == f.stem:
                r["d_p90"] = float(df["d_km"].quantile(0.9))
    return pd.DataFrame(rows)


def margin_of(tab, group, base, larger_is_worse=True):
    """Largest magnitude (in the direction away from nominal) still within the limits."""
    sub = tab[tab["group"] == group].copy()
    ok = (sub["d_med"] <= 2.0 * base["d_med"]) & (sub["feasible"] >= base["feasible"] - 0.10)
    sub["ok"] = ok
    return sub


def fig_margins(tab, out):
    base = tab[tab["group"] == "baseline"].iloc[0]
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.4), facecolor=SURFACE, sharey="row")
    for j, (g, title, xl, logx) in enumerate(MARGIN_PANELS):
        sub = tab[tab["group"] == g]
        nominal_x = {"nav": None, "delay": None, "lag": None, "rate": 45.0, "gain": 1.0, "bias": 0.0}[g]
        add = [] if nominal_x is None else [dict(magnitude=nominal_x, d_med=base["d_med"], d_p90=base["d_p90"],
                                                  feasible=base["feasible"])]
        sub = pd.concat([sub, pd.DataFrame(add)]).sort_values("magnitude")
        ax = axes[0, j]
        _style(ax, title, xl, "landing error [km]" if j == 0 else "", logy=True, logx=logx)
        ax.axhline(base["d_med"], color=MUTED, ls="--", lw=1.2)
        _line(ax, sub["magnitude"], sub["d_p90"], C[1], "90th percentile")
        _line(ax, sub["magnitude"], sub["d_med"], C[0], "median")
        ax = axes[1, j]
        _style(ax, "", xl, "feasible fraction" if j == 0 else "", logx=logx)
        ax.axhline(base["feasible"], color=MUTED, ls="--", lw=1.2)
        _line(ax, sub["magnitude"], sub["feasible"], C[2], "strictly feasible")
        ax.set_ylim(0, 1.02)
        if logx:
            for a in axes[:, j]:
                a.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8]))
    axes[0, 0].annotate("baseline median", (axes[0, 0].get_xlim()[0], base["d_med"]), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=INK2)
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(f"Stability margins of {LAW}: one perturbation class at a time on the nominal mission + 100 "
                 "cases at 0.1x VISTA dispersions (dashed: unperturbed baseline)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out / "fig_margins.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_disturbances(tab, out):
    base = tab[tab["group"] == "baseline"].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    x = np.arange(len(PHASES))
    for k, (key, ylab, logy) in enumerate([("d_med", "median landing error [km]", True),
                                           ("feasible", "feasible fraction", False)]):
        ax = axes[k]
        _style(ax, "impulsive kick at energy-to-go 0.7 / 0.35 / 0.08", "flight phase of the kick", ylab, logy=logy)
        ax.axhline(base[key], color=MUTED, ls="--", lw=1.2)
        for c, kick in zip(C, KICKS):
            sub = tab[tab["group"] == f"impulse_{kick}"].set_index("condition")
            y = [sub.loc[f"imp_{kick}_{ph}", key] if f"imp_{kick}_{ph}" in sub.index else np.nan for ph in PHASES]
            _line(ax, x, y, c, KICK_LABEL[kick])
        ax.set_xticks(x, PHASES)
        if not logy:
            ax.set_ylim(0, 1.02)
    ax = axes[2]
    _style(ax, "altitude-varying density perturbations", "", "median landing error [km]")
    names, vals, feas = ["baseline"], [base["d_med"]], [base["feasible"]]
    for _, r in tab[tab["group"].isin(["rho_wave", "rho_bump"])].sort_values(["group", "magnitude"]).iterrows():
        names.append(f"waves ±{r['magnitude'] * 100:.0f} %" if r["group"] == "rho_wave"
                     else f"bump {r['magnitude'] * 100:+.0f} %")
        vals.append(r["d_med"])
        feas.append(r["feasible"])
    xx = np.arange(len(names))
    ax.bar(xx, vals, width=0.6, color=C[0], edgecolor=SURFACE, linewidth=2)
    for xi, v, f in zip(xx, vals, feas):
        ax.annotate(f"{v:.1f} km\n{f * 100:.0f} % feas.", (xi, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7, color=INK2)
    ax.set_xticks(xx, names, rotation=20, ha="right")
    ax.set_ylim(0, max(vals) * 1.25)                       # headroom for the value labels
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Mid-flight disturbances (nominal + 100 cases at 0.1x; dashed: unperturbed baseline)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    fig.savefig(out / "fig_disturbances.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def sensitivity_table(d):
    df = pd.read_csv(d / "sensitivity" / "sensitivity.csv")
    rows = []
    for mode in ("closed", "open"):
        m = df[df["mode"] == mode]
        ref = m[m["param"] == "none"].iloc[0]
        lat_f = np.deg2rad(Cn.TARGET["lat_deg"])
        for _, r in m[m["param"] != "none"].iterrows():
            dn = (r["dlat_deg"] - ref["dlat_deg"]) * _DEG * Cn.R0 / 1e3
            de = (r["dlon_deg"] - ref["dlon_deg"]) * _DEG * Cn.R0 * np.cos(lat_f) / 1e3
            rows.append({"mode": mode, "param": r["param"], "mult": r["mult"], "delta": r["delta"],
                         "disp_km": float(np.hypot(dn, de)), "dfpa_f": r["dfpa_deg"] - ref["dfpa_deg"],
                         "dpsi_f": r["dpsi_deg"] - ref["dpsi_deg"],
                         "outcome": r["outcome"], "feasible": r["feasible"]})
    return pd.DataFrame(rows)


SENS_LABEL = {"ic_fpa_deg": "entry FPA 0.05 deg", "ic_V_mps": "entry speed 10 m/s", "ic_alt_m": "entry altitude 1 km",
              "ic_psi_deg": "entry heading 0.2 deg", "ic_lat_deg": "entry latitude 0.1 deg", "k_rho": "density 2 %",
              "k_CL": "C_L 2 %", "k_CD": "C_D 2 %", "k_mass": "mass 2 %"}


def fig_sensitivity(st, out):
    params = list(SENS_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE, sharey=True)
    for ax, m in zip(axes, (1.0, 5.0)):
        _style(ax, f"perturbation of {m:g} x the listed size (mean over +/-)", "terminal displacement [km]", "",
               logy=False, logx=True)
        y = np.arange(len(params))
        for c, mode, lab in ((C[0], "closed", "closed loop (policy)"), (C[1], "open", "open loop (nominal commands)")):
            v = [st[(st["mode"] == mode) & (st["param"] == p) & (st["mult"].abs() == m)]["disp_km"].mean() for p in params]
            ax.plot(np.maximum(v, 1e-3), y, "o", color=c, ms=7, markeredgecolor=SURFACE, markeredgewidth=1.2, label=lab)
        for yi, p in enumerate(params):
            vc = st[(st["mode"] == "closed") & (st["param"] == p) & (st["mult"].abs() == m)]["disp_km"].mean()
            vo = st[(st["mode"] == "open") & (st["param"] == p) & (st["mult"].abs() == m)]["disp_km"].mean()
            ax.plot([max(vc, 1e-3), max(vo, 1e-3)], [yi, yi], color=GRID, lw=2, zorder=0)
            ax.annotate(f"x{vo / max(vc, 1e-6):.0f}", (max(vo, 1e-3), yi), textcoords="offset points", xytext=(6, -3),
                        fontsize=7, color=INK2)
        ax.set_yticks(y, [SENS_LABEL[p] for p in params])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Terminal sensitivity on the nominal mission: closed loop vs open-loop replay "
                 "(annotation: open / closed attenuation ratio)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    fig.savefig(out / "fig_sensitivity.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _rtg_km(tr):
    lat_t, lon_t = Cn.TARGET["lat_deg"] * _DEG, Cn.TARGET["lon_deg"] * _DEG
    return np.array([great_circle_distance_m(la * _DEG, lo * _DEG, lat_t, lon_t)
                     for la, lo in zip(tr["lat_deg"], tr["lon_deg"])]) / 1e3


def fig_recovery(d, out):
    rd = d / "recovery"
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), facecolor=SURFACE, sharey=True)
    for ax, kick in zip(axes, KICKS):
        _style(ax, KICK_LABEL[kick], "energy-to-go (1 = entry, 0 = handover)", "range-to-go deviation [km]"
               if kick == KICKS[0] else "", logy=False)
        ax.set_yscale("symlog", linthresh=1.0)
        for mode, ls in (("closed", "-"), ("open", "--")):
            ref = pd.read_csv(rd / f"traj_{mode}_none_none.csv")
            e_ref, r_ref = ref["energy_togo"].to_numpy()[::-1], _rtg_km(ref)[::-1]
            for c, ph in zip(C, PHASES):
                f = rd / f"traj_{mode}_{kick}_{ph}.csv"
                if not f.exists():
                    continue
                tr = pd.read_csv(f)
                dev = _rtg_km(tr) - np.interp(tr["energy_togo"], e_ref, r_ref)
                ax.plot(tr["energy_togo"], dev, ls, color=c, lw=1.8,
                        label=f"{ph} kick, {mode} loop")
        ax.invert_xaxis()
        ax.axhline(0, color=MUTED, lw=1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Recovery after an impulsive kick on the nominal mission: deviation of range-to-go from the "
                 "undisturbed trajectory at equal energy (solid: policy, dashed: open-loop nominal commands)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(out / "fig_recovery.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def grid_floors(d, name):
    """Heat-rate floor (validation/heat_rate_floor.py) of every grid point, cached as
    <name>/floors.csv: a grid point whose floor exceeds 1 is heat-rate infeasible for every
    guidance law, so only the excess over max(1, floor) is counted against the policy."""
    fc = d / name / "floors.csv"
    if fc.exists():
        return pd.read_csv(fc)
    from dataclasses import fields
    from reentry_rl.envs.scenario import Scenario
    from reentry_rl.validation import heat_rate_floor as hf
    hf._init()
    keys = [f.name for f in fields(Scenario)]
    df = pd.read_csv(d / name / f"{name}.csv")
    rows = [hf._floor((r["id"], {k: r[k] for k in keys})) for _, r in df.iterrows()]
    fl = pd.DataFrame(rows)
    fl.to_csv(fc, index=False)
    return fl


def _with_floors(d, name, f):
    """Grid results with the cached floors (newer runs also log qdot_lb themselves: same values)."""
    return pd.read_csv(f).drop(columns="qdot_lb", errors="ignore").merge(grid_floors(d, name), on="id")


def _attributable(df):
    floor = np.maximum(1.0, df["qdot_lb"])
    return (df["outcome"] == "reached") & (df["max_Qdot"] <= floor + 1e-3) & (df["max_qbar"] <= 1.0) \
        & (df["max_n"] <= 1.0)


def fig_corridor(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURFACE)
    norm = LogNorm(vmin=0.3, vmax=300)
    specs = [("corridor", "ic_fpa_deg_grid", "ic_V_mps_grid", "entry corridor",
              "entry FPA offset [deg] (nominal -0.5)", "entry speed offset [m/s] (nominal 7450)"),
             ("footprint", "tc_lon_deg_grid", "tc_lat_deg_grid", "reachable target footprint",
              "target longitude offset [deg]", "target latitude offset [deg]")]
    im = None
    for ax, (name, kx, ky, title, xl, yl) in zip(axes, specs):
        f = d / name / f"{name}.csv"
        _style(ax, title, xl, yl)
        ax.grid(False)
        if not f.exists():
            continue
        df = _with_floors(d, name, f)
        xs, ys = np.sort(df[kx].unique()), np.sort(df[ky].unique())
        Z = np.full((len(ys), len(xs)), np.nan)
        for _, r in df.iterrows():
            Z[np.searchsorted(ys, r[ky]), np.searchsorted(xs, r[kx])] = r["d_km"]
        im = ax.pcolormesh(np.arange(len(xs) + 1) - 0.5, np.arange(len(ys) + 1) - 0.5, Z, norm=norm,
                           cmap="Blues_r", edgecolors=SURFACE, linewidth=1.5)
        att = _attributable(df)
        phys = df[~att & (df["qdot_lb"] > 1.0)]          # heat-rate floor > 1: no guidance can comply
        bad = df[~att & (df["qdot_lb"] <= 1.0)]
        ax.plot([np.searchsorted(xs, v) for v in phys[kx]], [np.searchsorted(ys, v) for v in phys[ky]], "o",
                mfc="none", mec=INK2, ms=6, mew=1.2, label="heat rate infeasible for any guidance (floor > 1)")
        ax.plot([np.searchsorted(xs, v) for v in bad[kx]], [np.searchsorted(ys, v) for v in bad[ky]], "x",
                color=C[1], ms=6, mew=1.6, label="guidance-attributable violation / not reached")
        # nominal: grid coordinates by interpolation (the grid need not contain 0)
        ax.plot([np.interp(0.0, xs, np.arange(len(xs)))], [np.interp(0.0, ys, np.arange(len(ys)))], "s",
                mfc="none", mec=INK, ms=12, mew=1.4, label="nominal")
        ax.set_xticks(range(len(xs))[::2], [f"{v:g}" for v in xs][::2])
        ax.set_yticks(range(len(ys))[::2], [f"{v:g}" for v in ys][::2])
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.17, top=0.88, wspace=0.25)
    if im is not None:
        cax = fig.add_axes([0.90, 0.17, 0.015, 0.71])     # own axis: never overlaps the panels
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("landing error [km]", fontsize=8, color=INK2)
        cb.ax.tick_params(labelsize=7, colors=INK2)
        cb.ax.yaxis.set_major_formatter(_PLAIN)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(f"Entry corridor and target footprint of {LAW} (deterministic grids, nominal model)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(out / "fig_corridor.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_jacobian(d, out):
    f = d / "jacobian" / "jacobian.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    _style(ax, "policy Jacobian norm ||d command / d normalized observation||_2 along the trajectories",
           "energy-to-go (1 = entry, 0 = handover)", "deg per unit normalized feature", logy=True)
    bins = np.linspace(0, 1, 41)
    disp = df[df["case"] != "nominal"]
    idx = np.digitize(disp["energy_togo"], bins)
    g = disp.groupby(idx)["jac_norm"]
    centers = np.array([0.5 * (bins[max(i - 1, 0)] + bins[min(i, len(bins) - 1)]) for i in g.median().index])
    ax.fill_between(centers, g.quantile(0.1).values, g.quantile(0.9).values, color=C[0], alpha=0.2, lw=0,
                    label="dispersed cases, 10-90 %")
    nom = df[df["case"] == "nominal"]
    ax.plot(nom["energy_togo"], nom["jac_norm"], color=C[0], lw=1.6, label="nominal")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "fig_jacobian.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return df


def _md(df, cols, fmt="{:.3g}"):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r) + " |")
    return "\n".join(lines)


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/stability_v26"
    out = d / "figures"
    out.mkdir(parents=True, exist_ok=True)
    global LAW
    sj = d / "suite.json"
    if sj.exists() and json.loads(sj.read_text()).get("guidance") == "fnpeg":
        LAW = "the FNPEG baseline"
    md = [f"# Stability analyses of {LAW}", ""]
    tab = load_margins(d)
    if not tab.empty:
        fig_margins(tab, out)
        fig_disturbances(tab, out)
        base = tab[tab["group"] == "baseline"].iloc[0]
        tab["within_margin"] = (tab["d_med"] <= 2.0 * base["d_med"]) & (tab["feasible"] >= base["feasible"] - 0.10)
        md += ["## Margin sweeps (nominal + 100 cases at 0.1x)", "",
               "within_margin: median landing error <= 2x baseline and feasible fraction >= baseline - 10 points.", "",
               _md(tab.sort_values(["group", "magnitude"]),
                   ["condition", "group", "magnitude", "d_med", "d_p90", "d_p95", "fpa_med", "psi_med", "feasible",
                    "reached", "robust_score", "bank_rate_rms", "reversals", "within_margin"]), ""]
    if (d / "sensitivity" / "sensitivity.csv").exists():
        st = sensitivity_table(d)
        fig_sensitivity(st, out)
        agg = (st.assign(absm=st["mult"].abs()).groupby(["param", "absm", "mode"])["disp_km"].mean()
               .unstack("mode").reset_index())
        agg["attenuation"] = agg["open"] / agg["closed"].clip(lower=1e-6)
        md += ["## Terminal sensitivity, closed vs open loop (nominal mission)", "",
               "Mean terminal displacement [km] over the + and - perturbation.", "",
               _md(agg, ["param", "absm", "closed", "open", "attenuation"]), ""]
    if (d / "recovery").exists():
        fig_recovery(d, out)
        rs = pd.read_csv(d / "recovery" / "recovery_summary.csv")
        md += ["## Recovery after impulsive kicks (nominal mission)", "",
               _md(rs, ["mode", "kick", "phase", "d_km", "dfpa_deg", "dpsi_deg", "max_ratio", "feasible"]), ""]
    fig_corridor(d, out)
    for name in ("corridor", "footprint"):
        f = d / name / f"{name}.csv"
        if f.exists():
            g = _with_floors(d, name, f)
            ok = (g["outcome"] == "reached") & g["feasible"].astype(bool)
            att = _attributable(g)
            md += [f"## {name}", "", f"{len(g)} grid points: reached and strictly feasible {ok.mean() * 100:.0f} %, "
                   f"heat-rate floor > 1 (infeasible for any guidance) {(g['qdot_lb'] > 1).mean() * 100:.0f} %, "
                   f"guidance-attributable feasible {att.mean() * 100:.0f} %, "
                   f"landing error <= 5 km {(g['d_km'] <= 5).mean() * 100:.0f} % "
                   f"(and attributable feasible {((g['d_km'] <= 5) & att).mean() * 100:.0f} %), "
                   f"median landing error {g['d_km'].median():.2f} km.", ""]
    jac = fig_jacobian(d, out)
    if jac is not None:
        md += ["## Policy Jacobian norm", "",
               f"Over {len(jac)} decisions on the nominal + 20 cases: median {jac['jac_norm'].median():.3g}, "
               f"p95 {jac['jac_norm'].quantile(0.95):.3g}, max {jac['jac_norm'].max():.3g} deg per unit "
               "normalized feature.", ""]
    comb = []
    for f in sorted((d / "combined").glob("*_summary.json")) if (d / "combined").exists() else []:
        s = json.loads(f.read_text())
        st_ = s["stats"]
        comb.append({"set": s["set"], "condition": s["condition"], "d_med": st_["d_km"]["median"],
                     "d_p95": st_["d_km"]["p95"], "fpa_med": st_["abs_dfpa_deg"]["median"],
                     "psi_med": st_["abs_dpsi_deg"]["median"], "feasible": s["feasible_rate"],
                     "robust_score": s["robust_score"]})
    if comb:
        md += ["## Combined realistic perturbations (300 cases per set)", "",
               "nav errors at spec + 1 s delay + 1 s lag + 30 deg/s bank rate + 10 % density waves.", "",
               _md(pd.DataFrame(comb), ["set", "condition", "d_med", "d_p95", "fpa_med", "psi_med", "feasible",
                                        "robust_score"]), ""]
    (d / "stability_tables.md").write_text("\n".join(md), encoding="utf-8")
    print(d / "stability_tables.md")


if __name__ == "__main__":
    main()
