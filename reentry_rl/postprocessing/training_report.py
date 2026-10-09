"""Training report of one run: learning curves, flight regimes, milestones, health.

Reads <run>/config.json, train_log.txt, eval/validation_history.csv,
eval/curriculum.csv and best/best.json, and writes to <run>/diag/report/:

  learning_curves.png  validation terminal state (landing error, peak path-
                       constraint ratio, FPA/heading errors, AoA spread) and
                       optimisation signals (training return, approx KL,
                       explained variance, value loss, learning rate) vs steps
  regimes.png          every validation rollout placed by peak constraint ratio
                       vs landing error, coloured by flight regime, and the
                       regime sequence over training
  report.json          best/last evaluations, milestone steps, regime statistics,
                       stagnation trends, PPO health

Flight regimes (deterministic nominal rollout at each validation):
  modulated  AoA spread > 1 deg along the trajectory
  locked     AoA held (spread <= 1 deg; in practice at the 40 deg limit)
crossed with feasibility (peak ratio <= 1.02, the selection-score margin).

    python -m reentry_rl.postprocessing.training_report --run results/stage2_v21
"""
import argparse
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

# Reference palette (dataviz skill): categorical slots 1-2, text and surface tokens
C_MOD, C_LOCK = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ALPHA_MOD_DEG = 1.0
FEAS = 1.02
MILESTONES_KM = (100, 50, 20, 10, 5, 1)


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


def _rolling(v, window):
    return pd.Series(v).rolling(window, center=True, min_periods=max(2, window // 3)).median().to_numpy()


def load(run):
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    vh = pd.read_csv(run / "eval" / "validation_history.csv")
    vh["modulated"] = vh["alpha_std_mean"] > ALPHA_MOD_DEG
    vh["feasible"] = vh["max_ratio_max"] <= FEAS
    cur = pd.read_csv(run / "eval" / "curriculum.csv") if (run / "eval" / "curriculum.csv").exists() else None
    recs = []                       # train_log.txt + train_log_resume*.txt, in order
    for log in [run / "train_log.txt"] + sorted(run.glob("train_log_resume*.txt")):
        if log.exists():
            recs += parse_train_log(log)
    best = json.loads((run / "best" / "best.json").read_text()) if (run / "best" / "best.json").exists() else None
    return run, cfg, vh, cur, recs, best


def learning_curves(run, vh, cur, recs, path):
    s = vh["step"].to_numpy() / 1e6
    fig, axes = plt.subplots(5, 2, figsize=(13, 15), facecolor=SURFACE)
    ax = axes.ravel()

    def regime_points(a, y, logy=False):
        plot = a.semilogy if logy else a.plot
        plot(s, y, "-", color=GRID, lw=1.0, zorder=1)
        for mask, col, lab in ((vh["modulated"], C_MOD, "AoA modulated"), (~vh["modulated"], C_LOCK, "AoA locked")):
            for feas, mk in ((True, "o"), (False, "x")):
                m = mask & (vh["feasible"] == feas)
                lbl = f"{lab}, {'feasible' if feas else 'over limit'}"
                plot(s[m], y[m], mk, color=col, ms=4 if feas else 5, label=lbl, zorder=2)

    regime_points(ax[0], vh["nominal_d_km"].to_numpy(), logy=True)
    ax[0].axhline(5.0, color=INK2, lw=0.8, ls="--")
    ax[0].text(s[-1], 5.0, " 5 km (curriculum)", fontsize=7, color=INK2, va="bottom", ha="right")
    ax[0].set_title("nominal landing error [km]", fontsize=9, loc="left")
    ax[0].legend(fontsize=7, frameon=False, ncol=2, loc="upper right", labelcolor=INK)

    regime_points(ax[1], vh["max_ratio_max"].to_numpy())
    ax[1].axhline(1.0, color=INK2, lw=0.8, ls="--")
    ax[1].axhline(FEAS, color=INK2, lw=0.6, ls=":")
    ax[1].set_title("peak path-constraint ratio (dashed: limit, dotted: 2 % margin)", fontsize=9, loc="left")

    regime_points(ax[2], vh["nominal_dfpa_deg"].abs().to_numpy())
    ax[2].set_title("|FPA error| at handover [deg]", fontsize=9, loc="left")
    regime_points(ax[3], vh["nominal_dpsi_deg"].abs().to_numpy())
    ax[3].set_title("|heading error| at handover [deg]", fontsize=9, loc="left")
    regime_points(ax[4], vh["alpha_std_mean"].to_numpy())
    ax[4].axhline(ALPHA_MOD_DEG, color=INK2, lw=0.6, ls=":")
    ax[4].set_title("AoA spread along the rollout [deg] (dotted: regime threshold)", fontsize=9, loc="left")

    t, r = series(recs, "ep_rew_mean")
    if t.size:
        ax[5].plot(t / 1e6, r, color=C_MOD, lw=0.5, alpha=0.4)
        ax[5].plot(t / 1e6, _rolling(r, 25), color=C_MOD, lw=1.8, label="rolling median")
        ax[5].legend(fontsize=7, frameon=False, labelcolor=INK)
    ax[5].set_title("training return ep_rew_mean (stochastic policy)", fontsize=9, loc="left")

    for a, key, title, logy in ((ax[6], "approx_kl", "approx KL per update (log)", True),
                                (ax[7], "explained_variance", "explained variance of the value function", False),
                                (ax[8], "value_loss", "value loss (log)", True),
                                (ax[9], "learning_rate", "learning rate", False)):
        t, v = series(recs, key)
        if t.size:
            (a.semilogy if logy else a.plot)(t / 1e6, v, color=C_MOD, lw=0.8)
        a.set_title(title, fontsize=9, loc="left")
    tk = (series(recs, "target_kl")[1] if series(recs, "target_kl")[0].size else None)
    if cur is not None and (cur["advanced"] == True).any():          # noqa: E712
        for st in cur.loc[cur["advanced"] == True, "step"]:         # noqa: E712
            for a in ax:
                a.axvline(st / 1e6, color=INK2, lw=0.8, ls="-.", alpha=0.6)
    for a in ax:
        _style(a)
        a.set_xlabel("training steps [M]", fontsize=8)
    fig.suptitle(f"{run.name}: learning curves (dash-dot: curriculum level change)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return tk


def regimes_figure(run, vh, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
    for mask, col, lab in ((vh["modulated"], C_MOD, "AoA modulated"), (~vh["modulated"], C_LOCK, "AoA locked")):
        a.semilogy(vh.loc[mask, "max_ratio_max"], vh.loc[mask, "nominal_d_km"], "o", color=col, ms=5,
                   alpha=0.8, label=f"{lab} ({int(mask.sum())})", markeredgecolor=SURFACE, markeredgewidth=0.8)
    a.axvline(1.0, color=INK2, lw=0.8, ls="--")
    a.axvline(FEAS, color=INK2, lw=0.6, ls=":")
    a.set_xlabel("peak path-constraint ratio")
    a.set_ylabel("nominal landing error [km]")
    a.set_title("every validation rollout: constraint margin vs landing error", fontsize=9, loc="left")
    a.legend(fontsize=8, frameon=False, labelcolor=INK)
    s = vh["step"].to_numpy() / 1e6
    code = np.where(vh["modulated"], 1.0, 0.0) + np.where(vh["feasible"], 0.0, 0.5)
    b.plot(s, vh["modulated"].astype(float), "-", color=GRID, lw=1.0)
    b.scatter(s, vh["modulated"].astype(float), c=np.where(vh["modulated"], C_MOD, C_LOCK), s=14, zorder=2)
    b.scatter(s[~vh["feasible"]], vh.loc[~vh["feasible"], "modulated"].astype(float) + 0.08,
              marker="x", color=INK2, s=14, label="over the 2 % margin")
    b.set_yticks([0, 1])
    b.set_yticklabels(["AoA locked", "AoA modulated"])
    b.set_ylim(-0.4, 1.5)
    b.set_xlabel("training steps [M]")
    b.set_title("regime of the deterministic policy over training", fontsize=9, loc="left")
    b.legend(fontsize=8, frameon=False, labelcolor=INK, loc="upper right")
    for ax in (a, b):
        _style(ax)
    fig.suptitle(f"{run.name}: flight regimes", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    _ = code


def _row(vh, i):
    r = vh.loc[i]
    return {"step": int(r["step"]), "d_km": round(float(r["nominal_d_km"]), 3),
            "dfpa_deg": round(float(r["nominal_dfpa_deg"]), 3), "dpsi_deg": round(float(r["nominal_dpsi_deg"]), 3),
            "peak_ratio": round(float(r["max_ratio_max"]), 4), "alpha_std_deg": round(float(r["alpha_std_mean"]), 3),
            "score": round(float(r["score"]), 3)}


def _trend(t, v, window):
    k = t >= t[-1] - window
    return float(np.polyfit(t[k] / 1e6, v[k], 1)[0]) if k.sum() > 3 else float("nan")


def build_report(run, cfg, vh, cur, recs, best, window=2_000_000):
    rep = {"run": run.name, "timesteps_config": cfg.get("timesteps"),
           "steps_evaluated": int(vh["step"].iloc[-1]), "n_validations": int(len(vh)),
           "best_by_score": _row(vh, int(vh["score"].idxmin())), "last": _row(vh, len(vh) - 1)}
    f = vh[vh["feasible"]]
    rep["best_feasible_d"] = _row(vh, int(f["nominal_d_km"].idxmin())) if len(f) else None
    rep["milestones_feasible_first_step"] = {
        f"d<{m}km": (int(f.loc[f["nominal_d_km"] < m, "step"].iloc[0]) if (f["nominal_d_km"] < m).any() else None)
        for m in MILESTONES_KM}
    reg = {}
    for name, m in (("modulated_feasible", vh["modulated"] & vh["feasible"]),
                    ("modulated_over_limit", vh["modulated"] & ~vh["feasible"]),
                    ("locked_feasible", ~vh["modulated"] & vh["feasible"]),
                    ("locked_over_limit", ~vh["modulated"] & ~vh["feasible"])):
        sub = vh[m]
        reg[name] = {"fraction": round(float(m.mean()), 3),
                     "median_d_km": (round(float(sub["nominal_d_km"].median()), 1) if len(sub) else None),
                     "min_d_km": (round(float(sub["nominal_d_km"].min()), 1) if len(sub) else None),
                     "median_peak_ratio": (round(float(sub["max_ratio_max"].median()), 3) if len(sub) else None)}
    rep["regimes"] = reg
    switches = int((vh["modulated"].astype(int).diff().abs() > 0).sum())
    rep["regime_switches"] = switches
    rep["spearman"] = {
        "d_vs_peak_ratio": round(float(vh["nominal_d_km"].rank().corr(vh["max_ratio_max"].rank())), 3),
        "d_vs_alpha_std": round(float(vh["nominal_d_km"].rank().corr(vh["alpha_std_mean"].rank())), 3)}
    t = vh["step"].to_numpy(float)
    rep["trend_last_window"] = {"window_steps": window,
                                "log10_d_slope_per_M": round(_trend(t, np.log10(vh["nominal_d_km"].to_numpy()), window), 4),
                                "steps_since_best": int(vh["step"].iloc[-1] - rep["best_by_score"]["step"])}
    tr, r = series(recs, "ep_rew_mean")
    if tr.size:
        rep["trend_last_window"]["return_slope_per_M"] = round(_trend(tr, r, window), 2)
        rep["return"] = {"first": float(r[0]), "max": float(r.max()), "last_window_median": float(np.median(r[tr >= tr[-1] - window]))}
    health = {}
    for key in ("approx_kl", "explained_variance", "value_loss", "clip_fraction"):
        tk, v = series(recs, key)
        if tk.size:
            health[key] = {"median": float(np.median(v)), "p95": float(np.quantile(v, 0.95)),
                           "max": float(v.max()), "min": float(v.min())}
    tkl = cfg.get("ppo_effective", {}).get("target_kl")
    if tkl and "approx_kl" in health:
        _tk, v = series(recs, "approx_kl")
        health["updates_kl_over_1p5_target"] = int((v > 1.5 * tkl).sum())
    rep["ppo_health"] = health
    if cur is not None:
        rep["curriculum"] = {"final_level": int(cur["level"].iloc[-1]),
                             "advances": [int(x) for x in cur.loc[cur["advanced"] == True, "step"]]}  # noqa: E712
    return rep


def main():
    ap = argparse.ArgumentParser(description="Training report of one run")
    ap.add_argument("--run", required=True)
    ap.add_argument("--window", type=int, default=2_000_000, help="trend window [steps]")
    args = ap.parse_args()
    run, cfg, vh, cur, recs, best = load(args.run)
    out = run / "diag" / "report"
    out.mkdir(parents=True, exist_ok=True)
    learning_curves(run, vh, cur, recs, out / "learning_curves.png")
    regimes_figure(run, vh, out / "regimes.png")
    rep = build_report(run, cfg, vh, cur, recs, best, args.window)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
