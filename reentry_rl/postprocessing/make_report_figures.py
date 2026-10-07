"""Generate every figure for the RL-guidance report (report/figures/*.pdf).

No titles; axis and legend labels capitalized with units in [ ]; default
matplotlib colors. Produces:
  validation_residual.pdf              node-per-node state residuals vs SCvx
  {reward,hpo}_altvel.pdf              altitude-velocity + corridor boundaries
  {reward,hpo}_groundtrack.pdf         ground track, target star, miss text
  {reward,hpo}_aoa_bank.pdf            AoA and bank vs time (twin axes)
where 'reward' = best reward-shaping policy (v16), 'hpo' = best HPO policy.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from reentry_rl.physics import constants as C
from reentry_rl.physics.geodesy import great_circle_distance_m
from reentry_rl.postprocessing.fig_scvx_vs_rl import boundary_alt_km

_DEG = np.pi / 180.0
OUT = Path("F:/reentry_RL/report/figures")
REF = Path(r"F:/MSc_thesis/Code/reentry_simulator")

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm",
                     "axes.linewidth": 0.8, "savefig.bbox": "tight"})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  -> {name}.pdf")


# ---------------------------------------------------------------- validation
def fig_validation():
    import reentry_rl.validation.benchmark_scp as b
    b.REF_ROOT = REF
    folder = REF / "results" / "validation" / "guidance_run_wb001_min_heatload_bank_aoa_free_time"
    from reentry_rl.validation.validate_physics import replay
    res, df, _ = replay(folder)
    t = df["tAbs_s"].to_numpy()
    series = [("alt_km", "Altitude [km]"), ("lat_deg", "Latitude [deg]"),
              ("lon_deg", "Longitude [deg]"), ("V_mps", "Speed [m/s]")]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for key, lab in series:
        ax.semilogy(t, np.maximum(res["errs"][key], 1e-12), lw=1.2, label=lab)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Node-per-node residual error")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    _save(fig, "validation_residual")


# --------------------------------------------------------- trajectory panels
def _corridor(ax):
    Vg = np.linspace(350.0, 7460.0, 150)
    ax.plot(Vg, boundary_alt_km(Vg, "Qdot", C.QDOT_MAX), "--", color="0.35", lw=0.9,
            label=r"$\dot{Q} = \dot{Q}_{\max}$")
    ax.plot(Vg, boundary_alt_km(Vg, "q", C.QBAR_MAX), "-.", color="0.35", lw=0.9,
            label=r"$\bar{q} = \bar{q}_{\max}$")
    ax.plot(Vg, boundary_alt_km(Vg, "n", C.N_MAX), ":", color="0.35", lw=1.1,
            label=r"$n = n_{\max}$")


def fig_altvel(df, tag):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(df["V_mps"], df["alt_km"], "-", color="C0", lw=1.7, label="RL trajectory")
    _corridor(ax)
    ax.set_xlabel("Earth-relative speed [m/s]")
    ax.set_ylabel("Altitude [km]")
    ax.set_xlim(7600, 250)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, f"{tag}_altvel")


def fig_groundtrack(df, tag):
    lat_t, lon_t = C.TARGET["lat_deg"], C.TARGET["lon_deg"]
    lat_f, lon_f = float(df["lat_deg"].iloc[-1]), float(df["lon_deg"].iloc[-1])
    miss = great_circle_distance_m(lat_f * _DEG, lon_f * _DEG, lat_t * _DEG, lon_t * _DEG) / 1e3
    fig, ax = plt.subplots(figsize=(3.1, 4.7))
    ax.plot(df["lon_deg"], df["lat_deg"], "-", color="C0", lw=1.5, label="RL trajectory")
    ax.plot(df["lon_deg"].iloc[0], df["lat_deg"].iloc[0], "o", color="0.3", ms=6, label="Entry")
    ax.plot(lon_t, lat_t, "*", color="C3", ms=15, label="Target")
    ax.text(lon_t + 1.4, lat_t - 7.0, f"Miss = {miss:.2f} km", fontsize=9, ha="left")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_xlim(-4, 16)
    ax.set_ylim(-4, 76)
    ax.set_aspect(1.0 / np.cos(lat_t * _DEG))     # undistorted ground scale at the target
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    _save(fig, f"{tag}_groundtrack")


def fig_aoa_bank(df, tag):
    t = df["t_s"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    l1, = ax.plot(t, df["alpha_deg"], color="C0", lw=1.5, label="Angle of attack [deg]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle of attack [deg]", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax2 = ax.twinx()
    l2, = ax2.plot(t, df["sigma_deg"], color="C1", lw=1.5, label="Bank angle [deg]")
    ax2.set_ylabel("Bank angle [deg]", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="best", fontsize=9)
    _save(fig, f"{tag}_aoa_bank")


def panels(csv, tag):
    df = pd.read_csv(csv)
    fig_altvel(df, tag)
    fig_aoa_bank(df, tag)


# ------------------------------------------------------------ learning curves
def fig_reward_learning(run):
    ev = np.load(f"{run}/evaluations.npz")
    steps = ev["timesteps"] / 1e6
    rew = ev["results"].mean(axis=1)
    # terminal error history parsed from the captured training log
    from reentry_rl.postprocessing.diagnose_run import parse_train_log, series
    recs = parse_train_log(f"{run}/train_log.txt")
    t_d, d = series(recs, "d_km")
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    l1, = ax.plot(steps, rew, color="C0", lw=1.2, label="Evaluation return")
    ax.set_xlabel("Training steps [millions]")
    ax.set_ylabel("Evaluation return", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax2 = ax.twinx()
    ok = np.isfinite(d)
    l2, = ax2.semilogy(t_d[ok] / 1e6, d[ok], color="C1", lw=1.0, label="Terminal position error [km]")
    ax2.set_ylabel("Terminal position error [km]", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper center", fontsize=9)
    _save(fig, "reward_learning")


SCVX_MAT = REF / "report" / "closed_loop_g&c_paper" / "figures" / "gen" / "_simcache_minq.mat"


def _load_scvx():
    import scipy.io as sio
    C = sio.loadmat(str(SCVX_MAT), squeeze_me=True, struct_as_record=False)["C"]
    idx = np.unique(np.linspace(0, len(C.V_mps) - 1, 4000).astype(int))
    return C, idx


def fig_compare_altvel(rl_csv):
    C, idx = _load_scvx()
    rl = pd.read_csv(rl_csv)
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    ax.plot(np.asarray(C.V_mps)[idx], np.asarray(C.h_km)[idx], "-", color="C1", lw=1.7,
            label="SCvx G\\&C (closed loop)")
    ax.plot(rl["V_mps"], rl["alt_km"], "-", color="C0", lw=1.7, label="RL guidance (PPO)")
    _corridor(ax)
    ax.set_xlabel("Earth-relative speed [m/s]")
    ax.set_ylabel("Altitude [km]")
    ax.set_xlim(7600, 250)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "compare_altvel")


def fig_compare_controls(rl_csv):
    C, idx = _load_scvx()
    rl = pd.read_csv(rl_csv)
    t_s = np.asarray(C.t)[idx]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    la1, = ax.plot(rl["t_s"], rl["alpha_deg"], "-", color="C0", lw=1.6,
                   label="Angle of attack, RL [deg]")
    la2, = ax.plot(t_s, np.asarray(C.alpha_deg)[idx], "--", color="C0", lw=1.4,
                   label="Angle of attack, SCvx [deg]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle of attack [deg]", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax2 = ax.twinx()
    lb1, = ax2.plot(rl["t_s"], rl["sigma_deg"], "-", color="C1", lw=1.6,
                    label="Bank angle, RL [deg]")
    lb2, = ax2.plot(t_s, np.asarray(C.sigma_deg)[idx], "--", color="C1", lw=1.4,
                    label="Bank angle, SCvx [deg]")
    ax2.set_ylabel("Bank angle [deg]", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax.legend(handles=[la1, la2, lb1, lb2], loc="upper center", fontsize=8.5, ncol=2)
    _save(fig, "compare_controls")


def fig_hpo_history():
    import optuna
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = optuna.load_study(study_name="stage2_polish_hpo",
                          storage="sqlite:///F:/reentry_RL/results/stage2_polish_hpo/study.db")
    comp = sorted((t.number, t.value) for t in s.trials
                  if t.state == TrialState.COMPLETE and t.value is not None)
    n = [c[0] for c in comp]
    v = np.array([c[1] for c in comp])
    best = np.minimum.accumulate(v)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.scatter(n, v, s=28, color="C0", label="Completed trial")
    ax.plot(n, best, color="C1", lw=1.5, label="Best so far")
    ax.axhline(1.0, ls="--", color="0.4", lw=0.9, label="Success disk [1 km]")
    ax.set_yscale("log")
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Terminal position error [km]")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "hpo_history")


def main():
    print("validation:")
    fig_validation()
    print("best reward-shaping policy (v16):")
    panels("F:/reentry_RL/results/stage2_v16/diag/best/eval_trajectory.csv", "reward")
    fig_reward_learning("F:/reentry_RL/results/stage2_v16")
    print("best HPO policy (trial 23):")
    panels("F:/reentry_RL/results/stage2_polish_hpo/diag_best/eval_trajectory.csv", "hpo")
    print("HPO study:")
    fig_hpo_history()


if __name__ == "__main__":
    main()
