# Re-entry Guidance & Control — Reinforcement Learning Architecture
## Design Document

| | |
|---|---|
| **Project** | RL guidance & control + non-linear simulator (`E:\reentry_RL`) |
| **Validation target** | MATLAB SCP framework `E:\Code\reentry_simulator`, vehicle **wb001**, **heat-load minimization only** |
| **Status** | Rev 4 — physics gate PASSED; Stage-1 **full training pipeline built & verified** (`train_sb3.py` + `common.py`: SubprocVecEnv, TensorBoard, EvalCallback + checkpoints, domain-metrics logging). `w_progress=20` enabled, substeps=3. **Ready to launch the multi-million-step Stage-1 run**, then Optuna HPO. |
| **Last updated** | 2026-06-17 (rev 4) |

> **How to use this document:** §1–§10 = architecture (updated with your answers). §11.1/§11.2 = decision logs (Rounds 1 & 2). §11.3 = candidate additional research questions you asked me to brainstorm — please select which are in scope. §12 = assumptions/risks/defaults.

---

## Table of Contents
0. [Purpose & status](#0-purpose)
1. [Scope & objective](#1-scope)
2. [Reference framework grounding](#2-reference)
3. [Locked design decisions](#3-decisions)
4. [Non-linear simulator (physics core)](#4-simulator)
5. [RL environment (Gymnasium)](#5-environment)
6. [Cascade stages](#6-cascade)
7. [Reward design vs. hyperparameter optimization (workflow)](#7-hpo)
8. [Validation vs the SCP reference](#8-validation)
9. [Repository / package layout](#9-repo)
10. [Build order & milestones](#10-build)
11. [Questions — §11.1 R1 · §11.2 R2 · §11.3 research questions](#11-questions)
12. [Assumptions, risks & defaults](#12-assumptions)
13. [Dependencies](#13-deps)

---

<a name="0-purpose"></a>
## 0. Purpose & status

Devise and implement a reinforcement-learning re-entry guidance & control architecture, together with the non-linear simulator in which it is trained and tested. The architecture is built up in three cascaded stages of increasing action-space dimensionality, validated against the existing MATLAB successive-convexification (SCP) framework for the **wb001** mission, restricted to the **heat-load minimization** objective.

The reward-transfer idea (*a reward that works for bank-only should also work once AoA is added*) is **one of several hypotheses to study — not the driving hypothesis** (per your note). The driving research question is **Q-GOAL(a): can an RL feedback policy match or beat SCP-optimal heat load for WB001** (§11.2); (b)/(c)/(d) and the candidates in §11.3 are additional. The architecture is still engineered so the reward is action-dimension-agnostic, which lets us *measure* the transfer cleanly.

---

<a name="1-scope"></a>
## 1. Scope & objective

- **Vehicle:** WB001 (NASA MSFC winged-body SSTO RLV).
- **Mission:** de-orbit re-entry, 100 km → 25 km.
- **Objective (only):** minimize integrated heat load `J = ∫ Q̇ dt`, `Q̇ = kQ·√ρ·Vⁿ` (Sutton–Graves).
- **Path constraints:** `Q̇ ≤ 1.5 MW/m²`, `n ≤ 2.5 g`, `q̄ ≤ 18 kPa`.
- **Free final time** for stages 1–2 (matches the reference convex-guidance assumption that attitude is tracked instantaneously).
- **Out of scope (for now):** other objectives (min-final-time, max-final-velocity), other vehicles (Apollo), Monte-Carlo robustness campaigns beyond the baseline comparison (added later).

---

<a name="2-reference"></a>
## 2. Reference framework grounding

All numbers below are extracted from `E:\Code\reentry_simulator` and reused **verbatim**.

### 2.1 WB001 vehicle (MCI / aero / actuators)

| Quantity | Value |
|---|---|
| Mass | 104 305 kg |
| Reference area `Sref` | 391.22 m² |
| Reference length / span `Lref` = `bref` | 29.2 m |
| `Ixx` / `Iyy` / `Izz` | 19 979 222 / 276 629 967 / 28 383 800 kg·m² |
| Products of inertia | 0 (diagonal J) |
| Aero model | Polynomial `CL(α,Mach)`, `CD(α,Mach)` (`wb001_polynomial_aero`) |
| Heat-rate model | `Q̇ = kQ·√ρ·V^n`, `kQ = 1.65e-4`, `n = 3.15` |
| Nominal AoA schedule | legacy tanh-like: 40° for V > 4570 m/s, tapering to 5° at low speed |
| Roll / pitch / yaw surfaces | δ_max = 18° / 20° / 25°, τ = 50 ms |
| RCS (each axis) | τ_max = 1.25 MN·m, Isp = 220 s, L_eff = 8 m, τ_lag = 30 ms |
| Attitude derivatives | Cm_α=−0.20, Cm_q=−3.5, Cm_δe=−0.50, Cn_β=+0.15, Cn_r=−0.30, Cn_δr=−0.07, Cl_β=−0.05, Cl_p=−0.35, Cl_δa=+0.05 (/rad) |

### 2.2 WB001 mission (IC / target / bounds)

| | Altitude | Longitude | Latitude | Velocity | FPA γ | Heading ψ |
|---|---|---|---|---|---|---|
| **Initial** | 100 km | 0° | 0° | 7450 m/s | −0.5° | 0° (due North) |
| **Target** | 25 km | 12°E | 70°N | *free* | −10° | 90° (due East) |

- SCP fixes 5 terminal states (alt, lon, lat, FPA, heading) as hard equalities; **V and bank free**. Both benchmark runs hit them essentially exactly (§2.7) — terminal miss ≈ metres, so the RL-vs-SCP terminal comparison is against a near-bullseye.
- State bounds: altitude 5–120 km, V 50–8500 m/s, γ ±89°, ψ ±360°, **bank σ ∈ [−85°,+85°] (signed)**, α ∈ [5°,40°].

### 2.3 Equations of motion (3-DOF translational, non-dimensional)

```
ṙ    = s · V·sinγ
lȯn  = s · V·cosγ·sinψ / (r·cos lat)
laṫ  = s · V·cosγ·cosψ / r
V̇    = s · ( −D + g_r·sinγ + g_δ·sinψ·cosγ + Ω²·r·cos lat·(sinγ·cos lat − cosγ·sin lat·cosψ) )
γ̇    = s · ( L·cosσ/V + V·cosγ/r + g_r·cosγ/V − g_δ·sinψ·sinγ/V
              + 2Ω·cos lat·sinψ + Ω²·r·cos lat·(cosγ·cos lat + sinγ·sin lat·cosψ)/V )
ψ̇    = s · ( L·sinσ/(V·cosγ) + V·cosγ·sinψ·tan(lat)/r − g_δ·sinψ/(V·cos lat)
              − 2Ω·(tanγ·cosψ·cos lat − sin lat) + Ω²·r·sin lat·cos lat·sinψ/(V·cosγ) )
σ̇    = u₁                          (bank-rate control)
α̇    = u₂                          (AoA-rate control, stage 2+ only)
```
`s = rateScale`; `Ω` Earth rotation; `g_r`/`g_δ` from rotating-J2 gravity; `L,D = q̄·Sref·{CL,CD}`, `q̄ = ½ρV²`.

### 2.4 Environment models to port (faithfully)

| Model | Reference file | Notes |
|---|---|---|
| Atmosphere | `atmosphere_properties.m` | US-1976, piecewise to 120 km geometric, analytic ρ, dρ/dh, T, a, da/dh |
| Gravity | `entry_gravity_j2.m` | rotating oblate-spheroid J2; `μ̄ = μ/(g0·R0²)`, `ReBar = J2RefRadius/R0` |
| Aero | `wb001_polynomial_aero.m` | polynomial CL/CD vs (α, Mach) |
| Heat rate | path-constraint #1 | Sutton–Graves `kQ·√ρ·V^3.15` |
| Non-dim constants | `cfg.earth.{R0, VScale, rateScale, Omega, mu, g0, J2ReferenceRadius}` | reused verbatim |

### 2.5 Heat-load objective (exact form)
```
J_heat = ∫₀^tf Q̇ dt    (Q̇ = kQ·√ρ·V^3.15  [W/m²])
```
Primary RL objective and primary comparison metric.

### 2.6 The three reference guidance modes

| Mode | stateDim | controlDim | optimizeAoA | freeFinalTime |
|---|---|---|---|---|
| `bank_free_time` | 7 | 1 (σ̇) | no | yes |
| `bank_aoa_free_time` | 8 | 2 (σ̇, α̇) | yes | yes |
| full 6-DOF (propagator) | 7/8 + 16 | LQR/MPC replaced | — | — |

### 2.7 Benchmark data available (extracted) — Q-BENCH1 RESOLVED

Benchmarks come from exported MATLAB results (no MATLAB re-run, no MATLAB Engine). Format (per folder): `guidance_executed_trajectory.txt` (CSV), `guidance_terminal_summary.txt`, `guidance_call_history.txt`, `guidance_profiles.{png,svg}`, `guidance_solution.mat`, `console_log.txt`.

**Both benchmarks are now in hand:**

| Stage | Folder | Mode | CSV cols | J_heat [J/m²] | t_f [s] | V_f [m/s] | terminal hit |
|---|---|---|---|---|---|---|---|
| **1** | `results/guidance_unit_test/guidance_run_wb001_20260617_191855` | `bank_free_time` | 9 (no α/uα) | **1.0832e9** | 1636.65 | 767 | 12.00002°, 70.0003°, γ −9.84°, ψ 89.76° |
| **2** | `results/validation/guidance_run_wb001_min_heatload_bank_aoa_free_time` | `bank_aoa_free_time` | 11 (incl. α, uα) | **1.072e9** | 1547.76 | 426 | 12.00008°, 70.00002°, γ −10.02°, ψ 89.94° |

- **Mode-matched `J_ref`** — Stage 1 normalizes/compares against 1.0832e9; Stage 2 against 1.072e9. Adding AoA buys ≈1.0 % heat-load and ≈89 s.
- **CSV columns:** bank+AoA = `tAbs_s, alt_km, lon_deg, lat_deg, V_mps, gamma_deg, psi_deg, sigma_deg, alpha_deg, uSigma_degps, uAlpha_degps`; bank-only drops `alpha_deg`/`uAlpha_degps` (α from schedule). The benchmark loader handles **both layouts** and reconstructs α from the schedule for the bank-only case. These CSVs (states + rate commands) drive the §4.5 validation gate.
- Episode horizon ≈ 1550–1640 steps at Δt = 1 s.
- *(Ignore the un-suffixed `results/validation/guidance_run_wb001_min_heatload` folder — it is a byte-identical duplicate of the bank+AoA run.)*

---

<a name="3-decisions"></a>
## 3. Locked design decisions

1. **Terminal BCs → footprint + windows.** Episode ends at 25 km altitude; terminal accuracy scored by a single continuous landing-error reward + FPA/heading bands (soft, §5.3). V and bank free.
2. **Action = attitude rates**, integrated in-env. σ̇ ∈ [−45,45]°/s, α̇ ∈ [−15,15]°/s, normalized to [−1,1].
3. **Stage 3 = hierarchical cascade.** Freeze trained Stage-2 guidance policy; train an inner attitude+allocation policy (6 actuator actions). Reward includes a propellant penalty (prefer free aero surfaces over RCS).
4. **Compute = NumPy first.** Doubles as the MATLAB-validation reference; SB3 `SubprocVecEnv`; JAX/GPU only if throughput-limited.

---

<a name="4-simulator"></a>
## 4. Non-linear simulator (physics core)

### 4.1 Modules (`reentry_rl/physics/`)
`constants.py` · `atmosphere.py` (US-76) · `gravity_j2.py` · `aero_wb001.py` (CL/CD + Q̇) · `eom_3dof.py` · `dynamics_6dof.py` (Euler kinematics, `ω̇ = J⁻¹(M_body − ω×Jω)`, six first-order actuators, propellant flow).

### 4.2 Two fidelities behind one core
- **Stages 1–2 (3-DOF):** translational only; α from command/schedule, σ from command (instantaneous attitude).
- **Stage 3 (6-DOF):** reference `simulator_state_rhs` minus the LQR/MPC — the policy replaces the controller.

### 4.3 Non-dimensionalization
Internal state non-dimensional as in the reference (`r/R0`, `V/VScale`, time via `rateScale`); obs/reward re-dimensionalized for the agent.

### 4.4 Integrator & decision timestep  ✓ *(Q-INT confirmed)*
Fixed-step **RK4**; guidance decision interval **Δt = 1.0 s** (sub-stepped); Stage-3 inner attitude step **~10–20 ms**.

### 4.5 Validation gate (`physics_validation` notebook)
Replay each benchmark CSV's commands (`uSigma`/`uAlpha`, or σ/α directly; α from schedule for the bank-only case) through the Python core and require node-for-node agreement of states + `J_heat` vs `guidance_executed_trajectory.txt` to a tight tolerance, for **both** benchmarks. **Blocks all downstream work until it passes.** No MATLAB Engine — pure CSV fixture comparison.

---

<a name="5-environment"></a>
## 5. RL environment (Gymnasium)

One base class, specialized per stage; single source of truth for obs/reward/termination.

### 5.1 Observation (stages 1–2)  ✓ *(Q-PHASE: velocity is the phase variable)*
```
[ h, V (PRIMARY phase variable, normalized), γ, ψ, σ, (α in stage 2),
  q̄/q̄max, Mach, Q̇/Q̇max, n/nmax,
  great-circle range-to-go to target, heading-error to target ]
```
Stage 3 appends `[φ, θ, ψ_e, p, q, r, β, δ_a, δ_e, δ_r, τ_roll, τ_pitch, τ_yaw, m_prop]`.
*(Integration stays in time; velocity is the observed progress/phase indicator, not a change of independent variable.)*
*(If RQ-GEN in §11.3 is in scope, the target (lat/lon/γ*/ψ*) is appended to the observation and randomized in training.)*

### 5.2 Action
| Stage | Action (normalized [−1,1]) | Dim |
|---|---|---|
| 1 | σ̇ | 1 |
| 2 | σ̇, α̇ | 2 |
| 3 (inner) | u_RCS,roll, u_RCS,pitch, u_RCS,yaw, u_δa, u_δe, u_δr | 6 |

Rates integrated in-env; angles clamped (σ ∈ [−85,85]°, α ∈ [5,40]°).

### 5.3 Reward  ✓ *(Q-TOL/Q-TOL2: ONE continuous terminal-error reward; FPA ±5°; heading ±15°; no V constraint)*
```
per-step:
  r_t = − w_Q   · (Q̇·Δt) / J_ref                      ← PRIMARY: heat-load increment
        − Σ_c w_path,c · barrier( margin to Q̇,n,q̄ limits )
        − w_act · ‖a_t‖²                                ← small smoothness/effort
        [stage 3 only: − w_prop · Δm_prop / m_ref ]     ← propellant penalty (RCS)

terminal (on crossing 25 km):  d = great-circle landing error [km]
  R_pos(d) =  + w_succ                          if d ≤ 1      ← success plateau, NO penalty
              − w_q·(d − 1)²                     if 1 < d ≤ 10 ← quadratic
              − w_q·81·exp( β·(d − 10) )         if d > 10     ← exponential, β = 18/81 (C1 at d=10)
              (success bonus smoothed across ~1 km so the total reward is continuous)
  r_T = R_pos(d) − w_fpa·|γ − γ*|/Δγ_tol − w_psi·|ψ − ψ*|/Δψ_tol
        Δγ_tol = 5°,  Δψ_tol = 15°,  no terminal velocity/Mach constraint

failure (skip-out, crash, OOB, NaN, t > t_max): r = − w_fail, episode ends
```
- **Single continuous terminal-error reward (Q-TOL2):** *one* policy, not two disk variants. No penalty below 1 km **plus a strong success bonus** (defines "success"); quadratic 1→10 km; exponential beyond 10 km; continuous (C1) across the 1 km and 10 km breakpoints. FPA/heading windows identical regardless of landing error. Pure **shaping** — outside the disk is penalized, never failed.
- `J_ref` is mode-matched (§2.7): 1.0832e9 (Stage 1), 1.072e9 (Stage 2).
- Only `w_act` + the stage-3 propellant term scale with action dimension → reward structure identical across stages (lets us measure transfer).
- Path constraints = soft smooth barriers (not hard kills), preserving gradient/credit; hard kill only on gross violation.

### 5.4 Termination
- **Success manifold:** altitude ≤ 25 km → terminal reward.
- **Failure:** altitude > 120 km · altitude < 5 km off-nominal · |γ|/V out of bounds · NaN · step budget exceeded.

### 5.5 Normalization
Observation via `VecNormalize` and/or fixed physical scales; reward scaled by `J_ref`.

---

<a name="6-cascade"></a>
## 6. Cascade stages

| Stage | Policy | Sim | Reference analogue | New element |
|---|---|---|---|---|
| **1** | bank-rate (1-D); α on legacy schedule | 3-DOF | `bank_free_time` | reward + HPO recipe |
| **2** | bank+AoA rate (2-D) | 3-DOF | `bank_aoa_free_time` | **reuse Stage-1 reward** → measure transfer |
| **3** | inner attitude+allocation (6-D) tracking **frozen Stage-2** | 6-DOF | guidance + attitude controller | hierarchical cascade + propellant reward |

✓ Stage-1 AoA = original WB001 schedule (Q-AOA-STAGE1). **Stage 3 detail:** Stage-2 policy supplies σ_cmd/α_cmd at the slow Δt; the inner policy runs at the fast inner step, outputs 6 actuator actions, tracks σ_cmd/α_cmd and regulates β→0; allocation learned under the propellant penalty. ✓ Two-timescale cadence confirmed (Q-CADENCE).

---

<a name="7-hpo"></a>
## 7. Reward design vs. hyperparameter optimization (workflow)

*(Answers Q-WEIGHTS — "first HPO then reward, or both at once?")*

**They are different things and are best done sequentially, reward first:**

1. **Reward design first (structure + coarse weights).** The reward defines *what* the agent optimizes. Fix the structure from domain knowledge and pick rough weights, using **default/reasonable hyperparameters**, until the agent (a) reliably solves the task (reaches the disk, respects constraints) and (b) the shaped reward is *aligned* with the true objective — i.e. higher reward ⇒ lower heat load while feasible (no reward-hacking). Mostly manual iteration + small targeted weight sweeps, judged by **task success**, not the final metric.

2. **HPO second, with the reward frozen.** Run Optuna over RL hyperparameters (LR, `n_steps`, batch, `n_epochs`, γ, GAE-λ, clip, entropy/vf coef, net arch). **Critically, the HPO objective is the true task metric — feasibility-gated `J_heat` — not the cumulative shaped reward.** That way HPO optimizes the real goal and cannot be fooled by shaping.

3. **Optional light refinement.** Once a competent policy exists, revisit a few penalty weights and re-run a shortened HPO. Iterate sparingly.

**Why not jointly, all at once?** (i) The search space explodes (reward weights × hyperparameters). (ii) Changing reward weights *moves the objective surface* — you'd optimize a moving target and risk reward-hacking. (iii) Confounded attribution: if both move, you can't tell what helped. (Joint/population-based reward+HPO search exists but is expensive and unnecessary here.)

**→ Default workflow: reward-first → HPO (metric = feasibility-gated `J_heat`) → optional refine.** Tooling: TPE sampler + MedianPruner; seeded envs; config-as-code (`config/wb001.yaml`); logged studies.

---

<a name="8-validation"></a>
## 8. Validation vs the SCP reference

Same IC, vehicle, constraints. Benchmark = exported MATLAB results (§2.7); mode-matched `J_ref` (Stage 1: 1.0832e9; Stage 2: 1.072e9).

| Metric | RL | SCP |
|---|---|---|
| **Heat load `J_heat` (primary)** | closed-loop policy rollout | receding-horizon SCP (benchmark) |
| Terminal miss (lon/lat/γ/ψ) | vs continuous-error reward | near-zero (hard fixes) |
| Peak path-constraint margins (Q̇, n, q̄) | ✓ | ✓ |
| Propellant (stage 3) | ✓ | ✓ |
| Control effort / smoothness | ✓ | ✓ |
| Inference wall-clock | per-step policy eval | per-call SCP solve |

**Post-processing, artifacts & comparison module** *(Q-STRUCTURE / Q-PLOTS)* — `reentry_rl/postprocessing/`:
- `plots.py` — reproduce the MATLAB `guidance_profiles` figure style (altitude/velocity/FPA/heading/bank/AoA + rate commands + path constraints vs time, plus ground-track / 3-D). Stage-3 adds attitude/body-rate/actuator/command-tracking panels.
- `report.py` — write a `guidance_terminal_summary`-equivalent for the RL run, **and save a per-run CSV time-series of states, controls, body rates, and path-constraint values vs time** (mirrors `guidance_executed_trajectory.txt`).
- `compare_vs_matlab.py` — RL↔SCP overlays + delta table (Δ`J_heat` %, terminal miss, peak constraint margins).
- **Saved artifacts (Q-PLOTS):** the **trained controller** (`model.zip` + `VecNormalize` stats) and the per-run **CSV** above, under `results/<stage>_<timestamp>/`.

Optional cross-check: replay the RL command history to confirm Python-sim vs MATLAB-sim agreement on the *RL-produced* trajectory.

---

<a name="9-repo"></a>
## 9. Repository / package layout  ✓ *(Q-STRUCTURE: package + notebooks; no numeric prefixes)*

```
reentry_rl/
  physics/         constants.py · atmosphere.py · gravity_j2.py · aero_wb001.py · eom_3dof.py · dynamics_6dof.py
  envs/            entry_env_base.py · stage1_bank.py · stage2_bank_aoa.py · stage3_full.py · rewards.py · normalizers.py
  validation/      crosscheck_matlab.py · benchmark_scp.py        (load/parse the exported MATLAB results; both CSV layouts)
  training/        hpo_optuna.py · train_sb3.py · callbacks.py
  postprocessing/  plots.py · report.py · compare_vs_matlab.py    (MATLAB-style plots + RL↔SCP comparison + CSV/model save)
  config/          wb001.yaml
notebooks/
  physics_validation.ipynb
  stage1_bank.ipynb
  stage2_bank_aoa.ipynb
  stage3_full.ipynb
  compare_vs_scp.ipynb
results/           <stage>_<timestamp>/  (model.zip, vecnormalize.pkl, trajectory CSV, plots, summary)
tests/             unit tests for physics modules vs MATLAB fixtures
DESIGN.md
```

---

<a name="10-build"></a>
## 10. Build order & milestones

| # | Milestone | Exit criterion |
|---|---|---|
| 1 ✓ | Physics port + MATLAB cross-validation (`physics_validation`) | **PASSED** — both benchmarks reproduced node-for-node: J_heat Δ < 0.001%, alt ≤ 20 m, V ≤ 0.9 m/s |
| 2a ✓ | Stage 1 — env + reward + smoke-train (`reentry_rl/envs`, `training/smoke_train_stage1.py`) | **DONE** — env self-test + PPO pipeline verified (save / eval / CSV) |
| 2b ✓ | Stage 1 — training pipeline (`train_sb3.py`, `common.py`) + reward iteration (`w_progress` on, substeps=3) | **DONE & verified** — subproc/TB/eval/ckpt/metrics pipeline clean |
| 2c | Stage 1 — launch multi-M-step run → Optuna HPO | policy completes entry within disk+windows; `J_heat` vs 1.0832e9; saves model + CSV |
| 3 | Stage 2 (bank+AoA) — reward reused | transfer measured; `J_heat` vs 1.072e9 |
| 4 | Stage 3 (hierarchical 6-DOF) | inner policy tracks guidance; propellant bounded; constraints held |
| 5 | Post-processing + compare vs SCP (all stages) | MATLAB-style plots + delta table + results dump |

---

<a name="11-questions"></a>
## 11. Questions

<a name="111"></a>
### 11.1 Decision log — Round 1 (RESOLVED)

| ID | Decision |
|---|---|
| **Q-MATLAB** | Exported MATLAB results; no re-run. Both benchmarks located (§2.7). |
| **Q-ENGINE** | No MATLAB Engine; validate against exported trajectory CSVs. |
| **Q-TOL / Q-TOL2** | *One* continuous terminal-error reward: no-penalty < 1 km + success bonus, quadratic 1–10 km, exponential > 10 km; FPA ±5°, heading ±15°; no velocity constraint (§5.3). |
| **Q-INT** | RK4, Δt = 1.0 s, Stage-3 inner step 10–20 ms. ✓ |
| **Q-PHASE** | **Velocity** = primary phase variable; include range-to-go + heading-error. ✓ |
| **Q-REACH** | Mission is not tight; footprint achievable. ✓ |
| **Q-CADENCE** | Two-timescale split confirmed. ✓ |
| **Q-WEIGHTS** | §7: reward-first → HPO (metric = feasibility-gated `J_heat`) → optional refine. |
| **Q-ALGO** | PPO throughout; SAC for Stage 3 if needed. ✓ |
| **Q-STRUCTURE** | Package + notebooks; drop numeric prefixes; post-processing/comparison module (§8, §9). ✓ |
| **Q-DISP** | Nominal IC first; dispersions later. ✓ |
| **Q-NAV** | Perfect navigation (full-state obs). ✓ |
| **Q-AOA-STAGE1** | Stage-1 α follows the original WB001 schedule. ✓ |

<a name="112"></a>
### 11.2 Decision log — Round 2 (RESOLVED)

| ID | Your answer → resolution |
|---|---|
| **Q-GOAL** | Primary = **(a)** RL match/beat SCP heat load. Additional = (b) one policy replacing the guidance+control stack, (c) onboard/real-time cost, (d) robustness under dispersions. You asked for more candidates → **§11.3**. |
| **Q-BENCH1** | "It exists now." Located: bank-only run `guidance_unit_test/guidance_run_wb001_20260617_191855` (J = 1.0832e9). Both benchmarks in §2.7. |
| **Q-TOL2** | *One* policy (not two), continuous terminal-error reward (no-penalty<1 km + success bonus / quadratic 1–10 km / exponential >10 km); FPA & heading windows identical; shaping (never a failure). Implemented in §5.3. |
| **Q-PLOTS** | Default plot set + **save the trained controller** + **save a per-run CSV** (states, controls, rates, constraints vs time). In §8. |

<a name="113"></a>
### 11.3 Additional research questions — SCOPE LOCKED

Legend: ✦ changes env/training design (decide **before Stage 1**) · ⊕ extra experiments only · ◦ falls out of planned work (≈free).

| ID | Research question |
|---|---|
| **RQ-OPT** ◦ | *Optimality gap & cause.* How close can a causal feedback policy get to the open-loop SCP optimum, and is the residual gap due to causality (no BVP), reaction time, or reward shaping? |
| **RQ-GEN** ✦ | *Generalization / target-conditioning.* Can **one** policy, conditioned on the target (and/or IC), fly a *family* of entries without retraining — the key practical edge over per-case convex re-solves? (Adds the target to the obs + randomizes it in training.) |
| **RQ-SAFE** ⊕ | *Hard-constraint satisfaction under uncertainty.* Does the soft-barrier reward keep Q̇/n/q̄ within limits across dispersions, and how does the violation rate compare to the SCP's hard constraints? |
| **RQ-SENS** ⊕ | *Reward & seed sensitivity.* How sensitive are heat load / terminal accuracy / constraint adherence to reward weights and random seed? (credibility of the RL-vs-SCP claim) |
| **RQ-FREQ** ⊕ | *Decision bandwidth.* How does guidance Δt (and the Stage-3 inner rate) trade optimality vs smoothness vs onboard cost? (ties into (c)) |
| **RQ-HIER** ⊕ | *Hierarchical vs end-to-end (Stage 3).* Does frozen-guidance hierarchy leave performance on the table vs a jointly trained / two-timescale single policy? (validates §3.3) |
| **RQ-CURR** ◦ | *Cascade warm-start transfer.* Does initializing Stage 2 from Stage 1 (and Stage 3 from Stage 2) speed training / improve performance vs cold starts? (the cascade's transfer hypothesis, complementary to reward-transfer) |
| **RQ-STRUCT** ◦ | *Recovered guidance structure.* Does the learned bank/AoA profile reproduce known entry structure (bank reversals, drag/energy tracking)? (trust & qualitative SCP comparison) |
| **RQ-PARETO** ⊕ | *Multi-objective trade (Stage 3).* Heat load vs propellant vs terminal accuracy Pareto front, and where the SCP+LQR reference sits on it. |

**Scope decision (LOCKED):**
- **In scope now:** **RQ-OPT, RQ-CURR, RQ-STRUCT** (all ◦ — answered by the planned stage builds + comparison, *no separate experiment campaigns*) + **(b)** Stage-3 single-policy thesis (inherent to the cascade) + **(c)** onboard/real-time cost (reported as a cheap metric in §8).
- **Deferred (no extra experiments for now):** **(d)** robustness/dispersions, and all ⊕ items (RQ-SAFE, RQ-SENS, RQ-FREQ, RQ-HIER, RQ-PARETO).
- **RQ-GEN ✦ deferred** → the env stays **single-target** (fixed WB001 target; no target-conditioning). Revisit later if desired.

Framework implications baked in now: (1) policy networks use a **transferable trunk** so Stage 2 can warm-start from Stage 1 and Stage 3 from Stage 2 (RQ-CURR); (2) the comparison module explicitly **reports the optimality gap** to SCP (RQ-OPT) and **overlays RL-vs-SCP bank/AoA profiles** to assess recovered structure (RQ-STRUCT).

---

<a name="12-assumptions"></a>
## 12. Assumptions, risks & defaults

**Assumptions**
- WB001 constants, EOM, atmosphere, gravity, aero, heat models reproduced exactly; non-dim constants verbatim.
- Coordinated flight (β→0) in stages 1–2; β regulated by the inner policy in stage 3.
- Bank signed (±85°), carries cross-range; policy learns reversals (no separate reversal logic).
- Free final time (stages 1–2); episode length emergent, terminated by the altitude manifold (~1550–1640 steps at the benchmarks).

**Risks**
1. **Physics fidelity mismatch** (highest) → §4.5 validation gate (both benchmarks) blocks downstream work.
2. **Causal feedback vs BVP terminal accuracy** → continuous terminal-error shaping; sub-1 km is aspirational for a causal policy (SCP hits ~0), measured not required.
3. **Reward transfer may not hold** when AoA added → reward is action-dimension-agnostic; Stage 2 measures it.
4. **Stage-3 two-timescale credit assignment / actuator redundancy** → hierarchical cascade + propellant penalty.
5. **Sample cost of 6-DOF training** → NumPy-first vectorized; escalate to JAX/GPU if needed.
6. ~~Missing Stage-1 benchmark~~ → **resolved**; both benchmarks in hand (§2.7).

---

<a name="13-deps"></a>
## 13. Dependencies

| Package | Purpose |
|---|---|
| Python 3.10+ | runtime |
| NumPy / SciPy | physics core, integration |
| Gymnasium | environment API |
| Stable-Baselines3 | PPO/SAC + VecEnv |
| Optuna | hyperparameter optimization |
| matplotlib / pandas | analysis, MATLAB-style plots, comparison |
| (optional) JAX | GPU-vectorized env if throughput-limited |
| ~~MATLAB Engine for Python~~ | **not needed** — validate against exported CSV fixtures |

---

*End of Rev 4. Scope locked; building the physics core + validation gate next.*
