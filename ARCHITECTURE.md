# Re-entry Guidance & Control — Full System Architecture

**A reinforcement-learning re-entry guidance & control system, and the SCvx trajectory-optimization framework it is validated against.**

| | |
|---|---|
| **RL system** | `E:\reentry_RL` (Python — Gymnasium / Stable-Baselines3 / Optuna) |
| **SCvx reference** | `E:\Code\reentry_simulator` (MATLAB — hp-pseudospectral successive convexification) |
| **Vehicle / mission** | NASA MSFC **WB001** winged-body RLV; de-orbit entry 100 km → 25 km; **heat-load minimization only** |
| **Primary research question** | Can a *causal RL feedback policy* match or beat the *open-loop SCvx-optimal* heat load for WB001? |
| **Companion docs** | [`DESIGN.md`](DESIGN.md) (decision log / requirements); this file (as-built architecture of both halves) |
| **Status (2026-06-29)** | Physics gate PASSED · Stage 1 characterized (bank-only Pareto wall) · Stage 2 feasible at Stage-1 parity (~130 km, heat below SCvx, AoA unbounded) but the AoA-40° pin / ~130 km wall is robust to reward shaping + exploration (§6.3) · env-vs-SCvx validation plots delivered (§5.3) · awaiting direction on the break-the-wall fork |

> This document describes **both** halves of the project and how they connect. Part I is the shared ground truth (vehicle + physics). Part II is the SCvx optimizer (the benchmark generator). Part III is the RL architecture (the system under development). Part IV is the validation/comparison bridge. Part V is current results.

---

## Table of Contents

1. [System overview & the two-framework relationship](#1-overview)
2. [Part I — Shared physical models (ground truth)](#2-physics)
3. [Part II — SCvx reference framework (MATLAB)](#3-scvx)
4. [Part III — RL architecture (Python)](#4-rl)
5. [Part IV — Validation & RL↔SCvx comparison](#5-validation)
6. [Part V — Results to date](#6-results)
7. [Repository layouts](#7-repos)
8. [Key design decisions & rationale](#8-decisions)
9. [Research questions](#9-rqs)

---

<a name="1-overview"></a>
## 1. System overview & the two-framework relationship

Two independent solvers attack the **same** optimal-control problem on the **same** vehicle and physics, so their solutions are directly comparable:

```
                       ┌──────────────────────────────────────────────┐
                       │      SHARED GROUND TRUTH  (Part I)            │
                       │  WB001 MCI · US-76 atmosphere · J2 gravity ·  │
                       │  polynomial aero · Sutton–Graves heating ·    │
                       │  3-DOF rotating-Earth EOM · path constraints  │
                       └───────────────┬───────────────┬──────────────┘
                                       │               │
                 verbatim port  ┌──────┘               └──────┐  verbatim port
                                ▼                              ▼
   ┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
   │  PART II — SCvx (MATLAB)             │   │  PART III — RL (Python)             │
   │  Open-loop optimal trajectory.      │   │  Closed-loop feedback policy.       │
   │  hp-flipped-LGR pseudospectral      │   │  Gymnasium env + PPO (SB3).         │
   │  transcription + successive         │   │  Cascade: bank → bank+AoA → 6-DOF.  │
   │  convexification (trust region,     │   │  Reward = soft objective + barriers.│
   │  virtual control, penalized merit). │   │  Optuna HPO. Trains IN the simulator.│
   │  Receding-horizon guidance wrapper. │   │                                     │
   └──────────────────┬──────────────────┘   └──────────────────┬──────────────────┘
                      │  exports trajectory CSV +                │  rolls out policy
                      │  J_heat benchmark (J_ref)                │  in the SAME physics
                      └────────────────┬─────────────────────────┘
                                       ▼
                       ┌──────────────────────────────────────────────┐
                       │   PART IV — VALIDATION & COMPARISON           │
                       │  (a) physics gate: replay SCvx commands in    │
                       │      the Python core → match node-for-node    │
                       │  (b) performance: RL J_heat  vs  mode-matched │
                       │      SCvx J_ref; terminal miss; constraints   │
                       └──────────────────────────────────────────────┘
```

- **SCvx** solves an open-loop boundary-value problem off-line: it *knows the whole horizon* and meets hard terminal conditions exactly. It is the **performance ceiling** and the source of the `J_ref` heat-load benchmark.
- **RL** learns a *causal* state→action feedback policy that runs forward in time with no preview. It can never satisfy 5 hard terminal equalities the way a BVP can, so its terminal conditions become **soft footprint + windows**, and the question is how close a feedback law gets to the open-loop optimum.
- The **bridge** is the verbatim physics port: the RL Python core reproduces the MATLAB models to < 0.001 % on `J_heat`, so any performance difference is attributable to *method* (causal feedback vs open-loop BVP), not to *modeling*.

---

<a name="2-physics"></a>
## 2. Part I — Shared physical models (ground truth)

Everything here is identical in both frameworks (the Python side is a verbatim port of the MATLAB models, validated node-for-node — §5.1).

### 2.1 WB001 vehicle (`vehicles/wb001/wb001_setup.m` → `reentry_rl/physics/constants.py`)

| Quantity | Value |
|---|---|
| Mass | 104 305 kg |
| Reference area `Sref` | 391.22 m² |
| Reference length / span `Lref = bref` | 29.2 m |
| Inertia `Ixx / Iyy / Izz` | 1.9979e7 / 2.7663e8 / 2.8384e7 kg·m² (products = 0) |
| Aero model | Polynomial `CL(α), CD(α)` (Mach-independent), `wb001_polynomial_aero.m` |
| Heat-rate model | Sutton–Graves `Q̇ = kQ·√ρ·Vⁿ`, `kQ = 1.65e-4`, `n = 3.15` |
| Nominal AoA schedule | 40° above 4570 m/s, tapering toward 5° at low speed (coeff 0.20705, ref 340 m/s); trim 20° |
| Path limits | `Q̇ ≤ 1.5 MW/m²`, `n ≤ 2.5 g`, `q̄ ≤ 18 kPa` |
| State bounds | h ∈ [5,120] km, V ∈ [50,8500] m/s, γ ∈ ±89°, **σ ∈ [−85°,+85°] (signed)**, α ∈ [5°,40°] |
| Rate-control bounds | `σ̇ ≤ ±45°/s`, `α̇ ≤ ±15°/s` |
| 6-DOF actuators (Stage 3) | roll/pitch/yaw surfaces δ_max = 18/20/25°, τ = 50 ms; RCS τ_max = 1.25 MN·m, Isp = 220 s, L_eff = 8 m, τ = 30 ms |

**Max L/D of WB001 ≈ 2.0 at α ≈ 16°**; at α = 40° L/D ≈ 1.07 (≈ half). This single fact drives the Stage-2 range behavior (§6.3).

### 2.2 Mission (IC / target)

| | Altitude | Lon | Lat | Velocity | FPA γ | Heading ψ |
|---|---|---|---|---|---|---|
| **Initial** | 100 km | 0° | 0° | 7450 m/s | −0.5° | 0° (North) |
| **Target** | 25 km | 12°E | 70°N | *free* | −10° | 90° (East) |

Great-circle initial range-to-go ≈ 7840 km; episode horizon ≈ 1550–1640 s.

### 2.3 Environment models

| Model | File (MATLAB → Python) | Notes |
|---|---|---|
| Atmosphere | `atmosphere_properties.m` → `atmosphere.py` | US-1976, piecewise to 120 km; ρ, T, a, dρ/dh |
| Gravity | `entry_gravity_j2.m` → `gravity_j2.py` | rotating oblate-spheroid J2; radial + latitudinal components |
| Geodesy | `ellipsoidal_altitude.m` → `geodesy.py` | WGS-84 ellipsoid altitude; great-circle range & bearing |
| Aero | `wb001_polynomial_aero.m` → `aero_wb001.py` | polynomial CL/CD; `L,D = q̄·Sref·{CL,CD}` |
| Heat rate | path-constraint #1 | `Q̇ = kQ·√ρ·V^3.15` |

### 2.4 Non-dimensionalization (identical constants)

```
length    r_nd = r / R0                  R0 = 6378 km,  g0 = 9.80665 m/s²
velocity  V_nd = V / VSCALE              VSCALE   = √(R0·g0) ≈ 7908 m/s
time      via RATE_SCALE = √(g0/R0)      TIME_SCALE = √(R0/g0) ≈ 806 s
Earth     OMEGA = ΩE / RATE_SCALE        ΩE = 7.2921159e-5 rad/s
force     L_nd = KFORCE·ρ·V_nd²·CL       KFORCE = R0·Sref/(2·mass)  (accel in g0 ⇒ n = hypot(L_nd,D_nd))
```

### 2.5 Equations of motion — 3-DOF translational, rotating oblate-J2 Earth

State `x = [r, lon, lat, V, γ, ψ, σ, (α)]` (non-dim except angles). Verbatim in `entry_dynamics_bank_aoa_no_control.m` and `eom_3dof.py`:

```
ṙ   = s·V·sinγ
lȯn = s·V·cosγ·sinψ / (r·cos lat)
laṫ = s·V·cosγ·cosψ / r
V̇   = s·( −D + g_r·sinγ + g_δ·sinψ·cosγ + Ω²·r·cos lat·(sinγ·cos lat − cosγ·sin lat·cosψ) )
γ̇   = s·( L·cosσ/V + V·cosγ/r + g_r·cosγ/V − g_δ·sinψ·sinγ/V
          + 2Ω·cos lat·sinψ + Ω²·r·cos lat·(cosγ·cos lat + sinγ·sin lat·cosψ)/V )
ψ̇   = s·( L·sinσ/(V·cosγ) + V·cosγ·sinψ·tan(lat)/r − g_δ·sinψ/(V·cos lat)
          − 2Ω·(tanγ·cosψ·cos lat − sin lat) + Ω²·r·sin lat·cos lat·sinψ/(V·cosγ) )
σ̇   = u₁                  (bank-rate control)
α̇   = u₂                  (AoA-rate control; Stage 2+ only)
```
`s = RATE_SCALE`; `g_r, g_δ` = radial/latitudinal J2 gravity; full Coriolis + centrifugal terms (rotating Earth).

### 2.6 Objective & path constraints (shared)

```
minimize   J_heat = ∫₀^tf Q̇ dt          Q̇ = kQ·√ρ·V^3.15   [W/m²]
subject to Q̇ ≤ 1.5 MW/m²,  n ≤ 2.5 g,  q̄ ≤ 18 kPa
```
`J_heat` is **the** objective for both solvers and the primary comparison metric.

---

<a name="3-scvx"></a>
## 3. Part II — SCvx reference framework (MATLAB)

A research-grade **hp-pseudospectral successive-convexification** trajectory optimizer with a **receding-horizon guidance** wrapper. It produces the open-loop optimal trajectories that serve as the RL benchmark.

### 3.1 Three guidance modes (map 1:1 onto the RL cascade)

| Mode | stateDim | controlDim | optimizeAoA | freeFinalTime | RL analogue |
|---|---|---|---|---|---|
| `bank_free_time` | 7 | 1 (σ̇) | no (α on schedule) | yes | Stage 1 |
| `bank_aoa_free_time` | 8 | 2 (σ̇, α̇) | yes | yes | Stage 2 |
| 6-DOF propagator + LQR/MPC | — | attitude controllers | — | — | Stage 3 |

Mode-dependent fields (`stateDim`, bounds, `targetState`, `terminalFixMask`, trust-region radii, selector matrices) are all set by the idempotent helper `guidance/apply_guidance_mode.m`.

### 3.2 Transcription — hp flipped Legendre–Gauss–Radau (`hp_flipped_lgr_setup.m`)

The continuous OCP is discretized by an **hp-pseudospectral** scheme: the horizon τ ∈ [0,1] is split into `numSegments` segments, each with `nodesPerSegment` flipped-Radau collocation nodes (node at the segment's right end; non-collocated point at the left).

- **Differentiation matrix** `D` (barycentric) maps node states to their τ-derivatives.
- **Radau quadrature weights** `w` integrate the running cost `∫Q̇ dt` exactly to polynomial order.
- Operators depending only on `(numSegments, nodesPerSegment)` are cached across SCP iterations and solves.
- Collocation (defect) condition per collocated node, for free final time `tf`:

  ```
  D·X  −  (tf / (2·numSegments))·f(X,U)  =  0
  ```

### 3.3 Successive convexification — penalized trust-region loop (`scp_entry_trustregion.m`)

Each SCP iteration linearizes the nonlinear dynamics + path constraints about the current iterate `ẑ`, solves a **convex subproblem**, and accepts/rejects via a trust-region merit test.

**Convex subproblem decision variables** (`build_subproblem_*`): `X` (states at all nodes), `U` (rate controls), `ν` (virtual control — defect relaxation, for recursive feasibility), `η / tη` (virtual-buffer / penalty epigraph variables).

**Linearized convex subproblem:**
```
minimize    ⟨∇J_heat(ẑ), X⟩ / 1e6   +   w_η·‖ν‖            (linearized heat integral + virtual-control penalty)
s.t.   D·X − (tf/2N)·[A(ẑ)·X + B·U] − ν = (tf/2N)·[f(ẑ) − A(ẑ)·ẑ]   (linearized collocation defects)
       pathGrad(ẑ)·X ≤ pathMax − pathValue(ẑ) + pathGrad(ẑ)·ẑ        (linearized Q̇, n, q̄ ≤ limits, all nodes)
       max(stateLower, ẑ−δ) ≤ X ≤ min(stateUpper, ẑ+δ)               (box TRUST REGION, radius δ)
       −controlMax ≤ U ≤ controlMax,   X(:,1) = x0 (fixed IC)
       X(terminalFixMask, N) = targetState                           (hard terminal BCs)
```
- **Objective** (`min_heatload`): the heat-load integral is linearized as `Σ_k (tf·w_k)·∇Q̇_k·X_k` (Radau-weighted gradient of the heat-rate path function), giving a **linear** cost in the convex variables.
- **Virtual control `ν`** on every defect guarantees the subproblem is always feasible (avoids artificial infeasibility); it is L1/L∞-penalized with weight `w_η` and driven to ~0 at convergence.
- **Trust region `δ`** is a per-state box (initial radius `δ0 = [20 km/R0, 20°, 20°, …]`, `δmin = 1e-6·δ0`, `δmax = 1e3·δ0`); the state scaling equals `δ0`.
- **Terminal BCs**: a `terminalFixMask` pins the 5 hard terminal states (alt, lon, lat, γ, ψ) to the target; **bank σ and final velocity V are left free** (the optimum picks them). Optional terminal-distance (SOC disk) and terminal-Mach constraints exist but are **disabled** by default (point-targeting).

**Trust-region accept/reject** (exact-penalty merit `M = J + μ_dyn·‖defect‖ + μ_path·‖pathViol‖`, `evaluate_merit.m`):
```
ρ = (exact merit decrease) / (predicted merit decrease)
accept if  exactDecrease ≥ ξ·predDecrease            (ξ = 0.30)
  on accept:  ẑ ← candidate;  grow δ ×βGrow only if ρ ≥ 0.9 (quality-gated grow)
  on reject:  shrink δ ×βShrink
penalty boost: after a reject streak, μ_dyn, w_η ×3 (e.g. 3e7→9e7→2.7e8→8.1e8) and δ partially reset
converged when  maxDefect ≤ defectTol  AND  scaledStep ≤ stepTol
```
The quality-gated grow (ρ ≥ 0.9) removes a parasitic accept/reject oscillation that an unconditional 5 %-grow caused. **Free final time** `tf` is itself a decision variable with its own trust radius `δ_tf`.

### 3.4 Receding-horizon guidance wrapper (the benchmark generator)

`solve_entry_case.m` runs either as a **standalone** open-loop solve or as a **guidance call** in a receding-horizon loop (`execute_guidance_profile.m`, `shift_solution_for_guidance.m`, `apply_guidance_mesh_schedule.m`): the SCP is re-solved over a shifting/shrinking horizon (MPC-style), the first portion of each solution is executed by propagating it through the nonlinear simulator, and the realized states + rate commands are logged. The exported `guidance_executed_trajectory.txt` from this loop **is the benchmark** the RL system is measured against.

### 3.5 6-DOF propagator + inner attitude control (Stage-3 context)

`propagator/simulator_state_rhs.m` adds Euler-angle kinematics, `ω̇ = J⁻¹(M_body − ω×Jω)`, six first-order actuators (3 RCS + 3 aero surfaces), and a propellant-flow model. The reference closes attitude with **LQR / MPC** roll/pitch/yaw controllers (`control/`). In the RL cascade these hand-tuned inner controllers are what the Stage-3 learned inner policy replaces.

### 3.6 Benchmark outputs used by the RL project

Per run folder: `guidance_executed_trajectory.txt` (CSV: states + rate commands), `guidance_terminal_summary.txt`, `guidance_call_history.txt`, `guidance_profiles.{png,svg}`, `guidance_solution.mat`, `console_log.txt`.

| Stage | Mode | J_heat (J/m²) | t_f (s) | Terminal hit |
|---|---|---|---|---|
| **1** | `bank_free_time` | **1.0832e9** | 1636.7 | lon 12.00002°, lat 70.0003°, γ −9.84°, ψ 89.76° |
| **2** | `bank_aoa_free_time` | **1.0720e9** | 1547.8 | lon 12.00008°, lat 70.00002°, γ −10.02°, ψ 89.94° |

Both hit the target to ~metres. Adding AoA buys ≈ 1 % heat load and ≈ 89 s. These two numbers (`J_REF_BANK_ONLY`, `J_REF_BANK_AOA`) are the **mode-matched references** baked into `constants.py`. The SCvx keeps AoA **high (≈ 28–40°)** throughout the bank+AoA solution.

---

<a name="4-rl"></a>
## 4. Part III — RL architecture (Python)

A reusable `reentry_rl/` package: a NumPy physics core, a Gymnasium environment specialized per cascade stage, a task-space reward, PPO training (SB3) with resumable long runs, Optuna HPO, and MATLAB-style post-processing.

### 4.1 Physics core (`reentry_rl/physics/`)

`constants.py` · `atmosphere.py` (US-76) · `gravity_j2.py` · `geodesy.py` (ellipsoid altitude, great-circle range/bearing) · `aero_wb001.py` (polynomial CL/CD + Sutton–Graves Q̇ + `path_quantities`) · `eom_3dof.py` (the EOM of §2.5 + an RK4 stepper). Pure NumPy, vectorized. This core **doubles as the MATLAB-validation reference** and as the env's dynamics.

### 4.2 Gymnasium environment (`reentry_rl/envs/`)

One base class `EntryEnvBase` (`entry_env_base.py`), specialized per stage. Single source of truth for observation, reward, termination.

- **Augmented integration state** `z = [r, lon, lat, V, γ, ψ, σ, (α)]`. `σ` is always a state driven by the bank-rate action; `α` is a state driven by the AoA-rate action only when `optimize_aoa` (Stage 2+), otherwise it follows the WB001 schedule.
- **Action = attitude RATES**, normalized to [−1,1], scaled to `σ̇ ≤ ±45°/s`, `α̇ ≤ ±15°/s`, **integrated in-env** with **RK4** (decision Δt = 1 s, `n_substeps = 3`). Angles clamped to bounds (σ ∈ [−85,85]°, α ∈ [5,40]°). *(Rate actions, not angle actions, so the policy commands a physically-rate-limited attitude profile — matching the SCvx control variables.)*

  | Stage | Action | Dim |
  |---|---|---|
  | 1 | σ̇ | 1 |
  | 2 | σ̇, α̇ | 2 |
  | 3 (inner) | 3× RCS torque + 3× surface deflection | 6 |

- **Observation (13-D, Stage 2; 12-D Stage 1)** — velocity is the *phase variable*:
  ```
  [ h/100km, V/7450, γ, sinψ, cosψ, σ/σmax, q̄/q̄max, Mach/25,
    Q̇/Q̇max, n/nmax, range-to-go/range0, heading-error/π, (α/40° in Stage 2) ]
  ```
  Perfect navigation (full-state observation). Wrapped via SB3 `VecNormalize` (obs only).

- **Termination**: success manifold = altitude ≤ 25 km → terminal reward; failure = altitude > 120 km, V < 50 m/s, |γ| > 89°, NaN, or step-budget exceeded → `−w_fail`.

### 4.3 Reward design (`rewards.py` + `entry_env_base._reward`)

Task-space, **action-dimension-agnostic** (so it transfers across stages — only `w_act`/propellant scale with action dim). Current as-built form:

```
per-step:
  r = − w_heat · ΔJ_heat / J_ref                                   (PRIMARY: heat increment; softened, heat ≈ SCvx)
      − w_path · [ barrier(Q̇/Q̇max) + barrier(q̄/q̄max) + barrier(n/nmax) ]   (soft feasibility, turn-on at 0.9·limit)
      − w_act  · ‖a‖²                                              (effort)
      − w_smooth · ‖a − a_prev‖²                                   (command smoothness — kills bank chatter)
      − w_aoa_lo · max(0, α_floor − α)²        [Stage 2 only]      (DIVE PROTECTION: keep AoA up → lift up)
      + w_progress · ( pos_pen(d_prev) − pos_pen(d_now) )          (DENSE potential shaping ⇒ credit assignment)

terminal (altitude ≤ 25 km), d = great-circle landing error [km]:
  r += (+w_succ)            if d ≤ 1 km                            (success bonus)
       (−w_pos·log10 d)     if d > 1 km                            (LOG position penalty, representative 1000s km→1 km)
  r −= w_fpa · min((Δγ/γ_tol)², 25)                               (terminal ENERGY term → arrive near γ* = −10°)
  success = (d ≤ 1 km)                                             (position-only success)

failure: r −= w_fail
```
where `pos_pen(d) = w_pos·log10(max(d,1))` and `barrier(x, soft=0.9) = (max(0,x−soft)/(1−soft))² + max(0,x−1)·(10+10·max(0,x−1))`.

**Design notes / lessons baked in:**
- The **dense potential-based shaping** telescopes to `−pos_pen(d_final)` (only the *end* position matters), but distributes that objective over the ~1600-step episode — essential for **credit assignment**; removing it stalled the policy at 400–700 km.
- The **LOG position penalty** is representative across scales (every decade of error costs `w_pos`); the steepness lever is `w_pos`, which steepens terminal *and* dense gradients together.
- **Path constraints are soft barriers** (preserve gradient), not hard kills.
- **FPA/heading were deferred** ("position first, velocity orientation later"); the **terminal energy (FPA) term is the latest addition** to break the Stage-2 range wall (§6.3) — it is physically *aligned* with range (a shallow arrival needs retained energy → better L/D → more range).

**Reward weights — meaning, code default, and role** (`RewardWeights` in `rewards.py`; every weight is overridable from the `train_sb3.py` CLI):

| weight | default | term | role / tuning lever |
|---|---|---|---|
| `w_heat` | 0.2 | `−w_heat·ΔJ/J_ref` | primary objective (softened — heat already ≈ SCvx); raising it pins AoA high (§6.3) |
| `w_path` | 3.5 (HPO 3.65) | soft barriers on Q̇/n/q̄ | feasibility; the main "don't violate" pressure |
| `w_act` | 0.05 | `−w_act·‖a‖²` | command effort; **also caps how large the rate commands grow** |
| `w_smooth` | 0.2 (HPO 0.38) | `−w_smooth·‖Δa‖²` | command smoothness; **the main clamp on bank-swing width** |
| `w_aoa_lo` | 0.0 → **1.0** (Stage 2) | `−w_aoa_lo·max(0,α_floor−α)²` | dive protection (soft, no bound); keeps AoA ≥ `aoa_floor_deg` |
| `aoa_floor_deg` | 20.0 | — | AoA below which the low-AoA penalty turns on |
| `w_progress` | 1.0 | dense potential shaping | distributes the terminal position objective → credit assignment |
| `w_pos` | 50 (Stage 2 150–250) | `w_pos·log10 d` | terminal + dense position steepness (the position pull) |
| `w_succ` | 100 | terminal bonus | reward for d ≤ 1 km |
| `w_fpa` | 0.0 → **4.0** (v4+) | `−w_fpa·min((Δγ/γ_tol)²,25)` | terminal energy term (arrive near γ* = −10°); `γ_tol = 5°` |
| `w_fail` | 50 | `−w_fail` | skip-out / crash / OOB / timeout |
| `w_psi` | 0.0 | (deferred) | terminal heading — kept 0 (symptom of undershoot, §6.3) |

### 4.4 The three-stage cascade (`stage1_bank.py`, `stage2_bank_aoa.py`, `stage3_full.py`)

| Stage | Policy | Sim | SCvx analogue | New element |
|---|---|---|---|---|
| **1** | bank-rate (1-D); α on WB001 schedule | 3-DOF | `bank_free_time` | reward + HPO recipe; `J_ref = 1.0832e9` |
| **2** | bank + AoA rate (2-D), **no AoA bounds** | 3-DOF | `bank_aoa_free_time` | reuse Stage-1 reward → measure transfer & whether AoA beats the bank-only wall; `J_ref = 1.0720e9` |
| **3** | inner attitude + allocation (6-D) tracking **frozen Stage-2** guidance | 6-DOF | guidance + LQR/MPC | hierarchical two-timescale cascade + propellant penalty (prefer free surfaces over RCS) |

Stage 3 is two-timescale: the frozen Stage-2 policy issues σ_cmd/α_cmd at the slow Δt; the inner policy runs at ~10–20 ms, outputs 6 actuator actions, tracks the commands and regulates sideslip β→0.

### 4.5 Training (`reentry_rl/training/`)

- **Algorithm**: PPO (SB3), `MlpPolicy` (net `[256,256]` default; `[512,512]` from HPO), CPU.
- **Vectorization**: `SubprocVecEnv` (Windows-`spawn`-safe — `ensure_child_importable` puts the repo on `PYTHONPATH`); `VecNormalize(norm_obs=True, norm_reward=False)`.
- **`train_sb3.py`** — the production trainer. CLI: `--stage`, `--timesteps`, `--n-envs`, `--hpo-best <best.json>` (load tuned PPO + reward weights), per-weight overrides (`--w-pos/--w-fpa/--w-progress/--w-aoa-lo/--w-heat/--w-path/--w-smooth/--w-act/…`), `--ent-coef`, **`--lr-decay`** (linear LR→0, a convergence aid), **`--resume <run_dir>`** (continue from latest checkpoint + VecNormalize; restores `num_timesteps` + LR schedule). `--tb` is **off by default** (the SB3 TensorBoard event-writer crashes on Windows at multi-million steps). *Exploration is controlled by `--ent-coef` (policy stochasticity) together with `--w-smooth`/`--w-act` (which gate how wide the rate commands can swing) — see §6.3.*
- **Callbacks**: `EvalCallback` (deterministic eval + best-model save), `CheckpointCallback` (with VecNormalize), and a custom `StageMetricsCallback` logging domain metrics (`d_km`, `J_ratio`, FPA/heading error, peak Q̇/n/q̄ ratios, success) so a long run is interpretable beyond `ep_rew_mean`.
- **Resumability is load-bearing**: the training machine sleeps overnight; `--resume` lets multi-million-step runs survive interruptions. *(Run from the `tudat-space` conda env: `C:/Users/acarr/anaconda3/envs/tudat-space/python.exe`; set `KMP_DUPLICATE_LIB_OK=TRUE` to avoid the MKL/OpenMP clash.)*

### 4.6 Hyperparameter optimization (`hpo_optuna.py`)

Optuna **TPE sampler + MedianPruner**, **SQLite-resumable** study. Joint search over reward weights `{w_path, w_far, w_smooth, w_psi, w_fpa, w_progress}` + PPO `{lr, n_steps, batch, n_epochs, γ, gae_λ, clip, ent_coef, net_width}` (LR-decay always on). **The HPO objective is the true task metric — a feasibility-gated `J_heat`/terminal-accuracy score, NOT the shaped reward** — so HPO optimizes the real goal and cannot be fooled by shaping. Workflow (per `DESIGN.md` §7): **reward-design first (manual) → HPO with reward frozen → optional light refine.**

### 4.7 Post-processing (`reentry_rl/postprocessing/`)

- **`plots.py`** — reproduces the MATLAB `guidance_profiles` figure from an `eval_trajectory.csv`: altitude-vs-velocity, ground track (target ★ + miss), q̄+n vs t (with limit lines), Q̇+heat-load, AoA+bank, bank-rate + range-to-go. Exports PNG (200 dpi) + SVG. Each run also saves the trained controller (`model.zip` + `vecnormalize.pkl`) and the per-run trajectory CSV under `results/<stage>_<tag>/`.
- **`validate_plots.py`** — the env-vs-SCvx overlay figures (§5.3): replays the SCvx command history through our core and overlays trajectories + controls + node-for-node residuals.

---

<a name="5-validation"></a>
## 5. Part IV — Validation & RL↔SCvx comparison

### 5.1 Physics validation gate (`reentry_rl/validation/`) — **PASSED**

Before any RL work, the Python core must reproduce the MATLAB benchmark trajectories node-for-node. `validate_physics.py` replays each benchmark CSV's command history (σ̇/α̇, or σ/α directly; α from schedule for bank-only) through the Python RK4 core and compares states + `J_heat` against `guidance_executed_trajectory.txt`. Result: **`J_heat` Δ < 0.001 %, altitude ≤ 20 m, V ≤ 0.9 m/s** for both benchmarks. `benchmark_scp.py` loads/parses the exported MATLAB results (handles both 9- and 11-column CSV layouts). This gate **blocks all downstream work**; no MATLAB Engine is required.

### 5.2 Performance comparison

| Metric | RL | SCvx |
|---|---|---|
| **Heat load `J_heat` (primary)** | closed-loop policy rollout | receding-horizon SCvx (benchmark) |
| Terminal miss (lon/lat/γ/ψ) | soft footprint + windows | near-zero (hard fixes) |
| Peak path-constraint margins (Q̇, n, q̄) | ✓ | ✓ |
| Control smoothness / effort | ✓ | ✓ |
| Onboard inference cost | per-step policy eval (µs) | per-call convex solve (ms–s) |

Mode-matched normalization: Stage-1 RL vs `J_REF_BANK_ONLY`, Stage-2 RL vs `J_REF_BANK_AOA`. The interesting asymmetry: SCvx wins terminal accuracy by construction (open-loop BVP); RL wins onboard cost (a forward pass vs an online convex solve). The heat-load comparison is only *fair at comparable terminal accuracy* — a policy that misses by 150 km can show a lower `J_heat` simply by flying an easier path.

### 5.3 Environment-validation overlay plots (`postprocessing/validate_plots.py`)

Beyond the numerical gate, this module drives **our** 3-DOF core with the **SCvx command history** (`replay_full`) and overlays the resulting trajectory on the SCvx trajectory — the *visual* counterpart of §5.1, confirming the simulator reproduces the reference dynamics under identical controls. Each 3×2 figure: altitude-vs-velocity, ground track, FPA + heading, **the SCvx control history** (the σ/α that drove both), heat flux + cumulative load, and node-for-node residuals. Results (`results/validation_plots/env_validation_{bank_only,bank_aoa}.png`):

| benchmark | max \|Δalt\| | max \|ΔV\| | ΔJ_heat |
|---|---|---|---|
| bank-only | 2.9 m | 0.10 m/s | +0.0006 % |
| bank + AoA | 19.7 m | 0.86 m/s | +0.0004 % |

The two curves lie on top of each other. The control-history panels are independently informative: the SCvx optimum swings **bank +45°→−60°** and **modulates AoA down to ~22°**, vividly showing the action-space usage the converged RL policy (bank ±8°, AoA 40°, §6.3) does *not* reach.

---

<a name="6-results"></a>
## 6. Part V — Results to date

### 6.1 Physics gate — PASSED (§5.1).

### 6.2 Stage 1 (bank-only) — characterized; bank-only Pareto wall confirmed

Feasible across all reward designs (peak n ≈ 0.5–0.9×, q̄ ≈ 0.6–0.9×, Q̇ ≈ 0.98×; 0 % violation), heat **≈ +2 %** of SCvx (primary metric essentially met). Smoothness solved (`w_smooth` → bank reversals 1440→2), convergence solved (LR decay). **Optuna HPO** confirmed a hard **position-vs-heading Pareto trade for bank-only**: no feasible config hits both — best feasible ≈ 130 km miss with heading stuck ≈ 49°, or heading in-window at ≈ 460 km. The SCvx proves both *are* simultaneously achievable open-loop, so this is a causal-feedback limitation, motivating Stage 2's extra control authority.

### 6.3 Stage 2 (bank + AoA, no bounds) — feasibility solved; breaking the range wall

The naive "same reward + add AoA" failed catastrophically (policy dove via low AoA → q̄ 5–11× limit). Diagnosis → fix sequence:
- An AoA-**rate** penalty couldn't stop a slow drift to low AoA (drift = tiny rate). → replaced with a **value-based low-AoA penalty** `w_aoa_lo·max(0, α_floor−α)²` (dive protection; soft, **no bound** on AoA).
- **v3 (4M steps): feasibility solved, decisively.** Final policy fully feasible with margin (max q̄ **0.12×**, n **0.47×**, Q̇ **0.95×**), heat **J/J_ref = 0.93 (below SCvx)**, best-eval miss **155 km** ≈ Stage-1 parity — achieved with **AoA fully unbounded**, meeting the directive.
- **Diagnosis of the residual miss**: it is an **undershoot driven by energy**, not steering. The policy pins **AoA at 40°** (L/D ≈ 1.07 ≈ *half* of WB001's max-L/D ≈ 2.0 at 16°), so it bleeds energy and descends too steeply (final γ ≈ −25° vs −10°) and runs out of altitude short while flying *straight at* the target (bank ≈ 0).

**Four 4M-step experiments to break below the wall — all converged to the same AoA-40° pin:**

| run | intervention | AoA | best miss | feasible | J/J_ref |
|---|---|---|---|---|---|
| v3 | dive protection only | 40° | 155 km | ✅ | 0.93 |
| v4 | + terminal energy term, `w_pos` 150→250 | 40° | **130 km** | ✅ | 0.94 |
| v5 | + heat reward cut 10× (`w_heat` 0.2→0.02) | 40° | 166 km | ✅ | 0.94 |
| v6 | + exploration (`ent_coef`↑, `w_smooth`/`w_act`↓) | 40° | 329 km | ✅ | 0.99 |

- **The finding:** AoA = 40° is a **robust local optimum** — simultaneously the *minimum-heat* profile (high lift → stay high → low ρ → low Q̇; the SCvx keeps AoA high for the same reason) and the *safest* (max-lift, gentlest descent, lowest q̄/n). Neither removing the heat reward (v5 — heat *stayed* 0.94× anyway) nor adding exploration (v6 — which only made the deterministic policy noisier and *worse*) pulled AoA off it. Bank likewise stays ≈ ±8° vs the SCvx's ±45–60°.
- **Why shaping/exploration can't break it:** the optimal trajectory (large bank + AoA modulated to ~16–22° for L/D) requires *global horizon planning*. Greedy local PPO sees only that lowering AoA / banking harder is **immediately** worse (steeper descent, tighter constraints), while the range payoff arrives ~1600 steps later — discounted to ~20 % at γ = 0.999. It is a **credit-assignment / global-planning gap, not an exploration-quantity one.** The env-validation control panels (§5.3) make the contrast concrete.
- **Status:** Stage-2's stated bar — Stage-1 parity (≈ 130 km), fully feasible, heat below SCvx, **AoA unbounded** — is **met**. Breaking *below* 130 km (proving AoA's authority beats bank-only) needs a mechanism change, not another reward knob. **Open fork (awaiting direction):** (A) recurrent policy (LSTM, `RecurrentPPO`) + richer energy-aware observations [the credit-assignment fix]; (B) direct max-L/D AoA shaping [fast "is it breakable at all?" test]; (C) accept parity → Stage 3 (6-DOF).

> **Net result so far:** adding AoA to the action space does **not** automatically beat bank-only. A from-scratch feed-forward PPO policy converges to the safe, heat-optimal high-AoA regime and reproduces the bank-only ~130 km causal-feedback wall — a clean negative that motivates either richer policy structure (memory + observations) or explicit guidance.

### 6.4 Experiment configurations (hyperparameters & reward weights per case)

**Shared PPO hyperparameters** — every Stage-2 run loads the **Stage-1 Optuna HPO winner (trial #0)** via `--hpo-best`, with `ent_coef` overridden and LR-decay on:

| hyperparameter | value | source |
|---|---|---|
| learning rate | 2.22e-4 → 0 (linear decay) | HPO trial #0 (`lr_base`) |
| `n_steps` | 1024 | HPO |
| `batch_size` | 256 | HPO |
| `n_epochs` | 5 | HPO |
| `gamma` (γ) | 0.99891 | HPO |
| `gae_lambda` (λ) | 0.9623 | HPO |
| `clip_range` | 0.2740 | HPO |
| net arch | [512, 512] tanh MLP | HPO |
| `ent_coef` | 0.005 (v6: **0.02**) | manual override |
| n_envs / n_substeps / seed / total steps | 8 / 3 / 0 / 4 M | fixed |

*(The base config is the best feasible config from the Stage-1 TPE+MedianPruner HPO over reward weights + PPO. `ent_coef` is overridden because the HPO-tuned 0.018 — found on 800 k-step trials — failed to converge on the 4 M-step runs; 0.005 converges cleanly, and v6 raised it to 0.02 specifically to test exploration.)*

**Per-case settings & outcomes** — only `ent_coef` and the listed reward weights change between runs; everything else is the shared config above. **Bold** marks each run's deliberate change vs the previous.

| run | ent | `w_heat` | `w_pos` | `w_fpa` | `w_smooth` | `w_act` | `w_aoa_lo` | outcome (best) |
|---|---|---|---|---|---|---|---|---|
| **v3** | 0.005 | 0.2 | 150 | 4.81 *(unused)* | 0.38 | 0.05 | **1.0** | feasible, 155 km, J 0.93 — dive solved |
| **v4** | 0.005 | 0.2 | **250** | **4.0** | 0.38 | 0.05 | 1.0 | feasible, **130 km**, J 0.94 — AoA still 40° |
| **v5** | 0.005 | **0.02** | 250 | 4.0 | 0.38 | 0.05 | 1.0 | feasible, 166 km, J 0.94 — heat not the pin |
| **v6** | **0.02** | 0.2 | 250 | 4.0 | **0.05** | **0.01** | 1.0 | feasible, 329 km — exploration didn't escape basin |

Constant across v3–v6: `w_path = 3.65`, `w_progress = 1.0`, `w_succ = 100`, `w_fail = 50`, `aoa_floor = 20°`, `w_psi = 0`. v3's `w_fpa = 4.81` was loaded from the HPO file but **unused** — the terminal energy term was only added to the code in v4 (before that the reward had no FPA term).

**Reading the series:** v3 establishes feasibility (the `w_aoa_lo` dive-protection); v4 adds the energy term + steeper position; v5 isolates heat (cut 10×); v6 isolates exploration (entropy up, command-regularization down). Every run holds AoA at 40° and lands at the ~130 km wall → the conclusion in §6.3.

**Stage-1 cases (provenance of the base config):** the bank-only campaign evolved the terminal reward quadratic/exponential → **LOG penalty**; raised `w_path` 1→3 to enforce feasibility; added `w_smooth` (0→0.2) killing bank chatter (1440→2 reversals); added LR-decay for convergence; then the Optuna HPO (above) mapped a hard **position-vs-heading Pareto trade** (best feasible ≈ 130 km / ψ 49°, *or* ψ in-window / ≈ 460 km), heat ≈ +2 % throughout. Full chronology in the project memory log.

---

<a name="7-repos"></a>
## 7. Repository layouts

### 7.1 RL system — `E:\reentry_RL`
```
reentry_rl/
  physics/         constants · atmosphere · gravity_j2 · geodesy · aero_wb001 · eom_3dof
  envs/            entry_env_base · stage1_bank · stage2_bank_aoa · (stage3_full) · rewards
  validation/      benchmark_scp · validate_physics            (load MATLAB results; node-for-node gate)
  training/        train_sb3 · common · hpo_optuna             (PPO + VecEnv + resume; Optuna HPO)
  postprocessing/  plots · validate_plots                      (profile figures; env-vs-SCvx overlays)
results/           <stage>_<tag>/  (model.zip, vecnormalize.pkl, config.json, ckpt/, best/,
                                    eval_trajectory.csv, eval_summary.txt, profile plots)
                   validation_plots/  (env_validation_{bank_only,bank_aoa}.png)
DESIGN.md  ARCHITECTURE.md  pyproject.toml
```

### 7.2 SCvx reference — `E:\Code\reentry_simulator`
```
guidance/
  scp/             scp_entry_trustregion · solve_entry_case · evaluate_merit · evaluate_iteration_metrics
  transcription/   hp_flipped_lgr_setup · quadrature_weights · compute_linearized_defects · decision_indices_*
  subproblem/      build_subproblem_* · solve_subproblem_*    (convex QP/SOCP per mode)
  linearization/   analytical_entry_jacobian_* · analytical_path_constraints_jacobian
  physics/         entry_dynamics_* · entry_gravity_j2 · aero_model · path_constraints_model · ellipsoidal_altitude
  guidance/        execute_guidance_profile · shift_solution_for_guidance   (receding-horizon wrapper)
  initialization/  preprocess_initial_guess · apply_initial_conditions · propagate_solution
  output/          export_guidance_txt · export_solution_txt · plot_*
vehicles/          vehicle_setup · wb001/wb001_setup · wb001_polynomial_aero · common/
propagator/        simulator_state_rhs · propagate_guidance_segment · propellant_flow_model   (6-DOF)
control/           lqr/ · mpc/ · models/ · sim/                (Stage-3 inner attitude controllers)
results/           guidance_unit_test/… · validation/…         (the exported benchmarks)
```

---

<a name="8-decisions"></a>
## 8. Key design decisions & rationale

| Decision | Rationale |
|---|---|
| **Soft terminal footprint + windows** (RL) vs hard BCs (SCvx) | A causal feedback policy cannot satisfy 5 hard terminal equalities the way an open-loop BVP can; scoring terminal accuracy with continuous shaping is the honest RL analogue. |
| **Action = attitude rates**, integrated in-env | Matches the SCvx control variables (σ̇, α̇), yields physically rate-limited profiles, and keeps the action space identical in structure across stages. |
| **Velocity as the observation phase variable** | Entry trajectories are naturally parameterized by energy/velocity; gives the policy a monotone progress signal. |
| **Action-dimension-agnostic reward** | Lets Stage 2 *reuse* the Stage-1 reward → cleanly *measures* reward transfer rather than assuming it. |
| **No AoA bounds in Stage 2** (soft penalty only) | User directive — the policy must learn safe AoA use from reward shaping, not hard limits; tests whether AoA authority genuinely helps. |
| **No SCvx warm-start** ("whatsoever") | The RL policy must be learned independently; SCvx is the *benchmark*, not an initializer. |
| **NumPy physics core first** | Doubles as the MATLAB-validation reference; escalate to JAX/GPU only if HPO is throughput-limited. |
| **Reward-first → HPO (metric = feasibility-gated J_heat) → refine** | Decouples *what* is optimized from *how well*; prevents reward-hacking and confounded attribution. |
| **Stage 3 = frozen-guidance hierarchy** | Two-timescale credit assignment is hard end-to-end; freezing Stage-2 guidance and learning only the inner allocation is tractable and matches the SCvx guidance+control split. |

---

<a name="9-rqs"></a>
## 9. Research questions

**Primary (Q-GOAL a):** can a causal RL feedback policy match/beat SCvx-optimal `J_heat` for WB001? **In scope alongside:** (b) one policy replacing the guidance+control stack (the cascade/Stage-3 thesis); (c) onboard/real-time cost (forward pass vs online convex solve); plus the "free" ◦ questions answered by the planned builds: **RQ-OPT** (optimality gap & its cause — causality vs reaction time vs shaping), **RQ-CURR** (cascade warm-start transfer), **RQ-STRUCT** (does the learned bank/AoA profile reproduce known entry structure). **Deferred** (no extra campaigns yet): (d) robustness/dispersions, target-conditioned generalization (RQ-GEN — env stays single-target), and the ⊕ sensitivity/bandwidth/Pareto studies. See `DESIGN.md` §11.3 for the full list and scope lock.

---

*Companion to `DESIGN.md` (requirements & decision log). This file is the as-built architecture of both the RL system and the SCvx reference, and the bridge between them.*
