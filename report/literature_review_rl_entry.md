# Reinforcement-Learning Guidance for Atmospheric Entry: Robustness, Stability and Evaluation Practice

*Literature review compiled 2026-10-24 for the WB001 RL entry-guidance paper (target venues: JGCD, JSR, Acta Astronautica, Aerospace Science and Technology, IEEE TAES, AIAA SciTech/GNC, ESA GNC). It extends and corrects [`literature_review.md`](literature_review.md) (2026-10-05), which covered reward design, terminal constraints and precision. This review is about the next claim: a **stable, robust** guidance law under uncertainty. Citations use Pandoc syntax (an @ sign followed by a citation key, inside brackets), and every key is in [`literature_review_rl_entry.bib`](literature_review_rl_entry.bib).*

---

## 0. How to read this review

**Verification.** Every reference was checked on 2026-10-24 against its primary metadata record: Crossref for DOIs, the arXiv API for preprints, and PMLR, JMLR, NeurIPS or publisher pages for proceedings without a Crossref DOI. Two items could not be fully verified. They are flagged in the text and in the `note` field of their BibTeX entries: `harpold1979shuttle` and `moseley1969apollo`. A third, `peng2025hrl`, has a DOI printed on the publisher page that Crossref does not resolve.

**Depth of reading.** Each source in the summary tables carries a reading level:

- **F**: full text read. Numbers are quoted from it, and for arXiv preprints from the preprint version.
- **A**: abstract or publisher summary only. Fields the abstract does not give are marked "n/r" (not reported in what was read). They are not claims that the paper omits them.
- **T**: title and metadata only.

Most AIAA and Elsevier full texts were paywalled or behind bot checks during this search. The quantitative comparisons below therefore lean on papers read in full: Gaudet's hypersonic papers, Su 2023, Jiang 2023, Bao 2025 and Hanson & Beard.

**Project state assumed** (`results/results/RUNS.md` on `point2-full-terminal-state`, commits up to f336a55):

- **Policy.** v24b (30M decisions) is a PPO policy (512×512 MLP, obs v25 with 31 features). Its actions are absolute bank and AoA targets tracked at actuator rate limits, with one decision per level of a √(energy-to-go) grid.
- **Constraints and training.** Path constraints use PID-Lagrangian multipliers and an incremental peak charge. Terminal angles are trained under an ε-curriculum. Training uses domain randomization at 0.1× the VISTA sigmas.
- **Realistic test (S3, 500 cases, 0.1×).** Median miss 1.36 km (IQM 1.36 km, 95 % CI [1.31, 1.41]). 24 % of cases end within 1 km. 79 % are strictly feasible, with heat rate the binding constraint (p95 peak ratio 1.011). Median |Δγ| is 1.16° and median |Δψ| 2.77°.
- **Full VISTA dispersions (SCvx `gc_ic_tc`, 1,001 cases, 1×).** RL median 10.5 km against SCvx 0.21 km. Feasible fractions are 34 % for RL and 39 % for SCvx.
- **Run in progress.** v25, launched 2026-10-24, trains under an ADR-style dispersion curriculum up to 1× and is selected on a robust score. It is evaluated on common-random-number scale sets S4_scale_x{0.1 … 1.25}.

---

## 1. Summary tables

### 1.1 RL guidance for entry, re-entry, hypersonic glide, Mars entry and aerocapture (primary)

| Work | Year · venue | Vehicle / phase | DOF | Algorithm | Observation → action | Uncertainty modeled | Constraints (handling) | Evaluation | Reported results | Src |
|---|---|---|---|---|---|---|---|---|---|---|
| Gaudet, Drozd, Furfaro [@gaudet2022approach; @gaudet2021approach_arxiv] | 2022 · AIAA SciTech | Hypersonic glider, booster separation 100 km / 7.45 km/s → 25 km / 1.5 km/s, ~8,000 km to randomized targets | 3-DOF, Coriolis included, exponential atmosphere | PPO with GRU layer (meta-RL) | Curved-Earth parallel-navigation velocity-field error and related states → **bank and AoA rate** commands at 0.2 Hz; AoA held at 40° until 4.57 km/s | ±3 % C_L, C_D, ρ (uniform); actuator lag, noise and failure; IC dispersions. Test-only: ±5/10 % aero/ρ, ±10 % mass/area, sensor scale-factor error, diverts, shorter range | Q̇ ≤ 3 MW/m², q̄ ≤ 50 kPa, load; a violation terminates the training episode | 5,000 episodes per test case, several cases outside the training distribution | In distribution: miss 47 ± 20 m (max 135 m), 100 % success (miss and altitude error < 1 km), 0 % violations. ±10 % aero/ρ: mean 74 m, max 3.9 km, 0.1 % violations. LQR tracking a GPOPS-II reference degraded and violated constraints. Terminal speed and altitude targeted; **no terminal γ or ψ** | F |
| Gaudet & Furfaro [@gaudet2023strike; @gaudet2021strike_arxiv] | 2023 · JSR | Hypersonic strike weapon, terminal phase: 25 km, ~3 km/s, 200 km to a maneuvering ground target | 3-DOF; preliminary 6-DOF integrated G&C | PPO, recurrent meta-RL | Radar-seeker-measurable observations → bank, AoA, sideslip **rates**, 5 Hz | 10 % aero coefficient and density variation, actuator failure, 0.1 s actuator lag, sensor scale-factor error, target motion, diverts | Heating, q̄, load, minimum impact speed 1.7 km/s | 5,000 episodes per case | In distribution: every miss < 5 m, constraints met. Generalization cases degrade with rare violations. Separate discounts for shaping (0.90) and terminal (0.995) rewards | F |
| Su, Wang, Chen [@su2023irl] | 2023 · Aerospace | CAV-H (min time) and RLV (min heat load) entry | 3-DOF | GAIL (inverse RL) + distributed PPO, from 50 GPOPS demonstrations | Normalized position/velocity → [generalized lift coefficient, bank] | Randomized initial states | Q̇, q̄, n; episode terminated on violation | 1,000 closed-loop runs per vehicle | Success 99.6 % (CAV-H) and 99.2 % (RLV), defined as constraints met and range-angle error < 0.25° / 0.5° (≈ 28 / 56 km). RLV heat load 3 % above GPOPS. 0.167 ms per command | F |
| Jiang, Wang, Bai, Li [@jiang2023her] | 2023 · Int. J. Aerospace Eng. | CAV-H glide phase, 60–64 km / 6.2–6.8 km/s | 3-DOF, rotating Earth | DDPG + hindsight experience replay; PPO and DDPG as controls | Basic state plus goal → single bank command (magnitude and sign), AoA from a fixed profile; 150 s policy step (9 decisions) | C_L, C_D error 0–10 % (uniform); IC and target dispersions | Q̇, q̄ ≤ 100 kPa, n ≤ 3; bank bounded by the H–V corridor | 1,000 Monte Carlo runs | Success means Δd < 300 km (handover to the dive), ΔH < 2 km, ΔT < 20 s: met in all runs. 0.094 s total compute vs 6.43 s for NPCG. Authors note that **dispersions beyond the training range require retraining** | F |
| Peng, Wang, Liu, Fan, Cheng [@peng2023modular] | 2023 · AST | Hypersonic entry, multi-mission (changing targets, emergency entry) | 3-DOF | DRL; separate AoA and bank modules | State → AoA profile and bank; non-uniform discretization with a **state-related independent variable** | Mission/target variation | Task constraints | n/r | Real-time trajectory generation over a wide reachable area | A |
| Peng, Wang, Liu, Fan [@peng2025hybrid] | 2025 · IEEE TAES | 3-D entry, any start/target in a large reachable area | 3-DOF | Dueling hybrid TD3 (discrete-continuous) | → bank **sign + magnitude** (hybrid action); AoA module; reward suppresses long-period oscillation and reversal count | n/r | Flight-capability requirements | n/r | "Autonomous robust trajectory planning" in a large reachable area | A |
| Peng et al. [@peng2025hrl] | 2025 · Acta Aeronaut. Astronaut. Sin. | Re-entry | n/r | Planning–correction hierarchical RL | Planning layer plans AoA/bank profiles; correction layer corrects at high frequency | Large model-parameter deviations | n/r | n/r | Handles larger deviations with better accuracy under large uncertainty; better mission adaptability and real-time performance than predictor–corrector guidance | A |
| Bao, Zhou, Wang, He, Tang [@bao2023onboard] | 2023 · Aeronaut. J. | Hypersonic glide, 3-D | 3-DOF | DDPG, actor pre-trained on convex-optimization trajectories | Motion state → controls | n/r | Path constraints in the reward | n/r | Real-time single-step planning; "significant improvement in terminal accuracy" and better generalization (comparison basis not in the abstract) | A |
| Xu, Han, Gao, Jing [@xu2025grukddpg] | 2025 · IFAC-PapersOnLine | X-34 RLV, descent and glide | 3-DOF | GRU-DDPG | Longitudinal module → bank magnitude; lateral module → bank sign | Disturbances | n/r | n/r | Better accuracy and robustness than DDPG | A |
| Liu et al. [@liu2025nfz; @liu2026timecontrol] | 2026 · Adv. Space Res.; 2026 · J. Zhejiang Univ. Sci. A | HGV entry with no-fly zones; time-controlled entry | 3-DOF | DRL around a reference flight profile, with a DNN time-to-go corrector | Heading-error-corridor threshold and observation corrected online | n/r | No-fly zones; arrival time | n/r | Robust in simulation across corridor widths; precise terminal-time control | A / T |
| Zhou, Wang, Li, Cui [@zhou2025rlv] | 2025 · IEEE TAES | RLV emergency return | 3-DOF, analytic models | Maximum-entropy RL with adaptive noise | Agent selects return profile and analytic model, then computes the 3-D trajectory | Varying return states | Load and thermal | n/r | Online planning across return scenarios | A |
| Gao, Zhou, Chen [@gao2024nfz] | 2024 · Proc. IMechE G | HGV, no-fly-zone avoidance stages | 3-DOF | RL for avoidance; predictor–corrector elsewhere; supervised net for reference commands | → continuous bank | "Vast initial state spaces" | No-fly zones | n/r | Computational efficiency, command stability, robustness | A |
| Li, Wang, Zhou [@li2024mbrl] | 2024 · AST | Entry with spatial no-fly zones | 3-DOF | Model-based RL inside a predictor–corrector | n/r | n/r | No-fly zones | n/r | Online computation ~80 % faster than a conventional predictor–corrector | A |
| Bao, Li, Xu, Tang, Yao [@bao2025morphing] | 2025 · Aerospace | Hypersonic morphing vehicle to the TAEM interface | 3-DOF | A* waypoints + **DDPG for sweep angle only**; analytic QEGC guidance with sliding mode | State → sweep angle | 3σ: H₀ 100 m, λ₀/φ₀ 0.5°, V₀ 100 m/s, γ₀ 0.5°, ψ₀ 0.5°, ρ 20 %, C_L/C_D 10 % (Gaussian) | Path constraints mapped to lift-coefficient bounds | 1,000 Monte Carlo runs | Range error 0.263 ± 0.184 km; 0.1 ms inference (122 kB model) | F |
| Wang, An, Li, Wang, Liu [@wang2025td3] | 2025 · Aerospace | RLV entry | 3-DOF | TD3 (three policy networks) | State → **drag-acceleration profile parameters**, re-planned periodically; bank from tracking; sign from reversal logic | Random ICs, aerodynamic deviation | Q̇ ≤ 3 MW/m², n ≤ 3, q̄ ≤ 50 kPa via the drag corridor | Three guidance tasks; no Monte Carlo campaign found in the text | < 0.01 s per command (~10² faster than baselines) | F |
| Cadamuro, Marchetti, Redondo Gutiérrez, Seelbinder [@cadamuro2026refex] | 2026 · JGCD | DLR ReFEx winged re-entry flight experiment | **6-DOF** high-fidelity GNC validation simulator | DRL (fully connected net) vs genetic programming | → real-time **corrections to precomputed reference commands** | External disturbances, model inaccuracies | n/r | n/r | Comparable to the mission baseline guidance and faster to execute. DRL commands are non-smooth; GP commands are smooth and readable | A |
| Jiang, Li, Furfaro [@jiang2019mars] | 2019 · Acta Astronaut. | Mars entry + powered descent, mid-lift high mass | 3-DOF | **RL selects the entry/powered-descent handover only**; hp-pseudospectral for both phases | — | Nominal and uncertainties | — | n/r | Integrated optimal guidance beats a successive-convexification comparison | A |
| Ghiugan, Staber, Lee [@ghiugan2026uranus] | 2026 · AIAA SciTech | Uranus aerocapture, bank modulation | n/r | Feedforward NN, GAM and RL vs numerical predictor–corrector | → bank | Dispersed conditions | Capture targets | n/r | Numerical predictor–corrector more accurate; ML much faster; RL less accurate in edge cases | A |
| Falcone & Putnam [@falcone2023aerobraking] | 2023 · IEEE TAES | Mars aerobraking campaign decisions | — | Parallel **domain-randomized** DRL | → aerobraking maneuver decisions | Large atmospheric variability | Thermal limits | n/r | Avoids thermal violations; parallel randomized training | A |
| Das, Wang, Niu [@das2023reentry; @das2024reentry] | 2023 · J. Phys. Conf. Ser.; 2024 · IET Conf. Proc. | CAV-H entry | 3-DOF, simplified Earth | PPO vs TD3 | → AoA and bank | — | Boundary and process constraints in the reward | Nominal trajectory | Nominal trajectories meeting mission requirements | A |
| Fabisch et al. [@fabisch2026attitude] | 2026 · arXiv | Spacecraft re-entry **attitude control** (AoA tracking with flaps) | 6-DOF attitude | Off-policy RL, plus hybrid RL + gain-scheduled PID | → actuator commands | Dynamics randomization: mass, inertia, flap actuator bandwidth | n/r | n/r | RL matches PID in distribution, but **out-of-distribution generalization is insufficient** without dynamics randomization; the hybrid is most robust | A |
| Wallace & Si [@wallace2024hsv] | 2024 · JGCD | Hypersonic vehicle **control** (unstable, non-minimum phase) | Longitudinal | Model-based continuous-time integral RL | — | Varying modeling errors and ICs | — | "Substantial and systematic numerical evaluations" | Convergence, optimality and **closed-loop stability guarantees**; better than LQR and feedback linearization | A |
| Song et al. [@song2022igc] | 2022 · Mathematics | HSV terminal guidance, integrated G&C | n/r | PPO | → control commands | Uncertain parameters, actuator faults | n/r | n/r | Feasible fault-tolerant IGC | A |

*No journal paper was found that trains an end-to-end RL bank-angle policy for Mars entry and evaluates it against MSL-class guidance. The Mars entries found use RL for the handover [@jiang2019mars], or are supervised or adaptive-NN tracking laws (§2.3). For aerocapture and aerobraking, the RL work found is one conference paper and one decision-level paper.*

### 1.2 Adjacent work whose methods transfer (powered descent and landing, missiles, other GNC)

| Work | Year · venue | Problem | Method relevant here | Uncertainty / evaluation | Takeaway for this project | Src |
|---|---|---|---|---|---|---|
| Gaudet, Linares, Furfaro [@gaudet2020adaptive; @gaudet2019adaptive_arxiv] | 2020 · Acta Astronaut. | Mars powered descent; asteroid landing | PPO; GRU policy and value function; recurrent vs MLP ablation | Engine failure, mass variation, sensor bias, unknown dynamics; **10,000 test episodes** per experiment | The recurrent policy was clearly better under engine failure, unobserved mass variation and altimeter/LIDAR-only sensing. The MLP was competitive under sensor bias and unknown asteroid dynamics. The authors state there is no guarantee of trajectory optimality | F |
| Gaudet, Linares, Furfaro [@gaudet2020sixdof] | 2020 · Adv. Space Res. | 6-DOF planetary landing | PPO, integrated G&C | Dispersions | 6-DOF integrated G&C is feasible with RL | T |
| Furfaro et al. [@furfaro2020zemzev] | 2020 · Acta Astronaut. | Planetary landing | RL adapts the gains of an analytic ZEM/ZEV law | — | Keeps a structured, analyzable law inside the loop | T |
| Federici & Furfaro [@federici2024meta] | 2025 · Neural Comput. Appl. | Planetary landing | Systematic comparison of meta-learning (recurrent) vs standard RL | — | Direct template for an MLP-vs-recurrent ablation | T |
| Carradori, Sagliano, Mooij [@carradori2026transformer; @carradori2025scitech] | 2026 · JGCD | Atmospheric booster landing | Meta-RL with Gated Transformer-XL; 6-DOF training and test | Atmospheric and aerodynamic uncertainty, navigation and control errors, dispersed ICs; perturbed Monte Carlo | Up to 99.7 % of runs meet all terminal requirements with a terminal patch; without it, vertical-velocity violations remain | A |
| Federici, Benedikter, Furfaro [@federici2025rlmpc] | 2026 · JGCD | Lunar pinpoint landing | RL learns the MPC cost; MPC enforces constraints | Trained in a perturbed environment and tested in a harsher one | Hybrid keeps constraint guarantees and is tested out of distribution | A |
| Scorsoglio et al. [@scorsoglio2022lunar; @scorsoglio2025lunar] | 2022 · JSR; 2025 · Neural Comput. Appl. | Lunar landing, image-based meta-RL GNC | Recurrent meta-RL | — | Integrated navigation + guidance with RL | T |
| Iafrate et al. [@iafrate2025propulsive] | 2025 · Acta Astronaut. | First-stage propulsive landing | DRL | — | Re-entry-adjacent booster landing | T |
| Zavoli & Federici [@zavoli2021robust]; Federici & Zavoli [@federici2023robust] | 2021 · JGCD; 2024 · Acta Astronaut. | Robust low-thrust transfer | PPO; meta-RL (2024) | Process noise, **observation noise**, control errors, missed thrust | Shows the uncertainty taxonomy reviewers recognize (process / observation / actuation) | A |
| Zavoli & Federici [@zavoli2026curriculum] | 2026 · JGCD (technical note) | Fuel-optimal G&C with constraints | Curriculum RL with constraints | — | Recent JGCD precedent for curriculum + constraints | T |
| Boone et al. [@boone2022obs] | 2022 · AIAA SciTech | CR3BP transfer | Train with observation noise; append uncertainty magnitude to the observation | Navigation uncertainty | Both measures increased robustness to navigation errors | A |
| Hovell & Ulrich [@hovell2021prox] | 2021 · JSR | Proximity operations | RL at the guidance level only | — | Pattern: learn guidance, keep a classical inner loop | T |
| Holt, Origer, Izzo [@holt2026bc]; Izzo et al. [@izzo2024optimality] | 2026 · JGCD; 2024 · Sci. Robot. | G&C networks | Behavioral cloning vs RL | — | Reviewers may ask why not imitation of the SCvx optimum | A |
| Gaudet, Furfaro, Linares [@gaudet2020angleonly]; Gaudet et al. [@gaudet2020exo] | 2020 · AST; 2021 · JSR | Interceptor guidance | Meta-RL with a parasitic attitude loop | Seeker and attitude-loop effects | Model the attitude loop inside the RL environment | T |
| He, Shin, Tsourdos [@he2021computational]; Lee et al. [@lee2023impactangle]; Qiu et al. [@qiu2023recorded] | 2021 · JAIS; 2023 · JGCD; 2024 · Def. Technol. | Missile homing, impact angle, recurrent RL guidance | — | — | Terminal-angle-constrained RL guidance (cf. the FPA requirement) | T |
| Yan et al. [@yan2024evasion] | 2024 · IJASS | Exo-atmospheric evasion | **Constrained RL** with a total energy limit | — | One of the few aerospace guidance papers using constrained-RL machinery | T |
| Chaudhary, Armellin, Holt, Sagliano [@chaudhary2026chance] | 2026 · AST | Robust trajectory optimization | Chance-constrained RL, distribution-agnostic | — | Probabilistic constraint formulation in aerospace RL | T |
| Gaudet & Furfaro [@gaudet2023igc] | 2023 · AIAA SciTech | Interceptor vs hypersonic threat | Fully vs partially integrated G&C, meta-RL | Aero perturbation, flexible body dynamics | Robustness to flexible-body dynamics shown by test | A |
| Dally & van Kampen [@dally2022sac] | 2022 · AIAA SciTech | Fault-tolerant flight control | SAC | Six failure cases, biased sensor noise, atmospheric disturbances | Robustness demonstrated by test cases, not margins | A |

### 1.3 This project in the same format

| Work | Status | Vehicle / phase | DOF | Algorithm | Observation → action | Uncertainty | Constraints | Evaluation | Results (v24b 30M) |
|---|---|---|---|---|---|---|---|---|---|
| This work | v24b done; v25 running | WB001 winged body, full entry 100 → 25 km, ~7,800 km downrange, entry heading north, target 70°N/12°E heading east | 3-DOF, rotating oblate Earth, ideal attitude loop with rate limits | PPO [@schulman2017ppo] (SB3), MLP 512×512 | 31 features incl. energy-to-go and running constraint peaks → **absolute bank/AoA targets** on a √(energy-to-go) decision grid | VISTA set (IC, TC, mass, C_L, C_D, ρ) at 0.1× in training; v25 curriculum up to 1×. No navigation noise, winds, actuator lag or attitude dynamics | Q̇ 1.5 MW/m², q̄ 18 kPa, n 2.5 g via **PID-Lagrangian** + incremental peak charge; terminal position (1 km), γ = −10°, ψ = 90° via ε-curriculum | 500 (0.1×) and 1,001 (1×) cases; same-physics SCvx closed-loop G&C baseline | 0.1×: median 1.36 km, 24 % ≤ 1 km, 79 % strictly feasible, \|Δγ\| 1.16°, \|Δψ\| 2.77°. 1×: median 10.5 km vs SCvx 0.21 km; feasible 34 % vs 39 % |

---

## 2. RL guidance for entry: what the literature does

Broader surveys of machine learning and RL in aerospace guidance and control: [@chai2021review; @izzo2019survey; @shirobokov2021survey; @tipaldi2022rl]. The main non-RL learning alternative is a network trained by supervision on optimal or numerically predicted trajectories [@sanchez2018realtime; @shi2021onboard; @cheng2020dnn]. Its RL counterpart is compared in [@holt2026bc].

### 2.1 End-to-end policies

Only the Gaudet line flies an end-to-end RL policy over a mission comparable to ours [@gaudet2022approach; @gaudet2021approach_arxiv]: 100 km / 7.45 km/s to 25 km over ~8,000 km. Its terminal state is position, altitude and **speed**, not flight-path angle and heading. Its dispersions are small (±3 % uniform on C_L, C_D and density in training, tested to ±10 %). The guidance runs at 0.2 Hz with AoA frozen at 40° above 4.57 km/s. The paper is still the benchmark reviewers will hold this project against. It has:

- a recurrent policy;
- actuator lag, noise and failure in training;
- 5,000-run tests per case, several outside the training distribution;
- an explicit generalization table;
- an LQR-tracking baseline.

Other end-to-end policies are evaluated less demandingly:

- Su et al. [@su2023irl] count a run as successful at a range-angle error below 0.25°, which is roughly 28 km on Earth.
- Jiang et al. [@jiang2023her] stop the glide phase at 300 km from the target.
- Peng et al. [@peng2023modular; @peng2025hybrid; @peng2025hrl] report capability over wide reachable areas rather than precision statistics, at least in their abstracts.
- Das et al. [@das2023reentry; @das2024reentry] show nominal trajectories.

**Design patterns that recur and bear on our choices:**

- **Bank sign versus magnitude.** A hybrid discrete-continuous action [@peng2025hybrid] or separate magnitude and sign modules [@xu2025grukddpg] are the common ways to avoid exploration problems with bank reversals. Our absolute-target formulation outputs a signed bank directly, as Jiang et al. [@jiang2023her] do with a single bank output that encodes both.
- **Independent variable.** Peng et al. [@peng2023modular] use a state-related independent variable to resolve step-size conflicts. Our √(energy-to-go) grid is the same idea pushed further, and is the closest precedent to cite.
- **Rates versus attitudes.** Gaudet commands rates. In this project, rate commands let AoA freeze at its bound: v21 found the saturated integrator gave the pinned command zero effect and zero gradient. Absolute targets make the 40° hold interior to the policy output. The literature offers no comparison, so the paper should present this as a finding with its ablation.

### 2.2 RL embedded in a classical architecture

Most recent entry-RL papers keep an analytic or numerical guidance core and use RL for one decision:

- no-fly-zone avoidance [@gao2024nfz; @li2024mbrl; @liu2025nfz];
- morphing [@bao2025morphing];
- drag-profile parameters [@wang2025td3];
- time-to-go correction [@liu2026timecontrol];
- profile and model selection [@zhou2025rlv];
- corrections to a reference [@cadamuro2026refex].

That design inherits the classical law's structure, and with it most of its robustness evidence. It is the comparison reviewers will raise against an end-to-end policy.

Cadamuro et al. [@cadamuro2026refex] matter most here. Theirs is the only JGCD-level entry-RL paper found that validates in a **6-DOF mission GNC simulator** (DLR ReFEx). They also report that DRL commands are **non-smooth** compared with genetic programming. Smoothness is therefore an observed weakness that a "stable guidance" paper must address with data.

### 2.3 Mars entry, aerocapture and aerobraking

- **Mars entry.** RL has been used for the entry-to-powered-descent handover [@jiang2019mars], not for bank guidance. Classical baselines in this area are MSL's entry terminal point controller [@mendeck2014msl], stochastic covariance-steering guidance compared against the Apollo final-phase law [@ridderhof2022stochastic], and adaptive Tianwen-1 guidance [@guo2022tianwen]. Learned components in recent entry work are supervised: Mars entry navigation with NN density adaptation [@giraldo2024navigation], deep-learning density estimation for entry guidance [@rataczak2025density] and NN-adaptive entry guidance [@giraldo2026adaptive].
- **Aerocapture.** Only one comparison of RL against a numerical predictor–corrector was found [@ghiugan2026uranus]. It concludes that the predictor–corrector stays more accurate and RL is faster but weaker in edge cases.
- **Aerobraking.** A domain-randomized DRL planner has been demonstrated [@falcone2023aerobraking].

*Implication:* a robustness result on a lifting entry vehicle with a full terminal state has no direct Mars or aerocapture counterpart. Mention it only as transfer potential, not as positioning.

### 2.4 Control-level RL during re-entry

Several papers apply RL to the **attitude/control** layer:

- re-entry attitude control against a gain-scheduled PID [@fabisch2026attitude];
- continuous-time integral RL with stability guarantees for hypersonic vehicle control [@wallace2024hsv];
- PPO integrated G&C under faults [@song2022igc];
- an actor-critic adaptive spiral-diving law with a Lyapunov proof [@wu2024spiral].

These are relevant because they show where stability proofs are tractable: low-order tracking dynamics with adaptive-dynamic-programming structure. That is unlike a long-horizon guidance policy.

### 2.5 Adjacent work: powered descent and missiles

Most robustness machinery transfers from landing. Recurrent meta-RL comes from Gaudet and Federici [@gaudet2020adaptive; @federici2024meta]. Transformer meta-RL under 6-DOF Monte Carlo comes from Carradori [@carradori2026transformer]. RL-tuned MPC that keeps constraint guarantees comes from Federici [@federici2025rlmpc]. Adversarial training is newer [@leiva2026adversarial]. Missile work adds terminal-angle constraints [@lee2023impactangle], seeker and attitude-loop modeling [@gaudet2020exo] and recurrent policies [@qiu2023recorded].

---

## 3. Robustness under uncertainty

### 3.1 Domain randomization and curricula

- **Origins and reviews.** Domain randomization (DR) originates in robotics [@tobin2017dr; @peng2018dynrand] and is reviewed in [@muratore2022review; @zhao2020sim2real].
- **Adaptive and automatic DR.** Automatic DR expands the randomization range as performance allows [@openai2019rubik]. Active DR learns which parameter regions to sample [@mehta2020adr]. Both sit in the broader automatic-curriculum literature [@portelas2020acl; @narvekar2020curriculum].
- **Aerospace use.** Uniform DR is used by Gaudet [@gaudet2022approach], Jiang [@jiang2023her] and Bao [@bao2025morphing]. Parallel randomized training is used by Falcone & Putnam [@falcone2023aerobraking], and dynamics randomization to recover out-of-distribution performance by Fabisch et al. [@fabisch2026attitude]. Automatic-curriculum RL has just been applied to hypersonic trajectory planning [@liu2026pqac] (title and metadata only, paywalled).

*Implication.* v25's ADR-style schedule (boundary 0.2 → 1.0 VISTA, advanced on success thresholds) matches [@openai2019rubik]. Its novelty for entry is the combination with a CMDP and with physically infeasible cases at high scale. Reviewers will ask:

- whether the curriculum beats uniform DR at the final scale (ablation);
- how the curriculum treats cases no guidance could save (§7).

### 3.2 Robust and adversarial RL

- **Classical robust RL** optimizes against a disturbing adversary [@morimoto2005robust; @pinto2017rarl] or against action perturbations [@tessler2019action].
- **Risk-averse model ensembles** train on the worst CVaR fraction of sampled models [@rajeswaran2017epopt].
- **Review** of the area: [@moos2022robust].
- **Observation-adversarial robustness** (state-adversarial MDPs) [@zhang2020samdp] is the formal counterpart of navigation-error robustness.
- **Aerospace.** Adversarial RL for planetary landing appeared in October 2026 [@leiva2026adversarial]; only the title and metadata were checked. Worst-case search with RL has been used to falsify a flight-control law [@braun2023worstcase].

*Implication.* An EPOpt-style tail objective, or adversarial dispersion sampling, would target exactly the 1× tail where v24b fails. The curriculum's boundary-heavy sampling (50 % at s_k) is a mild version of it.

### 3.3 Meta-RL and recurrent policies

Under randomized dynamics the problem is partially observable. The policy must infer C_L, C_D and density from the response, which a memoryless MLP can do only through observed proxies. Our measured drag and L/D features are such proxies.

- RL² [@duan2016rl2] and the aerospace meta-RL line use recurrent layers [@gaudet2020adaptive; @gaudet2022approach; @federici2022rendezvous; @federici2023robust; @federici2024meta; @scorsoglio2022lunar]. Carradori et al. use a transformer [@carradori2026transformer]; Xu et al. use GRU-DDPG [@xu2025grukddpg].
- Robotics adds teacher–student distillation and rapid motor adaptation, which estimate latent dynamics parameters online [@lee2020quadruped; @kumar2021rma].
- Gaudet's own ablation [@gaudet2019adaptive_arxiv] shows the recurrent policy winning clearly under engine failure and unobserved mass, while a non-recurrent MLP was competitive under sensor bias.

*Implication.* An MLP-vs-recurrent comparison at 1× VISTA is the experiment reviewers in this community expect [@gaudet2020adaptive; @federici2024meta]. A teacher policy with privileged access to C_L/C_D/ρ is a cheap upper bound: it shows how much of the 1× gap is information rather than capacity.

### 3.4 Navigation and sensor errors

Gaudet models sensor scale-factor error in test only and still finds high success [@gaudet2022approach]. The strike paper trains with it [@gaudet2023strike]. The LQR tracker in Gaudet's comparison was "so poor" under 1 % scale-factor noise that its results were omitted [@gaudet2021approach_arxiv]. Other references:

- observation-noise training and uncertainty-augmented observations [@boone2022obs];
- observation noise as one of three uncertainty classes [@zavoli2021robust];
- Mars entry navigation with NN density adaptation (supervised) [@giraldo2024navigation].

*Implication.* This project has **no navigation error at all**. For an aerospace venue a robustness claim without navigation error will be challenged. Minimum: inject Gaussian errors plus bias on position, velocity and measured drag in test. Better: train with them.

### 3.5 Actuator and attitude dynamics

The relevant precedents:

- actuator lag of 1 s and 0.1 s, rate limits, noise and failure [@gaudet2022approach; @gaudet2023strike];
- flap actuator bandwidth randomization [@fabisch2026attitude];
- a parasitic attitude loop inside the RL environment [@gaudet2020exo];
- flexible-body dynamics [@gaudet2023igc];
- 6-DOF validation [@cadamuro2026refex; @carradori2026transformer].

The project's SCvx baseline is a full closed-loop G&C, whereas the RL runs with an ideal attitude loop limited only by rate (RUNS.md, 2026-10-08). That asymmetry favors RL, and a reviewer will notice it.

*Implication.* Add a first-order attitude-response lag and a pure command delay, and sweep them (§5.5). State clearly which loop each method closes.

### 3.6 Out-of-distribution evaluation

Precedents for testing outside the training distribution:

- Gaudet tests explicitly outside training (±5/10 % aero, mass, diverts, shorter range) and reports graceful degradation [@gaudet2022approach].
- Federici et al. test in a harsher environment than training [@federici2025rlmpc].
- FNPEG was verified under dispersions "significantly higher than the design level" [@lu2017verification].
- Fabisch et al. find RL out-of-distribution generalization insufficient without randomization [@fabisch2026attitude].
- Jiang et al. state that their policy must be retrained beyond the training range [@jiang2023her].
- The ML-side taxonomy of zero-shot generalization is in [@kirk2023survey].

*Implication.* The S4_scale_x{0.1 … 1.25} sets with common random numbers are the right instrument. Report every metric as a curve against dispersion scale, for RL and SCvx on identical cases, including the 1.25× point beyond the training maximum. That curve is the core robustness figure.

---

## 4. Stability and verification of neural-network guidance

### 4.0 What "stable" has to mean in an entry-guidance paper

Entry guidance is a finite-horizon, time-varying (energy-indexed) problem with a terminal set, not regulation to an equilibrium. A reviewer will ask which of the following is meant. The paper should pick explicitly and give evidence for each one it claims.

1. **Closed-loop robust reachability.** For all dispersions in a set 𝒲, the trajectory stays inside the path-constraint set and reaches a terminal tolerance set. This is the guidance analogue of robust stability. It can be supported statistically (§4.6) or, for small subproblems, formally (§4.4).
2. **Incremental stability / bounded error growth.** Small state perturbations at any energy level produce bounded changes in the terminal error. This can be measured by finite-horizon sensitivity: the policy Jacobian and the linearized closed-loop transition along trajectories. Contraction-based certificates formalize it [@dawson2023certificates].
3. **Robustness margins.** How far a gain, delay or bias on actuation or sensing can be pushed before the property in point 1 fails. This is the classical-control analogue of gain, phase and delay margins (§4.5).
4. **Smooth, non-chattering commands.** Bounded command rates, a limited number of bank reversals, no limit cycles. Reviewers sometimes call this "stable" guidance [@cadamuro2026refex; @mysore2021caps].
5. **Training stability and reproducibility.** Consistent results across seeds. This is an algorithmic property, not a closed-loop one [@henderson2018matters; @agarwal2021precipice].

The JGCD editorial sets the tone [@lu2024editorial]:

- Simply applying ML to a GN&C problem is no longer novel.
- Authors must show why the approach is essential or superior to existing methods, and why it works.
- Authors must say when it fails, how failure can be detected, and what the consequences are.
- The editorial says this is "not nitpicking" because GN&C systems are mission critical.

### 4.1 What aerospace RL guidance papers actually report

In the entry/HGV papers reviewed (§1.1), robustness evidence is **Monte Carlo only**. None reports margins, Lyapunov certificates or formal verification. Gaudet states plainly that there is no guarantee of optimality for trajectories induced by the policy [@gaudet2019adaptive_arxiv].

Stability proofs appear only where the learned part is a low-order adaptive law:

- adaptive actor-critic tracking with a Lyapunov proof [@wu2024spiral];
- continuous-time integral RL control with closed-loop stability guarantees [@wallace2024hsv];
- RL tuning a Lyapunov-based Q-law, so stability holds by construction [@holt2021qlaw].

The most directly usable tool for this project is Scorsoglio, D'Ambrosio & Furfaro [@scorsoglio2026sindy]. They use sparse identification (SINDy) of the closed-loop dynamics induced by a trained RL policy to get an analytic surrogate. They then take a candidate Lyapunov function from the cost-to-go and check Lyapunov conditions over a bounded domain with a **PAC (probably approximately correct) bound** on the finite-sample verification. Demonstrations cover spacecraft attitude and asteroid hovering. It is the first JGCD paper found that gives an RL G&C policy a statistical stability certificate.

### 4.2 Lyapunov-based RL and learned certificates

- Safe model-based RL with stability guarantees, which expands a verified region of attraction [@berkenkamp2017safe].
- Lyapunov-constrained policy updates for CMDPs [@chow2018lyapunov].
- Jointly learned neural Lyapunov functions and controllers [@chang2019neural].
- Actor-critic with a Lyapunov critic and a stability guarantee [@han2020actorcritic].
- A control-theoretic stability certificate for RL policies [@jin2020stability].
- Survey of neural Lyapunov, barrier and contraction certificates [@dawson2023certificates].

These target equilibria or regions of attraction. For entry, the contraction and finite-horizon variants are the relevant ones (§4.0, points 1–2).

### 4.3 Control barrier functions, shields and run-time assurance

- **Theory.** CBF quadratic programs [@ames2017cbf; @ames2019cbf].
- **RL variants.** CBF-guided RL [@cheng2019rlcbf], shields [@alshiekh2018shield] and safety layers [@dalal2018safety].
- **Run-time assurance (RTA)** tutorial [@hobbs2023rta].
- **Aerospace RL with RTA or CBFs:** discrete CBFs for spacecraft inspection [@vanwijk2024cbf]; RTA for 6-DOF inspection [@dunlap2024rta]; adaptive DR with relaxed CBFs [@tammam2025safe]; space-processor timing of RL + RTA [@hamilton2025processor].
- **Hypersonic flight.** A viability-based action shield for longitudinal flight [@rastgoftar2026viability] (title and metadata only).

*Implication.* A q̄/Q̇ corridor filter on the bank/AoA targets is low-dimensional and would answer the editorial's question of what happens when the policy is wrong. It can be presented as an optional monitor so the "pure RL" result stays intact.

### 4.4 Reachability and formal verification of NN controllers

- **Tools:** Reluplex [@katz2017reluplex], Verisig [@ivanov2019verisig], NNV [@tran2020nnv], ReachNN [@huang2019reachnn], closed-loop reachability [@everett2021reach] and SDP-based robustness analysis [@fazlyab2022sdp].
- **Aerospace landmark.** The ACAS Xu line in JGCD: NN compression [@julian2019acas] and closed-loop reachability [@julian2021reach].
- **Probabilistic guarantees** for NN controllers [@katz2021probabilistic].

All of these scale to small networks and short horizons. A 512×512 policy with 31 inputs, ~800 decisions and nonlinear entry dynamics is far beyond exact closed-loop reachability. Reviewers will not expect formal verification of the full mission. They may expect the paper to say why it is not done and what replaces it (§4.6). A tractable formal piece would be a one-step output-range or Lipschitz certificate of the policy over the operational observation box.

### 4.5 Robustness margins for learned controllers

- Stability analysis of NN-controlled loops with integral quadratic constraints [@yin2022iqc].
- Training NN controllers with certified **disk margins**, for LTI plants with IQC-described uncertainty [@junnarkar2025margins].
- Lipschitz-bounded training [@pauli2022lipschitz; @barbara2025lipschitz].
- Smoothness regularization (CAPS) [@mysore2021caps].

None has been applied to entry guidance. They apply directly only to linearized subproblems, such as the lateral heading loop linearized around a nominal arc.

*Implication.* Use a "guidance margin" defined by experiment. Multiply a loop element (bank or AoA command gain, command delay, actuator lag, density or C_L scale) by a factor k and find the k at which the success probability of §4.0 point 1 drops below a threshold. Report it with confidence bounds. This is the Monte Carlo analogue of gain and delay margins. The literature's closest precedents are Gaudet's out-of-distribution table [@gaudet2022approach] and Lu's beyond-design-level verification [@lu2017verification].

### 4.6 Statistical verification and falsification

- **Sample-size logic** for probabilistic requirements with a stated consumer risk [@hanson2012mc; @hanson2010mc], with exact binomial bounds [@clopper1934binomial].
- **Scenario-approach guarantees** [@calafiore2006scenario; @campi2009scenario].
- **PAC-bounded Lyapunov checks** [@scorsoglio2026sindy].
- **Black-box falsification:** survey [@corso2021survey], adaptive stress testing [@lee2020ast], and RL-based worst-case search applied in JGCD [@braun2023worstcase].

*Implication.* The defensible "stability under uncertainty" evidence for this paper is:

- (i) a statistically sized Monte Carlo over the scale sweep with confidence bounds;
- (ii) experimental margins (§4.5);
- (iii) a falsification search that finds the worst dispersion vectors, for RL and SCvx alike;
- (iv) optionally, a PAC-type local certificate on a reduced model [@scorsoglio2026sindy].

---

## 5. Constrained RL for path constraints

### 5.1 How entry-RL papers handle path constraints

| Handling | Papers |
|---|---|
| Episode termination on violation | Gaudet [@gaudet2022approach] (training only; violations monitored in test), Su [@su2023irl], Jiang [@jiang2023her] |
| Reward penalty or barrier near the boundary | Su [@su2023irl], Bao 2023 [@bao2023onboard], Das [@das2023reentry] |
| Constraints mapped to action bounds (H–V corridor, lift-coefficient bounds, drag corridor) | Jiang [@jiang2023her], Bao 2025 [@bao2025morphing], Wang [@wang2025td3] |
| Classical law keeps the constraints; RL handles another decision | Gao [@gao2024nfz], Li [@li2024mbrl], Cadamuro [@cadamuro2026refex] |
| Constrained-RL machinery (CMDP, multipliers, chance constraints) | **None found in entry guidance.** In aerospace: Yan [@yan2024evasion] (exo-atmospheric), Chaudhary [@chaudhary2026chance], Zavoli & Federici [@zavoli2026curriculum], and the RL-learned MPC cost of Federici [@federici2025rlmpc] |

### 5.2 Constrained RL methods

- **Theory and benchmarks:** CMDPs [@altman1999cmdp]; CPO [@achiam2017cpo]; reward-constrained PPO [@tessler2019rcpo]; projection-based CPO [@yang2020pcpo]; penalized PPO (P3O) [@zhang2022p3o]; PPO-Lagrangian baselines in Safety Gym [@ray2019safetygym].
- **PID-Lagrangian** [@stooke2020pid], which this project uses: proportional and derivative terms damp the oscillation and overshoot of plain Lagrangian methods.
- **Tail and chance constraints:** CVaR or chance constraints in percentile-risk RL [@chow2018risk].
- **Surveys:** [@garcia2015survey; @gu2024review; @brunke2022safe].

### 5.3 Positioning

PID-Lagrangian multipliers with an **incremental (Markov) peak charge** have not been applied to entry guidance in the literature found. They are a defensible methodological contribution.

The v24 post-mortem showed a terminal-only peak charge to be non-Markov; the incremental charge telescopes to the same return. That argument is specific to peak-type constraints, which entry has (Q̇, q̄, n), and is worth stating formally. It relates to the CMDP literature's treatment of non-additive constraints. The constraint as implemented, E[max(0, peak/limit − 1)] ≤ budget, is an expectation constraint. Reviewers in aerospace think in probabilities: P(violation) ≤ p. Either report the chance-constraint view alongside, citing the percentile-risk formulation [@chow2018risk], or justify the expectation form.

### 5.4 Reporting expectations for constraints

- Per-constraint peak-ratio distributions with tails (p95, p99, max), not just "feasible yes/no".
- The probability of violation with a binomial confidence interval [@clopper1934binomial].
- Multiplier histories.
- An ablation against the fixed penalty used in v20–v22.

Gaudet reports per-constraint mean, σ and max against limits [@gaudet2022approach]. Jiang and Su plot all 1,000 runs against the corridor [@jiang2023her; @su2023irl].

---

## 6. Evaluation practice in the aerospace literature

### 6.1 Monte Carlo sizes and statistics

| Source | Runs per evaluated condition | Statistic reported |
|---|---|---|
| Gaudet et al. [@gaudet2022approach; @gaudet2023strike] | 5,000 per test case (≈10 cases) | Mean, σ, max of miss / altitude / speed error; success %; violation % |
| Gaudet et al. [@gaudet2020adaptive] | 10,000 per experiment | Mean, σ, max |
| Jiang et al. [@jiang2023her]; Su et al. [@su2023irl]; Bao et al. [@bao2025morphing] | 1,000 | Mean / min / max; success %; mean ± σ |
| Lu et al. (FNPEG, Orion high-fidelity simulator) [@lu2017verification] | Extensive testing vs PredGuid, dispersions beyond design level | Verification metrics defined in the paper (abstract read) |
| Ridderhof et al. [@ridderhof2022stochastic] | Monte Carlo | **1st and 99th percentile** range errors vs Apollo final phase |
| This project | 500 (0.1×), 1,001 (1×), 300 per scale (S4) | Median, IQM with bootstrap CI, p95, fractions within thresholds |

**How many runs are enough.** Hanson & Beard derive the run count from binomial failure analysis with a stated consumer risk (CR). NASA verification requirements typically use CR ≤ 10 %, i.e. 90 % confidence [@hanson2010mc; @hanson2012mc]. Their Table 1 lists:

- for an allowed failure rate of 0.27 % (a "3σ" requirement): N = 852 with 0 failures, 1,440 with 1, 1,970 with 2, 2,473 with 3;
- for 0.135 %: N = 1,705 with 0 failures, 2,880 with 1, 3,941 with 2.

Applied to this project (Clopper–Pearson [@clopper1934binomial], computed for this review):

- 500 runs with no failure bound the failure rate only below 0.46 % at 90 % confidence; 1,001 runs with no failure, below 0.23 %.
- 79 % strictly feasible on 500 cases has a 95 % interval of [75.2 %, 82.5 %]; 24 % within 1 km has [20.3 %, 28.0 %].
- No 3σ-type claim is possible yet. A "≥ 99.73 % success at 90 % confidence" claim needs ≥ 852 runs with zero failures, or ≥ 1,970 runs with at most two.

### 6.2 Dispersion tables

The papers that state dispersions:

- Gaudet: uniform ±3 % C_L/C_D/ρ, actuator lag, noise and failure parameters, IC ranges [@gaudet2022approach].
- Jiang: uniform 0–10 % C_L/C_D, IC ranges [@jiang2023her].
- Bao: Gaussian 3σ values for IC, density 20 % and aero 10 % [@bao2025morphing].

Hanson & Beard discuss input-distribution choices and the treatment of uniformly distributed parameters [@hanson2010mc]. The VISTA 1× set used here is larger than any of them: density σ 25 % and C_L/C_D σ 20 % (RUNS.md, robustness-phase header). That is a strength to state with a full dispersion table: distribution, σ or bounds, correlation, and whether each parameter is constant per run or altitude-dependent. Note that the RL papers read in full apply density as a single per-run scale factor, as this project does. Altitude-correlated density perturbations are the realistic case, and their absence will be noticed. Gaudet perturbs density by a scale only [@gaudet2022approach]; Ridderhof models altitude-dependent random variations [@ridderhof2022stochastic].

### 6.3 Reported metrics

**Common:**

- terminal miss (mean / σ / max, or percentiles);
- other terminal-state errors (altitude, speed, time);
- success rate against a stated tolerance;
- per-constraint peak statistics and violation rate;
- heat load against an optimum;
- number of bank reversals [@peng2025hybrid; @jiang2023her];
- CPU time per command.

**Rare but valuable:**

- 1st/99th percentiles [@ridderhof2022stochastic];
- tests outside the training distribution with degradation tables [@gaudet2022approach];
- beyond-design-level dispersions [@lu2017verification].

### 6.4 Baselines

- **Flight heritage.** Apollo final-phase / MSL entry terminal point controller [@mendeck2014msl; @moseley1969apollo]. Shuttle drag-tracking guidance [@harpold1979shuttle; @harpold1983shuttle].
- **Numerical predictor–correctors.** [@lu2008pc; @xue2010constrained; @lu2014unified], skip entry [@brunner2008skip; @brunner2012compare] and the verified FNPEG [@lu2017verification]. NPCG is the baseline Jiang compares against [@jiang2023her], and Cheng et al. use a DNN range predictor inside one [@cheng2020dnn].
- **Convex / SCP.** SOCP entry trajectory optimization [@liu2016socp]; SCP [@wang2017scp]; an online SOCP reference with QCQP tracking each guidance cycle [@wang2018autonomous]; improved SCP [@wang2020improvedscp]; SCvx convergence [@mao2016scvx]; tutorial [@malyuta2022convex].
- **Others used by RL papers:** LQR tracking a GPOPS-II optimum [@gaudet2022approach]; open-loop GPOPS optimality comparison [@su2023irl]; the mission baseline guidance [@cadamuro2026refex].

**Implication.** The same-physics SCvx closed-loop baseline is stronger than most RL papers use. Reviewers will still expect one **flight-heritage-class** law (FNPEG-type predictor–corrector or Apollo/Shuttle-type drag tracking) under the same dispersions. That shows where RL sits between "what flies" and "what optimizes".

### 6.5 Computational cost and onboard feasibility

| Source | Reported cost |
|---|---|
| Su et al. [@su2023irl] | 0.167 ms per command (CPU); GPOPS up to 28 s |
| Bao et al. [@bao2025morphing] | 0.1 ms inference, 122 kB model; 1.6 ms full cycle against a 50 ms budget |
| Jiang et al. [@jiang2023her] | 0.094 s for 9 calls vs 6.43 s for NPCG |
| Wang et al. [@wang2025td3] | < 0.01 s per command |
| Gaudet et al. [@gaudet2022approach] | Policy runs forward "in a few milliseconds" even in 6-DOF (qualitative) |
| Cheng et al. [@cheng2020dnn] | DNN predictor–corrector at 20 Hz |
| Wang & Lu [@wang2020improvedscp] | Improved SCP converges in seconds (MATLAB) |
| Hamilton et al. [@hamilton2025processor] | Timing of RL and RTA policies on space processors |

*Implication.* Report wall-clock per decision for the MLP and the SCvx cycle on the same CPU. If possible, also report it on an embedded-class processor [@hamilton2025processor].

### 6.6 RL-specific statistical practice

Use multiple seeds, with the number justified by a power analysis [@henderson2018matters; @colas2018seeds]. Report IQM with stratified-bootstrap confidence intervals and performance profiles rather than the best run [@agarwal2021precipice]. Separate tuning and evaluation sets [@patterson2024empirical].

The project already reports IQM with bootstrap CIs on fixed sets. Multiple training seeds of the final configuration are still missing.

---

## 7. Gap analysis positioned against this project

| Dimension | Literature (best case) | This project | Gap / claim |
|---|---|---|---|
| Mission scope | Gaudet: 100 → 25 km over ~8,000 km, 3-DOF [@gaudet2022approach]; most others glide phase only | 100 → 25 km, ~7,800 km, rotating oblate Earth | Comparable to the longest; state it |
| Terminal state | Position + altitude + speed [@gaudet2022approach]; distance/height/time [@jiang2023her]; range angle [@su2023irl]. No end-to-end RL with **terminal γ and ψ** found | Position (1 km), γ = −10°, ψ = 90° | **Novel**: full terminal state from an end-to-end policy. Currently ~1° / ~3°, so state precision honestly |
| Decision timing | Fixed-time steps; state-related independent variable [@peng2023modular] | √(energy-to-go) grid, absolute targets | New in this form; needs ablation (time steps, rate actions) |
| Path constraints | Termination, penalties, action bounds (§5.1); no CMDP in entry | PID-Lagrangian + incremental peak charge, three constraints | **Novel** for entry; needs fixed-penalty ablation and P(violation) reporting |
| Uncertainty: model | ±3 % → ±10 % [@gaudet2022approach]; 10 %/20 % 3σ [@bao2025morphing] | VISTA up to 1× (ρ σ 25 %, C_L/C_D σ 20 %) | **Larger than in any RL entry paper reviewed**; only constant-scale density so far |
| Uncertainty: navigation and sensors | Scale factor, noise [@gaudet2022approach; @gaudet2023strike; @boone2022obs] | None | **Gap reviewers will flag** |
| Uncertainty: actuators / attitude | Lag, noise, failure [@gaudet2022approach]; 6-DOF [@cadamuro2026refex; @carradori2026transformer] | Ideal attitude loop with rate limits | **Gap**; also unequal to the SCvx G&C baseline |
| Policy memory | Recurrent meta-RL standard [@gaudet2020adaptive; @federici2024meta; @carradori2026transformer] | MLP with running-peak and drag/L-D features | Ablation expected |
| Curriculum | ADR [@openai2019rubik]; automatic curriculum RL for HGV [@liu2026pqac] | ε-curriculum on terminal angles; ADR-style dispersion curriculum (v25) | Combination is new; needs ablation vs uniform DR |
| Baseline | NPCG [@jiang2023her], LQR tracking [@gaudet2022approach], GPOPS [@su2023irl], mission baseline [@cadamuro2026refex] | Same-physics closed-loop SCvx | **Stronger than typical**; add a heritage law |
| Monte Carlo size | 1,000–10,000 | 300–1,001 | Increase to ≥ 2,000 at the claim level (§6.1) |
| Stability evidence | None in entry RL; PAC-Lyapunov for RL G&C [@scorsoglio2026sindy] | None yet | **Opportunity**: first entry-RL paper with margins, falsification and statistical certification |
| Out-of-distribution behaviour | Degradation tables [@gaudet2022approach] | 1× is currently out of distribution (median 10.5 km vs 0.21 km) | v25 + scale curves (S4) address it directly |

**Claims the paper can make** if the experiments in §8 support them:

1. A single end-to-end RL policy flies the full entry to a full terminal state (position, γ, ψ) inside heat-rate, dynamic-pressure and load limits. No paper found does this.
2. The policy handles peak-type path constraints as a CMDP with PID-Lagrangian multipliers and a Markov incremental charge. No entry paper found uses CMDP methods.
3. The policy degrades gracefully, measured as robustness and margin curves against dispersion scale up to and beyond the training range, against a same-physics SCvx G&C and a heritage law. The comparison separates guidance failures from physically infeasible cases.
4. The policy is stable in the sense defined in §4.0, backed by margins, falsification and statistical certification. No entry paper found provides this.

---

## 8. What reviewers at these venues will expect for a robustness / stability claim

Priority: **M** = must have for JGCD/JSR/TAES; **S** = should have; **C** = could have. The concrete experiment uses this project's tooling where it exists.

| # | Expectation | Set by | Pri. | Concrete experiment for this project |
|---|---|---|---|---|
| R1 | Justify RL over existing methods; explain why it works, how it fails, how failure is detected | [@lu2024editorial] | M | Section comparing against SCvx and a heritage law on accuracy, feasibility, CPU and robustness. Failure-mode taxonomy with an online indicator, e.g. predicted energy or range infeasibility, running peak ratio |
| R2 | Documented dispersion table; Monte Carlo sized statistically with confidence bounds | [@hanson2012mc; @hanson2010mc; @clopper1934binomial; @gaudet2022approach; @lu2017verification] | M | Publish the VISTA table (distribution, σ, per-run or profile). ≥ 2,000 runs at the headline scale. Clopper–Pearson CIs on every fraction |
| R3 | Robustness vs uncertainty magnitude, including beyond the training range | [@gaudet2022approach; @lu2017verification; @federici2025rlmpc; @fabisch2026attitude] | M | S4 scale sweep 0.1 → 1.25× (extend to 1.5×) with common random numbers, RL and SCvx on identical cases. Curves of median / p95 miss, \|Δγ\|, \|Δψ\|, feasibility, success |
| R4 | Separate guidance failure from physical infeasibility | [@lu2017verification] (beyond-design testing); §4.0 definitions | M | Per case, run an **oracle** SCvx open-loop solve with the true dispersed parameters. Score RL only on cases the oracle can solve, and report both views |
| R5 | Navigation and sensor errors | [@gaudet2022approach; @gaudet2023strike; @boone2022obs; @zavoli2021robust; @zhang2020samdp] | M | Test with Gaussian noise + bias + scale factor on r, v and measured drag; then train with them. Report a noise-scale curve as in R3 |
| R6 | Actuator / attitude-loop effects | [@gaudet2022approach; @gaudet2020exo; @fabisch2026attitude; @cadamuro2026refex] | M | First-order attitude lag τ ∈ {0, 0.5, 1, 2} s, pure command delay sweep, AoA/bank bias. Find the delay or lag at which success drops below threshold (a **delay margin**) |
| R7 | Experimental robustness margins | [@junnarkar2025margins; @yin2022iqc] (concept); [@gaudet2022approach] (out-of-distribution tables) | M | Gain factor k on C_L, C_D, ρ, mass and command gain. Report k at which P(success) < 0.95 or 0.99, with CI, for RL and SCvx |
| R8 | Worst-case / falsification search | [@braun2023worstcase; @corso2021survey; @lee2020ast; @pinto2017rarl] | S | CMA-ES or RL adversary over the dispersion vector (bounded at 1× or 3σ) minimizing success margin. Report worst cases for RL and SCvx and what breaks |
| R9 | Memory / adaptation ablation | [@gaudet2020adaptive; @federici2024meta; @carradori2026transformer; @kumar2021rma] | S | MLP vs RecurrentPPO (LSTM) at 1×; privileged-information teacher as an upper bound |
| R10 | Curriculum and DR ablations | [@openai2019rubik; @mehta2020adr; @narvekar2020curriculum] | S | v25 ADR curriculum vs uniform DR at the final scale vs 0.1×-trained v24b, all on S4 |
| R11 | Constraint-method ablation and P(violation) | [@stooke2020pid; @ray2019safetygym; @achiam2017cpo; @chow2018risk] | M | Fixed penalty vs PID-Lagrangian (vs P3O optional). Peak-ratio tails (p95, p99, max) and violation probability with CI; multiplier traces |
| R12 | Seeds and aggregate statistics | [@henderson2018matters; @agarwal2021precipice; @colas2018seeds; @patterson2024empirical] | M | ≥ 5 training seeds of the final configuration; IQM with stratified-bootstrap CIs; selection and test sets disjoint |
| R13 | Command smoothness / chattering | [@cadamuro2026refex; @mysore2021caps; @peng2025hybrid] | S | Bank-reversal count, command-rate RMS and spectra, fraction of time at rate limit, compared with SCvx |
| R14 | Statistical stability certificate | [@scorsoglio2026sindy; @calafiore2006scenario; @katz2021probabilistic] | C | PAC-bounded check of bounded terminal-error sensitivity (finite-horizon Jacobian norm) over sampled states at each energy level, or a SINDy surrogate of the lateral loop |
| R15 | Local linear analysis | [@yin2022iqc; @dawson2023certificates] | C | Linearize the closed loop along nominal and dispersed trajectories; report terminal-error amplification and an empirical Lipschitz bound of the policy [@pauli2022lipschitz] |
| R16 | Formal verification: why not, and what replaces it | [@julian2021reach; @everett2021reach; @tran2020nnv; @katz2017reluplex] | C | One paragraph on scale limits, plus a one-step output-range check of the policy over the observation box |
| R17 | Safety monitor / run-time assurance discussion | [@hobbs2023rta; @dunlap2024rta; @ames2019cbf; @rastgoftar2026viability] | C | Optional q̄/Q̇ corridor filter on the bank/AoA targets; report how often it would intervene |
| R18 | Onboard cost | [@su2023irl; @bao2025morphing; @hamilton2025processor] | S | Time per decision for the MLP and SCvx cycle on the same CPU; memory footprint |
| R19 | Higher fidelity (6-DOF or attitude dynamics) | [@cadamuro2026refex; @carradori2026transformer; @gaudet2023strike] | C | At least a 6-DOF spot check with a simple attitude controller, or an explicit limitations statement |
| R20 | Reproducibility | [@patterson2024empirical; @henderson2018matters] | S | Release dispersion sets (S3/S4 JSON), seeds, evaluation scripts |

**Minimum set before submission:** R1–R7, R11, R12. Without them a robustness/stability claim is unlikely to pass review at JGCD or TAES, given the editorial stance [@lu2024editorial] and the precedent set by Gaudet's tables [@gaudet2022approach].

---

## 9. Corrections and updates to `report/literature_review.md` (2026-10-05)

1. **[6] Gaudet et al. 2021, arXiv:2107.14764.** Published as Gaudet, Drozd & Furfaro, AIAA SciTech 2022 Forum, doi:10.2514/6.2022-2214 [@gaudet2022approach]. The arXiv version also lists Meltzer. Facts confirmed from the full text: AoA held at 40° above 4.57 km/s, ±3 % aero/density uniform, 5,000-run tests, 47 ± 20 m miss in distribution. The earlier table's "high degree of accuracy" can now be quantified.
2. **[9] Gaudet et al. 2021, arXiv:2110.00634.** Published as Gaudet & Furfaro, "…via Reinforcement Metalearning", J. Spacecraft and Rockets 60(1):286–298, 2023, doi:10.2514/1.A35396 [@gaudet2023strike].
3. **§2.3 "separate discounts for shaping and terminal rewards."** Confirmed in the strike paper: discount 0.90 for shaping and 0.995 for the terminal reward [@gaudet2021strike_arxiv]. The earlier attribution to [17, 18] (6-DOF landing) is not verified here; cite the strike paper instead.
4. **[8] Su et al. 2023.** The method is GAIL (inverse RL) with distributed PPO trained from 50 GPOPS demonstrations. "Good terminal accuracy" means a range-angle error below 0.25° (≈ 28 km) for 99.6 % of 1,000 runs [@su2023irl].
5. **[10] Jiang et al. 2023.** The success criterion is Δd < 300 km at the end of the glide phase, a handover to the dive, so this is not a precision result. Evaluation is 1,000 Monte Carlo runs with 0–10 % C_L/C_D error, and the authors note retraining is needed beyond the training range [@jiang2023her].
6. **[11] Bao et al. 2025.** Correct as listed. Note that RL controls the sweep angle only, and that the dispersions are Gaussian with the 3σ values given in §1.1 [@bao2025morphing].
7. **[12] Shi et al.** IEEE TAES 61(2):2106–2119, 2025, doi:10.1109/TAES.2024.3466125 [@shi2024lateral].
8. **[5] Peng et al.** IEEE TAES 61(1):342–354, 2025, doi:10.1109/TAES.2024.3443776 [@peng2025hybrid]. A third Peng et al. paper (hierarchical RL, 2025) is added [@peng2025hrl].
9. **[13] Li et al. 2024.** The abstract supports "~80 % faster online than a conventional predictor–corrector". The earlier claim "cut offline training by 60 % vs PPO" could not be verified from what was accessible; treat it as unverified.
10. **[45] Carradori et al.** Confirmed: JGCD 49(4):1083–1100, 2026, doi:10.2514/1.G009331, "up to 99.7 %" with a terminal patch [@carradori2026transformer].
11. **[40] Federici & Furfaro.** Neural Comput. Appl. 37(22):17249–17271, 2025 (online 2024) [@federici2024meta]. **[15] Federici & Zavoli.** Acta Astronaut. 214:147–158, 2024 [@federici2023robust].
12. **[46] Das et al.** There are two papers: J. Phys. Conf. Ser. 2633:012005 (2023) and IET Conf. Proc. 2023(43):229–235 (2024) [@das2023reentry; @das2024reentry].
13. **† items.** Henderson et al. (AAAI 2018, doi:10.1609/aaai.v32i1.11694) and Agarwal et al. (NeurIPS 2021) are now verified [@henderson2018matters; @agarwal2021precipice].
14. **The earlier positioning table's row for this work** (bank/AoA rate actions, 0.27 km nominal) is superseded. The 0.27 km result was obtained with the AoA bound leak and is invalid (RUNS.md, evaluation-harness era). The current formulation uses absolute targets on an energy grid (§1.3).
15. **Consensus links** in the earlier reference list should be replaced by DOIs. All are now in the new `.bib`.
16. **New primary entries not in the earlier review:** Cadamuro et al. 2026 JGCD (6-DOF ReFEx validation of RL entry guidance) [@cadamuro2026refex]; Bao et al. 2023 [@bao2023onboard]; Liu et al. 2025/2026 [@liu2025nfz; @liu2026timecontrol]; Zhou et al. 2025 [@zhou2025rlv]; Wang et al. 2025 [@wang2025td3]; Jiang, Li & Furfaro 2019 (Mars) [@jiang2019mars]; Ghiugan et al. 2026 (aerocapture) [@ghiugan2026uranus]; Falcone & Putnam 2023 [@falcone2023aerobraking]; Fabisch et al. 2026 [@fabisch2026attitude]; Wallace & Si 2024 [@wallace2024hsv]; Scorsoglio et al. 2026 [@scorsoglio2026sindy]; and the JGCD editorial [@lu2024editorial].

---

## 10. Search and verification log

**Sources queried (2026-10-24).**

- OpenAlex API: keyword searches, plus venue-filtered searches over JGCD, JSR, Acta Astronautica, Aerospace Science and Technology and IEEE TAES for RL × {entry, reentry, hypersonic, glide, atmospheric}.
- Crossref API: DOI metadata and BibTeX source.
- Semantic Scholar API: venues and abstracts.
- arXiv API: preprints, journal references and comments.
- Consensus: one query; the service then rate-limited.
- General web search.
- Publisher and proceedings pages via a browser: AIAA ARC, MDPI, Wiley/Hindawi, PMLR, JMLR, NeurIPS, the CJA-Sinica journal site and NASA NTRS.

**Queries.** RL / deep RL / PPO / meta-RL × {entry, reentry, hypersonic glide vehicle, Mars entry, aerocapture, skip entry, lunar return, reusable launch vehicle, winged re-entry}. Also: domain randomization, adversarial and robust RL, recurrent policies, navigation noise, actuator delay; Lyapunov RL, control barrier functions, run-time assurance, reachability, formal verification, disk/stability margins, IQC; constrained / Lagrangian / chance-constrained RL in aerospace; Monte Carlo verification sample size; classical entry-guidance baselines.

**Access limits.** ScienceDirect presented a CAPTCHA, which was not bypassed. AIAA ARC full texts were not accessible except the free editorial [@lu2024editorial]. Elsevier and most AIAA papers were therefore read at abstract level only (marked A or T).

**Items not fully verified.**

- `harpold1979shuttle`: no DOI; details from secondary citations.
- `moseley1969apollo`: title, author and year from the index only.
- `peng2025hrl`: the DOI on the publisher page is not in Crossref.
- Elsevier abstracts that OpenAlex and Semantic Scholar do not carry (Peng 2023, Liu 2025, Liu 2026 PQAC-RL, Iafrate 2025, Federici & Zavoli 2024): the summaries in §1 rely on abstract text from Consensus, web-search snippets or titles. They are marked A or T accordingly.

---

## References

All keys resolve in `literature_review_rl_entry.bib`. Grouped by section:

- **RL entry / hypersonic (primary):** gaudet2022approach, gaudet2021approach_arxiv, gaudet2023strike, gaudet2021strike_arxiv, su2023irl, jiang2023her, peng2023modular, peng2025hybrid, peng2025hrl, bao2023onboard, xu2025grukddpg, liu2025nfz, liu2026timecontrol, zhou2025rlv, gao2024nfz, li2024mbrl, bao2025morphing, wang2025td3, cadamuro2026refex, jiang2019mars, ghiugan2026uranus, falcone2023aerobraking, das2023reentry, das2024reentry, fabisch2026attitude, wallace2024hsv, song2022igc, wu2024spiral, liu2026pqac, shi2024lateral, rastgoftar2026viability.
- **Adjacent GNC:** gaudet2020adaptive, gaudet2019adaptive_arxiv, gaudet2020sixdof, furfaro2020zemzev, federici2024meta, carradori2026transformer, carradori2025scitech, federici2025rlmpc, scorsoglio2022lunar, scorsoglio2025lunar, iafrate2025propulsive, zavoli2021robust, federici2023robust, federici2022rendezvous, zavoli2026curriculum, boone2022obs, hovell2021prox, holt2026bc, izzo2024optimality, gaudet2020angleonly, gaudet2020exo, he2021computational, lee2023impactangle, qiu2023recorded, yan2024evasion, chaudhary2026chance, gaudet2023igc, dally2022sac, leiva2026adversarial, holt2021qlaw.
- **Robustness (ML):** tobin2017dr, peng2018dynrand, openai2019rubik, mehta2020adr, muratore2022review, zhao2020sim2real, portelas2020acl, narvekar2020curriculum, morimoto2005robust, pinto2017rarl, tessler2019action, rajeswaran2017epopt, moos2022robust, zhang2020samdp, duan2016rl2, lee2020quadruped, kumar2021rma, kirk2023survey.
- **Stability / verification:** lu2024editorial, scorsoglio2026sindy, berkenkamp2017safe, chow2018lyapunov, chang2019neural, han2020actorcritic, jin2020stability, dawson2023certificates, ames2017cbf, ames2019cbf, cheng2019rlcbf, alshiekh2018shield, dalal2018safety, hobbs2023rta, vanwijk2024cbf, dunlap2024rta, tammam2025safe, hamilton2025processor, katz2017reluplex, ivanov2019verisig, tran2020nnv, huang2019reachnn, everett2021reach, fazlyab2022sdp, julian2019acas, julian2021reach, katz2021probabilistic, yin2022iqc, junnarkar2025margins, pauli2022lipschitz, barbara2025lipschitz, mysore2021caps, calafiore2006scenario, campi2009scenario, corso2021survey, lee2020ast, braun2023worstcase.
- **Constrained RL:** altman1999cmdp, achiam2017cpo, tessler2019rcpo, yang2020pcpo, zhang2022p3o, ray2019safetygym, stooke2020pid, chow2018risk, garcia2015survey, gu2024review, brunke2022safe.
- **Classical entry guidance and evaluation practice:** mendeck2014msl, moseley1969apollo, harpold1979shuttle, harpold1983shuttle, lu2008pc, xue2010constrained, lu2014unified, brunner2008skip, brunner2012compare, lu2017verification, liu2016socp, wang2017scp, wang2018autonomous, wang2020improvedscp, mao2016scvx, malyuta2022convex, ridderhof2022stochastic, guo2022tianwen, cheng2020dnn, shi2021onboard, sanchez2018realtime, rataczak2025density, giraldo2026adaptive, giraldo2024navigation, hanson2012mc, hanson2010mc, clopper1934binomial.
- **Surveys:** chai2021review, izzo2019survey, tipaldi2022rl, shirobokov2021survey.
- **RL methodology:** schulman2017ppo, henderson2018matters, agarwal2021precipice, colas2018seeds, patterson2024empirical.
