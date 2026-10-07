# Literature Review — RL Entry Guidance with Terminal and Path Constraints

*Compiled 2026-10-05. Scope set by the publication goal: improve the Stage 2 RL guidance (bank + AoA rate commands, 3-DOF) so that it (i) meets the full terminal state (position, flight path angle, heading), (ii) satisfies the heat rate / load / dynamic pressure path constraints, (iii) minimizes terminal error further, and (iv) is robust under uncertainty with Monte Carlo evidence. SCvx is used only as an external benchmark, never inside training.*

Reference numbers point to the list at the end. Items marked † come from general knowledge of the field rather than this search session; verify bibliographic details before citing.

---

## 1. Positioning: what the literature already covers and where the gap is

| Work | Vehicle / phase | Actions | Terminal conditions | Path constraints | Uncertainty | Reported accuracy |
|---|---|---|---|---|---|---|
| Gaudet et al. 2021 [6] | Hypersonic glider, approach phase | bank + AoA **rates** (same as ours) | position + **terminal speed** | heat rate, load, q̄ via reward | aero coefficients, actuator failure, sensor noise; recurrent meta-RL | "high degree of accuracy"; compared to LQR tracking an optimal trajectory |
| Gaudet et al. 2021 [9] | Hypersonic strike, terminal | bank, AoA, sideslip rates | position, min speed | heat, q̄, load | aero, actuator lag, sensor errors; preliminary 6-DOF | precision strike |
| Peng et al. 2023 [2], 2025 [5] | Entry, 3-D | separate AoA and bank modules; **hybrid discrete-continuous** bank (sign + magnitude), TD3 | position, wide reachable set | yes | changing targets | real-time, multi-mission |
| Jiang et al. 2023 [10] | HGV | bank magnitude + sign | multi-constraint posed as **multi-goal**, HER + DDPG | yes | initial bias, target change, aero deviation | — |
| Su et al. 2023 [8] | CAV-H and RLV | bank (PPO, inverse RL dense reward) | terminal accuracy | heat, load, q̄ | — | "good terminal accuracy" |
| Xu et al. 2025 [21] | X-34 RLV | GRU-DDPG, magnitude/sign split | position | yes | disturbances | better than DDPG |
| Bao et al. 2025 [11] | Morphing HGV, hierarchical | DDPG for sweep only; analytic QEGC guidance | position, velocity | yes | Monte Carlo | 0.263 ± 0.184 km |
| Shi et al. 2025 [12] | Classical lateral predictor-corrector | analytic | **terminal heading** | — | Monte Carlo | — (heading-constrained baseline) |
| **This work (current)** | WB001 winged body, full entry 100→25 km, 7843 km range | bank + AoA rates, PPO | position only (0.27 km nominal) | barrier penalty, feasible | **none yet** | 0.27 km, single IC |

**Gap the paper can claim.** The closest pure-RL work (Gaudet [6]) constrains position and speed, not flight path angle and heading, and uses a short approach phase. Most entry RL papers either split bank sign from magnitude and lean on analytic lateral logic, or embed RL inside a predictor-corrector [1, 13, 20]. A **single end-to-end RL policy flying the whole entry (100 → 25 km, ~7800 km range) to a full terminal state (position, γ, ψ) inside heat/load/q̄ constraints, with Monte Carlo robustness and a same-physics SCvx benchmark**, is not covered by any of the papers found. Your reward-design rules (uniform terminal anchor, episode-mean penalties, observation resolution matching the reward, Jensen/curvature argument) are a methodological contribution on their own; no paper in the search states them this explicitly.

---

## 2. Terminal constraints: flight path angle and heading

**2.1 Terminal reward on the full error vector.** Extend the terminal anchor from Φ(d) to Φ(e) with a normalized error e = [d/d_ref, Δγ/γ_ref, Δψ/ψ_ref]. Keep your own rules:
- charge it uniformly on every episode ending;
- keep the near field linear (dispersion-neutral) or mildly concave per your Jensen argument;
- add the matching observations at the resolution the reward acts on (Δγ, Δψ, and possibly their log-magnitudes near zero), as in v15.

**2.2 Tolerance annealing (ε-curriculum).** Zavoli & Federici [14] and the follow-up meta-RL work [15] handle hard terminal constraints in PPO with a penalty of the form max(0, ‖e‖ − ε). The tolerance ε shrinks during training, so the agent first learns to arrive roughly and is then progressively tightened. This maps directly onto adding γ and ψ: start with loose ε_γ, ε_ψ and anneal them. Hu et al. [16] add a *multistage exponential* terminal reward that keeps gradients informative as the errors become small, which addresses the same "last kilometres" problem you hit in v16–v19.

**2.3 Separate discounts for shaping and terminal rewards.** Gaudet et al. [17, 18] report that using different discount rates for shaping and terminal rewards significantly improves optimization. That is precisely the tension you resolved by hand in v8 and v10 (γ = 0.999 → 0.9999). Implementing two value heads (or two GAE streams) with γ_shape < γ_term would let the shaping stay local while the terminal reward is not attenuated. This is cheap to add on top of SB3 and gives a clean ablation.

**2.4 Guidance-aware dense shaping.** Gaudet [6] shapes the reward with a velocity-field tracking error based on parallel navigation, adapted to the curved Earth, which keeps the line of sight aligned with the velocity vector. For a **terminal heading** requirement, the analogous shaping aligns the velocity with a desired approach direction near the target (a curved-Earth version of impact-angle-constrained guidance). Zhang et al. [19] show RL handling terminal **angle + velocity** constraints in diving guidance. That paper uses an optimal-guidance baseline, though, which you have ruled out.

**2.5 Goal conditioning and hindsight relabelling.** Jiang et al. [10] turn the multiple terminal constraints into multiple goals and use HER with DDPG to reuse failed episodes. If you switch to an off-policy algorithm, a target-conditioned policy (target lat/lon/γ/ψ in the observation, randomized per episode) gives both terminal-state control and generalization. It also turns "different handover conditions" into an evaluation axis.

**2.6 Bank reversals as a hybrid action.** Heading control in entry is usually done through bank-sign reversals. Peng et al. [5] model the bank as a discrete sign plus a continuous magnitude (hybrid TD3) to avoid the trade-off between exploration and accuracy. Your rate-command formulation is smoother, but if heading convergence proves hard, a hybrid sign/magnitude head is the reference design to compare against.

**2.7 Independent variable.** Peng et al. [2] use a non-uniform discretization with a state-related independent variable, which resolves step-size conflicts between the long glide and the short terminal phase. For you this means stepping or observing in **energy** (or range-to-go) rather than fixed 1 s steps. Decision density then increases near the handover, exactly where γ and ψ must be trimmed. An energy-to-go observation is also the standard classical entry-guidance variable.

---

## 3. Path constraints

**3.1 Constrained RL instead of barrier tuning (pure RL, no solver).** Your barrier in Eq. (2) is a fixed penalty. The constrained-RL literature replaces fixed weights with adaptive multipliers that enforce a cost budget:
- PID-Lagrangian PPO (Stooke et al., ICML 2020) [22] damps the multiplier oscillations of plain Lagrangian PPO.
- P3O [23]: an exact penalty with a finite factor, with a multi-constraint extension (you have three constraints).
- Augmented-Lagrangian variants: APPO [24], PPO-EAL [25].
- Projection-based PCPO [26].

Formulation: one cost per constraint, c_k = max(0, g_k − 1) or the per-episode peak ratio, with budget ≈ 0. This also removes three weights from the HPO search space.

**3.2 Safety filters / run-time assurance (optional layer).** Control barrier functions:
- An optimization-based RTA keeps an RL spacecraft inside its constraints during and after training (van Wijk et al. [27]).
- Robust CBF layers [28], and RL-CBF [29], which also guides exploration.
- **CBF-RL** [30] applies the filter only during training, so the policy *internalizes* the constraint and needs no filter at deployment. This fits a "pure RL at run time" story well.
- Adaptive domain randomization combined with relaxed CBFs [31].

For entry, a CBF on q̄ or heat rate written as an altitude-velocity corridor bound, acting on the bank and AoA rate commands, is low-dimensional and closed-form for a scalar action. It is a monitor, not a trajectory optimizer, so it stays consistent with your "no SCvx in the loop" requirement.

**3.3 Reporting.** Gaudet [6, 9] treat constraints as terminal conditions on the episode (fail if violated). Whatever method you pick, report per-constraint peak-ratio distributions over the Monte Carlo, not just "feasible yes/no".

---

## 4. Minimizing terminal error (precision)

Your HPO showed that the limit is the learning dynamics (batch, rollout, action std). The literature offers three increasingly strong options.

**4.1 Exploration-noise control within RL.**
- Anneal the log-std on a schedule, or use state-dependent std (gSDE in SB3).
- Evaluate deterministically, as you already do.
- SAC with an entropy target annealed toward low values; Cheng et al. [32] use entropy-weighted adaptive SAC for lunar descent.

**4.2 Differentiable simulation / analytic policy gradients.** Your dynamics are smooth: no contacts, polynomial aero, an exponential atmosphere. That is the favourable regime for first-order gradients.
- **SHAC** [33]: truncated windows plus a critic to avoid exploding gradients over long horizons; one to two orders of magnitude faster than PPO in wall clock.
- **PODS** [34]: reports better asymptotic precision on tasks "that demand a high degree of accuracy and precision".
- **GI-PPO** [35]: blends analytic gradients into PPO adaptively. A natural fine-tuning stage after your current PPO policy.
- Suh et al. [36] explain when first-order gradients fail (stiffness, discontinuities). This matters for your termination events; smoothing the altitude crossing helps.
- Segmented APG for long horizons [37].
- Residual policy learning over a simple baseline improves asymptotic reward with first-order gradients [38].

This is still pure RL/learning, uses no external optimizer, and is a plausible route from 270 m to tens of metres. It is also a novelty claim for entry guidance; no entry paper using it was found.

**4.3 Fair policy selection.** Wijayatunga et al. [39] select policies on a **fixed grid of initial states** rather than on the best single evaluation. Your 0.27 km is the best of many evaluations across 41 trials, which is optimistic. A fixed evaluation grid plus multiple seeds removes that bias. Reviewers will expect it: Henderson et al. 2018 "Deep RL that Matters"†; Agarwal et al. 2021 "Deep RL at the Edge of the Statistical Precipice", with interquartile mean and bootstrap confidence intervals†.

---

## 5. Robustness under uncertainty

**5.1 Domain randomization.** Randomize the following during training:
- entry-interface state (altitude, speed, γ, ψ, lat/lon);
- atmospheric density (bias plus altitude-correlated perturbation);
- C_L/C_D scale factors and mass;
- navigation noise on observations;
- actuator lag on the rate commands.

This is the setup of Gaudet [6, 9] and of Federici et al. [40] in a pinpoint-landing comparison. Start narrow and widen (curriculum-based / adaptive DR [31, 41, 42]); uniform wide randomization from the start often slows convergence.

**5.2 Recurrent / meta-RL policies.** Under randomized dynamics the problem becomes a POMDP. Recurrent policies trained across the randomization adapt online:
- Gaudet et al. [43] compare non-recurrent and recurrent agents; the recurrent one adapts best.
- Federici et al. [15, 40, 44] show recurrent meta-RL meeting terminal conditions more accurately, including outside the training domain. They also feed the **previous action and reward** to the RNN.
- Carradori et al. [45] (JGCD 2026) use a **Gated Transformer-XL** policy for atmospheric rocket landing with 6-DOF Monte Carlo, reaching 99.7% compliance with a terminal patch.
- GRU-DDPG [21] for RLV entry.

In SB3 the recurrent route is `sb3_contrib.RecurrentPPO`.

**5.3 Monte Carlo protocol.**
- Several thousand runs over the dispersion set.
- Report terminal error in position, γ and ψ (mean, 3σ / 99th percentile), constraint peak ratios, and success rate.
- Run the same dispersion set through SCvx for the benchmark.
- Bao et al. [11] (0.263 ± 0.184 km) and Shi et al. [12] give comparable entry Monte Carlo numbers to position against.

---

## 6. Algorithm and infrastructure notes

- **PPO vs off-policy.** Das et al. [46] compare DDPG/PPO/TD3 on HGV entry; Peng [5] uses TD3. Off-policy methods reuse data, which matters if each episode is ~1500 steps. Keep PPO as the main method for continuity with the campaign, and add SAC/TD3 as an ablation.
- **Model-based RL.** Li et al. [13] use model-based RL for entry and cut offline training by 60% vs PPO, but inside a predictor-corrector. That is a useful reference for sample efficiency, not a template.
- **Throughput.** Moving the physics core to a vectorized JAX implementation (thousands of parallel environments on GPU) is what makes the domain-randomized, multi-seed Monte Carlo campaign feasible. It is also a prerequisite for §4.2.
- **Deployability.** Yang et al. [47] formally verify a compressed NN guidance policy (ReluVal). This is an optional discussion point on certification.

---

## 7. Recommended roadmap (pure RL, SCvx only as benchmark)

1. **Baseline hygiene (days).** Re-evaluate the 0.27 km policy on a fixed grid of perturbed initial states and 3–5 training seeds. This gives you the honest starting number and the evaluation harness for everything below.
2. **Full terminal state.**
   - Add Δγ, Δψ to the terminal anchor and observations.
   - Apply ε-annealing on the γ/ψ tolerances [14].
   - Use separate shaping/terminal discounts [17].
   - Optionally step on an energy-based independent variable [2].
   - Expect AoA to start modulating, because energy management is now required.
3. **Constrained RL for path constraints.** Use PID-Lagrangian or P3O with three costs [22, 23], ablated against your barrier. Optionally train with a CBF filter and deploy without it [30].
4. **Robustness.** Curriculum domain randomization [41], recurrent policy [43, 40], then a Monte Carlo campaign against SCvx on the same dispersions.
5. **Precision fine-tuning.** Analytic-gradient fine-tuning (SHAC / GI-PPO) in a differentiable JAX port [33, 35].
6. **Paper structure.** Physics validation → reward rules (your campaign) → constrained full-terminal-state policy → robustness → precision → SCvx comparison (accuracy, constraint margins, heat load, CPU time).

---

## References

[1] [Integrated entry guidance with no-fly zone constraint using reinforcement learning and predictor-corrector technique](https://consensus.app/papers/details/74cfbb201ffc5f2eb31c17a35de3d555/?utm_source=claude_desktop) — Gao et al., 2024, Proc. IMechE Part G.
[2] [Real-time adaptive entry trajectory generation with modular policy and deep reinforcement learning](https://consensus.app/papers/details/d15e90123e1a59139846f2fa6549589c/?utm_source=claude_desktop) — Peng et al., 2023, Aerospace Science and Technology.
[3] [Coordinated Design of Intelligent Morphing Decision and Entry Guidance for Morphing Hypersonic Glide Vehicles](https://consensus.app/papers/details/13efae826ac254eaba62041627892ec4/?utm_source=claude_desktop) — Zhang et al., 2025, J. Aerospace Engineering.
[4] [Reentry Guidance for Hypersonic Vehicle based on Reinforcement Learning](https://consensus.app/papers/details/4038d764ea7c5f2b85102e092769e99f/?utm_source=claude_desktop) — Liu et al., 2023, ICFEICT.
[5] [3-D Autonomous Entry Trajectory Planning via Hybrid Action Reinforcement Learning](https://consensus.app/papers/details/c56c97ea7e035879aa8f0191a1818303/?utm_source=claude_desktop) — Peng et al., 2025, IEEE TAES.
[6] [Adaptive Approach Phase Guidance for a Hypersonic Glider via Reinforcement Meta Learning](https://consensus.app/papers/details/ea3ab42dfb205338bc8b3c190c03e707/?utm_source=claude_desktop) — Gaudet et al., 2021, arXiv:2107.14764.
[7] [Collaborative penetration guidance strategy for hypersonic vehicles based on double-DQN](https://consensus.app/papers/details/ef294438147953e184382a540f874ccd/?utm_source=claude_desktop) — Teng et al., 2025, Aeronautical Journal.
[8] [A Real-Time and Optimal Hypersonic Entry Guidance Method Using Inverse Reinforcement Learning](https://consensus.app/papers/details/4981c674a5c457de8ca2ef3901f06db8/?utm_source=claude_desktop) — Su et al., 2023, Aerospace.
[9] [Terminal Adaptive Guidance for Autonomous Hypersonic Strike Weapons via Reinforcement Learning](https://consensus.app/papers/details/7d9be22709c6591f980cb4f2d94fd2d4/?utm_source=claude_desktop) — Gaudet et al., 2021, arXiv:2110.00634.
[10] [Intelligent Online Multiconstrained Reentry Guidance Based on Hindsight Experience Replay](https://consensus.app/papers/details/99531f63b68b5afd8543901a1b760f3e/?utm_source=claude_desktop) — Jiang et al., 2023, Int. J. Aerospace Engineering.
[11] [Coordinated Reentry Guidance with A* and Deep Reinforcement Learning for Hypersonic Morphing Vehicles Under Multiple No-Fly Zones](https://consensus.app/papers/details/3a4379013fae592897f9bf7f112c9c05/?utm_source=claude_desktop) — Bao et al., 2025, Aerospace.
[12] [Real-Time Lateral Predictor–Corrector Entry Guidance With Terminal Heading Angle Constraint](https://consensus.app/papers/details/7fbecd74a29e59c9a3c07b4dd5004a26/?utm_source=claude_desktop) — Shi et al., 2025, IEEE TAES.
[13] [Entry guidance for spatial no-fly zones avoidance via model-based reinforcement learning](https://consensus.app/papers/details/8eea59023b5d53d09d2859ed01ede767/?utm_source=claude_desktop) — Li et al., 2024, Aerospace Science and Technology.
[14] [Reinforcement Learning for Robust Trajectory Design of Interplanetary Missions](https://consensus.app/papers/details/c87c98235a3c59e2a95f1e85479dac98/?utm_source=claude_desktop) — Zavoli & Federici, 2021, JGCD.
[15] [Robust interplanetary trajectory design under multiple uncertainties via meta-reinforcement learning](https://consensus.app/papers/details/6d172523253f5ba0a28754808467fe5a/?utm_source=claude_desktop) — Federici et al., 2023, Acta Astronautica.
[16] [Densely Rewarded Reinforcement Learning for Robust Low-thrust Trajectory Optimization](https://consensus.app/papers/details/910946007e57556cacf521ff425ebd85/?utm_source=claude_desktop) — Hu et al., 2023, Advances in Space Research.
[17] [Deep Reinforcement Learning for Six Degree-of-Freedom Planetary Powered Descent and Landing](https://consensus.app/papers/details/fa9297fee0005352825db542a2409f53/?utm_source=claude_desktop) — Gaudet et al., 2018, arXiv.
[18] [Deep reinforcement learning for six degree-of-freedom planetary landing](https://consensus.app/papers/details/c8776e7d569b53b39057d1b9b2cac59c/?utm_source=claude_desktop) — Gaudet et al., 2020, Advances in Space Research.
[19] [Intelligent diving guidance with terminal angle and velocity constraints via deep reinforcement learning](https://consensus.app/papers/details/4c0530364ee75c6c86fef8b4dda60517/?utm_source=claude_desktop) — Zhang et al., 2024, Proc. IMechE Part G.
[20] [A prediction and correction reentry guidance method based on BP network and deep Q-learning network](https://consensus.app/papers/details/e7ac44403fa85e368f18f85724b28b79/?utm_source=claude_desktop) — Wang et al., 2025, J. Northwestern Polytechnical University.
[21] [Reentry Guidance Method for Reusable Launch Vehicles Based on GRU-DDPG](https://consensus.app/papers/details/818830e589975a5ebd90de91df53da39/?utm_source=claude_desktop) — Xu et al., 2025, IFAC-PapersOnLine.
[22] [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods](http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf) — Stooke, Achiam & Abbeel, 2020, ICML.
[23] [Penalized Proximal Policy Optimization for Safe Reinforcement Learning](https://consensus.app/papers/details/d7a86c298f3550a8b0300997fbfb6883/?utm_source=claude_desktop) — Zhang et al., 2022 (IJCAI).
[24] [Augmented Proximal Policy Optimization for Safe Reinforcement Learning](https://consensus.app/papers/details/79157883690b5f719ce568cce0121012/?utm_source=claude_desktop) — Dai et al., 2023 (AAAI).
[25] [PPO-EAL: Exact Augmented Lagrangian Proximal Policy Optimization for Safe Robotic Control](https://consensus.app/papers/details/918bdb3d372f5fdd90fd8695c6a72161/?utm_source=claude_desktop) — Ding et al., 2026, arXiv.
[26] [Projection-Based Constrained Policy Optimization](https://consensus.app/papers/details/358f17248b6054d79a5b558f3f240105/?utm_source=claude_desktop) — Yang et al., 2020 (ICLR).
[27] [Safe Spacecraft Inspection via Deep Reinforcement Learning and Discrete Control Barrier Functions](https://consensus.app/papers/details/ace2cea999215c1eb3b2a03a365f3ca8/?utm_source=claude_desktop) — van Wijk et al., 2024, J. Aerospace Information Systems.
[28] [Safe Reinforcement Learning Using Robust Control Barrier Functions](https://consensus.app/papers/details/1cb1ed1ebef05c4dbc38738d0c9e6763/?utm_source=claude_desktop) — Emam et al., 2021, IEEE RA-L.
[29] [End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks](https://consensus.app/papers/details/2330a261dfd352b2b20b4951ca66eaff/?utm_source=claude_desktop) — Cheng et al., 2019 (AAAI).
[30] [CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions](https://consensus.app/papers/details/aad3abb79d635034a9e2774168599c62/?utm_source=claude_desktop) — Yang et al., 2025, arXiv.
[31] [Safe Spacecraft Guidance Using Adaptive Domain Randomization and Relaxed Control Barrier Functions](https://consensus.app/papers/details/c940cbb283f8593eb5d25cded2e9ad9f/?utm_source=claude_desktop) — Tammam et al., 2025, IEEE SMC.
[32] [Powered Descent Flight Planning for Lunar Explorer Based on Deep Reinforcement Learning](https://consensus.app/papers/details/aac3450b85705548a4928e2e8d199a5f/?utm_source=claude_desktop) — Cheng et al., 2025, CCDC.
[33] [Accelerated Policy Learning with Parallel Differentiable Simulation](https://consensus.app/papers/details/21034edc3ed356f591c68c5f8c8242e9/?utm_source=claude_desktop) — Xu et al., 2022 (ICLR), SHAC.
[34] [PODS: Policy Optimization via Differentiable Simulation](https://consensus.app/papers/details/28433be09fc25c1bb69de7651e3b89b7/?utm_source=claude_desktop) — Zamora et al., 2021 (ICML).
[35] [Gradient Informed Proximal Policy Optimization](https://consensus.app/papers/details/26718812b65058bda90c5375a90e3cbc/?utm_source=claude_desktop) — Son et al., 2023 (NeurIPS).
[36] [Do Differentiable Simulators Give Better Policy Gradients?](https://consensus.app/papers/details/5030af624d7950ef8597f25872d67d17/?utm_source=claude_desktop) — Suh et al., 2022 (ICML).
[37] [Backpropagating Through Simulation: Analytic Policy Gradients for Sample and Learning Efficient Differentiable Continuous Control](https://consensus.app/papers/details/84407b53446654d3beb2fee7da590ce4/?utm_source=claude_desktop) — Deng, 2026, arXiv.
[38] [Residual Policy Learning for Perceptive Quadruped Control Using Differentiable Simulation](https://consensus.app/papers/details/cb0b6cdefffb52ff8a5bb6e8422c2b49/?utm_source=claude_desktop) — Luo et al., 2025, ICRA.
[39] [Robust trajectory design and guidance for far-range rendezvous using reinforcement learning with safety and observability considerations](https://consensus.app/papers/details/77790fd4efcb5d5f932802a59500ff50/?utm_source=claude_desktop) — Wijayatunga et al., 2025, Aerospace Science and Technology.
[40] [Improving reinforcement learning performance in spacecraft guidance and control through meta-learning: a comparison on planetary landing](https://consensus.app/papers/details/963bd146f4025e30bb03648cd426d96b/?utm_source=claude_desktop) — Federici et al., 2024, Neural Computing and Applications.
[41] [Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey](https://consensus.app/papers/details/6fed3f69f2e05030a719dd79579dd94c/?utm_source=claude_desktop) — Narvekar et al., 2020, JMLR.
[42] [Automatic Curriculum Learning For Deep RL: A Short Survey](https://consensus.app/papers/details/f41a4e91184758c980add8cf3458e98e/?utm_source=claude_desktop) — Portelas et al., 2020 (IJCAI).
[43] [Adaptive Guidance and Integrated Navigation with Reinforcement Meta-Learning](https://consensus.app/papers/details/1f0ccd797d1753da8cf6d3dc4c31d7dd/?utm_source=claude_desktop) — Gaudet, Linares & Furfaro, 2020, Acta Astronautica.
[44] [Meta-reinforcement learning for adaptive spacecraft guidance during finite-thrust rendezvous missions](https://consensus.app/papers/details/ebda6b02181555b3b1fef3352ee8ca4f/?utm_source=claude_desktop) — Federici et al., 2022, Acta Astronautica.
[45] [Transformer-Based Atmospheric Rocket Landing Guidance Using Meta-Reinforcement Learning](https://consensus.app/papers/details/8a9cfa0dd36c547d9ca3033befe91fd1/?utm_source=claude_desktop) — Carradori et al., 2026, JGCD.
[46] [Hypersonic vehicle reentry trajectory design based on reinforcement learning](https://consensus.app/papers/details/74fae2999d26584d900f4ebbe0331b5d/?utm_source=claude_desktop) — Das et al., 2024, CMAAE.
[47] [Model-Based Reinforcement Learning and Neural-Network-Based Policy Compression for Spacecraft Rendezvous on Resource-Constrained Embedded Systems](https://consensus.app/papers/details/74ba696fa2ae5aa9acef3c1f92f37c63/?utm_source=claude_desktop) — Yang et al., 2023, IEEE TII.

Further sources consulted: [Effectiveness of Warm-Start PPO for Guidance with Highly Constrained Nonlinear Fixed-Wing Dynamics (Sandia)](https://www.sandia.gov/app/uploads/sites/86/2023/03/Effectiveness_of_Warm_Start_PPO_for_Guidance_with_Highly_Constrained_Nonlinear_Fixed_Wing_Dynamics.pdf), relevant to your warm-start methodology.

† Henderson et al., "Deep Reinforcement Learning that Matters", AAAI 2018; Agarwal et al., "Deep RL at the Edge of the Statistical Precipice", NeurIPS 2021; Ng, Harada & Russell, "Policy invariance under reward transformations", ICML 1999 (potential-based shaping, already implicit in your Eq. 1).
