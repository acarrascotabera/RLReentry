# Stage-2 training runs — reward & hyperparameter log

Stage 2 = bank-rate + AoA-rate guidance (`Stage2BankAoaEnv`), WB001 heat-load
mission, target at (70°N, 12°E, 25 km), initial range ≈ 7,800 km. Success =
landing error ≤ 1 km. SCP reference heat load J_ref = 1.072e9 J/m².

Final `d_km` below is the deterministic end-of-run eval (`eval_summary.txt`);
"best" is the best deterministic eval seen during training.

## PPO hyperparameters (shared)

All runs use the Stage-1 Optuna best (`hpo_stage1_run1/best.json`) unless noted:

| param | value |
|---|---|
| learning rate | 2.22e-4, linear decay to 0 |
| n_steps / batch / epochs | 1024 / 256 / 5 |
| gamma / gae_lambda | 0.99891 / 0.9623 |
| clip_range / ent_coef | 0.274 / 0.005 |
| net_arch | 512×512 |
| n_envs / dt / substeps | 8 / 1 s / 3 |

## Legacy reward era (v1–v6): many-term reward

Reward terms: per-step heat (`w_heat·dJ/J_ref`), path-constraint barriers
(`w_path`), action magnitude (`w_act`), action change (`w_smooth`), AoA floor
(`w_aoa_lo`, v3+), dense log-position shaping (`w_progress·Δ(w_pos·log10 d)`),
terminal log-position penalty + FPA energy term (`w_fpa`), success bonus 100,
flat fail penalty 50.

| run | steps | w_pos | w_fpa | w_aoa_lo | w_smooth | w_act | ent | final d_km |
|---|---|---|---|---|---|---|---|---|
| v1 | 2M | 150 | 4.81 | – | 0.38 | 0.05 | 0.005 | 270.6 |
| v2 | 4M | 150 | 4.81 | – | 0.38 | 0.05 | 0.005 | aborted |
| v3 | 4M | 150 | 4.81 | 1.0 | 0.38 | 0.05 | 0.005 | 297.0 |
| v4 | 4M | 250 | 4.00 | 1.0 | 0.38 | 0.05 | 0.005 | **226.0** |
| v5 | 4M | 250 | 4.00 | 1.0 | 0.38 | 0.05 | 0.005 | 358.6 |
| v6 | 4M | 250 | 4.00 | 1.0 | 0.05 | 0.01 | 0.02 | 604.0 |

Verdict: plateau at 200–600 km regardless of weight tuning; heavy seed/run
variance. Motivated stripping the reward to position only.

## Minimal position-first reward era (v7+)

Reward reduced to 4 weights (2026-07-02): `w_pos` (log10 landing-error,
potential-based shaping), `w_succ` (≤1 km bonus), `w_fail`, `w_smooth`
(action-change penalty, anti-chattering). Heat / path barriers / action
magnitude / AoA terms / terminal FPA-heading all removed (still logged,
unrewarded). All runs 10M steps, seed 0.

| run | w_pos | w_succ | w_fail | w_smooth | terminal anchor | outcome |
|---|---|---|---|---|---|---|
| v7 | 100 | 100 | 50 | 0.05 | on landing only | **skip-out exploit.** Reached 231 km @4.7M, then learned to balloon past 120 km alt — failing (−50) was cheaper than landing far (−100·log10 d). Never lands after ~5M. |
| v8 | 100 | 100 | 50 | 0.05 | none (shaping only) | **dive-early attractor.** γ=0.9989 makes time expensive; ending at 5,400 km in 562 s out-discounts a 1,500 s closing glide. Frozen 0.2M→8.5M; final 4,503 km. |
| v9 | 100 | 100 | 50 | 0.05 | uniform: −w_pos·log10(d_end) charged on EVERY ending (+w_fail extra on fail/timeout) | **structure works, discount leaks.** Lands in 97% of evals — both v7/v8 attractors dead. Best 256 km @3.4M, but converged to 680–870 km (final eval 869 km): with γ=0.9989 the terminal anchor is discounted by γ^1500≈0.19, so precision end-game is worth ~8 units — train reward climbed to 10M while eval d_km flatlined. Overshoot end-game (passes ~200 km from target with excess energy); AoA chatters at full authority in high-q̄ phases; unrewarded constraint peaks n≈16 g, Q̇≈4×. |

### v9 reward (current)

```
r_t      = w_pos·(log10 d_prev − log10 d_cur)          # shaping (credit assignment)
           − w_smooth·‖a_t − a_{t−1}‖²                  # anti-chattering
r_end    = −w_pos·log10(d_end)                          # uniform terminal anchor
           + w_succ·[d ≤ 1 km]  −  w_fail·[fail/timeout]
w_pos=100, w_succ=100, w_fail=50, w_smooth=0.05
```

Lesson (v7 vs v8 vs v9): the terminal anchor must be charged identically on
every episode ending — charged on one outcome only creates outcome-selection
exploits; absent, discounting rewards ending the episode early.

### v10 — done (2026-07-04): found 43.5 km, then collapsed

Diagnosis from v9: the reward STRUCTURE is right (uniform anchor ⇒ no
degenerate attractors, 97% landing rate) but γ=0.9989 over ~1,500-step
episodes discounts the terminal anchor by ≈0.19 — the optimizer barely sees
the precision end-game, so it converged to a fast-approach/sloppy-landing
policy (train reward kept improving while eval d_km flatlined; v4's 226 km
had implicitly compensated with a ~5× heavier terminal term). Secondary:
AoA chatters at full authority (w_smooth=0.05 costs only ~0.2/step).

Selected changes (everything else identical to v9):

| knob | v9 | v10 | why |
|---|---|---|---|
| gamma | 0.99891 (HPO) | **0.9999** | γ^1500 ≈ 0.86: terminal anchor keeps ~86/100 of its weight across a full glide (was ~19). Landing 800→300 km is now worth ~37 discounted units (was ~8). |
| w_smooth | 0.05 | **0.2** | AoA buzz at full authority; still small vs w_pos so it can't fight position. |

Reward (unchanged form):
```
r_t   = 100·(log10 d_prev − log10 d_cur) − 0.2·‖a_t − a_{t−1}‖²
r_end = −100·log10(d_end) + 100·[d ≤ 1 km] − 50·[fail/timeout]
```

Command: `train_sb3 --stage stage2 --timesteps 10000000 --n-envs 8
--hpo-best hpo_stage1_run1/best.json --w-smooth 0.2 --gamma 0.9999
--lr-decay --seed 0 --outdir results/stage2_v10`

**Outcome.** Best model @200k steps rolls out to **43.5 km** — campaign best
by 5×, and the trajectory is everything at once: fully feasible (n peak 1.2 g,
q̄ 2.2 kPa, Q̇ touches the limit once), J_heat 0.95× the SCP optimum, bank rate
within ±0.3°/s, zero chattering, AoA steady at 40°. So the 4-weight reward's
optimum is the right trajectory and γ=0.9999 makes it visible. But PPO then
destroyed it: a large early update (stage-1-tuned lr 2.2e-4 / clip 0.274 vs
γ=0.9999-scale value targets) collapsed the policy into the v8-style
dive-early attractor at ~350k, and it ground from 5,700 to 4,200 km for the
remaining 9.5M steps without recovering (final eval 4,198 km). Lesson:
γ=0.9999 is right for the *objective* but too hot for from-scratch PPO at
these update sizes — the failure is optimization stability, not reward design.

### v11 — stopped early (2026-07-04): collapsed again — and revealed the real bug

Reward UNCHANGED (v10 proved the optimum is right). Strategy change: warm
start from v10's best model (43.5 km @200k, with its VecNormalize stats from
the 250k checkpoint) and polish gently for 10M steps — small LR, tight clip,
low entropy. New `--init-model/--init-vn/--lr/--clip-range` options in
train_sb3.

| knob | v10 | v11 | why |
|---|---|---|---|
| init | from scratch | **v10 best (43.5 km)** | exploit the found basin instead of re-rolling |
| lr | 2.2e-4, decay | **5e-5, decay** | polish, don't jump out of the basin |
| clip_range | 0.274 | **0.1** | cap per-update policy movement (what killed v10) |
| ent_coef | 0.005 | **0.003** | mild exploration; let action std shrink |
| gamma | 0.9999 | 0.9999 (inherited) | terminal anchor stays visible |

Command: `train_sb3 --stage stage2 --timesteps 10000000 --n-envs 8
--init-model results/stage2_v10/best/best_model.zip
--init-vn results/stage2_v10/ckpt/ppo_vecnormalize_250000_steps.pkl
--lr 5e-5 --lr-decay --clip-range 0.1 --ent-coef 0.003 --w-smooth 0.2
--seed 0 --outdir results/stage2_v11`

**Outcome.** Despite the tiny LR and clip, the warm-started 43 km policy
degraded immediately (best eval 634 km) and slid back into the dive attractor
(last ckpt: 5,338 km in 355 s). Stopped at ~5.5M. That ruled out optimizer
aggressiveness and exposed the actual bug: **the per-step smoothness penalty
is an episode-length × exploration-noise tax.** With action std σ≈1,
E‖a_t−a_{t−1}‖² ≈ 4, so w_smooth=0.2 costs ≈0.8/step of pure noise: a
1,500-step glide pays ~1,200, a 355-step dive ~280. The glide's position
advantage (~210) is buried — under the stochastic policy PPO *correctly*
preferred the dive, in v9/v10/v11 alike. The optimizer was never the problem.

### v12 — stopped at 6.3M (2026-07-04): 17.8 km best, then stagnation/regression

Per the directive: (1) close terminal position, (2) avoid chattering,
(3) respect path constraints — with the length/noise bias removed.

```
tier 1  r_t   = 100·(log10 d_prev − log10 d_cur)              # position shaping
        r_end = −100·log10(d_end) + 100·[d≤1 km] − 50·[fail/timeout]
tier 2  r_end −= 25 · mean_t ‖a_t − a_{t−1}‖²                 # chatter, EPISODE MEAN
tier 3  r_t  −= 1.0 · min(Σ path_barrier(ratio, soft=0.9), 3) # capped barrier
```

- Chatter as a terminal *mean* is episode-length-neutral: noise shifts every
  policy by the same constant, and the term bites as σ anneals. (w_smooth
  changed meaning: 0.2/step → 25·mean.)
- Path barriers return (softener 0.9, per-step cap 3): free on feasible
  trajectories — the v10-best glide pays ≈0 except grazing Q̇=1.01 — but
  buries dives (8–16× violations), aligning tier 3 with tier 1.
- Warm start from v10-best (43.5 km) with `--action-std 0.2` (new flag): at
  σ=0.2 the stochastic glide dominates the dive from step one, closing the
  remaining noise×length loophole in tier 3 during the anneal.

| knob | v11 | v12 |
|---|---|---|
| w_smooth | 0.2 (per step) | 25 (episode mean, terminal) |
| w_path / soft / cap | – | 1.0 / 0.9 / 3.0 |
| action std at init | ~0.96 (inherited) | 0.2 |
| lr / clip / ent | 5e-5 / 0.1 / 0.003 | 1e-4 / 0.15 / 0.001 |

Smoke check: warm start evals 42.6 km → 32.0 km after just 4k polish steps.

Command: `train_sb3 --stage stage2 --timesteps 10000000 --n-envs 8
--init-model results/stage2_v10/best/best_model.zip
--init-vn results/stage2_v10/ckpt/ppo_vecnormalize_250000_steps.pkl
--lr 1e-4 --lr-decay --clip-range 0.15 --ent-coef 0.001 --action-std 0.2
--seed 0 --outdir results/stage2_v12`

**Outcome.** The three-tier reward held: no dive collapse, feasible smooth
glides throughout. Best **17.8 km @0.35M** (campaign best; trajectory rides
Q̇=0.98, lands on the target track, AoA constant at 40°). Checkpoints improved
to 57 km @3.5M, then stagnated and regressed (352 km by 6.25M) — stopped.
Diagnosis of the 17 km floor: the plain log potential is too flat in the end-
game (17.8→1 km worth only ~125, same as one far-field decade) and the good
basin is shallow against its 50–300 km neighbors, so the policy wanders off.

### v13 — done (2026-07-05): 6.65 km best — the 17 km directive beaten

Reward change (tier 1 only; tiers 2-3 unchanged): the position potential gets
an extra w_near per decade inside d_near:

```
pen(d) = w_pos·log10(d) + w_near·min(log10(d), log10(d_near))
w_pos=100, w_near=300, d_near=50 km    (>50 km unchanged: 100/decade;
                                        <50 km: 400/decade)
```

Same pen() drives shaping and the uniform anchor, so the return stays
monotone in d_end across all endings — the exploit-immunity structure is
preserved, and the glide-vs-dive separation widens (~+800 vs ~−900). Closing
17.8→1 km is now worth 500 + 100 success (was 125 + 100), and the near-target
basin is ~4× deeper against 50–300 km neighbors (anti-drift).

Warm start from v12-best (17.8 km). Smoke check: **11.8 km after 4k steps** —
already under the 17 km directive. Stagnation watch armed at ~3M steps.

Command: `train_sb3 --stage stage2 --timesteps 10000000 --n-envs 8
--init-model results/stage2_v12/best/best_model.zip
--init-vn results/stage2_v12/ckpt/ppo_vecnormalize_250000_steps.pkl
--lr 1e-4 --lr-decay --clip-range 0.15 --ent-coef 0.001 --action-std 0.2
--seed 0 --outdir results/stage2_v13`

**Outcome.** Best **6.65 km @2.85M** (fully feasible: n 1.25 g, Q̇ 0.98,
J_heat 0.952×SCP), after a burst of best-model updates 27→20→10→10→6.6 km
right through the 2–3M window where earlier runs stagnated — the steepened
potential worked. Oscillation persisted (evals bounce 5–150 km; final
LR-decayed policy 34.9 km) but best-model checkpointing captures the dips.
End-game flaw found in the best trajectory: 1,590 s glass-smooth, then a
~20 s bang-bang burst (bank ±27°/s, AoA 40→30→40) trimming the impact point —
free under the episode-MEAN chatter term (20 steps dilute to ~0.6) and
invisible to the path barrier (low q̄ at 25 km). Reward loophole, not noise.

### v14 — stopped at 6M (2026-07-05): reward lever exhausted — observability binds

Directive: land inside 1 km, keep meeting path constraints, no end chatter.

| change | v13 | v14 | why |
|---|---|---|---|
| potential shelves | 100 + 300@50km | **+600 @10 km** | 1,000/decade inside 10 km: 6.6→1 km worth ~820 (was ~330) |
| w_succ | 100 | **300** | real magnet at the 1 km disk |
| burst guard | – | **w_endsmooth=1.0**: per-step ‖Δa‖² while d<50 km | bounded phase → no length tax; dense credit on the exact burst steps; smooth turns stay cheap |
| tiers 2/3 | 25·mean, path 1.0/0.9/cap 3 | unchanged | constraints already met; mean term still guards global chatter |

```
pen(d) = 100·log10 d + 300·min(log10 d, log10 50) + 600·min(log10 d, log10 10)
r_t    = pen(d_prev) − pen(d_cur) − 1.0·‖Δa‖²·[d<50] − 1.0·min(Σbarrier, 3)
r_end  = −pen(d_end) + 300·[d≤1 km] − 50·[fail/timeout] − 25·mean_t‖Δa‖²
```

Warm start from v13-best. Smoke: 6.21 km after 4k steps, value fn healthy.
Stagnation monitor armed at ~3M steps.

Command: `train_sb3 --stage stage2 --timesteps 10000000 --n-envs 8
--init-model results/stage2_v13/best/best_model.zip
--init-vn results/stage2_v13/ckpt/ppo_vecnormalize_2750000_steps.pkl
--lr 1e-4 --lr-decay --clip-range 0.15 --ent-coef 0.001 --action-std 0.2
--seed 0 --outdir results/stage2_v14`

**Outcome.** Monotone drift: 21.5 km at start → 450 km @3M → ~950 km @6M, no
recovery, best model frozen at 22.1 km from step 50k. Stopped. Diagnosis: the
observation normalizes range-to-go by the full 7,800 km mission, so at
single-km scale positions differ by ~1e-4 — numerically invisible to the
policy/value nets. The 1,000/decade shelf makes returns swing ±800 between
observationally IDENTICAL states; that residual is irreducible value noise,
and the steeper potential amplifies the end-game random walk instead of
anchoring it. Reward steepening is exhausted below ~5 km: **the binding
constraint is observability, not incentive.**

### v15 — running (launched 2026-07-05): log-range observation (user-approved)

MDP change (user sign-off given): a 14th observation feature
`log10(max(d_km, 0.1))/4`, appended LAST — per-decade resolution from
7,800 km down to ~100 m, matching the log reward's resolution structure.
Reward = v14's (three shelves, w_succ 300, burst guard) unchanged.

Warm start via obs-expansion surgery (`training/expand_obs.py`): rebuild the
net at 14 inputs, copy all weights, ZERO the new first-layer column → the
expanded model's rollout is bit-identical to v13-best (verified: 6.647 km,
same constraint profile). VecNormalize stats carried over, new dim
self-corrects. Note: pre-v15 models can no longer be rolled out on the
current env without the same surgery (obs dim moved 13→14).

Command: `train_sb3 --stage stage2 --timesteps 10000000 --n-envs 8
--init-model results/stage2_v13/expanded/model_expanded.zip
--init-vn results/stage2_v13/expanded/vecnormalize_expanded.pkl
--lr 1e-4 --lr-decay --clip-range 0.15 --ent-coef 0.001 --action-std 0.2
--seed 0 --outdir results/stage2_v15`

**Outcome** (done 2026-07-06; interrupted at 3.75M by a session restart and
resumed — note: resume RESETS EvalCallback's best-model state and
evaluations.npz). Best **9.1 km**, feasible (Q̇ 0.97). The log-range obs fixed
the v14-style divergence (excursions now recover), but evals oscillated
30–90 km with no plateau, and the diagnostic signature was decisive: the
STOCHASTIC train reward ran ~1,000 above the deterministic eval. Cause: the
steep log well is CONVEX in d, so by Jensen a dispersed landing cloud
out-scores its own center — the reward paid PPO to keep its action noise and
gamble on lucky hits. Log-shaped terminal rewards reward gamblers.

### v16 — running (launched 2026-07-06): anti-lottery reward

Three changes (position tier only; + ent_coef 0 so nothing props the std up):

```
pen(d) = 40·d                                  d ≤ 10 km   (LINEAR well:
         400 + 100·decades + 300·decades@50    d > 10 km    dispersion-neutral)
landing bonus = 300·(1 − d/5 km)^+                          (graded ramp, was
                                                             a d≤1 km cliff)
burst guard: w_endsmooth gated on alt < 35 km               (was range < 50 km)
```

- Linear well: Jensen-neutral — concentrating accuracy is the only way to
  earn; kills the dispersion incentive behind the v15 oscillation.
  Slope 40/km ≈ continues the old 1,000/decade gradient at the boundary;
  potential continuous at 10 km (=400) and unchanged beyond 50 km.
- Graded bonus: dense low-variance pull over exactly the 0-5 km band
  (the 1 km cliff only fired on lucky episodes — itself a lottery term).
  Terminal reward now strictly monotone: +300 at 0 → +200 at 1 km → −364 at 9 km.
- w_succ ramp applies on landing; uniform anchor unchanged on all endings.

| knob | v15 | v16 |
|---|---|---|
| near-field | log, 1,000/decade <10 km | linear, 40/km |
| landing bonus | 300 cliff at 1 km | 300 ramp from 5 km |
| burst-guard gate | range < 50 km | altitude < 35 km |
| ent_coef | 0.001 | 0.0 |

Warm start from v13-best expanded (6.65 km — still the strongest seed).
Smoke: behavior preserved (19.8 km on the smoke eval seed, = v15 smoke).
Cold-start control with this reward: deferred, revisit after v16.

**Outcome** (done 2026-07-07). Best **5.4 km @2.55M** (campaign best; feasible,
n 1.2 g, Q̇ 0.95, chatter gone — clean smooth bank profile). Anti-lottery fix
worked partially: the v15 Jensen inversion is gone in the near field. BUT the
train/eval gap REOPENED — train ep_rew_mean climbs to +200 while deterministic
eval stays pinned ~−500, and the deterministic policy DRIFTED off its own best
(5.4 km @2.55M → 56.6 km @10M). Two root causes, both now nailed down:

1. **σ-floor.** Policy std stuck flat at 0.18 the whole run (linear well gives
   no gradient on σ; ent_coef 0 doesn't shrink it). σ=0.18 on bank-rate ≈
   ±8°/s terminal noise → km-scale landing scatter → the deterministic mode is
   never sharpened and random-walks (best≠final). Load-factor peaks cross 1.0
   in the 2nd half as worse policies push harder (n up to 2.7×, unrewarded).

2. **AoA railed at 40.0° (std EXACTLY 0.0) — the real ceiling.** The vehicle
   flies bank-only; the entire Stage-2 second axis is unused, inherited from
   the warm-start lineage (v10→v16) and held at the ALPHA_MAX clip. The SCP
   physical optimum hits the target to ~4 m using terminal **AoA 28.6°** (and
   nails γ=−10°, ψ=90°). So 1 km is trivially feasible physically — the gap is
   that a bank-only policy cannot match a bank+AoA optimum, and NO position-
   term reshaping fixes that. Position-only optimization has reached the limit
   of bank-only control.

**Assessment / next.** The position-only era is exhausted at ~5 km on the warm
lineage. Decision (2026-07-07): test the cold start FIRST, with NO velocity-
orientation term — isolate whether a fresh policy alone breaks the bank-only
ceiling by discovering AoA modulation. Orientation term stays deferred.

### v17 — stopped at 3M (2026-07-07): cold start diverges — but frees AoA

**Outcome — mixed, decisive.** Two clean findings:

1. **AoA freeze WAS a warm-start scar.** The cold policy modulates AoA:
   std 9.15°, range 12.8–40.0° (vs the entire v10→v16 warm lineage frozen at
   40.0°, std 0.0). So nothing in the reward forbids AoA use — the warm lineage
   was simply stuck in the ALPHA_MAX basin.

2. **But cold start does NOT make it — it diverges on position.** Best-ever
   ~118 km @0.65M, then MONOTONIC divergence to 2,570 km @3M (not oscillation),
   σ still 0.55. Never came within 20× of the warm lineage's 5.4 km. Stopped.

Mechanism: the v16 reward is POLISH-tuned (linear near-field well, graded 5 km
bonus, burst guard as per-step ‖Δa‖² below 35 km alt) — it assumes an already-
good trajectory. A cold policy at 100+ km flying the low-altitude descent with
σ≈0.55 noise pays a large accumulated burst-guard penalty, which it minimizes
by NOT committing to the steep descent → range error grows. The polish terms
that consolidated the warm runs actively repel a cold exploratory policy. The
far-field log shaping alone isn't a strong enough attractor against σ≈0.55.

**Takeaway.** Cold-starting alone doesn't reach the target with the polish
reward; the warm lineage's 5.4 km (v16 best) remains the campaign best. The
frozen-AoA diagnosis is confirmed, so the productive path is to free AoA ON THE
WARM LINEAGE (perturb the AoA action head off the 40° rail from v16-best),
optionally with a cold-appropriate reward (burst guard OFF until late) if a
future cold attempt is wanted. Orientation term still deferred per directive.

### v18 — running (launched 2026-07-07): concave "precision well"

Back on the WARM lineage (v17 cold start diverged). Fall-back = v16-best
(5.4 km). New reward SHAPING — a genuinely different KIND, the mathematically
opposite correction to v15:

- v15 near-field CONVEX (log) → rewards dispersion → gambler (train≫eval).
- v16 near-field LINEAR → dispersion-NEUTRAL → σ has no gradient, stuck 0.18,
  deterministic policy drifts off its best.
- **v18 near-field CONCAVE** (`near_convexity` t=0.5: penalty convex-blended,
  continuity/endpoint preserving) + **concave landing bonus**
  `w_succ·(1−(d/succ_ramp)²)⁺` with succ_ramp=d_lin=10 km. Terminal reward is
  globally concave over 0–9 km (2nd-diff ≤ −3.5) → by Jensen a tight landing
  cluster beats a dispersed one → **PPO gets a downhill on σ for the first time**
  → action noise (hence terminal scatter) driven down. Near-target pull is also
  STRONGER than v16 (61–103/km vs 40/km), so both axes improve.

Everything else = v16 (log far field, uniform anchor, w_smooth mean chatter,
burst guard alt<35 km, path barrier) unchanged. Warm start v16-best; polish
config identical to v16 (lr 1e-4+decay, clip 0.15, ent 0, action-std 0.2) so
the ONLY change is the reward → any effect is attributable to the shaping.
Verified: continuity at 10 km, global concavity, Jensen anti-dispersion at
3/5.4/7 km, env self-test, warm smoke (12.4 km, behavior preserved).
Fallback if it fails: set near_convexity=0 (recovers v16 near field exactly)
+ revert the bonus, warm from v16-best.

DECISIVE 3M signals: (1) does σ fall below v16's stuck 0.18; (2) do train &
eval reward CONVERGE (v15/v16 they diverged); (3) does d_km push below 5.4 km.

**Outcome** (stopped ~5M, killed by a session teardown at 2.25M then resumed;
stopped for v19). The concave well WORKED as designed on 2 of 3 signals:
(1) σ annealed 0.20→0.15 — v16 was stuck flat at 0.18, so the anti-dispersion
curvature gave PPO a σ-downhill for the first time; (2) train & eval reward
CONVERGED (−525 vs −511) — the v15/v16 gambling gap (~700–1000) is gone.
BUT (3) position did NOT break through: d_km oscillated, slowly consolidating
toward ~25 km, never below 5.4 km. Diagnosis: σ=0.15 still = km-scale terminal
scatter, and it anneals too SLOWLY (0.05/3M) to reach precision within 10M.
The reward shape is right; the missing lever is forcing σ down hard — a
hyperparameter move, not a reward one.

### v19 — running (launched 2026-07-08): precision polish (reward + HP tuning)

User opened HP tuning. Warm from v16-best (5.4 km). Keep the v18 concave well
(it fixed the gambling) and add the σ/precision levers v18 lacked:

Reward: strengthen the terminal magnet — w_succ 300→500, succ_ramp 10→6 km
(concentrates the concave bonus near the target; pull 58–129/km in 0–3 km,
still concave/anti-dispersion over 0–5 km, kink safely at 6 km).
Hyperparameters (the real change): **action-std 0.06** (was 0.2 — a good 5 km
policy needs precision not exploration; low σ directly cuts terminal scatter),
lr 5e-5 (was 1e-4), clip 0.1 (was 0.15), batch 1024 (was 512, lower-variance
polish gradients), ent 0. New train_sb3 flags added: --batch-size/--n-steps/
--n-epochs. Smoke: 7.04 km after 4k steps at σ=0.06 (tighter than the 12–19 km
σ=0.2 smokes — low noise reproduces the trajectory precisely).

Bet: a good policy + strong concave near-target pull + LOW σ from the start
realizes the sub-km refinements that km-scale noise was washing out. Fallback:
v16-best (5.4 km) preserved. Trend check armed at ~2M (earlier read).

Command: `train_sb3 --stage stage2 --timesteps 10000000 --n-envs 8
--init-model results/stage2_v16/best/best_model.zip
--init-vn results/stage2_v16/ckpt/ppo_vecnormalize_2500000_steps.pkl
--w-succ 500 --succ-ramp-km 6 --action-std 0.06 --lr 5e-5 --lr-decay
--clip-range 0.1 --ent-coef 0.0 --batch-size 1024 --seed 0
--outdir results/stage2_v19`

**Outcome** (done 2026-07-09; interrupted at 1.75M and resumed — resume again
reset EvalCallback best-tracking). Dips to **5.13 km** around 1.55–1.7M, but
they fell between 250k checkpoints and the post-resume best_model.zip was
overwritten by a 7.9 km policy. Recoverable models: ckpt 1.5M → 12.7 km,
ckpt 1.75M → 11.7 km, best → 7.9 km, final → 12.3 km.

## Campaign close-out (2026-07-09)

**Campaign best: v16-best — 5.39 km, fully feasible** (Q̇ 0.95, n 0.48,
q̄ 0.12, J_heat 0.947×SCP reference), vs the closed-loop SCvx G&C benchmark
42 m. Position error trajectory across the campaign:
226 (v4) → 43.5 (v10) → 17.8 (v12) → 6.65 (v13) → **5.39 (v16)** km.
Comparison slide added to the MBDA presentation (page 15,
`presentation_MBDA/main.tex`) with a corridor-overlay altitude–velocity
figure (`reentry_rl/postprocessing/fig_scvx_vs_rl.py`): the RL trajectory
tracks the SCvx arc almost exactly through the hypersonic phase — the learned
policy recovers the optimal trajectory *shape*; terminal precision is what
model-based optimisation buys.

Operational lesson for future runs: EvalCallback best-model state resets on
--resume — copy best_model.zip aside before resuming, and consider denser
checkpoints (100k) when the eval trend is near a minimum.

## HPO era — algorithm hyperparameters (launched 2026-07-09)

### stage2_polish_hpo — running: Optuna TPE study over the PPO polish

Reward shaping paused at the v19 configuration (concave precision well,
w_succ 500 / succ_ramp 6 km — FROZEN, this study varies only the algorithm).
Every trial warm-starts from **v16-best (5.39 km)** + its VecNormalize; the
512×512 net is fixed by the warm start.

Search space (`hpo_optuna.py`, rewritten for this study): action_std
0.02–0.3 (log), lr 1e-5–3e-4 (log, linear decay per trial), clip 0.05–0.3,
ent_coef 1e-8–5e-3 (log), batch 256–2048, n_steps 1024–4096, epochs 3–15,
1−γ 5e-5–2e-3 (log), λ 0.90–0.99.

Objective: MIN over the trial of the feasibility-gated deterministic landing
error (gate: max constraint ratio ≤ 1.02, else +1000), rollout-scored every
100k steps → MedianPruner (5 startup trials, 400k warmup). **Every feasible
eval < 6 km saves model+stats instantly** (the v19 between-checkpoints
lesson). 40 trials × 1.5M steps, sequential on 8 envs (~1.4 h/trial,
pruning shortens); SQLite study at results/stage2_polish_hpo/study.db —
fully resumable. Smoke-validated (tiny trial reached 7.63 km, saves fired).

Command: `python -m reentry_rl.training.hpo_optuna --trials 40
--steps-per-trial 1500000 --eval-freq 100000 --n-envs 8
--init-model results/stage2_v16/best/best_model.zip
--init-vn results/stage2_v16/ckpt/ppo_vecnormalize_2500000_steps.pkl
--save-below 6.0 --study-name stage2_polish_hpo`

**MISSION SUCCESS (2026-07-10, trial 6): 0.868 km — inside the 1 km disk,
first `is_success` of the campaign.** Independently re-verified rollout:
d=0.868 km, max Q̇ 0.952 / n 0.482 / q̄ 0.124 (feasible with margin),
J_heat 0.948×SCP, 4 bank reversals. Model: `trial006_0.87km_700k.zip` (+vn).

Early-study picture (7 trials): 17.7 (t0) → 2.02 (t1) → 2.31 (t2) → 2.36 (t3)
→ 1.73 (t4) → pruned (t5) → **0.87 (t6)**. The decisive hyperparameters vs
the hand-tuned polish: **learning rate 1–4e-5** (7–10× lower than v16's),
**γ ≈ 0.9994–0.99994** (the v16-era γ=0.999 discounted the terminal anchor to
~20% over a 1,600 s episode — the discount was strangling the terminal
signal), long rollouts (n_steps 2048–4096), many epochs (7–13), tight-ish
clip. action_std proved secondary (2 km reached with σ 0.026 AND 0.258).
Trial 6 recipe: lr 2.47e-5, n_steps 4096, batch 256, epochs 12, λ 0.926,
clip 0.069, ent 2.2e-5, σ 0.021, γ 0.99960. Study continues toward 40 trials.

### Study complete (2026-07-11): **best 0.273 km — trial 23**

41 trials (11 complete / 29 pruned / 1 stale-failed), ~28 h total across two
legs (one session-restart interruption; SQLite resume worked as designed).
Verified rollout of `trial023_0.27km_300k.zip`: **d = 0.273 km, is_success,
max Q̇ 0.952 / n 0.482 / q̄ 0.124, J_heat 0.948×SCP (1,016 MJ/m²), 3 bank
reversals**, t_f 1,615 s, V_f ~ 300 m/s.

Winning recipe (trial 23): lr 1.41e-4, n_steps 4096, batch 256, epochs 5,
λ 0.908, clip 0.079, ent 3.6e-4, σ 0.021, γ 0.99965.
Optuna param importances: action_std 0.24 ≈ lr 0.21 ≈ n_steps 0.20 ≈
batch 0.19 ≫ λ 0.12 ≫ epochs/ent/clip/γ (<0.02 each). Within the searched
ranges the precision drivers were small σ + long rollouts + small batches;
both low-lr/many-epochs and high-lr/few-epochs styles reached ~1 km.

Sub-km configs found: t23 0.27, t30 0.71, t6 0.87 (plus 1.19/1.31/1.50).
**Final campaign arc: 226 km (v4) → 5.39 km (v16, reward design) →
0.273 km (HPO) vs SCvx closed-loop 42 m.** Presentation slide + comparison
figure updated to the 0.27 km trajectory (deck compiles, 39 pages).

Command: `train_sb3 --stage stage2 --timesteps 10000000 --n-envs 8
--init-model results/stage2_v16/best/best_model.zip
--init-vn results/stage2_v16/ckpt/ppo_vecnormalize_2500000_steps.pkl
--lr 1e-4 --lr-decay --clip-range 0.15 --ent-coef 0.0 --action-std 0.2
--seed 0 --outdir results/stage2_v18`

--- v17 superseded launch config (ran 0→3M before stop) ---

Fresh random init (no warm start), v16 reward and 14-dim log-range obs
unchanged. Config reverts from polish to exploration: default std 1.0 (was
0.2), ent 0.005 (was 0), clip 0.2 (was 0.15), lr 3e-4+decay (was 1e-4);
gamma 0.999, gae 0.95, net 256×256, n_steps 2048, batch 512, epochs 10 — all
DEFAULT_PPO. Purpose: does a fresh policy use the AoA axis (SCP optimum:
terminal AoA 28.6°) instead of railing it at 40° like the warm lineage?
Decisive signals at the 3M check: d_km trend, whether σ anneals, and
AoA std > 0 in the eval trajectory.

Command: `train_sb3 --stage stage2 --timesteps 10000000 --n-envs 8
--lr-decay --seed 0 --outdir results/stage2_v17`  (no --init-model /
--action-std / --ent-coef / --clip-range → DEFAULT_PPO cold start)

Command: `train_sb3 --stage stage2 --timesteps 10000000 --n-envs 8
--init-model results/stage2_v13/expanded/model_expanded.zip
--init-vn results/stage2_v13/expanded/vecnormalize_expanded.pkl
--lr 1e-4 --lr-decay --clip-range 0.15 --ent-coef 0.0 --action-std 0.2
--seed 0 --outdir results/stage2_v16`

## Evaluation-harness era (2026-10): re-measuring the campaign

Branch `point1-evaluation-harness`. Three measurement/physics issues found while
building a fixed, dispersion-aware evaluation protocol. Numbers below are
deterministic rollouts of the HPO-best policy `trial023_0.27km_300k.zip`.

| change | nominal d_km | max path ratio | note |
|---|---|---|---|
| as reported (July) | 0.273 | 0.952 | first 1 s step below 25 km, bounds after each RK4 substep |
| exact 25 km crossing | 0.418 | 0.952 | crossing located inside the substep (`exact_terminal`) |
| + saturated attitude integrator | **259.4** | **2.02** | `rate_saturation`: AoA/bank bounded inside every RK4 stage |

1. **Terminal measurement.** The handover state was read at the first step
   below 25 km. At V_f ~ 300 m/s one step is ~0.3 km of travel, the size of the
   result. Now located exactly (secant iteration inside the substep).
2. **AoA bound leak (decisive).** alpha <= 40 deg was enforced only after each
   RK4 substep, so the stages evaluated the aero at alpha + c*h*alpha_dot, up to
   5 deg beyond the limit. The whole v10 -> v16 -> HPO lineage commands
   alpha_dot = +3..+10 deg/s (raw action +0.22..+0.64) into the limit: the
   "AoA railed at 40 deg" was a hidden AoA modulation at ~40.5-41.6 deg
   effective. With the bound enforced inside the stages the HPO-best policy
   lands 259 km off and violates q-bar/n (ratio 2.02); v16-best goes 5.5 -> 317 km.
   The reward-design lessons of v1-v19 are unaffected (they are about reward
   structure), but every terminal-accuracy number of the campaign was obtained
   with AoA authority beyond the vehicle envelope and the SCvx bounds, and must
   be regenerated.
3. **Open-loop behaviour.** Even with the legacy dynamics the HPO-best policy is
   not robust: on the realistic one-at-a-time sweep (0.1x VISTA sigma, +-3 sigma)
   the median miss is 11.3 km (max 389 km, 21 % within 1 km); at VISTA scale the
   median is 115 km (max 3,542 km, 74 % feasible). Terminal angles are
   dfpa ~ -15 deg, dpsi ~ -70 deg in every case (never rewarded).
   Results: `results/baseline_eval/hpo_trial023/eval_legacy_integrator/`.

Legacy behaviour stays reproducible: `--legacy-terminal --legacy-integrator`
(train_sb3, hpo_optuna, evaluate_policy) gives 0.2728 km again.

### v20 — running (launched 2026-10-07): full terminal state, corrected dynamics

Branch `point2-full-terminal-state`. First run with the saturated attitude
integrator and the exact 25 km crossing, so its numbers are not comparable to
v1-v19/HPO (see above). Objective: the SCvx handover set — position within a
1 km radius, h/gamma/psi within 0.1 % (25 m, 0.01 deg, 0.09 deg) — inside the
path constraints.

- Warm start: HPO trial 23 expanded to the v20 observation layout (22 features,
  `results/stage2_v20_init/expand_report.json`; verified same trajectory). Under
  the corrected dynamics it starts at 259 km, dfpa -11.1 deg, dpsi -71.4 deg,
  q-bar/n ratio 2.0 — the run must first re-acquire position.
- Reward: v19 position tiers (w_succ 500, succ_ramp 6 km) + tier 1b FPA/heading
  channels under the `v20` curriculum (6 levels; level 0: c = 0.25, ramps 20 deg /
  90 deg, angle bonuses 250 each; advance on 2 consecutive passing validations
  after >= 300k steps at the level).
- PPO: HPO trial-23 settings (lr 1.41e-4 decayed, clip 0.079, ent 3.6e-4,
  n_steps 4096, batch 256, epochs 5, gamma 0.99965, lambda 0.908), action std
  0.021 (bank) / 0.1 (AoA rate — AoA exploration re-opened).
- Selection: ValidationCallback on S0_nominal, score_mode full
  (||(d/1 km, dfpa/1 deg, dpsi/5 deg)||, +1000 infeasible), every 100k steps.

Command: `train_sb3 --stage stage2 --timesteps 10000000 --n-envs 8
--init-model results/stage2_v20_init/model_expanded.zip
--init-vn results/stage2_v20_init/vecnormalize_expanded.pkl --obs-version v20
--action-std 0.021,0.1 --lr 1.41e-4 --lr-decay --clip-range 0.079
--ent-coef 3.6e-4 --w-succ 500 --succ-ramp-km 6 --curriculum v20
--val-set S0_nominal --score-mode full --eval-freq 100000 --save-below 5
--seed 0 --outdir results/stage2_v20`

**v20 outcome — stopped at 0.6M (2026-10-07):** PPO updates far outside the
trust region: approx_kl 2-14 per update (healthy ~0.01-0.05), clip_fraction
0.4-0.6, train return falling -773 -> -916, nominal d stuck at 250-285 km. Cause:
the HPO action std 0.021 was tuned to POLISH a converged trajectory; with that
std a 0.06 shift of the bank-rate mean is already KL ~ 4, so every lr-1.41e-4
update overshoots by ~100x while the policy has to re-learn its trajectory
under the corrected dynamics. (Feasibility was regained by 0.2M.)

### v20b — running (launched 2026-10-07 11:17): v20 with re-acquisition settings

Identical to v20 except the PPO/exploration settings, back to the v12-v16 recipe
that re-acquired position from 43 km: action std 0.1 (bank) / 0.15 (AoA rate),
lr 1e-4 decayed, clip 0.15, ent 0, plus target_kl 0.05 (new --target-kl).
First updates: approx_kl 0.002-0.006, clip_fraction 0.04-0.06, return
-813 -> -757 by 0.16M.

Command: as v20 with `--action-std 0.1,0.15 --lr 1e-4 --lr-decay --clip-range 0.15
--ent-coef 0.0 --target-kl 0.05 --outdir results/stage2_v20b`

**v20b outcome — stopped at 1.0M (2026-10-07):** PPO healthy (approx_kl
0.002-0.006) and AoA in use while infeasible (std ~7 deg), feasible from 0.5M
(peak ratio 3.6 -> 0.99) but AoA then locked again and the policy traded
position for heading: d 308 -> 349 km while dpsi 77 -> 65 deg, dfpa ~ -14 deg.
Cause: reward balance of curriculum level 0. It was sized for a warm start near
the target (0.4 km before the dynamics fix), but the start is now ~300 km out,
where the log position potential slope is ~0.15/km against ~4.3/deg for the
level-0 heading bonus (1 deg of heading ~ 30 km of position).

### v20c — running (launched 2026-10-07 12:02): position re-acquisition first

As v20b, but curriculum preset `v20r` (level 0 = angle channels OFF, advance
once d <= 5 km on 2 consecutive validations after >= 300k steps; then the six
v20 angle levels) and warm-started from v20b best (feasible, 308.6 km @0.6M;
copied to results/stage2_v20c_init/). First validation: 314 km feasible,
approx_kl ~0.003.

Command: as v20b with `--init-model results/stage2_v20c_init/model.zip
--init-vn results/stage2_v20c_init/vecnormalize.pkl --curriculum v20r
--outdir results/stage2_v20c`
