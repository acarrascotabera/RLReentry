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

**v20c outcome — stopped at 2.0M (2026-10-07):** position-only re-acquisition
worked at first (314 -> 59.6 km by 1.0M) but the policy then alternated
between two regimes: AoA modulated (std 6-7 deg) -> 60-106 km but q-bar 1.4-3.5x
the limit; AoA locked at 40 deg -> feasible but 100-180 km. The 2M rollout dives
the last 17 s with alpha = 5 deg and bank = -85 deg, reaching 25 km at
V = 1772 m/s (q-bar 3.5x). The per-step barrier capped at 3 priced that at
<= 51 units, cheaper than the position it bought.

### v20d — running (launched 2026-10-07 13:30): peak-violation charge

As v20c plus `--set-weight w_peak=1000`: -1000 * max(0, episode peak ratio - 1)
charged once on every ending (5 % overshoot = -50, the 3.5x dive = -2500).
Warm start: v20c best (0.9M, 73.6 km, peak ratio 1.007), copied to
results/stage2_v20d_init/.

**v20d interim (1.0M, still running):** feasible throughout (peak ratio ~0.98-0.99,
also under exploration noise) but drifting away: d 88 -> 157 -> 246 km, AoA
locked. Reward bookkeeping of the deterministic rollouts: 88 km -> position cost
1408 + barrier 52 + peak 5 = 1465; 246 km -> 1497 + 25 + 0 = 1522. The reward does
prefer the near-target policy, but by ~60 units for 160 km: the log far-field
potential is ~0.5/km out here, below the return noise and below the expected peak
penalty that exploration triggers near the constraint boundary. Root cause of
v20b-v20d: the position potential was designed for a warm start within a few km
of the target; under the corrected dynamics the start is 100-300 km out.

**v20d outcome — done (10M, 2026-10-07 20:56):** best = 0.1M (88 km, peak 1.005);
final 515 km, dfpa 14.2 deg, dpsi 94.9 deg, feasible. Collapse at ~1.8M (value
loss ~1e6, explained variance 0.3 for ~1M steps; dive to ~5,000 km at peak ratio
12): the 1000-weight peak charge made returns too variable for the critic. It
then spent ~6.5M steps recovering to ~500 km; flat over the last 1M (return
slope +0.4/M). Curves: results/curves/ (plot_chain_curves). At equal cumulative
chain steps the v20 chain (74 km at 1.5M) matched the old chain (60-300 km over
1-3M; 10 km first at ~3.4M, 0.27 km at ~6.3M + HPO).

### Monte Carlo on the SCvx campaigns' exact scenarios (2026-10-08)

Scored with identical metrics (scvx_benchmark.py, evaluate_policy.py); SCvx is full
closed-loop G&C, RL is 3-DOF guidance with an ideal attitude loop.

| method | campaign | median d [km] | p95 d [km] | median dfpa/dpsi [deg] | feasible | <= 1 km | VISTA tol | 0.1 % tol |
|---|---|---|---|---|---|---|---|---|
| SCvx G&C | gc_ic_tc (1001) | 0.21 | 307 | 0.30 / 0.62 | 39.5 % | 19.1 % | 18.7 % | 0.8 % |
| SCvx G&C | gc_models (501) | 0.33 | 102 | 0.54 / 1.08 | 33.7 % | 15.0 % | 14.2 % | 0.6 % |
| RL v20d best | gc_ic_tc | 256 | 655 | 12.6 / 71.1 | 19.0 % | 0 | 0 | 0 |
| RL v20d best | gc_models | 741 | 2,701 | 11.1 / 70.6 | 16.8 % | 0 | 0 | 0 |
| RL HPO t23 (corrected) | gc_ic_tc | 387 | 1,360 | 13.9 / 70.9 | 30.1 % | 0 | 0 | 0 |
| RL HPO t23 (corrected) | gc_models | 1,459 | 2,598 | 16.7 / 71.2 | 29.1 % | 0 | 0 | 0 |

The 0.1 % criterion is stricter than SCvx achieves (its nominal ends at
dfpa -0.097 deg, load factor 1.3 % over the limit).

### v21 — running (launched 2026-10-08 18:24): long horizon, bounded returns, steep far field

20M steps (lr decay over the full horizon), warm start from v20c best (0.9M,
73.6 km, critic healthy), curriculum v20r from the position-only level.
Reward changes vs v20d: `w_far = 2` per km between 10 and 500 km (88 -> 250 km
now costs 369 units instead of 45; near field unchanged) and `w_peak = 200`
(was 1000: 10 % overshoot -20, the 3.5x q-bar dive ~-550 incl. barrier, more than
the ~400 that 100 km of position is now worth). Reward normalization not used:
the warm-start critic was fitted on raw returns. PPO as v20b-v20d (std 0.1/0.15,
lr 1e-4 decayed, clip 0.15, target_kl 0.05). 378 steps/s -> ~14.7 h.
First validation: 118 km, approx_kl 0.002, explained variance 0.96.

Command: `train_sb3 --stage stage2 --timesteps 20000000 --n-envs 8
--init-model results/stage2_v21_init/model.zip --init-vn results/stage2_v21_init/vecnormalize.pkl
--obs-version v20 --action-std 0.1,0.15 --lr 1e-4 --lr-decay --clip-range 0.15 --ent-coef 0.0
--target-kl 0.05 --w-succ 500 --succ-ramp-km 6 --set-weight w_peak=200 --set-weight w_far=2
--curriculum v20r --val-set S0_nominal --score-mode full --eval-freq 100000 --ckpt-freq 500000
--save-below 5 --seed 0 --outdir results/stage2_v21`

**v21 outcome — done (20M, 2026-10-09 08:34).** Report: `results/stage2_v21/diag/report/`
(learning_curves.png, regimes.png, report.json); chains: `results/curves/`.

| | nominal d | realistic sweep median | realistic S3 median (IQM, 95 % CI) | VISTA S3 median | SCVX ic_tc / models median | feasible (realistic S3) |
|---|---|---|---|---|---|---|
| v21 best (15.1M) | **16.2 km** | 21.9 km | 103 km (109, [99, 121]) | 782 km | 264 / 795 km | 39 % |
| v21 final (20M) | 98.6 km | 102 km | 185 km (194, [182, 208]) | — | — | 69 % |
| v20d best | 88.0 km | 87.6 km | 114 km (116, [109, 124]) | 702 km | 256 / 741 km | 15 % |
| SCvx G&C | 0.016 km | — | — | — | 0.21 / 0.33 km | — |

Training analysis:
1. Longer training helped, slowly: feasible d < 100 km at 2.1M, < 50 km at 2.3M, < 20 km
   only at 15.1M (16.2 km, best under the corrected dynamics); never < 10 km; drifted to
   99 km by 20M as the lr decayed. Curriculum never left the position-only level.
2. Optimisation was healthy: approx_kl median 0.0023 (no target_kl hits), explained
   variance ~1 after 3M, and the training return tracks the deterministic landing error
   (Spearman -0.85). No reward/objective mismatch and no collapse; the late drift is
   PPO's own objective falling.
3. **AoA is effectively disabled.** In the 12.5M, best and final policies alpha sits at
   40 deg for 98.5-100 % of the steps; the raw AoA-rate mean at the limit is +0.38..+0.43
   (~3 std into the bound), so a negative rate is explored with p ~ 0.3 %/step, and the
   saturated integrator gives the pinned command zero effect and zero gradient. The
   "modulated" regime is only a terminal AoA dive (~1.5 % of the flight). v21 flies
   bank-only; SCvx modulates AoA (28 deg at the handover), and the July 0.27 km
   relied on the hidden >40 deg AoA channel.
4. Near-target feasible rollouts ride the constraint boundary (peak ratio 0.99-1.00):
   the fixed penalty weights decide the margin (19 % of evaluations over the limit).
5. Nominal-only training does not transfer: the nominal gain (88 -> 16 km) leaves the
   realistic-dispersion median at ~100 km and the SCvx scenarios at 264-795 km.

### v21 resumed — running (2026-10-09 16:14): +10M steps (to 30M), LR warm restart

`--resume results/stage2_v21 --timesteps 10000000 --lr-decay --lr-restart`: continues
from the 20M checkpoint with the same reward/curriculum/validation, the learning rate
restarted at 1e-4 and decayed to 0 over the new 10M (a plain resume would continue
the old decay from ~3e-5). Best-model state preserved (15.1M, 16.2 km). Log:
train_log_resume1.txt; segment record: resume_log.json. 407 steps/s -> ~6.8 h.

**v21 resumed outcome — done (30M, 2026-10-09 23:42).** New best **10.1 km** nominal
(29.2M, peak ratio 0.98, feasible); final 11.5 km; last 1M steps all 9.6-15 km. The
warm-restart segment is the cleanest learning of the campaign: d fell ~300 -> 10 km over
22-30M (~1 decade per 5.5M steps) with the training return rising monotonically to
~+1000 and no late drift. Still bank-only (alpha at 40 deg in every step), angle
channels never active (5 km gate not reached): dfpa ~14 deg, dpsi 56-84 deg.

| | nominal | realistic sweep median | realistic S3 median (IQM, 95 % CI) | realistic S3 <= 10 / 20 km | VISTA S3 | SCVX ic_tc / models | feasible (real. S3) |
|---|---|---|---|---|---|---|---|
| v21 best 15.1M | 16.2 km | 21.9 | 103 (109, [99, 121]) | — | 782 | 264 / 795 | 39 % |
| v21 best 29.2M | **10.1 km** | **16.0** | 108 (110, [100, 120]) | 1.6 % / 9.4 % | 946 | 260 / 983 | 48 % |
| v21 final 30M | 11.5 km | 16.0 | 96 (108, [98, 119]) | — | — | — | 40 % |

Nominal precision improves with training; robustness does not (nominal-only training):
the realistic-dispersion IQM is unchanged at ~109 km. Largest one-at-a-time
sensitivities (realistic +-3 sigma): C_L 310 km, C_D 294 km, entry speed 101 km.

### v22 — running (launched 2026-10-10 17:09): +10M steps, LR restart, training dispersions

Warm start from the v21 final model (30M, nominal 11.5 km; copied to
results/stage2_v22_init/). Same reward (w_far 2, w_peak 200, w_succ 500 / 6 km) and
curriculum v20r (restarts at the position-only level), PPO lr 1e-4 decayed to 0 over the
10M (the LR warm restart), clip 0.15, target_kl 0.05, learned action std kept.
New: **domain randomization** — every training episode draws a realistic dispersion
(`--train-dispersion ic,models --train-sigma-scale 0.1`: entry state + mass/C_L/C_D/density
at 0.1x the VISTA sigmas, the distribution of S2_val_x0.1 / S3_test_ic_models_x0.1).
Selection on **S2_val_x0.1** (nominal + 15 realistic dispersions, p90 of the full-error
score) every 250k steps (16 rollouts per evaluation). New run directory because the
selection criterion changed (v21 best.json is a nominal-only score). 414 steps/s.

**v22 outcome — done (10M, 2026-10-11 01:24).** Best (selection on S2_val_x0.1) at 9.75M:
validation median 26 km, 75 % of cases within limits. Training return -65 -> +600 monotone,
policy std annealed 0.087 -> 0.072, approx_kl median 0.003, explained variance >= 0.99.
Figures: results/stage2_v22/diag/report/, results/curves/mc_comparison.png.

| | nominal | realistic S3 median (IQM, 95 % CI) | <= 20 / 50 km | peak ratio p95 | feasible | VISTA S3 | SCVX ic_tc / models |
|---|---|---|---|---|---|---|---|
| v21 best (nominal training) | 10.1 | 108 (110, [100, 120]) | 9 % / 27 % | 2.04 | 48 % | 946 | 260 / 983 |
| **v22 best** (dispersion training) | 21.4 | **24.0 (25.1, [23.9, 26.7])** | 31 % / 83 % | **1.11** | **68 %** | 494 | 155 / 534 |
| v22 final | 16.0 | 20.3 (20.9, [20.1, 22.0]) | 48 % / 88 % | 1.23 | 41 % | 501 | — |
| SCvx G&C | 0.016 | — | — | — | — | — | 0.21 / 0.33 |

Training with realistic dispersions cut the realistic-test IQM 4.4x (110 -> 25 km) and
the worst-case constraint excess (p95 2.04 -> 1.11), at the cost of nominal precision
(10 -> 16-21 km). It also halved the error on the stress-level VISTA/SCvx scenarios it
never saw, which remain 2-3 orders above SCvx. The v22 CDFs start at ~14 km: no
realistic case lands closer, a precision floor of the bank-only policy (alpha at 40 deg
throughout). Best (selected for feasibility) vs final (more accurate, less feasible)
is the position / constraint-margin trade-off again.

### v23 — running (launched 2026-10-11 23:07): points B + C

Warm start from the v22 final model (10M; nominal 16.0 km, realistic S3 median 20.3 km,
41 % feasible), +10M steps with the LR restarted at 1e-4, same dispersion training
(ic+models, 0.1x VISTA) and selection (S2_val_x0.1, full score, every 250k).
- **B — angles earlier:** curriculum `v23`. The position-only level hands over once
  >= 50 % of the validation cases land within 30 km (and within the 2 % constraint
  margin); each angle level carries its own position criterion (30, 25, 20, 10, 5, 2 km)
  so the angle tolerances can tighten while the landing error is still tens of km.
- **C — PID-Lagrangian path constraints:** `--lagrangian` (training/lagrangian.py).
  E[max(0, peak_k/limit_k - 1)] <= 0.005 for heat rate, dynamic pressure and load
  factor, one multiplier per constraint on the episode peak excess (replaces w_peak),
  updated after every rollout: Kp 2000, Ki 1000, Kd 0, lam_init 200 (= the previous fixed
  w_peak), lam_max 2000. The per-step barrier is kept.
First rollout: load-factor and heat-rate overshoot in 12 % of the training episodes ->
lam_n 208, lam_Qdot 192.

**v23 outcome — done (10M, 2026-10-12 06:25).** Best by score at 8.75M (all 16 validation
cases within 1.9 % of the limits); final 5.0 km nominal. Training return 511 -> ~930,
approx_kl median 0.0046, explained variance >= 0.97, policy std 0.072 -> 0.063.
Lagrangian: lam_n hit the 2000 cap at 0.75M and stayed; load-factor mean excess fell
from 0.25 to 0.005-0.03; a q-bar violation at 2.3M drove lam_qbar to ~890 and it decayed
back over ~1.5M once satisfied; heat rate stayed within budget (lam ~190).
Curriculum: level 1 (angles on) from 1.75M, never level 2 (needs |dpsi| <= 45 deg on
half the cases; it reached 45-60 deg).

| | nominal | realistic S3 median (IQM, 95 % CI) | <= 10 / 20 km | feasible & <= 10 km | peak ratio p95 | feasible | SCVX ic_tc / models | median dfpa / dpsi |
|---|---|---|---|---|---|---|---|---|
| v22 best | 21.4 | 24.0 (25.1, [23.9, 26.7]) | 0 % / 31 % | 0 % | 1.105 | 68 % | 155 / 534 | 14.0 / 67.7 |
| **v23 best** | 9.2 | **9.4 (9.9, [9.5, 10.5])** | 57 % / 84 % | 50 % | **1.017** | **79 %** | 89 / 436 | 14.5 / 60.3 |
| **v23 final** | **5.0** | **7.5 (7.8, [7.5, 8.1])** | **71 % / 89 %** | **57 %** | 1.066 | 75 % | **64** / 291 | 17.0 / 57.0 |
| SCvx G&C | 0.016 | — | — | — | — | — | 0.21 / 0.33 | 0.30 / 0.62 |

B + C together: realistic-test IQM 25 -> 8-10 km (2.5-3x) while the constraint tail
tightened (p95 peak ratio 1.105 -> 1.017 for the best model; load-factor violations 19 % ->
8 %). The ~14 km floor of v22 is gone (7 % of realistic cases within 5 km for the final
model), so it was not a hard limit of bank-only flight. Heading improved 68 -> 57-60 deg;
FPA did not (14 -> 15-17 deg; the final model dives more steeply). AoA still at 40 deg.

### v23 resumed — running (2026-10-12 12:23): +10M steps (to 20M), warm restart

`--resume results/stage2_v23 --timesteps 10000000 --lr-decay --lr-restart` with the same
curriculum (v23, continues at level 1), PID-Lagrangian (multipliers restored: lam_n 2000,
lam_Qdot 191, lam_qbar 190), dispersion training and selection (best-so-far score 54.0
from 8.75M carried over). Before resuming, an anti-windup clamp was added to the PID
integral (I_k <= (lam_max - lam0)/Ki = 1.8; the load-factor integral had wound up to 9.0).
Log: train_log_resume1.txt; segment: resume_log.json. ~500 steps/s.

**v23 resumed outcome — done (20M, 2026-10-12 20:12).** Best by score at 16.5M (all cases
within 1.9 %); final 3.26 km nominal. Return median over the last 2M ~1320 (max 1490), still
rising; approx_kl median 0.0044; explained variance >= 0.97. Validation median 3.7-4.6 km
over the last 1M. Load factor still the binding constraint: mean training excess 3.5 %
(0-10M), 1.2 % (10-15M), 2.0 % (15-20M) vs the 0.5 % budget, lam_n mostly at the 2000 cap.
The extra precision comes with a steeper terminal dive (final model gamma_f ~ -31 deg vs
-10 deg target), which is also what drives the load-factor peaks. Curriculum stayed at
level 1 (heading 52-61 deg off, needs 45).

| v23 model | nominal | realistic S3 median (IQM, 95 % CI) | <= 5 / 10 km | feasible & <= 5 km | peak p95 | feasible | SCVX ic_tc median (<= 10 km) | SCVX models |
|---|---|---|---|---|---|---|---|---|
| 8.75M best | 9.25 | 9.4 (9.9, [9.5, 10.5]) | 0 % / 57 % | 0 % | 1.017 | 79 % | 89 (—) | 436 |
| 10M final | 5.01 | 7.5 (7.8, [7.5, 8.1]) | 7 % / 71 % | 6 % | 1.066 | 75 % | 64 (7 %) | 291 |
| 16.5M best | 4.82 | 6.3 (7.6, [6.9, 8.5]) | 29 % / 66 % | 27 % | 1.071 | 79 % | 101 (12 %) | 372 |
| **20M final** | **3.26** | **4.2 (4.2, [4.0, 4.3])** | **71 % / 89 %** | **57 %** | 1.162 | 75 % | **39 (18 %)** | **231** |

The second warm-restart segment halved the realistic-test error again (IQM 7.8 -> 4.2 km)
and the SCvx-scenario median (64 -> 39 km); no case within 1 km yet, and the constraint
tail of the final model widened (p95 1.07 -> 1.16).

## Reformulation era (2026-10-13): energy decisions, absolute attitude targets, from scratch

### v24 — stopped at 3.69M (2026-10-13 12:22 - 16:50): cold start, 20M decisions planned

No warm start. New formulation (commits 32e542e, 4ee0da6):
- **Energy as the independent variable** (`--step-mode energy --energy-steps 800`): one
  decision per level of a grid uniform in sqrt(normalized energy-to-go), integrated in 1/3 s
  substeps (boundary at substep resolution, handover crossing exact), steps capped at 10 s;
  ~817 decisions per nominal episode, ~190 in the last 200 s. Fixed-dE steps were rejected:
  dissipation ~ D*V, so they leave only ~87 decisions in the last 200 s. Heat load, barrier
  (time-integrated) and constraint peaks accumulated per substep. Constant-target check:
  identical trajectory to time stepping (heat load to 8e-8).
- **Absolute attitude targets** (`--action-mode target`): bank target in +-85 deg and AoA target
  in [5, 40] deg, tanh-mapped from a +-3 action box and tracked by the env at the actuator
  limits (45 / 15 deg/s). Holding 40 deg AoA is interior to the policy output (no dead zone).
  WB001 aero: max L/D 2.01 at 17 deg, 1.07 at 40 deg (max lift/drag); SCvx holds 40 deg until
  t ~ 1325 s (V ~ 2 km/s) and lands at 28.6 deg.
- **obs v24** (28): v20 + energy-to-go (linear, sqrt, log), range per energy-to-go, measured
  drag acceleration and L/D.
- **Curriculum v24** (loosened): first angle level passes at |dfpa| <= 20 deg, |dpsi| <= 65 deg
  (v23: 10 / 45, never passed); explicit tightening thresholds per level.
- PID-Lagrangian with cap 4000 and a 2M-decision warm-up at lam = 200 (a cold policy violates
  heat rate by ~2x in every episode); dispersion training (ic+models, 0.1x VISTA) from the start.
- PPO: 512x512, lr 3e-4 decayed, n_steps 2048 x 8, batch 512, 5 epochs, gamma 0.9995,
  lambda 0.95, clip 0.2, ent 0.005, target_kl 0.05, action std 0.5.
Throughput ~150-160 decisions/s at start (random policies loft: long, slow, capped steps).

**v24 outcome — stopped at 3.69M (local optimum, no escape).** Validation median 2,153 ->
1,360 km (0.75-1M), then 3,133 km (1.25M) and 4,200-4,400 km from 2M on; feasible 0 %.
The return plateaued at -2,410 from 1.0M to 2.0M with explained variance -> 0.999 (a
converged, deterministic mode), then stepped to -5,300 when the Lagrange warm-up ended and
stayed there (approx_kl 0.001-0.004; action std drifted 0.50 -> 0.89 under ent 0.005).
Heat-rate excess 0.71-0.77 on 100 % of the episodes with lam_Qdot pinned at the 4,000 cap
from ~2.05M (Kp*e alone moved it 200 -> ~1,700 in one rollout).

Diagnosis (3M checkpoint, nominal; open-loop constant targets):
- The policy holds AoA at 39-40 deg (pre-tanh 2.2-2.7) but banks ~80 deg right from entry
  (pre-tanh ~2.5), meets the heat pulse at 64 km instead of ~75 km (peak Qdot 1.82x) and
  reaches 25 km at t = 1,015 s, ~4,200 km short. The geometry needs a right turn (entry at
  0/0 heading north, target 70N/12E heading east), so the sign is right, the magnitude not.
- Heat rate is nearly binding at the best attitude: constant (alpha, bank) = (40, 0) gives
  peak Qdot 0.984x (overshoots ~430 km), (35, 0) 1.106x, (30, 0) 1.263x, (22.5, 0) 1.597x,
  (40, 60) 1.373x. Only lift-up at 40 deg through the pulse (t ~ 225 s, h ~ 75 km) is feasible.
- The reward is not the problem: with the 1M-2M weights (lam = 200) and the policy's own
  exploration noise, a noisy (40, 0) flight scores -1,980 (958 km) against -2,401 (3,998 km)
  for the 2M policy; with the current multipliers -2,426 vs -5,411. It is a local optimum
  in a flat far field: the position anchor is 1,760 at 500 km and 1,853 at 4,300 km (the
  linear w_far term stops at d_far = 500 km), so nothing pulls a policy that lands
  thousands of km off back.
- Why the cold start went there: in rate mode the zero action HOLDS the initial attitude
  (bank 0, alpha 40 = the near-feasible lift-up flight, ~430 km off), so every earlier
  policy started in the near-field basin — and v21 was not a true cold start either (it
  was initialized from stage2_v21_init, an expanded earlier model). In target mode the zero
  action is (bank 0, alpha 22.5 deg): a massive overshoot at 1.6x heat rate; correcting it
  drove the bank to the right saturation and past the target onto the flat short side.
- Credit assignment: the per-constraint Lagrangian charge was paid at the episode end on the
  episode PEAK, which is not in the observation (non-Markov return; ~600 decisions between
  the heat pulse and the charge).

### v24b — running (launched 2026-10-13): cold start with the v24 fixes, 20M decisions

Same formulation as v24 (energy decisions, absolute attitude targets, v24 curriculum,
dispersion training, selection) plus:
- **Initial action mean** (`--init-action-mean 0,2.5`): the action head's bias puts a fresh
  policy at bank 0, AoA 39.8 deg — the attitude-hold start of rate mode (lift-up, peak Qdot
  ~1.0x, ~430 km off). The policy remains free over the full [5, 40] deg / +-85 deg range.
- **Incremental peak charge** (`--peak-charge incremental`): w_peak and lam_k are paid at the
  decision where the running peak excess grows; it telescopes to the terminal charge
  (verified: identical return, difference 0.0) but lands at the heat pulse.
- **obs v25** (31): v24 + running peaks of Qdot, qbar, n over their limits (the charge is now
  a function of the observed state).
- **Lagrange ramp** (`--lag-rate 50`): multipliers rise at most 50 per rollout (cap 4000 in
  ~1.2M decisions) with conditional integration (no windup during the ramp); warm-up 1M.
- **ent_coef 0** (as v21-v23; the v24 std growth 0.50 -> 0.89 was entropy-driven).
- **Far field**: linear w_far = 2/km extended to d_far = 1,500 km (covers the noisy lift-up
  start at ~1,000 km).
Legacy paths unchanged (HPO-best regression return -660.1399573458849 identical to HEAD).

**v24b progress (2026-10-14 00:00, 6M).** Validation median 412 km (0.25M) -> 300-460 km
plateau (to 2.75M) -> 85 km (3.25M) -> 16 km (4.75M, curriculum level 0 -> 1) -> 4.2 km
(6M, level 1 -> 2; score 25.2). Monte Carlo of the 6M best (snap_best_6M, eval_mc_best_6M):

| model | nominal | realistic S3 median (IQM, 95 % CI) | <= 5 / 10 km | feasible & <= 5 km | peak p95 | feasible | viol. Qdot / n | median abs dfpa / dpsi | AoA mean (std, min) |
|---|---|---|---|---|---|---|---|---|---|
| v23 20M final | 3.26 | 4.17 (4.15, [4.01, 4.33]) | 71 % / 89 % | 57 % | 1.162 | 75 % | 14 % / 13 % | 15.3 / 58.6 deg | 40.0 (0.0, 40.0) |
| v24b 6M best | 3.57 | 3.93 (3.88, [3.77, 4.01]) | 84 % / 95 % | 70 % | 1.012 | 80 % | 19 % / 0.2 % | 20.3 / 34.7 deg | 39.1 (1.0, 36.6) |

From scratch in 6M decisions v24b matches/exceeds the v23 chain at 20M (itself warm-started
from older models) on position and constraints; heading better, FPA worse (gamma_f ~ -30 deg).

**v24b resumed (2026-10-14 06:04, at 10.5M -> 20M, curriculum preset v24b).** Level 2 from
6M to 10.5M: position kept improving (validation median 1.5-3.2 km from 8.75M) and heading
converged (|dpsi| 4-12 deg, from 30-75 deg), but FPA did not move (|dfpa| 17-29 deg against
the 12 deg pass threshold; return flat at ~3.5k for 2M decisions). The v24 level-2 FPA
bonus ramp (10 deg) is narrower than the error, so the concave bonus gave no gradient.
Resumed (`--resume --curriculum v24b`, same LR schedule: 1.42e-4 at 10.5M) with the FPA
ramps widened to ~2.5x the pass thresholds (levels 2-6: 30/15/8/2.5/0.6 deg); level,
multipliers (lam_Qdot 3976, lam_n 861) and selection best (25.2 at 6M) carried over.
Log: train_log_resume1.txt.

**v24b outcome — done (20M, 2026-10-14 18:52; 766.7 min of training in two segments).**
Curriculum advances at 4.75M, 6M, 15.75M, 16.25M, 18.75M (ends at level 5 of 6: pass needs
d <= 5 km, |dfpa| <= 1 deg, |dpsi| <= 3 deg). After the FPA-ramp fix (resume at 10.5M) the
median validation FPA error fell 28 -> 1-3 deg in 6M decisions while the position held at
1.3-2.8 km. Best by score at 18.5M (6.79: d 2.14 km, dfpa 2.4, dpsi 2.4 deg nominal); final
nominal 1.25 km, dfpa 2.31, dpsi 7.07 deg, J/J_ref 1.19 (heat load is not in the objective;
v23 0.996). AoA spread along the rollout grew 0.2 -> ~4 deg (modulated from 6M). PPO health:
approx_kl median 0.0044 (no update above 1.5x target), explained variance 0.999, action std
0.50 -> 0.25. Last 2M: validation log10 d still falling (-0.11 / M), return flat (level 5).
The load factor was the price of the shallower arrival: lam_n 900 -> 3,730 from 15M.

Monte Carlo (eval_mc_best_18p5M, eval_mc_final_20M; tables built from the per-case CSVs):

Realistic test S3_test_ic_models_x0.1 (500):

| model | median km | IQM [95 % CI] | <=1 / 5 / 10 km | feasible | feas. & <=5 km | peak p95 | viol. Qdot/qbar/n | median abs dfpa / dpsi | pos_1km / rel_1pct / vista_mc |
|---|---|---|---|---|---|---|---|---|---|
| v23 20M final | 4.17 | 4.15 [4.00, 4.32] | 0 / 71 / 89 % | 75 % | 57 % | 1.162 | 14/0/13 % | 15.3 / 58.6 deg | 0 / 0 / 0 % |
| v24b 6M best | 3.93 | 3.88 [3.75, 4.00] | 0 / 84 / 95 % | 80 % | 70 % | 1.012 | 19/0/0 % | 20.4 / 34.7 deg | 0 / 0 / 0 % |
| v24b 18.5M best | 2.22 | 2.29 [2.17, 2.43] | 3 / 96 / 100 % | 79 % | 77 % | 1.015 | 19/0/2 % | 1.69 / 2.63 deg | 3.0 / 0.4 / 18.0 % |
| **v24b 20M final** | **1.59** | **1.67 [1.59, 1.77]** | **12 / 98 / 100 %** | 74 % | 73 % | 1.018 | 22/0/4 % | 2.12 / 4.56 deg | 7.6 / 0.2 / 5.0 % |

SCvx gc_ic_tc campaign scenarios (1001, full VISTA dispersions = 10x the training scale):

| model | median km | IQM [95 % CI] | <=1 / 5 / 10 km | feasible | feas. & <=5 km | peak p95 | median abs dfpa / dpsi | pos_1km / rel_1pct / rel_0.1pct |
|---|---|---|---|---|---|---|---|---|
| SCvx closed-loop G&C | 0.21 | 1.93 [1.20, 2.75] | 62 / 67 / 71 % | 39 % | 21 % | 1.157 | 0.30 / 0.62 deg | 19.1 / 7.1 / 0.8 % |
| v23 20M final | 38.8 | 52.9 [45.5, 61.6] | 0 / 6 / 18 % | 31 % | 3 % | 1.604 | 15.9 / 63.5 deg | 0 / 0 / 0 % |
| v24b 18.5M best | 14.3 | 15.2 [13.9, 16.4] | 2 / 24 / 42 % | 30 % | 8 % | 1.738 | 4.3 / 13.1 deg | 0.2 / 0 / 0 % |
| v24b 20M final | 12.6 | 13.3 [12.1, 14.6] | 2 / 29 / 46 % | 32 % | 8 % | 1.614 | 4.8 / 10.9 deg | 0.6 / 0 / 0 % |

Other sets (v24b final vs v23 final): S3_test_ic_models (1x, 500) median 20.7 vs 203 km,
IQM 24.8 vs 335 km, feasible 25 vs 20 %; SCVX_gc_models (501) median 9.95 vs 231 km
(SCvx 0.33 km), feasible 35 vs 25 % (SCvx 34 %); S1_sweep_x0.1 median 1.31 vs 3.84 km,
100 vs 86 % within 5 km. 1x sweep: the largest errors come from lift (k_CL, up to 2,155 km
at the extreme multiplier), entry speed (peak ratio up to 2.9), entry altitude, drag and
density — v23's worst sweep errors were 3,500 km (k_CL) and 2,310 km (k_CD).

Conclusions: (1) the reformulation (energy-indexed decisions, absolute attitude targets,
Markov Lagrangian charge, attitude-hold initialization) trained from scratch beats the whole
warm-started v20-v23 chain on every metric on the realistic test: 2.5x lower landing error,
7x lower FPA error, 13x lower heading error, constraint tail p95 1.16 -> 1.02; (2) the FPA
plateau of v23/v24b-level-2 was a reward-shaping defect (bonus ramp narrower than the
error), not a capability limit; (3) the 0.1 % tolerance is still unmet (0 %): the angles
are ~2 deg / ~4 deg against 0.01 / 0.09 deg; (4) at the full VISTA dispersions the policy is
out of its training distribution — SCvx is far more precise in its successful cases (62 %
within 1 km) while the RL tail is shorter (RL beats SCvx beyond ~40 km) and feasibility is
comparable (32 vs 39 %); (5) heat rate stays the binding path constraint (1.6 % nominal
margin at the best attitude; 22 % of realistic cases exceed it, p95 1.018).

### v24b resumed (2) — aborted at 20.82M (2026-10-15 11:05 - 13:05): LR warm restart at 3e-4

`--resume results/stage2_v24b --timesteps 10000000 --lr-decay --lr-restart` with the same
env/reward flags, curriculum v24b (continues at level 5: pass at d <= 5 km, |dfpa| <= 1 deg,
|dpsi| <= 3 deg on 50 % of the validation cases), PID-Lagrangian restored (lam_Qdot 3161,
lam_n 3733), dispersion training 0.1x, selection best carried over (6.79 at 18.5M). LR back
to 3e-4 at 20M, decaying linearly to 0 at 30M. Log: train_log_resume2.txt.

Aborted: the restart peak was the run's lr_base, 3e-4 (the cold-start LR) — 3x the 1e-4
peak of v23's warm restarts — on a converged policy (action std 0.24). Within 0.8M
decisions: return 3,950 -> 2,090, validation median 1.9 -> 33 km (|dfpa| 2.6 -> 13 deg),
lam_n at the 4,000 cap (load-factor excess 9 %), clip fraction 0.065 -> 0.14, approx_kl
up to 0.02. Rolled back to the 20M checkpoint (ckpt/snapshots/eval-history rows after 20M,
Lagrange state and logs moved to aborted_restart_lr3e-4/; Lagrange state restored to its
20M values). `--lr` now sets lr_base on --resume (the restart peak).

### v24b resumed (3) — done (2026-10-15 13:07 - 2026-10-16 03:38): +10M decisions (to 30M), LR restart at 1e-4

Same as (2) but `--lr 1e-4` (v23's restart peak), decaying to 0 at 30M. Log:
train_log_resume2.txt. Throughput ~130 decisions/s (two MATLAB processes share the CPU).

**v24b resumed (3) outcome (30M; 869.2 min).** Stable restart (approx_kl median 0.0042, no
update above 1.5x target over the whole run; clip fraction 0.02-0.07). New bests by score
5.40 (24.5M) and 4.42 (26.75M: validation median 1.74 km, dfpa 1.05, dpsi 0.36 deg); the
final model (30M) reaches the nominal handover 0.91 km off (first nominal < 1 km), dfpa
1.40 deg, dpsi 0.03 deg, J/J_ref 1.22. Curriculum stayed at level 5 (heading met its 3 deg
threshold, FPA hovered at the 1 deg threshold). Load factor went from binding (lam_n at the
4,000 cap, 21-24M) to within budget (excess ~0.1 %, lam_n 3,012 at 30M); lam_Qdot released
3,161 -> 1,985. Last 2M: validation log10 d slope -0.04 / M, return flat (~4.19k).

| S3_test_ic_models_x0.1 (500) | median km | IQM [95 % CI] | <=1 / 5 / 10 km | feasible | feas. & <=5 km | peak p95 | viol. Qdot/qbar/n | median abs dfpa / dpsi | pos_1km / rel_1pct / vista_mc |
|---|---|---|---|---|---|---|---|---|---|
| v23 20M final | 4.17 | 4.15 [4.01, 4.33] | 0 / 71 / 89 % | 75 % | 57 % | 1.162 | 14/0/13 % | 15.3 / 58.6 deg | 0 / 0 / 0 % |
| v24b 20M final | 1.59 | 1.67 [1.59, 1.77] | 12 / 98 / 100 % | 74 % | 73 % | 1.018 | 22/0/4 % | 2.12 / 4.56 deg | 7.6 / 0.2 / 5.0 % |
| v24b 26.75M best | 1.78 | 1.85 [1.75, 1.95] | 10 / 99 / 100 % | 77 % | 76 % | 1.012 | 23/0/0 % | 0.74 / 1.82 deg | 8.0 / 1.8 / 32.8 % |
| **v24b 30M final** | **1.36** | **1.36 [1.31, 1.41]** | **24 / 100 / 100 %** | **79 %** | **79 %** | **1.011** | 21/0/0 % | 1.16 / 2.77 deg | 16.6 / 1.8 / 29.8 % |

| SCVX_gc_ic_tc (1001) | median km | IQM [95 % CI] | <=1 / 5 / 10 km | feasible | feas. & <=5 km | peak p95 | median abs dfpa / dpsi | pos_1km / rel_1pct / rel_0.1pct / vista_mc |
|---|---|---|---|---|---|---|---|---|
| SCvx closed-loop G&C | 0.21 | 1.93 [1.25, 2.73] | 62 / 67 / 71 % | 39 % | 21 % | 1.157 | 0.30 / 0.62 deg | 19.1 / 7.1 / 0.8 / 18.7 % |
| v24b 20M final | 12.6 | 13.3 [12.2, 14.7] | 2 / 29 / 46 % | 32 % | 8 % | 1.614 | 4.8 / 10.9 deg | 0.6 / 0 / 0 / 1.3 % |
| v24b 30M final | 10.5 | 12.3 [11.2, 13.5] | 2 / 28 / 49 % | 34 % | 10 % | 1.777 | 5.0 / 10.9 deg | 0.3 / 0.1 / 0 / 3.0 % |

Other sets (30M final vs 20M final): SCVX_gc_models median 5.29 vs 9.95 km (IQM 6.45 vs
10.7; SCvx 0.33 km, 3.06), feasible 35 % (SCvx 34 %); S3_test_ic_models (1x) median 13.9 vs
20.7 km; S1_sweep_x0.1 median 1.04 vs 1.31 km, 38 % within 1 km; S1_sweep median 3.66 vs
4.57 km. The 26.75M best has the smallest angle errors on the realistic test (0.74 / 1.82
deg, 32.8 % within the VISTA MC tolerance); the 30M final the smallest landing error.

Conclusions: the 1e-4 warm restart cut the realistic-test landing error 1.67 -> 1.36 km
(IQM; 24 % within 1 km), the angles ~2x (FPA 2.1 -> 0.7-1.2 deg, heading 4.6 -> 1.8-2.8 deg)
and removed the load-factor violations (4 % -> 0 %) — heat rate is the only constraint
still exceeded (21 % of the cases, p95 1.011). The 0.1 % tolerance is still 0 % (FPA ~1 deg
against 0.01 deg). At the full VISTA dispersions the gap to SCvx narrowed slightly but
remains a distribution problem (RL 2 % within 1 km vs 62 %; feasibility comparable).

## Robustness phase (2026-10-24): training under expanding dispersions

Goal (publication focus, aerospace venues): a guidance law that stays STABLE and drives the
vehicle to the handover under large uncertainty, degrading gracefully where the mission is
physically infeasible — not only feasibility and terminal precision at the realistic level.
Note on scale: the VISTA 1x dispersions are stress-test magnitudes (density sigma 25 %, CL/CD
20 %, mass 5 %, entry altitude 5 km, speed 60 m/s); 0.1x is the realistic level.

New tooling (this commit):
- **Dispersion curriculum, ADR-style** (training/dispersion_curriculum.py; OpenAI et al. 2019):
  the sampler draws the scale per episode (50 % at the boundary s_k, else uniform in
  [0, s_k]); the boundary advances 0.2 -> 0.35 -> 0.5 -> 0.7 -> 1.0 when, over the last 300
  boundary training episodes, >= 97 % reach the handover and the median landing error is
  <= 5 km (after >= 1.5M decisions at the level; forced after 4M). Groups ic + tc + models
  (terminal-condition dispersions now included, as in the SCvx gc_ic_tc campaign).
- **Robust selection score**: mean over cases of log10(1 + e), e = ||(d/1 km, dgamma/1 deg,
  dpsi/5 deg)|| + 100 x peak excess (+1e4 if not reached) — graded constraint excess instead of
  the 1.02 cliff, on S2_val_robust (24 cases: nominal + 8 @0.1x, 8 @0.5x, 7 @1x).
- **Parallel validation** (pool of 8 from a saved copy of the policy; training workers idle).
- **Evaluation sets for robustness-vs-uncertainty curves**: S4_scale_x{0.1, 0.25, 0.5, 0.75,
  1, 1.25}, 300 cases each, ic+tc+models, common random numbers (case i at scale s = case i at
  1x shrunk by s; verified to 1e-16).
- `--lag-restore-from` (start the multipliers from another run).
Legacy paths unchanged (HPO-best regression return -660.1399573458849).

Pre-training reference (v24b 30M final on S2_val_robust, after 8k smoke-test decisions):
median d 1.2 km nominal, 1.8 km @0.1x, 5.5 km @0.5x, 21.8 km @1x; feasible 100 / 88 / 38 / 29 %.

### v25 — running (launched 2026-10-24): warm start from v24b 30M, dispersion curriculum, 30M decisions

`--init-model results/stage2_v24b/model.zip --init-vn results/stage2_v24b/vecnormalize.pkl
--lr 1e-4 --lr-decay` (PPO settings of v24b: n_steps 2048 x 8, batch 512, 5 epochs, gamma
0.9995, clip 0.2, ent 0, target_kl 0.05); same env (obs v25, target actions, energy steps,
incremental peak charge); reward fixed at v24b's final curriculum level (w_ang_mult 1,
fpa/psi ramps 2.5 / 2 deg, angle bonuses 250) — no terminal curriculum, so the only thing
that changes during the run is the uncertainty; PID-Lagrangian from v24b's multipliers
(lam_Qdot 1985, lam_n 3012), cap 4000, ramp 50 / rollout; `--train-dispersion ic,tc,models
--disp-levels 0.2,0.35,0.5,0.7,1.0`; selection on S2_val_robust (robust score), validation
every 250k in a pool of 8.

**v25 to 5.0M.** Dispersion boundary 0.2 -> 0.35 (1.51M, boundary median 2.55 km) -> 0.5
(3.01M, 3.79 km) -> 0.7 (4.95M, 4.84 km); every boundary episode reached the handover.
Robust validation score 1.10 -> 0.958 (best, 4.0M); 1x-scale validation median 27-42 km
(0-1.5M) -> 7-15 km (2.75-5M). But from 3.75M the NOMINAL case violated a path constraint:
the 4.5M checkpoint pulls 1.099x the load-factor limit at the nominal handover (3M: 0.77x)
while reaching |dfpa| 0.17 deg — a sharper terminal pull-up bought with load factor; heat
rate stayed at 0.987x. Both lam_Qdot and lam_n sat at the shared 4,000 cap (mean excess
3.0 % / 3.7 % vs the 0.5 % budget): the heat-rate excess is partly unavoidable at 0.5x
(1.6 % nominal margin at the best attitude), the load-factor excess is not (it is the
pull-up), and at the shared cap 10 % load excess cost ~400 against the terminal-angle
rewards. Fix: per-constraint caps (`--lag-max-per n=10000`; Qdot and qbar stay at 4,000),
ramp 100 / rollout.

### v25 resumed — running (2026-10-24 18:38): from 5.0M to 30M, lam_n cap 10,000

`--resume results/stage2_v25 --timesteps 25000000 --lag-max 4000 --lag-max-per n=10000
--lag-rate 100`, all other flags unchanged (LR schedule continues: 8.3e-5 at 5M; dispersion
boundary 0.7 restored). Log: train_log_resume1.txt.

**v25 5M -> 9M (lam_n cap 10,000).** The load-factor violation at the nominal handover is
gone (8.0M checkpoint: nominal n 0.66x, |dfpa| 0.07 deg; 0.1x sweep n <= 0.71x), lam_n
4,000 -> 8,423. 1x-scale validation median 10.6 -> 4.7 km (8.0M). But the near-nominal cases
now break the heat rate by 1-2 % (8.0M: nominal peak 0.9966x — 0.3 % margin, against the
0.982x attainable at the best attitude; 21 % of the 0.1x sweep above the limit, max 1.021x;
validation feasibility at 0.1x 12-50 %). The boundary gate at 0.7x was not met (boundary
median 6.5-7.1 km, p90 30-126 km) and the curriculum forced 0.7 -> 1.0 at 8.95M.

**Heat-rate floor (`--qdot-floor`, commit cbf224f).** Per scenario, Qdot_lb = peak of the
open-loop lift-up flight (bank 0, AoA at its bound) from the initial state under the
episode's model factors (RK4 1 s; identical to 1/3 s; ~0.6 s per reset): the least peak any
guidance can reach. Only the excess above max(1, Qdot_lb) is charged and fed to the PID.
Nominal floor 0.982. On the CRN robustness sets (first 100 cases each) the floor exceeds the
limit — heat-rate infeasible for any guidance — in 14 % of the cases at 0.1x (p90 floor
1.005), 47 % at 0.5x (p90 1.096) and 57 % at 1x (p90 1.213, max 1.58). So most of the
heat-rate violations of v24b / v25 at the realistic level are physics, not guidance.

### v25 resumed (2) — running (2026-10-24 23:48): from 9.0M to 30M at the 1x boundary, heat-rate floor

`--resume results/stage2_v25 --timesteps 21000000 --qdot-floor`, all other flags as resume (1)
(lam_n cap 10,000, ramp 100). LR 7.0e-5 at 9M. Log: train_log_resume2.txt.

**v25 resumed (2) — aborted at 10.5M (1x boundary).** Training at the full VISTA scale (the
stress test: 57 % of the cases heat-rate infeasible for any guidance, boundary p90 landing
error 1,118 km, boundary mean peak excess 17 %) degraded the policy instead of hardening it:
robust validation score 1.19-1.29 (v24b start ~1.06, best 0.958 at 4.0M), the nominal case
infeasible on 6/6 validations, 0.1x feasibility 0 %, training return 1.4k -> 0.7-1.5k, value
loss 0.8-2.5e4, lam_n at the 10,000 cap (load-factor excess 2-15 %). Rolled back to the
9.0M checkpoint (end of the 0.7x phase); the 9.0-10.5M artifacts and logs are kept in
aborted_1x_boundary/. Lagrange state set to its 9.0M multipliers (integrals consistent).

**Design envelope.** The training distribution stops at 0.7x VISTA (~7x the realistic
level), where the 1x-scale validation median was still improving (10.6 -> 4.7 km during the
0.7x phase); 1x and 1.25x are off-design robustness tests (S4 sets).

### v25 resumed (3) — running (2026-10-25 01:48): from 9.0M to 30M at the 0.7x envelope, heat-rate floor

`--resume results/stage2_v25 --timesteps 21000000 --qdot-floor --disp-levels 0.2,0.35,0.5,0.7`
(0.7 is the last level), other flags as before. Log: train_log_resume2.txt.

**v25 9.0M -> 10.5M (0.7x envelope, heat-rate floor).** Near-nominal feasibility did not
recover (nominal infeasible on 5/6 validations, 0.1x 0-25 %); the 10.0M checkpoint lands
0.47 km off at nominal with load factor 0.63x but rides the heat-rate limit (peak 1.0006x,
floor 0.982). With the floor, the charged heat-rate excess is the avoidable part only, yet
lam_Qdot sat at its 4,000 cap with the mean attributable excess 1.1-1.8 % (budget 0.5 %).
Heat rate now gets the same 10,000 cap as the load factor.

### v25 resumed (4) — running (2026-10-25 02:48): from 10.5M to 30M, Qdot and n caps 10,000

`--lag-max-per n=10000,Qdot=10000`, other flags as resume (3). Log: train_log_resume3.txt.
