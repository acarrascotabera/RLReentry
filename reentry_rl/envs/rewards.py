"""Three-tier reward: position >> chattering >> path constraints.

Tier 1 — POSITION (the objective). Landing error d [km] enters through a
potential that is LOGARITHMIC far out (equal cost per decade — credit
assignment across 1000s of km) but LINEAR inside d_lin_km:
  * per step : potential-based shaping  pen(d_prev) - pen(d_cur)
  * ending   : uniform terminal anchor -pen(d_end) on EVERY ending, a graded
               success ramp +w_succ*(1 - d/succ_ramp_km)^+ on landing, and
               -w_fail extra on fail/timeout. (Uniformity matters: anchor-on-
               landing-only taught v7 to skip out; no anchor let v8 dive early.)

Near-field curvature controls how PPO treats its own action noise (Jensen):
  * -log10(d) is CONVEX in d  -> dispersed landings out-score their center
    -> PPO keeps sigma large and gambles (v15: train reward ~1,000 above eval,
    oscillation, no plateau).
  * linear well is NEUTRAL -> concentrating accuracy is not penalised but not
    rewarded either -> sigma has no gradient, gets stuck (~0.18) and the
    deterministic policy random-walks off its own best (v16: 5.4 km best @2.55M
    drifted to 56 km @10M).
  * CONCAVE well (`near_convexity` t>0: penalty CONVEX in d) -> a tight cluster
    beats a dispersed one -> for the first time PPO has a downhill on sigma, so
    the action noise (and the terminal scatter it causes) is driven DOWN. This
    is the v18 "precision well". The landing bonus is likewise made concave
    (1 - (d/succ_ramp)^2)^+ to reinforce anti-dispersion in the 0-succ_ramp band.
    t is a continuity- and endpoint-preserving linear<->quadratic blend, so
    t=0 recovers the v16 linear well exactly (clean fallback).

Tier 2 — CHATTERING, charged ONCE at the end as the episode-MEAN squared
action change. It must NOT be a per-step sum: with exploration noise sigma~1,
E‖a_t - a_{t-1}‖² ≈ 4, so a per-step penalty taxed a 1,500-step closing glide
~15x harder than a 355-step dive and the dive out-scored it — that (not
optimizer instability) is what collapsed v10 and v11. The mean is
episode-length-neutral: pure noise shifts every policy by the same constant,
and the term bites exactly when the action std anneals down.

Tier 3 — PATH CONSTRAINTS, a capped per-step soft barrier. Zero on feasible
trajectories (the 43.5 km v10-best glide never pays it), expensive on the
8-16x-violating dives — so it also buries the dive attractor. The cap bounds
the episode total so value scales stay sane.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class RewardWeights:
    # ---- tier 1: position ----
    w_pos: float = 100.0    # cost per decade of landing error (shaping + uniform anchor)
    w_near: float = 300.0   # EXTRA cost per decade inside d_near_km (end-game emphasis:
                            # the plain log left 17.8->1 km worth only ~125 units and
                            # v12 stagnated there; below d_near a decade costs w_pos+w_near)
    d_near_km: float = 50.0
    w_lin: float = 40.0     # near-field well scale [1/km]: pen(d_lin)=w_lin*d_lin, so
                            # 40/km ~ continues the old 1,000/decade gradient at 10 km and
                            # is 0 at d=0. Shape set by near_convexity below.
    d_lin_km: float = 10.0
    near_convexity: float = 0.5  # near-field linear<->quadratic blend t in [0,1]:
                            # pen(d)=w_lin*d_lin*[(1-t)(d/d_lin)+t(d/d_lin)^2]. t=0 -> v16
                            # linear (Jensen-neutral); t>0 -> penalty CONVEX -> reward
                            # concave -> landing dispersion is penalised -> drives sigma DOWN
                            # (the v18 precision well). Continuous & 0-at-target for all t.
    w_succ: float = 300.0   # CONCAVE landing bonus: w_succ*(1 - (d/succ_ramp_km)^2)^+ —
                            # dense pull from succ_ramp_km in AND anti-dispersion (concave;
                            # was linear in v16, a lottery cliff before that)
    succ_ramp_km: float = 10.0  # = d_lin_km: the bonus spans the whole near-field well so
                            # its (nonzero-slope) turn-off kink sits at 10 km, OUT of the
                            # ~5 km operating region — keeping the terminal reward cleanly
                            # concave there (curvature -14/km²) with no pro-dispersion pocket
                            # (a 5 km ramp put a convex kink right where the policy lands)
    w_fail: float = 50.0    # extra penalty on skip-out / crash / timeout
    # ---- tier 2: chattering ----
    w_smooth: float = 25.0  # * mean_t ‖a_t - a_{t-1}‖² at episode end (episode-mean:
                            # length-neutral, but dilutable by a short burst...)
    w_endsmooth: float = 1.0  # ...so ALSO a per-step ‖Δa‖² penalty gated to LOW altitude
                            # (alt < endsmooth_alt_m). The gated phase is short (~150-200 s),
                            # so no episode-length tax — it prices the end-game burst steps
                            # (v13's best flew 1,590 s glass-smooth then flipped bank/AoA
                            # full-scale for the last ~20 s to trim the impact point for
                            # free). Altitude-gated (was range<d_near) so trajectory shape
                            # cannot sidestep it.
    endsmooth_alt_m: float = 35e3
    # ---- tier 3: path constraints (per-step, capped) ----
    w_path: float = 1.0     # * min(sum of soft barriers, path_cap)
    path_soft: float = 0.9  # fraction of the limit where the barrier turns on
    path_cap: float = 3.0   # per-step cap on the summed barrier


def path_barrier(ratio, soft=0.9):
    """Smooth one-sided penalty on (constraint value / limit).

    0 below `soft`, quadratic up to the limit (=1 at the limit), steep beyond.
    Vectorized.
    """
    over = np.maximum(0.0, ratio - soft)
    base = (over / (1.0 - soft)) ** 2          # 0 at `soft`, 1 at the limit
    beyond = np.maximum(0.0, ratio - 1.0)      # only past the limit
    return base + beyond * (10.0 + 10.0 * beyond)


def position_penalty(d_km, w: RewardWeights):
    """Landing-error potential: CONVEX-blended well inside d_lin_km, LOG shelves outside.

    pen(d) = w_lin*d_lin*[(1-t)*(d/d_lin) + t*(d/d_lin)^2]   for d <= d_lin
           = w_lin*d_lin + (w_pos+w_near)*decades past d_lin, with the w_near
             shelf capped at d_near (so far field is w_pos/decade)   otherwise
    where t = near_convexity.

    Continuous at d_lin (= w_lin*d_lin for all t), zero at the target, monotone
    in d. The same potential drives the per-step shaping and the uniform terminal
    anchor, so the return stays monotone in d_end across all endings (exploit-
    free). t>0 makes the near field CONVEX -> the terminal reward is concave ->
    landing dispersion is penalised -> PPO drives its action noise down (v18).
    t=0 recovers the v16 linear (Jensen-neutral) well exactly."""
    d = max(float(d_km), 0.0)
    if d <= w.d_lin_km:
        x = d / w.d_lin_km
        t = w.near_convexity
        return w.w_lin * w.d_lin_km * ((1.0 - t) * x + t * x * x)
    logd = float(np.log10(d))
    log_lin = float(np.log10(w.d_lin_km))
    return (w.w_lin * w.d_lin_km
            + w.w_pos * (logd - log_lin)
            + w.w_near * (min(logd, float(np.log10(w.d_near_km))) - log_lin))
