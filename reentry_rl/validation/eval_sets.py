"""Generate the fixed evaluation sets (results/eval_sets/*.json).

Run once; the files are version controlled so every policy (and the SCvx
benchmark) is scored on identical scenarios. Dispersions follow the VISTA
Monte Carlo cases (see envs/scenario.py); "x0.1" sets scale sigma and clip by
0.1, the realistic level VISTA's stress-test magnitudes are derived from.

    S0_nominal               1     nominal mission
    S1_sweep                 90    one-at-a-time, 15 params x {-3..3} sigma (VISTA scale)
    S1_sweep_x0.1            90    same, realistic scale
    S2_val                   16    nominal + 15 draws ic+models (VISTA)     -> selection only
    S2_val_x0.1              16    nominal + 15 draws ic+models (realistic) -> selection only
    S3_test_ic_models        500   ic+models (VISTA)                         -> reporting
    S3_test_ic_models_x0.1   500   ic+models (realistic)                     -> reporting
    S3_test_ic_tc_models     500   ic+tc+models (VISTA, dispersed target)    -> reporting

Robustness study (python -m reentry_rl.validation.eval_sets --robust; written only if absent):
    S2_val_robust            24    nominal + 8 @0.1x, 8 @0.5x, 7 @1x, ic+tc+models -> selection only
    S4_scale_x<s>            300   ic+tc+models at s in {0.1, 0.25, 0.5, 0.75, 1.0, 1.25}, all
                                   from the SAME standard-normal draws (common random numbers:
                                   case i at scale s is case i at scale 1 shrunk by s), so the
                                   performance-vs-uncertainty curves compare like with like

Aero-capped robustness study (--robust-aero): S2_val_robust_a5 and S5_scale_x<s>, as above
but with the C_L / C_D sigma capped at 5 % (CRN with S4: only the aero factors differ).

    python -m reentry_rl.validation.eval_sets
"""
from pathlib import Path

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from reentry_rl.envs.scenario import NOMINAL, sample_scenarios, sweep_scenarios, save_set

OUT = Path(__file__).resolve().parents[2] / "results" / "eval_sets"


def build_all(outdir=OUT):
    made = []
    made.append(save_set(outdir / "S0_nominal.json", "S0_nominal", [NOMINAL],
                         meta={"purpose": "nominal mission"}))
    for scale, suffix in ((1.0, ""), (0.1, "_x0.1")):
        sw = sweep_scenarios(("ic", "tc", "models"), sigma_scale=scale)
        made.append(save_set(outdir / f"S1_sweep{suffix}.json", f"S1_sweep{suffix}",
                             [s for _, _, s in sw],
                             meta={"purpose": "one-at-a-time sensitivity", "sigma_scale": scale},
                             labels=[{"param": p, "multiplier": m} for p, m, _ in sw]))
    for scale, suffix, seed in ((1.0, "", 101), (0.1, "_x0.1", 102)):
        scen = [NOMINAL] + sample_scenarios(15, ("ic", "models"), scale, seed)
        made.append(save_set(outdir / f"S2_val{suffix}.json", f"S2_val{suffix}", scen,
                             meta={"purpose": "model selection only", "groups": ["ic", "models"],
                                   "sigma_scale": scale, "seed": seed, "first_is_nominal": True}))
    for groups, scale, name, seed in ((("ic", "models"), 1.0, "S3_test_ic_models", 2026),
                                      (("ic", "models"), 0.1, "S3_test_ic_models_x0.1", 2027),
                                      (("ic", "tc", "models"), 1.0, "S3_test_ic_tc_models", 2028)):
        made.append(save_set(outdir / f"{name}.json", name,
                             sample_scenarios(500, groups, scale, seed),
                             meta={"purpose": "reporting only", "groups": list(groups),
                                   "sigma_scale": scale, "seed": seed}))
    return made


ROBUST_SCALES = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25)
ROBUST_GROUPS = ("ic", "tc", "models")


def build_robust(outdir=OUT):
    """Sets for the robustness study; existing files are never overwritten."""
    made = []
    p = outdir / "S2_val_robust.json"
    if not p.exists():
        scen, labels = [NOMINAL], [{"disp_scale": 0.0}]
        for scale, n, seed in ((0.1, 8, 103), (0.5, 8, 104), (1.0, 7, 105)):
            scen += sample_scenarios(n, ROBUST_GROUPS, scale, seed)
            labels += [{"disp_scale": scale}] * n
        made.append(save_set(p, "S2_val_robust", scen, labels=labels,
                             meta={"purpose": "model selection only (mixed scales)",
                                   "groups": list(ROBUST_GROUPS), "first_is_nominal": True}))
    for scale in ROBUST_SCALES:
        name = f"S4_scale_x{scale:g}"
        p = outdir / f"{name}.json"
        if not p.exists():
            made.append(save_set(p, name, sample_scenarios(300, ROBUST_GROUPS, scale, 2030),
                                 labels=[{"disp_scale": scale}] * 300,
                                 meta={"purpose": "reporting only (robustness vs scale, CRN)",
                                       "groups": list(ROBUST_GROUPS), "sigma_scale": scale,
                                       "seed": 2030}))
    return made


AERO_CAP = 0.05


def build_robust_aero(outdir=OUT, aero_cap=AERO_CAP):
    """Robustness sets under the capped aerodynamic dispersions (C_L, C_D sigma <= 5 %).
    Same seeds and parameter order as S2_val_robust / S4, so case i differs from its S4
    counterpart only in the aerodynamic factors (common random numbers). Never overwritten."""
    made = []
    p = outdir / "S2_val_robust_a5.json"
    if not p.exists():
        scen, labels = [NOMINAL], [{"disp_scale": 0.0}]
        for scale, n, seed in ((0.1, 8, 103), (0.5, 8, 104), (1.0, 7, 105)):
            scen += sample_scenarios(n, ROBUST_GROUPS, scale, seed, aero_cap)
            labels += [{"disp_scale": scale}] * n
        made.append(save_set(p, "S2_val_robust_a5", scen, labels=labels,
                             meta={"purpose": "model selection only (mixed scales, aero cap)",
                                   "groups": list(ROBUST_GROUPS), "aero_cap": aero_cap,
                                   "first_is_nominal": True}))
    for scale in ROBUST_SCALES:
        name = f"S5_scale_x{scale:g}"
        p = outdir / f"{name}.json"
        if not p.exists():
            made.append(save_set(p, name, sample_scenarios(300, ROBUST_GROUPS, scale, 2030, aero_cap),
                                 labels=[{"disp_scale": scale}] * 300,
                                 meta={"purpose": "reporting only (robustness vs scale, CRN, aero cap)",
                                       "groups": list(ROBUST_GROUPS), "sigma_scale": scale,
                                       "aero_cap": aero_cap, "seed": 2030}))
    return made


if __name__ == "__main__":
    import sys
    if "--robust-aero" in sys.argv[1:]:
        made = build_robust_aero()
    elif "--robust" in sys.argv[1:]:
        made = build_robust()
    else:
        made = build_all()
    for p in made:
        print(p)
