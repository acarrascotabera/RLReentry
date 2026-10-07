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


if __name__ == "__main__":
    for p in build_all():
        print(p)
