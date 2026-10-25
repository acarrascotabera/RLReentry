"""Per-scenario heat-rate floor of an evaluation set: the peak Qdot / limit of the open-loop
lift-up flight (bank 0, AoA at its upper bound) from the scenario's initial state under its
model factors — the least peak any guidance can reach. Cases with a floor above 1 are
heat-rate infeasible for every guidance law; the difference between a policy's peak and
max(1, floor) is the guidance-attributable excess.

    python -m reentry_rl.validation.heat_rate_floor S4_scale_x0.1 S4_scale_x1 ... [--workers 8]

Writes results/eval_sets/floors/<set>.csv (id, qdot_lb).
"""
import argparse
import multiprocessing as mp
from pathlib import Path

if __package__ in (None, ""):
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "eval_sets" / "floors"
_ENV = {}


def _init():
    from reentry_rl.envs.stage2_bank_aoa import Stage2BankAoaEnv
    _ENV["env"] = Stage2BankAoaEnv(obs_version="v25", action_mode="target", step_mode="energy",
                                   qdot_floor=True)


def _floor(job):
    from reentry_rl.envs.scenario import Scenario
    i, sdict = job
    env = _ENV["env"]
    env.reset(options={"scenario": Scenario.from_dict(sdict)})
    return {"id": i, "qdot_lb": env._qdot_lb}


def floors_for(set_name, workers=8):
    from reentry_rl.envs.scenario import load_set
    from reentry_rl.validation.evaluate_policy import resolve_set
    _name, scen, _labels = load_set(resolve_set(set_name))
    jobs = [(i, s.to_dict()) for i, s in enumerate(scen)]
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        rows = pool.map(_floor, jobs, chunksize=4)
    df = pd.DataFrame(rows).sort_values("id")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{set_name}.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for s in args.sets:
        df = floors_for(s, args.workers)
        print(f"{s}: n={len(df)} floor median {df.qdot_lb.median():.3f} p90 {df.qdot_lb.quantile(0.9):.3f} "
              f"infeasible for any guidance {(df.qdot_lb > 1).mean():.1%}")


if __name__ == "__main__":
    main()
