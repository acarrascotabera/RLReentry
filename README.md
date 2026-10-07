# RLReentry — Reinforcement-learning guidance for atmospheric re-entry

PPO guidance policy for the WB001 winged-body heat-load re-entry mission
(100 km → 25 km handover, ~7,800 km range), trained in a 3-DOF rotating
oblate-Earth simulator that is a validated port of the SCvx reference framework
(VISTA). The policy commands bank-angle and angle-of-attack rates; SCvx is used
only as an external benchmark.

| Document | Content |
|---|---|
| [`report/main.tex`](report/main.tex) | Current technical report (physics validation, reward campaign v1–v19, HPO) |
| [`report/literature_review.md`](report/literature_review.md) | State of the art and roadmap |
| [`results/results/RUNS.md`](results/results/RUNS.md) | Run-by-run log of the training campaign |
| [`ARCHITECTURE.md`](ARCHITECTURE.md), [`DESIGN.md`](DESIGN.md) | Architecture and design decisions |

## Layout

```
reentry_rl/
  physics/         3-DOF EOM, US76 atmosphere, J2 gravity, WB001 aero
  envs/            Gymnasium environments, reward, scenarios
  training/        PPO training, Optuna HPO, observation-expansion surgery
  validation/      physics validation vs SCvx, Monte Carlo policy evaluation
  postprocessing/  figures and run diagnostics
report/            LaTeX report and figures
results/           training runs (models/checkpoints are not version controlled)
```

## Quick start

```bash
pip install -e .
python -m reentry_rl.validation.validate_physics
python -m reentry_rl.training.train_sb3 --stage stage2 --timesteps 10000000 --n-envs 8
```

Model files (`*.zip`, `*.pkl`), checkpoints and TensorBoard logs are excluded
from git; run configurations, summaries and the Optuna study are tracked.
