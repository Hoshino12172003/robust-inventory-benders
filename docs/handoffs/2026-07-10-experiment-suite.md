# PR #6 Handoff: Experiment Suite

## Summary

This PR adds a reproducible experiment suite for correctness validation, baseline comparison, ablation study, sensitivity analysis, and scalability analysis.

## Main Changes

- Added `src/experiment_suite.py`.
- Added formal experiment configs under `experiments/configs/`.
- Added result output support for `results.csv`, `summary.csv`, and `correctness_summary.csv`.
- Added `experiments/scripts/plot_results.py` for optional matplotlib plots.
- Added `tests/test_experiment_suite.py`.
- Extended instance generation with `demand_scale`, `capacity_factor`, `cost_scale`, and `service_level`.
- Added `python -m src.cli experiment-suite --config ...`.
- Updated README with experiment-suite usage and scope.

## Formal Experiment Configs

- `small_correctness.yaml` supports small exact validation.
- `baseline_comparison.yaml` includes small + medium.
- `ablation_study.yaml` is medium-focused and also includes medium-large.
- `sensitivity_gamma.yaml` varies `Gamma` on medium instances.
- `sensitivity_service.yaml` varies service levels on medium instances.
- `scalability.yaml` includes small + medium + large.

## Outputs

- `results.csv`: one row per method / variant / instance / seed run.
- `summary.csv`: aggregated success rate, runtime, gap, iteration, cut, and UB-validity statistics.
- `correctness_summary.csv`: exact-vs-Benders comparison for `small_correctness`.

## Experiment Methods

- `monolithic_gurobi`
- `standard_benders`
- `static_inexact_benders`
- `adaptive_gamma_benders`
- `adaptive_gap_benders`
- `adaptive_cut_benders`
- `proposed_adaptive_benders`
- `scenario_benders_full`

## Scope Notes

- This PR does not introduce RL/PPO.
- This PR does not change the mathematical model.
- This PR does not change the robust dual MILP core algorithm.
- `monolithic_gurobi` and `scenario_benders_full` are exact full-enumeration methods and are skipped when the full scenario count exceeds `max_scenarios`.
- Candidate scenarios remain heuristic and are not used as exact robust baselines.

## Verification

- `python scripts/check_hidden_unicode.py`: `No hidden Unicode characters found.`
- `pytest tests/test_experiment_suite.py -q`: `5 passed`
- `pytest tests -q`: `24 passed`

## PR Notes

- This PR adds `tests/test_experiment_suite.py`.
- This PR does not introduce RL/PPO.
- This PR does not change the mathematical model.
- This PR does not change the robust dual MILP core algorithm.

## Next Steps

- Run the formal `small_correctness` experiment locally and inspect `correctness_summary.csv`.
- Run `baseline_comparison` on small and medium instances before drafting the computational-results section.
- Use `ablation_study` to quantify the contribution of adaptive gap, Gamma continuation, and cut selection.
- Keep large-scale runs outside CI because they are intended for paper experiments, not unit tests.
