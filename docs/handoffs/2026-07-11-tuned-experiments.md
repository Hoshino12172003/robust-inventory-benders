# PR #8 Handoff: Tuned Experiments

## Summary

This PR tunes experiment parameters and collects tuned convergence results.

## Main Changes

- Added tuned experiment configs under `experiments/configs/`.
- Added tuned results under `experiments/results_tuned/`.
- Added `experiments/results_tuned/tuned_experiment_report.md`.

## Tuned Configs

- `baseline_comparison_tuned.yaml`
- `ablation_study_tuned.yaml`
- `sensitivity_gamma_tuned.yaml`
- `sensitivity_service_tuned.yaml`
- `scalability_tuned.yaml`

## Parameter Changes

- Increased `max_iterations` from 120 to 300.
- Increased tuned `time_limit` to 600 seconds.
- Kept `robust_dual_milp` as the main subproblem mode.
- Kept small correctness unchanged.
- Used positive `delta_cut` values so cut selection can be observed.
- Kept `no_cut_selection` as an ablation variant.

## Results Notes

- Small cases remain solved.
- Medium final gaps improved relative to PR #7 preliminary results.
- Medium and large cases still generally stop at `iteration_limit` under the strict solved criterion.
- Tuned results are diagnostic and not final paper-ready results.
- Further algorithmic or parameter tuning is still needed for strong medium / large convergence.

## Verification

- `python scripts/check_hidden_unicode.py`: `No hidden Unicode characters found.`
- `git diff --check`: passed.
- `python -m pytest tests/test_experiment_suite.py -q`: `6 passed`
- `python -m pytest tests -q`: `25 passed`
