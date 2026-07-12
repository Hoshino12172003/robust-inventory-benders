# Round 2 Tuning Suite Handoff

## Scope

This change adds medium-instance, tuning-seed-only configurations for a second diagnostic round. It does not change the mathematical model, robust-dual validity rules, or the selected-parameter file.

## Configurations

- `screen_relative_cut_wide.yaml` tests relative thresholds 0.00 through 0.50 with adaptive robust-subproblem gaps and K=2 multi-cut generation.
- `screen_master_gamma.yaml` compares master-gap settings and Gamma continuation designs. Its `relative_cut_threshold` is intentionally `null` until the wide threshold screen is reviewed.
- `confirm_equal_time_medium.yaml` is locked by `selected_algorithm_parameters.yaml` and compares standard Benders with the eventual selected proposed candidate under an equal 60-second limit.

The staged Gamma variant uses Gamma 0 for iterations 1-10, Gamma 1 for iterations 11-30, and Gamma 2 from iteration 31 onward.

## Safety

- Outputs use `experiments/results_diagnostics_round2/`, so round-1 diagnostics are not overwritten.
- Standard Benders remains protected by the experiment method mapping: fixed target Gamma, no cut selection, no adaptive master or robust-subproblem gap, K=1, and a fixed tight robust-subproblem gap.
- `selected_algorithm_parameters.yaml` remains pending and unchanged by this tuning round.

## Next Steps

1. Run `screen_relative_cut_wide.yaml` locally and select a threshold using convergence quality, cuts skipped, forced cuts, and runtime.
2. Put that threshold into `screen_master_gamma.yaml` before running it.
3. Complete all parameter selection before unlocking `confirm_equal_time_medium.yaml`.
