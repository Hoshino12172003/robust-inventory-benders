# Adaptive Secondary-Cut Generation Handoff

## Summary

This change moves the adaptive decision ahead of the no-good-constrained robust solve. The primary robust subproblem and every positively violated primary cut remain unchanged. Secondary solves are now optional and are attempted only when the lower bound has stalled and the configured resource checks permit them.

## Changes

- Normalized Gurobi status 9 to `time_limit` across Benders, monolithic, robust-dual, and experiment-summary paths.
- Added a rolling lower-bound improvement measure and a pre-solve secondary-generation gate.
- Added cooldown, cumulative subproblem-time share, remaining-time, and minimum solve-budget checks.
- Preserved the existing K=2 all-cuts behavior when adaptive generation is disabled; K=1 remains the default.
- Added iteration and result diagnostics for attempted, avoided, triggered, skipped, timed, duplicate, and added secondary solves/cuts.
- Added `screen_adaptive_secondary_generation.yaml` with the four requested equal-time medium-scale variants.
- Kept the robust model, primary-cut validity, target-Gamma evaluation, no-good constraints, and valid-UB rules unchanged.

## Verification

- Convergence diagnostics: 27 passed.
- Experiment suite: 19 passed.
- Full test suite: 65 passed.
- Hidden Unicode scan: `No hidden Unicode characters found.`
- `git diff --check`: clean.
- Python compilation: passed for the changed Python modules and tests.
- The complete suite was run in the E-drive project environment with the licensed Gurobi interpreter.

## Local Validation

Run in the E-drive project environment where the Gurobi license is available:

```powershell
python -m pytest tests/test_convergence_diagnostics.py -q
python -m pytest tests/test_experiment_suite.py -q
python -m pytest tests -q
python scripts/check_hidden_unicode.py
git diff --check
```

Do not run the medium tuning seeds until the PR is reviewed. The new experiment config is a template only and no result files are included.

## Review Focus

- Confirm that an adaptive gate closed decision cannot call the no-good secondary solver.
- Confirm that only the unconstrained target-Gamma robust solve contributes to the UB.
- Confirm that status 9 remains a completed time-limit run when it has an incumbent objective.
- Review the default gate thresholds before running the focused medium-scale screen.

## Branch Note

This work is stacked on the secondary-cut redesign from PR #12. Retarget it to `main` after PR #12 is merged.
