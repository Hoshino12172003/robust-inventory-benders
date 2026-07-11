# PR #7 Handoff: Experiment Results

## Summary

This PR adds experiment outputs generated from the formal experiment suite configs.

## Included Result Sets

- `experiments/results/small_correctness/`
- `experiments/results/baseline_comparison/`
- `experiments/results/ablation_study/`
- `experiments/results/sensitivity_gamma/`
- `experiments/results/sensitivity_service/`
- `experiments/results/scalability/`

## Files Included

- `results.csv`
- `summary.csv`
- `correctness_summary.csv` for `small_correctness`
- `experiment_report.md`
- `sensitivity_gamma/gamma_summary.csv`
- `sensitivity_service/service_summary.csv`
- generated instance JSON files under each experiment's `instances/` directory

## Notes

- This PR contains generated experiment artifacts only.
- This PR does not change experiment logic.
- This PR does not change the mathematical model.
- This PR does not change the robust dual MILP implementation.
- `small_correctness` reports `ok` for all three seeds in `correctness_summary.csv`.
- Small baseline rows are solved under the current solved-gap criterion.
- Medium and large rows are mostly completed at `iteration_limit`; these are preserved as generated.
- These results are preliminary and should not be treated as final paper results.

## Next PR

Open PR #8 to tune experiment parameters before final paper experiments:

- Increase the iteration limit to 300 or 500.
- Consider relaxing the solved gap for exploratory medium / large analysis.
- Increase the time limit if needed.
- Adjust adaptive gap and Gamma continuation settings so medium instances converge more reliably.

## Verification

- `python scripts/check_hidden_unicode.py`: `No hidden Unicode characters found.`
- `git diff --cached --check`: passed.
