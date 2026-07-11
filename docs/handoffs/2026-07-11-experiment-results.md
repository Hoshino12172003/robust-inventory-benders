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
- generated instance JSON files under each experiment's `instances/` directory

## Notes

- This PR contains generated experiment artifacts only.
- This PR does not change experiment logic.
- This PR does not change the mathematical model.
- This PR does not change the robust dual MILP implementation.
- `small_correctness` reports `ok` for all three seeds in `correctness_summary.csv`.
- Some medium-scale baseline rows are completed but not solved under the current final-gap criterion; these are preserved as generated.

## Verification

- `python scripts/check_hidden_unicode.py`: `No hidden Unicode characters found.`
- `git diff --cached --check`: passed.
