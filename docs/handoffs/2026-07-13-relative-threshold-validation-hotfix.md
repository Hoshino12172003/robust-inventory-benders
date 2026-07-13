# Relative Threshold Validation Hotfix Handoff

## Scope

This hotfix narrows the PR #10 relative-threshold validation without changing the mathematical model, Benders validity rules, or selected-parameter validation.

## Behavior

- Relative mode accepts a non-null top-level threshold.
- Relative mode also accepts configurations where every declared variant has its own non-null threshold.
- `screen_relative_cut_wide.yaml` therefore remains valid without a dummy top-level threshold.
- `screen_master_gamma.yaml` still fails before creating output directories while its threshold is unselected.
- A relative configuration with any variant missing its threshold fails with the same clear instruction.

## Validation

- `python -m pytest tests/test_experiment_suite.py -q`
- `python -m pytest tests -q`
- `git diff --check`
- `python scripts/check_hidden_unicode.py`
