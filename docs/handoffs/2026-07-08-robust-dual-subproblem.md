# Handoff: Implement Robust Dual MILP Subproblem

Date: 2026-07-08
Branch: `codex/robust-dual-subproblem`
Commit: `eed24d1`
PR: `https://github.com/Hoshino12172003/robust-inventory-benders/pull/4`

## Summary

- Added `src/robust_dual_subproblem.py` with the robust dual MILP subproblem for directly evaluating `Q^R(x; Gamma)`.
- Added `RobustDualSubproblemResult` with primal worst-demand metadata, dual values, Benders cut coefficients, runtime, status, and MIP gap.
- Added `algorithm.subproblem_mode` with supported values:
  - `robust_dual_milp`
  - `scenario_enumeration`
- Set `robust_dual_milp` as the default Benders subproblem mode for main experiments.
- Kept scenario enumeration as a benchmark / validation / heuristic mode.
- Updated Benders upper-bound logic so adaptive runs always update final UB using the target Gamma subproblem value.
- Added final metadata for active and target subproblem values, active Gamma, target Gamma, UB validity, and selected subproblem mode.
- Updated configs and README to document the new default and mode meanings.
- Added robust dual tests for full-enumeration agreement, cut exactness at the current point, cut validity at other points, and Benders integration.

## Verification

- Ran `pytest tests/test_robust_dual_subproblem.py -q`.
- Result: 4 passed.
- Ran `pytest tests -q`.
- Result: 14 passed.
- Hidden / bidirectional Unicode scan: no remaining hidden format/control characters found in changed files.

## Review Notes for ChatGPT

- Check the robust dual MILP objective against the paper derivation.
- Check the McCormick linearization bounds for `w = z * lambda` and `g = z * nu`.
- Check whether Benders metadata and iteration log fields are sufficient for experiment audit.
- Check whether the stacked PR base should remain PR #3 or be retargeted to `main` after PR #3 is merged.

## Next Steps

- After PR #3 is merged, retarget this PR to `main` if it is created as a stacked PR.
- Add larger experiment configs comparing `robust_dual_milp` against full scenario enumeration where full enumeration is tractable.
