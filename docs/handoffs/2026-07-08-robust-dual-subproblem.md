# Handoff: Implement Robust Dual MILP Subproblem

Date: 2026-07-08
Branch: `codex/robust-dual-subproblem`
Commit: `pending`
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
- Added robust dual `objective_bound` so non-optimal MILP solves can expose Gurobi's maximization upper bound.
- Updated Benders UB handling: optimal robust dual target subproblems use incumbent objective; non-optimal target subproblems use `objective_bound` as a conservative UB when available; otherwise the iteration is marked with `valid_UB=False`.
- Added iteration log and final metadata fields for subproblem status, MIP gap, objective bound, and whether the UB used a subproblem bound.
- Updated configs and README to document the new default and mode meanings.
- Added robust dual tests for full-enumeration agreement, cut exactness at the current point, cut validity at other points, and Benders integration.
- PR #3 has been merged, and this PR has been retargeted to `main`.

## Verification

- Ran `pytest tests/test_robust_dual_subproblem.py -q`.
- Result: 4 passed.
- Ran `pytest tests -q`.
- Result: 14 passed.
- Hidden / bidirectional Unicode scan: no remaining hidden format/control characters found in changed files.

## Review Notes for ChatGPT

- Check the robust dual MILP objective against the paper derivation.
- Check the McCormick linearization bounds for `w = z * lambda` and `g = z * nu`.
- Check whether Benders metadata and iteration log fields are sufficient for experiment audit, especially non-optimal robust dual subproblem cases.

## Next Steps

- Add larger experiment configs comparing `robust_dual_milp` against full scenario enumeration where full enumeration is tractable.
