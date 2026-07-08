# Handoff: Add Exact Scenario Mode and Scenario Metadata

Date: 2026-07-08
Branch: `codex/exact-scenario-mode`
Commit: `pending`
PR: `pending`

## Summary

- Added explicit `exact_scenarios` control to budgeted scenario enumeration.
- Exact mode now raises `ValueError("Exact scenario enumeration exceeds max_scenarios.")` instead of silently using candidate scenarios.
- Candidate fallback is now only allowed when `exact_scenarios` is false.
- Added scenario metadata to solve results:
  - `scenario_mode`
  - `exact_scenarios`
  - `num_scenarios_used`
  - `num_scenarios_total_estimated`
  - `max_scenarios`
- Updated README to distinguish exact full enumeration from heuristic candidate mode.
- Added tests for exact full enumeration, exact overflow failure, and candidate fallback metadata.

## Verification

- Ran `pytest tests -q`.
- Result: 6 passed.

## Review Notes for ChatGPT

- Check whether `scenario_mode` semantics are clear and correctly attached to both monolithic and Benders results.
- Check whether exact mode can still accidentally call `candidate_budget_scenarios`.
- Check whether candidate mode is sufficiently marked as heuristic / approximate in README and metadata.

## Next Steps

- Implement `robust_dual_subproblem` / `robust_dual_milp` in a future PR.
- Add experiment configs that explicitly set `exact_scenarios: true` for paper-grade runs.
