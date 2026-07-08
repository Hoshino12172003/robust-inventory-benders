# Handoff: Add Exact Scenario Mode and Scenario Metadata

Date: 2026-07-08
Branch: `codex/exact-scenario-mode`
Commit: `pending`
PR: `https://github.com/Hoshino12172003/robust-inventory-benders/pull/3`

## Summary

- Added `ScenarioEnumerationResult` to carry scenarios and enumeration metadata together.
- Added `enumerate_budget_scenarios_with_metadata(...)` as the primary scenario enumeration API.
- Kept `enumerate_budget_scenarios(...)` as a backward-compatible wrapper returning only scenarios.
- Exact mode now raises `ValueError("Exact scenario enumeration exceeds max_scenarios. Increase max_scenarios or set exact_scenarios=False.")` instead of silently using candidate scenarios.
- Candidate fallback is now only allowed when `exact_scenarios` is false.
- Added target-scenario metadata to Benders solve results:
  - `scenario_mode_target`
  - `exact_scenarios`
  - `num_target_scenarios_used`
  - `num_target_scenarios_total_estimated`
  - `max_scenarios`
  - `scenario_modes_by_gamma`
  - `heuristic_scenarios`
- Updated monolithic solve metadata while keeping monolithic as a small-scale exact benchmark that does not use candidate scenarios.
- Updated README to distinguish exact full enumeration from heuristic candidate mode and to state that `exact_scenarios: true` is the paper-experiment default.
- Added `tests/test_scenario_modes.py` for exact full enumeration, exact overflow failure, candidate fallback metadata, and Benders metadata.

## Verification

- Ran `pytest tests -q`.
- Result: 10 passed.

## Review Notes for ChatGPT

- Check whether `ScenarioEnumerationResult` makes exact vs candidate behavior clear enough for paper audit.
- Check whether `scenario_mode_target`, `scenario_modes_by_gamma`, and `heuristic_scenarios` are the right metadata names for experiment tables.
- Check whether exact mode can still accidentally call `candidate_budget_scenarios`.
- Check whether candidate mode is sufficiently marked as heuristic / approximate in README and metadata.

## Next Steps

- Implement `robust_dual_subproblem` / `robust_dual_milp` in a future PR.
- Add experiment configs that explicitly set `exact_scenarios: true` for paper-grade runs.
