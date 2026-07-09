# Handoff: Add Cut Selection Strategy

Date: 2026-07-09
Branch: `codex/cut-selection-strategy`
Commit: `a211813`
PR: `https://github.com/Hoshino12172003/robust-inventory-benders/pull/5`

## Summary

- Added violation-based Benders cut selection.
- Cut violation is computed as `cut_rhs(x^k) - theta^k`.
- Added algorithm settings:
  - `cut_selection_enabled`
  - `delta_cut`
  - `cut_violation_tol`
- Added iteration log fields for cut selection decisions:
  - `cut_selection_enabled`
  - `delta_cut`
  - `cut_rhs_current`
  - `cut_violation`
  - `cut_added`
  - `cut_skip_reason`
  - `cut_add_reason`
  - `cuts_added_total`
  - `cuts_skipped_total`
- Added final metadata for cut selection experiment summaries.
- Preserved scenario enumeration and robust dual MILP subproblem logic.
- Kept `objective_bound` restricted to conservative upper-bound updates; it is not used for cut generation.
- Added a target-Gamma safety mechanism that can force a positive-violation cut when skipping it would block progress.
- Updated default and experiment configs.
- Added README documentation for the cut selection strategy.
- Added `tests/test_cut_selection.py`.

## Verification

- Ran `pytest tests/test_cut_selection.py -q`.
- Result: 5 passed.
- Ran `pytest tests -q`.
- Result: 19 passed.
- Added and ran `scripts/check_hidden_unicode.py` across all git tracked files.
- Initial scan found U+FEFF at the beginning of `.gitignore`, `requirements.txt`, and `src/__init__.py`.
- Ran `python scripts/check_hidden_unicode.py --fix` to delete those hidden characters.
- Second scan result: `No hidden Unicode characters found.`
- Ran `git diff --check`.
- Result: no whitespace or conflict-marker errors.

## Review Notes for ChatGPT

- Check whether the cut violation calculation uses the cut incumbent solution rather than `objective_bound`.
- Check whether the forced target progress rule is conservative enough for large `delta_cut` settings.
- Check whether the cut selection metadata is sufficient for experiment tables.

## Next Steps

- Add experiment configs that compare `delta_cut = 0` with positive cut thresholds.
- Consider adding a max-skipped-cuts safeguard after larger experiments reveal practical behavior.
