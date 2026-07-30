# Final Medium-large fairness holdout delivery

## Scope

This change freezes the failed Large remediation Attempt 5 as a terminal development result and preregisters the only final Medium-large fairness holdout. It does not change the mathematical model, uncertainty set, Farkas certification, frozen candidate definitions, seeds, rho grid, tolerances, or time limits.

The Large archive is read-only and is identified by SHA256 `09B41862A5BFED724EDBEC1E64996B54AA878119F5C0DEDFE5B10126B2525A98`. Its decision is `stop_final_large_remediation`; no additional Large run is authorized.

## Final matrix

- Scale: `medium_large`
- Holdout seeds: 170–179
- Rho: 0, 0.01, 0.025, 0.05, 0.10
- Candidates: `single_cut` and `persistent_certified_cache_batch5`
- Tasks: 10 shared baselines plus 100 frontier tasks, 110 total
- Exact post-evaluation: 1,831 scenarios for every certified frontier result

The formal output directory is new and was absent before and after dry-run. Previous instances, anchors, runs, checkpoints, manifests, results, and summaries are not reused.

## Statistical independence

The independent unit is the seed. Each rho has exactly ten paired seed observations. The overall comparison first averages the five rho differences within each seed and then resamples the ten seed clusters. A seed cluster always carries all rho values and both methods together. Five per-rho Wilcoxon tests, when their preregistered conditions hold, receive Holm correction. The 50 seed-rho pairs are never treated as 50 independent observations.

## Verification

- Attempt 5 ZIP SHA and CRC independently verified; 2/2 pipeline tasks complete, baseline 1/1 certified, frontier 0/1 (`time_limit_uncertified`).
- Holdout access audit found no generated instance, solver result, baseline, anchor, run, or checkpoint for seeds 170–179. Repository planning declarations are recorded separately and do not count as access.
- Dry-run: 10 baselines, 100 frontier tasks, 110 unique run keys, zero duplicates, no instance generation, no solver call, and no output directory creation.
- The formal entry validates the frozen Large stop decision and holdout non-access evidence before creating output, generating instances, or invoking a solver.
- Solver-free targeted tests and all historical static audits pass.

This pull request must remain Draft until independent review. No formal holdout execution is authorized before merge and the next-stage review decision.
