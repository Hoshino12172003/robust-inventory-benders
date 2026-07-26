# Fairness scalability development protocol

Status: preregistered protocol only. Formal execution is not authorized by this PR.

## Scope and frozen scientific model

This stage investigates engineering scalability of the already frozen robust
regional-service fairness model. It does not change the objective, recourse,
cost budget, uncertainty set, cost anchor, success definition, or scientific
selection thresholds. It does not make claims about demographic or vulnerable
groups. Attempt 3 is retained only as a frozen runtime-pipeline incident and an
engineering motivation; its 57/60 and 28/60 counts and all seed-level outputs
are prohibited from parameter, rho, threshold, or algorithm selection.

The model remains the epsilon-constraint max-min formulation: minimize the
worst applicable regional shortage ratio `T`, with one scenario-specific
recourse policy satisfying both the original recourse constraints and the
scenario cost cap `(1 + rho) C_anchor`. Gamma remains 2, the exact extreme-point
set remains unchanged, the fairness algorithm time limit remains 1,800 seconds,
and only `certified_robust_optimal` is a solved run. PAR-2 remains algorithm
runtime for solved runs and twice 1,800 seconds otherwise. The frozen 80%
certified-solved gate is unchanged.

## Candidates and their only permitted differences

All candidates use identical masters, joint V1 precision control, Farkas cone,
fixed-scenario primal/Farkas certification, tolerances, bounds, and stopping
conditions.

1. `single_cut`: rebuild the complete separation MILP for each master point and
   add at most one independently certified cut.
2. `persistent_separation`: retain the same separation MILP across master
   iterations and update only current-point objective coefficients; add at most
   one independently certified cut.
3. `persistent_certified_cache`: additionally retain only historical scenario
   patterns as candidate hints. Every cache hit is resolved by a fresh
   fixed-scenario primal/Farkas certification at the current `(x,T,B_rho)`.
4. `persistent_certified_cache_batch5`: additionally request at most five
   distinct solution-pool patterns and add at most five cuts, each independently
   certified at the current point.

Solution-pool incumbents and cached patterns are candidate generators only.
Old rays and cuts are never reused. A current fixed-scenario infeasibility
certificate is required for every cut. A cache hit never certifies robust
feasibility. Only a valid objective bound from complete separation can certify
that the remaining uncertainty set contains no violation. False-positive
scenario exclusions are exact no-good constraints local to one separation call
and are removed before the next master iteration.

## New seeds and staged execution

Development labels are frozen to 160--169 for both scales. Seeds 130--159 are
sealed and may not be accessed. No formal seed is used by unit tests.

- S0: deterministic hand-built tiny instances and independent extensive-form
  comparisons only.
- S1: seeds 160--162, rho in `{0, 0.01}`, all four candidates. This is 3
  baselines plus 24 frontier tasks per scale.
- S2: seeds 160--169, rho in `{0, 0.01}`, all four candidates. S1 is a subset;
  the cumulative unique plan is 10 baselines plus 80 frontier tasks per scale.
- Full grid: allowed only after S2. A candidate must pass every correctness and
  certification check and obtain at least 16/20 `certified_robust_optimal`
  frontier runs at rho 0 and 0.01. Only the selected candidate may then run the
  three remaining rho values `{0.025, 0.05, 0.10}`, adding 30 tasks per scale.
  The complete staged unique plan is therefore 120 tasks per scale.

Formal execution requires a later pre-run audit and explicit authorization.
The protocol-only CLI refuses non-dry-run invocation.

## Frozen selection rule

Candidates are ordered lexicographically by:

1. mathematical and certification correctness;
2. certified solved count, higher first;
3. PAR-2, lower first;
4. separation runtime, lower first;
5. total wall runtime, lower first.

A correctness failure makes a candidate ineligible regardless of speed. S2
below 80% certified solved prevents the full rho grid. No observed result may
change these rules. Negative or inconclusive findings are retained.

## Required evidence

Iteration and result evidence includes `separation_model_build_runtime`,
`separation_optimize_runtime`, `cache_candidate_count`, `cache_hit_count`,
`certified_cached_cut_count`, `pool_candidate_count`,
`certified_batch_cut_count`, `duplicate_pattern_count`, `cuts_per_iteration`,
`total_iterations`, `algorithm_runtime`, and `total_wall_runtime`. Algorithm
runtime and end-to-end wall time remain distinct.

## Dry-run

Dry-run reads configuration only, performs no instance generation or solver
call, and does not create the formal output directory:

```powershell
python -m src.fairness_scalability_suite `
  --config experiments/configs/fairness_scalability_development_medium_large.yaml `
  --dry-run

python -m src.fairness_scalability_suite `
  --config experiments/configs/fairness_scalability_development_large.yaml `
  --dry-run
```
