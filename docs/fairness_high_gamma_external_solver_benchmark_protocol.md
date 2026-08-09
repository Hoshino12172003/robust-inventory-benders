# Certified fairness high-Gamma external-solver benchmark protocol

## Purpose and scope

This preregistered experiment tests structural sensitivity at higher budgeted uncertainty and compares the frozen certified Hybrid method with a solver-generic direct extensive form. It is a controlled small-scale benchmark, not a tuning stage. It does not alter Final Holdout, Gamma Attempt 3, D1, D2, or the minimal paired benchmark.

## Frozen matrix

- stage: `HIGH_GAMMA_EXTERNAL_BENCHMARK`
- scale: `small`, with 4 warehouses, 4 products, and 5 regions
- seeds: 185, 186, 187, 188, 189
- Gamma: 2, 3, 4
- rho: 0.025
- execution attempt: 1
- baseline: 15 independent Gamma-specific runs
- Hybrid frontier: 15 runs using `certified_hybrid_scenario_benders_fairness`
- direct frontier: 15 runs using `gurobi_direct_extensive_form`
- total: 45 runs

The uncertainty dimension is 20. Complete scenario counts are
`sum(comb(20,k), k=0..Gamma)`: 211, 1,351, and 6,196 for Gamma 2, 3, and 4. A baseline and certified anchor are unique to each seed-Gamma cell and are shared only by the two paired frontier methods in that same cell.

## Frozen scientific identity

Every instance archive, baseline, anchor, run, status, checkpoint, post-evaluation checkpoint, CSV projection, and manifest records scale, seed, Gamma, rho applicability, method, execution attempt, Git commit, config SHA, protocol SHA, candidate SHA, solver parameters, instance canonical SHA, instance archive file SHA, baseline run key, and anchor SHA. Canonical and file SHA fields are distinct. Cross-Gamma or cross-method checkpoint reuse fails closed.

## Hybrid method

The Hybrid candidate, mathematical model, fairness definition, budgeted uncertainty set, complete scenario recourse blocks, certified Farkas separation, final exact objective-bound certification, and scientific branching are unchanged. Threads=1, solver Seed=0, FeasibilityTol=1e-7, time limit=1,800 seconds, and at most one independently certified scenario/cut is committed per iteration. Committed scenario blocks and certified Farkas cuts are append-only.

The existing working candidate pool is the ephemeral, call-local set of distinct proposals returned by one separation call; it is not the committed master. A proposal not selected for commitment at the end of that call is counted as evicted for reporting only. Reporting-only counters are: maximum working candidate-pool size, eviction count, rediscovered previously evicted proposal count, duplicate proposal count, unique committed scenario blocks, committed Farkas cuts, and consecutive non-improving iterations. These counters never participate in selection, termination, branching, cache state, or master construction. Removing a working proposal never removes a committed master constraint.

## Direct extensive form

The direct method explicitly enumerates every scenario and creates complete scenario-specific recourse variables and constraints while sharing first-stage binary and inventory variables. Its first-stage budget, robust cost cap, regional fairness objective T, baseline anchor, and rho are identical to Hybrid. It does not call Hybrid separation, Farkas cuts, caches, candidate pools, or Gurobi Benders decomposition. `BendersStrategy=0` is recorded. When the installed Gurobi version exposes `BendersStrategy`, the runner explicitly sets it to zero. If that version does not expose the parameter, the runner records zero and the parameter absence itself establishes that built-in Benders cannot be enabled; any other parameter-setting error fails closed.

The 1,800-second algorithm limit covers model construction plus optimization. Remaining optimize time is `max(0, 1800-build_runtime)`; construction that exhausts the limit does not receive an additional optimize budget. The method records construction, optimize and total runtime; rows, columns, binaries, continuous variables and nonzeros; incumbent, objective bound and MIP gap; solver status; and memory/resource failure. A construction time limit or resource failure is a scientific performance result, not a pipeline failure.

Direct scientific success requires a completely built deterministic equivalent, an acceptable optimal solver status, finite lower and upper bounds with frozen relative gap at most 1e-4, and valid complete post-evaluation. Any time limit, resource failure, missing legal bound, invalid post-evaluation, or incomplete exact model is not certified and receives 3,600 seconds PAR-2.

## Common runtime and evaluation rules

- baseline, Hybrid, direct algorithm limit: 1,800 seconds
- post-evaluation limit: 30 seconds per scenario
- post-evaluation chunk size: 25
- PAR-2 multiplier: 2, based only on algorithm runtime
- algorithm, post-evaluation, aggregation/checkpoint, and total wall runtime remain distinct
- only final exact certified Hybrid or a completely solved direct deterministic equivalent, followed by valid post-evaluation, may be `certified_robust_optimal`

## S0 mathematical equivalence gate

Before formal authorization, three fixed non-formal tiny instances are tested at Gamma 0 through 4. Hybrid and direct use the same instance, valid baseline evidence, anchor, rho, solver parameters, and complete post-evaluation. Their objective T, legal lower/upper bound sandwich, cost cap, and post-evaluation values must agree within 1e-7 (or the stricter applicable frozen tolerance). These S0 checks are correctness tests and are not part of the 45 formal tasks or paper sample.

## Reporting and inference

Results include all preregistered cells. Uncertified methods remain in planned/completed rates and PAR-2; objective and cost differences are reported only when both methods certify. Seed is the independent unit. Gamma is repeated within seed, overall summaries aggregate Gamma within seed first, and any bootstrap resamples whole seed clusters. With five seeds, conclusions are descriptive and no statistical equivalence is claimed.

## Authorization and immutability

Formal execution requires the separately reviewed authorization JSON, a clean approved Git tree, an absent first-attempt output root, and strict `--resume`. `--overwrite` is unsupported. Authorization occurs before output creation, instance generation, solver configuration, or `gurobipy` import. The formal run may not expand the matrix, rerun earlier experiments, reuse prior benchmark results, or tune parameters from outcomes.
