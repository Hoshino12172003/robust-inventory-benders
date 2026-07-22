# Regional service fairness model development protocol

## Authorization and research question

The frozen diagnostic decision is `structural_fairness_gap`, with a valid
diagnosis and source `structural_not_recourse_degeneracy`.  This protocol asks
whether an interpretable robust cost allowance can improve the worst regional
service level.  It authorizes implementation and a future development run,
not validation, final testing, or a claim of positive results.

The scope remains regional service fairness.  Unit transportation cost is not
physical distance.  There are no observed population-vulnerability data, so
the study cannot make claims concerning low-income, older, racial, or other
socially disadvantaged groups.  Vulnerability weights may be considered only
in a later, data-supported protocol.

## Frozen design

The frozen baseline and cost source are
`joint_v1_core_point_strengthened`. Each instance runs that baseline once.
The full-precision certified conservative `SolveResult.upper_bound`, with
`valid_UB=true` and a final certified gap no greater than `tol`, is frozen as
\(C_{\rm anchor}\). The lower bound, midpoint, master objective, rounded
summary value, and a single-scenario cost are prohibited. No V3 parameter is
reselected. The model and Farkas cut derivation are in
`docs/robust_regional_fairness_model.md`.

The cost grid is frozen before development:

\[
\rho\in\{0.00,0.01,0.025,0.05,0.10\}.
\]

Every point must be retained and reported.  It is forbidden to display only a
favorable point.  The primary objective is the worst regional shortage rate
\(T\); robust minimum fill rate is \(1-T\).  The formal development config
does not enable a lexicographic cost stage because the scalable second-stage
cut derivation is not yet frozen.  The extensive-form oracle may use the
optional stage, with an absolute \(T\) tolerance of \(10^{-7}\), only to check
small-instance degeneracy.

Numerical tolerances are frozen as follows:

- cost absolute tolerance: \(10^{-6}\);
- cost relative tolerance: \(10^{-6}\);
- fairness feasibility tolerance: \(10^{-7}\);
- metric and zero-demand tolerance: \(10^{-9}\);
- algorithm termination tolerance: \(10^{-4}\).

The same recourse policy must satisfy the scenario cost cap and every regional
service cap.  A non-optimal, infeasible, missing, or numerically invalid
recourse/separation result is not silently discarded.

## Instances and isolation

Development seeds are exactly 120--129 for each scale:

| Scale | Dimensions (warehouses, products, regions) | Baseline runs | Frontier runs |
|---|---:|---:|---:|
| medium-large | (6, 6, 10) | 10 | 50 |
| large | (8, 8, 12) | 10 | 50 |

The two configurations are independent and use output directories
`experiments/results_regional_fairness_model/development_medium_large` and
`experiments/results_regional_fairness_model/development_large`.

Reserved and unused by this PR:

- validation: 130--139;
- final medium-large: 140--149;
- final large: 150--159.

Diagnostic seeds 110--119 cannot be reused.  No validation or final config is
created by this PR.

## Frozen baseline and algorithm controls

The baseline retains Gamma=2, schedule `[2]`, no Gamma continuation, the
robust-dual MILP, `joint_error_budget`, ratios 0.25/0.50, gap bounds
0.02/0.0001 and 0.05/0.0001, core-only strengthening, one cut, final
certification, no legacy selection, and no secondary generation.

The fairness master and Farkas separation reuse the same joint precision
policy.  A separation incumbent may add a valid cut; only the objective bound
can certify that no violated scenario/ray remains.  Final certification forces
both requested gaps to zero.  No automatic multi-instance parallelism is
introduced.

## Outputs and recovery

The runner uses stable keys for the per-seed baseline and every `(seed,rho)`
frontier point. Run records, resolved configuration, and manifest updates are
atomically replaced. Attempt 3 must start in a physically nonexistent output
directory; even an empty existing directory is rejected. The identity manifest
is atomically created before any instance, baseline, anchor, or run artifact.
Only `--resume` may enter an existing directory, and only when its schema-3
Attempt 3 identity exactly matches Git, config, protocol, candidate, run keys,
and the frozen prior-attempt history. Formal development rejects `--overwrite`.

The generic manifest and the atomic `fairness_development_manifest.json`
freeze the Git commit, canonical configuration SHA256, candidate SHA256, all
run keys, and every per-seed anchor. Each anchor records its source, decimal
and IEEE-754 hexadecimal value, baseline run key, `valid_UB`, baseline final
gap, Git commit, baseline config SHA256, and its own canonical SHA256. Each
fairness run records the same anchor SHA256, \(\rho\), \(B_\rho\), bounds,
requested gaps, cut count, cost/fairness scenario patterns encountered,
runtime, PAR-2, and final status.  Formal result analysis must additionally
recover scenario policies under the shared caps to report WGap, WWD, mean fill
rate, and opening/inventory changes.  That deterministic all-scenario reporting
pass has a 30-second per-scenario cap, is stored as post-evaluation, and is not
included in the fairness Benders algorithm runtime or PAR-2.

The post-evaluation recovery LP enforces the exact mathematical right-hand
sides `recourse_cost <= B_rho - first_stage_cost` and
`regional_shortage <= T * regional_demand`. The frozen feasibility tolerance
is assigned to the solver and used to verify residuals; it is not added to
those right-hand sides. This separation avoids classifying a policy on the
accepted numerical boundary as invalid because of floating-point
representation. Acceptance uses the frozen, scale-independent boundary
`acceptance_threshold = math.nextafter(feasibility_tolerance, math.inf)`.
Thus `floating_point_slack = acceptance_threshold - feasibility_tolerance` is
exactly one representable step at `1e-7`; it does not depend on costs, demands,
or model size. Each check stores tolerance, threshold, slack, unrounded `lhs`,
`rhs`, raw residual, nonnegative residual, decision, constraint type, scenario,
and region. Aggregations use `math.fsum`. Residuals above the boundary remain
uncertified/invalid rather than receiving any scale-dependent allowance.

Frontier records preserve the algorithm's own status in `algorithm_status`
and use `overall_status` (also exposed as the public `status`) for end-to-end
interpretation. The mutually exclusive public enumeration is
`certified_robust_optimal`, `master_optimal_but_robust_uncertified`,
`time_limit_uncertified`, `iteration_limit_uncertified`,
`certified_infeasible`, `infeasible_uncertified`,
`invalid_post_evaluation`, `numerical_uncertified`, `interrupted`,
`implementation_error`, and `unknown_uncertified`. Only
`certified_robust_optimal` contributes to optimal solved-rate. An
algorithm-level `optimal` value cannot conceal failed or missing robust
post-evaluation; unrecognized raw statuses are never promoted to success.

Attempt 3 records `execution_attempt: 3`, all previously accessed development
seeds 120--129, and structured prior identities. Attempt 1 used commit
`7bc8e81f91f4a4c7baf2c080af63a09ada1178d6` and seeds 120--126 and was
invalidated by the separation-certificate architecture defect. Attempt 2 used
commit `98c615767032bb6c57f28476bebc0392037fbf34` and seeds 120--129 and was
invalidated by the post-evaluation tolerance-boundary defect. Both are
scientifically invalid, are not reused, and cannot supply a baseline, anchor,
run, manifest, or summary to Attempt 3.

A single-writer lock protects each scale output directory. Every run record
and both manifests are atomically replaced. `--resume` validates config, Git,
candidate, baseline run, anchor, and rho identity before reuse. It skips only
certified successful records; `time_limit`, `iteration_limit`, failed, and
uncertified attempts remain explicit and are rerun by `--resume`. An
interruption cannot create a duplicate frontier key. Concurrent writers are
rejected rather than allowed to mix output.

## Pre-registered checks and metrics

Model validity requires:

- every scenario total cost is at most \(B_\rho\) plus frozen tolerance;
- every applicable regional shortage rate is at most \(T\) plus tolerance;
- actual price of fairness is no greater than \(\rho\) plus tolerance;
- optimal \(T\) is non-increasing as \(\rho\) increases, up to tolerance;
- diagnostic fair-best fixes the diagnostic first-stage decision and removes
  recourse degeneracy only;
- the integrated \(\rho=0\) model may select a different first-stage decision
  among policies inside the same certified cost boundary, so a cost-neutral
  fair reconfiguration may outperform diagnostic fair-best;
- if the first-stage decision is additionally fixed, \(\rho=0\) must agree
  with the corresponding fixed-x extensive form within frozen tolerances;
- with free first-stage decisions, numerical equality to diagnostic fair-best
  is neither required nor claimed, and any gain is interpreted as first-stage
  reconfiguration rather than recourse-degeneracy removal;
- lower bounds do not decrease, certified upper bounds do not increase, and
  requested MP/SP gaps do not increase;
- separation incumbents create cuts only, while bounds certify feasibility.

Fairness reporting includes robust minimum fill rate, WGap, WWD,
demand-weighted mean fill rate, the number and fraction of improved instances,
and regional service outcomes.  Algorithm reporting includes solved rate,
runtime, PAR-2, iterations, cost/fairness cut counts, distinct cost-worst and
fairness-worst patterns, and bound trajectories.

## Development decision rule

Correctness failures stop the affected candidate immediately.  If correctness
passes, a later validation-protocol PR is authorized only when:

1. each scale has at least 80% solved frontier runs; and
2. at one pre-registered positive \(\rho\), at least 4/10 instances in either
   scale improve robust minimum fill rate by at least 0.05 relative to
   \(\rho=0\).

The primary endpoint is the paired per-instance increase in robust minimum
fill rate relative to that instance's \(\rho=0\) point. The report must retain
the count with improvement at least 0.05 and the median improvement for every
scale and every frozen rho. Among positive rho values satisfying the rule, the
single development candidate is the **smallest eligible positive rho**. This
cost-parsimony rule is deterministic and cannot be replaced after observing
the frontier. No alternative algorithm is selected in this development stage.

If all runs are valid but the material-improvement rule fails, the outcome is
`stop_no_material_improvement`; this is a valid negative result.  If completion
is below 80% without a mathematical correctness failure, the outcome is
`development_inconclusive`.  Thresholds cannot be changed after seeing
development data.

Timeouts and uncertified runs are unsolved, remain in the report, and receive
PAR-2 equal to twice their frozen time limit. They are never silently omitted.

### Allowed validation changes

Only experiment name, `protocol_phase`, seeds (the reserved 130--139 set), and
an isolated output directory may change in a later validation-protocol PR.
The model, uncertainty set, instance generator, rho grid, V3 parameters,
fairness tolerances, time limits, success definition, PAR-2 definition,
separation/certification logic, and selection threshold cannot change.

## Commands

Dry-run only (does not generate instances or invoke Gurobi):

```powershell
python -m src.fairness_benders `
  --config experiments/configs/regional_fairness_development_medium_large.yaml `
  --dry-run

python -m src.fairness_benders `
  --config experiments/configs/regional_fairness_development_large.yaml `
  --dry-run
```

Future formal order, not executed in this PR:

```powershell
python -m src.fairness_benders `
  --config experiments/configs/regional_fairness_development_medium_large.yaml `
  --resume

# Only after medium-large correctness acceptance:
python -m src.fairness_benders `
  --config experiments/configs/regional_fairness_development_large.yaml `
  --resume
```
