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
`joint_v1_core_point_strengthened`.  Each instance runs that baseline once to
obtain certified \(C^*\).  No V3 parameter is reselected.  The model and Farkas
cut derivation are in `docs/robust_regional_fairness_model.md`.

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
frontier point.  Run records, resolved configuration, and manifest updates are
atomically replaced.  `--resume` skips complete successful records and refuses
to reinterpret a failed baseline as \(C^*\).  `--overwrite` is explicit and is
mutually exclusive with `--resume`.

The manifest freezes the Git commit and canonical configuration SHA256.  Each
fairness run records its baseline run key, \(C^*\), \(\rho\), \(B_\rho\), bounds,
requested gaps, cut count, cost/fairness scenario patterns encountered,
runtime, PAR-2, and final status.  Formal result analysis must additionally
recover scenario policies under the shared caps to report WGap, WWD, mean fill
rate, and opening/inventory changes.  That deterministic all-scenario reporting
pass has a 30-second per-scenario cap, is stored as post-evaluation, and is not
included in the fairness Benders algorithm runtime or PAR-2.

## Pre-registered checks and metrics

Model validity requires:

- every scenario total cost is at most \(B_\rho\) plus frozen tolerance;
- every applicable regional shortage rate is at most \(T\) plus tolerance;
- actual price of fairness is no greater than \(\rho\) plus tolerance;
- optimal \(T\) is non-increasing as \(\rho\) increases, up to tolerance;
- \(\rho=0\) uses the same cost-optimal feasible-set logic as diagnostic
  fair-best recourse (it need not retain the diagnostic's fixed first stage);
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

If all runs are valid but the material-improvement rule fails, the outcome is
`stop_no_material_improvement`; this is a valid negative result.  If completion
is below 80% without a mathematical correctness failure, the outcome is
`development_inconclusive`.  Thresholds cannot be changed after seeing
development data.

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
