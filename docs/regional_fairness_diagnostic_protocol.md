# Regional Service Fairness Diagnostic Protocol

## Authorization and purpose

This protocol is authorized only by the frozen V3 Final decision:

```yaml
decision: final_confirmed
selected_algorithm: joint_v1_core_point_strengthened
v3_status: completed
retuning_allowed: false
next_authorized_stage: fairness_diagnostic_only
```

The current PR is limited to `fairness_diagnostic_protocol_only`. It freezes a
descriptive post-solve diagnostic for regional service fairness (regional
service equity). It does not concern machine-learning fairness or protected
demographic attributes.

The diagnostic asks whether the cost-optimal budget-robust policy produces
material regional service differences, whether those differences persist
under a cost-optimal fair recourse selection, and whether a later
fairness-aware model-development protocol is justified. It does not presume
that a fairness constraint is desirable or that it will improve the model.

No base objective, constraint, dual, Benders cut, uncertainty-set definition,
or V3 parameter is changed. The diagnostic LPs never update the base run's LB,
UB, incumbent, or termination state and write only to a separate diagnostic
output tree.

## Audited model structure

Repository inspection confirms these indices and variables:

- warehouses are indexed by `i`, products by `j`, and regions by `r`;
- first-stage opening is `y[i]`, and inventory is `x[i,j]`;
- recourse transport is `q[i,r,j]`, shortage is `u[r,j]`, and service-level
  violation is `e[j]`;
- demand and demand deviation are stored as `[region][product]`;
- every warehouse-region-product transport variable exists. There is no
  prohibited-arc or geographic-distance field;
- the budget uncertainty pattern is binary over `(r,j)` with at most Gamma
  active deviations;
- the frozen base output serializes `best_y_values` and `best_x_values`, but it
  does not serialize regional transport or shortage allocations.

Because physical distance is absent, this protocol reports allocated unit
transport cost rather than inventing a distance measure. Structurally, every
region has all modelled warehouses reachable; `reachable_warehouse_count`
therefore equals `num_warehouses` under the current model. Any later
connectivity or distance extension changes the model and requires a separate
protocol.

The original recourse has positive transport, shortage, and service-violation
costs but may still have multiple cost-optimal allocations. One arbitrary
optimizer-returned allocation is therefore recorded as `default`, not treated
as the unique fairness outcome.

## Frozen regional metrics

For region `r` and extreme-point scenario `z`:

```text
D_r(z)  = sum_j d[r,j](z)
U_r(z)  = sum_j u[r,j](z)
FR_r(z) = 1 - U_r(z) / D_r(z)
```

If `D_r(z)=0` within `metric_tolerance=1e-9`, the region is marked
`not_applicable` with reason `zero_regional_demand`. It is excluded from the
scenario's min, max, standard-deviation, and Gini calculations. No division by
zero or zero-filled fairness value is allowed.

The demand-weighted mean fill rate is

```text
weighted_mean_FR(z) = 1 - sum_r U_r(z) / sum_r D_r(z).
```

Among applicable regions:

```text
DeltaFR(z) = max_r FR_r(z) - min_r FR_r(z)
FR_min(z)  = min_r FR_r(z)
WD(z)      = max_r {weighted_mean_FR(z) - FR_r(z)}.
```

For fixed first-stage inventory `x`, the robust diagnostic metrics are

```text
WGap(x)   = max_z DeltaFR(x,z)
WMinFR(x) = min_z FR_min(x,z)
WWD(x)    = max_z WD(x,z).
```

Primary metrics are fair-best robust WGap, fair-best robust WMinFR, default
robust WGap, and `default_WGap - fair_best_WGap`. Auxiliary metrics are regional
shortage rate, demand-weighted mean fill rate, regional fill-rate population
standard deviation, fill-rate Gini, regional transport cost, allocated unit
transport cost, and reachable warehouse count. Gini is descriptive only and
never controls the diagnostic decision.

Fill rates must lie in `[0,1]` within the frozen numerical tolerance. The
postprocessor clips only numerical noise within tolerance; a material shortage
above demand is a correctness failure.

## Cost-optimal degeneracy control

For each fixed saved `x` and scenario `z`, evaluation has two stages.

1. Solve the unchanged original linear recourse model to optimality and record
   `Q*(x,z)` and its default allocation.
2. Rebuild the same constraints, impose

   ```text
   Q(x,z) <= Q*(x,z) + epsilon_Q,
   epsilon_Q = 1e-6 + 1e-6 * max(1, abs(Q*(x,z))),
   ```

   and minimize the maximum applicable regional shortage rate.

Because regional demand is constant after `z` is fixed, all regional rate
constraints remain linear. The default allocation is a feasible reference for
the second LP. As a deterministic safeguard, the fair-best LP also constrains
its regional fill-rate gap not to exceed the feasible default gap plus
`metric_tolerance`; this prevents a tie in the primary max-shortage objective
from returning a needlessly worse gap.

The absolute and relative cost tolerances are frozen before any diagnostic run
and cannot be revised after results are observed. Both policies use identical
`x`, scenario ID, demand matrix, and Gamma usage. Fair-best cost must not exceed
the cost cap. It is a postprocessor result only and cannot update V3 bounds,
cuts, incumbents, or reported algorithm runtime.

The diagnostic categories have the following interpretation:

- `recourse_degeneracy_only`: default reaches the structural rule, fair-best
  does not, and at least 4/10 instances in a scale reduce WGap by at least
  0.05. A deterministic or lexicographic recourse rule is the next research
  question; this is not evidence for first-stage fairness constraints.
- `structural_fairness_gap`: fair-best still reaches a frozen material-gap
  rule. Only this outcome directly authorizes a separate fairness-aware model
  development protocol.
- `no_material_fairness_gap`: both scales meet the frozen low-gap rule and
  default does not reach the structural rule.
- `fairness_diagnostic_inconclusive`: every remaining valid outcome.

No fair-worst envelope is included in this protocol, avoiding a second
unbounded expansion of per-scenario solves. Such an envelope would require a
separate pre-registered auxiliary protocol and could not replace fair-best.

## Scenario scope

Gamma remains exactly 2. The diagnostic reuses the repository's exact budget
scenario enumeration and evaluates every binary extreme point with zero, one,
or two active `(region,product)` deviations. It includes the nominal scenario
and identifies the exact cost-worst and fairness-worst scenarios separately.
The cost-worst scenario is not assumed to be the fairness-worst scenario.

The counts are calculable from frozen dimensions without generating an
instance:

| Scale | Regions | Products | Units | Extreme points at Gamma 2 |
|---|---:|---:|---:|---:|
| medium-large | 10 | 6 | 60 | `1 + 60 + C(60,2) = 1831` |
| large | 12 | 8 | 96 | `1 + 96 + C(96,2) = 4657` |

Both are below the frozen exact-scenario safety limit of 5,000. If a future
configuration exceeds that limit, execution must stop. Sampling or a candidate
scenario subset is forbidden unless a new scenario-generation protocol is
reviewed first.

## Frozen runs and seed isolation

Only `joint_v1_core_point_strengthened` is included.

- medium-large diagnostic seeds: 110--119, giving 10 base optimization runs;
- large diagnostic seeds: 110--119, giving 10 base optimization runs;
- total: 20 base runs.

The repeated labels across scales identify different instances because the
scale dimensions differ. No V1, MP-only, secondary-only, full V3, or new
candidate is included. Seeds 75--109 are excluded.

The following future fairness-model seeds are reserved in metadata but are not
run or used to generate instances in this PR:

- development: 120--129;
- validation: 130--139;
- final medium-large: 140--149;
- final large: 150--159.

Diagnostic seeds 110--119 may inform whether fairness model development is
warranted, but cannot be reused for a later fairness model's validation or
final test.

## Frozen entry rule

For fair-best WGap, `structural_fairness_gap` requires all correctness checks
and either:

1. at least 4/10 instances in either scale have `WGap >= 0.10` and that
   scale's median WGap is at least 0.05; or
2. at least 8/20 combined instances have `WGap >= 0.10` and the combined
   median WGap is at least 0.05.

`recourse_degeneracy_only` requires the default WGap values to meet that same
structural rule, fair-best not to meet it, and at least 4/10 instances in at
least one scale to reduce WGap by at least 0.05.

`no_material_fairness_gap` requires, in both scales, no more than 1/10
fair-best instances with `WGap >= 0.10`, median fair-best WGap below 0.03, and
default not to meet the structural rule. Every other valid result is
`fairness_diagnostic_inconclusive`.

The values 0.10, 0.05, and 0.03 are pre-registered material-effect thresholds.
Counts at 0.05, 0.10, and 0.15 may be displayed as sensitivity summaries but
cannot change the main classification.

## Correctness audit

Every formal diagnostic must establish:

- the fixed `best_x_values` shape, hash, inventory upper bounds, and warehouse
  capacities match the base V3 output;
- neither default nor fair-best modifies `x`;
- the default objective reproduces the unchanged original recourse cost within
  tolerance;
- fair-best cost satisfies `Q* + epsilon_Q`;
- demand, supply, service, and nonnegativity constraints hold;
- default and fair-best share exactly the same `x` and `z`;
- scenario patterns are unique and their IDs, Gamma usage, and demand values
  are reproducible;
- zero-demand handling and fill-rate bounds are consistent;
- cost-worst and fairness-worst scenarios are separately recorded;
- diagnostic LPs never update LB, UB, cuts, or the base solution;
- no diagnostic output is written into development, validation, Final, or
  managerial-sensitivity directories.

Any failed correctness item invalidates the diagnosis; missing metrics must not
be filled with fabricated zero values.

## Frozen output contract

Formal outputs, if later authorized, are isolated under:

- `experiments/results_fairness_diagnostic/medium_large`;
- `experiments/results_fairness_diagnostic/large`.

Each scale must contain `results.csv`, `region_scenario_metrics.csv`,
`instance_summary.csv`, `resolved_config.yaml`, `run_manifest.json`,
`diagnosis.json`, and `audit_log.json`.

`region_scenario_metrics.csv` includes instance size, seed, scenario ID and
kind, region, regional demand and shortage, shortage/fill rate, applicability,
weighted mean fill rate, gap, worst-region deviation, dispersion metrics,
transport diagnostics, recourse policy, original/evaluated cost, cost
tolerance, Gamma usage, and the fixed-`x` SHA256.

`instance_summary.csv` includes seed, size, default/fair-best WGap, WMinFR and
WWD, nominal and cost-worst gaps, separately identified cost-worst and
fairness-worst scenarios, WGap reduction, category, scenario count, and the
fixed-`x` SHA256.

## Dry-run and execution boundary

The only allowed configuration commands in this PR are dry-runs:

```powershell
python -m src.experiment_suite `
  --config experiments/configs/regional_fairness_diagnostic_medium_large.yaml `
  --dry-run

python -m src.experiment_suite `
  --config experiments/configs/regional_fairness_diagnostic_large.yaml `
  --dry-run
```

Dry-run statically expands 10 runs per scale and reports 1,831 and 4,657
extreme-point scenarios. It must not generate instances, initialize Gurobi, or
create either formal output directory.

No formal diagnostic, fairness-model experiment, or managerial-sensitivity
experiment is authorized by this PR. A later diagnostic execution and its
read-only decision require separate tasks and separate review.
