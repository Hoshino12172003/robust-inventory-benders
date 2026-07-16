# Frozen managerial sensitivity protocol v1

## Purpose

This protocol studies model decisions and cost/service trade-offs with the already frozen `proposed_joint_rho025_050` method. It does not repeat the five-method runtime comparison and is not a new tuning stage. No sensitivity outcome is known or asserted at protocol-freeze time.

## Frozen common design

- Configuration: `experiments/configs/managerial_sensitivity_joint_v1.yaml`
- Experiment: `managerial_sensitivity_joint_v1`
- Output: `experiments/results_managerial/managerial_sensitivity_joint_v1`
- Instance size: `medium_large`
- New seeds: 30–39
- Variant: `proposed_joint_rho025_050` only
- Algorithm time limit: 900 seconds
- Iteration limit: 10,000
- Global tolerance: `1e-4`
- Exact post-evaluation time limit: 300 seconds, recorded separately
- Robust subproblem: `robust_dual_milp`
- Gamma continuation, cut selection, secondary selection/generation, and legacy adaptive schedules are disabled
- At most one cut is added per iteration

Selected algorithm parameters remain inherited from `experiments/configs/selected_algorithm_parameters.yaml`. The 190-run serial algorithm-limit upper bound is 190 × 900 seconds = 47.5 hours. This excludes separately recorded post-evaluation time and is not a runtime prediction. The suite does not automatically run multiple Gurobi processes in parallel.

## One-factor-at-a-time baseline

The common managerial baseline is:

| Factor | Baseline |
|---|---:|
| `gamma_target` | 2 |
| uniform `service_level` | 0.90 |
| `budget_factor` | 0.68 |
| `capacity_factor` | 1.25 |

The uniform service-level baseline is specific to this managerial protocol. It must not be described as identical to the product-specific random service levels in `final_evaluation_joint_v1`.

Only one factor changes within each axis:

| Axis | Values | Runs |
|---|---|---:|
| `gamma_target` | 0, 1, 2, 3, 4 | 50 |
| `service_level` | 0.82, 0.86, 0.90, 0.94 | 40 |
| `budget_factor` | 0.55, 0.62, 0.68, 0.75, 0.82 | 50 |
| `capacity_factor` | 1.05, 1.15, 1.25, 1.35, 1.45 | 50 |

Total: 190 runs. For Gamma sensitivity, every run uses `gamma_schedule = [gamma_target]`; no continuation occurs inside a run. Shortage penalties, service penalties, and demand-deviation ranges are outside this protocol.

Every expanded record explicitly stores the axis, value, baseline, seed, size, variant, and full resolved configuration.

## Independent managerial evaluation

Algorithm runtime ends before managerial post-evaluation begins. If the algorithm saved a valid first-stage incumbent, the post-evaluator:

1. fixes the saved best inventory decision;
2. solves the target-Gamma robust dual MILP with `MIPGap = 0` and its own time limit;
3. requires an optimal incumbent;
4. extracts one solver-selected worst-case demand pattern;
5. solves the linear recourse problem exactly for that demand;
6. reports the decision, cost, shortage, service-violation, and fill-rate metrics.

Robust dual optimization can have multiple tied worst-case demand patterns. Reported scenario-level metrics correspond to one solver-selected worst-case pattern and must not be described as the unique worst case.

The output metrics are:

- opened warehouses;
- total inventory, inventory by product, and inventory by warehouse;
- fixed opening, inventory, and total first-stage cost;
- worst-case recourse, transport, shortage, and service-violation cost;
- total worst-case demand;
- total and product-level shortage;
- total and product-level service violation;
- realized fill rate;
- active deviations and worst-case demand matrix;
- evaluation status, error, runtime, and validity flag.

Fill rate is `1 - total_shortage / total_worst_case_demand`. If total demand is zero, fill rate is recorded as null rather than dividing by zero. A failed post-evaluation records `managerial_metrics_valid = false`; it never substitutes guessed or partial metrics.

## Theory checks and nonclaims

For the same seed and otherwise identical model, the following objective-value directions can be checked:

1. increasing Gamma should not decrease the optimal robust objective;
2. increasing the service target should not decrease the optimal objective;
3. increasing `budget_factor` enlarges the feasible set, so the optimal objective should not increase;
4. increasing `capacity_factor` enlarges the feasible set, so the optimal objective should not increase.

These statements concern optimal objective values and require comparable certified solves. This protocol does **not** assume monotonicity of opened warehouse count, total inventory, total shortage, product inventory, or service violation. Discrete facility choices and cost substitution can make those metrics nonmonotone.

No paper conclusion may be written until the frozen runs are completed, audited, and analyzed in a separate PR.

## Persistence, recovery, and commands

Per-run records and `run_manifest.json` follow the same atomic-write, resume, and overwrite semantics as the large-scale protocol.

POSIX shell:

```bash
python -m src.managerial_sensitivity_suite \
  --config experiments/configs/managerial_sensitivity_joint_v1.yaml \
  --resume
```

PowerShell:

```powershell
python -m src.managerial_sensitivity_suite `
  --config experiments/configs/managerial_sensitivity_joint_v1.yaml `
  --resume
```

Configuration-only dry run:

```bash
python -m src.managerial_sensitivity_suite --config experiments/configs/managerial_sensitivity_joint_v1.yaml --dry-run
```

Static audit:

```bash
python -m src.extended_experiment_audit --output protocol_audit.json
```
