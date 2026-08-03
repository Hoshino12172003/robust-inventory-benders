# Gamma sensitivity reporting schema audit

This is a reporting-only audit. It does not change any optimization model, solver, certificate, run record, checkpoint, instance, or prior-attempt output.

## Attempt 2 boundary

The read-only field inventory of `E:\rfgs2\experiments\results_fh_gamma\ml_a2` found one completed baseline (`medium_large`, seed 180, Gamma 0), followed by one running frontier for the same cell. The frontier has no `run.json`, algorithm checkpoint, final exact certificate, or post-evaluation. `lg_a2` does not exist. Thus Attempt 2 contains no reusable frontier result and is frozen as `execution_incomplete`, `pipeline_identity_defect`, and `scientifically_usable=false`. Attempt 3 must not read any Attempt 1 or Attempt 2 artifact.

Attempt 3 subsequently completed the first `medium_large` seed 180, Gamma 0 baseline/frontier cell before the reporting projection failed. Its frontier has equal current lower/upper bounds, complete final exact certification, and valid one-scenario Gamma-0 post-evaluation. Those immutable optimization artifacts remain scientifically valid. The successor resume keeps their optimization identity at commit `b1b5e9908bbb685b8a852aff762f08ce7226aba1`, records the reviewed reporting and authorization commits separately in the manifest/audit log, and skips the completed instance, baseline, frontier, checkpoint, and post-evaluation.

## Production JSON-to-CSV projection

`I` is the instance's persisted warehouse order and `J` is its persisted product order. A frontier `x_values` payload is an `|I| x |J|` row-major JSON array and `y_values` is a length-`|I|` JSON array. Baseline uses the same order and dimensions under the distinct names `best_x_values` and `best_y_values`. Inventory is the finite sum over `I x J`; opened warehouses is the count of `y_i >= 0.5`. The projection rejects missing or extra rows/columns, mappings, booleans, strings, NaN, and infinity.

| CSV field | Source JSON path | Actual type after JSON read | Expected type | Projection | Status |
|---|---|---|---|---|---|
| inventory | frontier `result.x_values`; baseline `result.best_x_values` | nested list | exact `|I| x |J|` finite-number matrix | sum in persisted I/J order | fixed and strict |
| opened_warehouses | frontier `result.y_values`; baseline `result.best_y_values` | list | exact length-`|I|` finite-number vector | count values at least 0.5 | fixed and strict |
| objective_t | `result.objective_t` | number or null when no incumbent | finite number for certified frontier | direct | strict |
| robust_minimum_fill_rate | `result.robust_minimum_fill_rate` | number or null | finite number for certified frontier | direct; must equal `1-objective_t` | strict |
| baseline_robust_cost | frontier root `baseline_robust_cost`; baseline `result.upper_bound` | number | finite number | direct | strict |
| cost_budget | frontier root `cost_budget` | number | finite number | direct | strict |
| actual_robust_cost | `result.post_evaluation.actual_robust_cost` | number | finite number for certified frontier | direct | strict |
| actual_price_of_fairness | `result.post_evaluation.actual_price_of_fairness` | number | finite number for certified frontier | direct | strict |
| wminfr | `result.post_evaluation.wminfr` | number | finite number for certified frontier | actual worst-scenario minimum regional fill rate | strict label and value |
| minimum_weighted_mean_fill_rate | `result.post_evaluation.minimum_weighted_mean_fill_rate` | number | finite number for certified frontier | worst-scenario system demand-weighted mean fill rate | strict label and value |
| algorithm_runtime | `result.algorithm_runtime` | number | finite number | direct, excluding post-evaluation | strict |
| master_runtime | `result.master_runtime` | number | finite number | direct | strict |
| separation_runtime | frontier `result.separation_runtime`; baseline `result.subproblem_runtime` | number | finite number | direct with baseline name translation | strict |
| post_evaluation_wall_runtime | `result.post_evaluation_wall_runtime` | number | finite number | direct | strict |
| total_wall_runtime | `result.total_wall_runtime` | number | finite number | direct | strict |
| penalized_runtime_par2 | `result.penalized_runtime_par2` | number | finite number | frozen algorithm-runtime PAR-2 | strict |
| iterations | `result.iterations` | integer | nonnegative integer | direct | strict |
| scenario_block_count | `result.metadata.committed_scenario_count` | integer | nonnegative integer for frontier | direct | strict |
| certified_farkas_cut_count | `result.cuts` | integer | nonnegative integer for frontier | direct | strict |
| scientific_status | run root `scientific_status` | string | frozen status vocabulary | direct; certification additionally requires exact separation and valid post-evaluation | tightened |
| scale / seed / gamma / rho | run root identity | string / integer / integer / string | exact frozen run-plan values | direct | strict |
| baseline_run_key / anchor_sha256 | run root identity | string | exact paired baseline and anchor identity | direct | strict |

`robust_minimum_fill_rate = 1-T` is the certified minimum regional guarantee. `wminfr` and `minimum_weighted_mean_fill_rate` remain post-evaluation metrics and are not relabelled as the certificate.

## Runtime-failure prevention checks

- Aggregation reopens the matching immutable instance archive and verifies its canonical and identity SHA before projecting a run.
- Certified rows require a complete first-stage solution and valid post-evaluation with `errors=[]` and `objective_t_consistent=true`.
- A frontier cannot be classified certified unless both metadata and the final iteration record robust feasibility, final exact separation is present, the master is optimal, and the exact objective bound is acceptable.
- Uncertified rows are not counted as certified; absent incumbent/post-evaluation metrics remain explicitly `NOT_APPLICABLE` rather than fabricated zeros.
- The solver-free 60-task test writes production-shaped list matrices and vectors through JSON, aggregates both scales, then performs a second resume and compares `results.csv` and `summary.csv` byte for byte.
