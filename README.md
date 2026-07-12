# Robust Inventory Benders Research Prototype

This repository contains a reproducible Python prototype for a robust inventory allocation problem with budgeted demand uncertainty and Benders decomposition.

The current main research path combines:

- `robust_dual_milp`: robust dual MILP subproblem for evaluating `Q^R(x; Gamma)`.
- Adaptive master-problem MIP gap control.
- Gamma continuation from easier uncertainty budgets to the target budget.
- Violation-based Benders cut selection.

The project does not include RL/PPO training in the current implementation.

## Quick Start

```powershell
python -m src.cli generate --config configs/default.yaml
python -m src.cli solve --method adaptive_gap_gamma_benders --instance data/processed/instance.json
python -m src.cli experiment --config configs/experiment.yaml
```

## Methods

- `monolithic`: full-scenario robust monolithic benchmark for small exact validation.
- `standard_benders`: fixed target `Gamma`, exact or near-exact master solve, no cut selection.
- `inexact_benders`: fixed target `Gamma`, fixed relaxed master `MIPGap`.
- `adaptive_gap_gamma_benders`: adaptive master gap policy with optional Gamma continuation and cut selection.

The experiment suite exposes paper-facing names such as `standard_benders`, `static_inexact_benders`, `adaptive_gamma_benders`, `adaptive_gap_benders`, `adaptive_cut_benders`, `proposed_adaptive_benders`, `monolithic_gurobi`, and `scenario_benders_full`.

## Scenario Modes

1. `exact_scenarios: true`

This mode fully enumerates the budgeted uncertainty set `U(Gamma)`. It is the default setting for paper experiments and exact benchmarks. If the full scenario count exceeds `max_scenarios`, the program raises an error instead of silently switching to an approximation.

```yaml
robust:
  exact_scenarios: true
```

2. `exact_scenarios: false`

This mode allows `candidate_budget_scenarios` when full enumeration exceeds `max_scenarios`. It is a heuristic / approximate mode for exploratory large-scale runs. Candidate mode must not be reported as an exact robust optimum.

```yaml
robust:
  exact_scenarios: false
```

3. `monolithic`

The monolithic method is intended for small-scale exact validation. It uses full scenario enumeration and should be treated as an exact benchmark, not as a heuristic candidate-scenario solver.

## Subproblem Modes

The default paper-experiment setting is:

```yaml
algorithm:
  subproblem_mode: robust_dual_milp
```

- `robust_dual_milp`: solves the robust dual MILP subproblem derived from the paper model. This directly evaluates `Q^R(x; Gamma)` and is the recommended default for main experiments.
- `scenario_enumeration`: evaluates recourse over full or candidate demand scenarios. This mode remains available for small-scale exact benchmarks, validation against `robust_dual_milp`, and heuristic exploratory experiments.

## Cut Selection Strategy

The project supports Benders cut selection based on the violation of the candidate cut at the current master solution:

```text
v_k = cut_rhs(x^k) - theta^k
```

Absolute mode preserves the original `v_k >= delta_cut` rule. Relative mode uses the scale-independent value

```text
normalized_violation = max(0, cut_rhs(x^k) - theta^k)
                       / max(1, abs(theta^k), abs(cut_rhs(x^k)))
```

and adds a cut when it reaches `relative_cut_threshold`. Relative mode force-adds positively violated cuts near the final gap and after a configurable stall, so aggressive screening cannot permanently stop convergence.

```yaml
algorithm:
  cut_selection_enabled: true
  cut_selection_mode: relative
  relative_cut_threshold: 1.0e-4
  cut_violation_tol: 1.0e-8
  final_exact_gap: 1.0e-2
  cut_stall_patience: 5
```

For `robust_dual_milp`, cuts are generated only from the incumbent feasible dual solution through `cut.constant` and `cut.x_coefficients`. Target-Gamma UB evaluation always uses the finite maximization `objective_bound`, even when Gurobi reports `optimal` after satisfying a positive MIP gap. The bound is never used for cut generation.

The robust subproblem can optionally use an adaptive MIP gap selected from the previous valid global Benders gap. It starts with the coarsest configured gap when no finite UB exists and uses the tightest gap in the final phase. A nonoptimal feasible incumbent may define a valid cut; its maximization `objective_bound` may only define a conservative UB.

`max_cuts_per_iteration` defaults to `1`. Values above one request up to K distinct high-value robust demand patterns by excluding earlier binary patterns with no-good constraints. Additional solves may generate valid cuts but never replace the unconstrained solve used for UB evaluation.

## Experiment Suite

The formal experiment suite is implemented in `src/experiment_suite.py` and can be run directly:

```powershell
python -m src.experiment_suite --config experiments/configs/small_correctness.yaml
python -m src.experiment_suite --config experiments/configs/baseline_comparison.yaml
python -m src.experiment_suite --config experiments/configs/ablation_study.yaml
```

Convergence diagnostics are staged separately:

```powershell
python -m src.experiment_suite --config experiments/configs/diagnostic_medium.yaml
python -m src.experiment_suite --config experiments/configs/screen_relative_cut.yaml
python -m src.experiment_suite --config experiments/configs/screen_subproblem_gap.yaml
python -m src.experiment_suite --config experiments/configs/screen_multicut.yaml
```

Set `save_iteration_log: true` to write one CSV per run under `<output_dir>/iteration_logs/`. These logs record requested and achieved master/subproblem gaps, LB/UB trajectories, cut decisions, safety overrides, and timing. Result and summary files include time-to-gap metrics for 5%, 1%, 0.5%, and 0.1%, using blank values when a threshold is not reached.

Diagnostic tuning uses seeds `0, 1, 2`. Final evaluation reserves seeds `10` through `19`; they must not be used to select parameters. Fix the chosen settings in `experiments/configs/selected_algorithm_parameters.yaml` before running `final_evaluation_template.yaml`. The experiment suite validates and overlays every selected field, then writes the fully applied configuration to `<output_dir>/resolved_config.yaml`; missing selected values are an error rather than a fallback to defaults.

It is also available through the main CLI:

```powershell
python -m src.cli experiment-suite --config experiments/configs/baseline_comparison.yaml
```

### Formal Configurations

- `experiments/configs/small_correctness.yaml`: small exact validation with `monolithic_gurobi`, `scenario_benders_full`, `standard_benders`, and `proposed_adaptive_benders`.
- `experiments/configs/baseline_comparison.yaml`: small + medium comparison of strong baselines and the proposed method.
- `experiments/configs/ablation_study.yaml`: medium-focused module ablation over adaptive gap, Gamma continuation, and cut selection.
- `experiments/configs/sensitivity_gamma.yaml`: medium sensitivity analysis over `Gamma`.
- `experiments/configs/sensitivity_service.yaml`: medium sensitivity analysis over service levels.
- `experiments/configs/scalability.yaml`: small + medium + large scalability comparison.

### Output Files

Each suite run writes outputs under `experiments/results/<experiment_name>/`:

- `instances/`: generated JSON instances, one per seed and size.
- `results.csv`: one row per method or variant run.
- `summary.csv`: aggregate statistics by experiment, size, method, and variant.
- `correctness_summary.csv`: only for `small_correctness`, comparing exact and Benders objectives.

`results.csv` records solver status, objective, bounds, final gap, runtime, master/subproblem time, iterations, cut counts, scenario metadata, robust dual MILP metadata, and instance path. Fields that are not applicable are left blank.

`summary.csv` reports success rate, mean objective, mean runtime, mean final gap, mean iterations, mean cut counts, mean master/subproblem time, valid-UB rate, and speedup versus `standard_benders` when available.

`completed_rate` means a run ended with a usable incumbent / result: `optimal`, `iteration_limit`, or `time_limit` with an objective value. `solved_rate` means the run reached `optimal` status or the final Benders gap is within the configured tolerance. The legacy `success_rate` field is kept for compatibility and follows the stricter solved definition. A `time_limit` or `iteration_limit` row should not be interpreted as a strict solve unless its final gap also satisfies the tolerance.

### Experiment Scope

- `monolithic_gurobi` is only intended for small exact validation.
- `scenario_benders_full` uses `exact_scenarios: true`; if full enumeration exceeds `max_scenarios`, the run is recorded as skipped instead of falling back to candidate scenarios.
- Candidate scenarios are heuristic and must not be used as an exact robust baseline.
- Tests use very small instances only to verify the pipeline quickly. Formal configurations, not test fixtures, should be used for paper experiments and include small, medium, medium-large, and large settings.

## Optional Plotting

Basic plot generation is available:

```powershell
python experiments/scripts/plot_results.py --results experiments/results/baseline_comparison/results.csv --summary experiments/results/baseline_comparison/summary.csv --output-dir experiments/results/baseline_comparison/plots
```

The plotting script uses `matplotlib` when available and exits gracefully if it is not installed.

## Structure

- `src/instance.py`: synthetic inventory instance generation and JSON I/O.
- `src/scenarios.py`: budgeted demand scenario enumeration and candidate scenario generation.
- `src/subproblem.py`: recourse LP for a fixed demand scenario.
- `src/robust_dual_subproblem.py`: robust dual MILP subproblem.
- `src/monolithic.py`: full-scenario robust monolithic benchmark.
- `src/benders.py`: standard, inexact, and adaptive Benders loop.
- `src/policies.py`: master `MIPGap` policy interface.
- `src/experiment.py`: legacy lightweight experiment runner.
- `src/experiment_suite.py`: reproducible paper experiment suite.
