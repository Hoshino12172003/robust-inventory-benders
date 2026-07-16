# Frozen large-scale evaluation protocol v1

## Scope and research questions

This protocol evaluates whether the five algorithms frozen before the held-out medium-large evaluation retain their certification behavior and relative computational performance on the existing `large` instance definition. It is an out-of-sample scale extension, not a parameter-selection stage.

The evaluation asks:

1. What fraction of runs reaches the global tolerance within 1,800 seconds?
2. How do the methods compare under timeout-aware PAR-2 and time-to-gap metrics?
3. For runs that do not reach tolerance, what certified final gaps and bounds remain?
4. How do master and robust-subproblem time shares change at the larger scale?

No result or ranking is prespecified by this document.

## Frozen design

- Configuration: `experiments/configs/large_scale_evaluation_joint_v1.yaml`
- Experiment: `large_scale_evaluation_joint_v1`
- Output: `experiments/results_large/large_scale_evaluation_joint_v1`
- Size: `large` = 8 warehouses, 8 products, 12 regions
- Held-out seeds: 20–29
- Target uncertainty budget: `Gamma = 2`
- Target-only schedule: `[2]`; Gamma continuation is disabled
- Time limit: 1,800 seconds per run
- Iteration limit: 20,000
- Global tolerance: `1e-4`
- Robust subproblem: `robust_dual_milp`
- One Benders cut at most per iteration
- Cut selection, secondary cut selection/generation, legacy adaptive SP-gap scheduling, and legacy adaptive-gap control are disabled
- Iteration logging is enabled

The seed set is disjoint from tuning seeds 0–2, final medium-large seeds 10–19, and managerial seeds 30–39.

## Frozen algorithms

Each seed is run with the same five variants used by `final_evaluation_joint_v1`:

1. `standard_benders`
2. `static_inexact_benders`
3. `mp_adaptive_rho050`
4. `sp_adaptive_rho050`
5. `proposed_joint_rho025_050`

All selected parameters are inherited from and audited against `experiments/configs/selected_algorithm_parameters.yaml`. The MP/SP gap bounds, fixed baseline gaps, `rho_M`, `rho_S`, cut content, bound updates, termination tolerance, and final-certification trigger are not retuned for `large`.

The grid contains exactly 10 seeds × 5 methods = 50 runs. The serial time-limit upper bound is 50 × 1,800 seconds = 25 hours. This is a theoretical upper bound, not a runtime prediction. The suite does not automatically launch concurrent Gurobi processes.

## Timeout-aware reporting

Every terminal run remains in the dataset. A timed-out or otherwise unsolved run must not be deleted, replaced by a successful-only average, or silently rerun under different parameters.

The primary timeout-aware penalty is PAR-2:

```text
PAR-2 = runtime              if solved_to_tolerance is true
PAR-2 = 2 * time_limit       otherwise
```

Subsequent analysis must report:

- solved rate;
- PAR-2 over all 50 runs;
- runtime conditional on successful solution, clearly labeled;
- final gap for unsolved runs;
- times to 5%, 1%, 0.5%, and 0.1% global gap;
- performance profiles with timeout handling stated explicitly;
- master/subproblem time decomposition.

Raw rows retain status, objective, lower and upper bounds, valid-UB flags, target robust-subproblem status/bound, iteration count, time components, and all time-to-gap fields.

## Comparison families

The confirmatory family mirrors the medium-large evaluation:

- joint adaptive versus tight-tolerance inexact Benders;
- joint adaptive versus static inexact Benders.

The secondary ablation family is:

- joint adaptive versus MP-adaptive;
- joint adaptive versus SP-adaptive.

Any future statistical procedure must be specified in the independent analysis PR before interpreting the new results. This protocol does not authorize post-hoc parameter changes, seed replacement, timeout deletion, or comparison-family changes based on observed outcomes.

## Persistence and recovery

Each run has a stable key, per-run resolved configuration, instance JSON, status file, error file, result JSON, optional iteration log, configuration hash, and Git commit. Global CSV/JSON outputs are written through a temporary file and atomically replaced. `run_manifest.json` records expected, completed, solved, failed, skipped, and remaining counts.

Existing complete successful runs are skipped by default. `--resume` reruns only failed or incomplete records while continuing to skip complete successes. `--overwrite` is required to replace a complete success.

## Commands

POSIX shell:

```bash
python -m src.experiment_suite \
  --config experiments/configs/large_scale_evaluation_joint_v1.yaml \
  --resume
```

PowerShell:

```powershell
python -m src.experiment_suite `
  --config experiments/configs/large_scale_evaluation_joint_v1.yaml `
  --resume
```

Configuration-only dry run:

```bash
python -m src.experiment_suite --config experiments/configs/large_scale_evaluation_joint_v1.yaml --dry-run
```

Static audit:

```bash
python -m src.extended_experiment_audit --output protocol_audit.json
```

