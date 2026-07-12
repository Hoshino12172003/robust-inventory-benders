# Convergence Diagnostics Handoff

## Scope

This PR diagnoses the medium/large convergence bottleneck and adds optional diagnostics, relative cut selection, adaptive robust-subproblem gaps, multi-cut generation, and time-to-gap metrics. It does not change the inventory model, the budgeted uncertainty set, or the robust dual MILP derivation. Existing result directories are untouched.

## Verified Pre-Change Parameter Flow

- The master requested `MIPGap` is selected every iteration by `ExactGapPolicy`, `FixedGapPolicy`, or `RLInspiredGapPolicy`, then assigned to `model.Params.MIPGap` before optimization. The requested value therefore changes for adaptive runs; the previous log field called `realized_master_gap` was the solver's achieved `model.MIPGap`, not the requested value.
- Standard Benders uses `final_mip_gap`; static inexact Benders uses `initial_mip_gap`; the adaptive method selects within those endpoints from the previous Benders-gap trajectory.
- Active Gamma is selected by iteration from `gamma_schedule` only for `adaptive_gap_gamma_benders`. Standard and inexact methods force `[gamma_target]`.
- Every iteration evaluates `Gamma_target` for the final-model UB when active Gamma is smaller. When active Gamma equals the target, the same unconstrained robust solve serves both roles.
- A robust MILP incumbent supplies dual variables and cut coefficients. For target-Gamma UB evaluation, finite `ObjBound` is always used, including when Gurobi reports `optimal` after meeting a positive MIP gap. It is not used in a cut.
- Before this PR, robust subproblems always requested `final_mip_gap`, even early in the Benders run, while tuned experiments showed subproblem time dominating master time.
- The tuned absolute thresholds (`1e6` and `1e12`) depended on objective scale. The target-progress override often force-added a positive cut, so meaningful skipping was limited and difficult to interpret.
- Proposed, standard, and static-inexact variants use the same robust dual formulation and generally produce the same type of single incumbent cut. Their differences were primarily master MIPGap, Gamma continuation, and the absolute-threshold override; this explains similar cut trajectories in preliminary results.

## Files Changed

- `src/benders.py`: diagnostics, relative cut policy, adaptive subproblem-gap schedule, optional multi-cut loop, validity-preserving UB/cut handling, and metadata.
- `src/robust_dual_subproblem.py`: explicit no-incumbent results and optional no-good exclusions for distinct demand patterns.
- `src/experiment_suite.py`: iteration CSV output, time-to-gap result fields, and aggregate diagnostic metrics.
- `tests/test_convergence_diagnostics.py`: focused policy, no-good, no-incumbent, multi-cut, and safety tests.
- `tests/test_experiment_suite.py`: iteration-log and tuning/evaluation-seed checks.
- `experiments/configs/diagnostic_medium.yaml`: trajectory diagnosis.
- `experiments/configs/screen_relative_cut.yaml`: relative-threshold screen on tuning seeds.
- `experiments/configs/screen_subproblem_gap.yaml`: exact versus adaptive subproblem-gap screen.
- `experiments/configs/screen_multicut.yaml`: K=1,2,3 screen.
- `experiments/configs/final_evaluation_template.yaml`: reserved evaluation seeds 10-19.
- `experiments/configs/selected_algorithm_parameters.yaml`: intentionally pending until screens are completed.
- `experiments/configs/smoke_convergence_features.yaml`: very-small local smoke run for all new features.
- `README.md`: feature definitions and staged local commands.

## Mathematical Validity Decisions

- Master LB updates continue to use only `model.ObjBound`, never an inexact incumbent objective.
- Final-model UB evaluation always uses the unconstrained `Gamma_target` robust problem.
- A cut is available only when `has_incumbent=True`; no-incumbent results contain no fake objective or coefficients.
- The no-incumbent failure policy is to skip cut generation, preserve any earlier valid bounds, record the event, and continue until the configured time/iteration limit. No automatic retry is claimed in this PR.
- A nonoptimal incumbent remains a feasible dual solution and therefore defines a valid lower supporting cut.
- `objective_bound` is used only as a conservative maximization UB and never as a cut.
- Termination requires a valid finite target-Gamma objective bound in the current iteration; an incumbent objective cannot create a false zero gap.
- Additional no-good-constrained solves only provide cuts. Their bounds cannot update the unconstrained robust UB.
- Every additional solve contributes to runtime accounting even when it has no incumbent or repeats a pattern.
- Relative screening has final-phase and stall-patience overrides for positively violated cuts.

## Feature Flags And Defaults

- `save_iteration_log: false`
- `cut_selection_mode: absolute` for backward compatibility
- `relative_cut_threshold: 1.0e-4`
- `final_exact_gap: 1.0e-2`
- `cut_stall_patience: 5`
- `adaptive_subproblem_gap_enabled: false`
- `max_cuts_per_iteration: 1`

## Validation

Completed in the Codex sandbox:

- Python compilation passed for the three changed source modules and changed tests.
- All experiment YAML files parsed successfully.
- Hidden Unicode scan: `No hidden Unicode characters found.`
- `git diff --check` passed.
- Five pure convergence/metric tests passed, including scale invariance, relative safety decisions, adaptive subproblem-gap tightening, backward-compatible defaults, and time-to-gap aggregation.
- Three pure scenario-mode tests passed.
- Formal diagnostic configuration parsing passed and the final-evaluation lock was verified.

The named-user Gurobi license rejected the Codex sandbox account because it is licensed to `hu_jiaxin` while the sandbox OS user is `codexsandboxoffline`. The complete validation was therefore run in PyCharm under the licensed local user:

- `python -m pytest tests -q`: `35 passed in 2.09s`.
- `python -m src.experiment_suite --config experiments/configs/smoke_convergence_features.yaml`: completed successfully.
- The original 3-iteration zero-gap smoke result was superseded because it used the incumbent objective instead of the positive-gap objective bound for UB evaluation.
- Corrected smoke status: `optimal` in 4 iterations under the configured tolerance `0.001`.
- Corrected smoke UB: `6467.465885611927`; LB: `6462.072305841466`; final gap: `0.0008339556583452202`.
- The target robust cost equals the finite objective bound `5533.895885611927`; it is no longer the smaller incumbent used by the superseded run.
- Corrected smoke cut diagnostics: 5 cuts added, 3 skipped, K=2 generated 2 patterns per iteration on average, and 3 duplicate cuts were rejected.
- Corrected smoke subproblem diagnostics: no nonoptimal subproblems and no missing incumbents.
- `tests/test_convergence_diagnostics.py`: `14 passed`.
- `tests/test_experiment_suite.py`: `9 passed`.
- Full suite: `42 passed in 1.90s`.
- Experiment CSV and instance writers now emit LF directly; the final hidden Unicode scan is clean.

Run the smoke test locally with:

```powershell
python -m src.experiment_suite --config experiments/configs/smoke_convergence_features.yaml
```

The generated smoke evidence is stored under `experiments/results_diagnostics/smoke_convergence_features/`.

## Local Screens Still Required

```powershell
python -m src.experiment_suite --config experiments/configs/diagnostic_medium.yaml
python -m src.experiment_suite --config experiments/configs/screen_relative_cut.yaml
python -m src.experiment_suite --config experiments/configs/screen_subproblem_gap.yaml
python -m src.experiment_suite --config experiments/configs/screen_multicut.yaml
```

Do not run `final_evaluation_template.yaml` until `selected_algorithm_parameters.yaml` has been fixed from tuning-seed evidence.

## Limitations

- The parameter screens have not been run in Codex and no performance claim is made.
- Additional patterns are up to K distinct high-value patterns; they are not claimed to be exact top-K when solves are inexact.
- The selected-parameter file remains explicitly pending to prevent accidental use of evaluation seeds for tuning.
- Final evaluation now requires behavior-enabling selections for cut mode, adaptive robust-subproblem gaps, multi-cut count, and the optional per-iteration subproblem time budget.
- Selected proposed-method parameters are isolated from `standard_benders` and `static_inexact_benders`; both baselines force target-only Gamma, single-cut generation, disabled cut selection, and a fixed tight robust-subproblem gap.
