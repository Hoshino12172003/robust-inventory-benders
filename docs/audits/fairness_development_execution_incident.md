# Fairness development execution incident

## Identity and scope

- Original execution commit: `7bc8e81f91f4a4c7baf2c080af63a09ada1178d6`.
- Frozen config: `experiments/configs/regional_fairness_development_medium_large.yaml`.
- Frozen config byte SHA256: `E52FAA5EB93E43C6C6B611E47239DD98378DBF039BD9ECA3C49FF7B2D5AE59E1`.
- Read-only evidence directory: `E:\rf\experiments\results_regional_fairness_model\development_medium_large`.
- Observed interruption time: `2026-07-22T04:26:58Z`, from the last `status.json` update.
- Exception: `RuntimeError: Separation incumbent did not define a valid normalized Farkas ray.`
- Certificate implementation hotfix commit: `723b4eba4596d3b6299254a04f6f2000dcd6cbdf`.

The old directory was read but was not modified, resumed, overwritten, moved, or
deleted during this investigation. Large development was not started.

## Execution state at failure

The active task was:

`regional_fairness_development_medium_large__rho__0_01__medium_large__seed_126__robust_regional_fairness`

It has `status.json` with state `running`, but no `run.json`. The last completed
fairness task was seed 126 at `rho=0`; it ended with algorithm status `optimal`
but did not pass the complete scientific-success predicate. The most recent
successful fairness task was seed 125 at `rho=0.10`. The seed 126 frozen V3
baseline completed successfully immediately before the seed 126 frontier.

The authoritative per-run files contain 39 unique run directories, 38 complete
`run.json` files, 39 `status.json` files, one residual `running` status, and no
duplicate run key. Of the 38 complete records, 18 are successful and 20 are
unsuccessful. The atomic fairness manifest reports 38 complete, 18 solved, and
22 pending. The generic manifest is stale at 36 complete, 17 solved, 19 failed,
and 24 remaining because the process terminated before its next aggregate
refresh. This discrepancy is itself evidence of the abrupt exception; it is
not permission to reinterpret or repair the old directory.

No `error.txt`, audit log, solver log, or exception payload containing the
separation incumbent was written. Consequently the following requested values
cannot be reconstructed from the immutable evidence and are recorded as
`not_recorded`: incumbent objective, objective bound, MIP gap, deviation vector,
raw multipliers, normalization residual, minimum multiplier, maximum dual-cone
residual, scenario budget residual, and cut violation. Re-running seed 126 to
manufacture those fields is prohibited.

## Root cause and mathematical risk

The immediate trigger was an incumbent whose extracted multipliers failed the
normalization/nonnegativity/dual-cone validation. Because the incumbent payload
was not persisted, the evidence cannot distinguish solver numerical feasibility,
variable extraction, or McCormick-product inconsistency as the numerical trigger.
There is no evidence supporting a post-hoc claim that a particular residual or
sign was responsible.

The actionable implementation defect is independent of that missing detail:
the budgeted-uncertainty separation MILP supplied both the candidate scenario
and the proposed Farkas ray. A ray that failed validation caused an exception,
and there was no independent fixed-scenario recourse feasibility check. Directly
relaxing the feasibility tolerance would hide rather than resolve this defect.

The hotfix therefore makes the separation MILP a candidate-scenario generator
only. For every positive candidate it now:

1. solves the original fixed-scenario recourse feasibility LP using the current
   `y`, `x`, `T`, and cost budget;
2. discards the candidate without a cut if that LP is feasible, then excludes
   only that independently proven-feasible pattern and continues safely;
3. if the primal LP is infeasible, solves a separate continuous normalized
   Farkas LP with no uncertainty binaries or McCormick variables;
4. adds a cut only when the independent ray is optimal, finite, normalized,
   nonnegative, dual-cone feasible, and violated at the current master point;
5. returns an explicit uncertified status for every unresolved, interrupted,
   numeric, suboptimal, infeasible-without-certificate, or time-limited path.

The full separation objective bound remains the only proof that no violating
scenario remains. An incumbent, including a time-limit incumbent, can only
propose a scenario. A fixed-scenario feasible result never produces an
infeasibility cut. An invalid or unavailable ray never produces a cut.

## Scientific disposition and restart rule

All partial output under `E:\rf` from the original execution is invalid for
development selection, statistical summaries, or pooling. It must remain as
an immutable incident artifact.

The hotfix changes certificate provenance and observability only. It does not
change the fairness model, `rho` grid, seeds, cost anchor, time limits,
development thresholds, uncertainty set, frozen V3 parameters, or
`src/benders.py`. The two development YAML files and the mathematical protocol
retain their frozen byte hashes.

After this Draft PR is reviewed and merged, a restart is authorized only from a
fresh short worktree such as `E:\rf2` and nonexistent output directories. The
same seeds 120--129 and the same frozen configs must be used. The new manifest
must record `execution_restart_after_correctness_hotfix`. The old and new
outputs must never be merged, and `--resume` must not be pointed at `E:\rf`.
