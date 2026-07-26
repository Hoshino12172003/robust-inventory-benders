# Fairness scalability development protocol

Status: preregistered protocol only. Formal execution is not authorized by this PR.

## Attempt 1 quarantine and Attempt 2 identity

The first Medium-large S1 execution used commit
`22ce2d63a4ad8cea021bf2b6cbe60273c0c2919c`. It accessed seed 160 and stopped
after five atomic run records when the first post-evaluation chunk for
`persistent_certified_cache_batch5` could not create the 264-character Windows
temporary path. This is a physical path-length pipeline defect, not an
algorithm or mathematical result. Attempt 1 is `execution_incomplete`, is not
scientifically usable for candidate selection, and none of its instance,
baseline, anchor, run, checkpoint, result, summary, or manifest artifacts may
be resumed, imported, or reused.

The next execution identity is `execution_attempt: 2`. Both frozen configs use
new physical output directories and record the structured Attempt 1 history.
Attempt 2 must start with a nonexistent output directory under the merged
hotfix commit. Resume is permitted only for the same schema-2 Attempt 2
identity. The scientific run plan, candidates, seeds, rho values, model,
certification, time limits, and selection rules are unchanged.

## Scope and frozen scientific model

This stage investigates engineering scalability of the already frozen robust
regional-service fairness model. It does not change the objective, recourse,
cost budget, uncertainty set, cost anchor, success definition, or scientific
selection thresholds. It does not make claims about demographic or vulnerable
groups. Attempt 3 is retained only as a frozen runtime-pipeline incident and an
engineering motivation; its 57/60 and 28/60 counts and all seed-level outputs
are prohibited from parameter, rho, threshold, or algorithm selection.

The model remains the epsilon-constraint max-min formulation: minimize the
worst applicable regional shortage ratio `T`, with one scenario-specific
recourse policy satisfying both the original recourse constraints and the
scenario cost cap `(1 + rho) C_anchor`. Gamma remains 2, the exact extreme-point
set remains unchanged, the fairness algorithm time limit remains 1,800 seconds,
and only `certified_robust_optimal` is a solved run. PAR-2 remains algorithm
runtime for solved runs and twice 1,800 seconds otherwise. The frozen 80%
certified-solved gate is unchanged.

## Candidates and their only permitted differences

All candidates use identical masters, joint V1 precision control, Farkas cone,
fixed-scenario primal/Farkas certification, tolerances, bounds, and stopping
conditions.

1. `single_cut`: rebuild the complete separation MILP for each master point and
   add at most one independently certified cut.
2. `persistent_separation`: retain the same separation MILP across master
   iterations and update only current-point objective coefficients; add at most
   one independently certified cut.
3. `persistent_certified_cache`: additionally retain only historical scenario
   patterns as candidate hints. Every cache hit is resolved by a fresh
   fixed-scenario primal/Farkas certification at the current `(x,T,B_rho)`.
4. `persistent_certified_cache_batch5`: additionally request at most five
   distinct solution-pool patterns and add at most five cuts, each independently
   certified at the current point.

Solution-pool incumbents and cached patterns are candidate generators only.
Old rays and cuts are never reused. A current fixed-scenario infeasibility
certificate is required for every cut. A cache hit never certifies robust
feasibility. Only a valid objective bound from complete separation can certify
that the remaining uncertainty set contains no violation. False-positive
scenario exclusions are exact no-good constraints local to one separation call
and are removed before the next master iteration.

## New seeds and staged execution

Development labels are frozen to 160--169 for both scales. Seeds 130--159 are
sealed and may not be accessed. No formal seed is used by unit tests.

- S0: deterministic hand-built tiny instances and independent extensive-form
  comparisons only.
- S1: seeds 160--162, rho in `{0, 0.01}`, all four candidates. This is 3
  baselines plus 24 frontier tasks per scale.
- S2: seeds 160--169, rho in `{0, 0.01}`, all four candidates. S1 is a subset;
  the cumulative unique plan is 10 baselines plus 80 frontier tasks per scale.
- Full grid: allowed only after S2. A candidate must pass every correctness and
  certification check and obtain at least 16/20 `certified_robust_optimal`
  frontier runs at rho 0 and 0.01. Only the selected candidate may then run the
  three remaining rho values `{0.025, 0.05, 0.10}`, adding 30 tasks per scale.
  The complete staged unique plan is therefore 120 tasks per scale.

Formal execution requires a later pre-run audit and explicit authorization.
The runner is present so that its identity and recovery semantics can be
audited before authorization; this PR itself does not execute it.

## Stable execution and recovery contract

The cumulative machine plan has 27 S1 tasks, 90 S2 tasks, and 120 full-grid
tasks. S2 adds exactly 63 tasks (seven baselines and 56 frontier tasks) to S1;
full-grid adds exactly 30 tasks. A baseline key is unique by scale and seed and is never candidate
specific. Every frontier record locks the same instance, baseline run key,
certified conservative `C_anchor`, `float.hex`, and anchor SHA256 for its seed.
A non-certified baseline fails all frontier tasks for that seed closed.

Each seed/rho/candidate frontier task makes a fresh `solve_fairness_benders`
call. Persistent models, MIP state, cache patterns, pool solutions, no-good
constraints, timers, and iteration state live only within that call. No state,
including patterns or MIP starts, is passed across rho, candidate, or seed.

The scalability manifest schema is version 2. It atomically locks the Git
commit, input and resolved configuration hashes, protocol and frozen V3
candidate hashes, scale, cumulative stage, exact run specs, prior decision
hash, candidate definitions, Gurobi `Threads=1`, `Seed=0`,
`FeasibilityTol=1e-7`, both 1,800-second algorithm limits, post-evaluation
scenario/chunk settings, public status enum, PAR-2 basis, and runtime semantics.
Every algorithm checkpoint locks its run key, candidate, rho, anchor, Git and
configuration identities. Post-evaluation uses the existing atomic deterministic
scenario chunks and resumes only matching committed chunks. Corrupt or drifting
records/checkpoints fail closed. Attempt 2 rejects every schema-1 directory and
never searches outside its own output root. Final CSV files are rebuilt deterministically
from atomic run records, so a crash after `run.json` but before aggregation is
recoverable without resolving the task.

The frozen output schema comprises `scalability_development_manifest.json`,
`run_manifest.json`, `resolved_config.yaml`, `results.csv`, `summary.csv`,
`audit_log.json`, per-task `status.json` and `run.json`, an atomic whole-
algorithm checkpoint, and deterministic post-evaluation checkpoint chunks and
index. An interruption before the whole-algorithm checkpoint is committed
restarts only that frontier algorithm task; an interruption after it never
repeats the algorithm. Committed post-evaluation chunks are never resolved
again. No artifact may be imported from another output directory.

### Windows-portable physical layout

The canonical scientific `run_key` remains unchanged and is written to every
run/status record, both manifests, `results.csv`, and all checkpoint identities.
It is not used as a physical directory name. The physical id is frozen as:

```text
run_directory_id = "r_" + sha256(run_key UTF-8).hexdigest()[0:24]
```

Both manifests store `run_key_to_directory_id` and
`directory_id_to_run_key`. All ids are calculated before output initialization;
any hash collision fails closed. Resume locates artifacts only through the
validated full-key mapping and never infers scientific identity from a
directory name. Baseline records, status, algorithm checkpoints,
post-evaluation chunks, and final aggregation all use this same mapping.

Before creating the output directory, generating an instance, or configuring
Gurobi, the runner expands every run path and atomic temporary path, including
`run.json`, `status.json`, the algorithm checkpoint, the last possible
post-evaluation chunk and `.tmp`, checkpoint index, final post-evaluation file,
manifest, audit log, results, and summary. The frozen portable absolute-path
limit is 220 characters. A longer path fails closed. Dry-run reports the
maximum absolute length, longest path type and portability verdict without
creating any directory, instance, or solver model. The run,
`post_evaluation`, and `checkpoint` directories are explicitly created before
checkpoint writes; directory creation is not used as a substitute for the
short mapping.

`baseline_time_limit` is explicitly wired to the V3 baseline call,
`fairness_time_limit` to every fairness candidate call, and generic
`time_limit` remains the common frozen default. All three are 1,800 seconds,
so explicit wiring changes no frozen numerical limit. The manifest locks the
post-evaluation chunk size and solver parameters as well as all runtime fields.

Only `certified_robust_optimal` counts as solved. Public statuses distinguish
master-optimal but uncertified, time limit, infeasible, invalid post-evaluation,
implementation error, and interruption (with additional iteration-limit,
numeric, and unknown uncertified fail-closed states). PAR-2 remains based only on algorithm
runtime; post-evaluation solver/wall time, aggregation, checkpoint I/O, and total
wall time are separately reported.

S2 requires a frozen read-only S1 decision with its SHA256. Full-grid requires
a frozen S2 decision, at least 16/20 certified runs at each scale, correctness
approval, the exact preregistered ordering, and one separately hashed selected
candidate file. A missing, drifting, non-unique, or ineligible selection fails
closed. S1/S2 results cannot be used to alter the batch size or full-grid rho
set.

Stage advancement uses the same physical output and requires every prior-stage
run to have an atomic complete record. Already completed S1 keys are verified
and skipped during S2; they are never solved again. Resume requires exact
schema, Git, byte and resolved configuration hashes, protocol hash, frozen V3
candidate hash, scale, solver/post-evaluation settings, and stage-decision
identity. A pre-existing directory without that identity is rejected, even if
empty. There is no overwrite path.

## Frozen selection rule

Candidates are ordered lexicographically by:

1. mathematical and certification correctness;
2. certified solved count, higher first;
3. PAR-2, lower first;
4. separation runtime, lower first;
5. total wall runtime, lower first.

A correctness failure makes a candidate ineligible regardless of speed. S2
below 80% certified solved prevents the full rho grid. No observed result may
change these rules. Negative or inconclusive findings are retained.

## Required evidence

Iteration and result evidence includes `separation_model_build_runtime`,
`separation_optimize_runtime`, `cache_candidate_count`, `cache_hit_count`,
`certified_cached_cut_count`, `pool_candidate_count`,
`certified_batch_cut_count`, `duplicate_pattern_count`, `cuts_per_iteration`,
`total_iterations`, `algorithm_runtime`, and `total_wall_runtime`. Algorithm
runtime and end-to-end wall time remain distinct.

## Dry-run

Dry-run reads configuration only, performs no instance generation or solver
call, and does not create the formal output directory:

```powershell
python -m src.fairness_scalability_suite `
  --config experiments/configs/fairness_scalability_development_medium_large.yaml `
  --dry-run

python -m src.fairness_scalability_suite `
  --config experiments/configs/fairness_scalability_development_large.yaml `
  --dry-run
```

After a separate authorization, the same first-run/resume command is used. The
Attempt 2 S1 output directory must either not exist (atomic initialization) or
contain a matching schema-2 identity and exact bidirectional run-directory
mapping (resume):

```powershell
python -m src.fairness_scalability_suite `
  --config experiments/configs/fairness_scalability_development_medium_large.yaml `
  --stage s1 `
  --resume

python -m src.fairness_scalability_suite `
  --config experiments/configs/fairness_scalability_development_large.yaml `
  --stage s1 `
  --resume
```

S2 and full-grid use `--decision <frozen-decision.yaml>`. There is deliberately
no `--overwrite` option.
