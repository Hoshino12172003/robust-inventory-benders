# Medium-large Fairness Final Holdout Protocol

## Authorization and finality

This protocol is the final computational evaluation of the already frozen fairness
model. It does not authorize algorithm development, parameter revision, candidate
selection, Large remediation, S2, full-grid, or any additional attempt. The sole
formal configuration is
`experiments/configs/fairness_medium_large_final_holdout.yaml`. Its authorization
is valid only after the Draft PR that introduces this protocol is independently
reviewed and merged.

The final holdout implementation must not modify the fairness formulation, recourse,
uncertainty set, normalized Farkas cone, fixed-scenario certification, complete
separation objective-bound requirement, success definition, or the frozen V3
candidate file. Previous instances, baselines, anchors, runs, checkpoints, manifests,
results, and summaries are never reused.

## Frozen prerequisite decisions

Large fairness remediation Attempt 5 is frozen as an execution-complete scientific
failure. Its baseline is certified, but its sole frontier task is
`time_limit_uncertified`; `T=1`, its incumbent, and `complete` are not an optimal
fairness result. The final decision is `stop_final_large_remediation`; L1, M1, and
all additional Large runs remain unauthorized.

Seeds 170--179 may be used only after a fail-closed, read-only audit of controlled
Git configuration/document/manifest identities, known formal result directories,
frozen result archives, and existing instance identities. Text that merely reserves
these values as holdout seeds is not evidence of access. Any actual instance, run,
baseline, anchor, checkpoint, or solver result for one of these seeds invalidates the
entire seed set; no replacement seed may be chosen. Seeds 130--159 remain prohibited.

## Frozen matrix

- Scale: `medium_large`.
- Holdout seeds: 170, 171, 172, 173, 174, 175, 176, 177, 178, 179.
- Rho: 0.00, 0.01, 0.025, 0.05, 0.10.
- Candidates, in frozen order: `single_cut`,
  `persistent_certified_cache_batch5`.
- One certified cost baseline per seed: 10 baselines.
- Ten frontier tasks per seed: 100 frontier tasks.
- Total: 110 unique tasks.

Every seed's ten frontier tasks use exactly the same generated instance,
`baseline_run_key`, certified `C_anchor`, `anchor_value_hex`, and `anchor_sha256`.
Canonical scientific run keys are compact sorted UTF-8 JSON. Physical directories
are `r_` plus the first 24 lowercase hexadecimal characters of SHA256 of that run
key. Forward and reverse mappings are identity-locked; collisions fail closed.

`single_cut` has no persistent separator, cache, or solution pool and adds at most
one certified cut per iteration. `persistent_certified_cache_batch5` uses the
existing persistent separator, pattern-only cache, and solution-pool proposal
mechanism; every proposed cut is independently recertified at the current point and
at most five certified cuts are added. No third candidate, adaptive remediation
candidate, old ray, old cut, old violation, or old certification is permitted.

## Solver and certification identity

- Gurobi `Threads=1`, `Seed=0`, `FeasibilityTol=1e-7`.
- Baseline, fairness, and general algorithm limits: 1800 seconds.
- Gamma: 2; exact uncertainty scenarios.
- Post-evaluation: all 1,831 scenarios, 30 seconds per scenario, checkpoint chunks
  of 25, feasibility tolerance `1e-7`.
- PAR-2 multiplier: 2; basis: fairness `algorithm_runtime`; every uncertified task
  receives 3,600 seconds.

`certified_robust_optimal` requires all of the following: the fairness algorithm's
complete robust certification, a valid master bound and frozen gap, a legitimate
complete-separation objective bound proving no violating scenario, and a valid
1,831-scenario post-evaluation with no acceptance error. Time limits,
`master_optimal_but_robust_uncertified`, `separation_stalled_duplicate`, invalid
post-evaluation, identity mismatch, checkpoint failure, and implementation error do
not count as solved. A scientific time limit is recorded and the preregistered batch
continues. Implementation errors, identity drift, corruption, and invalid
post-evaluation stop the batch and fail closed.

The production baseline payload is the frozen schema: `best_y_values` is a finite
length-|I| vector and `best_x_values` is a finite |I|-by-|J| matrix in formal
instance order. Missing, extra, mapping, string, Boolean, nonfinite, or dimensionally
incorrect values fail closed. Instance, manifest, baseline, and anchor identities
remain bound.

## Outcomes and comparisons

Primary outputs include certification count/rate, all-run PAR-2, algorithm and total
wall runtime, post-evaluation wall runtime, separation and master runtime,
iterations, certified cuts, final gap, and scientific status.

The two methods are paired only at identical seed and rho. Each rho therefore has
exactly ten seed pairs. PAR-2 uses all 50 pairs. Runtime acceleration is reported as
all-run PAR-2 and separately for common-certified pairs. Objective `T`, robust cost,
and fairness outcomes are compared only when both methods are certified; uncertified
incumbents, bounds, and `T` values are never treated as optimal outcomes.

For each certified seed/rho result, report certified `C_anchor`,
`B_rho=(1+rho)C_anchor`, certified `T`, robust cost, worst regional shortage rate,
fairness improvement relative to the same method and seed at rho zero, and the cost
budget increment relative to baseline.

The frozen fairness-improvement quantity is the absolute reduction
`T(seed, rho=0, candidate) - T(seed, rho, candidate)`. It is reported only when both
the rho-zero reference and the target run are certified. Otherwise it is explicitly
`NOT_APPLICABLE`; no incumbent substitution or cross-seed reference is allowed.

### Statistical unit and preregistered inference

The independent unit is **seed**, never a seed-rho task.

- At each rho, the paired comparison contains ten seed pairs.
- Cluster bootstrap resamples the ten seeds. A sampled seed contributes all five
  rho values and both methods as one indivisible cluster.
- The overall comparison first takes the arithmetic mean of the five paired PAR-2
  differences within each seed and then bootstraps those ten seed-level aggregates.
- Use 10,000 bootstrap replicates, deterministic bootstrap seed 20260728, and a
  two-sided 95% percentile interval.
- A two-sided Wilcoxon signed-rank result is reported for a rho only when all ten
  pairs are present and at least six paired differences are nonzero. If rho-wise
  tests are reported, their five p-values use Holm correction.
- Failures have PAR-2=3,600 seconds. All preregistered results are disclosed
  regardless of significance.

Treating the 50 seed-rho tasks as 50 independent observations is forbidden.

## Identity, recovery, and outputs

The experiment has an independent manifest schema and execution attempt 1.
`previous_attempt_results_reused=false`. The output directory must not exist before
identity-locked initialization. `--overwrite` is unsupported; `--resume` is required
for both first initialization and recovery. Existing output without an exact manifest
identity fails closed.

Baseline and algorithm results are atomically checkpointed. Post-evaluation uses
identity-locked atomic scenario chunks and an index. Aggregation is rebuilt solely
from atomic run records. Resume must not duplicate a baseline, scientific run,
post-evaluation scenario/chunk, or output row. Corrupt records/checkpoints and any
Git/config/protocol/candidate/solver/instance/baseline/anchor drift fail closed.

Required outputs are `manifest.json`, `run_manifest.json`, `resolved_config.yaml`,
atomic run/status records, baseline/algorithm checkpoints, post-evaluation index and
chunks, `results.csv`, `summary.csv`, `paired_comparison.csv`,
`cost_fairness_frontier.csv`, `paired_statistics.json`, and `audit_log.json`.
Every path including atomic `.tmp` names must remain at most 220 characters on the
actual Windows worktree.

## Gate

A solver-free dry-run must report 10 baselines, 100 frontier tasks, 110 unique run
keys, zero duplicates, 1,831 scenarios, no generated instances, no solver call, an
absent output directory, and a passing Windows path check. Before this PR is merged,
formal execution remains unauthorized even though the configuration contains the
future activation fields. No Large, L0/L1/M1, original S2, full-grid, Attempt 4, or
holdout execution is authorized by protocol construction or review alone.
