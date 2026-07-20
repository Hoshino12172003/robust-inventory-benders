# Cut-Strengthened Joint V3 Final Protocol

## Authorization and scope

This protocol is created only after the validation decision was frozen as:

```yaml
decision: validation_pass
selected_candidate: joint_v1_core_point_strengthened
baseline: proposed_joint_rho025_050
next_authorized_stage: final_protocol_only
```

The final stage compares the frozen Joint V1 baseline with the frozen
core-point-only candidate on previously sealed instances. It does not select a
new method and is not a tuning stage. This PR creates only protocol text,
configurations, static auditing, and tests. It does not generate instances or
run final experiments.

The candidate remains a Magnanti-Wong-type core-point strengthened method; it
is not claimed to use a strictly Pareto-optimal cut. No fairness constraint,
managerial-sensitivity dimension, optimization-model change, or new
uncertainty set is introduced.

## Frozen configurations

Both configurations compare exactly:

- `proposed_joint_rho025_050` (Joint V1);
- `joint_v1_core_point_strengthened` (core-only candidate).

The medium-large configuration uses seeds 90--99, a 600-second time limit, and
10,000 maximum iterations. Its 10 seeds times 2 methods give 20 runs, written
only to `experiments/results_cut_v3/final_medium_large`.

The large configuration uses seeds 100--109, an 1,800-second time limit, and
20,000 maximum iterations. Its 10 seeds times 2 methods give 20 runs, written
only to `experiments/results_cut_v3/final_large`.

The two seed sets are disjoint and exclude development seeds 75--79 and
validation seeds 80--89. They must not be replaced, screened, regenerated, or
supplemented after results are observed.

Relative to the corresponding validation configuration, the only differences
are:

- `experiment_name`;
- `output_dir`;
- `random_seeds`;
- `protocol_phase: final`;
- `formal_inference_allowed: true` and the `final_analysis` metadata frozen
  below.

Instance dimensions, hardware/thread behavior, time limits, maximum
iterations, Gamma, tolerance, PAR-2 and success definitions, MP/SP precision,
core-point parameters, termination verification, and every algorithm switch
remain unchanged. The candidate lock SHA256 remains
`7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6`.

## Result handling

Final is confirmatory reporting, not another selection stage. Every configured
run must be retained and reported, including time limits, failures, and
unfavorable instances. No result may be deleted because it weakens the
candidate.

`solved_to_tolerance` retains the validation-pipeline definition: an objective
must exist, final gap must be finite, and final gap must be at most
`tol=1e-4`. PAR-2 equals algorithm runtime for a solved run and
`2 * time_limit` otherwise. Timeout runs retain their actual LB, UB, final gap,
iterations, and runtime. Failed or missing runs count as unsolved and make the
completeness/correctness audit fail; missing gaps or iterations are not filled
with zero or imputed.

Comparisons are paired by seed within each scale. Development and validation
results must not be pooled with final results to enlarge the sample or compute
the final effect size.

## Metrics

The primary final metrics are:

1. large paired mean PAR-2;
2. large solved rate.

Secondary metrics are medium-large mean PAR-2, mean iterations, paired win
counts, master/original-subproblem/core runtime, core success rate, core extra
runtime share, final gap, and solved rate. Successful-run runtime and unsolved
final gaps may be displayed descriptively but cannot replace the timeout-aware
primary analysis.

## Frozen confirmation rule

The correctness gate from validation remains mandatory: run completeness,
valid and monotone LB/UB, monotone requested MP/SP gaps, correct global gaps and
statuses, valid UB provenance, valid core cuts, fallback behavior, termination
verification isolation, one cut per iteration, and no secondary activity.

`final_confirmed` requires all of the following:

Medium-large:

- core-only solved rate is not below V1;
- core-only mean PAR-2 is no more than 103% of V1.

Large:

- core-only solved rate is not below V1;
- core-only mean PAR-2 is at least 7.5% lower than V1;
- core-only mean iterations is at least 15% lower than V1;
- core-only PAR-2 is no worse than V1 on at least 6/10 paired instances;
- core-only iterations is lower than V1 on at least 6/10 paired instances;
- core-only mean paired rank is better than V1.

If correctness or any necessary threshold fails, the recorded outcome is
`final_not_confirmed`. An unfavorable or uncertain final result must be
reported as observed. It does not authorize seed replacement, parameter
revision, repeated tuning, a new candidate, or rerunning the same data as a
new confirmation sample.

## Auxiliary statistical inference

Formal inference is allowed only in the following pre-registered auxiliary
form:

- resampling unit: paired large seed;
- estimand: arithmetic mean of `core PAR-2 - V1 PAR-2`;
- method: paired nonparametric percentile bootstrap;
- confidence level: 95%;
- resamples: 10,000;
- analysis RNG seed: 20260720;
- missing, failed, and timeout outcomes use the frozen PAR-2 handling before
  resampling.

The analysis RNG seed is not an instance seed and cannot generate or solve an
inventory instance. The confidence interval is auxiliary. It cannot replace,
relax, reinterpret, or override the pre-registered confirmation thresholds.
No p-value is required, and no post-hoc alternative bootstrap specification is
permitted after final results are observed.

## Dry-run and later execution

The following commands are allowed in this protocol PR because `--dry-run`
only expands the run plan and invokes static auditing; it does not create
instances, initialize Gurobi, or create result directories:

```powershell
python -m src.experiment_suite `
  --config experiments/configs/cut_strengthened_joint_v3_final_medium_large.yaml `
  --dry-run

python -m src.experiment_suite `
  --config experiments/configs/cut_strengthened_joint_v3_final_large.yaml `
  --dry-run
```

Expected expansion is 20 medium-large runs and 20 large runs.

Formal execution is not authorized by this implementation PR. After this
Draft PR is reviewed and merged, any later execution must use the frozen files
above with `--resume`, preserve every run, and be handled as a separate task.

## Post-final boundary

Managerial sensitivity remains paused until final results are reported and the
paper algorithm is determined. Fairness experiments remain outside this
protocol. Neither may be inferred from or silently added to final execution.
