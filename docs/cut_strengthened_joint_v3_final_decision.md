# Cut-Strengthened Joint V3 Final Decision

## Frozen decision

```yaml
decision: final_confirmed
selected_algorithm: joint_v1_core_point_strengthened
baseline: proposed_joint_rho025_050
v3_status: completed
retuning_allowed: false
seed_replacement_allowed: false
development_validation_pooling_allowed: false
next_authorized_stage: fairness_diagnostic_only
```

The frozen core-point-only candidate passed the Final correctness gate and all
confirmation thresholds pre-registered in
`docs/cut_strengthened_joint_v3_final_protocol.md`. The V3 algorithm stage is
therefore complete for the current base robust-inventory model. This decision
does not authorize any further V3 parameter revision, seed replacement, or
candidate selection.

Final evidence is reported separately from development and validation. Those
stages are not pooled to enlarge the Final sample or estimate the Final effect.

## Evidence identity

All Final runs identify commit
`11020383bfaf49b6f538f672089704f1cdf8b860`, the merge commit of the frozen
Final-protocol PR. It was independently confirmed to be the then-current
`origin/main` and therefore an ancestor of it.

| Evidence | SHA256 |
|---|---|
| `cut_strengthened_joint_v3_final_medium_large_results.zip` | `1388446BC75E44E8E8AFC9E7973F011B14B7172AEBC8BB400749C6FE7C1D1E7A` |
| `cut_strengthened_joint_v3_final_large_results.zip` | `6641DCD67F8BFD6FA15F7580459AD31148BFF7DF64034E16C1A36F98B78985F4` |
| `cut_strengthened_joint_v3_final_medium_large.yaml` | `1D41A19BB47218F2844C2BDFEADF9B044E8776DB944C37989EF8C26FEB9C0867` |
| `cut_strengthened_joint_v3_final_large.yaml` | `60FDF4A9A642485A46E473A25DDB7502198A84EEA927D9A60E670B764F8542F3` |
| `selected_cut_strengthened_joint_v3_candidate.yaml` | `7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6` |

Both archives passed complete ZIP CRC checks (`ZipFile.testzip() == None`).
The input-YAML values above are byte-level repository-file hashes. Manifest
`config_sha256` values use the pipeline's canonical resolved-configuration
hash domain and are intentionally not required to equal the raw YAML hashes.

## Completeness, correctness, and resume audit

The read-only audit found 20/20 unique runs at each scale. Medium-large seeds
were exactly 90--99 and large seeds exactly 100--109, with no overlap. Each
seed had exactly the frozen Joint V1 baseline and the frozen core-only method.
Each result tree contained exactly 20 `run.json`, 20 `status.json`, 20
iteration logs, and 20 empty `error.txt` files. There was no third method,
duplicate run key, missing result, failed run, partial state, or parameter
drift.

The large run was interrupted once and resumed in the same output directory.
Its final manifest records 20 expected, completed, and solved runs; zero
failed and remaining runs; and `skipped_run_count: 4`. The four skips are
complete successful runs preserved by `--resume`, not missing observations.
All 20 final run keys are unique and complete, their logs are continuous from
iteration 1 through the reported final iteration, and `run.json`,
`status.json`, `results.csv`, and the final log values agree. No interrupted,
running, partial, or failed state remains. The immutable Final archive contains
only one artifact set for each run, so the resumed evidence is structurally
consistent with the pipeline's non-overwriting skip behavior.

All 86 required read-only result-audit checks passed. In particular:

- LB was nondecreasing, UB was nonincreasing, and requested MP/SP gaps were
  nonincreasing in every run;
- every logged global gap matched
  `max(0, (UB-LB) / max(1, abs(UB)))` within the frozen numerical tolerance;
- final log bounds and gaps matched `results.csv`, every Final gap satisfied
  `tol=1e-4`, and PAR-2 equaled runtime for every solved run;
- every UB was valid and identified the original robust subproblem bound as
  its source; core auxiliary LP bounds never updated UB;
- accepted core cuts were recorded as dual feasible and satisfied the current
  point floor; rejected attempts retained the original V1 primary cut;
- no cut was generated without a subproblem incumbent, core strengthening was
  inactive during termination verification, and at most one cut was added per
  iteration;
- paired V1/core-only final intervals overlapped for every seed;
- stall-secondary trigger, solve, and added-cut counts were all zero;
- core auxiliary runtime was finite, nonnegative, and recorded independently.

The reusable audit reads ZIP members in place or reads an extracted directory.
It does not extract, edit, or rewrite its input. No Final ZIP, raw result,
instance, cache, or large binary is committed with this decision.

## Frozen Final metrics

### Medium-large

| Metric | V1 | Core-only |
|---|---:|---:|
| Solved | 10/10 | 10/10 |
| Mean PAR-2 (s) | 96.75627114000963 | 23.479954390006604 |
| Mean iterations | 924.8 | 265.6 |

Core-only reduced mean PAR-2 by 75.73288623738889% and mean iterations by
71.280276816609%. Its paired PAR-2 was no higher than V1 for 10/10 seeds, and
it used fewer iterations for 10/10 seeds.

The core mechanism recorded 2,013 attempts and 2,009 accepted cuts, an
acceptance rate of approximately 99.80%. Core auxiliary runtime was
approximately 5.75% of core-only algorithm runtime.

### Large

| Metric | V1 | Core-only |
|---|---:|---:|
| Solved | 10/10 | 10/10 |
| Mean PAR-2 (s) | 928.5956241400097 | 154.1141259899945 |
| Mean iterations | 2620.7 | 645.8 |
| Mean paired rank | 2.0 | 1.0 |

Core-only reduced mean PAR-2 by 83.40352657458163% and mean iterations by
75.35772885107033%. Its paired PAR-2 was no higher than V1 for 10/10 seeds,
and it used fewer iterations for 10/10 seeds.

The core mechanism recorded 4,650 attempts and 4,648 accepted cuts, an
acceptance rate of approximately 99.96%. Core auxiliary runtime was
approximately 4.70% of core-only algorithm runtime.

## Frozen confirmation gates

| Necessary condition | Result |
|---|---|
| Identity, completeness, and correctness | Pass |
| Medium-large solved rate not below V1 | Pass |
| Medium-large mean PAR-2 no more than 103% of V1 | Pass |
| Large solved rate not below V1 | Pass |
| Large mean PAR-2 reduction at least 7.5% | Pass (83.40%) |
| Large mean iteration reduction at least 15% | Pass (75.36%) |
| At least 6/10 large paired PAR-2 values no worse than V1 | Pass (10/10) |
| At least 6/10 large paired iteration values below V1 | Pass (10/10) |
| Core-only large mean paired rank better than V1 | Pass (1.0 vs 2.0) |

Because every necessary condition passed, the frozen outcome is
`final_confirmed`. This result cannot be revised through post-Final retuning,
alternative seeds, changed thresholds, or selective reruns.

## Auxiliary paired bootstrap

The auxiliary analysis uses only the ten paired large Final seeds and the
pre-registered estimand `core PAR-2 - V1 PAR-2`. The implementation is now
fully specified as `numpy.random.default_rng(20260720)`, 10,000 paired
nonparametric bootstrap resamples, and
`numpy.quantile(..., method="linear")` percentile endpoints.

- Mean paired difference: -774.4814981500152 seconds.
- 95% percentile interval: [-977.7605067112486, -594.9803125280666] seconds.

This interval is auxiliary evidence only. It does not replace or relax the
pre-registered confirmation gates, and it does not pool development or
validation observations.

## Post-Final boundary

This Final supports only the current base robust-inventory model. No fairness
model has been implemented or tested here, and this evidence cannot be used as
computational-performance evidence for a changed fairness model. A fairness
extension changes the model and must independently re-establish every affected
correctness and computational claim.

The only next authorized stage is `fairness_diagnostic_only`. That label does
not authorize formal fairness experiments, managerial sensitivity runs, or a
model change in this PR. V3 parameters remain frozen, and the MW-type
core-point strengthening is not claimed to be strictly Pareto-optimal.
