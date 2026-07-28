# Fairness scalability S1 Attempt 2 cross-scale freeze

## Frozen decision

This document freezes the existing Medium-large and Large S1 Attempt 2 evidence.
It is a derived, solver-free decision record, not a new optimization run.

```yaml
decision: no_existing_candidate_passes_cross_scale_s1
data_integrity_valid: true
medium_large_pipeline_valid: true
large_pipeline_valid: true
medium_large_certified_frontier: 20/24
large_certified_frontier: 0/24
large_candidate_certified_counts:
  single_cut: 0/6
  persistent_separation: 0/6
  persistent_certified_cache: 0/6
  persistent_certified_cache_batch5: 0/6
existing_candidate_selected: null
original_s2_authorized: false
full_grid_authorized: false
attempt4_authorized: false
current_results_usable_for_algorithm_development: true
current_large_results_usable_as_certified_fairness_results: false
next_authorized_stage: fairness_large_final_remediation_protocol_only
```

S1 has completed its screening function. None of the four existing candidates
meets the cross-scale scalability requirement because every candidate is 0/6
certified on Large. Cross-scale totals cannot override a zero-certification scale.

The original S2, full-grid, and Attempt 4 remain unauthorized. This result may
not be made to pass by increasing the time limit, weakening certification,
replacing a seed, or selectively rerunning tasks. The evidence may support the
design of at most one final new algorithm candidate. That work requires a
separate protocol, branch, output directory, and execution attempt. If that
candidate fails, no additional candidates are authorized.

## Immutable source archives

Both archives were hashed before and after the read-only audit and remained
unchanged. `ZipFile.testzip()` returned no bad member, and every JSON and CSV
member parsed successfully.

| Scale | SHA256 | Entries | Files | Explicit dirs | Inferred dirs | CRC |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Medium-large | `3919E271C40BDC86F5EE7FFA582A06DDD325D5BB4EAD453550ED17A6C62DB751` | 1,677 | 1,607 | 70 | 69 | valid |
| Large | `7A3C7BF75B0D6D3228B6AA9AC0E8B8E6799A4B4925949C89DBBCBF311C2D2376` | 117 | 87 | 30 | 29 | valid |

No archive, formal `run.json`, checkpoint, instance, or large result artifact is
committed by this freeze.

## Run identity

| Identity | Medium-large | Large |
| --- | --- | --- |
| schema / execution attempt | 2 / 2 | 2 / 2 |
| run commit | `29ae09e968a206b1987714317ff7528165372a46` | `ec33a047ecd60f4cb473260f1b3c4078726db776` |
| config file SHA256 | `31FED8028653E6F0D7132F61D73157188320ABA5486A0A66FEF950642D958893` | `CEB6025CD06DFBE91312827E738A47BF65FFCC2DCDEAFAD56EA5C3B9EE790801` |
| resolved config file SHA256 | `2475EE26B5EA27C4B45B41F5AB9A5326CAA6C3CBB5BF7387AFF12BB1FEEB98D8` | `583DEB4E9AFE00104C941DB24B96E6FD4ACEDADB687ACB5D067B715780C0C9F9` |
| resolved config canonical SHA256 | `89F9298A00488506C0D1416B67DC956E48A3E085AC177720BC52DD1820C28BCC` | `D24B5B1BB735E3FEA972348A41754DFF1A7635444F0726D65439ACB937589325` |
| protocol SHA256 | `240A702464FF524AAECEE00F2611EFA7882A64096CFA794C4147189A73C86623` | `CB64C7505F81296992164359E7B2C929AE2868F9364FADDE68631AFCA2CC78B4` |
| frozen V3 candidate SHA256 | `7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6` | same |

Canonical resolved config identity is
`PyYAML safe_dump(sort_keys=True, allow_unicode=True)`, encoded as UTF-8. It is
kept distinct from the SHA256 of the original file bytes.

Both scales use seeds 160--162; rho 0 and 0.01; candidates `single_cut`,
`persistent_separation`, `persistent_certified_cache`, and
`persistent_certified_cache_batch5`; Gurobi identity `Threads=1`, `Seed=0`,
`FeasibilityTol=1e-7`; baseline, fairness, and general time limits of 1,800
seconds; post-evaluation limit 30 seconds per exact scenario and chunk size 25;
and PAR-2 equal to twice the algorithm time limit for uncertified tasks, based
on `algorithm_runtime`. Attempt 1 history is retained and
`previous_attempt_results_reused=false` for both scales.

## Cross-commit scientific equivalence

The following Git blobs and byte SHA256 values are identical between the two
formal run commits.

| File | Git blob | SHA256 |
| --- | --- | --- |
| `src/fairness_scalability.py` | `d905de66ce99f22c9a077c7d1abaec863cd7f283` | `609261BD048F355E23176CFE66B2996AD1AFF40378E27CA26EBC411EA36E5BDA` |
| `src/fairness_benders.py` | `a1a65bb87fdc52d7b8e09da7b7641dd7c515c37b` | `289BEDB61939C9FF2FE2AA116209CA441745FCDE3BDD4E85309688A720DC372E` |
| `src/robust_regional_fairness.py` | `2d265f63179ac3d37048f5ca89b5b37b6e204a50` | `25A641F0D898E4C3113D06747CE1311FFE4E67BA339B8B4D2FDE11687A260318` |
| `src/benders.py` | `41f9e716a289df65a5ab05f848db42cda7bae49a` | `37967750EE1AAD5575A9B1FE0B050F012EC21DB58FA277FBEFAA5A48CFEF1D9F` |
| `src/scenarios.py` | `ab82c7dd67f43494116f01de26877d32777907e7` | `7294C60DC318F7678F8A4464DAF2CBD85E540842C6C3858BB1D30A9DE7915511` |
| frozen V3 candidate | `1f807df78f1f00eb3e88ee4f7c8bdd39016d41b8` | `7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6` |

Candidate definitions in both manifests and resolved configurations are also
identical. Cross-scale candidate comparison is therefore permitted.

## Coverage, mapping, baselines, and anchors

Each scale contains exactly 3 baselines and 24 frontier tasks: 27 `run.json`,
27 `status.json`, 27 complete states, zero pending, zero running, zero pipeline
failures, 27 unique run keys, and 27 unique physical directory IDs. Each seed
has one baseline and eight frontier tasks; every scale/candidate has six tasks.

For every task, the physical directory, `run.json`, `status.json`, manifest
forward mapping, manifest reverse mapping, and canonical run key agree with:

```text
r_ + sha256(canonical_run_key UTF-8).hexdigest()[:24]
```

No collision, omission, or reverse-mapping drift was found. All six baselines
are certified optimal with `valid_UB=true` and a gap within the frozen tolerance.
For every seed, `C_anchor` is exactly the saved baseline upper bound; the value,
`float.hex()`, and anchor SHA agree. Its eight frontier tasks share the same
instance SHA, baseline run key, anchor value, anchor hex, and anchor SHA.

## Independently reconstructed scientific status

| Candidate | Medium-large | Large | Cross-scale |
| --- | ---: | ---: | ---: |
| `single_cut` | 6/6 | 0/6 | 6/12 |
| `persistent_separation` | 6/6 | 0/6 | 6/12 |
| `persistent_certified_cache` | 2/6 | 0/6 | 2/12 |
| `persistent_certified_cache_batch5` | 6/6 | 0/6 | 6/12 |

Medium-large has 3/3 certified baselines and 20/24 certified frontier tasks.
The remaining frontier statuses are three `unknown_uncertified` tasks whose
algorithm status is `separation_stalled_duplicate`, and one
`time_limit_uncertified` task. Large has 3/3 certified baselines and 0/24
certified frontier tasks; all 24 are `time_limit_uncertified`. Pipeline
`failed_run_count` is zero for both scales, which is distinct from scientific
certification failure.

Neither `complete`, an algorithm status, an incumbent, an objective T, a lower
bound, nor a final iteration value is treated as robust certification. Large
uncertified numeric values are excluded from formal fairness results and
candidate selection.

## Checkpoint and post-evaluation evidence

Both scales contain 24 algorithm checkpoints, one for every frontier run. Each
checkpoint identity matches its run and its checkpoint result is a matching
subset of the final run result.

Medium-large has 20 successful post-evaluations, exactly the certified frontier
set. Each contains 1,831 ordered, uniquely keyed scenarios, a valid objective T
consistency result, no errors, and passing frozen acceptance evidence. All 20
indexes and all 1,480 chunk SHA256 values match. Large has no certified frontier
task and therefore correctly has no successful post-evaluation or post-evaluation
index; this is not a pipeline-loss finding.

## Reporting and timing reconstruction

The audit rebuilds 27 result projections per scale directly from `run.json` and
reconciles 216 metadata/iteration-log fields per scale. Both sets pass 216/216.
PAR-2 is not replaced with raw runtime: every uncertified Large task is 3,600
seconds. The three baseline PAR-2 values per scale are unambiguously derived
from their certified algorithm runtime because the baseline payload predates the
stored frontier PAR-2 field.

Large candidate means (six tasks each) are development evidence only:

| Candidate | Algorithm s | PAR-2 s | Separation s | Master s | Iterations | Cuts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_cut` | 1800.006314 | 3600 | 1785.775807 | 12.074705 | 556.333 | 555.333 |
| `persistent_separation` | 1800.006839 | 3600 | 1787.624911 | 12.036324 | 580.167 | 579.167 |
| `persistent_certified_cache` | 1800.017952 | 3600 | 139.775996 | 1656.794554 | 6266.833 | 6266.833 |
| `persistent_certified_cache_batch5` | 1800.038200 | 3600 | 241.233310 | 1556.359773 | 2311.667 | 11558.333 |

Thus separation dominates `single_cut` and `persistent_separation`; certified
cache shifts the bottleneck to the master; batch-5 reduces outer iterations but
creates substantially more master cuts. None obtains robust certification by
1,800 seconds, so these observations are not certified fairness outcomes.

## Frozen experiment matrix

`frozen_run_matrix.csv` contains 54 rows, 54 unique run keys, 54 unique
`(scale, task_type, seed, rho, candidate)` identities, and no blank cell.
Baseline rho, anchor, and baseline reference are explicitly `NOT_APPLICABLE`.
Frontier rows preserve their baseline and anchor identity and every row cites
the corresponding immutable source archive SHA256. Ordering is frozen by scale,
seed, task type, rho, and candidate.

`experiment_matrix_summary.csv` has 39 deterministic rows covering detailed
scale/task/candidate/rho groups, per-scale candidate rollups, per-scale totals,
cross-scale candidate rollups, and cross-scale totals. It verifies:

| Scope | Baseline | Frontier | Total |
| --- | ---: | ---: | ---: |
| Medium-large | 3 | 24 | 27 |
| Large | 3 | 24 | 27 |
| Cross-scale | 6 | 48 | 54 |

Each scale/candidate has `3 seeds × 2 rho = 6` frontier tasks; each candidate
has 12 cross-scale tasks. A candidate cannot enter the original S2 when any
scale, here Large, has no certified result.

## Deterministic derived evidence

Three independent builds (two temporary directories and the tracked analysis
directory) produced eight byte-identical files. `artifact_sha256.csv` indexes
the seven non-recursive artifacts; its own SHA256 is reported separately.

| Artifact | SHA256 |
| --- | --- |
| `decision.json` | `7EBFD4F22C5AF2B26E630722FB3F8D17E83FAEE9AA58248BDA32A386FFDE29B2` |
| `source_archive_provenance.json` | `0ACAC55C528386B76EC7A0386C9613BCBC22EBF9099BF7A394C007DBDCC3DF3A` |
| `medium_large_audit.json` | `79C6FB300E35616195487759A3FA7D7B4FAFC7032B99CF17DDBCC38E2F2F642E` |
| `large_audit.json` | `4AFE6C59141125DB24C5CA0848D003D2125AB7BDF63331D14EDB2638E69C8AF9` |
| `cross_scale_candidate_summary.csv` | `92D5F5A9BF17B98C7F9D98F1F758FD205765CC58C18A9D9225A3B3A1BF9EE699` |
| `frozen_run_matrix.csv` | `0DAB49DF711EDAE15B16778F90DEDC73249F0845FF9EFCF1F4182D39B5410AB2` |
| `experiment_matrix_summary.csv` | `9A4D6EACBB878B08E9B749B55B165BBB3A0308A859615BDDAC1469BFFFBEBA6B` |
| `artifact_sha256.csv` | `BC0781818CA2DD1F5964512BEEF5438CAD8181BE5FF8B0A992DE716E98DF2358` |

The derived files cite both source archive SHA256 values and are explicitly
labelled as audit/decision evidence, not formal optimization output.

## Scope controls

This freeze called no Gurobi solver, ran no formal configuration, generated no
instance, and did not access seeds 130--159. It did not run S2, full-grid,
Attempt 4, or a new Large experiment. It does not modify the fairness model,
separation/Farkas certification, rho, time limits, success definitions,
thresholds, source archives, or existing formal result directories. It develops
no new algorithm.

After independent review and merge of this Draft PR, its merge commit is the
only permitted base for `fairness_large_final_remediation_protocol`. That next
stage is limited to protocol, mathematical proof, tiny instances, and dry-run;
it is not authorization to run Large.
