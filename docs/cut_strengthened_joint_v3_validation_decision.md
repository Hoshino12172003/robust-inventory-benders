# Cut-Strengthened Joint V3 Validation Decision

## Frozen decision

```yaml
decision: validation_pass
selected_candidate: joint_v1_core_point_strengthened
baseline: proposed_joint_rho025_050
next_authorized_stage: final_protocol_only
formal_inference_allowed: false
```

The frozen core-only candidate passed every correctness gate and every
pre-registered validation threshold in
`docs/cut_strengthened_joint_v3_validation_protocol.md`. This decision only
authorizes a separate final-protocol PR. It does not authorize a final run,
does not establish the paper's final algorithm, and does not turn validation
results into final effect-size estimates.

No parameter was revised after development or validation. Development and
validation observations are not pooled, and no p-value or formal significance
test is reported.

## Evidence identity

The validation runs were produced by commit
`648556b1956008e93bfc8ac0459cdc3260ab93be`. That commit is the merge commit of
the frozen validation-protocol PR and was independently confirmed to be an
ancestor of the then-current `origin/main`.

| Evidence | SHA256 |
|---|---|
| `cut_strengthened_joint_v3_validation_medium_large_results.zip` | `2C54F54E8FC0EB78326228BB1C3069EBDAE6530DDAB7E332E8BB24F5CC3D5BCD` |
| `cut_strengthened_joint_v3_validation_large_results.zip` | `91D459DE451A99245BBE672D06AB4A0AB00787E576A7B1B5CB7E4973B926C780` |
| `cut_strengthened_joint_v3_validation_medium_large.yaml` | `EB7070B8045CFD3FC57B4F7DC906059F8C9CA60D9C0AD58B75CD6E8E98D41007` |
| `cut_strengthened_joint_v3_validation_large.yaml` | `44106F8A1F12D4CACA961439CA4B5EEBF8CA263AFAC567512CE541F4E80ACE27` |
| `selected_cut_strengthened_joint_v3_candidate.yaml` | `7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6` |

Both ZIP archives passed a full CRC check (`ZipFile.testzip() == None`). The
raw input-YAML hashes above are byte-level hashes of the repository files. The
manifests' internal `config_sha256` values use the experiment pipeline's
resolved-configuration hash domain; the two hash types are intentionally not
required to be equal.

## Completeness and correctness audit

The read-only audit found 20/20 runs for each scale, hence 40/40 overall. Seeds
were exactly 80--89 and each seed had exactly the frozen V1 baseline and the
frozen core-only candidate. There were no missing or duplicate pairs, failed,
skipped, remaining, or nonempty `error.txt` records. Seeds 90--109 remained
sealed and were neither generated nor accessed as experiment instances.

All 68 required audit checks passed. In particular:

- resolved run configurations retained the frozen scale definitions, time
  limits, maximum iterations, `tol=1e-4`, Gamma 2, `[2]` schedule, PAR-2
  semantics, V1 joint error budget, and frozen core-point parameters;
- LB was nondecreasing, UB was nonincreasing, and requested MP/SP gaps were
  nonincreasing in every run;
- every logged global gap exactly matched
  `max(0, (UB-LB) / max(1, abs(UB)))` within the audit tolerance;
- all UBs were valid and identified the original robust subproblem bound as
  their source; neither a core auxiliary LP nor a restricted secondary solve
  updated UB;
- accepted core cuts were recorded as dual feasible and satisfied the current
  point floor; failed strengthening attempts fell back to the original primary
  cut;
- no cut was created without a subproblem incumbent, core strengthening was
  disabled during final certification, and no iteration added more than one
  cut;
- each last iteration-log LB, UB, and gap matched `results.csv`, and the final
  intervals of the paired methods overlapped for every seed;
- the stall-secondary component had zero attempts and zero added cuts.

The audit reads ZIP members in place and never extracts, edits, or rewrites the
archives or result directories. The committed repository contains no raw
validation results or ZIP files.

## Frozen validation metrics

### Medium-large

| Metric | V1 | Core-only |
|---|---:|---:|
| Solved | 10/10 | 10/10 |
| Mean PAR-2 (s) | 124.44613205996575 | 33.350755810004195 |
| Mean iterations | 937.8 | 279.4 |

Core-only reduced mean PAR-2 by 73.20064894107454% and mean iterations by
70.20686713584986%. It was no worse than V1 in paired PAR-2 for 10/10 seeds and
used fewer iterations for 10/10 seeds.

### Large

| Metric | V1 | Core-only |
|---|---:|---:|
| Solved | 8/10 | 10/10 |
| Mean PAR-2 (s) | 1746.8291011099936 | 191.02528566997498 |
| Mean iterations | 2802.5 | 664.6 |
| Mean paired rank | 2.0 | 1.0 |

Core-only reduced mean PAR-2 by 89.06445481423505% and mean iterations by
76.28545941123996%. It was no worse than V1 in paired PAR-2 for 10/10 seeds and
used fewer iterations for 10/10 seeds. V1 reached the time limit on seeds 86
and 89; core-only solved all ten seeds.

### Core mechanism

The aggregate accepted/attempted ratio was approximately 99.82% on
medium-large and 99.98% on large. Core auxiliary runtime accounted for
approximately 6.21% and 5.01% of the corresponding core-only algorithm runtime.
The secondary component was never attempted.

## Pre-registered gate evaluation

| Required condition | Result |
|---|---|
| Correctness gate | Pass |
| Medium-large solved rate not below V1 | Pass |
| Medium-large mean PAR-2 no more than 103% of V1 | Pass |
| Large solved rate not below V1 | Pass |
| Large mean PAR-2 reduction at least 7.5% | Pass |
| Large mean iteration reduction at least 15% | Pass |
| At least 6/10 large paired PAR-2 results no worse than V1 | Pass (10/10) |
| At least 6/10 large paired iteration results below V1 | Pass (10/10) |
| Core-only mean paired rank better than V1 | Pass (1.0 vs 2.0) |

The resulting frozen decision is `validation_pass`. The validation data must
not be used for further tuning or another validation attempt. A final protocol,
if created, must be reviewed in a separate PR before any final run. Final seeds
90--109 remain sealed.

## Scope limits

This decision does not add fairness constraints, change the optimization model,
or change the demand uncertainty set. It does not run or authorize managerial
sensitivity experiments. The MW-type core-point strengthening is not claimed
to be strictly Pareto-optimal.
