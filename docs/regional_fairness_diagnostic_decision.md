# Regional Service Fairness Diagnostic Decision

## Frozen decision

```yaml
decision: structural_fairness_gap
diagnosis_valid: true
fairness_gap_source: structural_not_recourse_degeneracy
retuning_allowed: false
seed_replacement_allowed: false
threshold_revision_allowed: false
next_authorized_stage: fairness_model_development_protocol_only
```

This is a read-only, cross-scale decision based only on the pre-registered
regional service fairness diagnostic. It authorizes preparation and review of
a fairness-model **development protocol only**. It does not authorize a
fairness-aware optimization run, validation, Final experiment, seed use, or
model change.

## Evidence identity and integrity

The formal diagnostic was produced by Git commit
`ce96c183248044c024f046a9a2bbe29c6f0f6f04`. That commit is the merge commit
for PR #29 and is an ancestor of the decision branch's `origin/main` baseline.

| Evidence | Frozen SHA256 | Independently recomputed | CRC |
|---|---|---|---|
| `regional_fairness_diagnostic_medium_large_results.zip` | `A8011C1B4DF7ECAE317F7FDCECCF9CB0E042DB97174CCE6A32E96C6FD0070AE8` | exact match | all members passed |
| `regional_fairness_diagnostic_large_results.zip` | `2D64325B0C40330F54DA8644CB90E2BC0607983E79D98810988198B1EBB96AD7` | exact match | all members passed |

Frozen repository identities were also independently recomputed:

- medium-large config:
  `04D2CA32C31D7B2D3C9071583C4BC3897740B463D6AD945A8A52554A6317C79C`;
- large config:
  `7A40FF6CFEDB02F44D57C999377377B7EB25E406EBE417791CA7A0C22C2FB307`;
- diagnostic protocol:
  `EC7761D96C1D2A17F96EBA90BF4BFB520A9CE6359F938ACD7F294A10E7F24A38`;
- selected V3 candidate config:
  `7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6`.

The diagnostic manifests were `completed`. Their pending, failed, and
interrupted counts were all zero. All recorded final-output hashes and all
three locked base-output hashes were recomputed from ZIP members.

| Scale | Base `results.csv` | Base `summary.csv` | Base `run_manifest.json` |
|---|---|---|---|
| medium-large | `37780DD09B8F0FFE47F8D72EEDEEDDE7898524C2A93EE176E30D907A3FD6A4BE` | `17958D74C6A1A61289B1F2BEBDD370ECB3882C818874785ED70D3F248677CC9B` | `78245B5E18586C7F24CFDE7C0E7AFEDD6A2CE034305F1BBEE2751FDE3C29D49A` |
| large | `18BC0A3657E8EF001862260435F36120A1A9F3E94A859C15CD41A2BD0D75F294` | `7374B85C31B42EEED28CE500A143465F8F6B7D79F1902D6AEADD033BFDB00A81` | `CB326FF1225AE9D2D0A8AED516969A39C978BF7C66C1BC0B3F03232D4B677E37` |

## Run and scenario completeness

Both scales use seeds 110--119 and only
`joint_v1_core_point_strengthened`. Each contains ten unique, complete base
runs with the matching frozen first-stage inventory solution and Git/config
identity. There are no missing, duplicate, substituted, or additional seeds,
methods, scales, or run keys.

| Scale | Instances | Scenarios per instance | Instance-scenario pairs | Checkpoint blocks | Region-policy rows |
|---|---:|---:|---:|---:|---:|
| medium-large | 10 | 1,831 | 18,310 | 370 | 366,200 |
| large | 10 | 4,657 | 46,570 | 940 | 1,117,680 |

For every instance, the audit reconstructed the nominal, singleton, and pair
deviation patterns in the repository's deterministic Gamma=2 enumeration
order. Scenario keys, explicit deviation JSON, deviation SHA256, demand values,
Gamma usage, checkpoint ranges, and scenario IDs all matched. The regional
CSV was then reproduced row by row from the verified checkpoints; its full
primary key was unique and no extra or missing record was found.

## Independent recourse and metric audit

For every scenario, default and fair-best recourse were optimal, used the same
saved first-stage solution and demand scenario, and had no hidden invalid
record. Fair-best cost satisfied

```text
Q_fair <= Q_default + 1e-6 + 1e-6 * max(1, abs(Q_default)).
```

Regional demand and shortage were recomputed from the checkpoint matrices.
The audit independently reconstructed fill rate, minimum regional fill rate,
WGap, WWD, scenario extrema, and instance-level robust extrema. Zero-demand
handling was checked as `not_applicable`; it was never replaced by a fabricated
zero fill rate. Output fields report allocated unit transport **cost**, not a
physical distance that the current model does not contain.

## Independently recomputed scale statistics

| Scale | Default count WGap >= 0.10 | Default median WGap | Fair-best count WGap >= 0.10 | Fair-best median WGap | Reductions >= 0.05 | Signal |
|---|---:|---:|---:|---:|---:|---|
| medium-large | 10/10 | 0.39242929257889425 | 10/10 | 0.39239004663293100 | 0/10 | `structural_fairness_gap` |
| large | 10/10 | 0.28009684972496840 | 10/10 | 0.28002746849346244 | 0/10 | `structural_fairness_gap` |
| pooled | 20/20 | 0.32626575171524050 | 20/20 | 0.32610999680338876 | — | structural rule passed |

The negligible default-to-fair-best reductions do not satisfy the frozen
degeneracy criterion. The regional service difference therefore persists even
after selecting the fairest recourse within the frozen cost-optimal tolerance.

## Pre-registered joint gates

All structural gates passed:

- medium-large has at least 4/10 fair-best WGap values at or above 0.10;
- medium-large median fair-best WGap is at least 0.05;
- large has at least 4/10 fair-best WGap values at or above 0.10;
- large median fair-best WGap is at least 0.05;
- pooled evidence has at least 8/20 fair-best WGap values at or above 0.10;
- pooled median fair-best WGap is at least 0.05.

`recourse_degeneracy_only` does not apply because fair-best itself satisfies
the structural rule and no scale has four reductions of at least 0.05.
`no_material_fairness_gap` does not apply because all 20 fair-best values are
at least 0.10 and both medians exceed 0.05. The evidence is complete, so the
result is not `fairness_diagnostic_inconclusive`.

## Interpretation boundary

The result establishes a structural **regional service fairness/equity gap in
the current inventory model**. It does not establish unfairness toward a
socially disadvantaged or vulnerable population: the instances contain no
demographic vulnerability, income, protected-group, or social-disadvantage
data. It also does not show that a particular fairness constraint will improve
the system.

The next stage may only pre-register a fairness-model development protocol.
Changing the objective, constraints, uncertainty set, or fairness formulation
requires a separate reviewed protocol and new correctness tests. Seeds
120--159 remain sealed and unused. Diagnostic results must not be used to
retune V3, revise the materiality thresholds, replace seeds, or claim
validation/Final evidence for a future fairness-aware model.

## Audit result

The independent streaming audit passed 89/89 required checks. No optimization,
recourse solve, seed generation, or result mutation was performed during this
decision audit.
