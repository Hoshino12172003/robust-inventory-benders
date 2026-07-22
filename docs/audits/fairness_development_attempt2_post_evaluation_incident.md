# Fairness development attempt 2 post-evaluation incident

## Frozen disposition

```yaml
attempt: 2
initial_status: quarantined_pending_post_evaluation_audit
decision: implementation_blocker
scientific_selection_allowed: false
attempt_results_scientifically_valid: false
next_stage: correctness_hotfix_only
large_scale_authorized: false
```

The entire medium-large attempt is quarantined. The 22 records labelled
`certified_optimal` must not be selected or analysed separately, the 25
`invalid_post_evaluation` records must not be treated as ordinary performance
failures, and the 3 `uncertified_time_limit` records remain unsolved. No attempt
2 record may be combined with a later execution.

```yaml
execution_attempt_next: 3
development_seeds_previously_accessed: [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
previous_attempt_results_reused: false
prior_attempts:
  - attempt: 1
    git_commit: 7bc8e81f91f4a4c7baf2c080af63a09ada1178d6
    seeds_accessed: [120, 121, 122, 123, 124, 125, 126]
    scientifically_valid: false
    results_reused: false
    invalidation_reason: separation_certificate_architecture_defect
  - attempt: 2
    git_commit: 98c615767032bb6c57f28476bebc0392037fbf34
    seeds_accessed: [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
    scientifically_valid: false
    results_reused: false
    invalidation_reason: post_evaluation_tolerance_boundary_defect
```

## Input identity and immutable evidence

- Evidence directory (read only):
  `E:\rf2\experiments\results_regional_fairness_model\development_medium_large`
- execution Git commit:
  `98c615767032bb6c57f28476bebc0392037fbf34`
- frozen medium-large YAML byte SHA256:
  `E52FAA5EB93E43C6C6B611E47239DD98378DBF039BD9ECA3C49FF7B2D5AE59E1`
- canonical manifest config SHA256:
  `6ff5e3aefc245e49233a489b89ce5d8b3b01e3640805d9c122871c165dcadbf7`
- output `resolved_config.yaml` SHA256:
  `498237BA2CF2C2259078EAD1E741FC85CCF24FC56C0E41C3C815BAB850AF5BDD`
- `fairness_development_manifest.json` SHA256:
  `EF046372E39C496B7BEFD4316ED4A728355CE17FE489820B8E9FFEFAB1D2A869`
- `run_manifest.json` SHA256:
  `EF9B1993F42E3FDD537AFDF3543CF0E1AF9FB814D1B4B591DC1753CD273BF932`
- `results.csv` SHA256:
  `A21F9471C53869FF01C01184BE5FEB9E74B0CF65CF3DDF0D9A85114588044D2D`
- `summary.csv` SHA256:
  `8F1C4357B8781476455E9B1687A3C5BC2BF4CCDFDB9A2E216BCACC9A7CC25A5A`
- frozen V3 candidate SHA256:
  `7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6`

Hashes for all 60 immutable `run.json` records are stored in
`docs/audits/fairness_development_attempt2_run_sha256.csv`.

The manifest reports 60 completed tasks, 32 solved tasks, no pending task, and
24 resume skips. Ten baselines succeeded. Frontier classifications are 22
`certified_optimal`, 25 `invalid_post_evaluation`, and 3
`uncertified_time_limit`; no task remained `running`. Attempt 1 results were
not reused.

The last chronologically successful frontier record was seed 129, rho 0.025:
`regional_fairness_development_medium_large__rho__0_025__medium_large__seed_129__robust_regional_fairness`.
The last invalid record was seed 129, rho 0.10:
`regional_fairness_development_medium_large__rho__0_1__medium_large__seed_129__robust_regional_fairness`.

## Status cross-table

`C` = certified-optimal record, `I` = invalid post-evaluation record, and `U`
= uncertified time limit.

| seed | rho=0 | rho=.01 | rho=.025 | rho=.05 | rho=.10 |
|---:|:---:|:---:|:---:|:---:|:---:|
| 120 | C | I | C | I | I |
| 121 | C | C | C | I | I |
| 122 | C | C | I | I | I |
| 123 | I | C | U | U | U |
| 124 | I | C | I | C | I |
| 125 | C | I | C | I | I |
| 126 | I | I | C | I | C |
| 127 | C | C | C | C | C |
| 128 | C | I | I | C | I |
| 129 | I | I | C | I | I |

Every invalid record has the same stored failure class:
`Recovered policy violates the regional max-shortage-rate cap.` No invalid
record reports a cost-budget failure. The stored per-scenario error counts
range from 3 to 1,831 because the same boundary defect can affect some or all
of an instance's recovery LPs.

## Independent fixed-scenario audit

The audit used only saved `x`, `y`, `T`, `B_rho`, saved instance JSON, and the
deterministic 1,831-scenario enumeration. It did not solve a master problem,
change a first-stage solution, generate an instance, resume the batch, or
write the evidence directory. It did not change rho selection or generate any
new candidate fairness policy. These formal-seed fixed-policy forensic LPs
were used only to locate the implementation defect and are excluded from
paper tables, rho selection, and scientific statistics.

For all 25 invalid records, all 1,831 scenarios were rebuilt as an independent
fixed-scenario primal LP with the exact formal recourse constraints, exact
cost cap, exact regional shortage-rate cap, and the frozen Gurobi
`FeasibilityTol=1e-7`. In total, 45,775 LPs were solved.

- fixed-scenario LPs with optimal status: 45,775 / 45,775;
- runs with a true cost violation above `1e-7`: 0 / 25;
- runs with a true fairness violation above `1e-7`: 0 / 25;
- maximum robust cost-budget residual: `4.256435204297304e-09`;
- maximum worst-shortage-rate-minus-T residual: `4.62488380925663e-12`;
- maximum independently reconstructed primal residual:
  `4.256435204297304e-09`.

Consequently, there is no real violation in the 25 saved policies under the
frozen feasibility tolerance. Their saved separation logs are consistent with
this result: the last certification separation is `optimal`, incumbent and
objective bound are both zero, and the reason is
`objective_bound_proves_no_violation`. No worst independently reconstructed
scenario was stored as a false-positive exclusion in its final call.

## Root cause

The post-evaluation recovery LP added the full feasibility tolerance to each
mathematical cap:

```text
recourse_cost <= remaining_budget + tolerance
regional_shortage <= (T + tolerance) * regional_demand
```

It then checked the recovered solution against exactly the same tolerance
with a strict floating-point comparison. Cost minimization frequently chooses
the cheap-shortage regional solution exactly on `T + 1e-7`. Binary floating
representation produced a residual infinitesimally greater than `1e-7`, so a
mathematically accepted boundary policy was labelled invalid. A first-scenario
probe found the relaxed recovery residual at approximately `1e-7` for every
invalid run; the independent exact-cap LPs above reduced the maximum residual
to `4.6e-12`.

This is a post-evaluation implementation inconsistency. It is not a separation
bound error, scenario encoding error, no-good exclusion error, max/min reversal,
or mixing of incumbent and objective bound. Nonetheless, because the output
classification and solved-rate accounting are wrong, it is a correctness
blocker and attempt 2 is scientifically invalid as a whole.

## Correctness-only repair

Protocol identity history is retained rather than overwritten:

- original development protocol:
  `BBA9973BB8A4D660202FA3D99DBDC957DB00A8A336632667A09B9F47974E22B5`;
- first post-evaluation correction revision:
  `04D4A833A5FE018FA3120B1ECA7DDAC6F2BEF216DE4D27EAB806280B2261F16E`;
- Attempt 3 merge-blocker correction revision:
  `A3B13526778DE8049A03F47B01825474ABC562CB9E67F2355717435D3754FA5F`.

The first revision removed tolerance from the recovery LP right-hand sides and
introduced end-to-end status separation. The second freezes the one-step
`nextafter` acceptance boundary and evidence, complete prior-attempt identity,
fresh-output guard, and exhaustive public status enumeration. Neither revision
changes the mathematical model, rho grid, seeds, anchor, time limits, or
development/validation/final decision rules.

The repair does not change the mathematical model or any frozen tolerance.
The recovery LP now enforces the exact formal right-hand sides, while the
frozen tolerance is used as the solver feasibility tolerance and as the
independent acceptance tolerance. Result records also preserve
`algorithm_status` and add an end-to-end `overall_status`, distinguishing:

- `master_optimal_but_robust_uncertified`;
- `certified_robust_optimal`;
- `time_limit_uncertified`;
- `invalid_post_evaluation`;
- `implementation_error`.

The development YAML, rho grid, seeds, time limits, selection threshold,
uncertainty set, V3 parameters, `src/benders.py`, and robust model are unchanged.

After this hotfix is merged, a future attempt 3 requires a new Git commit and
a physically new output directory. It must start all 60 tasks from scratch and
must not import attempt 1 or attempt 2 baselines, anchors, records, manifests,
or summaries. Manifest schema 3 records
`execution_restart_after_post_evaluation_hotfix: true`,
`previous_attempt2_scientifically_invalid: true`,
`previous_attempt2_results_reused: false`, and all previously accessed attempt
2 seeds 120--129. Large remains unauthorized until a clean medium-large
attempt is accepted.
