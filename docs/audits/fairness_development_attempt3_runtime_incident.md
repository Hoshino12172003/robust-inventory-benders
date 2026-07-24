# Fairness development Attempt 3 runtime incident

## Frozen identity

- execution attempt: 3
- Git commit: `2becc7a2b2d42f783e72602567f4aa6fa72e0683`
- Large configuration SHA256:
  `358629EBB7BC15371B8D2295C6B7E468E9E872FA1AF2E9BA801982057096925B`
- protocol SHA256:
  `A3B13526778DE8049A03F47B01825474ABC562CB9E67F2355717435D3754FA5F`
- seeds accessed: 120--129
- previous attempt results reused: false

## Frozen execution state

Medium-large completed under Attempt 3. Large was stopped with 33 complete
`run.json` records. The 34th task,
`regional_fairness_development_large__rho__0_025__large__seed_125__robust_regional_fairness`,
had only a running status and no complete run record.

The Attempt 3 artifacts remain permanently read-only. No baseline, certified
cost anchor, run record, checkpoint, summary, result table, or manifest from
Attempt 3 may be resumed, migrated, imported, or reused.

## Decision

```yaml
status: scientifically_invalid
scientific_selection_allowed: false
results_reused: false
invalidation_reason: runtime_pipeline_and_timing_protocol_blocker
next_authorized_stage: attempt_4_pre_run_audit_only
```

The blocker was an execution-pipeline and timing-evidence defect:

- the 1,800-second fairness limit covered the Benders algorithm, not the whole
  frontier task;
- exact Large post-evaluation solved all 4,657 scenarios sequentially with a
  separate 30-second limit for each scenario;
- no global post-evaluation wall-clock bound existed;
- no scenario checkpoint, phase heartbeat, or fine-grained resume existed;
- interruption restarted the entire frontier task;
- PAR-2 intentionally excluded post-evaluation, while `results.csv` omitted
  post-evaluation and end-to-end wall runtimes;
- the dry-run envelope omitted post-evaluation work; and
- `baseline_time_limit` was reported by dry-run while the runtime path used the
  generic `time_limit` (both were 1,800 seconds, so Attempt 3 had no numerical
  time-limit drift).

This incident is not evidence that the fairness mathematics, separation model,
Farkas certification, uncertainty set, or frozen decision thresholds were
wrong. Attempt 3 nevertheless cannot enter development selection, statistical
summaries, figures, or paper results because its required execution and timing
evidence was incomplete.
