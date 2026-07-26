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
- schema version: 3
- execution attempt: 3

## Frozen execution state

The final read-only evidence in `E:\rf3` was independently checked without
modifying any artifact:

| Scale | completed_count | pending_count | solved_count | run.json | status.json |
|---|---:|---:|---:|---:|---:|
| Medium-large | 60 | 0 | 57 | 60 | 60 |
| Large | 60 | 0 | 28 | 60 | 60 |

All 120 status files record `complete`; no `running` state remains. Both
manifests record schema 3, execution attempt 3, the frozen Git commit, and
`previous_attempt_results_reused: false`.

Read-only evidence hashes are:

- Medium-large `fairness_development_manifest.json`:
  `E6AD8B08702E9FDBD44595A7F8753919E2233697CEA0867F84E5573055D7AAC1`;
- Medium-large `run_manifest.json`:
  `E0B458A2B3A4B3BD4B17D0E1B675A1C153EC6E70CD52BCFF041DA0781886C253`;
- Large `fairness_development_manifest.json`:
  `F4F75F6CB2AAC124137E6C4CE45B319743010D5D6138FBAB1165F63612046EB7`;
- Large `run_manifest.json`:
  `616AC5A571B3093BBF30F35AA51F6E97F4FFC0152C2BF3EA88E67400524686BC`.

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

The observed 57/60 and 28/60 solved counts are not scientific outcomes and
must not be used to adjust parameters, thresholds, rho levels, or the success
definition. No Attempt 3 artifact may be used for rho selection.
