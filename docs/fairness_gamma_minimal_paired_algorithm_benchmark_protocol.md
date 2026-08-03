# Minimal paired algorithm benchmark protocol (not authorized)

## Purpose

This protocol closes the strict paired-comparison gap identified after Gamma sensitivity Attempt 3. It is an algorithm-comparison experiment, not a model-selection, parameter-tuning, or sensitivity experiment.

## Frozen matrix

- Scales: `medium_large`, `large`
- Seeds: 180, 181, 182, 183, 184
- Gamma: 2
- Rho: 0.025
- Reference: certified single-cut Benders without complete scenario recourse blocks
- New tasks: ten reference frontier tasks only

The corresponding ten Hybrid cells, instances, model baselines, and anchors are read-only inputs from Gamma sensitivity Attempt 3. Hybrid must not be rerun.

## Identity and solver controls

- Each reference task must use the exact Attempt 3 instance canonical SHA, baseline run key, and anchor SHA.
- `Threads=1`, solver `Seed=0`, `FeasibilityTol=1e-7`.
- Algorithm time limit: 1,800 seconds.
- The same scientific-success definition and final exact separation objective-bound certification are required.
- Uncertified tasks receive PAR-2 = 3,600 seconds on the algorithm-runtime basis.

## Outcomes

Report certified solved rate, paired algorithm runtime where both methods certify, PAR-2, final gap, iterations, scenario blocks, and certified Farkas cuts. Timeout and uncertified results remain failures under the frozen definition.

## Statistical scope

The independent unit is the seed. Comparisons are paired within scale and seed. With five seeds per scale, emphasize paired trajectories and descriptive summaries; do not interpret non-significance as proof of equality.

## Authorization

```yaml
formal_run_authorized: false
next_authorized_stage: minimal_paired_algorithm_benchmark_protocol_review_only
```

This protocol does not authorize instance generation, solver use, reruns, selective completion, or any modification to Final Holdout or Gamma sensitivity conclusions.
