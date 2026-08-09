# Minimal paired benchmark delivery

## Scientific scope

This pipeline implements the ten-cell protocol in
`fairness_gamma_minimal_paired_algorithm_benchmark_protocol.md`. It runs only the
reference frontier for `medium_large` and `large`, seeds 180--184, Gamma 2 and
rho 0.025. It never regenerates a baseline and never reruns Hybrid.

Each task is paired to the immutable Gamma sensitivity Attempt 3 archive by the
tuple `(scale, seed, Gamma, rho, instance canonical SHA, instance file SHA,
baseline run key, anchor float hex, anchor SHA)`. The source ZIP is read-only;
algorithm checkpoints, incumbents, cuts, scenario blocks, caches, exact
certificates and post-evaluation records from Hybrid are not imported.

## Reference algorithm resolution

The merged protocol's reference is resolved to the existing production
`single_cut` path in `solve_fairness_benders`:

| Setting | Frozen value |
|---|---|
| Complete scenario recourse blocks | disabled |
| Persistent separation | disabled; rebuilt complete separation MILP |
| Certified scenario cache | disabled |
| Separation solution pool | disabled |
| Batch size | 1 |
| Maximum independently certified cuts per iteration | 1 |
| Certified Farkas fixed-scenario verification | required |
| Final exact objective-bound separation | required |
| Algorithm limit | 1,800 seconds |
| PAR-2 | algorithm runtime; 3,600 seconds if uncertified |
| Scientific success | final exact certification and valid complete post-evaluation |

Threads, solver Seed and FeasibilityTol are fixed to 1, 0 and `1e-7`. Gamma 0
or 1, other rho values, other seeds, new baselines, Hybrid reruns, full grids,
selective reruns and mathematical-model changes are outside authorization.

## Recovery and reporting

The output root is independently ignored by the exact repository rule
`/experiments/results_fgmpb/`. A completed algorithm checkpoint is reused only
when its full run identity matches. Post-evaluation uses the existing atomic,
chunked recovery implementation. Corrupt manifests, mappings, checkpoints,
source identities or authorization fail closed. `--overwrite` is unsupported.

`results.csv` includes every completed reference task. `summary.csv` keeps the
five planned seeds in each scale as the denominator. `paired_comparison.csv`
contains all ten planned pairs; uncertified reference tasks receive PAR-2 and
are not removed. Objective and cost differences are reported only when both
members certify. The seed is the independent statistical unit and no
equivalence claim is authorized.

## Operational boundary

Dry-run reads the frozen pairing catalog and source ZIP hash only. It does not
read instance payloads, import `gurobipy`, create output, or call a solver.
Formal execution additionally requires the reviewed authorization-only
successor commit and the exact short worktree root recorded in configuration.
