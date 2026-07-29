# Final cross-scale fairness holdout protocol

## Frozen scientific scope

This is the final experiment. No further algorithm development, candidate selection or parameter tuning is permitted. The only candidate is `certified_hybrid_scenario_benders_fairness` with exact Farkas certification, complete append-only scenario blocks, one independently certified scenario addition per iteration, and final exact separation. D1 and D2 are development evidence only.

The holdout matrix is frozen before any holdout instance is generated:

| Scale | Seeds | Rho | Baselines | Frontiers | Total |
| --- | --- | --- | ---: | ---: | ---: |
| medium_large | 170--179 | 0, 0.01, 0.025, 0.05, 0.10 | 10 | 50 | 60 |
| large | 170--179 | 0, 0.01, 0.025, 0.05, 0.10 | 10 | 50 | 60 |
| combined | 170--179 | five values | 20 | 100 | 120 |

Each scale/seed has one newly generated instance and one newly solved certified cost baseline. Its five frontier tasks share exactly that instance, baseline run key, certified `C_anchor`, `anchor_value_hex`, and `anchor_sha256`. Execution attempt is 1 in a new final-holdout experiment family. `previous_attempt_results_reused=false`. Seeds 130--169 and 180 onward, other rho values, other candidates and any full grid are forbidden.

Seeds 170--179 are reserved and must remain inaccessible until a separate independent pre-run authorization. Repository declarations that merely name the reserved set are not access. Before authorization, a fail-closed audit must again establish that no instance, solver run, baseline, anchor, checkpoint or result exists for any reserved seed. Any access evidence stops the experiment; the seed set may not be silently replaced.

## Frozen solver and certification identity

- Threads: 1
- solver Seed: 0
- FeasibilityTol: 1e-7
- Gamma: 2
- baseline algorithm limit: 1,800 seconds
- Hybrid fairness algorithm limit: 1,800 seconds
- post-evaluation limit: 30 seconds per scenario
- post-evaluation chunk size: 25
- medium-large post-evaluation scenarios: 1,831
- large post-evaluation scenarios: 4,657
- PAR-2 multiplier: 2
- PAR-2 basis: algorithm runtime

`certified_robust_optimal` requires a valid certified baseline anchor, a legal scenario-master solver bound, accepted global gap, final exact separation with a legal objective bound proving no violation, and valid complete post-evaluation. An incumbent, empty candidate set, repeated scenario, time limit or post-evaluation alone cannot certify success. Time limits are valid scientific failures and continue to the next preregistered task. Implementation errors, identity drift, corrupt checkpoints or invalid post-evaluation stop the batch fail closed.

All instances, baselines, anchors, manifests, scenario/cut checkpoints, post-evaluation chunks and aggregates are new. Stable canonical run keys use short hashed physical directories. Output is atomic and resumable; `--overwrite` is unsupported. Resume may not duplicate a baseline, scenario block, certified Farkas cut, post-evaluation scenario/chunk or run record.

## Frozen reporting and statistical unit

The independent experimental unit is the seed, never a seed-rho task. All 10 rho/scale observations belonging to a seed form one cluster.

- Certification counts and rates are reported by scale and rho, with all preregistered tasks disclosed.
- PAR-2 includes every frontier; uncertified tasks receive 3,600 seconds.
- Objective T, robust cost and fairness quantities are used as optimal results only for certified tasks.
- Cluster bootstrap resamples the 10 seeds. For each sampled seed, both scales and all five rho values move together.
- A per-rho cross-scale paired comparison contains exactly 10 seed pairs.
- An overall cross-scale comparison first aggregates the five rho values within each seed, then operates on 10 paired seed summaries; an equivalent seed-cluster bootstrap is allowed.
- If five rho-specific hypothesis tests are reported, their p-values use Holm correction. Conditions and all tests are disclosed without significance-based selection.
- The 100 seed-scale-rho frontier tasks must never be treated as 100 independent observations.

No post-hoc pass threshold, selective rerun, seed substitution, rho deletion or outcome-dependent analysis may be introduced. Data-integrity validity requires all 120 registered tasks to have coherent terminal records and zero implementation/identity/checkpoint/invalid-post-evaluation failures. Scientific time limits remain reported outcomes rather than pipeline errors.

## Authorization boundary

This protocol and its dry-run implementation do not authorize formal execution. `formal_run_authorized=false`. Formal mode must fail before output creation, instance generation, Gurobi configuration or any write. The only next stage is `fairness_hybrid_final_cross_scale_holdout_review_only`; after independent review, a separate pre-run authorization must re-audit seed non-access, the merged Git tree, paths and identities.
