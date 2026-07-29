# Controlled Large D2 protocol

## Frozen scope

D2 uses only `certified_hybrid_scenario_benders_fairness`, scale `large`, seeds 160--162, and rho values 0, 0.01, and 0.10. Each seed has one independently generated instance and one independently solved baseline; its three frontier runs share that seed's certified baseline anchor. The plan is exactly 3 baseline plus 9 frontier tasks. Execution attempt is 2, previous results are not reused, and the output directory is new.

No D1 instance, baseline, anchor, scenario, cut, checkpoint, manifest, result, or summary may be imported. D1 is read-only development evidence only. D1 configuration accepts only stage D1 and D2 configuration accepts only stage D2.

## Frozen mathematics and certification

The D1 mathematical model is unchanged: Gamma 2 budgeted uncertainty, complete scenario recourse blocks, append-only scenario and Farkas-cut sets, one new independently certified scenario per iteration, and exact Farkas certification. The master solver best bound is the lower bound; an incumbent is not a lower bound. Final robust feasibility requires complete exact separation with a legal objective bound. Post-evaluation checks the certified solution but cannot establish the algorithm certificate.

Threads=1, solver Seed=0, FeasibilityTol=1e-7, baseline and hybrid algorithm limits=1,800 seconds, post-evaluation limit=30 seconds per scenario, chunk size=25, and 4,657 scenarios are frozen. PAR-2 uses algorithm runtime only with multiplier 2.

## Success gate

D2 passes only with 3/3 complete optimal baselines having valid upper bounds and accepted gaps, and 9/9 complete frontier runs having algorithm status optimal, scientific status `certified_robust_optimal`, final exact separation certification, and valid 4,657-scenario post-evaluation. Implementation errors, invalid post-evaluations, identity/checkpoint/hash drift, and uncertified tasks must all be zero. Any uncertified frontier fails D2 and closes final holdout/full-grid authorization. Selective reruns, changed thresholds, and incumbent-based ranking are forbidden.

D2 is a controlled cross-seed/rho stability check, not a statistical significance study or final paper result.

## Recovery and identity

Manifest, run/status records, algorithm checkpoints, post-evaluation chunks/index, aggregation, results, and summary are atomic and resumable. Canonical run keys and short hashed physical directories are bijective. Scenario blocks and certified cuts are append-only. Resume validates stage, execution attempt, Git, config file/canonical identity, protocol, candidate, solver parameters, instance, baseline, anchor, checkpoint and post-evaluation identity. Drift, corruption, partial state, duplicate scientific objects, or old output fail closed. Overwrite is unsupported.

Formal execution is authorized only after this Draft PR is independently reviewed and merged. Before any output creation, instance generation, or solver configuration, production execution requires a clean detached worktree whose HEAD equals local `origin/main`, the exact frozen D2 config/protocol/candidate/decision SHAs, and the D1 approval decision. Dry-run is always solver-free and side-effect-free.

## Time envelope

Baseline solver limits total 3 x 1,800 = 5,400 seconds. Frontier algorithm limits total 9 x 1,800 = 16,200 seconds; combined algorithm envelope is 21,600 seconds (6 hours). Post-evaluation solver-limit envelope is 9 x 4,657 x 30 = 1,257,390 seconds (349.275 hours). These are solver-limit envelopes, not wall-time forecasts.
