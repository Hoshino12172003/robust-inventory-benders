# Certified Hybrid fairness Gamma sensitivity protocol

## 1. Scope and scientific purpose

This protocol freezes a single post-holdout sensitivity experiment for the already selected candidate `certified_hybrid_scenario_benders_fairness`. It studies only the budgeted-uncertainty parameter `Gamma` at the fixed fairness budget `rho=0.025`. It is not an algorithm-selection, parameter-tuning, or additional rho experiment. The final Hybrid holdout remains immutable and authoritative under PR #51 merge commit `13493ba63006604443f54f61799842dc2a3fbac9`, source ZIP SHA256 `BD253A4DE967143B8CD88E7DEADCB821D6062208CB71664B900F9BD8EBDF3839`, corrected-results SHA256 `50EB5823F4C7138E65FA36546B90EE081B48949D2F961F5AFDFAE098A7F0A496`, and paper-metrics SHA256 `044689ABF1ADD1C1FC217FCB5F46B8D280D8659865EE3A3707EBB9FE792F2E37`.

The questions are how increasing Gamma changes the certified baseline robust cost, fairness cost budget, certified minimum regional fill-rate guarantee, exact post-evaluation robust cost, inventory, opened warehouses, algorithm effort, and whether the structural trends agree between Medium-large and Large.

## 2. Frozen matrix and identities

- stage: `GAMMA_SENSITIVITY`
- scales: `medium_large`, `large`
- seeds: `180,181,182,183,184`
- Gamma: `0,1,2`
- rho: `0.025` for frontier and literal `NOT_APPLICABLE` for baseline
- candidate: `certified_hybrid_scenario_benders_fairness`
- candidate SHA256: `8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0`
- execution attempt: `1`
- previous-attempt results reused: `false`

For every scale x seed x Gamma there is one newly solved baseline and one frontier. Hence each scale has 15 baselines, 15 frontiers, and 30 runs; the combined matrix has exactly 30 baselines, 30 frontiers, and 60 runs. Gamma is part of the baseline identity because the certified cost anchor is Gamma-dependent:

\[
C_\rho(\Gamma)=(1+\rho)C^*(\Gamma).
\]

Every canonical run key includes stage, scale, task type, seed, Gamma, rho, candidate, and execution attempt. Its short physical directory is the prefix `r_` plus 24 hexadecimal digits from SHA256 of that exact canonical key. The manifest freezes both directions of the mapping; 60 unique run keys and 60 unique directory IDs are required, with zero duplicate or collision. A frontier may reference only the baseline and anchor with the same scale, seed, Gamma, and attempt. Baselines for different Gamma values are never shared.

All instances, baselines, anchors, runs, checkpoints, post-evaluation chunks, results, and summaries are new to this experiment family. Importing any such object from Final Holdout, D1, or D2 is prohibited. Same-seed runs across Gamma use the same deterministic random-number seed to provide the preregistered within-scale paired trajectory, while each Gamma receives its own stored instance identity and baseline identity.

## 3. Exact uncertainty sets

Medium-large has 60 demand components:

\[
|\mathcal U_0|=1,\qquad |\mathcal U_1|=1+60=61,\qquad
|\mathcal U_2|=1+60+\binom{60}{2}=1831.
\]

Large has 96 demand components:

\[
|\mathcal U_0|=1,\qquad |\mathcal U_1|=1+96=97,\qquad
|\mathcal U_2|=1+96+\binom{96}{2}=4657.
\]

Gamma 3 and 4 are outside scope and must be rejected. In particular,

\[
|\mathcal U_3^{large}|=1+96+\binom{96}{2}+\binom{96}{3}=147537,
\]

so complete post-evaluation grows sharply and is not part of this frozen sensitivity analysis.

## 4. Frozen solver and certificate settings

- Threads=1; solver Seed=0; FeasibilityTol=1e-7.
- Baseline and Hybrid algorithm time limits are both 1,800 seconds.
- Exact post-evaluation limit is 30 seconds per scenario; chunk size is 25.
- PAR-2 multiplier is 2 and its basis is `algorithm_runtime`.
- Complete scenario recourse blocks, certified Farkas separation, append-only scenarios/cuts, at most one independent certified scenario per iteration, and final exact separation are unchanged.
- A frontier is solved only when its scientific status is exactly `certified_robust_optimal`; post-evaluation alone cannot certify it.
- Post-evaluation is complete, chunked, resumable, and fail-closed on identity, ordering, cumulative-count, or SHA drift.

No mathematical model, fairness definition, uncertainty-set structure, Hybrid candidate, Farkas logic, scenario block, tolerance, or success definition may change. In particular, `src/benders.py` and `src/scenarios.py` are protected.

## 5. Seed isolation and fail-closed access audit

Seeds 180--184 remain inaccessible before a separately authorized formal run. A repository/config declaration that preregisters the numbers is not access. Actual access means evidence of a generated instance, solver run, baseline, anchor, run/status record, algorithm or post-evaluation checkpoint, or formal result for a reserved seed. The audit reads Git-tracked structured files, known configuration and formal-result roots, instance directories, manifests, run/status/checkpoint files, freeze evidence, and any supplied ZIP listing without extracting it. Detection stops the process; seeds are never silently replaced.

The audit reports four distinct categories: preregistration declaration, generated instance, solved run, and formal-result access. It examines both structured contents and path/member names, including opaque files and ZIP members. It is rerun immediately before any later formal authorization and before any output directory, instance generation, solver import, or solver configuration.

## 6. Output, checkpoint, and resume contract

Formal roots are short Windows-safe paths:

- `experiments/results_fh_gamma/ml_a1`
- `experiments/results_fh_gamma/lg_a1`

The formal detached worktree is frozen as `E:\rfgs`. Before execution, the runner expands every root artifact, run/status file, baseline and algorithm checkpoint, post-evaluation final/index/chunk, and atomic `.tmp` name under that exact absolute root and rejects any path longer than 220 characters. It also requires a clean detached HEAD equal to current `origin/main`.

The root manifest freezes schema, attempt, Git/config/protocol/candidate identities, solver settings, full matrix, run-key maps, and instance/baseline/anchor identities. Each run has atomic `run.json` and `status.json`. Frontier state also has an identity-bound append-only `algorithm_checkpoint.json` and a post-evaluation `index.json` plus SHA-bound ordered chunks. `results.csv` and `summary.csv` are atomic projections of committed run records. Corrupt JSON, missing identity, mapping drift, completed-run repetition, chunk gaps, SHA mismatch, or cumulative-count mismatch fails closed.

Only strict `--resume` is supported for a future authorized run. `--overwrite` does not exist. A committed result, Gamma-specific baseline, checkpoint, scenario, cut, post-evaluation chunk, or CSV row is never repeated. A later authorization-only change supplies a Git-tracked JSON authorization file; no runner change is needed after this PR. This protocol does not authorize that run.

## 11. Attempt 2 identity-incident isolation

Attempt 1 stopped during the first Medium-large, seed 180, Gamma 0 frontier before master construction because the initial-upper-bound identity gate compared the canonical JSON instance payload SHA with a YAML configuration SHA of the same instance. Attempt 1 is frozen as `execution_incomplete`, `pipeline_identity_defect`, and `scientifically_usable=false`; none of its instance, baseline, anchor, checkpoint, run, or aggregate artifacts may be reused.

Attempt 2 uses execution attempt 2 and the isolated output roots `experiments/results_fh_gamma/ml_a2` and `experiments/results_fh_gamma/lg_a2`. Its canonical instance identity is the SHA256 of the strict canonical JSON serialization of `InventoryInstance.to_dict()`. The identity binds scale, seed, Gamma, execution attempt, Git commit, Config SHA, Protocol SHA, and canonical instance payload SHA. Exact archive-file SHA, when reported, is named separately and is never substituted for the canonical instance payload SHA. The same canonical identity must be present in the archive, manifest, baseline, anchor, frontier, checkpoint, and initial-upper-bound solver contract. Any field drift remains fail closed.

## 7. Dry-run and solver-limit envelopes

Dry-run validates the frozen matrix and identities entirely in memory. It must not create a formal output root, generate or load a reserved-seed instance, configure/import/call Gurobi, or write protocol evidence. It reports the longest planned Windows absolute path and type.

The algorithm solver-limit envelope is

\[
60\times1800=108000\text{ seconds}=30\text{ hours}.
\]

Complete post-evaluation covers 9,465 Medium-large scenarios and 23,775 Large scenarios, 33,240 total. Its solver-limit envelope is

\[
33240\times30=997200\text{ seconds}\approx277\text{ hours}.
\]

These are sums of per-task solver limits, not wall-time forecasts. Final Holdout experience suggests actual time is normally much smaller, but no certification step is shortened on that basis.

## 8. Statistical protocol and reporting semantics

The independent unit is the seed, not seed x Gamma. Each scale x Gamma contains five seeds. Gamma comparisons are paired within the same scale and seed; complete trajectories and paired changes 0 to 1 and 1 to 2 are reported. Descriptive summaries are mean, median, sample standard deviation, IQR, minimum, and maximum. Optional bootstrap resamples whole seed clusters. Any family of multiple tests uses Holm correction. With n=5, non-significance is not evidence of no effect; conclusions are structural sensitivity and managerial interpretation, not new population-wide claims. Sensitivity results are never pooled with Final Holdout and never used to retune the algorithm.

Primary metrics are baseline robust cost, cost budget, exact actual robust cost, actual price of fairness, objective T, inventory, opened warehouses, algorithm/master/separation runtime, post-evaluation runtime, total wall runtime, PAR-2, iterations, scenario-block count, and certified Farkas-cut count.

The three fill-rate fields are never interchangeable:

- `robust_minimum_fill_rate = 1 - T`: certified lower guarantee for every applicable region and exact uncertainty scenario.
- `wminfr`: exact post-evaluation minimum over scenarios and regions.
- `minimum_weighted_mean_fill_rate`: minimum across scenarios of the system demand-weighted mean fill rate.

Only the first is labelled the certified minimum regional fill-rate guarantee.

## 9. Authorization boundary

```yaml
formal_run_authorized: false
next_authorized_stage: fairness_hybrid_gamma_sensitivity_pre_run_audit_only
```

This delivery authorizes only solver-free dry-run, static validation, checkpoint/resume unit tests with synthetic temporary data, and a fresh read-only pre-run audit. It does not authorize instance generation for seeds 180--184, Gurobi configuration or execution, formal optimization, selective reruns, Gamma 3/4, new rho values, or any mutation of frozen Holdout/D1/D2 artifacts.

After the Attempt 2 repair is merged, the separately reviewed authorization-only file must use schema `fairness_hybrid_gamma_sensitivity_authorization_v2`, set `formal_run_authorized=true`, bind the exact config/protocol/candidate SHA values, unchanged matrix, execution attempt 2, isolated Attempt 2 output roots, and repair merge commit, and set `next_authorized_stage=fairness_hybrid_gamma_sensitivity_attempt2_formal_run_only`. It must state `previous_attempt_results_reused=false`. The runner requires that file to be Git tracked and the repair merge commit to be an ancestor of the clean detached current `origin/main`. All authorization, Git, seed-access, existing-output identity, and path gates run before production dependencies are imported or any output is created.
