# Final Large fairness remediation protocol

Status: protocol only; no formal run is authorized. Base: `d6ae40c0bbe3d9af7c67cb7e71fb8ad45e64b033`.

## Frozen decision and scope

The cross-scale S1 decision and all artifacts under `analysis/fairness_scalability_s1_attempt2_cross_scale_freeze/` remain immutable: no existing candidate passes cross-scale S1, Medium-large certified 20/24, Large certified 0/24, no candidate is selected, and original S2, full-grid and Attempt 4 are unauthorized. Old results are development evidence only and are not validation evidence for this algorithm.

The paper-level hierarchy is frozen to Cost-Benders (V3 for `C*`), baseline single-cut Fair-Benders, and one new acceleration: `certified_adaptive_multicut_fair_benders`. Persistent separation, certified cache and fixed batch-5 remain S1 ablations, not new peer candidates. Reporting, paths, checkpoints and attempt history are not algorithmic contributions.

The new candidate may add only: the proved certified `T=1` initial upper bound; persistent separation with pattern-only cache; and deterministic adaptive selection of independently certified cuts. It may not change the fairness model, recourse, cost budget, uncertainty set, Farkas cone, fixed-scenario certification, objective-bound requirement, rho, anchor source, success state, `src/benders.py`, or frozen V3 candidate. Production implementation is explicitly deferred.

## Initial upper bound

The theorem, assumptions, proof, construction, evidence and fail-closed behavior are frozen in `docs/robust_regional_fairness_initial_upper_bound.md`. The result is `initial_T1_UB_proved: true` under its explicitly checked assumptions. The incumbent supplies only `UB=1`; it never supplies a lower bound or certification status.

## Certified adaptive multi-cut rule

At the current master point, cache entries contain deviation patterns only. Every cached or pool-proposed scenario is rerun through current-point fixed-scenario primal/Farkas certification. No old ray, cut, violation value or certification is reused. Every master cut has its own current-point certificate. Only a complete separation solve with a legal objective bound may certify absence of a violating scenario.

Represent each certified cut in canonical `(y,x,T)` coefficient order as `g(z)>=0`. Its raw violation is `max(0,-g(z_current))`; normalized violation is raw violation divided by `max(1, ||a||_1)`, where `a` is the nonconstant coefficient vector. Direction similarity is cosine similarity of L2-normalized `a` vectors. Zero-norm or nonfinite vectors fail closed.

Deduplicate exact pattern and cut hashes. Sort by normalized violation descending, raw violation descending, pattern SHA ascending, then cut SHA ascending. Always select the first valid violated cut. A secondary cut is eligible only if its normalized violation is at least 10% of the first cut and its cosine similarity to every selected cut is at most 0.98.

The per-iteration budget is determined solely from state known before selection. Define master share as previous master runtime divided by `max(1e-9, master+separation runtime)`. Define growth as previous master runtime divided by the median of up to three preceding positive master runtimes; missing history gives 1.0. Use the first matching rule: budget 1 during final certification, or at 5,000 active cuts, 0.80 share, or 2.00 growth; budget 2 at 2,500 cuts, 0.60 share, or 1.50 growth; budget 3 at 1,000 cuts or 0.40 share; otherwise budget 5. Thus at least one and at most five cuts are added when a certified violation exists, but diversity filtering may return fewer than the budget. The bands are frozen before outcomes and address the two observed S1 burdens: separation dominance and master growth. They must not be tuned using final success rates.

No cut deletion, retirement, deactivation or replacement is allowed. Adaptation controls only new rows. Resume restores canonical hashes, active rows, timing history and budget state exactly; replay must select the same cuts.

Metadata records cumulative candidates, cache hits and recertifications, certified cuts, selected cuts, rejection reasons and active cuts. Each iteration records candidate/pattern/cut hashes, raw and normalized violations, pairwise similarities, ordering keys, budget and trigger, selected count, active-cut count before/after, master/separation runtimes, master share/growth, complete-separation status and objective bound, and initial-UB evidence identity.

## Gates and matrices

S0 is design-only in this PR. It covers single region, symmetric regions, clear disparity, rho 0/0.01, Gamma 0/2, zero-demand policy, infeasible cost budget, valid baseline+T=1 UB, invalid-UB counterexamples, cache recertification, multiple certified cuts, ties, near-parallel cuts, budget 5-to-1 transition, and Benders/extensive-form agreement in T, first stage and certification. No mathematical solver test is run here.

L0 is Large seed 160, rho 0, one baseline plus one frontier (2 tasks). It requires implementation and pre-run audit approval. Pass requires `certified_robust_optimal`, no implementation error, no invalid post-evaluation, valid post-evaluation over 4,657 scenarios, valid UB and LB, within 1,800 algorithm seconds. Failure freezes `stop_final_large_remediation` and prohibits additional Large runs.

L1 is cumulative Large seeds 160--162 by rho 0/0.01: three baselines and six frontier tasks (9 total). The two L0 scientific identities are an exact subset: their stable run keys omit stage labels, L1 references their immutable evidence under identity lock, and only seven additional tasks execute. It opens only after L0 passes and requires at least 4/6 certified, zero implementation errors, zero invalid post-evaluations, and every successful post-evaluation valid. Failure stops before M1 or holdout.

M1 is Medium-large seeds 160--162 by rho 0/0.01: three baselines and six frontier tasks (9 total). It opens only after L1 reaches 4/6. The maximum cross-scale development matrix is therefore six baselines plus twelve frontier tasks, 18 total. Old candidates are never rerun.

## Identity, isolation and timing

Future configs use schema 3 and execution attempt 3, with complete prior-attempt history, `previous_attempt_results_reused=false`, stable canonical run keys, `r_` plus the first 24 SHA256 hex digits as physical directory, 220-character path preflight including `.tmp`, atomic checkpoints, post-evaluation chunks, resume and aggregation recovery. Identity locks Git, byte/canonical config hashes, protocol hash, candidate hash, solver parameters, instance, baseline and anchor. `--overwrite` is forbidden and identity mismatch fails closed.

The three future output directories are unique under `experiments/results_fairness_large_final_remediation/`; they must be absent before and after this protocol dry-run. Algorithm/baseline/general limits are 1,800 seconds. Post-evaluation is 30 seconds per scenario in chunks of 25. PAR-2 is twice algorithm runtime and total wall time separately accounts for algorithm, post-evaluation, aggregation and checkpoint I/O.

## Holdout

Seeds 130--159 remain prohibited. Seeds 170--179 are reserved, ungenerated and unsolved. The repository-history and frozen-evidence audit found no seed-access record for this range; because external result directories are deliberately not accessed in this protocol turn, a future pre-holdout audit must independently reconfirm all sanctioned result catalogs. Any record fails closed; no silent replacement is allowed. Only after algorithm, parameters, thresholds and code freeze may a separate authorization expose ten seeds by two rho values (20 frontier points), with a preregistered 16/20 certification threshold. This PR authorizes none of them.

## Decision

```yaml
initial_T1_UB_proved: true
decision: approve_for_implementation
next_authorized_stage: fairness_large_final_remediation_implementation_only
formal_run_authorized: false
```
