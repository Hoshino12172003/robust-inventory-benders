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

Pattern identity uses schema `fairness_deviation_pattern_v1`. Formal component order is the Cartesian sequence `[(r,j) for r in instance.R for j in instance.J]`; `instance.R` and `instance.J` are the ordered model index sequences frozen by the instance, never sets, incidental dictionary order, solver pool order or filesystem order. Canonical IDs are Unicode NFC strings. Values are integer 0 or 1 and length is exactly `|R||J|`; otherwise fail closed. The payload contains `schema`, ordered `[region_id,product_id]` pairs and values. It is serialized by `json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")`, with no BOM or trailing newline. `pattern_sha256` is the uppercase hexadecimal SHA256 of those bytes.

Cut identity uses schema `fairness_farkas_cut_v1`. Formal variable IDs are ordered as `y[i]` over `instance.I`, `x[i,j]` over `instance.I` then `instance.J`, followed by `T`; these IDs do not depend on Gurobi creation order or object identity. The payload includes schema, sense, every canonical variable ID and coefficient, constant, RHS, and Farkas normalization identity `nonnegative_multipliers_sum_to_one_v1`. Every numeric value is finite IEEE-754 binary64, `-0.0` is replaced by `+0.0`, and Python `float.hex()` is used; NaN and infinities fail closed. The same canonical JSON rule produces bytes and `cut_sha256` is uppercase hexadecimal SHA256. A cut hash is exact identity only; approximate redundancy never changes or rounds a hash.

Represent each certified cut in canonical `(y,x,T)` order as `g(z)>=0`. Raw violation is `max(0,-g(z_current))`; normalized violation is raw violation divided by `max(1,||a||_1)`. Direction similarity is cosine similarity of L2-normalized coefficient vectors; diversity is `1-max_similarity_to_selected`. Zero norms and nonfinite inputs fail closed.

Every selection metric is first converted to an integer bucket using exact `Decimal.from_float(binary64)/quantum`, precision 80 and `ROUND_HALF_EVEN`. Raw acceptance tolerance is `1e-7`; raw, normalized, relative-violation, cosine and diversity quanta are all `1e-9`, two orders below the scientific feasibility tolerance. The deadband is two buckets and affects selection only, never mathematical feasibility or final certification. Raw violation is accepted only strictly above tolerance plus deadband. Relative normalized violation is eligible only strictly above `0.10` plus deadband. Cosine values at or above `0.98` minus deadband are conservatively redundant. Thus values inside either deadband cannot gain priority or add a numerically ambiguous cut. Signed zero is one bucket and nonfinite values fail closed.

Exact hashes are deduplicated first. Primary-cut diversity is defined as one. Select the primary cut by quantized normalized violation descending, quantized diversity descending, quantized raw violation descending, pattern SHA ascending, then cut SHA ascending. Recompute quantized diversity after each selected cut and apply the same total order for every secondary choice. Creation order, object addresses and solver pool order never break ties.

The batch schedule depends only on checkpointed discrete state. Final certification uses one cut. Otherwise total certified master cuts below 1,000 use at most five; 1,000--2,999 use three; 3,000--4,999 use two; 5,000 or more use one. These bands are frozen from the S1 evidence that fixed batch-5 produced excessive master growth. They cannot be changed from L0, L1, M1 or holdout outcomes. Wall-clock, solver runtime and CPU time remain reporting fields only: machine, operating-system, scheduling and solver noise make them invalid drivers of a cross-machine and resume-deterministic scientific path.

No cut deletion, retirement, deactivation or replacement is allowed. Adaptation controls only new rows. Checkpoints freeze iteration, total certified cuts, schedule segment, quantized LB-improvement state, stall counter, pattern and cut hashes, complete candidate ordering, duplicate/redundancy decisions, and pattern/cut/quantization schema versions. LB improvement is `max(0,current_LB-previous_LB)` bucketed at `1e-9`; the stall counter counts consecutive zero buckets. Both are reporting-only under schedule v1 and cannot alter batch size or cut selection. Runtime history is not reconstructed into scientific state. After resume, identical checkpoint input must produce the identical batch size, ordering, selected hashes and next state.

Metadata records canonicalization schema versions, all quanta/deadbands, schedule, cumulative candidates, cache hits and recertifications, certified cuts, selected cuts, rejection reasons and active cuts. Each iteration records candidate/pattern/cut hashes, binary64 inputs and integer buckets for raw/normalized/relative violation, similarity and diversity, complete ordering keys, schedule segment, budget and trigger, selected count, active-cut count before/after, complete-separation status and objective bound, and initial-UB evidence identity. Master/separation/wall runtimes are reported separately and never appear in a selection key or branch predicate.

## Gates and matrices

S0 is design-only in this PR. It covers single region, symmetric regions, clear disparity, rho 0/0.01, Gamma 0/2, zero-demand policy, infeasible cost budget, valid baseline+T=1 UB, invalid-UB counterexamples, cache recertification, multiple certified cuts, ties, near-parallel cuts, budget 5-to-1 transition, and Benders/extensive-form agreement in T, first stage and certification. No mathematical solver test is run here.

L0 is Large seed 160, rho 0, one baseline plus one frontier (2 tasks). It requires implementation and pre-run audit approval. Pass requires `certified_robust_optimal`, no implementation error, no invalid post-evaluation, valid post-evaluation over 4,657 scenarios, valid UB and LB, within 1,800 algorithm seconds. Failure freezes `stop_final_large_remediation` and prohibits additional Large runs.

L1 is cumulative Large seeds 160--162 by rho 0/0.01: three baselines and six frontier tasks (9 total). The two L0 scientific identities are an exact subset: their stable run keys omit stage labels, L1 references their immutable evidence under identity lock, and only seven additional tasks execute. It opens only after L0 passes and requires at least 4/6 certified, zero implementation errors, zero invalid post-evaluations, and every successful post-evaluation valid. Failure stops before M1 or holdout.

M1 is Medium-large seeds 160--162 by rho 0/0.01: three baselines and six frontier tasks (9 total). It opens only after L1 reaches 4/6. The maximum cross-scale development matrix is therefore six baselines plus twelve frontier tasks, 18 total. Old candidates are never rerun.

## Identity, isolation and timing

Future configs use schema 3 and execution attempt 3, with complete prior-attempt history, `previous_attempt_results_reused=false`, stable canonical run keys, `r_` plus the first 24 SHA256 hex digits as physical directory, 220-character path preflight including `.tmp`, atomic checkpoints, post-evaluation chunks, resume and aggregation recovery. Identity locks Git, byte/canonical config hashes, protocol hash, candidate hash, pattern/cut canonicalization schemas, float encoding, quantization/deadband parameters, batch schedule, solver parameters, instance, baseline and anchor. `--overwrite` is forbidden and identity mismatch fails closed.

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
