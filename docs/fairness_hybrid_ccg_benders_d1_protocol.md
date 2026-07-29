# Certified Hybrid Scenario–Benders Fairness D1 protocol

## Frozen scope and prior node

The sole candidate is `certified_hybrid_scenario_benders_fairness`. D1 contains one Large baseline and one frontier run at seed 160 and rho 0. No other seed, rho, stage, prior output, or holdout is authorized. The original Attempt 5 archive remains immutable at SHA256 `09B41862A5BFED724EDBEC1E64996B54AA878119F5C0DEDFE5B10126B2525A98`; its frontier is `time_limit_uncertified` with LB 0, UB 1, gap 1 after 95 iterations and 121 cuts. This is evidence of a weak lower bound, not a certified fairness result.

## Literature boundary

The design uses four published ideas only as motivation, not as copied algorithms:

1. Goerigk et al., [A fast approximate column-and-constraint generation method for two-stage robust mixed-integer programs](https://doi.org/10.1016/j.ejor.2026.05.038), motivates scenario addition and legitimate second-stage bounds. D1 does not adopt approximate termination, adaptive time limits, or gap propagation.
2. Zhang et al., [Two-Stage Adaptive Robust Hub Location Problem Under Demand Uncertainty](https://doi.org/10.1002/nav.70043), motivates scenario-wise adjustable recourse for demand uncertainty. D1 retains this repository's inventory and regional-shortage recourse exactly.
3. Lefebvre, Schmidt and Thürauf, [Column generation in column-and-constraint generation for adjustable robust optimization](https://doi.org/10.1007/s12532-025-00300-3), motivates strengthening an adjustable robust master by explicit recourse columns/constraints. D1 adds complete scenario blocks rather than an approximate surrogate.
4. Glomb, Liers and Rösel, [A novel Pareto-optimal cut selection strategy for Benders Decomposition](https://doi.org/10.1007/s12532-025-00291-1), motivates deterministic selection of useful cuts. D1 does not implement their Pareto rule; it uses one certified maximum-normalized-violation scenario with SHA tie-breaking.

## Scenario-augmented master

Let `D_k` be an ordered set of uncertainty scenarios. The master keeps the original first-stage `y_i`, `x_ij`, first-stage budget, capacity, activation logic, and `T in [0,1]`. For every `s in D_k` it adds independent `q^s_irj`, `u^s_rj`, and `e^s_j` and exactly the production recourse constraints:

- `sum_i q^s_irj + u^s_rj >= d^s_rj`;
- `sum_r q^s_irj <= x_ij`;
- `sum_r u^s_rj - e^s_j <= (1-service_level_j) sum_r d^s_rj`;
- production transport, shortage, and service-violation costs;
- first-stage cost plus scenario recourse cost at most `B_rho`;
- for every region with demand above the frozen metric tolerance, `sum_j u^s_rj <= T sum_j d^s_rj`.

Zero-demand regions remain not applicable. Warehouse capacity and activation remain first-stage constraints. A committed scenario block is never deleted or replaced by a Farkas cut.

### Lower-bound theorem

Every robust-feasible first-stage solution has a legal recourse policy for every uncertainty scenario, hence in particular for all scenarios in `D_k`. Restricting the universal scenario requirement to `D_k` therefore enlarges the feasible set. Minimizing `T` over that relaxation gives a lower bound on the complete robust optimum. If `D_k` is contained in `D_(k+1)`, the latter feasible set is a subset of the former, so its optimal value and its valid solver best bound cannot decrease. The implementation updates the global LB by `max(previous_LB, master.ObjBound)` and never substitutes the master incumbent objective for the bound.

The scenario block calls the same production recourse-expression builder used by the extensive form and fixed-scenario policy model. No recourse row is added, removed, or relaxed.

## Initial scenarios

The canonical component order is `[(r,j) for r in instance.R for j in instance.J]`. The initial ordered set contains the nominal scenario, then one regional stress scenario per region. A regional stress scenario activates at most Gamma components in that region, ordered by descending `demand_deviation[r][j]`, then ascending canonical `(r,j)`; the stored active units themselves use canonical order. Scenario identity is compact canonical JSON (`sort_keys=True`, separators `(',',':')`, UTF-8, no trailing newline, `allow_nan=False`) over schema `fairness_hybrid_scenario_v1`, instance SHA, canonical component order, binary activity values, and binary64 demand values encoded with `float.hex()`. SHA256 is uppercase. Invalid dimensions, nonfinite demand, nonbinary activity, or duplicate SHA fail closed.

Attempt 5 scenarios are not initialization inputs. Dry-run records the symbolic initial-plan SHA without generating seed 160; formal execution records actual scenario SHA values after the new instance is created.

## Separation and deterministic scenario addition

Persistent separation, pattern-only cache, and solution-pool incumbents may propose scenarios. Every proposal must be re-certified at the current `(x,T,B_rho)` by production fixed-scenario primal/Farkas certification. Old rays, cuts, violations, and certifications are never reused. Each accepted Farkas cut remains independently certified.

At an intermediate iteration, all current-point certified strictly violating candidates are canonicalized and deduplicated by scenario SHA. D1 commits at most one new scenario: descending normalized violation, then ascending scenario SHA, then ascending canonical cut SHA. It may add that scenario's certified Farkas cut too. It never uses runtime, adaptive batch, or deletion.

An intermediate heuristic candidate advances the relaxation but cannot certify robust feasibility. Final termination always invokes complete exact separation. Empty cache/pool, no new scenario, duplicate scenario, or zero candidates cannot certify feasibility. Only a legal complete-separation objective bound proving no violation sets `robust_feasibility_certified=true`.

## Bounds and success

The certified baseline and the frozen T=1 theorem provide the initial UB. A lower T updates UB only after complete exact robust-feasibility certification. LB is the monotone maximum of valid scenario-master solver best bounds. Gap is `(UB-LB)/max(1,abs(UB))`.

`certified_robust_optimal` requires simultaneously: a valid master bound; frozen global gap tolerance; complete exact-separation objective-bound certification; and a valid full 4,657-scenario post-evaluation. Time limit, incumbent existence, duplicate scenario, missing post-evaluation, identity mismatch, checkpoint damage, or implementation failure is never solved.

## Atomic state and resume

The checkpoint identity locks Git, config, protocol, candidate, solver, instance, baseline, anchor, seed, scale, and rho. State contains the ordered scenario SHA list, canonical scenario payloads, committed cut SHA/payloads, LB, UB, gap, master bound, incumbent values plus their canonical SHA, iteration, and final-certification state. Scenario commit is atomic. A pre-commit interruption replays the same pending SHA against a freshly rebuilt master; a post-commit resume reconstructs the ordered committed blocks once. Duplicate, omission, order drift, partial state, corrupt JSON, or identity drift fails closed.

## D1 authorization

Only the checked-in D1 configuration may execute formally after merge and independent review. It expands to Large seed 160, rho 0, one baseline and one frontier. It uses Threads 1, solver Seed 0, FeasibilityTol `1e-7`, Gamma 2, 1,800 seconds for baseline and algorithm, 30 seconds per post-evaluation scenario, and PAR-2 multiplier 2. `--overwrite` is unsupported. The new output directory must not exist on first execution; `--resume` is required for both initialization and recovery. Seeds 161–179, Medium-large, D2, L1/M1, S2, full-grid, and old Attempt resume are rejected.

## Statistical unit boundary

D1 contains one development seed and supports no inferential claim. Any later pre-registered paired comparison must treat the seed, never a seed-rho task, as the independent unit. A cluster bootstrap resamples seeds and keeps every rho and both methods for a sampled seed in the same cluster. A comparison at one rho has one pair per seed. If five rho-specific hypotheses are tested, their p-values use Holm correction. An overall comparison first aggregates rho values within each seed or uses the same seed-cluster bootstrap. Fifty seed-rho tasks must never be represented as fifty independent observations.
