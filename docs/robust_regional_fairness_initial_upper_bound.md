# Certified initial upper bound for robust regional fairness

## Theorem

Let `(x_base,y_base)` be a first-stage solution with a certified robust cost upper bound `C*`. Assume every demand and every shortage-cost coefficient is nonnegative, all first-stage and other recourse costs are nonnegative, `C*` is finite and nonnegative, and `rho >= 0`. For every scenario `d` in the frozen uncertainty set, baseline certification supplies a feasible scenario-specific recourse `(q^d,u^d,e^d)` whose total cost is at most `C*`. It does not require one recourse vector to work for every scenario.

Define, componentwise,

`u'[r,j] = min(u^d[r,j], d[r,j])`,

and retain `q^d` and `e^d`. Then `(x_base,y_base,T=1)` is robustly feasible for budget `B_rho=(1+rho)C*`.

## Proof

Nonnegativity gives `0 <= u' <= d`. If `u^d <= d`, the demand inequality is unchanged. If `u^d > d`, replacing it by `d` leaves `sum_i q[i,r,j] + u'[r,j] >= d[r,j]`. Supply constraints involve only `q`; reducing `u` relaxes every service constraint `sum_r u[r,j]-e[j] <= (1-service_level[j])sum_r d[r,j]`. The first-stage capacity, opening and linking constraints are unchanged.

The shortage coefficients are nonnegative, so truncation cannot increase recourse cost; all other cost terms are unchanged. Thus each transformed recourse costs at most `C*`, and nonnegative `C*` with `rho>=0` implies `C* <= B_rho`.

For every region with positive total demand, summing `u'[r,j] <= d[r,j]` gives `sum_j u'[r,j]/sum_j d[r,j] <= 1`, exactly the fairness constraint at `T=1`. Regions whose total demand is at or below the frozen fairness-metric tolerance are excluded by the frozen zero-demand policy. In the exact zero-demand case, nonnegative component demands also give `u'=0`. Service-violation variables and their penalties do not invalidate the construction: `e` is retained, its constraint is relaxed, and its cost is unchanged.

Therefore every scenario has a legal recourse for the common first-stage decision and `T=1`, proving robust feasibility and a valid objective upper bound of one. This is not a lower bound, does not prove optimality, and does not replace complete separation or its objective-bound certificate. QED.

## Fail-closed construction

Before using the bound, implementation must verify: certified baseline status, `valid_UB=true`, accepted optimality gap, finite nonnegative `C*`, exact anchor value/hex/SHA identity, `rho>=0`, nonnegative demand and all cost coefficients, unchanged uncertainty-set identity, and unchanged first-stage identity. It then stores the baseline certificate, scenario-independent transformation rule, proof version and hashes. Any missing or false item yields `initial_t1_ub_certification_failed`; the run must not use the bound or silently fall back to an empirical MIP start.

Pseudocode:

```text
verify_all_assumptions_or_fail_closed()
B = (1 + rho) * certified_baseline_upper_bound
incumbent = (x_base, y_base, T=1)
for each d: witness rule is u_prime=min(u_base(d), d), q_prime=q_base(d), e_prime=e_base(d)
record proof/version/baseline/anchor/instance/config/protocol/candidate identities
set valid_upper_bound=1
do not set or alter lower_bound
do not set scientific_status=certified_robust_optimal
```

Required evidence fields are `initial_t1_ub_proved`, `proof_version`, `proof_assumptions_checked`, `baseline_run_key`, `baseline_certificate_sha256`, `instance_sha256`, `anchor_value`, `anchor_value_hex`, `anchor_sha256`, `rho`, `budget_value`, `incumbent_t`, `transformation_rule`, `valid_upper_bound`, `lower_bound_source`, and `failure_status`.

Future S0 tests must include all prescribed positive cases and counterexamples for negative demand, negative shortage cost, uncertified baseline, invalid anchor, and a deliberately infeasible budget. They may be executed only after implementation authorization.
