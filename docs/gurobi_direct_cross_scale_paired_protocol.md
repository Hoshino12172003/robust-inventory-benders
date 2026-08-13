# Gurobi direct cross-scale paired benchmark

This benchmark adds only ten direct deterministic-equivalent runs. It reuses,
read-only, the exact instances, cost baselines, anchors, and certified Hybrid
results from Gamma Attempt 3: medium-large and large, seeds 180--184,
Gamma 2, and rho 0.025. Hybrid and baseline runs are not repeated.

The direct formulation explicitly adds every one of 1,831 or 4,657 scenarios,
uses Threads 1, solver Seed 0, FeasibilityTol 1e-7, MIPGap 0, and disables
Gurobi built-in Benders. The 1,800-second algorithm clock includes model build
and optimization. An uncertified cell receives PAR-2 of 3,600 seconds. A direct
result is certified only after an exact complete model solve and full independent
post-evaluation. Results support statements about this formulation and paired
test matrix only, not claims that the Hybrid algorithm is generally better than
the Gurobi solver used internally by both methods.
