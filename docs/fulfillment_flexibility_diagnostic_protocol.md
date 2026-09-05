# Multi-warehouse fulfillment flexibility diagnostic protocol

Status: development diagnostic, preregistered before optimization.

## Scope and non-interference

This experiment screens whether adaptive multi-warehouse fulfillment has a
large enough managerial effect to justify a later formal experiment. It does
not modify the frozen uncertainty set, Gamma, V3 parameters, Benders or CCG
termination logic, any formal configuration, or any existing result.

The implementation is additive. Outputs are written only under
`experiments/results_fulfillment_flexibility/development`.

## Existing recourse semantics

The current recourse shipment variable is `q[i,r,j]`. In the existing model,
the demand constraint sums `q[i,r,j]` over every warehouse `i`; consequently,
one region can receive the same product from several warehouses in one demand
scenario. The warehouse-product availability constraint sums shipments over
regions and caps them by first-stage inventory `x[i,j]`.

## Eligibility modes

For each warehouse-region pair, the eligibility score is the arithmetic mean
of unit transportation cost over products. Ties are resolved by warehouse
index. The score is a transportation-cost measure, not physical distance.

- `full`: retain every original shipment arc;
- `k2`: retain the two lowest-transport-cost eligible warehouses per region;
- `k1`: retain the one lowest-transport-cost eligible warehouse per region.

For an ineligible pair `(i,r)`, `q[i,r,j] = 0` for every product and scenario.

Candidate eligibility is fixed before optimization. No warehouse is forced
open. Therefore the original first-stage feasible region is unchanged. If an
eligible warehouse is closed, the corresponding shipment remains unavailable.
Because shortage `u` and service-violation `e` variables provide relatively
complete recourse, this does not make the model mathematically infeasible; it
can increase shortage and penalty cost. Adding an eligible-opening constraint
would change the first-stage feasible region and is deliberately rejected.

## Development design

- scales: `small` and `medium`;
- seeds per scale: `190, 191, 192`;
- Gamma: `2`;
- rho: `0.025`;
- modes: `k1`, `k2`, `full`;
- exact finite scenario enumeration;
- one solver thread, solver seed zero, final MIP gap `1e-4`;
- maximum 300 seconds per optimization.

Seeds 190--192 are new development-only seeds. Existing repository evidence
uses seeds through 189; the new seeds are not validation or holdout units.
Small and medium scales are used because this screening requires several exact
extensive-form policy and fixed-first-stage solves. Larger scales are deferred
unless the screening recommendation is `proceed_to_formal_flexibility_experiment`.

For each scale-seed-mode cell, the mode first obtains its own robust cost
anchor. The re-optimized service model then minimizes worst-region shortage at
the frozen allowance `(1 + rho) * anchor`.

The fixed-first-stage diagnostic takes the `full` re-optimized warehouse
opening and inventory allocation and fixes them. It first calculates the
minimum robust cost of that fixed configuration under `k1`, `k2`, and `full`.
The three recourse restrictions are then compared under the common budget
`(1 + rho)` times the largest of those fixed-configuration cost anchors. This
prevents a restricted mode from being classified as service-infeasible merely
because the original full-mode cost cap is too small, while holding warehouse
opening and inventory exactly constant.

## Outcomes

The run records robust cost or certified upper bound, worst-region shortage,
minimum regional fill rate, the existing demand-weighted fill-rate definition,
transportation, inventory, opening, shortage and service-violation costs,
opened warehouses, active warehouse-region arcs, used warehouses by region,
certification, runtime, objective and bound consistency, and first-stage
configuration differences.

Paired comparisons are `k1` versus `k2`, `k2` versus `full`, and `k1` versus
`full`. Relative shortage reduction is `(T_k1 - T_K) / T_k1`; when `T_k1` is
zero, only the absolute difference is reported. Relative cost change is
`(C_K - C_k1) / C_k1`.

## Frozen screening rule

At the seed level, `k1` to `full` improvement above five percent on the pooled
median, with positive absolute improvement in a majority of seeds, supports
`proceed_to_formal_flexibility_experiment`. Improvements generally below one
percent, or mixed directions without a material absolute effect, support
`do_not_proceed`. Values between one and five percent or incomplete
certification produce `inconclusive`. Thresholds will not be changed after
observing results.

Development results are not paper-level statistical conclusions.
