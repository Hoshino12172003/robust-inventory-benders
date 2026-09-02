# Multi-warehouse coordination value experiment

Status: preregistered implementation protocol; formal optimization is not yet authorized.

## Research question

This isolated experiment quantifies the operational value of allowing an online
retail network to fulfill a region's realized demand from more than one open
warehouse. It does not modify, rerun, overwrite, or reinterpret any frozen
algorithm-development or holdout result.

The paired policies are:

1. `flexible_multiwarehouse`: every open warehouse with available inventory may
   serve every region after demand is observed;
2. `optimized_single_source`: before demand is observed, each region is assigned
   to exactly one open warehouse, and all scenario-specific shipments to that
   region must originate from the assigned warehouse.

The single-source assignment is optimized jointly with warehouse activation and
inventory positioning. It is not a manually disadvantaged nearest-warehouse
rule.

## Mathematical comparison

Both policies use the same warehouse, product, region, cost, capacity, demand,
and finite uncertainty-scenario inputs. Both first minimize robust total cost.
Let their certified cost anchors be `C_flex` and `C_single`.

The service-protection comparison uses the common absolute budget

`B(rho) = (1 + rho) * C_single`.

Under this common budget, each policy minimizes the maximum regional shortage
rate over all included scenarios. The primary allowance is `rho = 0.025`.
The preregistered sensitivity grid is `0, 0.01, 0.025, 0.05, 0.10`.

Using `C_single` as the common reference makes the comparison interpretable:
both policies receive the same monetary ceiling, while the flexible policy may
use coordination instead of additional facilities or inventory. Because the
flexible feasible set contains the single-source feasible set, a certified
flexible solution must not have a worse objective at a common budget.

## Data and statistical unit

The statistical unit is one Olist test week. All 17 previously prepared Olist
test weeks are included: the 12 validation-case weeks and the five sealed
confirmation weeks. Their existing instance and scenario files are read-only
inputs. File hashes and paths are frozen before any formal optimization.

Olist supplies observed transactions, regional demand, product grouping, and
geographic freight information. Candidate warehouses and economic parameters
remain calibrated modeling inputs and must not be represented as observed Olist
warehouse decisions.

## Outcomes

The primary paired outcome is the flexible-minus-single-source difference in
worst regional shortage rate at `rho = 0.025`.

Secondary outcomes are:

- robust total cost and its facility, inventory, transportation, shortage, and
  service-violation components;
- demand-weighted average shortage rate;
- number of region-scenario observations whose shortage rate exceeds 10%;
- number of regions whose worst shortage rate exceeds 10%;
- nonlocal fulfillment share, relative to each region's lowest average
  transportation-cost warehouse;
- mean and maximum number of warehouses actively serving a region in a
  scenario;
- total inventory, number of open facilities, opening pattern, and optimized
  single-source assignment.

Every week and every direction of effect is retained. No case may be removed
because its coordination benefit is small, zero, or operationally costly.

## Claim boundary

The experiment may support a claim that coordinated fulfillment has material
operational value only if the primary results are certified and the estimated
effect is reported with its full week-level distribution. It cannot establish
that multi-warehouse pooling is novel, that the Hybrid algorithm universally
outperforms CCG, or that Olist operated the calibrated warehouse network.

The formal report must distinguish mathematical weak dominance from empirical
materiality. A nonpositive flexible-minus-single shortage difference follows
from feasible-set containment; the magnitude, cost composition, and operating
conditions are empirical findings.

## Reproducibility and execution gate

All files are additive under `coordination_value_v1`. Existing repository files
and prior result directories remain unchanged. Formal execution requires:

1. a committed input catalog containing SHA-256 identities for all 34 source
   files;
2. a clean reviewed Git commit containing this protocol, implementation, and
   correctness tests;
3. an authorization file created after that commit is merged;
4. a previously absent formal output directory.

The implementation supports no overwrite option. Tiny synthetic correctness
tests and a solver-free dry run are allowed before formal authorization.
