# Formal multi-warehouse fulfillment-flexibility protocol

Status: preregistered protocol implementation; no formal optimization has been
authorized or executed.

> **FORMAL OPTIMIZATION PROHIBITED UNTIL PROTOCOL REVIEW AND AUTHORIZATION.**

## 1. Scientific question and evidence boundary

The formal experiment asks:

> How much cross-warehouse fulfillment flexibility is valuable under spatially
> uncertain demand, and how much of the potential pooling benefit can be
> captured with only a limited number of eligible fulfillment sources?

All modes use the same multi-warehouse network and the same two-stage model.
The experiment does not compare a multi-warehouse system with a single-
warehouse system. At the aggregate region-product level, `q[i,r,j](d)` is the
quantity of product `j` shipped from warehouse `i` to aggregate demand in
region `r` under scenario `d`. It does not represent splitting one customer's
parcel. A region contains many orders whose aggregate demand may be served by
different warehouses.

The operational mechanism is:

1. warehouse openings and inventory are selected before demand is observed;
2. spatial demand shocks can misalign positioned inventory and realized demand;
3. after demand is observed, access to alternative warehouses enlarges the
   admissible second-stage shipment set;
4. broader inventory access may mitigate localized shortage, at a physical
   fulfillment cost.

Synthetic formal evidence is a controlled mechanism confirmation. Olist remains
external managerial validation and M5 remains separate external computational
or model-robustness evidence. Neither external study may tune this protocol.

## 2. Frozen model boundary

This experiment is additive. It does not modify the original robust model,
first-stage semantics, uncertainty set, Gamma semantics, shortage or service
variables, Benders/CCG core, Hybrid v8, V3 parameters, termination or
certification logic, formal configurations, previous results, Olist, M5, or PR
#79 outputs. It adds no assignment binary, opening restriction, forced opening,
transshipment variable, order-level route, split-shipment binary, or physical-
distance assumption.

The original model is already two-stage:

- Stage 1: warehouse opening `y` and inventory positioning `x` before demand;
- Stage 2: adaptive aggregate shipment `q`, shortage `u`, and service slack `e`
  after demand.

The formal experiment changes only the size of the admissible Stage-2 shipment
set. It does not create the two-stage structure.

## 3. Fulfillment-flexibility parameter

For region `r`, define the arithmetic mean transportation-cost score

$$
\bar c_{ir}=\frac{1}{|J|}\sum_{j\in J}c_{irj}.
$$

Ties are resolved by warehouse index. Let `N_r^K` contain the `K` candidate
warehouses with the lowest values of `bar c_ir`. Eligibility is fixed before
optimization:

$$
q_{irj}(d)=0,
\qquad
i\notin\mathcal N_r^K,
\quad j\in J,
\quad d\in\mathcal U(\Gamma).
$$

The modes are:

- `k1`: restricted single-source eligibility benchmark;
- `k2`: limited pooling or limited fulfillment flexibility;
- `full`: full fulfillment flexibility; all original arcs remain available.

`K` is an experimental recourse-flexibility parameter, not a new core decision
variable and not the number of open warehouses. An eligible warehouse need not
open. Shortage variables retain mathematical recourse feasibility. The
first-stage feasible region is unchanged. “Lowest transportation cost” must never be rewritten as “nearest warehouse.”

## 4. Formal sample and frozen parameters

The primary sample contains twenty formal synthetic instances:

| Scale | Dimensions `(I,J,R)` | Seeds | Instances | Exact scenarios per instance |
|---|---:|---|---:|---:|
| medium_large | `(6,6,10)` | 230--239 | 10 | 1,831 |
| large | `(8,8,12)` | 230--239 | 10 | 4,657 |

The same seed labels across scales permit scale-specific and pooled reporting;
each scale-seed instance remains the paired experimental unit. Seeds were
frozen after the repository and archived-metadata audit and before generation
of any formal instance or result.

Main parameters are frozen at:

$$
\Gamma=2,
\qquad
\rho=0.025.
$$

There is no `K × Gamma × rho × scale` factorial and no tuning from formal or
external results.

### Preregistered operational fallback

The preferred sample is 10+10. Before any formal solver access, a protocol
amendment may activate the already frozen fallback seeds 230--234, giving 5+5,
only if the hardware/runtime gate in both configs fails. No different seeds or
post-result sample-size change is permitted. The fallback reduces the total
runtime envelope but not the per-instance memory of the large formulation;
therefore the hardware gate or a separately reviewed scalable exact backend is
still required. No fallback has been activated in this protocol PR.

## 5. Analysis A: re-optimized total system value

For every scale-seed-mode cell, solve the mode-specific robust cost anchor

$$
C_{\mathrm{anchor},K},
$$

set

$$
B_K=(1+\rho)C_{\mathrm{anchor},K},
$$

and minimize the existing worst-region shortage objective under the mode-
specific allowance. Warehouse opening, inventory positioning, and recourse all
re-optimize. The observed difference contains:

$$
\text{network configuration adaptation}
+\text{inventory positioning adaptation}
+\text{recourse flexibility}.
$$

Because the modes have different endogenous anchors and budgets, re-optimized
`T_k1 >= T_k2 >= T_full` is not required. Every non-monotone pair will be
reported without correction or post hoc explanation.

## 6. Analysis B: fixed-first-stage mechanism isolation

For each scale-seed instance, take the re-optimized full-mode solution
`(y_full*, x_full*)` and fix it exactly for all modes. Compute the three fixed-
configuration robust anchors and then define one common anchor and budget:

$$
C_{\mathrm{fixed,common}}
=
\max\{C_{\mathrm{fixed},k1},C_{\mathrm{fixed},k2},C_{\mathrm{fixed,full}}\},
$$

$$
B_{\mathrm{common}}
=
(1+\rho)C_{\mathrm{fixed,common}}.
$$

The three service models therefore share openings, inventory, scenarios,
Gamma, and cost allowance. Only shipment eligibility changes. This identifies
the incremental value of the size of the second-stage recourse set; it is not a
test of whether two-stage optimization works.

## 7. Primary outcome and marginal flexibility

The audited primary outcome remains

$$
T=
\max_{d\in\mathcal U(\Gamma)}
\max_{r\in R}
\frac{\sum_j u_{rj}(d)}{\sum_j d_{rj}}.
$$

Lower `T` is better. Minimum regional fill, demand-weighted fill, mean regional
shortage, and total shortage are secondary and cannot replace `T`.

For each instance and separately for re-optimized and fixed-first-stage
analyses, report

$$
\Delta_{12}=T_{k1}-T_{k2},
\qquad
\Delta_{2F}=T_{k2}-T_{full},
\qquad
\Delta_{1F}=T_{k1}-T_{full},
$$

$$
R_{12}=\frac{T_{k1}-T_{k2}}{T_{k1}},
\qquad
R_{1F}=\frac{T_{k1}-T_{full}}{T_{k1}},
$$

when `T_k1 > 0`, and

$$
\mathrm{Capture}_{k2}
=
\frac{T_{k1}-T_{k2}}{T_{k1}-T_{full}}
$$

when the denominator is nonzero. Capture is never truncated to `[0,1]`.
Values outside that interval, especially under endogenous re-optimized budgets,
remain in the output.

Diminishing returns are evaluated by the paired contrast
`Delta_12 > Delta_2F`. The report includes distributions, scale-specific and
pooled medians, counts, and exceptions. No conclusion about diminishing returns
is preregistered.

## 8. Cost and mechanism diagnostics

Robust cost is decomposed as

$$
C_{\mathrm{physical}}
=C_{\mathrm{opening}}+C_{\mathrm{inventory}}+C_{\mathrm{transport}},
$$

$$
C_{\mathrm{failure}}
=C_{\mathrm{shortage}}+C_{\mathrm{service\ violation}}.
$$

Robust total cost is also retained. The protocol tests whether flexibility
changes physical expenditure, failure expenditure, and total cost. It does not
assume that full lowers cost or that flexibility is free.

Post-processing uses existing `q`, `u`, `x`, and demand values to report:

- eligible and actually used warehouses per region;
- share of region-scenario pairs supplied by more than one warehouse;
- shipment concentration (HHI);
- unused inventory by warehouse-product;
- shortage by region-product;
- coexistence of shortage with unused same-product inventory elsewhere;
- a diagnostic sum of shortage matched by unused inventory at currently
  ineligible warehouses, without treating it as a conserved causal quantity;
- worst-shortage region and changes in its identity.

No additional optimization variable is introduced. The inaccessible-inventory
quantity is an exact post-processing diagnostic for the solved policy, but it
can double-count the same inventory across shortage regions; it will not be
interpreted as a causal flow or counterfactual reallocation amount.

For re-optimized solutions, report openings, opening vectors, Hamming distance,
total inventory, warehouse-product inventory vectors, normalized L1 inventory
distance, inventory concentration, and active warehouse-region arcs. These
separate network adaptation, inventory adaptation, and fixed-first-stage
recourse effects.

## 9. Frozen hypotheses and paired statistics

Primary hypothesis:

- H1: with fixed first-stage decisions and a common allowance, expanding
  eligibility from k1 to full reduces `T`.

Secondary hypotheses:

- H2: k2 improves `T` relative to k1;
- H3: full improves `T` relative to k2;
- H4: report the untruncated `Capture_k2` distribution without an arbitrary
  pass/fail threshold;
- H5: the direction remains consistent across medium_large and large.

Every comparison is paired by scale-seed instance. Reports include wins,
losses, ties, mean, median, mean and median relative effect, exact two-sided
sign test, deterministic 10,000-resample paired bootstrap confidence interval,
scale-specific results, and pooled results. Holm adjustment applies if the H2
and H3 p-values are formally interpreted. Effects, not p-values, lead the
interpretation.

## 10. Frozen formal classification

Classification is computed only after all 20 instances and all required
solutions certify. Otherwise reporting fails closed and no decision is issued.

- `confirm_strong_fulfillment_flexibility_mechanism`: primary sign-test
  `p < 0.05`, lower 95% bootstrap bound above zero, positive median in both
  scales, at least 15/20 wins, and pooled median relative reduction at least
  10%. The 10% formal magnitude criterion is distinct from the 5% development
  screening rule.
- `confirm_moderate_fulfillment_flexibility_mechanism`: lower confidence bound
  above zero, positive scale medians, at least 12/20 wins, and positive pooled
  median relative reduction, without satisfying the strong rule.
- `do_not_confirm_fulfillment_flexibility_mechanism`: upper confidence bound at
  or below zero, or nonpositive pooled median absolute effect.
- otherwise: `mixed_or_scale_dependent_fulfillment_flexibility`.

The decision also reports secondary effects, Holm-adjusted p-values, k2 capture,
diminishing-return count, re-optimized system effects, and certification.

## 11. Certification and immutable identity

Every main service solution must be optimal and certified. Each result records
status, incumbent, lower bound, objective-bound gap, recomputed `T`,
recomputation error, cost budget and residual, complete scenario count,
runtime, and certification. Reporting fails closed on missing or duplicate
scale-seed cells, seed/config/source/hash drift, nonfinite values, objective
inconsistency, cost-cap violation, incomplete scenarios, output overwrite, or
an uncertified result.

The later formal manifest and raw outputs bind source commit, protocol and
config hashes, runner and model-source hashes, seed list, eligibility source
hash, Gamma, rho, scale, mode, instance hash, full first-stage identity, and
common-budget identity. Existing formal output is never overwritten.

## 12. Full-mode regression

Before formal access, non-formal tiny regression instances compare `full` with
the existing unrestricted monolithic cost model and service extensive form.
Objectives, robust cost, opening and inventory feasibility, `T`, scenario
coverage, and feasible-region semantics must agree within frozen numerical
tolerances. Any discrepancy blocks authorization.

## 13. Task and runtime envelope

Each instance requires 12 optimization tasks:

- 3 re-optimized anchors;
- 3 re-optimized service models;
- 3 fixed-configuration anchors;
- 3 fixed-first-stage service models.

The preferred 20-instance design therefore has 240 tasks. With the frozen
1,800-second per-task limit, the conservative sequential envelope is 120 hours.
The exact large formulation has approximately 4.06 million columns before
mode-specific row differences. Static audit therefore requires at least 64 GiB
RAM and 250 GiB free disk or a separately reviewed eligibility-aware scalable
exact backend. Current-host inadequacy is a pre-result operational issue, not a
reason to inspect results and then alter the sample.

## 14. Execution gate and output isolation

The two configs set `formal_run_authorized: false`. The runner checks this gate
before importing the solver, generating a formal instance, enumerating its
scenarios, or creating the formal output directory. A later authorization must
be a reviewed identity-bound file and requires a protocol/config change.

Formal output is isolated at
`experiments/results_fulfillment_flexibility/formal`. Development outputs,
frozen result archives, PR #78, PR #79, Olist, and M5 are read-only and outside
the execution path.

The current task stops after protocol, audits, tests, full regression, and
solver-free dry-run. No formal optimization is permitted.
