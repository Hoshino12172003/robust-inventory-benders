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

The same seed labels across scales permit scale-specific and pooled reporting.
Within each scale, the scale-seed instance is paired. For pooled inference, the
independent statistical cluster is the synthetic seed because both scales use
the same seeded random stream. For seed `s`, the frozen pooled cluster effect is
the arithmetic mean of its medium_large and large effects. Pooled sign tests
and bootstrap intervals therefore use ten seed clusters, not twenty independent
scale-seed observations. Seeds were
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
therefore the high-memory hardware gate remains required in PR #80. A scalable
exact backend requires a separate implementation and review. No fallback has
been activated in this protocol PR.

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

This construction supplies the same **absolute** allowance to k1, k2, and full.
It must not be described as giving every mode exactly 2.5% above that mode's own
fixed-configuration minimum; the common anchor is the maximum of the three
fixed-mode anchors.

The fixed-first-stage comparison is conditional on the certified full-mode
`T`-optimal first-stage configuration selected by the frozen solution
procedure. It is not claimed to be invariant across every possible `T`-optimal
first-stage configuration.

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
recourse effects. A separate exploratory regional diagnostics table exports the
eligible and actually used warehouse counts for every region and the scenario
in which that region reaches its worst shortage rate.

Because the service model minimizes only `T`, opening, inventory, shipment, and
realized-cost solutions need not be unique. Cost decompositions, source-use and
concentration measures, active arcs, unused inventory, region-product shortage,
and first-stage vector differences are labelled exploratory diagnostics of one
certified `T`-optimal solution. They never determine the formal classification.
Confirmatory conclusions use only `Delta_12`, `Delta_2F`, `Delta_1F`, `R_12`,
`R_1F`, `Capture_k2`, and their frozen paired inference.

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

Every within-scale comparison is paired by scale-seed instance. Pooled reports
first average the two scale effects within seed and then treat the ten seeds as
independent clusters. Reports include wins,
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
  scales, at least 8/10 pooled cluster wins, and pooled median relative reduction at least
  10%. The 10% formal magnitude criterion is distinct from the 5% development
  screening rule.
- `confirm_moderate_fulfillment_flexibility_mechanism`: lower confidence bound
  above zero, positive scale medians, at least 6/10 pooled cluster wins, and positive pooled
  median relative reduction, without satisfying the strong rule.
- `do_not_confirm_fulfillment_flexibility_mechanism`: upper confidence bound at
  or below zero, or nonpositive pooled median absolute effect.
- otherwise: `mixed_or_scale_dependent_fulfillment_flexibility`.

The decision separately reports fixed-first-stage/common-budget and re-optimized
system effects, including their secondary Holm-adjusted p-values, k2 capture,
diminishing-return count, and certification. Only fixed-first-stage `k1` versus
`full` determines the primary classification.

## 11. Certification and immutable identity

Every main service solution must be optimal and certified. Each result records
status, incumbent, lower bound, objective-bound gap, recomputed `T`,
recomputation error, cost budget and residual, complete scenario count,
runtime, and certification. Reporting fails closed on missing or duplicate
scale-seed cells, seed/config/source/hash drift, nonfinite values, objective
inconsistency, cost-cap violation, incomplete scenarios, output overwrite, or
an uncertified result.

Every formal cost-anchor solve uses `MIPGap = 0` independently of the service-
model working gap. An anchor certifies only with optimal status and an absolute
incumbent-bound gap no greater than `1e-4`. The objective, best bound, absolute
gap, and status are recorded without rounding; an uncertified anchor blocks all
dependent service tasks.

The later formal manifest, immutable task checkpoints, and raw outputs bind source commit, protocol and
config hashes, runner and model-source hashes, seed list, eligibility source
hash, Gamma, rho, scale, mode, instance hash, full first-stage identity, and
common-budget identity. Task checkpoints also bind the instance-generator and
scenario-generator source hashes. Reporting independently recomputes current
frozen identities, verifies each raw result against the canonical certified
checkpoint-result hash, and recomputes anchor and common-budget links before
pooling. Existing certified evidence is never overwritten.

Gurobi major/minor version is a strict solver-environment identity field.
Exact package version, Python version and implementation, operating system,
platform string, and architecture are reproducibility metadata. A separate
Gurobi build identifier is recorded when exposed; current package metadata does
not expose one without importing the solver. Harmless metadata differences do
not invalidate evidence, but strict Gurobi major/minor identity and the frozen
solver-parameter identity must match.

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
RAM and 250 GiB free disk. The 64 GiB value is only a minimum gate, not proof of
sufficiency. PR #80 accepts only an identity-bound reviewed high-memory
extensive-form hardware qualification. A certified-backend qualification cannot
authorize execution until a separate Route B implementation PR supplies and
validates that backend. Current-host inadequacy is a pre-result operational
issue, not a reason to inspect results and then alter the sample.

## 14. Execution gate and output isolation

The two configs set `formal_run_authorized: false`, which protocol-stage
validation requires. A separately reviewed amendment may transition the config
to `true`; execution-stage validation then requires both an exact identity-bound
authorization and a separate identity-bound execution qualification. This gate
runs before solver import, instance generation, scenario enumeration, or output
creation. It also requires a clean relevant source tree; tracked modifications
or relevant untracked files fail closed and are never automatically cleaned or
stashed. The identity-bound authorization and hardware-qualification records
are execution records rather than executable sources and are the only excluded
paths; their contents are validated independently by the gate.

After the gate passes, each of the twelve optimization tasks per instance is
published as an immutable create-once checkpoint. Finalization uses an atomic
fail-if-exists hard-link operation so concurrent writers cannot overwrite an
existing target. Resume skips only
a certified, identity-valid checkpoint. Temporary, corrupt, uncertified, or
identity-incompatible files fail closed. Task identity binds scale, seed, mode,
task type, Gamma, rho, source, protocol/config/runner/eligibility/scenario/solver
identities and, where applicable, first-stage, anchor, and common-budget identities.

Formal output is isolated at
`experiments/results_fulfillment_flexibility/formal`. Development outputs,
frozen result archives, PR #78, PR #79, Olist, and M5 are read-only and outside
the execution path.

The current task stops after protocol, audits, tests, full regression, and
solver-free dry-run. No formal optimization is permitted.
