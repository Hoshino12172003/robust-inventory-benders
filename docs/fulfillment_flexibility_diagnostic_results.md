# Multi-warehouse fulfillment flexibility diagnostic results

Status: development-only evidence. These results do not replace, revise, or
enter any frozen formal experiment.

## Reproducibility record

- preregistered source commit: `2147ff6`;
- branch: `agent/fulfillment-flexibility-diagnostic`;
- command: `python -m src.fulfillment_flexibility_diagnostic --config experiments/configs/fulfillment_flexibility_diagnostic.yaml --stage all`;
- scales: small and medium;
- development seeds: 190, 191, and 192 per scale;
- uncertainty budget: Gamma = 2;
- service-protection allowance: rho = 0.025;
- solver controls: one thread, solver seed zero, MIP gap `1e-4`, and 300 seconds per optimization.

The complete preregistration is in
`docs/fulfillment_flexibility_diagnostic_protocol.md`. Machine-readable results
are under `experiments/results_fulfillment_flexibility/development`. Cost
components below are evaluated in the scenario that maximizes recourse cost;
they are not scenario averages.

## Certification and integrity checks

All 18 re-optimized cost anchors, 18 fixed-configuration cost anchors, and 36
service-protection models terminated with optimal status. All 36 service
solutions were certified. The maximum absolute difference between the model
objective and the recomputed worst-region shortage rate was
`1.08e-15`; the maximum objective--bound difference was `2.95e-6`.

The run created new files only in the dedicated development output directory.
It did not call or overwrite a frozen experiment configuration, seed, result,
or manuscript table.

## Primary results

Table 1 reports the mean worst-region shortage rate over three seeds. In the
re-optimized comparison, each eligibility mode obtains its own robust-cost
anchor and allowance. In the fixed-first-stage comparison, warehouse openings
and inventories from the full-mode solution are held constant and all modes
receive the same cost allowance.

**Table 1. Mean worst-region shortage rate by fulfillment mode**

| Evaluation | Scale | k1 | k2 | full |
|---|---:|---:|---:|---:|
| Re-optimized | Small | 0.4040 | 0.1622 | 0.1260 |
| Re-optimized | Medium | 0.5041 | 0.2166 | 0.1157 |
| Re-optimized | Pooled | 0.4541 | 0.1894 | 0.1208 |
| Fixed first stage | Small | 0.6884 | 0.3259 | 0.1260 |
| Fixed first stage | Medium | 0.6524 | 0.3283 | 0.1112 |
| Fixed first stage | Pooled | 0.6704 | 0.3271 | 0.1186 |

Table 2 gives paired effect sizes. A win means a strictly smaller worst-region
shortage rate for the more flexible mode. Relative reductions are computed
within each scale-seed pair before taking the median.

**Table 2. Paired flexibility effects over six development instances**

| Evaluation | Comparison | Wins | Mean absolute reduction | Median relative reduction | Mean relative robust-cost change |
|---|---|---:|---:|---:|---:|
| Re-optimized | k1 to k2 | 6/6 | 0.2647 | 64.36% | -36.89% |
| Re-optimized | k2 to full | 5/6 | 0.0686 | 33.73% | -18.33% |
| Re-optimized | k1 to full | 6/6 | 0.3332 | 72.78% | -50.96% |
| Fixed first stage | k1 to k2 | 6/6 | 0.3433 | 57.71% | 0.00% |
| Fixed first stage | k2 to full | 6/6 | 0.2085 | 51.50% | 0.00% |
| Fixed first stage | k1 to full | 6/6 | 0.5518 | 77.86% | 0.00% |

The sole non-monotone re-optimized pair was medium seed 191: k2 obtained
`T = 0.1437` and full obtained `T = 0.1443`. This difference is only 0.0006
and does not contradict feasible-set nesting because the two re-optimized
modes use different endogenous cost anchors and hence different cost caps.
Under the fixed configuration and common allowance, every instance satisfied
`T_k1 > T_k2 > T_full`.

## Cost and network mechanism

Table 3 decomposes the pooled re-optimized solutions. More flexible
fulfillment used more transportation, inventory, and facility capacity, but it
substantially reduced shortage and service-violation costs. Consequently, the
measured robust total cost fell rather than rose. This is not evidence that
flexibility is free: the protection components increased, whereas the avoided
failure costs decreased by more.

**Table 3. Mean re-optimized cost and network diagnostics**

| Mode | Open warehouses | Active warehouse-region arcs | Opening cost | Inventory cost | Transportation cost | Shortage cost | Service-violation cost | Robust total cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k1 | 3.33 | 6.00 | 717.00 | 3,176.74 | 2,673.60 | 8,119.49 | 12,304.49 | 26,991.33 |
| k2 | 3.83 | 12.00 | 814.83 | 3,504.83 | 3,044.35 | 4,149.39 | 4,835.72 | 16,349.14 |
| full | 4.00 | 26.00 | 852.50 | 3,674.11 | 3,286.98 | 2,224.80 | 1,553.80 | 11,592.19 |

Relative to k1, full flexibility increased mean transportation cost by 22.9%,
inventory cost by 15.7%, and opening cost by 18.9%. It reduced mean shortage
cost by 72.6% and service-violation cost by 87.4%. The resulting robust total
cost was 51.0% lower. These numbers support a pooling interpretation: broader
warehouse eligibility spends more on physical fulfillment but avoids a larger
amount of stockout-related cost.

The re-optimized k1 and full solutions selected different warehouse-opening
vectors in four of six instances (mean Hamming distance 0.667). Their inventory
vectors differed in all six instances; the mean normalized L1 difference was
1.096. Hence part of the re-optimized effect arises from changes in network
configuration and inventory positioning. The fixed-first-stage comparison
removes both channels. Its pooled mean worst-region shortage rate fell from
0.6704 to 0.1186, with a median paired reduction of 77.86%, demonstrating a
large recourse-flexibility effect even when openings, inventory, and the cost
allowance are identical.

## Answers to the diagnostic questions

1. **Does service protection improve as eligibility expands?** Yes. The fixed
   comparison yielded strict `k1 > k2 > full` nesting in all six instances.
   The re-optimized comparison yielded six k1-to-k2 wins, five k2-to-full wins,
   and six k1-to-full wins. The pooled median k1-to-full reduction was 72.78%.
2. **Is the effect caused by network design?** Partly. Openings changed in four
   instances and inventory positioning changed in all six. These are genuine
   design responses to the eligibility regime.
3. **Is there an independent recourse effect?** Yes. With the full-mode first
   stage fixed, expanded eligibility still reduced the pooled mean worst-region
   shortage rate by 0.5518 and won every paired comparison.
4. **Does flexibility reduce shortage without a proportional cost increase?**
   In this diagnostic, it did more: additional fulfillment expenditures were
   outweighed by avoided shortage and violation costs, so robust total cost
   declined. The result should be interpreted as development evidence, not as
   a population estimate or a claim that every network will exhibit this cost
   dominance.

## Frozen recommendation

The preregistered rule required positive k1-to-full improvement in at least
four of six seeds and a pooled median relative reduction of at least 5%. The
observed values were 6/6 and 72.78%, respectively. The machine-generated
decision is therefore:

`proceed_to_formal_flexibility_experiment`

This recommendation supports designing a separate, preregistered formal
multi-warehouse flexibility experiment. It does not authorize rewriting the
paper from these development seeds alone. The formal experiment should use
new frozen seeds or external data, retain k1/k2/full, and keep the
fixed-first-stage decomposition so that network-design and recourse effects
remain distinguishable.
