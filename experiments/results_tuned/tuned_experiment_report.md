# Tuned Experiment Report

## Scope

This report compares the preliminary PR #7 experiment outputs with the tuned PR #8 outputs.

The tuned runs changed experiment configuration parameters only. They did not change the mathematical model, the robust dual MILP derivation, or the core Benders implementation.

## Parameter Changes

- Increased `max_iterations` from 120 to 300 for tuned medium / large experiments.
- Increased tuned experiment `time_limit` to 600 seconds.
- Kept `robust_dual_milp` as the main subproblem mode.
- Kept small correctness results from PR #7 unchanged.
- Removed full scenario enumeration from tuned baseline comparison to avoid spending medium-scale time on scenario-enumeration benchmarking.
- Used positive `delta_cut` values in tuned configs. The ablation study uses a larger `delta_cut` so cut skipping is observable.

## High-Level Findings

- Small-scale results remain solved under the strict final-gap criterion.
- Medium-scale final gaps improved substantially compared with PR #7 preliminary results, but solved rates are still low under the strict `1e-4` solved criterion.
- Large-scale scalability rows remain completed but unsolved under the strict criterion.
- Cut selection is now observable in the tuned ablation study: the `full` variant records skipped cuts while `no_cut_selection` does not.
- These tuned results are improved diagnostics, not final paper-ready computational results.

## Baseline Comparison

| size | method | variant | prelim solved | tuned solved | prelim gap | tuned gap | prelim runtime | tuned runtime | tuned iterations | tuned cuts skipped |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium | proposed_adaptive_benders | proposed_adaptive_benders | 0.0000 | 0.0000 | 0.3318 | 0.0514 | 3.1838 | 15.4059 | 300.00 | 0.0000 |
| medium | scenario_benders_full | scenario_benders_full | 0.4000 |  | 0.0014 |  | 298.32 |  |  |  |
| medium | standard_benders | standard_benders | 0.0000 | 0.0000 | 0.3318 | 0.0514 | 3.1898 | 15.7830 | 300.00 | 0.0000 |
| medium | static_inexact_benders | static_inexact_benders | 0.0000 | 0.0000 | 0.3318 | 0.0514 | 3.1843 | 15.3507 | 300.00 | 0.0000 |
| small | proposed_adaptive_benders | proposed_adaptive_benders | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.4408 | 0.4384 | 37.0000 | 1.0000 |
| small | scenario_benders_full | scenario_benders_full | 1.0000 |  | 0.0000 |  | 1.4304 |  |  |  |
| small | standard_benders | standard_benders | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.4581 | 0.4728 | 37.0000 | 0.0000 |
| small | static_inexact_benders | static_inexact_benders | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.4364 | 0.4615 | 37.0000 | 0.0000 |

## Ablation Study

| size | method | variant | prelim solved | tuned solved | prelim gap | tuned gap | prelim runtime | tuned runtime | tuned iterations | tuned cuts skipped |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium | proposed_adaptive_benders | full | 0.0000 | 0.0000 | 0.3488 | 0.0630 | 3.4363 | 15.0783 | 300.00 | 2.0000 |
| medium | proposed_adaptive_benders | no_adaptive_gap | 0.0000 | 0.0000 | 0.3564 | 0.0524 | 3.7685 | 15.2022 | 300.00 | 2.0000 |
| medium | proposed_adaptive_benders | no_cut_selection | 0.0000 | 0.0000 | 0.3488 | 0.0669 | 3.3597 | 14.3265 | 300.00 | 0.0000 |
| medium | proposed_adaptive_benders | no_gamma_continuation | 0.0000 | 0.0000 | 0.3486 | 0.0624 | 3.7591 | 14.7334 | 300.00 | 0.0000 |
| medium | proposed_adaptive_benders | standard | 0.0000 | 0.0000 | 0.3318 | 0.0514 | 3.7193 | 15.2137 | 300.00 | 0.0000 |
| medium_large | proposed_adaptive_benders | full | 0.0000 |  | 0.4631 |  | 3.4650 |  |  |  |
| medium_large | proposed_adaptive_benders | no_adaptive_gap | 0.0000 |  | 0.4774 |  | 3.9071 |  |  |  |
| medium_large | proposed_adaptive_benders | no_cut_selection | 0.0000 |  | 0.4631 |  | 3.4950 |  |  |  |
| medium_large | proposed_adaptive_benders | no_gamma_continuation | 0.0000 |  | 0.4590 |  | 3.3693 |  |  |  |
| medium_large | proposed_adaptive_benders | standard | 0.0000 |  | 0.4656 |  | 4.1257 |  |  |  |

## Sensitivity: Gamma

| size | method | variant | prelim solved | tuned solved | prelim gap | tuned gap | prelim runtime | tuned runtime | tuned iterations | tuned cuts skipped |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium | proposed_adaptive_benders | proposed_adaptive_benders | 0.0000 | 0.0000 | 0.3056 | 0.0461 | 2.7712 | 12.6124 | 300.00 | 0.0000 |

## Sensitivity: Service Level

| size | method | variant | prelim solved | tuned solved | prelim gap | tuned gap | prelim runtime | tuned runtime | tuned iterations | tuned cuts skipped |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium | proposed_adaptive_benders | proposed_adaptive_benders | 0.0000 | 0.0000 | 0.2745 | 0.0182 | 2.6540 | 12.4196 | 300.00 | 0.0000 |

## Scalability

| size | method | variant | prelim solved | tuned solved | prelim gap | tuned gap | prelim runtime | tuned runtime | tuned iterations | tuned cuts skipped |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| large | proposed_adaptive_benders | proposed_adaptive_benders | 0.0000 | 0.0000 | 0.6441 | 0.5377 | 6.7405 | 26.0398 | 300.00 | 0.0000 |
| large | standard_benders | standard_benders | 0.0000 | 0.0000 | 0.6441 | 0.5377 | 6.7197 | 25.9324 | 300.00 | 0.0000 |
| medium | proposed_adaptive_benders | proposed_adaptive_benders | 0.0000 | 0.0000 | 0.2676 | 0.0450 | 3.9621 | 16.4619 | 300.00 | 0.0000 |
| medium | standard_benders | standard_benders | 0.0000 | 0.0000 | 0.2676 | 0.0450 | 4.3910 | 16.9829 | 300.00 | 0.0000 |
| small | proposed_adaptive_benders | proposed_adaptive_benders | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.6978 | 0.4790 | 38.3333 | 1.0000 |
| small | standard_benders | standard_benders | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.7045 | 0.4902 | 38.3333 | 0.0000 |

## Tuned Snapshot

| experiment | min solved_rate | max solved_rate | min mean_final_gap | max mean_final_gap |
| --- | ---: | ---: | ---: | ---: |
| baseline_comparison | 0.0000 | 1.0000 | 0.0000 | 0.0514 |
| ablation_study | 0.0000 | 0.0000 | 0.0514 | 0.0669 |
| sensitivity_gamma | 0.0000 | 0.0000 | 0.0461 | 0.0461 |
| sensitivity_service | 0.0000 | 0.0000 | 0.0182 | 0.0182 |
| scalability | 0.0000 | 1.0000 | 0.0000 | 0.5377 |

## Honest Interpretation

The tuning improves medium-scale final gaps, especially compared with the PR #7 preliminary medium rows that often had gaps around 0.3. However, most medium and large rows still stop at the iteration limit and do not satisfy the strict solved criterion.

Further work is needed before these can be presented as final paper results. A follow-up direction is to tune the algorithm itself, including adaptive gap scheduling, Gamma continuation timing, cut management, or stronger stopping / bounding strategies.
