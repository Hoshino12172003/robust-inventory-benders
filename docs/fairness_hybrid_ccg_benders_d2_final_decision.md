# Hybrid C&CG--Benders D2 final development decision

## Decision

The read-only audit of `fairness_hybrid_ccg_benders_d2_a3_large_results.zip` passed. The source archive SHA256 is `07746988F93B2CD6E6BDE7B08EB661FCB330B3CA7A3A874A772BE9ED24258271`; CRC, all identities, 12 run/status pairs, nine algorithm checkpoints, 1,683 post-evaluation chunks, and 41,913 post-evaluation scenario records were independently checked.

D2 contains 3/3 certified baselines and 9/9 certified Hybrid frontiers. Every frontier ended with a legal master bound, zero reported final gap, complete exact separation, `robust_feasibility_certified=true`, and a valid 4,657-scenario post-evaluation. The D2 decision is therefore `approve_final_cross_scale_holdout_protocol`.

This decision closes algorithm development and tuning. The following are frozen:

- candidate `certified_hybrid_scenario_benders_fairness`;
- exact Farkas certification;
- complete scenario recourse blocks;
- final exact separation and objective-bound certification;
- uncertainty set, fairness model, time limits, feasibility tolerance and success definition.

D1 and D2 remain development evidence only. They are not holdout observations and must not be pooled with final results. No D1/D2 instance, baseline, anchor, checkpoint, scenario, cut, result or summary may be reused in the final holdout.

The original D2 ZIP is not committed, rewritten or repacked. Machine-readable evidence is in `analysis/fairness_hybrid_ccg_benders_d2_decision/`.
