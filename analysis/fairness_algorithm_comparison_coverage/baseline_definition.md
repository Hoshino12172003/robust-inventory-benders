# Baseline definitions

## Model baseline

The model baseline is the cost-optimal robust model without the fairness constraint. It defines C*(Gamma) and therefore the fairness cost budget (1+rho)C*(Gamma). It is not an algorithmic competitor.

## Algorithm baselines

The algorithm baselines are certified Benders variants: single-cut, persistent separation, certified cache, batch-5 certified cache, and variants without complete scenario recourse blocks. Runtime comparisons are strictly paired only when scale, seed, Gamma, rho, instance, baseline, anchor, solver parameters, time limit, success definition, and final exact certification requirement all coincide.
