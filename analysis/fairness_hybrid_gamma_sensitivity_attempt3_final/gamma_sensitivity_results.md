# Gamma sensitivity results


Frozen source: `EE45A00AA341EE5EB2894DE43EE2F47022C27F1D29146FCFEC803236EF59DB6F`. The independent unit is the seed (n=5 per scale-Gamma cell).

Gamma differences are interpreted descriptively and through within-seed paired trajectories; seed-Gamma rows are not treated as independent replicates.


## Metric semantics


- `robust_minimum_fill_rate = 1 - objective_t` is the certified minimum regional service guarantee.

- `wminfr` is the post-evaluation fill rate of the worst scenario-region combination.

- `minimum_weighted_mean_fill_rate` is the demand-weighted system mean fill rate in the worst scenario.

These three quantities are distinct and are not relabelled as one another.


## Findings


For medium-large, mean baseline robust cost rose from 18562.05 at Gamma=0 to 20281.76 at Gamma=2, while the certified minimum regional fill-rate guarantee changed from 0.9522 to 0.9377. Mean algorithm runtime increased from 0.091s to 22.034s.

Within-seed paired mean changes for Gamma 0->1 and 1->2 were, respectively: baseline cost +896.01 and +823.71; certified minimum fill rate -0.0067 and -0.0078; algorithm runtime +2.696s and +19.248s.

For large, mean baseline robust cost rose from 25565.85 at Gamma=0 to 26789.54 at Gamma=2, while the certified minimum regional fill-rate guarantee changed from 0.9699 to 0.9641. Mean algorithm runtime increased from 0.117s to 76.231s.

Within-seed paired mean changes for Gamma 0->1 and 1->2 were, respectively: baseline cost +634.71 and +588.98; certified minimum fill rate -0.0023 and -0.0034; algorithm runtime +9.793s and +66.321s.


The small n=5 design supports structural sensitivity and managerial interpretation, not a claim that non-significance proves no effect.
