# EJOR algorithm-comparison assessment

Decision: `minimal_paired_algorithm_benchmark_recommended`.

The S1 archives provide auditable development evidence: medium-large certified counts are 6/6, 6/6, 2/6, and 6/6 for single-cut, persistent separation, certified cache, and batch-5 respectively; all four Large variants are 0/6 because they reached the time limit without certification, not because of pipeline errors. This supports a qualitative statement that the older certified methods did not scale reliably on the Large development screen.

However, the Gamma Attempt 3 Hybrid cells (seeds 180-184, Gamma=0/1/2, rho=0.025) have no fully identical algorithm-baseline cells. Consequently, the existing evidence does not support a strict paired runtime speedup claim. A minimal benchmark should add only ten reference frontiers: medium-large and large, seeds 180-184, Gamma=2, rho=0.025, using the exact Attempt 3 instances, baselines and anchors, Threads=1, solver Seed=0, FeasibilityTol=1e-7, and a 1800-second limit. Hybrid results remain read-only; no Hybrid rerun or tuning is permitted.
