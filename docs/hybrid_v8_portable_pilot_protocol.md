# Hybrid v8 portable finite-scenario pilot protocol

## Purpose

This development-only pilot checks that the frozen Hybrid v8 policy can be reproduced on repository-generated inventory instances outside the Olist-derived cases. It is a portability, correctness, and mechanism audit. It is not a new final holdout and cannot replace a later core implicit-uncertainty confirmation on untouched seeds.

## Frozen matrix

- seeds: 0, 1, 2;
- warehouses: 3;
- products: 3;
- regions: 4;
- Gamma: 2;
- exact scenario count: 79;
- fairness cost allowance: rho = 0.0001;
- methods: Hybrid v8, pure single-scenario CCG, Batch-4 CCG, complete Direct model;
- run order: cyclic rotation of the three decomposition methods by seed;
- Gurobi Threads: 1;
- Gurobi Seed: 0;
- final MIPGap: 0;
- final complete scenario certification is mandatory.

## Execution order

1. `prepare` generates only the three instances and their complete scenario lists.
2. The generated inputs are SHA-256 frozen before optimization.
3. The protocol, runner, config, candidate source identity, and input manifest are committed and pushed before `run`.
4. `run` solves a common cost anchor, complete Direct model, and all three decomposition methods for every seed.
5. `summarize` reports every preregistered cell without outcome-dependent deletion or rerun.

## Outcomes

- certification status;
- objective absolute error against Direct;
- runtime and paired ordering;
- iterations;
- complete scenario blocks;
- actually committed Farkas cuts.

Runtime is descriptive because the independent sample contains only three generated instances. Correctness requires every Hybrid v8 result to certify and agree with Direct within `1e-6`. Failure remains reported and does not authorize parameter changes.

## Boundary

This pilot uses public development seeds and an explicit finite scenario set. A later paper-level experiment must integrate the policy into the core implicit separator, preregister a new experiment family, and use seeds that were inaccessible during development. The old Final Holdout seeds 170--179 remain immutable and may not be reused for v8 selection or confirmation.
