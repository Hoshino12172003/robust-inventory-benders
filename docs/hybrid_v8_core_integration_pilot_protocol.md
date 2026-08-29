# Hybrid v8 core-integration development pilot

## Purpose

This development-only pilot checks that the frozen Hybrid v8 policy remains
correct after integration into the production implicit-separation solver.  It
compares the v8 and legacy Hybrid policies through the same master, separator,
solver parameters, instances, anchors, tolerances, and time limits.  A finite
79-scenario deterministic equivalent is used only to check the objective.

## Frozen design

- Seeds: 10, 11, and 12; these are development seeds and are not part of any
  previous or future formal holdout.
- Dimensions: 3 warehouses, 3 products, and 4 regions.
- Uncertainty budget: Gamma = 2, giving exactly 79 finite scenarios.
- Cost allowance: rho = 0.0001.
- Methods: core Hybrid v8, core legacy Hybrid, and Direct.
- Solver controls: one thread, solver seed 0, feasibility tolerance 1e-7,
  zero master MIP gap, and 120 seconds per Hybrid method.
- Method order alternates by seed.
- Correctness requires certified termination and absolute objective error no
  larger than 1e-6 relative to Direct.

## Sequencing and claim boundary

The `prepare` stage writes the cases and their hashes.  Those inputs, this
protocol, the configuration, runner, and candidate source must be committed and
pushed before `run` is authorized.  The `run` stage refuses changed inputs or
source files.  Results are committed separately after execution.

This pilot is an implementation check, not a formal holdout and not evidence
for replacing the paper's frozen experiments.  Its outcome may decide whether
the implementation is ready for a separately preregistered validation, but it
must not be used to tune the already frozen v8 policy.
