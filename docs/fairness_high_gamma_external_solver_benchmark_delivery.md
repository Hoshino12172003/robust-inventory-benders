# High-Gamma external-solver benchmark delivery

This delivery freezes a 45-task small-instance matrix and a strict runner for 15 Gamma-specific baselines, 15 certified Hybrid frontiers, and 15 paired Gurobi direct extensive-form frontiers. Formal optimization is not started by the implementation PR. The reviewed authorization file, clean merged tree, formal short worktree, absent output directory, seed-access gate, and final solver-free dry-run are all required before the user starts the run manually.

The pre-run mathematical gate uses three fixed non-formal, non-degenerate hand instances at Gamma 0 through 4. All 15 Hybrid/direct cells must agree on objective T and legal bound sandwiches within 1e-7 and both complete post-evaluations must satisfy the identical cost cap. These tests do not use seeds 185--189 and are excluded from the paper sample.
