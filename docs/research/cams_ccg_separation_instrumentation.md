# CAMS-CCG separation instrumentation research prototype

This branch adds behavior-neutral, opt-in timing and call accounting. It does not
change a solver parameter, stopping rule, tolerance, candidate order, cut,
scenario block, or scientific output when instrumentation is disabled. The
production default is disabled. This is a research instrumentation prototype,
not a new algorithm and not an ASBP implementation.

## Scope and clocks

Each ordinary or final-exact separation call has a canonical identity derived
from the run key, iteration, call index, and final-exact flag. Durations use
`time.perf_counter_ns()`. The mutually exclusive classified phases are model
preparation, separation MILP optimization, solution-pool extraction, cache
candidate processing, fixed-primal certification, Farkas certification,
candidate identity/de-duplication, deterministic selection, final-exact
preparation, and final-exact optimization. Time outside those scopes is recorded
as `separation_unclassified_ns`, never silently assigned to another phase.
Persistent model construction precedes the first separation call, so it is a
separate `persistent_model_setup` event rather than being backdated into that
call. Its total equals its model-prepare phase, preserving both real monotonic
start/end timestamps and timing conservation.

All solver attributes are nullable and carry a missing-reason field. NaN and
infinity are converted to unavailable states before canonical JSON encoding.
Ordinary and final-exact calls remain separate in the deterministic aggregator.

## Transaction and resume semantics

Call records remain pending until their master iteration commits. A checkpoint
contains schema identity, committed call identities, the pending identity,
cumulative committed values, the last committed iteration, and full ledgers.
On resume, a pending attempt is retained only as discarded evidence and is not
included in cumulative totals. Committed identities cannot be accepted twice.
The ledger is encoded as sorted, finite JSON so rebuilding a checkpoint from the
same state is byte stable.
When instrumentation is enabled, its schema is part of the checkpoint run
identity. An uninstrumented checkpoint or a checkpoint from another schema
therefore cannot be resumed as if it contained a complete compatible ledger.

## Interpretation limits

Instrumentation reports measurement facts only. Phase shares do not establish
a bottleneck, speedup, algorithm advantage, or statistical conclusion. No
development or formal optimization was run for this change. The tests use fake
clocks and mock scientific outputs and do not invoke Gurobi.
