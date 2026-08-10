# High-Gamma external benchmark Attempt 2 delivery

## Status

The implementation freezes HG1 as a baseline Gamma identity incident and prepares a full, isolated Attempt 2. The implementation PR alone does not authorize optimization. Formal execution remains blocked until a separate authorization-only PR is independently reviewed and merged.

## Frozen execution

- stage: `HIGH_GAMMA_EXTERNAL_BENCHMARK`
- execution attempt: 2
- output: `experiments/results_fh_ext/hg2`
- matrix: 15 fresh baseline + 15 fresh Hybrid + 15 fresh direct extensive-form tasks
- seeds: 185–189
- Gamma: 2, 3, 4
- rho: 0.025
- prior artifacts reused: false

Before either frontier begins, the requested Gamma must match the final baseline solver configuration, solver result, run/checkpoint, anchor, manifest, complete scenario count, and a checkpointed full-scenario T=1 lifting evaluation. Any discrepancy blocks the cell and the formal run fail closed.

The formal command will be frozen only by the authorization-only delivery. This document intentionally contains no executable authorized command.
