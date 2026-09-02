# Formal fulfillment-flexibility static complexity audit

Status: protocol-only static analysis; no formal optimization was executed.

## Instance and task matrix

| Scale | Seeds | Instances | Scenarios | Modes | Tasks per instance | Total tasks |
|---|---:|---:|---:|---|---:|---:|
| medium_large | 230--239 | 10 | 1,831 | k1, k2, full | 12 | 120 |
| large | 230--239 | 10 | 4,657 | k1, k2, full | 12 | 120 |
| Total | — | 20 | — | — | — | 240 |

Each instance contains three re-optimized cost anchors, three re-optimized
service solves, three fixed-configuration anchors, and three fixed-first-stage
service solves. At 1,800 seconds per task, the conservative sequential envelope
is 432,000 seconds, or 120 hours. This envelope is not a runtime prediction.

## Exact extensive-form dimensions

Ineligible shipment variables are omitted from k1/k2 rather than created and
fixed to zero. This is algebraically equivalent to `q[i,r,j](d)=0` for an
ineligible pair and does not add a first-stage restriction.

| Scale | Mode | Columns | Anchor rows | Service rows | Omitted ineligible `q` per scenario |
|---|---|---:|---:|---:|---:|
| medium_large | k1 | 230,749 | 188,636 | 206,946 | 300 |
| medium_large | k2 | 340,609 | 188,636 | 206,946 | 240 |
| medium_large | full | 780,049 | 188,636 | 206,946 | 0 |
| large | k1 | 931,473 | 787,106 | 842,990 | 672 |
| large | k2 | 1,378,545 | 787,106 | 842,990 | 576 |
| large | full | 4,060,977 | 787,106 | 842,990 | 0 |

The full figures reproduce the existing unrestricted deterministic-equivalent
dimensions. k1/k2 reduce columns but retain the same demand, supply, product-
service, cost-cap, and regional-service row families.

## Hardware and fallback gate

The current host reports approximately 32 GiB physical memory, below the
frozen 64 GiB primary gate. The formal configs additionally require at least
250 GiB free disk. Consequently, current-host formal execution is prohibited
even after scientific review unless:

1. a host satisfying both hardware thresholds is used; or
2. a separate Route B PR implements and validates an eligibility-aware scalable
   exact backend without changing model semantics or inspecting formal outcomes.

PR #80 itself does not accept backend qualification as execution authorization.

The preregistered 5+5 fallback reduces the total runtime envelope to 60 hours
but does not reduce the per-instance large/full memory requirement. It cannot
waive the hardware gate. It may only be activated before any formal instance
generation or solver access through a reviewed protocol amendment.

## Scientific and mathematical risks frozen for review

- Re-optimized modes have endogenous anchors and need not be monotone.
- Fixed-first-stage common-budget analysis is the primary mechanism evidence.
- The “inaccessible unused inventory” quantity is a post-processing diagnostic
  and may double-count inventory across shortage regions; it is not a causal
  flow measure.
- Exact extensive forms provide transparent certification but create a serious
  large/full memory burden. A later decomposition backend must reproduce full
  mode and restricted-arc semantics and pass the same certification tests.
- Twenty instances give a paired confirmatory sample, but inference remains
  conditional on the synthetic generator and modeled uncertainty set.

The machine-readable counterpart is
`analysis/fulfillment_flexibility_formal_protocol/static_audit.json`.
