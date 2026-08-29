# Hybrid v8 M5 external real-data holdout protocol

## Status and authorization

This document is protocol-only.  It authorizes neither raw-data inspection nor
optimization.  The protocol, configuration, ingestion code, and frozen Hybrid
v8 candidate must be reviewed and merged before the M5 files are processed.

## External data

The sole source is the official Kaggle **M5 Forecasting - Accuracy**
competition.  The required files are `sales_train_evaluation.csv` and
`calendar.csv`.  Their SHA-256 hashes, byte counts, row counts, and column
identities must be recorded immediately after download and before processing.
Unofficial mirrors are forbidden.

The M5 data contain actual daily unit sales for 3,049 products sold through 10
Walmart stores in California, Texas, and Wisconsin.  The experiment uses these
observations only as demand evidence.  Facility capacities, fixed costs,
inventory costs, transfer costs, shortage penalties, and service targets are
calibrated modeling inputs and must not be described as observed Walmart data.

## Frozen temporal split

- Factor-estimation period: `d_1` through `d_1717`.
- Calibration period: `d_1718` through `d_1857`.
- Formal holdout: `d_1858` through `d_1941`, partitioned chronologically into
  12 non-overlapping seven-day cases.

The holdout values must not be summarized, plotted, screened, or used for any
algorithm choice before this protocol is merged.  Every one of the 12 cases
must be reported regardless of direction, runtime, certification status, or
cut activation.

## Frozen model mapping

- Demand regions: all 10 `store_id` values; no store is selected by volume.
- Products: all seven `dept_id` values; no department is selected by volume.
- Candidate staging locations: the same 10 store locations.  They represent
  candidate inventory-staging positions, not observed Walmart warehouses.
- Demand is aggregated to store-department-day cells and then to the 12
  predeclared holdout weeks.
- Six uncertainty factors are estimated from positive normalized residuals in
  `d_1`--`d_1717` only.  Each store-department cell is assigned to exactly one
  factor by its largest absolute loading with deterministic lexical tie-breaks.
- Factor deviations are calibrated from `d_1718`--`d_1857` only at the frozen
  90th percentile.  The factor budget is Gamma = 2, producing exactly 22
  scenarios: the nominal case, six single-factor cases, and 15 two-factor
  cases.
- All non-demand parameters follow deterministic formulas recorded in the
  processing configuration.  No formula may be changed after holdout values
  are processed.

## Frozen algorithms and controls

The primary comparison is the merged Hybrid v8 policy against standard pure
single-scenario CCG.  Batch-4 CCG is the strong scenario-block ablation and the
complete Direct deterministic equivalent is the objective check.  All methods
use the same instance, scenarios, cost anchor, Gurobi backend, one thread,
solver seed zero, feasibility tolerance `1e-7`, final MIP gap zero, and a
1,800-second time limit.  Method order rotates deterministically by case.

Hybrid v8 remains fixed at up to four violation-ranked scenario blocks and at
most one max-coefficient-normalized Farkas cut with efficacy at least 0.10.
No post-holdout change to the candidate or comparator is permitted.

## Outcomes and decision rules

Primary outcomes are certification rate and paired PAR-2 runtime for Hybrid v8
versus pure CCG.  PAR-2 assigns 3,600 seconds to a timeout or uncertified run.
Secondary outcomes are paired raw runtime among jointly certified cases,
Direct-objective agreement, iterations, scenario blocks, and Farkas cuts.
Batch-4 comparisons quantify whether any improvement exceeds the contribution
of bounded multi-scenario submission.

Report the paired win/loss/tie counts, geometric mean runtime ratio on jointly
certified cases, and an exact two-sided sign test excluding ties.  With only 12
weekly cases, estimates and raw paired results take precedence over thresholded
significance claims.  A Farkas-cut advantage may be claimed only if cuts trigger
and the preregistered Batch-4 comparison supports it; otherwise the result must
be reported as absent or inconclusive.

## Reproducibility gates

1. Merge this protocol-only PR.
2. Download the two official files and record raw hashes without examining
   holdout values.
3. Run the deterministic processor and commit the raw manifest, processed
   inputs, case catalog, and hashes in a second PR.
4. Merge the input-freeze PR before any optimization.
5. Run all 12 cases once, archive every raw result and environment snapshot,
   and commit results separately.

The existing Olist experiments and all prior paper experiments remain immutable.
