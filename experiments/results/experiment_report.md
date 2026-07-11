# Experiment Results Report

## Scope

These results were generated from the formal experiment-suite configurations and are kept as preliminary computational artifacts.

They are useful for checking the pipeline and inspecting early behavior, but they should not be treated as final paper results.

## Main Observations

- Small correctness validation: all rows in `small_correctness/correctness_summary.csv` are `ok`.
- Small baseline comparison: all small-scale baseline rows are solved under the current solved-gap criterion.
- Medium and large runs: many rows are completed with an incumbent result but remain at `iteration_limit`.
- Medium / large results are therefore preliminary and should not yet be interpreted as final convergence evidence.

## Sensitivity Summaries

Additional grouped summaries were added:

- `sensitivity_gamma/gamma_summary.csv`
- `sensitivity_service/service_summary.csv`

These files aggregate objective, runtime, final gap, and worst-case cost by the relevant sensitivity parameter.

## Next PR

Use a separate PR #8 to tune experiment parameters before producing paper-ready results:

- Increase the iteration limit to 300 or 500.
- Consider increasing the time limit.
- Consider relaxing the solved-gap threshold for exploratory medium/large analysis.
- Adjust adaptive gap and Gamma continuation settings so medium instances can converge more reliably.
