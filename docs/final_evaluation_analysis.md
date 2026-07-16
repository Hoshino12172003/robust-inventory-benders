# Final evaluation analysis

## Purpose

The final-evaluation analysis is a deterministic, post-processing-only workflow for the frozen held-out experiment. It reads existing result files and never imports or calls the optimization solver.

## Required inputs

The input directory must contain:

- `results.csv` and `summary.csv`;
- `resolved_config.yaml`;
- exactly 50 CSV files under `iteration_logs/`;
- the frozen instance JSON files under `instances/`.

The repository-root archive `final_evaluation_joint_v1_results.zip` is optional. When present, it is hashed and included in the input manifest but is never modified.

## Command

From the repository root, run:

```powershell
python scripts/analyze_final_evaluation.py `
  --input-dir experiments/results_final/final_evaluation_joint_v1 `
  --analysis-config analysis/configs/final_evaluation_joint_v1.yaml `
  --output-dir analysis/outputs/final_evaluation_joint_v1
```

The same arguments can be placed on one line on other shells.

## Generated outputs

The output directory contains:

- `audit_checks.csv` and `audit_summary.json`;
- `input_manifest.json` and `input_manifest.sha256`;
- complete descriptive CSV tables and compact paper-facing LaTeX tables;
- seed-level results and deterministic runtime ranks;
- paired comparison, bootstrap, Wilcoxon, effect-size, and Holm-adjustment tables;
- PDF and 300-dpi PNG figures under `figures/`;
- `final_evaluation_report.md` generated from computed tables.

Generated outputs and the generated analysis ZIP are ignored by Git. Raw experiment files are not copied, rewritten, or committed.

## Audit failure behavior

Every required integrity condition is evaluated before statistics or figures are generated. The audit files are written first. If any required check fails, the command raises a clear integrity error and exits nonzero; downstream statistical outputs are not produced.

The audit covers the complete seed-by-method grid, status and bound validity, target-subproblem status, final gaps, frozen algorithm isolation, iteration counts, log identity, bound monotonicity, adaptive precision monotonicity, certification overrides, fallback handling, and per-seed objective/bound interval consistency.

## Statistical comparison families

The primary confirmatory family contains joint adaptive inexact Benders versus:

- tight-tolerance inexact Benders;
- static inexact Benders.

The secondary ablation family contains joint adaptive inexact Benders versus:

- MP-adaptive inexact Benders;
- SP-adaptive inexact Benders.

Holm correction is computed independently within each family. Confirmatory wording is based on the Holm-adjusted p-value, never the raw p-value alone. Secondary comparisons are presented as ablation evidence.

## Bootstrap and paired tests

Paired differences are defined as joint-method runtime minus comparator runtime. The configured NumPy random seed makes percentile bootstrap intervals deterministic. The pipeline uses 100,000 resamples for the paired mean and paired median, two-sided Wilcoxon signed-rank tests with the Pratt zero-difference convention, no continuity correction, and paired rank-biserial effect sizes. It records the actual Wilcoxon calculation method: `exact` when no paired difference is zero, and `approx` with Pratt handling when zero differences are present.

Two percentage-saving summaries are deliberately kept distinct:

- `mean_paired_percentage_saving_percent` is the arithmetic mean of the ten seed-level percentage savings;
- `median_paired_percentage_saving_percent` is the median of those seed-level percentage savings;
- `aggregate_mean_runtime_saving_percent` is `100 * (comparator mean runtime - proposed mean runtime) / comparator mean runtime`.

The aggregate ratio is not interchangeable with the mean of paired percentages, because seeds with different comparator runtimes receive different implicit weights. The Markdown report names both definitions and prefers the aggregate mean-runtime saving in its main narrative.

Sample standard deviation (`ddof=1`) is used in paper-facing descriptive tables.

## Convergence curves

The convergence plot uses a common 0--600 second grid. For each run, the most recent observed global gap is forward-filled. Once a run has completed, its terminal observed gap is carried forward over the remaining common time grid. The procedure does not interpolate between unobserved solver states. The plot shows the across-seed median and interquartile band on a logarithmic axis and marks the `1e-4` termination tolerance.

## Interpretation limits

The analysis is conditional on ten held-out medium-large instances and the recorded runtime environment. It does not establish universal superiority or mathematical optimality of the frozen ratios. Lower runtime can reflect lower per-iteration cost rather than fewer iterations. Repeated certification by static-inexact Benders is interpreted as late-stage difficulty under fixed loose precision, while bound validity remains governed by the audited certificates.

## Solver isolation

Neither `src/final_evaluation_analysis.py` nor `scripts/analyze_final_evaluation.py` imports `gurobipy`, the Benders solver, or any model-building module. Running this workflow never invokes Gurobi and never reruns a seed.
