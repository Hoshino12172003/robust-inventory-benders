from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


MERGE_COMMIT = "13493ba63006604443f54f61799842dc2a3fbac9"
MERGED_AT = "2026-07-30T02:58:06Z"
ZIP_SHA256 = "BD253A4DE967143B8CD88E7DEADCB821D6062208CB71664B900F9BD8EBDF3839"
RESULTS_SHA256 = "50EB5823F4C7138E65FA36546B90EE081B48949D2F961F5AFDFAE098A7F0A496"
PAPER_METRICS_SHA256 = "044689ABF1ADD1C1FC217FCB5F46B8D280D8659865EE3A3707EBB9FE792F2E37"
SOURCE_DIR = Path("analysis/fairness_hybrid_final_holdout_reconciliation")
RHOS = [0.0, 0.01, 0.025, 0.05, 0.1]
SCALES = ["medium_large", "large"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _mean(rows: Iterable[dict[str, Any]], name: str) -> float:
    values = [float(row[name]) for row in rows]
    return sum(values) / len(values)


def _main_table(groups: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    selected = [
        "objective_t", "robust_minimum_fill_rate", "minimum_weighted_mean_fill_rate",
        "wminfr", "actual_robust_cost", "actual_price_of_fairness", "algorithm_runtime",
        "post_evaluation_wall_runtime", "total_wall_runtime", "iterations", "scenario_count", "cut_count",
    ]
    rows: list[dict[str, Any]] = []
    for group in groups:
        row: dict[str, Any] = {"scale": group["scale"], "rho": group["rho"], "n": group["seed_count"]}
        for metric in selected:
            stats = group["metrics"][metric]
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_sd"] = stats["standard_deviation"]
        rows.append(row)
    header = "| Scale | rho | T | Certified min FR | Weighted mean FR | Realized min FR | Cost | Price of fairness | Algorithm s | Post-eval s | Iter. | Scenarios | Cuts |\n"
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    lines = [header, divider]
    for row in rows:
        cell = lambda name, digits=3: f"{_fmt(row[name + '_mean'], digits)} ± {_fmt(row[name + '_sd'], digits)}"
        lines.append(
            f"| {row['scale']} | {row['rho']:.3f} | {cell('objective_t')} | {cell('robust_minimum_fill_rate')} | "
            f"{cell('minimum_weighted_mean_fill_rate')} | {cell('wminfr')} | {cell('actual_robust_cost', 1)} | "
            f"{cell('actual_price_of_fairness')} | {cell('algorithm_runtime', 1)} | {cell('post_evaluation_wall_runtime', 1)} | "
            f"{cell('iterations', 1)} | {cell('scenario_count', 1)} | {cell('cut_count', 1)} |\n"
        )
    return rows, "".join(lines)


def _long_table(groups: list[dict[str, Any]], definitions: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        for metric, stats in group["metrics"].items():
            rows.append(
                {
                    "scale": group["scale"], "rho": group["rho"], "seed_count": group["seed_count"],
                    "metric": metric, "label": definitions[metric]["label"], "unit": definitions[metric]["unit"],
                    "mean": stats["mean"], "median": stats["median"], "standard_deviation": stats["standard_deviation"],
                    "iqr": stats["iqr"], "min": stats["min"], "max": stats["max"],
                }
            )
    return rows


def _figures(output: Path, groups: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "font.size": 9})
    colors = {"medium_large": "#1f77b4", "large": "#d62728"}
    labels = {"medium_large": "Medium-large", "large": "Large"}

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for scale in SCALES:
        data = [g for g in groups if g["scale"] == scale]
        x = [100 * g["metrics"]["actual_price_of_fairness"]["mean"] for g in data]
        y = [100 * g["metrics"]["robust_minimum_fill_rate"]["mean"] for g in data]
        xerr = [100 * g["metrics"]["actual_price_of_fairness"]["standard_deviation"] for g in data]
        yerr = [100 * g["metrics"]["robust_minimum_fill_rate"]["standard_deviation"] for g in data]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, marker="o", capsize=2.5, color=colors[scale], label=labels[scale])
        for rho, xv, yv in zip(RHOS, x, y):
            ax.annotate(f"ρ={rho:g}", (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Actual price of fairness (%)")
    ax.set_ylabel("Certified minimum regional fill rate (%)")
    ax.set_title("Final-holdout fairness–cost trade-off (mean ± SD across seeds)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output / "figure_fairness_cost_tradeoff.png", metadata={"Software": "matplotlib"})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True, constrained_layout=True)
    for ax, scale in zip(axes, SCALES):
        data = [g for g in groups if g["scale"] == scale]
        for metric, label, style in (
            ("algorithm_runtime", "Algorithm", "-o"),
            ("post_evaluation_wall_runtime", "Post-evaluation", "-s"),
            ("total_wall_runtime", "Total wall", "-^")
        ):
            ax.plot(RHOS, [g["metrics"][metric]["mean"] for g in data], style, label=label)
        ax.set_title(labels[scale])
        ax.set_xlabel("ρ")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Seconds (mean across 10 seeds)")
    axes[1].legend()
    fig.suptitle("Runtime components by scale and fairness budget")
    fig.savefig(output / "figure_runtime_scalability.png", metadata={"Software": "matplotlib"})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for ax, scale in zip(axes, SCALES):
        data = [g for g in groups if g["scale"] == scale]
        for metric, label, style in (("iterations", "Iterations", "-o"), ("scenario_count", "Scenario blocks", "-s"), ("cut_count", "Farkas cuts", "-^") ):
            ax.plot(RHOS, [g["metrics"][metric]["mean"] for g in data], style, label=label)
        ax.set_title(labels[scale])
        ax.set_xlabel("ρ")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Count (mean across 10 seeds)")
    axes[1].legend()
    fig.suptitle("Certified algorithm structure by scale and fairness budget")
    fig.savefig(output / "figure_algorithm_structure.png", metadata={"Software": "matplotlib"})
    plt.close(fig)


def generate(repo_root: Path, output: Path) -> dict[str, str]:
    source = repo_root / SOURCE_DIR
    metrics_path = source / "paper_metrics.json"
    results_path = source / "results.corrected.csv"
    if _sha(metrics_path) != PAPER_METRICS_SHA256 or _sha(results_path) != RESULTS_SHA256:
        raise ValueError("frozen derived source hash mismatch")
    metrics = _json(metrics_path)
    audit = _json(source / "final_holdout_audit.json")
    decision = _json(source / "decision.json")
    if audit["archive_sha256_before"] != audit["archive_sha256_after"] or audit["archive_sha256_before"] != ZIP_SHA256:
        raise ValueError("formal ZIP identity is not frozen")
    if metrics["scope"] != "FINAL_HOLDOUT_only_D1_D2_excluded" or decision["optimization_rerun_required"] is not False:
        raise ValueError("paper scope or scientific decision mismatch")
    output.mkdir(parents=True, exist_ok=False)

    freeze = {
        "status": "frozen_unique_final_paper_authority",
        "scope": "Hybrid FINAL_HOLDOUT only; D1/D2 excluded",
        "pr": "https://github.com/Hoshino12172003/robust-inventory-benders/pull/51",
        "merge_commit": MERGE_COMMIT,
        "merged_at": MERGED_AT,
        "source_zip_sha256": ZIP_SHA256,
        "results_corrected": {"path": str(SOURCE_DIR / "results.corrected.csv").replace("\\", "/"), "sha256": RESULTS_SHA256},
        "paper_metrics": {"path": str(SOURCE_DIR / "paper_metrics.json").replace("\\", "/"), "sha256": PAPER_METRICS_SHA256},
        "scientific_decision": decision,
        "immutability_rule": "Any change to one of these four identities creates a new, non-final analysis and must not silently replace this freeze.",
    }
    _write_json(output / "freeze_manifest.json", freeze)

    groups = metrics["scale_rho_summaries"]
    main_rows, main_md = _main_table(groups)
    _write_csv(output / "table_main_results.csv", main_rows)
    (output / "table_main_results.md").write_text(main_md, encoding="utf-8", newline="\n")
    _write_csv(output / "table_all_descriptive_statistics.csv", _long_table(groups, metrics["metric_definitions"]))
    _write_csv(output / "table_complete_seed_results.csv", sorted(metrics["complete_seed_results"], key=lambda row: (SCALES.index(row["scale"]), row["rho"], row["seed"])))
    _write_csv(output / "table_cross_scale_tests.csv", metrics["cross_scale_per_rho_paired"])
    _figures(output, groups)

    complete = metrics["complete_seed_results"]
    by_scale = {scale: [row for row in complete if row["scale"] == scale] for scale in SCALES}
    first_last = {(scale, rho): next(g for g in groups if g["scale"] == scale and g["rho"] == rho) for scale in SCALES for rho in (0.0, 0.1)}
    overall = metrics["cross_scale_overall_seed_aggregated"]
    max_certificate_difference = max(abs(row["robust_minimum_fill_rate"] - row["wminfr"]) for row in complete)
    chapter = f"""# Hybrid Final Holdout 实验结果（冻结草稿）

## 结果依据与统计设计

本节仅使用 Hybrid `FINAL_HOLDOUT`，不混入 D1/D2。唯一最终依据由 `freeze_manifest.json` 固定：原始 ZIP SHA256 为 `{ZIP_SHA256}`，PR #51 merge commit 为 `{MERGE_COMMIT}`，修正结果 CSV SHA256 为 `{RESULTS_SHA256}`，`paper_metrics.json` SHA256 为 `{PAPER_METRICS_SHA256}`。全部 120 个任务（20 个 baseline、100 个 frontier）通过只读审计；100 个 frontier 均完成 exact certification，13,050 个 post-evaluation chunk 的 SHA 全部正确。独立实验单位为 seed；每个 scale×ρ 包含 10 个配对 seed。跨规模 overall 统计先在 seed 内聚合五个 ρ，再进行 seed-cluster bootstrap；五个 per-ρ 检验使用 Holm 校正。

## 指标解释

`robust_minimum_fill_rate` 是算法证书 `1-T`，表示所有精确不确定场景、所有适用区域的最低 fill-rate 保证。`wminfr` 是 exact post-evaluation 直接观测到的 $\\min_s\\min_r$ 区域 fill rate；两者在 100 个任务上的最大绝对差为 `{max_certificate_difference:.3e}`。`minimum_weighted_mean_fill_rate` 则是 `min_s(1 - total_shortage_s / total_demand_s)`，即最坏场景下按需求加权的系统平均 fill rate，不能标为“最低区域 fill rate”。成本采用 exact post-evaluation 的 first-stage cost 加最坏 recourse cost。算法 runtime、post-evaluation wall time 与 total wall time分别报告，PAR-2 仅以算法 runtime 为基础。

## 公平—成本权衡

Medium-large 的平均认证最低 fill rate 从 ρ=0 的 `{first_last[('medium_large', 0.0)]['metrics']['robust_minimum_fill_rate']['mean']:.3f}` 提升到 ρ=0.10 的 `{first_last[('medium_large', 0.1)]['metrics']['robust_minimum_fill_rate']['mean']:.3f}`；Large 从 `{first_last[('large', 0.0)]['metrics']['robust_minimum_fill_rate']['mean']:.3f}` 提升到 `{first_last[('large', 0.1)]['metrics']['robust_minimum_fill_rate']['mean']:.3f}`。对应的实际 price of fairness 均由近 0 增至约 `{100 * first_last[('medium_large', 0.1)]['metrics']['actual_price_of_fairness']['mean']:.2f}%`（Medium-large）和 `{100 * first_last[('large', 0.1)]['metrics']['actual_price_of_fairness']['mean']:.2f}%`（Large）。图 `figure_fairness_cost_tradeoff.png` 展示了这一单调权衡；完整 mean、median、standard deviation、IQR、min、max 见 `table_all_descriptive_statistics.csv`，100 个 seed×ρ 明细见 `table_complete_seed_results.csv`。

## 可扩展性与跨规模比较

跨全部 50 个任务，Medium-large 的平均算法 runtime 为 `{_mean(by_scale['medium_large'], 'algorithm_runtime'):.2f}` 秒，Large 为 `{_mean(by_scale['large'], 'algorithm_runtime'):.2f}` 秒；平均 total wall time 分别为 `{_mean(by_scale['medium_large'], 'total_wall_runtime'):.2f}` 秒和 `{_mean(by_scale['large'], 'total_wall_runtime'):.2f}` 秒。运行时间分解见 `figure_runtime_scalability.png`，迭代、完整场景块和认证 Farkas cut 见 `figure_algorithm_structure.png`。Large−Medium-large 的 seed 内先聚合 ρ 后 T 差均值为 `{overall['large_minus_medium_large_objective_t']['mean']:.4f}`，cluster bootstrap 95% CI 为 `[{overall['cluster_bootstrap_95_percent_ci'][0]:.4f}, {overall['cluster_bootstrap_95_percent_ci'][1]:.4f}]`，配对置换 p 值为 `{overall['paired_permutation_pvalue']:.4f}`。五个 per-ρ Holm 校正 p 值均为 1.0，因此本 holdout 不支持两个规模在 T 上存在系统性差异的结论；这不等同于证明二者相同。

## Bound reconciliation 与科学有效性

唯一 crossing 出现在 Large、seed 172、ρ=0.10。历史 max-LB 比最终 UB 高 `2.0694404762322538e-05`，位于冻结容差 `1e-4` 内。该历史值只作为轨迹 ledger；论文认证下界采用最终当前 master solver best bound。该 bound 与最终 incumbent/UB 相等，且 final exact separation optimal、objective bound 为 `-0.0`，完整认证同一解鲁棒可行。因此该异常定性为 reporting/ledger 语义问题，不要求优化重跑，也不改变 100/100 certified 的科学状态。

## 报告边界

表和图均直接由冻结的 `paper_metrics.json` 生成。D1/D2 只能作为 development/scalability evidence 单独讨论，不进入本节估计、置信区间或显著性检验。任何改变 ZIP SHA、merge commit、修正 CSV SHA 或 paper-metrics SHA 的分析均不得沿用“唯一最终依据”标签。
"""
    (output / "experimental_results.md").write_text(chapter, encoding="utf-8", newline="\n")

    artifact_names = sorted(path.name for path in output.iterdir() if path.is_file())
    manifest_rows = [{"relative_path": name, "sha256": _sha(output / name)} for name in artifact_names]
    _write_csv(output / "artifact_sha256.csv", manifest_rows)
    return {row["relative_path"]: row["sha256"] for row in manifest_rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.repo_root, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
