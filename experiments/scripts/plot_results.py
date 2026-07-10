from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean_by(rows: list[dict[str, Any]], key: str, value: str) -> tuple[list[str], list[float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        numeric = _float(row.get(value))
        if numeric is not None:
            buckets[str(row.get(key, ""))].append(numeric)
    labels = sorted(buckets)
    means = [sum(buckets[label]) / len(buckets[label]) for label in labels]
    return labels, means


def _bar(plot, labels: list[str], values: list[float], title: str, ylabel: str, output: Path) -> None:
    fig, ax = plot.subplots(figsize=(8, 4.5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plot.close(fig)


def plot_results(results_csv: Path, summary_csv: Path, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot generation.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    results = _read_csv(results_csv) if results_csv.exists() else []
    summary = _read_csv(summary_csv) if summary_csv.exists() else []

    labels, values = _mean_by(summary, "method", "mean_runtime")
    if labels:
        _bar(plt, labels, values, "Runtime by method", "Mean runtime (s)", output_dir / "runtime_by_method.png")

    labels, values = _mean_by(summary, "method", "mean_final_gap")
    if labels:
        _bar(plt, labels, values, "Gap by method", "Mean final gap", output_dir / "gap_by_method.png")

    labels, values = _mean_by(summary, "method", "mean_cuts_added")
    if labels:
        _bar(plt, labels, values, "Cuts by method", "Mean cuts added", output_dir / "cuts_by_method.png")

    labels, values = _mean_by(summary, "instance_size", "mean_runtime")
    if labels:
        _bar(plt, labels, values, "Scalability runtime", "Mean runtime (s)", output_dir / "scalability_runtime.png")

    gamma_rows = [row for row in results if row.get("gamma_target")]
    labels, values = _mean_by(gamma_rows, "gamma_target", "objective")
    if labels:
        _bar(
            plt,
            labels,
            values,
            "Gamma sensitivity objective",
            "Mean objective",
            output_dir / "gamma_sensitivity_objective.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot experiment suite CSV outputs.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", default="experiments/results/plots")
    args = parser.parse_args()
    plot_results(Path(args.results), Path(args.summary), Path(args.output_dir))


if __name__ == "__main__":
    main()
