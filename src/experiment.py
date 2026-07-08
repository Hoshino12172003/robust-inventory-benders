from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .benders import solve_benders
from .instance import generate_instance, save_instance
from .monolithic import solve_monolithic
from .results import SolveResult


def solve_method(config: dict[str, Any], instance, method: str) -> SolveResult:
    if method == "monolithic":
        return solve_monolithic(config, instance)
    return solve_benders(config, instance, method)


def run_experiment(config: dict[str, Any]) -> list[SolveResult]:
    experiment_cfg = config.get("experiment", {})
    output_dir = Path(experiment_cfg.get("output_dir", "outputs/experiments"))
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = experiment_cfg.get("seeds", [config.get("seed", 42)])
    methods = experiment_cfg.get(
        "methods",
        ["monolithic", "standard_benders", "inexact_benders", "adaptive_gap_gamma_benders"],
    )

    results: list[SolveResult] = []
    for seed in seeds:
        instance = generate_instance(config, seed=int(seed))
        instance_path = output_dir / f"{instance.name}.json"
        save_instance(instance, instance_path)
        for method in methods:
            result = solve_method(config, instance, method)
            result.metadata["seed"] = seed
            result.metadata["instance"] = instance.name
            results.append(result)
            if result.iteration_log:
                _write_iteration_log(output_dir / f"{instance.name}_{method}_iterations.csv", result)

    _write_summary(output_dir / "summary.csv", results)
    _write_gamma_curve(output_dir / "gamma_curve.csv", results)
    return results


def _write_summary(path: Path, results: list[SolveResult]) -> None:
    rows = [result.summary_dict() for result in results]
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_iteration_log(path: Path, result: SolveResult) -> None:
    if not result.iteration_log:
        return
    keys = sorted({key for row in result.iteration_log for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(result.iteration_log)


def _write_gamma_curve(path: Path, results: list[SolveResult]) -> None:
    rows = []
    for result in results:
        for row in result.iteration_log:
            rows.append(
                {
                    "method": result.method,
                    "seed": result.metadata.get("seed"),
                    "instance": result.metadata.get("instance"),
                    "iteration": row["iteration"],
                    "gamma": row["gamma"],
                    "gap": row["gap"],
                    "upper_bound": row["upper_bound"],
                    "lower_bound": row["lower_bound"],
                    "target_worst_cost": row["target_worst_cost"],
                }
            )
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
