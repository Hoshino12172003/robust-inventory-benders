from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import gurobipy as gp
import numpy as np

from .benders import solve_benders
from .benders_zigzag_diagnostic import analyze_trajectory
from .config import load_config
from .experiment_protocol import atomic_write_csv, atomic_write_json, file_sha256, utc_now_iso
from .experiment_suite import _apply_selected_parameters, _apply_variant_config, _base_config
from .instance import load_instance
from .scenarios import count_budget_scenarios


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/benders_zigzag_diagnostic_210202.yaml"
OUTPUT_FILES = (
    "zigzag_diagnostic_summary.json",
    "zigzag_diagnostic_segments.csv",
    "zigzag_diagnostic_movements.csv",
    "zigzag_diagnostic_patterns.csv",
    "zigzag_trajectory_metrics.csv",
    "zigzag_trajectory_y.csv",
    "zigzag_trajectory_x.csv",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, timeout=10
    ).strip()


def _path(value: str) -> Path:
    return REPO_ROOT / value


def _verify_file(path: Path, expected_hash: str, label: str) -> None:
    _require(path.is_file(), f"Missing {label}: {path}")
    actual = file_sha256(path)
    _require(actual.lower() == expected_hash.lower(), f"{label} SHA-256 mismatch: {actual}")


def _resolve_solver(protocol: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    frozen = protocol["frozen_solver"]
    for label in ("protocol", "selected_parameters", "selected_candidate"):
        _verify_file(
            _path(frozen[f"{label}_path"]), frozen[f"{label}_sha256"], label
        )
    flat = load_config(_path(frozen["protocol_path"]))
    selected = _apply_selected_parameters(flat)
    base = _base_config(selected, "large", seed=0)
    method, _, resolved = _apply_variant_config(
        base,
        frozen["method"],
        selected["variant_settings"][frozen["variant"]],
    )
    return method, resolved


def preflight(config_path: Path) -> dict[str, Any]:
    protocol = load_config(config_path)
    _require(protocol["protocol"] == "benders_zigzag_diagnostic_v1", "Wrong protocol")
    _require(protocol["case_id"] == "210202", "Only Renault 210202 is authorized")
    _require(int(protocol["replay_time_limit_seconds"]) == 300, "Replay must remain 300 seconds")
    for label in ("instance", "index_metadata", "eligibility"):
        _verify_file(_path(protocol[f"{label}_path"]), protocol[f"{label}_sha256"], label)
    eligibility = json.loads(_path(protocol["eligibility_path"]).read_text(encoding="utf-8"))
    full = np.asarray(eligibility["Full"], dtype=int)
    _require(protocol["eligibility_mode"] == "Full", "Only Full eligibility is authorized")
    _require(full.shape == (15, 12) and bool((full == 1).all()), "Full eligibility changed")
    instance = load_instance(_path(protocol["instance_path"]))
    expected = protocol["expected_dimensions"]
    _require(
        (instance.num_warehouses, instance.num_products, instance.num_regions)
        == (expected["I"], expected["J"], expected["R"])
        == (15, 8, 12),
        "Instance dimensions changed",
    )
    _require(int(protocol["gamma"]) == 2, "Gamma changed")
    _require(count_budget_scenarios(instance, 2) == 4657, "Scenario count changed")
    _require(
        math.isclose(instance.budget, float(protocol["expected_budget"]), rel_tol=0.0, abs_tol=1e-9),
        "Budget changed",
    )
    method, solver = _resolve_solver(protocol)
    _require(method == "adaptive_gap_gamma_benders", "Frozen method changed")
    _require(solver["benders"]["time_limit"] == 1800, "Frozen formal time limit changed")
    _require(solver["robust"]["gamma_target"] == 2 and solver["robust"]["gamma_schedule"] == [2], "Frozen Gamma policy changed")
    algorithm = solver["algorithm"]
    _require(algorithm["subproblem_mode"] == "robust_dual_milp", "Subproblem mode changed")
    _require(algorithm["cut_strengthening_policy"] == "core_point", "PR #82 cut mode changed")
    _require(algorithm["max_cuts_per_iteration"] == 1, "Multi-cut is forbidden")
    _require(algorithm["cut_selection_enabled"] is False, "Cut selection changed")
    _require(algorithm["final_certification_enabled"] is True, "Certification changed")
    return {"protocol": protocol, "instance": instance, "method": method, "solver": solver}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        atomic_write_csv(path, rows, list(rows[0]))


def reanalyze_existing(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    protocol = load_config(config_path)
    output_root = _path(protocol["output_root"])

    def read_rows(filename: str) -> list[dict[str, str]]:
        with (output_root / filename).open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))

    raw_metrics = read_rows("zigzag_trajectory_metrics.csv")
    raw_y = read_rows("zigzag_trajectory_y.csv")
    raw_x = read_rows("zigzag_trajectory_x.csv")
    _require(len(raw_metrics) == len(raw_y) == len(raw_x), "Saved trajectory lengths differ")
    metrics = [
        {
            "iteration": int(row["iteration"]),
            "elapsed_time": float(row["elapsed_time"]),
            "LB": float(row["LB"]),
            "UB": float(row["UB"]),
            "gap": float(row["gap"]),
            "master_time": float(row["master_time"]),
            "cuts_added_total": int(row["cuts_added_total"]),
        }
        for row in raw_metrics
    ]
    y_fields = [field for field in raw_y[0] if field != "iteration"]
    x_fields = [field for field in raw_x[0] if field != "iteration"]
    _require(len(y_fields) == 15 and len(x_fields) == 120, "Saved trajectory dimensions changed")
    y_trajectory = [[float(row[field]) for field in y_fields] for row in raw_y]
    x_trajectory = [
        np.asarray([float(row[field]) for field in x_fields], dtype=float).reshape(15, 8).tolist()
        for row in raw_x
    ]
    diagnostic = analyze_trajectory(metrics, y_trajectory, x_trajectory)
    summary_path = output_root / "zigzag_diagnostic_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    segment_by_name = {row["segment"]: row for row in diagnostic["segments"]}
    summary.update(
        {
            "analysis_git_commit": _git_commit(),
            "final_100": segment_by_name["final_100"],
            "final_250": segment_by_name["final_250"],
            "classification": diagnostic["classification"],
            "classification_evidence": diagnostic["classification_evidence"],
            "unique_y_patterns": diagnostic["unique_y_patterns"],
            "y_pattern_switches": diagnostic["y_pattern_switches"],
            "short_cycles": diagnostic["short_cycles"],
            "movement_relationships": diagnostic["relationships"],
            "tail_movement_relationships": diagnostic["tail_relationships"],
            "enough_to_explain_tail_stagnation": diagnostic["enough_to_explain_tail_stagnation"],
            "stabilization_recommendation": diagnostic["stabilization_recommendation"],
        }
    )
    _write_csv(output_root / "zigzag_diagnostic_movements.csv", diagnostic["movements"])
    _write_csv(output_root / "zigzag_diagnostic_segments.csv", diagnostic["segments"])
    _write_csv(output_root / "zigzag_diagnostic_patterns.csv", diagnostic["patterns"])
    atomic_write_json(summary_path, summary)
    return summary


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    context = preflight(config_path)
    protocol = context["protocol"]
    output_root = _path(protocol["output_root"])
    existing = [output_root / name for name in OUTPUT_FILES if (output_root / name).exists()]
    if existing:
        raise RuntimeError(f"Refusing to repeat the one-shot replay; output exists: {existing[0]}")

    solver = deepcopy(context["solver"])
    solver["benders"]["time_limit"] = 300
    solver["diagnostics"] = {"record_first_stage_trajectory": True}
    commit = _git_commit()
    started = utc_now_iso()
    result = solve_benders(solver, context["instance"], context["method"])
    completed = utc_now_iso()
    _require(result.iteration_log, "Replay produced no iterations")
    _require(
        all("trajectory_y" in row and "trajectory_x" in row for row in result.iteration_log),
        "First-stage trajectory logging was incomplete",
    )

    metrics: list[dict[str, Any]] = []
    y_trajectory: list[list[float]] = []
    x_trajectory: list[list[list[float]]] = []
    for row in result.iteration_log:
        y_values = row["trajectory_y"]
        x_values = row["trajectory_x"]
        y_trajectory.append(y_values)
        x_trajectory.append(x_values)
        metrics.append(
            {
                "iteration": row["iteration"],
                "elapsed_time": row["elapsed_time"],
                "LB": row["LB"],
                "UB": row["UB"],
                "gap": row["gap"],
                "master_time": row["master_time"],
                "subproblem_time": row["subproblem_time"],
                "auxiliary_time": row["core_point_stage1_runtime"] + row["core_point_stage2_runtime"],
                "cuts_added_total": row["cuts_added_total"],
                "open_depots": row["open_depots"],
            }
        )

    diagnostic = analyze_trajectory(metrics, y_trajectory, x_trajectory)
    metadata = json.loads(_path(protocol["index_metadata_path"]).read_text(encoding="utf-8"))
    y_rows = [
        {"iteration": metrics[index]["iteration"], **{f"y_{code}": value for code, value in zip(metadata["depot_order"], values)}}
        for index, values in enumerate(y_trajectory)
    ]
    x_rows = []
    for index, matrix in enumerate(x_trajectory):
        row: dict[str, Any] = {"iteration": metrics[index]["iteration"]}
        for depot, values in zip(metadata["depot_order"], matrix):
            row.update({f"x_{depot}_{product}": value for product, value in zip(metadata["product_order"], values)})
        x_rows.append(row)

    segment_by_name = {row["segment"]: row for row in diagnostic["segments"]}
    runtime_total = result.master_runtime + result.subproblem_runtime
    auxiliary_runtime = float(result.metadata.get("core_point_total_runtime", 0.0))
    summary = {
        "schema": "benders_zigzag_diagnostic_v1",
        "diagnostic_only": True,
        "formal_result": False,
        "one_shot_replay": True,
        "historical_pr82_1800s_xy_recoverable": False,
        "historical_data_limitation": "PR #82 iteration log contains scalar metrics only; y/x cannot be reconstructed.",
        "input": {
            "case_id": "210202",
            "instance_sha256": protocol["instance_sha256"],
            "dimensions": {"I": 15, "J": 8, "R": 12},
            "gamma": 2,
            "eligibility": "Full",
            "time_limit_seconds": 300,
        },
        "provenance": {
            "git_commit": commit,
            "config_sha256": file_sha256(config_path),
            "started_at": started,
            "completed_at": completed,
            "solver_version": ".".join(str(value) for value in gp.gurobi.version()),
        },
        "replay": {
            "status": result.status,
            "iterations": result.iterations,
            "cuts": result.cuts,
            "runtime": result.runtime,
            "LB": result.lower_bound,
            "UB": result.upper_bound,
            "gap": result.gap,
        },
        "runtime_breakdown": {
            "master_seconds": result.master_runtime,
            "subproblem_seconds": result.subproblem_runtime,
            "auxiliary_seconds": auxiliary_runtime,
            "master_share_of_total_runtime": result.master_runtime / result.runtime,
            "subproblem_share_of_total_runtime": result.subproblem_runtime / result.runtime,
            "auxiliary_share_of_total_runtime": auxiliary_runtime / result.runtime,
            "accounted_core_seconds": runtime_total,
        },
        "final_100": segment_by_name["final_100"],
        "final_250": segment_by_name["final_250"],
        "classification": diagnostic["classification"],
        "classification_evidence": diagnostic["classification_evidence"],
        "unique_y_patterns": diagnostic["unique_y_patterns"],
        "y_pattern_switches": diagnostic["y_pattern_switches"],
        "short_cycles": diagnostic["short_cycles"],
        "movement_relationships": diagnostic["relationships"],
        "tail_movement_relationships": diagnostic["tail_relationships"],
        "enough_to_explain_tail_stagnation": diagnostic["enough_to_explain_tail_stagnation"],
        "stabilization_recommendation": diagnostic["stabilization_recommendation"],
        "interpretation_policy": "Classification uses only predeclared thresholds in src/benders_zigzag_diagnostic.py; it is diagnostic evidence, not a causal guarantee.",
        "historical_300s_comparators": {
            "PR82_original_baseline": {"iteration": 843, "LB": 116799.22154734991, "UB": 116854.79883500279, "gap": 0.00047560980128298635},
            "PR83_warm_start_diagnostic": {"iteration": 840, "LB": 116799.21974106538, "UB": 116854.79883500279, "gap": 0.0004756252587956587},
            "PR84_pareto_diagnostic": {"iteration": 860, "LB": 116792.68793882779, "UB": 116861.05322415945, "gap": 0.0005850134278742427},
        },
    }

    _write_csv(output_root / "zigzag_trajectory_metrics.csv", metrics)
    _write_csv(output_root / "zigzag_trajectory_y.csv", y_rows)
    _write_csv(output_root / "zigzag_trajectory_x.csv", x_rows)
    _write_csv(output_root / "zigzag_diagnostic_movements.csv", diagnostic["movements"])
    _write_csv(output_root / "zigzag_diagnostic_segments.csv", diagnostic["segments"])
    _write_csv(output_root / "zigzag_diagnostic_patterns.csv", diagnostic["patterns"])
    atomic_write_json(output_root / "zigzag_diagnostic_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the one-shot Renault Benders trajectory diagnostic")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--analyze-existing",
        action="store_true",
        help="Recompute diagnostic reports from saved trajectory CSVs without solving.",
    )
    args = parser.parse_args()
    summary = (
        reanalyze_existing(args.config.resolve())
        if args.analyze_existing
        else run(args.config.resolve())
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
