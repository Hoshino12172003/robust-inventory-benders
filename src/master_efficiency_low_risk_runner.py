from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from io import StringIO
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any

import numpy as np

from .benders import solve_benders
from .config import load_config
from .experiment_protocol import atomic_write_csv, atomic_write_json, file_sha256, git_commit, utc_now_iso
from .experiment_suite import _apply_selected_parameters, _apply_variant_config, _base_config
from .instance import generate_instance, load_instance


ROOT = Path(__file__).resolve().parents[1]
FINAL_PROTOCOL = ROOT / "experiments/configs/cut_strengthened_joint_v3_final_large.yaml"
RENAULT_INSTANCE = (
    ROOT
    / "real_data_studies/renault_formal_instances_v6/instances/210202_B1.00.json"
)
RENAULT_ELIGIBILITY = (
    ROOT
    / "real_data_studies/renault_formal_instances_v6/eligibility/210202_eligibility.json"
)
EXPECTED_RENAULT_SHA256 = "6792234516446485b95f4111f3826eb5d0cc03b440890a98b71207f50eb5524a"
EXPECTED_RENAULT_BUDGET = 84614.30513135396
REPORT_JSON = ROOT / "artifacts/master_efficiency_low_risk_report.json"
REPORT_CSV = ROOT / "artifacts/master_efficiency_low_risk_report.csv"
DEFAULT_BASELINE_REF = (
    "origin/codex/run-renault-certified-robust-baselines:"
    "real_data_studies/renault_robust_baseline_v1/results/iteration_logs/"
    "210202_B1.00_iterations.csv"
)


def _resolved_config(size: str, seed: int, time_limit: float) -> tuple[str, dict[str, Any]]:
    selected = _apply_selected_parameters(load_config(FINAL_PROTOCOL))
    base = _base_config(selected, size, seed)
    method, _, config = _apply_variant_config(
        base,
        "proposed_adaptive_benders",
        selected["variant_settings"]["joint_v1_core_point_strengthened"],
    )
    config["benders"]["time_limit"] = time_limit
    return method, config


def _valid_bounds(result: Any, tolerance: float = 1.0e-6) -> bool:
    return (
        result.lower_bound is not None
        and result.upper_bound is not None
        and math.isfinite(result.lower_bound)
        and math.isfinite(result.upper_bound)
        and result.lower_bound <= result.upper_bound + tolerance
    )


def _summary(case: str, variant: str, result: Any) -> dict[str, Any]:
    return {
        "case": case,
        "variant": variant,
        "status": result.status,
        "certified": result.status == "optimal" and result.gap is not None and result.gap <= 1.0e-4,
        "objective": result.objective,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "gap": result.gap,
        "iterations": result.iterations,
        "proposed_cuts": result.metadata["proposed_cuts_total"],
        "accepted_cuts": result.metadata["accepted_cuts_total"],
        "duplicate_cuts_rejected": result.metadata["exact_duplicates_rejected"],
        "master_runtime": result.master_runtime,
        "total_runtime": result.runtime,
        "valid_bounds": _valid_bounds(result),
    }


def _reduction(old: float, new: float) -> dict[str, float]:
    return {
        "absolute": old - new,
        "percent": 100.0 * (old - new) / old if old else 0.0,
    }


def run_paired() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for size, seed in (("small", 0), ("small", 1), ("medium", 0)):
        method, base_config = _resolved_config(size, seed, 120.0)
        instance = generate_instance(base_config, seed=seed)
        legacy_config = deepcopy(base_config)
        legacy_config["algorithm"]["master_efficiency_low_risk"] = False
        improved_config = deepcopy(base_config)
        improved_config["algorithm"]["master_efficiency_low_risk"] = True
        improved_config["algorithm"]["duplicate_cut_tolerance"] = 1.0e-10

        legacy = solve_benders(legacy_config, instance, method)
        improved = solve_benders(improved_config, instance, method)
        case = f"{size}_seed_{seed}"
        legacy_row = _summary(case, "legacy", legacy)
        improved_row = _summary(case, "low_risk", improved)
        rows.extend((legacy_row, improved_row))
        objective_match = (
            legacy.objective is not None
            and improved.objective is not None
            and math.isclose(legacy.objective, improved.objective, rel_tol=0.0, abs_tol=1.0e-6)
        )
        pairs.append(
            {
                "case": case,
                "objective_match": objective_match,
                "both_certified": legacy_row["certified"] and improved_row["certified"],
                "valid_bounds": legacy_row["valid_bounds"] and improved_row["valid_bounds"],
                "master_runtime_reduction": _reduction(legacy.master_runtime, improved.master_runtime),
                "total_runtime_reduction": _reduction(legacy.runtime, improved.runtime),
                "iteration_reduction": legacy.iterations - improved.iterations,
                "duplicate_cuts_rejected": improved.metadata["exact_duplicates_rejected"],
            }
        )

    legacy_rows = [row for row in rows if row["variant"] == "legacy"]
    improved_rows = [row for row in rows if row["variant"] == "low_risk"]
    return {
        "status": "PASS"
        if all(pair["objective_match"] and pair["both_certified"] and pair["valid_bounds"] for pair in pairs)
        else "FAIL",
        "cases": pairs,
        "rows": rows,
        "aggregate": {
            "master_runtime_reduction": _reduction(
                sum(row["master_runtime"] for row in legacy_rows),
                sum(row["master_runtime"] for row in improved_rows),
            ),
            "total_runtime_reduction": _reduction(
                sum(row["total_runtime"] for row in legacy_rows),
                sum(row["total_runtime"] for row in improved_rows),
            ),
            "iteration_reduction": sum(row["iterations"] for row in legacy_rows)
            - sum(row["iterations"] for row in improved_rows),
            "duplicate_cuts_rejected": sum(
                row["duplicate_cuts_rejected"] for row in improved_rows
            ),
        },
    }


def _baseline_rows(git_ref: str) -> list[dict[str, str]]:
    payload = subprocess.check_output(
        ["git", "show", git_ref], cwd=ROOT, text=True, encoding="utf-8", timeout=30
    )
    return list(csv.DictReader(StringIO(payload)))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _similarity_summary(log: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["nearest_cut_similarity"])
        for row in log[-500:]
        if row.get("nearest_cut_similarity") is not None
    ]
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(values),
        "count_ge_0_999": sum(value >= 0.999 for value in values),
    }


def _master_by_cut_quartile(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not log:
        return []
    groups = np.array_split(np.asarray(log, dtype=object), 4)
    return [
        {
            "quartile": index,
            "first_cut_count": int(group[0]["accumulated_cut_count"]),
            "last_cut_count": int(group[-1]["accumulated_cut_count"]),
            "mean_master_runtime": statistics.fmean(
                float(row["master_time"]) for row in group
            ),
        }
        for index, group in enumerate(groups, start=1)
        if len(group)
    ]


def run_renault_replay(baseline_ref: str) -> dict[str, Any]:
    if file_sha256(RENAULT_INSTANCE) != EXPECTED_RENAULT_SHA256:
        raise RuntimeError("Renault instance SHA-256 mismatch")
    instance = load_instance(RENAULT_INSTANCE)
    if (instance.num_warehouses, instance.num_products, instance.num_regions) != (15, 8, 12):
        raise RuntimeError("Renault dimensions changed")
    if not math.isclose(instance.budget, EXPECTED_RENAULT_BUDGET, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError("Renault budget changed")
    eligibility = json.loads(RENAULT_ELIGIBILITY.read_text(encoding="utf-8"))
    full = np.asarray(eligibility["Full"], dtype=int)
    if full.shape != (15, 12) or not bool((full == 1).all()):
        raise RuntimeError("Renault Full eligibility changed")

    method, config = _resolved_config("large", 0, 300.0)
    config["algorithm"]["master_efficiency_low_risk"] = True
    config["algorithm"]["duplicate_cut_tolerance"] = 1.0e-10
    improved = solve_benders(config, instance, method)

    baseline = [row for row in _baseline_rows(baseline_ref) if _float(row, "elapsed_time") <= 300.0]
    if not baseline:
        raise RuntimeError("No PR #82 baseline rows exist before 300 seconds")
    old = baseline[-1]
    old_master = sum(_float(row, "master_time") for row in baseline)
    old_subproblem = sum(_float(row, "subproblem_time") for row in baseline)
    old_core = sum(
        _float(row, "core_point_stage1_runtime") + _float(row, "core_point_stage2_runtime")
        for row in baseline
    )
    new_last = improved.iteration_log[-1]
    return {
        "status": improved.status,
        "not_a_formal_result": True,
        "time_cap_seconds": 300.0,
        "baseline_source": baseline_ref,
        "baseline": {
            "elapsed_time": _float(old, "elapsed_time"),
            "iteration": int(old["iteration"]),
            "lower_bound": _float(old, "lower_bound"),
            "upper_bound": _float(old, "upper_bound"),
            "gap": _float(old, "gap"),
            "accepted_cuts": int(old["cuts_added_total"]),
            "master_runtime": old_master,
            "subproblem_runtime": old_subproblem,
            "core_point_runtime": old_core,
        },
        "low_risk": {
            "elapsed_time": improved.runtime,
            "iteration": improved.iterations,
            "lower_bound": improved.lower_bound,
            "upper_bound": improved.upper_bound,
            "gap": improved.gap,
            "proposed_cuts": improved.metadata["proposed_cuts_total"],
            "accepted_cuts": improved.metadata["accepted_cuts_total"],
            "duplicate_cuts_rejected": improved.metadata["exact_duplicates_rejected"],
            "master_runtime": improved.master_runtime,
            "subproblem_runtime": improved.subproblem_runtime,
            "core_point_runtime": improved.metadata["core_point_total_runtime"],
            "valid_bounds": _valid_bounds(improved),
            "last_master_status": new_last["master_status_name"],
            "similarity_last_500": _similarity_summary(improved.iteration_log),
            "master_runtime_by_cut_quartile": _master_by_cut_quartile(
                improved.iteration_log
            ),
        },
        "comparison": {
            "master_runtime_reduction": _reduction(old_master, improved.master_runtime),
            "total_runtime_reduction": _reduction(_float(old, "elapsed_time"), improved.runtime),
            "iteration_change": improved.iterations - int(old["iteration"]),
            "accepted_cut_change": improved.metadata["accepted_cuts_total"]
            - int(old["cuts_added_total"]),
            "gap_change": improved.gap - _float(old, "gap") if improved.gap is not None else None,
        },
    }


def write_report(paired: dict[str, Any], renault: dict[str, Any] | None) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "git_commit": git_commit(ROOT),
        "scope": {
            "mathematical_model_changed": False,
            "uncertainty_set_changed": False,
            "certification_tolerance_changed": False,
            "formal_renault_result_replaced": False,
            "renault_210628_executed": False,
            "persistent_master_preexisting": True,
            "warm_start_preexisting": False,
            "exact_duplicate_filter_added": True,
            "near_duplicate_used_for_filtering": False,
        },
        "paired_correctness": paired,
        "renault_300s_diagnostic_replay": renault,
        "integrity_audit": "PASS"
        if paired["status"] == "PASS"
        and (renault is None or renault["low_risk"]["valid_bounds"])
        else "FAIL",
    }
    atomic_write_json(REPORT_JSON, report)
    csv_rows = paired["rows"]
    if renault is not None:
        for variant in ("baseline", "low_risk"):
            item = renault[variant]
            csv_rows.append(
                {
                    "case": "renault_210202_300s_diagnostic",
                    "variant": variant,
                    "status": renault.get("status") if variant == "low_risk" else "historical_time_cap",
                    "certified": False,
                    "objective": None,
                    "lower_bound": item["lower_bound"],
                    "upper_bound": item["upper_bound"],
                    "gap": item["gap"],
                    "iterations": item["iteration"],
                    "proposed_cuts": item.get("proposed_cuts"),
                    "accepted_cuts": item["accepted_cuts"],
                    "duplicate_cuts_rejected": item.get("duplicate_cuts_rejected"),
                    "master_runtime": item["master_runtime"],
                    "total_runtime": item["elapsed_time"],
                    "valid_bounds": item.get("valid_bounds", True),
                }
            )
    atomic_write_csv(REPORT_CSV, csv_rows, list(csv_rows[0]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renault-replay", action="store_true")
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE_REF)
    args = parser.parse_args()
    paired = run_paired()
    if paired["status"] != "PASS":
        write_report(paired, None)
        raise RuntimeError("Synthetic paired correctness failed; Renault replay blocked")
    renault = run_renault_replay(args.baseline_ref) if args.renault_replay else None
    report = write_report(paired, renault)
    print(json.dumps({"status": report["integrity_audit"], "report": str(REPORT_JSON)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
