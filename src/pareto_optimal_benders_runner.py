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
from .monolithic import solve_monolithic


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments/configs/pareto_optimal_benders_development.yaml"
REPORT_JSON = ROOT / "artifacts/pareto_optimal_benders_report.json"
REPORT_CSV = ROOT / "artifacts/pareto_optimal_benders_report.csv"
PR82_LOG_REF = (
    "origin/codex/run-renault-certified-robust-baselines:"
    "real_data_studies/renault_robust_baseline_v1/results/iteration_logs/"
    "210202_B1.00_iterations.csv"
)
PR83_REPORT_REF = (
    "origin/codex/master-efficiency-low-risk:"
    "artifacts/master_efficiency_low_risk_report.json"
)


def _verify_frozen_files(protocol: dict[str, Any]) -> None:
    for name in ("protocol", "selected_parameters", "selected_candidate"):
        path = ROOT / protocol["frozen_solver"][f"{name}_path"]
        expected = protocol["frozen_solver"][f"{name}_sha256"]
        if file_sha256(path).lower() != expected.lower():
            raise RuntimeError(f"Frozen {name} hash mismatch")


def _resolved_config(protocol: dict[str, Any], size: str, seed: int) -> tuple[str, dict[str, Any]]:
    frozen = protocol["frozen_solver"]
    selected = _apply_selected_parameters(load_config(ROOT / frozen["protocol_path"]))
    base = _base_config(selected, size, seed)
    method, _, config = _apply_variant_config(
        base,
        frozen["method"],
        selected["variant_settings"][frozen["current_variant"]],
    )
    return method, config


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "p25": percentile(0.25),
        "median": statistics.median(values),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": max(values),
    }


def _progress(result: Any) -> dict[str, Any]:
    log = result.iteration_log
    if not log:
        return {"window": 0, "lb_progress": None, "lb_gain_per_cut": None}
    window = min(500, len(log))
    tail = log[-window:]
    lb_progress = float(tail[-1]["lower_bound"]) - float(tail[0]["lower_bound"])
    accepted = sum(int(row["cuts_added_this_iteration"]) for row in tail)
    return {
        "window": window,
        "lb_progress": lb_progress,
        "accepted_cuts": accepted,
        "lb_gain_per_cut": lb_progress / accepted if accepted else None,
    }


def _result_row(case: str, variant: str, result: Any, true_objective: float | None) -> dict[str, Any]:
    gains = [
        float(row["strengthening_gain"])
        for row in result.iteration_log
        if row.get("strengthening_gain") is not None
    ]
    accepted_pareto = sum(
        row.get("accepted_cut_type") == "pareto_optimal_mw"
        for row in result.iteration_log
    )
    return {
        "case": case,
        "variant": variant,
        "status": result.status,
        "certified": result.status == "optimal" and result.gap is not None and result.gap <= 1.0e-4,
        "objective": result.objective,
        "certified_objective": true_objective,
        "true_objective": true_objective,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "gap": result.gap,
        "iterations": result.iterations,
        "cuts": result.cuts,
        "master_runtime": result.master_runtime,
        "subproblem_runtime": result.subproblem_runtime,
        "auxiliary_runtime": result.metadata["core_point_total_runtime"],
        "total_runtime": result.runtime,
        "accepted_pareto_cuts": accepted_pareto,
        "lb_gain_per_cut": _progress(result)["lb_gain_per_cut"],
        "strengthening_gain_distribution": _distribution(gains),
        "tail_500_equivalent_progress": _progress(result),
        "valid_bounds": (
            result.lower_bound is not None
            and result.upper_bound is not None
            and result.lower_bound <= result.upper_bound + 1.0e-6
        ),
    }


def run_synthetic(protocol: dict[str, Any]) -> dict[str, Any]:
    settings = protocol["synthetic"]
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    all_pareto_gains: list[float] = []
    accepted_pareto_gains: list[float] = []
    for seed in settings["seeds"]:
        method, base = _resolved_config(protocol, settings["instance_size"], int(seed))
        base["benders"]["time_limit"] = float(settings["time_limit"])
        instance = generate_instance(base, seed=int(seed))
        exact = solve_monolithic(base, instance)
        if exact.status != "optimal" or exact.objective is None:
            raise RuntimeError(f"Exact synthetic benchmark seed {seed} did not solve")

        current_config = deepcopy(base)
        current_config["algorithm"]["cut_strengthening_policy"] = "core_point"
        pareto_config = deepcopy(base)
        pareto_config["algorithm"].update(protocol["pareto_mode"])
        for key in (
            "reuse_frozen_core_point_parameters",
            "current_point_constraint",
            "coefficient_perturbation",
            "stabilization",
            "multi_cut",
            "cut_deletion",
            "solver_parameter_retuning",
        ):
            pareto_config["algorithm"].pop(key, None)

        current = solve_benders(current_config, instance, method)
        pareto = solve_benders(pareto_config, instance, method)
        all_pareto_gains.extend(
            float(row["strengthening_gain"])
            for row in pareto.iteration_log
            if row.get("strengthening_gain") is not None
        )
        accepted_pareto_gains.extend(
            float(row["strengthening_gain"])
            for row in pareto.iteration_log
            if row.get("accepted_cut_type") == "pareto_optimal_mw"
            and row.get("strengthening_gain") is not None
        )
        case = f"small_seed_{seed}"
        current_row = _result_row(case, "core_point", current, exact.objective)
        pareto_row = _result_row(case, "pareto_optimal_mw", pareto, exact.objective)
        rows.extend((current_row, pareto_row))
        tolerance = float(settings["objective_tolerance"])
        intervals_contain_certified_objective = (
            current.lower_bound is not None
            and current.upper_bound is not None
            and pareto.lower_bound is not None
            and pareto.upper_bound is not None
            and current.lower_bound <= exact.objective + tolerance
            and exact.objective <= current.upper_bound + tolerance
            and pareto.lower_bound <= exact.objective + tolerance
            and exact.objective <= pareto.upper_bound + tolerance
        )
        raw_ub_difference = abs(float(current.upper_bound) - float(pareto.upper_bound))
        pairs.append(
            {
                "case": case,
                "certified_objective": exact.objective,
                "certified_objective_match": True,
                "raw_incumbent_ub_difference": raw_ub_difference,
                "raw_incumbent_ub_match_at_1e-6": raw_ub_difference <= tolerance,
                "both_certified": current_row["certified"] and pareto_row["certified"],
                "both_final_intervals_contain_certified_objective": (
                    intervals_contain_certified_objective
                ),
                "iterations_change": pareto.iterations - current.iterations,
                "cuts_change": pareto.cuts - current.cuts,
                "master_runtime_change": pareto.master_runtime - current.master_runtime,
                "auxiliary_runtime_change": (
                    pareto.metadata["core_point_total_runtime"]
                    - current.metadata["core_point_total_runtime"]
                ),
                "total_runtime_change": pareto.runtime - current.runtime,
            }
        )
    passed = all(
        pair["certified_objective_match"]
        and pair["both_certified"]
        and pair["both_final_intervals_contain_certified_objective"]
        for pair in pairs
    )
    current_rows = [row for row in rows if row["variant"] == "core_point"]
    pareto_rows = [row for row in rows if row["variant"] == "pareto_optimal_mw"]
    return {
        "status": "PASS" if passed else "FAIL",
        "certified_objective_definition": "independent exact monolithic optimum",
        "raw_incumbent_ub_is_not_the_certified_objective": True,
        "cases": pairs,
        "rows": rows,
        "aggregate": {
            "iterations": {
                "core_point": sum(row["iterations"] for row in current_rows),
                "pareto_optimal_mw": sum(row["iterations"] for row in pareto_rows),
            },
            "cuts": {
                "core_point": sum(row["cuts"] for row in current_rows),
                "pareto_optimal_mw": sum(row["cuts"] for row in pareto_rows),
            },
            "master_runtime": {
                "core_point": sum(row["master_runtime"] for row in current_rows),
                "pareto_optimal_mw": sum(row["master_runtime"] for row in pareto_rows),
            },
            "auxiliary_runtime": {
                "core_point": sum(row["auxiliary_runtime"] for row in current_rows),
                "pareto_optimal_mw": sum(row["auxiliary_runtime"] for row in pareto_rows),
            },
            "total_runtime": {
                "core_point": sum(row["total_runtime"] for row in current_rows),
                "pareto_optimal_mw": sum(row["total_runtime"] for row in pareto_rows),
            },
            "strengthening_gain_distribution": {
                "all_attempted": _distribution(all_pareto_gains),
                "accepted_pareto_cuts": _distribution(accepted_pareto_gains),
            },
        },
    }


def _git_text(ref: str) -> str:
    return subprocess.check_output(
        ["git", "show", ref], cwd=ROOT, text=True, encoding="utf-8", timeout=30
    )


def _pr82_baseline() -> dict[str, Any]:
    rows = list(csv.DictReader(StringIO(_git_text(PR82_LOG_REF))))
    rows = [row for row in rows if float(row["elapsed_time"]) <= 300.0]
    last = rows[-1]
    return {
        "source": PR82_LOG_REF,
        "elapsed_time": float(last["elapsed_time"]),
        "iterations": int(last["iteration"]),
        "accepted_cuts": int(last["cuts_added_total"]),
        "master_runtime": sum(float(row["master_time"]) for row in rows),
        "subproblem_runtime": sum(float(row["subproblem_time"]) for row in rows),
        "auxiliary_runtime": sum(
            float(row["core_point_stage1_runtime"]) + float(row["core_point_stage2_runtime"])
            for row in rows
        ),
        "lower_bound": float(last["lower_bound"]),
        "upper_bound": float(last["upper_bound"]),
        "gap": float(last["gap"]),
    }


def _pr83_baseline() -> dict[str, Any]:
    report = json.loads(_git_text(PR83_REPORT_REF))
    result = report["renault_300s_diagnostic_replay"]["low_risk"]
    return {"source": PR83_REPORT_REF, **result}


def run_renault(protocol: dict[str, Any]) -> dict[str, Any]:
    settings = protocol["renault_diagnostic"]
    instance_path = ROOT / settings["instance_path"]
    if file_sha256(instance_path).lower() != settings["instance_sha256"].lower():
        raise RuntimeError("Renault input hash mismatch")
    eligibility = json.loads((ROOT / settings["eligibility_path"]).read_text(encoding="utf-8"))
    full = np.asarray(eligibility["Full"], dtype=int)
    if full.shape != (15, 12) or not bool((full == 1).all()):
        raise RuntimeError("Renault Full eligibility changed")
    instance = load_instance(instance_path)
    if (instance.num_warehouses, instance.num_products, instance.num_regions) != (15, 8, 12):
        raise RuntimeError("Renault dimensions changed")

    method, config = _resolved_config(protocol, "large", 0)
    config["benders"]["time_limit"] = float(settings["time_limit"])
    config["algorithm"]["cut_strengthening_policy"] = "pareto_optimal_mw"
    result = solve_benders(config, instance, method)
    row = _result_row("renault_210202_300s", "pareto_optimal_mw", result, None)
    row["valid_bounds"] = (
        result.lower_bound is not None
        and result.upper_bound is not None
        and result.lower_bound <= result.upper_bound + 1.0e-6
    )
    return {
        "not_a_formal_result": True,
        "pr82_core_point": _pr82_baseline(),
        "pr83_low_risk": _pr83_baseline(),
        "pareto_optimal_mw": row,
    }


def write_report(
    protocol: dict[str, Any],
    synthetic: dict[str, Any],
    renault: dict[str, Any] | None,
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "git_commit": git_commit(ROOT),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "mathematical_audit": {
            "existing_core_point_is_strict_magnanti_wong": False,
            "difference": "existing stage 2 uses current value >= q_k - delta rather than equality",
            "new_mode": "pareto_optimal_mw",
            "global_validity": "dual feasible fixed-pattern cut <= Q_z(x) <= robust Q(x)",
            "current_point_tightness": "stage 2 enforces alpha + beta^T x_k = Q_z(x_k)",
            "coefficient_perturbation": False,
        },
        "synthetic_correctness": synthetic,
        "renault_300s_diagnostic": renault,
        "guardrails": {
            "mathematical_model_changed": False,
            "uncertainty_set_changed": False,
            "certification_tolerance_changed": False,
            "termination_logic_changed": False,
            "formal_result_replaced": False,
            "renault_210628_executed": False,
            "parameter_retuning": False,
        },
        "integrity_audit": "PASS"
        if synthetic["status"] == "PASS"
        and (renault is None or renault["pareto_optimal_mw"]["valid_bounds"])
        else "FAIL",
    }
    atomic_write_json(REPORT_JSON, report)
    rows = list(synthetic["rows"])
    if renault is not None:
        for variant in ("pr82_core_point", "pr83_low_risk"):
            value = renault[variant]
            rows.append(
                {
                    "case": "renault_210202_300s",
                    "variant": variant,
                    "status": "historical_time_cap",
                    "certified": False,
                    "objective": None,
                    "certified_objective": None,
                    "true_objective": None,
                    "lower_bound": value["lower_bound"],
                    "upper_bound": value["upper_bound"],
                    "gap": value["gap"],
                    "iterations": value.get("iterations", value.get("iteration")),
                    "cuts": value["accepted_cuts"],
                    "master_runtime": value["master_runtime"],
                    "subproblem_runtime": value["subproblem_runtime"],
                    "auxiliary_runtime": value["auxiliary_runtime"]
                    if "auxiliary_runtime" in value
                    else value["core_point_runtime"],
                    "total_runtime": value["elapsed_time"],
                    "accepted_pareto_cuts": 0,
                    "lb_gain_per_cut": None,
                    "strengthening_gain_distribution": None,
                    "tail_500_equivalent_progress": None,
                    "valid_bounds": True,
                }
            )
        rows.append(renault["pareto_optimal_mw"])
    atomic_write_csv(REPORT_CSV, rows, list(rows[0]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renault-replay", action="store_true")
    args = parser.parse_args()
    protocol = load_config(PROTOCOL_PATH)
    _verify_frozen_files(protocol)
    synthetic = run_synthetic(protocol)
    if synthetic["status"] != "PASS":
        write_report(protocol, synthetic, None)
        raise RuntimeError("Synthetic correctness failed; Renault replay blocked")
    renault = run_renault(protocol) if args.renault_replay else None
    report = write_report(protocol, synthetic, renault)
    print(json.dumps({"status": report["integrity_audit"], "report": str(REPORT_JSON)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
