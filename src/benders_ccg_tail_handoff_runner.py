from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import gurobipy as gp
import numpy as np

from .benders import solve_benders
from .ccg_tail_handoff import deduplicate_patterns, relative_gap, solve_ccg
from .config import load_config
from .experiment_protocol import atomic_write_csv, atomic_write_json, file_sha256, utc_now_iso
from .experiment_suite import _apply_selected_parameters, _apply_variant_config, _base_config
from .instance import generate_instance, load_instance
from .monolithic import solve_monolithic
from .scenarios import count_budget_scenarios


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/benders_ccg_tail_handoff_210202.yaml"
METHODS = ("pure_ccg", "handoff_300", "handoff_600", "handoff_900")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _path(value: str) -> Path:
    return REPO_ROOT / value


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, timeout=10
    ).strip()


def _verify_file(path: Path, expected_hash: str, label: str) -> None:
    _require(path.is_file(), f"Missing {label}: {path}")
    actual = file_sha256(path)
    _require(actual.lower() == expected_hash.lower(), f"{label} SHA-256 mismatch: {actual}")


def _resolve_solver(protocol: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    frozen = protocol["frozen_solver"]
    for label in ("protocol", "selected_parameters", "selected_candidate"):
        _verify_file(_path(frozen[f"{label}_path"]), frozen[f"{label}_sha256"], label)
    selected = _apply_selected_parameters(load_config(_path(frozen["protocol_path"])))
    base = _base_config(selected, "large", seed=0)
    method, _, resolved = _apply_variant_config(
        base, frozen["method"], selected["variant_settings"][frozen["variant"]]
    )
    return method, resolved


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    protocol = load_config(config_path)
    _require(protocol["protocol"] == "benders_ccg_tail_handoff_diagnostic_v1", "Wrong protocol")
    _require(protocol["diagnostic_only"] is True, "Protocol must remain diagnostic-only")
    _require(protocol["case_id"] == "210202", "Only Renault 210202 is authorized")
    _require(protocol["handoff_seconds"] == [300, 600, 900], "Handoff points changed")
    _require(int(protocol["total_wall_clock_limit_seconds"]) == 1800, "Total limit changed")
    _require(math.isclose(float(protocol["certification_tolerance"]), 1e-4), "Tolerance changed")
    for label in ("instance", "index_metadata", "eligibility"):
        _verify_file(_path(protocol[f"{label}_path"]), protocol[f"{label}_sha256"], label)
    eligibility = json.loads(_path(protocol["eligibility_path"]).read_text(encoding="utf-8"))
    full = np.asarray(eligibility["Full"], dtype=int)
    _require(protocol["eligibility_mode"] == "Full", "Only Full eligibility is authorized")
    _require(full.shape == (15, 12) and bool((full == 1).all()), "Full eligibility changed")
    instance = load_instance(_path(protocol["instance_path"]))
    dimensions = protocol["expected_dimensions"]
    _require(
        (instance.num_warehouses, instance.num_products, instance.num_regions)
        == (dimensions["I"], dimensions["J"], dimensions["R"])
        == (15, 8, 12),
        "Instance dimensions changed",
    )
    _require(int(protocol["gamma"]) == 2 and count_budget_scenarios(instance, 2) == 4657, "Uncertainty set changed")
    _require(math.isclose(instance.budget, float(protocol["expected_budget"]), rel_tol=0.0, abs_tol=1e-9), "Budget changed")
    method, solver = _resolve_solver(protocol)
    _require(method == "adaptive_gap_gamma_benders", "Frozen Benders method changed")
    _require(solver["benders"]["time_limit"] == 1800, "Frozen Benders time limit changed")
    _require(solver["benders"]["tol"] == 1e-4, "Frozen certification tolerance changed")
    _require(solver["robust"]["gamma_target"] == 2 and solver["robust"]["gamma_schedule"] == [2], "Frozen Gamma policy changed")
    algorithm = solver["algorithm"]
    _require(algorithm["subproblem_mode"] == "robust_dual_milp", "Robust separation changed")
    _require(algorithm["cut_strengthening_policy"] == "core_point", "PR #82 Benders changed")
    _require(algorithm["max_cuts_per_iteration"] == 1 and algorithm["cut_selection_enabled"] is False, "Cut policy changed")
    _require(algorithm["final_certification_enabled"] is True, "Certification logic changed")
    return {"protocol": protocol, "instance": instance, "method": method, "solver": solver, "config_path": config_path}


def _synthetic_config() -> dict[str, Any]:
    return {
        "instance": {"num_warehouses": 2, "num_products": 2, "num_regions": 2, "budget_factor": 0.8},
        "robust": {"gamma_target": 2, "gamma_schedule": [2], "max_scenarios": 20, "exact_scenarios": True},
        "algorithm": {"subproblem_mode": "robust_dual_milp", "final_certification_enabled": True, "final_certification_no_cut_patience": 2},
        "benders": {"max_iterations": 200, "tol": 1e-8, "initial_mip_gap": 0.0, "final_mip_gap": 0.0, "time_limit": 30, "output_flag": False},
    }


def run_synthetic_gate(context: dict[str, Any]) -> dict[str, Any]:
    root = _path(context["protocol"]["output_root"])
    target = root / "synthetic_correctness.json"
    _require(not target.exists(), "Synthetic gate was already executed")
    cases = []
    for seed in context["protocol"]["synthetic_seeds"]:
        config = _synthetic_config()
        instance = generate_instance(config, seed=int(seed))
        exact = solve_monolithic(config, instance)
        benders = solve_benders(config, instance, "standard_benders")
        pure = solve_ccg(instance, gamma=2, time_limit=30, tolerance=1e-8)
        partial_config = deepcopy(config)
        partial_config["benders"]["max_iterations"] = 1
        partial_config["diagnostics"] = {"record_adversarial_patterns": True}
        partial = solve_benders(partial_config, instance, "standard_benders")
        patterns = [row["adversarial_pattern"] for row in partial.iteration_log if row.get("adversarial_pattern") is not None]
        hybrid = solve_ccg(
            instance,
            gamma=2,
            time_limit=30,
            tolerance=1e-8,
            inherited_patterns=patterns,
            initial_y=partial.metadata["best_y_values"],
            initial_x=partial.metadata["best_x_values"],
            initial_upper_bound=partial.upper_bound,
        )
        objectives = {"exact": exact.objective, "benders": benders.upper_bound, "pure_ccg": pure.upper_bound, "hybrid": hybrid.upper_bound}
        passed = bool(
            exact.status == "optimal"
            and benders.status == "optimal"
            and pure.certified
            and hybrid.certified
            and exact.objective is not None
            and all(value is not None and abs(float(value) - exact.objective) <= 1e-6 for name, value in objectives.items() if name != "exact")
            and pure.lower_bound is not None and pure.lower_bound <= exact.objective + 1e-6
            and hybrid.lower_bound is not None and hybrid.lower_bound <= exact.objective + 1e-6
        )
        cases.append(
            {
                "seed": seed,
                "status": "PASS" if passed else "FAIL",
                "objectives": objectives,
                "benders_iterations": benders.iterations,
                "pure_ccg_iterations": pure.iterations,
                "hybrid_ccg_iterations": hybrid.iterations,
                "hybrid_inherited_scenarios": hybrid.inherited_scenario_count,
            }
        )
    report = {
        "schema": "benders_ccg_synthetic_correctness_v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in cases) else "FAIL",
        "objective_absolute_tolerance": 1e-6,
        "cases": cases,
        "git_commit": _git_commit(),
        "timestamp": utc_now_iso(),
    }
    atomic_write_json(target, report)
    _require(report["status"] == "PASS", "Synthetic correctness failed; Renault is forbidden")
    return report


def _benders_best_incumbent_time(result: Any) -> float | None:
    if result.upper_bound is None:
        return None
    for row in result.iteration_log:
        if row.get("upper_bound") is not None and math.isclose(float(row["upper_bound"]), float(result.upper_bound), rel_tol=0.0, abs_tol=1e-8):
            return float(row["elapsed_time"])
    return None


def _write_method(root: Path, name: str, payload: dict[str, Any]) -> None:
    target = root / "methods" / f"{name}.json"
    _require(not target.exists(), f"Refusing to repeat completed method: {name}")
    atomic_write_json(target, payload)
    rows = payload["ccg"]["iteration_log"]
    if rows:
        atomic_write_csv(root / "iteration_logs" / f"{name}_ccg.csv", rows, list(rows[0]))
    benders_rows = payload.get("benders", {}).get("iteration_log", [])
    if benders_rows:
        atomic_write_csv(root / "iteration_logs" / f"{name}_benders.csv", benders_rows, list(benders_rows[0]))


def run_method(context: dict[str, Any], method_name: str) -> dict[str, Any]:
    _require(method_name in METHODS, f"Unknown method: {method_name}")
    root = _path(context["protocol"]["output_root"])
    gate = root / "synthetic_correctness.json"
    _require(gate.is_file() and json.loads(gate.read_text(encoding="utf-8"))["status"] == "PASS", "Synthetic PASS is required")
    total_limit = float(context["protocol"]["total_wall_clock_limit_seconds"])
    tolerance = float(context["protocol"]["certification_tolerance"])
    started = utc_now_iso()
    benders_payload: dict[str, Any] = {}

    if method_name == "pure_ccg":
        ccg = solve_ccg(context["instance"], gamma=2, time_limit=total_limit, tolerance=tolerance)
        total_runtime = ccg.runtime
        time_to_best = ccg.time_to_best_incumbent
    else:
        handoff_seconds = int(method_name.rsplit("_", 1)[1])
        benders_config = deepcopy(context["solver"])
        benders_config["benders"]["time_limit"] = handoff_seconds
        benders_config["diagnostics"] = {"record_adversarial_patterns": True}
        benders = solve_benders(benders_config, context["instance"], context["method"])
        _require(benders.metadata.get("valid_UB") is True and benders.upper_bound is not None, "Benders handoff lacks a valid UB")
        raw_patterns = [row["adversarial_pattern"] for row in benders.iteration_log if row.get("adversarial_pattern") is not None]
        inherited, duplicates = deduplicate_patterns(raw_patterns)
        remaining = max(1e-3, total_limit - benders.runtime)
        ccg = solve_ccg(
            context["instance"],
            gamma=2,
            time_limit=remaining,
            tolerance=tolerance,
            inherited_patterns=inherited,
            initial_y=benders.metadata["best_y_values"],
            initial_x=benders.metadata["best_x_values"],
            initial_upper_bound=benders.upper_bound,
        )
        total_runtime = benders.runtime + ccg.runtime
        benders_best_time = _benders_best_incumbent_time(benders)
        if ccg.time_to_best_incumbent is not None and ccg.time_to_best_incumbent > 0.0:
            time_to_best = benders.runtime + ccg.time_to_best_incumbent
        else:
            time_to_best = benders_best_time
        benders_payload = {
            "status": benders.status,
            "runtime": benders.runtime,
            "iterations": benders.iterations,
            "cuts": benders.cuts,
            "lower_bound": benders.lower_bound,
            "upper_bound": benders.upper_bound,
            "gap": benders.gap,
            "time_to_best_incumbent": benders_best_time,
            "raw_discovered_pattern_count": len(raw_patterns),
            "unique_inherited_pattern_count": len(inherited),
            "duplicate_patterns_removed": duplicates,
            "iteration_log": benders.iteration_log,
        }

    payload = {
        "schema": "benders_ccg_tail_handoff_method_v1",
        "method": method_name,
        "diagnostic_only": True,
        "provenance": {
            "git_commit": _git_commit(),
            "config_sha256": file_sha256(context["config_path"]),
            "input_sha256": context["protocol"]["instance_sha256"],
            "solver_version": ".".join(str(value) for value in gp.gurobi.version()),
            "started_at": started,
            "completed_at": utc_now_iso(),
        },
        "bound_validity": {
            "benders_bounds_are_phase_local": True,
            "ccg_lower_bound_from_restricted_scenario_master_only": True,
            "upper_bound_from_exact_adversarial_bound": True,
            "benders_and_ccg_lower_bounds_combined": False,
        },
        "benders": benders_payload,
        "ccg": ccg.to_dict(),
        "combined": {
            "certified": ccg.certified,
            "status": ccg.status,
            "total_runtime": total_runtime,
            "final_lower_bound": ccg.lower_bound,
            "final_upper_bound": ccg.upper_bound,
            "final_gap": relative_gap(ccg.upper_bound, ccg.lower_bound),
            "time_to_best_incumbent": time_to_best,
            "time_to_certification": (None if ccg.time_to_certification is None else total_runtime - ccg.runtime + ccg.time_to_certification),
            "master_dominated_ccg_tail": ccg.master_runtime > ccg.adversarial_runtime,
        },
    }
    _write_method(root, method_name, payload)
    return payload


def aggregate(context: dict[str, Any]) -> dict[str, Any]:
    root = _path(context["protocol"]["output_root"])
    methods = {
        name: json.loads((root / "methods" / f"{name}.json").read_text(encoding="utf-8"))
        for name in METHODS
    }
    baseline = context["protocol"]["benders_only_pr82"]
    baseline_gap = float(baseline["gap"])
    rows = []
    for name, payload in methods.items():
        combined = payload["combined"]
        ccg = payload["ccg"]
        gap = combined["final_gap"]
        rows.append(
            {
                "method": name,
                "certified": combined["certified"],
                "total_runtime": combined["total_runtime"],
                "final_LB": combined["final_lower_bound"],
                "final_UB": combined["final_upper_bound"],
                "final_gap": gap,
                "gap_reduction_vs_benders_only": None if gap is None else (baseline_gap - gap) / baseline_gap,
                "benders_runtime": payload.get("benders", {}).get("runtime", 0.0),
                "ccg_runtime": ccg["runtime"],
                "ccg_iterations": ccg["iterations"],
                "ccg_scenarios_added": ccg["scenarios_added"],
                "inherited_scenarios": ccg["inherited_scenario_count"],
                "final_active_scenarios": ccg["final_active_scenario_count"],
                "ccg_master_runtime": ccg["master_runtime"],
                "adversarial_runtime": ccg["adversarial_runtime"],
                "master_dominated_ccg_tail": combined["master_dominated_ccg_tail"],
            }
        )
    best = min(rows, key=lambda row: (not row["certified"], row["total_runtime"] if row["certified"] else row["final_gap"] if row["final_gap"] is not None else float("inf")))
    best_reduction = max((row["gap_reduction_vs_benders_only"] or -float("inf")) for row in rows)
    if any(row["certified"] for row in rows):
        outcome = "STRONG_SUCCESS"
    elif best_reduction >= 0.30:
        outcome = "SUBSTANTIAL_IMPROVEMENT"
    elif best_reduction < 0.10:
        outcome = "NO_MATERIAL_BENEFIT"
    else:
        outcome = "MARGINAL"
    report = {
        "schema": "benders_ccg_tail_handoff_summary_v1",
        "diagnostic_only": True,
        "synthetic_correctness": json.loads((root / "synthetic_correctness.json").read_text(encoding="utf-8")),
        "benders_only_pr82": baseline,
        "methods": rows,
        "best_method": best["method"],
        "outcome": outcome,
        "freeze_review_recommended": outcome in {"STRONG_SUCCESS", "SUBSTANTIAL_IMPROVEMENT"},
        "git_commit": _git_commit(),
        "timestamp": utc_now_iso(),
    }
    atomic_write_json(root / "summary.json", report)
    atomic_write_csv(root / "summary.csv", rows, list(rows[0]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the predeclared Benders-to-CCG tail handoff diagnostic")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("synthetic", *METHODS, "aggregate"), required=True)
    args = parser.parse_args()
    context = preflight(args.config.resolve())
    if args.stage == "synthetic":
        result = run_synthetic_gate(context)
    elif args.stage == "aggregate":
        result = aggregate(context)
    else:
        result = run_method(context, args.stage)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
