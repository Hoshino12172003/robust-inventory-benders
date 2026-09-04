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

from . import subproblem as recourse_module
from .benders import calculate_global_gap, solve_benders
from .config import load_config
from .experiment_protocol import atomic_write_csv, atomic_write_json, file_sha256, utc_now_iso
from .experiment_suite import _apply_selected_parameters, _apply_variant_config, _base_config
from .instance import InventoryInstance, load_instance
from .robust_dual_subproblem import RobustDualSubproblemResult, solve_robust_dual_subproblem
from .scenarios import DemandScenario, count_budget_scenarios


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = (
    REPO_ROOT / "experiments/configs/renault_robust_baseline_210202.yaml",
    REPO_ROOT / "experiments/configs/renault_robust_baseline_210628.yaml",
)
EXPECTED_CASE_ORDER = ("210202", "210628")


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, timeout=10
    ).strip()


def _absolute(path: str) -> Path:
    return REPO_ROOT / path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _verify_file(path: Path, expected_hash: str, label: str) -> None:
    _require(path.is_file(), f"Missing {label}: {path}")
    actual = file_sha256(path)
    _require(actual.lower() == expected_hash.lower(), f"{label} SHA-256 mismatch: {actual}")


def _resolve_solver_config(protocol: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    frozen = protocol["frozen_solver"]
    for prefix in ("protocol", "selected_parameters", "selected_candidate"):
        _verify_file(_absolute(frozen[f"{prefix}_path"]), frozen[f"{prefix}_sha256"], prefix)
    flat = load_config(_absolute(frozen["protocol_path"]))
    selected = _apply_selected_parameters(flat)
    variant_name = frozen["variant"]
    _require(variant_name in selected["variant_settings"], f"Unknown frozen variant: {variant_name}")
    base = _base_config(selected, "large", seed=0)
    solver_method, _, resolved = _apply_variant_config(
        base, frozen["method"], selected["variant_settings"][variant_name]
    )
    return solver_method, resolved


def _verify_full_eligibility(protocol: dict[str, Any]) -> dict[str, Any]:
    path = _absolute(protocol["eligibility_path"])
    _verify_file(path, protocol["eligibility_sha256"], "eligibility")
    eligibility = json.loads(path.read_text(encoding="utf-8"))
    full = np.asarray(eligibility["Full"], dtype=int)
    _require(protocol["eligibility_mode"] == "Full", "Only Full eligibility is authorized")
    _require(full.shape == (15, 12) and bool((full == 1).all()), "Full eligibility must be 15x12 all ones")
    return eligibility


def preflight(config_path: Path) -> dict[str, Any]:
    protocol = load_config(config_path)
    _require(protocol["protocol"] == "renault_robust_baseline_v1", "Wrong protocol")
    _require(protocol["case_id"] in EXPECTED_CASE_ORDER, "Unauthorized Renault case")
    instance_path = _absolute(protocol["instance_path"])
    metadata_path = _absolute(protocol["index_metadata_path"])
    _verify_file(instance_path, protocol["instance_sha256"], "instance")
    _verify_file(metadata_path, protocol["index_metadata_sha256"], "index metadata")
    _verify_full_eligibility(protocol)
    instance = load_instance(instance_path)
    dims = protocol["expected_dimensions"]
    _require(
        (instance.num_warehouses, instance.num_products, instance.num_regions)
        == (dims["I"], dims["J"], dims["R"]) == (15, 8, 12),
        "Instance dimensions changed",
    )
    _require(int(protocol["gamma"]) == 2, "Gamma must equal 2")
    _require(count_budget_scenarios(instance, 2) == int(protocol["expected_complete_scenarios"]) == 4657, "Scenario count changed")
    _require(math.isclose(instance.budget, float(protocol["expected_budget"]), rel_tol=0.0, abs_tol=1e-9), "Budget changed")
    solver_method, solver_config = _resolve_solver_config(protocol)
    _require(solver_method == "adaptive_gap_gamma_benders", "Frozen solver method changed")
    _require(solver_config["robust"]["gamma_target"] == 2, "Resolved Gamma changed")
    _require(solver_config["robust"]["gamma_schedule"] == [2], "Gamma continuation is forbidden")
    _require(solver_config["algorithm"]["subproblem_mode"] == "robust_dual_milp", "Wrong subproblem mode")
    _require(solver_config["algorithm"]["cut_strengthening_policy"] == "core_point", "Frozen algorithm changed")
    _require(solver_config["algorithm"]["final_certification_enabled"] is True, "Certification must be enabled")
    return {
        "protocol": protocol,
        "config_path": config_path,
        "config_sha256": file_sha256(config_path),
        "instance": instance,
        "solver_method": solver_method,
        "solver_config": solver_config,
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
    }


def _solve_recourse_with_primal(
    instance: InventoryInstance,
    scenario: DemandScenario,
    x_values: dict[tuple[int, int], float],
) -> tuple[Any, gp.Model]:
    captured: list[gp.Model] = []
    real_model = recourse_module.gp.Model

    def model_factory(*args: Any, **kwargs: Any) -> gp.Model:
        model = real_model(*args, **kwargs)
        captured.append(model)
        return model

    recourse_module.gp.Model = model_factory
    try:
        result = recourse_module.solve_recourse_subproblem(
            instance, scenario, x_values, output_flag=False
        )
    finally:
        recourse_module.gp.Model = real_model
    _require(len(captured) == 1, "Could not capture the solved recourse model")
    return result, captured[0]


def _provenance(
    protocol: dict[str, Any], config_hash: str, commit: str, timestamp: str
) -> dict[str, Any]:
    return {
        "input_sha256": protocol["instance_sha256"],
        "git_commit": commit,
        "config_sha256": config_hash,
        "timestamp": timestamp,
        "solver_version": ".".join(str(value) for value in gp.gurobi.version()),
    }


def _solution_arrays(instance: InventoryInstance, result: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(result.metadata["best_y_values"], dtype=float)
    x = np.asarray(result.metadata["best_x_values"], dtype=float)
    _require(y.shape == (instance.num_warehouses,), "Missing complete y solution")
    _require(x.shape == (instance.num_warehouses, instance.num_products), "Missing complete x solution")
    _require(bool(np.isfinite(y).all() and np.isfinite(x).all()), "Non-finite first-stage solution")
    return y, x


def _worst_case(
    instance: InventoryInstance,
    x: np.ndarray,
    time_limit: float,
) -> RobustDualSubproblemResult:
    values = {(i, j): float(x[i, j]) for i in instance.I for j in instance.J}
    result = solve_robust_dual_subproblem(
        instance,
        values,
        gamma=2,
        time_limit=time_limit,
        mip_gap=0.0,
        output_flag=False,
    )
    _require(result.status == "optimal" and result.has_incumbent, "Exact worst-case recertification failed")
    _require(result.objective is not None, "Worst-case objective is missing")
    return result


def _extract_recourse(
    instance: InventoryInstance,
    model: gp.Model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.empty((instance.num_warehouses, instance.num_regions, instance.num_products))
    u = np.empty((instance.num_regions, instance.num_products))
    e = np.empty(instance.num_products)
    for i in instance.I:
        for r in instance.R:
            for j in instance.J:
                q[i, r, j] = model.getVarByName(f"q[{i},{r},{j}]").X
    for r in instance.R:
        for j in instance.J:
            u[r, j] = model.getVarByName(f"u[{r},{j}]").X
    for j in instance.J:
        e[j] = model.getVarByName(f"e[{j}]").X
    return q, u, e


def _integrity_audit(
    instance: InventoryInstance,
    y: np.ndarray,
    x: np.ndarray,
    demand: np.ndarray,
    q: np.ndarray,
    u: np.ndarray,
    e: np.ndarray,
    dual_objective: float,
    recourse_objective: float,
    tolerance: float,
) -> dict[str, Any]:
    fixed = np.asarray(instance.fixed_cost)
    holding = np.asarray(instance.inventory_cost)
    volume = np.asarray(instance.volume)
    capacity = np.asarray(instance.capacity)
    inventory_ub = np.asarray(instance.inventory_ub)
    service_level = np.asarray(instance.service_level)
    first_stage = float(fixed @ y + np.sum(holding * x))
    violations = {
        "budget": max(0.0, first_stage - instance.budget),
        "capacity": float(np.maximum(0.0, np.sum(x * volume[None, :], axis=1) - capacity * y).max()),
        "logic": float(np.maximum(0.0, x - inventory_ub * y[:, None]).max()),
        "recourse_demand": float(np.maximum(0.0, demand - q.sum(axis=0) - u).max()),
        "recourse_supply": float(np.maximum(0.0, q.sum(axis=1) - x).max()),
        "recourse_service": float(
            np.maximum(
                0.0,
                u.sum(axis=0) - e - (1.0 - service_level) * demand.sum(axis=0),
            ).max()
        ),
        "nonnegativity": max(
            float(np.maximum(0.0, -y).max()),
            float(np.maximum(0.0, -x).max()),
            float(np.maximum(0.0, -q).max()),
            float(np.maximum(0.0, -u).max()),
            float(np.maximum(0.0, -e).max()),
        ),
        "primal_dual_objective": abs(recourse_objective - dual_objective),
    }
    scaled_duality_tolerance = tolerance * max(1.0, abs(recourse_objective), abs(dual_objective))
    constraint_pass = all(value <= tolerance for key, value in violations.items() if key != "primal_dual_objective")
    passed = constraint_pass and violations["primal_dual_objective"] <= scaled_duality_tolerance
    return {
        "status": "PASS" if passed else "FAIL",
        "tolerance": tolerance,
        "scaled_primal_dual_tolerance": scaled_duality_tolerance,
        "max_absolute_violations": violations,
    }


def _case_outputs(context: dict[str, Any], commit: str) -> dict[str, Any]:
    protocol = context["protocol"]
    instance: InventoryInstance = context["instance"]
    started_at = utc_now_iso()
    result = solve_benders(deepcopy(context["solver_config"]), instance, context["solver_method"])
    y, x = _solution_arrays(instance, result)
    first_stage = float(np.asarray(instance.fixed_cost) @ y + np.sum(np.asarray(instance.inventory_cost) * x))
    worst = _worst_case(instance, x, float(context["solver_config"]["benders"]["time_limit"]))
    demand = np.asarray(instance.base_demand) + np.asarray(instance.demand_deviation) * np.asarray(
        [[round(worst.z_values[r, j]) for j in instance.J] for r in instance.R]
    )
    scenario = DemandScenario(
        name="certified_exact_worst_case",
        active_units=tuple(sorted(key for key, value in worst.z_values.items() if round(value) == 1)),
        demand=tuple(tuple(float(value) for value in row) for row in demand),
    )
    x_values = {(i, j): float(x[i, j]) for i in instance.I for j in instance.J}
    recourse_result, recourse_model = _solve_recourse_with_primal(instance, scenario, x_values)
    q, u, e = _extract_recourse(instance, recourse_model)
    tolerance = float(protocol["integrity_tolerance"])
    audit = _integrity_audit(
        instance, y, x, demand, q, u, e, float(worst.objective), recourse_result.objective, tolerance
    )
    final_upper = first_stage + float(worst.objective)
    final_lower = result.lower_bound
    final_gap = calculate_global_gap(final_upper, final_lower) if final_lower is not None else None
    anomalies = []
    if result.status != "optimal":
        anomalies.append(f"benders_status={result.status}")
    if int(result.metadata.get("num_subproblem_nonoptimal", 0)):
        anomalies.append("nonoptimal_benders_subproblem")
    if int(result.metadata.get("num_subproblem_without_incumbent", 0)):
        anomalies.append("benders_subproblem_without_incumbent")
    if worst.status != "optimal":
        anomalies.append(f"worst_case_status={worst.status}")
    if audit["status"] != "PASS":
        anomalies.append("integrity_audit_failed")
    certified = (
        not anomalies
        and final_gap is not None
        and final_gap <= float(context["solver_config"]["benders"]["tol"])
    )
    completed_at = utc_now_iso()
    provenance = _provenance(protocol, context["config_sha256"], commit, completed_at)
    metadata = context["metadata"]
    depot_codes = metadata["depot_order"]
    product_codes = metadata["product_order"]
    region_codes = metadata["region_order"]
    inventory_spending = float(np.sum(np.asarray(instance.inventory_cost) * x))
    capacity_used = np.sum(x * np.asarray(instance.volume)[None, :], axis=1)
    region_demand = demand.sum(axis=1)
    region_shortage = u.sum(axis=1)
    fill_rates = 1.0 - region_shortage / region_demand
    worst_fill = float(fill_rates.min())
    worst_regions = [str(region_codes[r]) for r in instance.R if abs(fill_rates[r] - worst_fill) <= tolerance]

    allocation_rows = [
        {
            "case_id": protocol["case_id"],
            "depot_code": depot_codes[i],
            "product_code": product_codes[j],
            "x_ij": float(x[i, j]),
            "inventory_cost_contribution": float(instance.inventory_cost[i][j] * x[i, j]),
            "share_of_total_inventory_spending": (
                float(instance.inventory_cost[i][j] * x[i, j] / inventory_spending) if inventory_spending else 0.0
            ),
            **provenance,
        }
        for i in instance.I
        for j in instance.J
    ]
    depot_rows = [
        {
            "case_id": protocol["case_id"],
            "depot_code": depot_codes[i],
            "y_i": float(y[i]),
            "total_allocated_inventory": float(x[i].sum()),
            "capacity_used": float(capacity_used[i]),
            "capacity_utilization": (
                float(capacity_used[i] / (instance.capacity[i] * y[i])) if y[i] > tolerance else 0.0
            ),
            "fixed_cost_contribution": float(instance.fixed_cost[i] * y[i]),
            **provenance,
        }
        for i in instance.I
    ]
    service_rows = [
        {
            "case_id": protocol["case_id"],
            "region": region_codes[r],
            "total_demand": float(region_demand[r]),
            "total_shortage": float(region_shortage[r]),
            "shortage_rate": float(region_shortage[r] / region_demand[r]),
            "fill_rate": float(fill_rates[r]),
            **provenance,
        }
        for r in instance.R
    ]
    pattern_rows = [
        {
            "case_id": protocol["case_id"],
            "scenario": scenario.name,
            "region": region_codes[r],
            "product_code": product_codes[j],
            "base_demand": float(instance.base_demand[r][j]),
            "deviation": float(instance.demand_deviation[r][j]),
            "shocked_demand": float(demand[r, j]),
            "tie_rule": protocol["worst_case_tie_rule"],
            **provenance,
        }
        for r, j in scenario.active_units
    ]
    flow_rows = [
        {
            "case_id": protocol["case_id"],
            "depot_code": depot_codes[i],
            "region": region_codes[r],
            "product_code": product_codes[j],
            "q_irj": float(q[i, r, j]),
            **provenance,
        }
        for i in instance.I
        for r in instance.R
        for j in instance.J
    ]
    summary = {
        "case_id": protocol["case_id"],
        "solver_status": result.status,
        "certification_status": "certified_robust_optimal" if certified else "unresolved",
        "certified": certified,
        "final_lower_bound": final_lower,
        "final_upper_bound": final_upper,
        "final_optimality_gap": final_gap,
        "runtime": result.runtime,
        "iterations": result.iterations,
        "cuts_added": result.cuts,
        "open_depots": ";".join(depot_codes[i] for i in instance.I if y[i] >= 0.5),
        "first_stage_spending": first_stage,
        "budget": instance.budget,
        "budget_utilization": first_stage / instance.budget,
        "budget_slack": instance.budget - first_stage,
        "total_shortage": float(u.sum()),
        "FR_min": worst_fill,
        "FR_avg": float(fill_rates.mean()),
        "worst_served_region": ";".join(worst_regions),
        "average_regional_shortage_rate": float(np.mean(region_shortage / region_demand)),
        "worst_case_shocked_components": ";".join(
            f"{region_codes[r]}:{product_codes[j]}" for r, j in scenario.active_units
        ),
        "integrity_audit": audit["status"],
        "anomalies": ";".join(anomalies),
        **provenance,
    }
    result_json = {
        "schema": "renault_certified_robust_baseline_result_v1",
        "summary": summary,
        "run_started_at": started_at,
        "run_completed_at": completed_at,
        "input": {
            "instance_path": protocol["instance_path"],
            "instance_sha256": protocol["instance_sha256"],
            "eligibility_mode": "Full",
            "gamma": 2,
            "complete_scenario_count": 4657,
        },
        "solver": {
            "method": context["solver_method"],
            "frozen_variant": protocol["frozen_solver"]["variant"],
            "status": result.status,
            "original_upper_bound": result.upper_bound,
            "postsolve_exact_upper_bound": final_upper,
            "lower_bound": final_lower,
            "gap": final_gap,
            "runtime": result.runtime,
            "iterations": result.iterations,
            "cuts_added": result.cuts,
            "metadata": result.metadata,
        },
        "first_stage": {
            "y": y.tolist(),
            "x": x.tolist(),
            "spending": first_stage,
            "budget_utilization": first_stage / instance.budget,
            "budget_slack": instance.budget - first_stage,
        },
        "worst_case": {
            "scenario": scenario.name,
            "active_units": [
                {"region": region_codes[r], "product_code": product_codes[j]} for r, j in scenario.active_units
            ],
            "demand": demand.tolist(),
            "tie_rule": protocol["worst_case_tie_rule"],
            "robust_dual_status": worst.status,
            "robust_dual_mip_gap": worst.mip_gap,
            "robust_dual_objective": worst.objective,
            "robust_dual_objective_bound": worst.objective_bound,
        },
        "recourse": {
            "q": q.tolist(),
            "u": u.tolist(),
            "e": e.tolist(),
            "objective": recourse_result.objective,
            "FR_min": worst_fill,
            "FR_avg": float(fill_rates.mean()),
        },
        "integrity_audit": audit,
        "provenance": provenance,
    }
    return {
        "summary": summary,
        "result_json": result_json,
        "audit": {"case_id": protocol["case_id"], **audit, **provenance},
        "allocation_rows": allocation_rows,
        "depot_rows": depot_rows,
        "service_rows": service_rows,
        "pattern_rows": pattern_rows,
        "flow_rows": flow_rows,
        "iteration_log": result.iteration_log,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        atomic_write_csv(path, rows, list(rows[0]))


def _readback_audits(
    root: Path,
    completed: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    commit: str,
) -> list[dict[str, Any]]:
    context_by_case = {item["protocol"]["case_id"]: item for item in contexts}
    audits = []
    for output in completed:
        case_id = output["summary"]["case_id"]
        context = context_by_case[case_id]
        instance: InventoryInstance = context["instance"]
        path = root / "results" / f"{case_id}_B1.00_result.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        y = np.asarray(saved["first_stage"]["y"], dtype=float)
        x = np.asarray(saved["first_stage"]["x"], dtype=float)
        demand = np.asarray(saved["worst_case"]["demand"], dtype=float)
        q = np.asarray(saved["recourse"]["q"], dtype=float)
        u = np.asarray(saved["recourse"]["u"], dtype=float)
        e = np.asarray(saved["recourse"]["e"], dtype=float)
        audit = _integrity_audit(
            instance,
            y,
            x,
            demand,
            q,
            u,
            e,
            float(saved["worst_case"]["robust_dual_objective"]),
            float(saved["recourse"]["objective"]),
            float(context["protocol"]["integrity_tolerance"]),
        )
        provenance_pass = (
            saved["provenance"]["git_commit"] == commit
            and saved["provenance"]["input_sha256"] == context["protocol"]["instance_sha256"]
            and saved["provenance"]["config_sha256"] == context["config_sha256"]
        )
        audit["readback_revalidation"] = "PASS"
        audit["provenance_validation"] = "PASS" if provenance_pass else "FAIL"
        if not provenance_pass:
            audit["status"] = "FAIL"
        audits.append({"case_id": case_id, **audit, **saved["provenance"]})
    return audits


def _verify_manifest(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = root / item["path"]
        _require(path.stat().st_size == item["bytes"], f"Manifest byte count mismatch: {item['path']}")
        _require(file_sha256(path) == item["sha256"], f"Manifest hash mismatch: {item['path']}")


def _write_outputs(
    root: Path,
    completed: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    commit: str,
) -> None:
    results_dir = root / "results"
    audit_dir = root / "audit"
    for output in completed:
        case_id = output["summary"]["case_id"]
        atomic_write_json(results_dir / f"{case_id}_B1.00_result.json", output["result_json"])
        _write_csv(results_dir / "iteration_logs" / f"{case_id}_B1.00_iterations.csv", output["iteration_log"])
    for filename, key in (
        ("summary.csv", "summary"),
        ("first_stage_allocation.csv", "allocation_rows"),
        ("depot_summary.csv", "depot_rows"),
        ("regional_service.csv", "service_rows"),
        ("worst_case_patterns.csv", "pattern_rows"),
        ("recourse_flows.csv", "flow_rows"),
    ):
        rows: list[dict[str, Any]] = []
        for output in completed:
            value = output[key]
            rows.extend(value if isinstance(value, list) else [value])
        _write_csv(results_dir / filename, rows)
    atomic_write_json(
        root / "protocol.json",
        {
            "schema": "renault_robust_baseline_protocol_v1",
            "git_commit": commit,
            "case_order": list(EXPECTED_CASE_ORDER),
            "completed_cases": [item["summary"]["case_id"] for item in completed],
            "configs": [
                {"path": str(context["config_path"].relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": context["config_sha256"]}
                for context in contexts
            ],
            "constraints": {
                "budget": "B_ref_only",
                "gamma": 2,
                "eligibility": "Full",
                "tuning_after_results": False,
                "k_experiments": False,
                "algorithm_comparison": False,
                "core_model_change": False,
            },
        },
    )
    readback_audits = _readback_audits(root, completed, contexts, commit)
    overall = "PASS" if completed and all(item["status"] == "PASS" for item in readback_audits) else "FAIL"
    atomic_write_json(
        audit_dir / "result_integrity_audit.json",
        {
            "schema": "renault_result_integrity_audit_v1",
            "overall": overall,
            "written_results_reloaded": True,
            "cases": readback_audits,
        },
    )
    manifest_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"
    )
    atomic_write_json(
        root / "manifest.json",
        {
            "schema": "renault_robust_baseline_manifest_v1",
            "git_commit": commit,
            "created_at": utc_now_iso(),
            "files": [
                {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in manifest_files
            ],
        },
    )
    _verify_manifest(root)


def run_formal_baselines() -> list[dict[str, Any]]:
    contexts = [preflight(path) for path in CONFIG_PATHS]
    _require(tuple(item["protocol"]["case_id"] for item in contexts) == EXPECTED_CASE_ORDER, "Case order changed")
    roots = {_absolute(item["protocol"]["result_root"]) for item in contexts}
    _require(len(roots) == 1, "Cases must share one result root")
    root = roots.pop()
    _require(not root.exists(), f"Formal result directory already exists: {root}")
    commit = _git_commit()
    completed: list[dict[str, Any]] = []
    for context in contexts:
        output = _case_outputs(context, commit)
        completed.append(output)
        _write_outputs(root, completed, contexts, commit)
        if not output["summary"]["certified"]:
            raise RuntimeError(f"{output['summary']['case_id']} unresolved; second case is not authorized")
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the two frozen Renault robust baselines serially")
    parser.add_argument("--preflight", action="store_true", help="validate protocol and inputs without solving")
    args = parser.parse_args()
    if args.preflight:
        results = [preflight(path) for path in CONFIG_PATHS]
        print(json.dumps({"preflight": "PASS", "cases": [item["protocol"]["case_id"] for item in results]}))
        return 0
    completed = run_formal_baselines()
    print(json.dumps({"status": "PASS", "cases": [item["summary"]["case_id"] for item in completed]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
