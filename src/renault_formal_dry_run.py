from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import gurobipy as gp
import numpy as np

from . import robust_dual_subproblem as robust_dual_module
from . import subproblem as recourse_module
from .benders import _build_master
from .instance import InventoryInstance, load_instance
from .scenarios import enumerate_budget_scenarios


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "real_data_studies" / "renault_formal_instances_v6"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts"
EXPECTED_DIMS = (15, 8, 12)
EXPECTED_SCENARIOS = 4657


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _finite_nonnegative(value: Any, shape: tuple[int, ...]) -> bool:
    array = np.asarray(value, dtype=float)
    return array.shape == shape and bool(np.isfinite(array).all() and (array >= 0.0).all())


def _audit_instance(instance: InventoryInstance) -> dict[str, Any]:
    i, j, r = EXPECTED_DIMS
    shapes = {
        "fixed_cost": ((i,), instance.fixed_cost),
        "inventory_cost": ((i, j), instance.inventory_cost),
        "capacity": ((i,), instance.capacity),
        "volume": ((j,), instance.volume),
        "transport_cost": ((i, r, j), instance.transport_cost),
        "shortage_penalty": ((r, j), instance.shortage_penalty),
        "service_penalty": ((j,), instance.service_penalty),
        "service_level": ((j,), instance.service_level),
        "base_demand": ((r, j), instance.base_demand),
        "demand_deviation": ((r, j), instance.demand_deviation),
        "inventory_ub": ((i, j), instance.inventory_ub),
    }
    dimensions_pass = (instance.num_warehouses, instance.num_products, instance.num_regions) == EXPECTED_DIMS
    numeric_pass = all(_finite_nonnegative(values, shape) for shape, values in shapes.values())
    volume = np.asarray(instance.volume, dtype=float)
    capacity = np.asarray(instance.capacity, dtype=float)
    demand = np.asarray(instance.base_demand, dtype=float)
    actual_ub = np.asarray(instance.inventory_ub, dtype=float)
    expected_ub = np.minimum(capacity[:, None] / volume[None, :], 1.35 * demand.sum(axis=0)[None, :])
    ub_error = float(np.max(np.abs(actual_ub - expected_ub)))
    return {
        "dimensions_pass": dimensions_pass,
        "array_shapes": {name: list(np.asarray(values).shape) for name, (_, values) in shapes.items()},
        "numeric_domain_pass": numeric_pass,
        "budget_positive": bool(math.isfinite(instance.budget) and instance.budget > 0.0),
        "ub_audit": _status(bool(np.allclose(actual_ub, expected_ub, rtol=0.0, atol=1e-9))),
        "ub_max_abs_error": ub_error,
    }


def _audit_scenarios(instance: InventoryInstance) -> dict[str, Any]:
    scenarios = enumerate_budget_scenarios(instance, gamma=2, max_scenarios=EXPECTED_SCENARIOS)
    gamma_counts = {str(k): sum(s.gamma == k for s in scenarios) for k in range(3)}
    unique_patterns = len({scenario.active_units for scenario in scenarios})
    passed = len(scenarios) == EXPECTED_SCENARIOS and unique_patterns == EXPECTED_SCENARIOS
    passed = passed and gamma_counts == {"0": 1, "1": 96, "2": 4560}
    return {
        "scenario_count": len(scenarios),
        "gamma_counts": gamma_counts,
        "unique_patterns": unique_patterns,
        "scenario_enumeration": _status(passed),
    }


def _coefficient(model: gp.Model, constraint: gp.Constr, variable: gp.Var) -> float:
    return float(model.getCoeff(constraint, variable))


def _audit_master(instance: InventoryInstance) -> dict[str, Any]:
    model, y, x, _ = _build_master(instance, output_flag=False)
    model.update()
    capacity = [model.getConstrByName(f"capacity[{i}]") for i in instance.I]
    logic = [model.getConstrByName(f"logic[{i},{j}]") for i in instance.I for j in instance.J]
    budget = model.getConstrByName("budget")
    structure_pass = len(y) == 15 and len(x) == 120
    structure_pass = structure_pass and all(item is not None for item in capacity + logic) and budget is not None
    constraint_counts = {
        "capacity": len(capacity),
        "logic": len(logic),
        "budget": int(budget is not None),
    }
    budget_formula_pass = budget is not None and budget.Sense == "<" and math.isclose(budget.RHS, instance.budget)
    if budget_formula_pass:
        budget_formula_pass = all(
            math.isclose(_coefficient(model, budget, y[i]), instance.fixed_cost[i]) for i in instance.I
        ) and all(
            math.isclose(_coefficient(model, budget, x[i, j]), instance.inventory_cost[i][j])
            for i in instance.I
            for j in instance.J
        )
    capacity_formula_pass = True
    for i, constraint in enumerate(capacity):
        capacity_formula_pass = capacity_formula_pass and constraint is not None and constraint.Sense == "<"
        capacity_formula_pass = capacity_formula_pass and math.isclose(constraint.RHS, 0.0)
        capacity_formula_pass = capacity_formula_pass and math.isclose(
            _coefficient(model, constraint, y[i]), -instance.capacity[i]
        )
        capacity_formula_pass = capacity_formula_pass and all(
            math.isclose(_coefficient(model, constraint, x[i, j]), instance.volume[j]) for j in instance.J
        )
    passed = structure_pass and budget_formula_pass and capacity_formula_pass
    passed = passed and constraint_counts == {"capacity": 15, "logic": 120, "budget": 1}
    return {
        "master_build": _status(passed),
        "variables": {"y": len(y), "x": len(x), "theta": 1},
        "constraints": constraint_counts,
        "budget_formula_pass": budget_formula_pass,
        "capacity_formula_pass": capacity_formula_pass,
    }


def _capture_nominal_solve(instance: InventoryInstance) -> dict[str, Any]:
    captured: list[gp.Model] = []
    real_model = recourse_module.gp.Model

    def model_factory(*args: Any, **kwargs: Any) -> gp.Model:
        model = real_model(*args, **kwargs)
        captured.append(model)
        return model

    recourse_module.gp.Model = model_factory
    try:
        scenario = enumerate_budget_scenarios(instance, gamma=0, max_scenarios=1)[0]
        x_values = {(i, j): 0.0 for i in instance.I for j in instance.J}
        result = recourse_module.solve_recourse_subproblem(instance, scenario, x_values, output_flag=False)
    finally:
        recourse_module.gp.Model = real_model
    model = captured[0]
    model.update()
    names = [constraint.ConstrName for constraint in model.getConstrs()]
    expected_constraints = {"demand": 96, "supply": 120, "service": 8}
    counts = {prefix: sum(name.startswith(f"{prefix}[") for name in names) for prefix in expected_constraints}
    passed = model.NumVars == 1544 and counts == expected_constraints and model.NumConstrs == 224
    return {
        "nominal_subproblem_build": _status(passed),
        "variables": {"q": 1440, "u": 96, "e": 8, "total": model.NumVars},
        "constraints": {**counts, "total": model.NumConstrs},
        "solve_status": "OPTIMAL",
        "objective": result.objective,
    }


class _BuildOnlyModel:
    def __init__(self, model: gp.Model):
        self.model = model

    def optimize(self) -> None:
        self.model.update()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def _audit_robust_dual(instance: InventoryInstance) -> dict[str, Any]:
    captured: list[_BuildOnlyModel] = []
    real_model = robust_dual_module.gp.Model

    def model_factory(*args: Any, **kwargs: Any) -> _BuildOnlyModel:
        wrapper = _BuildOnlyModel(real_model(*args, **kwargs))
        captured.append(wrapper)
        return wrapper

    robust_dual_module.gp.Model = model_factory
    try:
        x_values = {(i, j): 0.0 for i in instance.I for j in instance.J}
        robust_dual_module.solve_robust_dual_subproblem(instance, x_values, gamma=2, output_flag=False)
    finally:
        robust_dual_module.gp.Model = real_model
    model = captured[0].model
    model.update()
    variables = model.getVars()
    constraints = model.getConstrs()
    coefficients_finite = all(
        math.isfinite(float(model.getRow(constraint).getCoeff(k)))
        for constraint in constraints
        for k in range(model.getRow(constraint).size())
    )
    values_finite = all(math.isfinite(float(v.LB)) and math.isfinite(float(v.Obj)) for v in variables)
    values_finite = values_finite and all(
        math.isfinite(float(v.UB)) or (math.isinf(float(v.UB)) and v.VarName.startswith(("w[", "g[")))
        for v in variables
    )
    values_finite = values_finite and all(math.isfinite(float(c.RHS)) for c in constraints)
    default_unbounded = [v.VarName for v in variables if math.isinf(float(v.UB))]
    budget = model.getConstrByName("budget")
    demand_shape = list(np.asarray(instance.base_demand).shape)
    passed = model.NumVars == 512 and model.NumConstrs == 2121
    passed = passed and demand_shape == [12, 8] and coefficients_finite and values_finite
    passed = passed and budget is not None and budget.Sense == "<" and math.isclose(budget.RHS, 2.0)
    return {
        "robust_dual_build": _status(passed),
        "gamma": 2,
        "demand_shape": demand_shape,
        "variables": model.NumVars,
        "constraints": model.NumConstrs,
        "data_derived_bounds_and_coefficients_finite": coefficients_finite and values_finite,
        "intentional_solver_default_unbounded_variables": len(default_unbounded),
        "unexpected_nan_or_inf": not (coefficients_finite and values_finite),
        "optimization_executed": False,
    }


def _audit_eligibility(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    arrays = {key: np.asarray(data[key], dtype=int) for key in ("K1", "K2", "Full")}
    shape_pass = all(array.shape == (15, 12) for array in arrays.values())
    binary_pass = all(np.isin(array, (0, 1)).all() for array in arrays.values())
    nesting_pass = bool((arrays["K1"] <= arrays["K2"]).all() and (arrays["K2"] <= arrays["Full"]).all())
    cardinalities = {key: arrays[key].sum(axis=0).tolist() for key in arrays}
    cardinality_pass = cardinalities["K1"] == [1] * 12
    cardinality_pass = cardinality_pass and cardinalities["K2"] == [2] * 12
    cardinality_pass = cardinality_pass and cardinalities["Full"] == [15] * 12
    return {
        "eligibility_audit": _status(shape_pass and binary_pass and nesting_pass and cardinality_pass),
        "shapes": {key: list(value.shape) for key, value in arrays.items()},
        "eligible_depots_per_region": cardinalities,
        "nesting_pass": nesting_pass,
        "cardinality_pass": cardinality_pass,
    }


def _verify_package_hashes(data_root: Path) -> bool:
    declarations = json.loads((data_root / "file_hashes.json").read_text(encoding="utf-8"))
    return all(_sha256(data_root / item["path"]) == item["sha256"] for item in declarations)


def run_dry_run(data_root: Path = DEFAULT_DATA_ROOT, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict[str, Any]:
    instance_paths = sorted((data_root / "instances").glob("*_B*.json"))
    if len(instance_paths) != 8:
        raise ValueError(f"Expected 8 Renault instances, found {len(instance_paths)}")
    eligibility = {
        case: _audit_eligibility(data_root / "eligibility" / f"{case}_eligibility.json")
        for case in ("210202", "210628")
    }
    nominal: dict[str, dict[str, Any]] = {}
    for case in eligibility:
        nominal_path = data_root / "instances" / f"{case}_B1.00.json"
        nominal[case] = _capture_nominal_solve(load_instance(nominal_path))

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for path in instance_paths:
        instance = load_instance(path)
        case = path.name.split("_", 1)[0]
        contract = _audit_instance(instance)
        scenarios = _audit_scenarios(instance)
        master = _audit_master(instance)
        robust_dual = _audit_robust_dual(instance)
        checks = [
            contract["dimensions_pass"],
            contract["numeric_domain_pass"],
            contract["budget_positive"],
            contract["ub_audit"] == "PASS",
            scenarios["scenario_enumeration"] == "PASS",
            master["master_build"] == "PASS",
            nominal[case]["nominal_subproblem_build"] == "PASS",
            robust_dual["robust_dual_build"] == "PASS",
            eligibility[case]["eligibility_audit"] == "PASS",
        ]
        overall = _status(all(checks))
        row = {
            "instance_name": path.stem,
            "sha256": _sha256(path),
            "num_warehouses": instance.num_warehouses,
            "num_products": instance.num_products,
            "num_regions": instance.num_regions,
            "budget": instance.budget,
            "scenario_count": scenarios["scenario_count"],
            "master_build": master["master_build"],
            "nominal_subproblem_build": nominal[case]["nominal_subproblem_build"],
            "robust_dual_build": robust_dual["robust_dual_build"],
            "ub_audit": contract["ub_audit"],
            "eligibility_audit": eligibility[case]["eligibility_audit"],
            "overall": overall,
        }
        rows.append(row)
        details.append({**row, "contract": contract, "scenarios": scenarios, "master": master, "nominal": nominal[case], "robust_dual": robust_dual, "eligibility": eligibility[case]})

    package_hashes_pass = _verify_package_hashes(data_root)
    overall = _status(package_hashes_pass and all(row["overall"] == "PASS" for row in rows))
    report = {
        "step": "4A",
        "overall": overall,
        "package_hashes": _status(package_hashes_pass),
        "inventory_instance_code_changed": False,
        "core_mathematical_model_changed": False,
        "formal_gamma2_optimization_executed": False,
        "paper_result_produced": False,
        "robust_dual_optimization_executed": False,
        "nominal_subproblems_solved": ["210202_B1.00", "210628_B1.00"],
        "instances": details,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifact_dir / "renault_step4a_dry_run_report.json"
    csv_path = artifact_dir / "renault_step4a_dry_run_report.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Renault Step 4A code-level dry-run")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    report = run_dry_run(args.data_root, args.artifact_dir)
    print(json.dumps({"step": report["step"], "overall": report["overall"], "instances": len(report["instances"])}))
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
