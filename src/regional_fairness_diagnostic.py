from __future__ import annotations

from dataclasses import asdict, dataclass, field
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from typing import Any, Iterable, Mapping

import gurobipy as gp
from gurobipy import GRB

from .instance import InventoryInstance
from .scenarios import DemandScenario, enumerate_budget_scenarios_with_metadata
from .status import gurobi_status_name


DEFAULT_COST_ABSOLUTE_TOLERANCE = 1.0e-6
DEFAULT_COST_RELATIVE_TOLERANCE = 1.0e-6
DEFAULT_METRIC_TOLERANCE = 1.0e-9
MATERIAL_GAP_THRESHOLD = 0.10
STRUCTURAL_MEDIAN_THRESHOLD = 0.05
NO_MATERIAL_MEDIAN_THRESHOLD = 0.03
DEGENERACY_REDUCTION_THRESHOLD = 0.05

REGION_SCENARIO_FIELDS = [
    "diagnostic_run_key",
    "base_run_key",
    "instance_name",
    "experiment_name",
    "scale",
    "method",
    "seed",
    "base_git_commit",
    "base_config_sha256",
    "resolved_config_sha256",
    "scenario_key",
    "scenario_index",
    "scenario_type",
    "is_nominal",
    "is_cost_worst",
    "is_fairness_worst",
    "deviation_pattern",
    "deviation_pattern_sha256",
    "region_id",
    "recourse_variant",
    "default_recourse_status",
    "fair_best_recourse_status",
    "default_recourse_cost",
    "fair_best_recourse_cost",
    "cost_absolute_tolerance",
    "cost_relative_tolerance",
    "invalid_reason",
    "instance_size",
    "scenario_id",
    "scenario_kind",
    "region",
    "regional_demand",
    "regional_shortage",
    "shortage_rate",
    "fill_rate",
    "fill_rate_applicable",
    "not_applicable_reason",
    "weighted_mean_fill_rate",
    "fill_rate_gap",
    "worst_region_deviation",
    "fill_rate_standard_deviation",
    "fill_rate_gini",
    "regional_transport_cost",
    "allocated_transport_units",
    "allocated_unit_transport_cost",
    "reachable_warehouse_count",
    "recourse_policy",
    "original_recourse_cost",
    "evaluated_recourse_cost",
    "cost_tolerance",
    "scenario_gamma_usage",
    "first_stage_x_sha256",
]

INSTANCE_SUMMARY_FIELDS = [
    "diagnostic_run_key",
    "base_run_key",
    "instance_name",
    "experiment_name",
    "method",
    "base_git_commit",
    "base_config_sha256",
    "resolved_config_sha256",
    "seed",
    "size",
    "default_WGap",
    "fair_best_WGap",
    "default_WMinFR",
    "fair_best_WMinFR",
    "default_WWD",
    "fair_best_WWD",
    "nominal_gap",
    "cost_worst_gap",
    "cost_worst_scenario",
    "fairness_worst_scenario",
    "default_minus_fair_best_WGap",
    "diagnosis_category",
    "scenario_count",
    "first_stage_x_sha256",
]


def deviation_pattern_payload(
    instance: InventoryInstance,
    scenario: DemandScenario,
) -> list[dict[str, Any]]:
    """Return a deterministic, human-auditable representation of active deviations."""
    return [
        {
            "region_id": int(region),
            "product_id": int(product),
            "deviation_value": float(instance.demand_deviation[region][product]),
            "base_demand": float(instance.base_demand[region][product]),
            "realized_demand": float(scenario.demand[region][product]),
        }
        for region, product in sorted(scenario.active_units)
    ]


def deviation_pattern_sha256(pattern: list[dict[str, Any]]) -> str:
    payload = json.dumps(pattern, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_scenario_key(
    instance: InventoryInstance,
    scenario: DemandScenario,
    scenario_index: int,
) -> str:
    pattern_hash = deviation_pattern_sha256(deviation_pattern_payload(instance, scenario))
    return f"scenario_{int(scenario_index):05d}_{pattern_hash[:16]}"


@dataclass(frozen=True)
class RecourseAllocation:
    policy: str
    scenario_id: str
    status: str
    objective: float | None
    original_optimal_cost: float | None
    cost_tolerance: float | None
    shipment_values: list[list[list[float]]] = field(default_factory=list)
    shortage_values: list[list[float]] = field(default_factory=list)
    service_violation_values: list[float] = field(default_factory=list)
    transport_cost_by_region: list[float] = field(default_factory=list)
    allocated_units_by_region: list[float] = field(default_factory=list)
    runtime: float = 0.0
    first_stage_x_sha256: str = ""
    gamma_usage: int = 0
    original_cost_reproduced: bool = False
    cost_cap_satisfied: bool = False
    constraints_satisfied: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FairnessDiagnosticInstanceResult:
    instance_size: str
    seed: int
    valid: bool
    scenario_count: int
    region_scenario_metrics: list[dict[str, Any]]
    instance_summary: dict[str, Any]
    audit: dict[str, Any]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def first_stage_x_sha256(x_values: list[list[float]]) -> str:
    payload = json.dumps(
        [[float(value) for value in row] for row in x_values],
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def combined_cost_tolerance(
    optimal_cost: float,
    *,
    absolute_tolerance: float = DEFAULT_COST_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_COST_RELATIVE_TOLERANCE,
) -> float:
    values = (optimal_cost, absolute_tolerance, relative_tolerance)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Cost tolerance inputs must be finite.")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("Cost tolerances must be nonnegative.")
    return float(absolute_tolerance) + float(relative_tolerance) * max(
        1.0, abs(float(optimal_cost))
    )


def _gini(values: list[float]) -> float | None:
    if not values:
        return None
    mean = statistics.mean(values)
    if abs(mean) <= DEFAULT_METRIC_TOLERANCE:
        return 0.0
    pairwise = sum(abs(left - right) for left in values for right in values)
    return pairwise / (2.0 * len(values) * len(values) * mean)


def summarize_regional_service(
    demand_values: list[list[float]],
    shortage_values: list[list[float]],
    *,
    transport_cost_by_region: list[float] | None = None,
    allocated_units_by_region: list[float] | None = None,
    reachable_warehouse_count: int | None = None,
    metric_tolerance: float = DEFAULT_METRIC_TOLERANCE,
) -> dict[str, Any]:
    if not math.isfinite(float(metric_tolerance)) or metric_tolerance < 0.0:
        raise ValueError("metric_tolerance must be finite and nonnegative.")
    if len(demand_values) != len(shortage_values):
        raise ValueError("Demand and shortage must have the same region count.")
    if any(len(demand) != len(shortage) for demand, shortage in zip(demand_values, shortage_values)):
        raise ValueError("Demand and shortage product dimensions differ.")
    region_count = len(demand_values)
    transport = transport_cost_by_region or [0.0] * region_count
    allocated = allocated_units_by_region or [0.0] * region_count
    if len(transport) != region_count or len(allocated) != region_count:
        raise ValueError("Regional transport arrays have the wrong length.")

    rows: list[dict[str, Any]] = []
    applicable_fill_rates: list[float] = []
    total_demand = 0.0
    total_shortage = 0.0
    for region, (demand_row, shortage_row) in enumerate(zip(demand_values, shortage_values)):
        demand = sum(float(value) for value in demand_row)
        shortage = sum(float(value) for value in shortage_row)
        if not math.isfinite(demand) or not math.isfinite(shortage):
            raise ValueError("Demand and shortage must be finite.")
        if demand < -metric_tolerance or shortage < -metric_tolerance:
            raise ValueError("Demand and shortage must be nonnegative.")
        if shortage > demand + metric_tolerance:
            raise ValueError("Regional shortage cannot exceed regional demand.")
        demand = max(0.0, demand)
        shortage = min(max(0.0, shortage), demand)
        total_demand += demand
        total_shortage += shortage
        applicable = demand > metric_tolerance
        fill_rate = None if not applicable else min(1.0, max(0.0, 1.0 - shortage / demand))
        if fill_rate is not None:
            applicable_fill_rates.append(fill_rate)
        unit_transport = None if allocated[region] <= metric_tolerance else transport[region] / allocated[region]
        rows.append(
            {
                "region": region,
                "regional_demand": demand,
                "regional_shortage": shortage,
                "shortage_rate": None if not applicable else shortage / demand,
                "fill_rate": fill_rate,
                "fill_rate_applicable": applicable,
                "not_applicable_reason": "" if applicable else "zero_regional_demand",
                "regional_transport_cost": float(transport[region]),
                "allocated_transport_units": float(allocated[region]),
                "allocated_unit_transport_cost": None if unit_transport is None else float(unit_transport),
                "reachable_warehouse_count": reachable_warehouse_count,
            }
        )

    weighted = None if total_demand <= metric_tolerance else 1.0 - total_shortage / total_demand
    gap = None if not applicable_fill_rates else max(applicable_fill_rates) - min(applicable_fill_rates)
    minimum = None if not applicable_fill_rates else min(applicable_fill_rates)
    worst_deviation = (
        None
        if weighted is None or not applicable_fill_rates
        else max(weighted - fill_rate for fill_rate in applicable_fill_rates)
    )
    standard_deviation = (
        None
        if not applicable_fill_rates
        else statistics.pstdev(applicable_fill_rates)
    )
    gini = _gini(applicable_fill_rates)
    for row in rows:
        row.update(
            {
                "weighted_mean_fill_rate": weighted,
                "fill_rate_gap": gap,
                "worst_region_deviation": worst_deviation,
                "fill_rate_standard_deviation": standard_deviation,
                "fill_rate_gini": gini,
            }
        )
    return {
        "regions": rows,
        "total_demand": total_demand,
        "total_shortage": total_shortage,
        "weighted_mean_fill_rate": weighted,
        "fill_rate_gap": gap,
        "minimum_fill_rate": minimum,
        "worst_region_deviation": worst_deviation,
        "fill_rate_standard_deviation": standard_deviation,
        "fill_rate_gini": gini,
        "applicable_region_count": len(applicable_fill_rates),
        "not_applicable_region_count": region_count - len(applicable_fill_rates),
    }


def _validate_first_stage(instance: InventoryInstance, x_values: list[list[float]], tolerance: float) -> None:
    if len(x_values) != instance.num_warehouses or any(
        len(row) != instance.num_products for row in x_values
    ):
        raise ValueError("best_x_values has the wrong shape.")
    for i in instance.I:
        for j in instance.J:
            value = float(x_values[i][j])
            if not math.isfinite(value) or value < -tolerance:
                raise ValueError("best_x_values must be finite and nonnegative.")
            if value > instance.inventory_ub[i][j] + tolerance:
                raise ValueError("best_x_values violates an inventory upper bound.")
        used_capacity = sum(instance.volume[j] * float(x_values[i][j]) for j in instance.J)
        if used_capacity > instance.capacity[i] + tolerance:
            raise ValueError("best_x_values violates warehouse capacity.")


def _build_recourse_model(
    instance: InventoryInstance,
    scenario: DemandScenario,
    x_values: list[list[float]],
    *,
    time_limit: float,
    output_flag: bool,
) -> tuple[gp.Model, Any, Any, Any, Any, Any, Any]:
    model = gp.Model(f"fairness_recourse_{scenario.name}")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.Method = 1
    model.Params.TimeLimit = max(1.0e-3, float(time_limit))
    q = model.addVars(instance.I, instance.R, instance.J, lb=0.0, name="q")
    u = model.addVars(instance.R, instance.J, lb=0.0, name="u")
    e = model.addVars(instance.J, lb=0.0, name="e")
    for r in instance.R:
        for j in instance.J:
            model.addConstr(
                gp.quicksum(q[i, r, j] for i in instance.I) + u[r, j]
                >= scenario.demand[r][j],
                name=f"demand[{r},{j}]",
            )
    for i in instance.I:
        for j in instance.J:
            model.addConstr(
                gp.quicksum(q[i, r, j] for r in instance.R) <= x_values[i][j],
                name=f"supply[{i},{j}]",
            )
    for j in instance.J:
        model.addConstr(
            gp.quicksum(u[r, j] for r in instance.R) - e[j]
            <= (1.0 - instance.service_level[j])
            * sum(scenario.demand[r][j] for r in instance.R),
            name=f"service[{j}]",
        )
    transport = gp.quicksum(
        instance.transport_cost[i][r][j] * q[i, r, j]
        for i in instance.I
        for r in instance.R
        for j in instance.J
    )
    shortage = gp.quicksum(
        instance.shortage_penalty[r][j] * u[r, j]
        for r in instance.R
        for j in instance.J
    )
    service = gp.quicksum(instance.service_penalty[j] * e[j] for j in instance.J)
    return model, q, u, e, transport, shortage, service


def _allocation_from_model(
    *,
    instance: InventoryInstance,
    scenario: DemandScenario,
    x_values: list[list[float]],
    model: gp.Model,
    q: Any,
    u: Any,
    e: Any,
    evaluated_cost: Any,
    policy: str,
    original_optimal_cost: float,
    cost_tolerance: float,
    runtime: float,
    metric_tolerance: float,
) -> RecourseAllocation:
    shipments = [
        [[float(q[i, r, j].X) for j in instance.J] for r in instance.R]
        for i in instance.I
    ]
    shortages = [[float(u[r, j].X) for j in instance.J] for r in instance.R]
    service_values = [float(e[j].X) for j in instance.J]
    transport_by_region = [
        sum(
            instance.transport_cost[i][r][j] * shipments[i][r][j]
            for i in instance.I
            for j in instance.J
        )
        for r in instance.R
    ]
    allocated_by_region = [
        sum(shipments[i][r][j] for i in instance.I for j in instance.J)
        for r in instance.R
    ]
    demand_ok = all(
        sum(shipments[i][r][j] for i in instance.I) + shortages[r][j]
        + metric_tolerance
        >= scenario.demand[r][j]
        for r in instance.R
        for j in instance.J
    )
    supply_ok = all(
        sum(shipments[i][r][j] for r in instance.R)
        <= x_values[i][j] + metric_tolerance
        for i in instance.I
        for j in instance.J
    )
    service_ok = all(
        sum(shortages[r][j] for r in instance.R) - service_values[j]
        <= (1.0 - instance.service_level[j])
        * sum(scenario.demand[r][j] for r in instance.R)
        + metric_tolerance
        for j in instance.J
    )
    objective = float(evaluated_cost.getValue())
    return RecourseAllocation(
        policy=policy,
        scenario_id=scenario.name,
        status="optimal",
        objective=objective,
        original_optimal_cost=float(original_optimal_cost),
        cost_tolerance=float(cost_tolerance),
        shipment_values=shipments,
        shortage_values=shortages,
        service_violation_values=service_values,
        transport_cost_by_region=[float(value) for value in transport_by_region],
        allocated_units_by_region=[float(value) for value in allocated_by_region],
        runtime=max(0.0, float(runtime)),
        first_stage_x_sha256=first_stage_x_sha256(x_values),
        gamma_usage=scenario.gamma,
        original_cost_reproduced=abs(objective - original_optimal_cost) <= cost_tolerance + metric_tolerance,
        cost_cap_satisfied=objective <= original_optimal_cost + cost_tolerance + metric_tolerance,
        constraints_satisfied=demand_ok and supply_ok and service_ok,
    )


def solve_default_and_fair_best_recourse(
    instance: InventoryInstance,
    scenario: DemandScenario,
    x_values: list[list[float]],
    *,
    cost_absolute_tolerance: float = DEFAULT_COST_ABSOLUTE_TOLERANCE,
    cost_relative_tolerance: float = DEFAULT_COST_RELATIVE_TOLERANCE,
    metric_tolerance: float = DEFAULT_METRIC_TOLERANCE,
    time_limit: float = 30.0,
    output_flag: bool = False,
) -> tuple[RecourseAllocation, RecourseAllocation]:
    _validate_first_stage(instance, x_values, metric_tolerance)
    start = time.perf_counter()
    model, q, u, e, transport, shortage, service = _build_recourse_model(
        instance, scenario, x_values, time_limit=time_limit, output_flag=output_flag
    )
    try:
        cost = transport + shortage + service
        model.setObjective(cost, GRB.MINIMIZE)
        model.optimize()
        status = gurobi_status_name(model.Status)
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Default recourse ended with status {status}.")
        q_star = float(model.ObjVal)
        tolerance = combined_cost_tolerance(
            q_star,
            absolute_tolerance=cost_absolute_tolerance,
            relative_tolerance=cost_relative_tolerance,
        )
        default = _allocation_from_model(
            instance=instance,
            scenario=scenario,
            x_values=x_values,
            model=model,
            q=q,
            u=u,
            e=e,
            evaluated_cost=cost,
            policy="default",
            original_optimal_cost=q_star,
            cost_tolerance=tolerance,
            runtime=time.perf_counter() - start,
            metric_tolerance=metric_tolerance,
        )
    finally:
        model.dispose()
    default_metrics = summarize_regional_service(
        [list(row) for row in scenario.demand],
        default.shortage_values,
        metric_tolerance=metric_tolerance,
    )

    fair_start = time.perf_counter()
    fair_model, fq, fu, fe, ftransport, fshortage, fservice = _build_recourse_model(
        instance, scenario, x_values, time_limit=time_limit, output_flag=output_flag
    )
    try:
        fair_cost = ftransport + fshortage + fservice
        fair_model.addConstr(fair_cost <= q_star + tolerance, name="original_cost_cap")
        applicable_regions = [
            r for r in instance.R if sum(scenario.demand[r][j] for j in instance.J) > metric_tolerance
        ]
        if applicable_regions:
            max_shortage_rate = fair_model.addVar(lb=0.0, name="max_regional_shortage_rate")
            min_shortage_rate = fair_model.addVar(lb=0.0, name="min_regional_shortage_rate")
            for r in applicable_regions:
                regional_demand = sum(scenario.demand[r][j] for j in instance.J)
                rate = gp.quicksum(fu[r, j] for j in instance.J) / regional_demand
                fair_model.addConstr(rate <= max_shortage_rate, name=f"max_shortage_rate[{r}]")
                fair_model.addConstr(min_shortage_rate <= rate, name=f"min_shortage_rate[{r}]")
            default_gap = float(default_metrics["fill_rate_gap"] or 0.0)
            fair_model.addConstr(
                max_shortage_rate - min_shortage_rate <= default_gap + metric_tolerance,
                name="fairness_not_worse_than_default",
            )
            fair_model.setObjective(max_shortage_rate, GRB.MINIMIZE)
        else:
            fair_model.setObjective(fair_cost, GRB.MINIMIZE)
        fair_model.optimize()
        fair_status = gurobi_status_name(fair_model.Status)
        if fair_model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Fair-best recourse ended with status {fair_status}.")
        fair = _allocation_from_model(
            instance=instance,
            scenario=scenario,
            x_values=x_values,
            model=fair_model,
            q=fq,
            u=fu,
            e=fe,
            evaluated_cost=fair_cost,
            policy="fair_best",
            original_optimal_cost=q_star,
            cost_tolerance=tolerance,
            runtime=time.perf_counter() - fair_start,
            metric_tolerance=metric_tolerance,
        )
    finally:
        fair_model.dispose()
    return default, fair


def _scenario_rows(
    *,
    instance_size: str,
    seed: int,
    scenario: DemandScenario,
    scenario_kind: str,
    allocation: RecourseAllocation,
    instance: InventoryInstance,
    metric_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = summarize_regional_service(
        [list(row) for row in scenario.demand],
        allocation.shortage_values,
        transport_cost_by_region=allocation.transport_cost_by_region,
        allocated_units_by_region=allocation.allocated_units_by_region,
        reachable_warehouse_count=instance.num_warehouses,
        metric_tolerance=metric_tolerance,
    )
    rows: list[dict[str, Any]] = []
    for region in metrics["regions"]:
        row = {
            "instance_size": instance_size,
            "seed": seed,
            "scenario_id": scenario.name,
            "scenario_kind": scenario_kind,
            **region,
            "recourse_policy": allocation.policy,
            "original_recourse_cost": allocation.original_optimal_cost,
            "evaluated_recourse_cost": allocation.objective,
            "cost_tolerance": allocation.cost_tolerance,
            "scenario_gamma_usage": scenario.gamma,
            "first_stage_x_sha256": allocation.first_stage_x_sha256,
        }
        rows.append({field: row.get(field) for field in REGION_SCENARIO_FIELDS})
    return rows, metrics


def _instance_category(default_gap: float, fair_gap: float) -> str:
    if fair_gap >= MATERIAL_GAP_THRESHOLD:
        return "structural_fairness_gap"
    if default_gap >= MATERIAL_GAP_THRESHOLD and default_gap - fair_gap >= DEGENERACY_REDUCTION_THRESHOLD:
        return "recourse_degeneracy_only"
    if fair_gap < NO_MATERIAL_MEDIAN_THRESHOLD and default_gap < MATERIAL_GAP_THRESHOLD:
        return "no_material_fairness_gap"
    return "fairness_diagnostic_inconclusive"


def evaluate_fairness_diagnostic_instance(
    instance: InventoryInstance,
    *,
    instance_size: str,
    seed: int,
    best_x_values: list[list[float]],
    gamma: int = 2,
    max_scenarios: int = 5000,
    exact_scenarios: bool = True,
    cost_absolute_tolerance: float = DEFAULT_COST_ABSOLUTE_TOLERANCE,
    cost_relative_tolerance: float = DEFAULT_COST_RELATIVE_TOLERANCE,
    metric_tolerance: float = DEFAULT_METRIC_TOLERANCE,
    recourse_time_limit: float = 30.0,
    output_flag: bool = False,
) -> FairnessDiagnosticInstanceResult:
    if gamma != 2:
        raise ValueError("The frozen regional fairness diagnostic requires Gamma=2.")
    enumeration = enumerate_budget_scenarios_with_metadata(
        instance,
        gamma,
        max_scenarios=max_scenarios,
        exact_scenarios=exact_scenarios,
    )
    if enumeration.scenario_mode != "full" or not exact_scenarios:
        raise ValueError("Fairness diagnosis requires all budget extreme-point scenarios; sampling is forbidden.")
    pattern_keys = [scenario.active_units for scenario in enumeration.scenarios]
    if len(pattern_keys) != len(set(pattern_keys)):
        raise ValueError("Duplicate uncertainty scenarios detected.")

    x_hash = first_stage_x_sha256(best_x_values)
    records: list[tuple[DemandScenario, RecourseAllocation, RecourseAllocation, dict[str, Any], dict[str, Any]]] = []
    errors: list[str] = []
    for scenario in enumeration.scenarios:
        try:
            default, fair = solve_default_and_fair_best_recourse(
                instance,
                scenario,
                best_x_values,
                cost_absolute_tolerance=cost_absolute_tolerance,
                cost_relative_tolerance=cost_relative_tolerance,
                metric_tolerance=metric_tolerance,
                time_limit=recourse_time_limit,
                output_flag=output_flag,
            )
            default_metrics = summarize_regional_service(
                [list(row) for row in scenario.demand], default.shortage_values, metric_tolerance=metric_tolerance
            )
            fair_metrics = summarize_regional_service(
                [list(row) for row in scenario.demand], fair.shortage_values, metric_tolerance=metric_tolerance
            )
            if default.first_stage_x_sha256 != x_hash or fair.first_stage_x_sha256 != x_hash:
                raise RuntimeError("Default and fair-best did not use the same fixed first-stage solution.")
            if not default.constraints_satisfied or not fair.constraints_satisfied:
                raise RuntimeError("A recourse solution failed feasibility checks.")
            if not fair.cost_cap_satisfied:
                raise RuntimeError("Fair-best recourse exceeded the frozen cost tolerance.")
            if (
                fair_metrics["fill_rate_gap"] is not None
                and default_metrics["fill_rate_gap"] is not None
                and fair_metrics["fill_rate_gap"] > default_metrics["fill_rate_gap"] + metric_tolerance
            ):
                raise RuntimeError("Fair-best regional fill-rate gap exceeded the default gap.")
            records.append((scenario, default, fair, default_metrics, fair_metrics))
        except Exception as exc:  # noqa: BLE001 - invalid diagnostics must be explicit.
            errors.append(f"{scenario.name}: {type(exc).__name__}: {exc}")

    if errors or len(records) != len(enumeration.scenarios):
        return FairnessDiagnosticInstanceResult(
            instance_size=instance_size,
            seed=seed,
            valid=False,
            scenario_count=len(enumeration.scenarios),
            region_scenario_metrics=[],
            instance_summary={},
            audit={
                "fixed_first_stage_sha256": x_hash,
                "all_scenarios_evaluated": False,
                "diagnostic_updates_benders_bounds": False,
            },
            errors=errors,
        )

    cost_worst = max(records, key=lambda item: float(item[1].original_optimal_cost or -math.inf))[0].name
    fairness_worst = max(
        records,
        key=lambda item: float(item[4]["fill_rate_gap"] if item[4]["fill_rate_gap"] is not None else -math.inf),
    )[0].name
    rows: list[dict[str, Any]] = []
    for scenario, default, fair, _default_metrics, _fair_metrics in records:
        kinds = ["budget_extreme"]
        if scenario.gamma == 0:
            kinds.append("nominal")
        if scenario.name == cost_worst:
            kinds.append("cost_worst")
        if scenario.name == fairness_worst:
            kinds.append("fairness_worst")
        kind = "|".join(kinds)
        default_rows, _ = _scenario_rows(
            instance_size=instance_size,
            seed=seed,
            scenario=scenario,
            scenario_kind=kind,
            allocation=default,
            instance=instance,
            metric_tolerance=metric_tolerance,
        )
        fair_rows, _ = _scenario_rows(
            instance_size=instance_size,
            seed=seed,
            scenario=scenario,
            scenario_kind=kind,
            allocation=fair,
            instance=instance,
            metric_tolerance=metric_tolerance,
        )
        rows.extend(default_rows + fair_rows)

    def robust(policy_index: int, metric: str, function: Any) -> float | None:
        values = [item[policy_index][metric] for item in records if item[policy_index][metric] is not None]
        return None if not values else float(function(values))

    default_wgap = float(max(item[3]["fill_rate_gap"] for item in records if item[3]["fill_rate_gap"] is not None))
    fair_wgap = float(max(item[4]["fill_rate_gap"] for item in records if item[4]["fill_rate_gap"] is not None))
    nominal_record = next(item for item in records if item[0].gamma == 0)
    cost_record = next(item for item in records if item[0].name == cost_worst)
    summary = {
        "seed": seed,
        "size": instance_size,
        "default_WGap": default_wgap,
        "fair_best_WGap": fair_wgap,
        "default_WMinFR": robust(3, "minimum_fill_rate", min),
        "fair_best_WMinFR": robust(4, "minimum_fill_rate", min),
        "default_WWD": robust(3, "worst_region_deviation", max),
        "fair_best_WWD": robust(4, "worst_region_deviation", max),
        "nominal_gap": nominal_record[4]["fill_rate_gap"],
        "cost_worst_gap": cost_record[4]["fill_rate_gap"],
        "cost_worst_scenario": cost_worst,
        "fairness_worst_scenario": fairness_worst,
        "default_minus_fair_best_WGap": default_wgap - fair_wgap,
        "diagnosis_category": _instance_category(default_wgap, fair_wgap),
        "scenario_count": len(records),
        "first_stage_x_sha256": x_hash,
    }
    summary = {field: summary.get(field) for field in INSTANCE_SUMMARY_FIELDS}
    return FairnessDiagnosticInstanceResult(
        instance_size=instance_size,
        seed=seed,
        valid=True,
        scenario_count=len(records),
        region_scenario_metrics=rows,
        instance_summary=summary,
        audit={
            "fixed_first_stage_sha256": x_hash,
            "all_scenarios_evaluated": True,
            "scenario_patterns_unique": True,
            "gamma": gamma,
            "cost_worst_scenario": cost_worst,
            "fairness_worst_scenario": fairness_worst,
            "cost_worst_and_fairness_worst_separately_recorded": True,
            "diagnostic_updates_benders_bounds": False,
            "default_and_fair_best_same_x": True,
            "default_and_fair_best_same_scenarios": True,
        },
        errors=[],
    )


def _structural_rule(values_by_scale: Mapping[str, list[float]]) -> bool:
    per_scale = any(
        len(values) == 10
        and sum(value >= MATERIAL_GAP_THRESHOLD for value in values) >= 4
        and statistics.median(values) >= STRUCTURAL_MEDIAN_THRESHOLD
        for values in values_by_scale.values()
    )
    combined = [value for values in values_by_scale.values() for value in values]
    combined_rule = (
        len(combined) == 20
        and sum(value >= MATERIAL_GAP_THRESHOLD for value in combined) >= 8
        and statistics.median(combined) >= STRUCTURAL_MEDIAN_THRESHOLD
    )
    return per_scale or combined_rule


def classify_fairness_diagnostic(
    instance_summaries: Iterable[Mapping[str, Any]],
    *,
    correctness_checks_passed: bool,
) -> dict[str, Any]:
    summaries = list(instance_summaries)
    grouped: dict[str, list[Mapping[str, Any]]] = {"medium_large": [], "large": []}
    for summary in summaries:
        size = str(summary.get("size"))
        if size not in grouped:
            raise ValueError(f"Unexpected diagnostic size: {size}")
        grouped[size].append(summary)
    if any(len(rows) != 10 for rows in grouped.values()):
        raise ValueError("A frozen diagnostic decision requires 10 instances per scale.")
    fair = {
        size: [float(row["fair_best_WGap"]) for row in rows]
        for size, rows in grouped.items()
    }
    default = {
        size: [float(row["default_WGap"]) for row in rows]
        for size, rows in grouped.items()
    }
    fair_structural = _structural_rule(fair)
    default_structural = _structural_rule(default)
    degeneracy_reductions = {
        size: sum(
            float(row["default_WGap"]) - float(row["fair_best_WGap"])
            >= DEGENERACY_REDUCTION_THRESHOLD
            for row in rows
        )
        for size, rows in grouped.items()
    }
    no_material = all(
        sum(value >= MATERIAL_GAP_THRESHOLD for value in fair[size]) <= 1
        and statistics.median(fair[size]) < NO_MATERIAL_MEDIAN_THRESHOLD
        for size in grouped
    ) and not default_structural
    if not correctness_checks_passed:
        category = "fairness_diagnostic_invalid"
        next_stage = "none"
    elif fair_structural:
        category = "structural_fairness_gap"
        next_stage = "fairness_model_development_protocol_only"
    elif default_structural and any(count >= 4 for count in degeneracy_reductions.values()):
        category = "recourse_degeneracy_only"
        next_stage = "lexicographic_recourse_rule_protocol_only"
    elif no_material:
        category = "no_material_fairness_gap"
        next_stage = "no_fairness_model_development"
    else:
        category = "fairness_diagnostic_inconclusive"
        next_stage = "separate_inconclusive_resolution_protocol_required"
    return {
        "decision": category,
        "next_authorized_stage": next_stage,
        "correctness_gate_required": True,
        "correctness_checks_passed": bool(correctness_checks_passed),
        "fair_best_structural_rule": fair_structural,
        "default_structural_rule": default_structural,
        "degeneracy_reduction_count_by_scale": degeneracy_reductions,
        "scale_statistics": {
            size: {
                "fair_best_count_at_least_0_10": sum(value >= MATERIAL_GAP_THRESHOLD for value in fair[size]),
                "fair_best_median_WGap": statistics.median(fair[size]),
                "default_count_at_least_0_10": sum(value >= MATERIAL_GAP_THRESHOLD for value in default[size]),
                "default_median_WGap": statistics.median(default[size]),
            }
            for size in grouped
        },
        "sensitivity_thresholds": [0.05, 0.10, 0.15],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or safely resume the frozen regional fairness diagnostic."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    try:
        if args.dry_run:
            from .config import load_config
            from .experiment_suite import experiment_dry_run_report

            report = experiment_dry_run_report(load_config(args.config))
        else:
            from .regional_fairness_pipeline import run_regional_fairness_pipeline

            report = run_regional_fairness_pipeline(
                args.config,
                resume=bool(args.resume),
                overwrite=bool(args.overwrite),
            )
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    except KeyboardInterrupt:
        print("Regional fairness diagnostic interrupted safely.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001 - CLI failures must be explicit and nonzero.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
