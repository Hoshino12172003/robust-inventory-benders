from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from .instance import InventoryInstance
from .robust_dual_subproblem import solve_robust_dual_subproblem
from .status import gurobi_status_name


@dataclass(frozen=True)
class ManagerialEvaluationResult:
    opened_warehouses: int | None = None
    total_inventory: float | None = None
    inventory_by_product: list[float] | None = None
    inventory_by_warehouse: list[float] | None = None
    fixed_opening_cost: float | None = None
    inventory_cost: float | None = None
    first_stage_cost: float | None = None
    worst_case_recourse_cost: float | None = None
    transport_cost: float | None = None
    shortage_cost: float | None = None
    service_violation_cost: float | None = None
    total_worst_case_demand: float | None = None
    total_shortage: float | None = None
    shortage_by_product: list[float] | None = None
    service_violation: float | None = None
    service_violation_by_product: list[float] | None = None
    realized_fill_rate: float | None = None
    worst_case_active_deviations: list[dict[str, int]] | None = None
    worst_case_demand_values: list[list[float]] | None = None
    managerial_evaluation_status: str = "not_run"
    managerial_evaluation_runtime: float = 0.0
    managerial_metrics_valid: bool = False
    managerial_evaluation_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _RecourseMetrics:
    status: str
    objective: float | None
    transport_cost: float | None
    shortage_cost: float | None
    service_violation_cost: float | None
    shortage_values: list[list[float]] = field(default_factory=list)
    service_violation_values: list[float] = field(default_factory=list)
    error: str = ""


def invalid_managerial_evaluation(
    status: str,
    error: str,
    runtime: float,
) -> ManagerialEvaluationResult:
    return ManagerialEvaluationResult(
        managerial_evaluation_status=status,
        managerial_evaluation_runtime=max(0.0, float(runtime)),
        managerial_metrics_valid=False,
        managerial_evaluation_error=str(error),
    )


def summarize_managerial_metrics(
    *,
    instance: InventoryInstance,
    y_values: list[float],
    x_values: list[list[float]],
    demand_values: list[list[float]],
    active_deviations: list[dict[str, int]],
    recourse_objective: float,
    transport_cost: float,
    shortage_cost: float,
    service_violation_cost: float,
    shortage_values: list[list[float]],
    service_violation_values: list[float],
    runtime: float,
) -> ManagerialEvaluationResult:
    if len(y_values) != instance.num_warehouses:
        raise ValueError("best_y_values has the wrong number of warehouses.")
    if len(x_values) != instance.num_warehouses or any(
        len(row) != instance.num_products for row in x_values
    ):
        raise ValueError("best_x_values has the wrong shape.")
    if len(demand_values) != instance.num_regions or any(
        len(row) != instance.num_products for row in demand_values
    ):
        raise ValueError("worst-case demand has the wrong shape.")
    if len(shortage_values) != instance.num_regions or any(
        len(row) != instance.num_products for row in shortage_values
    ):
        raise ValueError("shortage values have the wrong shape.")
    if len(service_violation_values) != instance.num_products:
        raise ValueError("service violation values have the wrong shape.")

    clean_y = [float(value) for value in y_values]
    clean_x = [[float(value) for value in row] for row in x_values]
    inventory_by_product = [
        sum(clean_x[i][j] for i in instance.I) for j in instance.J
    ]
    inventory_by_warehouse = [sum(clean_x[i]) for i in instance.I]
    fixed_opening_cost = sum(instance.fixed_cost[i] * clean_y[i] for i in instance.I)
    inventory_cost = sum(
        instance.inventory_cost[i][j] * clean_x[i][j]
        for i in instance.I
        for j in instance.J
    )
    shortage_by_product = [
        sum(float(shortage_values[r][j]) for r in instance.R) for j in instance.J
    ]
    total_shortage = sum(shortage_by_product)
    total_demand = sum(float(demand_values[r][j]) for r in instance.R for j in instance.J)
    fill_rate = None if abs(total_demand) <= 1e-12 else 1.0 - total_shortage / total_demand

    return ManagerialEvaluationResult(
        opened_warehouses=sum(1 for value in clean_y if value >= 0.5),
        total_inventory=sum(inventory_by_product),
        inventory_by_product=inventory_by_product,
        inventory_by_warehouse=inventory_by_warehouse,
        fixed_opening_cost=float(fixed_opening_cost),
        inventory_cost=float(inventory_cost),
        first_stage_cost=float(fixed_opening_cost + inventory_cost),
        worst_case_recourse_cost=float(recourse_objective),
        transport_cost=float(transport_cost),
        shortage_cost=float(shortage_cost),
        service_violation_cost=float(service_violation_cost),
        total_worst_case_demand=float(total_demand),
        total_shortage=float(total_shortage),
        shortage_by_product=shortage_by_product,
        service_violation=float(sum(float(value) for value in service_violation_values)),
        service_violation_by_product=[float(value) for value in service_violation_values],
        realized_fill_rate=None if fill_rate is None else float(fill_rate),
        worst_case_active_deviations=active_deviations,
        worst_case_demand_values=[
            [float(value) for value in row] for row in demand_values
        ],
        managerial_evaluation_status="optimal",
        managerial_evaluation_runtime=max(0.0, float(runtime)),
        managerial_metrics_valid=True,
        managerial_evaluation_error="",
    )


def _solve_recourse_for_metrics(
    instance: InventoryInstance,
    x_values: list[list[float]],
    demand_values: list[list[float]],
    *,
    time_limit: float,
    output_flag: bool,
) -> _RecourseMetrics:
    model = gp.Model("managerial_recourse")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.Method = 1
    model.Params.TimeLimit = max(1e-3, float(time_limit))

    q = model.addVars(instance.I, instance.R, instance.J, lb=0.0, name="q")
    u = model.addVars(instance.R, instance.J, lb=0.0, name="u")
    e = model.addVars(instance.J, lb=0.0, name="e")
    for r in instance.R:
        for j in instance.J:
            model.addConstr(
                gp.quicksum(q[i, r, j] for i in instance.I) + u[r, j]
                >= demand_values[r][j],
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
            * sum(demand_values[r][j] for r in instance.R),
            name=f"service[{j}]",
        )

    transport_expression = gp.quicksum(
        instance.transport_cost[i][r][j] * q[i, r, j]
        for i in instance.I
        for r in instance.R
        for j in instance.J
    )
    shortage_expression = gp.quicksum(
        instance.shortage_penalty[r][j] * u[r, j]
        for r in instance.R
        for j in instance.J
    )
    service_expression = gp.quicksum(
        instance.service_penalty[j] * e[j] for j in instance.J
    )
    model.setObjective(
        transport_expression + shortage_expression + service_expression,
        GRB.MINIMIZE,
    )
    model.optimize()
    status = gurobi_status_name(model.Status)
    if model.Status != GRB.OPTIMAL:
        return _RecourseMetrics(
            status=status,
            objective=None,
            transport_cost=None,
            shortage_cost=None,
            service_violation_cost=None,
            error=f"Exact recourse evaluation ended with status {status}.",
        )
    return _RecourseMetrics(
        status=status,
        objective=float(model.ObjVal),
        transport_cost=float(transport_expression.getValue()),
        shortage_cost=float(shortage_expression.getValue()),
        service_violation_cost=float(service_expression.getValue()),
        shortage_values=[
            [float(u[r, j].X) for j in instance.J] for r in instance.R
        ],
        service_violation_values=[float(e[j].X) for j in instance.J],
    )


def evaluate_managerial_solution(
    instance: InventoryInstance,
    *,
    best_y_values: list[float] | None,
    best_x_values: list[list[float]] | None,
    gamma_target: int,
    time_limit: float = 300.0,
    output_flag: bool = False,
) -> ManagerialEvaluationResult:
    start = time.perf_counter()
    if best_y_values is None or best_x_values is None:
        return invalid_managerial_evaluation(
            "missing_first_stage_incumbent",
            "No JSON-serializable best_y_values/best_x_values were saved.",
            time.perf_counter() - start,
        )
    try:
        if len(best_x_values) != instance.num_warehouses or any(
            len(row) != instance.num_products for row in best_x_values
        ):
            raise ValueError("best_x_values has the wrong shape.")
        if len(best_y_values) != instance.num_warehouses:
            raise ValueError("best_y_values has the wrong shape.")
        x_dict = {
            (i, j): float(best_x_values[i][j])
            for i in instance.I
            for j in instance.J
        }
        robust = solve_robust_dual_subproblem(
            instance,
            x_dict,
            gamma_target,
            time_limit=max(1e-3, float(time_limit)),
            mip_gap=0.0,
            output_flag=output_flag,
        )
        if robust.status != "optimal" or not robust.has_incumbent:
            return invalid_managerial_evaluation(
                f"robust_dual_{robust.status}",
                "Exact robust-dual evaluation did not return an optimal incumbent; no metrics were inferred.",
                time.perf_counter() - start,
            )
        demand_values = [
            [float(robust.demand_values[r, j]) for j in instance.J]
            for r in instance.R
        ]
        remaining = float(time_limit) - (time.perf_counter() - start)
        if remaining <= 0.0:
            return invalid_managerial_evaluation(
                "managerial_evaluation_time_limit",
                "No time remained for exact recourse evaluation.",
                time.perf_counter() - start,
            )
        recourse = _solve_recourse_for_metrics(
            instance,
            best_x_values,
            demand_values,
            time_limit=remaining,
            output_flag=output_flag,
        )
        if recourse.status != "optimal" or recourse.objective is None:
            return invalid_managerial_evaluation(
                f"recourse_{recourse.status}",
                recourse.error,
                time.perf_counter() - start,
            )
        active_deviations = [
            {"region": r, "product": j}
            for r in instance.R
            for j in instance.J
            if int(round(robust.z_values[r, j])) == 1
        ]
        return summarize_managerial_metrics(
            instance=instance,
            y_values=best_y_values,
            x_values=best_x_values,
            demand_values=demand_values,
            active_deviations=active_deviations,
            recourse_objective=recourse.objective,
            transport_cost=float(recourse.transport_cost),
            shortage_cost=float(recourse.shortage_cost),
            service_violation_cost=float(recourse.service_violation_cost),
            shortage_values=recourse.shortage_values,
            service_violation_values=recourse.service_violation_values,
            runtime=time.perf_counter() - start,
        )
    except Exception as exc:  # noqa: BLE001 - failure must be recorded without invented metrics.
        return invalid_managerial_evaluation(
            "managerial_evaluation_failed",
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - start,
        )
