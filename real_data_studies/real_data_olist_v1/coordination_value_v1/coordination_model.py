from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import gurobipy as gp
from gurobipy import GRB

from src.instance import InventoryInstance
from src.scenarios import DemandScenario
from src.status import gurobi_status_name


Policy = Literal["flexible_multiwarehouse", "optimized_single_source"]


@dataclass
class ScenarioVariables:
    q: gp.tupledict
    u: gp.tupledict
    e: gp.tupledict
    recourse: gp.LinExpr


def _first_stage_expression(instance: InventoryInstance, y, x) -> gp.LinExpr:
    return (
        gp.quicksum(instance.fixed_cost[i] * y[i] for i in instance.I)
        + gp.quicksum(
            instance.inventory_cost[i][j] * x[i, j]
            for i in instance.I
            for j in instance.J
        )
    )


def _build_first_stage(
    instance: InventoryInstance,
    policy: Policy,
    *,
    time_limit: float,
    mip_gap: float,
    feasibility_tolerance: float,
    threads: int,
    solver_seed: int,
):
    if policy not in ("flexible_multiwarehouse", "optimized_single_source"):
        raise ValueError(f"unsupported fulfillment policy: {policy}")
    model = gp.Model(f"coordination_{policy}")
    model.Params.OutputFlag = 0
    model.Params.Threads = int(threads)
    model.Params.Seed = int(solver_seed)
    model.Params.MIPGap = float(mip_gap)
    model.Params.FeasibilityTol = float(feasibility_tolerance)
    model.Params.TimeLimit = float(time_limit)
    y = model.addVars(instance.I, vtype=GRB.BINARY, name="y")
    x = model.addVars(instance.I, instance.J, lb=0.0, name="x")
    for i in instance.I:
        model.addConstr(
            gp.quicksum(instance.volume[j] * x[i, j] for j in instance.J)
            <= instance.capacity[i] * y[i],
            name=f"warehouse_capacity[{i}]",
        )
        for j in instance.J:
            model.addConstr(
                x[i, j] <= instance.inventory_ub[i][j] * y[i],
                name=f"inventory_opening_link[{i},{j}]",
            )
    first_stage = _first_stage_expression(instance, y, x)
    model.addConstr(first_stage <= instance.budget, name="capital_budget")

    z = None
    if policy == "optimized_single_source":
        z = model.addVars(instance.I, instance.R, vtype=GRB.BINARY, name="z")
        for r in instance.R:
            model.addConstr(
                gp.quicksum(z[i, r] for i in instance.I) == 1,
                name=f"single_source_assignment[{r}]",
            )
            for i in instance.I:
                model.addConstr(z[i, r] <= y[i], name=f"assignment_opening_link[{i},{r}]")
    return model, y, x, z, first_stage


def _add_recourse(
    model: gp.Model,
    instance: InventoryInstance,
    scenario: DemandScenario,
    scenario_index: int,
    policy: Policy,
    x,
    z,
) -> ScenarioVariables:
    q = model.addVars(
        instance.I,
        instance.R,
        instance.J,
        lb=0.0,
        name=f"q[{scenario_index}]",
    )
    u = model.addVars(
        instance.R,
        instance.J,
        lb=0.0,
        name=f"u[{scenario_index}]",
    )
    e = model.addVars(instance.J, lb=0.0, name=f"e[{scenario_index}]")
    for r in instance.R:
        for j in instance.J:
            demand = float(scenario.demand[r][j])
            model.addConstr(
                gp.quicksum(q[i, r, j] for i in instance.I) + u[r, j] >= demand,
                name=f"demand_balance[{scenario_index},{r},{j}]",
            )
            if policy == "optimized_single_source":
                for i in instance.I:
                    model.addConstr(
                        q[i, r, j] <= demand * z[i, r],
                        name=f"single_source_shipment[{scenario_index},{i},{r},{j}]",
                    )
    for i in instance.I:
        for j in instance.J:
            model.addConstr(
                gp.quicksum(q[i, r, j] for r in instance.R) <= x[i, j],
                name=f"inventory_availability[{scenario_index},{i},{j}]",
            )
    for j in instance.J:
        total_product_demand = sum(float(scenario.demand[r][j]) for r in instance.R)
        model.addConstr(
            gp.quicksum(u[r, j] for r in instance.R) - e[j]
            <= (1.0 - instance.service_level[j]) * total_product_demand,
            name=f"product_service_target[{scenario_index},{j}]",
        )
    recourse = (
        gp.quicksum(
            instance.transport_cost[i][r][j] * q[i, r, j]
            for i in instance.I
            for r in instance.R
            for j in instance.J
        )
        + gp.quicksum(
            instance.shortage_penalty[r][j] * u[r, j]
            for r in instance.R
            for j in instance.J
        )
        + gp.quicksum(instance.service_penalty[j] * e[j] for j in instance.J)
    )
    return ScenarioVariables(q=q, u=u, e=e, recourse=recourse)


def _nearest_warehouse_by_region(instance: InventoryInstance) -> dict[int, int]:
    return {
        r: min(
            instance.I,
            key=lambda i: sum(instance.transport_cost[i][r][j] for j in instance.J)
            / max(1, instance.num_products),
        )
        for r in instance.R
    }


def _extract_metrics(
    instance: InventoryInstance,
    scenarios: list[DemandScenario],
    policy: Policy,
    y,
    x,
    z,
    recourse_variables: list[ScenarioVariables],
    first_stage_value: float,
    severe_shortage_threshold: float,
) -> dict:
    tolerance = 1.0e-7
    facility_cost = sum(instance.fixed_cost[i] * y[i].X for i in instance.I)
    inventory_cost = sum(
        instance.inventory_cost[i][j] * x[i, j].X
        for i in instance.I
        for j in instance.J
    )
    nearest = _nearest_warehouse_by_region(instance)
    scenario_rows = []
    all_shortage = 0.0
    all_demand = 0.0
    all_shipments = 0.0
    nonlocal_shipments = 0.0
    source_counts = []
    severe_observations = 0
    region_worst = {r: 0.0 for r in instance.R}
    for index, scenario in enumerate(scenarios):
        variables = recourse_variables[index]
        transport_cost = sum(
            instance.transport_cost[i][r][j] * variables.q[i, r, j].X
            for i in instance.I
            for r in instance.R
            for j in instance.J
        )
        shortage_cost = sum(
            instance.shortage_penalty[r][j] * variables.u[r, j].X
            for r in instance.R
            for j in instance.J
        )
        service_violation_cost = sum(
            instance.service_penalty[j] * variables.e[j].X for j in instance.J
        )
        shortage_units = sum(
            variables.u[r, j].X for r in instance.R for j in instance.J
        )
        demand_units = sum(
            float(scenario.demand[r][j]) for r in instance.R for j in instance.J
        )
        worst_regional_rate = 0.0
        for r in instance.R:
            region_demand = sum(float(scenario.demand[r][j]) for j in instance.J)
            region_shortage = sum(variables.u[r, j].X for j in instance.J)
            rate = region_shortage / region_demand if region_demand > 0 else 0.0
            region_worst[r] = max(region_worst[r], rate)
            worst_regional_rate = max(worst_regional_rate, rate)
            severe_observations += int(rate > severe_shortage_threshold + tolerance)
            active_sources = sum(
                any(variables.q[i, r, j].X > tolerance for j in instance.J)
                for i in instance.I
            )
            source_counts.append(active_sources)
        shipment_units = sum(
            variables.q[i, r, j].X
            for i in instance.I
            for r in instance.R
            for j in instance.J
        )
        scenario_nonlocal = sum(
            variables.q[i, r, j].X
            for i in instance.I
            for r in instance.R
            for j in instance.J
            if i != nearest[r]
        )
        all_shortage += shortage_units
        all_demand += demand_units
        all_shipments += shipment_units
        nonlocal_shipments += scenario_nonlocal
        scenario_rows.append(
            {
                "scenario": scenario.name,
                "transport_cost": transport_cost,
                "shortage_cost": shortage_cost,
                "service_violation_cost": service_violation_cost,
                "recourse_cost": transport_cost + shortage_cost + service_violation_cost,
                "shortage_units": shortage_units,
                "demand_units": demand_units,
                "shortage_rate": shortage_units / demand_units if demand_units > 0 else 0.0,
                "worst_regional_shortage_rate": worst_regional_rate,
                "nonlocal_fulfillment_share": (
                    scenario_nonlocal / shipment_units if shipment_units > 0 else 0.0
                ),
            }
        )
    assignment = None
    if z is not None:
        assignment = {
            str(r): next(i for i in instance.I if z[i, r].X > 0.5) for r in instance.R
        }
    recourse_costs = [row["recourse_cost"] for row in scenario_rows]
    return {
        "policy": policy,
        "first_stage_cost": first_stage_value,
        "facility_cost": facility_cost,
        "inventory_cost": inventory_cost,
        "robust_recourse_cost": max(recourse_costs),
        "robust_total_cost": first_stage_value + max(recourse_costs),
        "mean_transport_cost": sum(row["transport_cost"] for row in scenario_rows) / len(scenario_rows),
        "worst_transport_cost": max(row["transport_cost"] for row in scenario_rows),
        "mean_shortage_cost": sum(row["shortage_cost"] for row in scenario_rows) / len(scenario_rows),
        "worst_shortage_cost": max(row["shortage_cost"] for row in scenario_rows),
        "worst_regional_shortage_rate": max(region_worst.values()),
        "demand_weighted_shortage_rate": all_shortage / all_demand if all_demand > 0 else 0.0,
        "severe_region_scenario_count": severe_observations,
        "severe_region_count": sum(
            value > severe_shortage_threshold + tolerance for value in region_worst.values()
        ),
        "nonlocal_fulfillment_share": (
            nonlocal_shipments / all_shipments if all_shipments > 0 else 0.0
        ),
        "mean_active_sources_per_region_scenario": (
            sum(source_counts) / len(source_counts) if source_counts else 0.0
        ),
        "maximum_active_sources_per_region_scenario": max(source_counts, default=0),
        "open_facility_count": sum(y[i].X > 0.5 for i in instance.I),
        "open_facilities": [i for i in instance.I if y[i].X > 0.5],
        "total_inventory_units": sum(x[i, j].X for i in instance.I for j in instance.J),
        "nearest_warehouse_by_region": {str(key): value for key, value in nearest.items()},
        "single_source_assignment": assignment,
        "scenario_metrics": scenario_rows,
    }


def solve_policy(
    instance: InventoryInstance,
    scenarios: list[DemandScenario],
    policy: Policy,
    *,
    objective: Literal["cost_anchor", "service_protection"],
    severe_shortage_threshold: float,
    time_limit: float,
    mip_gap: float,
    feasibility_tolerance: float,
    threads: int,
    solver_seed: int,
    cost_budget: float | None = None,
) -> dict:
    if not scenarios:
        raise ValueError("at least one demand scenario is required")
    if objective == "service_protection" and cost_budget is None:
        raise ValueError("service protection requires an absolute cost budget")
    start = time.perf_counter()
    model, y, x, z, first_stage = _build_first_stage(
        instance,
        policy,
        time_limit=time_limit,
        mip_gap=mip_gap,
        feasibility_tolerance=feasibility_tolerance,
        threads=threads,
        solver_seed=solver_seed,
    )
    recourse_variables = [
        _add_recourse(model, instance, scenario, index, policy, x, z)
        for index, scenario in enumerate(scenarios)
    ]
    t = None
    theta = None
    if objective == "cost_anchor":
        theta = model.addVar(lb=0.0, name="theta")
        for index, variables in enumerate(recourse_variables):
            model.addConstr(theta >= variables.recourse, name=f"worst_recourse[{index}]")
        model.setObjective(first_stage + theta, GRB.MINIMIZE)
    elif objective == "service_protection":
        t = model.addVar(lb=0.0, ub=1.0, name="worst_regional_shortage_rate")
        for index, (scenario, variables) in enumerate(zip(scenarios, recourse_variables)):
            model.addConstr(
                first_stage + variables.recourse <= float(cost_budget),
                name=f"common_cost_budget[{index}]",
            )
            for r in instance.R:
                region_demand = sum(float(scenario.demand[r][j]) for j in instance.J)
                model.addConstr(
                    gp.quicksum(variables.u[r, j] for j in instance.J)
                    <= region_demand * t,
                    name=f"regional_shortage_protection[{index},{r}]",
                )
        model.setObjective(t, GRB.MINIMIZE)
    else:
        raise ValueError(f"unsupported objective: {objective}")
    model.optimize()
    status = gurobi_status_name(model.Status)
    result = {
        "policy": policy,
        "objective": objective,
        "status": status,
        "runtime_seconds": time.perf_counter() - start,
        "solution_count": int(model.SolCount),
        "objective_value": float(model.ObjVal) if model.SolCount else None,
        "lower_bound": float(model.ObjBound) if model.SolCount else None,
        "mip_gap": float(model.MIPGap) if model.SolCount else None,
        "cost_budget": cost_budget,
        "certified": status == "optimal" and model.SolCount > 0,
    }
    if model.SolCount:
        first_stage_value = float(first_stage.getValue())
        result["metrics"] = _extract_metrics(
            instance,
            scenarios,
            policy,
            y,
            x,
            z,
            recourse_variables,
            first_stage_value,
            severe_shortage_threshold,
        )
        result["y_values"] = [float(y[i].X) for i in instance.I]
        result["x_values"] = [
            [float(x[i, j].X) for j in instance.J] for i in instance.I
        ]
        if t is not None:
            result["worst_regional_shortage_rate"] = float(t.X)
        if theta is not None:
            result["worst_recourse_cost"] = float(theta.X)
    model.dispose()
    return result
