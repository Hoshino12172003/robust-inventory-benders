from __future__ import annotations

import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from .instance import InventoryInstance
from .results import SolveResult
from .scenarios import enumerate_budget_scenarios, scenario_metadata


def first_stage_cost_expr(instance: InventoryInstance, y: gp.tupledict, x: gp.tupledict) -> gp.LinExpr:
    return gp.quicksum(instance.fixed_cost[i] * y[i] for i in instance.I) + gp.quicksum(
        instance.inventory_cost[i][j] * x[i, j] for i in instance.I for j in instance.J
    )


def solve_monolithic(config: dict[str, Any], instance: InventoryInstance) -> SolveResult:
    robust_cfg = config.get("robust", {})
    benders_cfg = config.get("benders", {})
    gamma = int(robust_cfg.get("gamma_target", 0))
    max_scenarios = int(robust_cfg.get("max_scenarios", 5000))
    exact_scenarios = bool(robust_cfg.get("exact_scenarios", True))
    scenarios = enumerate_budget_scenarios(instance, gamma, max_scenarios, exact_scenarios=exact_scenarios)

    start = time.perf_counter()
    model = gp.Model("robust_inventory_monolithic")
    model.Params.OutputFlag = 1 if benders_cfg.get("output_flag", False) else 0
    model.Params.TimeLimit = float(benders_cfg.get("time_limit", 120))

    y = model.addVars(instance.I, vtype=GRB.BINARY, name="y")
    x = model.addVars(instance.I, instance.J, lb=0.0, name="x")
    theta = model.addVar(lb=0.0, name="theta")

    for i in instance.I:
        model.addConstr(
            gp.quicksum(instance.volume[j] * x[i, j] for j in instance.J) <= instance.capacity[i] * y[i],
            name=f"capacity[{i}]",
        )
        for j in instance.J:
            model.addConstr(x[i, j] <= instance.inventory_ub[i][j] * y[i], name=f"logic[{i},{j}]")

    model.addConstr(first_stage_cost_expr(instance, y, x) <= instance.budget, name="budget")

    for s, scenario in enumerate(scenarios):
        q = model.addVars(instance.I, instance.R, instance.J, lb=0.0, name=f"q_{s}")
        u = model.addVars(instance.R, instance.J, lb=0.0, name=f"u_{s}")
        e = model.addVars(instance.J, lb=0.0, name=f"e_{s}")

        for r in instance.R:
            for j in instance.J:
                model.addConstr(
                    gp.quicksum(q[i, r, j] for i in instance.I) + u[r, j] >= scenario.demand[r][j],
                    name=f"demand[{s},{r},{j}]",
                )
        for i in instance.I:
            for j in instance.J:
                model.addConstr(
                    gp.quicksum(q[i, r, j] for r in instance.R) <= x[i, j],
                    name=f"supply[{s},{i},{j}]",
                )
        for j in instance.J:
            model.addConstr(
                gp.quicksum(u[r, j] for r in instance.R) - e[j]
                <= (1.0 - instance.service_level[j]) * sum(scenario.demand[r][j] for r in instance.R),
                name=f"service[{s},{j}]",
            )

        scenario_cost = (
            gp.quicksum(
                instance.transport_cost[i][r][j] * q[i, r, j]
                for i in instance.I
                for r in instance.R
                for j in instance.J
            )
            + gp.quicksum(instance.shortage_penalty[r][j] * u[r, j] for r in instance.R for j in instance.J)
            + gp.quicksum(instance.service_penalty[j] * e[j] for j in instance.J)
        )
        model.addConstr(theta >= scenario_cost, name=f"robust_theta[{s}]")

    first_stage = first_stage_cost_expr(instance, y, x)
    model.setObjective(first_stage + theta, GRB.MINIMIZE)
    model.optimize()
    runtime = time.perf_counter() - start

    status = "optimal" if model.Status == GRB.OPTIMAL else f"gurobi_status_{model.Status}"
    objective = float(model.ObjVal) if model.SolCount else None
    first_stage_value = float(first_stage.getValue()) if model.SolCount else None
    robust_cost = float(theta.X) if model.SolCount else None
    bound = float(model.ObjBound) if model.SolCount else None
    gap = float(model.MIPGap) if model.SolCount and model.IsMIP else 0.0
    return SolveResult(
        method="monolithic",
        status=status,
        objective=objective,
        lower_bound=bound,
        upper_bound=objective,
        gap=gap,
        runtime=runtime,
        robust_cost=robust_cost,
        first_stage_cost=first_stage_value,
        gamma_target=gamma,
        metadata={
            "num_scenarios": len(scenarios),
            **scenario_metadata(instance, gamma, max_scenarios, exact_scenarios, len(scenarios)),
        },
    )
