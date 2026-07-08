from __future__ import annotations

from dataclasses import dataclass
import time

import gurobipy as gp
from gurobipy import GRB

from .instance import InventoryInstance
from .scenarios import DemandScenario


@dataclass(frozen=True)
class SubproblemResult:
    scenario_name: str
    objective: float
    constant: float
    x_coefficients: dict[tuple[int, int], float]
    runtime: float

    def cut_value(self, x_values: dict[tuple[int, int], float]) -> float:
        return self.constant + sum(coef * x_values[key] for key, coef in self.x_coefficients.items())


def solve_recourse_subproblem(
    instance: InventoryInstance,
    scenario: DemandScenario,
    x_values: dict[tuple[int, int], float],
    output_flag: bool = False,
) -> SubproblemResult:
    start = time.perf_counter()
    model = gp.Model(f"recourse_{scenario.name}")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.Method = 1

    q = model.addVars(instance.I, instance.R, instance.J, lb=0.0, name="q")
    u = model.addVars(instance.R, instance.J, lb=0.0, name="u")
    e = model.addVars(instance.J, lb=0.0, name="e")

    demand_ct = {}
    for r in instance.R:
        for j in instance.J:
            demand_ct[r, j] = model.addConstr(
                gp.quicksum(q[i, r, j] for i in instance.I) + u[r, j] >= scenario.demand[r][j],
                name=f"demand[{r},{j}]",
            )

    supply_ct = {}
    for i in instance.I:
        for j in instance.J:
            supply_ct[i, j] = model.addConstr(
                gp.quicksum(q[i, r, j] for r in instance.R) <= x_values[i, j],
                name=f"supply[{i},{j}]",
            )

    service_ct = {}
    for j in instance.J:
        rhs = (1.0 - instance.service_level[j]) * sum(scenario.demand[r][j] for r in instance.R)
        service_ct[j] = model.addConstr(
            gp.quicksum(u[r, j] for r in instance.R) - e[j] <= rhs,
            name=f"service[{j}]",
        )

    model.setObjective(
        gp.quicksum(
            instance.transport_cost[i][r][j] * q[i, r, j]
            for i in instance.I
            for r in instance.R
            for j in instance.J
        )
        + gp.quicksum(instance.shortage_penalty[r][j] * u[r, j] for r in instance.R for j in instance.J)
        + gp.quicksum(instance.service_penalty[j] * e[j] for j in instance.J),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Subproblem {scenario.name} did not solve to optimality: status {model.Status}")

    constant = 0.0
    for r in instance.R:
        for j in instance.J:
            constant += scenario.demand[r][j] * demand_ct[r, j].Pi
    for j in instance.J:
        rhs = (1.0 - instance.service_level[j]) * sum(scenario.demand[r][j] for r in instance.R)
        constant += rhs * service_ct[j].Pi

    x_coefficients = {(i, j): supply_ct[i, j].Pi for i in instance.I for j in instance.J}
    return SubproblemResult(
        scenario_name=scenario.name,
        objective=float(model.ObjVal),
        constant=float(constant),
        x_coefficients={k: float(v) for k, v in x_coefficients.items()},
        runtime=time.perf_counter() - start,
    )
