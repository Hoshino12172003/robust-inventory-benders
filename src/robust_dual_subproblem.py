from __future__ import annotations

from dataclasses import dataclass
import math
import time

import gurobipy as gp
from gurobipy import GRB

from .instance import InventoryInstance
from .status import gurobi_status_name


@dataclass(frozen=True)
class RobustDualSubproblemResult:
    objective: float | None
    z_values: dict[tuple[int, int], float]
    lambda_values: dict[tuple[int, int], float]
    mu_values: dict[tuple[int, int], float]
    nu_values: dict[int, float]
    demand_values: dict[tuple[int, int], float]
    constant: float
    x_coefficients: dict[tuple[int, int], float]
    runtime: float
    status: str
    objective_bound: float | None
    mip_gap: float | None
    has_incumbent: bool
    requested_mip_gap: float | None

    def cut_value(self, x_values: dict[tuple[int, int], float]) -> float:
        if not self.has_incumbent:
            raise ValueError("A robust dual cut requires a feasible incumbent.")
        return self.constant + sum(coef * x_values[key] for key, coef in self.x_coefficients.items())


def solve_robust_dual_subproblem(
    instance: InventoryInstance,
    x_values: dict[tuple[int, int], float],
    gamma: int,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    output_flag: bool = False,
    excluded_patterns: list[dict[tuple[int, int], int]] | None = None,
) -> RobustDualSubproblemResult:
    start = time.perf_counter()
    gamma = min(max(0, int(gamma)), instance.num_regions * instance.num_products)

    model = gp.Model(f"robust_dual_gamma_{gamma}")
    model.Params.OutputFlag = 1 if output_flag else 0
    if time_limit is not None:
        model.Params.TimeLimit = max(1e-3, float(time_limit))
    if mip_gap is not None:
        model.Params.MIPGap = max(0.0, float(mip_gap))

    lambda_ub = {
        (r, j): float(instance.shortage_penalty[r][j] + instance.service_penalty[j])
        for r in instance.R
        for j in instance.J
    }
    mu_ub = {
        (i, j): max(lambda_ub[r, j] for r in instance.R)
        for i in instance.I
        for j in instance.J
    }

    lam = model.addVars(instance.R, instance.J, lb=0.0, ub=lambda_ub, name="lambda")
    mu = model.addVars(instance.I, instance.J, lb=0.0, ub=mu_ub, name="mu")
    nu = model.addVars(instance.J, lb=0.0, ub={j: instance.service_penalty[j] for j in instance.J}, name="nu")
    z = model.addVars(instance.R, instance.J, vtype=GRB.BINARY, name="z")
    w = model.addVars(instance.R, instance.J, lb=0.0, name="w")
    g = model.addVars(instance.R, instance.J, lb=0.0, name="g")

    for i in instance.I:
        for r in instance.R:
            for j in instance.J:
                model.addConstr(
                    lam[r, j] - mu[i, j] <= instance.transport_cost[i][r][j],
                    name=f"dual_q[{i},{r},{j}]",
                )

    for r in instance.R:
        for j in instance.J:
            model.addConstr(
                lam[r, j] - nu[j] <= instance.shortage_penalty[r][j],
                name=f"dual_u[{r},{j}]",
            )

    model.addConstrs((nu[j] <= instance.service_penalty[j] for j in instance.J), name="dual_e")
    model.addConstr(gp.quicksum(z[r, j] for r in instance.R for j in instance.J) <= gamma, name="budget")
    for pattern_index, pattern in enumerate(excluded_patterns or []):
        ones = [key for key, value in pattern.items() if int(round(value)) == 1]
        zeros = [key for key, value in pattern.items() if int(round(value)) == 0]
        model.addConstr(
            gp.quicksum(z[key] for key in ones) - gp.quicksum(z[key] for key in zeros) <= len(ones) - 1,
            name=f"exclude_pattern[{pattern_index}]",
        )

    for r in instance.R:
        for j in instance.J:
            lam_ub = lambda_ub[r, j]
            nu_ub = float(instance.service_penalty[j])

            model.addConstr(w[r, j] <= lam_ub * z[r, j], name=f"mccormick_w_ub_z[{r},{j}]")
            model.addConstr(w[r, j] <= lam[r, j], name=f"mccormick_w_ub_lam[{r},{j}]")
            model.addConstr(w[r, j] >= lam[r, j] - lam_ub * (1.0 - z[r, j]), name=f"mccormick_w_lb[{r},{j}]")

            model.addConstr(g[r, j] <= nu_ub * z[r, j], name=f"mccormick_g_ub_z[{r},{j}]")
            model.addConstr(g[r, j] <= nu[j], name=f"mccormick_g_ub_nu[{r},{j}]")
            model.addConstr(g[r, j] >= nu[j] - nu_ub * (1.0 - z[r, j]), name=f"mccormick_g_lb[{r},{j}]")

    objective = (
        gp.quicksum(instance.base_demand[r][j] * lam[r, j] for r in instance.R for j in instance.J)
        + gp.quicksum(instance.demand_deviation[r][j] * w[r, j] for r in instance.R for j in instance.J)
        - gp.quicksum(x_values[i, j] * mu[i, j] for i in instance.I for j in instance.J)
        - gp.quicksum(
            (1.0 - instance.service_level[j])
            * (
                gp.quicksum(instance.base_demand[r][j] for r in instance.R) * nu[j]
                + gp.quicksum(instance.demand_deviation[r][j] * g[r, j] for r in instance.R)
            )
            for j in instance.J
        )
    )
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()

    status = gurobi_status_name(model.Status)
    has_incumbent = model.SolCount > 0
    objective_bound = None
    try:
        objective_bound = float(model.ObjBound)
    except (AttributeError, gp.GurobiError):
        pass
    if objective_bound is not None and not math.isfinite(objective_bound):
        objective_bound = None
    if not has_incumbent:
        return RobustDualSubproblemResult(
            objective=None,
            z_values={},
            lambda_values={},
            mu_values={},
            nu_values={},
            demand_values={},
            constant=0.0,
            x_coefficients={},
            runtime=time.perf_counter() - start,
            status=status,
            objective_bound=objective_bound,
            mip_gap=None,
            has_incumbent=False,
            requested_mip_gap=mip_gap,
        )

    z_values = {(r, j): float(z[r, j].X) for r in instance.R for j in instance.J}
    lambda_values = {(r, j): float(lam[r, j].X) for r in instance.R for j in instance.J}
    mu_values = {(i, j): float(mu[i, j].X) for i in instance.I for j in instance.J}
    nu_values = {j: float(nu[j].X) for j in instance.J}
    demand_values = {
        (r, j): float(instance.base_demand[r][j] + instance.demand_deviation[r][j] * round(z_values[r, j]))
        for r in instance.R
        for j in instance.J
    }

    constant = sum(demand_values[r, j] * lambda_values[r, j] for r in instance.R for j in instance.J)
    constant -= sum(
        (1.0 - instance.service_level[j])
        * sum(demand_values[r, j] for r in instance.R)
        * nu_values[j]
        for j in instance.J
    )
    x_coefficients = {(i, j): -mu_values[i, j] for i in instance.I for j in instance.J}
    mip_gap_value = float(model.MIPGap) if model.IsMIP and model.SolCount else None

    return RobustDualSubproblemResult(
        objective=float(model.ObjVal),
        z_values=z_values,
        lambda_values=lambda_values,
        mu_values=mu_values,
        nu_values=nu_values,
        demand_values=demand_values,
        constant=float(constant),
        x_coefficients=x_coefficients,
        runtime=time.perf_counter() - start,
        status=status,
        objective_bound=objective_bound,
        mip_gap=mip_gap_value,
        has_incumbent=True,
        requested_mip_gap=mip_gap,
    )
