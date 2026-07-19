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


@dataclass(frozen=True)
class FixedPatternDualLPResult:
    objective: float | None
    lambda_values: dict[tuple[int, int], float]
    mu_values: dict[tuple[int, int], float]
    nu_values: dict[int, float]
    demand_values: dict[tuple[int, int], float]
    constant: float
    x_coefficients: dict[tuple[int, int], float]
    runtime: float
    status: str
    has_solution: bool
    dual_feasible: bool
    num_binary_variables: int
    is_mip: bool

    def cut_value(self, x_values: dict[tuple[int, int], float]) -> float:
        if not self.has_solution:
            raise ValueError("A fixed-pattern dual cut requires an optimal solution.")
        return self.constant + sum(
            coefficient * x_values[key]
            for key, coefficient in self.x_coefficients.items()
        )


@dataclass(frozen=True)
class CorePointStrengtheningSolveResult:
    strengthened_cut: RobustDualSubproblemResult | None
    stage1_status: str
    stage1_runtime: float
    stage1_objective: float | None
    stage2_status: str
    stage2_runtime: float
    original_value_at_current: float | None
    strengthened_value_at_current: float | None
    original_value_at_core: float | None
    strengthened_value_at_core: float | None
    current_value_floor: float | None
    dual_feasible: bool
    auxiliary_bound_used_for_ub: bool
    failure_reason: str | None


def discretize_robust_pattern(
    instance: InventoryInstance,
    z_values: dict[tuple[int, int], float],
    *,
    tolerance: float = 1.0e-5,
) -> dict[tuple[int, int], int] | None:
    expected = {(r, j) for r in instance.R for j in instance.J}
    if set(z_values) != expected:
        return None
    pattern: dict[tuple[int, int], int] = {}
    for key in sorted(expected):
        value = float(z_values[key])
        if not math.isfinite(value):
            return None
        rounded = int(round(value))
        if rounded not in {0, 1} or abs(value - rounded) > tolerance:
            return None
        pattern[key] = rounded
    return pattern


def _fixed_pattern_demand(
    instance: InventoryInstance,
    pattern: dict[tuple[int, int], int],
) -> dict[tuple[int, int], float]:
    return {
        (r, j): float(
            instance.base_demand[r][j]
            + instance.demand_deviation[r][j] * pattern[r, j]
        )
        for r in instance.R
        for j in instance.J
    }


def _dual_cut_components(
    instance: InventoryInstance,
    demand_values: dict[tuple[int, int], float],
    lambda_values: dict[tuple[int, int], float],
    mu_values: dict[tuple[int, int], float],
    nu_values: dict[int, float],
) -> tuple[float, dict[tuple[int, int], float]]:
    constant = sum(
        demand_values[r, j] * lambda_values[r, j]
        for r in instance.R
        for j in instance.J
    )
    constant -= sum(
        (1.0 - instance.service_level[j])
        * sum(demand_values[r, j] for r in instance.R)
        * nu_values[j]
        for j in instance.J
    )
    return float(constant), {
        (i, j): -float(mu_values[i, j])
        for i in instance.I
        for j in instance.J
    }


def _fixed_pattern_dual_feasible(
    instance: InventoryInstance,
    lambda_values: dict[tuple[int, int], float],
    mu_values: dict[tuple[int, int], float],
    nu_values: dict[int, float],
    *,
    tolerance: float = 1.0e-6,
) -> bool:
    if any(value < -tolerance or not math.isfinite(value) for value in lambda_values.values()):
        return False
    if any(value < -tolerance or not math.isfinite(value) for value in mu_values.values()):
        return False
    if any(value < -tolerance or not math.isfinite(value) for value in nu_values.values()):
        return False
    if any(nu_values[j] > instance.service_penalty[j] + tolerance for j in instance.J):
        return False
    if any(
        lambda_values[r, j] - mu_values[i, j]
        > instance.transport_cost[i][r][j] + tolerance
        for i in instance.I
        for r in instance.R
        for j in instance.J
    ):
        return False
    return not any(
        lambda_values[r, j] - nu_values[j]
        > instance.shortage_penalty[r][j] + tolerance
        for r in instance.R
        for j in instance.J
    )


def solve_fixed_pattern_dual_lp(
    instance: InventoryInstance,
    objective_x_values: dict[tuple[int, int], float],
    z_values: dict[tuple[int, int], float],
    *,
    time_limit: float,
    output_flag: bool = False,
    current_x_values: dict[tuple[int, int], float] | None = None,
    current_value_floor: float | None = None,
) -> FixedPatternDualLPResult:
    """Solve a continuous fixed-pattern dual LP used only for cut strengthening."""
    start = time.perf_counter()
    pattern = discretize_robust_pattern(instance, z_values)
    if pattern is None:
        return FixedPatternDualLPResult(
            None, {}, {}, {}, {}, 0.0, {}, 0.0, "invalid_pattern", False, False, 0, False
        )
    demand_values = _fixed_pattern_demand(instance, pattern)
    model = gp.Model("fixed_pattern_dual_lp")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.TimeLimit = max(0.0, float(time_limit))
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
    nu = model.addVars(
        instance.J,
        lb=0.0,
        ub={j: instance.service_penalty[j] for j in instance.J},
        name="nu",
    )
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

    def affine_objective(x_values: dict[tuple[int, int], float]) -> gp.LinExpr:
        return (
            gp.quicksum(
                demand_values[r, j] * lam[r, j]
                for r in instance.R
                for j in instance.J
            )
            - gp.quicksum(
                x_values[i, j] * mu[i, j]
                for i in instance.I
                for j in instance.J
            )
            - gp.quicksum(
                (1.0 - instance.service_level[j])
                * sum(demand_values[r, j] for r in instance.R)
                * nu[j]
                for j in instance.J
            )
        )

    if current_x_values is not None or current_value_floor is not None:
        if current_x_values is None or current_value_floor is None:
            raise ValueError("current_x_values and current_value_floor must be supplied together")
        model.addConstr(
            affine_objective(current_x_values) >= float(current_value_floor),
            name="current_point_value_floor",
        )
    model.setObjective(affine_objective(objective_x_values), GRB.MAXIMIZE)
    model.optimize()
    runtime = time.perf_counter() - start
    status = gurobi_status_name(model.Status)
    optimal = model.Status == GRB.OPTIMAL and model.SolCount > 0
    if not optimal:
        return FixedPatternDualLPResult(
            None,
            {},
            {},
            {},
            demand_values,
            0.0,
            {},
            runtime,
            status,
            False,
            False,
            int(model.NumBinVars),
            bool(model.IsMIP),
        )
    lambda_values = {(r, j): float(lam[r, j].X) for r in instance.R for j in instance.J}
    mu_values = {(i, j): float(mu[i, j].X) for i in instance.I for j in instance.J}
    nu_values = {j: float(nu[j].X) for j in instance.J}
    constant, coefficients = _dual_cut_components(
        instance,
        demand_values,
        lambda_values,
        mu_values,
        nu_values,
    )
    return FixedPatternDualLPResult(
        objective=float(model.ObjVal),
        lambda_values=lambda_values,
        mu_values=mu_values,
        nu_values=nu_values,
        demand_values=demand_values,
        constant=constant,
        x_coefficients=coefficients,
        runtime=runtime,
        status=status,
        has_solution=True,
        dual_feasible=_fixed_pattern_dual_feasible(
            instance,
            lambda_values,
            mu_values,
            nu_values,
        ),
        num_binary_variables=int(model.NumBinVars),
        is_mip=bool(model.IsMIP),
    )


def solve_core_point_strengthened_dual_cut(
    instance: InventoryInstance,
    current_x_values: dict[tuple[int, int], float],
    core_x_values: dict[tuple[int, int], float],
    original_cut: RobustDualSubproblemResult,
    *,
    stage1_time_limit: float,
    stage2_time_limit: float,
    remaining_global_time: float,
    current_abs_tol: float,
    current_rel_tol: float,
    output_flag: bool = False,
) -> CorePointStrengtheningSolveResult:
    stage1_limit = min(float(stage1_time_limit), max(0.0, float(remaining_global_time)))
    original_current = original_cut.cut_value(current_x_values)
    original_core = original_cut.cut_value(core_x_values)
    if stage1_limit <= 0.0:
        return CorePointStrengtheningSolveResult(
            None, "not_run", 0.0, None, "not_run", 0.0,
            original_current, None, original_core, None, None, False, False,
            "stage1_time_unavailable",
        )
    stage1 = solve_fixed_pattern_dual_lp(
        instance,
        current_x_values,
        original_cut.z_values,
        time_limit=stage1_limit,
        output_flag=output_flag,
    )
    if stage1.status != "optimal" or not stage1.has_solution or stage1.objective is None:
        return CorePointStrengtheningSolveResult(
            None, stage1.status, stage1.runtime, stage1.objective, "not_run", 0.0,
            original_current, None, original_core, None, None, False, False,
            "stage1_not_optimal",
        )
    delta = float(current_abs_tol) + float(current_rel_tol) * max(1.0, abs(stage1.objective))
    remaining_after_stage1 = max(0.0, float(remaining_global_time) - stage1.runtime)
    actual_stage2_limit = min(float(stage2_time_limit), remaining_after_stage1)
    if actual_stage2_limit <= 0.0:
        return CorePointStrengtheningSolveResult(
            None, stage1.status, stage1.runtime, stage1.objective, "not_run", 0.0,
            original_current, None, original_core, None, stage1.objective - delta,
            False, False, "stage2_time_unavailable",
        )
    stage2 = solve_fixed_pattern_dual_lp(
        instance,
        core_x_values,
        original_cut.z_values,
        time_limit=actual_stage2_limit,
        output_flag=output_flag,
        current_x_values=current_x_values,
        current_value_floor=stage1.objective - delta,
    )
    if stage2.status != "optimal" or not stage2.has_solution:
        return CorePointStrengtheningSolveResult(
            None, stage1.status, stage1.runtime, stage1.objective,
            stage2.status, stage2.runtime, original_current, None, original_core,
            None, stage1.objective - delta, stage2.dual_feasible, False,
            "stage2_not_optimal",
        )
    strengthened_current = stage2.cut_value(current_x_values)
    strengthened_core = stage2.cut_value(core_x_values)
    pattern = discretize_robust_pattern(instance, original_cut.z_values)
    if pattern is None:
        raise RuntimeError("Stage 2 used a pattern that could no longer be discretized")
    strengthened_cut = RobustDualSubproblemResult(
        # Preserve RobustDualSubproblemResult's convention: objective is the
        # incumbent cut value at the current first-stage point, not the
        # auxiliary core-point objective.
        objective=strengthened_current,
        z_values={key: float(value) for key, value in pattern.items()},
        lambda_values=stage2.lambda_values,
        mu_values=stage2.mu_values,
        nu_values=stage2.nu_values,
        demand_values=stage2.demand_values,
        constant=stage2.constant,
        x_coefficients=stage2.x_coefficients,
        runtime=stage1.runtime + stage2.runtime,
        status="optimal",
        objective_bound=None,
        mip_gap=None,
        has_incumbent=True,
        requested_mip_gap=None,
    )
    return CorePointStrengtheningSolveResult(
        strengthened_cut, stage1.status, stage1.runtime, stage1.objective,
        stage2.status, stage2.runtime, original_current, strengthened_current,
        original_core, strengthened_core, stage1.objective - delta,
        stage2.dual_feasible, False, None,
    )


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
