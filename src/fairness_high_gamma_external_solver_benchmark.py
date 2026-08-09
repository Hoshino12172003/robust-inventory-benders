"""Direct deterministic-equivalent fairness benchmark.

`gurobipy` is imported only inside the production solve function.  Importing
this module is therefore safe for authorization gates and dry-runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

from .instance import InventoryInstance
from .scenarios import enumerate_budget_scenarios_with_metadata


@dataclass(frozen=True)
class DirectExtensiveFormResult:
    status: str
    scientific_model_status: str
    complete_model_built: bool
    resource_failure: bool
    resource_failure_detail: str | None
    objective_t: float | None
    robust_minimum_fill_rate: float | None
    lower_bound: float | None
    upper_bound: float | None
    incumbent: float | None
    objective_bound: float | None
    gap: float | None
    mip_gap: float | None
    y_values: list[float] | None
    x_values: list[list[float]] | None
    baseline_cost: float
    rho: float
    cost_budget: float
    scenario_count: int
    model_build_runtime: float
    optimize_runtime: float
    algorithm_runtime: float
    rows: int
    columns: int
    binaries: int
    continuous_variables: int
    nonzeros: int
    solver_status_code: int | None
    solver_status: str
    node_count: float
    simplex_iterations: float
    benders_strategy: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _empty_result(*, status: str, detail: str | None, baseline_cost: float, rho: float,
                  scenario_count: int, build: float, optimize: float, rows: int = 0,
                  columns: int = 0, binaries: int = 0, continuous_variables: int = 0,
                  nonzeros: int = 0, solver_code: int | None = None,
                  solver_status: str = "not_optimized") -> DirectExtensiveFormResult:
    return DirectExtensiveFormResult(
        status=status, scientific_model_status=status, complete_model_built=False,
        resource_failure=status == "resource_failure", resource_failure_detail=detail,
        objective_t=None, robust_minimum_fill_rate=None, lower_bound=None, upper_bound=None,
        incumbent=None, objective_bound=None,
        gap=None, mip_gap=None, y_values=None, x_values=None, baseline_cost=float(baseline_cost),
        rho=float(rho), cost_budget=(1.0 + float(rho)) * float(baseline_cost),
        scenario_count=scenario_count, model_build_runtime=build, optimize_runtime=optimize,
        algorithm_runtime=build + optimize, rows=rows, columns=columns, binaries=binaries,
        continuous_variables=continuous_variables, nonzeros=nonzeros, solver_status_code=solver_code,
        solver_status=solver_status, node_count=0.0, simplex_iterations=0.0,
        benders_strategy=0,
    )


def solve_gurobi_direct_extensive_form(
    instance: InventoryInstance,
    *,
    baseline_cost: float,
    rho: float,
    gamma: int,
    expected_scenario_count: int,
    solver_parameters: dict[str, Any],
    time_limit: float,
    output_flag: bool = False,
) -> DirectExtensiveFormResult:
    """Build and solve the complete deterministic equivalent under one clock."""
    import gurobipy as gp
    from gurobipy import GRB

    start = time.perf_counter()
    limit = float(time_limit)
    model = None
    try:
        enumeration = enumerate_budget_scenarios_with_metadata(
            instance, int(gamma), max_scenarios=int(expected_scenario_count), exact_scenarios=True,
        )
        if len(enumeration.scenarios) != int(expected_scenario_count):
            raise ValueError("direct extensive-form scenario count identity mismatch")
        model = gp.Model("fairness_high_gamma_direct_extensive_form")
        model.Params.OutputFlag = 1 if output_flag else 0
        model.Params.Threads = int(solver_parameters["Threads"])
        model.Params.Seed = int(solver_parameters["Seed"])
        model.Params.FeasibilityTol = float(solver_parameters["FeasibilityTol"])
        model.Params.MIPGap = 0.0
        try:
            model.setParam("BendersStrategy", 0)
        except gp.GurobiError as exc:
            if "Unknown parameter" not in str(exc):
                raise
        y = model.addVars(instance.I, vtype=GRB.BINARY, name="y")
        x = model.addVars(instance.I, instance.J, lb=0.0, name="x")
        t = model.addVar(lb=0.0, ub=1.0, name="T")
        for i in instance.I:
            model.addConstr(gp.quicksum(instance.volume[j] * x[i, j] for j in instance.J)
                            <= instance.capacity[i] * y[i], name=f"capacity[{i}]")
            for j in instance.J:
                model.addConstr(x[i, j] <= instance.inventory_ub[i][j] * y[i],
                                name=f"logic[{i},{j}]")
        first_stage = gp.quicksum(instance.fixed_cost[i] * y[i] for i in instance.I) + gp.quicksum(
            instance.inventory_cost[i][j] * x[i, j] for i in instance.I for j in instance.J)
        model.addConstr(first_stage <= instance.budget, name="first_stage_budget")
        cost_budget = (1.0 + float(rho)) * float(baseline_cost)
        complete = True
        for index, scenario in enumerate(enumeration.scenarios):
            if time.perf_counter() - start >= limit:
                complete = False
                break
            q = model.addVars(instance.I, instance.R, instance.J, lb=0.0, name=f"q_{index}")
            u = model.addVars(instance.R, instance.J, lb=0.0, name=f"u_{index}")
            e = model.addVars(instance.J, lb=0.0, name=f"e_{index}")
            for r in instance.R:
                for j in instance.J:
                    model.addConstr(gp.quicksum(q[i, r, j] for i in instance.I) + u[r, j]
                                    >= scenario.demand[r][j], name=f"demand[{index},{r},{j}]")
            for i in instance.I:
                for j in instance.J:
                    model.addConstr(gp.quicksum(q[i, r, j] for r in instance.R) <= x[i, j],
                                    name=f"supply[{index},{i},{j}]")
            for j in instance.J:
                model.addConstr(gp.quicksum(u[r, j] for r in instance.R) - e[j]
                                <= (1.0 - instance.service_level[j]) *
                                sum(scenario.demand[r][j] for r in instance.R),
                                name=f"service[{index},{j}]")
            recourse_cost = gp.quicksum(
                instance.transport_cost[i][r][j] * q[i, r, j]
                for i in instance.I for r in instance.R for j in instance.J
            ) + gp.quicksum(
                instance.shortage_penalty[r][j] * u[r, j]
                for r in instance.R for j in instance.J
            ) + gp.quicksum(instance.service_penalty[j] * e[j] for j in instance.J)
            model.addConstr(first_stage + recourse_cost <= cost_budget, name=f"cost_cap[{index}]")
            for r in instance.R:
                regional_demand = sum(scenario.demand[r][j] for j in instance.J)
                if regional_demand > 1e-9:
                    model.addConstr(gp.quicksum(u[r, j] for j in instance.J) <= t * regional_demand,
                                    name=f"regional_fairness[{index},{r}]")
        model.setObjective(t, GRB.MINIMIZE)
        model.update()
        build_runtime = time.perf_counter() - start
        sizes = {
            "rows": int(model.NumConstrs), "columns": int(model.NumVars),
            "binaries": int(model.NumBinVars),
            "continuous_variables": int(model.NumVars - model.NumBinVars - model.NumIntVars),
            "nonzeros": int(model.NumNZs),
        }
        if not complete or build_runtime >= limit:
            return _empty_result(status="model_build_time_limit", detail=None,
                                 baseline_cost=baseline_cost, rho=rho,
                                 scenario_count=expected_scenario_count, build=build_runtime,
                                 optimize=0.0, **sizes)
        remaining = limit - build_runtime
        model.Params.TimeLimit = max(1e-3, remaining)
        optimize_start = time.perf_counter()
        model.optimize()
        optimize_runtime = time.perf_counter() - optimize_start
        status_names = {
            GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
            GRB.INFEASIBLE: "infeasible", GRB.INF_OR_UNBD: "inf_or_unbd",
            GRB.UNBOUNDED: "unbounded", GRB.INTERRUPTED: "interrupted",
            GRB.NUMERIC: "numeric",
        }
        solver_status = status_names.get(model.Status, f"status_{model.Status}")
        has_solution = int(model.SolCount) > 0
        upper = float(model.ObjVal) if has_solution and math.isfinite(float(model.ObjVal)) else None
        lower = float(model.ObjBound) if math.isfinite(float(model.ObjBound)) else None
        gap = (max(0.0, upper - lower) / max(1.0, abs(upper))) if upper is not None and lower is not None else None
        certified = model.Status == GRB.OPTIMAL and upper is not None and lower is not None and gap <= 1e-4
        return DirectExtensiveFormResult(
            status="optimal" if certified else solver_status,
            scientific_model_status="complete_exact_model_optimal" if certified else f"complete_model_{solver_status}",
            complete_model_built=True, resource_failure=False, resource_failure_detail=None,
            objective_t=upper if certified else None,
            robust_minimum_fill_rate=1.0 - upper if certified and upper is not None else None,
            lower_bound=lower, upper_bound=upper, incumbent=upper, objective_bound=lower, gap=gap,
            mip_gap=float(model.MIPGap) if has_solution and math.isfinite(float(model.MIPGap)) else None,
            y_values=[float(y[i].X) for i in instance.I] if certified else None,
            x_values=[[float(x[i, j].X) for j in instance.J] for i in instance.I] if certified else None,
            baseline_cost=float(baseline_cost), rho=float(rho), cost_budget=cost_budget,
            scenario_count=expected_scenario_count, model_build_runtime=build_runtime,
            optimize_runtime=optimize_runtime, algorithm_runtime=build_runtime + optimize_runtime,
            **sizes, solver_status_code=int(model.Status), solver_status=solver_status,
            node_count=float(model.NodeCount), simplex_iterations=float(model.IterCount),
            benders_strategy=0,
        )
    except (MemoryError, gp.GurobiError) as exc:
        if isinstance(exc, gp.GurobiError) and getattr(exc, "errno", None) not in {
            GRB.Error.OUT_OF_MEMORY,
            GRB.Error.SIZE_LIMIT_EXCEEDED,
            GRB.Error.EXCEED_2B_NONZEROS,
            GRB.Error.FAILED_TO_CREATE_MODEL,
        }:
            raise
        elapsed = time.perf_counter() - start
        rows = int(model.NumConstrs) if model is not None else 0
        columns = int(model.NumVars) if model is not None else 0
        binaries = int(model.NumBinVars) if model is not None else 0
        continuous_variables = int(model.NumVars - model.NumBinVars - model.NumIntVars) if model is not None else 0
        nonzeros = int(model.NumNZs) if model is not None else 0
        return _empty_result(status="resource_failure", detail=f"{type(exc).__name__}:{exc}",
                             baseline_cost=baseline_cost, rho=rho,
                             scenario_count=expected_scenario_count, build=elapsed,
                             optimize=0.0, rows=rows, columns=columns, binaries=binaries,
                             continuous_variables=continuous_variables, nonzeros=nonzeros)
    finally:
        if model is not None:
            model.dispose()
