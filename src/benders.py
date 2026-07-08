from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from .instance import InventoryInstance
from .policies import ExactGapPolicy, FixedGapPolicy, GapPolicy, GapPolicyState, RLInspiredGapPolicy
from .results import SolveResult
from .scenarios import DemandScenario, enumerate_budget_scenarios
from .subproblem import SubproblemResult, solve_recourse_subproblem


@dataclass(frozen=True)
class BendersSettings:
    method: str
    gamma_target: int
    gamma_schedule: list[int]
    max_scenarios: int
    max_iterations: int
    tol: float
    initial_mip_gap: float
    final_mip_gap: float
    time_limit: float
    output_flag: bool


def _settings(config: dict[str, Any], method: str) -> BendersSettings:
    robust_cfg = config.get("robust", {})
    benders_cfg = config.get("benders", {})
    gamma_target = int(robust_cfg.get("gamma_target", 0))
    raw_schedule = robust_cfg.get("gamma_schedule") or list(range(gamma_target + 1))
    schedule = [min(gamma_target, max(0, int(v))) for v in raw_schedule]
    if not schedule or schedule[-1] != gamma_target:
        schedule.append(gamma_target)
    if method in {"standard_benders", "inexact_benders"}:
        schedule = [gamma_target]
    return BendersSettings(
        method=method,
        gamma_target=gamma_target,
        gamma_schedule=schedule,
        max_scenarios=int(robust_cfg.get("max_scenarios", 5000)),
        max_iterations=int(benders_cfg.get("max_iterations", 80)),
        tol=float(benders_cfg.get("tol", 1e-4)),
        initial_mip_gap=float(benders_cfg.get("initial_mip_gap", 0.08)),
        final_mip_gap=float(benders_cfg.get("final_mip_gap", 1e-4)),
        time_limit=float(benders_cfg.get("time_limit", 120)),
        output_flag=bool(benders_cfg.get("output_flag", False)),
    )


def _first_stage_expr(instance: InventoryInstance, y: gp.tupledict, x: gp.tupledict) -> gp.LinExpr:
    return gp.quicksum(instance.fixed_cost[i] * y[i] for i in instance.I) + gp.quicksum(
        instance.inventory_cost[i][j] * x[i, j] for i in instance.I for j in instance.J
    )


def _first_stage_value(
    instance: InventoryInstance,
    y_values: dict[int, float],
    x_values: dict[tuple[int, int], float],
) -> float:
    return sum(instance.fixed_cost[i] * y_values[i] for i in instance.I) + sum(
        instance.inventory_cost[i][j] * x_values[i, j] for i in instance.I for j in instance.J
    )


def _build_master(instance: InventoryInstance, output_flag: bool) -> tuple[gp.Model, gp.tupledict, gp.tupledict, gp.Var]:
    model = gp.Model("robust_inventory_benders_master")
    model.Params.OutputFlag = 1 if output_flag else 0
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

    first_stage = _first_stage_expr(instance, y, x)
    model.addConstr(first_stage <= instance.budget, name="budget")
    model.setObjective(first_stage + theta, GRB.MINIMIZE)
    return model, y, x, theta


def _make_gap_policy(settings: BendersSettings) -> GapPolicy:
    if settings.method == "standard_benders":
        return ExactGapPolicy(settings.final_mip_gap)
    if settings.method == "inexact_benders":
        return FixedGapPolicy(settings.initial_mip_gap)
    return RLInspiredGapPolicy(lower=settings.final_mip_gap, upper=settings.initial_mip_gap)


def _gamma_for_iteration(settings: BendersSettings, iteration: int) -> int:
    if settings.method != "adaptive_gap_gamma_benders":
        return settings.gamma_target
    return settings.gamma_schedule[min(iteration, len(settings.gamma_schedule) - 1)]


def _solve_worst_recourse(
    instance: InventoryInstance,
    scenarios: list[DemandScenario],
    x_values: dict[tuple[int, int], float],
    output_flag: bool,
) -> tuple[SubproblemResult, float]:
    start = time.perf_counter()
    worst: SubproblemResult | None = None
    for scenario in scenarios:
        result = solve_recourse_subproblem(instance, scenario, x_values, output_flag=output_flag)
        if worst is None or result.objective > worst.objective + 1e-8:
            worst = result
    if worst is None:
        raise RuntimeError("No recourse scenarios were available.")
    return worst, time.perf_counter() - start


def _add_cut(model: gp.Model, x: gp.tupledict, theta: gp.Var, cut: SubproblemResult, cut_index: int) -> None:
    model.addConstr(
        theta
        >= cut.constant + gp.quicksum(cut.x_coefficients[i, j] * x[i, j] for i, j in cut.x_coefficients),
        name=f"benders_cut[{cut_index}]_{cut.scenario_name}",
    )


def solve_benders(config: dict[str, Any], instance: InventoryInstance, method: str) -> SolveResult:
    if method not in {"standard_benders", "inexact_benders", "adaptive_gap_gamma_benders"}:
        raise ValueError(f"Unknown Benders method: {method}")

    settings = _settings(config, method)
    target_scenarios = enumerate_budget_scenarios(instance, settings.gamma_target, settings.max_scenarios)
    scenario_cache = {
        gamma: enumerate_budget_scenarios(instance, gamma, settings.max_scenarios)
        for gamma in sorted(set(settings.gamma_schedule + [settings.gamma_target]))
    }

    start = time.perf_counter()
    model, y, x, theta = _build_master(instance, settings.output_flag)
    upper_bound = float("inf")
    lower_bound = -float("inf")
    best_first_stage = None
    best_robust_cost = None
    best_objective = None
    cuts = 0
    master_runtime = 0.0
    subproblem_runtime = 0.0
    log: list[dict[str, Any]] = []
    status = "iteration_limit"
    current_gap = 1.0
    previous_gap = 1.0
    gap_policy = _make_gap_policy(settings)

    for iteration in range(settings.max_iterations):
        remaining = max(1e-3, settings.time_limit - (time.perf_counter() - start))
        if remaining <= 1e-3:
            status = "time_limit"
            break

        active_gamma = _gamma_for_iteration(settings, iteration)
        policy_state = GapPolicyState(
            iteration=iteration + 1,
            benders_gap=current_gap,
            previous_benders_gap=previous_gap,
            lower_bound=None if lower_bound == -float("inf") else lower_bound,
            upper_bound=None if upper_bound == float("inf") else upper_bound,
        )
        selected_mip_gap = gap_policy.select_gap(policy_state)
        model.Params.MIPGap = selected_mip_gap
        model.Params.TimeLimit = remaining

        master_start = time.perf_counter()
        model.optimize()
        master_runtime += time.perf_counter() - master_start

        if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} or model.SolCount == 0:
            status = f"gurobi_status_{model.Status}"
            break

        x_values = {(i, j): float(x[i, j].X) for i in instance.I for j in instance.J}
        y_values = {i: float(y[i].X) for i in instance.I}
        first_stage = _first_stage_value(instance, y_values, x_values)

        active_worst, active_sub_time = _solve_worst_recourse(
            instance, scenario_cache[active_gamma], x_values, settings.output_flag
        )
        target_worst, target_sub_time = _solve_worst_recourse(
            instance, target_scenarios, x_values, settings.output_flag
        )
        subproblem_runtime += active_sub_time + target_sub_time

        candidate_upper = first_stage + target_worst.objective
        if candidate_upper < upper_bound:
            upper_bound = candidate_upper
            best_first_stage = first_stage
            best_robust_cost = target_worst.objective
            best_objective = candidate_upper

        lower_bound = max(lower_bound, float(model.ObjBound))
        previous_gap = current_gap
        gap = max(0.0, (upper_bound - lower_bound) / max(1.0, abs(upper_bound)))
        current_gap = gap
        log.append(
            {
                "iteration": iteration + 1,
                "gamma": active_gamma,
                "mip_gap": selected_mip_gap,
                "realized_master_gap": float(model.MIPGap) if model.IsMIP else 0.0,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "gap": gap,
                "log_gap": policy_state.log_gap,
                "gap_improvement": policy_state.gap_improvement,
                "master_objective": float(model.ObjVal),
                "theta": float(theta.X),
                "first_stage_cost": first_stage,
                "active_worst_cost": active_worst.objective,
                "target_worst_cost": target_worst.objective,
                "active_scenario": active_worst.scenario_name,
                "target_scenario": target_worst.scenario_name,
                "cuts": cuts,
            }
        )

        _add_cut(model, x, theta, active_worst, cuts)
        cuts += 1

        if active_gamma == settings.gamma_target and gap <= settings.tol:
            status = "optimal"
            break

    runtime = time.perf_counter() - start
    final_gap = None
    if upper_bound < float("inf") and lower_bound > -float("inf"):
        final_gap = max(0.0, (upper_bound - lower_bound) / max(1.0, abs(upper_bound)))

    return SolveResult(
        method=method,
        status=status,
        objective=best_objective,
        lower_bound=lower_bound if lower_bound > -float("inf") else None,
        upper_bound=upper_bound if upper_bound < float("inf") else None,
        gap=final_gap,
        runtime=runtime,
        iterations=len(log),
        cuts=cuts,
        master_runtime=master_runtime,
        subproblem_runtime=subproblem_runtime,
        robust_cost=best_robust_cost,
        first_stage_cost=best_first_stage,
        gamma_target=settings.gamma_target,
        metadata={
            "num_target_scenarios": len(target_scenarios),
            "gamma_schedule": ",".join(str(v) for v in settings.gamma_schedule),
        },
        iteration_log=log,
    )
