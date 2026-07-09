from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from .instance import InventoryInstance
from .policies import ExactGapPolicy, FixedGapPolicy, GapPolicy, GapPolicyState, RLInspiredGapPolicy
from .results import SolveResult
from .robust_dual_subproblem import RobustDualSubproblemResult, solve_robust_dual_subproblem
from .scenarios import DemandScenario, ScenarioEnumerationResult, enumerate_budget_scenarios_with_metadata
from .subproblem import SubproblemResult, solve_recourse_subproblem


@dataclass(frozen=True)
class BendersSettings:
    method: str
    gamma_target: int
    gamma_schedule: list[int]
    max_scenarios: int
    exact_scenarios: bool
    subproblem_mode: str
    cut_selection_enabled: bool
    delta_cut: float
    cut_violation_tol: float
    max_iterations: int
    tol: float
    initial_mip_gap: float
    final_mip_gap: float
    time_limit: float
    output_flag: bool


def _settings(config: dict[str, Any], method: str) -> BendersSettings:
    algorithm_cfg = config.get("algorithm", {})
    robust_cfg = config.get("robust", {})
    benders_cfg = config.get("benders", {})
    gamma_target = int(robust_cfg.get("gamma_target", 0))
    raw_schedule = robust_cfg.get("gamma_schedule") or list(range(gamma_target + 1))
    schedule = [min(gamma_target, max(0, int(v))) for v in raw_schedule]
    if not schedule or schedule[-1] != gamma_target:
        schedule.append(gamma_target)
    if method in {"standard_benders", "inexact_benders"}:
        schedule = [gamma_target]
    subproblem_mode = str(algorithm_cfg.get("subproblem_mode", "robust_dual_milp"))
    if subproblem_mode not in {"scenario_enumeration", "robust_dual_milp"}:
        raise ValueError(f"Unknown subproblem_mode: {subproblem_mode}")
    return BendersSettings(
        method=method,
        gamma_target=gamma_target,
        gamma_schedule=schedule,
        max_scenarios=int(robust_cfg.get("max_scenarios", 5000)),
        exact_scenarios=bool(robust_cfg.get("exact_scenarios", True)),
        subproblem_mode=subproblem_mode,
        cut_selection_enabled=bool(algorithm_cfg.get("cut_selection_enabled", True)),
        delta_cut=float(algorithm_cfg.get("delta_cut", 0.0)),
        cut_violation_tol=float(algorithm_cfg.get("cut_violation_tol", 1e-8)),
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


def _add_cut(
    model: gp.Model,
    x: gp.tupledict,
    theta: gp.Var,
    cut: SubproblemResult | RobustDualSubproblemResult,
    cut_index: int,
) -> None:
    cut_name = getattr(cut, "scenario_name", "robust_dual")
    model.addConstr(
        theta
        >= cut.constant + gp.quicksum(cut.x_coefficients[i, j] * x[i, j] for i, j in cut.x_coefficients),
        name=f"benders_cut[{cut_index}]_{cut_name}",
    )


def solve_benders(config: dict[str, Any], instance: InventoryInstance, method: str) -> SolveResult:
    if method not in {"standard_benders", "inexact_benders", "adaptive_gap_gamma_benders"}:
        raise ValueError(f"Unknown Benders method: {method}")

    settings = _settings(config, method)
    target_enum: ScenarioEnumerationResult | None = None
    target_scenarios: list[DemandScenario] = []
    scenario_cache: dict[int, ScenarioEnumerationResult] = {}
    if settings.subproblem_mode == "scenario_enumeration":
        target_enum = enumerate_budget_scenarios_with_metadata(
            instance,
            settings.gamma_target,
            settings.max_scenarios,
            exact_scenarios=settings.exact_scenarios,
        )
        target_scenarios = target_enum.scenarios
        scenario_cache = {
            gamma: enumerate_budget_scenarios_with_metadata(
                instance,
                gamma,
                settings.max_scenarios,
                exact_scenarios=settings.exact_scenarios,
            )
            for gamma in sorted(set(settings.gamma_schedule + [settings.gamma_target]))
        }

    def solve_robust_dual(
        gamma: int,
        x_current: dict[tuple[int, int], float],
        remaining_time: float,
    ) -> RobustDualSubproblemResult:
        return solve_robust_dual_subproblem(
            instance,
            x_current,
            gamma,
            time_limit=remaining_time,
            mip_gap=settings.final_mip_gap,
            output_flag=settings.output_flag,
        )

    start = time.perf_counter()
    model, y, x, theta = _build_master(instance, settings.output_flag)
    upper_bound = float("inf")
    lower_bound = -float("inf")
    best_first_stage = None
    best_robust_cost = None
    best_objective = None
    cuts = 0
    cuts_skipped = 0
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

        if settings.subproblem_mode == "scenario_enumeration":
            if target_enum is None:
                raise RuntimeError("Scenario enumeration metadata is not initialized.")
            active_enum = scenario_cache[active_gamma]
            active_scenarios = active_enum.scenarios
            active_cut, active_sub_time = _solve_worst_recourse(
                instance,
                active_scenarios,
                x_values,
                settings.output_flag,
            )
            target_cut, target_sub_time = _solve_worst_recourse(
                instance,
                target_scenarios,
                x_values,
                settings.output_flag,
            )
            active_scenario_name = active_cut.scenario_name
            target_scenario_name = target_cut.scenario_name
            active_scenario_mode = active_enum.scenario_mode
            target_scenario_mode = target_enum.scenario_mode
            active_subproblem_status = "optimal"
            target_subproblem_status = "optimal"
            active_subproblem_mip_gap = None
            target_subproblem_mip_gap = None
            target_subproblem_objective_bound = target_cut.objective
        else:
            active_cut = solve_robust_dual(active_gamma, x_values, remaining)
            active_sub_time = active_cut.runtime
            if active_gamma == settings.gamma_target:
                target_cut = active_cut
                target_sub_time = 0.0
            else:
                target_cut = solve_robust_dual(settings.gamma_target, x_values, remaining)
                target_sub_time = target_cut.runtime
            active_scenario_name = "robust_dual_milp"
            target_scenario_name = "robust_dual_milp"
            active_scenario_mode = "not_applicable"
            target_scenario_mode = "not_applicable"
            active_subproblem_status = active_cut.status
            target_subproblem_status = target_cut.status
            active_subproblem_mip_gap = active_cut.mip_gap
            target_subproblem_mip_gap = target_cut.mip_gap
            target_subproblem_objective_bound = target_cut.objective_bound
        subproblem_runtime += active_sub_time + target_sub_time

        ub_uses_subproblem_bound = False
        valid_ub = True
        conservative_target_cost = target_cut.objective
        if settings.subproblem_mode == "robust_dual_milp" and target_cut.status != "optimal":
            if target_cut.objective_bound is None:
                valid_ub = False
                conservative_target_cost = None
            else:
                conservative_target_cost = target_cut.objective_bound
                ub_uses_subproblem_bound = True

        candidate_upper = None if conservative_target_cost is None else first_stage + conservative_target_cost
        if candidate_upper is not None and candidate_upper < upper_bound:
            upper_bound = candidate_upper
            best_first_stage = first_stage
            best_robust_cost = conservative_target_cost
            best_objective = candidate_upper

        lower_bound = max(lower_bound, float(model.ObjBound))
        previous_gap = current_gap
        if upper_bound < float("inf"):
            gap = max(0.0, (upper_bound - lower_bound) / max(1.0, abs(upper_bound)))
        else:
            gap = 1.0
        current_gap = gap
        theta_current = float(theta.X)
        cut_rhs_current = active_cut.cut_value(x_values)
        cut_violation = cut_rhs_current - theta_current
        cut_added = False
        cut_skip_reason = None
        cut_add_reason = None
        if settings.subproblem_mode == "robust_dual_milp" and active_cut.status not in {
            "optimal",
            "time_limit",
            "suboptimal",
        }:
            add_cut = False
            cut_skip_reason = "no_incumbent"
        elif not settings.cut_selection_enabled:
            add_cut = True
        else:
            add_cut = cut_violation >= settings.delta_cut - settings.cut_violation_tol
            if not add_cut:
                cut_skip_reason = "low_violation"
                if (
                    active_gamma == settings.gamma_target
                    and gap > settings.tol
                    and cut_violation > settings.cut_violation_tol
                ):
                    add_cut = True
                    cut_skip_reason = None
                    cut_add_reason = "forced_target_progress"

        if add_cut:
            _add_cut(model, x, theta, active_cut, cuts)
            cuts += 1
            cut_added = True
        else:
            cuts_skipped += 1

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
                "theta": theta_current,
                "first_stage_cost": first_stage,
                "active_worst_cost": active_cut.objective,
                "target_worst_cost": target_cut.objective,
                "active_subproblem_value": active_cut.objective,
                "target_subproblem_value": target_cut.objective,
                "active_subproblem_status": active_subproblem_status,
                "target_subproblem_status": target_subproblem_status,
                "active_subproblem_mip_gap": active_subproblem_mip_gap,
                "target_subproblem_mip_gap": target_subproblem_mip_gap,
                "target_subproblem_objective": target_cut.objective,
                "target_subproblem_objective_bound": target_subproblem_objective_bound,
                "ub_uses_subproblem_bound": ub_uses_subproblem_bound,
                "valid_UB": valid_ub,
                "active_gamma": active_gamma,
                "gamma_target": settings.gamma_target,
                "active_scenario": active_scenario_name,
                "target_scenario": target_scenario_name,
                "active_scenario_mode": active_scenario_mode,
                "target_scenario_mode": target_scenario_mode,
                "cut_selection_enabled": settings.cut_selection_enabled,
                "delta_cut": settings.delta_cut,
                "cut_rhs_current": cut_rhs_current,
                "cut_violation": cut_violation,
                "cut_added": cut_added,
                "cut_skip_reason": cut_skip_reason,
                "cut_add_reason": cut_add_reason,
                "cuts_added_total": cuts,
                "cuts_skipped_total": cuts_skipped,
                "cuts": cuts,
            }
        )

        if active_gamma == settings.gamma_target and gap <= settings.tol:
            status = "optimal"
            break

    runtime = time.perf_counter() - start
    final_gap = None
    if upper_bound < float("inf") and lower_bound > -float("inf"):
        final_gap = max(0.0, (upper_bound - lower_bound) / max(1.0, abs(upper_bound)))

    last_log = log[-1] if log else {}
    if settings.subproblem_mode == "scenario_enumeration":
        if target_enum is None:
            raise RuntimeError("Scenario enumeration metadata is not initialized.")
        scenario_modes_by_gamma = ",".join(
            f"{gamma}:{enum.scenario_mode}" for gamma, enum in sorted(scenario_cache.items())
        )
        heuristic_scenarios = any(enum.scenario_mode == "candidate" for enum in scenario_cache.values())
        scenario_metadata: dict[str, Any] = {
            "scenario_mode_target": target_enum.scenario_mode,
            "exact_scenarios": settings.exact_scenarios,
            "num_target_scenarios_used": target_enum.num_scenarios_used,
            "num_target_scenarios_total_estimated": target_enum.num_scenarios_total_estimated,
            "max_scenarios": target_enum.max_scenarios,
            "scenario_modes_by_gamma": scenario_modes_by_gamma,
            "heuristic_scenarios": heuristic_scenarios,
            "num_target_scenarios": len(target_scenarios),
        }
    else:
        scenario_metadata = {
            "scenario_mode_target": "not_applicable",
            "exact_scenarios": "not_applicable",
            "num_target_scenarios_used": "not_applicable",
            "num_target_scenarios_total_estimated": "not_applicable",
            "max_scenarios": settings.max_scenarios,
            "scenario_modes_by_gamma": "not_applicable",
            "heuristic_scenarios": False,
            "num_target_scenarios": "not_applicable",
        }

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
            "subproblem_mode": settings.subproblem_mode,
            "gamma_schedule": ",".join(str(v) for v in settings.gamma_schedule),
            "cut_selection_enabled": settings.cut_selection_enabled,
            "delta_cut": settings.delta_cut,
            "cut_violation_tol": settings.cut_violation_tol,
            "cuts_added_total": cuts,
            "cuts_skipped_total": cuts_skipped,
            "last_cut_violation": last_log.get("cut_violation"),
            "last_cut_added": last_log.get("cut_added"),
            "last_cut_skip_reason": last_log.get("cut_skip_reason"),
            "active_subproblem_value": last_log.get("active_subproblem_value"),
            "target_subproblem_value": last_log.get("target_subproblem_value"),
            "active_subproblem_status": last_log.get("active_subproblem_status"),
            "target_subproblem_status": last_log.get("target_subproblem_status"),
            "active_subproblem_mip_gap": last_log.get("active_subproblem_mip_gap"),
            "target_subproblem_mip_gap": last_log.get("target_subproblem_mip_gap"),
            "target_subproblem_objective": last_log.get("target_subproblem_objective"),
            "target_subproblem_objective_bound": last_log.get("target_subproblem_objective_bound"),
            "ub_uses_subproblem_bound": last_log.get("ub_uses_subproblem_bound"),
            "valid_UB": last_log.get("valid_UB"),
            "active_gamma": last_log.get("active_gamma"),
            "gamma_target": settings.gamma_target,
            **scenario_metadata,
        },
        iteration_log=log,
    )
