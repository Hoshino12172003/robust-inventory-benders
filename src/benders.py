from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB

from .instance import InventoryInstance
from .policies import ExactGapPolicy, FixedGapPolicy, GapPolicy, GapPolicyState, RLInspiredGapPolicy
from .precision_policy import (
    PrecisionPolicyConfig,
    PrecisionPolicyDecision,
    WorkloadAwarePrecisionConfig,
    initialize_precision_state,
    initialize_workload_aware_state,
    precision_policy_config,
    select_joint_error_budget_precision,
    select_workload_aware_precision,
    update_workload_time_ema,
    valid_global_gap_for_precision,
    workload_aware_precision_config,
)
from .results import SolveResult
from .robust_dual_subproblem import RobustDualSubproblemResult, solve_robust_dual_subproblem
from .scenarios import DemandScenario, ScenarioEnumerationResult, enumerate_budget_scenarios_with_metadata
from .status import gurobi_status_name
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
    cut_selection_mode: str
    relative_cut_threshold: float
    cut_violation_tol: float
    final_exact_gap: float
    cut_stall_patience: int
    adaptive_secondary_cut_selection_enabled: bool
    secondary_cut_warmup_cuts: int
    secondary_cut_master_time_share_trigger: float
    secondary_cut_recent_master_time_trigger: float
    adaptive_secondary_generation_enabled: bool
    secondary_generation_lb_window: int
    secondary_generation_stall_threshold: float
    secondary_generation_cooldown_iterations: int
    secondary_generation_max_subproblem_time_share: float
    secondary_generation_min_remaining_time: float
    secondary_generation_min_solve_budget: float
    final_certification_enabled: bool
    final_certification_no_cut_patience: int
    precision_config: PrecisionPolicyConfig
    workload_precision_config: WorkloadAwarePrecisionConfig | None
    adaptive_subproblem_gap_enabled: bool
    subproblem_gap_schedule: list[dict[str, float]]
    max_cuts_per_iteration: int
    subproblem_time_budget_per_iteration: float | None
    max_iterations: int
    tol: float
    initial_mip_gap: float
    final_mip_gap: float
    time_limit: float
    output_flag: bool


@dataclass(frozen=True)
class AdditionalCutBatch:
    cuts: list[RobustDualSubproblemResult]
    runtime: float
    nonoptimal_count: int
    without_incumbent_count: int
    duplicate_patterns_rejected: int


@dataclass(frozen=True)
class SecondaryGenerationDecision:
    attempt: bool
    trigger_reason: str | None
    skipped_reason: str | None
    recent_relative_lb_improvement: float | None
    cooldown_remaining: int


@dataclass(frozen=True)
class FinalCertificationState:
    active: bool = False
    triggered: bool = False
    trigger_iteration: int | None = None
    count: int = 0
    iterations: int = 0
    consecutive_no_useful_primary_cuts: int = 0
    exit_reason: str | None = None


@dataclass(frozen=True)
class FinalCertificationTransition:
    state: FinalCertificationState
    triggered_this_iteration: bool
    reason: str | None


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
    cut_selection_mode = str(algorithm_cfg.get("cut_selection_mode", "absolute"))
    if cut_selection_mode not in {"absolute", "relative"}:
        raise ValueError(f"Unknown cut_selection_mode: {cut_selection_mode}")
    raw_subproblem_schedule = algorithm_cfg.get("subproblem_gap_schedule") or [
        {"global_gap_above": 0.0, "mip_gap": float(benders_cfg.get("final_mip_gap", 1e-4))}
    ]
    subproblem_schedule = [
        {
            "global_gap_above": float(item["global_gap_above"]),
            "mip_gap": max(0.0, float(item["mip_gap"])),
        }
        for item in raw_subproblem_schedule
    ]
    precision_config = precision_policy_config(
        algorithm_cfg,
        fixed_master_gap=float(benders_cfg.get("initial_mip_gap", 0.08)),
        fixed_subproblem_gap=float(benders_cfg.get("final_mip_gap", 1e-4)),
        legacy_subproblem_gaps=[item["mip_gap"] for item in subproblem_schedule],
    )
    workload_precision_config = (
        workload_aware_precision_config(algorithm_cfg)
        if precision_config.precision_policy == "workload_aware_joint"
        else None
    )
    return BendersSettings(
        method=method,
        gamma_target=gamma_target,
        gamma_schedule=schedule,
        max_scenarios=int(robust_cfg.get("max_scenarios", 5000)),
        exact_scenarios=bool(robust_cfg.get("exact_scenarios", True)),
        subproblem_mode=subproblem_mode,
        cut_selection_enabled=bool(algorithm_cfg.get("cut_selection_enabled", True)),
        delta_cut=float(algorithm_cfg.get("delta_cut", 0.0)),
        cut_selection_mode=cut_selection_mode,
        relative_cut_threshold=float(algorithm_cfg.get("relative_cut_threshold", 1e-4)),
        cut_violation_tol=float(algorithm_cfg.get("cut_violation_tol", 1e-8)),
        final_exact_gap=float(algorithm_cfg.get("final_exact_gap", 1e-2)),
        cut_stall_patience=max(1, int(algorithm_cfg.get("cut_stall_patience", 5))),
        adaptive_secondary_cut_selection_enabled=bool(
            algorithm_cfg.get("adaptive_secondary_cut_selection_enabled", False)
        ),
        secondary_cut_warmup_cuts=max(
            1, int(algorithm_cfg.get("secondary_cut_warmup_cuts", 50))
        ),
        secondary_cut_master_time_share_trigger=max(
            1e-6,
            float(algorithm_cfg.get("secondary_cut_master_time_share_trigger", 0.35)),
        ),
        secondary_cut_recent_master_time_trigger=max(
            1e-6,
            float(algorithm_cfg.get("secondary_cut_recent_master_time_trigger", 0.5)),
        ),
        adaptive_secondary_generation_enabled=bool(
            algorithm_cfg.get("adaptive_secondary_generation_enabled", False)
        ),
        secondary_generation_lb_window=max(
            1, int(algorithm_cfg.get("secondary_generation_lb_window", 5))
        ),
        secondary_generation_stall_threshold=max(
            0.0,
            float(algorithm_cfg.get("secondary_generation_stall_threshold", 1e-4)),
        ),
        secondary_generation_cooldown_iterations=max(
            0, int(algorithm_cfg.get("secondary_generation_cooldown_iterations", 5))
        ),
        secondary_generation_max_subproblem_time_share=min(
            1.0,
            max(
                0.0,
                float(
                    algorithm_cfg.get(
                        "secondary_generation_max_subproblem_time_share",
                        0.75,
                    )
                ),
            ),
        ),
        secondary_generation_min_remaining_time=max(
            0.0,
            float(algorithm_cfg.get("secondary_generation_min_remaining_time", 2.0)),
        ),
        secondary_generation_min_solve_budget=max(
            1e-3,
            float(algorithm_cfg.get("secondary_generation_min_solve_budget", 1.0)),
        ),
        final_certification_enabled=bool(
            algorithm_cfg.get("final_certification_enabled", False)
        ),
        final_certification_no_cut_patience=max(
            1, int(algorithm_cfg.get("final_certification_no_cut_patience", 2))
        ),
        precision_config=precision_config,
        workload_precision_config=workload_precision_config,
        adaptive_subproblem_gap_enabled=bool(algorithm_cfg.get("adaptive_subproblem_gap_enabled", False)),
        subproblem_gap_schedule=subproblem_schedule,
        max_cuts_per_iteration=max(1, int(algorithm_cfg.get("max_cuts_per_iteration", 1))),
        subproblem_time_budget_per_iteration=(
            float(algorithm_cfg["subproblem_time_budget_per_iteration"])
            if algorithm_cfg.get("subproblem_time_budget_per_iteration") is not None
            else None
        ),
        max_iterations=int(benders_cfg.get("max_iterations", 80)),
        tol=float(benders_cfg.get("tol", 1e-4)),
        initial_mip_gap=float(benders_cfg.get("initial_mip_gap", 0.08)),
        final_mip_gap=float(benders_cfg.get("final_mip_gap", 1e-4)),
        time_limit=float(benders_cfg.get("time_limit", 120)),
        output_flag=bool(benders_cfg.get("output_flag", False)),
    )


def calculate_cut_violations(rhs: float, theta: float) -> tuple[float, float]:
    """Return nonnegative absolute and scale-independent cut violations."""
    absolute = max(0.0, float(rhs) - float(theta))
    normalized = absolute / max(1.0, abs(float(theta)), abs(float(rhs)))
    return absolute, normalized


def certification_mip_gap(certification_active: bool, configured_gap: float) -> float:
    return 0.0 if certification_active else float(configured_gap)


def update_final_certification(
    state: FinalCertificationState,
    *,
    enabled: bool,
    iteration: int,
    active_gamma: int,
    gamma_target: int,
    valid_ub: bool,
    global_gap: float,
    tol: float,
    useful_primary_cut_added: bool,
    no_cut_patience: int,
) -> FinalCertificationTransition:
    """Advance persistent certification after one completed Benders iteration."""
    if not enabled:
        return FinalCertificationTransition(
            FinalCertificationState(),
            False,
            "disabled",
        )

    if state.active:
        certification_iterations = state.iterations + 1
        if useful_primary_cut_added:
            return FinalCertificationTransition(
                FinalCertificationState(
                    active=False,
                    triggered=state.triggered,
                    trigger_iteration=state.trigger_iteration,
                    count=state.count,
                    iterations=certification_iterations,
                    consecutive_no_useful_primary_cuts=0,
                    exit_reason="useful_primary_cut_added",
                ),
                False,
                "useful_primary_cut_added",
            )
        if active_gamma == gamma_target and valid_ub and global_gap <= tol:
            return FinalCertificationTransition(
                FinalCertificationState(
                    active=False,
                    triggered=state.triggered,
                    trigger_iteration=state.trigger_iteration,
                    count=state.count,
                    iterations=certification_iterations,
                    consecutive_no_useful_primary_cuts=(
                        state.consecutive_no_useful_primary_cuts
                    ),
                    exit_reason="gap_tolerance_met",
                ),
                False,
                "gap_tolerance_met",
            )
        consecutive = state.consecutive_no_useful_primary_cuts
        if active_gamma == gamma_target and valid_ub and global_gap > tol:
            consecutive += 1
        return FinalCertificationTransition(
            FinalCertificationState(
                active=True,
                triggered=state.triggered,
                trigger_iteration=state.trigger_iteration,
                count=state.count,
                iterations=certification_iterations,
                consecutive_no_useful_primary_cuts=consecutive,
                exit_reason=None,
            ),
            False,
            "certification_active",
        )

    qualifying_iteration = (
        active_gamma == gamma_target
        and valid_ub
        and global_gap > tol
        and not useful_primary_cut_added
    )
    consecutive = (
        state.consecutive_no_useful_primary_cuts + 1
        if qualifying_iteration
        else 0
    )
    trigger = consecutive >= max(1, int(no_cut_patience))
    if trigger:
        return FinalCertificationTransition(
            FinalCertificationState(
                active=True,
                triggered=True,
                trigger_iteration=(
                    state.trigger_iteration
                    if state.trigger_iteration is not None
                    else iteration
                ),
                count=state.count + 1,
                iterations=state.iterations,
                consecutive_no_useful_primary_cuts=consecutive,
                exit_reason=None,
            ),
            True,
            "target_gamma_no_useful_primary_cut_patience",
        )
    return FinalCertificationTransition(
        FinalCertificationState(
            active=False,
            triggered=state.triggered,
            trigger_iteration=state.trigger_iteration,
            count=state.count,
            iterations=state.iterations,
            consecutive_no_useful_primary_cuts=consecutive,
            exit_reason=state.exit_reason,
        ),
        False,
        "waiting_for_no_useful_primary_cut_patience",
    )


def recent_relative_lb_improvement(
    lower_bound_history: list[float],
    rolling_window: int,
) -> float | None:
    required_points = max(1, int(rolling_window)) + 1
    if len(lower_bound_history) < required_points:
        return None
    recent = lower_bound_history[-required_points:]
    start = float(recent[0])
    end = float(recent[-1])
    if not math.isfinite(start) or not math.isfinite(end):
        return None
    return max(0.0, end - start) / max(1.0, abs(start), abs(end))


def secondary_generation_decision(
    *,
    enabled: bool,
    max_cuts_per_iteration: int,
    primary_has_incumbent: bool,
    lower_bound_history: list[float],
    rolling_window: int,
    stall_threshold: float,
    current_iteration: int,
    last_secondary_solve_iteration: int | None,
    cooldown_iterations: int,
    cumulative_subproblem_time_share: float,
    max_subproblem_time_share: float,
    remaining_time: float,
    available_budget: float,
    min_remaining_time: float,
    min_solve_budget: float,
    certification_active: bool = False,
) -> SecondaryGenerationDecision:
    recent_improvement = recent_relative_lb_improvement(
        lower_bound_history,
        rolling_window,
    )
    if certification_active:
        return SecondaryGenerationDecision(
            False, None, "final_certification", recent_improvement, 0
        )
    if max_cuts_per_iteration <= 1:
        return SecondaryGenerationDecision(
            False, None, "single_cut_mode", recent_improvement, 0
        )
    if not primary_has_incumbent:
        return SecondaryGenerationDecision(
            False, None, "no_primary_incumbent", recent_improvement, 0
        )

    if not enabled:
        if remaining_time <= 1e-3 or available_budget <= 1e-3:
            return SecondaryGenerationDecision(
                False, None, "insufficient_secondary_budget", recent_improvement, 0
            )
        return SecondaryGenerationDecision(
            True, "all_secondary_cuts", None, recent_improvement, 0
        )

    if remaining_time < min_remaining_time:
        return SecondaryGenerationDecision(
            False, None, "insufficient_remaining_time", recent_improvement, 0
        )
    if available_budget < min_solve_budget:
        return SecondaryGenerationDecision(
            False, None, "insufficient_secondary_budget", recent_improvement, 0
        )

    cooldown_remaining = 0
    if last_secondary_solve_iteration is not None and cooldown_iterations > 0:
        iterations_since_solve = current_iteration - last_secondary_solve_iteration
        if iterations_since_solve <= cooldown_iterations:
            cooldown_remaining = cooldown_iterations - iterations_since_solve + 1
            return SecondaryGenerationDecision(
                False,
                None,
                "cooldown",
                recent_improvement,
                cooldown_remaining,
            )

    if cumulative_subproblem_time_share >= max_subproblem_time_share:
        return SecondaryGenerationDecision(
            False, None, "subproblem_time_share", recent_improvement, cooldown_remaining
        )
    if recent_improvement is None:
        return SecondaryGenerationDecision(
            False, None, "insufficient_lb_history", None, cooldown_remaining
        )
    if recent_improvement > stall_threshold:
        return SecondaryGenerationDecision(
            False, None, "lb_progress", recent_improvement, cooldown_remaining
        )
    return SecondaryGenerationDecision(
        True, "lb_stall", None, recent_improvement, cooldown_remaining
    )


def select_subproblem_mip_gap(
    global_gap: float | None,
    has_finite_upper_bound: bool,
    schedule: list[dict[str, float]],
    final_exact_gap: float,
) -> float:
    if not schedule:
        raise ValueError("subproblem_gap_schedule must not be empty")
    coarsest = max(item["mip_gap"] for item in schedule)
    tightest = min(item["mip_gap"] for item in schedule)
    if not has_finite_upper_bound or global_gap is None:
        return coarsest
    if global_gap <= final_exact_gap:
        return tightest
    for item in sorted(schedule, key=lambda row: row["global_gap_above"], reverse=True):
        if global_gap > item["global_gap_above"]:
            return item["mip_gap"]
    return tightest


def relative_cut_decision(
    absolute_violation: float,
    normalized_violation: float,
    threshold: float,
    tolerance: float,
    active_gamma: int,
    gamma_target: int,
    global_gap: float,
    final_exact_gap: float,
) -> tuple[bool, str | None, str | None]:
    if absolute_violation <= tolerance:
        return False, "not_violated", None
    if normalized_violation >= threshold:
        return True, None, None
    if active_gamma == gamma_target and global_gap <= final_exact_gap:
        return True, None, "final_exact_phase"
    return False, "low_relative_violation", None


def primary_cut_decision(
    absolute_violation: float,
    tolerance: float,
    duplicate: bool = False,
) -> tuple[bool, str | None]:
    if duplicate:
        return False, "duplicate_cut"
    if absolute_violation <= tolerance:
        return False, "not_violated"
    return True, None


def marginal_normalized_violation(
    secondary_rhs: float,
    primary_rhs: float,
    theta_current: float,
) -> float:
    baseline = max(float(theta_current), float(primary_rhs))
    marginal_absolute = max(0.0, float(secondary_rhs) - baseline)
    return marginal_absolute / max(1.0, abs(float(secondary_rhs)), abs(baseline))


def adaptive_secondary_cut_threshold(
    base_threshold: float,
    cuts_in_master: int,
    warmup_cuts: int,
    master_time_share: float,
    master_time_share_trigger: float,
    recent_master_runtime: float,
    recent_master_time_trigger: float,
    global_gap: float,
    final_exact_gap: float,
    final_exact_phase: bool,
) -> float:
    if final_exact_phase:
        return 0.0

    master_is_small = (
        cuts_in_master < warmup_cuts
        and master_time_share < master_time_share_trigger
        and recent_master_runtime < recent_master_time_trigger
    )
    if master_is_small:
        return 0.0

    pressure = max(
        1.0,
        cuts_in_master / max(1, warmup_cuts),
        master_time_share / max(1e-12, master_time_share_trigger),
        recent_master_runtime / max(1e-12, recent_master_time_trigger),
    )
    gap_scale = max(0.25, min(1.0, global_gap / max(1e-12, 5.0 * final_exact_gap)))
    return min(1.0, max(0.0, base_threshold) * pressure * gap_scale)


def _pattern_key(cut: RobustDualSubproblemResult) -> tuple[int, ...]:
    return tuple(int(round(cut.z_values[key])) for key in sorted(cut.z_values))


def _cut_key(cut: SubproblemResult | RobustDualSubproblemResult, digits: int = 8) -> tuple[float, ...]:
    coefficients = tuple(round(cut.x_coefficients[key], digits) for key in sorted(cut.x_coefficients))
    return (round(cut.constant, digits), *coefficients)


def target_upper_cost(
    subproblem_mode: str,
    target_cut: SubproblemResult | RobustDualSubproblemResult,
) -> tuple[float | None, bool, bool]:
    """Return target robust cost, UB validity, and whether a MILP bound was used."""
    if subproblem_mode == "scenario_enumeration":
        return float(target_cut.objective), True, False
    objective_bound = target_cut.objective_bound
    if objective_bound is None or not math.isfinite(float(objective_bound)):
        return None, False, False
    return float(objective_bound), True, True


def calculate_global_gap(upper_bound: float, lower_bound: float) -> float:
    return max(0.0, (upper_bound - lower_bound) / max(1.0, abs(upper_bound)))


def should_terminate_benders(
    active_gamma: int,
    gamma_target: int,
    valid_ub: bool,
    global_gap: float,
    tol: float,
) -> bool:
    return active_gamma == gamma_target and valid_ub and global_gap <= tol


def generate_additional_robust_cuts(
    primary_cut: RobustDualSubproblemResult,
    max_cuts_per_iteration: int,
    time_budget: float,
    solve_extra: Callable[
        [list[dict[tuple[int, int], int]], float],
        RobustDualSubproblemResult,
    ],
) -> AdditionalCutBatch:
    if max_cuts_per_iteration <= 1 or not primary_cut.has_incumbent or time_budget <= 1e-3:
        return AdditionalCutBatch([], 0.0, 0, 0, 0)

    excluded_patterns = [{key: int(round(value)) for key, value in primary_cut.z_values.items()}]
    seen_patterns = {_pattern_key(primary_cut)}
    cuts: list[RobustDualSubproblemResult] = []
    runtime = 0.0
    nonoptimal_count = 0
    without_incumbent_count = 0
    duplicate_patterns = 0

    for _ in range(1, max_cuts_per_iteration):
        remaining = time_budget - runtime
        if remaining <= 1e-3:
            break
        extra_cut = solve_extra(excluded_patterns, remaining)
        runtime += extra_cut.runtime
        if extra_cut.status != "optimal":
            nonoptimal_count += 1
        if not extra_cut.has_incumbent:
            without_incumbent_count += 1
            break
        pattern = _pattern_key(extra_cut)
        if pattern in seen_patterns:
            duplicate_patterns += 1
            break
        seen_patterns.add(pattern)
        excluded_patterns.append({key: int(round(value)) for key, value in extra_cut.z_values.items()})
        cuts.append(extra_cut)

    return AdditionalCutBatch(
        cuts=cuts,
        runtime=runtime,
        nonoptimal_count=nonoptimal_count,
        without_incumbent_count=without_incumbent_count,
        duplicate_patterns_rejected=duplicate_patterns,
    )


def generate_gated_additional_robust_cuts(
    decision: SecondaryGenerationDecision,
    primary_cut: RobustDualSubproblemResult,
    max_cuts_per_iteration: int,
    time_budget: float,
    solve_extra: Callable[
        [list[dict[tuple[int, int], int]], float],
        RobustDualSubproblemResult,
    ],
) -> AdditionalCutBatch:
    if not decision.attempt:
        return AdditionalCutBatch([], 0.0, 0, 0, 0)
    return generate_additional_robust_cuts(
        primary_cut,
        max_cuts_per_iteration,
        time_budget,
        solve_extra,
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
        requested_mip_gap: float,
        excluded_patterns: list[dict[tuple[int, int], int]] | None = None,
    ) -> RobustDualSubproblemResult:
        return solve_robust_dual_subproblem(
            instance,
            x_current,
            gamma,
            time_limit=remaining_time,
            mip_gap=requested_mip_gap,
            output_flag=settings.output_flag,
            excluded_patterns=excluded_patterns,
        )

    start = time.perf_counter()
    model, y, x, theta = _build_master(instance, settings.output_flag)
    upper_bound = float("inf")
    lower_bound = -float("inf")
    best_first_stage = None
    best_robust_cost = None
    best_objective = None
    best_y_values: list[float] | None = None
    best_x_values: list[list[float]] | None = None
    cuts = 0
    cuts_skipped = 0
    secondary_cuts_added = 0
    secondary_cuts_skipped = 0
    secondary_solves_attempted = 0
    secondary_solves_avoided = 0
    duplicate_cuts_rejected = 0
    duplicate_patterns_rejected = 0
    additional_subproblem_time = 0.0
    subproblem_nonoptimal = 0
    subproblem_without_incumbent = 0
    requested_subproblem_gaps: list[float] = []
    cuts_generated_counts: list[int] = []
    known_cut_keys: set[tuple[float, ...]] = set()
    iterations_without_useful_cut = 0
    iterations_without_secondary_cut = 0
    last_secondary_solve_iteration: int | None = None
    lower_bound_history: list[float] = []
    master_runtime = 0.0
    subproblem_runtime = 0.0
    log: list[dict[str, Any]] = []
    status = "iteration_limit"
    current_gap = 1.0
    previous_gap = 1.0
    gap_policy = _make_gap_policy(settings)
    certification_state = FinalCertificationState()
    precision_state = initialize_precision_state(settings.precision_config)
    workload_state = initialize_workload_aware_state(settings.precision_config)
    workload_master_weights: list[float] = []
    workload_subproblem_weights: list[float] = []
    workload_fallback_count = 0

    for iteration in range(settings.max_iterations):
        remaining = max(1e-3, settings.time_limit - (time.perf_counter() - start))
        if remaining <= 1e-3:
            status = "time_limit"
            break

        active_gamma = _gamma_for_iteration(settings, iteration)
        certification_active = certification_state.active
        policy_state = GapPolicyState(
            iteration=iteration + 1,
            benders_gap=current_gap,
            previous_benders_gap=previous_gap,
            lower_bound=None if lower_bound == -float("inf") else lower_bound,
            upper_bound=None if upper_bound == float("inf") else upper_bound,
        )
        legacy_master_gap = gap_policy.select_gap(policy_state)
        legacy_subproblem_gap = (
            select_subproblem_mip_gap(
                current_gap,
                upper_bound < float("inf"),
                settings.subproblem_gap_schedule,
                settings.final_exact_gap,
            )
            if settings.adaptive_subproblem_gap_enabled
            else settings.precision_config.fixed_subproblem_gap
        )
        workload_decision = None
        if settings.precision_config.precision_policy == "workload_aware_joint":
            if settings.workload_precision_config is None:
                raise RuntimeError("Workload-aware precision config is unavailable")
            workload_decision = select_workload_aware_precision(
                settings.precision_config,
                settings.workload_precision_config,
                workload_state,
                upper_bound=(None if upper_bound == float("inf") else upper_bound),
                lower_bound=(None if lower_bound == -float("inf") else lower_bound),
                update_state=not certification_active,
            )
            workload_state = workload_decision.next_state
            precision_decision = workload_decision.precision_decision
        elif settings.precision_config.precision_policy == "joint_error_budget":
            precision_decision = select_joint_error_budget_precision(
                settings.precision_config,
                precision_state,
                upper_bound=(None if upper_bound == float("inf") else upper_bound),
                lower_bound=(None if lower_bound == -float("inf") else lower_bound),
                update_state=not certification_active,
            )
            precision_state = precision_decision.next_state
        else:
            precision_gap, precision_fallback = valid_global_gap_for_precision(
                None if upper_bound == float("inf") else upper_bound,
                None if lower_bound == -float("inf") else lower_bound,
            )
            precision_decision = PrecisionPolicyDecision(
                valid_global_gap_for_precision=precision_gap,
                fallback_used=precision_fallback,
                master_candidate_gap=legacy_master_gap,
                master_previous_gap=legacy_master_gap,
                master_selected_gap=legacy_master_gap,
                subproblem_candidate_gap=legacy_subproblem_gap,
                subproblem_previous_gap=legacy_subproblem_gap,
                subproblem_selected_gap=legacy_subproblem_gap,
                next_state=precision_state,
            )
        selected_mip_gap = certification_mip_gap(
            certification_active,
            precision_decision.master_selected_gap,
        )
        requested_subproblem_gap = certification_mip_gap(
            certification_active,
            precision_decision.subproblem_selected_gap,
        )
        requested_subproblem_gaps.append(requested_subproblem_gap)
        model.Params.MIPGap = selected_mip_gap
        model.Params.TimeLimit = remaining

        master_start = time.perf_counter()
        model.optimize()
        master_elapsed = time.perf_counter() - master_start
        master_runtime += master_elapsed

        master_status = gurobi_status_name(model.Status)
        if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} or model.SolCount == 0:
            status = master_status
            break
        if master_status == "time_limit":
            status = "time_limit"

        x_values = {(i, j): float(x[i, j].X) for i in instance.I for j in instance.J}
        y_values = {i: float(y[i].X) for i in instance.I}
        first_stage = _first_stage_value(instance, y_values, x_values)
        additional_batch = AdditionalCutBatch([], 0.0, 0, 0, 0)
        secondary_generation = SecondaryGenerationDecision(
            False,
            None,
            "scenario_enumeration",
            recent_relative_lb_improvement(
                lower_bound_history,
                settings.secondary_generation_lb_window,
            ),
            0,
        )
        generation_subproblem_time_share = 0.0

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
            active_candidates: list[SubproblemResult | RobustDualSubproblemResult] = [active_cut]
        else:
            active_cut = solve_robust_dual(active_gamma, x_values, remaining, requested_subproblem_gap)
            active_sub_time = active_cut.runtime
            if active_gamma == settings.gamma_target:
                target_cut = active_cut
                target_sub_time = 0.0
            else:
                target_remaining = max(1e-3, settings.time_limit - (time.perf_counter() - start))
                target_cut = solve_robust_dual(
                    settings.gamma_target,
                    x_values,
                    target_remaining,
                    requested_subproblem_gap,
                )
                target_sub_time = target_cut.runtime
            active_candidates = [active_cut] if active_cut.has_incumbent else []
            global_remaining = max(0.0, settings.time_limit - (time.perf_counter() - start))
            total_subproblem_budget = min(
                global_remaining,
                settings.subproblem_time_budget_per_iteration or global_remaining,
            )
            additional_budget = max(0.0, total_subproblem_budget - active_sub_time - target_sub_time)
            elapsed_before_secondary = max(1e-12, time.perf_counter() - start)
            projected_subproblem_time = subproblem_runtime + active_sub_time + target_sub_time
            generation_subproblem_time_share = (
                projected_subproblem_time / elapsed_before_secondary
            )
            secondary_generation = secondary_generation_decision(
                enabled=settings.adaptive_secondary_generation_enabled,
                max_cuts_per_iteration=settings.max_cuts_per_iteration,
                primary_has_incumbent=active_cut.has_incumbent,
                lower_bound_history=lower_bound_history,
                rolling_window=settings.secondary_generation_lb_window,
                stall_threshold=settings.secondary_generation_stall_threshold,
                current_iteration=iteration + 1,
                last_secondary_solve_iteration=last_secondary_solve_iteration,
                cooldown_iterations=settings.secondary_generation_cooldown_iterations,
                cumulative_subproblem_time_share=generation_subproblem_time_share,
                max_subproblem_time_share=(
                    settings.secondary_generation_max_subproblem_time_share
                ),
                remaining_time=global_remaining,
                available_budget=additional_budget,
                min_remaining_time=settings.secondary_generation_min_remaining_time,
                min_solve_budget=settings.secondary_generation_min_solve_budget,
                certification_active=certification_active,
            )
            if secondary_generation.attempt:
                secondary_solves_attempted += 1
                last_secondary_solve_iteration = iteration + 1
            elif settings.max_cuts_per_iteration > 1:
                secondary_solves_avoided += 1
            additional_batch = generate_gated_additional_robust_cuts(
                secondary_generation,
                active_cut,
                settings.max_cuts_per_iteration,
                additional_budget,
                lambda excluded, extra_remaining: solve_robust_dual(
                    active_gamma,
                    x_values,
                    extra_remaining,
                    requested_subproblem_gap,
                    excluded_patterns=excluded,
                ),
            )
            iteration_additional_subproblem_time = additional_batch.runtime
            additional_subproblem_time += iteration_additional_subproblem_time
            subproblem_nonoptimal += additional_batch.nonoptimal_count
            subproblem_without_incumbent += additional_batch.without_incumbent_count
            duplicate_patterns_rejected += additional_batch.duplicate_patterns_rejected
            active_candidates.extend(additional_batch.cuts)
            active_scenario_name = "robust_dual_milp"
            target_scenario_name = "robust_dual_milp"
            active_scenario_mode = "not_applicable"
            target_scenario_mode = "not_applicable"
            active_subproblem_status = active_cut.status
            target_subproblem_status = target_cut.status
            active_subproblem_mip_gap = active_cut.mip_gap
            target_subproblem_mip_gap = target_cut.mip_gap
            target_subproblem_objective_bound = target_cut.objective_bound
            for evaluated_cut in {id(active_cut): active_cut, id(target_cut): target_cut}.values():
                if evaluated_cut.status != "optimal":
                    subproblem_nonoptimal += 1
                if not evaluated_cut.has_incumbent:
                    subproblem_without_incumbent += 1
        if settings.subproblem_mode == "scenario_enumeration":
            iteration_additional_subproblem_time = 0.0
        iteration_subproblem_time = active_sub_time + target_sub_time + iteration_additional_subproblem_time
        subproblem_runtime += iteration_subproblem_time

        if workload_decision is not None and not certification_active:
            if settings.workload_precision_config is None:
                raise RuntimeError("Workload-aware precision config is unavailable")
            workload_master_weights.append(
                workload_decision.master_weight_selected
            )
            workload_subproblem_weights.append(
                workload_decision.subproblem_weight_selected
            )
            if workload_decision.fallback_used:
                workload_fallback_count += 1
            workload_state = update_workload_time_ema(
                settings.workload_precision_config,
                workload_state,
                master_time=master_elapsed,
                subproblem_time=iteration_subproblem_time,
            )

        conservative_target_cost, valid_ub, ub_uses_subproblem_bound = target_upper_cost(
            settings.subproblem_mode,
            target_cut,
        )

        previous_lower_bound = lower_bound
        previous_upper_bound = upper_bound
        candidate_upper = None if conservative_target_cost is None else first_stage + conservative_target_cost
        if candidate_upper is not None and candidate_upper < upper_bound:
            upper_bound = candidate_upper
            best_first_stage = first_stage
            best_robust_cost = conservative_target_cost
            best_objective = candidate_upper
            best_y_values = [float(y_values[i]) for i in instance.I]
            best_x_values = [
                [float(x_values[i, j]) for j in instance.J]
                for i in instance.I
            ]

        lower_bound = max(lower_bound, float(model.ObjBound))
        lower_bound_history.append(lower_bound)
        previous_gap = current_gap
        if upper_bound < float("inf"):
            gap = calculate_global_gap(upper_bound, lower_bound)
        else:
            gap = 1.0
        current_gap = gap
        theta_current = float(theta.X)
        primary_rhs = theta_current
        if active_candidates:
            primary_rhs = active_candidates[0].cut_value(x_values)
        elapsed_before_cut_selection = max(1e-12, time.perf_counter() - start)
        master_time_share = master_runtime / elapsed_before_cut_selection
        final_exact_phase = (
            active_gamma == settings.gamma_target and gap <= settings.final_exact_gap
        )
        adaptive_secondary_threshold = adaptive_secondary_cut_threshold(
            settings.relative_cut_threshold,
            cuts,
            settings.secondary_cut_warmup_cuts,
            master_time_share,
            settings.secondary_cut_master_time_share_trigger,
            master_elapsed,
            settings.secondary_cut_recent_master_time_trigger,
            gap,
            settings.final_exact_gap,
            final_exact_phase,
        )
        secondary_selection_threshold = 0.0 if final_exact_phase else (
            adaptive_secondary_threshold
            if settings.adaptive_secondary_cut_selection_enabled
            else settings.relative_cut_threshold
        )
        cut_decisions: list[dict[str, Any]] = []
        for candidate_index, candidate_cut in enumerate(active_candidates):
            cut_role = "primary" if candidate_index == 0 else "secondary"
            cut_rhs = candidate_cut.cut_value(x_values)
            raw_violation = cut_rhs - theta_current
            absolute_violation, normalized_violation = calculate_cut_violations(cut_rhs, theta_current)
            marginal_violation = (
                0.0
                if cut_role == "primary"
                else marginal_normalized_violation(
                    cut_rhs,
                    primary_rhs,
                    theta_current,
                )
            )
            duplicate = settings.max_cuts_per_iteration > 1 and _cut_key(candidate_cut) in known_cut_keys
            add_cut = False
            skip_reason = None
            add_reason = None
            active_threshold = 0.0
            if cut_role == "primary":
                add_cut, skip_reason = primary_cut_decision(
                    absolute_violation,
                    settings.cut_violation_tol,
                    duplicate,
                )
            else:
                if duplicate:
                    skip_reason = "duplicate_cut"
                elif absolute_violation <= settings.cut_violation_tol:
                    skip_reason = "not_violated"
                elif final_exact_phase:
                    add_cut = True
                    add_reason = "final_exact_phase"
                elif not settings.cut_selection_enabled:
                    add_cut = True
                elif settings.cut_selection_mode == "absolute":
                    active_threshold = settings.delta_cut
                    add_cut = raw_violation >= settings.delta_cut - settings.cut_violation_tol
                    if not add_cut:
                        skip_reason = "low_violation"
                else:
                    active_threshold = secondary_selection_threshold
                    add_cut, skip_reason, add_reason = relative_cut_decision(
                        absolute_violation,
                        marginal_violation,
                        active_threshold,
                        settings.cut_violation_tol,
                        active_gamma,
                        settings.gamma_target,
                        gap,
                        settings.final_exact_gap,
                    )
            if duplicate:
                duplicate_cuts_rejected += 1
            cut_decisions.append(
                {
                    "cut": candidate_cut,
                    "cut_role": cut_role,
                    "rhs": cut_rhs,
                    "raw_violation": raw_violation,
                    "absolute_violation": absolute_violation,
                    "normalized_violation": normalized_violation,
                    "marginal_normalized_violation": marginal_violation,
                    "active_threshold": active_threshold,
                    "add": add_cut,
                    "skip_reason": skip_reason,
                    "add_reason": add_reason,
                }
            )

        if (
            settings.cut_selection_enabled
            and settings.cut_selection_mode == "relative"
            and not any(
                decision["add"] for decision in cut_decisions if decision["cut_role"] == "secondary"
            )
            and iterations_without_secondary_cut + 1 >= settings.cut_stall_patience
        ):
            eligible = [
                decision
                for decision in cut_decisions
                if decision["cut_role"] == "secondary"
                and decision["absolute_violation"] > settings.cut_violation_tol
                and decision["skip_reason"] != "duplicate_cut"
            ]
            if eligible:
                forced = max(
                    eligible,
                    key=lambda decision: decision["marginal_normalized_violation"],
                )
                forced["add"] = True
                forced["skip_reason"] = None
                forced["add_reason"] = "stall_patience"

        cuts_added_this_iteration = 0
        cuts_skipped_this_iteration = 0
        for decision in cut_decisions:
            if decision["add"]:
                candidate_cut = decision["cut"]
                _add_cut(model, x, theta, candidate_cut, cuts)
                cuts += 1
                cuts_added_this_iteration += 1
                known_cut_keys.add(_cut_key(candidate_cut))
                if decision["cut_role"] == "secondary":
                    secondary_cuts_added += 1
            else:
                cuts_skipped += 1
                cuts_skipped_this_iteration += 1
                if decision["cut_role"] == "secondary":
                    secondary_cuts_skipped += 1
        if not active_candidates:
            cuts_skipped += 1
            cuts_skipped_this_iteration += 1
        if cuts_added_this_iteration:
            iterations_without_useful_cut = 0
        else:
            iterations_without_useful_cut += 1
        if any(
            decision["add"] for decision in cut_decisions if decision["cut_role"] == "secondary"
        ):
            iterations_without_secondary_cut = 0
        else:
            iterations_without_secondary_cut += 1
        cuts_generated_this_iteration = len(active_candidates)
        cuts_generated_counts.append(cuts_generated_this_iteration)
        primary_decision = cut_decisions[0] if cut_decisions else None
        cut_rhs_current = primary_decision["rhs"] if primary_decision else None
        cut_violation = primary_decision["raw_violation"] if primary_decision else None
        absolute_cut_violation = primary_decision["absolute_violation"] if primary_decision else None
        normalized_cut_violation = primary_decision["normalized_violation"] if primary_decision else None
        cut_added = bool(primary_decision and primary_decision["add"])
        cut_skip_reason = primary_decision["skip_reason"] if primary_decision else "no_incumbent"
        cut_add_reason = primary_decision["add_reason"] if primary_decision else None
        secondary_cut_decisions = [
            {
                "index": index,
                "normalized_violation": decision["normalized_violation"],
                "marginal_normalized_violation": decision[
                    "marginal_normalized_violation"
                ],
                "added": decision["add"],
                "skip_reason": decision["skip_reason"],
                "active_threshold": decision["active_threshold"],
            }
            for index, decision in enumerate(cut_decisions[1:], start=1)
        ]
        secondary_generated_cut_added = any(
            decision["add"]
            for decision in cut_decisions
            if decision["cut_role"] == "secondary"
        )
        secondary_generated_cut_duplicate = (
            additional_batch.duplicate_patterns_rejected > 0
            or any(
                decision["skip_reason"] == "duplicate_cut"
                for decision in cut_decisions
                if decision["cut_role"] == "secondary"
            )
        )
        forced_cut_added = any(decision["add_reason"] is not None for decision in cut_decisions)
        forced_cut_reason = next(
            (decision["add_reason"] for decision in cut_decisions if decision["add_reason"] is not None),
            None,
        )
        certification_transition = update_final_certification(
            certification_state,
            enabled=settings.final_certification_enabled,
            iteration=iteration + 1,
            active_gamma=active_gamma,
            gamma_target=settings.gamma_target,
            valid_ub=valid_ub,
            global_gap=gap,
            tol=settings.tol,
            useful_primary_cut_added=cut_added,
            no_cut_patience=settings.final_certification_no_cut_patience,
        )
        certification_state = certification_transition.state
        lb_improvement = (
            None if previous_lower_bound == -float("inf") else max(0.0, lower_bound - previous_lower_bound)
        )
        ub_improvement = (
            None if previous_upper_bound == float("inf") else max(0.0, previous_upper_bound - upper_bound)
        )
        log.append(
            {
                "iteration": iteration + 1,
                "gamma": active_gamma,
                "mip_gap": selected_mip_gap,
                "requested_master_mip_gap": selected_mip_gap,
                "precision_policy": settings.precision_config.precision_policy,
                "valid_global_gap_for_precision": (
                    precision_decision.valid_global_gap_for_precision
                ),
                "precision_gap_fallback_used": precision_decision.fallback_used,
                "adaptive_master_precision_enabled": (
                    settings.precision_config.adaptive_master_precision_enabled
                ),
                "adaptive_subproblem_precision_enabled": (
                    settings.precision_config.adaptive_subproblem_precision_enabled
                ),
                "master_gap_candidate": precision_decision.master_candidate_gap,
                "master_gap_previous": precision_decision.master_previous_gap,
                "master_gap_selected": precision_decision.master_selected_gap,
                "subproblem_gap_candidate": (
                    precision_decision.subproblem_candidate_gap
                ),
                "subproblem_gap_previous": (
                    precision_decision.subproblem_previous_gap
                ),
                "subproblem_gap_selected": (
                    precision_decision.subproblem_selected_gap
                ),
                "master_error_budget_ratio": (
                    settings.precision_config.master_error_budget_ratio
                ),
                "subproblem_error_budget_ratio": (
                    settings.precision_config.subproblem_error_budget_ratio
                ),
                "monotone_precision_tightening": (
                    settings.precision_config.monotone_precision_tightening
                ),
                "workload_policy_active": (
                    workload_decision is not None and not certification_active
                ),
                "workload_ema_decay": (
                    settings.workload_precision_config.ema_decay
                    if workload_decision is not None
                    and settings.workload_precision_config is not None
                    else None
                ),
                "workload_master_time_ema": (
                    workload_decision.master_time_ema
                    if workload_decision is not None and not certification_active
                    else None
                ),
                "workload_subproblem_time_ema": (
                    workload_decision.subproblem_time_ema
                    if workload_decision is not None and not certification_active
                    else None
                ),
                "workload_master_share_raw": (
                    workload_decision.master_share_raw
                    if workload_decision is not None and not certification_active
                    else None
                ),
                "workload_master_weight_selected": (
                    workload_decision.master_weight_selected
                    if workload_decision is not None and not certification_active
                    else None
                ),
                "workload_subproblem_weight_selected": (
                    workload_decision.subproblem_weight_selected
                    if workload_decision is not None and not certification_active
                    else None
                ),
                "workload_master_ratio_selected": (
                    workload_decision.master_ratio_selected
                    if workload_decision is not None and not certification_active
                    else None
                ),
                "workload_subproblem_ratio_selected": (
                    workload_decision.subproblem_ratio_selected
                    if workload_decision is not None and not certification_active
                    else None
                ),
                "workload_total_error_budget_ratio": (
                    settings.workload_precision_config.total_error_budget_ratio
                    if workload_decision is not None
                    and settings.workload_precision_config is not None
                    else None
                ),
                "workload_fallback_used": (
                    workload_decision.fallback_used
                    if workload_decision is not None and not certification_active
                    else False
                ),
                "workload_fallback_reason": (
                    workload_decision.fallback_reason
                    if workload_decision is not None and not certification_active
                    else None
                ),
                "realized_master_gap": float(model.MIPGap) if model.IsMIP else 0.0,
                "achieved_master_mip_gap": float(model.MIPGap) if model.IsMIP else 0.0,
                "master_status": int(model.Status),
                "master_best_bound": float(model.ObjBound),
                "master_time": master_elapsed,
                "lower_bound": lower_bound,
                "LB": lower_bound,
                "upper_bound": upper_bound,
                "UB": upper_bound,
                "gap": gap,
                "global_gap": gap,
                "lb_improvement": lb_improvement,
                "ub_improvement": ub_improvement,
                "elapsed_time": time.perf_counter() - start,
                "log_gap": policy_state.log_gap,
                "gap_improvement": policy_state.gap_improvement,
                "master_objective": float(model.ObjVal),
                "theta": theta_current,
                "theta_current": theta_current,
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
                "target_robust_evaluation_used": True,
                "subproblem_requested_mip_gap": requested_subproblem_gap,
                "subproblem_achieved_mip_gap": active_subproblem_mip_gap,
                "subproblem_status": active_subproblem_status,
                "subproblem_incumbent_objective": active_cut.objective,
                "subproblem_objective_bound": getattr(active_cut, "objective_bound", active_cut.objective),
                "subproblem_time": iteration_subproblem_time,
                "additional_subproblem_time": iteration_additional_subproblem_time,
                "subproblem_has_incumbent": getattr(active_cut, "has_incumbent", True),
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
                "absolute_cut_violation": absolute_cut_violation,
                "normalized_cut_violation": normalized_cut_violation,
                "secondary_cut_decisions": secondary_cut_decisions,
                "secondary_active_threshold": secondary_selection_threshold,
                "secondary_cuts_added_total": secondary_cuts_added,
                "secondary_cuts_skipped_total": secondary_cuts_skipped,
                "master_time_share": master_time_share,
                "secondary_solve_attempted": secondary_generation.attempt,
                "secondary_solve_trigger_reason": secondary_generation.trigger_reason,
                "secondary_solve_skipped_reason": secondary_generation.skipped_reason,
                "recent_relative_lb_improvement": (
                    secondary_generation.recent_relative_lb_improvement
                ),
                "secondary_solve_cooldown_remaining": (
                    secondary_generation.cooldown_remaining
                ),
                "secondary_solve_runtime": iteration_additional_subproblem_time,
                "secondary_generation_subproblem_time_share": (
                    generation_subproblem_time_share
                ),
                "secondary_generated_cut_added": secondary_generated_cut_added,
                "secondary_generated_cut_duplicate": secondary_generated_cut_duplicate,
                "secondary_solves_avoided_total": secondary_solves_avoided,
                "final_certification_active": certification_active,
                "final_certification_triggered_this_iteration": (
                    certification_transition.triggered_this_iteration
                ),
                "final_certification_trigger_iteration": (
                    certification_state.trigger_iteration
                ),
                "final_certification_reason": certification_transition.reason,
                "final_certification_count": certification_state.count,
                "consecutive_no_useful_primary_cuts": (
                    certification_state.consecutive_no_useful_primary_cuts
                ),
                "certification_forced_master_mip_gap": (
                    0.0 if certification_active else None
                ),
                "certification_forced_subproblem_mip_gap": (
                    0.0 if certification_active else None
                ),
                "secondary_solve_disabled_by_certification": (
                    certification_active and settings.max_cuts_per_iteration > 1
                ),
                "cut_added": cut_added,
                "cut_skip_reason": cut_skip_reason,
                "cut_add_reason": cut_add_reason,
                "cuts_added_total": cuts,
                "cuts_skipped_total": cuts_skipped,
                "cuts_generated_this_iteration": cuts_generated_this_iteration,
                "cuts_added_this_iteration": cuts_added_this_iteration,
                "cuts_skipped_this_iteration": cuts_skipped_this_iteration,
                "forced_cut_added": forced_cut_added,
                "forced_cut_reason": forced_cut_reason,
                "cuts": cuts,
            }
        )

        if should_terminate_benders(
            active_gamma,
            settings.gamma_target,
            valid_ub,
            gap,
            settings.tol,
        ):
            status = "optimal"
            break

    runtime = time.perf_counter() - start
    final_gap = None
    if upper_bound < float("inf") and lower_bound > -float("inf"):
        final_gap = calculate_global_gap(upper_bound, lower_bound)

    last_log = log[-1] if log else {}
    certification_exit_reason = certification_state.exit_reason
    if certification_state.active:
        certification_exit_reason = status
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
            "cut_selection_mode": settings.cut_selection_mode,
            "relative_cut_threshold": settings.relative_cut_threshold,
            "cut_violation_tol": settings.cut_violation_tol,
            "final_exact_gap": settings.final_exact_gap,
            "cut_stall_patience": settings.cut_stall_patience,
            "adaptive_secondary_cut_selection_enabled": (
                settings.adaptive_secondary_cut_selection_enabled
            ),
            "secondary_cut_warmup_cuts": settings.secondary_cut_warmup_cuts,
            "secondary_cut_master_time_share_trigger": (
                settings.secondary_cut_master_time_share_trigger
            ),
            "secondary_cut_recent_master_time_trigger": (
                settings.secondary_cut_recent_master_time_trigger
            ),
            "adaptive_secondary_generation_enabled": (
                settings.adaptive_secondary_generation_enabled
            ),
            "secondary_generation_lb_window": settings.secondary_generation_lb_window,
            "secondary_generation_stall_threshold": (
                settings.secondary_generation_stall_threshold
            ),
            "secondary_generation_cooldown_iterations": (
                settings.secondary_generation_cooldown_iterations
            ),
            "secondary_generation_max_subproblem_time_share": (
                settings.secondary_generation_max_subproblem_time_share
            ),
            "secondary_generation_min_remaining_time": (
                settings.secondary_generation_min_remaining_time
            ),
            "secondary_generation_min_solve_budget": (
                settings.secondary_generation_min_solve_budget
            ),
            "final_certification_enabled": settings.final_certification_enabled,
            "final_certification_triggered": certification_state.triggered,
            "final_certification_trigger_iteration": (
                certification_state.trigger_iteration
            ),
            "final_certification_count": certification_state.count,
            "final_certification_iterations": certification_state.iterations,
            "final_certification_exit_reason": certification_exit_reason,
            "precision_policy": settings.precision_config.precision_policy,
            "adaptive_master_precision_enabled": (
                settings.precision_config.adaptive_master_precision_enabled
            ),
            "adaptive_subproblem_precision_enabled": (
                settings.precision_config.adaptive_subproblem_precision_enabled
            ),
            "master_gap_max": settings.precision_config.master_gap_max,
            "master_gap_min": settings.precision_config.master_gap_min,
            "subproblem_gap_max": settings.precision_config.subproblem_gap_max,
            "subproblem_gap_min": settings.precision_config.subproblem_gap_min,
            "fixed_master_mip_gap": settings.precision_config.fixed_master_gap,
            "fixed_subproblem_mip_gap": settings.precision_config.fixed_subproblem_gap,
            "master_error_budget_ratio": (
                settings.precision_config.master_error_budget_ratio
            ),
            "subproblem_error_budget_ratio": (
                settings.precision_config.subproblem_error_budget_ratio
            ),
            "monotone_precision_tightening": (
                settings.precision_config.monotone_precision_tightening
            ),
            "workload_aware_policy_enabled": (
                settings.precision_config.precision_policy
                == "workload_aware_joint"
            ),
            "workload_final_master_time_ema": (
                workload_state.master_time_ema
                if settings.precision_config.precision_policy
                == "workload_aware_joint"
                else None
            ),
            "workload_final_subproblem_time_ema": (
                workload_state.subproblem_time_ema
                if settings.precision_config.precision_policy
                == "workload_aware_joint"
                else None
            ),
            "workload_final_master_weight": (
                workload_master_weights[-1] if workload_master_weights else None
            ),
            "workload_final_subproblem_weight": (
                workload_subproblem_weights[-1]
                if workload_subproblem_weights
                else None
            ),
            "workload_mean_master_weight": (
                sum(workload_master_weights) / len(workload_master_weights)
                if workload_master_weights
                else None
            ),
            "workload_mean_subproblem_weight": (
                sum(workload_subproblem_weights) / len(workload_subproblem_weights)
                if workload_subproblem_weights
                else None
            ),
            "workload_fallback_count": workload_fallback_count,
            "cuts_added_total": cuts,
            "cuts_skipped_total": cuts_skipped,
            "secondary_cuts_added_total": secondary_cuts_added,
            "secondary_cuts_skipped_total": secondary_cuts_skipped,
            "secondary_solves_attempted_total": secondary_solves_attempted,
            "secondary_solves_avoided_total": secondary_solves_avoided,
            "last_secondary_solve_attempted": last_log.get("secondary_solve_attempted"),
            "last_secondary_solve_trigger_reason": last_log.get(
                "secondary_solve_trigger_reason"
            ),
            "last_secondary_solve_skipped_reason": last_log.get(
                "secondary_solve_skipped_reason"
            ),
            "last_recent_relative_lb_improvement": last_log.get(
                "recent_relative_lb_improvement"
            ),
            "last_secondary_cut_decisions": last_log.get("secondary_cut_decisions"),
            "last_secondary_active_threshold": last_log.get("secondary_active_threshold"),
            "last_cut_violation": last_log.get("cut_violation"),
            "last_normalized_cut_violation": last_log.get("normalized_cut_violation"),
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
            "adaptive_subproblem_gap_enabled": settings.adaptive_subproblem_gap_enabled,
            "subproblem_gap_schedule": settings.subproblem_gap_schedule,
            "last_subproblem_requested_mip_gap": last_log.get("subproblem_requested_mip_gap"),
            "last_subproblem_achieved_mip_gap": last_log.get("subproblem_achieved_mip_gap"),
            "mean_subproblem_requested_mip_gap": (
                sum(requested_subproblem_gaps) / len(requested_subproblem_gaps)
                if requested_subproblem_gaps
                else None
            ),
            "num_subproblem_nonoptimal": subproblem_nonoptimal,
            "num_subproblem_without_incumbent": subproblem_without_incumbent,
            "max_cuts_per_iteration": settings.max_cuts_per_iteration,
            "mean_cuts_generated_per_iteration": (
                sum(cuts_generated_counts) / len(cuts_generated_counts) if cuts_generated_counts else None
            ),
            "duplicate_cuts_rejected": duplicate_cuts_rejected,
            "duplicate_patterns_rejected": duplicate_patterns_rejected,
            "additional_subproblem_time": additional_subproblem_time,
            "best_y_values": best_y_values,
            "best_x_values": best_x_values,
            **scenario_metadata,
        },
        iteration_log=log,
    )
