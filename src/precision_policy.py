from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


PRECISION_POLICIES = {"legacy", "joint_error_budget", "workload_aware_joint"}


@dataclass(frozen=True)
class PrecisionPolicyConfig:
    precision_policy: str
    adaptive_master_precision_enabled: bool
    adaptive_subproblem_precision_enabled: bool
    master_gap_max: float
    master_gap_min: float
    subproblem_gap_max: float
    subproblem_gap_min: float
    master_error_budget_ratio: float
    subproblem_error_budget_ratio: float
    monotone_precision_tightening: bool
    fixed_master_gap: float
    fixed_subproblem_gap: float


@dataclass(frozen=True)
class PrecisionPolicyState:
    previous_master_gap: float
    previous_subproblem_gap: float


@dataclass(frozen=True)
class PrecisionPolicyDecision:
    valid_global_gap_for_precision: float
    fallback_used: bool
    master_candidate_gap: float
    master_previous_gap: float
    master_selected_gap: float
    subproblem_candidate_gap: float
    subproblem_previous_gap: float
    subproblem_selected_gap: float
    next_state: PrecisionPolicyState


@dataclass(frozen=True)
class WorkloadAwarePrecisionConfig:
    ema_decay: float
    total_error_budget_ratio: float
    master_weight_min: float
    master_weight_max: float
    time_epsilon: float
    initial_master_weight: float
    initial_subproblem_weight: float


@dataclass(frozen=True)
class WorkloadAwarePrecisionState:
    precision_state: PrecisionPolicyState
    master_time_ema: float | None = None
    subproblem_time_ema: float | None = None
    time_observations: int = 0
    last_time_update_error: str | None = None


@dataclass(frozen=True)
class WorkloadAwarePrecisionDecision:
    precision_decision: PrecisionPolicyDecision
    next_state: WorkloadAwarePrecisionState
    policy_active: bool
    master_time_ema: float | None
    subproblem_time_ema: float | None
    master_share_raw: float | None
    master_weight_selected: float
    subproblem_weight_selected: float
    master_ratio_selected: float
    subproblem_ratio_selected: float
    total_error_budget_ratio: float
    fallback_used: bool
    fallback_reason: str | None


def _finite_in_range(name: str, value: Any, minimum: float, maximum: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return number


def _finite_nonnegative(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def precision_policy_config(
    algorithm_cfg: dict[str, Any],
    *,
    fixed_master_gap: float,
    fixed_subproblem_gap: float,
    legacy_subproblem_gaps: list[float],
) -> PrecisionPolicyConfig:
    policy = str(algorithm_cfg.get("precision_policy", "legacy"))
    if policy not in PRECISION_POLICIES:
        options = ", ".join(sorted(PRECISION_POLICIES))
        raise ValueError(f"precision_policy must be one of: {options}")

    legacy_gaps = [float(value) for value in legacy_subproblem_gaps]
    default_subproblem_max = max(legacy_gaps) if legacy_gaps else fixed_master_gap
    default_subproblem_min = min(legacy_gaps) if legacy_gaps else fixed_master_gap
    master_gap_max = _finite_nonnegative(
        "master_gap_max",
        algorithm_cfg.get("master_gap_max", fixed_master_gap),
    )
    master_gap_min = _finite_nonnegative(
        "master_gap_min",
        algorithm_cfg.get("master_gap_min", min(fixed_master_gap, master_gap_max)),
    )
    subproblem_gap_max = _finite_nonnegative(
        "subproblem_gap_max",
        algorithm_cfg.get("subproblem_gap_max", default_subproblem_max),
    )
    subproblem_gap_min = _finite_nonnegative(
        "subproblem_gap_min",
        algorithm_cfg.get("subproblem_gap_min", default_subproblem_min),
    )
    if master_gap_min > master_gap_max:
        raise ValueError("master_gap_min must be less than or equal to master_gap_max")
    if subproblem_gap_min > subproblem_gap_max:
        raise ValueError(
            "subproblem_gap_min must be less than or equal to subproblem_gap_max"
        )

    fixed_master = _finite_nonnegative(
        "fixed_master_mip_gap",
        algorithm_cfg.get("fixed_master_mip_gap", fixed_master_gap),
    )
    fixed_subproblem = _finite_nonnegative(
        "fixed_subproblem_mip_gap",
        algorithm_cfg.get("fixed_subproblem_mip_gap", fixed_subproblem_gap),
    )
    return PrecisionPolicyConfig(
        precision_policy=policy,
        adaptive_master_precision_enabled=bool(
            algorithm_cfg.get("adaptive_master_precision_enabled", False)
        ),
        adaptive_subproblem_precision_enabled=bool(
            algorithm_cfg.get("adaptive_subproblem_precision_enabled", False)
        ),
        master_gap_max=master_gap_max,
        master_gap_min=master_gap_min,
        subproblem_gap_max=subproblem_gap_max,
        subproblem_gap_min=subproblem_gap_min,
        master_error_budget_ratio=_finite_nonnegative(
            "master_error_budget_ratio",
            algorithm_cfg.get("master_error_budget_ratio", 0.5),
        ),
        subproblem_error_budget_ratio=_finite_nonnegative(
            "subproblem_error_budget_ratio",
            algorithm_cfg.get("subproblem_error_budget_ratio", 0.5),
        ),
        monotone_precision_tightening=bool(
            algorithm_cfg.get("monotone_precision_tightening", True)
        ),
        fixed_master_gap=fixed_master,
        fixed_subproblem_gap=fixed_subproblem,
    )


def workload_aware_precision_config(
    algorithm_cfg: dict[str, Any],
) -> WorkloadAwarePrecisionConfig:
    decay = _finite_in_range(
        "workload_ema_decay",
        algorithm_cfg.get("workload_ema_decay", 0.80),
        0.0,
        1.0,
    )
    if decay >= 1.0:
        raise ValueError("workload_ema_decay must be less than 1")
    total_ratio = _finite_nonnegative(
        "workload_total_error_budget_ratio",
        algorithm_cfg.get("workload_total_error_budget_ratio", 0.75),
    )
    if total_ratio <= 0.0:
        raise ValueError("workload_total_error_budget_ratio must be positive")
    master_min = _finite_in_range(
        "workload_master_weight_min",
        algorithm_cfg.get("workload_master_weight_min", 1.0 / 3.0),
        0.0,
        1.0,
    )
    master_max = _finite_in_range(
        "workload_master_weight_max",
        algorithm_cfg.get("workload_master_weight_max", 2.0 / 3.0),
        0.0,
        1.0,
    )
    if master_min > master_max:
        raise ValueError(
            "workload_master_weight_min must not exceed workload_master_weight_max"
        )
    time_epsilon = _finite_nonnegative(
        "workload_time_epsilon",
        algorithm_cfg.get("workload_time_epsilon", 1.0e-9),
    )
    if time_epsilon <= 0.0:
        raise ValueError("workload_time_epsilon must be positive")
    initial_master = _finite_in_range(
        "workload_initial_master_weight",
        algorithm_cfg.get("workload_initial_master_weight", 1.0 / 3.0),
        master_min,
        master_max,
    )
    initial_subproblem = _finite_in_range(
        "workload_initial_subproblem_weight",
        algorithm_cfg.get("workload_initial_subproblem_weight", 2.0 / 3.0),
        0.0,
        1.0,
    )
    if not math.isclose(initial_master + initial_subproblem, 1.0, abs_tol=1e-12):
        raise ValueError("workload initial weights must sum to 1")
    return WorkloadAwarePrecisionConfig(
        ema_decay=decay,
        total_error_budget_ratio=total_ratio,
        master_weight_min=master_min,
        master_weight_max=master_max,
        time_epsilon=time_epsilon,
        initial_master_weight=initial_master,
        initial_subproblem_weight=initial_subproblem,
    )


def initialize_precision_state(config: PrecisionPolicyConfig) -> PrecisionPolicyState:
    return PrecisionPolicyState(
        previous_master_gap=config.master_gap_max,
        previous_subproblem_gap=config.subproblem_gap_max,
    )


def initialize_workload_aware_state(
    precision_config: PrecisionPolicyConfig,
) -> WorkloadAwarePrecisionState:
    return WorkloadAwarePrecisionState(
        precision_state=initialize_precision_state(precision_config),
    )


def valid_global_gap_for_precision(
    upper_bound: float | None,
    lower_bound: float | None,
) -> tuple[float, bool]:
    if (
        upper_bound is None
        or lower_bound is None
        or not math.isfinite(float(upper_bound))
        or not math.isfinite(float(lower_bound))
    ):
        return 1.0, True
    gap = max(
        0.0,
        (float(upper_bound) - float(lower_bound))
        / max(1.0, abs(float(upper_bound))),
    )
    return gap, False


def error_budget_candidate(
    global_gap: float,
    minimum_gap: float,
    maximum_gap: float,
    error_budget_ratio: float,
) -> float:
    return max(
        float(minimum_gap),
        min(float(maximum_gap), float(error_budget_ratio) * float(global_gap)),
    )


def select_joint_error_budget_precision(
    config: PrecisionPolicyConfig,
    state: PrecisionPolicyState,
    *,
    upper_bound: float | None,
    lower_bound: float | None,
    update_state: bool = True,
) -> PrecisionPolicyDecision:
    gap, fallback = valid_global_gap_for_precision(upper_bound, lower_bound)
    master_candidate = error_budget_candidate(
        gap,
        config.master_gap_min,
        config.master_gap_max,
        config.master_error_budget_ratio,
    )
    subproblem_candidate = error_budget_candidate(
        gap,
        config.subproblem_gap_min,
        config.subproblem_gap_max,
        config.subproblem_error_budget_ratio,
    )

    master_adaptive = (
        min(state.previous_master_gap, master_candidate)
        if config.monotone_precision_tightening
        else master_candidate
    )
    subproblem_adaptive = (
        min(state.previous_subproblem_gap, subproblem_candidate)
        if config.monotone_precision_tightening
        else subproblem_candidate
    )
    master_selected = (
        master_adaptive
        if config.adaptive_master_precision_enabled
        else config.fixed_master_gap
    )
    subproblem_selected = (
        subproblem_adaptive
        if config.adaptive_subproblem_precision_enabled
        else config.fixed_subproblem_gap
    )

    next_state = state
    if update_state:
        next_state = PrecisionPolicyState(
            previous_master_gap=(
                master_adaptive
                if config.adaptive_master_precision_enabled
                else state.previous_master_gap
            ),
            previous_subproblem_gap=(
                subproblem_adaptive
                if config.adaptive_subproblem_precision_enabled
                else state.previous_subproblem_gap
            ),
        )
    return PrecisionPolicyDecision(
        valid_global_gap_for_precision=gap,
        fallback_used=fallback,
        master_candidate_gap=master_candidate,
        master_previous_gap=state.previous_master_gap,
        master_selected_gap=master_selected,
        subproblem_candidate_gap=subproblem_candidate,
        subproblem_previous_gap=state.previous_subproblem_gap,
        subproblem_selected_gap=subproblem_selected,
        next_state=next_state,
    )


def _usable_time(value: float | None) -> tuple[float | None, str | None]:
    if value is None:
        return None, "missing_time_history"
    number = float(value)
    if not math.isfinite(number):
        return None, "nonfinite_time_history"
    if number < 0.0:
        return None, "negative_time_history"
    return number, None


def select_workload_aware_precision(
    precision_config: PrecisionPolicyConfig,
    workload_config: WorkloadAwarePrecisionConfig,
    state: WorkloadAwarePrecisionState,
    *,
    upper_bound: float | None,
    lower_bound: float | None,
    update_state: bool = True,
) -> WorkloadAwarePrecisionDecision:
    gap, global_gap_fallback = valid_global_gap_for_precision(
        upper_bound,
        lower_bound,
    )
    master_ema, master_error = _usable_time(state.master_time_ema)
    subproblem_ema, subproblem_error = _usable_time(state.subproblem_time_ema)
    fallback_reasons: list[str] = []
    if global_gap_fallback:
        fallback_reasons.append("invalid_global_gap")
    if master_error is not None:
        fallback_reasons.append(f"master_{master_error}")
    if subproblem_error is not None:
        fallback_reasons.append(f"subproblem_{subproblem_error}")

    master_share_raw: float | None = None
    if master_ema is not None and subproblem_ema is not None:
        total_time = master_ema + subproblem_ema
        if total_time <= workload_config.time_epsilon:
            fallback_reasons.append("time_sum_too_small")
        else:
            master_share_raw = master_ema / max(
                workload_config.time_epsilon,
                total_time,
            )

    fallback_used = bool(fallback_reasons)
    if fallback_used:
        master_weight = workload_config.initial_master_weight
        subproblem_weight = workload_config.initial_subproblem_weight
    else:
        if master_share_raw is None:
            raise RuntimeError("Valid workload history did not produce a master share")
        master_weight = max(
            workload_config.master_weight_min,
            min(workload_config.master_weight_max, master_share_raw),
        )
        subproblem_weight = 1.0 - master_weight

    master_ratio = workload_config.total_error_budget_ratio * master_weight
    subproblem_ratio = (
        workload_config.total_error_budget_ratio * subproblem_weight
    )
    master_candidate = error_budget_candidate(
        gap,
        precision_config.master_gap_min,
        precision_config.master_gap_max,
        master_ratio,
    )
    subproblem_candidate = error_budget_candidate(
        gap,
        precision_config.subproblem_gap_min,
        precision_config.subproblem_gap_max,
        subproblem_ratio,
    )
    master_adaptive = (
        min(state.precision_state.previous_master_gap, master_candidate)
        if precision_config.monotone_precision_tightening
        else master_candidate
    )
    subproblem_adaptive = (
        min(state.precision_state.previous_subproblem_gap, subproblem_candidate)
        if precision_config.monotone_precision_tightening
        else subproblem_candidate
    )
    master_selected = (
        master_adaptive
        if precision_config.adaptive_master_precision_enabled
        else precision_config.fixed_master_gap
    )
    subproblem_selected = (
        subproblem_adaptive
        if precision_config.adaptive_subproblem_precision_enabled
        else precision_config.fixed_subproblem_gap
    )
    next_precision_state = state.precision_state
    if update_state:
        next_precision_state = PrecisionPolicyState(
            previous_master_gap=(
                master_adaptive
                if precision_config.adaptive_master_precision_enabled
                else state.precision_state.previous_master_gap
            ),
            previous_subproblem_gap=(
                subproblem_adaptive
                if precision_config.adaptive_subproblem_precision_enabled
                else state.precision_state.previous_subproblem_gap
            ),
        )
    next_state = WorkloadAwarePrecisionState(
        precision_state=next_precision_state,
        master_time_ema=state.master_time_ema,
        subproblem_time_ema=state.subproblem_time_ema,
        time_observations=state.time_observations,
        last_time_update_error=state.last_time_update_error,
    )
    precision_decision = PrecisionPolicyDecision(
        valid_global_gap_for_precision=gap,
        fallback_used=global_gap_fallback,
        master_candidate_gap=master_candidate,
        master_previous_gap=state.precision_state.previous_master_gap,
        master_selected_gap=master_selected,
        subproblem_candidate_gap=subproblem_candidate,
        subproblem_previous_gap=state.precision_state.previous_subproblem_gap,
        subproblem_selected_gap=subproblem_selected,
        next_state=next_precision_state,
    )
    return WorkloadAwarePrecisionDecision(
        precision_decision=precision_decision,
        next_state=next_state,
        policy_active=True,
        master_time_ema=master_ema,
        subproblem_time_ema=subproblem_ema,
        master_share_raw=master_share_raw,
        master_weight_selected=master_weight,
        subproblem_weight_selected=subproblem_weight,
        master_ratio_selected=master_ratio,
        subproblem_ratio_selected=subproblem_ratio,
        total_error_budget_ratio=workload_config.total_error_budget_ratio,
        fallback_used=fallback_used,
        fallback_reason=";".join(fallback_reasons) if fallback_reasons else None,
    )


def update_workload_time_ema(
    config: WorkloadAwarePrecisionConfig,
    state: WorkloadAwarePrecisionState,
    *,
    master_time: float | None,
    subproblem_time: float | None,
    update_state: bool = True,
) -> WorkloadAwarePrecisionState:
    if not update_state:
        return state
    master_value, master_error = _usable_time(master_time)
    subproblem_value, subproblem_error = _usable_time(subproblem_time)
    errors = [
        reason
        for reason in (
            f"master_{master_error}" if master_error else None,
            f"subproblem_{subproblem_error}" if subproblem_error else None,
        )
        if reason is not None
    ]
    if errors:
        return WorkloadAwarePrecisionState(
            precision_state=state.precision_state,
            master_time_ema=None,
            subproblem_time_ema=None,
            time_observations=state.time_observations,
            last_time_update_error=";".join(errors),
        )
    if master_value is None or subproblem_value is None:
        raise RuntimeError("Validated workload times unexpectedly became unavailable")
    previous_master, previous_master_error = _usable_time(state.master_time_ema)
    previous_subproblem, previous_subproblem_error = _usable_time(
        state.subproblem_time_ema
    )
    has_valid_history = (
        previous_master_error is None
        and previous_subproblem_error is None
        and previous_master is not None
        and previous_subproblem is not None
    )
    if has_valid_history:
        master_ema = (
            config.ema_decay * previous_master
            + (1.0 - config.ema_decay) * master_value
        )
        subproblem_ema = (
            config.ema_decay * previous_subproblem
            + (1.0 - config.ema_decay) * subproblem_value
        )
    else:
        master_ema = master_value
        subproblem_ema = subproblem_value
    return WorkloadAwarePrecisionState(
        precision_state=state.precision_state,
        master_time_ema=master_ema,
        subproblem_time_ema=subproblem_ema,
        time_observations=state.time_observations + 1,
        last_time_update_error=None,
    )
