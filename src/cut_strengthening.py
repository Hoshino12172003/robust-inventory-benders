from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


CUT_STRENGTHENING_POLICIES = {
    "none",
    "core_point",
    "stall_secondary",
    "core_point_stall_secondary",
}


@dataclass(frozen=True)
class CutStrengtheningConfig:
    policy: str
    core_point_update_weight: float
    core_point_min_distance: float
    core_point_stage1_time_limit: float
    core_point_stage2_time_limit: float
    core_point_min_remaining_time: float
    core_point_min_global_gap: float
    core_point_current_abs_tol: float
    core_point_current_rel_tol: float
    core_point_min_normalized_improvement: float
    v3_secondary_lb_window: int
    v3_secondary_stall_threshold: float
    v3_secondary_cooldown_iterations: int
    v3_secondary_min_global_gap: float
    v3_secondary_min_remaining_time: float
    v3_secondary_max_time_per_attempt: float
    v3_secondary_max_time_fraction_of_remaining: float
    v3_secondary_max_extra_time_share: float
    v3_secondary_pattern_memory: int

    @property
    def core_point_enabled(self) -> bool:
        return self.policy in {"core_point", "core_point_stall_secondary"}

    @property
    def secondary_enabled(self) -> bool:
        return self.policy in {"stall_secondary", "core_point_stall_secondary"}


@dataclass(frozen=True)
class CorePointState:
    core_x: dict[tuple[int, int], float] | None = None
    observations: int = 0


@dataclass(frozen=True)
class CorePointAttemptDecision:
    attempt: bool
    skipped_reason: str | None
    distance: float | None


@dataclass(frozen=True)
class V3SecondaryCutDecision:
    attempt: bool
    trigger_reason: str | None
    skipped_reason: str | None
    recent_lb_improvement: float | None
    cooldown_remaining: int
    time_limit: float | None


@dataclass(frozen=True)
class CorePointCutAcceptance:
    accepted: bool
    fallback_reason: str | None
    normalized_improvement: float | None


@dataclass(frozen=True)
class V3SecondaryPatternMemory:
    patterns: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class V3SecondaryCutAcceptance:
    accepted: bool
    skip_reason: str | None


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _nonnegative(name: str, value: Any) -> float:
    number = _finite(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _positive(name: str, value: Any) -> float:
    number = _finite(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _unit_interval(name: str, value: Any) -> float:
    number = _finite(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _positive_unit_interval(name: str, value: Any) -> float:
    number = _finite(name, value)
    if not 0.0 < number <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return number


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    finite = _finite(name, value)
    number = int(finite)
    if finite != float(number) or number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    finite = _finite(name, value)
    number = int(finite)
    if finite != float(number) or number < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return number


def cut_strengthening_config(algorithm_cfg: dict[str, Any]) -> CutStrengtheningConfig:
    policy = str(algorithm_cfg.get("cut_strengthening_policy", "none"))
    if policy not in CUT_STRENGTHENING_POLICIES:
        options = ", ".join(sorted(CUT_STRENGTHENING_POLICIES))
        raise ValueError(f"cut_strengthening_policy must be one of: {options}")
    return CutStrengtheningConfig(
        policy=policy,
        core_point_update_weight=_unit_interval(
            "core_point_update_weight",
            algorithm_cfg.get("core_point_update_weight", 0.50),
        ),
        core_point_min_distance=_nonnegative(
            "core_point_min_distance",
            algorithm_cfg.get("core_point_min_distance", 1.0e-9),
        ),
        core_point_stage1_time_limit=_positive(
            "core_point_stage1_time_limit",
            algorithm_cfg.get("core_point_stage1_time_limit", 2.0),
        ),
        core_point_stage2_time_limit=_positive(
            "core_point_stage2_time_limit",
            algorithm_cfg.get("core_point_stage2_time_limit", 2.0),
        ),
        core_point_min_remaining_time=_nonnegative(
            "core_point_min_remaining_time",
            algorithm_cfg.get("core_point_min_remaining_time", 10.0),
        ),
        core_point_min_global_gap=_nonnegative(
            "core_point_min_global_gap",
            algorithm_cfg.get("core_point_min_global_gap", 5.0e-4),
        ),
        core_point_current_abs_tol=_nonnegative(
            "core_point_current_abs_tol",
            algorithm_cfg.get("core_point_current_abs_tol", 1.0e-7),
        ),
        core_point_current_rel_tol=_nonnegative(
            "core_point_current_rel_tol",
            algorithm_cfg.get("core_point_current_rel_tol", 1.0e-8),
        ),
        core_point_min_normalized_improvement=_nonnegative(
            "core_point_min_normalized_improvement",
            algorithm_cfg.get("core_point_min_normalized_improvement", 1.0e-7),
        ),
        v3_secondary_lb_window=_positive_int(
            "v3_secondary_lb_window",
            algorithm_cfg.get("v3_secondary_lb_window", 5),
        ),
        v3_secondary_stall_threshold=_nonnegative(
            "v3_secondary_stall_threshold",
            algorithm_cfg.get("v3_secondary_stall_threshold", 1.0e-4),
        ),
        v3_secondary_cooldown_iterations=_nonnegative_int(
            "v3_secondary_cooldown_iterations",
            algorithm_cfg.get("v3_secondary_cooldown_iterations", 10),
        ),
        v3_secondary_min_global_gap=_nonnegative(
            "v3_secondary_min_global_gap",
            algorithm_cfg.get("v3_secondary_min_global_gap", 1.0e-3),
        ),
        v3_secondary_min_remaining_time=_nonnegative(
            "v3_secondary_min_remaining_time",
            algorithm_cfg.get("v3_secondary_min_remaining_time", 30.0),
        ),
        v3_secondary_max_time_per_attempt=_positive(
            "v3_secondary_max_time_per_attempt",
            algorithm_cfg.get("v3_secondary_max_time_per_attempt", 10.0),
        ),
        v3_secondary_max_time_fraction_of_remaining=_positive_unit_interval(
            "v3_secondary_max_time_fraction_of_remaining",
            algorithm_cfg.get("v3_secondary_max_time_fraction_of_remaining", 0.05),
        ),
        v3_secondary_max_extra_time_share=_positive_unit_interval(
            "v3_secondary_max_extra_time_share",
            algorithm_cfg.get("v3_secondary_max_extra_time_share", 0.10),
        ),
        v3_secondary_pattern_memory=_positive_int(
            "v3_secondary_pattern_memory",
            algorithm_cfg.get("v3_secondary_pattern_memory", 10),
        ),
    )


def initialize_core_point_state() -> CorePointState:
    return CorePointState()


def core_point_distance(
    current_x: dict[tuple[int, int], float],
    core_x: dict[tuple[int, int], float],
) -> float:
    if set(current_x) != set(core_x):
        raise ValueError("current_x and core_x must use identical keys")
    return math.sqrt(sum((float(current_x[key]) - float(core_x[key])) ** 2 for key in current_x))


def update_core_point_state(
    state: CorePointState,
    current_x: dict[tuple[int, int], float],
    update_weight: float,
    *,
    update_state: bool = True,
) -> CorePointState:
    if not update_state:
        return state
    eta = _unit_interval("core_point_update_weight", update_weight)
    cleaned = {key: float(value) for key, value in current_x.items()}
    if any(not math.isfinite(value) for value in cleaned.values()):
        raise ValueError("current_x must contain only finite values")
    if state.core_x is None:
        return CorePointState(core_x=cleaned, observations=1)
    if set(state.core_x) != set(cleaned):
        raise ValueError("Core-point update requires identical x keys")
    updated = {
        key: eta * float(state.core_x[key]) + (1.0 - eta) * cleaned[key]
        for key in cleaned
    }
    return CorePointState(core_x=updated, observations=state.observations + 1)


def should_attempt_core_point_strengthening(
    config: CutStrengtheningConfig,
    state: CorePointState,
    current_x: dict[tuple[int, int], float],
    *,
    subproblem_mode: str,
    primary_has_incumbent: bool,
    primary_absolute_violation: float,
    primary_violation_tolerance: float = 0.0,
    global_gap: float,
    remaining_time: float,
    certification_active: bool,
) -> CorePointAttemptDecision:
    if not config.core_point_enabled:
        return CorePointAttemptDecision(False, "component_disabled", None)
    if subproblem_mode != "robust_dual_milp":
        return CorePointAttemptDecision(False, "subproblem_mode", None)
    if certification_active:
        return CorePointAttemptDecision(False, "final_certification", None)
    if state.core_x is None or state.observations <= 0:
        return CorePointAttemptDecision(False, "core_point_unavailable", None)
    if not primary_has_incumbent:
        return CorePointAttemptDecision(False, "primary_without_incumbent", None)
    if primary_absolute_violation <= _nonnegative(
        "primary_violation_tolerance",
        primary_violation_tolerance,
    ):
        return CorePointAttemptDecision(False, "primary_not_violated", None)
    if not math.isfinite(global_gap) or global_gap <= config.core_point_min_global_gap:
        return CorePointAttemptDecision(False, "global_gap_too_small", None)
    if not math.isfinite(remaining_time) or remaining_time < config.core_point_min_remaining_time:
        return CorePointAttemptDecision(False, "insufficient_remaining_time", None)
    distance = core_point_distance(current_x, state.core_x)
    if distance < config.core_point_min_distance:
        return CorePointAttemptDecision(False, "core_point_too_close", distance)
    return CorePointAttemptDecision(True, None, distance)


def normalized_core_improvement(strengthened_value: float, original_value: float) -> float:
    strengthened = _finite("strengthened_value", strengthened_value)
    original = _finite("original_value", original_value)
    return max(0.0, strengthened - original) / max(
        1.0,
        abs(strengthened),
        abs(original),
    )


def core_point_cut_acceptance(
    *,
    stage1_optimal: bool,
    stage2_optimal: bool,
    dual_feasible: bool,
    strengthened_value_at_current: float | None,
    current_value_floor: float | None,
    original_value_at_current: float | None,
    strengthened_value_at_core: float | None,
    original_value_at_core: float | None,
    current_tolerance: float,
    minimum_normalized_improvement: float,
    duplicate: bool,
    original_primary_violated: bool,
    certification_active: bool,
) -> CorePointCutAcceptance:
    if not stage1_optimal:
        return CorePointCutAcceptance(False, "stage1_not_optimal", None)
    if not stage2_optimal:
        return CorePointCutAcceptance(False, "stage2_not_optimal", None)
    if not dual_feasible:
        return CorePointCutAcceptance(False, "dual_infeasible", None)
    scalar_values = (
        strengthened_value_at_current,
        current_value_floor,
        original_value_at_current,
        strengthened_value_at_core,
        original_value_at_core,
    )
    if any(value is None or not math.isfinite(float(value)) for value in scalar_values):
        return CorePointCutAcceptance(False, "missing_strengthening_value", None)
    strengthened_current = float(strengthened_value_at_current)
    floor = float(current_value_floor)
    original_current = float(original_value_at_current)
    tolerance = _nonnegative("current_tolerance", current_tolerance)
    if strengthened_current < floor - 1.0e-7:
        return CorePointCutAcceptance(False, "current_point_floor_violated", None)
    if strengthened_current < original_current - tolerance - 1.0e-7:
        return CorePointCutAcceptance(False, "weaker_at_current_point", None)
    improvement = normalized_core_improvement(
        float(strengthened_value_at_core),
        float(original_value_at_core),
    )
    if improvement <= 0.0:
        return CorePointCutAcceptance(False, "no_core_point_improvement", improvement)
    if improvement <= _nonnegative(
        "minimum_normalized_improvement",
        minimum_normalized_improvement,
    ):
        return CorePointCutAcceptance(False, "improvement_below_threshold", improvement)
    if duplicate:
        return CorePointCutAcceptance(False, "duplicate_strengthened_cut", improvement)
    if not original_primary_violated:
        return CorePointCutAcceptance(False, "original_primary_not_violated", improvement)
    if certification_active:
        return CorePointCutAcceptance(False, "final_certification", improvement)
    return CorePointCutAcceptance(True, None, improvement)


def recent_relative_lb_improvement(
    lower_bound_history: list[float],
    window: int,
) -> float | None:
    window = _positive_int("v3_secondary_lb_window", window)
    if len(lower_bound_history) < window:
        return None
    values = [float(value) for value in lower_bound_history[-window:]]
    if any(not math.isfinite(value) for value in values):
        return None
    first, last = values[0], values[-1]
    return max(0.0, last - first) / max(1.0, abs(first), abs(last))


def should_attempt_v3_secondary_cut(
    config: CutStrengtheningConfig,
    *,
    subproblem_mode: str,
    active_gamma: int,
    target_gamma: int,
    certification_active: bool,
    primary_has_incumbent: bool,
    primary_pattern_valid: bool,
    global_gap: float,
    lower_bound_history: list[float],
    current_iteration: int,
    last_attempt_iteration: int | None,
    remaining_time: float,
    extra_runtime: float,
    elapsed_time: float,
) -> V3SecondaryCutDecision:
    improvement = recent_relative_lb_improvement(
        lower_bound_history,
        config.v3_secondary_lb_window,
    )
    cooldown = 0
    if last_attempt_iteration is not None:
        cooldown = max(
            0,
            config.v3_secondary_cooldown_iterations
            - (current_iteration - last_attempt_iteration),
        )
    reason: str | None = None
    if not config.secondary_enabled:
        reason = "component_disabled"
    elif subproblem_mode != "robust_dual_milp":
        reason = "subproblem_mode"
    elif active_gamma != target_gamma:
        reason = "active_gamma_not_target"
    elif certification_active:
        reason = "final_certification"
    elif current_iteration <= 1:
        reason = "first_iteration"
    elif not primary_has_incumbent:
        reason = "primary_without_incumbent"
    elif not primary_pattern_valid:
        reason = "invalid_primary_pattern"
    elif not math.isfinite(global_gap) or global_gap <= config.v3_secondary_min_global_gap:
        reason = "global_gap_too_small"
    elif improvement is None:
        reason = "insufficient_lb_history"
    elif improvement > config.v3_secondary_stall_threshold:
        reason = "lb_progressing"
    elif cooldown > 0:
        reason = "cooldown"
    elif not math.isfinite(remaining_time) or remaining_time < config.v3_secondary_min_remaining_time:
        reason = "insufficient_remaining_time"
    elif elapsed_time > 0.0 and extra_runtime / elapsed_time > config.v3_secondary_max_extra_time_share:
        reason = "extra_time_share_limit"
    secondary_limit = min(
        config.v3_secondary_max_time_per_attempt,
        config.v3_secondary_max_time_fraction_of_remaining * max(0.0, remaining_time),
    )
    if reason is None and secondary_limit < 1.0:
        reason = "secondary_time_limit_below_one_second"
    if reason is not None:
        return V3SecondaryCutDecision(
            False,
            None,
            reason,
            improvement,
            cooldown,
            None,
        )
    return V3SecondaryCutDecision(
        True,
        "lb_stall",
        None,
        improvement,
        0,
        secondary_limit,
    )


def update_secondary_pattern_memory(
    state: V3SecondaryPatternMemory,
    pattern: tuple[int, ...],
    maximum_size: int,
) -> V3SecondaryPatternMemory:
    limit = _positive_int("v3_secondary_pattern_memory", maximum_size)
    values = tuple(int(value) for value in pattern)
    if any(value not in {0, 1} for value in values):
        raise ValueError("Secondary pattern memory accepts binary patterns only")
    retained = [existing for existing in state.patterns if existing != values]
    retained.append(values)
    return V3SecondaryPatternMemory(patterns=tuple(retained[-limit:]))


def v3_secondary_cut_acceptance(
    *,
    has_incumbent: bool,
    pattern_differs_from_primary: bool,
    pattern_in_memory: bool,
    duplicate_cut: bool,
    absolute_violation: float,
    violation_tolerance: float,
    certification_active: bool,
    already_added_this_iteration: bool,
) -> V3SecondaryCutAcceptance:
    if not has_incumbent:
        return V3SecondaryCutAcceptance(False, "no_incumbent")
    if not pattern_differs_from_primary:
        return V3SecondaryCutAcceptance(False, "same_as_primary_pattern")
    if pattern_in_memory:
        return V3SecondaryCutAcceptance(False, "pattern_in_memory")
    if duplicate_cut:
        return V3SecondaryCutAcceptance(False, "duplicate_cut")
    if _finite("absolute_violation", absolute_violation) <= _nonnegative(
        "violation_tolerance",
        violation_tolerance,
    ):
        return V3SecondaryCutAcceptance(False, "not_violated")
    if certification_active:
        return V3SecondaryCutAcceptance(False, "final_certification")
    if already_added_this_iteration:
        return V3SecondaryCutAcceptance(False, "secondary_already_added")
    return V3SecondaryCutAcceptance(True, None)


def pattern_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    if len(left) != len(right):
        raise ValueError("Patterns must have equal length")
    return sum(int(a != b) for a, b in zip(left, right))
