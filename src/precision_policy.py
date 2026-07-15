from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


PRECISION_POLICIES = {"legacy", "joint_error_budget"}


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


def initialize_precision_state(config: PrecisionPolicyConfig) -> PrecisionPolicyState:
    return PrecisionPolicyState(
        previous_master_gap=config.master_gap_max,
        previous_subproblem_gap=config.subproblem_gap_max,
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
