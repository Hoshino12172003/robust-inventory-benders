from __future__ import annotations

import pytest

from src.cut_strengthening import (
    V3SecondaryPatternMemory,
    cut_strengthening_config,
    pattern_distance,
    recent_relative_lb_improvement,
    should_attempt_v3_secondary_cut,
    update_secondary_pattern_memory,
    v3_secondary_cut_acceptance,
)
from src.instance import generate_instance
from src.robust_dual_subproblem import (
    discretize_robust_pattern,
    solve_robust_dual_subproblem,
)


def _algorithm() -> dict[str, object]:
    return {
        "cut_strengthening_policy": "stall_secondary",
        "v3_secondary_lb_window": 5,
        "v3_secondary_stall_threshold": 1.0e-4,
        "v3_secondary_cooldown_iterations": 10,
        "v3_secondary_min_global_gap": 1.0e-3,
        "v3_secondary_min_remaining_time": 30.0,
        "v3_secondary_max_time_per_attempt": 10.0,
        "v3_secondary_max_time_fraction_of_remaining": 0.05,
        "v3_secondary_max_extra_time_share": 0.10,
        "v3_secondary_pattern_memory": 10,
    }


def _decision(**overrides: object):
    values = {
        "subproblem_mode": "robust_dual_milp",
        "active_gamma": 2,
        "target_gamma": 2,
        "certification_active": False,
        "primary_has_incumbent": True,
        "primary_pattern_valid": True,
        "global_gap": 0.1,
        "lower_bound_history": [100.0, 100.0, 100.0, 100.0, 100.0],
        "current_iteration": 6,
        "last_attempt_iteration": None,
        "remaining_time": 100.0,
        "extra_runtime": 1.0,
        "elapsed_time": 100.0,
    }
    values.update(overrides)
    return should_attempt_v3_secondary_cut(cut_strengthening_config(_algorithm()), **values)


def test_no_good_excludes_primary_pattern_and_returns_different_pattern() -> None:
    config = {
        "seed": 211,
        "instance": {"num_warehouses": 2, "num_products": 2, "num_regions": 2, "budget_factor": 0.8},
    }
    instance = generate_instance(config, seed=211)
    x_values = {(i, j): 0.0 for i in instance.I for j in instance.J}
    primary = solve_robust_dual_subproblem(instance, x_values, gamma=1, mip_gap=0.0)
    pattern = discretize_robust_pattern(instance, primary.z_values)
    assert pattern is not None
    secondary = solve_robust_dual_subproblem(
        instance,
        x_values,
        gamma=1,
        mip_gap=0.0,
        excluded_patterns=[pattern],
    )
    assert secondary.has_incumbent
    assert tuple(round(primary.z_values[key]) for key in sorted(primary.z_values)) != tuple(
        round(secondary.z_values[key]) for key in sorted(secondary.z_values)
    )


def test_pattern_memory_is_unique_and_bounded() -> None:
    memory = V3SecondaryPatternMemory()
    for index in range(15):
        pattern = tuple((index >> bit) & 1 for bit in range(4))
        memory = update_secondary_pattern_memory(memory, pattern, maximum_size=10)
    assert len(memory.patterns) == 10
    last = memory.patterns[-1]
    memory = update_secondary_pattern_memory(memory, last, maximum_size=10)
    assert len(memory.patterns) == 10
    assert memory.patterns[-1] == last


def test_pattern_distance_is_hamming_distance() -> None:
    assert pattern_distance((1, 0, 1, 0), (0, 0, 1, 1)) == 2


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"has_incumbent": False}, "no_incumbent"),
        ({"pattern_differs_from_primary": False}, "same_as_primary_pattern"),
        ({"pattern_in_memory": True}, "pattern_in_memory"),
        ({"duplicate_cut": True}, "duplicate_cut"),
        ({"absolute_violation": 1.0e-7}, "not_violated"),
        ({"certification_active": True}, "final_certification"),
        ({"already_added_this_iteration": True}, "secondary_already_added"),
    ],
)
def test_secondary_cut_acceptance_guards(
    overrides: dict[str, object],
    reason: str,
) -> None:
    values = {
        "has_incumbent": True,
        "pattern_differs_from_primary": True,
        "pattern_in_memory": False,
        "duplicate_cut": False,
        "absolute_violation": 1.0,
        "violation_tolerance": 1.0e-6,
        "certification_active": False,
        "already_added_this_iteration": False,
    }
    values.update(overrides)
    decision = v3_secondary_cut_acceptance(**values)
    assert not decision.accepted
    assert decision.skip_reason == reason


def test_violated_distinct_nonduplicate_secondary_cut_is_accepted() -> None:
    decision = v3_secondary_cut_acceptance(
        has_incumbent=True,
        pattern_differs_from_primary=True,
        pattern_in_memory=False,
        duplicate_cut=False,
        absolute_violation=1.0,
        violation_tolerance=1.0e-6,
        certification_active=False,
        already_added_this_iteration=False,
    )
    assert decision.accepted
    assert decision.skip_reason is None


def test_stall_window_requires_complete_history() -> None:
    assert recent_relative_lb_improvement([1.0, 1.0, 1.0, 1.0], 5) is None
    decision = _decision(lower_bound_history=[1.0, 1.0, 1.0, 1.0])
    assert not decision.attempt
    assert decision.skipped_reason == "insufficient_lb_history"


def test_progressing_lower_bound_does_not_trigger() -> None:
    decision = _decision(lower_bound_history=[100.0, 101.0, 102.0, 103.0, 104.0])
    assert not decision.attempt
    assert decision.skipped_reason == "lb_progressing"


def test_stall_triggers_with_capped_time_limit() -> None:
    decision = _decision(remaining_time=100.0)
    assert decision.attempt
    assert decision.trigger_reason == "lb_stall"
    assert decision.time_limit == pytest.approx(5.0)


def test_cooldown_is_enforced() -> None:
    decision = _decision(last_attempt_iteration=5, current_iteration=10)
    assert not decision.attempt
    assert decision.skipped_reason == "cooldown"
    assert decision.cooldown_remaining == 5


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"global_gap": 1.0e-4}, "global_gap_too_small"),
        ({"remaining_time": 20.0}, "insufficient_remaining_time"),
        ({"extra_runtime": 11.0, "elapsed_time": 100.0}, "extra_time_share_limit"),
        ({"certification_active": True}, "final_certification"),
        ({"current_iteration": 1}, "first_iteration"),
        ({"primary_has_incumbent": False}, "primary_without_incumbent"),
        ({"primary_pattern_valid": False}, "invalid_primary_pattern"),
        ({"active_gamma": 1}, "active_gamma_not_target"),
    ],
)
def test_secondary_trigger_guards(overrides: dict[str, object], reason: str) -> None:
    decision = _decision(**overrides)
    assert not decision.attempt
    assert decision.skipped_reason == reason


def test_secondary_time_limit_below_one_second_is_skipped() -> None:
    algorithm = _algorithm()
    algorithm["v3_secondary_min_remaining_time"] = 0.0
    decision = should_attempt_v3_secondary_cut(
        cut_strengthening_config(algorithm),
        subproblem_mode="robust_dual_milp",
        active_gamma=2,
        target_gamma=2,
        certification_active=False,
        primary_has_incumbent=True,
        primary_pattern_valid=True,
        global_gap=0.1,
        lower_bound_history=[1.0] * 5,
        current_iteration=6,
        last_attempt_iteration=None,
        remaining_time=10.0,
        extra_runtime=0.0,
        elapsed_time=10.0,
    )
    assert not decision.attempt
    assert decision.skipped_reason == "secondary_time_limit_below_one_second"


@pytest.mark.parametrize("field", [
    "v3_secondary_stall_threshold",
    "v3_secondary_min_global_gap",
    "v3_secondary_min_remaining_time",
    "v3_secondary_max_time_per_attempt",
    "v3_secondary_max_time_fraction_of_remaining",
    "v3_secondary_max_extra_time_share",
])
def test_secondary_config_rejects_nonfinite_values(field: str) -> None:
    values = _algorithm()
    values[field] = float("nan")
    with pytest.raises(ValueError):
        cut_strengthening_config(values)
