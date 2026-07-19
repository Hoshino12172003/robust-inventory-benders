from __future__ import annotations

import math

import numpy as np
import pytest

import src.robust_dual_subproblem as robust_module
from src.cut_strengthening import (
    CorePointState,
    core_point_cut_acceptance,
    cut_strengthening_config,
    initialize_core_point_state,
    normalized_core_improvement,
    should_attempt_core_point_strengthening,
    update_core_point_state,
)
from src.instance import InventoryInstance, generate_instance
from src.robust_dual_subproblem import (
    FixedPatternDualLPResult,
    solve_core_point_strengthened_dual_cut,
    solve_fixed_pattern_dual_lp,
    solve_robust_dual_subproblem,
)
from src.scenarios import DemandScenario
from src.subproblem import solve_recourse_subproblem


def _config() -> dict:
    return {
        "seed": 201,
        "instance": {
            "num_warehouses": 2,
            "num_products": 2,
            "num_regions": 2,
            "budget_factor": 0.8,
        },
        "robust": {"gamma_target": 1, "gamma_schedule": [1]},
        "algorithm": {
            "cut_strengthening_policy": "core_point",
            "subproblem_mode": "robust_dual_milp",
        },
    }


def _sample_x(instance: InventoryInstance, scale: float) -> dict[tuple[int, int], float]:
    return {
        (i, j): min(
            instance.inventory_ub[i][j],
            scale * sum(instance.base_demand[r][j] for r in instance.R),
        )
        for i in instance.I
        for j in instance.J
    }


def _failed_lp(status: str) -> FixedPatternDualLPResult:
    return FixedPatternDualLPResult(
        objective=None,
        lambda_values={},
        mu_values={},
        nu_values={},
        demand_values={},
        constant=0.0,
        x_coefficients={},
        runtime=0.01,
        status=status,
        has_solution=False,
        dual_feasible=False,
        num_binary_variables=0,
        is_mip=False,
    )


def test_fixed_pattern_model_is_a_continuous_lp_and_matches_exact_recourse() -> None:
    instance = generate_instance(_config(), seed=201)
    x_values = _sample_x(instance, 0.25)
    pattern = {(r, j): float((r + j) % 2) for r in instance.R for j in instance.J}
    result = solve_fixed_pattern_dual_lp(
        instance,
        x_values,
        pattern,
        time_limit=5.0,
    )
    demand = tuple(
        tuple(result.demand_values[r, j] for j in instance.J)
        for r in instance.R
    )
    scenario = DemandScenario("fixed", tuple(key for key, value in pattern.items() if value), demand)
    recourse = solve_recourse_subproblem(instance, scenario, x_values)
    assert result.status == "optimal"
    assert result.num_binary_variables == 0
    assert result.is_mip is False
    assert result.dual_feasible
    assert result.objective == pytest.approx(recourse.objective, abs=1.0e-5)


def test_core_point_two_stage_cut_preserves_current_value_and_is_dual_feasible() -> None:
    instance = generate_instance(_config(), seed=202)
    current = _sample_x(instance, 0.20)
    core = _sample_x(instance, 0.35)
    original = solve_robust_dual_subproblem(instance, current, gamma=1, mip_gap=0.0)
    result = solve_core_point_strengthened_dual_cut(
        instance,
        current,
        core,
        original,
        stage1_time_limit=5.0,
        stage2_time_limit=5.0,
        remaining_global_time=15.0,
        current_abs_tol=1.0e-7,
        current_rel_tol=1.0e-8,
    )
    assert result.stage1_status == "optimal"
    assert result.stage2_status == "optimal"
    assert result.strengthened_cut is not None
    assert result.dual_feasible
    assert result.auxiliary_bound_used_for_ub is False
    assert result.strengthened_value_at_current >= result.current_value_floor - 1.0e-6


def test_strengthened_fixed_pattern_cut_is_valid_at_random_inventory_points() -> None:
    instance = generate_instance(_config(), seed=203)
    current = _sample_x(instance, 0.22)
    core = _sample_x(instance, 0.38)
    original = solve_robust_dual_subproblem(instance, current, gamma=1, mip_gap=0.0)
    result = solve_core_point_strengthened_dual_cut(
        instance,
        current,
        core,
        original,
        stage1_time_limit=5.0,
        stage2_time_limit=5.0,
        remaining_global_time=15.0,
        current_abs_tol=1.0e-7,
        current_rel_tol=1.0e-8,
    )
    assert result.strengthened_cut is not None
    rng = np.random.default_rng(203)
    for _ in range(3):
        test_x = {
            (i, j): float(rng.uniform(0.0, instance.inventory_ub[i][j]))
            for i in instance.I
            for j in instance.J
        }
        exact = solve_robust_dual_subproblem(instance, test_x, gamma=1, mip_gap=0.0)
        assert result.strengthened_cut.cut_value(test_x) <= exact.objective + 1.0e-5


def test_stage1_nonoptimal_stops_before_stage2(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = generate_instance(_config(), seed=204)
    current = _sample_x(instance, 0.2)
    original = solve_robust_dual_subproblem(instance, current, gamma=1, mip_gap=0.0)
    calls = 0

    def fake_solver(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _failed_lp("time_limit")

    monkeypatch.setattr(robust_module, "solve_fixed_pattern_dual_lp", fake_solver)
    result = solve_core_point_strengthened_dual_cut(
        instance, current, _sample_x(instance, 0.3), original,
        stage1_time_limit=2.0, stage2_time_limit=2.0,
        remaining_global_time=10.0, current_abs_tol=1.0e-7,
        current_rel_tol=1.0e-8,
    )
    assert calls == 1
    assert result.failure_reason == "stage1_not_optimal"
    assert result.strengthened_cut is None


def test_stage2_nonoptimal_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = generate_instance(_config(), seed=205)
    current = _sample_x(instance, 0.2)
    original = solve_robust_dual_subproblem(instance, current, gamma=1, mip_gap=0.0)
    real_solver = robust_module.solve_fixed_pattern_dual_lp
    calls = 0

    def fake_solver(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_solver(*args, **kwargs)
        return _failed_lp("time_limit")

    monkeypatch.setattr(robust_module, "solve_fixed_pattern_dual_lp", fake_solver)
    result = solve_core_point_strengthened_dual_cut(
        instance, current, _sample_x(instance, 0.3), original,
        stage1_time_limit=2.0, stage2_time_limit=2.0,
        remaining_global_time=10.0, current_abs_tol=1.0e-7,
        current_rel_tol=1.0e-8,
    )
    assert calls == 2
    assert result.failure_reason == "stage2_not_optimal"
    assert result.strengthened_cut is None


def test_normalized_improvement_and_acceptance_thresholds() -> None:
    assert normalized_core_improvement(110.0, 100.0) == pytest.approx(10.0 / 110.0)
    accepted = core_point_cut_acceptance(
        stage1_optimal=True,
        stage2_optimal=True,
        dual_feasible=True,
        strengthened_value_at_current=100.0,
        current_value_floor=99.9,
        original_value_at_current=100.0,
        strengthened_value_at_core=110.0,
        original_value_at_core=100.0,
        current_tolerance=0.1,
        minimum_normalized_improvement=1.0e-7,
        duplicate=False,
        original_primary_violated=True,
        certification_active=False,
    )
    assert accepted.accepted
    weak = core_point_cut_acceptance(
        stage1_optimal=True, stage2_optimal=True, dual_feasible=True,
        strengthened_value_at_current=100.0, current_value_floor=99.9,
        original_value_at_current=100.0, strengthened_value_at_core=100.0,
        original_value_at_core=100.0, current_tolerance=0.1,
        minimum_normalized_improvement=1.0e-7, duplicate=False,
        original_primary_violated=True, certification_active=False,
    )
    assert not weak.accepted
    assert weak.fallback_reason == "no_core_point_improvement"


def test_duplicate_strengthened_cut_is_rejected() -> None:
    decision = core_point_cut_acceptance(
        stage1_optimal=True, stage2_optimal=True, dual_feasible=True,
        strengthened_value_at_current=100.0, current_value_floor=99.9,
        original_value_at_current=100.0, strengthened_value_at_core=110.0,
        original_value_at_core=100.0, current_tolerance=0.1,
        minimum_normalized_improvement=1.0e-7, duplicate=True,
        original_primary_violated=True, certification_active=False,
    )
    assert not decision.accepted
    assert decision.fallback_reason == "duplicate_strengthened_cut"


def test_improvement_must_strictly_exceed_frozen_threshold() -> None:
    normalized = normalized_core_improvement(110.0, 100.0)
    decision = core_point_cut_acceptance(
        stage1_optimal=True, stage2_optimal=True, dual_feasible=True,
        strengthened_value_at_current=100.0, current_value_floor=99.9,
        original_value_at_current=100.0, strengthened_value_at_core=110.0,
        original_value_at_core=100.0, current_tolerance=0.1,
        minimum_normalized_improvement=normalized, duplicate=False,
        original_primary_violated=True, certification_active=False,
    )
    assert not decision.accepted
    assert decision.fallback_reason == "improvement_below_threshold"


def test_first_iteration_has_no_core_point_and_uses_previous_state_only() -> None:
    config = cut_strengthening_config(_config()["algorithm"])
    current = {(0, 0): 2.0}
    initial = initialize_core_point_state()
    decision = should_attempt_core_point_strengthening(
        config, initial, current, subproblem_mode="robust_dual_milp",
        primary_has_incumbent=True, primary_absolute_violation=1.0,
        global_gap=0.1, remaining_time=30.0, certification_active=False,
    )
    assert not decision.attempt
    assert decision.skipped_reason == "core_point_unavailable"
    after_first = update_core_point_state(initial, current, 0.5)
    assert after_first.core_x == current
    assert after_first.observations == 1


def test_core_point_update_is_convex_combination() -> None:
    state = CorePointState(core_x={(0, 0): 2.0, (0, 1): 4.0}, observations=1)
    updated = update_core_point_state(
        state,
        {(0, 0): 6.0, (0, 1): 8.0},
        0.5,
    )
    assert updated.core_x == {(0, 0): 4.0, (0, 1): 6.0}
    assert updated.observations == 2


def test_certification_skips_strengthening_and_preserves_core_state() -> None:
    config = cut_strengthening_config(_config()["algorithm"])
    state = CorePointState(core_x={(0, 0): 2.0}, observations=1)
    decision = should_attempt_core_point_strengthening(
        config, state, {(0, 0): 4.0}, subproblem_mode="robust_dual_milp",
        primary_has_incumbent=True, primary_absolute_violation=1.0,
        global_gap=0.1, remaining_time=30.0, certification_active=True,
    )
    assert not decision.attempt
    assert decision.skipped_reason == "final_certification"
    assert update_core_point_state(state, {(0, 0): 4.0}, 0.5, update_state=False) == state


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_core_configuration_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError):
        cut_strengthening_config({"core_point_stage1_time_limit": value})


@pytest.mark.parametrize(
    "field",
    [
        "v3_secondary_max_time_fraction_of_remaining",
        "v3_secondary_max_extra_time_share",
    ],
)
def test_secondary_time_ratios_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError):
        cut_strengthening_config({field: 0.0})
