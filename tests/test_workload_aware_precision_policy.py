from __future__ import annotations

import math

import pytest

from src.precision_policy import (
    WorkloadAwarePrecisionState,
    initialize_workload_aware_state,
    precision_policy_config,
    select_joint_error_budget_precision,
    select_workload_aware_precision,
    update_workload_time_ema,
    workload_aware_precision_config,
)


def _algorithm(policy: str = "workload_aware_joint") -> dict[str, object]:
    return {
        "precision_policy": policy,
        "adaptive_master_precision_enabled": True,
        "adaptive_subproblem_precision_enabled": True,
        "master_gap_min": 0.0001,
        "master_gap_max": 0.02,
        "subproblem_gap_min": 0.0001,
        "subproblem_gap_max": 0.05,
        "master_error_budget_ratio": 0.25,
        "subproblem_error_budget_ratio": 0.50,
        "monotone_precision_tightening": True,
        "fixed_master_mip_gap": 0.02,
        "fixed_subproblem_mip_gap": 0.05,
        "workload_ema_decay": 0.80,
        "workload_total_error_budget_ratio": 0.75,
        "workload_master_weight_min": 1.0 / 3.0,
        "workload_master_weight_max": 2.0 / 3.0,
        "workload_time_epsilon": 1.0e-9,
        "workload_initial_master_weight": 1.0 / 3.0,
        "workload_initial_subproblem_weight": 2.0 / 3.0,
    }


def _configs(policy: str = "workload_aware_joint"):
    algorithm = _algorithm(policy)
    precision = precision_policy_config(
        algorithm,
        fixed_master_gap=0.02,
        fixed_subproblem_gap=0.05,
        legacy_subproblem_gaps=[0.05],
    )
    workload = workload_aware_precision_config(algorithm)
    return precision, workload


def _state(master: float | None, subproblem: float | None):
    precision, _workload = _configs()
    initial = initialize_workload_aware_state(precision)
    return WorkloadAwarePrecisionState(
        precision_state=initial.precision_state,
        master_time_ema=master,
        subproblem_time_ema=subproblem,
        time_observations=1 if master is not None and subproblem is not None else 0,
    )


def test_initial_fallback_exactly_restores_v1_ratios() -> None:
    precision, workload = _configs()
    decision = select_workload_aware_precision(
        precision,
        workload,
        initialize_workload_aware_state(precision),
        upper_bound=100.0,
        lower_bound=90.0,
    )
    assert decision.fallback_used
    assert decision.master_weight_selected == pytest.approx(1.0 / 3.0)
    assert decision.subproblem_weight_selected == pytest.approx(2.0 / 3.0)
    assert decision.master_ratio_selected == pytest.approx(0.25)
    assert decision.subproblem_ratio_selected == pytest.approx(0.50)


@pytest.mark.parametrize("master,subproblem", [(1.0, 9.0), (5.0, 5.0), (9.0, 1.0)])
def test_dynamic_weights_are_bounded_and_ratios_sum_to_total(master: float, subproblem: float) -> None:
    precision, workload = _configs()
    decision = select_workload_aware_precision(
        precision,
        workload,
        _state(master, subproblem),
        upper_bound=100.0,
        lower_bound=99.0,
    )
    assert not decision.fallback_used
    assert 1.0 / 3.0 <= decision.master_weight_selected <= 2.0 / 3.0
    assert decision.master_ratio_selected + decision.subproblem_ratio_selected == pytest.approx(0.75)


def test_master_share_increase_moves_ratios_in_opposite_directions() -> None:
    precision, workload = _configs()
    low = select_workload_aware_precision(
        precision, workload, _state(2.0, 8.0), upper_bound=100.0, lower_bound=99.0
    )
    high = select_workload_aware_precision(
        precision, workload, _state(8.0, 2.0), upper_bound=100.0, lower_bound=99.0
    )
    assert high.master_ratio_selected >= low.master_ratio_selected
    assert high.subproblem_ratio_selected <= low.subproblem_ratio_selected


def test_ema_initialization_and_update_are_exact() -> None:
    precision, workload = _configs()
    state = initialize_workload_aware_state(precision)
    state = update_workload_time_ema(workload, state, master_time=2.0, subproblem_time=6.0)
    assert state.master_time_ema == pytest.approx(2.0)
    assert state.subproblem_time_ema == pytest.approx(6.0)
    state = update_workload_time_ema(workload, state, master_time=6.0, subproblem_time=2.0)
    assert state.master_time_ema == pytest.approx(2.8)
    assert state.subproblem_time_ema == pytest.approx(5.2)
    assert state.time_observations == 2


@pytest.mark.parametrize(
    "master,subproblem,reason",
    [
        (math.nan, 1.0, "nonfinite"),
        (1.0, math.inf, "nonfinite"),
        (-1.0, 1.0, "negative"),
        (0.0, 0.0, "time_sum_too_small"),
    ],
)
def test_invalid_time_history_uses_fallback(master: float, subproblem: float, reason: str) -> None:
    precision, workload = _configs()
    decision = select_workload_aware_precision(
        precision,
        workload,
        _state(master, subproblem),
        upper_bound=100.0,
        lower_bound=99.0,
    )
    assert decision.fallback_used
    assert reason in str(decision.fallback_reason)
    assert decision.master_ratio_selected == pytest.approx(0.25)
    assert decision.subproblem_ratio_selected == pytest.approx(0.50)


def test_candidate_gaps_stay_bounded_and_selected_gaps_only_tighten() -> None:
    precision, workload = _configs()
    state = _state(8.0, 2.0)
    first = select_workload_aware_precision(
        precision, workload, state, upper_bound=100.0, lower_bound=99.0
    )
    second = select_workload_aware_precision(
        precision, workload, first.next_state, upper_bound=100.0, lower_bound=99.9
    )
    third = select_workload_aware_precision(
        precision, workload, second.next_state, upper_bound=100.0, lower_bound=95.0
    )
    for decision in (first, second, third):
        assert 0.0001 <= decision.precision_decision.master_candidate_gap <= 0.02
        assert 0.0001 <= decision.precision_decision.subproblem_candidate_gap <= 0.05
    assert second.precision_decision.master_selected_gap <= first.precision_decision.master_selected_gap
    assert third.precision_decision.master_selected_gap <= second.precision_decision.master_selected_gap
    assert second.precision_decision.subproblem_selected_gap <= first.precision_decision.subproblem_selected_gap
    assert third.precision_decision.subproblem_selected_gap <= second.precision_decision.subproblem_selected_gap


def test_certification_style_no_update_preserves_precision_and_ema_state() -> None:
    precision, workload = _configs()
    state = update_workload_time_ema(
        workload,
        initialize_workload_aware_state(precision),
        master_time=4.0,
        subproblem_time=2.0,
    )
    decision = select_workload_aware_precision(
        precision,
        workload,
        state,
        upper_bound=100.0,
        lower_bound=99.0,
        update_state=False,
    )
    after = update_workload_time_ema(
        workload,
        decision.next_state,
        master_time=100.0,
        subproblem_time=100.0,
        update_state=False,
    )
    assert decision.next_state == state
    assert after == state


def test_v1_joint_error_budget_decision_is_unchanged() -> None:
    precision, _workload = _configs("joint_error_budget")
    state = initialize_workload_aware_state(precision).precision_state
    decision = select_joint_error_budget_precision(
        precision,
        state,
        upper_bound=100.0,
        lower_bound=99.0,
    )
    assert decision.master_candidate_gap == pytest.approx(0.0025)
    assert decision.subproblem_candidate_gap == pytest.approx(0.005)
