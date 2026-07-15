from __future__ import annotations

from copy import deepcopy

import pytest

import src.benders as benders_module
from src.benders import (
    FinalCertificationState,
    _settings,
    adaptive_secondary_cut_threshold,
    certification_mip_gap,
    calculate_global_gap,
    calculate_cut_violations,
    generate_additional_robust_cuts,
    generate_gated_additional_robust_cuts,
    marginal_normalized_violation,
    primary_cut_decision,
    recent_relative_lb_improvement,
    relative_cut_decision,
    secondary_generation_decision,
    select_subproblem_mip_gap,
    should_terminate_benders,
    solve_benders,
    target_upper_cost,
    update_final_certification,
)
from src.instance import generate_instance
from src.precision_policy import (
    PrecisionPolicyState,
    error_budget_candidate,
    initialize_precision_state,
    precision_policy_config,
    select_joint_error_budget_precision,
    valid_global_gap_for_precision,
)
from src.robust_dual_subproblem import solve_robust_dual_subproblem
from src.robust_dual_subproblem import RobustDualSubproblemResult
from src.experiment_suite import _summary_rows, _time_to_gap_metrics
from src.results import SolveResult


def tiny_config() -> dict:
    return {
        "seed": 81,
        "instance": {
            "num_warehouses": 2,
            "num_products": 2,
            "num_regions": 2,
            "budget_factor": 0.7,
        },
        "robust": {"gamma_target": 1, "gamma_schedule": [0, 1], "max_scenarios": 100},
        "algorithm": {
            "subproblem_mode": "robust_dual_milp",
            "cut_selection_enabled": True,
            "cut_selection_mode": "relative",
            "relative_cut_threshold": 1e-4,
            "cut_violation_tol": 1e-8,
            "final_exact_gap": 1e-2,
            "cut_stall_patience": 2,
            "adaptive_subproblem_gap_enabled": True,
            "subproblem_gap_schedule": [
                {"global_gap_above": 0.10, "mip_gap": 0.05},
                {"global_gap_above": 0.01, "mip_gap": 0.005},
                {"global_gap_above": 0.00, "mip_gap": 0.0001},
            ],
            "max_cuts_per_iteration": 1,
        },
        "benders": {
            "max_iterations": 20,
            "tol": 1e-3,
            "initial_mip_gap": 0.05,
            "final_mip_gap": 1e-5,
            "time_limit": 30,
            "output_flag": False,
        },
    }


def fake_robust_cut(
    *,
    objective: float | None = 90.0,
    objective_bound: float | None = 100.0,
    status: str = "optimal",
    runtime: float = 1.0,
    pattern: tuple[int, int] = (1, 0),
    has_incumbent: bool = True,
) -> RobustDualSubproblemResult:
    z_values = {(0, 0): float(pattern[0]), (1, 0): float(pattern[1])} if has_incumbent else {}
    return RobustDualSubproblemResult(
        objective=objective,
        z_values=z_values,
        lambda_values={},
        mu_values={},
        nu_values={},
        demand_values={},
        constant=objective or 0.0,
        x_coefficients={(0, 0): 0.0} if has_incumbent else {},
        runtime=runtime,
        status=status,
        objective_bound=objective_bound,
        mip_gap=0.1 if objective != objective_bound else 0.0,
        has_incumbent=has_incumbent,
        requested_mip_gap=0.1,
    )


def test_normalized_violation_is_scale_independent() -> None:
    absolute, normalized = calculate_cut_violations(120.0, 100.0)
    scaled_absolute, scaled_normalized = calculate_cut_violations(120000.0, 100000.0)
    assert absolute == pytest.approx(20.0)
    assert scaled_absolute == pytest.approx(20000.0)
    assert normalized == pytest.approx(scaled_normalized)


def test_relative_cut_threshold_and_safety_decisions() -> None:
    assert relative_cut_decision(2.0, 2e-3, 1e-3, 1e-8, 0, 1, 0.5, 0.01)[0]
    early = relative_cut_decision(2.0, 2e-4, 1e-3, 1e-8, 0, 1, 0.5, 0.01)
    assert early == (False, "low_relative_violation", None)
    final = relative_cut_decision(2.0, 2e-4, 1e-3, 1e-8, 1, 1, 0.005, 0.01)
    assert final == (True, None, "final_exact_phase")
    assert relative_cut_decision(1e-10, 1e-10, 0.0, 1e-8, 1, 1, 0.0, 0.01)[0] is False


def test_primary_cut_is_never_screened_by_relative_threshold() -> None:
    assert primary_cut_decision(0.25, 1e-8) == (True, None)
    assert primary_cut_decision(1e-10, 1e-8) == (False, "not_violated")
    assert primary_cut_decision(0.25, 1e-8, duplicate=True) == (
        False,
        "duplicate_cut",
    )


def test_secondary_rhs_below_or_equal_to_primary_has_no_marginal_value() -> None:
    assert marginal_normalized_violation(90.0, 100.0, 10.0) == 0.0
    assert marginal_normalized_violation(100.0, 100.0, 10.0) == 0.0


def test_secondary_rhs_above_primary_has_scaled_marginal_value() -> None:
    marginal = marginal_normalized_violation(101.0, 100.0, 10.0)
    assert marginal == pytest.approx(1.0 / 101.0)


def test_marginal_value_uses_theta_when_primary_is_not_violated() -> None:
    marginal = marginal_normalized_violation(102.0, 90.0, 100.0)
    assert marginal == pytest.approx(2.0 / 102.0)


def test_marginal_value_is_scale_independent() -> None:
    original = marginal_normalized_violation(110.0, 100.0, 10.0)
    scaled = marginal_normalized_violation(1100.0, 1000.0, 100.0)
    assert scaled == pytest.approx(original)


def test_secondary_cut_can_be_screened_and_final_phase_restores_it() -> None:
    marginal = marginal_normalized_violation(101.0, 100.0, 10.0)
    screened = relative_cut_decision(1.0, marginal, 0.20, 1e-8, 2, 2, 0.05, 0.01)
    assert screened == (False, "low_relative_violation", None)
    restored = relative_cut_decision(1.0, marginal, 0.20, 1e-8, 2, 2, 0.005, 0.01)
    assert restored == (True, None, "final_exact_phase")


def test_adaptive_secondary_threshold_responds_to_master_pressure() -> None:
    warmup = adaptive_secondary_cut_threshold(
        0.1, 10, 50, 0.10, 0.35, 0.05, 0.5, 0.20, 0.01, False
    )
    pressured = adaptive_secondary_cut_threshold(
        0.1, 100, 50, 0.50, 0.35, 0.75, 0.5, 0.20, 0.01, False
    )
    final = adaptive_secondary_cut_threshold(
        0.1, 100, 50, 0.50, 0.35, 0.75, 0.5, 0.005, 0.01, True
    )
    assert warmup == 0.0
    assert pressured > 0.1
    assert final == 0.0


def test_secondary_generation_gate_closed_does_not_call_solver() -> None:
    primary = fake_robust_cut(pattern=(1, 0))
    decision = secondary_generation_decision(
        enabled=True,
        max_cuts_per_iteration=2,
        primary_has_incumbent=True,
        lower_bound_history=[100.0, 100.0],
        rolling_window=5,
        stall_threshold=1e-4,
        current_iteration=3,
        last_secondary_solve_iteration=None,
        cooldown_iterations=5,
        cumulative_subproblem_time_share=0.2,
        max_subproblem_time_share=0.75,
        remaining_time=30.0,
        available_budget=10.0,
        min_remaining_time=5.0,
        min_solve_budget=2.0,
    )
    calls = 0

    def unexpected_solve(
        _excluded: list[dict[tuple[int, int], int]],
        _remaining: float,
    ) -> RobustDualSubproblemResult:
        nonlocal calls
        calls += 1
        return fake_robust_cut(pattern=(0, 1))

    batch = generate_gated_additional_robust_cuts(
        decision,
        primary,
        2,
        10.0,
        unexpected_solve,
    )

    assert decision.attempt is False
    assert decision.skipped_reason == "insufficient_lb_history"
    assert calls == 0
    assert batch.cuts == []


def test_lb_stall_opens_secondary_generation_gate() -> None:
    history = [100.0, 100.002, 100.003, 100.004]
    assert recent_relative_lb_improvement(history, 3) == pytest.approx(0.004 / 100.004)

    decision = secondary_generation_decision(
        enabled=True,
        max_cuts_per_iteration=2,
        primary_has_incumbent=True,
        lower_bound_history=history,
        rolling_window=3,
        stall_threshold=1e-4,
        current_iteration=5,
        last_secondary_solve_iteration=None,
        cooldown_iterations=2,
        cumulative_subproblem_time_share=0.2,
        max_subproblem_time_share=0.75,
        remaining_time=30.0,
        available_budget=10.0,
        min_remaining_time=5.0,
        min_solve_budget=2.0,
    )

    assert decision.attempt is True
    assert decision.trigger_reason == "lb_stall"


def test_subproblem_time_share_distinguishes_conservative_and_permissive_gates() -> None:
    common = {
        "enabled": True,
        "max_cuts_per_iteration": 2,
        "primary_has_incumbent": True,
        "lower_bound_history": [100.0, 100.0, 100.0, 100.0],
        "rolling_window": 3,
        "stall_threshold": 1e-4,
        "current_iteration": 10,
        "last_secondary_solve_iteration": None,
        "cooldown_iterations": 5,
        "cumulative_subproblem_time_share": 0.80,
        "remaining_time": 30.0,
        "available_budget": 10.0,
        "min_remaining_time": 5.0,
        "min_solve_budget": 2.0,
    }

    conservative = secondary_generation_decision(
        **common,
        max_subproblem_time_share=0.75,
    )
    permissive = secondary_generation_decision(
        **common,
        max_subproblem_time_share=0.95,
    )

    assert conservative.attempt is False
    assert conservative.skipped_reason == "subproblem_time_share"
    assert permissive.attempt is True
    assert permissive.trigger_reason == "lb_stall"


def test_cooldown_temporarily_closes_secondary_generation_gate() -> None:
    decision = secondary_generation_decision(
        enabled=True,
        max_cuts_per_iteration=2,
        primary_has_incumbent=True,
        lower_bound_history=[100.0, 100.0, 100.0, 100.0],
        rolling_window=3,
        stall_threshold=1e-4,
        current_iteration=7,
        last_secondary_solve_iteration=6,
        cooldown_iterations=3,
        cumulative_subproblem_time_share=0.2,
        max_subproblem_time_share=0.75,
        remaining_time=30.0,
        available_budget=10.0,
        min_remaining_time=5.0,
        min_solve_budget=2.0,
    )

    assert decision.attempt is False
    assert decision.skipped_reason == "cooldown"
    assert decision.cooldown_remaining == 3


def test_insufficient_remaining_time_closes_secondary_generation_gate() -> None:
    decision = secondary_generation_decision(
        enabled=True,
        max_cuts_per_iteration=2,
        primary_has_incumbent=True,
        lower_bound_history=[100.0, 100.0, 100.0, 100.0],
        rolling_window=3,
        stall_threshold=1e-4,
        current_iteration=10,
        last_secondary_solve_iteration=None,
        cooldown_iterations=0,
        cumulative_subproblem_time_share=0.2,
        max_subproblem_time_share=0.75,
        remaining_time=4.0,
        available_budget=4.0,
        min_remaining_time=5.0,
        min_solve_budget=2.0,
    )

    assert decision.attempt is False
    assert decision.skipped_reason == "insufficient_remaining_time"


def test_single_cut_mode_never_attempts_secondary_generation() -> None:
    decision = secondary_generation_decision(
        enabled=True,
        max_cuts_per_iteration=1,
        primary_has_incumbent=True,
        lower_bound_history=[100.0, 100.0, 100.0, 100.0],
        rolling_window=3,
        stall_threshold=1e-4,
        current_iteration=10,
        last_secondary_solve_iteration=None,
        cooldown_iterations=0,
        cumulative_subproblem_time_share=0.0,
        max_subproblem_time_share=1.0,
        remaining_time=30.0,
        available_budget=30.0,
        min_remaining_time=1.0,
        min_solve_budget=1.0,
    )

    assert decision.attempt is False
    assert decision.skipped_reason == "single_cut_mode"


def test_adaptive_subproblem_gap_tightens() -> None:
    schedule = [
        {"global_gap_above": 0.10, "mip_gap": 0.05},
        {"global_gap_above": 0.05, "mip_gap": 0.02},
        {"global_gap_above": 0.01, "mip_gap": 0.005},
        {"global_gap_above": 0.00, "mip_gap": 0.0001},
    ]
    assert select_subproblem_mip_gap(None, False, schedule, 0.01) == pytest.approx(0.05)
    assert select_subproblem_mip_gap(0.07, True, schedule, 0.01) == pytest.approx(0.02)
    assert select_subproblem_mip_gap(0.005, True, schedule, 0.01) == pytest.approx(0.0001)


def joint_precision_config(**overrides: object):
    values = {
        "precision_policy": "joint_error_budget",
        "adaptive_master_precision_enabled": True,
        "adaptive_subproblem_precision_enabled": True,
        "master_gap_max": 0.02,
        "master_gap_min": 0.0001,
        "subproblem_gap_max": 0.05,
        "subproblem_gap_min": 0.0001,
        "fixed_subproblem_mip_gap": 0.01,
        "master_error_budget_ratio": 0.5,
        "subproblem_error_budget_ratio": 1.0,
        "monotone_precision_tightening": True,
    }
    values.update(overrides)
    return precision_policy_config(
        values,
        fixed_master_gap=0.01,
        fixed_subproblem_gap=0.01,
        legacy_subproblem_gaps=[0.05, 0.0001],
    )


def test_joint_error_budget_candidate_formulas_and_clipping() -> None:
    config = joint_precision_config()
    decision = select_joint_error_budget_precision(
        config,
        initialize_precision_state(config),
        upper_bound=100.0,
        lower_bound=98.0,
    )
    assert decision.valid_global_gap_for_precision == pytest.approx(0.02)
    assert decision.master_candidate_gap == pytest.approx(0.01)
    assert decision.subproblem_candidate_gap == pytest.approx(0.02)
    assert error_budget_candidate(1.0, 0.0001, 0.02, 2.0) == pytest.approx(0.02)
    assert error_budget_candidate(1.0e-8, 0.0001, 0.02, 0.5) == pytest.approx(
        0.0001
    )


def test_joint_error_budget_precision_is_monotone_for_both_solvers() -> None:
    config = joint_precision_config()
    state = initialize_precision_state(config)
    master_gaps = []
    subproblem_gaps = []
    for lower_bound in (90.0, 99.0, 95.0, 99.9):
        decision = select_joint_error_budget_precision(
            config,
            state,
            upper_bound=100.0,
            lower_bound=lower_bound,
        )
        state = decision.next_state
        master_gaps.append(decision.master_selected_gap)
        subproblem_gaps.append(decision.subproblem_selected_gap)
    assert master_gaps == sorted(master_gaps, reverse=True)
    assert subproblem_gaps == sorted(subproblem_gaps, reverse=True)


@pytest.mark.parametrize(
    ("master_enabled", "subproblem_enabled", "expected_master", "expected_subproblem"),
    [
        (True, False, 0.01, 0.01),
        (False, True, 0.01, 0.02),
        (True, True, 0.01, 0.02),
        (False, False, 0.01, 0.01),
    ],
)
def test_master_and_subproblem_adaptation_are_independently_switchable(
    master_enabled: bool,
    subproblem_enabled: bool,
    expected_master: float,
    expected_subproblem: float,
) -> None:
    config = joint_precision_config(
        adaptive_master_precision_enabled=master_enabled,
        adaptive_subproblem_precision_enabled=subproblem_enabled,
    )
    initial = initialize_precision_state(config)
    decision = select_joint_error_budget_precision(
        config,
        initial,
        upper_bound=100.0,
        lower_bound=98.0,
    )
    assert decision.master_selected_gap == pytest.approx(expected_master)
    assert decision.subproblem_selected_gap == pytest.approx(expected_subproblem)
    assert decision.next_state.previous_master_gap == pytest.approx(
        expected_master if master_enabled else initial.previous_master_gap
    )
    assert decision.next_state.previous_subproblem_gap == pytest.approx(
        expected_subproblem if subproblem_enabled else initial.previous_subproblem_gap
    )


def test_precision_gap_fallback_is_logged_but_never_certifies() -> None:
    gap, fallback = valid_global_gap_for_precision(None, 10.0)
    assert gap == 1.0
    assert fallback is True
    assert should_terminate_benders(2, 2, False, gap, 1.0e-4) is False


def test_certification_freezes_pre_certification_precision_state() -> None:
    config = joint_precision_config()
    pre_certification = select_joint_error_budget_precision(
        config,
        initialize_precision_state(config),
        upper_bound=100.0,
        lower_bound=98.0,
    ).next_state
    during_certification = select_joint_error_budget_precision(
        config,
        pre_certification,
        upper_bound=100.0,
        lower_bound=99.9,
        update_state=False,
    )
    assert during_certification.next_state == pre_certification
    resumed = select_joint_error_budget_precision(
        config,
        during_certification.next_state,
        upper_bound=100.0,
        lower_bound=95.0,
    )
    assert resumed.master_previous_gap == pytest.approx(
        pre_certification.previous_master_gap
    )
    assert resumed.subproblem_previous_gap == pytest.approx(
        pre_certification.previous_subproblem_gap
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("precision_policy", "unknown", "precision_policy must be one of"),
        ("master_gap_max", float("inf"), "master_gap_max must be finite"),
        ("master_gap_min", -0.1, "master_gap_min must be finite"),
        ("subproblem_gap_max", float("nan"), "subproblem_gap_max must be finite"),
        ("subproblem_gap_min", -0.1, "subproblem_gap_min must be finite"),
        (
            "fixed_master_mip_gap",
            -0.1,
            "fixed_master_mip_gap must be finite",
        ),
        (
            "master_error_budget_ratio",
            -0.1,
            "master_error_budget_ratio must be finite",
        ),
        (
            "subproblem_error_budget_ratio",
            float("inf"),
            "subproblem_error_budget_ratio must be finite",
        ),
    ],
)
def test_precision_policy_rejects_invalid_configuration(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        joint_precision_config(**{field: value})


def test_explicit_fixed_master_gap_overrides_initial_gap_fallback() -> None:
    explicit = joint_precision_config(fixed_master_mip_gap=0.007)
    fallback = joint_precision_config()

    assert explicit.fixed_master_gap == pytest.approx(0.007)
    assert fallback.fixed_master_gap == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("minimum_field", "maximum_field"),
    [
        ("master_gap_min", "master_gap_max"),
        ("subproblem_gap_min", "subproblem_gap_max"),
    ],
)
def test_precision_policy_rejects_minimum_above_maximum(
    minimum_field: str,
    maximum_field: str,
) -> None:
    with pytest.raises(ValueError, match="must be less than or equal"):
        joint_precision_config(**{minimum_field: 0.02, maximum_field: 0.01})


def _advance_certification(
    state: FinalCertificationState,
    iteration: int,
    *,
    active_gamma: int = 2,
    valid_ub: bool = True,
    gap: float = 0.1,
    useful_cut: bool = False,
):
    return update_final_certification(
        state,
        enabled=True,
        iteration=iteration,
        active_gamma=active_gamma,
        gamma_target=2,
        valid_ub=valid_ub,
        global_gap=gap,
        tol=1.0e-4,
        useful_primary_cut_added=useful_cut,
        no_cut_patience=2,
    )


def test_final_certification_triggers_after_target_no_cut_patience() -> None:
    first = _advance_certification(FinalCertificationState(), 1)
    assert first.state.active is False
    assert first.state.consecutive_no_useful_primary_cuts == 1

    second = _advance_certification(first.state, 2)
    assert second.triggered_this_iteration is True
    assert second.state.active is True
    assert second.state.triggered is True
    assert second.state.trigger_iteration == 2
    assert second.state.count == 1


@pytest.mark.parametrize(
    ("active_gamma", "valid_ub"),
    [(1, True), (2, False)],
)
def test_invalid_ub_or_non_target_gamma_cannot_trigger_certification(
    active_gamma: int,
    valid_ub: bool,
) -> None:
    state = FinalCertificationState()
    for iteration in (1, 2, 3):
        transition = _advance_certification(
            state,
            iteration,
            active_gamma=active_gamma,
            valid_ub=valid_ub,
        )
        state = transition.state
    assert state.active is False
    assert state.triggered is False
    assert state.count == 0


def test_certification_forces_master_and_subproblem_gaps_to_zero() -> None:
    assert certification_mip_gap(True, 0.02) == 0.0
    assert certification_mip_gap(True, 0.0001) == 0.0
    assert certification_mip_gap(False, 0.02) == pytest.approx(0.02)


def test_certification_suppresses_optional_secondary_solves() -> None:
    decision = secondary_generation_decision(
        enabled=False,
        max_cuts_per_iteration=2,
        primary_has_incumbent=True,
        lower_bound_history=[100.0, 100.0, 100.0],
        rolling_window=2,
        stall_threshold=1e-4,
        current_iteration=3,
        last_secondary_solve_iteration=None,
        cooldown_iterations=0,
        cumulative_subproblem_time_share=0.1,
        max_subproblem_time_share=0.95,
        remaining_time=30.0,
        available_budget=10.0,
        min_remaining_time=5.0,
        min_solve_budget=2.0,
        certification_active=True,
    )
    calls = 0

    def unexpected_solve(
        _excluded: list[dict[tuple[int, int], int]],
        _remaining: float,
    ) -> RobustDualSubproblemResult:
        nonlocal calls
        calls += 1
        return fake_robust_cut(pattern=(0, 1))

    batch = generate_gated_additional_robust_cuts(
        decision,
        fake_robust_cut(pattern=(1, 0)),
        2,
        10.0,
        unexpected_solve,
    )
    assert decision.attempt is False
    assert decision.skipped_reason == "final_certification"
    assert batch.cuts == []
    assert calls == 0


def test_useful_primary_cut_exits_certification_and_allows_later_retrigger() -> None:
    state = _advance_certification(FinalCertificationState(), 1).state
    state = _advance_certification(state, 2).state

    active = _advance_certification(state, 3)
    assert active.state.active is True
    assert active.triggered_this_iteration is False
    assert active.state.count == 1

    exited = _advance_certification(active.state, 4, useful_cut=True)
    assert exited.state.active is False
    assert exited.state.consecutive_no_useful_primary_cuts == 0
    assert exited.state.exit_reason == "useful_primary_cut_added"

    waiting = _advance_certification(exited.state, 5)
    retriggered = _advance_certification(waiting.state, 6)
    assert retriggered.triggered_this_iteration is True
    assert retriggered.state.active is True
    assert retriggered.state.count == 2


def test_no_cut_never_substitutes_for_valid_benders_termination() -> None:
    state = _advance_certification(FinalCertificationState(), 1).state
    state = _advance_certification(state, 2).state
    assert state.active is True
    assert should_terminate_benders(2, 2, True, 0.1, 1.0e-4) is False
    assert should_terminate_benders(2, 2, False, 0.0, 1.0e-4) is False
    assert should_terminate_benders(1, 2, True, 0.0, 1.0e-4) is False
    assert should_terminate_benders(2, 2, True, 1.0e-4, 1.0e-4) is True


def test_solver_persists_certification_precision_and_suppresses_secondary() -> None:
    config = tiny_config()
    config["robust"]["gamma_schedule"] = [1]
    config["algorithm"].update(
        {
            "cut_selection_enabled": False,
            "cut_violation_tol": 1.0e12,
            "adaptive_secondary_generation_enabled": False,
            "max_cuts_per_iteration": 2,
            "final_certification_enabled": True,
            "final_certification_no_cut_patience": 2,
        }
    )
    config["benders"]["max_iterations"] = 3
    config["benders"]["tol"] = 0.0
    instance = generate_instance(config, seed=89)

    result = solve_benders(config, instance, "adaptive_gap_gamma_benders")

    assert result.status == "iteration_limit"
    assert len(result.iteration_log) == 3
    assert result.iteration_log[1][
        "final_certification_triggered_this_iteration"
    ] is True
    certified = result.iteration_log[2]
    assert certified["final_certification_active"] is True
    assert certified["requested_master_mip_gap"] == 0.0
    assert certified["subproblem_requested_mip_gap"] == 0.0
    assert certified["secondary_solve_attempted"] is False
    assert certified["secondary_solve_skipped_reason"] == "final_certification"
    assert certified["secondary_solve_disabled_by_certification"] is True
    assert result.metadata["final_certification_triggered"] is True
    assert result.metadata["final_certification_count"] == 1
    assert result.metadata["final_certification_iterations"] == 1
    assert result.metadata["final_certification_exit_reason"] == "iteration_limit"


def test_new_features_are_backward_compatible_by_default() -> None:
    config = tiny_config()
    config["algorithm"] = {"subproblem_mode": "robust_dual_milp"}
    settings = _settings(config, "standard_benders")
    assert settings.cut_selection_mode == "absolute"
    assert settings.adaptive_subproblem_gap_enabled is False
    assert settings.max_cuts_per_iteration == 1
    assert settings.precision_config.precision_policy == "legacy"
    assert settings.precision_config.adaptive_master_precision_enabled is False
    assert settings.precision_config.adaptive_subproblem_precision_enabled is False
    assert settings.final_certification_enabled is False
    assert settings.final_certification_no_cut_patience == 2


def test_positive_mip_gap_target_ub_always_uses_objective_bound() -> None:
    target = fake_robust_cut(objective=90.0, objective_bound=100.0, status="optimal")
    robust_cost, valid_ub, uses_bound = target_upper_cost("robust_dual_milp", target)
    assert robust_cost == pytest.approx(100.0)
    assert valid_ub is True
    assert uses_bound is True
    candidate_upper = 10.0 + robust_cost
    assert calculate_global_gap(candidate_upper, 100.0) == pytest.approx(10.0 / 110.0)
    assert calculate_global_gap(candidate_upper, 100.0) > 0.0


def test_missing_target_objective_bound_is_not_a_valid_ub() -> None:
    target = fake_robust_cut(objective=90.0, objective_bound=None, status="time_limit")
    assert target_upper_cost("robust_dual_milp", target) == (None, False, False)


def test_solver_does_not_falsely_converge_when_optimal_status_has_bound_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    config = tiny_config()
    config["robust"]["gamma_schedule"] = [1]
    config["algorithm"]["adaptive_subproblem_gap_enabled"] = False
    config["algorithm"]["max_cuts_per_iteration"] = 1
    config["benders"]["max_iterations"] = 1
    instance = generate_instance(config, seed=86)

    def fake_subproblem(*_args: object, **_kwargs: object) -> RobustDualSubproblemResult:
        return RobustDualSubproblemResult(
            objective=100.0,
            z_values={(r, j): 0.0 for r in instance.R for j in instance.J},
            lambda_values={},
            mu_values={},
            nu_values={},
            demand_values={},
            constant=100.0,
            x_coefficients={(i, j): 0.0 for i in instance.I for j in instance.J},
            runtime=0.01,
            status="optimal",
            objective_bound=200.0,
            mip_gap=0.5,
            has_incumbent=True,
            requested_mip_gap=0.1,
        )

    monkeypatch.setattr(benders_module, "solve_robust_dual_subproblem", fake_subproblem)
    result = solve_benders(config, instance, "standard_benders")
    assert result.status == "iteration_limit"
    assert result.robust_cost == pytest.approx(200.0)
    assert result.gap is not None and result.gap > 0.0
    assert result.iteration_log[0]["target_subproblem_objective"] == pytest.approx(100.0)
    assert result.iteration_log[0]["target_subproblem_objective_bound"] == pytest.approx(200.0)
    assert result.iteration_log[0]["ub_uses_subproblem_bound"] is True


def test_additional_runtime_counts_no_incumbent_and_duplicate_pattern() -> None:
    primary = fake_robust_cut(pattern=(1, 0))
    no_incumbent = fake_robust_cut(
        objective=None,
        objective_bound=120.0,
        status="time_limit",
        runtime=2.5,
        has_incumbent=False,
    )
    no_incumbent_batch = generate_additional_robust_cuts(
        primary,
        max_cuts_per_iteration=2,
        time_budget=10.0,
        solve_extra=lambda _excluded, _remaining: no_incumbent,
    )
    assert no_incumbent_batch.runtime == pytest.approx(2.5)
    assert no_incumbent_batch.without_incumbent_count == 1
    assert no_incumbent_batch.cuts == []

    duplicate = fake_robust_cut(runtime=3.25, pattern=(1, 0))
    duplicate_batch = generate_additional_robust_cuts(
        primary,
        max_cuts_per_iteration=2,
        time_budget=10.0,
        solve_extra=lambda _excluded, _remaining: duplicate,
    )
    assert duplicate_batch.runtime == pytest.approx(3.25)
    assert duplicate_batch.duplicate_patterns_rejected == 1
    assert duplicate_batch.cuts == []


def test_k_one_skips_extra_solves_and_extra_bounds_cannot_affect_ub() -> None:
    primary = fake_robust_cut(objective=90.0, objective_bound=100.0)
    calls = 0

    def unexpected_solve(_excluded: list[dict[tuple[int, int], int]], _remaining: float) -> RobustDualSubproblemResult:
        nonlocal calls
        calls += 1
        return fake_robust_cut(objective=80.0, objective_bound=1e9, pattern=(0, 1))

    single = generate_additional_robust_cuts(primary, 1, 10.0, unexpected_solve)
    assert calls == 0
    assert single.runtime == 0.0
    assert single.cuts == []

    multiple = generate_additional_robust_cuts(primary, 2, 10.0, unexpected_solve)
    assert len(multiple.cuts) == 1
    assert multiple.cuts[0].objective_bound == pytest.approx(1e9)
    robust_cost, valid_ub, _ = target_upper_cost("robust_dual_milp", primary)
    assert robust_cost == pytest.approx(100.0)
    assert valid_ub is True


def test_time_to_gap_metrics_use_first_reached_iteration() -> None:
    result = SolveResult(
        method="proposed_adaptive_benders",
        status="iteration_limit",
        objective=100.0,
        lower_bound=99.0,
        upper_bound=100.0,
        gap=0.01,
        runtime=8.0,
        subproblem_runtime=6.0,
        iteration_log=[
            {"iteration": 1, "global_gap": 0.08, "elapsed_time": 2.0, "lb_improvement": None},
            {"iteration": 2, "global_gap": 0.04, "elapsed_time": 5.0, "lb_improvement": 3.0},
            {"iteration": 3, "global_gap": 0.01, "elapsed_time": 8.0, "lb_improvement": 1.0},
        ],
    )
    metrics = _time_to_gap_metrics(result)
    assert metrics["reached_gap_5pct"] is True
    assert metrics["time_to_gap_5pct"] == pytest.approx(5.0)
    assert metrics["iteration_to_gap_1pct"] == 3
    assert metrics["reached_gap_05pct"] is False
    assert metrics["subproblem_time_share"] == pytest.approx(0.75)
    assert metrics["mean_lb_improvement_per_iteration"] == pytest.approx(2.0)

    row = {
        "experiment_name": "diagnostic",
        "instance_size": "medium",
        "method": "proposed_adaptive_benders",
        "variant_name": "full",
        "status": "iteration_limit",
        "objective": 100.0,
        "runtime": 8.0,
        "final_gap": 0.01,
        "iterations": 3,
        "cuts_added_total": 2,
        "cuts_skipped_total": 1,
        "master_time": 2.0,
        "subproblem_time": 6.0,
        "valid_UB": True,
        **metrics,
    }
    summary = _summary_rows([row])[0]
    assert summary["gap_5pct_rate"] == pytest.approx(1.0)
    assert summary["gap_05pct_rate"] == pytest.approx(0.0)
    assert summary["mean_time_to_gap_5pct"] == pytest.approx(5.0)


def test_gurobi_status_nine_is_completed_in_summary_metrics() -> None:
    row = {
        "experiment_name": "status_normalization",
        "instance_size": "medium",
        "method": "standard_benders",
        "variant_name": "standard_benders",
        "status": 9,
        "objective": 120.0,
        "runtime": 60.0,
        "final_gap": 0.03,
        "iterations": 150,
        "cuts_added_total": 150,
        "cuts_skipped_total": 0,
        "master_time": 10.0,
        "subproblem_time": 50.0,
        "valid_UB": True,
    }

    summary = _summary_rows([row])[0]

    assert summary["num_completed"] == 1
    assert summary["completed_rate"] == pytest.approx(1.0)
    assert summary["mean_objective"] == pytest.approx(120.0)
    assert summary["mean_runtime"] == pytest.approx(60.0)


def test_no_good_constraint_returns_distinct_pattern_and_valid_cut() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=82)
    x_values = {(i, j): 0.5 * instance.inventory_ub[i][j] for i in instance.I for j in instance.J}
    primary = solve_robust_dual_subproblem(instance, x_values, gamma=1, mip_gap=0.0)
    excluded = [{key: int(round(value)) for key, value in primary.z_values.items()}]
    secondary = solve_robust_dual_subproblem(
        instance,
        x_values,
        gamma=1,
        mip_gap=0.0,
        excluded_patterns=excluded,
    )
    assert primary.has_incumbent
    assert secondary.has_incumbent
    assert {key: round(value) for key, value in primary.z_values.items()} != {
        key: round(value) for key, value in secondary.z_values.items()
    }
    exact = solve_robust_dual_subproblem(instance, x_values, gamma=1, mip_gap=0.0)
    assert secondary.cut_value(x_values) <= exact.objective + 1e-6


def test_no_incumbent_result_cannot_generate_a_cut() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=85)
    x_values = {(i, j): 0.0 for i in instance.I for j in instance.J}
    zero_pattern = {(r, j): 0 for r in instance.R for j in instance.J}
    result = solve_robust_dual_subproblem(
        instance,
        x_values,
        gamma=0,
        mip_gap=0.0,
        excluded_patterns=[zero_pattern],
    )
    assert result.has_incumbent is False
    assert result.objective is None
    with pytest.raises(ValueError, match="feasible incumbent"):
        result.cut_value(x_values)


def test_multicut_k_one_preserves_default_and_k_two_adds_diagnostics() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=83)
    all_primary_config = deepcopy(config)
    all_primary_config["algorithm"]["cut_selection_enabled"] = False
    all_primary_config["algorithm"]["max_cuts_per_iteration"] = 1
    all_primary = solve_benders(all_primary_config, instance, "adaptive_gap_gamma_benders")
    screened_primary_config = deepcopy(config)
    screened_primary_config["algorithm"]["relative_cut_threshold"] = 1e9
    screened_primary_config["algorithm"]["max_cuts_per_iteration"] = 1
    screened_primary = solve_benders(
        screened_primary_config,
        instance,
        "adaptive_gap_gamma_benders",
    )
    assert screened_primary.metadata["max_cuts_per_iteration"] == 1
    assert screened_primary.objective == pytest.approx(all_primary.objective)
    assert screened_primary.lower_bound == pytest.approx(all_primary.lower_bound)
    assert screened_primary.cuts == all_primary.cuts
    assert all(
        row["cut_added"]
        for row in screened_primary.iteration_log
        if row["absolute_cut_violation"] is not None
        and row["absolute_cut_violation"] > config["algorithm"]["cut_violation_tol"]
    )

    multi_config = deepcopy(config)
    multi_config["algorithm"]["max_cuts_per_iteration"] = 2
    multiple = solve_benders(multi_config, instance, "adaptive_gap_gamma_benders")
    assert multiple.metadata["max_cuts_per_iteration"] == 2
    assert multiple.metadata["mean_cuts_generated_per_iteration"] >= 1.0
    assert "additional_subproblem_time" in multiple.metadata
    assert "last_secondary_cut_decisions" in multiple.metadata
    for row in multiple.iteration_log:
        for decision in row["secondary_cut_decisions"]:
            assert {
                "normalized_violation",
                "marginal_normalized_violation",
                "added",
                "skip_reason",
                "active_threshold",
            }.issubset(decision)
    assert multiple.objective is not None


def test_high_relative_threshold_does_not_skip_primary_cut() -> None:
    config = tiny_config()
    config["algorithm"]["relative_cut_threshold"] = 2.0
    config["algorithm"]["cut_stall_patience"] = 1
    config["benders"]["max_iterations"] = 2
    instance = generate_instance(config, seed=84)
    result = solve_benders(config, instance, "adaptive_gap_gamma_benders")
    assert all(
        row["cut_skip_reason"] != "low_relative_violation"
        for row in result.iteration_log
    )
    assert any(row["cut_added"] for row in result.iteration_log)
