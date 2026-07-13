from __future__ import annotations

from copy import deepcopy

import pytest

import src.benders as benders_module
from src.benders import (
    _settings,
    adaptive_secondary_cut_threshold,
    calculate_global_gap,
    calculate_cut_violations,
    generate_additional_robust_cuts,
    marginal_normalized_violation,
    primary_cut_decision,
    relative_cut_decision,
    select_subproblem_mip_gap,
    solve_benders,
    target_upper_cost,
)
from src.instance import generate_instance
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


def test_new_features_are_backward_compatible_by_default() -> None:
    config = tiny_config()
    config["algorithm"] = {"subproblem_mode": "robust_dual_milp"}
    settings = _settings(config, "standard_benders")
    assert settings.cut_selection_mode == "absolute"
    assert settings.adaptive_subproblem_gap_enabled is False
    assert settings.max_cuts_per_iteration == 1


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
