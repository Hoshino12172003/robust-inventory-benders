from __future__ import annotations

import pytest

from src.fairness_benders import solve_fairness_benders
from src.robust_regional_fairness import separate_robust_fairness_feasibility, solve_fairness_extensive_form
from tests.test_robust_regional_fairness import tiny_instance


FROZEN_PRECISION = {
    "precision_policy": "joint_error_budget",
    "adaptive_master_precision_enabled": True,
    "adaptive_subproblem_precision_enabled": True,
    "master_gap_max": 0.02,
    "master_gap_min": 0.0001,
    "subproblem_gap_max": 0.05,
    "subproblem_gap_min": 0.0001,
    "fixed_master_mip_gap": 0.02,
    "fixed_subproblem_mip_gap": 0.05,
    "master_error_budget_ratio": 0.25,
    "subproblem_error_budget_ratio": 0.50,
    "monotone_precision_tightening": True,
}


@pytest.mark.parametrize("gamma", [0, 2])
def test_fairness_benders_matches_tiny_extensive_form(gamma: int) -> None:
    instance = tiny_instance()
    baseline = 30.0 if gamma == 0 else 42.0
    extensive = solve_fairness_extensive_form(
        instance,
        baseline_cost=baseline,
        rho=0.10,
        gamma=gamma,
        max_scenarios=10,
        mip_gap=0.0,
    )
    benders = solve_fairness_benders(
        instance,
        baseline_cost=baseline,
        rho=0.10,
        gamma=gamma,
        algorithm_config=FROZEN_PRECISION,
        max_iterations=100,
        time_limit=60.0,
        tol=1e-6,
    )
    assert extensive.status == "optimal"
    assert benders.status == "optimal"
    assert benders.objective_t == pytest.approx(extensive.objective_t, abs=2e-6)
    assert benders.lower_bound <= extensive.objective_t + 2e-6
    assert benders.upper_bound >= extensive.objective_t - 2e-6
    assert all(not row["fairness_cut_core_point_strengthened"] for row in benders.iteration_log)
    assert benders.cuts_with_cost_component <= benders.cuts
    assert benders.cuts_with_fairness_component <= benders.cuts
    if gamma == 0:
        assert benders.y_values == pytest.approx(extensive.y_values, abs=2e-6)
        assert benders.x_values[0] == pytest.approx(extensive.x_values[0], abs=2e-6)


def test_final_benders_solution_is_certified_by_separation_bound() -> None:
    instance = tiny_instance()
    result = solve_fairness_benders(
        instance,
        baseline_cost=30.0,
        rho=0.10,
        gamma=0,
        algorithm_config=FROZEN_PRECISION,
        max_iterations=100,
        time_limit=60.0,
        tol=1e-6,
    )
    certificate = separate_robust_fairness_feasibility(
        instance,
        y_values=result.y_values,
        x_values=result.x_values,
        t_value=result.objective_t,
        cost_budget_value=result.cost_budget,
        gamma=0,
        mip_gap=0.0,
    )
    assert certificate.robust_feasibility_certified
    assert certificate.objective_bound <= 1e-7


def test_cost_and_fairness_worst_patterns_are_separately_named() -> None:
    instance = tiny_instance()
    result = solve_fairness_benders(
        instance,
        baseline_cost=42.0,
        rho=0.10,
        gamma=2,
        algorithm_config=FROZEN_PRECISION,
        max_iterations=100,
        time_limit=60.0,
        tol=1e-6,
    )
    assert result.status == "optimal"
    assert "separation_patterns_seen" in result.to_dict()
    assert result.metadata["cost_and_fairness_worst_separately_identified_in_post_evaluation"] is True
    assert result.metadata["same_recourse_satisfies_cost_and_fairness"] is True


def test_non_joint_precision_policy_is_rejected() -> None:
    instance = tiny_instance()
    with pytest.raises(ValueError, match="joint_error_budget"):
        solve_fairness_benders(
            instance,
            baseline_cost=30.0,
            rho=0.0,
            gamma=0,
            algorithm_config={"precision_policy": "legacy"},
            max_iterations=2,
            time_limit=2.0,
        )
