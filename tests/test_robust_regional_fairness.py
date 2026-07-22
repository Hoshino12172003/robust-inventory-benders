from __future__ import annotations

import math

import pytest
from gurobipy import GRB

import src.robust_regional_fairness as fairness_module
from src.instance import InventoryInstance
from src.regional_fairness_diagnostic import (
    solve_default_and_fair_best_recourse,
    summarize_regional_service,
)
from src.robust_regional_fairness import (
    FairnessFarkasRay,
    FixedScenarioCertificate,
    certify_fixed_scenario_fairness_feasibility,
    cost_tolerance,
    evaluate_fairness_solution,
    farkas_ray_validation,
    fairness_cost_budget,
    fairness_cut_from_ray,
    separate_robust_fairness_feasibility,
    separation_bound_certifies,
    solve_fairness_extensive_form,
    solve_scenario_policy_with_shared_caps,
    validate_farkas_ray,
)
from src.scenarios import DemandScenario, enumerate_budget_scenarios


def tiny_instance(
    *,
    regions: int = 2,
    products: int = 1,
    symmetric: bool = False,
    zero_second_demand: bool = False,
) -> InventoryInstance:
    demand = [[5.0] for _ in range(regions)]
    if zero_second_demand and regions > 1:
        demand[1][0] = 0.0
    return InventoryInstance(
        name="hand_built_fairness_instance",
        num_warehouses=1,
        num_products=products,
        num_regions=regions,
        fixed_cost=[0.0],
        inventory_cost=[[5.0 for _ in range(products)]],
        capacity=[30.0],
        volume=[1.0 for _ in range(products)],
        budget=150.0,
        transport_cost=[
            [[0.0 for _ in range(products)] for _ in range(regions)]
        ],
        shortage_penalty=[
            [10.0 if symmetric or r == 0 else 1.0 for _ in range(products)]
            for r in range(regions)
        ],
        service_penalty=[100.0 for _ in range(products)],
        service_level=[0.0 for _ in range(products)],
        base_demand=demand,
        demand_deviation=[[1.0 for _ in range(products)] for _ in range(regions)],
        inventory_ub=[[30.0 for _ in range(products)]],
    )


def scenario(instance: InventoryInstance, active: tuple[tuple[int, int], ...] = ()) -> DemandScenario:
    demand = [row[:] for row in instance.base_demand]
    for r, j in active:
        demand[r][j] += instance.demand_deviation[r][j]
    return DemandScenario("manual", active, tuple(tuple(row) for row in demand))


def cost_neutral_reconfiguration_instance() -> InventoryInstance:
    return InventoryInstance(
        name="cost_neutral_reconfiguration",
        num_warehouses=2,
        num_products=1,
        num_regions=2,
        fixed_cost=[0.0, 0.0],
        inventory_cost=[[0.0], [0.0]],
        capacity=[10.0, 10.0],
        volume=[1.0],
        budget=0.0,
        transport_cost=[[[0.0], [10.0]], [[10.0], [0.0]]],
        shortage_penalty=[[1.0], [1.0]],
        service_penalty=[100.0],
        service_level=[0.0],
        base_demand=[[5.0], [5.0]],
        demand_deviation=[[0.0], [0.0]],
        inventory_ub=[[10.0], [10.0]],
    )


def test_budget_and_tolerance_are_frozen_epsilon_constraint_quantities() -> None:
    budget = fairness_cost_budget(100.0, 0.025)
    assert budget.budget == pytest.approx(102.5)
    assert cost_tolerance(100.0) == pytest.approx(1.01e-4)
    with pytest.raises(ValueError):
        fairness_cost_budget(100.0, -0.01)
    with pytest.raises(ValueError):
        fairness_cost_budget(math.inf, 0.0)


def test_single_region_extensive_form_has_no_regional_gap() -> None:
    instance = tiny_instance(regions=1)
    result = solve_fairness_extensive_form(
        instance, baseline_cost=30.0, rho=0.0, gamma=0, max_scenarios=10
    )
    assert result.status == "optimal"
    assert result.wgap == pytest.approx(0.0, abs=1e-8)
    assert result.robust_minimum_fill_rate == pytest.approx(1.0 - result.objective_t)


def test_symmetric_regions_receive_equal_service() -> None:
    instance = tiny_instance(symmetric=True)
    result = solve_fairness_extensive_form(
        instance, baseline_cost=50.0, rho=0.0, gamma=0, max_scenarios=10
    )
    assert result.status == "optimal"
    policy = result.scenario_policies[0]
    assert policy.fill_rates[0] == pytest.approx(policy.fill_rates[1], abs=1e-7)
    assert policy.fill_rate_gap == pytest.approx(0.0, abs=1e-7)


def test_degenerate_symmetric_recourse_preserves_primary_fairness_value() -> None:
    instance = tiny_instance(symmetric=True)
    primary = solve_fairness_extensive_form(
        instance,
        baseline_cost=50.0,
        rho=0.0,
        gamma=0,
        max_scenarios=10,
        lexicographic_cost_stage=False,
    )
    lexicographic = solve_fairness_extensive_form(
        instance,
        baseline_cost=50.0,
        rho=0.0,
        gamma=0,
        max_scenarios=10,
        lexicographic_cost_stage=True,
    )
    assert primary.status == lexicographic.status == "optimal"
    assert lexicographic.objective_t == pytest.approx(primary.objective_t, abs=1e-7)


def test_hand_built_instance_has_material_regional_gap_at_rho_zero() -> None:
    result = solve_fairness_extensive_form(
        tiny_instance(), baseline_cost=30.0, rho=0.0, gamma=0, max_scenarios=10
    )
    assert result.status == "optimal"
    assert result.objective_t > 0.5
    assert result.robust_minimum_fill_rate < 0.5


def test_zero_demand_region_is_not_applicable() -> None:
    instance = tiny_instance(zero_second_demand=True)
    result = solve_fairness_extensive_form(
        instance, baseline_cost=25.0, rho=0.0, gamma=0, max_scenarios=10
    )
    assert result.status == "optimal"
    policy = result.scenario_policies[0]
    assert policy.fill_rates[1] is None
    assert policy.fill_rate_gap == pytest.approx(0.0, abs=1e-8)


def test_cost_budget_can_make_model_infeasible() -> None:
    instance = tiny_instance()
    result = solve_fairness_extensive_form(
        instance, baseline_cost=0.0, rho=0.0, gamma=0, max_scenarios=10
    )
    assert result.status == "infeasible"
    assert result.objective_t is None


def test_shared_policy_uses_one_recourse_for_cost_and_fairness() -> None:
    instance = tiny_instance()
    demand_scenario = scenario(instance)
    x_values = [[5.0]]
    y_values = [1.0]
    default, fair = solve_default_and_fair_best_recourse(
        instance, demand_scenario, x_values, time_limit=30.0
    )
    fair_metrics = summarize_regional_service(
        [list(row) for row in demand_scenario.demand], fair.shortage_values
    )
    t_value = 1.0 - float(fair_metrics["minimum_fill_rate"])
    cap = 25.0 + default.original_optimal_cost + default.cost_tolerance
    shared = solve_scenario_policy_with_shared_caps(
        instance,
        demand_scenario,
        y_values=y_values,
        x_values=x_values,
        t_value=t_value,
        cost_budget_value=cap,
    )
    assert shared.recourse_cost <= default.original_optimal_cost + default.cost_tolerance + 1e-6
    assert shared.minimum_fill_rate >= 1.0 - t_value - 1e-7


def test_rho_zero_allows_cost_neutral_first_stage_reconfiguration() -> None:
    instance = cost_neutral_reconfiguration_instance()
    demand_scenario = scenario(instance)
    fixed_x = [[10.0], [0.0]]
    default, fixed_fair = solve_default_and_fair_best_recourse(
        instance, demand_scenario, fixed_x, time_limit=30.0
    )
    fixed_metrics = summarize_regional_service(
        [list(row) for row in demand_scenario.demand], fixed_fair.shortage_values
    )
    fixed_t = 1.0 - float(fixed_metrics["minimum_fill_rate"])
    integrated = solve_fairness_extensive_form(
        instance,
        baseline_cost=default.original_optimal_cost,
        rho=0.0,
        gamma=0,
        max_scenarios=10,
        lexicographic_cost_stage=False,
    )
    assert integrated.status == "optimal"
    assert integrated.actual_robust_cost <= default.original_optimal_cost + 1e-7
    assert integrated.objective_t < fixed_t - 0.5
    assert integrated.x_values != fixed_x


def test_gamma_zero_and_two_extreme_points_are_exact() -> None:
    instance = tiny_instance(regions=2)
    assert len(enumerate_budget_scenarios(instance, 0, max_scenarios=10)) == 1
    assert len(enumerate_budget_scenarios(instance, 2, max_scenarios=10)) == 4


def test_cost_worst_and_fairness_worst_scenarios_can_differ() -> None:
    result = solve_fairness_extensive_form(
        tiny_instance(), baseline_cost=42.0, rho=0.10, gamma=2, max_scenarios=10
    )
    assert result.status == "optimal"
    assert result.cost_worst_scenario == "g2_r0j0_r1j0"
    assert result.fairness_worst_scenario == "g1_r0j0"
    assert result.cost_worst_scenario != result.fairness_worst_scenario


def test_post_evaluation_recomputes_all_shared_cap_scenarios() -> None:
    instance = tiny_instance()
    solution = solve_fairness_extensive_form(
        instance,
        baseline_cost=42.0,
        rho=0.10,
        gamma=2,
        max_scenarios=10,
        lexicographic_cost_stage=False,
    )
    evaluation = evaluate_fairness_solution(
        instance,
        y_values=solution.y_values,
        x_values=solution.x_values,
        t_value=solution.objective_t,
        baseline_cost=42.0,
        rho=0.10,
        gamma=2,
        max_scenarios=10,
    )
    assert evaluation.valid
    assert evaluation.scenario_count == 4
    assert evaluation.actual_price_of_fairness <= 0.10 + 1e-6
    assert evaluation.wminfr >= 1.0 - solution.objective_t - 1e-6
    assert evaluation.objective_t_consistent is True
    assert evaluation.realized_worst_shortage_rate <= solution.objective_t + 1e-6
    assert evaluation.cost_worst_scenario is not None
    assert evaluation.fairness_worst_scenario is not None


def test_farkas_separation_finds_infeasible_candidate_and_valid_cut() -> None:
    instance = tiny_instance()
    separation = separate_robust_fairness_feasibility(
        instance,
        y_values=[0.0],
        x_values=[[0.0]],
        t_value=0.0,
        cost_budget_value=30.0,
        gamma=0,
        mip_gap=0.0,
    )
    assert separation.status == "optimal"
    assert separation.cut is not None
    assert separation.cut_certificate_source == "fixed_scenario_normalized_farkas_lp"
    assert separation.fixed_scenario_certificate is not None
    assert separation.fixed_scenario_certificate.infeasibility_certified
    assert validate_farkas_ray(instance, separation.cut.ray)
    ray_sum = (
        sum(value for row in separation.cut.ray.demand for value in row)
        + sum(value for row in separation.cut.ray.supply for value in row)
        + sum(separation.cut.ray.service)
        + separation.cut.ray.cost
        + sum(separation.cut.ray.regional_fairness)
    )
    assert ray_sum == pytest.approx(1.0, abs=1e-7)
    assert separation.objective is not None and separation.objective > 1e-7
    assert separation.cut.value([0.0], [[0.0]], 0.0) == pytest.approx(-separation.objective, abs=1e-6)

    feasible = solve_fairness_extensive_form(
        instance, baseline_cost=30.0, rho=0.0, gamma=0, max_scenarios=10
    )
    assert feasible.status == "optimal"
    assert separation.cut.value(feasible.y_values, feasible.x_values, feasible.objective_t) >= -1e-6


def test_invalid_dual_ray_is_rejected() -> None:
    instance = tiny_instance()
    invalid = FairnessFarkasRay(
        demand=[[1.0], [1.0]],
        supply=[[0.0]],
        service=[0.0],
        cost=0.0,
        regional_fairness=[0.0, 0.0],
    )
    assert not validate_farkas_ray(instance, invalid)


@pytest.mark.parametrize("normalization_sum", [0.0, 0.5])
def test_zero_or_under_normalized_ray_is_rejected(normalization_sum: float) -> None:
    instance = tiny_instance()
    ray = FairnessFarkasRay(
        demand=[[0.0], [0.0]],
        supply=[[normalization_sum]],
        service=[0.0],
        cost=0.0,
        regional_fairness=[0.0, 0.0],
    )
    diagnostics = farkas_ray_validation(instance, ray)
    assert not diagnostics.valid
    assert diagnostics.normalization_residual == pytest.approx(1.0 - normalization_sum)


def test_negative_multiplier_and_dual_cone_residual_are_reported() -> None:
    instance = tiny_instance()
    negative = FairnessFarkasRay(
        demand=[[-1.0e-4], [0.0]],
        supply=[[1.0001]],
        service=[0.0],
        cost=0.0,
        regional_fairness=[0.0, 0.0],
    )
    negative_diagnostics = farkas_ray_validation(instance, negative)
    assert not negative_diagnostics.valid
    assert negative_diagnostics.minimum_multiplier == pytest.approx(-1.0e-4)

    cone_violation = FairnessFarkasRay(
        demand=[[1.0], [0.0]],
        supply=[[0.0]],
        service=[0.0],
        cost=0.0,
        regional_fairness=[0.0, 0.0],
    )
    cone_diagnostics = farkas_ray_validation(instance, cone_violation)
    assert not cone_diagnostics.valid
    assert cone_diagnostics.maximum_dual_cone_violation == pytest.approx(1.0)


def test_fixed_scenario_primal_rejects_milp_false_positive_without_a_cut() -> None:
    instance = tiny_instance()
    certificate = certify_fixed_scenario_fairness_feasibility(
        instance,
        y_values=[1.0],
        x_values=[[10.0]],
        t_value=1.0,
        cost_budget_value=100.0,
        demand_values=[row[:] for row in instance.base_demand],
        time_limit=30.0,
    )
    assert certificate.primal_status == "optimal"
    assert certificate.primal_feasible
    assert not certificate.infeasibility_certified
    assert certificate.ray is None


def test_fixed_scenario_infeasibility_has_independent_valid_farkas_cut() -> None:
    instance = tiny_instance()
    demand = [row[:] for row in instance.base_demand]
    certificate = certify_fixed_scenario_fairness_feasibility(
        instance,
        y_values=[0.0],
        x_values=[[0.0]],
        t_value=0.0,
        cost_budget_value=30.0,
        demand_values=demand,
        time_limit=30.0,
    )
    assert certificate.primal_status == "infeasible"
    assert not certificate.primal_feasible
    assert certificate.infeasibility_certified
    assert certificate.ray_status == "optimal"
    assert certificate.ray is not None
    assert certificate.ray_validation is not None
    assert certificate.ray_validation.valid
    cut = fairness_cut_from_ray(
        instance,
        cost_budget_value=30.0,
        demand_values=demand,
        ray=certificate.ray,
        active_deviations=[],
    )
    assert cut.value([0.0], [[0.0]], 0.0) < -1e-7


def test_separation_false_positive_is_excluded_without_generating_a_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = tiny_instance()

    def fixed_feasible(*args: object, **kwargs: object) -> FixedScenarioCertificate:
        return FixedScenarioCertificate(
            primal_status="optimal",
            primal_feasible=True,
            infeasibility_certified=False,
            primal_runtime=0.0,
            certification_reason="fixed_scenario_primal_feasible",
        )

    monkeypatch.setattr(
        fairness_module,
        "certify_fixed_scenario_fairness_feasibility",
        fixed_feasible,
    )
    separation = separate_robust_fairness_feasibility(
        instance,
        y_values=[0.0],
        x_values=[[0.0]],
        t_value=0.0,
        cost_budget_value=30.0,
        gamma=0,
        mip_gap=0.0,
    )
    assert separation.cut is None
    assert not separation.robust_feasibility_certified
    assert separation.false_positive_scenarios_excluded == 1


def test_candidate_without_fixed_scenario_certificate_returns_uncertified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = tiny_instance()

    def fixed_uncertified(*args: object, **kwargs: object) -> FixedScenarioCertificate:
        return FixedScenarioCertificate(
            primal_status="infeasible",
            primal_feasible=False,
            infeasibility_certified=False,
            primal_runtime=0.0,
            ray_status="numeric",
            certification_reason="fixed_scenario_farkas_certificate_unavailable",
        )

    monkeypatch.setattr(
        fairness_module,
        "certify_fixed_scenario_fairness_feasibility",
        fixed_uncertified,
    )
    separation = separate_robust_fairness_feasibility(
        instance,
        y_values=[0.0],
        x_values=[[0.0]],
        t_value=0.0,
        cost_budget_value=30.0,
        gamma=0,
        mip_gap=0.0,
    )
    assert separation.status == "uncertified_infeasible"
    assert separation.cut is None
    assert not separation.robust_feasibility_certified
    assert separation.certification_reason == "fixed_scenario_farkas_certificate_unavailable"


@pytest.mark.parametrize(
    "status",
    [GRB.INTERRUPTED, GRB.NUMERIC, GRB.SUBOPTIMAL, GRB.INFEASIBLE, GRB.UNBOUNDED],
)
def test_non_whitelisted_separation_status_never_certifies(status: int) -> None:
    certified, reason = separation_bound_certifies(status, -1.0, 1e-7)
    assert not certified
    assert "not_certifiable" in reason


def test_time_limit_requires_a_proving_objective_bound() -> None:
    assert separation_bound_certifies(GRB.TIME_LIMIT, 1e-8, 1e-7)[0]
    assert not separation_bound_certifies(GRB.TIME_LIMIT, 1e-4, 1e-7)[0]


def test_fairness_cut_sign_regression_would_be_detected() -> None:
    instance = tiny_instance()
    demand = [row[:] for row in instance.base_demand]
    ray = FairnessFarkasRay(
        demand=[[0.5], [0.5]],
        supply=[[0.5]],
        service=[0.0],
        cost=0.0,
        regional_fairness=[0.0, 0.0],
    )
    cut = fairness_cut_from_ray(
        instance,
        cost_budget_value=30.0,
        demand_values=demand,
        ray=ray,
        active_deviations=[],
    )
    correct = cut.value([0.0], [[0.0]], 0.0)
    wrong_sign = -correct
    assert correct < 0.0
    assert wrong_sign > 0.0


def test_rho_frontier_cannot_worsen_optimal_t() -> None:
    instance = tiny_instance()
    low = solve_fairness_extensive_form(
        instance, baseline_cost=30.0, rho=0.0, gamma=0, max_scenarios=10
    )
    high = solve_fairness_extensive_form(
        instance, baseline_cost=30.0, rho=1.0, gamma=0, max_scenarios=10
    )
    assert low.status == high.status == "optimal"
    assert high.objective_t <= low.objective_t + 1e-7
    assert high.robust_minimum_fill_rate >= low.robust_minimum_fill_rate - 1e-7
