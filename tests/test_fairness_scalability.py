from __future__ import annotations

import json

import pytest

from src.fairness_benders import solve_fairness_benders
from src.fairness_scalability import (
    CertifiedScenarioCache,
    PersistentFairnessSeparation,
    SCALABILITY_CANDIDATES,
    validate_scalability_strategy,
)
from src.robust_regional_fairness import FixedScenarioCertificate, solve_fairness_extensive_form
from tests.test_fairness_benders_against_extensive_form import FROZEN_PRECISION
from tests.test_robust_regional_fairness import tiny_instance


def test_strategy_validation_is_closed() -> None:
    assert tuple(validate_scalability_strategy(v) for v in SCALABILITY_CANDIDATES) == SCALABILITY_CANDIDATES
    with pytest.raises(ValueError):
        validate_scalability_strategy("uncertified_fast_path")


def test_cache_stores_patterns_and_recertifies_each_current_point() -> None:
    instance = tiny_instance()
    cache = CertifiedScenarioCache()
    assert cache.add([{"region": 0, "product": 0}])
    assert not cache.add([{"region": 0, "product": 0}])
    calls: list[float] = []

    def certifier(*args, **kwargs):
        calls.append(float(kwargs["t_value"]))
        return FixedScenarioCertificate(
            primal_status="optimal",
            primal_feasible=True,
            infeasibility_certified=False,
            primal_runtime=0.0,
            certification_reason="fixed_scenario_primal_feasible",
        )

    for t_value in (0.2, 0.4):
        batch = cache.certify_current_point(
            instance,
            y_values=[1.0],
            x_values=[[10.0]],
            t_value=t_value,
            cost_budget_value=100.0,
            time_limit=1.0,
            feasibility_tolerance=1.0e-7,
            max_cuts=1,
            output_flag=False,
            certifier=certifier,
        )
        assert batch.cuts == []
    assert calls == [0.2, 0.4]
    assert not hasattr(cache, "_rays")


def test_false_positive_exclusion_is_local_to_one_separation_call() -> None:
    instance = tiny_instance()
    calls = 0

    def feasible_certifier(*args, **kwargs):
        nonlocal calls
        calls += 1
        return FixedScenarioCertificate(
            primal_status="optimal",
            primal_feasible=True,
            infeasibility_certified=False,
            primal_runtime=0.0,
            certification_reason="synthetic_fixed_primal_feasible",
        )

    separator = PersistentFairnessSeparation(instance, gamma=0)
    try:
        baseline_constraints = separator.model.NumConstrs
        for _ in range(2):
            result = separator.separate(
                y_values=[0.0],
                x_values=[[0.0]],
                t_value=0.0,
                cost_budget_value=0.0,
                mip_gap=0.0,
                time_limit=5.0,
                certifier=feasible_certifier,
            )
            assert result.robust_feasibility_certified is False
            assert separator.model.NumConstrs == baseline_constraints
        assert calls == 2
    finally:
        separator.dispose()


@pytest.mark.parametrize("strategy", SCALABILITY_CANDIDATES)
@pytest.mark.parametrize("gamma,baseline", [(0, 30.0), (2, 42.0)])
def test_each_candidate_matches_independent_tiny_extensive_form(
    strategy: str, gamma: int, baseline: float
) -> None:
    instance = tiny_instance()
    extensive = solve_fairness_extensive_form(
        instance,
        baseline_cost=baseline,
        rho=0.10,
        gamma=gamma,
        max_scenarios=10,
        mip_gap=0.0,
    )
    config = {**FROZEN_PRECISION, "fairness_scalability_strategy": strategy}
    result = solve_fairness_benders(
        instance,
        baseline_cost=baseline,
        rho=0.10,
        gamma=gamma,
        algorithm_config=config,
        max_iterations=100,
        time_limit=60.0,
        tol=1.0e-6,
    )
    assert result.status == "optimal"
    assert result.objective_t == pytest.approx(extensive.objective_t, abs=2.0e-6)
    assert result.metadata["fairness_scalability_strategy"] == strategy
    assert result.metadata["total_iterations"] == result.iterations
    assert json.dumps(result.to_dict(), allow_nan=False)
    assert all(row["cuts_per_iteration"] <= (5 if strategy.endswith("batch5") else 1) for row in result.iteration_log)
    assert all(
        not row["robust_feasibility_certified"]
        or row["separation_objective_bound"] <= 1.0e-7
        for row in result.iteration_log
    )


def test_default_path_remains_single_cut() -> None:
    result = solve_fairness_benders(
        tiny_instance(),
        baseline_cost=30.0,
        rho=0.10,
        gamma=0,
        algorithm_config=FROZEN_PRECISION,
        max_iterations=100,
        time_limit=60.0,
        tol=1.0e-6,
    )
    assert result.metadata["fairness_scalability_strategy"] == "single_cut"
    assert max(row["cuts_per_iteration"] for row in result.iteration_log) <= 1
