from __future__ import annotations

import pytest

from src.benders import solve_benders
from src.instance import generate_instance
from src.scenarios import count_budget_scenarios, enumerate_budget_scenarios_with_metadata


def tiny_config() -> dict:
    return {
        "seed": 17,
        "instance": {
            "num_warehouses": 2,
            "num_products": 1,
            "num_regions": 2,
            "budget_factor": 0.7,
        },
        "robust": {
            "gamma_target": 1,
            "max_scenarios": 50,
            "exact_scenarios": True,
            "gamma_schedule": [0, 1],
        },
        "benders": {
            "max_iterations": 40,
            "tol": 1e-4,
            "initial_mip_gap": 0.05,
            "final_mip_gap": 1e-5,
            "time_limit": 60,
            "output_flag": False,
        },
    }


def test_exact_scenarios_metadata_full_enumeration_under_limit() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=17)
    gamma = 1
    total = count_budget_scenarios(instance, gamma)

    result = enumerate_budget_scenarios_with_metadata(
        instance,
        gamma,
        max_scenarios=total,
        exact_scenarios=True,
    )

    assert result.scenario_mode == "full"
    assert result.exact_scenarios is True
    assert result.num_scenarios_used == result.num_scenarios_total_estimated
    assert len(result.scenarios) == count_budget_scenarios(instance, gamma)


def test_exact_scenarios_metadata_raises_over_limit() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=18)

    with pytest.raises(ValueError, match="Exact scenario enumeration exceeds max_scenarios"):
        enumerate_budget_scenarios_with_metadata(
            instance,
            gamma=1,
            max_scenarios=1,
            exact_scenarios=True,
        )


def test_candidate_scenarios_metadata_allowed_when_not_exact() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=19)

    result = enumerate_budget_scenarios_with_metadata(
        instance,
        gamma=1,
        max_scenarios=1,
        exact_scenarios=False,
    )

    assert result.scenario_mode == "candidate"
    assert result.exact_scenarios is False
    assert result.num_scenarios_used <= result.max_scenarios
    assert result.num_scenarios_total_estimated > result.max_scenarios


def test_solve_benders_records_scenario_mode_metadata() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=20)

    result = solve_benders(config, instance, "standard_benders")

    assert "exact_scenarios" in result.metadata
    assert "scenario_mode_target" in result.metadata
    assert "num_target_scenarios_used" in result.metadata
    assert "num_target_scenarios_total_estimated" in result.metadata
    assert "heuristic_scenarios" in result.metadata
