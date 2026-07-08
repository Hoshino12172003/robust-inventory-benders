from __future__ import annotations

from src.benders import solve_benders
from src.instance import generate_instance
from src.monolithic import solve_monolithic
from src.scenarios import count_budget_scenarios, enumerate_budget_scenarios
from src.subproblem import solve_recourse_subproblem


def tiny_config() -> dict:
    return {
        "seed": 7,
        "instance": {
            "num_warehouses": 2,
            "num_products": 1,
            "num_regions": 2,
            "budget_factor": 0.7,
        },
        "robust": {"gamma_target": 1, "max_scenarios": 50, "gamma_schedule": [0, 1]},
        "benders": {
            "max_iterations": 40,
            "tol": 1e-4,
            "initial_mip_gap": 0.05,
            "final_mip_gap": 1e-5,
            "time_limit": 60,
            "output_flag": False,
        },
    }


def test_subproblem_cut_matches_current_point() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=7)
    scenario = enumerate_budget_scenarios(instance, 1)[-1]
    x_values = {(i, j): 0.0 for i in instance.I for j in instance.J}
    result = solve_recourse_subproblem(instance, scenario, x_values)
    assert abs(result.cut_value(x_values) - result.objective) <= 1e-5


def test_standard_benders_matches_monolithic_on_tiny_instance() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=8)
    mono = solve_monolithic(config, instance)
    benders = solve_benders(config, instance, "standard_benders")
    assert mono.objective is not None
    assert benders.objective is not None
    assert abs(mono.objective - benders.objective) <= 1e-3


def test_adaptive_benders_converges() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=9)
    result = solve_benders(config, instance, "adaptive_gap_gamma_benders")
    assert result.objective is not None
    assert result.upper_bound is not None
    assert result.lower_bound is not None
    assert result.gap is not None
    assert result.gap <= 1e-3


def test_exact_scenarios_full_enumeration_when_under_limit() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=10)
    total = count_budget_scenarios(instance, 1)
    scenarios = enumerate_budget_scenarios(instance, 1, max_scenarios=total, exact_scenarios=True)
    assert len(scenarios) == total


def test_exact_scenarios_raise_when_over_limit() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=11)
    try:
        enumerate_budget_scenarios(instance, 1, max_scenarios=1, exact_scenarios=True)
    except ValueError as exc:
        assert "Exact scenario enumeration exceeds max_scenarios" in str(exc)
    else:
        raise AssertionError("Expected exact scenario enumeration to fail when over max_scenarios.")


def test_candidate_fallback_records_metadata() -> None:
    config = tiny_config()
    config["instance"] = {
        "num_warehouses": 2,
        "num_products": 2,
        "num_regions": 3,
        "budget_factor": 0.7,
    }
    config["robust"] = {
        "gamma_target": 2,
        "max_scenarios": 5,
        "exact_scenarios": False,
        "gamma_schedule": [0, 1, 2],
    }
    config["algorithm"] = {"subproblem_mode": "scenario_enumeration"}
    instance = generate_instance(config, seed=12)
    result = solve_benders(config, instance, "standard_benders")
    assert result.metadata["scenario_mode_target"] == "candidate"
    assert result.metadata["exact_scenarios"] is False
    assert result.metadata["num_target_scenarios_used"] <= 5
    assert result.metadata["num_target_scenarios_total_estimated"] > 5
    assert result.metadata["max_scenarios"] == 5
    assert result.metadata["heuristic_scenarios"] is True
