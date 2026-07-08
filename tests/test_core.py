from __future__ import annotations

from src.benders import solve_benders
from src.instance import generate_instance
from src.monolithic import solve_monolithic
from src.scenarios import enumerate_budget_scenarios
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
