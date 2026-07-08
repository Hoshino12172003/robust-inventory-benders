from __future__ import annotations

import numpy as np

from src.benders import solve_benders
from src.instance import InventoryInstance, generate_instance
from src.robust_dual_subproblem import solve_robust_dual_subproblem
from src.scenarios import enumerate_budget_scenarios
from src.subproblem import solve_recourse_subproblem


def tiny_config() -> dict:
    return {
        "seed": 23,
        "instance": {
            "num_warehouses": 2,
            "num_products": 2,
            "num_regions": 2,
            "budget_factor": 0.8,
        },
        "robust": {
            "gamma_target": 1,
            "max_scenarios": 50,
            "exact_scenarios": True,
            "gamma_schedule": [0, 1],
        },
        "algorithm": {"subproblem_mode": "robust_dual_milp"},
        "benders": {
            "max_iterations": 60,
            "tol": 1e-4,
            "initial_mip_gap": 0.05,
            "final_mip_gap": 1e-6,
            "time_limit": 60,
            "output_flag": False,
        },
    }


def sample_x(instance: InventoryInstance, scale: float = 0.55) -> dict[tuple[int, int], float]:
    values = {}
    for i in instance.I:
        for j in instance.J:
            product_demand = sum(instance.base_demand[r][j] for r in instance.R)
            values[i, j] = min(instance.inventory_ub[i][j], product_demand * scale / instance.num_warehouses)
    return values


def full_enumeration_value(instance: InventoryInstance, x_values: dict[tuple[int, int], float], gamma: int) -> float:
    scenarios = enumerate_budget_scenarios(instance, gamma, max_scenarios=1000, exact_scenarios=True)
    return max(solve_recourse_subproblem(instance, scenario, x_values).objective for scenario in scenarios)


def test_robust_dual_matches_full_scenario_enumeration() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=23)
    gamma = 1
    x_values = sample_x(instance)

    enumerated = full_enumeration_value(instance, x_values, gamma)
    robust_dual = solve_robust_dual_subproblem(instance, x_values, gamma, mip_gap=0.0)

    assert abs(enumerated - robust_dual.objective) <= 1e-5


def test_robust_dual_cut_matches_current_point() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=24)
    x_values = sample_x(instance)

    result = solve_robust_dual_subproblem(instance, x_values, gamma=1, mip_gap=0.0)

    assert abs(result.cut_value(x_values) - result.objective) <= 1e-5


def test_robust_dual_cut_validity() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=25)
    gamma = 1
    rng = np.random.default_rng(25)
    x_values = sample_x(instance)
    cut = solve_robust_dual_subproblem(instance, x_values, gamma, mip_gap=0.0)

    for _ in range(5):
        x_test = {
            (i, j): float(rng.uniform(0.0, instance.inventory_ub[i][j]))
            for i in instance.I
            for j in instance.J
        }
        true_qr = solve_robust_dual_subproblem(instance, x_test, gamma, mip_gap=0.0)
        assert cut.cut_value(x_test) <= true_qr.objective + 1e-6


def test_benders_runs_with_robust_dual_milp() -> None:
    config = tiny_config()
    instance = generate_instance(config, seed=26)

    result = solve_benders(config, instance, "standard_benders")

    assert result.objective is not None
    assert result.status in {"optimal", "iteration_limit", "time_limit"}
    assert result.metadata["subproblem_mode"] == "robust_dual_milp"
    assert result.gap is None or result.gap <= 1e-3 or result.status in {"iteration_limit", "time_limit"}
