from __future__ import annotations

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coordination_model import solve_policy  # noqa: E402
from src.instance import InventoryInstance  # noqa: E402
from src.scenarios import DemandScenario  # noqa: E402


def tiny_instance() -> InventoryInstance:
    return InventoryInstance(
        name="coordination_tiny",
        num_warehouses=2,
        num_products=1,
        num_regions=2,
        fixed_cost=[1.0, 1.0],
        inventory_cost=[[1.0], [1.0]],
        capacity=[8.0, 8.0],
        volume=[1.0],
        budget=50.0,
        transport_cost=[[[1.0], [4.0]], [[4.0], [1.0]]],
        shortage_penalty=[[20.0], [20.0]],
        service_penalty=[30.0],
        service_level=[0.8],
        base_demand=[[5.0], [5.0]],
        demand_deviation=[[5.0], [5.0]],
        inventory_ub=[[8.0], [8.0]],
    )


def tiny_scenarios() -> list[DemandScenario]:
    return [
        DemandScenario("nominal", tuple(), ((5.0,), (5.0,))),
        DemandScenario("region_0_surge", ((0, 0),), ((10.0,), (5.0,))),
        DemandScenario("region_1_surge", ((1, 0),), ((5.0,), (10.0,))),
    ]


def solve(instance, scenarios, policy, objective, cost_budget=None):
    return solve_policy(
        instance,
        scenarios,
        policy,
        objective=objective,
        cost_budget=cost_budget,
        severe_shortage_threshold=0.1,
        time_limit=30.0,
        mip_gap=1.0e-8,
        feasibility_tolerance=1.0e-8,
        threads=1,
        solver_seed=0,
    )


def test_flexible_policy_contains_single_source_and_can_strictly_improve_service() -> None:
    instance = tiny_instance()
    scenarios = tiny_scenarios()
    flexible_anchor = solve(instance, scenarios, "flexible_multiwarehouse", "cost_anchor")
    single_anchor = solve(instance, scenarios, "optimized_single_source", "cost_anchor")
    assert flexible_anchor["certified"]
    assert single_anchor["certified"]
    assert flexible_anchor["objective_value"] <= single_anchor["objective_value"] + 1.0e-6

    common_budget = single_anchor["objective_value"] * 1.025
    flexible = solve(
        instance,
        scenarios,
        "flexible_multiwarehouse",
        "service_protection",
        common_budget,
    )
    single = solve(
        instance,
        scenarios,
        "optimized_single_source",
        "service_protection",
        common_budget,
    )
    assert flexible["certified"]
    assert single["certified"]
    assert flexible["objective_value"] <= single["objective_value"] + 1.0e-6
    assert flexible["objective_value"] + 1.0e-6 < single["objective_value"]
    assert single["metrics"]["maximum_active_sources_per_region_scenario"] <= 1
    assert flexible["metrics"]["robust_total_cost"] <= common_budget + 1.0e-5
    assert single["metrics"]["robust_total_cost"] <= common_budget + 1.0e-5
    print(
        {
            "status": "tiny_correctness_passed",
            "flexible_anchor": flexible_anchor["objective_value"],
            "single_anchor": single_anchor["objective_value"],
            "flexible_t": flexible["objective_value"],
            "single_t": single["objective_value"],
        }
    )


if __name__ == "__main__":
    test_flexible_policy_contains_single_source_and_can_strictly_improve_service()
