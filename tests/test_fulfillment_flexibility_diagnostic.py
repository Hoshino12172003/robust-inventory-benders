from __future__ import annotations

from src.fulfillment_flexibility_diagnostic import build_eligibility, solve_cost_anchor, solve_service
from src.instance import InventoryInstance
from src.monolithic import solve_monolithic
from src.robust_regional_fairness import solve_fairness_extensive_form
from src.scenarios import DemandScenario, enumerate_budget_scenarios


def instance() -> InventoryInstance:
    return InventoryInstance(
        name="flex_test", num_warehouses=3, num_products=1, num_regions=2,
        fixed_cost=[1, 1, 1], inventory_cost=[[1], [1], [1]], capacity=[8, 8, 8],
        volume=[1], budget=50, transport_cost=[[[1], [5]], [[2], [1]], [[3], [2]]],
        shortage_penalty=[[20], [20]], service_penalty=[30], service_level=[0.8],
        base_demand=[[5], [5]], demand_deviation=[[5], [5]], inventory_ub=[[8], [8], [8]])


def scenarios():
    return [DemandScenario("a", ((0, 0),), ((10.0,), (5.0,))),
            DemandScenario("b", ((1, 0),), ((5.0,), (10.0,)))]


SETTINGS = {"time_limit": 30, "mip_gap": 1e-8, "feasibility_tolerance": 1e-8,
            "threads": 1, "solver_seed": 0}


def test_eligibility_construction():
    assert build_eligibility(instance(), "k1") == {0: (0,), 1: (1,)}
    assert build_eligibility(instance(), "k2") == {0: (0, 1), 1: (1, 2)}
    assert build_eligibility(instance(), "full") == {0: (0, 1, 2), 1: (0, 1, 2)}


def test_arc_restriction_and_mode_consistency():
    inst, sc = instance(), scenarios()
    anchors = {mode: solve_cost_anchor(inst, sc, mode, SETTINGS) for mode in ("k1", "k2", "full")}
    results = {mode: solve_service(inst, sc, mode, anchors[mode]["objective"], 0.025, SETTINGS)
               for mode in ("k1", "k2", "full")}
    assert all(value["certified"] for value in results.values())
    assert results["k1"]["metrics"]["active_warehouse_region_arcs"] <= 2
    assert results["k2"]["metrics"]["active_warehouse_region_arcs"] <= 4
    assert results["full"]["objective_t"] <= results["k2"]["objective_t"] + 1e-7


def test_fixed_first_stage_evaluation():
    inst, sc = instance(), scenarios()
    anchor = solve_cost_anchor(inst, sc, "full", SETTINGS)
    full = solve_service(inst, sc, "full", anchor["objective"], 0.025, SETTINGS)
    fixed_anchors = {
        mode: solve_cost_anchor(
            inst, sc, mode, SETTINGS,
            fixed_y=full["y_values"], fixed_x=full["x_values"]
        )
        for mode in ("k1", "k2", "full")
    }
    common_anchor = max(value["objective"] for value in fixed_anchors.values())
    fixed = {mode: solve_service(inst, sc, mode, common_anchor, 0.025, SETTINGS,
                                 fixed_y=full["y_values"], fixed_x=full["x_values"])
             for mode in ("k1", "k2", "full")}
    assert all(value["certified"] for value in fixed.values())
    assert fixed["full"]["objective_t"] <= fixed["k2"]["objective_t"] + 1e-7
    assert fixed["k2"]["objective_t"] <= fixed["k1"]["objective_t"] + 1e-7


def test_full_mode_regresses_to_original_extensive_form():
    inst = instance()
    sc = enumerate_budget_scenarios(inst, 1, max_scenarios=50, exact_scenarios=True)
    config = {
        "robust": {"gamma_target": 1, "max_scenarios": 50, "exact_scenarios": True},
        "benders": {"time_limit": 30, "output_flag": False},
    }
    original_anchor = solve_monolithic(config, inst)
    diagnostic_anchor = solve_cost_anchor(inst, sc, "full", SETTINGS)
    assert original_anchor.objective is not None
    assert diagnostic_anchor["certified"]
    assert abs(original_anchor.objective - diagnostic_anchor["objective"]) <= 1e-6

    original_service = solve_fairness_extensive_form(
        inst,
        baseline_cost=diagnostic_anchor["objective"],
        rho=0.025,
        gamma=1,
        max_scenarios=50,
        time_limit=30,
        mip_gap=1e-8,
        lexicographic_cost_stage=False,
    )
    diagnostic_service = solve_service(
        inst, sc, "full", diagnostic_anchor["objective"], 0.025, SETTINGS
    )
    assert original_service.status == "optimal"
    assert diagnostic_service["certified"]
    assert abs(original_service.objective_t - diagnostic_service["objective_t"]) <= 1e-7
