from __future__ import annotations

import json

import pytest

from src.instance import InventoryInstance
from src.managerial_evaluation import (
    evaluate_managerial_solution,
    invalid_managerial_evaluation,
    summarize_managerial_metrics,
)


def _instance() -> InventoryInstance:
    return InventoryInstance(
        name="managerial_test",
        num_warehouses=2,
        num_products=2,
        num_regions=2,
        fixed_cost=[10.0, 20.0],
        inventory_cost=[[1.0, 2.0], [3.0, 4.0]],
        capacity=[100.0, 100.0],
        volume=[1.0, 1.0],
        budget=1000.0,
        transport_cost=[
            [[1.0, 2.0], [1.5, 2.5]],
            [[2.0, 3.0], [2.5, 3.5]],
        ],
        shortage_penalty=[[5.0, 6.0], [7.0, 8.0]],
        service_penalty=[9.0, 10.0],
        service_level=[0.9, 0.9],
        base_demand=[[10.0, 20.0], [30.0, 40.0]],
        demand_deviation=[[1.0, 2.0], [3.0, 4.0]],
        inventory_ub=[[100.0, 100.0], [100.0, 100.0]],
    )


def test_managerial_metrics_inventory_opening_shortage_fill_and_costs() -> None:
    result = summarize_managerial_metrics(
        instance=_instance(),
        y_values=[1.0, 0.0],
        x_values=[[5.0, 6.0], [7.0, 8.0]],
        demand_values=[[10.0, 20.0], [30.0, 40.0]],
        active_deviations=[{"region": 1, "product": 0}],
        recourse_objective=50.0,
        transport_cost=20.0,
        shortage_cost=21.0,
        service_violation_cost=9.0,
        shortage_values=[[1.0, 2.0], [3.0, 4.0]],
        service_violation_values=[0.5, 1.5],
        runtime=0.2,
    )

    assert result.managerial_metrics_valid is True
    assert result.opened_warehouses == 1
    assert result.total_inventory == 26.0
    assert result.inventory_by_product == [12.0, 14.0]
    assert result.inventory_by_warehouse == [11.0, 15.0]
    assert result.fixed_opening_cost == 10.0
    assert result.inventory_cost == 70.0
    assert result.first_stage_cost == 80.0
    assert result.total_shortage == 10.0
    assert result.shortage_by_product == [4.0, 6.0]
    assert result.service_violation == 2.0
    assert result.realized_fill_rate == pytest.approx(0.9)
    assert result.worst_case_recourse_cost == 50.0
    assert result.transport_cost + result.shortage_cost + result.service_violation_cost == 50.0


def test_zero_demand_fill_rate_is_explicitly_null() -> None:
    instance = _instance()
    result = summarize_managerial_metrics(
        instance=instance,
        y_values=[0.0, 0.0],
        x_values=[[0.0, 0.0], [0.0, 0.0]],
        demand_values=[[0.0, 0.0], [0.0, 0.0]],
        active_deviations=[],
        recourse_objective=0.0,
        transport_cost=0.0,
        shortage_cost=0.0,
        service_violation_cost=0.0,
        shortage_values=[[0.0, 0.0], [0.0, 0.0]],
        service_violation_values=[0.0, 0.0],
        runtime=0.0,
    )
    assert result.total_worst_case_demand == 0.0
    assert result.realized_fill_rate is None


def test_failure_result_has_no_invented_metrics_and_is_json_serializable() -> None:
    result = invalid_managerial_evaluation("robust_dual_time_limit", "no optimum", 3.0)
    payload = result.to_dict()

    assert result.managerial_metrics_valid is False
    assert result.total_shortage is None
    assert result.service_violation is None
    assert result.opened_warehouses is None
    json.dumps(payload)


def test_missing_first_stage_incumbent_fails_without_calling_gurobi() -> None:
    result = evaluate_managerial_solution(
        _instance(),
        best_y_values=None,
        best_x_values=None,
        gamma_target=2,
        time_limit=1.0,
    )
    assert result.managerial_metrics_valid is False
    assert result.managerial_evaluation_status == "missing_first_stage_incumbent"
    assert result.total_inventory is None
