from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest

from src.fairness_hybrid_gamma_attempt3_final import (
    AuditError,
    GAMMAS,
    SEEDS,
    SCALES,
    _directory_id,
    _project,
    _run_key,
    audit_gamma_archive,
)


ARCHIVE = Path(r"E:\论文代码\fairness_hybrid_gamma_sensitivity_attempt3_results.zip")


def _record(task_type: str = "frontier") -> dict:
    result = {
        "algorithm_runtime": 1.0,
        "master_runtime": 0.4,
        "separation_runtime": 0.5,
        "post_evaluation_wall_runtime": 0.2,
        "total_wall_runtime": 1.2,
        "penalized_runtime_par2": 1.0,
        "iterations": 2,
    }
    if task_type == "frontier":
        result.update({
            "x_values": [[1.0, 2.0], [3.0, 4.0]],
            "y_values": [1.0, 0.0],
            "objective_t": 0.1,
            "robust_minimum_fill_rate": 0.9,
            "cuts": 2,
            "metadata": {"committed_scenario_count": 3},
            "post_evaluation": {
                "valid": True,
                "actual_robust_cost": 101.0,
                "actual_price_of_fairness": 0.01,
                "wminfr": 0.9,
                "minimum_weighted_mean_fill_rate": 0.95,
            },
        })
    else:
        result.update({"best_x_values": [[1.0, 2.0], [3.0, 4.0]], "best_y_values": [1.0, 0.0], "upper_bound": 100.0})
    return {
        "task_type": task_type,
        "scientific_status": "certified_robust_optimal",
        "baseline_robust_cost": 100.0,
        "cost_budget": 102.5,
        "result": result,
    }


def _instance() -> dict:
    return {"identity": {}, "instance": {"num_warehouses": 2, "num_products": 2}}


def test_frozen_plan_has_sixty_unique_keys_and_directories() -> None:
    keys = [_run_key(scale, seed, gamma, task) for scale in SCALES for seed in SEEDS for gamma in GAMMAS for task in ("baseline", "frontier")]
    assert len(keys) == len(set(keys)) == 60
    assert len({_directory_id(key) for key in keys}) == 60
    assert all(json.loads(key)["gamma"] in GAMMAS for key in keys)


def test_production_projection_uses_strict_matrix_and_vector_schema() -> None:
    row = _project(_record(), _instance())
    assert row["inventory"] == 10.0
    assert row["opened_warehouses"] == 1
    assert row["scenario_block_count"] == 3


@pytest.mark.parametrize(
    "value",
    [
        [[1.0, 2.0]],
        [[1.0], [2.0]],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[True, 2.0], [3.0, 4.0]],
        [["1", 2.0], [3.0, 4.0]],
        [[math.nan, 2.0], [3.0, 4.0]],
        [[math.inf, 2.0], [3.0, 4.0]],
    ],
)
def test_projection_rejects_dimension_and_numeric_drift(value: object) -> None:
    record = _record()
    record["result"]["x_values"] = value
    with pytest.raises(AuditError):
        _project(record, _instance())


def test_projection_rejects_y_schema_drift() -> None:
    record = _record()
    record["result"]["y_values"] = [1.0, True]
    with pytest.raises(AuditError):
        _project(record, _instance())


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="formal read-only archive is not mounted")
def test_formal_archive_full_solver_free_audit() -> None:
    report, rows = audit_gamma_archive(ARCHIVE)
    assert report["decision"] == "approve_gamma_sensitivity_attempt3"
    assert report["coverage"]["chunk"] == 1350
    assert report["coverage"]["acceptance_evidence"] == 413220
    assert report["post_evaluation"]["maximum_acceptance_residual"] == pytest.approx(2.9103830456733704e-11)
    assert len(rows) == 60
    assert "gurobipy" not in sys.modules
