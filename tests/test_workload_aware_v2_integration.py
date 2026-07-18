from __future__ import annotations

import json

import pytest

from src.benders import _settings, certification_mip_gap, solve_benders
from src.instance import generate_instance


WORKLOAD_LOG_FIELDS = {
    "workload_policy_active",
    "workload_ema_decay",
    "workload_master_time_ema",
    "workload_subproblem_time_ema",
    "workload_master_share_raw",
    "workload_master_weight_selected",
    "workload_subproblem_weight_selected",
    "workload_master_ratio_selected",
    "workload_subproblem_ratio_selected",
    "workload_total_error_budget_ratio",
    "workload_fallback_used",
    "workload_fallback_reason",
}


def _config(policy: str = "workload_aware_joint") -> dict[str, object]:
    return {
        "seed": 141,
        "instance": {
            "num_warehouses": 2,
            "num_products": 2,
            "num_regions": 2,
            "budget_factor": 0.7,
        },
        "robust": {"gamma_target": 1, "gamma_schedule": [1], "max_scenarios": 100},
        "algorithm": {
            "subproblem_mode": "robust_dual_milp",
            "cut_selection_enabled": False,
            "adaptive_secondary_cut_selection_enabled": False,
            "adaptive_secondary_generation_enabled": False,
            "adaptive_subproblem_gap_enabled": False,
            "max_cuts_per_iteration": 1,
            "precision_policy": policy,
            "adaptive_master_precision_enabled": True,
            "adaptive_subproblem_precision_enabled": True,
            "master_gap_max": 0.02,
            "master_gap_min": 0.0001,
            "subproblem_gap_max": 0.05,
            "subproblem_gap_min": 0.0001,
            "fixed_master_mip_gap": 0.02,
            "fixed_subproblem_mip_gap": 0.05,
            "master_error_budget_ratio": 0.25,
            "subproblem_error_budget_ratio": 0.50,
            "monotone_precision_tightening": True,
            "workload_ema_decay": 0.80,
            "workload_total_error_budget_ratio": 0.75,
            "workload_master_weight_min": 1.0 / 3.0,
            "workload_master_weight_max": 2.0 / 3.0,
            "workload_time_epsilon": 1.0e-9,
            "workload_initial_master_weight": 1.0 / 3.0,
            "workload_initial_subproblem_weight": 2.0 / 3.0,
        },
        "benders": {
            "max_iterations": 3,
            "tol": 0.0,
            "initial_mip_gap": 0.02,
            "final_mip_gap": 1e-5,
            "time_limit": 30,
            "output_flag": False,
        },
    }


def test_v2_solver_logs_workload_state_and_json_serializable_metadata() -> None:
    config = _config()
    instance = generate_instance(config, seed=141)
    result = solve_benders(config, instance, "adaptive_gap_gamma_benders")

    assert result.iteration_log
    assert all(WORKLOAD_LOG_FIELDS <= set(row) for row in result.iteration_log)
    first = result.iteration_log[0]
    assert first["workload_policy_active"] is True
    assert first["workload_fallback_used"] is True
    assert first["workload_master_ratio_selected"] == pytest.approx(0.25)
    assert first["workload_subproblem_ratio_selected"] == pytest.approx(0.50)

    normal_rows = [row for row in result.iteration_log if row["workload_policy_active"]]
    master_gaps = [float(row["requested_master_mip_gap"]) for row in normal_rows]
    subproblem_gaps = [float(row["subproblem_requested_mip_gap"]) for row in normal_rows]
    assert master_gaps == sorted(master_gaps, reverse=True)
    assert subproblem_gaps == sorted(subproblem_gaps, reverse=True)
    assert result.metadata["workload_aware_policy_enabled"] is True
    assert result.metadata["workload_final_master_time_ema"] is not None
    assert result.metadata["workload_final_subproblem_time_ema"] is not None
    json.dumps(result.summary_dict(), allow_nan=False)


def test_v1_settings_and_logs_remain_free_of_dynamic_weights() -> None:
    config = _config("joint_error_budget")
    settings = _settings(config, "adaptive_gap_gamma_benders")
    assert settings.precision_config.precision_policy == "joint_error_budget"
    assert settings.workload_precision_config is None
    instance = generate_instance(config, seed=142)
    result = solve_benders(config, instance, "adaptive_gap_gamma_benders")
    assert result.metadata["workload_aware_policy_enabled"] is False
    assert result.metadata["workload_final_master_weight"] is None
    assert result.metadata["workload_mean_master_weight"] is None
    assert result.metadata["workload_fallback_count"] == 0
    assert all(row["workload_policy_active"] is False for row in result.iteration_log)
    assert all(row["workload_master_weight_selected"] is None for row in result.iteration_log)


def test_certification_forces_zero_without_redefining_policy_gap() -> None:
    assert certification_mip_gap(True, 0.02) == 0.0
    assert certification_mip_gap(True, 0.05) == 0.0
    assert certification_mip_gap(False, 0.02) == pytest.approx(0.02)
