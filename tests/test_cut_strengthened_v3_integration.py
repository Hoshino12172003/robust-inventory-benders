from __future__ import annotations

from copy import deepcopy
import json

import pytest

import src.benders as benders_module
from src.benders import _settings, solve_benders
from src.instance import generate_instance


V3_LOG_FIELDS = {
    "cut_strengthening_policy",
    "core_point_available",
    "core_point_observations",
    "core_point_attempted",
    "core_point_auxiliary_bound_used_for_UB",
    "v3_secondary_attempted",
    "v3_secondary_runtime",
    "v3_secondary_cut_added",
    "v3_secondary_objective_bound",
    "v3_secondary_bound_used_for_UB",
}


def _config(policy: str = "none") -> dict:
    return {
        "seed": 221,
        "instance": {
            "num_warehouses": 2,
            "num_products": 2,
            "num_regions": 2,
            "budget_factor": 0.75,
        },
        "robust": {"gamma_target": 1, "gamma_schedule": [1], "max_scenarios": 100},
        "algorithm": {
            "subproblem_mode": "robust_dual_milp",
            "cut_selection_enabled": False,
            "adaptive_secondary_cut_selection_enabled": False,
            "adaptive_secondary_generation_enabled": False,
            "adaptive_subproblem_gap_enabled": False,
            "precision_policy": "joint_error_budget",
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
            "final_certification_enabled": False,
            "cut_strengthening_policy": policy,
            "core_point_min_distance": 0.0,
            "core_point_min_remaining_time": 0.0,
            "core_point_min_global_gap": 0.0,
            "core_point_stage1_time_limit": 2.0,
            "core_point_stage2_time_limit": 2.0,
            "v3_secondary_lb_window": 1,
            "v3_secondary_stall_threshold": 1.0,
            "v3_secondary_cooldown_iterations": 0,
            "v3_secondary_min_global_gap": 0.0,
            "v3_secondary_min_remaining_time": 0.0,
            "v3_secondary_max_time_per_attempt": 2.0,
            "v3_secondary_max_time_fraction_of_remaining": 0.5,
            "v3_secondary_max_extra_time_share": 1.0,
            "v3_secondary_pattern_memory": 2,
            "max_cuts_per_iteration": 2 if "secondary" in policy else 1,
        },
        "benders": {
            "max_iterations": 3,
            "tol": 0.0,
            "initial_mip_gap": 0.02,
            "final_mip_gap": 0.0001,
            "time_limit": 30,
            "output_flag": False,
        },
    }


def test_policy_none_exactly_restores_v1_and_never_calls_v3_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implicit = _config()
    implicit["algorithm"].pop("cut_strengthening_policy")
    explicit = _config("none")
    instance = generate_instance(explicit, seed=221)

    helper_calls = 0
    real_helper = benders_module.solve_core_point_strengthened_dual_cut

    def counted_helper(*args, **kwargs):
        nonlocal helper_calls
        helper_calls += 1
        return real_helper(*args, **kwargs)

    robust_calls = 0
    real_robust = benders_module.solve_robust_dual_subproblem

    def counted_robust(*args, **kwargs):
        nonlocal robust_calls
        robust_calls += 1
        return real_robust(*args, **kwargs)

    monkeypatch.setattr(benders_module, "solve_core_point_strengthened_dual_cut", counted_helper)
    monkeypatch.setattr(benders_module, "solve_robust_dual_subproblem", counted_robust)
    explicit_result = solve_benders(explicit, instance, "adaptive_gap_gamma_benders")
    explicit_robust_calls = robust_calls
    implicit_result = solve_benders(implicit, instance, "adaptive_gap_gamma_benders")

    assert helper_calls == 0
    assert explicit_robust_calls == explicit_result.iterations
    assert robust_calls - explicit_robust_calls == implicit_result.iterations
    assert explicit_result.objective == pytest.approx(implicit_result.objective)
    assert explicit_result.lower_bound == pytest.approx(implicit_result.lower_bound)
    assert explicit_result.upper_bound == pytest.approx(implicit_result.upper_bound)
    assert explicit_result.metadata["cut_strengthening_policy"] == "none"
    assert explicit_result.metadata["v3_total_extra_cut_runtime"] == 0.0


def test_core_only_logs_are_serializable_and_auxiliary_never_updates_ub() -> None:
    config = _config("core_point")
    instance = generate_instance(config, seed=222)
    result = solve_benders(config, instance, "adaptive_gap_gamma_benders")
    assert result.iteration_log
    assert all(V3_LOG_FIELDS <= set(row) for row in result.iteration_log)
    assert all(row["core_point_auxiliary_bound_used_for_UB"] is False for row in result.iteration_log)
    assert all(row["v3_secondary_attempted"] is False for row in result.iteration_log)
    assert result.metadata["v3_secondary_solve_count"] == 0
    json.dumps(result.summary_dict(), allow_nan=False)
    json.dumps(result.iteration_log, allow_nan=False)


def test_secondary_only_uses_at_most_two_cuts_and_never_uses_restricted_bound_for_ub() -> None:
    config = _config("stall_secondary")
    instance = generate_instance(config, seed=223)
    result = solve_benders(config, instance, "adaptive_gap_gamma_benders")
    assert all(row["cuts_added_this_iteration"] <= 2 for row in result.iteration_log)
    assert all(row["v3_secondary_bound_used_for_UB"] is False for row in result.iteration_log)
    assert result.metadata["core_point_attempt_count"] == 0
    assert result.metadata["v3_secondary_cuts_added"] <= result.metadata["v3_secondary_solve_count"]


def test_secondary_cut_never_counts_as_useful_primary_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("stall_secondary")
    instance = generate_instance(config, seed=224)
    captured: list[bool] = []
    original = benders_module.update_final_certification

    def capture(*args, **kwargs):
        captured.append(bool(kwargs["useful_primary_cut_added"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(benders_module, "update_final_certification", capture)
    result = solve_benders(config, instance, "adaptive_gap_gamma_benders")
    assert captured == [bool(row["cut_added"]) for row in result.iteration_log]


def test_full_v3_keeps_v1_precision_policy_and_metadata_contract() -> None:
    config = _config("core_point_stall_secondary")
    settings = _settings(config, "adaptive_gap_gamma_benders")
    assert settings.precision_config.precision_policy == "joint_error_budget"
    assert settings.cut_strengthening_config.core_point_enabled
    assert settings.cut_strengthening_config.secondary_enabled
    instance = generate_instance(config, seed=225)
    result = solve_benders(config, instance, "adaptive_gap_gamma_benders")
    required_metadata = {
        "cut_strengthening_policy",
        "core_point_attempt_count",
        "core_point_success_count",
        "core_point_fallback_count",
        "core_point_total_runtime",
        "v3_secondary_trigger_count",
        "v3_secondary_solve_count",
        "v3_secondary_cut_added_count",
        "v3_secondary_total_runtime",
        "v3_total_extra_cut_runtime",
        "v3_extra_cut_runtime_share",
        "v3_primary_cuts_added",
        "v3_secondary_cuts_added",
    }
    assert required_metadata <= set(result.metadata)
    assert result.metadata["precision_policy"] == "joint_error_budget"
    assert result.metadata["cut_strengthening_policy"] == "core_point_stall_secondary"
    assert result.metadata["valid_UB"] in {True, False}
