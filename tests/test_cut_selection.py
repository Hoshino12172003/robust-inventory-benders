from __future__ import annotations

from src.benders import solve_benders
from src.instance import generate_instance


# Tests keep cut selection audit fields visible without changing algorithm behavior.
def cut_selection_config() -> dict:
    return {
        "seed": 31,
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
        "algorithm": {
            "subproblem_mode": "robust_dual_milp",
            "cut_selection_enabled": True,
            "delta_cut": 0.0,
            "cut_violation_tol": 1e-8,
        },
        "benders": {
            "max_iterations": 40,
            "tol": 1e-4,
            "initial_mip_gap": 0.05,
            "final_mip_gap": 1e-6,
            "time_limit": 60,
            "output_flag": False,
        },
    }


def test_default_cut_selection_keeps_benders_progress() -> None:
    config = cut_selection_config()
    instance = generate_instance(config, seed=31)

    result = solve_benders(config, instance, "standard_benders")

    assert result.status in {"optimal", "iteration_limit", "time_limit"}
    assert result.metadata["cut_selection_enabled"] is True
    assert "cuts_added_total" in result.metadata
    assert "cuts_skipped_total" in result.metadata
    assert result.metadata["cuts_added_total"] >= 1


def test_high_delta_cut_selection_skips_low_violation_cut() -> None:
    config = cut_selection_config()
    config["algorithm"]["delta_cut"] = 1e9
    config["benders"]["max_iterations"] = 1
    instance = generate_instance(config, seed=32)

    result = solve_benders(config, instance, "adaptive_gap_gamma_benders")

    assert result.metadata["cuts_skipped_total"] >= 1
    assert any(not row["cut_added"] for row in result.iteration_log)
    assert any(row["cut_skip_reason"] == "low_violation" for row in result.iteration_log)


def test_disabled_cut_selection_keeps_old_add_cut_behavior() -> None:
    config = cut_selection_config()
    config["algorithm"]["cut_selection_enabled"] = False
    config["algorithm"]["delta_cut"] = 1e9
    config["benders"]["max_iterations"] = 2
    instance = generate_instance(config, seed=33)

    result = solve_benders(config, instance, "standard_benders")

    assert result.metadata["cut_selection_enabled"] is False
    assert result.metadata["cuts_added_total"] >= 1
    assert result.metadata["cuts_skipped_total"] == 0
    assert all(row["cut_added"] for row in result.iteration_log)


def test_cut_violation_log_matches_cut_rhs_minus_theta() -> None:
    config = cut_selection_config()
    instance = generate_instance(config, seed=34)

    result = solve_benders(config, instance, "standard_benders")
    first = result.iteration_log[0]

    assert abs(first["cut_violation"] - (first["cut_rhs_current"] - first["theta"])) <= 1e-6
    assert abs(first["cut_rhs_current"] - first["active_subproblem_value"]) <= 1e-5


def test_objective_bound_is_not_used_for_cut_generation() -> None:
    config = cut_selection_config()
    instance = generate_instance(config, seed=35)

    result = solve_benders(config, instance, "standard_benders")
    first = result.iteration_log[0]

    assert "target_subproblem_objective_bound" in first
    assert first["cut_rhs_current"] <= first["active_subproblem_value"] + 1e-6
