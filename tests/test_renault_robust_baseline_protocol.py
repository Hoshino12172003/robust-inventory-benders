from __future__ import annotations

from src.renault_robust_baseline_runner import CONFIG_PATHS, EXPECTED_CASE_ORDER, preflight


def test_only_two_b1_full_gamma2_cases_are_registered() -> None:
    contexts = [preflight(path) for path in CONFIG_PATHS]
    assert tuple(item["protocol"]["case_id"] for item in contexts) == EXPECTED_CASE_ORDER
    for context in contexts:
        protocol = context["protocol"]
        assert protocol["instance_path"].endswith("_B1.00.json")
        assert protocol["eligibility_mode"] == "Full"
        assert protocol["gamma"] == 2
        assert protocol["expected_complete_scenarios"] == 4657


def test_solver_resolution_matches_frozen_final_policy() -> None:
    for path in CONFIG_PATHS:
        context = preflight(path)
        config = context["solver_config"]
        assert context["solver_method"] == "adaptive_gap_gamma_benders"
        assert config["robust"]["gamma_target"] == 2
        assert config["robust"]["gamma_schedule"] == [2]
        assert config["algorithm"]["subproblem_mode"] == "robust_dual_milp"
        assert config["algorithm"]["precision_policy"] == "joint_error_budget"
        assert config["algorithm"]["cut_strengthening_policy"] == "core_point"
        assert config["algorithm"]["final_certification_enabled"] is True
        assert config["algorithm"]["max_cuts_per_iteration"] == 1
        assert config["benders"]["time_limit"] == 1800
        assert config["benders"]["max_iterations"] == 20000
        assert config["benders"]["tol"] == 1.0e-4
