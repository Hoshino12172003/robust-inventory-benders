from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .experiment_protocol import file_sha256
from .experiment_suite import INSTANCE_SIZES
from .fairness_benders import development_run_plan
from .scenarios import count_budget_scenarios


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = {
    "regional_fairness_development_medium_large": ROOT / "experiments/configs/regional_fairness_development_medium_large.yaml",
    "regional_fairness_development_large": ROOT / "experiments/configs/regional_fairness_development_large.yaml",
}
FROZEN_HASHES = {
    "experiments/configs/selected_algorithm_parameters.yaml": "50B275578A127B349BDDA47FF161680048CD1C0C8845EA85E707949BDFA29D25",
    "experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml": "7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6",
    "docs/cut_strengthened_joint_v3_final_decision.md": "1E9EB741056331CCCC5A456BFA7858C9FB1B423C3C5DC904B602315B23B72594",
    "docs/regional_fairness_diagnostic_decision.md": "4676C64C4B09DE26A246B6ECF104B16FF9F0B84F3DDFAAC62808869A438A04C7",
    "experiments/configs/regional_fairness_diagnostic_medium_large.yaml": "04D2CA32C31D7B2D3C9071583C4BC3897740B463D6AD945A8A52554A6317C79C",
    "experiments/configs/regional_fairness_diagnostic_large.yaml": "7A40FF6CFEDB02F44D57C999377377B7EB25E406EBE417791CA7A0C22C2FB307",
    "docs/regional_fairness_diagnostic_protocol.md": "EC7761D96C1D2A17F96EBA90BF4BFB520A9CE6359F938ACD7F294A10E7F24A38",
    "src/benders.py": "37967750EE1AAD5575A9B1FE0B050F012EC21DB58FA277FBEFAA5A48CFEF1D9F",
    "src/subproblem.py": "63AACB578BA5C2131424D5C103E3B8F7AA4408028670329E4A724EE43CB69EC1",
    "src/robust_dual_subproblem.py": "EC20EE9A736585AD0E2273FD77D5A362FB75E900B7997DE9C60F6CC3AED16008",
    "src/scenarios.py": "7294C60DC318F7678F8A4464DAF2CBD85E540842C6C3858BB1D30A9DE7915511",
}
EXPECTED_SEEDS = list(range(120, 130))
EXPECTED_RHOS = [0.0, 0.01, 0.025, 0.05, 0.10]
EXPECTED_CANDIDATE_HASH = FROZEN_HASHES[
    "experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml"
]
EXPECTED_ATTEMPT3_PROTOCOL_HASH = (
    "A3B13526778DE8049A03F47B01825474ABC562CB9E67F2355717435D3754FA5F"
)


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping.")
    return value


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = "") -> None:
    checks.append({"check": name, "passed": bool(passed), "required": True, "detail": detail})


def _finite(config: Mapping[str, Any], field: str, lower: float, upper: float | None = None) -> bool:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    return float(value) >= lower and (upper is None or float(value) <= upper)


def _without_allowed_scale_differences(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    for field in (
        "experiment_name",
        "output_dir",
        "instance_sizes",
        "baseline_time_limit",
        "fairness_time_limit",
        "time_limit",
        "max_iterations",
    ):
        value.pop(field, None)
    return value


def audit_fairness_development(
    *,
    config_overrides: Mapping[str, dict[str, Any]] | None = None,
    allow_existing_output: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "attempt3_protocol_sha256",
        file_sha256(ROOT / "docs/regional_fairness_development_protocol.md").upper()
        == EXPECTED_ATTEMPT3_PROTOCOL_HASH,
        file_sha256(ROOT / "docs/regional_fairness_development_protocol.md").upper(),
    )
    configs = {
        name: deepcopy(config_overrides[name]) if config_overrides and name in config_overrides else _load(path)
        for name, path in CONFIG_PATHS.items()
    }
    for relative, expected in FROZEN_HASHES.items():
        path = ROOT / relative
        actual = file_sha256(path).upper() if path.exists() else "missing"
        _check(checks, f"frozen_{relative.replace('/', '_').replace('.', '_')}", actual == expected, actual)

    candidate = _load(ROOT / "experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml")
    candidate_algorithm = candidate["algorithm"]
    algorithm_fields = (
        "precision_policy",
        "adaptive_master_precision_enabled",
        "adaptive_subproblem_precision_enabled",
        "master_gap_max",
        "master_gap_min",
        "subproblem_gap_max",
        "subproblem_gap_min",
        "fixed_master_mip_gap",
        "fixed_subproblem_mip_gap",
        "master_error_budget_ratio",
        "subproblem_error_budget_ratio",
        "monotone_precision_tightening",
        "cut_strengthening_policy",
        "core_point_update_weight",
        "core_point_min_distance",
        "core_point_stage1_time_limit",
        "core_point_stage2_time_limit",
        "core_point_min_remaining_time",
        "core_point_min_global_gap",
        "core_point_current_abs_tol",
        "core_point_current_rel_tol",
        "core_point_min_normalized_improvement",
        "max_cuts_per_iteration",
        "cut_selection_enabled",
        "adaptive_secondary_cut_selection_enabled",
        "adaptive_secondary_generation_enabled",
        "adaptive_subproblem_gap_enabled",
        "adaptive_gap_enabled",
        "final_certification_enabled",
        "final_certification_no_cut_patience",
    )
    for name, config in configs.items():
        prefix = name.replace("regional_fairness_development_", "")
        expected_size = "medium_large" if prefix == "medium_large" else "large"
        _check(checks, f"{prefix}_experiment_identity", config.get("experiment_name") == name)
        _check(checks, f"{prefix}_phase", config.get("protocol_phase") == "fairness_model_development")
        _check(checks, f"{prefix}_authorization", config.get("authorization") == "fairness_model_development_protocol_only")
        _check(checks, f"{prefix}_seeds_exact_120_129", config.get("random_seeds") == EXPECTED_SEEDS)
        _check(checks, f"{prefix}_size", config.get("instance_sizes") == [expected_size])
        _check(
            checks,
            f"{prefix}_only_frozen_candidate",
            config.get("variants") == ["joint_v1_core_point_strengthened"]
            and config.get("baseline_method") == "joint_v1_core_point_strengthened",
        )
        _check(checks, f"{prefix}_candidate_hash", str(config.get("candidate_config_sha256", "")).upper() == EXPECTED_CANDIDATE_HASH)
        _check(
            checks,
            f"{prefix}_candidate_parameters_frozen",
            all(config.get(field) == candidate_algorithm.get(field) for field in algorithm_fields),
        )
        _check(
            checks,
            f"{prefix}_gamma_frozen",
            config.get("gamma_target") == 2
            and config.get("gamma_schedule") == [2]
            and config.get("gamma_continuation_enabled") is False,
        )
        _check(
            checks,
            f"{prefix}_legacy_features_off",
            config.get("cut_selection_enabled") is False
            and config.get("adaptive_secondary_cut_selection_enabled") is False
            and config.get("adaptive_secondary_generation_enabled") is False,
        )
        fairness = config.get("fairness_development", {})
        _check(checks, f"{prefix}_rho_grid", fairness.get("rho_grid") == EXPECTED_RHOS)
        _check(
            checks,
            f"{prefix}_same_recourse",
            fairness.get("same_recourse_for_cost_and_fairness") is True,
        )
        _check(
            checks,
            f"{prefix}_separation_mode",
            fairness.get("separation_mode") == "budgeted_uncertainty_farkas_milp"
            and fairness.get("separation_requires_objective_bound_for_certification") is True,
        )
        _check(
            checks,
            f"{prefix}_certified_anchor",
            fairness.get("baseline_cost_anchor_source") == "solve_result.upper_bound"
            and fairness.get("baseline_cost_anchor_requires_valid_ub") is True
            and fairness.get("baseline_cost_anchor_precision") == "ieee754_float_and_hex",
        )
        _check(
            checks,
            f"{prefix}_ray_and_status_certification",
            fairness.get("farkas_ray_normalization") == "sum_multipliers_equals_one"
            and fairness.get("certifiable_separation_statuses") == ["optimal", "time_limit"],
        )
        _check(
            checks,
            f"{prefix}_core_boundary",
            fairness.get("baseline_core_point_strengthening_enabled") is True
            and fairness.get("fairness_cut_core_point_strengthening_enabled") is False,
        )
        _check(
            checks,
            f"{prefix}_tolerances",
            all(
                _finite(fairness, field, 0.0)
                for field in (
                    "lexicographic_T_absolute_tolerance",
                    "cost_absolute_tolerance",
                    "cost_relative_tolerance",
                    "feasibility_tolerance",
                    "metric_tolerance",
                    "post_evaluation_time_limit_per_scenario",
                )
            ),
        )
        reserved = fairness.get("reserved_future_seeds", {})
        continue_rule = fairness.get("development_continue_rule", {})
        _check(
            checks,
            f"{prefix}_development_decision_rule",
            continue_rule.get("correctness_required") is True
            and continue_rule.get("minimum_solved_rate_each_scale") == 0.80
            and continue_rule.get("material_minimum_fill_rate_improvement") == 0.05
            and continue_rule.get("minimum_instances_improved") == 4
            and continue_rule.get("eligible_positive_rho") == [0.01, 0.025, 0.05, 0.10]
            and continue_rule.get("candidate_selection_rule") == "smallest_eligible_positive_rho"
            and continue_rule.get("validation_allowed_changes")
            == ["experiment_name", "protocol_phase", "random_seeds", "output_dir"],
        )
        _check(
            checks,
            f"{prefix}_future_seeds_reserved",
            reserved.get("validation") == list(range(130, 140))
            and reserved.get("final_medium_large") == list(range(140, 150))
            and reserved.get("final_large") == list(range(150, 160)),
        )
        all_configured_seeds = set(config.get("random_seeds", []))
        _check(
            checks,
            f"{prefix}_no_diagnostic_or_future_seed_use",
            not all_configured_seeds.intersection(set(range(110, 120)) | set(range(130, 160))),
        )
        plan = development_run_plan(config)
        _check(checks, f"{prefix}_baseline_run_count_10", plan["baseline_run_count"] == 10)
        _check(checks, f"{prefix}_frontier_run_count_50", plan["fairness_frontier_run_count"] == 50)
        expected_scenarios = 1831 if expected_size == "medium_large" else 4657
        _check(checks, f"{prefix}_scenario_count", plan["scenario_count_by_size"].get(expected_size) == expected_scenarios)
        output = ROOT / str(config.get("output_dir", ""))
        _check(
            checks,
            f"{prefix}_formal_output_absent",
            allow_existing_output or not output.exists(),
            str(output),
        )
        _check(
            checks,
            f"{prefix}_output_isolated",
            "results_regional_fairness_model/development_" in str(config.get("output_dir", "")).replace("\\", "/"),
        )

    medium = configs["regional_fairness_development_medium_large"]
    large = configs["regional_fairness_development_large"]
    _check(
        checks,
        "scale_configs_only_allowed_differences",
        _without_allowed_scale_differences(medium) == _without_allowed_scale_differences(large),
    )
    _check(
        checks,
        "development_outputs_do_not_overlap",
        medium.get("output_dir") != large.get("output_dir"),
    )
    future_configs = list((ROOT / "experiments/configs").glob("regional_fairness_*validation*.yaml")) + list(
        (ROOT / "experiments/configs").glob("regional_fairness_*final*.yaml")
    )
    _check(checks, "no_validation_or_final_configs_created", not future_configs, [str(path) for path in future_configs])
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    _check(checks, "fairness_model_results_ignored", "experiments/results_regional_fairness_model/" in gitignore)
    model_doc = (ROOT / "docs/robust_regional_fairness_model.md").read_text(encoding="utf-8")
    protocol_doc = (ROOT / "docs/regional_fairness_development_protocol.md").read_text(encoding="utf-8")
    _check(checks, "model_prohibits_recourse_splicing", "prohibited" in model_doc and "same" in model_doc)
    _check(
        checks,
        "transport_cost_not_distance",
        "not a physical distance" in model_doc
        and "not physical distance" in " ".join(protocol_doc.split()),
    )
    _check(checks, "no_social_vulnerability_claim", "cannot support claims" in model_doc and "cannot make claims" in protocol_doc)
    _check(checks, "negative_result_rule_frozen", "stop_no_material_improvement" in protocol_doc)
    _check(
        checks,
        "candidate_selection_rule_frozen",
        "smallest eligible positive rho" in protocol_doc
        and "Allowed validation changes" in protocol_doc,
    )
    runner = (ROOT / "src/fairness_benders.py").read_text(encoding="utf-8")
    separation = (ROOT / "src/robust_regional_fairness.py").read_text(encoding="utf-8")
    _check(
        checks,
        "certified_anchor_implemented",
        '"source": "solve_result.upper_bound"' in runner
        and 'result.get("valid_UB") is True' in runner,
    )
    _check(
        checks,
        "single_writer_and_atomic_manifest",
        "SingleWriterLock" in runner
        and "fairness_development_manifest.json" in runner
        and "atomic_write_json" in runner,
    )
    _check(
        checks,
        "separation_candidate_requires_fixed_scenario_certificate",
        "certify_fixed_scenario_fairness_feasibility" in separation
        and "fixed_scenario_normalized_farkas_lp" in separation
        and "candidate_scenario_only" in runner,
    )
    _check(
        checks,
        "invalid_milp_ray_never_directly_generates_cut",
        "ray=fixed.ray" in separation
        and "cut=incumbent_cut" not in separation
        and 'cut_certificate_source="fixed_scenario_normalized_farkas_lp"' in separation
        and "Separation incumbent did not define a valid normalized Farkas ray" not in separation,
    )
    _check(
        checks,
        "fresh_execution_manifest_records_correctness_restart",
        '"execution_restart_after_correctness_hotfix": True' in runner
        and '"previous_attempt_scientifically_invalid": True' in runner
        and '"previous_attempt_results_reused": False' in runner
        and "EXECUTION_ATTEMPT = 3" in runner
        and "PREVIOUS_ATTEMPT_SEEDS = list(range(120, 130))" in runner
        and '"prior_attempts": PRIOR_ATTEMPTS' in runner,
    )
    _check(
        checks,
        "fresh_execution_manifest_records_post_evaluation_restart",
        '"execution_restart_after_post_evaluation_hotfix": True' in runner
        and '"previous_attempt2_scientifically_invalid": True' in runner
        and '"previous_attempt2_results_reused": False' in runner
        and "POST_EVALUATION_INVALID_ATTEMPT_SEEDS = list(range(120, 130))" in runner,
    )
    _check(
        checks,
        "post_evaluation_uses_nextafter_only_acceptance_boundary",
        "math.nextafter(allowed, math.inf)" in separation
        and '"floating_point_slack": slack' in separation
        and "32.0 * max" not in separation
        and "math.ulp(" not in separation,
    )
    _check(
        checks,
        "post_evaluation_records_constraint_acceptance_evidence",
        '"acceptance_threshold": threshold' in separation
        and '"lhs": left' in separation
        and '"rhs": right' in separation
        and '"raw_residual": raw_residual' in separation
        and '"constraint_type": str(constraint_type)' in separation
        and '"scenario_id": scenario_id' in separation
        and '"region_id": region_id' in separation,
    )
    _check(
        checks,
        "attempt3_requires_fresh_output_and_prohibits_overwrite",
        "def _prepare_attempt3_output(" in runner
        and "Fresh Attempt 3 requires an output directory that does not exist." in runner
        and "Attempt 3 resume requires a complete identity manifest." in runner
        and "--overwrite is prohibited for frozen fairness development." in runner,
    )
    _check(
        checks,
        "fixed_scenario_primal_uses_frozen_feasibility_tolerance",
        "primal.Params.FeasibilityTol = tolerance" in separation
        and "maximum_primal_constraint_violation" in separation,
    )
    _check(
        checks,
        "false_positive_exclusions_are_local_and_auditable",
        "excluded_candidate_evidence: list[dict[str, Any]] = []" in separation
        and "exclude_fixed_feasible_candidate" in separation
        and "fixed_primal_feasible_but_separation_incumbent_ray_valid" in separation,
    )
    _check(
        checks,
        "runner_atomically_records_certificate_exceptions",
        "def _record_failed_task(" in runner
        and 'state="failed"' in runner
        and "except BaseException as error" in runner,
    )
    _check(
        checks,
        "post_evaluation_caps_are_not_relaxed_by_tolerance",
        "recourse_cost <= remaining," in separation
        and "<= float(t_value) * demand" in separation
        and "remaining + float(feasibility_tolerance)" not in separation
        and "float(t_value) + float(feasibility_tolerance)" not in separation
        and "model.Params.FeasibilityTol = tolerance" in separation,
    )
    _check(
        checks,
        "frontier_status_distinguishes_algorithm_and_end_to_end_certification",
        "def fairness_frontier_overall_status(" in runner
        and '"certified_robust_optimal"' in runner
        and '"master_optimal_but_robust_uncertified"' in runner
        and '"time_limit_uncertified"' in runner
        and '"iteration_limit_uncertified"' in runner
        and '"certified_infeasible"' in runner
        and '"infeasible_uncertified"' in runner
        and '"invalid_post_evaluation"' in runner
        and '"numerical_uncertified"' in runner
        and '"interrupted"' in runner
        and '"implementation_error"' in runner
        and '"unknown_uncertified"' in runner
        and 'payload["algorithm_status"] = result.status' in runner
        and 'payload["status"] = payload["overall_status"]' in runner,
    )
    incident = ROOT / "docs/audits/fairness_development_attempt2_post_evaluation_incident.md"
    run_hashes = ROOT / "docs/audits/fairness_development_attempt2_run_sha256.csv"
    incident_text = incident.read_text(encoding="utf-8") if incident.exists() else ""
    hash_lines = run_hashes.read_text(encoding="utf-8").splitlines() if run_hashes.exists() else []
    _check(
        checks,
        "attempt2_post_evaluation_incident_is_quarantined",
        "decision: implementation_blocker" in incident_text
        and "scientific_selection_allowed: false" in incident_text
        and "next_stage: correctness_hotfix_only" in incident_text
        and "45,775" in incident_text
        and len(hash_lines) == 61,
    )
    return {
        "audit_name": "robust_regional_fairness_development_protocol",
        "passed": all(check["passed"] for check in checks if check.get("required", True)),
        "passed_count": sum(check["passed"] for check in checks),
        "check_count": len(checks),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen fairness-model development protocol")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_fairness_development()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
