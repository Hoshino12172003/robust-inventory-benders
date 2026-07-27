from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .experiment_protocol import file_sha256
from .fairness_scalability import SCALABILITY_CANDIDATES
from .fairness_scalability_runner import (
    PUBLIC_STATUSES,
    SCALABILITY_MANIFEST_SCHEMA_VERSION,
    WINDOWS_PORTABLE_PATH_LIMIT,
    cumulative_run_plan,
    path_portability_report,
    run_directory_id,
    stage_new_specs,
    validate_runtime_config,
)
from .fairness_scalability_suite import dry_run_report


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "medium_large": ROOT / "experiments/configs/fairness_scalability_development_medium_large.yaml",
    "large": ROOT / "experiments/configs/fairness_scalability_development_large.yaml",
}
FROZEN_HASHES = {
    "experiments/configs/selected_algorithm_parameters.yaml": "50B275578A127B349BDDA47FF161680048CD1C0C8845EA85E707949BDFA29D25",
    "experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml": "7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6",
    "src/benders.py": "37967750EE1AAD5575A9B1FE0B050F012EC21DB58FA277FBEFAA5A48CFEF1D9F",
    "src/scenarios.py": "7294C60DC318F7678F8A4464DAF2CBD85E540842C6C3858BB1D30A9DE7915511",
}
EXPECTED_SEEDS = list(range(160, 170))
SEALED_SEEDS = set(range(130, 160))
EXPECTED_RHOS = [0.0, 0.01, 0.025, 0.05, 0.10]
EXPECTED_PROTOCOL_HASH = "CB64C7505F81296992164359E7B2C929AE2868F9364FADDE68631AFCA2CC78B4"
EXPECTED_CONFIG_HASHES = {
    "medium_large": "31FED8028653E6F0D7132F61D73157188320ABA5486A0A66FEF950642D958893",
    "large": "CEB6025CD06DFBE91312827E738A47BF65FFCC2DCDEAFAD56EA5C3B9EE790801",
}


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def _add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = "") -> None:
    checks.append({"check": name, "passed": bool(passed), "required": True, "detail": detail})


def _without_size_fields(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    for key in ("experiment_name", "output_dir", "instance_sizes", "max_iterations"):
        value.pop(key, None)
    return value


def audit_fairness_scalability(
    *, config_overrides: Mapping[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    configs = {
        size: deepcopy(config_overrides[size])
        if config_overrides and size in config_overrides
        else _load(path)
        for size, path in CONFIGS.items()
    }
    _add(
        checks,
        "protocol_sha256",
        file_sha256(ROOT / "docs/fairness_scalability_development_protocol.md").upper()
        == EXPECTED_PROTOCOL_HASH,
    )
    for size, path in CONFIGS.items():
        _add(
            checks,
            f"{size}_config_sha256",
            file_sha256(path).upper() == EXPECTED_CONFIG_HASHES[size],
            file_sha256(path).upper(),
        )
    for relative, expected in FROZEN_HASHES.items():
        actual = file_sha256(ROOT / relative).upper()
        _add(checks, f"frozen_{relative.replace('/', '_')}", actual == expected, actual)

    source = (ROOT / "src/fairness_scalability.py").read_text(encoding="utf-8")
    benders_source = (ROOT / "src/fairness_benders.py").read_text(encoding="utf-8")
    runner_source = (ROOT / "src/fairness_scalability_runner.py").read_text(encoding="utf-8")
    reporting_source = (ROOT / "src/fairness_scalability_results_audit.py").read_text(encoding="utf-8")
    suite_source = (ROOT / "src/fairness_scalability_suite.py").read_text(encoding="utf-8")
    post_source = (ROOT / "src/fairness_post_evaluation.py").read_text(encoding="utf-8")
    _add(checks, "old_ray_not_cached", "self._patterns" in source and "self._rays" not in source)
    _add(checks, "cache_recertifies_current_point", "certify_current_point" in source and "certifier(" in source)
    _add(checks, "call_local_exclusions_removed", "self.model.remove(temporary_exclusions)" in source)
    _add(checks, "objective_bound_certification_preserved", "separation_partition_certifies" in source)
    _add(checks, "benders_success_requires_certification", "separation.robust_feasibility_certified" in benders_source)
    _add(checks, "formal_runner_schema", SCALABILITY_MANIFEST_SCHEMA_VERSION == 2)
    _add(
        checks,
        "formal_cli_flags",
        all(flag in suite_source for flag in ('"--config"', '"--stage"', '"--resume"', '"--dry-run"'))
        and '"--overwrite"' not in suite_source,
    )
    _add(checks, "atomic_scalability_manifest", "atomic_write_json(_manifest_path" in runner_source)
    _add(checks, "atomic_run_manifest", "atomic_write_json(" in runner_source and "_run_manifest_path(output_dir)" in runner_source)
    _add(checks, "fresh_candidate_solver_call", "frontier_solver(" in runner_source and "solve_fairness_benders(" in runner_source)
    _add(checks, "post_checkpoint_pipeline", "checkpointed_fairness_post_evaluation(" in runner_source)
    _add(checks, "shared_baseline_anchor", "baseline_run_key" in runner_source and "anchor_sha256" in runner_source and "anchor_value_hex" in runner_source)
    _add(checks, "stage_decision_gates", "validate_stage_decision(" in runner_source and "at least 16/20" in runner_source)
    _add(checks, "strict_resume_identity", "Scalability resume identity mismatch" in runner_source and "Existing output lacks a valid scalability identity manifest" in runner_source)
    _add(
        checks,
        "short_run_directory_mapping",
        'hashlib.sha256(str(run_key).encode("utf-8"))' in runner_source
        and '"run_key_to_directory_id"' in runner_source
        and '"directory_id_to_run_key"' in runner_source
        and "run-directory hash collision" in runner_source,
    )
    _add(
        checks,
        "portable_path_preflight_before_solver",
        "assert_windows_portable_paths(portability_report)" in runner_source
        and runner_source.index("assert_windows_portable_paths(portability_report)")
        < runner_source.index("(deps.configure_solver or _configure_gurobi)"),
    )
    _add(
        checks,
        "atomic_temp_paths_preflighted",
        "_atomic_temporary_path(path)" in runner_source
        and "WINDOWS_PORTABLE_PATH_LIMIT = 220" in runner_source,
    )
    _add(
        checks,
        "post_checkpoint_directories_explicitly_created",
        '(root / "checkpoint").mkdir(parents=True, exist_ok=True)' in post_source
        and 'post_root.mkdir(parents=True, exist_ok=True)' in runner_source,
    )
    _add(
        checks,
        "attempt1_quarantined_attempt2_fresh",
        "SCALABILITY_EXECUTION_ATTEMPT = 2" in runner_source
        and "windows_path_length_pipeline_defect" in runner_source
        and "previous_attempt_results_reused" in runner_source,
    )
    _add(
        checks,
        "reporting_csv_projection_complete",
        "aggregate_records(" in runner_source
        and "RESULT_FIELDS" in runner_source
        and all(
            field in reporting_source
            for field in (
                "separation_runtime", "separation_model_build_runtime",
                "separation_optimize_runtime", "master_runtime",
                "cache_candidate_count", "cache_hit_count",
                "certified_cached_cut_count", "pool_candidate_count",
                "certified_batch_cut_count", "duplicate_pattern_count",
                "cuts_per_iteration", "total_iterations", "cuts",
                "algorithm_runtime", "penalized_runtime_par2",
                "total_wall_runtime",
            )
        ),
    )
    _add(
        checks,
        "reporting_projection_semantics_explicit",
        "task_metadata_total_verified_against_iteration_log_sum" in reporting_source
        and "task_metadata_mean_verified_against_iteration_log_mean" in reporting_source,
    )
    _add(
        checks,
        "resolved_config_hash_semantics_split",
        all(
            field in runner_source
            for field in (
                "resolved_config_file_sha256",
                "resolved_config_canonical_sha256",
                "resolved_config_canonicalization",
            )
        )
        and "PyYAML safe_dump(sort_keys=True, allow_unicode=True), UTF-8"
        in reporting_source,
    )
    _add(
        checks,
        "scientific_status_schema",
        all(
            status in PUBLIC_STATUSES
            for status in (
                "certified_robust_optimal",
                "master_optimal_but_robust_uncertified",
                "time_limit_uncertified",
                "infeasible",
                "invalid_post_evaluation",
                "implementation_error",
                "interrupted",
            )
        ),
    )

    for size, config in configs.items():
        prefix = size
        try:
            validate_runtime_config(config)
            runtime_config_valid = True
        except (TypeError, ValueError):
            runtime_config_valid = False
        _add(checks, f"{prefix}_runtime_config_identity", runtime_config_valid)
        seeds = config.get("development_seeds", [])
        candidates = config.get("scalability_candidates", [])
        settings = config.get("candidate_settings", {})
        certification = config.get("certification", {})
        _add(checks, f"{prefix}_protocol_only", config.get("authorization") == "protocol_only_no_formal_execution")
        _add(
            checks,
            f"{prefix}_attempt2_identity",
            config.get("execution_attempt") == 2
            and config.get("previous_attempt_results_reused") is False
            and config.get("prior_attempts", [{}])[0].get("seeds_accessed") == [160],
        )
        _add(checks, f"{prefix}_seeds_160_169", seeds == EXPECTED_SEEDS)
        _add(checks, f"{prefix}_random_seeds_match", config.get("random_seeds") == EXPECTED_SEEDS)
        _add(checks, f"{prefix}_sealed_seeds_unused", not (set(seeds) & SEALED_SEEDS))
        _add(checks, f"{prefix}_s1", config.get("s1_seeds") == [160, 161, 162])
        _add(checks, f"{prefix}_candidate_set", tuple(candidates) == SCALABILITY_CANDIDATES)
        _add(checks, f"{prefix}_rho_grid", config.get("rho_grid") == EXPECTED_RHOS)
        _add(checks, f"{prefix}_s1_s2_rhos", config.get("s1_rho_grid") == [0.0, 0.01] and config.get("s2_rho_grid") == [0.0, 0.01])
        _add(checks, f"{prefix}_time_limits", all(config.get(k) == 1800 for k in ("baseline_time_limit", "fairness_time_limit", "time_limit")))
        _add(checks, f"{prefix}_gamma", config.get("gamma_target") == 2 and config.get("gamma_schedule") == [2] and config.get("gamma_continuation_enabled") is False)
        _add(checks, f"{prefix}_v1_precision", config.get("precision_policy") == "joint_error_budget" and config.get("master_error_budget_ratio") == 0.25 and config.get("subproblem_error_budget_ratio") == 0.50)
        _add(checks, f"{prefix}_candidate_hash", str(config.get("candidate_config_sha256", "")).upper() == FROZEN_HASHES["experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml"])
        _add(checks, f"{prefix}_single_cut_limit", settings.get("single_cut", {}).get("max_cuts_per_iteration") == 1)
        _add(checks, f"{prefix}_persistent_limit", settings.get("persistent_separation", {}).get("max_cuts_per_iteration") == 1)
        _add(checks, f"{prefix}_cache_limit", settings.get("persistent_certified_cache", {}).get("max_cuts_per_iteration") == 1)
        _add(checks, f"{prefix}_batch5_limit", settings.get("persistent_certified_cache_batch5", {}).get("max_cuts_per_iteration") == 5)
        _add(checks, f"{prefix}_certification_rules", certification.get("incumbent_role") == "candidate_scenario_only" and certification.get("cached_pattern_role") == "candidate_scenario_only" and certification.get("old_ray_reuse_allowed") is False and certification.get("old_cut_reuse_allowed") is False and certification.get("current_point_fixed_scenario_certification_required") is True and certification.get("complete_separation_objective_bound_required_for_robust_feasibility") is True)
        _add(checks, f"{prefix}_s2_gate", config.get("full_grid_gate") == {"correctness_required": True, "minimum_certified_solved_rate": 0.8, "minimum_certified_solved_count": 16, "denominator": 20})
        _add(checks, f"{prefix}_selection_order", config.get("selection_order") == ["mathematical_and_certification_correctness", "certified_solved_count_descending", "par2_ascending", "separation_runtime_ascending", "total_wall_runtime_ascending"])
        report = dry_run_report(config) if runtime_config_valid else None
        _add(checks, f"{prefix}_dry_run_counts", report is not None and report["s1"]["total_tasks"] == 27 and report["s2_cumulative"]["total_tasks"] == 90 and report["complete_staged_unique_tasks"] == 120)
        _add(checks, f"{prefix}_scenario_count", report is not None and report["scenario_count"] == (1831 if size == "medium_large" else 4657))
        _add(checks, f"{prefix}_no_formal_output", report is not None and report["output_dir_exists"] is False)
        if runtime_config_valid:
            s1 = cumulative_run_plan(config, "s1")
            s2 = cumulative_run_plan(config, "s2")
            full = cumulative_run_plan(config, "full-grid", selected_candidate="single_cut")
            _add(
                checks,
                f"{prefix}_machine_plan_counts",
                (len(s1), len(s2), len(full)) == (27, 90, 120)
                and tuple(len(stage_new_specs(plan, stage)) for plan, stage in (
                    (s1, "s1"), (s2, "s2"), (full, "full-grid")
                )) == (27, 63, 30),
            )
            _add(
                checks,
                f"{prefix}_one_baseline_per_seed",
                all(
                    len([spec for spec in plan if spec.task_type == "baseline"])
                    == len({spec.seed for spec in plan})
                    for plan in (s1, s2, full)
                ),
            )
            portability = path_portability_report(
                Path(str(config["output_dir"])),
                s1,
                scenario_count=(1831 if size == "medium_large" else 4657),
                chunk_size=int(config["post_evaluation"]["checkpoint_chunk_size"]),
            )
            _add(
                checks,
                f"{prefix}_windows_portable_paths",
                portability["windows_portability_check"] is True
                and portability["max_absolute_path_length"] <= WINDOWS_PORTABLE_PATH_LIMIT
                and portability["atomic_temporary_paths_checked"] is True,
                {
                    key: portability[key]
                    for key in (
                        "windows_portable_path_limit",
                        "max_absolute_path_length",
                        "longest_path_type",
                        "longest_path",
                        "windows_portability_check",
                        "atomic_temporary_paths_checked",
                    )
                },
            )
            _add(
                checks,
                f"{prefix}_all_s1_run_directories_short_and_unique",
                len({run_directory_id(spec.run_key) for spec in s1}) == 27
                and all(len(run_directory_id(spec.run_key)) == 26 for spec in s1),
            )
    _add(checks, "configs_equal_except_size", _without_size_fields(configs["medium_large"]) == _without_size_fields(configs["large"]))
    passed = sum(item["passed"] for item in checks)
    return {"status": "passed" if passed == len(checks) else "failed", "passed_checks": passed, "total_checks": len(checks), "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the frozen fairness scalability protocol")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = audit_fairness_scalability()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
