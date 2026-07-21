from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from .config import load_config
from .cut_strengthened_v3_audit import FROZEN_CONFIG_SHA256
from .cut_strengthened_v3_final_audit import _absolute_lock_paths, _effective_variant
from .experiment_protocol import atomic_write_json, file_sha256, git_commit, utc_now_iso
from .experiment_suite import INSTANCE_SIZES, _apply_selected_parameters, experiment_run_specs


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROTOCOL_SHA256 = "ec7761d96c1d2a17f96eba90bf4bfb520a9ce6359f938acd7f294a10e7f24a38"
DIAGNOSTIC_SEEDS = list(range(110, 120))
RESERVED_SEEDS = {
    "development": list(range(120, 130)),
    "validation": list(range(130, 140)),
    "final_medium_large": list(range(140, 150)),
    "final_large": list(range(150, 160)),
}
PRE_DIAGNOSTIC_SEEDS = set(range(0, 110))
FUTURE_RESERVED_SEEDS = set(range(120, 160))
EXPECTED_VARIANTS = ["joint_v1_core_point_strengthened"]
EXPECTED_VARIANT_SETTINGS = {
    "joint_v1_core_point_strengthened": {
        "cut_strengthening_policy": "core_point",
        "max_cuts_per_iteration": 1,
    }
}
EXPECTED_CONFIGS = {
    "regional_fairness_diagnostic_medium_large.yaml": {
        "size": "medium_large",
        "time_limit": 600,
        "max_iterations": 10000,
        "output_dir": "experiments/results_fairness_diagnostic/medium_large",
        "scenario_count": 1831,
        "sha256": "04d2ca32c31d7b2d3c9071583c4bc3897740b463d6ad945a8a52554a6317c79c",
    },
    "regional_fairness_diagnostic_large.yaml": {
        "size": "large",
        "time_limit": 1800,
        "max_iterations": 20000,
        "output_dir": "experiments/results_fairness_diagnostic/large",
        "scenario_count": 4657,
        "sha256": "7a40ff6cfedb02f44d57c999377377b7eb25e406ebe417791ca7a0c22c2fb307",
    },
}
FROZEN_FINAL_FILES = {
    "docs/cut_strengthened_joint_v3_final_decision.md": "1e9eb741056331cccc5a456bfa7858c9fb1b423c3c5dc904b602315b23b72594",
    "experiments/configs/cut_strengthened_joint_v3_final_medium_large.yaml": "1d41a19bb47218f2844c2bdfeadf9b044e8776db944c37989ef8c26feb9c0867",
    "experiments/configs/cut_strengthened_joint_v3_final_large.yaml": "60fdf4a9a642485a46e473a25ddb7502198a84eea927d9a60e670b764f8542f3",
    "experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml": "7e8aaf39de8c100b4ce9b46256a074fbd324b07ddc347d256494ed070d4e0eb6",
}
FROZEN_MODEL_FILES = {
    "src/benders.py": "37967750ee1aad5575a9b1fe0b050f012ec21db58fa277fbefaa5a48cfef1d9f",
    "src/instance.py": "6efbe6f93534d1621a1fc5bf110f6745f9d3529d5c9edde3dfe1828b41633ecc",
    "src/subproblem.py": "63aacb578ba5c2131424d5c103e3b8f7aa4408028670329e4a724ee43cb69ec1",
    "src/robust_dual_subproblem.py": "ec20ee9a736585ad0e2273fd77d5a362fb75e900b7997de9c60f6cc3aed16008",
    "src/scenarios.py": "7294c60dc318f7678f8a4464daf2cbd85e540842c6c3858bb1d30a9de7915511",
}
EXPECTED_OUTPUT_FILES = [
    "results.csv",
    "region_scenario_metrics.csv",
    "instance_summary.csv",
    "resolved_config.yaml",
    "run_manifest.json",
    "diagnostic_run_manifest.json",
    "diagnosis.json",
    "audit_log.json",
    "checkpoint/index.json",
]
FORBIDDEN_MODEL_KEYS = {
    "fairness_penalty",
    "fairness_weight",
    "fairness_constraint",
    "equity_penalty",
    "protected_group",
    "uncertainty_set",
    "uncertainty_budget",
    "managerial_sensitivity",
    "sensitivity_axis",
}


def _check(name: str, passed: bool, details: Any = "") -> dict[str, Any]:
    return {"check": name, "required": True, "passed": bool(passed), "details": details}


def _recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_recursive_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_recursive_keys(item))
    return keys


def _scenario_count(size: str, gamma: int) -> int:
    dimensions = INSTANCE_SIZES[size]
    units = dimensions["num_products"] * dimensions["num_regions"]
    return sum(math.comb(units, k) for k in range(gamma + 1))


def _normalized(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    for field in (
        "experiment_name",
        "output_dir",
        "instance_sizes",
        "time_limit",
        "max_iterations",
    ):
        value.pop(field, None)
    return value


def audit_regional_fairness_diagnostic(
    repo_root: str | Path | None = None,
    *,
    require_absent_outputs: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    config_dir = root / "experiments/configs"
    document_path = root / "docs/regional_fairness_diagnostic_protocol.md"
    evaluator_path = root / "src/regional_fairness_diagnostic.py"
    pipeline_path = root / "src/regional_fairness_pipeline.py"
    pipeline_test_path = root / "tests/test_regional_fairness_pipeline.py"
    checks: list[dict[str, Any]] = []

    for name, expected_hash in FROZEN_CONFIG_SHA256.items():
        path = config_dir / name
        actual = file_sha256(path).lower() if path.exists() else "missing"
        checks.append(_check(f"frozen_{name}_unchanged", actual == expected_hash, actual))
    for relative, expected_hash in {**FROZEN_FINAL_FILES, **FROZEN_MODEL_FILES}.items():
        path = root / relative
        actual = file_sha256(path).lower() if path.exists() else "missing"
        checks.append(_check(f"frozen_{relative.replace('/', '_')}_unchanged", actual == expected_hash, actual))

    final_decision = (root / "docs/cut_strengthened_joint_v3_final_decision.md").read_text(
        encoding="utf-8"
    )
    checks.append(
        _check(
            "final_decision_authorizes_diagnostic_only",
            all(
                token in final_decision
                for token in (
                    "decision: final_confirmed",
                    "selected_algorithm: joint_v1_core_point_strengthened",
                    "v3_status: completed",
                    "retuning_allowed: false",
                    "next_authorized_stage: fairness_diagnostic_only",
                )
            ),
        )
    )

    configs: list[dict[str, Any]] = []
    outputs: set[str] = set()
    for filename, expected in EXPECTED_CONFIGS.items():
        path = config_dir / filename
        config = load_config(path) if path.exists() else {}
        configs.append(config)
        prefix = filename.removesuffix(".yaml")
        diagnostic = config.get("fairness_diagnostic", {})
        seeds = set(config.get("random_seeds", []))
        actual_hash = file_sha256(path).lower() if path.exists() else "missing"
        output_dir = str(config.get("output_dir", ""))
        outputs.add(output_dir)
        scenario_count = _scenario_count(str(expected["size"]), 2)
        checks.extend(
            [
                _check(f"{prefix}_sha256_frozen", actual_hash == expected["sha256"], actual_hash),
                _check(f"{prefix}_seeds_exact_110_119", config.get("random_seeds") == DIAGNOSTIC_SEEDS, config.get("random_seeds")),
                _check(f"{prefix}_no_pre_diagnostic_or_future_seed_use", seeds.isdisjoint(PRE_DIAGNOSTIC_SEEDS | FUTURE_RESERVED_SEEDS)),
                _check(f"{prefix}_only_frozen_core_candidate", config.get("variants") == EXPECTED_VARIANTS and config.get("variant_settings") == EXPECTED_VARIANT_SETTINGS),
                _check(f"{prefix}_exactly_10_base_runs", len(experiment_run_specs(config)) == 10),
                _check(
                    f"{prefix}_scale_and_algorithm_limits_frozen",
                    config.get("instance_sizes") == [expected["size"]]
                    and config.get("time_limit") == expected["time_limit"]
                    and config.get("max_iterations") == expected["max_iterations"]
                    and config.get("tol") == 1.0e-4,
                ),
                _check(
                    f"{prefix}_base_model_switches_frozen",
                    config.get("gamma_target") == 2
                    and config.get("gamma_schedule") == [2]
                    and config.get("gamma_continuation_enabled") is False
                    and config.get("subproblem_mode") == "robust_dual_milp"
                    and config.get("cut_selection_enabled") is False
                    and config.get("adaptive_secondary_generation_enabled") is False
                    and config.get("max_cuts_per_iteration") == 1,
                ),
                _check(
                    f"{prefix}_diagnostic_scope_only",
                    config.get("protocol_phase") == "fairness_diagnostic"
                    and config.get("diagnostic_authorization") == "fairness_diagnostic_protocol_only"
                    and config.get("formal_inference_allowed") is False,
                ),
                _check(
                    f"{prefix}_exact_scenario_policy_and_count",
                    diagnostic.get("gamma") == 2
                    and diagnostic.get("scenario_policy") == "all_budget_extreme_points"
                    and diagnostic.get("exact_scenarios") is True
                    and diagnostic.get("max_scenarios") == 5000
                    and scenario_count == expected["scenario_count"]
                    and scenario_count <= diagnostic.get("max_scenarios", 0),
                    scenario_count,
                ),
                _check(
                    f"{prefix}_cost_tolerance_frozen",
                    diagnostic.get("cost_absolute_tolerance") == 1.0e-6
                    and diagnostic.get("cost_relative_tolerance") == 1.0e-6
                    and diagnostic.get("metric_tolerance") == 1.0e-9,
                ),
                _check(
                    f"{prefix}_decision_thresholds_frozen",
                    diagnostic.get("thresholds")
                    == {
                        "material_gap": 0.10,
                        "structural_median": 0.05,
                        "no_material_median": 0.03,
                        "degeneracy_reduction": 0.05,
                        "sensitivity_levels": [0.05, 0.10, 0.15],
                    },
                ),
                _check(f"{prefix}_future_seed_reservations_frozen", diagnostic.get("reserved_future_seeds") == RESERVED_SEEDS),
                _check(f"{prefix}_required_output_schema_frozen", diagnostic.get("required_outputs") == EXPECTED_OUTPUT_FILES),
                _check(f"{prefix}_checkpoint_chunk_size_frozen", diagnostic.get("checkpoint_scenario_chunk_size") == 50),
                _check(
                    f"{prefix}_isolated_absent_output",
                    output_dir == expected["output_dir"]
                    and "results_fairness_diagnostic" in output_dir
                    and (
                        not require_absent_outputs
                        or not (root / output_dir).exists()
                    ),
                    output_dir,
                ),
                _check(f"{prefix}_no_model_fairness_or_managerial_keys", _recursive_keys(config).isdisjoint(FORBIDDEN_MODEL_KEYS), sorted(_recursive_keys(config) & FORBIDDEN_MODEL_KEYS)),
            ]
        )
        try:
            resolved = _apply_selected_parameters(_absolute_lock_paths(config, config_dir))
            effective = _effective_variant(resolved, "joint_v1_core_point_strengthened")["algorithm"]
            candidate_lock = load_config(
                config_dir / "selected_cut_strengthened_joint_v3_candidate.yaml"
            )
            frozen_algorithm = candidate_lock["algorithm"]
            explicit_core_fields = {
                key: value
                for key, value in frozen_algorithm.items()
                if key.startswith("core_point_")
                or key
                in {
                    "cut_strengthening_policy",
                    "max_cuts_per_iteration",
                    "cut_selection_enabled",
                    "adaptive_secondary_cut_selection_enabled",
                    "adaptive_secondary_generation_enabled",
                    "adaptive_subproblem_gap_enabled",
                    "adaptive_gap_enabled",
                    "final_certification_enabled",
                    "final_certification_no_cut_patience",
                }
            }
            candidate_ok = (
                all(
                    effective.get(key, config.get(key)) == value
                    for key, value in frozen_algorithm.items()
                )
                and all(config.get(key) == value for key, value in explicit_core_fields.items())
                and effective.get("precision_policy") == "joint_error_budget"
                and effective.get("cut_strengthening_policy") == "core_point"
                and effective.get("max_cuts_per_iteration") == 1
                and effective.get("adaptive_secondary_generation_enabled") is False
            )
            error = ""
        except Exception as exc:  # noqa: BLE001 - malformed lock is an audit failure.
            candidate_ok = False
            error = f"{type(exc).__name__}: {exc}"
        checks.append(_check(f"{prefix}_effective_candidate_parameters_frozen", candidate_ok, error))

    checks.extend(
        [
            _check("diagnostic_scale_configs_differ_only_as_registered", len(configs) == 2 and _normalized(configs[0]) == _normalized(configs[1])),
            _check("diagnostic_outputs_are_distinct", len(outputs) == 2),
            _check(
                "formal_result_directories_absent",
                not require_absent_outputs
                or not (root / "experiments/results_fairness_diagnostic").exists(),
            ),
        ]
    )

    document = document_path.read_text(encoding="utf-8") if document_path.exists() else ""
    checks.extend(
        [
            _check(
                "protocol_document_sha256_frozen",
                document_path.exists()
                and file_sha256(document_path).lower() == EXPECTED_PROTOCOL_SHA256,
                file_sha256(document_path).lower() if document_path.exists() else "missing",
            ),
            _check(
                "protocol_freezes_metric_definitions",
                all(token in document for token in ("WGap", "WMinFR", "WWD", "fair-best", "not_applicable", "Gini")),
            ),
            _check(
                "protocol_freezes_decision_categories",
                all(
                    token in document
                    for token in (
                        "structural_fairness_gap",
                        "recourse_degeneracy_only",
                        "no_material_fairness_gap",
                        "fairness_diagnostic_inconclusive",
                        "0.10",
                        "0.05",
                        "0.03",
                    )
                ),
            ),
            _check(
                "protocol_keeps_future_seeds_sealed",
                all(token in document for token in ("120--129", "130--139", "140--149", "150--159")),
            ),
            _check(
                "protocol_documents_formal_resume_and_atomic_chunks",
                all(
                    token in document
                    for token in (
                        "python -m src.regional_fairness_diagnostic",
                        "--resume",
                        "checkpoint_scenario_chunk_size: 50",
                        "diagnostic_run_manifest.json",
                        "os.replace",
                        "single-writer lock",
                    )
                ),
            ),
            _check(
                "postprocessor_isolated_from_benders",
                evaluator_path.exists()
                and "diagnostic_updates_benders_bounds\": False" in evaluator_path.read_text(encoding="utf-8")
                and "solve_benders" not in evaluator_path.read_text(encoding="utf-8"),
            ),
            _check(
                "formal_cli_entrypoint_and_resume_present",
                evaluator_path.exists()
                and all(
                    token in evaluator_path.read_text(encoding="utf-8")
                    for token in ("def main()", 'parser.add_argument("--resume"', "run_regional_fairness_pipeline")
                ),
            ),
            _check(
                "diagnostic_manifest_identity_and_schema_present",
                pipeline_path.exists()
                and all(
                    token in pipeline_path.read_text(encoding="utf-8")
                    for token in (
                        "DIAGNOSTIC_MANIFEST_SCHEMA_VERSION",
                        '"diagnostic_run_key"',
                        '"base_input_identity"',
                        '"base_results_sha256"',
                        '"protocol_document_sha256"',
                        '"final_outputs"',
                    )
                ),
            ),
            _check(
                "atomic_chunk_checkpoint_and_index_present",
                pipeline_path.exists()
                and all(
                    token in pipeline_path.read_text(encoding="utf-8")
                    for token in (
                        "atomic_write_json(path, checkpoint)",
                        "_write_checkpoint_index",
                        "checkpoint_is_resume_source_of_truth",
                        "after_checkpoint_commit_before_index",
                    )
                ),
            ),
            _check(
                "single_writer_and_interrupt_handling_present",
                pipeline_path.exists()
                and all(
                    token in pipeline_path.read_text(encoding="utf-8")
                    for token in ("SingleWriterLock", "os.O_EXCL", "except KeyboardInterrupt", '"interrupted"')
                ),
            ),
            _check(
                "traceability_schema_present",
                evaluator_path.exists()
                and all(
                    token in evaluator_path.read_text(encoding="utf-8")
                    for token in (
                        '"diagnostic_run_key"',
                        '"base_run_key"',
                        '"instance_name"',
                        '"scenario_key"',
                        '"deviation_pattern"',
                        '"default_recourse_status"',
                        '"fair_best_recourse_status"',
                        '"invalid_reason"',
                    )
                ),
            ),
            _check(
                "resume_fault_injection_tests_present",
                pipeline_test_path.exists()
                and all(
                    token in pipeline_test_path.read_text(encoding="utf-8")
                    for token in (
                        "test_fault_injection_resume_matches_clean_run",
                        "after_checkpoint_commit_before_index",
                        "after_all_chunks_before_aggregation",
                        "after_region_csv_before_diagnosis",
                        "test_keyboard_interrupt_is_atomic_and_resumable",
                        "test_single_writer_lock_refuses_concurrent_writer",
                    )
                ),
            ),
        ]
    )
    failed = [check["check"] for check in checks if not check["passed"]]
    return {
        "audit_name": "regional_fairness_diagnostic_protocol",
        "created_at": utc_now_iso(),
        "git_commit": git_commit(root),
        "authorization": "fairness_diagnostic_protocol_only",
        "all_required_checks_passed": not failed,
        "required_check_count": len(checks),
        "passed_check_count": sum(check["passed"] for check in checks),
        "failed_checks": failed,
        "scenario_counts": {
            expected["size"]: expected["scenario_count"] for expected in EXPECTED_CONFIGS.values()
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen regional fairness diagnostic protocol.")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_regional_fairness_diagnostic()
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_required_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
