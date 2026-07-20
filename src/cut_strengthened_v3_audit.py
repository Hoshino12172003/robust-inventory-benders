from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Any

from .config import load_config
from .cut_strengthening import cut_strengthening_config
from .experiment_protocol import atomic_write_json, file_sha256, git_commit, utc_now_iso
from .experiment_suite import (
    _apply_selected_parameters,
    _apply_variant_config,
    _base_config,
    _variant_specs,
    experiment_run_specs,
)
from .robust_dual_subproblem import solve_fixed_pattern_dual_lp


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "experiments/configs"
DOCUMENT_PATH = REPO_ROOT / "docs/cut_strengthened_joint_v3_protocol.md"
DECISION_DOCUMENT_PATH = REPO_ROOT / "docs/cut_strengthened_joint_v3_development_decision.md"
SELECTED_CANDIDATE_CONFIG_NAME = "selected_cut_strengthened_joint_v3_candidate.yaml"
SELECTED_CANDIDATE_CONFIG_SHA256 = (
    "7e8aaf39de8c100b4ce9b46256a074fbd324b07ddc347d256494ed070d4e0eb6"
)
DEVELOPMENT_EXPERIMENT_COMMIT = "8eabc10f9248878f4f5e409bbcd75ead288e168b"
MEDIUM_RESULTS_ZIP = "cut_strengthened_joint_v3_development_medium_large_results.zip"
MEDIUM_RESULTS_ZIP_SHA256 = (
    "D778D9B988BB360BBFF898A7A80887D8E68103E525512617EBC924C9DC76C492"
)
LARGE_RESULTS_ZIP = "cut_strengthened_joint_v3_development_large_results.zip"
LARGE_RESULTS_ZIP_SHA256 = (
    "8516D345BDB752F946CB5643B10A560C14B6B7DBF8109F2BCE5F2A78521826E4"
)
FROZEN_CONFIG_SHA256 = {
    "selected_algorithm_parameters.yaml": "50b275578a127b349bdda47ff161680048cd1c0c8845ea85e707949bdfa29d25",
    "final_evaluation_joint_v1.yaml": "efa7d3406687d4a7a7a99726eaa19f604f0f5b10cf9f38709420dcec8bf1195f",
    "large_scale_evaluation_joint_v1.yaml": "689d5b8e1ca7b277137a27a75ecb1405da98b2adcd9a9a11481c48d5a5bda539",
    "managerial_sensitivity_joint_v1.yaml": "b7d7880a3f106c1a7a2560b3d0a30a23369980a2fbece3498e21d541310be796",
}
DEVELOPMENT_SEEDS = set(range(75, 80))
RESERVED_VALIDATION_SEEDS = set(range(80, 90))
RESERVED_MEDIUM_FINAL_SEEDS = set(range(90, 100))
RESERVED_LARGE_FINAL_SEEDS = set(range(100, 110))
PREVIOUS_SEEDS = set(range(0, 75))
EXPECTED_VARIANTS = [
    "proposed_joint_rho025_050",
    "joint_v1_core_point_strengthened",
    "joint_v1_stall_secondary_cut",
    "proposed_cut_strengthened_joint_v3",
]
EXPECTED_CONFIGS = {
    "cut_strengthened_joint_v3_development_medium_large.yaml": {
        "size": "medium_large",
        "time_limit": 600,
        "max_iterations": 10000,
    },
    "cut_strengthened_joint_v3_development_large.yaml": {
        "size": "large",
        "time_limit": 1800,
        "max_iterations": 20000,
    },
}
EXPECTED_VALIDATION_CONFIG_NAMES = {
    "cut_strengthened_joint_v3_validation_medium_large.yaml",
    "cut_strengthened_joint_v3_validation_large.yaml",
}


def _check(name: str, passed: bool, details: Any = "") -> dict[str, Any]:
    return {"check": name, "required": True, "passed": bool(passed), "details": details}


def _variant_config(raw: dict[str, Any], name: str) -> dict[str, Any]:
    resolved = _apply_selected_parameters(raw)
    variants = {
        variant_name: (method, settings)
        for variant_name, method, settings in _variant_specs(resolved)
    }
    method, settings = variants[name]
    base = _base_config(
        resolved,
        str(resolved["instance_sizes"][0]),
        int(resolved["random_seeds"][0]),
    )
    _solver_method, _flags, config = _apply_variant_config(base, method, settings)
    return config


def audit_cut_strengthened_v3(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    config_dir = root / "experiments/configs"
    document = (root / "docs/cut_strengthened_joint_v3_protocol.md").read_text(encoding="utf-8")
    decision_document = (
        root / "docs/cut_strengthened_joint_v3_development_decision.md"
    ).read_text(encoding="utf-8")
    configs = {name: load_config(config_dir / name) for name in EXPECTED_CONFIGS}
    selected_candidate_path = config_dir / SELECTED_CANDIDATE_CONFIG_NAME
    selected_candidate = load_config(selected_candidate_path)
    checks: list[dict[str, Any]] = []

    for name, expected_hash in FROZEN_CONFIG_SHA256.items():
        actual = file_sha256(config_dir / name).lower()
        checks.append(_check(f"frozen_{name}_unchanged", actual == expected_hash, actual))

    final = load_config(config_dir / "final_evaluation_joint_v1.yaml")
    selected = load_config(config_dir / "selected_algorithm_parameters.yaml")
    selected_algorithm = selected_candidate.get("algorithm", {})
    selected_components = selected_candidate.get("components", {})
    selected_robust = selected_candidate.get("robust", {})
    selected_evidence = selected_candidate.get("evidence", {})
    selected_strengthening = cut_strengthening_config(selected_algorithm)
    development_config_paths = [config_dir / name for name in EXPECTED_CONFIGS]
    development_config_paths.append(selected_candidate_path)
    used_development_seeds = {
        int(seed)
        for path in development_config_paths
        for seed in load_config(path).get("random_seeds", [])
    }
    checks.extend(
        [
            _check("v1_precision_policy_joint_error_budget", selected.get("precision_policy") == "joint_error_budget"),
            _check("v1_cut_strengthening_default_none", _base_config(final, "medium_large", 10)["algorithm"]["cut_strengthening_policy"] == "none"),
            _check(
                "selected_v3_candidate_config_hash_frozen",
                file_sha256(selected_candidate_path).lower()
                == SELECTED_CANDIDATE_CONFIG_SHA256,
                file_sha256(selected_candidate_path).lower(),
            ),
            _check(
                "selected_v3_candidate_is_core_only",
                selected_candidate.get("selected_variant")
                == "joint_v1_core_point_strengthened"
                and selected_algorithm.get("cut_strengthening_policy") == "core_point"
                and selected_components.get("core_point_strengthening_enabled") is True
                and selected_components.get("stall_secondary_enabled") is False,
            ),
            _check(
                "selected_v3_uses_frozen_joint_precision",
                selected_algorithm.get("precision_policy") == "joint_error_budget"
                and selected_algorithm.get("precision_policy") != "workload_aware_joint"
                and math.isclose(
                    float(selected_algorithm.get("master_error_budget_ratio")),
                    0.25,
                )
                and math.isclose(
                    float(selected_algorithm.get("subproblem_error_budget_ratio")),
                    0.50,
                )
                and selected_algorithm.get("monotone_precision_tightening") is True,
            ),
            _check(
                "selected_v3_gap_bounds_match_v1",
                math.isclose(float(selected_algorithm.get("master_gap_max")), 0.02)
                and math.isclose(float(selected_algorithm.get("master_gap_min")), 0.0001)
                and math.isclose(float(selected_algorithm.get("subproblem_gap_max")), 0.05)
                and math.isclose(float(selected_algorithm.get("subproblem_gap_min")), 0.0001),
            ),
            _check(
                "selected_v3_core_parameters_frozen",
                math.isclose(selected_strengthening.core_point_update_weight, 0.50)
                and math.isclose(selected_strengthening.core_point_min_distance, 1.0e-9)
                and math.isclose(selected_strengthening.core_point_stage1_time_limit, 2.0)
                and math.isclose(selected_strengthening.core_point_stage2_time_limit, 2.0)
                and math.isclose(selected_strengthening.core_point_min_remaining_time, 10.0)
                and math.isclose(selected_strengthening.core_point_min_global_gap, 5.0e-4)
                and math.isclose(selected_strengthening.core_point_current_abs_tol, 1.0e-7)
                and math.isclose(selected_strengthening.core_point_current_rel_tol, 1.0e-8)
                and math.isclose(
                    selected_strengthening.core_point_min_normalized_improvement,
                    1.0e-7,
                ),
            ),
            _check(
                "selected_v3_single_cut_and_legacy_modules_disabled",
                selected_algorithm.get("max_cuts_per_iteration") == 1
                and selected_algorithm.get("cut_selection_enabled") is False
                and selected_algorithm.get("adaptive_secondary_cut_selection_enabled") is False
                and selected_algorithm.get("adaptive_secondary_generation_enabled") is False
                and selected_algorithm.get("adaptive_subproblem_gap_enabled") is False
                and selected_components.get("workload_aware_v2_enabled") is False,
            ),
            _check(
                "selected_v3_gamma_frozen_without_continuation",
                selected_robust.get("gamma_target") == 2
                and selected_robust.get("gamma_schedule") == [2]
                and selected_robust.get("gamma_continuation_enabled") is False,
            ),
            _check(
                "selected_v3_evidence_commit_and_hashes",
                selected_candidate.get("development_experiment_commit")
                == DEVELOPMENT_EXPERIMENT_COMMIT
                and selected_evidence.get("medium_large_results_zip", {}).get("filename")
                == MEDIUM_RESULTS_ZIP
                and selected_evidence.get("medium_large_results_zip", {}).get("sha256")
                == MEDIUM_RESULTS_ZIP_SHA256
                and selected_evidence.get("large_results_zip", {}).get("filename")
                == LARGE_RESULTS_ZIP
                and selected_evidence.get("large_results_zip", {}).get("sha256")
                == LARGE_RESULTS_ZIP_SHA256,
            ),
            _check(
                "selected_v3_no_revision_or_validation_started",
                selected_candidate.get("parameter_revision_used") is False
                and selected_candidate.get("selection_frozen") is True
                and selected_candidate.get("validation_started") is False
                and selected_candidate.get("final_test_started") is False
                and selected_candidate.get("formal_statistical_inference_allowed") is False,
            ),
            _check(
                "development_does_not_use_reserved_seeds_80_109",
                used_development_seeds.isdisjoint(
                    RESERVED_VALIDATION_SEEDS
                    | RESERVED_MEDIUM_FINAL_SEEDS
                    | RESERVED_LARGE_FINAL_SEEDS
                ),
                sorted(used_development_seeds),
            ),
            _check(
                "only_expected_validation_configs_and_no_final_config",
                {
                    path.name
                    for path in config_dir.glob(
                        "cut_strengthened_joint_v3_validation*.yaml"
                    )
                }
                == EXPECTED_VALIDATION_CONFIG_NAMES
                and not any(config_dir.glob("*cut_strengthened_joint_v3*final*.yaml")),
            ),
        ]
    )

    for filename, expected in EXPECTED_CONFIGS.items():
        raw = configs[filename]
        resolved = _apply_selected_parameters(raw)
        specs = experiment_run_specs(resolved)
        checks.extend(
            [
                _check(f"{filename}_run_count_20", len(specs) == 20, len(specs)),
                _check(f"{filename}_development_seeds", set(raw.get("random_seeds", [])) == DEVELOPMENT_SEEDS),
                _check(f"{filename}_instance_size", raw.get("instance_sizes") == [expected["size"]]),
                _check(f"{filename}_variants", raw.get("variants") == EXPECTED_VARIANTS),
                _check(
                    f"{filename}_limits",
                    raw.get("time_limit") == expected["time_limit"]
                    and raw.get("max_iterations") == expected["max_iterations"],
                ),
            ]
        )
        expected_policy_and_cuts = {
            "proposed_joint_rho025_050": ("none", 1),
            "joint_v1_core_point_strengthened": ("core_point", 1),
            "joint_v1_stall_secondary_cut": ("stall_secondary", 2),
            "proposed_cut_strengthened_joint_v3": ("core_point_stall_secondary", 2),
        }
        for variant, (policy, max_cuts) in expected_policy_and_cuts.items():
            run = _variant_config(raw, variant)
            algorithm = run["algorithm"]
            checks.append(
                _check(
                    f"{filename}_{variant}_effective_settings",
                    algorithm.get("precision_policy") == "joint_error_budget"
                    and algorithm.get("precision_policy") != "workload_aware_joint"
                    and algorithm.get("cut_strengthening_policy") == policy
                    and algorithm.get("max_cuts_per_iteration") == max_cuts
                    and algorithm.get("subproblem_mode") == "robust_dual_milp"
                    and run.get("gamma_continuation_enabled") is False
                    and run["robust"].get("gamma_schedule") == [2]
                    and algorithm.get("cut_selection_enabled") is False
                    and algorithm.get("adaptive_secondary_cut_selection_enabled") is False
                    and algorithm.get("adaptive_secondary_generation_enabled") is False,
                )
            )

    seed_groups = [
        DEVELOPMENT_SEEDS,
        RESERVED_VALIDATION_SEEDS,
        RESERVED_MEDIUM_FINAL_SEEDS,
        RESERVED_LARGE_FINAL_SEEDS,
    ]
    checks.extend(
        [
            _check(
                "new_seed_groups_pairwise_disjoint",
                all(
                    seed_groups[left].isdisjoint(seed_groups[right])
                    for left in range(len(seed_groups))
                    for right in range(left + 1, len(seed_groups))
                ),
            ),
            _check("new_seeds_disjoint_from_0_74", PREVIOUS_SEEDS.isdisjoint(set().union(*seed_groups))),
            _check("validation_reserved_80_89", RESERVED_VALIDATION_SEEDS == set(range(80, 90))),
            _check("final_reserved_90_109", RESERVED_MEDIUM_FINAL_SEEDS == set(range(90, 100)) and RESERVED_LARGE_FINAL_SEEDS == set(range(100, 110))),
        ]
    )

    representative = configs["cut_strengthened_joint_v3_development_medium_large.yaml"]
    full_algorithm = _variant_config(representative, "proposed_cut_strengthened_joint_v3")["algorithm"]
    strengthening = cut_strengthening_config(full_algorithm)
    checks.extend(
        [
            _check(
                "core_parameters_frozen",
                math.isclose(strengthening.core_point_update_weight, 0.50)
                and math.isclose(strengthening.core_point_min_distance, 1.0e-9)
                and math.isclose(strengthening.core_point_stage1_time_limit, 2.0)
                and math.isclose(strengthening.core_point_stage2_time_limit, 2.0)
                and math.isclose(strengthening.core_point_min_remaining_time, 10.0)
                and math.isclose(strengthening.core_point_min_global_gap, 5.0e-4)
                and math.isclose(strengthening.core_point_current_abs_tol, 1.0e-7)
                and math.isclose(strengthening.core_point_current_rel_tol, 1.0e-8)
                and math.isclose(strengthening.core_point_min_normalized_improvement, 1.0e-7),
            ),
            _check(
                "secondary_parameters_frozen",
                strengthening.v3_secondary_lb_window == 5
                and math.isclose(strengthening.v3_secondary_stall_threshold, 1.0e-4)
                and strengthening.v3_secondary_cooldown_iterations == 10
                and math.isclose(strengthening.v3_secondary_min_global_gap, 1.0e-3)
                and math.isclose(strengthening.v3_secondary_min_remaining_time, 30.0)
                and math.isclose(strengthening.v3_secondary_max_time_per_attempt, 10.0)
                and math.isclose(strengthening.v3_secondary_max_time_fraction_of_remaining, 0.05)
                and math.isclose(strengthening.v3_secondary_max_extra_time_share, 0.10)
                and strengthening.v3_secondary_pattern_memory == 10,
            ),
        ]
    )

    lp_source = inspect.getsource(solve_fixed_pattern_dual_lp)
    benders_source = (root / "src/benders.py").read_text(encoding="utf-8")
    checks.extend(
        [
            _check("fixed_pattern_solver_is_continuous_lp", "GRB.BINARY" not in lp_source and "addVars" in lp_source),
            _check("core_auxiliary_never_updates_ub", '"core_point_auxiliary_bound_used_for_UB"' in benders_source and "auxiliary_bound_used_for_ub = False" in benders_source),
            _check("secondary_bound_never_updates_ub", '"v3_secondary_bound_used_for_UB"' in benders_source and "v3_secondary_bound_used_for_ub = False" in benders_source),
            _check(
                "development_rules_frozen_in_document",
                all(token in document for token in ("103%", "降低 5%", "降低 10%", "至少 3 个", "差距小于 1%", "继续采用 V1")),
            ),
            _check(
                "validation_rules_frozen_in_document",
                all(token in document for token in ("降低 7.5%", "降低 15%", "至少 6/10", "平均名次优于")),
            ),
            _check("managerial_sensitivity_postponed", "管理敏感性继续暂停" in document),
            _check("mw_type_not_pareto_claim", "Magnanti-Wong-type" in document and "不声称该割严格 Pareto-optimal" in document),
            _check(
                "development_decision_document_frozen",
                all(
                    token in decision_document
                    for token in (
                        DEVELOPMENT_EXPERIMENT_COMMIT,
                        MEDIUM_RESULTS_ZIP,
                        MEDIUM_RESULTS_ZIP_SHA256,
                        LARGE_RESULTS_ZIP,
                        LARGE_RESULTS_ZIP_SHA256,
                        SELECTED_CANDIDATE_CONFIG_SHA256.upper(),
                        "joint_v1_core_point_strengthened",
                        "full V3 虽通过两个规模的门槛，但不被选为 validation 候选",
                        "Development 阶段仅用于机制筛选和候选冻结，不进行正式统计推断",
                        "Validation 尚未开始",
                    )
                ),
            ),
        ]
    )
    failed = [check["check"] for check in checks if not check["passed"]]
    return {
        "audit_name": "cut_strengthened_joint_v3_protocol",
        "created_at": utc_now_iso(),
        "git_commit": git_commit(root),
        "all_required_checks_passed": not failed,
        "required_check_count": len(checks),
        "passed_check_count": sum(check["passed"] for check in checks),
        "failed_checks": failed,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the cut-strengthened Joint V3 protocol.")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_cut_strengthened_v3()
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_required_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
