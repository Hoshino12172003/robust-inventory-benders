from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .experiment_protocol import atomic_write_json, file_sha256, git_commit, utc_now_iso
from .experiment_suite import (
    INSTANCE_SIZES,
    SELECTED_ALGORITHM_FIELDS,
    SELECTED_EXPERIMENT_FIELDS,
    _apply_selected_parameters,
    _apply_variant_config,
    _base_config,
    experiment_run_specs,
)
from .managerial_sensitivity_suite import managerial_run_config, managerial_run_specs


REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTED_CONFIG = REPO_ROOT / "experiments/configs/selected_algorithm_parameters.yaml"
FINAL_CONFIG = REPO_ROOT / "experiments/configs/final_evaluation_joint_v1.yaml"
LARGE_CONFIG = REPO_ROOT / "experiments/configs/large_scale_evaluation_joint_v1.yaml"
MANAGERIAL_CONFIG = REPO_ROOT / "experiments/configs/managerial_sensitivity_joint_v1.yaml"

EXPECTED_SELECTED_SHA256 = "50b275578a127b349bdda47ff161680048cd1c0c8845ea85e707949bdfa29d25"
EXPECTED_FINAL_SHA256 = "efa7d3406687d4a7a7a99726eaa19f604f0f5b10cf9f38709420dcec8bf1195f"
TUNING_SEEDS = {0, 1, 2}
FINAL_SEEDS = set(range(10, 20))
LARGE_SEEDS = set(range(20, 30))
MANAGERIAL_SEEDS = set(range(30, 40))
FROZEN_VARIANTS = [
    "standard_benders",
    "static_inexact_benders",
    "mp_adaptive_rho050",
    "sp_adaptive_rho050",
    "proposed_joint_rho025_050",
]
EXPECTED_AXES: dict[str, dict[str, Any]] = {
    "gamma_target": {"baseline_value": 2, "values": [0, 1, 2, 3, 4]},
    "service_level": {
        "baseline_value": 0.90,
        "values": [0.82, 0.86, 0.90, 0.94],
    },
    "budget_factor": {
        "baseline_value": 0.68,
        "values": [0.55, 0.62, 0.68, 0.75, 0.82],
    },
    "capacity_factor": {
        "baseline_value": 1.25,
        "values": [1.05, 1.15, 1.25, 1.35, 1.45],
    },
}


def _check(name: str, passed: bool, details: Any = "", required: bool = True) -> dict[str, Any]:
    return {
        "check": name,
        "required": required,
        "passed": bool(passed),
        "details": details,
    }


def _selected_values_match(config: dict[str, Any], selected: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches = [
        field
        for field in SELECTED_ALGORITHM_FIELDS + SELECTED_EXPERIMENT_FIELDS
        if config.get(field) != selected.get(field)
    ]
    return not mismatches, mismatches


def audit_protocols(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    selected_path = root / "experiments/configs/selected_algorithm_parameters.yaml"
    final_path = root / "experiments/configs/final_evaluation_joint_v1.yaml"
    large_path = root / "experiments/configs/large_scale_evaluation_joint_v1.yaml"
    managerial_path = root / "experiments/configs/managerial_sensitivity_joint_v1.yaml"

    selected = load_config(selected_path)
    final = load_config(final_path)
    large_raw = load_config(large_path)
    managerial_raw = load_config(managerial_path)
    large = _apply_selected_parameters(large_raw)
    managerial = _apply_selected_parameters(managerial_raw)
    large_specs = experiment_run_specs(large)
    managerial_specs = managerial_run_specs(managerial)

    seed_sets = [TUNING_SEEDS, FINAL_SEEDS, LARGE_SEEDS, MANAGERIAL_SEEDS]
    seed_disjoint = all(
        seed_sets[left].isdisjoint(seed_sets[right])
        for left in range(len(seed_sets))
        for right in range(left + 1, len(seed_sets))
    )
    large_match, large_mismatches = _selected_values_match(large, selected)
    managerial_match, managerial_mismatches = _selected_values_match(managerial, selected)
    # Gamma is the managerial factor itself, so each expanded run replaces only gamma_schedule/target.
    managerial_mismatches = [
        field for field in managerial_mismatches if field != "gamma_schedule"
    ]
    managerial_match = not managerial_mismatches

    _solver_method, _flags, static_large = _apply_variant_config(
        _base_config(large, "large", seed=20),
        "static_inexact_benders",
        large_raw.get("variant_settings", {}).get("static_inexact_benders", {}),
    )
    managerial_document = (
        root / "docs/managerial_sensitivity_protocol.md"
    ).read_text(encoding="utf-8")

    common_large_checks = {
        "subproblem_mode": "robust_dual_milp",
        "gamma_continuation_enabled": False,
        "cut_selection_enabled": False,
        "adaptive_secondary_cut_selection_enabled": False,
        "adaptive_secondary_generation_enabled": False,
        "adaptive_subproblem_gap_enabled": False,
        "adaptive_gap_enabled": False,
        "max_cuts_per_iteration": 1,
    }
    common_managerial_checks = dict(common_large_checks)
    checks = [
        _check(
            "seed_sets_pairwise_disjoint",
            seed_disjoint,
            {
                "tuning": sorted(TUNING_SEEDS),
                "final": sorted(FINAL_SEEDS),
                "large": sorted(LARGE_SEEDS),
                "managerial": sorted(MANAGERIAL_SEEDS),
            },
        ),
        _check("large_seed_set", set(large_raw.get("random_seeds", [])) == LARGE_SEEDS),
        _check(
            "managerial_seed_set",
            set(managerial_raw.get("random_seeds", [])) == MANAGERIAL_SEEDS,
        ),
        _check("large_run_count_50", len(large_specs) == 50, len(large_specs)),
        _check(
            "managerial_run_count_190", len(managerial_specs) == 190, len(managerial_specs)
        ),
        _check(
            "large_uses_five_frozen_variants",
            large_raw.get("variants") == FROZEN_VARIANTS,
            large_raw.get("variants"),
        ),
        _check(
            "large_variant_settings_match_final_evaluation",
            large_raw.get("variant_settings") == final.get("variant_settings"),
            {
                "large": large_raw.get("variant_settings"),
                "final": final.get("variant_settings"),
            },
        ),
        _check(
            "managerial_uses_only_proposed",
            managerial_raw.get("variants") == ["proposed_joint_rho025_050"],
            managerial_raw.get("variants"),
        ),
        _check(
            "large_instance_size",
            large_raw.get("instance_sizes") == ["large"],
            large_raw.get("instance_sizes"),
        ),
        _check(
            "managerial_instance_size",
            managerial_raw.get("instance_sizes") == ["medium_large"],
            managerial_raw.get("instance_sizes"),
        ),
        _check(
            "large_size_definition_8_8_12",
            INSTANCE_SIZES.get("large")
            == {"num_warehouses": 8, "num_products": 8, "num_regions": 12},
            INSTANCE_SIZES.get("large"),
        ),
        _check(
            "large_static_inexact_gaps_002_002",
            static_large["algorithm"]["fixed_master_mip_gap"] == 0.02
            and static_large["algorithm"]["fixed_subproblem_mip_gap"] == 0.02,
        ),
        _check(
            "large_common_settings",
            all(large.get(key) == value for key, value in common_large_checks.items()),
            {key: large.get(key) for key in common_large_checks},
        ),
        _check(
            "managerial_common_settings",
            all(
                managerial.get(key) == value
                for key, value in common_managerial_checks.items()
            ),
            {key: managerial.get(key) for key in common_managerial_checks},
        ),
        _check(
            "large_gamma_schedule_target_only",
            large.get("gamma_target") == 2 and large.get("gamma_schedule") == [2],
            {"gamma_target": large.get("gamma_target"), "gamma_schedule": large.get("gamma_schedule")},
        ),
        _check(
            "large_proposed_parameters_frozen",
            large_match,
            {"mismatches": large_mismatches},
        ),
        _check(
            "managerial_proposed_parameters_frozen",
            managerial_match,
            {"mismatches": managerial_mismatches},
        ),
        _check(
            "selected_algorithm_parameters_unchanged",
            file_sha256(selected_path).lower() == EXPECTED_SELECTED_SHA256,
            file_sha256(selected_path).lower(),
        ),
        _check(
            "final_evaluation_config_unchanged",
            file_sha256(final_path).lower() == EXPECTED_FINAL_SHA256,
            file_sha256(final_path).lower(),
        ),
        _check(
            "no_final_seed_reuse",
            FINAL_SEEDS.isdisjoint(set(large_raw.get("random_seeds", [])))
            and FINAL_SEEDS.isdisjoint(set(managerial_raw.get("random_seeds", []))),
        ),
        _check(
            "managerial_axes_and_levels",
            managerial_raw.get("sensitivity_axes") == EXPECTED_AXES,
            managerial_raw.get("sensitivity_axes"),
        ),
        _check(
            "managerial_gamma_runs_use_target_only_schedule",
            all(
                (lambda run: run["robust"]["gamma_schedule"] == [run["robust"]["gamma_target"]])(
                    managerial_run_config(managerial, spec)
                )
                for spec in managerial_specs
            ),
        ),
        _check(
            "managerial_baseline",
            managerial_raw.get("baseline")
            == {
                "gamma_target": 2,
                "service_level": 0.90,
                "budget_factor": 0.68,
                "capacity_factor": 1.25,
            },
            managerial_raw.get("baseline"),
        ),
        _check(
            "managerial_protocol_document_matches_levels",
            all(
                expected in managerial_document
                for expected in (
                    "| `gamma_target` | 0, 1, 2, 3, 4 | 50 |",
                    "| `service_level` | 0.82, 0.86, 0.90, 0.94 | 40 |",
                    "| `budget_factor` | 0.55, 0.62, 0.68, 0.75, 0.82 | 50 |",
                    "| `capacity_factor` | 1.05, 1.15, 1.25, 1.35, 1.45 | 50 |",
                )
            ),
        ),
        _check(
            "large_runtime_protocol",
            large_raw.get("time_limit") == 1800
            and large_raw.get("max_iterations") == 20000
            and float(large_raw.get("tol")) == 1.0e-4,
        ),
        _check(
            "managerial_runtime_protocol",
            managerial_raw.get("time_limit") == 900
            and managerial_raw.get("managerial_evaluation_time_limit") == 300
            and managerial_raw.get("max_iterations") == 10000
            and float(managerial_raw.get("tol")) == 1.0e-4,
        ),
        _check(
            "frozen_final_seed_declaration",
            selected.get("tuning_seeds_used") == [0, 1, 2]
            and selected.get("final_evaluation_seeds") == list(range(10, 20))
            and final.get("random_seeds") == list(range(10, 20)),
        ),
    ]
    failed = [check["check"] for check in checks if check["required"] and not check["passed"]]
    return {
        "audit_name": "extended_experiment_protocol_v1",
        "created_at": utc_now_iso(),
        "git_commit": git_commit(root),
        "all_required_checks_passed": not failed,
        "required_check_count": sum(1 for check in checks if check["required"]),
        "passed_check_count": sum(
            1 for check in checks if check["required"] and check["passed"]
        ),
        "failed_checks": failed,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen extended experiment protocols.")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_protocols()
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_required_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
