from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .config import load_config
from .experiment_protocol import atomic_write_json, file_sha256, git_commit, utc_now_iso
from .experiment_suite import (
    _apply_selected_parameters,
    _apply_variant_config,
    _base_config,
    _variant_specs,
    experiment_run_specs,
)
from .precision_policy import (
    initialize_workload_aware_state,
    precision_policy_config,
    select_workload_aware_precision,
    workload_aware_precision_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "experiments/configs"
DOCUMENT_PATH = REPO_ROOT / "docs/workload_aware_joint_v2_protocol.md"

FROZEN_CONFIG_SHA256 = {
    "selected_algorithm_parameters.yaml": "50b275578a127b349bdda47ff161680048cd1c0c8845ea85e707949bdfa29d25",
    "final_evaluation_joint_v1.yaml": "efa7d3406687d4a7a7a99726eaa19f604f0f5b10cf9f38709420dcec8bf1195f",
    "large_scale_evaluation_joint_v1.yaml": "689d5b8e1ca7b277137a27a75ecb1405da98b2adcd9a9a11481c48d5a5bda539",
    "managerial_sensitivity_joint_v1.yaml": "b7d7880a3f106c1a7a2560b3d0a30a23369980a2fbece3498e21d541310be796",
}

DEVELOPMENT_SEEDS = set(range(40, 45))
VALIDATION_SEEDS = set(range(45, 55))
RESERVED_MEDIUM_LARGE_FINAL_SEEDS = set(range(55, 65))
RESERVED_LARGE_FINAL_SEEDS = set(range(65, 75))
PREVIOUSLY_USED_SEEDS = set(range(0, 40))
EXPECTED_VARIANTS = [
    "mp_adaptive_rho050",
    "proposed_joint_rho025_050",
    "proposed_workload_aware_joint_v2",
]
V2_VARIANT = "proposed_workload_aware_joint_v2"
EXPECTED_CONFIGS = {
    "workload_aware_joint_v2_development_medium_large.yaml": {
        "seeds": DEVELOPMENT_SEEDS,
        "size": "medium_large",
        "runs": 15,
        "time_limit": 600,
        "max_iterations": 10000,
        "phase": "development",
    },
    "workload_aware_joint_v2_development_large.yaml": {
        "seeds": DEVELOPMENT_SEEDS,
        "size": "large",
        "runs": 15,
        "time_limit": 1800,
        "max_iterations": 20000,
        "phase": "development",
    },
    "workload_aware_joint_v2_validation_medium_large.yaml": {
        "seeds": VALIDATION_SEEDS,
        "size": "medium_large",
        "runs": 30,
        "time_limit": 600,
        "max_iterations": 10000,
        "phase": "validation",
    },
    "workload_aware_joint_v2_validation_large.yaml": {
        "seeds": VALIDATION_SEEDS,
        "size": "large",
        "runs": 30,
        "time_limit": 1800,
        "max_iterations": 20000,
        "phase": "validation",
    },
}

SELECTION_THRESHOLDS = {
    "medium_large_v1_par2_ratio_max": 1.03,
    "medium_large_mean_degradation_percentage_points_max": 3,
    "large_v1_par2_reduction_min": 0.05,
    "large_mp_only_par2_reduction_min": 0.03,
    "large_instance_par2_noninferior_count_min": 6,
    "large_validation_instance_count": 10,
}


def _check(name: str, passed: bool, details: Any = "") -> dict[str, Any]:
    return {"check": name, "required": True, "passed": bool(passed), "details": details}


def _resolved_variant(config: dict[str, Any], variant_name: str) -> dict[str, Any]:
    resolved = _apply_selected_parameters(config)
    base = _base_config(resolved, str(resolved["instance_sizes"][0]), int(resolved["random_seeds"][0]))
    methods = {name: (method, settings) for name, method, settings in _variant_specs(resolved)}
    method, settings = methods[variant_name]
    _method, _flags, run_config = _apply_variant_config(
        base,
        method,
        settings,
    )
    return run_config


def audit_workload_aware_v2(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    config_dir = root / "experiments/configs"
    configs = {name: load_config(config_dir / name) for name in EXPECTED_CONFIGS}
    checks: list[dict[str, Any]] = []

    for name, expected_hash in FROZEN_CONFIG_SHA256.items():
        actual_hash = file_sha256(config_dir / name).lower()
        checks.append(_check(f"frozen_{name}_unchanged", actual_hash == expected_hash, actual_hash))

    all_new_seed_sets = [
        DEVELOPMENT_SEEDS,
        VALIDATION_SEEDS,
        RESERVED_MEDIUM_LARGE_FINAL_SEEDS,
        RESERVED_LARGE_FINAL_SEEDS,
    ]
    checks.append(
        _check(
            "new_seed_sets_pairwise_disjoint",
            all(
                all_new_seed_sets[i].isdisjoint(all_new_seed_sets[j])
                for i in range(len(all_new_seed_sets))
                for j in range(i + 1, len(all_new_seed_sets))
            ),
        )
    )
    checks.append(
        _check(
            "all_new_seeds_disjoint_from_0_39",
            PREVIOUSLY_USED_SEEDS.isdisjoint(set().union(*all_new_seed_sets)),
        )
    )
    checks.append(
        _check(
            "final_seeds_reserved_55_74",
            RESERVED_MEDIUM_LARGE_FINAL_SEEDS == set(range(55, 65))
            and RESERVED_LARGE_FINAL_SEEDS == set(range(65, 75)),
        )
    )

    for name, expected in EXPECTED_CONFIGS.items():
        raw = configs[name]
        resolved = _apply_selected_parameters(raw)
        specs = experiment_run_specs(resolved)
        v2_run = _resolved_variant(raw, V2_VARIANT)
        v2_algorithm = v2_run["algorithm"]
        checks.extend(
            [
                _check(f"{name}_run_count", len(specs) == expected["runs"], len(specs)),
                _check(f"{name}_seed_set", set(raw.get("random_seeds", [])) == expected["seeds"]),
                _check(f"{name}_instance_size", raw.get("instance_sizes") == [expected["size"]]),
                _check(f"{name}_variants", raw.get("variants") == EXPECTED_VARIANTS, raw.get("variants")),
                _check(
                    f"{name}_runtime_limits",
                    raw.get("time_limit") == expected["time_limit"]
                    and raw.get("max_iterations") == expected["max_iterations"],
                ),
                _check(f"{name}_phase", raw.get("protocol_phase") == expected["phase"]),
                _check(
                    f"{name}_correctness_controls",
                    v2_algorithm.get("subproblem_mode") == "robust_dual_milp"
                    and v2_run.get("gamma_continuation_enabled") is False
                    and v2_run["robust"].get("gamma_schedule") == [2]
                    and v2_algorithm.get("cut_selection_enabled") is False
                    and v2_algorithm.get("adaptive_secondary_generation_enabled") is False
                    and v2_algorithm.get("adaptive_secondary_cut_selection_enabled") is False
                    and v2_algorithm.get("max_cuts_per_iteration") == 1,
                ),
                _check(
                    f"{name}_v2_precision_policy_and_bounds",
                    v2_algorithm.get("precision_policy") == "workload_aware_joint"
                    and math.isclose(float(v2_algorithm.get("master_gap_min")), 0.0001)
                    and math.isclose(float(v2_algorithm.get("master_gap_max")), 0.02)
                    and math.isclose(float(v2_algorithm.get("subproblem_gap_min")), 0.0001)
                    and math.isclose(float(v2_algorithm.get("subproblem_gap_max")), 0.05),
                ),
            ]
        )

    reference = configs["workload_aware_joint_v2_development_medium_large.yaml"]
    v1_algorithm = _resolved_variant(reference, "proposed_joint_rho025_050")["algorithm"]
    v2_algorithm = _resolved_variant(reference, V2_VARIANT)["algorithm"]
    precision = precision_policy_config(
        v2_algorithm,
        fixed_master_gap=0.02,
        fixed_subproblem_gap=0.05,
        legacy_subproblem_gaps=[0.05],
    )
    workload = workload_aware_precision_config(v2_algorithm)
    initial = select_workload_aware_precision(
        precision,
        workload,
        initialize_workload_aware_state(precision),
        upper_bound=None,
        lower_bound=None,
    )

    checks.extend(
        [
            _check("v1_precision_policy_remains_joint_error_budget", v1_algorithm["precision_policy"] == "joint_error_budget"),
            _check("v2_uses_independent_precision_policy", v2_algorithm["precision_policy"] == "workload_aware_joint"),
            _check(
                "v2_gap_bounds_equal_v1",
                all(
                    math.isclose(float(v2_algorithm[key]), float(v1_algorithm[key]), rel_tol=0.0, abs_tol=1e-15)
                    for key in ("master_gap_min", "master_gap_max", "subproblem_gap_min", "subproblem_gap_max")
                )
                and math.isclose(float(v2_algorithm["master_gap_max"]), 0.02)
                and math.isclose(float(v2_algorithm["subproblem_gap_max"]), 0.05),
            ),
            _check("workload_total_error_budget_ratio", math.isclose(workload.total_error_budget_ratio, 0.75)),
            _check(
                "workload_initial_weights",
                math.isclose(workload.initial_master_weight, 1.0 / 3.0)
                and math.isclose(workload.initial_subproblem_weight, 2.0 / 3.0),
            ),
            _check(
                "workload_initial_ratios_restore_v1",
                math.isclose(initial.master_ratio_selected, 0.25)
                and math.isclose(initial.subproblem_ratio_selected, 0.50),
            ),
            _check(
                "workload_master_weight_bounds",
                0.0 <= workload.master_weight_min <= workload.initial_master_weight
                <= workload.master_weight_max <= 1.0,
            ),
            _check("workload_ema_decay_range", 0.0 <= workload.ema_decay < 1.0),
            _check(
                "workload_defaults_frozen",
                math.isclose(workload.ema_decay, 0.80)
                and math.isclose(workload.master_weight_min, 1.0 / 3.0)
                and math.isclose(workload.master_weight_max, 2.0 / 3.0)
                and math.isclose(workload.time_epsilon, 1.0e-9),
            ),
        ]
    )

    document = (root / "docs/workload_aware_joint_v2_protocol.md").read_text(encoding="utf-8")
    threshold_tokens = (
        "1.03",
        "3 个百分点",
        "5%",
        "3%",
        "至少 6 个",
        "选择 V1",
        "[55, 56, 57, 58, 59, 60, 61, 62, 63, 64]",
        "[65, 66, 67, 68, 69, 70, 71, 72, 73, 74]",
    )
    checks.append(
        _check(
            "selection_thresholds_frozen_in_document",
            all(token in document for token in threshold_tokens),
            SELECTION_THRESHOLDS,
        )
    )

    failed = [check["check"] for check in checks if not check["passed"]]
    return {
        "audit_name": "workload_aware_joint_v2_protocol",
        "created_at": utc_now_iso(),
        "git_commit": git_commit(root),
        "all_required_checks_passed": not failed,
        "required_check_count": len(checks),
        "passed_check_count": sum(check["passed"] for check in checks),
        "failed_checks": failed,
        "selection_thresholds": SELECTION_THRESHOLDS,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the workload-aware Joint V2 protocol.")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_workload_aware_v2()
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_required_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
