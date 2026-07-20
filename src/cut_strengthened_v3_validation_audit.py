from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import load_config
from .cut_strengthened_v3_audit import (
    FROZEN_CONFIG_SHA256,
    SELECTED_CANDIDATE_CONFIG_NAME,
    SELECTED_CANDIDATE_CONFIG_SHA256,
)
from .experiment_protocol import (
    atomic_write_json,
    file_sha256,
    git_commit,
    stable_run_key,
    utc_now_iso,
)
from .experiment_suite import (
    _apply_selected_parameters,
    _apply_variant_config,
    _base_config,
    _variant_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DECISION_DOCUMENT = "cut_strengthened_joint_v3_validation_decision.md"
VALIDATION_SEEDS = list(range(80, 90))
FINAL_SEEDS = set(range(90, 110))
EXPECTED_VARIANTS = [
    "proposed_joint_rho025_050",
    "joint_v1_core_point_strengthened",
]
EXPECTED_VARIANT_SETTINGS = {
    "proposed_joint_rho025_050": {
        "cut_strengthening_policy": "none",
        "max_cuts_per_iteration": 1,
    },
    "joint_v1_core_point_strengthened": {
        "cut_strengthening_policy": "core_point",
        "max_cuts_per_iteration": 1,
    },
}
EXPECTED_VALIDATION_CONFIGS = {
    "cut_strengthened_joint_v3_validation_medium_large.yaml": {
        "development": "cut_strengthened_joint_v3_development_medium_large.yaml",
        "instance_size": "medium_large",
        "time_limit": 600,
        "max_iterations": 10000,
        "output_dir": "experiments/results_cut_v3/validation_medium_large",
        "sha256": "eb7070b8045cfd3fc57b4f7dc906059f8c9ca60d9c0ad58b75cd6e8e98d41007",
    },
    "cut_strengthened_joint_v3_validation_large.yaml": {
        "development": "cut_strengthened_joint_v3_development_large.yaml",
        "instance_size": "large",
        "time_limit": 1800,
        "max_iterations": 20000,
        "output_dir": "experiments/results_cut_v3/validation_large",
        "sha256": "44106f8a1f12d4caca961439ca4b5eebf8ca263afac567512ce541f4e80ace27",
    },
}
ALLOWED_DEVELOPMENT_DIFFERENCE_FIELDS = {
    "experiment_name",
    "output_dir",
    "random_seeds",
    "variants",
    "variant_settings",
    "protocol_phase",
    "candidate_parameters_must_be_fixed_from",
    "candidate_config_sha256",
}
FORBIDDEN_VALIDATION_KEYS = {
    "fairness",
    "equity",
    "protected_group",
    "demographic_group",
    "sensitivity_axis",
    "sensitivity_axes",
    "managerial_evaluation_time_limit",
    "service_level",
    "budget_factor",
    "capacity_factor",
    "demand_deviation_factor",
    "uncertainty_set",
    "uncertainty_budget",
}


def _check(name: str, passed: bool, details: Any = "") -> dict[str, Any]:
    return {"check": name, "required": True, "passed": bool(passed), "details": details}


def _normalized_for_development_comparison(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    for field in ALLOWED_DEVELOPMENT_DIFFERENCE_FIELDS:
        normalized.pop(field, None)
    return normalized


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


def _absolute_lock_paths(config: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    resolved = deepcopy(config)
    selected_path = Path(str(resolved["parameters_must_be_fixed_from"])).name
    candidate_path = Path(str(resolved["candidate_parameters_must_be_fixed_from"])).name
    resolved["parameters_must_be_fixed_from"] = str(config_dir / selected_path)
    resolved["candidate_parameters_must_be_fixed_from"] = str(config_dir / candidate_path)
    return resolved


def _effective_variant(config: dict[str, Any], variant_name: str) -> dict[str, Any]:
    variants = {
        name: (method, settings)
        for name, method, settings in _variant_specs(config)
    }
    method, settings = variants[variant_name]
    base = _base_config(
        config,
        str(config["instance_sizes"][0]),
        int(config["random_seeds"][0]),
    )
    _solver_method, _flags, effective = _apply_variant_config(base, method, settings)
    return effective


def _run_key_set(config: dict[str, Any]) -> set[str]:
    return {
        stable_run_key(
            experiment_name=str(config["experiment_name"]),
            sensitivity_axis=None,
            sensitivity_value=None,
            instance_size=str(size),
            seed=int(seed),
            variant_name=str(variant),
        )
        for seed in config["random_seeds"]
        for size in config["instance_sizes"]
        for variant in config["variants"]
    }


def audit_cut_strengthened_v3_validation(
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    config_dir = root / "experiments/configs"
    document_path = root / "docs/cut_strengthened_joint_v3_validation_protocol.md"
    decision_document_path = root / "docs" / VALIDATION_DECISION_DOCUMENT
    checks: list[dict[str, Any]] = []

    for name, expected_hash in FROZEN_CONFIG_SHA256.items():
        path = config_dir / name
        actual = file_sha256(path).lower() if path.exists() else "missing"
        checks.append(_check(f"frozen_{name}_unchanged", actual == expected_hash, actual))

    candidate_path = config_dir / SELECTED_CANDIDATE_CONFIG_NAME
    candidate_hash = file_sha256(candidate_path).lower() if candidate_path.exists() else "missing"
    checks.append(
        _check(
            "selected_candidate_sha256_frozen",
            candidate_hash == SELECTED_CANDIDATE_CONFIG_SHA256,
            candidate_hash,
        )
    )

    decision_document = (
        decision_document_path.read_text(encoding="utf-8")
        if decision_document_path.exists()
        else ""
    )
    checks.extend(
        [
            _check(
                "validation_decision_is_frozen_pass",
                all(
                    token in decision_document
                    for token in (
                        "decision: validation_pass",
                        "selected_candidate: joint_v1_core_point_strengthened",
                        "next_authorized_stage: final_protocol_only",
                        "formal_inference_allowed: false",
                        "648556b1956008e93bfc8ac0459cdc3260ab93be",
                    )
                ),
            ),
            _check(
                "validation_decision_keeps_final_seeds_sealed",
                "90--109 remain sealed" in decision_document
                and "does not authorize a final run" in decision_document,
            ),
            _check(
                "read_only_validation_results_audit_available",
                (root / "src/cut_v3_validation_results_audit.py").exists(),
            ),
        ]
    )

    all_validation_outputs: set[str] = set()
    all_validation_keys: set[str] = set()
    all_development_outputs: set[str] = set()
    all_development_keys: set[str] = set()
    for filename, expected in EXPECTED_VALIDATION_CONFIGS.items():
        path = config_dir / filename
        raw = load_config(path) if path.exists() else {}
        development_path = config_dir / str(expected["development"])
        development = load_config(development_path) if development_path.exists() else {}
        prefix = filename.removesuffix(".yaml")

        actual_hash = file_sha256(path).lower() if path.exists() else "missing"
        checks.extend(
            [
                _check(
                    f"{prefix}_config_sha256_frozen",
                    actual_hash == expected["sha256"],
                    actual_hash,
                ),
                _check(
                    f"{prefix}_seeds_exact_80_89",
                    raw.get("random_seeds") == VALIDATION_SEEDS,
                    raw.get("random_seeds"),
                ),
                _check(
                    f"{prefix}_final_seeds_excluded",
                    set(raw.get("random_seeds", [])).isdisjoint(FINAL_SEEDS),
                ),
                _check(
                    f"{prefix}_only_v1_and_core_candidate",
                    raw.get("variants") == EXPECTED_VARIANTS
                    and raw.get("variant_settings") == EXPECTED_VARIANT_SETTINGS,
                    raw.get("variants"),
                ),
                _check(
                    f"{prefix}_run_count_20",
                    len(raw.get("random_seeds", []))
                    * len(raw.get("instance_sizes", []))
                    * len(raw.get("variants", []))
                    == 20,
                ),
                _check(
                    f"{prefix}_scale_and_limits_frozen",
                    raw.get("instance_sizes") == [expected["instance_size"]]
                    and raw.get("time_limit") == expected["time_limit"]
                    and raw.get("max_iterations") == expected["max_iterations"],
                ),
                _check(
                    f"{prefix}_candidate_lock_reference",
                    Path(str(raw.get("candidate_parameters_must_be_fixed_from", ""))).name
                    == SELECTED_CANDIDATE_CONFIG_NAME
                    and str(raw.get("candidate_config_sha256", "")).lower()
                    == SELECTED_CANDIDATE_CONFIG_SHA256,
                ),
                _check(
                    f"{prefix}_validation_phase_only",
                    raw.get("protocol_phase") == "validation"
                    and raw.get("formal_inference_allowed") is False,
                ),
                _check(
                    f"{prefix}_development_equivalence",
                    _normalized_for_development_comparison(raw)
                    == _normalized_for_development_comparison(development),
                ),
                _check(
                    f"{prefix}_no_fairness_managerial_or_new_uncertainty_keys",
                    _recursive_keys(raw).isdisjoint(FORBIDDEN_VALIDATION_KEYS),
                    sorted(_recursive_keys(raw) & FORBIDDEN_VALIDATION_KEYS),
                ),
                _check(
                    f"{prefix}_isolated_output_directory",
                    raw.get("output_dir") == expected["output_dir"]
                    and raw.get("output_dir") != development.get("output_dir")
                    and "validation" in str(raw.get("output_dir", ""))
                    and "development" not in str(raw.get("output_dir", ""))
                    and "final" not in str(raw.get("output_dir", "")),
                    raw.get("output_dir"),
                ),
            ]
        )

        try:
            resolved = _apply_selected_parameters(_absolute_lock_paths(raw, config_dir))
            v1 = _effective_variant(resolved, "proposed_joint_rho025_050")["algorithm"]
            candidate = _effective_variant(
                resolved, "joint_v1_core_point_strengthened"
            )["algorithm"]
            effective_ok = (
                v1.get("precision_policy") == "joint_error_budget"
                and candidate.get("precision_policy") == "joint_error_budget"
                and v1.get("master_error_budget_ratio") == 0.25
                and v1.get("subproblem_error_budget_ratio") == 0.50
                and candidate.get("master_error_budget_ratio") == 0.25
                and candidate.get("subproblem_error_budget_ratio") == 0.50
                and v1.get("cut_strengthening_policy") == "none"
                and candidate.get("cut_strengthening_policy") == "core_point"
                and v1.get("max_cuts_per_iteration") == 1
                and candidate.get("max_cuts_per_iteration") == 1
                and candidate.get("adaptive_secondary_generation_enabled") is False
                and candidate.get("cut_selection_enabled") is False
            )
            resolution_error = ""
        except Exception as exc:  # noqa: BLE001 - audit reports malformed locks.
            resolved = raw
            effective_ok = False
            resolution_error = f"{type(exc).__name__}: {exc}"
        checks.append(
            _check(
                f"{prefix}_effective_v1_and_frozen_core_only",
                effective_ok,
                resolution_error,
            )
        )

        if raw:
            all_validation_outputs.add(str(raw.get("output_dir")))
            all_validation_keys.update(_run_key_set(raw))
        if development:
            all_development_outputs.add(str(development.get("output_dir")))
            all_development_keys.update(_run_key_set(development))

    actual_validation_names = {
        path.name
        for path in config_dir.glob("cut_strengthened_joint_v3_validation*.yaml")
    }
    checks.extend(
        [
            _check(
                "only_expected_validation_configs_exist",
                actual_validation_names == set(EXPECTED_VALIDATION_CONFIGS),
                sorted(actual_validation_names),
            ),
            _check(
                "no_v3_final_config_created",
                not any(config_dir.glob("*cut_strengthened_joint_v3*final*.yaml")),
            ),
            _check(
                "resume_outputs_and_run_keys_isolated_from_development",
                all_validation_outputs.isdisjoint(all_development_outputs)
                and all_validation_keys.isdisjoint(all_development_keys),
            ),
        ]
    )

    document = document_path.read_text(encoding="utf-8") if document_path.exists() else ""
    checks.append(
        _check(
            "validation_decision_rules_frozen_in_document",
            all(
                token in document
                for token in (
                    "Validation **pass**",
                    "Validation **fail**",
                    "Validation **inconclusive**",
                    "7.5%",
                    "15%",
                    "至少 6 个",
                    "2 × time_limit",
                    "seeds 90–109",
                    "不得合并进行正式统计推断",
                )
            ),
        )
    )

    failed = [check["check"] for check in checks if not check["passed"]]
    return {
        "audit_name": "cut_strengthened_joint_v3_validation_protocol",
        "created_at": utc_now_iso(),
        "git_commit": git_commit(root),
        "all_required_checks_passed": not failed,
        "required_check_count": len(checks),
        "passed_check_count": sum(check["passed"] for check in checks),
        "failed_checks": failed,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the frozen cut-strengthened Joint V3 validation protocol."
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_cut_strengthened_v3_validation()
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_required_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
