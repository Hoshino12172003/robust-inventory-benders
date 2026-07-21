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
MEDIUM_FINAL_SEEDS = list(range(90, 100))
LARGE_FINAL_SEEDS = list(range(100, 110))
PRE_FINAL_SEEDS = set(range(75, 90))
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
EXPECTED_FINAL_ANALYSIS = {
    "primary_metrics": ["large_paired_mean_par2", "large_solved_rate"],
    "secondary_metrics": [
        "medium_large_mean_par2",
        "mean_iterations",
        "paired_win_counts",
        "master_subproblem_core_time",
        "core_success_rate",
        "core_extra_time_share",
        "final_gap",
        "solved_rate",
    ],
    "confirmation_outcomes": {
        "confirmed": "final_confirmed",
        "not_confirmed": "final_not_confirmed",
    },
    "paired_comparison_key": "seed",
    "par2_unsolved_multiplier": 2,
    "development_validation_pooling_allowed": False,
    "seed_replacement_allowed": False,
    "retuning_allowed": False,
    "auxiliary_inference": {
        "enabled": True,
        "role": "auxiliary_only",
        "scope": "large",
        "estimand": "mean_paired_par2_difference_core_minus_v1",
        "method": "paired_nonparametric_bootstrap_percentile",
        "confidence_level": 0.95,
        "resamples": 10000,
        "analysis_random_seed": 20260720,
        "replaces_confirmation_thresholds": False,
    },
}
EXPECTED_FINAL_CONFIGS = {
    "cut_strengthened_joint_v3_final_medium_large.yaml": {
        "validation": "cut_strengthened_joint_v3_validation_medium_large.yaml",
        "instance_size": "medium_large",
        "seeds": MEDIUM_FINAL_SEEDS,
        "time_limit": 600,
        "max_iterations": 10000,
        "output_dir": "experiments/results_cut_v3/final_medium_large",
        "sha256": "1d41a19bb47218f2844c2bdfeadf9b044e8776db944c37989ef8c26feb9c0867",
    },
    "cut_strengthened_joint_v3_final_large.yaml": {
        "validation": "cut_strengthened_joint_v3_validation_large.yaml",
        "instance_size": "large",
        "seeds": LARGE_FINAL_SEEDS,
        "time_limit": 1800,
        "max_iterations": 20000,
        "output_dir": "experiments/results_cut_v3/final_large",
        "sha256": "60fdf4a9a642485a46e473a25ddb7502198a84eea927d9a60e670b764f8542f3",
    },
}
ALLOWED_VALIDATION_DIFFERENCE_FIELDS = {
    "experiment_name",
    "output_dir",
    "random_seeds",
    "protocol_phase",
    "formal_inference_allowed",
    "final_analysis",
}
FORBIDDEN_FINAL_KEYS = {
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


def _normalized_for_validation_comparison(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    for field in ALLOWED_VALIDATION_DIFFERENCE_FIELDS:
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
    for field in (
        "parameters_must_be_fixed_from",
        "candidate_parameters_must_be_fixed_from",
    ):
        resolved[field] = str(config_dir / Path(str(resolved[field])).name)
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


def audit_cut_strengthened_v3_final(
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    config_dir = root / "experiments/configs"
    document_path = root / "docs/cut_strengthened_joint_v3_final_protocol.md"
    decision_path = root / "docs/cut_strengthened_joint_v3_final_decision.md"
    results_audit_path = root / "src/cut_v3_final_results_audit.py"
    validation_decision_path = root / "docs/cut_strengthened_joint_v3_validation_decision.md"
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

    decision = (
        validation_decision_path.read_text(encoding="utf-8")
        if validation_decision_path.exists()
        else ""
    )
    checks.append(
        _check(
            "validation_pass_authorizes_final_protocol_only",
            all(
                token in decision
                for token in (
                    "decision: validation_pass",
                    "selected_candidate: joint_v1_core_point_strengthened",
                    "next_authorized_stage: final_protocol_only",
                    "648556b1956008e93bfc8ac0459cdc3260ab93be",
                )
            ),
        )
    )

    final_outputs: set[str] = set()
    final_run_keys: set[str] = set()
    prior_outputs: set[str] = set()
    prior_run_keys: set[str] = set()
    final_seed_sets: list[set[int]] = []

    for filename, expected in EXPECTED_FINAL_CONFIGS.items():
        path = config_dir / filename
        raw = load_config(path) if path.exists() else {}
        validation_path = config_dir / str(expected["validation"])
        validation = load_config(validation_path) if validation_path.exists() else {}
        prefix = filename.removesuffix(".yaml")
        seeds = set(raw.get("random_seeds", []))
        final_seed_sets.append({int(seed) for seed in seeds})
        actual_hash = file_sha256(path).lower() if path.exists() else "missing"
        output_dir = str(raw.get("output_dir", ""))
        output_path = root / output_dir

        checks.extend(
            [
                _check(
                    f"{prefix}_config_sha256_frozen",
                    actual_hash == expected["sha256"],
                    actual_hash,
                ),
                _check(
                    f"{prefix}_seeds_exact",
                    raw.get("random_seeds") == expected["seeds"],
                    raw.get("random_seeds"),
                ),
                _check(
                    f"{prefix}_pre_final_seeds_excluded",
                    seeds.isdisjoint(PRE_FINAL_SEEDS),
                    sorted(seeds & PRE_FINAL_SEEDS),
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
                    f"{prefix}_final_phase_and_analysis_frozen",
                    raw.get("protocol_phase") == "final"
                    and raw.get("formal_inference_allowed") is True
                    and raw.get("final_analysis") == EXPECTED_FINAL_ANALYSIS,
                ),
                _check(
                    f"{prefix}_validation_equivalence",
                    _normalized_for_validation_comparison(raw)
                    == _normalized_for_validation_comparison(validation),
                ),
                _check(
                    f"{prefix}_no_fairness_managerial_or_new_uncertainty_keys",
                    _recursive_keys(raw).isdisjoint(FORBIDDEN_FINAL_KEYS),
                    sorted(_recursive_keys(raw) & FORBIDDEN_FINAL_KEYS),
                ),
                _check(
                    f"{prefix}_isolated_output_directory",
                    output_dir == expected["output_dir"]
                    and "final" in output_dir
                    and "development" not in output_dir
                    and "validation" not in output_dir
                    and output_dir != str(validation.get("output_dir", "")),
                    output_dir,
                ),
                _check(
                    f"{prefix}_no_instances_or_results_generated",
                    not output_path.exists(),
                    str(output_path),
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
                and candidate.get("cut_selection_enabled") is False
                and candidate.get("adaptive_secondary_cut_selection_enabled") is False
                and candidate.get("adaptive_secondary_generation_enabled") is False
                and candidate.get("adaptive_subproblem_gap_enabled") is False
                and candidate.get("subproblem_mode") == "robust_dual_milp"
                and resolved.get("gamma_continuation_enabled") is False
                and resolved.get("gamma_schedule") == [2]
            )
            resolution_error = ""
        except Exception as exc:  # noqa: BLE001 - audit reports malformed locks.
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
            final_outputs.add(output_dir)
            final_run_keys.update(_run_key_set(raw))
        if validation:
            prior_outputs.add(str(validation.get("output_dir")))
            prior_run_keys.update(_run_key_set(validation))

    for prior_name in (
        "cut_strengthened_joint_v3_development_medium_large.yaml",
        "cut_strengthened_joint_v3_development_large.yaml",
    ):
        prior_path = config_dir / prior_name
        if prior_path.exists():
            prior = load_config(prior_path)
            prior_outputs.add(str(prior.get("output_dir")))
            prior_run_keys.update(_run_key_set(prior))

    actual_final_names = {
        path.name for path in config_dir.glob("cut_strengthened_joint_v3_final*.yaml")
    }
    checks.extend(
        [
            _check(
                "only_expected_final_configs_exist",
                actual_final_names == set(EXPECTED_FINAL_CONFIGS),
                sorted(actual_final_names),
            ),
            _check(
                "final_seed_groups_disjoint",
                len(final_seed_sets) == 2
                and final_seed_sets[0].isdisjoint(final_seed_sets[1]),
            ),
            _check(
                "final_outputs_and_resume_keys_isolated",
                final_outputs.isdisjoint(prior_outputs)
                and final_run_keys.isdisjoint(prior_run_keys),
            ),
        ]
    )

    document = document_path.read_text(encoding="utf-8") if document_path.exists() else ""
    final_decision = decision_path.read_text(encoding="utf-8") if decision_path.exists() else ""
    checks.extend(
        [
            _check(
                "final_confirmation_rules_frozen_in_document",
                all(
                    token in document
                    for token in (
                        "final_confirmed",
                        "final_not_confirmed",
                        "103%",
                        "7.5%",
                        "15%",
                        "at least 6/10",
                        "mean paired rank",
                    )
                ),
            ),
            _check(
                "auxiliary_inference_frozen_in_document",
                all(
                    token in document
                    for token in (
                        "paired nonparametric percentile bootstrap",
                        "95%",
                        "10,000",
                        "20260720",
                        "cannot replace",
                    )
                ),
            ),
            _check(
                "final_not_a_retuning_or_seed_replacement_stage",
                "does not authorize seed replacement" in document
                and "not a tuning stage" in document,
            ),
            _check(
                "final_decision_frozen_after_read_only_audit",
                all(
                    token in final_decision
                    for token in (
                        "decision: final_confirmed",
                        "selected_algorithm: joint_v1_core_point_strengthened",
                        "v3_status: completed",
                        "retuning_allowed: false",
                        "seed_replacement_allowed: false",
                        "development_validation_pooling_allowed: false",
                        "next_authorized_stage: fairness_diagnostic_only",
                    )
                ),
            ),
            _check(
                "final_decision_evidence_identity_frozen",
                all(
                    token in final_decision
                    for token in (
                        "11020383bfaf49b6f538f672089704f1cdf8b860",
                        "1388446BC75E44E8E8AFC9E7973F011B14B7172AEBC8BB400749C6FE7C1D1E7A",
                        "6641DCD67F8BFD6FA15F7580459AD31148BFF7DF64034E16C1A36F98B78985F4",
                        "skipped_run_count: 4",
                        "-774.4814981500152",
                        "-977.7605067112486",
                        "-594.9803125280666",
                    )
                ),
            ),
            _check(
                "read_only_final_results_audit_present",
                results_audit_path.exists()
                and "def audit_cut_v3_final_results(" in results_audit_path.read_text(encoding="utf-8"),
            ),
        ]
    )

    failed = [check["check"] for check in checks if not check["passed"]]
    return {
        "audit_name": "cut_strengthened_joint_v3_final_protocol",
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
        description="Audit the frozen cut-strengthened Joint V3 final protocol."
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_cut_strengthened_v3_final()
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_required_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
