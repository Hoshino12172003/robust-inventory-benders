from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .experiment_protocol import file_sha256
from .fairness_scalability import SCALABILITY_CANDIDATES
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
EXPECTED_PROTOCOL_HASH = "20DB7AEF62E0C0958C2AC99D51D5F3F326639468CAAC8620419FB9639DDB4C91"
EXPECTED_CONFIG_HASHES = {
    "medium_large": "26A3E952C38E07EE0075CE91158992508762FA973634A53849D5F5C2FAF82B5C",
    "large": "56DDAA42F9ED9F393D0767B05697AE7DD28C6A4D3D04AE5240ED1C443C951107",
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
    _add(checks, "old_ray_not_cached", "self._patterns" in source and "self._rays" not in source)
    _add(checks, "cache_recertifies_current_point", "certify_current_point" in source and "certifier(" in source)
    _add(checks, "call_local_exclusions_removed", "self.model.remove(temporary_exclusions)" in source)
    _add(checks, "objective_bound_certification_preserved", "separation_partition_certifies" in source)
    _add(checks, "benders_success_requires_certification", "separation.robust_feasibility_certified" in benders_source)

    for size, config in configs.items():
        prefix = size
        seeds = config.get("development_seeds", [])
        candidates = config.get("scalability_candidates", [])
        settings = config.get("candidate_settings", {})
        certification = config.get("certification", {})
        _add(checks, f"{prefix}_protocol_only", config.get("authorization") == "protocol_only_no_formal_execution")
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
        report = dry_run_report(config)
        _add(checks, f"{prefix}_dry_run_counts", report["s1"]["total_tasks"] == 27 and report["s2_cumulative"]["total_tasks"] == 90 and report["complete_staged_unique_tasks"] == 120)
        _add(checks, f"{prefix}_scenario_count", report["scenario_count"] == (1831 if size == "medium_large" else 4657))
        _add(checks, f"{prefix}_no_formal_output", report["output_dir_exists"] is False)
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
