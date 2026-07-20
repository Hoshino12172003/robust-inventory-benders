from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.cut_strengthened_v3_audit import (
    DEVELOPMENT_SEEDS,
    EXPECTED_CONFIGS,
    EXPECTED_VARIANTS,
    PREVIOUS_SEEDS,
    RESERVED_LARGE_FINAL_SEEDS,
    RESERVED_MEDIUM_FINAL_SEEDS,
    RESERVED_VALIDATION_SEEDS,
    audit_cut_strengthened_v3,
)
from src.experiment_suite import (
    _apply_selected_parameters,
    _apply_variant_config,
    _base_config,
    _variant_specs,
    experiment_dry_run_report,
    experiment_run_specs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments/configs"


@pytest.mark.parametrize("filename,expected", EXPECTED_CONFIGS.items())
def test_development_config_expands_to_twenty_runs(
    filename: str,
    expected: dict[str, object],
) -> None:
    raw = load_config(CONFIG_DIR / filename)
    resolved = _apply_selected_parameters(raw)
    specs = experiment_run_specs(resolved)
    assert len(specs) == 20
    assert {spec.seed for spec in specs} == DEVELOPMENT_SEEDS
    assert {spec.instance_size for spec in specs} == {expected["size"]}
    assert sorted({spec.variant_name for spec in specs}) == sorted(EXPECTED_VARIANTS)


def test_four_variants_keep_v1_precision_and_use_distinct_cut_policies() -> None:
    raw = load_config(CONFIG_DIR / "cut_strengthened_joint_v3_development_medium_large.yaml")
    resolved = _apply_selected_parameters(raw)
    base = _base_config(resolved, "medium_large", 75)
    expected = {
        "proposed_joint_rho025_050": ("none", 1),
        "joint_v1_core_point_strengthened": ("core_point", 1),
        "joint_v1_stall_secondary_cut": ("stall_secondary", 2),
        "proposed_cut_strengthened_joint_v3": ("core_point_stall_secondary", 2),
    }
    for name, method, settings in _variant_specs(resolved):
        _solver, _flags, run = _apply_variant_config(base, method, settings)
        policy, max_cuts = expected[name]
        assert run["algorithm"]["precision_policy"] == "joint_error_budget"
        assert run["algorithm"]["precision_policy"] != "workload_aware_joint"
        assert run["algorithm"]["cut_strengthening_policy"] == policy
        assert run["algorithm"]["max_cuts_per_iteration"] == max_cuts
        assert run["algorithm"]["adaptive_secondary_generation_enabled"] is False


def test_seed_groups_are_pairwise_disjoint_and_do_not_reuse_zero_to_seventy_four() -> None:
    groups = [
        DEVELOPMENT_SEEDS,
        RESERVED_VALIDATION_SEEDS,
        RESERVED_MEDIUM_FINAL_SEEDS,
        RESERVED_LARGE_FINAL_SEEDS,
    ]
    assert all(
        groups[left].isdisjoint(groups[right])
        for left in range(len(groups))
        for right in range(left + 1, len(groups))
    )
    assert PREVIOUS_SEEDS.isdisjoint(set().union(*groups))


@pytest.mark.parametrize(
    "filename,expected_hours",
    [
        ("cut_strengthened_joint_v3_development_medium_large.yaml", 20 * 600 / 3600),
        ("cut_strengthened_joint_v3_development_large.yaml", 20 * 1800 / 3600),
    ],
)
def test_dry_run_reports_twenty_runs_without_audit_errors(
    filename: str,
    expected_hours: float,
) -> None:
    report = experiment_dry_run_report(load_config(CONFIG_DIR / filename))
    assert report["total_run_count"] == 20
    assert report["methods"] == EXPECTED_VARIANTS
    assert report["theoretical_maximum_hours"] == pytest.approx(expected_hours)
    assert report["protocol_audit_errors"] == []
    assert report["automatic_parallelism_enabled"] is False


def test_v3_static_audit_passes_and_v1_hashes_are_unchanged() -> None:
    report = audit_cut_strengthened_v3(ROOT)
    assert report["all_required_checks_passed"], report["failed_checks"]
    assert report["passed_check_count"] == report["required_check_count"]


def test_only_frozen_validation_configs_and_no_final_configs_exist() -> None:
    assert {
        path.name
        for path in CONFIG_DIR.glob("cut_strengthened_joint_v3_validation*.yaml")
    } == {
        "cut_strengthened_joint_v3_validation_medium_large.yaml",
        "cut_strengthened_joint_v3_validation_large.yaml",
    }
    assert not list(CONFIG_DIR.glob("cut_strengthened_joint_v3_final*.yaml"))
    assert "experiments/results_cut_v3/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
