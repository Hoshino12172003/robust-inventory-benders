from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.experiment_suite import (
    _apply_selected_parameters,
    _apply_variant_config,
    _base_config,
    _variant_specs,
    experiment_dry_run_report,
    experiment_run_specs,
)
from src.workload_aware_v2_audit import (
    DEVELOPMENT_SEEDS,
    EXPECTED_CONFIGS,
    EXPECTED_VARIANTS,
    PREVIOUSLY_USED_SEEDS,
    RESERVED_LARGE_FINAL_SEEDS,
    RESERVED_MEDIUM_LARGE_FINAL_SEEDS,
    VALIDATION_SEEDS,
    audit_workload_aware_v2,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments/configs"


@pytest.mark.parametrize("filename,expected", EXPECTED_CONFIGS.items())
def test_v2_config_expansion_is_frozen(filename: str, expected: dict[str, object]) -> None:
    config = load_config(CONFIG_DIR / filename)
    resolved = _apply_selected_parameters(config)
    specs = experiment_run_specs(resolved)
    assert len(specs) == expected["runs"]
    assert {spec.seed for spec in specs} == expected["seeds"]
    assert {spec.instance_size for spec in specs} == {expected["size"]}
    assert sorted({spec.variant_name for spec in specs}) == sorted(EXPECTED_VARIANTS)


def test_development_validation_and_reserved_seeds_are_isolated() -> None:
    groups = [
        DEVELOPMENT_SEEDS,
        VALIDATION_SEEDS,
        RESERVED_MEDIUM_LARGE_FINAL_SEEDS,
        RESERVED_LARGE_FINAL_SEEDS,
    ]
    assert all(
        groups[left].isdisjoint(groups[right])
        for left in range(len(groups))
        for right in range(left + 1, len(groups))
    )
    assert PREVIOUSLY_USED_SEEDS.isdisjoint(set().union(*groups))


def test_v1_and_v2_resolve_to_distinct_precision_policies() -> None:
    raw = load_config(CONFIG_DIR / "workload_aware_joint_v2_development_medium_large.yaml")
    config = _apply_selected_parameters(raw)
    base = _base_config(config, "medium_large", seed=40)
    settings = config["variant_settings"]
    methods = {name: method for name, method, _variant in _variant_specs(config)}
    _method, _flags, v1 = _apply_variant_config(
        base, methods["proposed_joint_rho025_050"], settings["proposed_joint_rho025_050"]
    )
    _method, _flags, v2 = _apply_variant_config(
        base, methods["proposed_workload_aware_joint_v2"], settings["proposed_workload_aware_joint_v2"]
    )
    assert v1["algorithm"]["precision_policy"] == "joint_error_budget"
    assert v2["algorithm"]["precision_policy"] == "workload_aware_joint"
    assert v2["algorithm"]["master_gap_max"] == v1["algorithm"]["master_gap_max"] == 0.02
    assert v2["algorithm"]["subproblem_gap_max"] == v1["algorithm"]["subproblem_gap_max"] == 0.05


@pytest.mark.parametrize(
    "filename,expected_runs,expected_hours",
    [
        ("workload_aware_joint_v2_development_medium_large.yaml", 15, 2.5),
        ("workload_aware_joint_v2_development_large.yaml", 15, 7.5),
        ("workload_aware_joint_v2_validation_medium_large.yaml", 30, 5.0),
        ("workload_aware_joint_v2_validation_large.yaml", 30, 15.0),
    ],
)
def test_dry_run_expands_without_protocol_errors(
    filename: str,
    expected_runs: int,
    expected_hours: float,
) -> None:
    report = experiment_dry_run_report(load_config(CONFIG_DIR / filename))
    assert report["total_run_count"] == expected_runs
    assert report["theoretical_maximum_hours"] == pytest.approx(expected_hours)
    assert report["methods"] == EXPECTED_VARIANTS
    assert report["automatic_parallelism_enabled"] is False
    assert report["protocol_audit_errors"] == []


def test_static_v2_protocol_audit_passes() -> None:
    report = audit_workload_aware_v2(ROOT)
    assert report["all_required_checks_passed"], report["failed_checks"]
    assert report["passed_check_count"] == report["required_check_count"]
