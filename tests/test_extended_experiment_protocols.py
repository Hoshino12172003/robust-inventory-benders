from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.extended_experiment_audit import audit_protocols
from src.experiment_suite import (
    SELECTED_PARAMETER_FIELDS,
    _apply_selected_parameters,
    experiment_dry_run_report,
    experiment_run_specs,
)
from src.managerial_sensitivity_suite import (
    managerial_dry_run_report,
    managerial_run_config,
    managerial_run_specs,
)


CONFIG_DIR = Path("experiments/configs")


def test_large_scale_protocol_expands_to_exactly_50_runs() -> None:
    config = _apply_selected_parameters(
        load_config(CONFIG_DIR / "large_scale_evaluation_joint_v1.yaml")
    )
    specs = experiment_run_specs(config)

    assert len(specs) == 50
    assert {spec.seed for spec in specs} == set(range(20, 30))
    assert {spec.instance_size for spec in specs} == {"large"}
    assert {spec.variant_name for spec in specs} == {
        "standard_benders",
        "static_inexact_benders",
        "mp_adaptive_rho050",
        "sp_adaptive_rho050",
        "proposed_joint_rho025_050",
    }


def test_managerial_protocol_expands_to_exactly_190_runs_by_axis() -> None:
    config = _apply_selected_parameters(
        load_config(CONFIG_DIR / "managerial_sensitivity_joint_v1.yaml")
    )
    specs = managerial_run_specs(config)
    counts: dict[str, int] = {}
    for spec in specs:
        counts[str(spec.sensitivity_axis)] = counts.get(str(spec.sensitivity_axis), 0) + 1

    assert len(specs) == 190
    assert counts == {
        "gamma_target": 50,
        "service_level": 40,
        "budget_factor": 50,
        "capacity_factor": 50,
    }
    assert {spec.seed for spec in specs} == set(range(30, 40))
    assert {spec.variant_name for spec in specs} == {"proposed_joint_rho025_050"}


def test_all_protocol_seed_sets_are_disjoint() -> None:
    seed_sets = [set([0, 1, 2]), set(range(10, 20)), set(range(20, 30)), set(range(30, 40))]
    assert all(
        seed_sets[left].isdisjoint(seed_sets[right])
        for left in range(len(seed_sets))
        for right in range(left + 1, len(seed_sets))
    )


def test_managerial_baselines_and_target_only_gamma_schedule() -> None:
    config = _apply_selected_parameters(
        load_config(CONFIG_DIR / "managerial_sensitivity_joint_v1.yaml")
    )
    specs = managerial_run_specs(config)
    gamma_spec = next(
        spec
        for spec in specs
        if spec.sensitivity_axis == "gamma_target" and spec.sensitivity_value == 4
    )
    service_spec = next(
        spec
        for spec in specs
        if spec.sensitivity_axis == "service_level" and spec.sensitivity_value == 0.82
    )
    budget_spec = next(
        spec
        for spec in specs
        if spec.sensitivity_axis == "budget_factor" and spec.sensitivity_value == 0.75
    )
    capacity_spec = next(
        spec
        for spec in specs
        if spec.sensitivity_axis == "capacity_factor" and spec.sensitivity_value == 1.35
    )

    gamma_run = managerial_run_config(config, gamma_spec)
    service_run = managerial_run_config(config, service_spec)
    budget_run = managerial_run_config(config, budget_spec)
    capacity_run = managerial_run_config(config, capacity_spec)

    assert gamma_spec.baseline_value == 2
    assert gamma_run["robust"]["gamma_target"] == 4
    assert gamma_run["robust"]["gamma_schedule"] == [4]
    assert service_spec.baseline_value == 0.90
    assert service_run["instance"]["service_level"] == 0.82
    assert budget_spec.baseline_value == 0.68
    assert budget_run["instance"]["budget_factor"] == 0.75
    assert capacity_spec.baseline_value == 1.25
    assert capacity_run["instance"]["capacity_factor"] == 1.35


def test_frozen_proposed_parameters_are_inherited_without_drift() -> None:
    selected = load_config(CONFIG_DIR / "selected_algorithm_parameters.yaml")
    for name in (
        "large_scale_evaluation_joint_v1.yaml",
        "managerial_sensitivity_joint_v1.yaml",
    ):
        resolved = _apply_selected_parameters(load_config(CONFIG_DIR / name))
        for field in SELECTED_PARAMETER_FIELDS:
            assert resolved[field] == selected[field]


def test_dry_run_reports_required_counts_and_serial_upper_bounds() -> None:
    large = experiment_dry_run_report(
        load_config(CONFIG_DIR / "large_scale_evaluation_joint_v1.yaml")
    )
    managerial = managerial_dry_run_report(
        load_config(CONFIG_DIR / "managerial_sensitivity_joint_v1.yaml")
    )

    assert large["total_run_count"] == 50
    assert large["theoretical_maximum_hours"] == 25.0
    assert managerial["total_run_count"] == 190
    assert managerial["theoretical_maximum_hours"] == 47.5
    assert not large["protocol_audit_errors"]
    assert not managerial["protocol_audit_errors"]


def test_static_protocol_audit_passes_and_frozen_files_are_unchanged() -> None:
    report = audit_protocols()
    assert report["all_required_checks_passed"] is True
    assert not report["failed_checks"]
    by_name = {check["check"]: check for check in report["checks"]}
    assert by_name["selected_algorithm_parameters_unchanged"]["passed"] is True
    assert by_name["final_evaluation_config_unchanged"]["passed"] is True

