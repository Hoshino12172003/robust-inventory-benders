from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.fairness_benders import _validate_resume_record_identity, development_run_plan
from src.fairness_development_audit import CONFIG_PATHS, audit_fairness_development


def load_configs() -> dict[str, dict]:
    return {
        name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, path in CONFIG_PATHS.items()
    }


def failed(report: dict) -> set[str]:
    return {check["check"] for check in report["checks"] if not check["passed"]}


def test_frozen_development_protocol_passes_static_audit() -> None:
    report = audit_fairness_development()
    assert report["passed"], failed(report)
    assert report["check_count"] >= 60


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda c: c.update(random_seeds=list(range(121, 131))), "medium_large_seeds_exact_120_129"),
        (lambda c: c.update(variants=["joint_v1_core_point_strengthened", "other"]), "medium_large_only_frozen_candidate"),
        (lambda c: c.update(master_error_budget_ratio=0.30), "medium_large_candidate_parameters_frozen"),
        (lambda c: c.update(candidate_config_sha256="0" * 64), "medium_large_candidate_hash"),
        (lambda c: c["fairness_development"].update(rho_grid=[0.0, 0.05]), "medium_large_rho_grid"),
        (
            lambda c: c["fairness_development"].update(same_recourse_for_cost_and_fairness=False),
            "medium_large_same_recourse",
        ),
    ],
)
def test_protocol_drift_is_rejected(mutation, expected: str) -> None:
    configs = load_configs()
    target = configs["regional_fairness_development_medium_large"]
    mutation(target)
    report = audit_fairness_development(config_overrides=configs)
    assert expected in failed(report)


def test_plans_freeze_ten_baselines_and_fifty_frontier_runs_per_scale() -> None:
    for config in load_configs().values():
        plan = development_run_plan(config)
        assert plan["baseline_run_count"] == 10
        assert plan["fairness_frontier_run_count"] == 50
        assert plan["total_computational_run_count"] == 60
        assert plan["instances_generated"] is False
        assert plan["solver_called"] is False


def test_dry_run_plan_reports_exact_scenario_counts_without_instances(tmp_path: Path) -> None:
    configs = load_configs()
    medium = development_run_plan(configs["regional_fairness_development_medium_large"])
    large = development_run_plan(configs["regional_fairness_development_large"])
    assert medium["scenario_count_by_size"] == {"medium_large": 1831}
    assert large["scenario_count_by_size"] == {"large": 4657}
    assert not list(tmp_path.iterdir())


def test_future_seeds_are_reserved_but_not_in_run_plan() -> None:
    for config in load_configs().values():
        assert development_run_plan(config)["seeds"] == list(range(120, 130))
        assert not set(development_run_plan(config)["seeds"]).intersection(range(130, 160))


def test_scale_configs_differ_only_in_pre_registered_operational_fields() -> None:
    configs = load_configs()
    medium = deepcopy(configs["regional_fairness_development_medium_large"])
    large = deepcopy(configs["regional_fairness_development_large"])
    for field in (
        "experiment_name",
        "output_dir",
        "instance_sizes",
        "baseline_time_limit",
        "fairness_time_limit",
        "time_limit",
        "max_iterations",
    ):
        medium.pop(field)
        large.pop(field)
    assert medium == large


def test_no_validation_or_final_fairness_model_configs_exist() -> None:
    root = next(iter(CONFIG_PATHS.values())).parent
    assert not list(root.glob("regional_fairness_*validation*.yaml"))
    assert not list(root.glob("regional_fairness_*final*.yaml"))


def test_resume_rejects_config_or_commit_identity_drift() -> None:
    record = {"run_key": "key", "config_sha256": "config", "git_commit": "commit"}
    _validate_resume_record_identity(record, config_hash="config", commit="commit", run_key="key")
    with pytest.raises(ValueError, match="Config identity"):
        _validate_resume_record_identity(record, config_hash="other", commit="commit", run_key="key")
    with pytest.raises(ValueError, match="Git-commit identity"):
        _validate_resume_record_identity(record, config_hash="config", commit="other", run_key="key")
