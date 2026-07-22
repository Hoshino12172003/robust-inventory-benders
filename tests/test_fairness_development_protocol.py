from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest
import yaml

from src.fairness_benders import (
    FAIRNESS_DEVELOPMENT_MANIFEST_SCHEMA_VERSION,
    EXECUTION_ATTEMPT,
    POST_EVALUATION_INVALID_ATTEMPT_SEEDS,
    PRIOR_ATTEMPTS,
    PREVIOUS_ATTEMPT_SEEDS,
    _attempt_identity,
    _certified_baseline_anchor,
    _development_run_keys,
    _record_failed_task,
    _prepare_attempt3_output,
    _validate_development_manifest_identity,
    _validate_frontier_anchor_identity,
    _validate_resume_record_identity,
    _write_fairness_result_tables,
    _write_record,
    development_run_plan,
    fairness_frontier_overall_status,
    run_fairness_development,
)
from src.experiment_protocol import config_sha256
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


def test_baseline_anchor_uses_certified_upper_bound_not_objective() -> None:
    record = {
        "solved_to_tolerance": True,
        "result": {
            "status": "optimal",
            "objective": 90.0,
            "lower_bound": 95.0,
            "upper_bound": 100.0,
            "gap": 1e-5,
            "valid_UB": True,
        },
    }
    anchor = _certified_baseline_anchor(
        record,
        baseline_run_key="baseline",
        config_hash="config",
        commit="commit",
        candidate_config_sha256="A" * 64,
        tolerance=1e-4,
    )
    assert anchor["source"] == "solve_result.upper_bound"
    assert anchor["value"] == 100.0
    assert anchor["value_hex"] == float(100.0).hex()
    assert anchor["value"] != record["result"]["objective"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result.update(valid_UB=False),
        lambda result: result.update(upper_bound=None),
        lambda result: result.update(status="time_limit"),
        lambda result: result.update(gap=0.1),
    ],
)
def test_uncertified_baseline_cannot_become_cost_anchor(mutation) -> None:
    result = {"status": "optimal", "upper_bound": 100.0, "gap": 1e-5, "valid_UB": True}
    mutation(result)
    with pytest.raises(RuntimeError, match="certified feasible robust upper bound"):
        _certified_baseline_anchor(
            {"solved_to_tolerance": True, "result": result},
            baseline_run_key="baseline",
            config_hash="config",
            commit="commit",
            candidate_config_sha256="A" * 64,
            tolerance=1e-4,
        )


def test_frontier_resume_locks_anchor_and_rho_identity() -> None:
    anchor = {"anchor_sha256": "anchor"}
    record = {"baseline_run_key": "base", "baseline_anchor_sha256": "anchor", "rho": 0.05}
    _validate_frontier_anchor_identity(record, anchor=anchor, baseline_run_key="base", rho=0.05)
    with pytest.raises(ValueError, match="C_anchor identity"):
        _validate_frontier_anchor_identity(
            {**record, "baseline_anchor_sha256": "other"},
            anchor=anchor,
            baseline_run_key="base",
            rho=0.05,
        )


def test_development_manifest_rejects_run_plan_or_commit_drift() -> None:
    config = load_configs()["regional_fairness_development_medium_large"]
    keys = ["a", "b"]
    manifest = {
        **_attempt_identity(
            config=config, config_hash="config", commit="commit", run_keys=keys
        ),
        "execution_restart_after_correctness_hotfix": True,
        "previous_attempt_scientifically_invalid": True,
        "execution_restart_after_post_evaluation_hotfix": True,
        "previous_attempt2_scientifically_invalid": True,
        "previous_attempt2_results_reused": False,
        "post_evaluation_invalid_attempt_seeds": POST_EVALUATION_INVALID_ATTEMPT_SEEDS,
        "baseline_anchor_source": "solve_result.upper_bound",
        "run_keys": keys,
    }
    _validate_development_manifest_identity(
        manifest,
        config=config,
        config_hash="config",
        commit="commit",
        run_keys=keys,
    )
    with pytest.raises(ValueError, match="git_commit"):
        _validate_development_manifest_identity(
            manifest,
            config=config,
            config_hash="config",
            commit="other",
            run_keys=keys,
        )
    with pytest.raises(ValueError, match="previous_attempt_results_reused"):
        _validate_development_manifest_identity(
            {**manifest, "previous_attempt_results_reused": True},
            config=config,
            config_hash="config",
            commit="commit",
            run_keys=keys,
        )
    with pytest.raises(ValueError, match="execution_restart_after_correctness_hotfix"):
        _validate_development_manifest_identity(
            {**manifest, "execution_restart_after_correctness_hotfix": False},
            config=config,
            config_hash="config",
            commit="commit",
            run_keys=keys,
        )
    with pytest.raises(ValueError, match="previous_attempt2_results_reused"):
        _validate_development_manifest_identity(
            {**manifest, "previous_attempt2_results_reused": True},
            config=config,
            config_hash="config",
            commit="commit",
            run_keys=keys,
        )


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_certificate"),
    [
        (RuntimeError("certificate failure"), "failed", "uncertified_exception"),
        (KeyboardInterrupt(), "interrupted", "uncertified_interrupted"),
    ],
)
def test_running_task_is_atomically_replaced_by_failure_evidence(
    tmp_path: Path,
    error: BaseException,
    expected_status: str,
    expected_certificate: str,
) -> None:
    _record_failed_task(
        tmp_path,
        run_key="synthetic-run",
        task_type="fairness_frontier",
        instance_size="tiny",
        seed=1,
        method="robust_regional_fairness",
        commit="commit",
        config_hash="config",
        error=error,
        rho=0.05,
        baseline_run_key="baseline",
        anchor={"anchor_sha256": "anchor"},
    )
    run = yaml.safe_load(
        (tmp_path / "runs" / "synthetic-run" / "run.json").read_text(encoding="utf-8")
    )
    status = yaml.safe_load(
        (tmp_path / "runs" / "synthetic-run" / "status.json").read_text(encoding="utf-8")
    )
    assert run["state"] == "failed"
    assert run["result"]["status"] == expected_status
    assert run["result"]["algorithm_status"] == expected_status
    expected_public = "interrupted" if expected_status == "interrupted" else "implementation_error"
    assert run["result"]["overall_status"] == expected_public
    assert run["certification_status"] == expected_certificate
    assert status["state"] == "failed"
    assert status["certification_status"] == expected_certificate


@pytest.mark.parametrize(
    ("algorithm_status", "algorithm_solved", "attempted", "valid", "expected"),
    [
        ("optimal", True, True, True, "certified_robust_optimal"),
        ("optimal", True, True, False, "invalid_post_evaluation"),
        ("optimal", False, False, False, "master_optimal_but_robust_uncertified"),
        ("time_limit", False, False, False, "time_limit_uncertified"),
        ("iteration_limit", False, False, False, "iteration_limit_uncertified"),
        ("numeric", False, False, False, "numerical_uncertified"),
        ("infeasible", False, False, False, "infeasible_uncertified"),
        ("interrupted", False, False, False, "interrupted"),
        ("unexpected", False, False, False, "unknown_uncertified"),
    ],
)
def test_frontier_status_never_hides_uncertified_robust_result(
    algorithm_status: str,
    algorithm_solved: bool,
    attempted: bool,
    valid: bool,
    expected: str,
) -> None:
    assert fairness_frontier_overall_status(
        algorithm_status=algorithm_status,
        algorithm_solved=algorithm_solved,
        post_evaluation_attempted=attempted,
        post_evaluation_valid=valid,
    ) == expected


def _prepare_identity(
    output: Path,
    config: dict,
    *,
    resume: bool = False,
    overwrite: bool = False,
    commit: str = "attempt3-commit",
) -> list[str]:
    keys = _development_run_keys(config)
    _prepare_attempt3_output(
        output,
        resume=resume,
        overwrite=overwrite,
        config=config,
        config_hash=config_sha256(config),
        commit=commit,
        run_keys=keys,
    )
    return keys


def test_attempt3_fresh_output_identity_and_resume_are_strict(tmp_path: Path) -> None:
    config = load_configs()["regional_fairness_development_medium_large"]
    output = tmp_path / "rf3"
    keys = _prepare_identity(output, config)
    manifest = json.loads((output / "fairness_development_manifest.json").read_text())
    assert len(keys) == 60
    assert manifest["execution_attempt"] == EXECUTION_ATTEMPT == 3
    assert manifest["development_seeds_previously_accessed"] == list(range(120, 130))
    assert manifest["prior_attempts"] == PRIOR_ATTEMPTS
    assert manifest["previous_attempt_results_reused"] is False
    assert not (output / "runs").exists()
    _prepare_identity(output, config, resume=True)


def test_attempt3_rejects_existing_empty_or_prior_attempt_directory(tmp_path: Path) -> None:
    config = load_configs()["regional_fairness_development_medium_large"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="does not exist"):
        _prepare_identity(empty, config)
    for name in ("attempt1", "attempt2"):
        prior = tmp_path / name
        prior.mkdir()
        (prior / "fairness_development_manifest.json").write_text(
            json.dumps({"schema_version": 3, "execution_attempt": int(name[-1])})
        )
        with pytest.raises(ValueError, match="identity mismatch"):
            _prepare_identity(prior, config, resume=True)


def test_attempt3_resume_rejects_identity_drift_and_external_run(tmp_path: Path) -> None:
    config = load_configs()["regional_fairness_development_medium_large"]
    output = tmp_path / "rf3"
    keys = _prepare_identity(output, config)
    with pytest.raises(ValueError, match="identity mismatch"):
        _prepare_identity(output, config, resume=True, commit="different")

    manifest_path = output / "fairness_development_manifest.json"
    original = json.loads(manifest_path.read_text())
    for field in ("config_sha256", "protocol_sha256", "candidate_config_sha256"):
        mutated = deepcopy(original)
        mutated[field] = "0" * 64
        manifest_path.write_text(json.dumps(mutated))
        with pytest.raises(ValueError, match="identity mismatch"):
            _prepare_identity(output, config, resume=True)
    manifest_path.write_text(json.dumps(original))

    (output / "resolved_config.yaml").write_text("foreign: true\n")
    with pytest.raises(ValueError, match="resolved config identity"):
        _prepare_identity(output, config, resume=True)
    (output / "resolved_config.yaml").unlink()

    injected = output / "runs" / keys[0]
    injected.mkdir(parents=True)
    (injected / "run.json").write_text(
        json.dumps({"run_key": keys[0], "config_sha256": "foreign", "git_commit": "foreign"})
    )
    with pytest.raises(ValueError, match="Config identity"):
        _prepare_identity(output, config, resume=True)


def test_attempt3_rejects_missing_identity_and_overwrite(tmp_path: Path) -> None:
    config = load_configs()["regional_fairness_development_medium_large"]
    missing = tmp_path / "missing-identity"
    missing.mkdir()
    with pytest.raises(ValueError, match="complete identity manifest"):
        _prepare_identity(missing, config, resume=True)
    with pytest.raises(ValueError, match="overwrite is prohibited"):
        _prepare_identity(tmp_path / "new", config, overwrite=True)
    with pytest.raises(ValueError, match="overwrite is prohibited"):
        run_fairness_development(config, overwrite=True)


def test_certified_infeasible_is_complete_but_not_optimal_solved() -> None:
    assert fairness_frontier_overall_status(
        algorithm_status="infeasible",
        algorithm_solved=False,
        post_evaluation_attempted=False,
        post_evaluation_valid=False,
        infeasibility_certified=True,
    ) == "certified_infeasible"


def test_summary_uses_public_status_enum_consistently(tmp_path: Path) -> None:
    records = [
        ("certified", "certified_robust_optimal", True),
        ("infeasible", "infeasible_uncertified", False),
        ("iteration", "iteration_limit_uncertified", False),
    ]
    for key, public_status, solved in records:
        _write_record(
            tmp_path,
            key,
            {
                **{
                    "execution_attempt": 3,
                    "previous_attempt_results_reused": False,
                    "prior_attempts": PRIOR_ATTEMPTS,
                },
                "run_key": key,
                "task_type": "fairness_frontier",
                "instance_size": "tiny",
                "seed": 1,
                "method": "robust_regional_fairness",
                "rho": 0.0,
                "overall_status": public_status,
                "algorithm_status": public_status,
                "certification_status": public_status,
                "solved_to_tolerance": solved,
                "result": {"status": public_status},
            },
        )
    _write_fairness_result_tables(tmp_path, [item[0] for item in records])
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as stream:
        summary = next(csv.DictReader(stream))
    assert summary["solved_count"] == "1"
    assert summary["optimal_count"] == "1"
    assert summary["uncertified_count"] == "2"
    assert summary["infeasible_count"] == "1"
    assert summary["execution_attempt"] == "3"
    assert summary["previous_attempt_results_reused"] == "False"
