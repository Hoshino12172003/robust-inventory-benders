from __future__ import annotations

import json
from pathlib import Path

import src.experiment_suite as suite
from src.experiment_protocol import (
    ProtocolRunSpec,
    atomic_write_json,
    build_run_manifest,
    decide_run_action,
    penalized_runtime_par2,
    stable_run_key,
)
from src.results import SolveResult


def test_par2_uses_runtime_for_solved_and_double_limit_otherwise() -> None:
    assert penalized_runtime_par2(
        solved_to_tolerance=True, runtime=12.5, time_limit=100
    ) == 12.5
    assert penalized_runtime_par2(
        solved_to_tolerance=False, runtime=12.5, time_limit=100
    ) == 200.0


def test_run_key_is_stable_and_includes_sensitivity_coordinates() -> None:
    kwargs = {
        "experiment_name": "managerial_sensitivity_joint_v1",
        "sensitivity_axis": "budget_factor",
        "sensitivity_value": 0.68,
        "instance_size": "medium_large",
        "seed": 30,
        "variant_name": "proposed_joint_rho025_050",
    }
    first = stable_run_key(**kwargs)
    second = stable_run_key(**kwargs)
    assert first == second
    assert "budget_factor" in first
    assert "seed_30" in first


def test_resume_and_overwrite_decisions() -> None:
    success = {
        "state": "complete",
        "success": True,
        "result": {"status": "time_limit"},
    }
    failure = {
        "state": "complete",
        "success": False,
        "result": {"status": "failed"},
    }
    incomplete = {"state": "running", "success": False}

    assert decide_run_action(success, resume=False, overwrite=False) == "skip_success"
    assert decide_run_action(success, resume=True, overwrite=False) == "skip_success"
    assert decide_run_action(success, resume=False, overwrite=True) == "run_overwrite"
    assert decide_run_action(failure, resume=False, overwrite=False) == "skip_incomplete"
    assert decide_run_action(failure, resume=True, overwrite=False) == "run_resume"
    assert decide_run_action(incomplete, resume=True, overwrite=False) == "run_resume"
    assert decide_run_action(None, resume=False, overwrite=False) == "run"


def test_manifest_counts_complete_solved_failed_and_remaining(tmp_path: Path) -> None:
    output = tmp_path / "experiment"
    keys = ["run_a", "run_b", "run_c"]
    atomic_write_json(
        output / "runs/run_a/run.json",
        {
            "state": "complete",
            "success": True,
            "solved_to_tolerance": True,
            "result": {"status": "optimal"},
        },
    )
    atomic_write_json(
        output / "runs/run_b/run.json",
        {
            "state": "complete",
            "success": False,
            "solved_to_tolerance": False,
            "result": {"status": "failed"},
        },
    )

    manifest = build_run_manifest(
        output_dir=output,
        run_keys=keys,
        config_hash="abc",
        commit="def",
        skipped_run_count=1,
    )
    assert manifest["expected_run_count"] == 3
    assert manifest["completed_run_count"] == 2
    assert manifest["solved_run_count"] == 1
    assert manifest["failed_run_count"] == 1
    assert manifest["skipped_run_count"] == 1
    assert manifest["remaining_run_count"] == 1


def test_suite_resume_skips_success_reruns_failure_and_overwrite_forces_run(
    tmp_path: Path, monkeypatch
) -> None:
    calls = {"count": 0}

    def fake_solve(config, instance, method, variant):
        calls["count"] += 1
        return (
            SolveResult(
                method=method,
                status="optimal",
                objective=100.0,
                lower_bound=99.999,
                upper_bound=100.0,
                gap=1.0e-5,
                runtime=2.0,
                iterations=3,
                master_runtime=0.8,
                subproblem_runtime=1.1,
                gamma_target=2,
                metadata={
                    "subproblem_mode": "robust_dual_milp",
                    "valid_UB": True,
                    "best_y_values": [1.0, 0.0],
                    "best_x_values": [[1.0, 2.0], [0.0, 0.0]],
                },
            ),
            {
                "adaptive_gap_enabled": False,
                "gamma_continuation_enabled": False,
                "cut_selection_enabled": False,
            },
        )

    monkeypatch.setattr(suite, "_solve_experiment_method", fake_solve)
    config = {
        "experiment_name": "resume_test",
        "output_dir": str(tmp_path / "out"),
        "random_seeds": [7],
        "instance_sizes": ["very_small"],
        "methods": ["standard_benders"],
        "time_limit": 10,
        "max_iterations": 10,
        "tol": 1e-4,
        "gamma_target": 2,
        "gamma_schedule": [2],
        "gamma_continuation_enabled": False,
        "subproblem_mode": "robust_dual_milp",
        "cut_selection_enabled": False,
        "save_iteration_log": True,
    }

    first = suite.run_experiment_suite(config)
    assert calls["count"] == 1
    suite.run_experiment_suite(config)
    assert calls["count"] == 1

    spec = suite.experiment_run_specs(config)[0]
    record_path = Path(first["output_dir"]) / "runs" / spec.run_key / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["success"] = False
    record["result"]["status"] = "failed"
    atomic_write_json(record_path, record)

    first_stage = record["result"]
    assert first_stage["best_y_values"] == [1.0, 0.0]
    assert first_stage["best_x_values"] == [[1.0, 2.0], [0.0, 0.0]]
    json.dumps(first_stage["best_y_values"])
    json.dumps(first_stage["best_x_values"])

    suite.run_experiment_suite(config, resume=True)
    assert calls["count"] == 2
    suite.run_experiment_suite(config, overwrite=True)
    assert calls["count"] == 3


def test_atomic_json_write_replaces_complete_file_without_leaving_temp(
    tmp_path: Path,
) -> None:
    target = tmp_path / "run.json"
    atomic_write_json(target, {"version": 1, "payload": "old"})
    atomic_write_json(target, {"version": 2, "payload": "new"})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "payload": "new",
        "version": 2,
    }
    assert not target.with_name(f".{target.name}.tmp").exists()
