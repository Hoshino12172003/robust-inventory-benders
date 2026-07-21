from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml

from src.experiment_protocol import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_yaml,
    config_sha256,
    file_sha256,
    git_commit,
)
from src.experiment_suite import _apply_selected_parameters, experiment_run_specs
from src.instance import InventoryInstance
from src.regional_fairness_diagnostic import RecourseAllocation, first_stage_x_sha256
from src.regional_fairness_pipeline import (
    DiagnosticCheckpointError,
    DiagnosticIdentityError,
    DiagnosticLockError,
    PipelineDependencies,
    run_regional_fairness_pipeline,
)
from src.scenarios import enumerate_budget_scenarios_with_metadata


ROOT = Path(__file__).resolve().parents[1]


def _tiny_instance(seed: int) -> InventoryInstance:
    return InventoryInstance(
        name=f"synthetic_fairness_seed{seed}",
        num_warehouses=2,
        num_products=1,
        num_regions=2,
        fixed_cost=[0.0, 0.0],
        inventory_cost=[[0.0], [0.0]],
        capacity=[10.0, 10.0],
        volume=[1.0],
        budget=100.0,
        transport_cost=[[[1.0], [1.0]], [[1.0], [1.0]]],
        shortage_penalty=[[10.0], [10.0]],
        service_penalty=[20.0],
        service_level=[0.0],
        base_demand=[[8.0], [8.0]],
        demand_deviation=[[1.0], [1.0]],
        inventory_ub=[[10.0], [10.0]],
    )


def _config(root: Path) -> Path:
    config_dir = root / "experiments/configs"
    config_dir.mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    (root / "docs/regional_fairness_diagnostic_protocol.md").write_text(
        "synthetic protocol", encoding="utf-8"
    )
    (config_dir / "selected_cut_strengthened_joint_v3_candidate.yaml").write_text(
        "selected_variant: joint_v1_core_point_strengthened\n", encoding="utf-8"
    )
    value: dict[str, Any] = {
        "experiment_name": "synthetic_regional_fairness_diagnostic",
        "output_dir": "outputs/fairness",
        "random_seeds": list(range(1, 11)),
        "instance_sizes": ["medium_large"],
        "variants": ["joint_v1_core_point_strengthened"],
        "variant_settings": {
            "joint_v1_core_point_strengthened": {
                "cut_strengthening_policy": "core_point",
                "max_cuts_per_iteration": 1,
            }
        },
        "protocol_phase": "fairness_diagnostic",
        "fairness_diagnostic": {
            "gamma": 2,
            "max_scenarios": 5000,
            "exact_scenarios": True,
            "cost_absolute_tolerance": 1.0e-6,
            "cost_relative_tolerance": 1.0e-6,
            "metric_tolerance": 1.0e-9,
            "recourse_time_limit": 1.0,
            "checkpoint_scenario_chunk_size": 2,
        },
    }
    path = config_dir / "synthetic.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


class SyntheticBaseRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, bool]] = []

    def __call__(self, config: dict[str, Any], *, resume: bool, overwrite: bool) -> dict[str, Path]:
        self.calls.append({"resume": resume, "overwrite": overwrite})
        output = Path(config["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        resolved = _apply_selected_parameters(config)
        atomic_write_yaml(output / "resolved_config.yaml", resolved)
        commit = git_commit(Path.cwd())
        rows: list[dict[str, Any]] = []
        specs = experiment_run_specs(resolved)
        for spec in specs:
            instance = _tiny_instance(spec.seed)
            instance_path = output / "instances" / f"{instance.name}.json"
            atomic_write_json(instance_path, instance.to_dict())
            result = {
                "run_key": spec.run_key,
                "variant_name": spec.variant_name,
                "method": spec.variant_name,
                "seed": spec.seed,
                "instance_size": spec.instance_size,
                "instance_name": instance.name,
                "git_commit": commit,
                "config_sha256": f"base-config-{spec.seed}",
                "instance_path": str(instance_path.resolve()),
                "best_x_values": [[5.0], [5.0]],
                "status": "optimal",
                "solved_to_tolerance": True,
            }
            record = {
                "run_key": spec.run_key,
                "state": "complete",
                "success": True,
                "solved_to_tolerance": True,
                "git_commit": commit,
                "result": result,
            }
            atomic_write_json(output / "runs" / spec.run_key / "run.json", record)
            rows.append(result)
        atomic_write_csv(
            output / "results.csv",
            rows,
            [
                "run_key",
                "variant_name",
                "method",
                "seed",
                "instance_size",
                "instance_name",
                "git_commit",
                "config_sha256",
                "instance_path",
                "status",
                "solved_to_tolerance",
            ],
        )
        atomic_write_csv(output / "summary.csv", [{"run_count": 10}], ["run_count"])
        atomic_write_json(
            output / "run_manifest.json",
            {
                "expected_run_count": 10,
                "completed_run_count": 10,
                "solved_run_count": 10,
                "failed_run_count": 0,
                "remaining_run_count": 0,
                "skipped_run_count": 0,
                "config_sha256": config_sha256(resolved),
                "git_commit": commit,
            },
        )
        return {"output_dir": output}


class InterruptingBaseRunner(SyntheticBaseRunner):
    def __init__(self) -> None:
        super().__init__()
        self.interrupted = False

    def __call__(self, config: dict[str, Any], *, resume: bool, overwrite: bool) -> dict[str, Path]:
        if not self.interrupted:
            self.interrupted = True
            self.calls.append({"resume": resume, "overwrite": overwrite})
            output = Path(config["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            resolved = _apply_selected_parameters(config)
            atomic_write_json(
                output / "run_manifest.json",
                {
                    "expected_run_count": 10,
                    "completed_run_count": 0,
                    "solved_run_count": 0,
                    "failed_run_count": 0,
                    "remaining_run_count": 10,
                    "config_sha256": config_sha256(resolved),
                    "git_commit": git_commit(Path.cwd()),
                },
            )
            raise KeyboardInterrupt
        return super().__call__(config, resume=resume, overwrite=overwrite)


def _scenario_solver(
    instance: InventoryInstance,
    scenario: Any,
    best_x_values: list[list[float]],
    **_kwargs: Any,
) -> tuple[RecourseAllocation, RecourseAllocation]:
    x_hash = first_stage_x_sha256(best_x_values)
    tolerance = 71.0e-6
    common = {
        "scenario_id": scenario.name,
        "status": "optimal",
        "objective": 70.0,
        "original_optimal_cost": 70.0,
        "cost_tolerance": tolerance,
        "shipment_values": [],
        "service_violation_values": [0.0],
        "runtime": 0.01,
        "first_stage_x_sha256": x_hash,
        "gamma_usage": scenario.gamma,
        "original_cost_reproduced": True,
        "cost_cap_satisfied": True,
        "constraints_satisfied": True,
    }
    default = RecourseAllocation(
        policy="default",
        shortage_values=[[6.0], [0.0]],
        transport_cost_by_region=[2.0, 8.0],
        allocated_units_by_region=[2.0, 8.0],
        **common,
    )
    fair = RecourseAllocation(
        policy="fair_best",
        shortage_values=[[3.0], [3.0]],
        transport_cost_by_region=[5.0, 5.0],
        allocated_units_by_region=[5.0, 5.0],
        **common,
    )
    return default, fair


def _dependencies(
    runner: SyntheticBaseRunner,
    fault: Any = None,
    solver: Any = _scenario_solver,
) -> PipelineDependencies:
    return PipelineDependencies(
        base_runner=runner,
        scenario_enumerator=enumerate_budget_scenarios_with_metadata,
        scenario_solver=solver,
        fault_injector=fault,
    )


def _run(root: Path, runner: SyntheticBaseRunner, *, fault: Any = None, solver: Any = _scenario_solver) -> dict[str, Any]:
    return run_regional_fairness_pipeline(
        _config(root) if not (root / "experiments/configs/synthetic.yaml").exists() else root / "experiments/configs/synthetic.yaml",
        resume=True,
        dependencies=_dependencies(runner, fault, solver),
        strict_protocol_audit=False,
    )


def test_new_run_recovers_base_and_writes_complete_schema(tmp_path: Path) -> None:
    runner = SyntheticBaseRunner()
    report = _run(tmp_path, runner)
    output = tmp_path / "outputs/fairness"
    assert report["status"] == "completed"
    assert runner.calls == [{"resume": True, "overwrite": False}]
    manifest = json.loads((output / "diagnostic_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["expected_chunk_count"] == 20
    assert manifest["completed_chunk_count"] == 20
    assert manifest["pending_chunk_count"] == 0
    assert manifest["base_results_files"]["results.csv"] == file_sha256(output / "results.csv")
    with (output / "region_scenario_metrics.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 10 * 4 * 2 * 2
    required = {
        "diagnostic_run_key",
        "base_run_key",
        "instance_name",
        "experiment_name",
        "scale",
        "method",
        "seed",
        "base_git_commit",
        "base_config_sha256",
        "resolved_config_sha256",
        "scenario_key",
        "scenario_index",
        "scenario_type",
        "deviation_pattern",
        "deviation_pattern_sha256",
        "region_id",
        "recourse_variant",
        "default_recourse_status",
        "fair_best_recourse_status",
        "default_recourse_cost",
        "fair_best_recourse_cost",
        "cost_absolute_tolerance",
        "cost_relative_tolerance",
        "regional_demand",
        "regional_shortage",
        "fill_rate",
        "invalid_reason",
    }
    assert required <= set(rows[0])
    keys = [
        (row["diagnostic_run_key"], row["base_run_key"], row["scenario_key"], row["recourse_variant"], row["region_id"])
        for row in rows
    ]
    assert len(keys) == len(set(keys))
    pattern = json.loads(next(row["deviation_pattern"] for row in rows if row["deviation_pattern"] != "[]"))
    assert {"region_id", "product_id", "deviation_value", "base_demand", "realized_demand"} <= set(pattern[0])


@pytest.mark.parametrize(
    "event",
    [
        "during_chunk",
        "after_checkpoint_commit_before_index",
        "after_all_chunks_before_aggregation",
        "before_final_aggregation_commit",
        "after_region_csv_before_diagnosis",
    ],
)
def test_fault_injection_resume_matches_clean_run(tmp_path: Path, event: str) -> None:
    clean_root = tmp_path / "clean"
    interrupted_root = tmp_path / "interrupted"
    clean_runner = SyntheticBaseRunner()
    _run(clean_root, clean_runner)
    clean_output = clean_root / "outputs/fairness"

    triggered = False

    def fault(name: str, _context: Any) -> None:
        nonlocal triggered
        if name == event and not triggered:
            triggered = True
            raise RuntimeError(f"injected {event}")

    runner = SyntheticBaseRunner()
    with pytest.raises(RuntimeError, match="injected"):
        _run(interrupted_root, runner, fault=fault)
    manifest = json.loads(
        (interrupted_root / "outputs/fairness/diagnostic_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    _run(interrupted_root, runner)
    resumed_output = interrupted_root / "outputs/fairness"
    assert runner.calls == [{"resume": True, "overwrite": False}]
    for filename in ("region_scenario_metrics.csv", "instance_summary.csv"):
        with (clean_output / filename).open(newline="", encoding="utf-8") as source:
            clean_rows = list(csv.DictReader(source))
        with (resumed_output / filename).open(newline="", encoding="utf-8") as source:
            resumed_rows = list(csv.DictReader(source))
        for row in clean_rows + resumed_rows:
            row.pop("diagnostic_run_key", None)
        assert clean_rows == resumed_rows
    clean_diagnosis = json.loads((clean_output / "diagnosis.json").read_text(encoding="utf-8"))
    resumed_diagnosis = json.loads((resumed_output / "diagnosis.json").read_text(encoding="utf-8"))
    clean_diagnosis.pop("diagnostic_run_key")
    resumed_diagnosis.pop("diagnostic_run_key")
    assert clean_diagnosis == resumed_diagnosis


def test_keyboard_interrupt_is_atomic_and_resumable(tmp_path: Path) -> None:
    triggered = False

    def interrupt(name: str, _context: Any) -> None:
        nonlocal triggered
        if name == "during_chunk" and not triggered:
            triggered = True
            raise KeyboardInterrupt

    runner = SyntheticBaseRunner()
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, runner, fault=interrupt)
    manifest_path = tmp_path / "outputs/fairness/diagnostic_run_manifest.json"
    interrupted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert interrupted_manifest["status"] == "interrupted"
    assert interrupted_manifest["interrupted_count"] == 1
    assert interrupted_manifest["interrupted_chunk_count"] == 1
    _run(tmp_path, runner)
    completed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert completed_manifest["status"] == "completed"
    assert completed_manifest["interrupted_count"] == 1


def test_base_stage_keyboard_interrupt_resumes_through_existing_runner(tmp_path: Path) -> None:
    runner = InterruptingBaseRunner()
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, runner)
    manifest_path = tmp_path / "outputs/fairness/diagnostic_run_manifest.json"
    interrupted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert interrupted_manifest["status"] == "interrupted"
    assert interrupted_manifest["interrupted_count"] == 1
    _run(tmp_path, runner)
    completed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert completed_manifest["status"] == "completed"
    assert completed_manifest["interrupted_count"] == 1
    assert runner.calls == [
        {"resume": True, "overwrite": False},
        {"resume": True, "overwrite": False},
    ]


def test_completed_resume_is_idempotent_and_does_not_rerun_base(tmp_path: Path) -> None:
    runner = SyntheticBaseRunner()
    _run(tmp_path, runner)
    output = tmp_path / "outputs/fairness"
    before = (output / "region_scenario_metrics.csv").read_bytes()
    _run(tmp_path, runner)
    after = (output / "region_scenario_metrics.csv").read_bytes()
    assert before == after
    assert len(runner.calls) == 1


def test_resume_refuses_config_identity_drift(tmp_path: Path) -> None:
    runner = SyntheticBaseRunner()
    _run(tmp_path, runner)
    path = tmp_path / "experiments/configs/synthetic.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["fairness_diagnostic"]["recourse_time_limit"] = 2.0
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(DiagnosticIdentityError, match="identity mismatch"):
        run_regional_fairness_pipeline(
            path,
            resume=True,
            dependencies=_dependencies(runner),
            strict_protocol_audit=False,
        )


def test_resume_refuses_changed_base_results_hash(tmp_path: Path) -> None:
    runner = SyntheticBaseRunner()
    _run(tmp_path, runner)
    output = tmp_path / "outputs/fairness"
    (output / "results.csv").write_text("foreign\n", encoding="utf-8")
    with pytest.raises(DiagnosticIdentityError, match="hashes changed"):
        _run(tmp_path, runner)


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_resume_rejects_corrupt_or_missing_committed_checkpoint(tmp_path: Path, damage: str) -> None:
    runner = SyntheticBaseRunner()
    _run(tmp_path, runner)
    output = tmp_path / "outputs/fairness"
    index = json.loads((output / "checkpoint/index.json").read_text(encoding="utf-8"))
    checkpoint = output / index["entries"][0]["relative_path"]
    if damage == "corrupt":
        checkpoint.write_text("{broken", encoding="utf-8")
    else:
        checkpoint.unlink()
    with pytest.raises(DiagnosticCheckpointError):
        _run(tmp_path, runner)


def test_nonoptimal_recourse_marks_diagnosis_invalid(tmp_path: Path) -> None:
    def failed_solver(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("recourse status time_limit")

    runner = SyntheticBaseRunner()
    with pytest.raises(RuntimeError, match="invalid"):
        _run(tmp_path, runner, solver=failed_solver)
    output = tmp_path / "outputs/fairness"
    diagnosis = json.loads((output / "diagnosis.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "diagnostic_run_manifest.json").read_text(encoding="utf-8"))
    assert diagnosis["decision"] == "fairness_diagnostic_invalid"
    assert manifest["status"] == "failed"
    assert manifest["failed_chunk_count"] == 20


def test_single_writer_lock_refuses_concurrent_writer(tmp_path: Path) -> None:
    path = _config(tmp_path)
    output = tmp_path / "outputs/fairness"
    output.mkdir(parents=True)
    atomic_write_json(
        output / ".regional_fairness_diagnostic.lock",
        {"pid": __import__("os").getpid(), "host": "test"},
    )
    with pytest.raises(DiagnosticLockError):
        run_regional_fairness_pipeline(
            path,
            resume=True,
            dependencies=_dependencies(SyntheticBaseRunner()),
            strict_protocol_audit=False,
        )


def test_cli_dry_run_does_not_create_formal_outputs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src import regional_fairness_diagnostic as cli

    output = ROOT / "experiments/results_fairness_diagnostic/medium_large"
    assert not output.exists()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "regional_fairness_diagnostic",
            "--config",
            str(ROOT / "experiments/configs/regional_fairness_diagnostic_medium_large.yaml"),
            "--dry-run",
        ],
    )
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report["total_run_count"] == 10
    assert report["scenario_count_by_size"] == {"medium_large": 1831}
    assert not output.exists()
