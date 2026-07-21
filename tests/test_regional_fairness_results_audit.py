from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile

import pytest
import yaml

from src.experiment_protocol import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_yaml,
    config_sha256,
    file_sha256,
)
from src.experiment_suite import _apply_selected_parameters, experiment_run_specs
from src.instance import InventoryInstance
from src.regional_fairness_diagnostic import RecourseAllocation, first_stage_x_sha256
from src.regional_fairness_pipeline import PipelineDependencies, run_regional_fairness_pipeline
from src.regional_fairness_results_audit import (
    ResultSource,
    ScaleExpectation,
    _pattern_sha256,
    _recompute_metrics,
    audit_regional_fairness_results,
    classify_joint_fairness,
)
from src.scenarios import enumerate_budget_scenarios_with_metadata


def _tiny_instance(seed: int) -> InventoryInstance:
    return InventoryInstance(
        name=f"audit_synthetic_seed{seed}",
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


class _BaseRunner:
    def __call__(self, config: dict, *, resume: bool, overwrite: bool) -> dict[str, Path]:
        assert resume and not overwrite
        output = Path(config["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        resolved = _apply_selected_parameters(config)
        atomic_write_yaml(output / "resolved_config.yaml", resolved)
        rows: list[dict] = []
        for spec in experiment_run_specs(resolved):
            instance = _tiny_instance(spec.seed)
            instance_path = output / "instances" / f"{instance.name}.json"
            atomic_write_json(instance_path, instance.to_dict())
            result = {
                "run_key": spec.run_key,
                "experiment_name": config["experiment_name"],
                "variant_name": spec.variant_name,
                "method": spec.variant_name,
                "seed": spec.seed,
                "instance_size": spec.instance_size,
                "instance_name": instance.name,
                "git_commit": "unknown",
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
                "git_commit": "unknown",
                "config_sha256": result["config_sha256"],
                "result": result,
            }
            run_dir = output / "runs" / spec.run_key
            atomic_write_json(run_dir / "run.json", record)
            atomic_write_json(
                run_dir / "status.json",
                {
                    "run_key": spec.run_key,
                    "state": "complete",
                    "success": True,
                    "solved_to_tolerance": True,
                    "status": "optimal",
                },
            )
            (run_dir / "error.txt").write_text("", encoding="utf-8")
            rows.append(result)
        atomic_write_csv(
            output / "results.csv",
            rows,
            [
                "run_key",
                "experiment_name",
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
                "git_commit": "unknown",
            },
        )
        return {"output_dir": output}


def _solver(
    instance: InventoryInstance,
    scenario,
    best_x_values: list[list[float]],
    **_kwargs,
) -> tuple[RecourseAllocation, RecourseAllocation]:
    common = {
        "scenario_id": scenario.name,
        "status": "optimal",
        "objective": 70.0,
        "original_optimal_cost": 70.0,
        "cost_tolerance": 71.0e-6,
        "shipment_values": [],
        "service_violation_values": [0.0],
        "runtime": 0.01,
        "first_stage_x_sha256": first_stage_x_sha256(best_x_values),
        "gamma_usage": scenario.gamma,
        "original_cost_reproduced": True,
        "cost_cap_satisfied": True,
        "constraints_satisfied": True,
    }
    return (
        RecourseAllocation(
            policy="default",
            shortage_values=[[6.0], [0.0]],
            transport_cost_by_region=[2.0, 8.0],
            allocated_units_by_region=[2.0, 8.0],
            **common,
        ),
        RecourseAllocation(
            policy="fair_best",
            shortage_values=[[3.0], [3.0]],
            transport_cost_by_region=[5.0, 5.0],
            allocated_units_by_region=[5.0, 5.0],
            **common,
        ),
    )


def _build(root: Path, name: str) -> tuple[Path, ScaleExpectation]:
    config_dir = root / "experiments/configs"
    config_dir.mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    protocol = root / "docs/regional_fairness_diagnostic_protocol.md"
    candidate = config_dir / "selected_cut_strengthened_joint_v3_candidate.yaml"
    protocol.write_text("synthetic regional fairness protocol", encoding="utf-8")
    candidate.write_text("selected_variant: joint_v1_core_point_strengthened\n", encoding="utf-8")
    config = {
        "experiment_name": name,
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
    config_path = config_dir / "synthetic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run_regional_fairness_pipeline(
        config_path,
        resume=True,
        dependencies=PipelineDependencies(
            base_runner=_BaseRunner(),
            scenario_enumerator=enumerate_budget_scenarios_with_metadata,
            scenario_solver=_solver,
        ),
        strict_protocol_audit=False,
    )
    output = root / "outputs/fairness"
    expectation = ScaleExpectation(
        label=name,
        instance_size="medium_large",
        experiment_name=name,
        seeds=tuple(range(1, 11)),
        scenario_count=4,
        num_regions=2,
        num_products=1,
        checkpoint_count=20,
        archive_sha256=None,
        config_sha256=file_sha256(config_path),
        config_filename=None,
        run_commit="unknown",
        protocol_sha256=file_sha256(protocol),
        candidate_sha256=file_sha256(candidate),
        chunk_size=2,
    )
    return output, expectation


def _fixture(tmp_path: Path):
    medium_root = tmp_path / "medium"
    large_root = tmp_path / "large"
    medium, medium_exp = _build(medium_root, "synthetic_medium")
    large, large_exp = _build(large_root, "synthetic_large")
    return medium, large, medium_exp, large_exp, medium_root, large_root


def _audit(fixture):
    medium, large, medium_exp, large_exp, medium_root, _large_root = fixture
    return audit_regional_fairness_results(
        medium,
        large,
        repo_root=medium_root,
        expectations={"medium_large": medium_exp, "large": large_exp},
    )


def test_complete_synthetic_evidence_passes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    # Both roots contain byte-identical protocol and candidate files, so one repo root is sufficient.
    report = _audit(fixture)
    assert report["all_required_checks_passed"], report["failed_checks"]
    assert report["diagnosis_valid"]


def test_zip_hash_and_crc_are_independently_checked(tmp_path: Path) -> None:
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("results.csv", "a\n1\n")
    expected = file_sha256(path)
    with ResultSource(path) as source:
        assert source.archive_sha256() == expected
        assert source.crc_error() is None
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("results.csv")
        offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    data = bytearray(path.read_bytes())
    data[offset] ^= 1
    path.write_bytes(data)
    with ResultSource(path) as source:
        assert source.crc_error() == "results.csv"


@pytest.mark.parametrize("mutation", ["identity", "output_hash", "missing_checkpoint", "duplicate_row"])
def test_corruption_missing_duplicate_and_identity_drift_fail(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    medium = fixture[0]
    if mutation == "identity":
        manifest = json.loads((medium / "diagnostic_run_manifest.json").read_text(encoding="utf-8"))
        manifest["diagnostic_code_git_commit"] = "drift"
        (medium / "diagnostic_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "output_hash":
        with (medium / "instance_summary.csv").open("a", encoding="utf-8") as handle:
            handle.write("corrupt\n")
    elif mutation == "missing_checkpoint":
        checkpoint = next((medium / "checkpoint").glob("base_*/chunk_*.json"))
        checkpoint.unlink()
    else:
        path = medium / "region_scenario_metrics.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows.append(dict(rows[0]))
        atomic_write_csv(path, rows, list(rows[0]))
        manifest = json.loads((medium / "diagnostic_run_manifest.json").read_text(encoding="utf-8"))
        manifest["final_outputs"]["region_scenario_metrics.csv"] = file_sha256(path)
        atomic_write_json(medium / "diagnostic_run_manifest.json", manifest)
    assert not _audit(fixture)["all_required_checks_passed"]


@pytest.mark.parametrize("mutation", ["pattern", "recourse_status", "cost_tolerance"])
def test_pattern_recourse_and_cost_tolerance_drift_fail(tmp_path: Path, mutation: str) -> None:
    fixture = _fixture(tmp_path)
    medium = fixture[0]
    index_path = medium / "checkpoint/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = index["entries"][0]
    checkpoint_path = medium / entry["relative_path"]
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record = checkpoint["scenario_records"][0]
    if mutation == "pattern":
        record["deviation_pattern_sha256"] = "0" * 64
    elif mutation == "recourse_status":
        record["fair_best"]["status"] = "time_limit"
    else:
        record["fair_best"]["cost_tolerance"] = 1.0
    atomic_write_json(checkpoint_path, checkpoint)
    entry["sha256"] = file_sha256(checkpoint_path)
    atomic_write_json(index_path, index)
    manifest = json.loads((medium / "diagnostic_run_manifest.json").read_text(encoding="utf-8"))
    manifest["final_outputs"]["checkpoint/index.json"] = file_sha256(index_path)
    atomic_write_json(medium / "diagnostic_run_manifest.json", manifest)
    assert not _audit(fixture)["all_required_checks_passed"]


def test_metric_recomputation_and_zero_demand() -> None:
    metrics = _recompute_metrics([[0.0], [10.0]], [[0.0], [2.0]], metric_tolerance=1e-9)
    assert not metrics["regions"][0]["fill_rate_applicable"]
    assert metrics["regions"][0]["not_applicable_reason"] == "zero_regional_demand"
    assert metrics["regions"][1]["fill_rate"] == pytest.approx(0.8)
    assert metrics["fill_rate_gap"] == pytest.approx(0.0)


def test_deviation_pattern_hash_is_deterministic() -> None:
    pattern = [{"region_id": 1, "product_id": 0, "deviation_value": 2.0, "base_demand": 8.0, "realized_demand": 10.0}]
    assert _pattern_sha256(pattern) == _pattern_sha256(json.loads(json.dumps(pattern)))


@pytest.mark.parametrize(
    "default,fair,expected",
    [
        ([0.20] * 10, [0.20] * 10, "structural_fairness_gap"),
        ([0.20] * 10, [0.01] * 10, "recourse_degeneracy_only"),
        ([0.01] * 10, [0.01] * 10, "no_material_fairness_gap"),
        ([0.04] * 10, [0.04] * 10, "fairness_diagnostic_inconclusive"),
    ],
)
def test_all_four_joint_categories(default: list[float], fair: list[float], expected: str) -> None:
    report = classify_joint_fairness(
        {
            "medium_large": {"default_WGap": default, "fair_best_WGap": fair},
            "large": {"default_WGap": default, "fair_best_WGap": fair},
        },
        correctness_valid=True,
    )
    assert report["decision"] == expected


def test_invalid_evidence_never_returns_a_substantive_decision() -> None:
    report = classify_joint_fairness({}, correctness_valid=False)
    assert report["decision"] == "fairness_diagnostic_invalid"
    assert not report["diagnosis_valid"]
