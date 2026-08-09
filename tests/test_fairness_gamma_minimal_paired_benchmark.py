from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from src.experiment_protocol import atomic_write_json
from src.experiment_protocol import file_sha256
from src.fairness_gamma_minimal_paired_benchmark_audit import build_source_catalog, static_audit
from src.fairness_gamma_minimal_paired_benchmark_runner import (
    BenchmarkGateError, Dependencies, STAGE, _final_certificate, classify_status, dry_run, execute_plan,
    expand_plan, formal_git_gate, load_catalog, load_yaml, run_directory_id,
    validate_authorization, validate_solution_payload,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/fairness_gamma_minimal_paired_benchmark.yaml"
SOURCE_ZIP = Path(r"E:\论文代码\fairness_hybrid_gamma_sensitivity_attempt3_results.zip")
S1_ZIP = Path(r"E:\论文代码\fairness_scalability_s1_attempt2_medium_large_results.zip")


def _instance(_source: Path, _cell: dict) -> dict:
    return {"num_warehouses": 2, "num_products": 2}


def _result(runtime: float = 0.1) -> dict:
    return {
        "status": "optimal", "runtime": runtime, "gap": 0.0, "iterations": 1, "cuts": 1,
        "objective_t": 0.2, "x_values": [[1.0, 2.0], [3.0, 4.0]], "y_values": [1.0, 1.0],
        "robust_feasibility_certified": True,
        "iteration_log": [{
            "certification_active": True, "robust_feasibility_certified": True,
            "master_status": 2, "separation_status": "optimal",
            "separation_requested_mip_gap": 0.0, "separation_objective_bound": -0.0,
            "master_time": 0.02, "subproblem_time": 0.03,
        }],
    }


class FakePipeline:
    def __init__(self) -> None:
        self.solve_calls = 0
        self.post_calls = 0

    def dependencies(self) -> Dependencies:
        def solve(_config: dict, _instance_value: dict, _cell: dict) -> dict:
            self.solve_calls += 1
            return _result()

        def post(_config: dict, _instance_value: dict, _result_value: dict, _cell: dict, identity: dict, root: Path):
            self.post_calls += 1
            expected = 1831 if identity["scale"] == "medium_large" else 4657
            value = {
                "valid": True, "errors": [], "objective_t_consistent": True,
                "scenario_count": expected, "actual_robust_cost": 100.0,
            }
            root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(root / "post_evaluation_final.json", value)
            atomic_write_json(root / "post_evaluation_index.json", {"scenario_count": expected})
            return value, {"post_evaluation_wall_runtime": 0.2}

        return Dependencies(solve, post, lambda value: value)


def _config_and_cells() -> tuple[dict, list[dict]]:
    config = load_yaml(CONFIG_PATH)
    return config, load_catalog(config)


def test_exact_plan_and_short_directory_identity() -> None:
    rows = expand_plan()
    assert len(rows) == 10
    assert {(r["scale"], r["seed"], r["gamma"], r["rho"]) for r in rows} == {
        (scale, seed, 2, "0.025") for scale in ("medium_large", "large") for seed in range(180, 185)
    }
    assert len({r["run_key"] for r in rows}) == len({r["run_directory_id"] for r in rows}) == 10
    assert all(r["run_directory_id"] == run_directory_id(r["run_key"]) for r in rows)


def test_dry_run_is_solver_free_and_side_effect_free() -> None:
    before = "gurobipy" in sys.modules
    report = dry_run(CONFIG_PATH)
    assert report["gurobipy_imported"] is before
    assert report["gurobipy_imported_by_dry_run"] is False
    assert report["reference_frontier"] == 10
    assert report["baseline_new"] == report["hybrid_new"] == 0
    assert report["unique_run_keys"] == report["unique_directory_ids"] == 10
    assert report["instances_generated"] is report["solver_called"] is False
    assert report["source_instance_payloads_read"] is False
    assert report["longest_windows_path_length"] < 220


@pytest.mark.skipif(not SOURCE_ZIP.exists(), reason="formal source ZIP is not mounted")
def test_source_archive_catalog_is_exact_and_read_only() -> None:
    before = file_sha256(SOURCE_ZIP)
    rebuilt = build_source_catalog(SOURCE_ZIP)
    frozen = json.loads((ROOT / "analysis/fairness_gamma_minimal_paired_benchmark_protocol/source_pairing_catalog.json").read_text(encoding="utf-8"))
    assert rebuilt == frozen
    assert file_sha256(SOURCE_ZIP) == before
    assert len(rebuilt["cells"]) == 10


@pytest.mark.skipif(not SOURCE_ZIP.exists(), reason="formal source ZIP is not mounted")
def test_benchmark_static_audit_protects_solver_core_and_source() -> None:
    report = static_audit(ROOT, SOURCE_ZIP)
    assert report["decision"] == "approve_benchmark_protocol"
    assert all(report["checks"].values())


@pytest.mark.parametrize("bad", [True, "1", float("nan"), float("inf")])
def test_solution_numeric_schema_rejects_nonfinite_and_non_numeric(bad: object) -> None:
    result = _result()
    result["x_values"][0][0] = bad
    with pytest.raises(BenchmarkGateError):
        validate_solution_payload(result, _instance(Path(), {}))


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(x_values=[[1.0, 2.0]]),
    lambda r: r.update(x_values=[[1.0], [2.0]]),
    lambda r: r.update(x_values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    lambda r: r.update(x_values={"0,0": 1.0}),
    lambda r: r.update(y_values=[1.0]),
])
def test_solution_dimensions_and_order_fail_closed(mutation) -> None:
    result = _result()
    mutation(result)
    with pytest.raises(BenchmarkGateError):
        validate_solution_payload(result, _instance(Path(), {}))


def test_baseline_production_field_names_are_validated() -> None:
    baseline = {"best_x_values": [[1.0, 2.0], [3.0, 4.0]], "best_y_values": [1.0, 1.0]}
    validate_solution_payload(baseline, _instance(Path(), {}), baseline=True)
    with pytest.raises(BenchmarkGateError):
        validate_solution_payload({"x_values": baseline["best_x_values"], "y_values": baseline["best_y_values"]}, _instance(Path(), {}), baseline=True)


def test_certification_and_post_evaluation_are_both_required() -> None:
    post = {"valid": True, "errors": [], "objective_t_consistent": True, "scenario_count": 1831}
    assert classify_status(_result(), post, 1831) == "certified_robust_optimal"
    bad = _result()
    bad["iteration_log"][-1]["separation_objective_bound"] = 1e-3
    assert classify_status(bad, post, 1831) != "certified_robust_optimal"
    assert classify_status(_result(), {**post, "valid": False}, 1831) == "invalid_post_evaluation"


@pytest.mark.skipif(not S1_ZIP.exists(), reason="S1 production archive is not mounted")
def test_production_single_cut_final_certificate_schema() -> None:
    import zipfile
    with zipfile.ZipFile(S1_ZIP) as archive:
        result = next(
            json.loads(archive.read(name))["result"]
            for name in archive.namelist()
            if name.endswith("/run.json") and json.loads(archive.read(name)).get("candidate") == "single_cut"
        )
    assert result.get("robust_feasibility_certified") is None
    assert _final_certificate(result) is True


def test_reference_candidate_disables_every_acceleration() -> None:
    candidate = load_yaml(ROOT / "experiments/configs/fairness_gamma_minimal_paired_reference_candidate.yaml")
    assert candidate["complete_scenario_recourse_blocks_enabled"] is False
    assert candidate["persistent_separation_enabled"] is False
    assert candidate["certified_scenario_cache_enabled"] is False
    assert candidate["separation_solution_pool_enabled"] is False
    assert candidate["batch_size"] == candidate["max_certified_cuts_per_iteration"] == 1
    assert candidate["certified_farkas_separation_required"] is True
    assert candidate["final_exact_separation_required"] is True


@pytest.mark.skipif(not SOURCE_ZIP.exists(), reason="formal source ZIP is not mounted")
def test_synthetic_ten_task_disk_pipeline_and_second_resume_are_deterministic(tmp_path: Path) -> None:
    config, cells = _config_and_cells()
    output = tmp_path / "out"
    first = FakePipeline()
    result = execute_plan(CONFIG_PATH, config, cells, output, commit="1" * 40, dependencies=first.dependencies(), source_zip=SOURCE_ZIP, source_instance_loader=_instance)
    assert result["completed"] == 10 and first.solve_calls == first.post_calls == 10
    assert len(list(output.glob("runs/*/run.json"))) == 10
    assert len(list(output.glob("runs/*/status.json"))) == 10
    files = {name: (output / name).read_bytes() for name in ("results.csv", "summary.csv", "paired_comparison.csv", "manifest.json", "run_manifest.json")}
    second = FakePipeline()
    execute_plan(CONFIG_PATH, config, cells, output, commit="1" * 40, dependencies=second.dependencies(), source_zip=SOURCE_ZIP, source_instance_loader=_instance)
    assert second.solve_calls == second.post_calls == 0
    assert files == {name: (output / name).read_bytes() for name in files}


@pytest.mark.skipif(not SOURCE_ZIP.exists(), reason="formal source ZIP is not mounted")
def test_resume_rejects_manifest_and_checkpoint_identity_corruption(tmp_path: Path) -> None:
    config, cells = _config_and_cells()
    output = tmp_path / "out"
    pipeline = FakePipeline()
    execute_plan(CONFIG_PATH, config, cells, output, commit="2" * 40, dependencies=pipeline.dependencies(), source_zip=SOURCE_ZIP, source_instance_loader=_instance)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["run_key_to_directory_id"] = {}
    atomic_write_json(output / "manifest.json", manifest)
    with pytest.raises(BenchmarkGateError, match="mapping"):
        execute_plan(CONFIG_PATH, config, cells, output, commit="2" * 40, dependencies=FakePipeline().dependencies(), source_zip=SOURCE_ZIP, source_instance_loader=_instance)


@pytest.mark.skipif(not SOURCE_ZIP.exists(), reason="formal source ZIP is not mounted")
def test_resume_rejects_checkpoint_corruption(tmp_path: Path) -> None:
    config, cells = _config_and_cells()
    output = tmp_path / "out"
    execute_plan(CONFIG_PATH, config, cells, output, commit="3" * 40, dependencies=FakePipeline().dependencies(), source_zip=SOURCE_ZIP, source_instance_loader=_instance)
    row = expand_plan()[0]
    run_root = output / "runs" / row["run_directory_id"]
    (run_root / "run.json").unlink()
    (run_root / "status.json").unlink()
    checkpoint = json.loads((run_root / "algorithm_checkpoint.json").read_text())
    checkpoint["identity"]["gamma"] = 1
    atomic_write_json(run_root / "algorithm_checkpoint.json", checkpoint)
    with pytest.raises(BenchmarkGateError, match="checkpoint identity"):
        execute_plan(CONFIG_PATH, config, cells, output, commit="3" * 40, dependencies=FakePipeline().dependencies(), source_zip=SOURCE_ZIP, source_instance_loader=_instance)


@pytest.mark.skipif(not SOURCE_ZIP.exists(), reason="formal source ZIP is not mounted")
def test_post_interruption_resumes_without_repeating_algorithm(tmp_path: Path) -> None:
    config, cells = _config_and_cells()
    output = tmp_path / "out"
    calls = {"solve": 0, "post": 0}

    def solve(_config, _instance_value, _cell):
        calls["solve"] += 1
        return _result()

    def interrupted_post(*_args, **_kwargs):
        calls["post"] += 1
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_plan(CONFIG_PATH, config, cells, output, commit="4" * 40, dependencies=Dependencies(solve, interrupted_post, lambda v: v), source_zip=SOURCE_ZIP, source_instance_loader=_instance)
    assert calls == {"solve": 1, "post": 1}
    recovery = FakePipeline()
    execute_plan(CONFIG_PATH, config, cells, output, commit="4" * 40, dependencies=recovery.dependencies(), source_zip=SOURCE_ZIP, source_instance_loader=_instance)
    assert recovery.solve_calls == 9
    assert recovery.post_calls == 10


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def test_real_git_gate_allows_only_ignored_output_and_authorization_successor(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / ".gitignore").write_text("/experiments/results_fgmpb/\n", encoding="utf-8")
    (repo / "code.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "code.py")
    _git(repo, "commit", "-m", "implementation")
    implementation = _git(repo, "rev-parse", "HEAD")
    auth_path = "experiments/configs/auth.json"
    (repo / "experiments/configs").mkdir(parents=True)
    (repo / auth_path).write_text("{}\n", encoding="utf-8")
    _git(repo, "add", auth_path)
    _git(repo, "commit", "-m", "authorization")
    auth = {"authorized_implementation_commit": implementation, "authorization_only_path": auth_path, "output_relative_path": "experiments/results_fgmpb/a1"}
    formal_git_gate(repo, auth)
    output = repo / "experiments/results_fgmpb/a1"
    output.mkdir(parents=True)
    (output / "checkpoint.json").write_text("{}", encoding="utf-8")
    formal_git_gate(repo, auth)
    (repo / "unrelated.txt").write_text("x", encoding="utf-8")
    with pytest.raises(BenchmarkGateError, match="untracked"):
        formal_git_gate(repo, auth)
    (repo / "unrelated.txt").unlink()
    (repo / "code.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(BenchmarkGateError, match="tracked"):
        formal_git_gate(repo, auth)


def test_authorization_identity_and_scope_drift_fail_closed(tmp_path: Path) -> None:
    config = load_yaml(CONFIG_PATH)
    auth_path = ROOT / "experiments/configs/fairness_gamma_minimal_paired_benchmark_authorization.json"
    authorization, digest = validate_authorization(auth_path, CONFIG_PATH.resolve(), config, Path(r"E:\rfab1"))
    assert authorization["formal_run_authorized"] is True and len(digest) == 64
    drifted = deepcopy(authorization)
    drifted["gamma"] = [1, 2]
    drift_path = tmp_path / "authorization.json"
    drift_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(BenchmarkGateError, match="authorization gamma mismatch"):
        validate_authorization(drift_path, CONFIG_PATH.resolve(), config, Path(r"E:\rfab1"))
