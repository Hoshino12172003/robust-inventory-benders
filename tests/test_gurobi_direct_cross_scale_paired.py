from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import yaml

from src.gurobi_direct_cross_scale_paired import (
    Dependencies, DirectBenchmarkError, EXPECTED_SCENARIOS, STAGE,
    _scientific_status, dry_run, execute, expand_plan, load_catalog, load_yaml,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/gurobi_direct_cross_scale_paired.yaml"


def test_plan_is_exactly_the_paired_ten_cells() -> None:
    rows = expand_plan()
    assert [(row["scale"], row["seed"]) for row in rows] == [
        (scale, seed) for scale in ("medium_large", "large") for seed in range(180, 185)
    ]
    assert len({row["run_key"] for row in rows}) == len({row["run_directory_id"] for row in rows}) == 10


def test_dry_run_is_solver_free_and_verifies_source() -> None:
    before = "gurobipy" in sys.modules
    report = dry_run(CONFIG)
    assert report["planned_direct_runs"] == report["source_cells_verified"] == 10
    assert report["hybrid_reruns"] == report["baseline_reruns"] == 0
    assert report["solver_called"] is False
    assert report["gurobipy_imported_by_dry_run"] is False
    assert ("gurobipy" in sys.modules) is before


def test_catalog_pairs_exact_instance_anchor_and_hybrid_identities() -> None:
    config = load_yaml(CONFIG); cells = load_catalog(ROOT, config)
    assert len(cells) == 10
    assert all(cell["source_hybrid_scientific_status"] == "certified_robust_optimal" for cell in cells)
    assert all(cell["baseline_scientific_status"] == "certified_robust_optimal" for cell in cells)
    assert len({cell["instance_canonical_sha256"] for cell in cells}) == 10


def test_certification_requires_exact_model_and_post_evaluation() -> None:
    result = {"status": "optimal", "complete_model_built": True, "gap": 0.0, "objective_t": 0.2}
    post = {"valid": True, "errors": [], "scenario_count": 1831, "objective_t_consistent": True}
    assert _scientific_status(result, post, 1831) == "certified_robust_optimal"
    assert _scientific_status({**result, "status": "time_limit"}, None, 1831) == "time_limit_uncertified"
    assert _scientific_status(result, {**post, "scenario_count": 1}, 1831) == "invalid_post_evaluation"


class FakeDependencies:
    def __init__(self) -> None:
        self.solves = 0

    def value(self) -> Dependencies:
        def solve(_config, _instance, cell):
            self.solves += 1
            scale = cell["scale"]
            return {
                "status": "optimal", "complete_model_built": True, "gap": 0.0,
                "objective_t": cell["source_hybrid_objective_t"], "algorithm_runtime": 1.0,
                "y_values": [1.0], "x_values": [[1.0]], "rows": 1, "columns": 2,
                "nonzeros": 3, "node_count": 0.0, "scenario_count": EXPECTED_SCENARIOS[scale],
            }
        def post(_config, _instance, result, cell, _identity, _root):
            return ({"valid": True, "errors": [], "scenario_count": EXPECTED_SCENARIOS[cell["scale"]],
                     "objective_t_consistent": True}, {})
        return Dependencies(solve, post, lambda value: value)


def test_mock_execute_and_resume_do_not_repeat_direct_runs(tmp_path: Path) -> None:
    config = load_yaml(CONFIG); config["output_dir"] = str(tmp_path / "out")
    custom = tmp_path / "config.yaml"; custom.write_text(yaml.safe_dump(config), encoding="utf-8")
    first = FakeDependencies(); result = execute(custom, resume=True, dependencies=first.value())
    assert result["completed"] == 10 and first.solves == 10
    second = FakeDependencies(); execute(custom, resume=True, dependencies=second.value())
    assert second.solves == 0
    rows = list(__import__("csv").DictReader((tmp_path / "out/paired_results.csv").open(encoding="utf-8")))
    assert len(rows) == 10
    assert all(float(row["objective_abs_difference"]) == 0.0 for row in rows)


def test_resume_uses_direct_checkpoint_without_repeating_solve(tmp_path: Path) -> None:
    config = load_yaml(CONFIG); config["output_dir"] = str(tmp_path / "out")
    custom = tmp_path / "config.yaml"; custom.write_text(yaml.safe_dump(config), encoding="utf-8")
    first = FakeDependencies(); execute(custom, resume=True, dependencies=first.value())
    row = expand_plan()[0]
    run_root = tmp_path / "out/runs" / row["run_directory_id"]
    (run_root / "run.json").unlink(); (run_root / "status.json").unlink()
    recovery = FakeDependencies(); execute(custom, resume=True, dependencies=recovery.value())
    assert recovery.solves == 0
    assert (run_root / "run.json").is_file()


def test_execution_requires_resume() -> None:
    with pytest.raises(DirectBenchmarkError, match="requires --resume"):
        execute(CONFIG, resume=False, dependencies=FakeDependencies().value())


def test_scope_drift_fails_closed(tmp_path: Path) -> None:
    config = load_yaml(CONFIG); config["seeds"] = [180]
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(DirectBenchmarkError, match="seeds drift"):
        dry_run(path)
