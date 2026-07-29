from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.experiment_protocol import file_sha256
from src.fairness_hybrid_ccg_benders import INITIAL_UPPER_BOUND_EXPECTED_IDENTITY_FIELDS
from src.fairness_hybrid_ccg_benders_runner import HybridDependencies
from src.fairness_hybrid_final_holdout_audit import (
    D2_ARCHIVE_SHA256,
    audit_d2_archive,
    write_freeze_evidence,
)
from src.fairness_hybrid_final_holdout_static_audit import audit as static_audit
import src.fairness_hybrid_final_holdout_runner as runner
from tests.test_robust_regional_fairness import tiny_instance


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/fairness_hybrid_final_cross_scale_holdout.yaml"
D2_ARCHIVE = Path(r"E:\论文代码\fairness_hybrid_ccg_benders_d2_a3_large_results.zip")


def _fake_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    scales = {}
    for scale, frozen in runner.SCALES.items():
        output = tmp_path / scale
        scales[scale] = {**frozen, "output_dir": str(output)}
        config["scales"][scale]["output_dir"] = str(output)
    monkeypatch.setattr(runner, "SCALES", scales)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
    monkeypatch.setattr(runner, "EXPECTED_CONFIG_SHA256", file_sha256(path).upper())
    return config, path


def _dependencies(calls: dict[str, int], *, certified: bool = True) -> HybridDependencies:
    def generate(config, seed):
        calls["generate"] += 1
        assert config["scale"] in {"medium_large", "large"}
        assert seed in range(170, 180)
        return tiny_instance()

    def baseline(_config, _instance, _seed, params):
        calls["baseline"] += 1
        assert params == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}
        return {
            "status": "optimal", "valid_UB": True, "gap": 0.0,
            "upper_bound": 30.0, "runtime": 1.0,
            "best_y_values": [1.0], "best_x_values": [[5.0]],
        }

    def frontier(config, _instance, baseline_record, anchor, expected, _checkpoint, params, _row):
        calls["frontier"] += 1
        assert tuple(expected) == INITIAL_UPPER_BOUND_EXPECTED_IDENTITY_FIELDS
        assert expected["baseline_run_key"] == baseline_record["run_key"]
        assert expected["anchor_sha256"] == anchor["anchor_sha256"]
        assert params == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}
        if not certified:
            return {
                "status": "time_limit", "gap": 1.0, "lower_bound": 0.0,
                "upper_bound": 1.0, "objective_t": 1.0, "runtime": 1800.0,
                "y_values": [1.0], "x_values": [[5.0]], "iteration_log": [],
                "metadata": {"robust_feasibility_certified": False},
            }
        assert config["scenario_count"] in {1831, 4657}
        return {
            "status": "optimal", "gap": 0.0, "lower_bound": 0.5,
            "upper_bound": 0.5, "objective_t": 0.5, "runtime": 2.0,
            "y_values": [1.0], "x_values": [[5.0]],
            "iteration_log": [{
                "master_status": "optimal",
                "robust_feasibility_certified": True,
                "final_exact_separation_performed": True,
                "separation_objective_bound": 0.0,
            }],
            "metadata": {
                "full_separation_objective_bound_required": True,
                "robust_feasibility_certified": True,
            },
        }

    def post(config, *_args):
        calls["post"] += 1
        return (
            {"valid": True, "scenario_count": config["scenario_count"], "objective_t_consistent": True, "errors": []},
            {"post_evaluation_solver_runtime": 1.0, "post_evaluation_wall_runtime": 1.0,
             "aggregation_runtime": 0.1, "checkpoint_io_runtime": 0.1},
        )

    def configure(params):
        calls["configure"] += 1
        assert params == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}

    return HybridDependencies(generate, baseline, frontier, post, configure)


def test_final_holdout_plan_is_exact_and_unique() -> None:
    rows = runner.expand_plan()
    assert len(rows) == len({row["run_key"] for row in rows}) == 120
    assert {row["scale"] for row in rows} == {"medium_large", "large"}
    for scale in ("medium_large", "large"):
        scale_rows = [row for row in rows if row["scale"] == scale]
        assert sum(row["task_type"] == "baseline" for row in scale_rows) == 10
        assert sum(row["task_type"] == "frontier" for row in scale_rows) == 50
        for seed in range(170, 180):
            seed_rows = [row for row in scale_rows if row["seed"] == seed]
            assert sum(row["task_type"] == "baseline" for row in seed_rows) == 1
            assert sum(row["task_type"] == "frontier" for row in seed_rows) == 5


def test_dry_run_is_solver_free_side_effect_free_and_portable() -> None:
    outputs = [ROOT / value["output_dir"] for value in runner.SCALES.values()]
    assert not any(path.exists() for path in outputs)
    report = runner.dry_run(CONFIG)
    assert report["baseline"] == 20 and report["frontier"] == 100 and report["total"] == 120
    assert report["instances_generated"] is report["solver_called"] is False
    assert report["output_dirs_exist"] is False
    assert report["windows_path_check"] is True and report["longest_path_length"] < 220
    assert report["reserved_seed_access_audit_passed"] is True
    assert report["independent_unit"] == "seed" and report["seed_cluster_count"] == 10
    assert not any(path.exists() for path in outputs)


def test_formal_gate_fails_before_any_output_or_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = [ROOT / value["output_dir"] for value in runner.SCALES.values()]
    monkeypatch.setattr(
        runner,
        "_formal_git_gate",
        lambda _root: (_ for _ in ()).throw(runner.RemediationGateError("formal_run_not_authorized")),
    )
    with pytest.raises(Exception, match="formal_run_not_authorized"):
        runner.run_holdout(CONFIG, resume=True)
    assert not any(path.exists() for path in outputs)


def test_fake_authorized_cross_scale_pipeline_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, path = _fake_config(tmp_path, monkeypatch)
    calls = {key: 0 for key in ("generate", "baseline", "frontier", "post", "configure")}
    deps = _dependencies(calls)
    report = runner.run_holdout(path, resume=True, dependencies=deps, test_authorization=True)
    assert report["completed_run_count"] == report["certified_solved_count"] == 120
    assert calls == {"generate": 20, "baseline": 20, "frontier": 100, "post": 100, "configure": 1}
    for scale in runner.SCALES:
        manifest = report["manifests"][scale]
        assert manifest["completed_run_count"] == 60
        assert manifest["baseline_certified_count"] == 10
        assert manifest["frontier_certified_count"] == 50
        for seed in range(170, 180):
            identities = [value for value in manifest["run_identities"].values() if value["seed"] == seed]
            assert len(identities) == 5
            assert len({value["baseline_run_key"] for value in identities}) == 1
            assert len({value["anchor_sha256"] for value in identities}) == 1
            assert len({value["instance_sha256"] for value in identities}) == 1
        output = Path(config["scales"][scale]["output_dir"])
        assert len((output / "results.csv").read_text(encoding="utf-8").splitlines()) == 61
    runner.run_holdout(path, resume=True, dependencies=deps, test_authorization=True)
    assert calls == {"generate": 20, "baseline": 20, "frontier": 100, "post": 100, "configure": 2}


def test_uncertified_frontiers_never_count_as_solved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config, path = _fake_config(tmp_path, monkeypatch)
    calls = {key: 0 for key in ("generate", "baseline", "frontier", "post", "configure")}
    report = runner.run_holdout(
        path,
        resume=True,
        dependencies=_dependencies(calls, certified=False),
        test_authorization=True,
    )
    assert calls["frontier"] == 100 and calls["post"] == 0
    assert report["completed_run_count"] == 120
    assert report["certified_solved_count"] == 20


def test_resume_manifest_identity_drift_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, path = _fake_config(tmp_path, monkeypatch)
    calls = {key: 0 for key in ("generate", "baseline", "frontier", "post", "configure")}
    deps = _dependencies(calls)
    runner.run_holdout(path, resume=True, dependencies=deps, test_authorization=True)
    manifest_path = Path(config["scales"]["large"]["output_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["execution_attempt"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(Exception, match="resume identity mismatch"):
        runner.run_holdout(path, resume=True, dependencies=deps, test_authorization=True)


def test_requires_resume_and_rejects_dependency_substitution() -> None:
    with pytest.raises(Exception, match="requires --resume"):
        runner.run_holdout(CONFIG, resume=False)
    calls = {key: 0 for key in ("generate", "baseline", "frontier", "post", "configure")}
    with pytest.raises(Exception, match="test_authorization"):
        runner.run_holdout(CONFIG, resume=True, dependencies=_dependencies(calls))


def test_production_frontier_locks_protocol_and_checkpoint_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    class Result:
        def to_dict(self):
            return {"status": "time_limit"}

    def fake_solver(*_args, **kwargs):
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(runner, "solve_certified_hybrid_scenario_benders_fairness", fake_solver)
    expected = {field: "x" for field in INITIAL_UPPER_BOUND_EXPECTED_IDENTITY_FIELDS}
    config = runner._scale_config(runner.load_config(CONFIG), "large")
    runner._production_frontier(
        config, tiny_instance(), {}, {}, expected, tmp_path / "checkpoint.json",
        {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}, {"run_key": "run", "rho": "0"},
    )
    assert captured["execution_protocol_sha256"] == runner.PROTOCOL_SHA256
    assert captured["checkpoint_identity"] == {"run_key": "run", **expected}
    assert captured["time_limit"] == 1800.0 and captured["feasibility_tolerance"] == 1e-7


@pytest.mark.parametrize(("scale", "scenario_count"), [("medium_large", 1831), ("large", 4657)])
def test_production_post_evaluation_propagates_frozen_scale_identity(
    scale: str, scenario_count: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    captured = {}

    def fake(*_args, **kwargs):
        captured.update(kwargs)
        evaluation = type("Evaluation", (), {"to_dict": lambda self: {"valid": True}})()
        timing = type("Timing", (), {
            "solver_runtime": 0.0, "wall_runtime": 0.0,
            "aggregation_runtime": 0.0, "checkpoint_io_runtime": 0.0,
        })()
        return evaluation, timing

    monkeypatch.setattr(runner, "checkpointed_fairness_post_evaluation", fake)
    config = runner._scale_config(runner.load_config(CONFIG), scale)
    runner._production_post_evaluate(
        config, tiny_instance(), {"y_values": [1.0], "x_values": [[5.0]], "objective_t": 0.5},
        {"anchor_sha256": "A" * 64, "value": 30.0},
        {"resolved_config_file_sha256": "B" * 64, "git_commit": "c" * 40},
        tmp_path, {"run_key": "run", "rho": "0"},
    )
    assert captured["max_scenarios"] == scenario_count
    assert captured["per_scenario_time_limit"] == 30.0
    assert captured["chunk_size"] == 25
    assert captured["run_execution_attempt"] == 1


@pytest.mark.parametrize("stage", ["D1", "D2", "L0", "L1", "M1", "S2", "full-grid"])
def test_other_stages_cannot_borrow_holdout_protocol(stage: str) -> None:
    with pytest.raises(Exception, match="only FINAL_HOLDOUT"):
        runner.main(["--config", str(CONFIG), "--stage", stage, "--dry-run"])


def test_seed_access_audit_distinguishes_declaration_from_evidence(tmp_path: Path) -> None:
    declaration = tmp_path / "declaration.json"
    declaration.write_text(json.dumps({"reserved_seeds": list(range(170, 180))}), encoding="utf-8")
    assert runner.structured_seed_access_evidence(tmp_path) == []
    evidence = tmp_path / "run.json"
    evidence.write_text(json.dumps({"seed": 170, "run_key": "formal-run"}), encoding="utf-8")
    assert runner.structured_seed_access_evidence(tmp_path) == ["run.json"]


def test_statistics_freeze_seed_as_independent_unit() -> None:
    config = runner.load_config(CONFIG)
    statistics = config["statistics"]
    assert statistics["independent_unit"] == "seed"
    assert statistics["cluster_members"] == ["scale", "rho"]
    assert statistics["cluster_bootstrap_resamples_seed"] is True
    assert statistics["per_rho_cross_scale_pair_count"] == 10
    assert statistics["multiple_testing"] == "Holm_if_five_rho_tests"
    assert statistics["prohibit_seed_rho_independence"] is True


def test_protocol_evidence_rebuild_is_deterministic(tmp_path: Path) -> None:
    first = runner.write_protocol_evidence(CONFIG, tmp_path / "first")
    second = runner.write_protocol_evidence(CONFIG, tmp_path / "second")
    assert first == second
    assert len((tmp_path / "first/frozen_run_plan.csv").read_text(encoding="utf-8").splitlines()) == 121


def test_committed_evidence_hashes_match_lf_bytes() -> None:
    decision_dir = ROOT / "analysis/fairness_hybrid_ccg_benders_d2_decision"
    rows = (decision_dir / "artifact_sha256.csv").read_text(encoding="utf-8").splitlines()[1:]
    for row in rows:
        name, expected = row.split(",")
        assert file_sha256(decision_dir / name).upper() == expected
    assert file_sha256(decision_dir / "decision.json").upper() == runner.D2_DECISION_SHA256


@pytest.mark.skipif(not D2_ARCHIVE.exists(), reason="formal D2 archive is not available")
def test_real_d2_archive_and_deterministic_freeze(tmp_path: Path) -> None:
    before = D2_ARCHIVE.read_bytes()
    result = audit_d2_archive(D2_ARCHIVE)
    assert result["status"] == "pass"
    assert result["source_archive_sha256"] == D2_ARCHIVE_SHA256
    assert result["baseline_certified_count"] == 3
    assert result["frontier_certified_count"] == 9
    assert result["post_evaluation_chunk_count"] == 1683
    assert result["post_evaluation_scenario_total"] == 9 * 4657
    first = write_freeze_evidence(result, tmp_path / "first")
    second = write_freeze_evidence(result, tmp_path / "second")
    assert first == second
    assert D2_ARCHIVE.read_bytes() == before


def test_committed_d2_decision_is_locked_to_archive() -> None:
    decision = json.loads(
        (ROOT / "analysis/fairness_hybrid_ccg_benders_d2_decision/decision.json").read_text(encoding="utf-8")
    )
    assert decision["decision"] == "approve_final_cross_scale_holdout_protocol"
    assert decision["source_archive_sha256"] == D2_ARCHIVE_SHA256
    assert decision["algorithm_development_closed"] is True
    assert decision["final_holdout_formal_run_authorized"] is False


def test_static_audit_passes() -> None:
    result = static_audit(ROOT)
    assert result["status"] == "pass" and result["passed"] == result["total"]
