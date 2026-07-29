from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from src.experiment_protocol import file_sha256
from src.fairness_hybrid_ccg_benders_d2_audit import audit
import src.fairness_hybrid_ccg_benders_d2_runner as runner
from src.fairness_hybrid_ccg_benders_runner import HybridDependencies
from tests.test_robust_regional_fairness import tiny_instance


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/fairness_hybrid_ccg_benders_d2.yaml"


def _fake_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = tmp_path / "new_d2_output"
    config["output_dir"] = str(output)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(runner, "OUTPUT_RELATIVE", str(output))
    monkeypatch.setattr(runner, "EXPECTED_CONFIG_SHA256", file_sha256(path).upper())
    return config, path


def _dependencies(calls: dict[str, int], *, certified: bool = True) -> HybridDependencies:
    def generate(_config, seed):
        calls["generate"] += 1
        assert seed in [160, 161, 162]
        return tiny_instance()

    def baseline(_config, _instance, _seed, params):
        calls["baseline"] += 1
        assert params == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}
        return {
            "status": "optimal", "valid_UB": True, "gap": 0.0,
            "upper_bound": 30.0, "runtime": 1.0,
            "best_y_values": [1.0], "best_x_values": [[5.0]],
        }

    def frontier(_config, _instance, baseline_record, anchor, expected, _checkpoint, params, row):
        calls["frontier"] += 1
        assert expected["baseline_run_key"] == baseline_record["run_key"]
        assert expected["anchor_sha256"] == anchor["anchor_sha256"]
        assert expected["stage"] == "D2" and expected["execution_attempt"] == 2
        assert expected["protocol_sha256"] == runner.PROTOCOL_SHA256
        assert params == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}
        if not certified:
            return {
                "status": "time_limit", "gap": 1.0, "lower_bound": 0.0,
                "upper_bound": 1.0, "objective_t": 1.0, "runtime": 1800.0,
                "y_values": [1.0], "x_values": [[5.0]], "iteration_log": [],
                "metadata": {"robust_feasibility_certified": False},
            }
        return {
            "status": "optimal", "gap": 0.0, "lower_bound": 0.5,
            "upper_bound": 0.5, "objective_t": 0.5, "runtime": 2.0,
            "y_values": [1.0], "x_values": [[5.0]],
            "iteration_log": [{
                "master_status": "optimal", "robust_feasibility_certified": True,
                "final_exact_separation_performed": True,
                "separation_objective_bound": 0.0,
            }],
            "metadata": {
                "full_separation_objective_bound_required": True,
                "robust_feasibility_certified": True,
            },
        }

    def post(*_args):
        calls["post"] += 1
        return (
            {"valid": True, "scenario_count": 4657, "objective_t_consistent": True, "errors": []},
            {"post_evaluation_solver_runtime": 1.0, "post_evaluation_wall_runtime": 1.0,
             "aggregation_runtime": 0.1, "checkpoint_io_runtime": 0.1},
        )

    def configure(params):
        calls["configure"] += 1
        assert params == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}

    return HybridDependencies(generate, baseline, frontier, post, configure)


def test_d2_plan_and_dry_run_are_exact_and_side_effect_free() -> None:
    output = ROOT / runner.OUTPUT_RELATIVE
    assert not output.exists()
    rows = runner.expand_d2_plan()
    report = runner.dry_run(CONFIG)
    assert len(rows) == len({row["run_key"] for row in rows}) == 12
    assert sum(row["task_type"] == "baseline" for row in rows) == 3
    assert sum(row["task_type"] == "frontier" for row in rows) == 9
    assert report["total"] == 12 and report["uncertainty_scenarios"] == 4657
    assert report["instances_generated"] is report["solver_called"] is False
    assert report["output_dir_exists"] is False and report["windows_path_check"] is True
    assert report["longest_path_length"] < 220
    assert not output.exists()


@pytest.mark.parametrize("stage", ["D1", "L0", "L1", "M1", "S2", "full-grid"])
def test_non_d2_stage_is_rejected(stage: str) -> None:
    with pytest.raises(Exception, match="only D2"):
        runner.main(["--config", str(CONFIG), "--stage", stage, "--dry-run"])


def test_formal_gate_rejects_before_output_or_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, path = _fake_config(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_formal_git_gate", lambda _root: (_ for _ in ()).throw(runner.RemediationGateError("formal_run_not_authorized")))
    with pytest.raises(Exception, match="formal_run_not_authorized"):
        runner.run_d2(path, resume=True)
    assert not Path(config["output_dir"]).exists()


def test_fake_authorized_end_to_end_resume_and_shared_anchor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, path = _fake_config(tmp_path, monkeypatch)
    calls = {key: 0 for key in ("generate", "baseline", "frontier", "post", "configure")}
    deps = _dependencies(calls)
    manifest = runner.run_d2(path, resume=True, dependencies=deps, test_authorization=True)
    assert calls == {"generate": 3, "baseline": 3, "frontier": 9, "post": 9, "configure": 1}
    assert manifest["completed_run_count"] == 12
    assert manifest["certified_solved_count"] == 12
    assert manifest["d2_gate"]["passed"] is True
    for seed in runner.SEEDS:
        identities = [value for value in manifest["run_identities"].values() if value["seed"] == seed]
        assert len(identities) == 3
        assert len({value["baseline_run_key"] for value in identities}) == 1
        assert len({value["anchor_sha256"] for value in identities}) == 1
        assert len({value["instance_sha256"] for value in identities}) == 1
    results = list((Path(config["output_dir"]) / "results.csv").read_text(encoding="utf-8").splitlines())
    assert len(results) == 13
    runner.run_d2(path, resume=True, dependencies=deps, test_authorization=True)
    assert calls == {"generate": 3, "baseline": 3, "frontier": 9, "post": 9, "configure": 2}


def test_uncertified_frontiers_continue_but_never_count_as_solved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config, path = _fake_config(tmp_path, monkeypatch)
    calls = {key: 0 for key in ("generate", "baseline", "frontier", "post", "configure")}
    manifest = runner.run_d2(
        path, resume=True, dependencies=_dependencies(calls, certified=False),
        test_authorization=True,
    )
    assert calls["frontier"] == 9 and calls["post"] == 0
    assert manifest["completed_run_count"] == 12
    assert manifest["certified_solved_count"] == 3
    assert manifest["d2_gate"]["passed"] is False
    assert manifest["d2_gate"]["final_holdout_or_full_grid_authorized"] is False


def test_resume_identity_drift_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, path = _fake_config(tmp_path, monkeypatch)
    calls = {key: 0 for key in ("generate", "baseline", "frontier", "post", "configure")}
    deps = _dependencies(calls)
    runner.run_d2(path, resume=True, dependencies=deps, test_authorization=True)
    manifest_path = Path(config["output_dir"]) / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["identity"]["execution_attempt"] = 99
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="identity mismatch"):
        runner.run_d2(path, resume=True, dependencies=deps, test_authorization=True)


def test_d2_post_evaluation_uses_unambiguous_attempt_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}
    def fake(*_args, **kwargs):
        captured.update(kwargs)
        return type("E", (), {"to_dict": lambda self: {"valid": True}})(), type(
            "T", (), {"solver_runtime": 0.0, "wall_runtime": 0.0,
                       "aggregation_runtime": 0.0, "checkpoint_io_runtime": 0.0}
        )()
    monkeypatch.setattr(runner, "checkpointed_fairness_post_evaluation", fake)
    result = {"y_values": [1.0], "x_values": [[5.0]], "objective_t": 0.5}
    anchor = {"anchor_sha256": "A" * 64, "value": 30.0}
    identity = {"resolved_config_file_sha256": "B" * 64, "git_commit": "c" * 40}
    row = {"run_key": "run", "rho": "0.00"}
    runner._production_post_evaluate_d2(
        yaml.safe_load(CONFIG.read_text(encoding="utf-8")), tiny_instance(), result,
        anchor, identity, tmp_path, row,
    )
    assert captured["run_execution_attempt"] == 2
    assert captured["post_evaluation_pipeline_generation"] == 4


def test_static_audit_passes() -> None:
    result = audit(ROOT)
    assert result["status"] == "pass" and result["passed"] == result["total"]
