from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest
import yaml

import src.fairness_scalability_runner as runner
from src.experiment_protocol import atomic_write_json, file_sha256
from src.fairness_scalability_runner import (
    SCALABILITY_MANIFEST_SCHEMA_VERSION,
    ScalabilityDependencies,
    ScalabilityRunSpec,
    cumulative_run_plan,
    run_scalability_stage,
    stage_new_specs,
    validate_stage_decision,
)
from tests.test_robust_regional_fairness import tiny_instance


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/fairness_scalability_development_medium_large.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_exact_cumulative_and_new_task_counts() -> None:
    config = load_config()
    s1 = cumulative_run_plan(config, "s1")
    s2 = cumulative_run_plan(config, "s2")
    full = cumulative_run_plan(config, "full-grid", selected_candidate="single_cut")
    assert (len(s1), len(s2), len(full)) == (27, 90, 120)
    assert (len(stage_new_specs(s1, "s1")), len(stage_new_specs(s2, "s2")), len(stage_new_specs(full, "full-grid"))) == (27, 63, 30)
    for specs, baseline_count in ((s1, 3), (s2, 10), (full, 10)):
        baselines = [spec for spec in specs if spec.task_type == "baseline"]
        assert len(baselines) == baseline_count
        assert len({(spec.scale, spec.seed) for spec in baselines}) == baseline_count
        assert len({spec.run_key for spec in specs}) == len(specs)
        assert all(spec.introduced_stage in spec.run_key for spec in specs)


def test_full_grid_contains_only_one_selected_candidate_for_new_rhos() -> None:
    specs = cumulative_run_plan(load_config(), "full-grid", selected_candidate="persistent_separation")
    added = stage_new_specs(specs, "full-grid")
    assert {spec.candidate for spec in added} == {"persistent_separation"}
    assert {spec.rho for spec in added} == {0.025, 0.05, 0.10}


def test_s2_and_full_grid_decision_gates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="decision"):
        validate_stage_decision("s2", None)
    s1 = tmp_path / "s1.yaml"
    s1.write_text(
        yaml.safe_dump({
            "decision": "s1_pass",
            "next_authorized_stage": "s2",
            "mathematical_and_certification_correctness": True,
        }),
        encoding="utf-8",
    )
    assert validate_stage_decision("s2", s1)[1] == file_sha256(s1).upper()

    selected = tmp_path / "selected.yaml"
    selected.write_text(yaml.safe_dump({"selected_candidate": "single_cut"}), encoding="utf-8")
    decision = {
        "decision": "s2_candidate_selected",
        "next_authorized_stage": "full-grid",
        "mathematical_and_certification_correctness": True,
        "selected_candidate": "single_cut",
        "selected_candidate_config": str(selected),
        "selected_candidate_config_sha256": file_sha256(selected).upper(),
        "selection_order": [
            "mathematical_and_certification_correctness",
            "certified_solved_count_descending",
            "par2_ascending",
            "separation_runtime_ascending",
            "total_wall_runtime_ascending",
        ],
        "scale_results": {
            "medium_large": {"certified_solved_count": 16, "denominator": 20},
            "large": {"certified_solved_count": 16, "denominator": 20},
        },
    }
    path = tmp_path / "s2.yaml"
    path.write_text(yaml.safe_dump(decision), encoding="utf-8")
    assert validate_stage_decision("full-grid", path)[2] == "single_cut"
    decision["scale_results"]["large"]["certified_solved_count"] = 15
    path.write_text(yaml.safe_dump(decision), encoding="utf-8")
    with pytest.raises(ValueError, match="16/20"):
        validate_stage_decision("full-grid", path)


def synthetic_specs(config: dict) -> list[ScalabilityRunSpec]:
    scale = str(config["instance_sizes"][0])
    return [
        ScalabilityRunSpec("synthetic-baseline", "s1", "baseline", scale, 1, None, "joint_v1_core_point_strengthened"),
        ScalabilityRunSpec("synthetic-a", "s1", "frontier", scale, 1, 0.0, "single_cut"),
        ScalabilityRunSpec("synthetic-b", "s1", "frontier", scale, 1, 0.0, "persistent_separation"),
    ]


def synthetic_dependencies(calls: dict[str, list]) -> ScalabilityDependencies:
    def generate(*args, **kwargs):
        calls["generate"].append(kwargs["seed"])
        return tiny_instance()

    def baseline(config, instance, *, scale, seed):
        calls["baseline"].append((scale, seed))
        return {
            "status": "optimal",
            "valid_UB": True,
            "gap": 0.0,
            "upper_bound": 30.0,
            "runtime": 1.0,
            "iterations": 1,
        }

    def frontier(config, instance, *, anchor, rho, candidate):
        calls["frontier"].append((anchor, rho, candidate, object()))
        return {
            "status": "optimal",
            "gap": 0.0,
            "objective_t": 0.2,
            "y_values": [1.0],
            "x_values": [[5.0]],
            "runtime": 2.0,
            "iterations": 2,
        }

    def post(config, instance, **kwargs):
        calls["post"].append(kwargs["spec"].run_key)
        return {"valid": True, "objective_t_consistent": True}, {
            "post_evaluation_solver_runtime": 0.5,
            "post_evaluation_wall_runtime": 0.6,
            "aggregation_runtime": 0.1,
            "checkpoint_io_runtime": 0.05,
        }

    return ScalabilityDependencies(
        generate_instance=generate,
        solve_baseline=baseline,
        solve_frontier=frontier,
        post_evaluate=post,
        configure_solver=lambda settings: calls["solver_settings"].append(dict(settings)),
    )


def prepare_synthetic_run(monkeypatch, tmp_path: Path):
    config = deepcopy(load_config())
    config["output_dir"] = str(tmp_path / "new-container" / "output")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    specs = synthetic_specs(config)
    monkeypatch.setattr(runner, "cumulative_run_plan", lambda *args, **kwargs: specs)
    monkeypatch.setattr(runner, "stage_new_specs", lambda values, stage: list(values))
    calls = {key: [] for key in ("generate", "baseline", "frontier", "post", "solver_settings")}
    return config, config_path, specs, calls, synthetic_dependencies(calls)


def test_shared_baseline_anchor_candidate_isolation_and_outputs(monkeypatch, tmp_path: Path) -> None:
    config, config_path, specs, calls, deps = prepare_synthetic_run(monkeypatch, tmp_path)
    run_scalability_stage(config, config_path=config_path, stage="s1", resume=True, dependencies=deps)
    output = Path(config["output_dir"])
    assert calls["generate"] == [1]
    assert len(calls["baseline"]) == 1
    assert len(calls["frontier"]) == 2
    assert len({id(call[3]) for call in calls["frontier"]}) == 2
    records = [json.loads((output / "runs" / spec.run_key / "run.json").read_text(encoding="utf-8")) for spec in specs]
    frontier = [record for record in records if record["task_type"] == "frontier"]
    assert len({record["baseline_run_key"] for record in frontier}) == 1
    assert len({record["anchor_sha256"] for record in frontier}) == 1
    assert len({record["anchor_value_hex"] for record in frontier}) == 1
    assert all(record["scientific_status"] == "certified_robust_optimal" for record in records)
    assert all(
        record["result"].get("post_evaluation_runtime_excluded_from_algorithm_runtime") is True
        for record in frontier
    )
    assert all(
        record["result"]["penalized_runtime_par2"] == record["result"]["algorithm_runtime"]
        for record in frontier
    )
    assert calls["solver_settings"] == [{"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7}]
    manifest = json.loads((output / "scalability_development_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCALABILITY_MANIFEST_SCHEMA_VERSION
    assert manifest["completed_run_count"] == 3
    assert manifest["solved_run_count"] == 3
    assert manifest["post_evaluation"]["checkpoint_chunk_size"] == 25
    assert (output / "results.csv").is_file()
    assert (output / "summary.csv").is_file()
    assert (output / "run_manifest.json").is_file()
    assert (output / "audit_log.json").is_file()
    for record, spec in zip(records, specs):
        status = json.loads(
            (output / "runs" / spec.run_key / "status.json").read_text(encoding="utf-8")
        )
        assert status["state"] == record["state"] == "complete"
        assert status["scientific_status"] == record["scientific_status"]
    with (output / "results.csv").open(encoding="utf-8", newline="") as handle:
        result_rows = list(csv.DictReader(handle))
    assert len(result_rows) == 3
    assert len({row["run_key"] for row in result_rows}) == 3
    assert {row["scientific_status"] for row in result_rows} == {"certified_robust_optimal"}


def test_interrupt_after_atomic_run_and_resume_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    config, config_path, specs, calls, deps = prepare_synthetic_run(monkeypatch, tmp_path)
    injected = {"done": False}

    def interrupt(phase, spec):
        if phase == "after_run_record" and not injected["done"]:
            injected["done"] = True
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_scalability_stage(
            config, config_path=config_path, stage="s1", resume=True,
            dependencies=deps, failure_injector=interrupt,
        )
    assert len(calls["baseline"]) == 1
    assert len(calls["frontier"]) == 1
    run_scalability_stage(config, config_path=config_path, stage="s1", resume=True, dependencies=deps)
    assert len(calls["baseline"]) == 1
    assert len(calls["frontier"]) == 2
    before = (Path(config["output_dir"]) / "results.csv").read_bytes()
    run_scalability_stage(config, config_path=config_path, stage="s1", resume=True, dependencies=deps)
    assert len(calls["frontier"]) == 2
    assert (Path(config["output_dir"]) / "results.csv").read_bytes() == before


def test_resume_identity_and_corrupt_checkpoint_fail_closed(monkeypatch, tmp_path: Path) -> None:
    config, config_path, specs, calls, deps = prepare_synthetic_run(monkeypatch, tmp_path)
    run_scalability_stage(config, config_path=config_path, stage="s1", resume=True, dependencies=deps)
    drifted = deepcopy(config)
    drifted["gurobi_parameters"]["Seed"] = 1
    with pytest.raises(ValueError, match="Gurobi|identity"):
        run_scalability_stage(drifted, config_path=config_path, stage="s1", resume=True, dependencies=deps)

    output = Path(config["output_dir"])
    target = specs[1]
    (output / "runs" / target.run_key / "run.json").unlink()
    checkpoint = output / "runs" / target.run_key / "algorithm_checkpoint.json"
    checkpoint.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint is corrupt"):
        run_scalability_stage(config, config_path=config_path, stage="s1", resume=True, dependencies=deps)


def test_existing_empty_output_and_missing_resume_are_rejected(monkeypatch, tmp_path: Path) -> None:
    config, config_path, _specs, _calls, deps = prepare_synthetic_run(monkeypatch, tmp_path)
    Path(config["output_dir"]).mkdir(parents=True)
    with pytest.raises(ValueError, match="identity manifest"):
        run_scalability_stage(config, config_path=config_path, stage="s1", resume=True, dependencies=deps)
    with pytest.raises(ValueError, match="requires --resume"):
        run_scalability_stage(config, config_path=config_path, stage="s1", resume=False, dependencies=deps)


def test_s2_adds_only_new_seeds_and_does_not_rerun_s1(monkeypatch, tmp_path: Path) -> None:
    config, config_path, s1_specs, calls, deps = prepare_synthetic_run(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "stage_new_specs", lambda values, stage: [v for v in values if v.introduced_stage == stage])
    run_scalability_stage(config, config_path=config_path, stage="s1", resume=True, dependencies=deps)
    s2_specs = s1_specs + [
        ScalabilityRunSpec("synthetic-baseline-2", "s2", "baseline", "medium_large", 2, None, "joint_v1_core_point_strengthened"),
        ScalabilityRunSpec("synthetic-frontier-2", "s2", "frontier", "medium_large", 2, 0.0, "single_cut"),
    ]
    monkeypatch.setattr(runner, "cumulative_run_plan", lambda config, stage, **kwargs: s1_specs if stage == "s1" else s2_specs)
    decision = tmp_path / "s1-decision.yaml"
    decision.write_text(
        yaml.safe_dump({
            "decision": "s1_pass",
            "next_authorized_stage": "s2",
            "mathematical_and_certification_correctness": True,
        }),
        encoding="utf-8",
    )
    run_scalability_stage(
        config,
        config_path=config_path,
        stage="s2",
        resume=True,
        decision_path=decision,
        dependencies=deps,
    )
    assert calls["baseline"] == [("medium_large", 1), ("medium_large", 2)]
    assert [entry[2] for entry in calls["frontier"]] == [
        "single_cut", "persistent_separation", "single_cut"
    ]
    manifest = json.loads(
        (Path(config["output_dir"]) / "scalability_development_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["expected_run_count"] == 5
    assert manifest["new_run_count"] == 2
    assert manifest["completed_run_count"] == 5


def test_duplicate_plan_and_config_file_identity_drift_fail_closed(monkeypatch, tmp_path: Path) -> None:
    config, config_path, specs, calls, deps = prepare_synthetic_run(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "cumulative_run_plan", lambda *args, **kwargs: specs + [specs[-1]])
    with pytest.raises(ValueError, match="Duplicate scalability run key"):
        run_scalability_stage(config, config_path=config_path, stage="s1", resume=True, dependencies=deps)

    monkeypatch.setattr(runner, "cumulative_run_plan", lambda *args, **kwargs: specs)
    run_scalability_stage(config, config_path=config_path, stage="s1", resume=True, dependencies=deps)
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config_file_sha256"):
        run_scalability_stage(config, config_path=config_path, stage="s1", resume=True, dependencies=deps)


def test_uncertified_baseline_blocks_every_frontier_without_solving(monkeypatch, tmp_path: Path) -> None:
    config, config_path, specs, calls, deps = prepare_synthetic_run(monkeypatch, tmp_path)

    def uncertified(*args, **kwargs):
        calls["baseline"].append((kwargs["scale"], kwargs["seed"]))
        return {"status": "time_limit", "valid_UB": True, "gap": 0.2, "upper_bound": 30.0, "runtime": 1800.0}

    deps = ScalabilityDependencies(
        generate_instance=deps.generate_instance,
        solve_baseline=uncertified,
        solve_frontier=deps.solve_frontier,
        post_evaluate=deps.post_evaluate,
        configure_solver=deps.configure_solver,
    )
    run_scalability_stage(config, config_path=config_path, stage="s1", resume=True, dependencies=deps)
    assert calls["frontier"] == []
    records = [
        json.loads((Path(config["output_dir"]) / "runs" / spec.run_key / "run.json").read_text(encoding="utf-8"))
        for spec in specs if spec.task_type == "frontier"
    ]
    assert {record["algorithm_status"] for record in records} == {"baseline_uncertified"}
    assert {record["scientific_status"] for record in records} == {"implementation_error"}
