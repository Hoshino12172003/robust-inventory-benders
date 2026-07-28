from __future__ import annotations

from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys

import pytest
import yaml

import src.fairness_medium_large_final_holdout_runner as runner
from src.experiment_protocol import file_sha256
from src.fairness_medium_large_final_holdout_audit import static_audit
from src.fairness_medium_large_final_holdout_runner import (
    CANDIDATES,
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_FILE_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    HOLDOUT_SEEDS,
    HoldoutDependencies,
    RHOS,
    dry_run_report,
    final_holdout_run_plan,
    paired_statistics,
    run_directory_id,
    run_final_holdout,
    validate_production_baseline_payload,
    validate_runtime_config,
)
from tests.test_robust_regional_fairness import tiny_instance


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/fairness_medium_large_final_holdout.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_file_identities() -> None:
    config = load_config()
    assert file_sha256(CONFIG_PATH).upper() == EXPECTED_FILE_SHA256
    assert file_sha256(ROOT / config["protocol_document"]).upper() == EXPECTED_PROTOCOL_SHA256
    assert file_sha256(ROOT / config["candidate_parameters_must_be_fixed_from"]).upper() == EXPECTED_CANDIDATE_SHA256
    validate_runtime_config(config, config_path=CONFIG_PATH)


def test_exact_110_task_matrix_and_shared_baselines() -> None:
    specs = final_holdout_run_plan()
    assert len(specs) == 110
    assert len({spec.run_key for spec in specs}) == 110
    assert len({run_directory_id(spec.run_key) for spec in specs}) == 110
    assert {spec.seed for spec in specs} == set(HOLDOUT_SEEDS)
    assert {spec.rho for spec in specs if spec.rho is not None} == set(RHOS)
    assert {spec.candidate for spec in specs if spec.task_type == "frontier"} == set(CANDIDATES)
    for seed in HOLDOUT_SEEDS:
        assert sum(spec.seed == seed and spec.task_type == "baseline" for spec in specs) == 1
        assert sum(spec.seed == seed and spec.task_type == "frontier" for spec in specs) == 10


def test_solver_free_dry_run_has_no_side_effect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config()
    config["output_dir"] = str(tmp_path / "absent")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(runner, "EXPECTED_FILE_SHA256", file_sha256(path).upper())
    before = set(tmp_path.iterdir())
    report = dry_run_report(config, config_path=path)
    assert (report["baseline_count"], report["frontier_count"], report["total"]) == (10, 100, 110)
    assert report["unique_run_keys"] == 110
    assert report["duplicate_run_keys"] == 0
    assert report["scenario_count"] == 1831
    assert report["instances_generated"] is False
    assert report["solver_called"] is False
    assert report["output_dir_exists"] is False
    assert report["windows_portability_check"] is True
    assert set(tmp_path.iterdir()) == before
    assert static_audit(config, report)["status"] == "pass"


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("holdout_seeds", list(range(171, 181)), "seeds"),
        ("rho_grid", [0.0, 0.01], "rho"),
        ("candidates", ["single_cut"], "candidate"),
        ("formal_run_authorized", False, "formal_run_authorized"),
        ("execution_attempt", 2, "attempt"),
    ],
)
def test_identity_drift_fails_closed(field: str, value, match: str) -> None:
    config = load_config()
    config[field] = value
    with pytest.raises(ValueError, match=match):
        validate_runtime_config(config)


@pytest.mark.parametrize("evidence_name", ["large_attempt5_audit", "holdout_seed_access_audit"])
def test_prerequisite_evidence_identity_drift_fails_closed(evidence_name: str) -> None:
    config = load_config()
    config["prerequisite_evidence"][evidence_name]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="evidence SHA"):
        validate_runtime_config(config)


def test_production_baseline_payload_schema_is_strict() -> None:
    instance = tiny_instance()
    valid = {"best_y_values": [1.0], "best_x_values": [[2.0]]}
    evidence = validate_production_baseline_payload(instance, valid)
    assert evidence["warehouse_count"] == 1
    assert evidence["product_count"] == 1
    invalid = (
        {"best_y_values": {}, "best_x_values": [[2.0]]},
        {"best_y_values": [1.0, 2.0], "best_x_values": [[2.0]]},
        {"best_y_values": [True], "best_x_values": [[2.0]]},
        {"best_y_values": [1.0], "best_x_values": {"0": [2.0]}},
        {"best_y_values": [1.0], "best_x_values": [[2.0, 3.0]]},
        {"best_y_values": [1.0], "best_x_values": [[float("nan")]]},
        {"best_y_values": [1.0], "best_x_values": [[float("inf")]]},
        {"best_y_values": [1.0], "best_x_values": [["2.0"]]},
    )
    for payload in invalid:
        with pytest.raises(ValueError):
            validate_production_baseline_payload(instance, payload)


def fake_dependencies(calls: dict[str, list]) -> HoldoutDependencies:
    def generate(*args, **kwargs):
        calls["generate"].append(kwargs["seed"])
        return tiny_instance()

    def baseline(config, instance, *, scale, seed):
        calls["baseline"].append((scale, seed))
        return {
            "status": "optimal",
            "valid_UB": True,
            "gap": 0.0,
            "lower_bound": 29.0,
            "upper_bound": 30.0,
            "runtime": 1.0,
            "algorithm_runtime": 1.0,
            "iterations": 1,
            "cuts": 1,
            "best_y_values": [1.0],
            "best_x_values": [[5.0]],
        }

    def frontier(config, instance, *, anchor, rho, candidate):
        calls["frontier"].append((anchor, rho, candidate))
        return {
            "status": "optimal",
            "gap": 0.0,
            "lower_bound": 0.2,
            "upper_bound": 0.2,
            "objective_t": 0.2,
            "cost_budget": (1.0 + rho) * anchor,
            "y_values": [1.0],
            "x_values": [[5.0]],
            "runtime": 2.0,
            "algorithm_runtime": 2.0,
            "separation_runtime": 1.0,
            "master_runtime": 0.5,
            "iterations": 2,
            "cuts": 2,
            "metadata": {},
            "iteration_log": [],
        }

    def post(config, instance, **kwargs):
        calls["post"].append(kwargs["spec"].run_key)
        return {
            "valid": True,
            "scenario_count": 1831,
            "objective_t_consistent": True,
            "errors": [],
            "actual_robust_cost": 30.0,
            "realized_worst_shortage_rate": 0.2,
        }, {
            "post_evaluation_solver_runtime": 0.5,
            "post_evaluation_wall_runtime": 0.6,
            "aggregation_runtime": 0.1,
            "checkpoint_io_runtime": 0.05,
        }

    return HoldoutDependencies(
        generate_instance=generate,
        solve_baseline=baseline,
        solve_frontier=frontier,
        post_evaluate=post,
        configure_solver=lambda settings: calls["solver"].append(dict(settings)),
    )


def prepare_fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = load_config()
    config["output_dir"] = str(tmp_path / "container" / "holdout")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(runner, "EXPECTED_FILE_SHA256", file_sha256(path).upper())
    calls = {key: [] for key in ("generate", "baseline", "frontier", "post", "solver")}
    return config, path, calls, fake_dependencies(calls)


def test_fake_authorized_end_to_end_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, path, calls, deps = prepare_fake(tmp_path, monkeypatch)
    output = run_final_holdout(
        config, config_path=path, resume=True, dependencies=deps, test_authorization=True
    )
    assert calls["generate"] == list(HOLDOUT_SEEDS)
    assert len(calls["baseline"]) == 10
    assert len(calls["frontier"]) == 100
    assert len(calls["post"]) == 100
    assert calls["solver"] == [{"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7}]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed_run_count"] == 110
    assert manifest["certified_solved_count"] == 110
    assert manifest["previous_attempt_results_reused"] is False
    assert len(manifest["baseline_anchors"]) == 10
    with (output / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 110
    assert all(row["scientific_status"] == "certified_robust_optimal" for row in rows)
    with (output / "paired_comparison.csv").open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 50
    statistics_payload = json.loads((output / "paired_statistics.json").read_text(encoding="utf-8"))
    assert statistics_payload["independent_unit"] == "seed"
    assert statistics_payload["seed_rho_tasks_treated_as_independent"] is False
    before = {key: list(value) for key, value in calls.items()}
    run_final_holdout(
        config, config_path=path, resume=True, dependencies=deps, test_authorization=True
    )
    assert calls["generate"] == before["generate"]
    assert calls["baseline"] == before["baseline"]
    assert calls["frontier"] == before["frontier"]
    assert calls["post"] == before["post"]


def test_scientific_time_limit_continues_and_uses_par2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, path, calls, deps = prepare_fake(tmp_path, monkeypatch)
    original = deps.solve_frontier
    assert original is not None

    def frontier(config, instance, *, anchor, rho, candidate):
        payload = original(config, instance, anchor=anchor, rho=rho, candidate=candidate)
        if candidate == "single_cut":
            payload.update({"status": "time_limit", "gap": 1.0, "algorithm_runtime": 1800.0})
        return payload

    deps = HoldoutDependencies(
        generate_instance=deps.generate_instance,
        solve_baseline=deps.solve_baseline,
        solve_frontier=frontier,
        post_evaluate=deps.post_evaluate,
        configure_solver=deps.configure_solver,
    )
    output = run_final_holdout(
        config, config_path=path, resume=True, dependencies=deps, test_authorization=True
    )
    with (output / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    single = [row for row in rows if row["candidate"] == "single_cut"]
    batch = [row for row in rows if row["candidate"] == "persistent_certified_cache_batch5"]
    assert len(single) == len(batch) == 50
    assert all(row["scientific_status"] == "time_limit_uncertified" for row in single)
    assert all(float(row["penalized_runtime_par2"]) == 3600.0 for row in single)
    assert all(row["scientific_status"] == "certified_robust_optimal" for row in batch)
    assert len(calls["frontier"]) == 100


def test_corrupt_checkpoint_and_identity_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, path, calls, deps = prepare_fake(tmp_path, monkeypatch)
    injected = {"done": False}

    def interrupt(phase, spec):
        if phase == "before_frontier" and not injected["done"]:
            injected["done"] = True
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_final_holdout(
            config, config_path=path, resume=True, dependencies=deps,
            test_authorization=True, failure_injector=interrupt,
        )
    output = Path(config["output_dir"])
    target = next(spec for spec in final_holdout_run_plan() if spec.task_type == "frontier")
    checkpoint = output / "runs" / run_directory_id(target.run_key) / "algorithm_checkpoint.json"
    checkpoint.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint is corrupt"):
        run_final_holdout(
            config, config_path=path, resume=True, dependencies=deps, test_authorization=True
        )


def test_dependency_substitution_requires_explicit_test_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, path, _calls, deps = prepare_fake(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="test_authorization"):
        run_final_holdout(config, config_path=path, resume=True, dependencies=deps)
    assert not Path(config["output_dir"]).exists()


def test_seed_cluster_statistics_never_treats_50_tasks_as_independent() -> None:
    config = load_config()
    pairs = []
    for seed in HOLDOUT_SEEDS:
        for rho in RHOS:
            pairs.append(
                {
                    "seed": seed,
                    "rho": rho,
                    "paired_par2_difference_batch5_minus_single": float(seed - 175) + rho,
                    "single_cut_certified": True,
                    "batch5_certified": True,
                }
            )
    result = paired_statistics(pairs, config)
    assert result["status"] == "complete"
    assert result["independent_unit"] == "seed"
    assert result["overall"]["bootstrap"]["clusters"] == 10
    assert all(item["pair_count"] == 10 for item in result["per_rho"].values())
    assert result["multiple_testing_correction"] == "Holm"


def test_paired_statistics_rejects_duplicate_seed_rho_as_independent_observation() -> None:
    config = load_config()
    pairs = [
        {
            "seed": seed,
            "rho": rho,
            "paired_par2_difference_batch5_minus_single": 0.0,
            "single_cut_certified": False,
            "batch5_certified": False,
        }
        for seed in HOLDOUT_SEEDS
        for rho in RHOS
    ]
    pairs[-1] = dict(pairs[0])
    with pytest.raises(ValueError, match="exactly one pair per seed and rho"):
        paired_statistics(pairs, config)


def test_no_production_solver_or_frozen_candidate_files_changed() -> None:
    protected = (
        "src/benders.py",
        "src/scenarios.py",
        "src/fairness_benders.py",
        "src/fairness_scalability.py",
        "src/robust_regional_fairness.py",
        "experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml",
    )
    import subprocess

    completed = subprocess.run(
        ["git", "diff", "--quiet", "origin/main", "--", *protected], cwd=ROOT
    )
    assert completed.returncode == 0
