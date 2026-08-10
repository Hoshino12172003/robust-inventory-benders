from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys
import subprocess

import pytest
import src.fairness_high_gamma_external_solver_benchmark_runner as runner_module

from src.fairness_high_gamma_external_solver_benchmark import DirectExtensiveFormResult
from src.fairness_high_gamma_external_solver_benchmark_hg1_incident import (
    HG1_SHA256,
    audit_hg1,
    write_freeze,
)
from src.fairness_high_gamma_external_solver_benchmark_runner import (
    Dependencies,
    GAMMAS,
    HighGammaGateError,
    SEEDS,
    _baseline_method_config,
    _frontier_scientific,
    _git_gate,
    _par2,
    _solution_values,
    dry_run,
    expand_plan,
    formal_run,
    run_directory_id,
    scenario_count,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/fairness_high_gamma_external_solver_benchmark_attempt2.yaml"
HG1_ZIP = Path(r"E:\论文代码\fairness_high_gamma_external_solver_benchmark_hg1_results.zip")


def _instance(seed: int) -> dict:
    return {"name": f"small_{seed}", "num_warehouses": 4, "num_products": 4,
            "num_regions": 5, "base_demand": [[40.0] * 4 for _ in range(5)],
            "demand_deviation": [[10.0] * 4 for _ in range(5)]}


def fake_dependencies(calls: Counter[str]) -> Dependencies:
    def generate(config: dict, seed: int) -> dict:
        calls["instance"] += 1
        return _instance(seed)

    def baseline(config: dict, instance: dict, seed: int, solver: dict) -> dict:
        calls["baseline"] += 1
        gamma = config["gamma_value"]
        return {"status": "optimal", "valid_UB": True, "gap": 0.0,
                "lower_bound": 100.0 + gamma,
                "upper_bound": 100.0 + gamma, "runtime": 1.0,
                "best_x_values": [[1.0] * 4 for _ in range(4)], "best_y_values": [1.0] * 4,
                "gamma": gamma, "requested_gamma": gamma, "gamma_target": gamma,
                "active_gamma": gamma, "active_gamma_policy": "fixed_requested_gamma",
                "gamma_schedule": [gamma], "scenario_count": scenario_count(gamma),
                "max_scenarios": scenario_count(gamma),
                "instance_canonical_sha256": runner_module.sha256_value(instance)}

    def anchor(record: dict, common: dict, tolerance: float) -> dict:
        value = float(record["result"]["upper_bound"])
        payload = {"baseline_run_key": common["baseline_run_key"], "value": value,
                   "value_hex": value.hex(), "valid_UB": True, **common}
        return {**payload, "anchor_sha256": __import__("hashlib").sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest().upper()}

    def hybrid(config: dict, instance: dict, baseline_record: dict, anchor_value: dict,
               common: dict, checkpoint: Path, solver: dict, row: dict) -> dict:
        calls["hybrid"] += 1
        t = 0.1 + row["gamma"] * 0.001
        return {"status": "optimal", "runtime": 2.0, "lower_bound": t, "upper_bound": t,
                "gap": 0.0, "objective_t": t, "robust_minimum_fill_rate": 1 - t,
                "x_values": [[2.0] * 4 for _ in range(4)], "y_values": [1.0] * 4,
                "iterations": 2, "cuts": 1,
                "metadata": {"robust_feasibility_certified": True, "committed_scenario_count": 2,
                             "committed_scenario_sha256_values": ["s0", "s1"],
                             "committed_farkas_cut_sha256_values": ["c1"]},
                "iteration_log": [{"lower_bound": 0.0, "pool_candidate_count": 2,
                                   "cuts_per_iteration": 1, "duplicate_pattern_count": 0,
                                   "scenario_count": 1},
                                  {"lower_bound": t, "pool_candidate_count": 0,
                                   "cuts_per_iteration": 0, "duplicate_pattern_count": 0,
                                   "scenario_count": 2,
                                   "committed_scenario_sha256": "s1",
                                   "committed_farkas_cut_sha256": "c1",
                                   "final_exact_separation_performed": True,
                                   "robust_feasibility_certified": True,
                                   "separation_objective_bound": -0.0}], "cuts": 1}

    def direct(config: dict, instance: dict, baseline_record: dict,
               anchor_value: dict, row: dict) -> dict:
        calls["direct"] += 1
        t = 0.1 + row["gamma"] * 0.001
        return DirectExtensiveFormResult(
            status="optimal", scientific_model_status="complete_exact_model_optimal",
            complete_model_built=True, resource_failure=False, resource_failure_detail=None,
            objective_t=t, robust_minimum_fill_rate=1-t, lower_bound=t, upper_bound=t,
            incumbent=t, objective_bound=t,
            gap=0.0, mip_gap=0.0, y_values=[1.0] * 4, x_values=[[2.0] * 4 for _ in range(4)],
            baseline_cost=anchor_value["value"], rho=0.025,
            cost_budget=anchor_value["value"] * 1.025, scenario_count=scenario_count(row["gamma"]),
            model_build_runtime=1.0, optimize_runtime=2.0, algorithm_runtime=3.0,
            rows=100, columns=200, binaries=4, continuous_variables=196, nonzeros=500,
            solver_status_code=2, solver_status="optimal", node_count=1.0,
            simplex_iterations=10.0, benders_strategy=0).to_dict()

    def post(config: dict, instance: dict, result: dict, anchor_value: dict,
             identity: dict, root: Path, row: dict) -> tuple[dict, dict]:
        calls["post"] += 1
        return ({"valid": True, "errors": [], "objective_t_consistent": True,
                 "scenario_count": scenario_count(row["gamma"]),
                 "actual_robust_cost": anchor_value["value"] * 1.02,
                 "instance_canonical_sha256": identity["instance_canonical_sha256"]},
                {"post_evaluation_solver_runtime": 0.1, "post_evaluation_wall_runtime": 0.2,
                 "aggregation_runtime": 0.01, "checkpoint_io_runtime": 0.01})

    return Dependencies(generate, lambda value: value, lambda value: value, baseline, anchor,
                        hybrid, direct, post, lambda solver: calls.update(configure=1))


def test_matrix_and_scenario_counts() -> None:
    assert [scenario_count(gamma) for gamma in GAMMAS] == [211, 1351, 6196]
    rows = expand_plan()
    assert len(rows) == 45
    assert Counter(row["task_type"] for row in rows) == {
        "baseline": 15, "hybrid_frontier": 15, "direct_extensive_frontier": 15}
    assert len({row["run_key"] for row in rows}) == len({row["run_directory_id"] for row in rows}) == 45
    assert all(row["run_directory_id"] == run_directory_id(row["run_key"]) for row in rows)
    with pytest.raises(HighGammaGateError):
        scenario_count(1)
    with pytest.raises(HighGammaGateError):
        scenario_count(5)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_production_baseline_method_config_uses_requested_gamma(gamma: int) -> None:
    config = runner_module.load_yaml(CONFIG)
    method, resolved = _baseline_method_config({**config, "gamma_value": gamma}, 185)
    assert method == "adaptive_gap_gamma_benders"
    assert resolved["gamma"] == gamma
    assert resolved["gamma_continuation_enabled"] is False
    assert resolved["robust"]["gamma_target"] == gamma
    assert resolved["robust"]["gamma_schedule"] == [gamma]
    assert resolved["robust"]["max_scenarios"] == scenario_count(gamma)


def test_dry_run_has_no_side_effect_or_solver_import(tmp_path: Path) -> None:
    before = "gurobipy" in sys.modules
    report = dry_run(CONFIG, root_override=tmp_path)
    assert report["total"] == 45 and report["baselines"] == 15
    assert report["instances_generated"] is report["solver_called"] is False
    assert report["output_dir_exists"] is False
    assert report["gurobipy_imported"] is False
    assert ("gurobipy" in sys.modules) == before
    assert not any(tmp_path.iterdir())


def test_scientific_status_and_par2() -> None:
    timeout = {"status": "time_limit", "complete_model_built": True,
               "resource_failure": False, "lower_bound": 0.1, "upper_bound": 0.2, "gap": 0.1}
    assert _frontier_scientific("direct_extensive_frontier", timeout, None, 211, 1e-4) == "time_limit_uncertified"
    assert _par2("time_limit_uncertified", 2.0) == 3600.0
    crossing = {"status": "optimal", "complete_model_built": True,
                "resource_failure": False, "lower_bound": 0.2, "upper_bound": 0.1, "gap": 0.0}
    assert _frontier_scientific("direct_extensive_frontier", crossing,
                                {"valid": True, "errors": [], "objective_t_consistent": True,
                                 "scenario_count": 211}, 211, 1e-4) == "robust_uncertified"


def test_synthetic_45_task_disk_pipeline_and_zero_repeat_resume(tmp_path: Path) -> None:
    calls: Counter[str] = Counter()
    deps = fake_dependencies(calls)
    result = formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
                        test_authorization=True, test_root=tmp_path)
    assert result["completed_run_count"] == 45
    output = tmp_path / "experiments/results_fh_ext/hg2"
    assert len(list(output.glob("runs/*/run.json"))) == 45
    assert len(list(output.glob("runs/*/status.json"))) == 45
    assert len(list(output.glob("runs/*/algorithm_checkpoint.json"))) == 45
    assert calls["instance"] == calls["baseline"] == calls["hybrid"] == calls["direct"] == 15
    assert calls["post"] == 45
    first_results = (output / "results.csv").read_bytes()
    first_summary = (output / "summary.csv").read_bytes()
    calls.clear()
    formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
               test_authorization=True, test_root=tmp_path)
    assert calls == Counter({"configure": 1})
    assert (output / "results.csv").read_bytes() == first_results
    assert (output / "summary.csv").read_bytes() == first_summary
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for seed in SEEDS:
        for gamma in GAMMAS:
            cell = f"s{seed}_g{gamma}"
            expected = {"requested_gamma": gamma, "gamma_target": gamma,
                        "active_gamma": gamma, "gamma_schedule": [gamma],
                        "scenario_count": scenario_count(gamma)}
            for field, value in expected.items():
                assert manifest["baseline_identities"][cell][field] == value
                assert manifest["baseline_anchors"][cell][field] == value
                assert manifest["baseline_t1_lifting"][cell][field] == value
            paired = []
            for row in expand_plan():
                if row["seed"] == seed and row["gamma"] == gamma and row["task_type"] != "baseline":
                    paired.append(json.loads((output / "runs" / row["run_directory_id"] / "run.json").read_text(encoding="utf-8")))
            assert len(paired) == 2
            for field in ("instance_canonical_sha256", "baseline_run_key", "anchor_sha256",
                          "anchor_value_hex", "cost_budget"):
                assert paired[0][field] == paired[1][field]


def test_baseline_gamma_mismatch_blocks_frontiers(tmp_path: Path) -> None:
    calls: Counter[str] = Counter()
    original = fake_dependencies(calls)

    def mismatched(config, instance, seed, solver):
        result = original.solve_baseline(config, instance, seed, solver)
        result["active_gamma"] = 2 if config["gamma_value"] != 2 else 3
        return result

    deps = Dependencies(original.generate_instance, original.serialize_instance,
                        original.deserialize_instance, mismatched, original.make_anchor,
                        original.solve_hybrid, original.solve_direct, original.post_evaluate,
                        original.configure_solver)
    with pytest.raises(HighGammaGateError, match="Gamma identity mismatch"):
        formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
                   test_authorization=True, test_root=tmp_path)
    assert calls["hybrid"] == calls["direct"] == 0


def test_infeasible_t1_lifting_blocks_both_frontiers(tmp_path: Path) -> None:
    calls: Counter[str] = Counter()
    original = fake_dependencies(calls)

    def invalid_post(config, instance, result, anchor, identity, root, row):
        evaluation, timing = original.post_evaluate(
            config, instance, result, anchor, identity, root, row)
        if row["task_type"] == "baseline":
            evaluation["actual_robust_cost"] = anchor["value"] * 2.0
        return evaluation, timing

    deps = Dependencies(original.generate_instance, original.serialize_instance,
                        original.deserialize_instance, original.solve_baseline,
                        original.make_anchor, original.solve_hybrid, original.solve_direct,
                        invalid_post, original.configure_solver)
    with pytest.raises(HighGammaGateError, match="cost budget"):
        formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
                   test_authorization=True, test_root=tmp_path)
    assert calls["hybrid"] == calls["direct"] == 0


def test_hybrid_append_only_and_final_exact_are_required() -> None:
    calls: Counter[str] = Counter()
    deps = fake_dependencies(calls)
    row = next(item for item in expand_plan() if item["task_type"] == "hybrid_frontier")
    result = deps.solve_hybrid({}, _instance(185), {}, {}, {}, Path("checkpoint"), {}, row)
    post = {"valid": True, "errors": [], "objective_t_consistent": True,
            "scenario_count": scenario_count(row["gamma"])}
    assert _frontier_scientific("hybrid_frontier", result, post,
                                scenario_count(row["gamma"]), 1e-4) == "certified_robust_optimal"
    broken = json.loads(json.dumps(result))
    broken["iteration_log"][-1]["scenario_count"] = 0
    assert _frontier_scientific("hybrid_frontier", broken, post,
                                scenario_count(row["gamma"]), 1e-4) == "robust_uncertified"
    broken = json.loads(json.dumps(result))
    broken["iteration_log"][-1]["final_exact_separation_performed"] = False
    assert _frontier_scientific("hybrid_frontier", broken, post,
                                scenario_count(row["gamma"]), 1e-4) == "robust_uncertified"


def test_direct_result_schema_is_solver_free() -> None:
    assert "gurobipy" not in DirectExtensiveFormResult.__module__
    assert len(SEEDS) == 5


@pytest.mark.skipif(not HG1_ZIP.exists(), reason="external immutable HG1 archive unavailable")
def test_hg1_incident_freeze_is_read_only_and_deterministic(tmp_path: Path) -> None:
    before = __import__("hashlib").sha256(HG1_ZIP.read_bytes()).hexdigest().upper()
    audit, rows = audit_hg1(HG1_ZIP)
    assert before == HG1_SHA256 == audit["source_archive_sha256_after"]
    assert audit["baseline_internal_gamma_mismatch_count"] == 10
    assert audit["scientifically_usable_gamma2_subset"] is True
    assert audit["scientifically_usable_gamma3_4"] is False
    assert len(rows) == 15
    first, second = tmp_path / "first", tmp_path / "second"
    write_freeze(HG1_ZIP, first)
    write_freeze(HG1_ZIP, second)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()}
    assert __import__("hashlib").sha256(HG1_ZIP.read_bytes()).hexdigest().upper() == before


@pytest.mark.parametrize("mutation", [
    lambda value: value["x_values"].pop(),
    lambda value: value["x_values"][0].pop(),
    lambda value: value["x_values"][0].append(1.0),
    lambda value: value["x_values"][0].__setitem__(0, True),
    lambda value: value["x_values"][0].__setitem__(0, "1"),
    lambda value: value["x_values"][0].__setitem__(0, float("nan")),
    lambda value: value["y_values"].pop(),
    lambda value: value["y_values"].__setitem__(0, float("inf")),
])
def test_solution_projection_rejects_shape_type_and_nonfinite(mutation) -> None:
    value = {"x_values": [[1.0] * 4 for _ in range(4)], "y_values": [1.0] * 4}
    mutation(value)
    with pytest.raises(HighGammaGateError):
        _solution_values(value, "hybrid_frontier")


def test_manifest_and_checkpoint_damage_fail_closed(tmp_path: Path) -> None:
    deps = fake_dependencies(Counter())
    formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
               test_authorization=True, test_root=tmp_path)
    output = tmp_path / "experiments/results_fh_ext/hg2"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["identity"]["rho"] = 0.1
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HighGammaGateError, match="manifest mismatch"):
        formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
                   test_authorization=True, test_root=tmp_path)


def test_completed_algorithm_checkpoint_damage_fails_closed(tmp_path: Path) -> None:
    deps = fake_dependencies(Counter())
    formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
               test_authorization=True, test_root=tmp_path)
    checkpoint_path = next((tmp_path / "experiments/results_fh_ext/hg2").glob(
        "runs/*/algorithm_checkpoint.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["checkpoint_sha256"] = "0" * 64
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(HighGammaGateError, match="checkpoint corrupt"):
        formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
                   test_authorization=True, test_root=tmp_path)


def test_completed_baseline_lifting_tree_drift_fails_closed(tmp_path: Path) -> None:
    deps = fake_dependencies(Counter())
    formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
               test_authorization=True, test_root=tmp_path)
    output = tmp_path / "experiments/results_fh_ext/hg2"
    row = next(item for item in expand_plan() if item["task_type"] == "baseline")
    lifting_root = output / "runs" / row["run_directory_id"] / "baseline_t1_lifting"
    lifting_root.mkdir(parents=True, exist_ok=True)
    (lifting_root / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(HighGammaGateError, match="lifting checkpoint tree drift"):
        formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
                   test_authorization=True, test_root=tmp_path)


def test_interrupted_hybrid_uses_separate_internal_checkpoint_and_resumes(tmp_path: Path) -> None:
    calls: Counter[str] = Counter()
    original = fake_dependencies(calls)
    interrupted = {"done": False}

    def hybrid(*args):
        checkpoint = args[5]
        assert checkpoint.name == "hybrid_internal_checkpoint.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text('{"synthetic_internal":true}', encoding="utf-8")
        if not interrupted["done"]:
            interrupted["done"] = True
            raise KeyboardInterrupt
        return original.solve_hybrid(*args)

    deps = Dependencies(original.generate_instance, original.serialize_instance,
                        original.deserialize_instance, original.solve_baseline,
                        original.make_anchor, hybrid, original.solve_direct,
                        original.post_evaluate, original.configure_solver)
    with pytest.raises(KeyboardInterrupt):
        formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
                   test_authorization=True, test_root=tmp_path)
    output = tmp_path / "experiments/results_fh_ext/hg2"
    internal = next(output.glob("runs/*/hybrid_internal_checkpoint.json"))
    assert internal.exists() and not (internal.parent / "algorithm_checkpoint.json").exists()
    formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
               test_authorization=True, test_root=tmp_path)
    assert len(list(output.glob("runs/*/run.json"))) == 45
    assert len(list(output.glob("runs/*/algorithm_checkpoint.json"))) == 45


def test_formal_windows_path_has_margin() -> None:
    report = dry_run(CONFIG, root_override=Path(r"E:\rfext2"))
    assert report["longest_windows_path_length"] < 220


def test_post_evaluation_interruption_resumes_without_resolve(tmp_path: Path) -> None:
    calls: Counter[str] = Counter()
    original = fake_dependencies(calls)
    interrupted = {"done": False}

    def post(*args):
        calls["post"] += 1
        if not interrupted["done"]:
            interrupted["done"] = True
            raise KeyboardInterrupt
        calls["post"] -= 1
        return original.post_evaluate(*args)

    deps = Dependencies(original.generate_instance, original.serialize_instance,
                        original.deserialize_instance, original.solve_baseline,
                        original.make_anchor, original.solve_hybrid, original.solve_direct,
                        post, original.configure_solver)
    with pytest.raises(KeyboardInterrupt):
        formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
                   test_authorization=True, test_root=tmp_path)
    formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
               test_authorization=True, test_root=tmp_path)
    assert calls["baseline"] == calls["hybrid"] == calls["direct"] == 15


def test_aggregation_interruption_resumes_without_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Counter[str] = Counter()
    deps = fake_dependencies(calls)
    real = runner_module.aggregate
    interrupted = {"done": False}

    def aggregate_once(*args, **kwargs):
        if not interrupted["done"]:
            interrupted["done"] = True
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(runner_module, "aggregate", aggregate_once)
    with pytest.raises(KeyboardInterrupt):
        formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
                   test_authorization=True, test_root=tmp_path)
    formal_run(CONFIG, resume=True, authorization_file=None, dependencies=deps,
               test_authorization=True, test_root=tmp_path)
    assert calls["baseline"] == calls["hybrid"] == calls["direct"] == 15
    assert len(list((tmp_path / "experiments/results_fh_ext/hg2").glob("runs/*/run.json"))) == 45


def test_real_git_gate_ignores_only_frozen_output_root(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("/experiments/results_fh_ext/\n", encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True)
    output = tmp_path / "experiments/results_fh_ext/hg2"
    output.mkdir(parents=True)
    (output / "checkpoint.json").write_text("{}", encoding="utf-8")
    assert len(_git_gate(tmp_path, {"formal_worktree_root": str(tmp_path)})) == 40
    unrelated = tmp_path / "unrelated.tmp"
    unrelated.write_text("x", encoding="utf-8")
    with pytest.raises(HighGammaGateError, match="dirty"):
        _git_gate(tmp_path, {"formal_worktree_root": str(tmp_path)})
    unrelated.unlink()
    (tmp_path / "tracked.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(HighGammaGateError, match="dirty"):
        _git_gate(tmp_path, {"formal_worktree_root": str(tmp_path)})
