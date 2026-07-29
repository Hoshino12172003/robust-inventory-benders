from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import gurobipy as gp
from gurobipy import GRB
import pytest
import yaml

from src.experiment_protocol import config_sha256, file_sha256
from src.fairness_benders import _build_master
from src.fairness_hybrid_ccg_benders import (
    CANDIDATE_SHA256,
    _first_stage_expression,
    _checkpoint_hash,
    _incumbent_identity,
    _load_checkpoint,
    _write_checkpoint,
    add_complete_scenario_block,
    canonical_scenario_payload,
    initial_scenario_plan_identity,
    initial_scenarios,
    scenario_from_payload,
    scenario_sha256,
    select_one_new_scenario,
    solve_certified_hybrid_scenario_benders_fairness,
)
from src.fairness_hybrid_ccg_benders_d1_audit import audit
import src.fairness_hybrid_ccg_benders_runner as runner
from src.fairness_hybrid_ccg_benders_runner import HybridDependencies, dry_run, expand_d1_plan, run_d1
from src.robust_regional_fairness import fairness_cost_budget, solve_fairness_extensive_form
from src.scenarios import _scenario_from_units, enumerate_budget_scenarios
from tests.test_robust_regional_fairness import tiny_instance


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/fairness_hybrid_ccg_benders_d1.yaml"


def _solve_restricted(instance, scenarios, baseline_cost=50.0):
    model, _y, x, t = _build_master(instance, False)
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0
    first = _first_stage_expression(instance, _y, x)
    budget = fairness_cost_budget(baseline_cost, 0.0).budget
    for scenario in scenarios:
        digest = scenario_sha256(instance, scenario)
        add_complete_scenario_block(model, instance, scenario, x, t, first_stage=first, cost_budget=budget, scenario_sha=digest)
    model.optimize()
    result = (model.Status, None if model.SolCount == 0 else float(t.X), float(model.ObjBound))
    model.dispose()
    return result


@pytest.mark.parametrize("gamma", [0, 2])
def test_s0_complete_scenario_master_matches_extensive_form(gamma: int) -> None:
    instance = tiny_instance()
    scenarios = enumerate_budget_scenarios(instance, gamma, max_scenarios=20, exact_scenarios=True)
    status, objective, bound = _solve_restricted(instance, scenarios)
    extensive = solve_fairness_extensive_form(
        instance, baseline_cost=50.0, rho=0.0, gamma=gamma,
        max_scenarios=20, time_limit=30.0, mip_gap=0.0, lexicographic_cost_stage=False,
    )
    assert status == GRB.OPTIMAL
    assert extensive.status == "optimal"
    assert objective == pytest.approx(extensive.objective_t, abs=1e-7)
    assert bound == pytest.approx(extensive.objective_t, abs=1e-7)


def test_scenario_master_is_relaxation_and_lb_monotone() -> None:
    instance = tiny_instance()
    scenarios = enumerate_budget_scenarios(instance, 2, max_scenarios=20, exact_scenarios=True)
    _, restricted, restricted_bound = _solve_restricted(instance, scenarios[:1])
    _, full, full_bound = _solve_restricted(instance, scenarios)
    assert restricted <= full + 1e-8
    assert restricted_bound <= full_bound + 1e-8


def test_regression_scenario_block_strictly_strengthens_zero_cut_lb() -> None:
    instance = tiny_instance()
    model, y, x, t = _build_master(instance, False)
    model.Params.OutputFlag = 0
    model.optimize()
    pure_cut_lb = float(model.ObjBound)
    scenario = _scenario_from_units(instance, ())
    add_complete_scenario_block(
        model, instance, scenario, x, t, first_stage=_first_stage_expression(instance, y, x),
        cost_budget=30.0, scenario_sha=scenario_sha256(instance, scenario),
    )
    model.optimize()
    strengthened = float(model.ObjBound)
    model.dispose()
    assert pure_cut_lb == pytest.approx(0.0)
    assert strengthened > pure_cut_lb + 1e-6


def test_initial_scenarios_are_deterministic_and_regional() -> None:
    instance = tiny_instance(regions=2, products=1)
    first = initial_scenarios(instance, 2)
    second = initial_scenarios(instance, 2)
    assert [scenario_sha256(instance, value) for value in first] == [scenario_sha256(instance, value) for value in second]
    assert len(first) == 3
    assert first[0].active_units == ()
    assert first[1].active_units == ((0, 0),)
    assert first[2].active_units == ((1, 0),)


def test_scenario_canonical_round_trip_and_drift_rejection() -> None:
    instance = tiny_instance()
    scenario = _scenario_from_units(instance, ((1, 0),))
    payload = canonical_scenario_payload(instance, scenario)
    rebuilt = scenario_from_payload(payload)
    assert scenario_sha256(instance, rebuilt) == scenario_sha256(instance, scenario)
    bad = deepcopy(payload)
    bad["values"].append(1)
    with pytest.raises(Exception, match="component identity"):
        scenario_from_payload(bad)


def test_candidate_selection_uses_violation_then_scenario_then_cut_sha() -> None:
    instance = tiny_instance()
    cut_a = SimpleNamespace(active_deviations=[{"region": 0, "product": 0}])
    cut_b = SimpleNamespace(active_deviations=[{"region": 1, "product": 0}])
    candidates = [
        SimpleNamespace(cut=cut_b, normalized_violation_bucket=9, cut_sha256="B" * 64),
        SimpleNamespace(cut=cut_a, normalized_violation_bucket=10, cut_sha256="A" * 64),
    ]
    selected = select_one_new_scenario(instance, candidates, set())
    assert selected is not None and selected[0] is candidates[1]
    assert select_one_new_scenario(instance, candidates, {scenario_sha256(instance, selected[1])})[0] is candidates[0]


def test_duplicate_scenario_is_not_added_again() -> None:
    instance = tiny_instance()
    cut = SimpleNamespace(active_deviations=[{"region": 0, "product": 0}])
    candidate = SimpleNamespace(cut=cut, normalized_violation_bucket=1, cut_sha256="A" * 64)
    scenario = _scenario_from_units(instance, ((0, 0),))
    assert select_one_new_scenario(instance, [candidate], {scenario_sha256(instance, scenario)}) is None


@pytest.mark.parametrize("point", ["before", "after"])
def test_atomic_checkpoint_resume_preserves_scenario_order(tmp_path: Path, point: str) -> None:
    instance = tiny_instance()
    scenarios = initial_scenarios(instance, 2)
    payloads = {scenario_sha256(instance, value): canonical_scenario_payload(instance, value) for value in scenarios}
    order = list(payloads)
    state = {
        "iteration": 1, "committed_scenario_sha256_values": order,
        "scenario_payloads_by_sha256": payloads, "committed_farkas_cut_sha256_values": [],
        "cut_payloads_by_sha256": {}, "lower_bound": 0.2, "upper_bound": 1.0,
        "gap": 0.8, "master_solver_best_bound": 0.2, "best_y": [1.0],
        "best_x": [[5.0]], "incumbent_identity_sha256": _incumbent_identity([1.0], [[5.0]], 1.0),
        "final_certification_state": "not_certified", "iteration_log": [],
    }
    path = tmp_path / f"{point}.json"
    _write_checkpoint(path, {"run": "d1"}, state)
    assert _load_checkpoint(path, {"run": "d1"})["committed_scenario_sha256_values"] == order


def test_checkpoint_damage_and_identity_drift_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    _write_checkpoint(path, {"run": "d1"}, {"iteration": 0})
    with pytest.raises(Exception, match="identity"):
        _load_checkpoint(path, {"run": "other"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state"]["iteration"] = 9
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="hash"):
        _load_checkpoint(path, {"run": "d1"})


def test_d1_plan_and_dry_run_are_exact_and_side_effect_free() -> None:
    output = ROOT / "experiments/results_fairness_hybrid_ccg_benders/development_d1_large_seed160_rho0"
    assert not output.exists()
    rows = expand_d1_plan()
    report = dry_run(CONFIG)
    assert len(rows) == 2 and len({row["run_key"] for row in rows}) == 2
    assert report["total"] == 2 and report["seed"] == 160 and report["rho"] == 0.0
    assert report["initial_scenario_count"] == 13
    assert report["instances_generated"] is report["solver_called"] is False
    assert report["output_dir_exists"] is False and report["windows_path_check"] is True
    assert not output.exists()


@pytest.mark.parametrize("stage", ["L0", "L1", "M1", "D2", "S2", "full-grid"])
def test_non_d1_stage_is_rejected(stage: str) -> None:
    with pytest.raises(Exception, match="only D1"):
        runner.main(["--config", str(CONFIG), "--stage", stage, "--dry-run"])


def _fake_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = tmp_path / "new_d1_output"
    config["output_dir"] = str(output)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(runner, "OUTPUT_RELATIVE", str(output))
    monkeypatch.setattr(runner, "EXPECTED_CONFIG_SHA256", file_sha256(path).upper())
    return config, path


def test_fake_authorized_end_to_end_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config, path = _fake_config(tmp_path, monkeypatch)
    calls = {"generate": 0, "baseline": 0, "frontier": 0, "post": 0, "configure": 0}
    def generate(_config, seed):
        calls["generate"] += 1
        assert seed == 160
        return tiny_instance()
    def baseline(_config, _instance, _seed, params):
        calls["baseline"] += 1
        assert params == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}
        return {"status": "optimal", "valid_UB": True, "gap": 0.0, "upper_bound": 30.0,
                "runtime": 1.0, "best_y_values": [1.0], "best_x_values": [[5.0]]}
    def frontier(*_args):
        calls["frontier"] += 1
        return {"status": "optimal", "gap": 0.0, "lower_bound": 0.5, "upper_bound": 0.5,
                "objective_t": 0.5, "runtime": 2.0, "y_values": [1.0], "x_values": [[5.0]],
                "iteration_log": [{"master_status": "optimal", "robust_feasibility_certified": True,
                                   "separation_objective_bound": 0.0}],
                "metadata": {"full_separation_objective_bound_required": True, "robust_feasibility_certified": True}}
    def post(*_args):
        calls["post"] += 1
        return ({"valid": True, "scenario_count": 4657, "objective_t_consistent": True, "errors": []},
                {"post_evaluation_solver_runtime": 1.0, "post_evaluation_wall_runtime": 1.0,
                 "aggregation_runtime": 0.1, "checkpoint_io_runtime": 0.1})
    deps = HybridDependencies(generate, baseline, frontier, post, lambda params: calls.__setitem__("configure", calls["configure"] + 1))
    manifest = run_d1(path, resume=True, dependencies=deps, test_authorization=True)
    assert manifest["completed_run_count"] == 2 and manifest["certified_solved_count"] == 2
    assert calls == {"generate": 1, "baseline": 1, "frontier": 1, "post": 1, "configure": 1}
    run_d1(path, resume=True, dependencies=deps, test_authorization=True)
    assert calls["generate"] == 1 and calls["baseline"] == 1 and calls["frontier"] == 1 and calls["post"] == 1


def test_dependency_substitution_requires_test_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config, path = _fake_config(tmp_path, monkeypatch)
    deps = HybridDependencies(None, None, None, None, None)  # type: ignore[arg-type]
    with pytest.raises(Exception, match="test_authorization"):
        run_d1(path, resume=True, dependencies=deps)
    assert not Path(_config["output_dir"]).exists()


def test_baseline_checkpoint_resume_does_not_resolve_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config, path = _fake_config(tmp_path, monkeypatch)
    calls = {"baseline": 0}
    def baseline(_config, _instance, _seed, _params):
        calls["baseline"] += 1
        return {"status": "optimal", "valid_UB": True, "gap": 0.0, "upper_bound": 30.0,
                "runtime": 1.0, "best_y_values": [1.0], "best_x_values": [[5.0]]}
    deps = HybridDependencies(
        lambda _config, _seed: tiny_instance(), baseline,
        lambda *_args: {"status": "time_limit", "gap": 1.0, "lower_bound": 0.0, "upper_bound": 1.0,
                       "objective_t": 1.0, "runtime": 1.0, "y_values": [1.0], "x_values": [[5.0]],
                       "iteration_log": [], "metadata": {"full_separation_objective_bound_required": True}},
        lambda *_args: ({}, {}), lambda _params: None,
    )
    def interrupt(point, _payload):
        if point == "after_baseline_checkpoint":
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_d1(path, resume=True, dependencies=deps, test_authorization=True, failure_injector=interrupt)
    assert calls["baseline"] == 1
    run_d1(path, resume=True, dependencies=deps, test_authorization=True)
    assert calls["baseline"] == 1


def test_baseline_checkpoint_damage_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, path = _fake_config(tmp_path, monkeypatch)
    calls = {"baseline": 0}
    def baseline(_config, _instance, _seed, _params):
        calls["baseline"] += 1
        return {"status": "optimal", "valid_UB": True, "gap": 0.0, "upper_bound": 30.0,
                "runtime": 1.0, "best_y_values": [1.0], "best_x_values": [[5.0]]}
    deps = HybridDependencies(
        lambda _config, _seed: tiny_instance(), baseline,
        lambda *_args: {"status": "time_limit", "runtime": 1.0, "metadata": {}},
        lambda *_args: ({}, {}), lambda _params: None,
    )
    def interrupt(point, _payload):
        if point == "after_baseline_checkpoint":
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_d1(path, resume=True, dependencies=deps, test_authorization=True, failure_injector=interrupt)
    checkpoint = Path(config["output_dir"]) / "runs" / expand_d1_plan()[0]["run_directory_id"] / "baseline_checkpoint.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["result"]["upper_bound"] = 31.0
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="checkpoint hash"):
        run_d1(path, resume=True, dependencies=deps, test_authorization=True)
    assert calls["baseline"] == 1


def test_static_audit_passes_without_formal_side_effect() -> None:
    result = audit(ROOT)
    assert result["status"] == "pass"
    assert result["dry_run"]["solver_called"] is False


def test_old_t1_ub_default_candidate_identity_regressions() -> None:
    from tests.test_fairness_large_final_remediation_implementation import baseline_evidence, upper_bound_identity
    from src.fairness_large_final_remediation import construct_initial_t1_upper_bound
    instance = tiny_instance()
    record, anchor = baseline_evidence(instance, gamma=0)
    result = construct_initial_t1_upper_bound(
        instance, baseline_record=record, anchor=anchor, rho=0.0, tolerance=1e-7,
        expected_identity=upper_bound_identity(instance, record, anchor),
    )
    assert result.value == 1.0


def test_s0_hybrid_solver_reaches_exact_certification_and_resume(tmp_path: Path) -> None:
    from tests.test_fairness_large_final_remediation_implementation import baseline_evidence, upper_bound_identity
    instance = tiny_instance()
    record, anchor = baseline_evidence(instance, gamma=0)
    record["candidate_sha256"] = CANDIDATE_SHA256
    anchor["candidate_sha256"] = CANDIDATE_SHA256
    anchor["anchor_sha256"] = config_sha256({key: value for key, value in anchor.items() if key != "anchor_sha256"})
    expected = upper_bound_identity(instance, record, anchor)
    checkpoint = tmp_path / "hybrid.json"
    result = solve_certified_hybrid_scenario_benders_fairness(
        instance, baseline_record=record, anchor=anchor, expected_identity=expected,
        solver_parameters={"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7},
        rho=0.0, gamma=0, max_iterations=20, time_limit=30.0, tol=1e-6,
        checkpoint_path=checkpoint, checkpoint_identity={"run_key": "tiny-hybrid"},
    )
    assert result.status == "optimal"
    assert result.metadata["robust_feasibility_certified"] is True
    assert result.lower_bound == pytest.approx(result.upper_bound, abs=1e-6)
    certified_entries = [entry for entry in result.iteration_log if entry["robust_feasibility_certified"]]
    assert certified_entries and all(entry["final_exact_separation_performed"] for entry in certified_entries)
    resumed = solve_certified_hybrid_scenario_benders_fairness(
        instance, baseline_record=record, anchor=anchor, expected_identity=expected,
        solver_parameters={"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7},
        rho=0.0, gamma=0, max_iterations=20, time_limit=30.0, tol=1e-6,
        checkpoint_path=checkpoint, checkpoint_identity={"run_key": "tiny-hybrid"},
    )
    assert resumed.objective_t == pytest.approx(result.objective_t, abs=1e-8)
    assert resumed.metadata["committed_scenario_sha256_values"] == result.metadata["committed_scenario_sha256_values"]
    damaged = json.loads(checkpoint.read_text(encoding="utf-8"))
    damaged["state"]["best_y"][0] = 0.0
    damaged.pop("checkpoint_sha256")
    damaged["checkpoint_sha256"] = _checkpoint_hash(damaged)
    checkpoint.write_text(json.dumps(damaged), encoding="utf-8")
    with pytest.raises(Exception, match="incumbent identity"):
        solve_certified_hybrid_scenario_benders_fairness(
            instance, baseline_record=record, anchor=anchor, expected_identity=expected,
            solver_parameters={"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7},
            rho=0.0, gamma=0, max_iterations=20, time_limit=30.0, tol=1e-6,
            checkpoint_path=checkpoint, checkpoint_identity={"run_key": "tiny-hybrid"},
        )


@pytest.mark.parametrize("fault_point", ["before_scenario_commit", "after_scenario_commit_checkpoint"])
def test_s0_scenario_commit_interrupt_resumes_without_duplicate(
    tmp_path: Path, fault_point: str,
) -> None:
    from tests.test_fairness_large_final_remediation_implementation import baseline_evidence, upper_bound_identity
    instance = tiny_instance()
    record, anchor = baseline_evidence(instance, gamma=2)
    record["candidate_sha256"] = CANDIDATE_SHA256
    anchor["candidate_sha256"] = CANDIDATE_SHA256
    anchor["anchor_sha256"] = config_sha256({key: value for key, value in anchor.items() if key != "anchor_sha256"})
    expected = upper_bound_identity(instance, record, anchor)
    checkpoint = tmp_path / f"{fault_point}.json"
    raised = {"value": False}
    def interrupt(point, _payload):
        if point == fault_point and not raised["value"]:
            raised["value"] = True
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        solve_certified_hybrid_scenario_benders_fairness(
            instance, baseline_record=record, anchor=anchor, expected_identity=expected,
            solver_parameters={"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7},
            rho=0.0, gamma=2, max_iterations=20, time_limit=30.0, tol=1e-6,
            checkpoint_path=checkpoint, checkpoint_identity={"run_key": fault_point},
            failure_injector=interrupt,
        )
    resumed = solve_certified_hybrid_scenario_benders_fairness(
        instance, baseline_record=record, anchor=anchor, expected_identity=expected,
        solver_parameters={"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7},
        rho=0.0, gamma=2, max_iterations=20, time_limit=30.0, tol=1e-6,
        checkpoint_path=checkpoint, checkpoint_identity={"run_key": fault_point},
    )
    committed = resumed.metadata["committed_scenario_sha256_values"]
    assert resumed.status == "optimal"
    assert len(committed) == len(set(committed))
