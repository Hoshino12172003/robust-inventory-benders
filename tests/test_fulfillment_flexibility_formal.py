from __future__ import annotations

from pathlib import Path
import json

import pytest
import yaml

from src.experiment_suite import INSTANCE_SIZES
from src.fulfillment_flexibility_formal_reporting import (
    bootstrap_mean_ci,
    exact_two_sided_sign_test,
    holm_adjust,
    _comparison_statistics,
    _validate_manifest_family,
    FormalReportingError,
)
from src.fulfillment_flexibility_formal_runner import (
    FALLBACK_SEEDS,
    FORMAL_SEEDS,
    FormalOptimizationProhibited,
    FormalProtocolError,
    _add_recourse,
    _base_model,
    build_eligibility,
    dry_run,
    assert_formal_execution_gate,
    execution_qualification_identity,
    execute_formal_config,
    full_mode_regression,
    load_config,
    model_size_estimate,
    protocol_identity,
    run_or_resume_certified_task,
    seed_nonreuse_audit,
    solve_cost_anchor,
    solve_service,
    task_matrix,
    validate_config,
    validate_execution_config,
    validate_protocol_config,
)
from src.instance import generate_instance
from src.scenarios import enumerate_budget_scenarios


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ROOT / "experiments/configs/fulfillment_flexibility_formal_medium_large.yaml",
    ROOT / "experiments/configs/fulfillment_flexibility_formal_large.yaml",
]


def _instance(seed: int = 0):
    return generate_instance(
        {
            "seed": seed,
            "instance": {
                **INSTANCE_SIZES["very_small"],
                "budget_factor": 0.68,
                "capacity_factor": 1.25,
            },
            "robust": {"gamma_target": 1},
        },
        seed=seed,
    )


def _settings() -> dict[str, float | int]:
    return {
        "time_limit": 120.0,
        "mip_gap": 0.0,
        "feasibility_tolerance": 1.0e-7,
        "threads": 1,
        "solver_seed": 0,
    }


def test_configs_freeze_matrix_parameters_and_gate() -> None:
    configs = [load_config(path) for path in CONFIGS]
    for config in configs:
        validate_config(config)
        assert config["formal_run_authorized"] is False
        assert config["seeds"] == list(FORMAL_SEEDS)
        assert config["fallback_seeds"] == list(FALLBACK_SEEDS)
        assert config["gamma"] == 2
        assert config["rho"] == 0.025
        assert config["analysis"]["independent_unit"] == "synthetic_seed_cluster"
        assert config["analysis"]["pooled_cluster_effect"] == "arithmetic_mean_across_scales"
    tasks = task_matrix(configs)
    assert len(tasks) == 240
    assert sum(task["scale"] == "medium_large" for task in tasks) == 120
    assert sum(task["scale"] == "large" for task in tasks) == 120


def test_seed_audit_proves_formal_nonreuse() -> None:
    audit = seed_nonreuse_audit(ROOT)
    assert audit["formal_seeds"] == list(FORMAL_SEEDS)
    assert audit["external_development_reservation"]["seeds"] == [190, 191, 192]
    assert audit["formal_seed_conflicts"] == {}
    assert audit["formal_seeds_untouched"] is True
    assert max(audit["main_structured_seed_values"]) == 189


def test_eligibility_and_arc_omission_are_exact() -> None:
    instance = _instance()
    k1 = build_eligibility(instance, "k1")
    k2 = build_eligibility(instance, "k2")
    full = build_eligibility(instance, "full")
    for region in instance.R:
        assert len(k1[region]) == 1
        assert len(k2[region]) == 2
        assert set(k1[region]).issubset(k2[region])
        assert set(k2[region]).issubset(full[region])
        ranked = sorted(
            instance.I,
            key=lambda i: (
                sum(instance.transport_cost[i][region][j] for j in instance.J)
                / instance.num_products,
                i,
            ),
        )
        assert k1[region] == tuple(ranked[:1])
        assert k2[region] == tuple(ranked[:2])

    scenario = enumerate_budget_scenarios(
        instance, 1, max_scenarios=5000, exact_scenarios=True
    )[0]
    model, _y, x, _first, gp = _base_model(
        instance, fixed_y=None, fixed_x=None, settings=_settings()
    )
    block = _add_recourse(model, instance, scenario, 0, x, k1, gp)
    expected = {
        (i, r, j)
        for r in instance.R for i in k1[r] for j in instance.J
    }
    assert set(block["q"].keys()) == expected
    model.dispose()


def test_fixed_first_stage_uses_identical_configuration_and_common_budget() -> None:
    instance = _instance(seed=2)
    scenarios = enumerate_budget_scenarios(
        instance, 1, max_scenarios=5000, exact_scenarios=True
    )
    anchor = solve_cost_anchor(instance, scenarios, "full", _settings())
    full = solve_service(
        instance, scenarios, "full", anchor["objective"], 0.025, _settings()
    )
    fixed_anchors = {
        mode: solve_cost_anchor(
            instance,
            scenarios,
            mode,
            _settings(),
            fixed_y=full["y_values"],
            fixed_x=full["x_values"],
        )
        for mode in ("k1", "k2", "full")
    }
    common = max(result["objective"] for result in fixed_anchors.values())
    fixed = {
        mode: solve_service(
            instance,
            scenarios,
            mode,
            common,
            0.025,
            _settings(),
            fixed_y=full["y_values"],
            fixed_x=full["x_values"],
        )
        for mode in ("k1", "k2", "full")
    }
    assert all(result["certified"] for result in fixed.values())
    assert len({result["cost_budget"] for result in fixed.values()}) == 1
    assert all(result["y_values"] == full["y_values"] for result in fixed.values())
    assert all(result["x_values"] == full["x_values"] for result in fixed.values())
    assert fixed["k1"]["objective_t"] + 1.0e-7 >= fixed["k2"]["objective_t"]
    assert fixed["k2"]["objective_t"] + 1.0e-7 >= fixed["full"]["objective_t"]


def test_full_mode_regression_matches_unrestricted_oracles() -> None:
    regression = full_mode_regression()
    assert regression["formal_seed_accessed"] is False
    assert regression["status"] == "pass"
    assert len(regression["cases"]) == 2
    for case in regression["cases"]:
        assert case["cost_objective_difference"] <= 1.0e-7
        assert case["T_difference"] <= 1.0e-7
        assert case["formal_first_stage_feasibility"]["feasible"]
        assert case["original_first_stage_feasibility"]["feasible"]


def test_static_dimensions_and_hardware_sensitive_large_full() -> None:
    medium, large = [load_config(path) for path in CONFIGS]
    assert model_size_estimate(medium, "full")["columns"] == 780049
    assert model_size_estimate(large, "full")["columns"] == 4060977
    assert model_size_estimate(large, "k1")["columns"] < model_size_estimate(large, "k2")["columns"]
    assert model_size_estimate(large, "k2")["columns"] < model_size_estimate(large, "full")["columns"]


def test_formal_gate_fails_before_instance_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("formal instance generation must not be reached")

    monkeypatch.setattr(
        "src.fulfillment_flexibility_formal_runner._formal_instance", forbidden
    )
    with pytest.raises(FormalOptimizationProhibited, match="FORMAL OPTIMIZATION PROHIBITED"):
        execute_formal_config(CONFIGS[0], ROOT)


def test_solver_free_dry_run_keeps_output_absent() -> None:
    output = ROOT / "experiments/results_fulfillment_flexibility/formal"
    assert not output.exists()
    result = dry_run(CONFIGS, ROOT)
    assert result["task_count"] == 240
    assert result["formal_solver_imported_or_called"] is False
    assert result["formal_output_exists"] is False
    assert len(result["gate_failures"]) == 2
    assert not output.exists()


def test_paired_statistics_are_exact_deterministic_and_untruncated() -> None:
    sign = exact_two_sided_sign_test([1.0] * 20)
    assert sign == {
        "wins": 20,
        "losses": 0,
        "ties": 0,
        "non_ties": 20,
        "p_value": pytest.approx(2 / (2 ** 20)),
    }
    assert bootstrap_mean_ci([0.1, 0.2, 0.3]) == bootstrap_mean_ci([0.1, 0.2, 0.3])
    adjusted = holm_adjust({"h2": 0.01, "h3": 0.04})
    assert adjusted == {"h2": pytest.approx(0.02), "h3": pytest.approx(0.04)}
    delta_12, delta_1f = 0.7, 0.5
    assert delta_12 / delta_1f == pytest.approx(1.4)


def test_protocol_uses_correct_managerial_boundary() -> None:
    protocol = (ROOT / "docs/fulfillment_flexibility_formal_protocol.md").read_text(
        encoding="utf-8"
    )
    assert "FORMAL OPTIMIZATION PROHIBITED" in protocol
    assert "restricted single-source eligibility benchmark" in protocol
    assert "limited pooling" in protocol
    assert "full fulfillment flexibility" in protocol
    assert "not the number of open warehouses" in protocol
    assert "nearest warehouse" in protocol
    assert "must never be rewritten" in protocol
    assert "ten seed clusters" in protocol
    assert "exploratory diagnostics" in protocol


def _future_authorized_config(tmp_path: Path) -> Path:
    config = load_config(CONFIGS[0])
    config["formal_run_authorized"] = True
    config["authorization_file"] = str(tmp_path / "authorization.json")
    config["execution_qualification_file"] = str(tmp_path / "qualification.json")
    path = tmp_path / "future_authorized.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_authorization_state_transition_is_structurally_satisfiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = load_config(CONFIGS[0])
    validate_protocol_config(protocol)
    with pytest.raises(FormalOptimizationProhibited):
        validate_execution_config(protocol)

    future = _future_authorized_config(tmp_path)
    validate_execution_config(load_config(future))
    authorization = {
        "formal_optimization_authorized": True,
        "identity": protocol_identity(future, ROOT),
    }
    qualification = {
        "reviewed": True,
        "qualification_type": "high_memory_hardware",
        "identity": execution_qualification_identity(future, ROOT),
        "qualification_status": "pass",
        "qualified_total_ram_bytes": 128 * 1024 ** 3,
        "qualified_free_disk_bytes": 300 * 1024 ** 3,
        "scientifically_usable": False,
        "paper_evidence": False,
        "formal_sample": False,
    }
    Path(load_config(future)["authorization_file"]).write_text(
        json.dumps(authorization), encoding="utf-8"
    )
    Path(load_config(future)["execution_qualification_file"]).write_text(
        json.dumps(qualification), encoding="utf-8"
    )
    monkeypatch.setattr(
        "src.fulfillment_flexibility_formal_runner.current_host_resources",
        lambda _root: {
            "total_ram_bytes": 128 * 1024 ** 3,
            "free_disk_bytes": 300 * 1024 ** 3,
        },
    )
    assert_formal_execution_gate(future, ROOT)


def test_authorization_and_qualification_identity_drift_fail_closed(tmp_path: Path) -> None:
    future = _future_authorized_config(tmp_path)
    identity = protocol_identity(future, ROOT)
    authorization_path = Path(load_config(future)["authorization_file"])
    qualification_path = Path(load_config(future)["execution_qualification_file"])
    authorization_path.write_text(json.dumps({
        "formal_optimization_authorized": True, "identity": {**identity, "gamma": 99}
    }), encoding="utf-8")
    qualification_path.write_text(json.dumps({
        "reviewed": True, "qualification_type": "high_memory_hardware",
        "identity": execution_qualification_identity(future, ROOT),
        "qualification_status": "pass",
        "qualified_total_ram_bytes": 128 * 1024 ** 3,
        "qualified_free_disk_bytes": 300 * 1024 ** 3,
        "scientifically_usable": False, "paper_evidence": False, "formal_sample": False,
    }), encoding="utf-8")
    with pytest.raises(FormalOptimizationProhibited, match="authorization identity drifted"):
        assert_formal_execution_gate(future, ROOT)


def test_current_host_cannot_pass_high_memory_execution_gate(tmp_path: Path) -> None:
    future = _future_authorized_config(tmp_path)
    config = load_config(future)
    Path(config["authorization_file"]).write_text(json.dumps({
        "formal_optimization_authorized": True,
        "identity": protocol_identity(future, ROOT),
    }), encoding="utf-8")
    Path(config["execution_qualification_file"]).write_text(json.dumps({
        "reviewed": True, "qualification_type": "high_memory_hardware",
        "identity": execution_qualification_identity(future, ROOT),
        "qualification_status": "pass",
        "qualified_total_ram_bytes": 128 * 1024 ** 3,
        "qualified_free_disk_bytes": 300 * 1024 ** 3,
        "scientifically_usable": False, "paper_evidence": False, "formal_sample": False,
    }), encoding="utf-8")
    with pytest.raises(FormalOptimizationProhibited, match="current execution host"):
        assert_formal_execution_gate(future, ROOT)


def test_atomic_certified_task_resume_and_fail_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    identity = {"seed": 0, "mode": "k1", "task_type": "test"}
    calls = []
    result, resumed = run_or_resume_certified_task(
        checkpoint, identity, lambda: calls.append(1) or {"certified": True, "status": "optimal"}
    )
    assert result["certified"] is True and resumed is False and calls == [1]
    result, resumed = run_or_resume_certified_task(
        checkpoint, identity, lambda: calls.append(2) or {"certified": True}
    )
    assert result["certified"] is True and resumed is True and calls == [1]
    with pytest.raises(FormalProtocolError, match="identity mismatch"):
        run_or_resume_certified_task(checkpoint, {**identity, "mode": "full"}, lambda: None)

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{", encoding="utf-8")
    with pytest.raises(FormalProtocolError, match="corrupted"):
        run_or_resume_certified_task(corrupt, identity, lambda: None)
    incomplete = tmp_path / ".interrupted.tmp"
    incomplete.write_text("partial", encoding="utf-8")
    fresh = tmp_path / "fresh.json"
    _, resumed = run_or_resume_certified_task(
        fresh, identity, lambda: {"certified": True, "status": "optimal"}
    )
    assert resumed is False and incomplete.exists()


def test_mixed_manifest_identity_fails_closed() -> None:
    common = {
        "source_commit": "a", "protocol_sha256": "b", "runner_sha256": "c",
        "reporting_sha256": "d", "eligibility_sha256": "e", "gamma": 2,
        "rho_hex": float(0.025).hex(), "formal_seeds": list(FORMAL_SEEDS),
    }
    manifests = {
        scale: {"identity": {**common}, "scale": scale, "seeds": list(FORMAL_SEEDS)}
        for scale in ("medium_large", "large")
    }
    manifests["large"]["identity"]["runner_sha256"] = "different"
    with pytest.raises(FormalReportingError, match="mixed formal manifest identity"):
        _validate_manifest_family(manifests)


def test_pooled_statistics_use_ten_seed_clusters() -> None:
    rows = []
    for seed in FORMAL_SEEDS:
        for scale, effect in (("medium_large", 0.1), ("large", 0.3)):
            rows.append({
                "seed": seed, "scale": scale, "comparison": "k1_vs_full",
                "absolute_T_reduction": effect,
                "relative_T_reduction": effect,
            })
    statistics = _comparison_statistics(rows, "k1_vs_full")
    assert statistics["pooled_independent_cluster_count"] == 10
    assert statistics["wins"] == 10
    assert statistics["mean_effect"] == pytest.approx(0.2)
    assert set(statistics["cluster_effects_by_seed"]) == {str(seed) for seed in FORMAL_SEEDS}


def test_eligibility_matches_frozen_development_reference_semantics() -> None:
    instance = _instance(seed=4)
    for mode, count in (("k1", 1), ("k2", 2), ("full", instance.num_warehouses)):
        reference = {}
        for region in instance.R:
            reference[region] = (
                tuple(instance.I)
                if mode == "full"
                else tuple(sorted(
                    instance.I,
                    key=lambda warehouse: (
                        sum(instance.transport_cost[warehouse][region][product] for product in instance.J)
                        / instance.num_products,
                        warehouse,
                    ),
                )[:count])
            )
        assert build_eligibility(instance, mode) == reference


def _all_arc_fixed_zero_oracle(instance, scenarios, mode: str, anchor: float | None = None):
    from gurobipy import GRB

    model, _y, x, first, gp = _base_model(
        instance, fixed_y=None, fixed_x=None, settings=_settings()
    )
    eligibility = build_eligibility(instance, mode)
    theta = None if anchor is not None else model.addVar(lb=0.0, name="theta")
    t = model.addVar(lb=0.0, ub=1.0, name="T") if anchor is not None else None
    for index, scenario in enumerate(scenarios):
        q = model.addVars(instance.I, instance.R, instance.J, lb=0.0, name=f"q_{index}")
        u = model.addVars(instance.R, instance.J, lb=0.0, name=f"u_{index}")
        e = model.addVars(instance.J, lb=0.0, name=f"e_{index}")
        for i in instance.I:
            for r in instance.R:
                if i not in eligibility[r]:
                    for j in instance.J:
                        q[i, r, j].UB = 0.0
        for r in instance.R:
            for j in instance.J:
                model.addConstr(gp.quicksum(q[i, r, j] for i in instance.I) + u[r, j]
                                >= scenario.demand[r][j])
        for i in instance.I:
            for j in instance.J:
                model.addConstr(gp.quicksum(q[i, r, j] for r in instance.R) <= x[i, j])
        for j in instance.J:
            model.addConstr(
                gp.quicksum(u[r, j] for r in instance.R) - e[j]
                <= (1.0 - instance.service_level[j])
                * sum(scenario.demand[r][j] for r in instance.R)
            )
        cost = (
            gp.quicksum(instance.transport_cost[i][r][j] * q[i, r, j]
                        for i in instance.I for r in instance.R for j in instance.J)
            + gp.quicksum(instance.shortage_penalty[r][j] * u[r, j]
                          for r in instance.R for j in instance.J)
            + gp.quicksum(instance.service_penalty[j] * e[j] for j in instance.J)
        )
        if anchor is None:
            model.addConstr(theta >= cost)
        else:
            model.addConstr(first + cost <= (1.0 + 0.025) * anchor)
            for r in instance.R:
                demand = sum(scenario.demand[r][j] for j in instance.J)
                model.addConstr(gp.quicksum(u[r, j] for j in instance.J) <= t * demand)
    model.Params.MIPGap = 0.0
    model.setObjective(first + theta if anchor is None else t, GRB.MINIMIZE)
    model.optimize()
    value = float(model.ObjVal)
    model.dispose()
    return value


def test_k1_k2_arc_omission_matches_all_arc_fixed_zero_oracle() -> None:
    instance = _instance(seed=5)
    scenarios = enumerate_budget_scenarios(instance, 1, max_scenarios=5000, exact_scenarios=True)
    for mode in ("k1", "k2"):
        anchor = solve_cost_anchor(instance, scenarios, mode, _settings())
        assert anchor["certified"] is True
        oracle_anchor = _all_arc_fixed_zero_oracle(instance, scenarios, mode)
        assert anchor["objective"] == pytest.approx(oracle_anchor, abs=1.0e-7)
        service = solve_service(
            instance, scenarios, mode, anchor["objective"], 0.025, _settings()
        )
        oracle_t = _all_arc_fixed_zero_oracle(instance, scenarios, mode, anchor["objective"])
        assert service["objective_t"] == pytest.approx(oracle_t, abs=1.0e-7)
