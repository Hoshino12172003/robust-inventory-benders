from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment_suite import INSTANCE_SIZES
from src.fulfillment_flexibility_formal_reporting import (
    bootstrap_mean_ci,
    exact_two_sided_sign_test,
    holm_adjust,
)
from src.fulfillment_flexibility_formal_runner import (
    FALLBACK_SEEDS,
    FORMAL_SEEDS,
    FormalOptimizationProhibited,
    _add_recourse,
    _base_model,
    build_eligibility,
    dry_run,
    execute_formal_config,
    full_mode_regression,
    load_config,
    model_size_estimate,
    seed_nonreuse_audit,
    solve_cost_anchor,
    solve_service,
    task_matrix,
    validate_config,
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
