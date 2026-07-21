from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import pytest
import yaml

import src.experiment_suite as experiment_suite_module
import src.regional_fairness_diagnostic as diagnostic_module
from src.config import load_config
from src.cut_strengthened_v3_audit import FROZEN_CONFIG_SHA256
from src.experiment_suite import experiment_dry_run_report
from src.instance import InventoryInstance
from src.regional_fairness_diagnostic import (
    classify_fairness_diagnostic,
    evaluate_fairness_diagnostic_instance,
    solve_default_and_fair_best_recourse,
    summarize_regional_service,
)
from src.regional_fairness_diagnostic_audit import (
    EXPECTED_CONFIGS,
    FROZEN_FINAL_FILES,
    FROZEN_MODEL_FILES,
    audit_regional_fairness_diagnostic,
)
from src.scenarios import (
    DemandScenario,
    ScenarioEnumerationResult,
    count_budget_scenarios,
    enumerate_budget_scenarios,
)
from src.subproblem import solve_recourse_subproblem


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments/configs"


def _tiny_instance(*, products: int = 1) -> InventoryInstance:
    base = [[8.0 for _ in range(products)] for _ in range(2)]
    return InventoryInstance(
        name="synthetic_fairness_unit_test",
        num_warehouses=2,
        num_products=products,
        num_regions=2,
        fixed_cost=[0.0, 0.0],
        inventory_cost=[[0.0] * products, [0.0] * products],
        capacity=[20.0 * products, 20.0 * products],
        volume=[1.0] * products,
        budget=100.0,
        transport_cost=[
            [[1.0] * products, [1.0] * products],
            [[1.0] * products, [1.0] * products],
        ],
        shortage_penalty=[[10.0] * products, [10.0] * products],
        service_penalty=[20.0] * products,
        service_level=[0.0] * products,
        base_demand=base,
        demand_deviation=[[1.0] * products, [1.0] * products],
        inventory_ub=[[20.0] * products, [20.0] * products],
    )


def _nominal(instance: InventoryInstance) -> DemandScenario:
    return DemandScenario(
        name="g0_base",
        active_units=(),
        demand=tuple(tuple(row) for row in instance.base_demand),
    )


def _summaries(default: list[float], fair: list[float]) -> list[dict[str, object]]:
    return [
        {"size": size, "seed": index, "default_WGap": default[index], "fair_best_WGap": fair[index]}
        for size in ("medium_large", "large")
        for index in range(10)
    ]


def _write_yaml(path: Path, value: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _audit_sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for relative in [
        *(f"experiments/configs/{name}" for name in FROZEN_CONFIG_SHA256),
        *FROZEN_FINAL_FILES,
        *FROZEN_MODEL_FILES,
        *(f"experiments/configs/{name}" for name in EXPECTED_CONFIGS),
        "docs/regional_fairness_diagnostic_protocol.md",
        "src/regional_fairness_diagnostic.py",
    ]:
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return root


def test_equal_service_has_zero_gap() -> None:
    metrics = summarize_regional_service([[10.0], [20.0]], [[2.0], [4.0]])
    assert metrics["fill_rate_gap"] == pytest.approx(0.0)
    assert metrics["weighted_mean_fill_rate"] == pytest.approx(0.8)


def test_one_region_shortage_and_multi_product_aggregation() -> None:
    metrics = summarize_regional_service(
        [[4.0, 6.0], [3.0, 7.0]],
        [[1.0, 1.0], [0.0, 0.0]],
    )
    assert metrics["regions"][0]["regional_demand"] == 10.0
    assert metrics["regions"][0]["regional_shortage"] == 2.0
    assert metrics["regions"][0]["fill_rate"] == pytest.approx(0.8)
    assert metrics["fill_rate_gap"] == pytest.approx(0.2)


def test_zero_demand_region_is_not_applicable() -> None:
    metrics = summarize_regional_service([[0.0], [10.0]], [[0.0], [2.0]])
    assert metrics["not_applicable_region_count"] == 1
    assert metrics["regions"][0]["fill_rate"] is None
    assert metrics["regions"][0]["not_applicable_reason"] == "zero_regional_demand"
    assert metrics["fill_rate_gap"] == pytest.approx(0.0)


def test_fair_best_reuses_x_respects_cost_cap_and_never_worsens_gap() -> None:
    instance = _tiny_instance()
    x_values = [[5.0], [5.0]]
    default, fair = solve_default_and_fair_best_recourse(instance, _nominal(instance), x_values)
    default_metrics = summarize_regional_service(instance.base_demand, default.shortage_values)
    fair_metrics = summarize_regional_service(instance.base_demand, fair.shortage_values)
    assert default.first_stage_x_sha256 == fair.first_stage_x_sha256
    assert fair.objective <= default.objective + float(fair.cost_tolerance) + 1.0e-9
    assert fair_metrics["fill_rate_gap"] <= default_metrics["fill_rate_gap"] + 1.0e-9
    assert default.constraints_satisfied and fair.constraints_satisfied


def test_default_recourse_cost_matches_existing_exact_recourse() -> None:
    instance = _tiny_instance()
    scenario = _nominal(instance)
    x_values = [[5.0], [5.0]]
    default, _fair = solve_default_and_fair_best_recourse(instance, scenario, x_values)
    existing = solve_recourse_subproblem(
        instance,
        scenario,
        {(i, j): x_values[i][j] for i in instance.I for j in instance.J},
    )
    assert default.objective == pytest.approx(existing.objective, abs=1.0e-7)
    assert default.original_cost_reproduced


def test_gamma_two_extreme_enumeration_and_scenario_roles() -> None:
    instance = _tiny_instance()
    assert count_budget_scenarios(instance, 2) == 4
    assert len(enumerate_budget_scenarios(instance, 2)) == 4
    result = evaluate_fairness_diagnostic_instance(
        instance,
        instance_size="synthetic",
        seed=1,
        best_x_values=[[5.0], [5.0]],
    )
    assert result.valid, result.errors
    assert result.scenario_count == 4
    assert result.audit["cost_worst_and_fairness_worst_separately_recorded"]
    assert result.instance_summary["cost_worst_scenario"]
    assert result.instance_summary["fairness_worst_scenario"]
    kinds = {row["scenario_kind"] for row in result.region_scenario_metrics}
    assert any("nominal" in kind for kind in kinds)
    assert any("cost_worst" in kind for kind in kinds)
    assert any("fairness_worst" in kind for kind in kinds)


def test_duplicate_scenario_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _tiny_instance()
    duplicate = _nominal(instance)
    monkeypatch.setattr(
        diagnostic_module,
        "enumerate_budget_scenarios_with_metadata",
        lambda *_args, **_kwargs: ScenarioEnumerationResult(
            scenarios=[duplicate, duplicate],
            scenario_mode="full",
            exact_scenarios=True,
            num_scenarios_used=2,
            num_scenarios_total_estimated=2,
            max_scenarios=5000,
        ),
    )
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_fairness_diagnostic_instance(
            instance,
            instance_size="synthetic",
            seed=1,
            best_x_values=[[5.0], [5.0]],
        )


def test_structural_threshold_boundary_and_correctness_gate() -> None:
    fair = [0.10] * 4 + [0.05] * 2 + [0.0] * 4
    report = classify_fairness_diagnostic(
        _summaries(fair, fair), correctness_checks_passed=True
    )
    assert report["decision"] == "structural_fairness_gap"
    invalid = classify_fairness_diagnostic(
        _summaries(fair, fair), correctness_checks_passed=False
    )
    assert invalid["decision"] == "fairness_diagnostic_invalid"
    assert invalid["next_authorized_stage"] == "none"


def test_recourse_degeneracy_only_classification() -> None:
    default = [0.10] * 4 + [0.05] * 2 + [0.0] * 4
    fair = [0.0] * 10
    report = classify_fairness_diagnostic(
        _summaries(default, fair), correctness_checks_passed=True
    )
    assert report["decision"] == "recourse_degeneracy_only"


def test_no_material_and_inconclusive_boundaries() -> None:
    no_material = classify_fairness_diagnostic(
        _summaries([0.02] * 10, [0.02] * 10), correctness_checks_passed=True
    )
    assert no_material["decision"] == "no_material_fairness_gap"
    at_boundary = classify_fairness_diagnostic(
        _summaries([0.03] * 10, [0.03] * 10), correctness_checks_passed=True
    )
    assert at_boundary["decision"] == "fairness_diagnostic_inconclusive"


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("seed_out_of_range", "regional_fairness_diagnostic_medium_large_seeds_exact_110_119"),
        ("second_algorithm", "regional_fairness_diagnostic_medium_large_only_frozen_core_candidate"),
        ("v3_parameter", "regional_fairness_diagnostic_medium_large_effective_candidate_parameters_frozen"),
        ("future_seed", "regional_fairness_diagnostic_medium_large_no_pre_diagnostic_or_future_seed_use"),
    ],
)
def test_static_audit_rejects_protocol_drift(
    tmp_path: Path,
    mutation: str,
    failed_check: str,
) -> None:
    root = _audit_sandbox(tmp_path)
    path = root / "experiments/configs/regional_fairness_diagnostic_medium_large.yaml"
    config = load_config(path)
    if mutation == "seed_out_of_range":
        config["random_seeds"][-1] = 109
    elif mutation == "second_algorithm":
        config["variants"].append("proposed_joint_rho025_050")
    elif mutation == "v3_parameter":
        config["core_point_update_weight"] = 0.6
    else:
        config["random_seeds"][-1] = 120
    _write_yaml(path, config)
    report = audit_regional_fairness_diagnostic(root)
    assert not report["all_required_checks_passed"]
    assert failed_check in report["failed_checks"]


def test_static_audit_rejects_output_conflict(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    (root / "experiments/results_fairness_diagnostic/medium_large").mkdir(parents=True)
    report = audit_regional_fairness_diagnostic(root)
    assert not report["all_required_checks_passed"]
    assert "regional_fairness_diagnostic_medium_large_isolated_absent_output" in report["failed_checks"]
    assert "formal_result_directories_absent" in report["failed_checks"]


@pytest.mark.parametrize(
    ("filename", "expected_count"),
    [(name, int(expected["scenario_count"])) for name, expected in EXPECTED_CONFIGS.items()],
)
def test_fairness_dry_run_is_static_and_does_not_generate_instances(
    filename: str,
    expected_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment_suite_module,
        "generate_instance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fairness dry-run must not generate instances")
        ),
    )
    config = load_config(CONFIG_DIR / filename)
    report = experiment_dry_run_report(config)
    assert report["total_run_count"] == 10
    assert report["methods"] == ["joint_v1_core_point_strengthened"]
    assert report["seeds"] == list(range(110, 120))
    assert list(report["scenario_count_by_size"].values()) == [expected_count]
    assert report["protocol_audit_errors"] == []
    assert not (ROOT / config["output_dir"]).exists()


def test_normal_static_audit_passes() -> None:
    report = audit_regional_fairness_diagnostic(ROOT)
    assert report["all_required_checks_passed"], report["failed_checks"]
    assert report["passed_check_count"] == report["required_check_count"]
