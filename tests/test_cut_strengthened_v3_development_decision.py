from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.cut_strengthened_v3_audit import (
    DEVELOPMENT_EXPERIMENT_COMMIT,
    LARGE_RESULTS_ZIP_SHA256,
    MEDIUM_RESULTS_ZIP_SHA256,
    RESERVED_LARGE_FINAL_SEEDS,
    RESERVED_MEDIUM_FINAL_SEEDS,
    RESERVED_VALIDATION_SEEDS,
    SELECTED_CANDIDATE_CONFIG_NAME,
    SELECTED_CANDIDATE_CONFIG_SHA256,
    audit_cut_strengthened_v3,
)
from src.experiment_protocol import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments/configs"
SELECTED_PATH = CONFIG_DIR / SELECTED_CANDIDATE_CONFIG_NAME
DECISION_PATH = ROOT / "docs/cut_strengthened_joint_v3_development_decision.md"


def test_selected_candidate_is_frozen_core_only_with_v1_precision() -> None:
    selected = load_config(SELECTED_PATH)
    algorithm = selected["algorithm"]
    components = selected["components"]

    assert selected["selected_variant"] == "joint_v1_core_point_strengthened"
    assert selected["development_experiment_commit"] == DEVELOPMENT_EXPERIMENT_COMMIT
    assert selected["selection_frozen"] is True
    assert selected["random_seeds"] == [75, 76, 77, 78, 79]
    assert selected["parameter_revision_used"] is False
    assert selected["validation_started"] is False
    assert selected["final_test_started"] is False

    assert algorithm["precision_policy"] == "joint_error_budget"
    assert algorithm["master_error_budget_ratio"] == pytest.approx(0.25)
    assert algorithm["subproblem_error_budget_ratio"] == pytest.approx(0.50)
    assert algorithm["cut_strengthening_policy"] == "core_point"
    assert algorithm["max_cuts_per_iteration"] == 1
    assert components["core_point_strengthening_enabled"] is True
    assert components["stall_secondary_enabled"] is False
    assert components["workload_aware_v2_enabled"] is False


def test_selected_candidate_freezes_core_and_disables_legacy_modules() -> None:
    selected = load_config(SELECTED_PATH)
    algorithm = selected["algorithm"]
    robust = selected["robust"]

    expected_core = {
        "core_point_update_weight": 0.50,
        "core_point_min_distance": 1.0e-9,
        "core_point_stage1_time_limit": 2.0,
        "core_point_stage2_time_limit": 2.0,
        "core_point_min_remaining_time": 10.0,
        "core_point_min_global_gap": 5.0e-4,
        "core_point_current_abs_tol": 1.0e-7,
        "core_point_current_rel_tol": 1.0e-8,
        "core_point_min_normalized_improvement": 1.0e-7,
    }
    for field, expected in expected_core.items():
        assert algorithm[field] == pytest.approx(expected)

    assert algorithm["cut_selection_enabled"] is False
    assert algorithm["adaptive_secondary_cut_selection_enabled"] is False
    assert algorithm["adaptive_secondary_generation_enabled"] is False
    assert algorithm["adaptive_subproblem_gap_enabled"] is False
    assert algorithm["adaptive_gap_enabled"] is False
    assert robust == {
        "gamma_target": 2,
        "gamma_schedule": [2],
        "gamma_continuation_enabled": False,
    }


def test_selected_candidate_evidence_and_file_hash_are_frozen() -> None:
    selected = load_config(SELECTED_PATH)
    evidence = selected["evidence"]

    assert file_sha256(SELECTED_PATH).lower() == SELECTED_CANDIDATE_CONFIG_SHA256
    assert evidence["medium_large_results_zip"]["sha256"] == MEDIUM_RESULTS_ZIP_SHA256
    assert evidence["large_results_zip"]["sha256"] == LARGE_RESULTS_ZIP_SHA256


def test_validation_seeds_are_isolated_and_final_seeds_remain_unused() -> None:
    validation_paths = list(
        CONFIG_DIR.glob("cut_strengthened_joint_v3_validation*.yaml")
    )
    assert len(validation_paths) == 2
    for path in validation_paths:
        assert set(load_config(path)["random_seeds"]) == RESERVED_VALIDATION_SEEDS

    non_validation_used: set[int] = set()
    for path in CONFIG_DIR.glob("*cut_strengthened_joint_v3*.yaml"):
        if path in validation_paths:
            continue
        non_validation_used.update(
            int(seed) for seed in load_config(path).get("random_seeds", [])
        )
    assert non_validation_used.isdisjoint(RESERVED_VALIDATION_SEEDS)
    assert non_validation_used.isdisjoint(
        RESERVED_MEDIUM_FINAL_SEEDS | RESERVED_LARGE_FINAL_SEEDS
    )
    assert not list(CONFIG_DIR.glob("*cut_strengthened_joint_v3*final*.yaml"))


def test_secondary_and_full_v3_remain_as_unselected_ablation_variants() -> None:
    development = load_config(
        CONFIG_DIR / "cut_strengthened_joint_v3_development_medium_large.yaml"
    )
    selected = load_config(SELECTED_PATH)

    assert "joint_v1_stall_secondary_cut" in development["variants"]
    assert "proposed_cut_strengthened_joint_v3" in development["variants"]
    assert selected["selected_variant"] not in {
        "joint_v1_stall_secondary_cut",
        "proposed_cut_strengthened_joint_v3",
    }


def test_decision_document_preserves_development_interpretation_boundary() -> None:
    document = DECISION_PATH.read_text(encoding="utf-8")

    assert DEVELOPMENT_EXPERIMENT_COMMIT in document
    assert MEDIUM_RESULTS_ZIP_SHA256 in document
    assert LARGE_RESULTS_ZIP_SHA256 in document
    assert "full V3 虽通过两个规模的门槛，但不被选为 validation 候选" in document
    assert "Development 阶段仅用于机制筛选和候选冻结，不进行正式统计推断" in document
    assert "Validation 尚未开始" in document
    assert "不称为严格 Pareto-optimal cut" in document


def test_extended_v3_audit_accepts_the_frozen_development_decision() -> None:
    report = audit_cut_strengthened_v3(ROOT)
    assert report["all_required_checks_passed"], report["failed_checks"]
    assert report["passed_check_count"] == report["required_check_count"]
