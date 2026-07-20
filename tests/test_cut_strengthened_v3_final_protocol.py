from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

import src.experiment_suite as experiment_suite_module
from src.config import load_config
from src.cut_strengthened_v3_audit import FROZEN_CONFIG_SHA256
from src.cut_strengthened_v3_final_audit import (
    EXPECTED_FINAL_CONFIGS,
    EXPECTED_VARIANTS,
    LARGE_FINAL_SEEDS,
    MEDIUM_FINAL_SEEDS,
    _normalized_for_validation_comparison,
    audit_cut_strengthened_v3_final,
)
from src.experiment_protocol import file_sha256
from src.experiment_suite import (
    _apply_selected_parameters,
    experiment_dry_run_report,
    experiment_run_specs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments/configs"


def _write_yaml(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _audit_sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    config_dir = root / "experiments/configs"
    docs_dir = root / "docs"
    config_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    names = set(FROZEN_CONFIG_SHA256) | {
        "selected_cut_strengthened_joint_v3_candidate.yaml",
        "cut_strengthened_joint_v3_development_medium_large.yaml",
        "cut_strengthened_joint_v3_development_large.yaml",
        "cut_strengthened_joint_v3_validation_medium_large.yaml",
        "cut_strengthened_joint_v3_validation_large.yaml",
        *EXPECTED_FINAL_CONFIGS,
    }
    for name in names:
        shutil.copy2(CONFIG_DIR / name, config_dir / name)
    shutil.copy2(
        ROOT / "docs/cut_strengthened_joint_v3_validation_decision.md",
        docs_dir / "cut_strengthened_joint_v3_validation_decision.md",
    )
    shutil.copy2(
        ROOT / "docs/cut_strengthened_joint_v3_final_protocol.md",
        docs_dir / "cut_strengthened_joint_v3_final_protocol.md",
    )
    return root


@pytest.mark.parametrize("filename,expected", EXPECTED_FINAL_CONFIGS.items())
def test_final_configs_expand_to_twenty_static_runs(
    filename: str,
    expected: dict[str, object],
) -> None:
    raw = load_config(CONFIG_DIR / filename)
    resolved = _apply_selected_parameters(raw)
    specs = experiment_run_specs(resolved)

    assert len(specs) == 20
    assert sorted({spec.seed for spec in specs}) == expected["seeds"]
    assert {spec.instance_size for spec in specs} == {expected["instance_size"]}
    assert sorted({spec.variant_name for spec in specs}) == sorted(EXPECTED_VARIANTS)
    assert raw["protocol_phase"] == "final"
    assert raw["formal_inference_allowed"] is True
    assert file_sha256(CONFIG_DIR / filename).lower() == expected["sha256"]


@pytest.mark.parametrize("filename,expected", EXPECTED_FINAL_CONFIGS.items())
def test_final_diff_from_validation_is_limited_to_allowed_fields(
    filename: str,
    expected: dict[str, object],
) -> None:
    final = load_config(CONFIG_DIR / filename)
    validation = load_config(CONFIG_DIR / str(expected["validation"]))
    assert _normalized_for_validation_comparison(final) == (
        _normalized_for_validation_comparison(validation)
    )


@pytest.mark.parametrize("filename,expected", EXPECTED_FINAL_CONFIGS.items())
def test_final_dry_run_only_expands_plan(
    filename: str,
    expected: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_instance_generation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not generate final instances")

    monkeypatch.setattr(
        experiment_suite_module,
        "generate_instance",
        forbidden_instance_generation,
    )
    report = experiment_dry_run_report(load_config(CONFIG_DIR / filename))
    assert report["total_run_count"] == 20
    assert report["methods"] == EXPECTED_VARIANTS
    assert report["seeds"] == expected["seeds"]
    assert report["output_dir"] == expected["output_dir"]
    assert report["protocol_audit_errors"] == []
    assert report["automatic_parallelism_enabled"] is False
    assert not (ROOT / str(expected["output_dir"])).exists()


def test_normal_final_protocol_passes() -> None:
    report = audit_cut_strengthened_v3_final(ROOT)
    assert report["all_required_checks_passed"], report["failed_checks"]
    assert report["passed_check_count"] == report["required_check_count"]


def test_final_seed_out_of_range_fails(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    path = root / "experiments/configs/cut_strengthened_joint_v3_final_medium_large.yaml"
    config = load_config(path)
    config["random_seeds"][-1] = 110
    _write_yaml(path, config)

    report = audit_cut_strengthened_v3_final(root)
    assert not report["all_required_checks_passed"]
    assert "cut_strengthened_joint_v3_final_medium_large_seeds_exact" in report["failed_checks"]


def test_final_seed_groups_cannot_overlap(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    path = root / "experiments/configs/cut_strengthened_joint_v3_final_medium_large.yaml"
    config = load_config(path)
    config["random_seeds"] = LARGE_FINAL_SEEDS
    _write_yaml(path, config)

    report = audit_cut_strengthened_v3_final(root)
    assert not report["all_required_checks_passed"]
    assert "final_seed_groups_disjoint" in report["failed_checks"]


def test_development_or_validation_seed_use_fails(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    path = root / "experiments/configs/cut_strengthened_joint_v3_final_large.yaml"
    config = load_config(path)
    config["random_seeds"] = list(range(80, 90))
    _write_yaml(path, config)

    report = audit_cut_strengthened_v3_final(root)
    assert not report["all_required_checks_passed"]
    assert "cut_strengthened_joint_v3_final_large_pre_final_seeds_excluded" in report["failed_checks"]


def test_third_method_fails(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    path = root / "experiments/configs/cut_strengthened_joint_v3_final_large.yaml"
    config = load_config(path)
    config["variants"].append("proposed_cut_strengthened_joint_v3")
    config["variant_settings"]["proposed_cut_strengthened_joint_v3"] = {
        "cut_strengthening_policy": "core_point_stall_secondary",
        "max_cuts_per_iteration": 2,
    }
    _write_yaml(path, config)

    report = audit_cut_strengthened_v3_final(root)
    assert not report["all_required_checks_passed"]
    assert "cut_strengthened_joint_v3_final_large_only_v1_and_core_candidate" in report["failed_checks"]


def test_frozen_parameter_drift_fails(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    path = root / "experiments/configs/cut_strengthened_joint_v3_final_medium_large.yaml"
    config = load_config(path)
    config["core_point_update_weight"] = 0.60
    _write_yaml(path, config)

    report = audit_cut_strengthened_v3_final(root)
    assert not report["all_required_checks_passed"]
    assert "cut_strengthened_joint_v3_final_medium_large_validation_equivalence" in report["failed_checks"]


def test_output_directory_conflict_fails(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    config_dir = root / "experiments/configs"
    path = config_dir / "cut_strengthened_joint_v3_final_large.yaml"
    validation = load_config(
        config_dir / "cut_strengthened_joint_v3_validation_large.yaml"
    )
    config = load_config(path)
    config["output_dir"] = validation["output_dir"]
    _write_yaml(path, config)

    report = audit_cut_strengthened_v3_final(root)
    assert not report["all_required_checks_passed"]
    assert "cut_strengthened_joint_v3_final_large_isolated_output_directory" in report["failed_checks"]
    assert "final_outputs_and_resume_keys_isolated" in report["failed_checks"]


def test_candidate_sha_mismatch_fails(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    path = root / "experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml"
    candidate = load_config(path)
    candidate["algorithm"]["core_point_update_weight"] = 0.60
    _write_yaml(path, candidate)

    report = audit_cut_strengthened_v3_final(root)
    assert not report["all_required_checks_passed"]
    assert "selected_candidate_sha256_frozen" in report["failed_checks"]


def test_unexpected_final_result_file_fails(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    output = root / "experiments/results_cut_v3/final_medium_large"
    output.mkdir(parents=True)
    (output / "results.csv").write_text("not,formal,data\n", encoding="utf-8")

    report = audit_cut_strengthened_v3_final(root)
    assert not report["all_required_checks_passed"]
    assert "cut_strengthened_joint_v3_final_medium_large_no_instances_or_results_generated" in report["failed_checks"]


def test_final_seed_sets_are_exact_and_disjoint() -> None:
    assert MEDIUM_FINAL_SEEDS == list(range(90, 100))
    assert LARGE_FINAL_SEEDS == list(range(100, 110))
    assert set(MEDIUM_FINAL_SEEDS).isdisjoint(LARGE_FINAL_SEEDS)
