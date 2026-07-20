from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from src.config import load_config
from src.cut_strengthened_v3_audit import FROZEN_CONFIG_SHA256
from src.cut_strengthened_v3_validation_audit import (
    EXPECTED_VALIDATION_CONFIGS,
    EXPECTED_VARIANTS,
    FINAL_SEEDS,
    VALIDATION_SEEDS,
    audit_cut_strengthened_v3_validation,
)
from src.experiment_protocol import file_sha256
from src.experiment_suite import (
    _apply_selected_parameters,
    _apply_variant_config,
    _base_config,
    _variant_specs,
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
        *EXPECTED_VALIDATION_CONFIGS,
    }
    for name in names:
        shutil.copy2(CONFIG_DIR / name, config_dir / name)
    shutil.copy2(
        ROOT / "docs/cut_strengthened_joint_v3_validation_protocol.md",
        docs_dir / "cut_strengthened_joint_v3_validation_protocol.md",
    )
    return root


@pytest.mark.parametrize("filename,expected", EXPECTED_VALIDATION_CONFIGS.items())
def test_validation_configs_expand_to_twenty_paired_runs(
    filename: str,
    expected: dict[str, object],
) -> None:
    raw = load_config(CONFIG_DIR / filename)
    resolved = _apply_selected_parameters(raw)
    specs = experiment_run_specs(resolved)

    assert len(specs) == 20
    assert sorted({spec.seed for spec in specs}) == VALIDATION_SEEDS
    assert {spec.seed for spec in specs}.isdisjoint(FINAL_SEEDS)
    assert {spec.instance_size for spec in specs} == {expected["instance_size"]}
    assert [name for name, _method, _settings in _variant_specs(resolved)] == EXPECTED_VARIANTS

    effective = {}
    base = _base_config(resolved, str(expected["instance_size"]), 80)
    for name, method, settings in _variant_specs(resolved):
        _solver, _flags, run = _apply_variant_config(base, method, settings)
        effective[name] = run["algorithm"]
    assert effective["proposed_joint_rho025_050"]["cut_strengthening_policy"] == "none"
    candidate = effective["joint_v1_core_point_strengthened"]
    assert candidate["cut_strengthening_policy"] == "core_point"
    assert candidate["precision_policy"] == "joint_error_budget"
    assert candidate["master_error_budget_ratio"] == pytest.approx(0.25)
    assert candidate["subproblem_error_budget_ratio"] == pytest.approx(0.50)
    assert candidate["max_cuts_per_iteration"] == 1
    assert candidate["adaptive_secondary_generation_enabled"] is False


@pytest.mark.parametrize("filename,expected", EXPECTED_VALIDATION_CONFIGS.items())
def test_validation_dry_run_is_audited_without_solver_calls(
    filename: str,
    expected: dict[str, object],
) -> None:
    report = experiment_dry_run_report(load_config(CONFIG_DIR / filename))
    assert report["total_run_count"] == 20
    assert report["methods"] == EXPECTED_VARIANTS
    assert report["seeds"] == VALIDATION_SEEDS
    assert report["output_dir"] == expected["output_dir"]
    assert report["automatic_parallelism_enabled"] is False
    assert report["protocol_audit_errors"] == []


def test_normal_validation_protocol_and_hashes_pass() -> None:
    report = audit_cut_strengthened_v3_validation(ROOT)
    assert report["all_required_checks_passed"], report["failed_checks"]
    for filename, expected in EXPECTED_VALIDATION_CONFIGS.items():
        assert file_sha256(CONFIG_DIR / filename).lower() == expected["sha256"]


def test_seed_out_of_range_fails_audit(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    path = root / "experiments/configs/cut_strengthened_joint_v3_validation_large.yaml"
    config = load_config(path)
    config["random_seeds"][-1] = 90
    _write_yaml(path, config)

    report = audit_cut_strengthened_v3_validation(root)
    assert not report["all_required_checks_passed"]
    assert "cut_strengthened_joint_v3_validation_large_seeds_exact_80_89" in report["failed_checks"]
    assert "cut_strengthened_joint_v3_validation_large_final_seeds_excluded" in report["failed_checks"]


def test_adding_third_candidate_fails_audit(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    path = root / "experiments/configs/cut_strengthened_joint_v3_validation_medium_large.yaml"
    config = load_config(path)
    config["variants"].append("proposed_cut_strengthened_joint_v3")
    config["variant_settings"]["proposed_cut_strengthened_joint_v3"] = {
        "cut_strengthening_policy": "core_point_stall_secondary",
        "max_cuts_per_iteration": 2,
    }
    _write_yaml(path, config)

    report = audit_cut_strengthened_v3_validation(root)
    assert not report["all_required_checks_passed"]
    assert "cut_strengthened_joint_v3_validation_medium_large_only_v1_and_core_candidate" in report["failed_checks"]


def test_modifying_frozen_candidate_parameter_fails_audit(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    path = root / "experiments/configs/cut_strengthened_joint_v3_validation_large.yaml"
    config = load_config(path)
    config["core_point_update_weight"] = 0.60
    _write_yaml(path, config)

    report = audit_cut_strengthened_v3_validation(root)
    assert not report["all_required_checks_passed"]
    assert "cut_strengthened_joint_v3_validation_large_development_equivalence" in report["failed_checks"]


def test_modifying_frozen_v1_config_fails_audit(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    path = root / "experiments/configs/selected_algorithm_parameters.yaml"
    config = load_config(path)
    config["master_error_budget_ratio"] = 0.30
    _write_yaml(path, config)

    report = audit_cut_strengthened_v3_validation(root)
    assert not report["all_required_checks_passed"]
    assert "frozen_selected_algorithm_parameters.yaml_unchanged" in report["failed_checks"]


def test_output_directory_conflict_fails_audit(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    config_dir = root / "experiments/configs"
    validation_path = config_dir / "cut_strengthened_joint_v3_validation_large.yaml"
    development = load_config(
        config_dir / "cut_strengthened_joint_v3_development_large.yaml"
    )
    validation = load_config(validation_path)
    validation["output_dir"] = development["output_dir"]
    _write_yaml(validation_path, validation)

    report = audit_cut_strengthened_v3_validation(root)
    assert not report["all_required_checks_passed"]
    assert "cut_strengthened_joint_v3_validation_large_isolated_output_directory" in report["failed_checks"]
    assert "resume_outputs_and_run_keys_isolated_from_development" in report["failed_checks"]


def test_candidate_sha_mismatch_fails_audit_and_runtime_lock(tmp_path: Path) -> None:
    root = _audit_sandbox(tmp_path)
    candidate_path = root / "experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml"
    candidate = load_config(candidate_path)
    candidate["algorithm"]["core_point_update_weight"] = 0.60
    _write_yaml(candidate_path, candidate)

    report = audit_cut_strengthened_v3_validation(root)
    assert not report["all_required_checks_passed"]
    assert "selected_candidate_sha256_frozen" in report["failed_checks"]

    config = load_config(
        CONFIG_DIR / "cut_strengthened_joint_v3_validation_medium_large.yaml"
    )
    config["candidate_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Frozen V3 candidate SHA256 mismatch"):
        _apply_selected_parameters(config)
