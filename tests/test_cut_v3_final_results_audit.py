from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest
import yaml

from src.cut_strengthened_v3_final_audit import EXPECTED_FINAL_ANALYSIS
from src.cut_v3_final_results_audit import (
    CORE_VARIANT,
    EXPECTED_VARIANTS,
    FINAL_COMMIT,
    V1_VARIANT,
    audit_cut_v3_final_results,
)
from src.experiment_protocol import config_sha256, file_sha256


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run_config(scale: str, seed: int, variant: str) -> dict[str, object]:
    dimensions = (6, 6, 10) if scale == "medium_large" else (8, 8, 12)
    return {
        "seed": seed,
        "adaptive_gap_enabled": False,
        "gamma_continuation_enabled": False,
        "instance": {
            "num_warehouses": dimensions[0],
            "num_products": dimensions[1],
            "num_regions": dimensions[2],
        },
        "robust": {"gamma_target": 2, "gamma_schedule": [2]},
        "algorithm": {
            "subproblem_mode": "robust_dual_milp",
            "cut_selection_enabled": False,
            "adaptive_secondary_cut_selection_enabled": False,
            "adaptive_secondary_generation_enabled": False,
            "adaptive_subproblem_gap_enabled": False,
            "final_certification_enabled": True,
            "precision_policy": "joint_error_budget",
            "master_gap_max": 0.02,
            "master_gap_min": 0.0001,
            "subproblem_gap_max": 0.05,
            "subproblem_gap_min": 0.0001,
            "master_error_budget_ratio": 0.25,
            "subproblem_error_budget_ratio": 0.50,
            "monotone_precision_tightening": True,
            "cut_strengthening_policy": "none" if variant == V1_VARIANT else "core_point",
            "max_cuts_per_iteration": 1,
            "core_point_update_weight": 0.50,
            "core_point_min_distance": 1.0e-9,
            "core_point_stage1_time_limit": 2.0,
            "core_point_stage2_time_limit": 2.0,
            "core_point_min_remaining_time": 10.0,
            "core_point_min_global_gap": 5.0e-4,
            "core_point_current_abs_tol": 1.0e-7,
            "core_point_current_rel_tol": 1.0e-8,
            "core_point_min_normalized_improvement": 1.0e-7,
        },
        "benders": {
            "max_iterations": 10000 if scale == "medium_large" else 20000,
            "tol": 1.0e-4,
            "time_limit": 600.0 if scale == "medium_large" else 1800.0,
        },
    }


def _build_scale(root: Path, scale: str) -> Path:
    result_root = root / scale
    result_root.mkdir(parents=True)
    experiment = f"cut_strengthened_joint_v3_final_{scale}"
    seeds = range(90, 100) if scale == "medium_large" else range(100, 110)
    limit = 600.0 if scale == "medium_large" else 1800.0
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for variant in EXPECTED_VARIANTS:
            runtime = 100.0 if variant == V1_VARIANT else 20.0
            iterations = 2 if variant == V1_VARIANT else 1
            run_key = f"{experiment}__none__none__{scale}__seed_{seed}__{variant}"
            log_name = f"{run_key}.csv"
            row: dict[str, object] = {
                "experiment_name": experiment,
                "instance_size": scale,
                "seed": seed,
                "variant_name": variant,
                "status": "optimal",
                "solved_to_tolerance": True,
                "objective": 100.005,
                "lower_bound": 100.0,
                "upper_bound": 100.005,
                "final_gap": (100.005 - 100.0) / 100.005,
                "runtime": runtime,
                "time_limit": limit,
                "penalized_runtime_par2": runtime,
                "iterations": iterations,
                "master_time": runtime * 0.4,
                "subproblem_time": runtime * 0.5,
                "valid_UB": True,
                "ub_uses_subproblem_bound": True,
                "core_point_attempt_count": 1 if variant == CORE_VARIANT else 0,
                "core_point_success_count": 1 if variant == CORE_VARIANT else 0,
                "core_point_total_runtime": 1.0 if variant == CORE_VARIANT else 0.0,
                "core_point_stage1_total_runtime": 0.5 if variant == CORE_VARIANT else 0.0,
                "core_point_stage2_total_runtime": 0.5 if variant == CORE_VARIANT else 0.0,
                "v3_secondary_trigger_count": 0,
                "v3_secondary_solve_count": 0,
                "v3_secondary_cut_added_count": 0,
                "v3_secondary_cuts_added": 0,
                "run_key": run_key,
                "iteration_log_path": f"results\\iteration_logs\\{log_name}",
                "git_commit": FINAL_COMMIT,
            }
            rows.append(row)
            log: list[dict[str, object]] = []
            for iteration in range(1, iterations + 1):
                lb = 99.0 if iteration < iterations else 100.0
                ub = 101.0 if iteration < iterations else 100.005
                gap = (ub - lb) / max(1.0, abs(ub))
                attempted = variant == CORE_VARIANT
                log.append(
                    {
                        "iteration": iteration,
                        "LB": lb,
                        "UB": ub,
                        "global_gap": gap,
                        "requested_master_mip_gap": 0.001 if iteration < iterations else 0.0001,
                        "subproblem_requested_mip_gap": 0.001 if iteration < iterations else 0.0001,
                        "target_robust_evaluation_used": True,
                        "subproblem_has_incumbent": True,
                        "cut_added": True,
                        "cut_rhs_current": 10.0,
                        "cuts_added_this_iteration": 1,
                        "final_certification_active": False,
                        "core_point_attempted": attempted,
                        "core_point_stage1_status": "optimal" if attempted else "not_attempted",
                        "core_point_stage2_status": "optimal" if attempted else "not_attempted",
                        "core_point_current_value_floor": 9.9 if attempted else "",
                        "core_point_dual_feasible": attempted,
                        "core_point_original_value_at_current": 10.0 if attempted else "",
                        "core_point_strengthened_value_at_current": 10.0 if attempted else "",
                        "core_point_cut_accepted": attempted,
                        "core_point_cut_fallback_reason": "",
                        "core_point_auxiliary_bound_used_for_UB": False,
                        "v3_secondary_attempted": False,
                        "v3_secondary_cut_added": False,
                        "v3_secondary_bound_used_for_UB": False,
                    }
                )
            _write_csv(result_root / "iteration_logs" / log_name, log)
            run_dir = result_root / "runs" / run_key
            run_dir.mkdir(parents=True)
            resolved = _run_config(scale, seed, variant)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_key": run_key,
                        "state": "complete",
                        "success": True,
                        "solved_to_tolerance": True,
                        "git_commit": FINAL_COMMIT,
                        "config_sha256": config_sha256(resolved),
                        "result": row,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "run_key": run_key,
                        "state": "complete",
                        "success": True,
                        "solved_to_tolerance": True,
                        "status": "optimal",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(resolved, sort_keys=False),
                encoding="utf-8",
            )
            (run_dir / "error.txt").write_text("", encoding="utf-8")
    _write_csv(result_root / "results.csv", rows)
    _write_csv(result_root / "summary.csv", [{"scale": scale, "runs": 20}])
    top = {
        "experiment_name": experiment,
        "random_seeds": list(seeds),
        "instance_sizes": [scale],
        "variants": list(EXPECTED_VARIANTS),
        "time_limit": limit,
        "max_iterations": 10000 if scale == "medium_large" else 20000,
        "tol": 1.0e-4,
        "gamma_target": 2,
        "gamma_schedule": [2],
        "formal_inference_allowed": True,
        "protocol_phase": "final",
        "candidate_config_sha256": "7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6",
        "final_analysis": EXPECTED_FINAL_ANALYSIS,
    }
    manifest = {
        "expected_run_count": 20,
        "completed_run_count": 20,
        "solved_run_count": 20,
        "failed_run_count": 0,
        "remaining_run_count": 0,
        "skipped_run_count": 0 if scale == "medium_large" else 4,
        "git_commit": FINAL_COMMIT,
        "config_sha256": config_sha256(top),
    }
    (result_root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (result_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(top, sort_keys=False), encoding="utf-8"
    )
    return result_root


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    return _build_scale(tmp_path, "medium_large"), _build_scale(tmp_path, "large")


def _audit(medium: Path, large: Path):
    return audit_cut_v3_final_results(
        medium,
        large,
        repo_root=ROOT,
        expected_archive_sha256={"medium_large": None, "large": None},
    )


def _zip_tree(source: Path, target: Path) -> Path:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, f"{source.name}/{path.relative_to(source).as_posix()}")
    return target


def _update_run_record(root: Path, row: dict[str, str]) -> None:
    path = root / "runs" / row["run_key"] / "run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["result"].update(row)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_normal_final_results_and_legal_resume_pass(tmp_path: Path) -> None:
    medium, large = _fixture(tmp_path)
    report = _audit(medium, large)
    assert report["all_required_checks_passed"], report["failed_checks"]
    assert report["decision"] == "final_confirmed"
    assert report["scales"]["large"]["observed_seeds"] == list(range(100, 110))
    assert report["auxiliary_bootstrap"]["valid"]


def test_zip_sha_mismatch_is_invalid(tmp_path: Path) -> None:
    medium, large = _fixture(tmp_path)
    medium_zip = _zip_tree(medium, tmp_path / "medium.zip")
    large_zip = _zip_tree(large, tmp_path / "large.zip")
    report = audit_cut_v3_final_results(
        medium_zip,
        large_zip,
        repo_root=ROOT,
        expected_archive_sha256={"medium_large": "0" * 64, "large": file_sha256(large_zip)},
    )
    assert report["decision"] == "invalid_run"
    assert "medium_large_archive_sha256" in report["failed_checks"]


def test_git_commit_mismatch_is_invalid(tmp_path: Path) -> None:
    medium, large = _fixture(tmp_path)
    rows = _read_csv(large / "results.csv")
    rows[0]["git_commit"] = "bad"
    _write_csv(large / "results.csv", rows)
    assert _audit(medium, large)["decision"] == "invalid_run"


def test_seed_replacement_or_overlap_is_invalid(tmp_path: Path) -> None:
    medium, large = _fixture(tmp_path)
    rows = _read_csv(large / "results.csv")
    rows[0]["seed"] = "99"
    _write_csv(large / "results.csv", rows)
    report = _audit(medium, large)
    assert report["decision"] == "invalid_run"
    assert "large_exact_final_seeds" in report["failed_checks"]
    assert "final_seed_groups_exact_and_disjoint" in report["failed_checks"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "third"])
def test_run_set_integrity_failures_are_invalid(tmp_path: Path, mutation: str) -> None:
    medium, large = _fixture(tmp_path)
    rows = _read_csv(large / "results.csv")
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    else:
        extra = dict(rows[0])
        extra["variant_name"] = "third_method"
        extra["run_key"] += "__third"
        rows.append(extra)
    _write_csv(large / "results.csv", rows)
    assert _audit(medium, large)["decision"] == "invalid_run"


def test_frozen_parameter_drift_is_invalid(tmp_path: Path) -> None:
    medium, large = _fixture(tmp_path)
    path = next(large.glob(f"runs/*{CORE_VARIANT}/resolved_config.yaml"))
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["algorithm"]["core_point_update_weight"] = 0.6
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    assert _audit(medium, large)["decision"] == "invalid_run"


@pytest.mark.parametrize("field,value", [("valid_UB", "False"), ("penalized_runtime_par2", "999")])
def test_invalid_bound_or_par2_is_invalid(tmp_path: Path, field: str, value: str) -> None:
    medium, large = _fixture(tmp_path)
    rows = _read_csv(medium / "results.csv")
    rows[0][field] = value
    _write_csv(medium / "results.csv", rows)
    assert _audit(medium, large)["decision"] == "invalid_run"


def test_core_auxiliary_bound_used_for_ub_is_invalid(tmp_path: Path) -> None:
    medium, large = _fixture(tmp_path)
    path = next(medium.glob(f"iteration_logs/*{CORE_VARIANT}.csv"))
    rows = _read_csv(path)
    rows[0]["core_point_auxiliary_bound_used_for_UB"] = "True"
    _write_csv(path, rows)
    assert _audit(medium, large)["decision"] == "invalid_run"


def test_discontinuous_iteration_log_is_invalid(tmp_path: Path) -> None:
    medium, large = _fixture(tmp_path)
    path = next(large.glob(f"iteration_logs/*{V1_VARIANT}.csv"))
    rows = _read_csv(path)
    rows[0]["iteration"] = "2"
    _write_csv(path, rows)
    assert _audit(medium, large)["decision"] == "invalid_run"


def test_interrupted_state_residue_is_invalid(tmp_path: Path) -> None:
    medium, large = _fixture(tmp_path)
    path = next(large.glob("runs/*/status.json"))
    status = json.loads(path.read_text(encoding="utf-8"))
    status["state"] = "interrupted"
    path.write_text(json.dumps(status), encoding="utf-8")
    assert _audit(medium, large)["decision"] == "invalid_run"


def test_skipped_resume_with_missing_result_is_invalid(tmp_path: Path) -> None:
    medium, large = _fixture(tmp_path)
    rows = _read_csv(large / "results.csv")[:-1]
    _write_csv(large / "results.csv", rows)
    assert _audit(medium, large)["decision"] == "invalid_run"


def test_valid_evidence_failing_performance_gate_is_not_confirmed(tmp_path: Path) -> None:
    medium, large = _fixture(tmp_path)
    rows = _read_csv(large / "results.csv")
    for row in rows:
        if row["variant_name"] == CORE_VARIANT:
            row["runtime"] = "110"
            row["penalized_runtime_par2"] = "110"
            _update_run_record(large, row)
    _write_csv(large / "results.csv", rows)
    report = _audit(medium, large)
    assert report["decision"] == "final_not_confirmed"
    assert not report["decision_gates"]["large_mean_par2_reduction_at_least_7_5_percent"]


def test_bootstrap_is_fixed_seed_reproducible(tmp_path: Path) -> None:
    medium, large = _fixture(tmp_path)
    first = _audit(medium, large)["auxiliary_bootstrap"]
    second = _audit(medium, large)["auxiliary_bootstrap"]
    assert first == second
    assert first["rng"] == "numpy.random.default_rng"
    assert first["quantile_method"] == "linear"
    assert first["resamples"] == 10_000
    assert first["analysis_random_seed"] == 20260720
