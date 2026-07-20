from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import yaml

from src.cut_v3_validation_results_audit import (
    CORE_VARIANT,
    EXPECTED_VARIANTS,
    V1_VARIANT,
    audit_cut_v3_validation_results,
)
from src.experiment_protocol import file_sha256


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
    limit = 600.0 if scale == "medium_large" else 1800.0
    iterations = 10000 if scale == "medium_large" else 20000
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
            "max_iterations": iterations,
            "tol": 1.0e-4,
            "time_limit": limit,
        },
    }


def _build_scale(root: Path, scale: str) -> Path:
    result_root = root / scale
    result_root.mkdir(parents=True)
    experiment = f"cut_strengthened_joint_v3_validation_{scale}"
    limit = 600.0 if scale == "medium_large" else 1800.0
    max_iterations = 10000 if scale == "medium_large" else 20000
    result_rows: list[dict[str, object]] = []
    solved_total = 0
    for seed in range(80, 90):
        for variant in EXPECTED_VARIANTS:
            unsolved = scale == "large" and variant == V1_VARIANT and seed in {88, 89}
            solved = not unsolved
            solved_total += int(solved)
            runtime = limit if unsolved else (100.0 if variant == V1_VARIANT else 20.0)
            par2 = 2.0 * limit if unsolved else runtime
            lb = 90.0 if unsolved else 100.0
            ub = 100.0 if unsolved else 100.005
            gap = max(0.0, (ub - lb) / max(1.0, abs(ub)))
            iterations = 100 if variant == V1_VARIANT else 50
            run_key = f"{experiment}__none__none__{scale}__seed_{seed}__{variant}"
            log_name = f"{run_key}.csv"
            row = {
                "experiment_name": experiment,
                "instance_size": scale,
                "seed": seed,
                "variant_name": variant,
                "status": "optimal" if solved else "time_limit",
                "solved_to_tolerance": solved,
                "objective": ub,
                "lower_bound": lb,
                "upper_bound": ub,
                "final_gap": gap,
                "runtime": runtime,
                "time_limit": limit,
                "penalized_runtime_par2": par2,
                "iterations": iterations,
                "valid_UB": True,
                "ub_uses_subproblem_bound": True,
                "core_point_attempt_count": 10 if variant == CORE_VARIANT else 0,
                "core_point_success_count": 10 if variant == CORE_VARIANT else 0,
                "core_point_total_runtime": 1.0 if variant == CORE_VARIANT else 0.0,
                "v3_secondary_solve_count": 0,
                "v3_secondary_cut_added_count": 0,
                "v3_secondary_cuts_added": 0,
                "run_key": run_key,
                "iteration_log_path": f"results\\iteration_logs\\{log_name}",
                "git_commit": "648556b1956008e93bfc8ac0459cdc3260ab93be",
            }
            result_rows.append(row)

            log = [{
                "iteration": iterations,
                "LB": lb,
                "UB": ub,
                "global_gap": gap,
                "requested_master_mip_gap": 0.0001,
                "subproblem_requested_mip_gap": 0.0001,
                "target_robust_evaluation_used": True,
                "subproblem_has_incumbent": True,
                "cut_added": True,
                "cuts_added_this_iteration": 1,
                "final_certification_active": False,
                "core_point_attempted": variant == CORE_VARIANT,
                "core_point_stage1_status": "optimal" if variant == CORE_VARIANT else "not_attempted",
                "core_point_stage2_status": "optimal" if variant == CORE_VARIANT else "not_attempted",
                "core_point_current_value_floor": 9.9 if variant == CORE_VARIANT else "",
                "core_point_dual_feasible": variant == CORE_VARIANT,
                "core_point_original_value_at_current": 10.0 if variant == CORE_VARIANT else "",
                "core_point_strengthened_value_at_current": 10.0 if variant == CORE_VARIANT else "",
                "core_point_cut_accepted": variant == CORE_VARIANT,
                "core_point_cut_fallback_reason": "",
                "core_point_auxiliary_bound_used_for_UB": False,
                "v3_secondary_attempted": False,
                "v3_secondary_cut_added": False,
                "v3_secondary_bound_used_for_UB": False,
            }]
            _write_csv(result_root / "iteration_logs" / log_name, log)

            run_dir = result_root / "runs" / run_key
            run_dir.mkdir(parents=True)
            run_payload = {
                "run_key": run_key,
                "git_commit": "648556b1956008e93bfc8ac0459cdc3260ab93be",
                "result": {"run_key": run_key},
            }
            status = {
                "run_key": run_key,
                "state": "complete",
                "success": True,
                "solved_to_tolerance": solved,
                "status": row["status"],
            }
            (run_dir / "run.json").write_text(json.dumps(run_payload), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
            (run_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(_run_config(scale, seed, variant), sort_keys=False),
                encoding="utf-8",
            )
            (run_dir / "error.txt").write_text("", encoding="utf-8")

    _write_csv(result_root / "results.csv", result_rows)
    manifest = {
        "expected_run_count": 20,
        "completed_run_count": 20,
        "solved_run_count": solved_total,
        "failed_run_count": 0,
        "skipped_run_count": 0,
        "remaining_run_count": 0,
        "git_commit": "648556b1956008e93bfc8ac0459cdc3260ab93be",
    }
    (result_root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    top = {
        "experiment_name": experiment,
        "random_seeds": list(range(80, 90)),
        "instance_sizes": [scale],
        "variants": list(EXPECTED_VARIANTS),
        "time_limit": limit,
        "max_iterations": max_iterations,
        "tol": 1.0e-4,
        "gamma_target": 2,
        "gamma_schedule": [2],
        "formal_inference_allowed": False,
        "protocol_phase": "validation",
        "candidate_config_sha256": "7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6",
    }
    (result_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(top, sort_keys=False), encoding="utf-8"
    )
    return result_root


def _zip_tree(source: Path, target: Path) -> Path:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, f"{source.name}/{path.relative_to(source).as_posix()}")
    return target


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    medium_dir = _build_scale(tmp_path, "medium_large")
    large_dir = _build_scale(tmp_path, "large")
    medium = _zip_tree(medium_dir, tmp_path / "medium.zip")
    large = _zip_tree(large_dir, tmp_path / "large.zip")
    hashes = {"medium_large": file_sha256(medium), "large": file_sha256(large)}
    return medium, large, hashes


def _audit(medium: Path, large: Path, hashes: dict[str, str]):
    return audit_cut_v3_validation_results(
        medium,
        large,
        repo_root=ROOT,
        expected_archive_sha256=hashes,
    )


def test_normal_synthetic_validation_results_pass(tmp_path: Path) -> None:
    medium, large, hashes = _fixture(tmp_path)
    report = _audit(medium, large, hashes)
    assert report["all_required_checks_passed"], report["failed_checks"]
    assert report["decision"] == "validation_pass"


def test_sha_mismatch_fails(tmp_path: Path) -> None:
    medium, large, hashes = _fixture(tmp_path)
    hashes["medium_large"] = "0" * 64
    report = _audit(medium, large, hashes)
    assert "medium_large_archive_sha256" in report["failed_checks"]
    assert report["decision"] != "validation_pass"


def test_seed_out_of_range_fails(tmp_path: Path) -> None:
    medium_dir = _build_scale(tmp_path, "medium_large")
    large_dir = _build_scale(tmp_path, "large")
    rows = _read_csv(medium_dir / "results.csv")
    rows[0]["seed"] = "90"
    _write_csv(medium_dir / "results.csv", rows)
    medium = _zip_tree(medium_dir, tmp_path / "medium.zip")
    large = _zip_tree(large_dir, tmp_path / "large.zip")
    report = _audit(medium, large, {"medium_large": file_sha256(medium), "large": file_sha256(large)})
    assert "medium_large_exact_seeds_80_89" in report["failed_checks"]
    assert "medium_large_sealed_seeds_90_109_absent" in report["failed_checks"]


def test_missing_run_fails(tmp_path: Path) -> None:
    medium_dir = _build_scale(tmp_path, "medium_large")
    large_dir = _build_scale(tmp_path, "large")
    rows = _read_csv(large_dir / "results.csv")[:-1]
    _write_csv(large_dir / "results.csv", rows)
    medium = _zip_tree(medium_dir, tmp_path / "medium.zip")
    large = _zip_tree(large_dir, tmp_path / "large.zip")
    report = _audit(medium, large, {"medium_large": file_sha256(medium), "large": file_sha256(large)})
    assert "large_exactly_20_rows" in report["failed_checks"]
    assert report["decision"] != "validation_pass"


def test_third_method_fails(tmp_path: Path) -> None:
    medium_dir = _build_scale(tmp_path, "medium_large")
    large_dir = _build_scale(tmp_path, "large")
    rows = _read_csv(medium_dir / "results.csv")
    extra = dict(rows[0])
    extra["variant_name"] = "unregistered_third_method"
    extra["run_key"] += "__third"
    rows.append(extra)
    _write_csv(medium_dir / "results.csv", rows)
    medium = _zip_tree(medium_dir, tmp_path / "medium.zip")
    large = _zip_tree(large_dir, tmp_path / "large.zip")
    report = _audit(medium, large, {"medium_large": file_sha256(medium), "large": file_sha256(large)})
    assert "medium_large_only_v1_and_core" in report["failed_checks"]


def test_parameter_drift_fails(tmp_path: Path) -> None:
    medium_dir = _build_scale(tmp_path, "medium_large")
    large_dir = _build_scale(tmp_path, "large")
    config_path = next(large_dir.glob(f"runs/*{CORE_VARIANT}/resolved_config.yaml"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["algorithm"]["core_point_update_weight"] = 0.6
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    medium = _zip_tree(medium_dir, tmp_path / "medium.zip")
    large = _zip_tree(large_dir, tmp_path / "large.zip")
    report = _audit(medium, large, {"medium_large": file_sha256(medium), "large": file_sha256(large)})
    assert "large_resolved_configs_frozen" in report["failed_checks"]


def test_invalid_ub_fails(tmp_path: Path) -> None:
    medium_dir = _build_scale(tmp_path, "medium_large")
    large_dir = _build_scale(tmp_path, "large")
    rows = _read_csv(large_dir / "results.csv")
    rows[0]["valid_UB"] = "False"
    _write_csv(large_dir / "results.csv", rows)
    medium = _zip_tree(medium_dir, tmp_path / "medium.zip")
    large = _zip_tree(large_dir, tmp_path / "large.zip")
    report = _audit(medium, large, {"medium_large": file_sha256(medium), "large": file_sha256(large)})
    assert "large_valid_bounds" in report["failed_checks"]


def test_par2_error_fails(tmp_path: Path) -> None:
    medium_dir = _build_scale(tmp_path, "medium_large")
    large_dir = _build_scale(tmp_path, "large")
    rows = _read_csv(medium_dir / "results.csv")
    rows[0]["penalized_runtime_par2"] = "999"
    _write_csv(medium_dir / "results.csv", rows)
    medium = _zip_tree(medium_dir, tmp_path / "medium.zip")
    large = _zip_tree(large_dir, tmp_path / "large.zip")
    report = _audit(medium, large, {"medium_large": file_sha256(medium), "large": file_sha256(large)})
    assert "medium_large_par2_correct" in report["failed_checks"]


def test_core_auxiliary_bound_used_for_ub_fails(tmp_path: Path) -> None:
    medium_dir = _build_scale(tmp_path, "medium_large")
    large_dir = _build_scale(tmp_path, "large")
    log_path = next(medium_dir.glob(f"iteration_logs/*{CORE_VARIANT}.csv"))
    rows = _read_csv(log_path)
    rows[0]["core_point_auxiliary_bound_used_for_UB"] = "True"
    _write_csv(log_path, rows)
    medium = _zip_tree(medium_dir, tmp_path / "medium.zip")
    large = _zip_tree(large_dir, tmp_path / "large.zip")
    report = _audit(medium, large, {"medium_large": file_sha256(medium), "large": file_sha256(large)})
    assert "medium_large_only_original_robust_bound_updates_ub" in report["failed_checks"]


def test_failed_necessary_gate_never_returns_pass(tmp_path: Path) -> None:
    medium_dir = _build_scale(tmp_path, "medium_large")
    large_dir = _build_scale(tmp_path, "large")
    rows = _read_csv(large_dir / "results.csv")
    for row in rows:
        if row["variant_name"] == CORE_VARIANT:
            row["runtime"] = "1750"
            row["penalized_runtime_par2"] = "1750"
    _write_csv(large_dir / "results.csv", rows)
    medium = _zip_tree(medium_dir, tmp_path / "medium.zip")
    large = _zip_tree(large_dir, tmp_path / "large.zip")
    report = _audit(medium, large, {"medium_large": file_sha256(medium), "large": file_sha256(large)})
    assert report["decision"] == "validation_fail"
    assert not report["decision_gates"]["large_mean_par2_reduction_at_least_7_5_percent"]
