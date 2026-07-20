from __future__ import annotations

import csv
import json
from pathlib import Path

from src.cut_v3_development_analysis import (
    EXPECTED_VARIANTS,
    analyze_development,
    analyze_result_directory,
    markdown_report,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_results(root: Path, *, large: bool) -> None:
    profiles = {
        "proposed_joint_rho025_050": (100.0, 100.0),
        "joint_v1_core_point_strengthened": ((94.0, 88.0) if large else (101.0, 95.0)),
        "joint_v1_stall_secondary_cut": ((100.0, 92.0) if large else (105.0, 98.0)),
        "proposed_cut_strengthened_joint_v3": ((93.0, 85.0) if large else (99.0, 94.0)),
    }
    rows: list[dict[str, object]] = []
    # Synthetic row identifiers deliberately avoid every development,
    # validation, and final-test seed reserved by the protocol.
    for seed in range(301, 306):
        for variant in EXPECTED_VARIANTS:
            par2, iterations = profiles[variant]
            log_name = f"{seed}_{variant}.csv"
            _write_csv(
                root / "iteration_logs" / log_name,
                [
                    {
                        "LB": 10.0,
                        "UB": 20.0,
                        "requested_master_mip_gap": 0.02,
                        "subproblem_requested_mip_gap": 0.05,
                        "core_point_auxiliary_bound_used_for_UB": False,
                        "v3_secondary_bound_used_for_UB": False,
                        "subproblem_has_incumbent": True,
                        "cut_added": True,
                        "final_certification_active": False,
                    },
                    {
                        "LB": 11.0,
                        "UB": 19.0,
                        "requested_master_mip_gap": 0.01,
                        "subproblem_requested_mip_gap": 0.02,
                        "core_point_auxiliary_bound_used_for_UB": False,
                        "v3_secondary_bound_used_for_UB": False,
                        "subproblem_has_incumbent": True,
                        "cut_added": True,
                        "final_certification_active": False,
                    },
                ],
            )
            rows.append(
                {
                    "seed": seed,
                    "variant_name": variant,
                    "run_key": f"{seed}::{variant}",
                    "status": "optimal",
                    "solved_to_tolerance": True,
                    "valid_UB": True,
                    "lower_bound": 100.0,
                    "upper_bound": 100.001,
                    "final_gap": 1.0e-5,
                    "runtime": par2,
                    "penalized_runtime_par2": par2,
                    "iterations": iterations,
                    "master_time": 10.0,
                    "subproblem_time": 20.0,
                    "core_point_total_runtime": 2.0 if "core" in variant or "proposed_cut" in variant else 0.0,
                    "core_point_attempt_count": 5 if "core" in variant or "proposed_cut" in variant else 0,
                    "core_point_success_count": 4 if "core" in variant or "proposed_cut" in variant else 0,
                    "v3_secondary_total_runtime": 3.0 if "secondary" in variant or "proposed_cut" in variant else 0.0,
                    "v3_secondary_trigger_count": 3 if "secondary" in variant or "proposed_cut" in variant else 0,
                    "v3_secondary_cut_added_count": 2 if "secondary" in variant or "proposed_cut" in variant else 0,
                    "v3_total_extra_cut_runtime": 5.0 if variant == "proposed_cut_strengthened_joint_v3" else 2.0,
                    "iteration_log_path": log_name,
                }
            )
    _write_csv(root / "results.csv", rows)


def test_analysis_checks_completeness_and_iteration_invariants(tmp_path: Path) -> None:
    result_dir = tmp_path / "medium"
    _make_results(result_dir, large=False)
    report = analyze_result_directory(result_dir)
    assert report["actual_run_count"] == 20
    assert report["run_count_complete"]
    assert report["all_expected_variants_present"]
    assert report["all_iteration_checks_passed"]
    assert report["all_solved_runs_have_valid_ub"]
    assert report["all_runs_have_valid_lower_bound"]
    assert report["all_runs_have_valid_ub"]


def test_analysis_calculates_component_rates_and_times(tmp_path: Path) -> None:
    result_dir = tmp_path / "large"
    _make_results(result_dir, large=True)
    report = analyze_result_directory(result_dir)
    core = report["method_summaries"]["joint_v1_core_point_strengthened"]
    full = report["method_summaries"]["proposed_cut_strengthened_joint_v3"]
    assert core["core_point_success_rate"] == 0.8
    assert full["secondary_add_rate"] == 2.0 / 3.0
    assert full["mean_core_point_time"] == 2.0
    assert full["mean_secondary_time"] == 3.0
    assert full["mean_original_subproblem_time"] == 17.0


def test_frozen_selection_rule_selects_best_eligible_candidate(tmp_path: Path) -> None:
    medium = tmp_path / "medium"
    large = tmp_path / "large"
    _make_results(medium, large=False)
    _make_results(large, large=True)
    report = analyze_development(medium, large)
    judgment = report["candidate_judgment"]
    assert judgment["selected_candidate"] == "proposed_cut_strengthened_joint_v3"
    assert judgment["decision"] == "freeze_one_candidate"
    assert judgment["configuration_or_parameter_changes_performed"] is False
    assert report["benefits_vs_v1"]["large"]["proposed_cut_strengthened_joint_v3"][
        "par2_reduction_fraction"
    ] == 0.07
    json.dumps(report, allow_nan=False)
    markdown = markdown_report(report)
    assert "read-only" in markdown
    assert "proposed_cut_strengthened_joint_v3" in markdown
    assert "Relative benefits versus V1" in markdown


def test_incomplete_results_are_reported_not_fabricated(tmp_path: Path) -> None:
    result_dir = tmp_path / "missing"
    report = analyze_result_directory(result_dir)
    assert report["actual_run_count"] == 0
    assert not report["run_count_complete"]
    assert not report["all_iteration_checks_passed"]
