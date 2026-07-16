from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.final_evaluation_analysis import (
    AnalysisIntegrityError,
    FORBIDDEN_REPORT_PHRASES,
    assert_audit_passes,
    audit_frames,
    bootstrap_interval,
    build_input_manifest,
    certification_summary,
    generate_report,
    holm_adjust_by_family,
    method_summary,
    paired_comparisons,
    performance_profile,
    runtime_ranks,
)


METHODS = [
    "standard_benders",
    "static_inexact_benders",
    "mp_adaptive_rho050",
    "sp_adaptive_rho050",
    "proposed_joint_rho025_050",
]
SEEDS = list(range(10, 20))


def analysis_config() -> dict:
    return {
        "experiment_name": "synthetic_final",
        "expected_seeds": SEEDS,
        "expected_instance_size": "medium_large",
        "tolerance": 1e-4,
        "method_order": METHODS,
        "reference_method": "proposed_joint_rho025_050",
        "paper_labels": {
            method: {"English": method, "Chinese": method} for method in METHODS
        },
        "comparison_families": {
            "primary_confirmatory": [
                ["proposed_joint_rho025_050", "standard_benders"],
                ["proposed_joint_rho025_050", "static_inexact_benders"],
            ],
            "secondary_ablation": [
                ["proposed_joint_rho025_050", "mp_adaptive_rho050"],
                ["proposed_joint_rho025_050", "sp_adaptive_rho050"],
            ],
        },
        "bootstrap_seed": 20260716,
        "bootstrap_resamples": 500,
        "confidence_level": 0.95,
    }


def synthetic_inputs() -> tuple[pd.DataFrame, dict[tuple[int, str], pd.DataFrame], dict]:
    rows = []
    logs = {}
    runtime_offsets = {
        "standard_benders": 5.0,
        "static_inexact_benders": 8.0,
        "mp_adaptive_rho050": 3.0,
        "sp_adaptive_rho050": 4.0,
        "proposed_joint_rho025_050": 2.0,
    }
    for seed in SEEDS:
        for method in METHODS:
            adaptive_master = method in {
                "mp_adaptive_rho050",
                "proposed_joint_rho025_050",
            }
            adaptive_subproblem = method in {
                "sp_adaptive_rho050",
                "proposed_joint_rho025_050",
            }
            fixed_master = 0.0001 if method in {"standard_benders", "sp_adaptive_rho050"} else 0.02
            fixed_subproblem = 0.0001 if method in {"standard_benders", "mp_adaptive_rho050"} else 0.02 if method == "static_inexact_benders" else 0.05
            certification_enabled = method != "standard_benders"
            rows.append(
                {
                    "seed": seed,
                    "variant_name": method,
                    "instance_size": "medium_large",
                    "status": "optimal",
                    "valid_UB": True,
                    "subproblem_mode": "robust_dual_milp",
                    "target_subproblem_status": "optimal",
                    "final_gap": 5e-5,
                    "num_subproblem_nonoptimal": 0,
                    "num_subproblem_without_incumbent": 0,
                    "gamma_target": 2,
                    "gamma_schedule": "2",
                    "max_cuts_per_iteration": 1,
                    "cut_selection_enabled": False,
                    "adaptive_secondary_cut_selection_enabled": False,
                    "adaptive_secondary_generation_enabled": False,
                    "secondary_solves_attempted_total": 0,
                    "secondary_cuts_added_total": 0,
                    "secondary_cuts_skipped_total": 0,
                    "iterations": 2,
                    "objective": 50.0 + seed * 0.01,
                    "best_bound": 49.999 + seed * 0.01,
                    "runtime": 20.0 + runtime_offsets[method] + (seed - 10) * 0.1,
                    "master_time": 8.0,
                    "subproblem_time": 10.0,
                    "precision_policy": "legacy" if method in {"standard_benders", "static_inexact_benders"} else "joint_error_budget",
                    "adaptive_master_precision_enabled": adaptive_master,
                    "adaptive_subproblem_precision_enabled": adaptive_subproblem,
                    "fixed_master_mip_gap": fixed_master,
                    "fixed_subproblem_mip_gap": fixed_subproblem,
                    "master_error_budget_ratio": 0.50 if method == "mp_adaptive_rho050" else 0.25,
                    "subproblem_error_budget_ratio": 0.50,
                    "final_certification_enabled": certification_enabled,
                    "final_certification_triggered": method == "static_inexact_benders",
                    "final_certification_count": 1 if method == "static_inexact_benders" else 0,
                    "final_certification_iterations": 1 if method == "static_inexact_benders" else 0,
                    "time_to_gap_5pct": 5.0,
                    "time_to_gap_1pct": 10.0,
                    "time_to_gap_05pct": 12.0,
                    "time_to_gap_01pct": 18.0,
                }
            )
            master_selected = [fixed_master, 0.01 if adaptive_master else fixed_master]
            subproblem_selected = [fixed_subproblem, 0.02 if adaptive_subproblem else fixed_subproblem]
            logs[(seed, method)] = pd.DataFrame(
                {
                    "iteration": [1, 2],
                    "seed": [seed, seed],
                    "variant_name": [method, method],
                    "LB": [0.0, 10.0],
                    "UB": [100.0, 50.0],
                    "global_gap": [1.0, 0.8],
                    "elapsed_time": [1.0, 2.0],
                    "precision_gap_fallback_used": [True, False],
                    "valid_global_gap_for_precision": [1.0, 0.8],
                    "adaptive_master_precision_enabled": [adaptive_master] * 2,
                    "adaptive_subproblem_precision_enabled": [adaptive_subproblem] * 2,
                    "master_gap_selected": master_selected,
                    "subproblem_gap_selected": subproblem_selected,
                    "requested_master_mip_gap": master_selected,
                    "subproblem_requested_mip_gap": subproblem_selected,
                    "final_certification_active": [False, False],
                    "secondary_solve_attempted": [False, False],
                    "secondary_cuts_added_total": [0, 0],
                    "secondary_cuts_skipped_total": [0, 0],
                }
            )
    return pd.DataFrame(rows), logs, {"final_certification_no_cut_patience": 2}


def require_failure(results: pd.DataFrame, logs: dict, resolved: dict) -> None:
    audit = audit_frames(results, logs, analysis_config(), resolved)
    with pytest.raises(AnalysisIntegrityError):
        assert_audit_passes(audit)


def test_exact_expected_seed_variant_grid_passes() -> None:
    results, logs, resolved = synthetic_inputs()
    audit = audit_frames(results, logs, analysis_config(), resolved)
    assert audit["passed"].all(), audit.loc[~audit["passed"], ["check", "details"]]


def test_missing_row_fails() -> None:
    results, logs, resolved = synthetic_inputs()
    require_failure(results.iloc[:-1].copy(), logs, resolved)


def test_duplicate_row_fails() -> None:
    results, logs, resolved = synthetic_inputs()
    duplicate = pd.concat([results, results.iloc[[0]]], ignore_index=True)
    require_failure(duplicate, logs, resolved)


def test_tuning_seed_contamination_fails() -> None:
    results, logs, resolved = synthetic_inputs()
    results.loc[0, "seed"] = 0
    require_failure(results, logs, resolved)


def test_invalid_ub_fails() -> None:
    results, logs, resolved = synthetic_inputs()
    results.loc[0, "valid_UB"] = False
    require_failure(results, logs, resolved)


def test_nonoptimal_target_subproblem_fails() -> None:
    results, logs, resolved = synthetic_inputs()
    results.loc[0, "target_subproblem_status"] = "time_limit"
    require_failure(results, logs, resolved)


def test_final_gap_above_tolerance_fails() -> None:
    results, logs, resolved = synthetic_inputs()
    results.loc[0, "final_gap"] = 2e-4
    require_failure(results, logs, resolved)


def test_nonmonotone_lb_fails() -> None:
    results, logs, resolved = synthetic_inputs()
    logs[(10, "standard_benders")].loc[:, "LB"] = [10.0, 9.0]
    require_failure(results, logs, resolved)


def test_nonmonotone_ub_fails() -> None:
    results, logs, resolved = synthetic_inputs()
    logs[(10, "standard_benders")].loc[:, "UB"] = [50.0, 60.0]
    require_failure(results, logs, resolved)


def test_nonmonotone_adaptive_precision_fails() -> None:
    results, logs, resolved = synthetic_inputs()
    logs[(10, "mp_adaptive_rho050")].loc[:, "master_gap_selected"] = [0.01, 0.02]
    logs[(10, "mp_adaptive_rho050")].loc[:, "requested_master_mip_gap"] = [0.01, 0.02]
    require_failure(results, logs, resolved)


def test_iteration_count_mismatch_fails() -> None:
    results, logs, resolved = synthetic_inputs()
    results.loc[0, "iterations"] = 3
    require_failure(results, logs, resolved)


def test_wrong_proposed_ratios_fail() -> None:
    results, logs, resolved = synthetic_inputs()
    mask = results["variant_name"] == "proposed_joint_rho025_050"
    results.loc[mask, "master_error_budget_ratio"] = 0.50
    require_failure(results, logs, resolved)


def test_baseline_contamination_fails() -> None:
    results, logs, resolved = synthetic_inputs()
    mask = results["variant_name"] == "standard_benders"
    results.loc[mask, "adaptive_master_precision_enabled"] = True
    require_failure(results, logs, resolved)


def test_bootstrap_output_is_deterministic() -> None:
    values = np.array([-3.0, -2.0, 1.0, 4.0])
    first = bootstrap_interval(values, np.mean, seed=7, resamples=1000, confidence_level=0.95)
    second = bootstrap_interval(values, np.mean, seed=7, resamples=1000, confidence_level=0.95)
    assert first == second


def test_holm_is_independent_within_families() -> None:
    frame = pd.DataFrame(
        {
            "comparator": ["a", "b", "c", "d"],
            "comparison_family": ["primary", "primary", "secondary", "secondary"],
            "raw_p_value": [0.01, 0.04, 0.03, 0.04],
        }
    )
    adjusted = holm_adjust_by_family(frame).set_index("comparator")
    assert adjusted.loc["a", "holm_adjusted_p_value"] == pytest.approx(0.02)
    assert adjusted.loc["b", "holm_adjusted_p_value"] == pytest.approx(0.04)
    assert adjusted.loc["c", "holm_adjusted_p_value"] == pytest.approx(0.06)
    assert adjusted.loc["d", "holm_adjusted_p_value"] == pytest.approx(0.06)


def test_runtime_ranks_handle_ties_deterministically() -> None:
    frame = pd.DataFrame(
        {
            "seed": [10, 10, 10],
            "variant_name": ["a", "b", "c"],
            "runtime": [1.0, 1.0, 2.0],
        }
    )
    ranks = runtime_ranks(frame).set_index("variant_name")
    assert ranks.loc["a", "runtime_rank"] == pytest.approx(1.5)
    assert ranks.loc["b", "runtime_rank"] == pytest.approx(1.5)
    assert ranks.loc["c", "runtime_rank"] == pytest.approx(3.0)


def test_performance_profile_values_are_correct() -> None:
    frame = pd.DataFrame(
        {
            "seed": [10, 10, 11, 11],
            "variant_name": ["a", "b", "a", "b"],
            "runtime": [1.0, 2.0, 4.0, 2.0],
        }
    )
    profile = performance_profile(frame, ["a", "b"], np.array([1.0, 2.0]))
    pivot = profile.pivot(index="tau", columns="method", values="fraction")
    assert pivot.loc[1.0, "a"] == pytest.approx(0.5)
    assert pivot.loc[1.0, "b"] == pytest.approx(0.5)
    assert pivot.loc[2.0, "a"] == pytest.approx(1.0)
    assert pivot.loc[2.0, "b"] == pytest.approx(1.0)


def test_generated_tables_contain_all_five_methods() -> None:
    results, _, _ = synthetic_inputs()
    summary = method_summary(results, analysis_config())
    assert summary["method"].tolist() == METHODS


def test_generated_report_avoids_forbidden_overclaiming() -> None:
    results, _, _ = synthetic_inputs()
    config = analysis_config()
    summary = method_summary(results, config)
    cert = certification_summary(results, config)
    ranks = runtime_ranks(results)
    _, comparisons, _ = paired_comparisons(results, ranks, config)
    report = generate_report(summary, comparisons, cert, config)
    lowered = report.lower()
    assert all(phrase not in lowered for phrase in FORBIDDEN_REPORT_PHRASES)
    assert "tight-tolerance inexact Benders" in report


def test_input_hashes_are_deterministic_without_absolute_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    input_dir = repo / "experiments" / "results_final" / "final_evaluation_joint_v1"
    (input_dir / "instances").mkdir(parents=True)
    (input_dir / "iteration_logs").mkdir()
    for name, content in (
        ("results.csv", "a\n1\n"),
        ("summary.csv", "a\n1\n"),
        ("resolved_config.yaml", "a: 1\n"),
    ):
        (input_dir / name).write_text(content, encoding="utf-8")
    (input_dir / "instances" / "instance.json").write_text("{}\n", encoding="utf-8")
    (input_dir / "iteration_logs" / "log.csv").write_text("a\n1\n", encoding="utf-8")
    config_path = repo / "analysis" / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("a: 1\n", encoding="utf-8")

    first = build_input_manifest(input_dir, config_path, repo)
    second = build_input_manifest(input_dir, config_path, repo)
    assert first == second
    serialized = json.dumps(first)
    assert str(repo) not in serialized
    assert all(not Path(entry["path"]).is_absolute() for entry in first["files"])
