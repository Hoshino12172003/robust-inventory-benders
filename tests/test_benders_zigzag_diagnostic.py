from __future__ import annotations

import math

import pytest

from src.benders import solve_benders
from src.benders_zigzag_diagnostic import (
    analyze_trajectory,
    calculate_movements,
    pattern_concentration,
    summarize_segments,
)
from src.instance import generate_instance


def _metrics(count: int) -> list[dict[str, float | int]]:
    return [
        {
            "iteration": index + 1,
            "elapsed_time": float(index + 1),
            "LB": float(10 + index),
            "UB": 20.0,
            "gap": float(10 - index) / 20.0,
            "master_time": 0.1 * (index + 1),
            "cuts_added_total": index + 1,
        }
        for index in range(count)
    ]


def test_hamming_and_added_removed_depots() -> None:
    movements = calculate_movements(
        _metrics(2),
        [[1, 0, 1, 0], [0, 1, 1, 0]],
        [[[1.0]], [[1.0]]],
    )
    assert movements[0]["y_hamming"] == 2
    assert movements[0]["y_hamming_normalized"] == pytest.approx(0.5)
    assert movements[0]["y_added"] == 1
    assert movements[0]["y_removed"] == 1


def test_normalized_l1_and_l2_match_protocol_definitions() -> None:
    movements = calculate_movements(
        _metrics(2),
        [[1], [1]],
        [[[3.0, 4.0]], [[0.0, 8.0]]],
    )
    row = movements[0]
    assert row["x_l1"] == pytest.approx(7.0)
    assert row["x_l2"] == pytest.approx(5.0)
    assert row["x_normalized_l1"] == pytest.approx(7.0 / 7.5)
    assert row["x_normalized_l2"] == pytest.approx(1.0)


def test_period_two_y_and_near_x_cycles_are_detected() -> None:
    movements = calculate_movements(
        _metrics(3),
        [[1, 0], [0, 1], [1, 0]],
        [[[1.0, 2.0]], [[4.0, 2.0]], [[1.0 + 1e-7, 2.0]]],
    )
    assert movements[-1]["y_cycle_p2"] is True
    assert movements[-1]["x_cycle_p2"] is True
    assert movements[-1]["x_return_relative_l1_p2"] <= 1e-6


def test_cycle_detection_rejects_non_return_and_short_history() -> None:
    movements = calculate_movements(
        _metrics(3),
        [[1], [0], [1]],
        [[[1.0]], [[2.0]], [[1.1]]],
    )
    assert movements[0]["y_cycle_p2"] is False
    assert movements[-1]["y_cycle_p2"] is True
    assert movements[-1]["x_cycle_p2"] is False


def test_zero_movement_and_identical_pattern() -> None:
    movements = calculate_movements(
        _metrics(2),
        [[1, 0], [1, 0]],
        [[[0.0, 0.0]], [[0.0, 0.0]]],
    )
    row = movements[0]
    assert row["y_exact_same"] is True
    assert row["x_normalized_l1"] == 0.0
    assert all(not row[f"y_cycle_p{period}"] for period in (2, 3, 4, 5))
    assert all(not row[f"x_cycle_p{period}"] for period in (2, 3, 4, 5))
    patterns = pattern_concentration([[1, 0], [1, 0]])
    assert patterns[0]["count"] == 2
    assert patterns[0]["share"] == 1.0
    assert patterns[0]["mean_run_length"] == 2.0


def test_segments_use_disjoint_percentile_bands_and_bounded_tails() -> None:
    metrics = _metrics(20)
    ys = [[index % 2] for index in range(20)]
    xs = [[[float(index)]] for index in range(20)]
    movements = calculate_movements(metrics, ys, xs)
    segments = {row["segment"]: row for row in summarize_segments(movements, 20)}
    assert segments["first_25pct"]["end_iteration"] == 5
    assert segments["25_to_50pct"]["start_iteration"] == 6
    assert segments["final_10pct"]["start_iteration"] == 19
    assert segments["final_100"]["movement_count"] == 19


def test_analysis_reports_unique_patterns_and_finite_correlations() -> None:
    metrics = _metrics(8)
    ys = [[index % 2, (index // 2) % 2] for index in range(8)]
    xs = [[[float(index), float(index % 3)]] for index in range(8)]
    report = analyze_trajectory(metrics, ys, xs)
    assert report["unique_y_patterns"] == 4
    assert report["classification"] in {
        "STRONG_ZIGZAG",
        "MODERATE_ZIGZAG",
        "WEAK_ZIGZAG",
        "NO_MEANINGFUL_ZIGZAG",
        "INCONCLUSIVE",
    }
    pearson = report["relationships"]["correlations"]["master_runtime"]["pearson"]
    assert pearson is None or math.isfinite(pearson)


def test_mismatched_or_nonfinite_trajectories_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        calculate_movements(_metrics(2), [[1]], [[[1.0]], [[2.0]]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        calculate_movements(_metrics(2), [[1], [1]], [[[1.0]], [[float("nan")]]])


def test_first_stage_trajectory_logging_is_explicitly_opt_in() -> None:
    config = {
        "instance": {"num_warehouses": 2, "num_products": 1, "num_regions": 2, "budget_factor": 0.7},
        "robust": {"gamma_target": 1, "gamma_schedule": [1], "max_scenarios": 10},
        "benders": {
            "max_iterations": 2,
            "tol": 1e-4,
            "initial_mip_gap": 0.05,
            "final_mip_gap": 1e-5,
            "time_limit": 30,
            "output_flag": False,
        },
        "diagnostics": {"record_first_stage_trajectory": True},
    }
    instance = generate_instance(config, seed=17)
    result = solve_benders(config, instance, "standard_benders")
    assert result.iteration_log
    assert all(len(row["trajectory_y"]) == 2 for row in result.iteration_log)
    assert all(len(row["trajectory_x"]) == 2 for row in result.iteration_log)
    assert all(len(row["trajectory_x"][0]) == 1 for row in result.iteration_log)
