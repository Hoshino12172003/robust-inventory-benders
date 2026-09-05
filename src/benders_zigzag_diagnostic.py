from __future__ import annotations

from collections import Counter
import hashlib
import math
from typing import Any

import numpy as np
from scipy.stats import rankdata


NEAR_RETURN_TOLERANCE = 1.0e-6
PERIODS = (2, 3, 4, 5)


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _correlation(left: list[float], right: list[float], *, ranks: bool = False) -> float | None:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.size < 2 or y.size != x.size:
        return None
    if ranks:
        x = rankdata(x)
        y = rankdata(y)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _y_signature(values: np.ndarray) -> str:
    return "".join("1" if value >= 0.5 else "0" for value in values)


def _x_hash(values: np.ndarray) -> str:
    rounded = np.round(values, decimals=8).astype("<f8", copy=False)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def calculate_movements(
    metrics: list[dict[str, Any]],
    y_trajectory: list[list[float]],
    x_trajectory: list[list[list[float]]],
    *,
    near_return_tolerance: float = NEAR_RETURN_TOLERANCE,
) -> list[dict[str, Any]]:
    if not (len(metrics) == len(y_trajectory) == len(x_trajectory)):
        raise ValueError("Metrics, y, and x trajectories must have equal lengths.")
    if near_return_tolerance < 0.0:
        raise ValueError("near_return_tolerance must be nonnegative.")
    if len(metrics) < 2:
        return []

    ys = [np.asarray(values, dtype=float) for values in y_trajectory]
    xs = [np.asarray(values, dtype=float) for values in x_trajectory]
    y_shape = ys[0].shape
    x_shape = xs[0].shape
    if len(y_shape) != 1 or len(x_shape) != 2:
        raise ValueError("Expected one-dimensional y and two-dimensional x trajectories.")
    if any(values.shape != y_shape for values in ys) or any(values.shape != x_shape for values in xs):
        raise ValueError("Trajectory dimensions changed between iterations.")
    if not all(np.isfinite(values).all() for values in (*ys, *xs)):
        raise ValueError("Trajectory contains NaN or Inf.")

    seen_y: set[str] = {_y_signature(ys[0])}
    movements: list[dict[str, Any]] = []
    for index in range(1, len(metrics)):
        current_y = ys[index]
        previous_y = ys[index - 1]
        current_x = xs[index]
        previous_x = xs[index - 1]
        current_binary = current_y >= 0.5
        previous_binary = previous_y >= 0.5
        changed = current_binary != previous_binary
        delta_x = current_x - previous_x
        x_l1 = float(np.sum(np.abs(delta_x)))
        x_l2 = float(np.linalg.norm(delta_x))
        normalized_l1 = x_l1 / max(
            1.0,
            0.5 * (float(np.sum(np.abs(current_x))) + float(np.sum(np.abs(previous_x)))),
        )
        normalized_l2 = x_l2 / max(1.0, float(np.linalg.norm(previous_x)))
        signature = _y_signature(current_y)
        seen_y.add(signature)

        row: dict[str, Any] = {
            "iteration": int(metrics[index]["iteration"]),
            "previous_iteration": int(metrics[index - 1]["iteration"]),
            "elapsed_time": float(metrics[index]["elapsed_time"]),
            "LB": float(metrics[index]["LB"]),
            "UB": float(metrics[index]["UB"]),
            "gap": float(metrics[index]["gap"]),
            "master_runtime": float(metrics[index]["master_time"]),
            "cumulative_cuts": int(metrics[index]["cuts_added_total"]),
            "y_hamming": int(np.sum(changed)),
            "y_hamming_normalized": float(np.mean(changed)),
            "y_added": int(np.sum(current_binary & ~previous_binary)),
            "y_removed": int(np.sum(previous_binary & ~current_binary)),
            "y_exact_same": bool(not np.any(changed)),
            "unique_y_patterns_to_date": len(seen_y),
            "y_pattern": signature,
            "x_l1": x_l1,
            "x_l2": x_l2,
            "x_normalized_l1": normalized_l1,
            "x_normalized_l2": normalized_l2,
            "x_rounded_hash": _x_hash(current_x),
            "lb_improvement": float(metrics[index]["LB"]) - float(metrics[index - 1]["LB"]),
            "gap_reduction": float(metrics[index - 1]["gap"]) - float(metrics[index]["gap"]),
        }
        for period in PERIODS:
            if index >= period:
                row[f"y_cycle_p{period}"] = (
                    signature == _y_signature(ys[index - period])
                    and signature != _y_signature(previous_y)
                )
                denominator = max(1.0, float(np.sum(np.abs(current_x))))
                relative_return = float(np.sum(np.abs(current_x - xs[index - period]))) / denominator
                adjacent_relative_movement = x_l1 / denominator
                row[f"x_cycle_p{period}"] = (
                    relative_return <= near_return_tolerance
                    and adjacent_relative_movement > near_return_tolerance
                )
                row[f"x_return_relative_l1_p{period}"] = relative_return
            else:
                row[f"y_cycle_p{period}"] = False
                row[f"x_cycle_p{period}"] = False
                row[f"x_return_relative_l1_p{period}"] = None
        movements.append(row)
    return movements


def _segment_indices(total_iterations: int, movement_count: int) -> dict[str, list[int]]:
    iteration_numbers = np.arange(2, total_iterations + 1)
    if movement_count != len(iteration_numbers):
        raise ValueError("Movement count does not match the iteration count.")
    q25 = math.ceil(0.25 * total_iterations)
    q50 = math.ceil(0.50 * total_iterations)
    q75 = math.ceil(0.75 * total_iterations)
    q90 = math.ceil(0.90 * total_iterations)
    masks = {
        "first_25pct": iteration_numbers <= q25,
        "25_to_50pct": (iteration_numbers > q25) & (iteration_numbers <= q50),
        "50_to_75pct": (iteration_numbers > q50) & (iteration_numbers <= q75),
        "75_to_90pct": (iteration_numbers > q75) & (iteration_numbers <= q90),
        "final_10pct": iteration_numbers > q90,
    }
    result = {name: np.flatnonzero(mask).tolist() for name, mask in masks.items()}
    for length in (500, 250, 100):
        start = max(0, movement_count - length)
        result[f"final_{length}"] = list(range(start, movement_count))
    return result


def summarize_segments(
    movements: list[dict[str, Any]], total_iterations: int
) -> list[dict[str, Any]]:
    segments = _segment_indices(total_iterations, len(movements))
    output: list[dict[str, Any]] = []
    for name, indices in segments.items():
        rows = [movements[index] for index in indices]
        hamming = [float(row["y_hamming"]) for row in rows]
        normalized_l1 = [float(row["x_normalized_l1"]) for row in rows]
        cycle_counts = {
            f"y_cycles_p{period}": sum(bool(row[f"y_cycle_p{period}"]) for row in rows)
            for period in PERIODS
        }
        cycle_counts.update(
            {
                f"x_cycles_p{period}": sum(bool(row[f"x_cycle_p{period}"]) for row in rows)
                for period in PERIODS
            }
        )
        output.append(
            {
                "segment": name,
                "movement_count": len(rows),
                "start_iteration": rows[0]["iteration"] if rows else None,
                "end_iteration": rows[-1]["iteration"] if rows else None,
                "median_y_hamming": _percentile(hamming, 50),
                "p90_y_hamming": _percentile(hamming, 90),
                "fraction_y_hamming_positive": (
                    sum(value > 0.0 for value in hamming) / len(rows) if rows else None
                ),
                "median_x_normalized_l1": _percentile(normalized_l1, 50),
                "p90_x_normalized_l1": _percentile(normalized_l1, 90),
                "fraction_x_normalized_l1_gt_0_01": (
                    sum(value > 0.01 for value in normalized_l1) / len(rows) if rows else None
                ),
                "fraction_x_normalized_l1_gt_0_05": (
                    sum(value > 0.05 for value in normalized_l1) / len(rows) if rows else None
                ),
                "fraction_x_normalized_l1_gt_0_10": (
                    sum(value > 0.10 for value in normalized_l1) / len(rows) if rows else None
                ),
                "unique_y_patterns": len({row["y_pattern"] for row in rows}),
                "short_cycles_total": sum(cycle_counts.values()),
                "lb_improvement": sum(float(row["lb_improvement"]) for row in rows),
                "mean_master_runtime": (
                    float(np.mean([row["master_runtime"] for row in rows])) if rows else None
                ),
                **cycle_counts,
            }
        )
    return output


def pattern_concentration(y_trajectory: list[list[float]]) -> list[dict[str, Any]]:
    signatures = [_y_signature(np.asarray(values, dtype=float)) for values in y_trajectory]
    counts = Counter(signatures)
    runs: dict[str, list[int]] = {signature: [] for signature in counts}
    if signatures:
        current = signatures[0]
        length = 1
        for signature in signatures[1:]:
            if signature == current:
                length += 1
            else:
                runs[current].append(length)
                current = signature
                length = 1
        runs[current].append(length)
    total = len(signatures)
    return [
        {
            "rank": rank,
            "y_pattern": signature,
            "count": count,
            "share": count / total,
            "mean_run_length": float(np.mean(runs[signature])),
            "switches_into_pattern": max(0, len(runs[signature]) - (1 if signatures and signatures[0] == signature else 0)),
        }
        for rank, (signature, count) in enumerate(counts.most_common(10), start=1)
    ]


def movement_relationships(movements: list[dict[str, Any]]) -> dict[str, Any]:
    movement = [float(row["x_normalized_l1"]) for row in movements]
    outcomes = {
        "lb_improvement": [float(row["lb_improvement"]) for row in movements],
        "master_runtime": [float(row["master_runtime"]) for row in movements],
        "gap_reduction": [float(row["gap_reduction"]) for row in movements],
    }
    correlations = {
        outcome: {
            "pearson": _correlation(movement, values),
            "spearman": _correlation(movement, values, ranks=True),
        }
        for outcome, values in outcomes.items()
    }
    bins = (("small", -float("inf"), 0.01), ("medium", 0.01, 0.05), ("large", 0.05, float("inf")))
    binned: list[dict[str, Any]] = []
    for name, low, high in bins:
        selected = [
            row for row in movements
            if float(row["x_normalized_l1"]) <= high and (name == "small" or float(row["x_normalized_l1"]) > low)
        ]
        binned.append(
            {
                "movement_bin": name,
                "count": len(selected),
                "mean_lb_improvement": float(np.mean([row["lb_improvement"] for row in selected])) if selected else None,
                "mean_master_runtime": float(np.mean([row["master_runtime"] for row in selected])) if selected else None,
                "mean_gap_reduction": float(np.mean([row["gap_reduction"] for row in selected])) if selected else None,
            }
        )
    return {"correlations": correlations, "bins": binned}


def classify_zigzag(final_100: dict[str, Any], final_250: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    cycle_fraction = max(
        final_100["short_cycles_total"] / max(1, 8 * final_100["movement_count"]),
        final_250["short_cycles_total"] / max(1, 8 * final_250["movement_count"]),
    )
    signals = {
        "frequent_y_switching": max(final_100["fraction_y_hamming_positive"], final_250["fraction_y_hamming_positive"]) > 0.50,
        "large_median_x_movement": max(final_100["median_x_normalized_l1"], final_250["median_x_normalized_l1"]) > 0.05,
        "frequent_very_large_x_movement": max(final_100["fraction_x_normalized_l1_gt_0_10"], final_250["fraction_x_normalized_l1_gt_0_10"]) > 0.30,
        "significant_short_cycles": cycle_fraction > 0.10,
        "large_movement_with_flat_lb": (
            max(final_100["median_x_normalized_l1"], final_250["median_x_normalized_l1"]) > 0.05
            and max(final_100["lb_improvement"], final_250["lb_improvement"]) <= 1.0e-4
        ),
    }
    strong_count = sum(signals.values())
    moderate = (
        max(final_100["fraction_y_hamming_positive"], final_250["fraction_y_hamming_positive"]) > 0.25
        or max(final_100["median_x_normalized_l1"], final_250["median_x_normalized_l1"]) > 0.01
        or max(final_100["fraction_x_normalized_l1_gt_0_05"], final_250["fraction_x_normalized_l1_gt_0_05"]) > 0.20
        or cycle_fraction > 0.05
    )
    negligible = (
        max(final_100["fraction_y_hamming_positive"], final_250["fraction_y_hamming_positive"]) <= 0.10
        and max(final_100["median_x_normalized_l1"], final_250["median_x_normalized_l1"]) <= 0.01
        and max(final_100["fraction_x_normalized_l1_gt_0_05"], final_250["fraction_x_normalized_l1_gt_0_05"]) <= 0.05
        and cycle_fraction == 0.0
    )
    if strong_count >= 2:
        classification = "STRONG_ZIGZAG"
    elif moderate:
        classification = "MODERATE_ZIGZAG"
    elif negligible:
        classification = "NO_MEANINGFUL_ZIGZAG"
    elif any(row > 0.0 for row in (final_100["fraction_y_hamming_positive"], final_250["fraction_y_hamming_positive"])):
        classification = "WEAK_ZIGZAG"
    else:
        classification = "INCONCLUSIVE"
    return classification, {"signals": signals, "short_cycle_fraction": cycle_fraction, "strong_signal_count": strong_count}


def analyze_trajectory(
    metrics: list[dict[str, Any]],
    y_trajectory: list[list[float]],
    x_trajectory: list[list[list[float]]],
) -> dict[str, Any]:
    movements = calculate_movements(metrics, y_trajectory, x_trajectory)
    segments = summarize_segments(movements, len(metrics))
    by_name = {row["segment"]: row for row in segments}
    classification, evidence = classify_zigzag(by_name["final_100"], by_name["final_250"])
    relationships = movement_relationships(movements)
    patterns = pattern_concentration(y_trajectory)
    tail = by_name["final_250"]
    enough_to_explain_tail = classification in {"STRONG_ZIGZAG", "MODERATE_ZIGZAG"}
    return {
        "movements": movements,
        "segments": segments,
        "patterns": patterns,
        "relationships": relationships,
        "classification": classification,
        "classification_evidence": evidence,
        "unique_y_patterns": len({_y_signature(np.asarray(row)) for row in y_trajectory}),
        "y_pattern_switches": sum(not row["y_exact_same"] for row in movements),
        "short_cycles": {
            f"{kind}_p{period}": sum(bool(row[f"{kind}_cycle_p{period}"]) for row in movements)
            for kind in ("y", "x")
            for period in PERIODS
        },
        "enough_to_explain_tail_stagnation": enough_to_explain_tail,
        "stabilization_recommendation": (
            "LIGHT_TRUST_REGION_REVIEW" if enough_to_explain_tail and tail["fraction_y_hamming_positive"] > 0.50
            else "LEVEL_METHOD_REVIEW" if enough_to_explain_tail
            else "DO_NOT_ADD_STABILIZATION"
        ),
    }
