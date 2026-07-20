from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .experiment_protocol import atomic_write_json, atomic_write_text, utc_now_iso


V1_VARIANT = "proposed_joint_rho025_050"
NEW_VARIANTS = [
    "joint_v1_core_point_strengthened",
    "joint_v1_stall_secondary_cut",
    "proposed_cut_strengthened_joint_v3",
]
EXPECTED_VARIANTS = [V1_VARIANT, *NEW_VARIANTS]
COMPONENT_COUNTS = {
    "joint_v1_core_point_strengthened": 1,
    "joint_v1_stall_secondary_cut": 1,
    "proposed_cut_strengthened_joint_v3": 2,
}


def _float(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    return float(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _nondecreasing(values: list[float], tolerance: float = 1.0e-7) -> bool:
    return all(right + tolerance >= left for left, right in zip(values, values[1:]))


def _nonincreasing(values: list[float], tolerance: float = 1.0e-7) -> bool:
    return all(right <= left + tolerance for left, right in zip(values, values[1:]))


def _resolve_log_path(result_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    direct = result_dir / path
    if direct.exists():
        return direct
    return result_dir / "iteration_logs" / path.name


def run_correctness_checks(result_dir: Path, row: dict[str, str]) -> dict[str, bool]:
    log_path = _resolve_log_path(result_dir, row.get("iteration_log_path", ""))
    logs = _read_csv(log_path)
    lower_bounds = [_float(item.get("LB")) for item in logs if item.get("LB") not in {None, ""}]
    upper_bounds = [_float(item.get("UB")) for item in logs if item.get("UB") not in {None, ""}]
    master_gaps = [
        _float(item.get("requested_master_mip_gap"))
        for item in logs
        if item.get("requested_master_mip_gap") not in {None, ""}
    ]
    subproblem_gaps = [
        _float(item.get("subproblem_requested_mip_gap"))
        for item in logs
        if item.get("subproblem_requested_mip_gap") not in {None, ""}
    ]
    accepted_core_rows = [item for item in logs if _bool(item.get("core_point_cut_accepted"))]
    optimal_stage2_rows = [
        item for item in logs if str(item.get("core_point_stage2_status", "")).lower() == "optimal"
    ]
    secondary_added_rows = [item for item in logs if _bool(item.get("v3_secondary_cut_added"))]
    certification_rows = [item for item in logs if _bool(item.get("final_certification_active"))]
    return {
        "iteration_log_present": bool(logs),
        "lb_nondecreasing": _nondecreasing(lower_bounds),
        "ub_nonincreasing": _nonincreasing(upper_bounds),
        "master_gap_nonincreasing": _nonincreasing(master_gaps),
        "subproblem_gap_nonincreasing": _nonincreasing(subproblem_gaps),
        "core_auxiliary_never_updates_ub": all(
            not _bool(item.get("core_point_auxiliary_bound_used_for_UB"))
            for item in logs
        ),
        "secondary_bound_never_updates_ub": all(
            not _bool(item.get("v3_secondary_bound_used_for_UB"))
            for item in logs
        ),
        "accepted_core_cuts_dual_feasible": all(
            _bool(item.get("core_point_dual_feasible"))
            for item in accepted_core_rows
        ),
        "optimal_core_stage2_current_floor_satisfied": all(
            item.get("core_point_strengthened_value_at_current") not in {None, ""}
            and item.get("core_point_current_value_floor") not in {None, ""}
            and _float(item.get("core_point_strengthened_value_at_current"))
            + 1.0e-7
            >= _float(item.get("core_point_current_value_floor"))
            for item in optimal_stage2_rows
        ),
        "optimal_core_stage2_not_weaker_than_original_beyond_delta": all(
            item.get("core_point_strengthened_value_at_current") not in {None, ""}
            and item.get("core_point_original_value_at_current") not in {None, ""}
            and item.get("core_point_stage1_objective") not in {None, ""}
            and item.get("core_point_current_value_floor") not in {None, ""}
            and _float(item.get("core_point_strengthened_value_at_current"))
            + (
                _float(item.get("core_point_stage1_objective"))
                - _float(item.get("core_point_current_value_floor"))
            )
            + 1.0e-7
            >= _float(item.get("core_point_original_value_at_current"))
            for item in optimal_stage2_rows
        ),
        "primary_cut_never_added_without_incumbent": all(
            not _bool(item.get("cut_added")) or _bool(item.get("subproblem_has_incumbent"))
            for item in logs
        ),
        "secondary_cut_never_added_without_incumbent": all(
            _bool(item.get("v3_secondary_has_incumbent"))
            for item in secondary_added_rows
        ),
        "secondary_pattern_differs_from_primary": all(
            item.get("v3_secondary_pattern_distance") not in {None, ""}
            and _float(item.get("v3_secondary_pattern_distance")) > 0.0
            for item in secondary_added_rows
        ),
        "v3_disabled_during_final_certification": all(
            not _bool(item.get("core_point_attempted"))
            and not _bool(item.get("v3_secondary_attempted"))
            for item in certification_rows
        ),
    }


def summarize_method(rows: list[dict[str, str]]) -> dict[str, Any]:
    solved = [_bool(row.get("solved_to_tolerance")) for row in rows]
    runtimes = [_float(row.get("runtime")) for row in rows]
    par2 = [_float(row.get("penalized_runtime_par2")) for row in rows]
    iterations = [_float(row.get("iterations")) for row in rows]
    core_attempts = sum(_float(row.get("core_point_attempt_count")) for row in rows)
    core_successes = sum(_float(row.get("core_point_success_count")) for row in rows)
    secondary_triggers = sum(_float(row.get("v3_secondary_trigger_count")) for row in rows)
    secondary_added = sum(_float(row.get("v3_secondary_cut_added_count")) for row in rows)
    return {
        "run_count": len(rows),
        "failed_count": sum(str(row.get("status", "")).lower() == "failed" for row in rows),
        "solved_count": sum(solved),
        "solved_rate": sum(solved) / len(rows) if rows else 0.0,
        "mean_runtime": _mean(runtimes),
        "mean_par2": _mean(par2),
        "median_par2": _median(par2),
        "mean_iterations": _mean(iterations),
        "median_iterations": _median(iterations),
        "mean_master_time": _mean([_float(row.get("master_time")) for row in rows]),
        "mean_original_subproblem_time": _mean(
            [
                max(
                    0.0,
                    _float(row.get("subproblem_time"))
                    - _float(row.get("v3_secondary_total_runtime")),
                )
                for row in rows
            ]
        ),
        "mean_core_point_time": _mean([_float(row.get("core_point_total_runtime")) for row in rows]),
        "mean_secondary_time": _mean([_float(row.get("v3_secondary_total_runtime")) for row in rows]),
        "mean_extra_cut_time": _mean([_float(row.get("v3_total_extra_cut_runtime")) for row in rows]),
        "core_point_success_rate": core_successes / core_attempts if core_attempts else None,
        "secondary_trigger_rate": secondary_triggers / len(rows) if rows else 0.0,
        "secondary_add_rate": secondary_added / secondary_triggers if secondary_triggers else None,
        "timeout_count": sum(str(row.get("status", "")).lower() == "time_limit" for row in rows),
    }


def analyze_result_directory(result_dir: str | Path, expected_run_count: int = 20) -> dict[str, Any]:
    root = Path(result_dir)
    rows = _read_csv(root / "results.csv")
    by_variant = {
        variant: [row for row in rows if row.get("variant_name") == variant]
        for variant in EXPECTED_VARIANTS
    }
    correctness_by_run: dict[str, dict[str, bool]] = {}
    for row in rows:
        key = row.get("run_key") or f"{row.get('seed')}::{row.get('variant_name')}"
        correctness_by_run[str(key)] = run_correctness_checks(root, row)
    valid_lower_bounds = all(
        row.get("lower_bound") not in {None, ""}
        and math.isfinite(_float(row.get("lower_bound")))
        for row in rows
    )
    valid_upper_bounds = all(
        _bool(row.get("valid_UB"))
        and row.get("upper_bound") not in {None, ""}
        and math.isfinite(_float(row.get("upper_bound")))
        for row in rows
    )
    per_variant_correctness = {
        variant: bool(items)
        and all(bool(row.get("status")) for row in items)
        and all(str(row.get("status", "")).lower() != "failed" for row in items)
        and all(
            row.get("final_gap") not in {None, ""}
            and math.isfinite(_float(row.get("final_gap")))
            for row in items
        )
        and all(
            row.get("lower_bound") not in {None, ""}
            and math.isfinite(_float(row.get("lower_bound")))
            for row in items
        )
        and all(
            _bool(row.get("valid_UB"))
            and row.get("upper_bound") not in {None, ""}
            and math.isfinite(_float(row.get("upper_bound")))
            for row in items
        )
        and all(
            all(correctness_by_run[
                str(row.get("run_key") or f"{row.get('seed')}::{row.get('variant_name')}")
            ].values())
            for row in items
        )
        for variant, items in by_variant.items()
    }
    return {
        "result_dir": str(root),
        "result_file_present": (root / "results.csv").exists(),
        "expected_run_count": expected_run_count,
        "actual_run_count": len(rows),
        "run_count_complete": len(rows) == expected_run_count,
        "variant_counts": {variant: len(items) for variant, items in by_variant.items()},
        "all_expected_variants_present": all(len(items) == 5 for items in by_variant.values()),
        "all_statuses_recorded": all(bool(row.get("status")) for row in rows),
        "all_final_gaps_recorded": all(row.get("final_gap") not in {None, ""} for row in rows),
        "all_runs_have_valid_lower_bound": bool(rows) and valid_lower_bounds,
        "all_runs_have_valid_ub": bool(rows) and valid_upper_bounds,
        "all_solved_runs_have_valid_ub": all(
            not _bool(row.get("solved_to_tolerance")) or _bool(row.get("valid_UB"))
            for row in rows
        ),
        "correctness_by_run": correctness_by_run,
        "all_iteration_checks_passed": bool(correctness_by_run)
        and all(all(checks.values()) for checks in correctness_by_run.values()),
        "per_variant_correctness": per_variant_correctness,
        "method_summaries": {
            variant: summarize_method(items) for variant, items in by_variant.items()
        },
        "rows": rows,
    }


def _paired_noninferior_count(
    rows: list[dict[str, str]],
    candidate: str,
    *,
    ratio: float = 1.0,
) -> int:
    by_key = {(int(row["seed"]), row["variant_name"]): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    return sum(
        _float(by_key[seed, candidate].get("penalized_runtime_par2"))
        <= ratio * _float(by_key[seed, V1_VARIANT].get("penalized_runtime_par2"))
        for seed in seeds
        if (seed, candidate) in by_key and (seed, V1_VARIANT) in by_key
    )


def _systematic_tail_degradation(rows: list[dict[str, str]], candidate: str) -> bool:
    worse = 5 - _paired_noninferior_count(rows, candidate, ratio=1.03)
    return worse >= 3


def development_candidate_judgment(
    medium: dict[str, Any],
    large: dict[str, Any],
) -> dict[str, Any]:
    medium_v1 = medium["method_summaries"][V1_VARIANT]
    large_v1 = large["method_summaries"][V1_VARIANT]
    judgments: dict[str, Any] = {}
    eligible: list[str] = []
    for variant in NEW_VARIANTS:
        medium_summary = medium["method_summaries"][variant]
        large_summary = large["method_summaries"][variant]
        correctness = (
            medium["per_variant_correctness"].get(variant, False)
            and large["per_variant_correctness"].get(variant, False)
            and medium_summary["failed_count"] == 0
            and large_summary["failed_count"] == 0
        )
        medium_gate = (
            medium_summary["solved_rate"] >= medium_v1["solved_rate"]
            and (medium_summary["mean_par2"] or float("inf"))
            <= 1.03 * (medium_v1["mean_par2"] or 0.0)
            and not _systematic_tail_degradation(medium["rows"], variant)
        )
        large_pair_count = _paired_noninferior_count(large["rows"], variant)
        large_gate = (
            large_summary["solved_rate"] >= large_v1["solved_rate"]
            and (large_summary["mean_par2"] or float("inf"))
            <= 0.95 * (large_v1["mean_par2"] or 0.0)
            and (large_summary["mean_iterations"] or float("inf"))
            <= 0.90 * (large_v1["mean_iterations"] or 0.0)
            and large_pair_count >= 3
            and all(
                row.get("v3_total_extra_cut_runtime") not in {None, ""}
                for row in large["rows"]
                if row.get("variant_name") == variant
            )
        )
        passed = correctness and medium_gate and large_gate
        if passed:
            eligible.append(variant)
        judgments[variant] = {
            "correctness_gate_passed": correctness,
            "medium_gate_passed": medium_gate,
            "large_gate_passed": large_gate,
            "large_par2_noninferior_instance_count": large_pair_count,
            "eligible": passed,
        }
    ranked = sorted(
        eligible,
        key=lambda variant: (
            large["method_summaries"][variant]["mean_par2"],
            large["method_summaries"][variant]["mean_iterations"],
            large["method_summaries"][variant]["timeout_count"],
            medium["method_summaries"][variant]["mean_par2"],
            COMPONENT_COUNTS[variant],
        ),
    )
    selected = ranked[0] if ranked else None
    if len(ranked) >= 2:
        best = large["method_summaries"][ranked[0]]["mean_par2"]
        second = large["method_summaries"][ranked[1]]["mean_par2"]
        if best and abs(second - best) / best < 0.01:
            selected = min(ranked[:2], key=lambda variant: COMPONENT_COUNTS[variant])
    return {
        "judgments": judgments,
        "eligible_candidates": ranked,
        "selected_candidate": selected,
        "decision": "continue_with_v1" if selected is None else "freeze_one_candidate",
        "configuration_or_parameter_changes_performed": False,
    }


def analyze_development(
    medium_dir: str | Path,
    large_dir: str | Path,
) -> dict[str, Any]:
    medium = analyze_result_directory(medium_dir)
    large = analyze_result_directory(large_dir)
    benefits_vs_v1: dict[str, dict[str, dict[str, float | None]]] = {}
    for scale, section in (("medium_large", medium), ("large", large)):
        v1 = section["method_summaries"][V1_VARIANT]
        benefits_vs_v1[scale] = {}
        for variant in NEW_VARIANTS:
            candidate = section["method_summaries"][variant]
            v1_par2 = v1["mean_par2"]
            v1_iterations = v1["mean_iterations"]
            benefits_vs_v1[scale][variant] = {
                "par2_reduction_fraction": (
                    (v1_par2 - candidate["mean_par2"]) / v1_par2
                    if v1_par2 and candidate["mean_par2"] is not None
                    else None
                ),
                "iteration_reduction_fraction": (
                    (v1_iterations - candidate["mean_iterations"]) / v1_iterations
                    if v1_iterations and candidate["mean_iterations"] is not None
                    else None
                ),
                "solved_rate_difference": candidate["solved_rate"] - v1["solved_rate"],
            }
    return {
        "analysis_name": "cut_strengthened_joint_v3_development",
        "created_at": utc_now_iso(),
        "read_only": True,
        "medium_large": medium,
        "large": large,
        "benefits_vs_v1": benefits_vs_v1,
        "candidate_judgment": development_candidate_judgment(medium, large),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Cut-Strengthened Joint V3 Development Analysis",
        "",
        "This report is read-only and does not modify algorithm parameters or configurations.",
        "",
        "| Scale | Runs | Complete | Iteration checks |",
        "| --- | ---: | --- | --- |",
    ]
    for key, label in (("medium_large", "medium-large"), ("large", "large")):
        section = report[key]
        lines.append(
            f"| {label} | {section['actual_run_count']}/{section['expected_run_count']} "
            f"| {section['run_count_complete']} | {section['all_iteration_checks_passed']} |"
        )
        lines.extend(
            [
                "",
                f"### {label} method summary",
                "",
                "| Method | Solved rate | Mean PAR-2 | Median PAR-2 | Mean iterations | Core time | Secondary time |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for variant in EXPECTED_VARIANTS:
            summary = section["method_summaries"][variant]
            lines.append(
                f"| `{variant}` | {summary['solved_rate']:.3f} | "
                f"{summary['mean_par2']} | {summary['median_par2']} | "
                f"{summary['mean_iterations']} | {summary['mean_core_point_time']} | "
                f"{summary['mean_secondary_time']} |"
            )
    lines.extend(["", "## Candidate judgment", ""])
    judgment = report["candidate_judgment"]
    lines.append(f"Decision: `{judgment['decision']}`")
    lines.append(f"Selected candidate: `{judgment['selected_candidate']}`")
    lines.extend(
        [
            "",
            "| Candidate | Correctness | Medium gate | Large gate | Eligible |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for variant in NEW_VARIANTS:
        item = judgment["judgments"][variant]
        lines.append(
            f"| `{variant}` | {item['correctness_gate_passed']} | "
            f"{item['medium_gate_passed']} | {item['large_gate_passed']} | "
            f"{item['eligible']} |"
        )
    lines.extend(["", "## Relative benefits versus V1", ""])
    lines.extend(
        [
            "| Scale | Candidate | PAR-2 reduction | Iteration reduction | Solved-rate difference |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for scale in ("medium_large", "large"):
        for variant in NEW_VARIANTS:
            benefit = report["benefits_vs_v1"][scale][variant]
            lines.append(
                f"| {scale} | `{variant}` | {benefit['par2_reduction_fraction']} | "
                f"{benefit['iteration_reduction_fraction']} | "
                f"{benefit['solved_rate_difference']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze V3 development results without solving.")
    parser.add_argument(
        "--medium-dir",
        default="experiments/results_cut_v3/development_medium_large",
    )
    parser.add_argument(
        "--large-dir",
        default="experiments/results_cut_v3/development_large",
    )
    parser.add_argument("--output-dir", default="experiments/results_cut_v3/development_analysis")
    args = parser.parse_args()
    report = analyze_development(args.medium_dir, args.large_dir)
    output = Path(args.output_dir)
    atomic_write_json(output / "cut_v3_development_analysis.json", report)
    atomic_write_text(
        output / "cut_v3_development_analysis.md",
        markdown_report(report),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
