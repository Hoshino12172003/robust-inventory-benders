from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats


class AnalysisIntegrityError(RuntimeError):
    """Raised when frozen final-evaluation inputs fail a required audit."""


@dataclass(frozen=True)
class AnalysisInputs:
    results: pd.DataFrame
    summary: pd.DataFrame
    resolved_config: dict[str, Any]
    iteration_logs: dict[tuple[int, str], pd.DataFrame]


REQUIRED_RESULT_COLUMNS = {
    "seed",
    "variant_name",
    "instance_size",
    "status",
    "valid_UB",
    "subproblem_mode",
    "target_subproblem_status",
    "final_gap",
    "num_subproblem_nonoptimal",
    "num_subproblem_without_incumbent",
    "gamma_target",
    "gamma_schedule",
    "max_cuts_per_iteration",
    "cut_selection_enabled",
    "adaptive_secondary_cut_selection_enabled",
    "adaptive_secondary_generation_enabled",
    "secondary_solves_attempted_total",
    "secondary_cuts_added_total",
    "secondary_cuts_skipped_total",
    "iterations",
    "objective",
    "best_bound",
    "runtime",
    "master_time",
    "subproblem_time",
    "precision_policy",
    "adaptive_master_precision_enabled",
    "adaptive_subproblem_precision_enabled",
    "fixed_master_mip_gap",
    "fixed_subproblem_mip_gap",
    "master_error_budget_ratio",
    "subproblem_error_budget_ratio",
    "final_certification_enabled",
    "final_certification_triggered",
    "final_certification_count",
    "final_certification_iterations",
    "time_to_gap_5pct",
    "time_to_gap_1pct",
    "time_to_gap_05pct",
    "time_to_gap_01pct",
}

REQUIRED_LOG_COLUMNS = {
    "iteration",
    "seed",
    "variant_name",
    "LB",
    "UB",
    "global_gap",
    "elapsed_time",
    "precision_gap_fallback_used",
    "valid_global_gap_for_precision",
    "adaptive_master_precision_enabled",
    "adaptive_subproblem_precision_enabled",
    "master_gap_selected",
    "subproblem_gap_selected",
    "requested_master_mip_gap",
    "subproblem_requested_mip_gap",
    "final_certification_active",
    "secondary_solve_attempted",
    "secondary_cuts_added_total",
    "secondary_cuts_skipped_total",
}

FORBIDDEN_REPORT_PHRASES = (
    "universally optimal",
    "dominates every instance",
    "proves the ratios are mathematically optimal",
    "exact benders for the fixed 1e-4 baseline",
    "error decomposition theorem",
)

PALETTE = {
    "standard_benders": "#0072B2",
    "static_inexact_benders": "#E69F00",
    "mp_adaptive_rho050": "#009E73",
    "sp_adaptive_rho050": "#CC79A7",
    "proposed_joint_rho025_050": "#D55E00",
}


def load_analysis_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment_name",
        "expected_seeds",
        "expected_instance_size",
        "tolerance",
        "method_order",
        "reference_method",
        "paper_labels",
        "comparison_families",
        "bootstrap_seed",
        "bootstrap_resamples",
        "confidence_level",
    }
    missing = sorted(required - set(config))
    if missing:
        raise AnalysisIntegrityError(
            f"Analysis configuration is missing: {', '.join(missing)}"
        )
    if config["reference_method"] not in config["method_order"]:
        raise AnalysisIntegrityError("reference_method is not in method_order")
    return config


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", "", "nan", "none"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def _bool_series(series: pd.Series) -> pd.Series:
    return series.map(_as_bool)


def _schedule(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [int(item) for item in parsed]
    if "," in text:
        return [int(float(item.strip())) for item in text.split(",")]
    return [int(float(text))]


def _close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    left_value = float(left)
    right_value = float(right)
    scale = max(1.0, abs(left_value), abs(right_value))
    return abs(left_value - right_value) <= tolerance * scale


def _monotone(values: pd.Series, *, increasing: bool, tolerance: float = 1e-8) -> bool:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if len(array) <= 1 or not np.isfinite(array).all():
        return len(array) <= 1 or np.isfinite(array).all()
    differences = np.diff(array)
    scales = np.maximum.reduce(
        [np.ones(len(differences)), np.abs(array[:-1]), np.abs(array[1:])]
    )
    if increasing:
        return bool(np.all(differences >= -tolerance * scales))
    return bool(np.all(differences <= tolerance * scales))


def _check_record(
    records: list[dict[str, Any]],
    name: str,
    passed: bool,
    details: str,
) -> None:
    records.append(
        {
            "check": name,
            "required": True,
            "passed": bool(passed),
            "details": details,
        }
    )


def audit_frames(
    results: pd.DataFrame,
    iteration_logs: Mapping[tuple[int, str], pd.DataFrame],
    config: Mapping[str, Any],
    resolved_config: Mapping[str, Any] | None = None,
    input_presence: Mapping[str, bool] | None = None,
) -> pd.DataFrame:
    """Return all required audit checks without modifying any input."""

    records: list[dict[str, Any]] = []
    presence = {
        "results.csv": True,
        "summary.csv": True,
        "resolved_config.yaml": True,
        "iteration_logs": True,
    }
    if input_presence is not None:
        presence.update(input_presence)
    for name, present in presence.items():
        _check_record(records, f"input_exists::{name}", present, str(present))

    missing_columns = sorted(REQUIRED_RESULT_COLUMNS - set(results.columns))
    _check_record(
        records,
        "results_required_columns",
        not missing_columns,
        "missing=" + ",".join(missing_columns),
    )
    missing_log_columns = sorted(
        REQUIRED_LOG_COLUMNS
        - set().union(*(set(frame.columns) for frame in iteration_logs.values()))
        if iteration_logs
        else REQUIRED_LOG_COLUMNS
    )
    _check_record(
        records,
        "iteration_logs_required_columns",
        not missing_log_columns,
        "missing=" + ",".join(missing_log_columns),
    )
    if missing_columns or missing_log_columns:
        return pd.DataFrame(records)

    expected_seeds = [int(seed) for seed in config["expected_seeds"]]
    methods = [str(method) for method in config["method_order"]]
    expected_grid = {(seed, method) for seed in expected_seeds for method in methods}
    seeds = pd.to_numeric(results["seed"], errors="coerce").astype("Int64")
    actual_grid = set(zip(seeds.astype(int), results["variant_name"].astype(str)))
    duplicate_count = int(results.duplicated(["seed", "variant_name"]).sum())

    _check_record(records, "results_row_count", len(results) == 50, f"rows={len(results)}")
    _check_record(
        records,
        "seed_variant_grid",
        actual_grid == expected_grid and len(results) == len(expected_grid),
        f"expected={len(expected_grid)},actual={len(actual_grid)}",
    )
    _check_record(records, "no_duplicate_runs", duplicate_count == 0, f"duplicates={duplicate_count}")
    _check_record(
        records,
        "expected_seeds",
        sorted(seeds.unique().tolist()) == expected_seeds,
        f"seeds={sorted(seeds.unique().tolist())}",
    )
    tuning = sorted(set(seeds.astype(int)) & {0, 1, 2})
    _check_record(records, "no_tuning_seeds", not tuning, f"contamination={tuning}")
    actual_methods = sorted(results["variant_name"].astype(str).unique().tolist())
    _check_record(
        records,
        "expected_variants",
        set(actual_methods) == set(methods),
        f"variants={actual_methods}",
    )
    size_values = sorted(results["instance_size"].astype(str).unique().tolist())
    _check_record(
        records,
        "expected_instance_size",
        size_values == [str(config["expected_instance_size"])],
        f"sizes={size_values}",
    )
    _check_record(
        records,
        "all_status_optimal",
        results["status"].astype(str).str.lower().eq("optimal").all(),
        str(results["status"].value_counts().to_dict()),
    )
    _check_record(
        records,
        "all_valid_ub",
        _bool_series(results["valid_UB"]).all(),
        str(_bool_series(results["valid_UB"]).value_counts().to_dict()),
    )
    _check_record(
        records,
        "robust_dual_mode",
        results["subproblem_mode"].astype(str).eq("robust_dual_milp").all(),
        str(results["subproblem_mode"].value_counts().to_dict()),
    )
    _check_record(
        records,
        "target_subproblems_optimal",
        results["target_subproblem_status"].astype(str).str.lower().eq("optimal").all(),
        str(results["target_subproblem_status"].value_counts().to_dict()),
    )
    gaps = pd.to_numeric(results["final_gap"], errors="coerce")
    tolerance = float(config["tolerance"])
    _check_record(
        records,
        "final_gap_within_tolerance",
        np.isfinite(gaps).all() and gaps.le(tolerance + 1e-12).all(),
        f"max={gaps.max()}",
    )
    for field in ("num_subproblem_nonoptimal", "num_subproblem_without_incumbent"):
        values = pd.to_numeric(results[field], errors="coerce")
        _check_record(
            records,
            f"zero::{field}",
            values.eq(0).all(),
            f"sum={values.sum()}",
        )
    _check_record(
        records,
        "gamma_target_two",
        pd.to_numeric(results["gamma_target"], errors="coerce").eq(2).all(),
        str(results["gamma_target"].value_counts().to_dict()),
    )
    schedules = results["gamma_schedule"].map(_schedule)
    _check_record(
        records,
        "target_gamma_schedule_only",
        schedules.map(lambda value: value == [2]).all(),
        str(sorted({tuple(value) for value in schedules})),
    )
    _check_record(
        records,
        "single_cut_k1",
        pd.to_numeric(results["max_cuts_per_iteration"], errors="coerce").eq(1).all(),
        str(results["max_cuts_per_iteration"].value_counts().to_dict()),
    )
    for field in (
        "cut_selection_enabled",
        "adaptive_secondary_cut_selection_enabled",
        "adaptive_secondary_generation_enabled",
    ):
        values = _bool_series(results[field])
        _check_record(records, f"disabled::{field}", (~values).all(), str(values.value_counts().to_dict()))

    result_secondary_zero = all(
        pd.to_numeric(results[field], errors="coerce").fillna(0).eq(0).all()
        for field in (
            "secondary_solves_attempted_total",
            "secondary_cuts_added_total",
            "secondary_cuts_skipped_total",
        )
    )
    log_secondary_zero = all(
        (~_bool_series(frame["secondary_solve_attempted"])).all()
        and pd.to_numeric(frame["secondary_cuts_added_total"], errors="coerce").fillna(0).eq(0).all()
        and pd.to_numeric(frame["secondary_cuts_skipped_total"], errors="coerce").fillna(0).eq(0).all()
        for frame in iteration_logs.values()
    )
    _check_record(
        records,
        "no_secondary_activity",
        result_secondary_zero and log_secondary_zero,
        f"results={result_secondary_zero},logs={log_secondary_zero}",
    )

    actual_log_keys = set(iteration_logs)
    _check_record(records, "iteration_log_count", len(iteration_logs) == 50, f"count={len(iteration_logs)}")
    _check_record(
        records,
        "iteration_log_grid",
        actual_log_keys == expected_grid,
        f"missing={sorted(expected_grid - actual_log_keys)},extra={sorted(actual_log_keys - expected_grid)}",
    )

    row_lookup = {
        (int(row.seed), str(row.variant_name)): row
        for row in results.itertuples(index=False)
    }
    count_ok = True
    identity_ok = True
    lb_ok = True
    ub_ok = True
    mp_monotone_ok = True
    sp_monotone_ok = True
    outside_equal_ok = True
    certification_zero_ok = True
    fallback_ok = True
    for key, frame in iteration_logs.items():
        ordered = frame.sort_values("iteration", kind="stable").reset_index(drop=True)
        result_row = row_lookup.get(key)
        count_ok &= result_row is not None and len(ordered) == int(result_row.iterations)
        identity_ok &= (
            pd.to_numeric(ordered["seed"], errors="coerce").eq(key[0]).all()
            and ordered["variant_name"].astype(str).eq(key[1]).all()
        )
        lb_ok &= _monotone(ordered["LB"], increasing=True)
        ub_ok &= _monotone(ordered["UB"], increasing=False)
        adaptive_master = _bool_series(ordered["adaptive_master_precision_enabled"])
        adaptive_subproblem = _bool_series(ordered["adaptive_subproblem_precision_enabled"])
        if adaptive_master.any():
            mp_monotone_ok &= _monotone(
                ordered.loc[adaptive_master, "master_gap_selected"], increasing=False
            )
        if adaptive_subproblem.any():
            sp_monotone_ok &= _monotone(
                ordered.loc[adaptive_subproblem, "subproblem_gap_selected"], increasing=False
            )
        certification = _bool_series(ordered["final_certification_active"])
        outside = ~certification
        outside_equal_ok &= all(
            _close(left, right)
            for left, right in zip(
                ordered.loc[outside, "master_gap_selected"],
                ordered.loc[outside, "requested_master_mip_gap"],
            )
        )
        outside_equal_ok &= all(
            _close(left, right)
            for left, right in zip(
                ordered.loc[outside, "subproblem_gap_selected"],
                ordered.loc[outside, "subproblem_requested_mip_gap"],
            )
        )
        if certification.any():
            certification_zero_ok &= (
                pd.to_numeric(
                    ordered.loc[certification, "requested_master_mip_gap"],
                    errors="coerce",
                ).eq(0.0).all()
                and pd.to_numeric(
                    ordered.loc[certification, "subproblem_requested_mip_gap"],
                    errors="coerce",
                ).eq(0.0).all()
            )
        first = ordered.iloc[0]
        fallback_rows = ordered[_bool_series(ordered["precision_gap_fallback_used"])]
        fallback_ok &= (
            _as_bool(first["precision_gap_fallback_used"])
            and _close(first["valid_global_gap_for_precision"], 1.0)
            and pd.to_numeric(
                fallback_rows["valid_global_gap_for_precision"], errors="coerce"
            ).map(lambda value: _close(value, 1.0)).all()
        )

    for name, value in (
        ("iteration_count_matches", count_ok),
        ("iteration_log_identity", identity_ok),
        ("lb_nondecreasing", lb_ok),
        ("ub_nonincreasing", ub_ok),
        ("adaptive_master_gap_nonincreasing", mp_monotone_ok),
        ("adaptive_subproblem_gap_nonincreasing", sp_monotone_ok),
        ("outside_certification_selected_equals_requested", outside_equal_ok),
        ("certification_requested_gaps_zero", certification_zero_ok),
        ("first_iteration_precision_fallback", fallback_ok),
    ):
        _check_record(records, name, value, str(value))

    by_variant = {name: results[results["variant_name"] == name] for name in methods}

    proposed = by_variant.get("proposed_joint_rho025_050", pd.DataFrame())
    proposed_ok = (
        len(proposed) == len(expected_seeds)
        and proposed["precision_policy"].astype(str).eq("joint_error_budget").all()
        and _bool_series(proposed["adaptive_master_precision_enabled"]).all()
        and _bool_series(proposed["adaptive_subproblem_precision_enabled"]).all()
        and pd.to_numeric(proposed["master_error_budget_ratio"], errors="coerce").map(lambda value: _close(value, 0.25)).all()
        and pd.to_numeric(proposed["subproblem_error_budget_ratio"], errors="coerce").map(lambda value: _close(value, 0.50)).all()
    )
    _check_record(records, "proposed_policy_exact", proposed_ok, str(proposed_ok))

    mp_only = by_variant.get("mp_adaptive_rho050", pd.DataFrame())
    mp_ok = (
        len(mp_only) == len(expected_seeds)
        and _bool_series(mp_only["adaptive_master_precision_enabled"]).all()
        and (~_bool_series(mp_only["adaptive_subproblem_precision_enabled"])).all()
        and pd.to_numeric(mp_only["master_error_budget_ratio"], errors="coerce").map(lambda value: _close(value, 0.50)).all()
        and pd.to_numeric(mp_only["fixed_subproblem_mip_gap"], errors="coerce").map(lambda value: _close(value, 0.0001)).all()
    )
    _check_record(records, "mp_only_policy_exact", mp_ok, str(mp_ok))

    sp_only = by_variant.get("sp_adaptive_rho050", pd.DataFrame())
    sp_ok = (
        len(sp_only) == len(expected_seeds)
        and (~_bool_series(sp_only["adaptive_master_precision_enabled"])).all()
        and _bool_series(sp_only["adaptive_subproblem_precision_enabled"]).all()
        and pd.to_numeric(sp_only["subproblem_error_budget_ratio"], errors="coerce").map(lambda value: _close(value, 0.50)).all()
        and pd.to_numeric(sp_only["fixed_master_mip_gap"], errors="coerce").map(lambda value: _close(value, 0.0001)).all()
    )
    _check_record(records, "sp_only_policy_exact", sp_ok, str(sp_ok))

    standard = by_variant.get("standard_benders", pd.DataFrame())
    standard_ok = (
        len(standard) == len(expected_seeds)
        and (~_bool_series(standard["adaptive_master_precision_enabled"])).all()
        and (~_bool_series(standard["adaptive_subproblem_precision_enabled"])).all()
        and pd.to_numeric(standard["fixed_master_mip_gap"], errors="coerce").map(lambda value: _close(value, 0.0001)).all()
        and pd.to_numeric(standard["fixed_subproblem_mip_gap"], errors="coerce").map(lambda value: _close(value, 0.0001)).all()
        and (~_bool_series(standard["final_certification_enabled"])).all()
    )
    _check_record(
        records,
        "standard_baseline_matches_frozen_spec",
        standard_ok,
        str(standard_ok),
    )

    static = by_variant.get("static_inexact_benders", pd.DataFrame())
    patience = None if resolved_config is None else resolved_config.get("final_certification_no_cut_patience")
    static_ok = (
        len(static) == len(expected_seeds)
        and (~_bool_series(static["adaptive_master_precision_enabled"])).all()
        and (~_bool_series(static["adaptive_subproblem_precision_enabled"])).all()
        and pd.to_numeric(static["fixed_master_mip_gap"], errors="coerce").map(lambda value: _close(value, 0.02)).all()
        and pd.to_numeric(static["fixed_subproblem_mip_gap"], errors="coerce").map(lambda value: _close(value, 0.02)).all()
        and _bool_series(static["final_certification_enabled"]).all()
        and patience == 2
    )
    _check_record(
        records,
        "static_baseline_matches_frozen_spec",
        static_ok,
        f"passed={static_ok},patience={patience}",
    )

    objective_ok = True
    objective_details: list[str] = []
    for seed, group in results.groupby("seed", sort=True):
        lower = pd.to_numeric(group["best_bound"], errors="coerce").to_numpy(float)
        upper = pd.to_numeric(group["objective"], errors="coerce").to_numpy(float)
        scale = max(1.0, float(np.max(np.abs(np.concatenate([lower, upper])))))
        consistent = (
            np.isfinite(lower).all()
            and np.isfinite(upper).all()
            and np.all(lower <= upper + tolerance * scale)
            and float(np.max(lower)) <= float(np.min(upper)) + tolerance * scale
        )
        objective_ok &= bool(consistent)
        objective_details.append(f"{int(seed)}:{consistent}")
    _check_record(
        records,
        "objective_bound_intervals_consistent",
        objective_ok,
        ";".join(objective_details),
    )
    return pd.DataFrame(records)


def assert_audit_passes(audit: pd.DataFrame) -> None:
    failed = audit.loc[~audit["passed"].astype(bool), "check"].astype(str).tolist()
    if failed:
        raise AnalysisIntegrityError("Required audit checks failed: " + ", ".join(failed))


def load_inputs(input_dir: Path) -> tuple[AnalysisInputs, dict[str, bool]]:
    paths = {
        "results.csv": input_dir / "results.csv",
        "summary.csv": input_dir / "summary.csv",
        "resolved_config.yaml": input_dir / "resolved_config.yaml",
        "iteration_logs": input_dir / "iteration_logs",
    }
    presence = {name: path.exists() for name, path in paths.items()}
    missing = [name for name, present in presence.items() if not present]
    if missing:
        raise AnalysisIntegrityError("Missing required final-evaluation input: " + ", ".join(missing))
    results = pd.read_csv(paths["results.csv"])
    summary = pd.read_csv(paths["summary.csv"])
    resolved = yaml.safe_load(paths["resolved_config.yaml"].read_text(encoding="utf-8"))
    logs: dict[tuple[int, str], pd.DataFrame] = {}
    for path in sorted(paths["iteration_logs"].glob("*.csv"), key=lambda item: item.name):
        frame = pd.read_csv(path)
        if frame.empty or "seed" not in frame or "variant_name" not in frame:
            raise AnalysisIntegrityError(f"Invalid iteration log schema: {path.name}")
        seed_values = pd.to_numeric(frame["seed"], errors="raise").astype(int).unique()
        variant_values = frame["variant_name"].astype(str).unique()
        if len(seed_values) != 1 or len(variant_values) != 1:
            raise AnalysisIntegrityError(f"Mixed seed or variant in iteration log: {path.name}")
        key = (int(seed_values[0]), str(variant_values[0]))
        if key in logs:
            raise AnalysisIntegrityError(f"Duplicate iteration log for seed/variant: {key}")
        logs[key] = frame
    return AnalysisInputs(results, summary, resolved, logs), presence


def runtime_ranks(results: pd.DataFrame) -> pd.DataFrame:
    output = results[["seed", "variant_name", "runtime"]].copy()
    output["runtime"] = pd.to_numeric(output["runtime"], errors="raise")
    output["runtime_rank"] = output.groupby("seed")["runtime"].rank(
        method="average", ascending=True
    )
    return output.sort_values(["seed", "runtime_rank", "variant_name"], kind="stable")


def method_summary(results: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in config["method_order"]:
        group = results[results["variant_name"] == method].copy()
        runtime = pd.to_numeric(group["runtime"], errors="coerce")
        rows.append(
            {
                "method": method,
                "paper_label_english": config["paper_labels"][method]["English"],
                "paper_label_chinese": config["paper_labels"][method]["Chinese"],
                "n": len(group),
                "optimal_count": int(group["status"].astype(str).str.lower().eq("optimal").sum()),
                "valid_ub_count": int(_bool_series(group["valid_UB"]).sum()),
                "runtime_mean": runtime.mean(),
                "runtime_sd_sample": runtime.std(ddof=1),
                "runtime_median": runtime.median(),
                "runtime_q1": runtime.quantile(0.25),
                "runtime_q3": runtime.quantile(0.75),
                "runtime_min": runtime.min(),
                "runtime_max": runtime.max(),
                "mean_iterations": pd.to_numeric(group["iterations"], errors="coerce").mean(),
                "mean_master_time": pd.to_numeric(group["master_time"], errors="coerce").mean(),
                "mean_subproblem_time": pd.to_numeric(group["subproblem_time"], errors="coerce").mean(),
                "mean_time_to_gap_5pct": pd.to_numeric(group["time_to_gap_5pct"], errors="coerce").mean(),
                "mean_time_to_gap_1pct": pd.to_numeric(group["time_to_gap_1pct"], errors="coerce").mean(),
                "mean_time_to_gap_05pct": pd.to_numeric(group["time_to_gap_05pct"], errors="coerce").mean(),
                "mean_time_to_gap_01pct": pd.to_numeric(group["time_to_gap_01pct"], errors="coerce").mean(),
                "final_certification_trigger_count": int(_bool_series(group["final_certification_triggered"]).sum()),
                "total_certification_count": int(pd.to_numeric(group["final_certification_count"], errors="coerce").sum()),
                "mean_final_gap": pd.to_numeric(group["final_gap"], errors="coerce").mean(),
            }
        )
    return pd.DataFrame(rows)


def certification_summary(results: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for method in config["method_order"]:
        group = results[results["variant_name"] == method]
        triggered = _bool_series(group["final_certification_triggered"])
        counts = pd.to_numeric(group["final_certification_count"], errors="coerce")
        iterations = pd.to_numeric(group["final_certification_iterations"], errors="coerce")
        rows.append(
            {
                "method": method,
                "paper_label_english": config["paper_labels"][method]["English"],
                "n": len(group),
                "trigger_count": int(triggered.sum()),
                "trigger_rate": float(triggered.mean()),
                "total_certification_count": int(counts.sum()),
                "mean_certification_count": float(counts.mean()),
                "mean_certification_iterations": float(iterations.mean()),
                "max_certification_count": int(counts.max()),
            }
        )
    return pd.DataFrame(rows)


def objective_consistency(results: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows = []
    for seed, group in results.groupby("seed", sort=True):
        lower = pd.to_numeric(group["best_bound"], errors="coerce")
        upper = pd.to_numeric(group["objective"], errors="coerce")
        scale = max(1.0, lower.abs().max(), upper.abs().max())
        rows.append(
            {
                "seed": int(seed),
                "max_lower_bound": lower.max(),
                "min_upper_bound": upper.min(),
                "interval_overlap": lower.max() <= upper.min() + tolerance * scale,
                "objective_range": upper.max() - upper.min(),
                "bound_range": lower.max() - lower.min(),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_interval(
    values: np.ndarray | list[float],
    statistic: Callable[[np.ndarray], float],
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a finite nonempty vector")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    batch_size = min(10000, resamples)
    offset = 0
    while offset < resamples:
        count = min(batch_size, resamples - offset)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        samples = array[indices]
        estimates[offset : offset + count] = np.apply_along_axis(statistic, 1, samples)
        offset += count
    alpha = 1.0 - confidence_level
    return (
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[~np.isclose(differences, 0.0, atol=1e-12, rtol=0.0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def wilcoxon_signed_rank(
    differences: np.ndarray | list[float],
) -> dict[str, float | str | bool]:
    """Run the prespecified two-sided paired test and expose its actual method."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Wilcoxon differences must be a finite nonempty vector")
    has_zero = bool(np.isclose(values, 0.0, atol=1e-12, rtol=0.0).any())
    calculation_method = "approx" if has_zero else "exact"
    if np.isclose(values, 0.0, atol=1e-12, rtol=0.0).all():
        statistic, p_value = 0.0, 1.0
    else:
        test = stats.wilcoxon(
            values,
            zero_method="pratt",
            correction=False,
            alternative="two-sided",
            method=calculation_method,
        )
        statistic, p_value = float(test.statistic), float(test.pvalue)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "zero_method": "pratt",
        "alternative": "two-sided",
        "continuity_correction": False,
        "calculation_method": calculation_method,
    }


def holm_adjust_by_family(
    frame: pd.DataFrame,
    *,
    family_column: str = "comparison_family",
    p_column: str = "raw_p_value",
) -> pd.DataFrame:
    output = frame.copy()
    output["holm_adjusted_p_value"] = np.nan
    for _, group in output.groupby(family_column, sort=False):
        ordered = group.sort_values([p_column, "comparator"], kind="stable")
        count = len(ordered)
        scaled = np.array(
            [min(1.0, (count - index) * float(value)) for index, value in enumerate(ordered[p_column])]
        )
        adjusted = np.maximum.accumulate(scaled)
        output.loc[ordered.index, "holm_adjusted_p_value"] = adjusted
    return output


def paired_comparisons(
    results: pd.DataFrame,
    ranks: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = str(config["reference_method"])
    rank_lookup = ranks.set_index(["seed", "variant_name"])["runtime_rank"]
    family_lookup: dict[str, str] = {}
    for family, comparisons in config["comparison_families"].items():
        for left, right in comparisons:
            if left != reference:
                raise AnalysisIntegrityError(
                    f"Comparison family {family} does not use reference method first"
                )
            family_lookup[str(right)] = str(family)

    pivot = results.pivot(index="seed", columns="variant_name", values="runtime").astype(float)
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    comparators = [method for method in config["method_order"] if method != reference]
    for comparator_index, comparator in enumerate(comparators):
        proposed_runtime = pivot[reference]
        comparator_runtime = pivot[comparator]
        differences = (proposed_runtime - comparator_runtime).to_numpy(float)
        paired_savings = 100.0 * (
            comparator_runtime - proposed_runtime
        ) / comparator_runtime
        aggregate_saving = 100.0 * (
            float(comparator_runtime.mean()) - float(proposed_runtime.mean())
        ) / float(comparator_runtime.mean())
        for seed in pivot.index:
            difference = float(proposed_runtime.loc[seed] - comparator_runtime.loc[seed])
            outcome = "win" if difference < -1e-12 else "loss" if difference > 1e-12 else "tie"
            detail_rows.append(
                {
                    "seed": int(seed),
                    "reference_method": reference,
                    "comparator": comparator,
                    "proposed_runtime": float(proposed_runtime.loc[seed]),
                    "comparator_runtime": float(comparator_runtime.loc[seed]),
                    "difference_proposed_minus_comparator": difference,
                    "percentage_difference_vs_comparator": 100.0 * difference / float(comparator_runtime.loc[seed]),
                    "outcome": outcome,
                    "proposed_runtime_rank": float(rank_lookup.loc[(seed, reference)]),
                    "comparator_runtime_rank": float(rank_lookup.loc[(seed, comparator)]),
                }
            )
        mean_ci = bootstrap_interval(
            differences,
            np.mean,
            seed=int(config["bootstrap_seed"]) + comparator_index * 10,
            resamples=int(config["bootstrap_resamples"]),
            confidence_level=float(config["confidence_level"]),
        )
        median_ci = bootstrap_interval(
            differences,
            np.median,
            seed=int(config["bootstrap_seed"]) + comparator_index * 10 + 1,
            resamples=int(config["bootstrap_resamples"]),
            confidence_level=float(config["confidence_level"]),
        )
        wilcoxon = wilcoxon_signed_rank(differences)
        summary_rows.append(
            {
                "reference_method": reference,
                "comparator": comparator,
                "comparison_family": family_lookup[comparator],
                "n_pairs": len(differences),
                "wins": int(np.sum(differences < -1e-12)),
                "losses": int(np.sum(differences > 1e-12)),
                "ties": int(np.sum(np.isclose(differences, 0.0, atol=1e-12, rtol=0.0))),
                "paired_mean_difference": float(np.mean(differences)),
                "paired_median_difference": float(np.median(differences)),
                "mean_paired_percentage_saving_percent": float(
                    np.mean(paired_savings)
                ),
                "median_paired_percentage_saving_percent": float(
                    np.median(paired_savings)
                ),
                "aggregate_mean_runtime_saving_percent": aggregate_saving,
                "bootstrap_mean_ci_lower": mean_ci[0],
                "bootstrap_mean_ci_upper": mean_ci[1],
                "bootstrap_median_ci_lower": median_ci[0],
                "bootstrap_median_ci_upper": median_ci[1],
                "wilcoxon_statistic": wilcoxon["statistic"],
                "raw_p_value": wilcoxon["p_value"],
                "wilcoxon_zero_method": wilcoxon["zero_method"],
                "wilcoxon_alternative": wilcoxon["alternative"],
                "wilcoxon_continuity_correction": wilcoxon[
                    "continuity_correction"
                ],
                "wilcoxon_calculation_method": wilcoxon["calculation_method"],
                "paired_rank_biserial": _rank_biserial(differences),
            }
        )
    details = pd.DataFrame(detail_rows)
    summary = holm_adjust_by_family(pd.DataFrame(summary_rows))
    statistical = summary[
        [
            "reference_method",
            "comparator",
            "comparison_family",
            "wilcoxon_statistic",
            "raw_p_value",
            "holm_adjusted_p_value",
            "wilcoxon_zero_method",
            "wilcoxon_alternative",
            "wilcoxon_continuity_correction",
            "wilcoxon_calculation_method",
            "paired_rank_biserial",
        ]
    ].copy()
    return details, summary, statistical


def performance_profile(
    results: pd.DataFrame,
    method_order: list[str],
    tau_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    pivot = results.pivot(index="seed", columns="variant_name", values="runtime").astype(float)
    ratios = pivot.div(pivot.min(axis=1), axis=0)
    if tau_grid is None:
        maximum = max(1.0, float(np.ceil(ratios.max().max() * 100.0) / 100.0))
        tau_grid = np.linspace(1.0, maximum, 300)
    rows = []
    for method in method_order:
        for tau in tau_grid:
            rows.append(
                {
                    "method": method,
                    "tau": float(tau),
                    "fraction": float((ratios[method] <= tau + 1e-12).mean()),
                }
            )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    results: pd.DataFrame,
    iteration_logs: Mapping[tuple[int, str], pd.DataFrame],
    config: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(config["method_order"])
    labels = {method: config["paper_labels"][method]["English"] for method in methods}
    positions = np.arange(len(methods))
    generated: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 5.5))
    runtime_values = [
        pd.to_numeric(results.loc[results["variant_name"] == method, "runtime"], errors="coerce").to_numpy(float)
        for method in methods
    ]
    box = ax.boxplot(runtime_values, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(PALETTE[method])
        patch.set_alpha(0.45)
    for index, (method, values) in enumerate(zip(methods, runtime_values)):
        offsets = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(index + offsets, values, color=PALETTE[method], edgecolor="black", linewidth=0.3, s=30, zorder=3)
    ax.set_xticks(positions, [labels[method] for method in methods], rotation=18, ha="right")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Held-out runtime distribution")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, output_dir / "runtime_distribution")

    reference = str(config["reference_method"])
    comparators = [method for method in methods if method != reference]
    pivot = results.pivot(index="seed", columns="variant_name", values="runtime").astype(float)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for index, comparator in enumerate(comparators):
        differences = pivot[reference] - pivot[comparator]
        offsets = np.linspace(-0.10, 0.10, len(differences))
        ax.scatter(index + offsets, differences, color=PALETTE[comparator], edgecolor="black", linewidth=0.3, s=34, label=labels[comparator])
        ax.plot([index - 0.15, index + 0.15], [differences.median()] * 2, color="black", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(np.arange(len(comparators)), [labels[method] for method in comparators], rotation=16, ha="right")
    ax.set_ylabel("Joint minus comparator runtime (seconds)")
    ax.set_title("Paired held-out runtime differences")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, output_dir / "paired_runtime")

    profile = performance_profile(results, methods)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in methods:
        subset = profile[profile["method"] == method]
        ax.step(subset["tau"], subset["fraction"], where="post", color=PALETTE[method], linewidth=2, label=labels[method])
    ax.set_xlim(left=1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Performance ratio (runtime / best seed runtime)")
    ax.set_ylabel("Fraction of seeds")
    ax.set_title("Held-out runtime performance profile")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save_figure(fig, output_dir / "performance_profile")

    time_grid = np.arange(0.0, 601.0, 1.0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in methods:
        trajectories = []
        for seed in config["expected_seeds"]:
            frame = iteration_logs[(int(seed), method)].sort_values("elapsed_time")
            elapsed = pd.to_numeric(frame["elapsed_time"], errors="coerce").to_numpy(float)
            gap = pd.to_numeric(frame["global_gap"], errors="coerce").to_numpy(float)
            series = pd.Series(gap, index=elapsed)
            series = series[~series.index.duplicated(keep="last")].sort_index()
            reindexed = series.reindex(series.index.union(time_grid)).sort_index().ffill().reindex(time_grid)
            trajectories.append(reindexed.to_numpy(float))
        values = np.asarray(trajectories, dtype=float)
        observed = np.any(np.isfinite(values), axis=0)
        observed_grid = time_grid[observed]
        observed_values = values[:, observed]
        median = np.nanmedian(observed_values, axis=0)
        q1 = np.nanquantile(observed_values, 0.25, axis=0)
        q3 = np.nanquantile(observed_values, 0.75, axis=0)
        floor = float(config["tolerance"]) / 100.0
        ax.plot(observed_grid, np.maximum(median, floor), color=PALETTE[method], linewidth=2, label=labels[method])
        ax.fill_between(observed_grid, np.maximum(q1, floor), np.maximum(q3, floor), color=PALETTE[method], alpha=0.13)
    ax.axhline(float(config["tolerance"]), color="black", linestyle="--", linewidth=1.2, label="Termination tolerance")
    ax.set_yscale("log")
    ax.set_xlim(0, 600)
    ax.set_xlabel("Elapsed time (seconds)")
    ax.set_ylabel("Global gap")
    ax.set_title("Held-out convergence trajectories")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, ncol=2)
    _save_figure(fig, output_dir / "convergence_gap")

    means = results.groupby("variant_name")[["runtime", "master_time", "subproblem_time"]].mean().reindex(methods)
    residual = (means["runtime"] - means["master_time"] - means["subproblem_time"]).clip(lower=0.0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(positions, means["master_time"], label="Master", color="#56B4E9")
    ax.bar(positions, means["subproblem_time"], bottom=means["master_time"], label="Robust subproblem", color="#E69F00")
    ax.bar(positions, residual, bottom=means["master_time"] + means["subproblem_time"], label="Residual overhead", color="#999999")
    ax.set_xticks(positions, [labels[method] for method in methods], rotation=18, ha="right")
    ax.set_ylabel("Mean time (seconds)")
    ax.set_title("Mean runtime decomposition")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, output_dir / "time_decomposition")

    cert = certification_summary(results, config).set_index("method").reindex(methods)
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    width = 0.38
    ax1.bar(positions - width / 2, cert["trigger_rate"], width, color="#0072B2", label="Trigger rate")
    ax1.set_ylabel("Certification trigger rate")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.bar(positions + width / 2, cert["total_certification_count"], width, color="#D55E00", label="Total certification count")
    ax2.set_ylabel("Total certification count")
    ax1.set_xticks(positions, [labels[method] for method in methods], rotation=18, ha="right")
    ax1.set_title("Final-certification behavior")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.grid(axis="y", alpha=0.25)
    _save_figure(fig, output_dir / "certification_counts")

    for stem in (
        "runtime_distribution",
        "paired_runtime",
        "performance_profile",
        "convergence_gap",
        "time_decomposition",
        "certification_counts",
    ):
        generated.extend([output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"])
    return generated


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(repo_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_input_manifest(
    input_dir: Path,
    analysis_config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    files = [input_dir / "results.csv", input_dir / "summary.csv", input_dir / "resolved_config.yaml"]
    files.extend(sorted((input_dir / "instances").glob("*.json"), key=lambda path: path.name))
    files.extend(sorted((input_dir / "iteration_logs").glob("*.csv"), key=lambda path: path.name))
    archive = repo_root / "final_evaluation_joint_v1_results.zip"
    if archive.exists():
        files.append(archive)
    entries = []
    for path in sorted(files, key=lambda item: item.relative_to(repo_root).as_posix()):
        entries.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "analysis_configuration": {
            "path": analysis_config_path.relative_to(repo_root).as_posix(),
            "sha256": _sha256(analysis_config_path),
        },
        "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "final_algorithm_frozen_v1_peeled_commit": _git_value(
            repo_root, "rev-parse", "final-algorithm-frozen-v1^{}"
        ),
        "files": entries,
    }


def write_manifest(manifest: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    manifest_path = output_dir / "input_manifest.json"
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    checksum_path = output_dir / "input_manifest.sha256"
    checksum_path.write_text(f"{digest}  input_manifest.json\n", encoding="utf-8", newline="\n")
    return manifest_path, checksum_path


def generate_report(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    certification: pd.DataFrame,
    config: Mapping[str, Any],
) -> str:
    reference = str(config["reference_method"])
    summary_by_method = summary.set_index("method")
    reference_row = summary_by_method.loc[reference]
    fastest_mean = summary.loc[summary["runtime_mean"].idxmin(), "method"]
    fastest_median = summary.loc[summary["runtime_median"].idxmin(), "method"]
    decomposition = {
        method: (
            float(row.runtime_mean),
            float(row.mean_master_time),
            float(row.mean_subproblem_time),
        )
        for method, row in summary_by_method.iterrows()
    }
    primary_lines = []
    secondary_lines = []
    for row in comparisons.itertuples(index=False):
        comparator_label = config["paper_labels"][row.comparator]["English"]
        line = (
            f"- Versus {comparator_label}: wins/losses/ties = "
            f"{row.wins}/{row.losses}/{row.ties}; aggregate mean-runtime saving = "
            f"{row.aggregate_mean_runtime_saving_percent:.2f}%; paired mean difference "
            f"{row.paired_mean_difference:.3f} s (bootstrap 95% CI "
            f"[{row.bootstrap_mean_ci_lower:.3f}, {row.bootstrap_mean_ci_upper:.3f}] s). "
            f"The mean and median seed-level paired percentage savings are "
            f"{row.mean_paired_percentage_saving_percent:.2f}% and "
            f"{row.median_paired_percentage_saving_percent:.2f}%, respectively; these "
            f"pairwise percentage summaries are distinct from the aggregate saving. "
            f"Raw p = {row.raw_p_value:.6g}; "
            f"Holm-adjusted p = {row.holm_adjusted_p_value:.6g}."
        )
        if row.comparison_family == "primary_confirmatory":
            if row.holm_adjusted_p_value < 0.05:
                line += " This primary comparison meets the prespecified adjusted 0.05 criterion."
            else:
                line += " This primary comparison does not meet the prespecified adjusted 0.05 criterion."
            primary_lines.append(line)
        else:
            line += " This ablation comparison is interpreted as exploratory."
            secondary_lines.append(line)

    static_cert = certification.set_index("method").loc["static_inexact_benders"]
    speed_statement = (
        "The joint method has the lowest mean and median runtime in these held-out runs."
        if fastest_mean == reference and fastest_median == reference
        else (
            f"The lowest mean runtime is observed for `{fastest_mean}`, while the lowest "
            f"median runtime is observed for `{fastest_median}`."
        )
    )
    wilcoxon_methods = ", ".join(
        sorted(set(comparisons["wilcoxon_calculation_method"].astype(str)))
    )
    report = f"""# Final evaluation report: {config['experiment_name']}

## 1. Experimental integrity

All required seed-by-method cells, frozen settings, valid bounds, and iteration logs passed the prespecified audit. The analysis uses ten held-out seeds (10--19) at the medium-large size and does not rerun the optimization experiment.

## 2. Correctness and certification

All 50 runs terminated with an optimal solver status, a valid conservative upper bound, and a final global gap at or below {float(config['tolerance']):.1e}. Bound intervals overlap within the declared numerical tolerance. The standard baseline is described as tight-tolerance inexact Benders.

## 3. Runtime comparison

{speed_statement} The joint method's mean runtime is {float(reference_row.runtime_mean):.3f} s and its median runtime is {float(reference_row.runtime_median):.3f} s (sample SD {float(reference_row.runtime_sd_sample):.3f} s).

## 4. Paired comparison

### Primary confirmatory family

{chr(10).join(primary_lines)}

### Secondary ablation family

{chr(10).join(secondary_lines)}

## 5. Statistical inference

Wilcoxon signed-rank tests are two-sided, use the Pratt zero-difference convention, disable continuity correction, and record the actual calculation method ({wilcoxon_methods} in these comparisons). Exact calculation is used when there are no zero differences; an approximate Pratt calculation is used when zero differences are present. Holm adjustment is applied separately within the primary and secondary families. Bootstrap intervals use {int(config['bootstrap_resamples'])} deterministic percentile resamples at the {100 * float(config['confidence_level']):.0f}% level. Adjusted p-values, rather than raw p-values alone, govern confirmatory wording.

## 6. MP/SP time decomposition

For the joint method, mean total/master/robust-subproblem times are {decomposition[reference][0]:.3f}/{decomposition[reference][1]:.3f}/{decomposition[reference][2]:.3f} s. Differences in total runtime can arise from lower per-iteration computational cost rather than fewer iterations; iteration counts and component times should therefore be read together.

## 7. Convergence-stage interpretation

The convergence figure uses forward-filled observed solver states on a common 0--600 s grid, without interpolation between observations. After a run completes, its terminal observed gap is carried forward over the remainder of the common grid. Median trajectories and interquartile bands summarize the timing of progress but do not establish a continuous-time convergence law.

## 8. Final-certification behavior

Static inexact Benders triggered certification in {int(static_cert.trigger_count)} of {int(static_cert.n)} runs and accumulated {int(static_cert.total_certification_count)} certification episodes. Repeated exact certification under fixed loose precision indicates late-stage difficulty and should not be interpreted as a defect in bound validity.

## 9. Limitations

The inference is conditional on ten medium-large held-out instances, a single implementation, and the prespecified comparison families. Runtime measurements may depend on the software and hardware environment. The selected ratios are empirically frozen policy choices, not a mathematical optimality result.

## 10. Paper-ready conclusions

The held-out analysis supports cautious comparison of joint adaptive inexact Benders with tight-tolerance, static-inexact, and one-sided adaptive alternatives. Claims about runtime advantage are restricted to the computed paired evidence and Holm-adjusted tests. Secondary MP-only and SP-only comparisons remain ablation evidence.
"""
    lowered = report.lower()
    present = [phrase for phrase in FORBIDDEN_REPORT_PHRASES if phrase in lowered]
    if present:
        raise AnalysisIntegrityError("Generated report contains forbidden wording: " + ", ".join(present))
    return report


def _latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _paper_latex_table(
    headers: list[str], rows: list[list[str]], alignment: str
) -> str:
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def method_summary_latex(summary: pd.DataFrame) -> str:
    """Return the compact paper-facing descriptive table."""
    rows: list[list[str]] = []
    for row in summary.itertuples(index=False):
        rows.append(
            [
                _latex_escape(row.paper_label_english),
                f"{int(row.n)}",
                f"{row.runtime_mean:.3f} \\(\\pm\\) {row.runtime_sd_sample:.3f}",
                f"{row.runtime_median:.3f}",
                f"{row.runtime_q1:.3f}--{row.runtime_q3:.3f}",
                f"{row.mean_iterations:.1f}",
                f"{row.mean_master_time:.3f}",
                f"{row.mean_subproblem_time:.3f}",
                f"{row.mean_time_to_gap_1pct:.3f}",
                f"{row.mean_time_to_gap_01pct:.3f}",
                f"{int(row.final_certification_trigger_count)}",
            ]
        )
    headers = [
        "Method",
        r"$n$",
        r"Runtime mean \(\pm\) sample SD (s)",
        "Runtime median (s)",
        "Runtime Q1--Q3 (s)",
        "Mean iterations",
        "Mean master time (s)",
        "Mean subproblem time (s)",
        r"Mean time to 1\% gap (s)",
        r"Mean time to 0.1\% gap (s)",
        "Certification triggers",
    ]
    return _paper_latex_table(headers, rows, "l" + "r" * 10)


def paired_comparison_summary_latex(
    comparisons: pd.DataFrame, config: Mapping[str, Any]
) -> str:
    """Return the compact paper-facing paired-comparison table."""
    family_labels = {
        "primary_confirmatory": "Primary confirmatory",
        "secondary_ablation": "Secondary ablation",
    }
    rows: list[list[str]] = []
    for row in comparisons.itertuples(index=False):
        comparator_label = config["paper_labels"][row.comparator]["English"]
        family_label = family_labels.get(
            row.comparison_family, str(row.comparison_family)
        )
        rows.append(
            [
                _latex_escape(comparator_label),
                _latex_escape(family_label),
                f"{row.wins}/{row.losses}/{row.ties}",
                f"{row.paired_mean_difference:.3f} "
                f"[{row.bootstrap_mean_ci_lower:.3f}, {row.bootstrap_mean_ci_upper:.3f}]",
                f"{row.paired_median_difference:.3f} "
                f"[{row.bootstrap_median_ci_lower:.3f}, {row.bootstrap_median_ci_upper:.3f}]",
                f"{row.raw_p_value:.6g}",
                f"{row.holm_adjusted_p_value:.6g}",
                f"{row.paired_rank_biserial:.3f}",
            ]
        )
    headers = [
        "Comparator",
        "Family",
        "W/L/T",
        r"Paired mean difference [95\% CI] (s)",
        r"Paired median difference [95\% CI] (s)",
        "Raw Wilcoxon $p$",
        "Holm-adjusted $p$",
        "Paired rank-biserial",
    ]
    return _paper_latex_table(headers, rows, "ll" + "r" * 6)


def _write_table(frame: pd.DataFrame, csv_path: Path) -> None:
    frame.to_csv(csv_path, index=False, lineterminator="\n")


def run_analysis(
    input_dir: Path,
    analysis_config_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    input_dir = input_dir.resolve()
    analysis_config_path = analysis_config_path.resolve()
    output_dir = output_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    config = load_analysis_config(analysis_config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs, presence = load_inputs(input_dir)
    audit = audit_frames(
        inputs.results,
        inputs.iteration_logs,
        config,
        resolved_config=inputs.resolved_config,
        input_presence=presence,
    )
    audit_path = output_dir / "audit_checks.csv"
    audit.to_csv(audit_path, index=False, lineterminator="\n")
    failed = audit.loc[~audit["passed"].astype(bool), "check"].astype(str).tolist()
    audit_summary = {
        "all_required_checks_passed": not failed,
        "required_check_count": int(len(audit)),
        "passed_check_count": int(audit["passed"].astype(bool).sum()),
        "failed_checks": failed,
    }
    audit_summary_path = output_dir / "audit_summary.json"
    audit_summary_path.write_text(
        json.dumps(audit_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    assert_audit_passes(audit)

    manifest = build_input_manifest(input_dir, analysis_config_path, repo_root)
    manifest_path, manifest_checksum_path = write_manifest(manifest, output_dir)

    summary = method_summary(inputs.results, config)
    seed_results = inputs.results.sort_values(["seed", "variant_name"], kind="stable")
    ranks = runtime_ranks(inputs.results)
    cert = certification_summary(inputs.results, config)
    objective = objective_consistency(inputs.results, float(config["tolerance"]))
    details, comparison_summary, statistical = paired_comparisons(inputs.results, ranks, config)

    _write_table(summary, output_dir / "method_summary.csv")
    (output_dir / "method_summary.tex").write_text(
        method_summary_latex(summary), encoding="utf-8", newline="\n"
    )
    _write_table(seed_results, output_dir / "seed_level_results.csv")
    _write_table(ranks, output_dir / "seed_runtime_ranks.csv")
    _write_table(cert, output_dir / "certification_summary.csv")
    _write_table(objective, output_dir / "objective_consistency.csv")
    _write_table(details, output_dir / "paired_runtime_differences.csv")
    _write_table(comparison_summary, output_dir / "paired_comparison_summary.csv")
    (output_dir / "paired_comparison_summary.tex").write_text(
        paired_comparison_summary_latex(comparison_summary, config),
        encoding="utf-8",
        newline="\n",
    )
    _write_table(statistical, output_dir / "statistical_tests.csv")

    figures = generate_figures(
        inputs.results,
        inputs.iteration_logs,
        config,
        output_dir / "figures",
    )
    report_path = output_dir / "final_evaluation_report.md"
    report_path.write_text(
        generate_report(summary, comparison_summary, cert, config),
        encoding="utf-8",
        newline="\n",
    )
    outputs = {
        "audit_checks": audit_path,
        "audit_summary": audit_summary_path,
        "manifest": manifest_path,
        "manifest_checksum": manifest_checksum_path,
        "method_summary": output_dir / "method_summary.csv",
        "paired_comparison_summary": output_dir / "paired_comparison_summary.csv",
        "statistical_tests": output_dir / "statistical_tests.csv",
        "report": report_path,
    }
    outputs.update({path.stem: path for path in figures})
    return outputs
