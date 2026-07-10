from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .benders import solve_benders
from .config import load_config
from .instance import InventoryInstance, generate_instance, save_instance
from .monolithic import solve_monolithic
from .results import SolveResult
from .scenarios import count_budget_scenarios


INSTANCE_SIZES: dict[str, dict[str, int]] = {
    "very_small": {"num_warehouses": 2, "num_products": 2, "num_regions": 2},
    "small": {"num_warehouses": 3, "num_products": 3, "num_regions": 4},
    "medium": {"num_warehouses": 5, "num_products": 5, "num_regions": 8},
    "medium_large": {"num_warehouses": 6, "num_products": 6, "num_regions": 10},
    "large": {"num_warehouses": 8, "num_products": 8, "num_regions": 12},
}

RESULT_FIELDS = [
    "experiment_name",
    "instance_name",
    "instance_size",
    "seed",
    "method",
    "variant_name",
    "subproblem_mode",
    "status",
    "objective",
    "best_bound",
    "final_gap",
    "runtime",
    "master_time",
    "subproblem_time",
    "iterations",
    "cuts_added_total",
    "cuts_skipped_total",
    "last_cut_violation",
    "last_cut_added",
    "gamma_target",
    "gamma_schedule",
    "adaptive_gap_enabled",
    "gamma_continuation_enabled",
    "cut_selection_enabled",
    "delta_cut",
    "exact_scenarios",
    "scenario_mode_target",
    "heuristic_scenarios",
    "num_target_scenarios_used",
    "num_target_scenarios_total_estimated",
    "valid_UB",
    "ub_uses_subproblem_bound",
    "target_subproblem_status",
    "target_subproblem_mip_gap",
    "target_subproblem_objective_bound",
    "total_shortage",
    "service_violation",
    "first_stage_cost",
    "inventory_cost",
    "worst_case_cost",
    "error_message",
    "instance_path",
]

SUMMARY_FIELDS = [
    "experiment_name",
    "instance_size",
    "method",
    "variant_name",
    "num_runs",
    "num_completed",
    "completed_rate",
    "num_solved",
    "solved_rate",
    "num_success",
    "success_rate",
    "mean_objective",
    "std_objective",
    "mean_runtime",
    "std_runtime",
    "mean_final_gap",
    "mean_iterations",
    "mean_cuts_added",
    "mean_cuts_skipped",
    "mean_master_time",
    "mean_subproblem_time",
    "mean_valid_UB_rate",
    "speedup_vs_standard_benders",
    "runtime_saving_vs_standard",
]

COMPLETED_STATUSES = {"optimal", "iteration_limit", "time_limit"}
SOLVE_TOLERANCE = 1e-4


def _as_list(value: Any, default: list[Any] | None = None) -> list[Any]:
    if value is None:
        return [] if default is None else list(default)
    if isinstance(value, list):
        return value
    return [value]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _base_config(exp_cfg: dict[str, Any], size_name: str, seed: int, alpha: float | None = None) -> dict[str, Any]:
    size_cfg = dict(INSTANCE_SIZES[size_name])
    size_cfg.update(exp_cfg.get("instance_overrides", {}))
    if alpha is not None:
        size_cfg["service_level"] = alpha

    gamma_target = int(exp_cfg.get("gamma_target", 2))
    gamma_schedule = exp_cfg.get("gamma_schedule") or list(range(gamma_target + 1))
    return {
        "seed": seed,
        "instance": {
            **size_cfg,
            "budget_factor": float(exp_cfg.get("budget_factor", 0.68)),
            "demand_scale": float(exp_cfg.get("demand_scale", 1.0)),
            "capacity_factor": float(exp_cfg.get("capacity_factor", 1.25)),
            "cost_scale": float(exp_cfg.get("cost_scale", 1.0)),
        },
        "robust": {
            "gamma_target": gamma_target,
            "gamma_schedule": gamma_schedule,
            "max_scenarios": int(exp_cfg.get("max_scenarios", 5000)),
            "exact_scenarios": bool(exp_cfg.get("exact_scenarios", True)),
        },
        "algorithm": {
            "subproblem_mode": exp_cfg.get("subproblem_mode", "robust_dual_milp"),
            "cut_selection_enabled": bool(exp_cfg.get("cut_selection_enabled", True)),
            "delta_cut": float(exp_cfg.get("delta_cut", 0.0)),
            "cut_violation_tol": float(exp_cfg.get("cut_violation_tol", 1e-8)),
        },
        "benders": {
            "max_iterations": int(exp_cfg.get("max_iterations", 80)),
            "tol": float(exp_cfg.get("tol", 1e-4)),
            "initial_mip_gap": float(exp_cfg.get("mip_gap", exp_cfg.get("initial_mip_gap", 0.05))),
            "final_mip_gap": float(exp_cfg.get("final_mip_gap", 1e-4)),
            "time_limit": float(exp_cfg.get("time_limit", 120)),
            "output_flag": bool(exp_cfg.get("output_flag", False)),
        },
    }


def _apply_variant_config(
    config: dict[str, Any],
    method: str,
    variant: dict[str, Any],
) -> tuple[str, dict[str, bool], dict[str, Any]]:
    config = deepcopy(config)
    has_variant_overrides = bool(variant)
    flags = {
        "adaptive_gap_enabled": bool(variant.get("adaptive_gap_enabled", False)),
        "gamma_continuation_enabled": bool(variant.get("gamma_continuation_enabled", False)),
        "cut_selection_enabled": bool(variant.get("cut_selection_enabled", False)),
    }
    gamma_target = int(config["robust"]["gamma_target"])

    config["algorithm"]["cut_selection_enabled"] = flags["cut_selection_enabled"]
    if not flags["gamma_continuation_enabled"]:
        config["robust"]["gamma_schedule"] = [gamma_target]
    if not flags["adaptive_gap_enabled"]:
        final_gap = float(config["benders"]["final_mip_gap"])
        config["benders"]["initial_mip_gap"] = final_gap

    solver_method = "adaptive_gap_gamma_benders"
    if method == "standard_benders":
        solver_method = "standard_benders"
        config["algorithm"]["cut_selection_enabled"] = False
        config["robust"]["gamma_schedule"] = [gamma_target]
        config["benders"]["initial_mip_gap"] = float(config["benders"]["final_mip_gap"])
    elif method == "static_inexact_benders":
        solver_method = "inexact_benders"
        config["algorithm"]["cut_selection_enabled"] = False
        config["robust"]["gamma_schedule"] = [gamma_target]
    elif method == "adaptive_gamma_benders":
        solver_method = "adaptive_gap_gamma_benders"
        config["algorithm"]["cut_selection_enabled"] = False
        config["benders"]["initial_mip_gap"] = float(config["benders"]["final_mip_gap"])
    elif method == "adaptive_gap_benders":
        solver_method = "adaptive_gap_gamma_benders"
        config["algorithm"]["cut_selection_enabled"] = False
        config["robust"]["gamma_schedule"] = [gamma_target]
    elif method == "adaptive_cut_benders":
        solver_method = "standard_benders"
        config["algorithm"]["cut_selection_enabled"] = True
    elif method == "proposed_adaptive_benders":
        solver_method = "adaptive_gap_gamma_benders"
        if not has_variant_overrides:
            config["algorithm"]["cut_selection_enabled"] = True
    elif method == "scenario_benders_full":
        solver_method = "standard_benders"
        config["algorithm"]["subproblem_mode"] = "scenario_enumeration"
        config["algorithm"]["cut_selection_enabled"] = False
        config["robust"]["exact_scenarios"] = True
        config["robust"]["gamma_schedule"] = [gamma_target]
    elif method == "monolithic_gurobi":
        solver_method = "monolithic_gurobi"
        config["robust"]["exact_scenarios"] = True
    elif method not in {"standard_benders", "static_inexact_benders"}:
        raise ValueError(f"Unsupported experiment method: {method}")

    if method in {"standard_benders", "static_inexact_benders", "scenario_benders_full", "monolithic_gurobi"}:
        flags = {
            "adaptive_gap_enabled": False,
            "gamma_continuation_enabled": False,
            "cut_selection_enabled": bool(config["algorithm"].get("cut_selection_enabled", False)),
        }
    elif method == "adaptive_gamma_benders":
        flags = {
            "adaptive_gap_enabled": False,
            "gamma_continuation_enabled": True,
            "cut_selection_enabled": False,
        }
    elif method == "adaptive_gap_benders":
        flags = {
            "adaptive_gap_enabled": True,
            "gamma_continuation_enabled": False,
            "cut_selection_enabled": False,
        }
    elif method == "adaptive_cut_benders":
        flags = {
            "adaptive_gap_enabled": False,
            "gamma_continuation_enabled": False,
            "cut_selection_enabled": True,
        }
    elif method == "proposed_adaptive_benders" and not has_variant_overrides:
        flags = {
            "adaptive_gap_enabled": True,
            "gamma_continuation_enabled": True,
            "cut_selection_enabled": True,
        }

    return solver_method, flags, config


def _should_skip_full_scenarios(instance: InventoryInstance, config: dict[str, Any], method: str) -> tuple[bool, str]:
    if method not in {"monolithic_gurobi", "scenario_benders_full"}:
        return False, ""
    total = count_budget_scenarios(instance, int(config["robust"]["gamma_target"]))
    max_scenarios = int(config["robust"]["max_scenarios"])
    if total > max_scenarios:
        return True, f"too_many_scenarios: {total} > {max_scenarios}"
    return False, ""


def _solve_experiment_method(
    config: dict[str, Any],
    instance: InventoryInstance,
    method: str,
    variant: dict[str, Any],
) -> tuple[SolveResult, dict[str, bool]]:
    solver_method, flags, method_config = _apply_variant_config(config, method, variant)
    skip, reason = _should_skip_full_scenarios(instance, method_config, method)
    if skip:
        return (
            SolveResult(
                method=method,
                status="skipped",
                objective=None,
                lower_bound=None,
                upper_bound=None,
                gap=None,
                runtime=0.0,
                metadata={"error_message": reason},
            ),
            flags,
        )
    if solver_method == "monolithic_gurobi":
        result = solve_monolithic(method_config, instance)
    else:
        result = solve_benders(method_config, instance, solver_method)
    result.method = method
    return result, flags


def _result_row(
    exp_name: str,
    size_name: str,
    seed: int,
    method: str,
    variant_name: str,
    result: SolveResult,
    flags: dict[str, bool],
    instance: InventoryInstance,
    instance_path: Path,
    error_message: str = "",
) -> dict[str, Any]:
    meta = result.metadata
    return {
        "experiment_name": exp_name,
        "instance_name": instance.name,
        "instance_size": size_name,
        "seed": seed,
        "method": method,
        "variant_name": variant_name,
        "subproblem_mode": meta.get("subproblem_mode"),
        "status": result.status,
        "objective": result.objective,
        "best_bound": result.lower_bound,
        "final_gap": result.gap,
        "runtime": result.runtime,
        "master_time": result.master_runtime,
        "subproblem_time": result.subproblem_runtime,
        "iterations": result.iterations,
        "cuts_added_total": meta.get("cuts_added_total", result.cuts),
        "cuts_skipped_total": meta.get("cuts_skipped_total"),
        "last_cut_violation": meta.get("last_cut_violation"),
        "last_cut_added": meta.get("last_cut_added"),
        "gamma_target": result.gamma_target or meta.get("gamma_target"),
        "gamma_schedule": meta.get("gamma_schedule"),
        "adaptive_gap_enabled": flags.get("adaptive_gap_enabled"),
        "gamma_continuation_enabled": flags.get("gamma_continuation_enabled"),
        "cut_selection_enabled": meta.get("cut_selection_enabled", flags.get("cut_selection_enabled")),
        "delta_cut": meta.get("delta_cut"),
        "exact_scenarios": meta.get("exact_scenarios"),
        "scenario_mode_target": meta.get("scenario_mode_target", meta.get("scenario_mode")),
        "heuristic_scenarios": meta.get("heuristic_scenarios"),
        "num_target_scenarios_used": meta.get("num_target_scenarios_used", meta.get("num_scenarios_used")),
        "num_target_scenarios_total_estimated": meta.get(
            "num_target_scenarios_total_estimated",
            meta.get("num_scenarios_total_estimated"),
        ),
        "valid_UB": meta.get("valid_UB"),
        "ub_uses_subproblem_bound": meta.get("ub_uses_subproblem_bound"),
        "target_subproblem_status": meta.get("target_subproblem_status"),
        "target_subproblem_mip_gap": meta.get("target_subproblem_mip_gap"),
        "target_subproblem_objective_bound": meta.get("target_subproblem_objective_bound"),
        # TODO: derive second-stage shortage/service metrics from stored worst-case recourse solutions.
        "total_shortage": meta.get("total_shortage"),
        "service_violation": meta.get("service_violation"),
        "first_stage_cost": result.first_stage_cost,
        "inventory_cost": meta.get("inventory_cost"),
        "worst_case_cost": result.robust_cost,
        "error_message": error_message or meta.get("error_message"),
        "instance_path": str(instance_path),
    }


def _failure_row(
    exp_name: str,
    size_name: str,
    seed: int,
    method: str,
    variant_name: str,
    flags: dict[str, bool],
    instance: InventoryInstance,
    instance_path: Path,
    exc: Exception,
) -> dict[str, Any]:
    row = _result_row(
        exp_name,
        size_name,
        seed,
        method,
        variant_name,
        SolveResult(method=method, status="failed", objective=None, lower_bound=None, upper_bound=None, gap=None, runtime=0.0),
        flags,
        instance,
        instance_path,
        error_message=str(exc),
    )
    row["status"] = "failed"
    return row


def _variant_specs(exp_cfg: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    if "variants" in exp_cfg:
        variants = exp_cfg.get("variant_settings", {})
        return [(name, "proposed_adaptive_benders", variants.get(name, {})) for name in exp_cfg["variants"]]
    methods = exp_cfg.get("methods") or _as_list(exp_cfg.get("method"), ["proposed_adaptive_benders"])
    return [(str(method), str(method), {}) for method in methods]


def _expanded_dimensions(exp_cfg: dict[str, Any]) -> list[tuple[int, str, float | None, int | None, float | None]]:
    seeds = [int(seed) for seed in exp_cfg.get("random_seeds", [0])]
    sizes = [str(size) for size in exp_cfg.get("instance_sizes", ["small"])]
    gamma_values = _as_list(exp_cfg.get("gamma_target_values"))
    alpha_values = _as_list(exp_cfg.get("alpha_values"))
    if gamma_values:
        return [(seed, size, None, int(gamma), None) for seed in seeds for size in sizes for gamma in gamma_values]
    if alpha_values and len(alpha_values) > 1 and exp_cfg.get("experiment_name") == "sensitivity_service":
        return [(seed, size, float(alpha), None, float(alpha)) for seed in seeds for size in sizes for alpha in alpha_values]
    return [(seed, size, None, None, None) for seed in seeds for size in sizes]


def _summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in results:
        key = (
            str(row.get("experiment_name", "")),
            str(row.get("instance_size", "")),
            str(row.get("method", "")),
            str(row.get("variant_name", "")),
        )
        groups.setdefault(key, []).append(row)

    standard_runtime: dict[tuple[str, str], float] = {}
    for (exp_name, size_name, method, _variant), rows in groups.items():
        if method == "standard_benders":
            runtimes = [float(r["runtime"]) for r in rows if _is_completed(r) and r.get("runtime") not in (None, "")]
            mean_runtime = _mean(runtimes)
            if mean_runtime and mean_runtime > 0:
                standard_runtime[(exp_name, size_name)] = mean_runtime

    summaries = []
    for (exp_name, size_name, method, variant_name), rows in sorted(groups.items()):
        completed = [row for row in rows if _is_completed(row)]
        solved = [row for row in rows if _is_solved(row)]
        objectives = [float(r["objective"]) for r in completed if r.get("objective") not in (None, "")]
        runtimes = [float(r["runtime"]) for r in completed if r.get("runtime") not in (None, "")]
        gaps = [float(r["final_gap"]) for r in completed if r.get("final_gap") not in (None, "")]
        iterations = [float(r["iterations"]) for r in completed if r.get("iterations") not in (None, "")]
        cuts_added = [float(r["cuts_added_total"]) for r in completed if r.get("cuts_added_total") not in (None, "")]
        cuts_skipped = [float(r["cuts_skipped_total"]) for r in completed if r.get("cuts_skipped_total") not in (None, "")]
        master_time = [float(r["master_time"]) for r in completed if r.get("master_time") not in (None, "")]
        subproblem_time = [float(r["subproblem_time"]) for r in completed if r.get("subproblem_time") not in (None, "")]
        valid_ub_values = [
            1.0 if r.get("valid_UB") in {True, "True", "true", 1} else 0.0
            for r in completed
            if r.get("valid_UB") not in (None, "")
        ]
        mean_runtime = _mean(runtimes)
        base_runtime = standard_runtime.get((exp_name, size_name))
        speedup = base_runtime / mean_runtime if base_runtime and mean_runtime and mean_runtime > 0 else None
        saving = (base_runtime - mean_runtime) / base_runtime if base_runtime and mean_runtime else None
        summaries.append(
            {
                "experiment_name": exp_name,
                "instance_size": size_name,
                "method": method,
                "variant_name": variant_name,
                "num_runs": len(rows),
                "num_completed": len(completed),
                "completed_rate": len(completed) / len(rows) if rows else 0.0,
                "num_solved": len(solved),
                "solved_rate": len(solved) / len(rows) if rows else 0.0,
                "num_success": len(solved),
                "success_rate": len(solved) / len(rows) if rows else 0.0,
                "mean_objective": _mean(objectives),
                "std_objective": _std(objectives),
                "mean_runtime": mean_runtime,
                "std_runtime": _std(runtimes),
                "mean_final_gap": _mean(gaps),
                "mean_iterations": _mean(iterations),
                "mean_cuts_added": _mean(cuts_added),
                "mean_cuts_skipped": _mean(cuts_skipped),
                "mean_master_time": _mean(master_time),
                "mean_subproblem_time": _mean(subproblem_time),
                "mean_valid_UB_rate": _mean(valid_ub_values),
                "speedup_vs_standard_benders": speedup,
                "runtime_saving_vs_standard": saving,
            }
        )
    return summaries


def _is_completed(row: dict[str, Any]) -> bool:
    return row.get("status") in COMPLETED_STATUSES and row.get("objective") not in (None, "")


def _is_solved(row: dict[str, Any]) -> bool:
    if row.get("objective") in (None, ""):
        return False
    if row.get("status") == "optimal":
        return True
    gap = row.get("final_gap")
    if gap in (None, ""):
        return False
    return float(gap) <= SOLVE_TOLERANCE


def _correctness_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault((int(row["seed"]), str(row["instance_name"])), {})[str(row["method"])] = row

    rows = []
    for (seed, instance_name), by_method in sorted(grouped.items()):
        mono = by_method.get("monolithic_gurobi", {})
        scen = by_method.get("scenario_benders_full", {})
        standard = by_method.get("standard_benders", {})
        proposed = by_method.get("proposed_adaptive_benders", {})
        mono_obj = mono.get("objective")
        scen_obj = scen.get("objective")
        standard_obj = standard.get("objective")
        proposed_obj = proposed.get("objective")
        diff_mono_standard = _diff(mono_obj, standard_obj)
        diff_scen_standard = _diff(scen_obj, standard_obj)
        diff_mono_proposed = _diff(mono_obj, proposed_obj)
        diff_scen_proposed = _diff(scen_obj, proposed_obj)
        rows.append(
            {
                "seed": seed,
                "instance_name": instance_name,
                "monolithic_objective": mono_obj,
                "scenario_benders_objective": scen_obj,
                "standard_benders_objective": standard_obj,
                "proposed_objective": proposed_obj,
                "abs_diff_monolithic_vs_standard": diff_mono_standard[0],
                "rel_diff_monolithic_vs_standard": diff_mono_standard[1],
                "abs_diff_scenario_vs_robust_dual": diff_scen_standard[0],
                "rel_diff_scenario_vs_robust_dual": diff_scen_standard[1],
                "abs_diff_monolithic_vs_proposed": diff_mono_proposed[0],
                "rel_diff_monolithic_vs_proposed": diff_mono_proposed[1],
                "abs_diff_scenario_vs_proposed": diff_scen_proposed[0],
                "rel_diff_scenario_vs_proposed": diff_scen_proposed[1],
                "status": _correctness_status(
                    mono_obj,
                    scen_obj,
                    standard_obj,
                    proposed_obj,
                    [diff_mono_standard, diff_scen_standard, diff_mono_proposed, diff_scen_proposed],
                ),
            }
        )
    return rows


def _diff(lhs: Any, rhs: Any) -> tuple[float | None, float | None]:
    if lhs in (None, "") or rhs in (None, ""):
        return None, None
    abs_diff = abs(float(lhs) - float(rhs))
    rel_diff = abs_diff / max(1.0, abs(float(lhs)))
    return abs_diff, rel_diff


def _within_tolerance(diff: tuple[float | None, float | None]) -> bool:
    abs_diff, rel_diff = diff
    if abs_diff is None or rel_diff is None:
        return False
    return abs_diff <= 1e-4 or rel_diff <= 1e-4


def _correctness_status(
    monolithic_objective: Any,
    scenario_benders_objective: Any,
    standard_benders_objective: Any,
    proposed_objective: Any,
    diffs: list[tuple[float | None, float | None]],
) -> str:
    if monolithic_objective in (None, ""):
        return "missing_monolithic"
    if scenario_benders_objective in (None, ""):
        return "missing_scenario_benders"
    if standard_benders_objective in (None, ""):
        return "missing_robust_dual_benders"
    if proposed_objective in (None, ""):
        return "missing_proposed"
    return "ok" if all(_within_tolerance(diff) for diff in diffs) else "check"


def run_experiment_suite(config: dict[str, Any]) -> dict[str, Path]:
    exp_name = str(config.get("experiment_name", "experiment_suite"))
    output_dir = Path(str(config.get("output_dir", f"experiments/results/{exp_name}")))
    instances_dir = output_dir / "instances"
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    variants = _variant_specs(config)
    for seed, size_name, _alpha_key, gamma_override, alpha_value in _expanded_dimensions(config):
        run_cfg = _base_config(config, size_name, seed, alpha=alpha_value)
        if gamma_override is not None:
            run_cfg["robust"]["gamma_target"] = gamma_override
            run_cfg["robust"]["gamma_schedule"] = list(range(gamma_override + 1))
        instance = generate_instance(run_cfg, seed=seed)
        instance_path = instances_dir / f"{instance.name}.json"
        save_instance(instance, instance_path)
        for variant_name, method, variant in variants:
            try:
                method_cfg = deepcopy(run_cfg)
                result, flags = _solve_experiment_method(method_cfg, instance, method, variant)
                row = _result_row(exp_name, size_name, seed, method, variant_name, result, flags, instance, instance_path)
                if gamma_override is not None:
                    row["gamma_target"] = gamma_override
                if alpha_value is not None:
                    row["alpha"] = alpha_value
                results.append(row)
            except Exception as exc:  # noqa: BLE001 - experiments must keep running after failed methods.
                flags = {
                    "adaptive_gap_enabled": bool(variant.get("adaptive_gap_enabled", False)),
                    "gamma_continuation_enabled": bool(variant.get("gamma_continuation_enabled", False)),
                    "cut_selection_enabled": bool(variant.get("cut_selection_enabled", False)),
                }
                results.append(_failure_row(exp_name, size_name, seed, method, variant_name, flags, instance, instance_path, exc))

    results_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.csv"
    _write_csv(results_path, results, RESULT_FIELDS + ["alpha"])
    _write_csv(summary_path, _summary_rows(results), SUMMARY_FIELDS)
    if exp_name == "small_correctness":
        _write_csv(
            output_dir / "correctness_summary.csv",
            _correctness_rows(results),
            [
                "seed",
                "instance_name",
                "monolithic_objective",
                "scenario_benders_objective",
                "standard_benders_objective",
                "proposed_objective",
                "abs_diff_monolithic_vs_standard",
                "rel_diff_monolithic_vs_standard",
                "abs_diff_scenario_vs_robust_dual",
                "rel_diff_scenario_vs_robust_dual",
                "abs_diff_monolithic_vs_proposed",
                "rel_diff_monolithic_vs_proposed",
                "abs_diff_scenario_vs_proposed",
                "rel_diff_scenario_vs_proposed",
                "status",
            ],
        )
    return {"results": results_path, "summary": summary_path, "output_dir": output_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robust inventory experiment suite.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    outputs = run_experiment_suite(load_config(args.config))
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
