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
from .status import normalize_run_status


INSTANCE_SIZES: dict[str, dict[str, int]] = {
    "very_small": {"num_warehouses": 2, "num_products": 2, "num_regions": 2},
    "small": {"num_warehouses": 3, "num_products": 3, "num_regions": 4},
    "medium": {"num_warehouses": 5, "num_products": 5, "num_regions": 8},
    "medium_large": {"num_warehouses": 6, "num_products": 6, "num_regions": 10},
    "large": {"num_warehouses": 8, "num_products": 8, "num_regions": 12},
}

SELECTED_ALGORITHM_FIELDS = (
    "cut_selection_mode",
    "adaptive_subproblem_gap_enabled",
    "adaptive_secondary_cut_selection_enabled",
    "secondary_cut_warmup_cuts",
    "secondary_cut_master_time_share_trigger",
    "secondary_cut_recent_master_time_trigger",
    "adaptive_secondary_generation_enabled",
    "secondary_generation_lb_window",
    "secondary_generation_stall_threshold",
    "secondary_generation_cooldown_iterations",
    "secondary_generation_max_subproblem_time_share",
    "secondary_generation_min_remaining_time",
    "secondary_generation_min_solve_budget",
    "relative_cut_threshold",
    "cut_violation_tol",
    "final_exact_gap",
    "cut_stall_patience",
    "subproblem_gap_schedule",
    "max_cuts_per_iteration",
    "subproblem_time_budget_per_iteration",
)
NULLABLE_SELECTED_ALGORITHM_FIELDS = {"subproblem_time_budget_per_iteration"}

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
    "secondary_cuts_added_total",
    "secondary_cuts_skipped_total",
    "last_cut_violation",
    "last_cut_added",
    "gamma_target",
    "gamma_schedule",
    "adaptive_gap_enabled",
    "gamma_continuation_enabled",
    "cut_selection_enabled",
    "delta_cut",
    "cut_selection_mode",
    "relative_cut_threshold",
    "cut_violation_tol",
    "final_exact_gap",
    "cut_stall_patience",
    "adaptive_secondary_cut_selection_enabled",
    "secondary_cut_warmup_cuts",
    "secondary_cut_master_time_share_trigger",
    "secondary_cut_recent_master_time_trigger",
    "adaptive_secondary_generation_enabled",
    "secondary_generation_lb_window",
    "secondary_generation_stall_threshold",
    "secondary_generation_cooldown_iterations",
    "secondary_generation_max_subproblem_time_share",
    "secondary_generation_min_remaining_time",
    "secondary_generation_min_solve_budget",
    "final_certification_enabled",
    "final_certification_triggered",
    "final_certification_trigger_iteration",
    "final_certification_count",
    "final_certification_iterations",
    "final_certification_exit_reason",
    "precision_policy",
    "adaptive_master_precision_enabled",
    "adaptive_subproblem_precision_enabled",
    "master_gap_max",
    "master_gap_min",
    "subproblem_gap_max",
    "subproblem_gap_min",
    "fixed_master_mip_gap",
    "fixed_subproblem_mip_gap",
    "master_error_budget_ratio",
    "subproblem_error_budget_ratio",
    "monotone_precision_tightening",
    "secondary_solves_attempted_total",
    "secondary_solves_avoided_total",
    "last_secondary_solve_trigger_reason",
    "last_secondary_solve_skipped_reason",
    "last_recent_relative_lb_improvement",
    "adaptive_subproblem_gap_enabled",
    "subproblem_gap_schedule",
    "last_subproblem_requested_mip_gap",
    "last_subproblem_achieved_mip_gap",
    "num_subproblem_nonoptimal",
    "num_subproblem_without_incumbent",
    "max_cuts_per_iteration",
    "mean_cuts_generated_per_iteration",
    "duplicate_cuts_rejected",
    "duplicate_patterns_rejected",
    "additional_subproblem_time",
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
    "iteration_log_path",
    "reached_gap_5pct",
    "reached_gap_1pct",
    "reached_gap_05pct",
    "reached_gap_01pct",
    "time_to_gap_5pct",
    "time_to_gap_1pct",
    "time_to_gap_05pct",
    "time_to_gap_01pct",
    "iteration_to_gap_5pct",
    "iteration_to_gap_1pct",
    "iteration_to_gap_05pct",
    "iteration_to_gap_01pct",
    "subproblem_time_share",
    "mean_lb_improvement_per_iteration",
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
    "gap_5pct_rate",
    "gap_1pct_rate",
    "gap_05pct_rate",
    "gap_01pct_rate",
    "mean_time_to_gap_5pct",
    "mean_time_to_gap_1pct",
    "mean_time_to_gap_05pct",
    "mean_time_to_gap_01pct",
    "mean_subproblem_time_share",
    "mean_lb_improvement_per_iteration",
]

ITERATION_LOG_FIELDS = [
    "iteration",
    "instance_name",
    "seed",
    "method",
    "variant_name",
    "active_gamma",
    "gamma_target",
    "requested_master_mip_gap",
    "achieved_master_mip_gap",
    "master_status",
    "master_objective",
    "master_best_bound",
    "master_time",
    "LB",
    "UB",
    "global_gap",
    "lb_improvement",
    "ub_improvement",
    "target_robust_evaluation_used",
    "subproblem_requested_mip_gap",
    "subproblem_achieved_mip_gap",
    "subproblem_status",
    "subproblem_incumbent_objective",
    "subproblem_objective_bound",
    "subproblem_time",
    "subproblem_has_incumbent",
    "cut_rhs_current",
    "theta_current",
    "absolute_cut_violation",
    "normalized_cut_violation",
    "cut_added",
    "cut_skip_reason",
    "cuts_generated_this_iteration",
    "cuts_added_this_iteration",
    "cuts_skipped_this_iteration",
    "cuts_added_total",
    "cuts_skipped_total",
    "secondary_cuts_added_total",
    "secondary_cuts_skipped_total",
    "forced_cut_added",
    "forced_cut_reason",
    "secondary_cut_decisions",
    "secondary_active_threshold",
    "master_time_share",
    "secondary_solve_attempted",
    "secondary_solve_trigger_reason",
    "secondary_solve_skipped_reason",
    "recent_relative_lb_improvement",
    "secondary_solve_cooldown_remaining",
    "secondary_solve_runtime",
    "secondary_generation_subproblem_time_share",
    "secondary_generated_cut_added",
    "secondary_generated_cut_duplicate",
    "secondary_solves_avoided_total",
    "final_certification_active",
    "final_certification_triggered_this_iteration",
    "final_certification_trigger_iteration",
    "final_certification_reason",
    "final_certification_count",
    "consecutive_no_useful_primary_cuts",
    "certification_forced_master_mip_gap",
    "certification_forced_subproblem_mip_gap",
    "secondary_solve_disabled_by_certification",
    "precision_policy",
    "valid_global_gap_for_precision",
    "precision_gap_fallback_used",
    "adaptive_master_precision_enabled",
    "adaptive_subproblem_precision_enabled",
    "master_gap_candidate",
    "master_gap_previous",
    "master_gap_selected",
    "subproblem_gap_candidate",
    "subproblem_gap_previous",
    "subproblem_gap_selected",
    "master_error_budget_ratio",
    "subproblem_error_budget_ratio",
    "monotone_precision_tightening",
    "elapsed_time",
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
        if any(isinstance(item, (dict, list, tuple)) for item in value):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
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
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
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
            "cut_selection_mode": str(exp_cfg.get("cut_selection_mode", "absolute")),
            "relative_cut_threshold": float(exp_cfg.get("relative_cut_threshold", 1e-4)),
            "final_exact_gap": float(exp_cfg.get("final_exact_gap", 1e-2)),
            "cut_stall_patience": int(exp_cfg.get("cut_stall_patience", 5)),
            "adaptive_secondary_cut_selection_enabled": bool(
                exp_cfg.get("adaptive_secondary_cut_selection_enabled", False)
            ),
            "secondary_cut_warmup_cuts": int(exp_cfg.get("secondary_cut_warmup_cuts", 50)),
            "secondary_cut_master_time_share_trigger": float(
                exp_cfg.get("secondary_cut_master_time_share_trigger", 0.35)
            ),
            "secondary_cut_recent_master_time_trigger": float(
                exp_cfg.get("secondary_cut_recent_master_time_trigger", 0.5)
            ),
            "adaptive_secondary_generation_enabled": bool(
                exp_cfg.get("adaptive_secondary_generation_enabled", False)
            ),
            "secondary_generation_lb_window": int(
                exp_cfg.get("secondary_generation_lb_window", 5)
            ),
            "secondary_generation_stall_threshold": float(
                exp_cfg.get("secondary_generation_stall_threshold", 1e-4)
            ),
            "secondary_generation_cooldown_iterations": int(
                exp_cfg.get("secondary_generation_cooldown_iterations", 5)
            ),
            "secondary_generation_max_subproblem_time_share": float(
                exp_cfg.get("secondary_generation_max_subproblem_time_share", 0.75)
            ),
            "secondary_generation_min_remaining_time": float(
                exp_cfg.get("secondary_generation_min_remaining_time", 2.0)
            ),
            "secondary_generation_min_solve_budget": float(
                exp_cfg.get("secondary_generation_min_solve_budget", 1.0)
            ),
            "final_certification_enabled": bool(
                exp_cfg.get("final_certification_enabled", False)
            ),
            "final_certification_no_cut_patience": int(
                exp_cfg.get("final_certification_no_cut_patience", 2)
            ),
            "precision_policy": str(exp_cfg.get("precision_policy", "legacy")),
            "adaptive_master_precision_enabled": bool(
                exp_cfg.get("adaptive_master_precision_enabled", False)
            ),
            "adaptive_subproblem_precision_enabled": bool(
                exp_cfg.get("adaptive_subproblem_precision_enabled", False)
            ),
            "master_gap_max": float(
                exp_cfg.get(
                    "master_gap_max",
                    exp_cfg.get("mip_gap", exp_cfg.get("initial_mip_gap", 0.05)),
                )
            ),
            "master_gap_min": float(
                exp_cfg.get("master_gap_min", exp_cfg.get("final_mip_gap", 1e-4))
            ),
            "subproblem_gap_max": float(
                exp_cfg.get("subproblem_gap_max", exp_cfg.get("final_mip_gap", 1e-4))
            ),
            "subproblem_gap_min": float(
                exp_cfg.get("subproblem_gap_min", exp_cfg.get("final_mip_gap", 1e-4))
            ),
            "fixed_subproblem_mip_gap": float(
                exp_cfg.get(
                    "fixed_subproblem_mip_gap",
                    exp_cfg.get("final_mip_gap", 1e-4),
                )
            ),
            "master_error_budget_ratio": float(
                exp_cfg.get("master_error_budget_ratio", 0.5)
            ),
            "subproblem_error_budget_ratio": float(
                exp_cfg.get("subproblem_error_budget_ratio", 0.5)
            ),
            "monotone_precision_tightening": bool(
                exp_cfg.get("monotone_precision_tightening", True)
            ),
            "adaptive_subproblem_gap_enabled": bool(
                exp_cfg.get("adaptive_subproblem_gap_enabled", False)
            ),
            "subproblem_gap_schedule": exp_cfg.get("subproblem_gap_schedule"),
            "max_cuts_per_iteration": int(exp_cfg.get("max_cuts_per_iteration", 1)),
            "subproblem_time_budget_per_iteration": exp_cfg.get("subproblem_time_budget_per_iteration"),
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
    for key in (
        "cut_selection_mode",
        "relative_cut_threshold",
        "cut_violation_tol",
        "final_exact_gap",
        "cut_stall_patience",
        "adaptive_secondary_cut_selection_enabled",
        "secondary_cut_warmup_cuts",
        "secondary_cut_master_time_share_trigger",
        "secondary_cut_recent_master_time_trigger",
        "adaptive_secondary_generation_enabled",
        "secondary_generation_lb_window",
        "secondary_generation_stall_threshold",
        "secondary_generation_cooldown_iterations",
        "secondary_generation_max_subproblem_time_share",
        "secondary_generation_min_remaining_time",
        "secondary_generation_min_solve_budget",
        "final_certification_enabled",
        "final_certification_no_cut_patience",
        "precision_policy",
        "adaptive_master_precision_enabled",
        "adaptive_subproblem_precision_enabled",
        "master_gap_max",
        "master_gap_min",
        "subproblem_gap_max",
        "subproblem_gap_min",
        "fixed_subproblem_mip_gap",
        "master_error_budget_ratio",
        "subproblem_error_budget_ratio",
        "monotone_precision_tightening",
        "adaptive_subproblem_gap_enabled",
        "subproblem_gap_schedule",
        "max_cuts_per_iteration",
        "subproblem_time_budget_per_iteration",
    ):
        if key in variant:
            config["algorithm"][key] = variant[key]
    if flags["gamma_continuation_enabled"] and "gamma_schedule" in variant:
        config["robust"]["gamma_schedule"] = [int(value) for value in variant["gamma_schedule"]]
    for key in ("max_iterations", "tol", "initial_mip_gap", "final_mip_gap", "time_limit"):
        if key in variant:
            config["benders"][key] = variant[key]
    if not flags["gamma_continuation_enabled"]:
        config["robust"]["gamma_schedule"] = [gamma_target]
    configured_initial_mip_gap = float(config["benders"]["initial_mip_gap"])
    if not flags["adaptive_gap_enabled"]:
        final_gap = float(config["benders"]["final_mip_gap"])
        config["benders"]["initial_mip_gap"] = final_gap

    solver_method = "adaptive_gap_gamma_benders"
    if method == "standard_benders":
        solver_method = "standard_benders"
        config["algorithm"]["cut_selection_enabled"] = False
        config["algorithm"]["adaptive_secondary_cut_selection_enabled"] = False
        config["algorithm"]["adaptive_secondary_generation_enabled"] = False
        config["algorithm"]["adaptive_subproblem_gap_enabled"] = False
        config["algorithm"]["precision_policy"] = "legacy"
        config["algorithm"]["adaptive_master_precision_enabled"] = False
        config["algorithm"]["adaptive_subproblem_precision_enabled"] = False
        config["algorithm"]["final_certification_enabled"] = False
        config["algorithm"]["fixed_subproblem_mip_gap"] = float(
            config["benders"]["final_mip_gap"]
        )
        config["algorithm"]["subproblem_gap_schedule"] = [
            {"global_gap_above": 0.0, "mip_gap": float(config["benders"]["final_mip_gap"])}
        ]
        config["algorithm"]["max_cuts_per_iteration"] = 1
        config["algorithm"]["subproblem_time_budget_per_iteration"] = None
        config["robust"]["gamma_schedule"] = [gamma_target]
        config["benders"]["initial_mip_gap"] = float(config["benders"]["final_mip_gap"])
    elif method == "static_inexact_benders":
        solver_method = "inexact_benders"
        config["algorithm"]["cut_selection_enabled"] = False
        config["algorithm"]["adaptive_secondary_cut_selection_enabled"] = False
        config["algorithm"]["adaptive_secondary_generation_enabled"] = False
        config["algorithm"]["adaptive_subproblem_gap_enabled"] = False
        config["algorithm"]["precision_policy"] = "legacy"
        config["algorithm"]["adaptive_master_precision_enabled"] = False
        config["algorithm"]["adaptive_subproblem_precision_enabled"] = False
        config["benders"]["initial_mip_gap"] = configured_initial_mip_gap
        config["algorithm"]["fixed_subproblem_mip_gap"] = float(
            variant.get("fixed_subproblem_mip_gap", configured_initial_mip_gap)
        )
        config["algorithm"]["subproblem_gap_schedule"] = [
            {
                "global_gap_above": 0.0,
                "mip_gap": config["algorithm"]["fixed_subproblem_mip_gap"],
            }
        ]
        config["algorithm"]["max_cuts_per_iteration"] = 1
        config["algorithm"]["subproblem_time_budget_per_iteration"] = None
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


def _time_to_gap_metrics(result: SolveResult) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    thresholds = [("5pct", 0.05), ("1pct", 0.01), ("05pct", 0.005), ("01pct", 0.001)]
    for label, threshold in thresholds:
        reached = next(
            (
                row
                for row in result.iteration_log
                if row.get("global_gap") is not None and float(row["global_gap"]) <= threshold
            ),
            None,
        )
        metrics[f"reached_gap_{label}"] = reached is not None
        metrics[f"time_to_gap_{label}"] = reached.get("elapsed_time") if reached else None
        metrics[f"iteration_to_gap_{label}"] = reached.get("iteration") if reached else None
    metrics["subproblem_time_share"] = (
        result.subproblem_runtime / result.runtime if result.runtime > 0 else None
    )
    improvements = [
        float(row["lb_improvement"])
        for row in result.iteration_log
        if row.get("lb_improvement") is not None
    ]
    metrics["mean_lb_improvement_per_iteration"] = _mean(improvements)
    return metrics


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
        "status": normalize_run_status(result.status),
        "objective": result.objective,
        "best_bound": result.lower_bound,
        "final_gap": result.gap,
        "runtime": result.runtime,
        "master_time": result.master_runtime,
        "subproblem_time": result.subproblem_runtime,
        "iterations": result.iterations,
        "cuts_added_total": meta.get("cuts_added_total", result.cuts),
        "cuts_skipped_total": meta.get("cuts_skipped_total"),
        "secondary_cuts_added_total": meta.get("secondary_cuts_added_total"),
        "secondary_cuts_skipped_total": meta.get("secondary_cuts_skipped_total"),
        "last_cut_violation": meta.get("last_cut_violation"),
        "last_cut_added": meta.get("last_cut_added"),
        "gamma_target": result.gamma_target or meta.get("gamma_target"),
        "gamma_schedule": meta.get("gamma_schedule"),
        "adaptive_gap_enabled": flags.get("adaptive_gap_enabled"),
        "gamma_continuation_enabled": flags.get("gamma_continuation_enabled"),
        "cut_selection_enabled": meta.get("cut_selection_enabled", flags.get("cut_selection_enabled")),
        "delta_cut": meta.get("delta_cut"),
        "cut_selection_mode": meta.get("cut_selection_mode"),
        "relative_cut_threshold": meta.get("relative_cut_threshold"),
        "cut_violation_tol": meta.get("cut_violation_tol"),
        "final_exact_gap": meta.get("final_exact_gap"),
        "cut_stall_patience": meta.get("cut_stall_patience"),
        "adaptive_secondary_cut_selection_enabled": meta.get(
            "adaptive_secondary_cut_selection_enabled"
        ),
        "secondary_cut_warmup_cuts": meta.get("secondary_cut_warmup_cuts"),
        "secondary_cut_master_time_share_trigger": meta.get(
            "secondary_cut_master_time_share_trigger"
        ),
        "secondary_cut_recent_master_time_trigger": meta.get(
            "secondary_cut_recent_master_time_trigger"
        ),
        "adaptive_secondary_generation_enabled": meta.get(
            "adaptive_secondary_generation_enabled"
        ),
        "secondary_generation_lb_window": meta.get("secondary_generation_lb_window"),
        "secondary_generation_stall_threshold": meta.get(
            "secondary_generation_stall_threshold"
        ),
        "secondary_generation_cooldown_iterations": meta.get(
            "secondary_generation_cooldown_iterations"
        ),
        "secondary_generation_max_subproblem_time_share": meta.get(
            "secondary_generation_max_subproblem_time_share"
        ),
        "secondary_generation_min_remaining_time": meta.get(
            "secondary_generation_min_remaining_time"
        ),
        "secondary_generation_min_solve_budget": meta.get(
            "secondary_generation_min_solve_budget"
        ),
        "final_certification_enabled": meta.get("final_certification_enabled"),
        "final_certification_triggered": meta.get("final_certification_triggered"),
        "final_certification_trigger_iteration": meta.get(
            "final_certification_trigger_iteration"
        ),
        "final_certification_count": meta.get("final_certification_count"),
        "final_certification_iterations": meta.get(
            "final_certification_iterations"
        ),
        "final_certification_exit_reason": meta.get(
            "final_certification_exit_reason"
        ),
        "precision_policy": meta.get("precision_policy"),
        "adaptive_master_precision_enabled": meta.get(
            "adaptive_master_precision_enabled"
        ),
        "adaptive_subproblem_precision_enabled": meta.get(
            "adaptive_subproblem_precision_enabled"
        ),
        "master_gap_max": meta.get("master_gap_max"),
        "master_gap_min": meta.get("master_gap_min"),
        "subproblem_gap_max": meta.get("subproblem_gap_max"),
        "subproblem_gap_min": meta.get("subproblem_gap_min"),
        "fixed_master_mip_gap": meta.get("fixed_master_mip_gap"),
        "fixed_subproblem_mip_gap": meta.get("fixed_subproblem_mip_gap"),
        "master_error_budget_ratio": meta.get("master_error_budget_ratio"),
        "subproblem_error_budget_ratio": meta.get(
            "subproblem_error_budget_ratio"
        ),
        "monotone_precision_tightening": meta.get(
            "monotone_precision_tightening"
        ),
        "secondary_solves_attempted_total": meta.get("secondary_solves_attempted_total"),
        "secondary_solves_avoided_total": meta.get("secondary_solves_avoided_total"),
        "last_secondary_solve_trigger_reason": meta.get(
            "last_secondary_solve_trigger_reason"
        ),
        "last_secondary_solve_skipped_reason": meta.get(
            "last_secondary_solve_skipped_reason"
        ),
        "last_recent_relative_lb_improvement": meta.get(
            "last_recent_relative_lb_improvement"
        ),
        "adaptive_subproblem_gap_enabled": meta.get("adaptive_subproblem_gap_enabled"),
        "subproblem_gap_schedule": meta.get("subproblem_gap_schedule"),
        "last_subproblem_requested_mip_gap": meta.get("last_subproblem_requested_mip_gap"),
        "last_subproblem_achieved_mip_gap": meta.get("last_subproblem_achieved_mip_gap"),
        "num_subproblem_nonoptimal": meta.get("num_subproblem_nonoptimal"),
        "num_subproblem_without_incumbent": meta.get("num_subproblem_without_incumbent"),
        "max_cuts_per_iteration": meta.get("max_cuts_per_iteration"),
        "mean_cuts_generated_per_iteration": meta.get("mean_cuts_generated_per_iteration"),
        "duplicate_cuts_rejected": meta.get("duplicate_cuts_rejected"),
        "duplicate_patterns_rejected": meta.get("duplicate_patterns_rejected"),
        "additional_subproblem_time": meta.get("additional_subproblem_time"),
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
        **_time_to_gap_metrics(result),
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
        method_names = {
            "standard_benders",
            "static_inexact_benders",
            "adaptive_gamma_benders",
            "adaptive_gap_benders",
            "adaptive_cut_benders",
            "proposed_adaptive_benders",
            "scenario_benders_full",
            "monolithic_gurobi",
        }
        return [
            (name, name if name in method_names else "proposed_adaptive_benders", variants.get(name, {}))
            for name in exp_cfg["variants"]
        ]
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
        gap_reach_rates = {
            label: _mean(
                [
                    1.0 if r.get(f"reached_gap_{label}") in {True, "True", "true", 1} else 0.0
                    for r in rows
                ]
            )
            for label in ("5pct", "1pct", "05pct", "01pct")
        }
        mean_times_to_gap = {
            label: _mean(
                [
                    float(r[f"time_to_gap_{label}"])
                    for r in rows
                    if r.get(f"time_to_gap_{label}") not in (None, "")
                ]
            )
            for label in ("5pct", "1pct", "05pct", "01pct")
        }
        subproblem_shares = [
            float(r["subproblem_time_share"])
            for r in completed
            if r.get("subproblem_time_share") not in (None, "")
        ]
        lb_improvements = [
            float(r["mean_lb_improvement_per_iteration"])
            for r in completed
            if r.get("mean_lb_improvement_per_iteration") not in (None, "")
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
                "gap_5pct_rate": gap_reach_rates["5pct"],
                "gap_1pct_rate": gap_reach_rates["1pct"],
                "gap_05pct_rate": gap_reach_rates["05pct"],
                "gap_01pct_rate": gap_reach_rates["01pct"],
                "mean_time_to_gap_5pct": mean_times_to_gap["5pct"],
                "mean_time_to_gap_1pct": mean_times_to_gap["1pct"],
                "mean_time_to_gap_05pct": mean_times_to_gap["05pct"],
                "mean_time_to_gap_01pct": mean_times_to_gap["01pct"],
                "mean_subproblem_time_share": _mean(subproblem_shares),
                "mean_lb_improvement_per_iteration": _mean(lb_improvements),
            }
        )
    return summaries


def _is_completed(row: dict[str, Any]) -> bool:
    return (
        normalize_run_status(row.get("status")) in COMPLETED_STATUSES
        and row.get("objective") not in (None, "")
    )


def _is_solved(row: dict[str, Any]) -> bool:
    if row.get("objective") in (None, ""):
        return False
    if normalize_run_status(row.get("status")) == "optimal":
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


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)


def _write_iteration_log(
    output_dir: Path,
    experiment_name: str,
    instance_name: str,
    seed: int,
    method: str,
    variant_name: str,
    iteration_log: list[dict[str, Any]],
) -> Path:
    filename = _safe_filename(
        f"{experiment_name}__{instance_name}__seed_{seed}__{method}__{variant_name}"
    )
    path = output_dir / "iteration_logs" / f"{filename}.csv"
    rows = []
    for source in iteration_log:
        row = dict(source)
        row.update(
            {
                "instance_name": instance_name,
                "seed": seed,
                "method": method,
                "variant_name": variant_name,
            }
        )
        rows.append(row)
    _write_csv(path, rows, ITERATION_LOG_FIELDS)
    return path


def _validate_relative_threshold_config(config: dict[str, Any]) -> None:
    if config.get("cut_selection_mode") != "relative":
        return
    if config.get("relative_cut_threshold") is not None:
        return

    variant_names = config.get("variants", [])
    variant_settings = config.get("variant_settings", {})
    variants_have_thresholds = bool(variant_names) and all(
        variant_settings.get(name, {}).get("relative_cut_threshold") is not None
        for name in variant_names
    )
    if variants_have_thresholds:
        return

    raise ValueError(
        "relative_cut_threshold must be selected before running "
        "screen_master_gamma.yaml. Run screen_relative_cut_wide.yaml first."
    )


def run_experiment_suite(config: dict[str, Any]) -> dict[str, Path]:
    config = deepcopy(config)
    exp_name = str(config.get("experiment_name", "experiment_suite"))
    selected_parameters_path = config.get("parameters_must_be_fixed_from")
    if selected_parameters_path:
        selected = load_config(str(selected_parameters_path))
        if selected.get("selection_status") != "selected":
            raise ValueError(
                "Final evaluation is locked until selected_algorithm_parameters.yaml has selection_status: selected."
            )
        absent = [field for field in SELECTED_ALGORITHM_FIELDS if field not in selected]
        if absent:
            raise ValueError(f"Selected algorithm parameters are missing: {', '.join(absent)}")
        missing = [
            field
            for field in SELECTED_ALGORITHM_FIELDS
            if field not in NULLABLE_SELECTED_ALGORITHM_FIELDS and selected.get(field) is None
        ]
        if missing:
            raise ValueError(f"Selected algorithm parameters are missing: {', '.join(missing)}")
        if selected["cut_selection_mode"] not in {"absolute", "relative"}:
            raise ValueError("Selected cut_selection_mode must be 'absolute' or 'relative'.")
        if (
            selected["cut_selection_mode"] != "relative"
            and selected.get("relative_cut_threshold") is not None
        ):
            raise ValueError(
                "relative_cut_threshold requires selected cut_selection_mode='relative'."
            )
        if not isinstance(selected["adaptive_subproblem_gap_enabled"], bool):
            raise ValueError("Selected adaptive_subproblem_gap_enabled must be true or false.")
        if not isinstance(selected["adaptive_secondary_cut_selection_enabled"], bool):
            raise ValueError(
                "Selected adaptive_secondary_cut_selection_enabled must be true or false."
            )
        if not isinstance(selected["adaptive_secondary_generation_enabled"], bool):
            raise ValueError(
                "Selected adaptive_secondary_generation_enabled must be true or false."
            )
        warmup_cuts = selected["secondary_cut_warmup_cuts"]
        if (
            isinstance(warmup_cuts, bool)
            or not isinstance(warmup_cuts, int)
            or warmup_cuts <= 0
        ):
            raise ValueError("Selected secondary_cut_warmup_cuts must be a positive integer.")
        for trigger_field in (
            "secondary_cut_master_time_share_trigger",
            "secondary_cut_recent_master_time_trigger",
        ):
            trigger_value = selected[trigger_field]
            if (
                isinstance(trigger_value, bool)
                or not isinstance(trigger_value, (int, float))
                or not math.isfinite(float(trigger_value))
                or float(trigger_value) <= 0.0
            ):
                raise ValueError(f"Selected {trigger_field} must be a positive finite value.")
        lb_window = selected["secondary_generation_lb_window"]
        if isinstance(lb_window, bool) or not isinstance(lb_window, int) or lb_window <= 0:
            raise ValueError("Selected secondary_generation_lb_window must be a positive integer.")
        stall_threshold = selected["secondary_generation_stall_threshold"]
        if (
            isinstance(stall_threshold, bool)
            or not isinstance(stall_threshold, (int, float))
            or not math.isfinite(float(stall_threshold))
            or float(stall_threshold) < 0.0
        ):
            raise ValueError(
                "Selected secondary_generation_stall_threshold must be a finite nonnegative value."
            )
        cooldown_iterations = selected["secondary_generation_cooldown_iterations"]
        if (
            isinstance(cooldown_iterations, bool)
            or not isinstance(cooldown_iterations, int)
            or cooldown_iterations < 0
        ):
            raise ValueError(
                "Selected secondary_generation_cooldown_iterations must be a nonnegative integer."
            )
        max_time_share = selected["secondary_generation_max_subproblem_time_share"]
        if (
            isinstance(max_time_share, bool)
            or not isinstance(max_time_share, (int, float))
            or not math.isfinite(float(max_time_share))
            or not 0.0 < float(max_time_share) <= 1.0
        ):
            raise ValueError(
                "Selected secondary_generation_max_subproblem_time_share must be finite and in (0, 1]."
            )
        min_remaining_time = selected["secondary_generation_min_remaining_time"]
        if (
            isinstance(min_remaining_time, bool)
            or not isinstance(min_remaining_time, (int, float))
            or not math.isfinite(float(min_remaining_time))
            or float(min_remaining_time) < 0.0
        ):
            raise ValueError(
                "Selected secondary_generation_min_remaining_time must be a finite nonnegative value."
            )
        min_solve_budget = selected["secondary_generation_min_solve_budget"]
        if (
            isinstance(min_solve_budget, bool)
            or not isinstance(min_solve_budget, (int, float))
            or not math.isfinite(float(min_solve_budget))
            or float(min_solve_budget) <= 0.0
        ):
            raise ValueError(
                "Selected secondary_generation_min_solve_budget must be a positive finite value."
            )
        for field in SELECTED_ALGORITHM_FIELDS:
            config[field] = deepcopy(selected[field])
    _validate_relative_threshold_config(config)
    output_dir = Path(str(config.get("output_dir", f"experiments/results/{exp_name}")))
    instances_dir = output_dir / "instances"
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )

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
                if bool(config.get("save_iteration_log", False)):
                    iteration_log_path = _write_iteration_log(
                        output_dir,
                        exp_name,
                        instance.name,
                        seed,
                        method,
                        variant_name,
                        result.iteration_log,
                    )
                    row["iteration_log_path"] = str(iteration_log_path)
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
    return {
        "results": results_path,
        "summary": summary_path,
        "resolved_config": resolved_config_path,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robust inventory experiment suite.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    outputs = run_experiment_suite(load_config(args.config))
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
