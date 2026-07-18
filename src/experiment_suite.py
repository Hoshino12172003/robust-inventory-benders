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
from .experiment_protocol import (
    ProtocolRunSpec,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    config_sha256,
    decide_run_action,
    git_commit,
    load_run_record,
    penalized_runtime_par2,
    theoretical_maximum_hours,
    update_run_manifest,
    utc_now_iso,
    write_run_state,
)
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
    "subproblem_mode",
    "cut_selection_enabled",
    "cut_selection_mode",
    "final_certification_enabled",
    "final_certification_no_cut_patience",
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
SELECTED_EXPERIMENT_FIELDS = (
    "adaptive_gap_enabled",
    "gamma_continuation_enabled",
    "gamma_schedule",
)
SELECTED_PARAMETER_FIELDS = SELECTED_ALGORITHM_FIELDS + SELECTED_EXPERIMENT_FIELDS
WORKLOAD_AWARE_CONFIG_FIELDS = (
    "workload_ema_decay",
    "workload_total_error_budget_ratio",
    "workload_master_weight_min",
    "workload_master_weight_max",
    "workload_time_epsilon",
    "workload_initial_master_weight",
    "workload_initial_subproblem_weight",
)
NULLABLE_SELECTED_ALGORITHM_FIELDS = {
    "secondary_cut_warmup_cuts",
    "secondary_cut_master_time_share_trigger",
    "secondary_cut_recent_master_time_trigger",
    "secondary_generation_lb_window",
    "secondary_generation_stall_threshold",
    "secondary_generation_cooldown_iterations",
    "secondary_generation_max_subproblem_time_share",
    "secondary_generation_min_remaining_time",
    "secondary_generation_min_solve_budget",
    "relative_cut_threshold",
    "subproblem_gap_schedule",
    "subproblem_time_budget_per_iteration",
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
    "solved_to_tolerance",
    "objective",
    "best_bound",
    "lower_bound",
    "upper_bound",
    "final_gap",
    "runtime",
    "time_limit",
    "penalized_runtime_par2",
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
    "workload_aware_policy_enabled",
    "workload_final_master_time_ema",
    "workload_final_subproblem_time_ema",
    "workload_final_master_weight",
    "workload_final_subproblem_weight",
    "workload_mean_master_weight",
    "workload_mean_subproblem_weight",
    "workload_fallback_count",
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
    "best_y_values",
    "best_x_values",
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
    "run_key",
    "config_sha256",
    "git_commit",
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
    "mean_penalized_runtime_par2",
    "mean_solved_runtime",
    "mean_unsolved_final_gap",
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
    "workload_policy_active",
    "workload_ema_decay",
    "workload_master_time_ema",
    "workload_subproblem_time_ema",
    "workload_master_share_raw",
    "workload_master_weight_selected",
    "workload_subproblem_weight_selected",
    "workload_master_ratio_selected",
    "workload_subproblem_ratio_selected",
    "workload_total_error_budget_ratio",
    "workload_fallback_used",
    "workload_fallback_reason",
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
    atomic_write_csv(path, rows, fields, value_encoder=_csv_value)


def _configured_or_default(
    config: dict[str, Any],
    field: str,
    default: Any,
) -> Any:
    value = config.get(field)
    return default if value is None else value


def _base_config(exp_cfg: dict[str, Any], size_name: str, seed: int, alpha: float | None = None) -> dict[str, Any]:
    size_cfg = dict(INSTANCE_SIZES[size_name])
    size_cfg.update(exp_cfg.get("instance_overrides", {}))
    if alpha is not None:
        size_cfg["service_level"] = alpha

    gamma_target = int(exp_cfg.get("gamma_target", 2))
    gamma_schedule = exp_cfg.get("gamma_schedule") or list(range(gamma_target + 1))
    return {
        "seed": seed,
        "adaptive_gap_enabled": bool(exp_cfg.get("adaptive_gap_enabled", False)),
        "gamma_continuation_enabled": bool(
            exp_cfg.get("gamma_continuation_enabled", False)
        ),
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
            "relative_cut_threshold": float(
                _configured_or_default(exp_cfg, "relative_cut_threshold", 1e-4)
            ),
            "final_exact_gap": float(exp_cfg.get("final_exact_gap", 1e-2)),
            "cut_stall_patience": int(exp_cfg.get("cut_stall_patience", 5)),
            "adaptive_secondary_cut_selection_enabled": bool(
                exp_cfg.get("adaptive_secondary_cut_selection_enabled", False)
            ),
            "secondary_cut_warmup_cuts": int(
                _configured_or_default(exp_cfg, "secondary_cut_warmup_cuts", 50)
            ),
            "secondary_cut_master_time_share_trigger": float(
                _configured_or_default(
                    exp_cfg,
                    "secondary_cut_master_time_share_trigger",
                    0.35,
                )
            ),
            "secondary_cut_recent_master_time_trigger": float(
                _configured_or_default(
                    exp_cfg,
                    "secondary_cut_recent_master_time_trigger",
                    0.5,
                )
            ),
            "adaptive_secondary_generation_enabled": bool(
                exp_cfg.get("adaptive_secondary_generation_enabled", False)
            ),
            "secondary_generation_lb_window": int(
                _configured_or_default(exp_cfg, "secondary_generation_lb_window", 5)
            ),
            "secondary_generation_stall_threshold": float(
                _configured_or_default(
                    exp_cfg,
                    "secondary_generation_stall_threshold",
                    1e-4,
                )
            ),
            "secondary_generation_cooldown_iterations": int(
                _configured_or_default(
                    exp_cfg,
                    "secondary_generation_cooldown_iterations",
                    5,
                )
            ),
            "secondary_generation_max_subproblem_time_share": float(
                _configured_or_default(
                    exp_cfg,
                    "secondary_generation_max_subproblem_time_share",
                    0.75,
                )
            ),
            "secondary_generation_min_remaining_time": float(
                _configured_or_default(
                    exp_cfg,
                    "secondary_generation_min_remaining_time",
                    2.0,
                )
            ),
            "secondary_generation_min_solve_budget": float(
                _configured_or_default(
                    exp_cfg,
                    "secondary_generation_min_solve_budget",
                    1.0,
                )
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
            "fixed_master_mip_gap": float(
                exp_cfg.get(
                    "fixed_master_mip_gap",
                    exp_cfg.get("mip_gap", exp_cfg.get("initial_mip_gap", 0.05)),
                )
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
            "workload_ema_decay": float(
                exp_cfg.get("workload_ema_decay", 0.80)
            ),
            "workload_total_error_budget_ratio": float(
                exp_cfg.get("workload_total_error_budget_ratio", 0.75)
            ),
            "workload_master_weight_min": float(
                exp_cfg.get("workload_master_weight_min", 1.0 / 3.0)
            ),
            "workload_master_weight_max": float(
                exp_cfg.get("workload_master_weight_max", 2.0 / 3.0)
            ),
            "workload_time_epsilon": float(
                exp_cfg.get("workload_time_epsilon", 1.0e-9)
            ),
            "workload_initial_master_weight": float(
                exp_cfg.get("workload_initial_master_weight", 1.0 / 3.0)
            ),
            "workload_initial_subproblem_weight": float(
                exp_cfg.get("workload_initial_subproblem_weight", 2.0 / 3.0)
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
    flags = {
        "adaptive_gap_enabled": bool(
            variant.get(
                "adaptive_gap_enabled",
                config.get("adaptive_gap_enabled", False),
            )
        ),
        "gamma_continuation_enabled": bool(
            variant.get(
                "gamma_continuation_enabled",
                config.get("gamma_continuation_enabled", False),
            )
        ),
        "cut_selection_enabled": bool(
            variant.get(
                "cut_selection_enabled",
                config["algorithm"].get("cut_selection_enabled", False),
            )
        ),
    }
    config["adaptive_gap_enabled"] = flags["adaptive_gap_enabled"]
    config["gamma_continuation_enabled"] = flags["gamma_continuation_enabled"]
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
        "fixed_master_mip_gap",
        "fixed_subproblem_mip_gap",
        "master_error_budget_ratio",
        "subproblem_error_budget_ratio",
        "monotone_precision_tightening",
        *WORKLOAD_AWARE_CONFIG_FIELDS,
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
    if (
        not flags["adaptive_gap_enabled"]
        and config["algorithm"].get("precision_policy", "legacy") == "legacy"
    ):
        final_gap = float(config["benders"]["final_mip_gap"])
        config["benders"]["initial_mip_gap"] = final_gap
        config["algorithm"]["fixed_master_mip_gap"] = final_gap

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
        config["algorithm"]["fixed_master_mip_gap"] = float(
            config["benders"]["final_mip_gap"]
        )
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
        config["algorithm"]["fixed_master_mip_gap"] = configured_initial_mip_gap
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
    elif method == "proposed_adaptive_benders":
        flags = {
            "adaptive_gap_enabled": bool(config["adaptive_gap_enabled"]),
            "gamma_continuation_enabled": bool(
                config["gamma_continuation_enabled"]
            ),
            "cut_selection_enabled": bool(
                config["algorithm"].get("cut_selection_enabled", False)
            ),
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
    time_limit: float | None = None,
    solve_tolerance: float = SOLVE_TOLERANCE,
) -> dict[str, Any]:
    meta = result.metadata
    normalized_status = normalize_run_status(result.status)
    solved_to_tolerance = bool(
        result.objective is not None
        and result.gap is not None
        and float(result.gap) <= float(solve_tolerance)
    )
    configured_time_limit = float(time_limit) if time_limit is not None else float(result.runtime)
    return {
        "experiment_name": exp_name,
        "instance_name": instance.name,
        "instance_size": size_name,
        "seed": seed,
        "method": method,
        "variant_name": variant_name,
        "subproblem_mode": meta.get("subproblem_mode"),
        "status": normalized_status,
        "solved_to_tolerance": solved_to_tolerance,
        "objective": result.objective,
        "best_bound": result.lower_bound,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "final_gap": result.gap,
        "runtime": result.runtime,
        "time_limit": configured_time_limit,
        "penalized_runtime_par2": penalized_runtime_par2(
            solved_to_tolerance=solved_to_tolerance,
            runtime=result.runtime,
            time_limit=configured_time_limit,
        ),
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
        "workload_aware_policy_enabled": meta.get(
            "workload_aware_policy_enabled", False
        ),
        "workload_final_master_time_ema": meta.get(
            "workload_final_master_time_ema"
        ),
        "workload_final_subproblem_time_ema": meta.get(
            "workload_final_subproblem_time_ema"
        ),
        "workload_final_master_weight": meta.get(
            "workload_final_master_weight"
        ),
        "workload_final_subproblem_weight": meta.get(
            "workload_final_subproblem_weight"
        ),
        "workload_mean_master_weight": meta.get(
            "workload_mean_master_weight"
        ),
        "workload_mean_subproblem_weight": meta.get(
            "workload_mean_subproblem_weight"
        ),
        "workload_fallback_count": meta.get("workload_fallback_count", 0),
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
        "best_y_values": meta.get("best_y_values"),
        "best_x_values": meta.get("best_x_values"),
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
    time_limit: float | None = None,
    solve_tolerance: float = SOLVE_TOLERANCE,
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
        time_limit=time_limit,
        solve_tolerance=solve_tolerance,
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
        par2_values = [
            float(r["penalized_runtime_par2"])
            for r in rows
            if r.get("penalized_runtime_par2") not in (None, "")
        ]
        solved_runtimes = [
            float(r["runtime"])
            for r in solved
            if r.get("runtime") not in (None, "")
        ]
        unsolved_gaps = [
            float(r["final_gap"])
            for r in rows
            if r not in solved and r.get("final_gap") not in (None, "")
        ]
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
                "mean_penalized_runtime_par2": _mean(par2_values),
                "mean_solved_runtime": _mean(solved_runtimes),
                "mean_unsolved_final_gap": _mean(unsolved_gaps),
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
    explicit = row.get("solved_to_tolerance")
    if explicit not in (None, ""):
        return explicit in {True, "True", "true", 1, "1"}
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
    run_key: str | None = None,
) -> Path:
    filename = _safe_filename(
        run_key
        or f"{experiment_name}__{instance_name}__seed_{seed}__{method}__{variant_name}"
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


def _apply_selected_parameters(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    selected_parameters_path = config.get("parameters_must_be_fixed_from")
    if not selected_parameters_path:
        return config

    selected = load_config(str(selected_parameters_path))
    if selected.get("selection_status") != "selected":
        raise ValueError(
            "Final evaluation is locked until selected_algorithm_parameters.yaml has selection_status: selected."
        )
    absent = [field for field in SELECTED_PARAMETER_FIELDS if field not in selected]
    if absent:
        raise ValueError(f"Selected algorithm parameters are missing: {', '.join(absent)}")
    missing = [
        field
        for field in SELECTED_PARAMETER_FIELDS
        if field not in NULLABLE_SELECTED_ALGORITHM_FIELDS
        and selected.get(field) is None
    ]
    if missing:
        raise ValueError(f"Selected algorithm parameters are missing: {', '.join(missing)}")

    boolean_fields = (
        "cut_selection_enabled",
        "final_certification_enabled",
        "adaptive_master_precision_enabled",
        "adaptive_subproblem_precision_enabled",
        "monotone_precision_tightening",
        "adaptive_subproblem_gap_enabled",
        "adaptive_secondary_cut_selection_enabled",
        "adaptive_secondary_generation_enabled",
        "adaptive_gap_enabled",
        "gamma_continuation_enabled",
    )
    for field in boolean_fields:
        if not isinstance(selected[field], bool):
            raise ValueError(f"Selected {field} must be true or false.")

    if selected["precision_policy"] not in {"legacy", "joint_error_budget"}:
        raise ValueError(
            "Selected precision_policy must be 'legacy' or 'joint_error_budget'."
        )
    if selected["cut_selection_mode"] not in {"absolute", "relative"}:
        raise ValueError("Selected cut_selection_mode must be 'absolute' or 'relative'.")
    if (
        selected["cut_selection_mode"] != "relative"
        and selected.get("relative_cut_threshold") is not None
    ):
        raise ValueError(
            "relative_cut_threshold requires selected cut_selection_mode='relative'."
        )
    if (
        selected["cut_selection_enabled"]
        and selected["cut_selection_mode"] == "relative"
        and selected.get("relative_cut_threshold") is None
    ):
        raise ValueError(
            "Selected relative_cut_threshold is required for relative cut selection."
        )

    for field in (
        "master_gap_max",
        "master_gap_min",
        "subproblem_gap_max",
        "subproblem_gap_min",
        "fixed_master_mip_gap",
        "fixed_subproblem_mip_gap",
        "master_error_budget_ratio",
        "subproblem_error_budget_ratio",
    ):
        value = selected[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"Selected {field} must be a finite nonnegative value.")
    if float(selected["master_gap_min"]) > float(selected["master_gap_max"]):
        raise ValueError("Selected master_gap_min must not exceed master_gap_max.")
    if float(selected["subproblem_gap_min"]) > float(selected["subproblem_gap_max"]):
        raise ValueError("Selected subproblem_gap_min must not exceed subproblem_gap_max.")

    patience = selected["final_certification_no_cut_patience"]
    if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
        raise ValueError(
            "Selected final_certification_no_cut_patience must be a positive integer."
        )
    max_cuts = selected["max_cuts_per_iteration"]
    if isinstance(max_cuts, bool) or not isinstance(max_cuts, int) or max_cuts <= 0:
        raise ValueError("Selected max_cuts_per_iteration must be a positive integer.")
    gamma_schedule = selected["gamma_schedule"]
    if (
        not isinstance(gamma_schedule, list)
        or not gamma_schedule
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in gamma_schedule)
    ):
        raise ValueError("Selected gamma_schedule must be a nonempty list of nonnegative integers.")

    if selected["adaptive_secondary_cut_selection_enabled"]:
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

    if selected["adaptive_secondary_generation_enabled"]:
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

    for field in SELECTED_PARAMETER_FIELDS:
        config[field] = deepcopy(selected[field])
    return config


def experiment_run_specs(config: dict[str, Any]) -> list[ProtocolRunSpec]:
    exp_name = str(config.get("experiment_name", "experiment_suite"))
    variants = _variant_specs(config)
    specs: list[ProtocolRunSpec] = []
    for seed, size_name, _alpha_key, gamma_override, alpha_value in _expanded_dimensions(config):
        axis = "gamma_target" if gamma_override is not None else "service_level" if alpha_value is not None else None
        value: int | float | None = gamma_override if gamma_override is not None else alpha_value
        for variant_name, _method, _variant in variants:
            specs.append(
                ProtocolRunSpec(
                    experiment_name=exp_name,
                    instance_size=size_name,
                    seed=seed,
                    variant_name=variant_name,
                    sensitivity_axis=axis,
                    sensitivity_value=value,
                )
            )
    return specs


def experiment_dry_run_report(config: dict[str, Any]) -> dict[str, Any]:
    resolved = _apply_selected_parameters(config)
    specs = experiment_run_specs(resolved)
    time_limit = float(resolved.get("time_limit", 0.0))
    audit_errors: list[str] = []
    experiment_name = str(resolved.get("experiment_name", ""))
    if experiment_name.startswith("workload_aware_joint_v2_"):
        try:
            from .workload_aware_v2_audit import audit_workload_aware_v2

            audit = audit_workload_aware_v2()
            audit_errors = [
                str(check["check"])
                for check in audit["checks"]
                if check.get("required", True) and not check.get("passed", False)
            ]
        except Exception as exc:  # noqa: BLE001 - dry-run reports audit failures.
            audit_errors = [f"audit_execution_failed: {exc}"]
    elif experiment_name in {
        "large_scale_evaluation_joint_v1",
        "managerial_sensitivity_joint_v1",
    }:
        try:
            from .extended_experiment_audit import audit_protocols

            audit = audit_protocols()
            audit_errors = [
                str(check["check"])
                for check in audit["checks"]
                if check.get("required", True) and not check.get("passed", False)
            ]
        except Exception as exc:  # noqa: BLE001 - dry-run must report audit import failures.
            audit_errors = [f"audit_execution_failed: {exc}"]
    return {
        "experiment_name": resolved.get("experiment_name"),
        "total_run_count": len(specs),
        "run_count_by_axis": {"none": len(specs)},
        "seeds": sorted({spec.seed for spec in specs}),
        "instance_sizes": sorted({spec.instance_size for spec in specs}),
        "methods": [name for name, _method, _variant in _variant_specs(resolved)],
        "output_dir": resolved.get("output_dir"),
        "time_limit_seconds": time_limit,
        "theoretical_maximum_seconds": len(specs) * time_limit,
        "theoretical_maximum_hours": theoretical_maximum_hours(len(specs), time_limit),
        "serial_upper_bound_not_runtime_prediction": True,
        "automatic_parallelism_enabled": False,
        "protocol_audit_errors": audit_errors,
    }


def run_experiment_suite(
    config: dict[str, Any],
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    config = _apply_selected_parameters(config)
    exp_name = str(config.get("experiment_name", "experiment_suite"))
    _validate_relative_threshold_config(config)
    output_dir = Path(str(config.get("output_dir", f"experiments/results/{exp_name}")))
    instances_dir = output_dir / "instances"
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = atomic_write_yaml(output_dir / "resolved_config.yaml", config)

    config_hash = config_sha256(config)
    commit = git_commit(Path.cwd())
    specs = experiment_run_specs(config)
    run_keys = [spec.run_key for spec in specs]
    skipped_run_count = 0
    results: list[dict[str, Any]] = []
    variants = _variant_specs(config)
    spec_index = 0
    manifest_path = update_run_manifest(
        output_dir=output_dir,
        run_keys=run_keys,
        config_hash=config_hash,
        commit=commit,
        skipped_run_count=skipped_run_count,
    )

    for seed, size_name, _alpha_key, gamma_override, alpha_value in _expanded_dimensions(config):
        run_cfg = _base_config(config, size_name, seed, alpha=alpha_value)
        if gamma_override is not None:
            run_cfg["robust"]["gamma_target"] = gamma_override
            run_cfg["robust"]["gamma_schedule"] = list(range(gamma_override + 1))
        instance = generate_instance(run_cfg, seed=seed)
        instance_path = instances_dir / f"{instance.name}.json"
        atomic_write_json(instance_path, instance.to_dict())
        for variant_name, method, variant in variants:
            spec = specs[spec_index]
            spec_index += 1
            run_key = spec.run_key
            existing = load_run_record(output_dir, run_key)
            action = decide_run_action(existing, resume=resume, overwrite=overwrite)
            if action.startswith("skip"):
                skipped_run_count += 1
                if existing and isinstance(existing.get("result"), dict):
                    results.append(dict(existing["result"]))
                manifest_path = update_run_manifest(
                    output_dir=output_dir,
                    run_keys=run_keys,
                    config_hash=config_hash,
                    commit=commit,
                    skipped_run_count=skipped_run_count,
                )
                continue

            run_dir = output_dir / "runs" / run_key
            resolution_error: Exception | None = None
            try:
                _solver_method, _resolved_flags, method_resolved_config = _apply_variant_config(
                    deepcopy(run_cfg), method, variant
                )
            except Exception as exc:  # noqa: BLE001 - unsupported methods become failed rows.
                resolution_error = exc
                method_resolved_config = deepcopy(run_cfg)
                method_resolved_config["resolution_error"] = f"{type(exc).__name__}: {exc}"
            run_config_hash = config_sha256(method_resolved_config)
            atomic_write_yaml(run_dir / "resolved_config.yaml", method_resolved_config)
            write_run_state(
                output_dir,
                run_key,
                state="running",
                details={
                    "config_sha256": run_config_hash,
                    "git_commit": commit,
                    "started_at": utc_now_iso(),
                    "action": action,
                },
            )
            try:
                if resolution_error is not None:
                    raise resolution_error
                result, flags = _solve_experiment_method(
                    deepcopy(run_cfg), instance, method, variant
                )
                row = _result_row(
                    exp_name,
                    size_name,
                    seed,
                    method,
                    variant_name,
                    result,
                    flags,
                    instance,
                    instance_path,
                    time_limit=float(run_cfg["benders"]["time_limit"]),
                    solve_tolerance=float(run_cfg["benders"]["tol"]),
                )
                if bool(config.get("save_iteration_log", False)):
                    iteration_log_path = _write_iteration_log(
                        output_dir,
                        exp_name,
                        instance.name,
                        seed,
                        method,
                        variant_name,
                        result.iteration_log,
                        run_key=run_key,
                    )
                    row["iteration_log_path"] = str(iteration_log_path)
                if gamma_override is not None:
                    row["gamma_target"] = gamma_override
                if alpha_value is not None:
                    row["alpha"] = alpha_value
                success = row["status"] not in {"failed", "skipped"}
                atomic_write_text(run_dir / "error.txt", "")
            except Exception as exc:  # noqa: BLE001 - experiments must keep running after failed methods.
                flags = {
                    "adaptive_gap_enabled": bool(variant.get("adaptive_gap_enabled", False)),
                    "gamma_continuation_enabled": bool(variant.get("gamma_continuation_enabled", False)),
                    "cut_selection_enabled": bool(variant.get("cut_selection_enabled", False)),
                }
                row = _failure_row(
                    exp_name,
                    size_name,
                    seed,
                    method,
                    variant_name,
                    flags,
                    instance,
                    instance_path,
                    exc,
                    time_limit=float(run_cfg["benders"]["time_limit"]),
                    solve_tolerance=float(run_cfg["benders"]["tol"]),
                )
                success = False
                atomic_write_text(run_dir / "error.txt", f"{type(exc).__name__}: {exc}\n")
                if bool(config.get("save_iteration_log", False)):
                    row["iteration_log_path"] = str(
                        _write_iteration_log(
                            output_dir,
                            exp_name,
                            instance.name,
                            seed,
                            method,
                            variant_name,
                            [],
                            run_key=run_key,
                        )
                    )

            row.update(
                {
                    "run_key": run_key,
                    "config_sha256": run_config_hash,
                    "git_commit": commit,
                }
            )
            results.append(row)
            now = utc_now_iso()
            record = {
                "run_key": run_key,
                "state": "complete",
                "success": success,
                "solved_to_tolerance": bool(row.get("solved_to_tolerance")),
                "created_at": (existing or {}).get("created_at", now),
                "updated_at": now,
                "config_sha256": run_config_hash,
                "git_commit": commit,
                "result": row,
            }
            atomic_write_json(run_dir / "run.json", record)
            write_run_state(
                output_dir,
                run_key,
                state="complete",
                details={
                    "success": success,
                    "solved_to_tolerance": bool(row.get("solved_to_tolerance")),
                    "status": row.get("status"),
                },
            )
            manifest_path = update_run_manifest(
                output_dir=output_dir,
                run_keys=run_keys,
                config_hash=config_hash,
                commit=commit,
                skipped_run_count=skipped_run_count,
            )

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
        "run_manifest": manifest_path,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robust inventory experiment suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(experiment_dry_run_report(config), ensure_ascii=False, indent=2))
        return
    outputs = run_experiment_suite(
        config,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
