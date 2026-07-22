from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import yaml

from .benders import solve_benders
from .experiment_protocol import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_yaml,
    config_sha256,
    decide_run_action,
    git_commit,
    load_run_record,
    penalized_runtime_par2,
    read_json,
    stable_run_key,
    update_run_manifest,
    utc_now_iso,
    write_run_state,
)
from .experiment_suite import (
    INSTANCE_SIZES,
    _apply_selected_parameters,
    _apply_variant_config,
    _base_config,
)
from .instance import InventoryInstance, generate_instance
from .precision_policy import (
    initialize_precision_state,
    precision_policy_config,
    select_joint_error_budget_precision,
)
from .regional_fairness_pipeline import SingleWriterLock
from .robust_regional_fairness import (
    FAIRNESS_FEASIBILITY_TOLERANCE,
    FairnessFeasibilityCut,
    evaluate_fairness_solution,
    fairness_cost_budget,
    separate_robust_fairness_feasibility,
)
from .status import gurobi_status_name


@dataclass(frozen=True)
class FairnessBendersResult:
    status: str
    objective_t: float | None
    robust_minimum_fill_rate: float | None
    lower_bound: float | None
    upper_bound: float | None
    gap: float | None
    runtime: float
    iterations: int
    cuts: int
    cuts_with_cost_component: int
    cuts_with_fairness_component: int
    joint_cost_fairness_cuts: int
    baseline_cost: float
    rho: float
    cost_budget: float
    y_values: list[float] | None
    x_values: list[list[float]] | None
    master_runtime: float
    separation_runtime: float
    separation_patterns_seen: list[list[dict[str, int]]] = field(default_factory=list)
    iteration_log: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def relative_gap(upper_bound: float | None, lower_bound: float | None) -> float | None:
    if upper_bound is None or lower_bound is None:
        return None
    return max(0.0, (float(upper_bound) - float(lower_bound)) / max(1.0, abs(float(upper_bound))))


def _build_master(instance: InventoryInstance, output_flag: bool) -> tuple[gp.Model, Any, Any, Any]:
    model = gp.Model("robust_regional_fairness_master")
    model.Params.OutputFlag = 1 if output_flag else 0
    y = model.addVars(instance.I, vtype=GRB.BINARY, name="y")
    x = model.addVars(instance.I, instance.J, lb=0.0, name="x")
    t = model.addVar(lb=0.0, ub=1.0, name="T")
    for i in instance.I:
        model.addConstr(
            gp.quicksum(instance.volume[j] * x[i, j] for j in instance.J)
            <= instance.capacity[i] * y[i],
            name=f"capacity[{i}]",
        )
        for j in instance.J:
            model.addConstr(x[i, j] <= instance.inventory_ub[i][j] * y[i], name=f"logic[{i},{j}]")
    first_stage = gp.quicksum(instance.fixed_cost[i] * y[i] for i in instance.I) + gp.quicksum(
        instance.inventory_cost[i][j] * x[i, j] for i in instance.I for j in instance.J
    )
    model.addConstr(first_stage <= instance.budget, name="first_stage_budget")
    model.setObjective(t, GRB.MINIMIZE)
    return model, y, x, t


def _add_fairness_cut(
    model: gp.Model,
    y: Any,
    x: Any,
    t: Any,
    cut: FairnessFeasibilityCut,
    index: int,
) -> None:
    model.addConstr(
        cut.constant
        + gp.quicksum(cut.y_coefficients[i] * y[i] for i in range(len(cut.y_coefficients)))
        + gp.quicksum(
            cut.x_coefficients[i][j] * x[i, j]
            for i in range(len(cut.x_coefficients))
            for j in range(len(cut.x_coefficients[i]))
        )
        + cut.t_coefficient * t
        >= 0.0,
        name=f"fairness_feasibility_cut[{index}]",
    )


def _cut_key(cut: FairnessFeasibilityCut, digits: int = 10) -> tuple[Any, ...]:
    return (
        round(cut.constant, digits),
        tuple(round(value, digits) for value in cut.y_coefficients),
        tuple(round(value, digits) for row in cut.x_coefficients for value in row),
        round(cut.t_coefficient, digits),
    )


def solve_fairness_benders(
    instance: InventoryInstance,
    *,
    baseline_cost: float,
    rho: float,
    gamma: int = 2,
    algorithm_config: dict[str, Any] | None = None,
    max_iterations: int = 10000,
    time_limit: float = 600.0,
    tol: float = 1.0e-4,
    feasibility_tolerance: float = FAIRNESS_FEASIBILITY_TOLERANCE,
    output_flag: bool = False,
) -> FairnessBendersResult:
    """Constraint-generation algorithm for the robust max-min service model.

    The original frozen V3 solver is used to obtain ``baseline_cost`` outside
    this routine.  Its core-point strengthening remains unchanged there.  New
    fairness feasibility cuts are Farkas cuts in ``(y,x,T)`` and are deliberately
    not passed through the V3 recourse-cut core-point LP, whose validity proof
    applies to a different cut family.
    """
    start = time.perf_counter()
    budget = fairness_cost_budget(baseline_cost, rho)
    cfg = deepcopy(algorithm_config or {})
    precision = precision_policy_config(
        cfg,
        fixed_master_gap=float(cfg.get("fixed_master_mip_gap", 0.02)),
        fixed_subproblem_gap=float(cfg.get("fixed_subproblem_mip_gap", 0.05)),
        legacy_subproblem_gaps=[0.05, 0.0001],
    )
    if precision.precision_policy != "joint_error_budget":
        raise ValueError("The fairness development algorithm must retain precision_policy=joint_error_budget.")
    state = initialize_precision_state(precision)
    model, y, x, t = _build_master(instance, output_flag)
    lower_bound: float | None = None
    upper_bound: float | None = None
    best_y: list[float] | None = None
    best_x: list[list[float]] | None = None
    cuts = 0
    cost_component_cuts = 0
    fairness_component_cuts = 0
    joint_component_cuts = 0
    seen_cuts: set[tuple[Any, ...]] = set()
    master_runtime = 0.0
    separation_runtime = 0.0
    log: list[dict[str, Any]] = []
    patterns_seen: list[list[dict[str, int]]] = []
    status = "iteration_limit"
    certification_active = False

    try:
        for iteration in range(1, int(max_iterations) + 1):
            elapsed = time.perf_counter() - start
            remaining = float(time_limit) - elapsed
            if remaining <= 0.0:
                status = "time_limit"
                break
            decision = select_joint_error_budget_precision(
                precision,
                state,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
                update_state=not certification_active,
            )
            state = decision.next_state
            master_gap = 0.0 if certification_active else decision.master_selected_gap
            subproblem_gap = 0.0 if certification_active else decision.subproblem_selected_gap
            model.Params.MIPGap = master_gap
            model.Params.TimeLimit = max(1.0e-3, remaining)
            master_start = time.perf_counter()
            model.optimize()
            master_elapsed = time.perf_counter() - master_start
            master_runtime += master_elapsed
            master_status = gurobi_status_name(model.Status)
            if model.SolCount <= 0:
                status = "infeasible" if model.Status == GRB.INFEASIBLE else master_status
                break
            candidate_lb = float(model.ObjBound)
            lower_bound = candidate_lb if lower_bound is None else max(lower_bound, candidate_lb)
            candidate_t = float(t.X)
            candidate_y = [float(y[i].X) for i in instance.I]
            candidate_x = [[float(x[i, j].X) for j in instance.J] for i in instance.I]
            remaining = float(time_limit) - (time.perf_counter() - start)
            if remaining <= 0.0:
                status = "time_limit"
                break
            separation = separate_robust_fairness_feasibility(
                instance,
                y_values=candidate_y,
                x_values=candidate_x,
                t_value=candidate_t,
                cost_budget_value=budget.budget,
                gamma=gamma,
                mip_gap=subproblem_gap,
                time_limit=remaining,
                feasibility_tolerance=feasibility_tolerance,
                output_flag=output_flag,
            )
            separation_runtime += separation.runtime
            cut_added = False
            duplicate_cut = False
            cut_value_at_candidate: float | None = None
            if separation.cut is not None:
                cut_value_at_candidate = separation.cut.value(candidate_y, candidate_x, candidate_t)
                key = _cut_key(separation.cut)
                if key not in seen_cuts and cut_value_at_candidate < -float(feasibility_tolerance):
                    _add_fairness_cut(model, y, x, t, separation.cut, cuts)
                    seen_cuts.add(key)
                    cuts += 1
                    has_cost_component = separation.cut.ray.cost > feasibility_tolerance
                    has_fairness_component = any(
                        value > feasibility_tolerance
                        for value in separation.cut.ray.regional_fairness
                    )
                    cost_component_cuts += int(has_cost_component)
                    fairness_component_cuts += int(has_fairness_component)
                    joint_component_cuts += int(
                        has_cost_component and has_fairness_component
                    )
                    cut_added = True
                    patterns_seen.append(separation.cut.active_deviations)
                else:
                    duplicate_cut = key in seen_cuts
            if separation.robust_feasibility_certified:
                if upper_bound is None or candidate_t < upper_bound:
                    upper_bound = candidate_t
                    best_y = candidate_y
                    best_x = candidate_x
            gap = relative_gap(upper_bound, lower_bound)
            log.append(
                {
                    "iteration": iteration,
                    "master_status": master_status,
                    "master_requested_mip_gap": master_gap,
                    "separation_status": separation.status,
                    "separation_requested_mip_gap": subproblem_gap,
                    "separation_objective": separation.objective,
                    "separation_objective_bound": separation.objective_bound,
                    "separation_has_incumbent": separation.has_incumbent,
                    "robust_feasibility_certified": separation.robust_feasibility_certified,
                    "separation_certification_reason": separation.certification_reason,
                    "separation_candidate_active_deviations": separation.candidate_active_deviations,
                    "separation_incumbent_ray_validation": (
                        None
                        if separation.incumbent_ray_validation is None
                        else separation.incumbent_ray_validation.to_dict()
                    ),
                    "fixed_scenario_certificate": (
                        None
                        if separation.fixed_scenario_certificate is None
                        else separation.fixed_scenario_certificate.to_dict()
                    ),
                    "separation_false_positive_scenarios_excluded": (
                        separation.false_positive_scenarios_excluded
                    ),
                    "separation_cut_certificate_source": separation.cut_certificate_source,
                    "cut_added": cut_added,
                    "cut_has_cost_component": (
                        None
                        if separation.cut is None
                        else separation.cut.ray.cost > feasibility_tolerance
                    ),
                    "cut_has_fairness_component": (
                        None
                        if separation.cut is None
                        else any(
                            value > feasibility_tolerance
                            for value in separation.cut.ray.regional_fairness
                        )
                    ),
                    "duplicate_cut": duplicate_cut,
                    "cut_value_at_candidate": cut_value_at_candidate,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "global_gap": gap,
                    "candidate_t": candidate_t,
                    "certification_active": certification_active,
                    "fairness_cut_core_point_strengthened": False,
                    "fairness_cut_core_point_skip_reason": "different_farkas_cut_family",
                    "master_runtime": master_elapsed,
                    "separation_runtime": separation.runtime,
                }
            )
            if upper_bound is not None and gap is not None and gap <= float(tol):
                if certification_active:
                    if separation.robust_feasibility_certified and model.Status == GRB.OPTIMAL:
                        status = "optimal"
                        break
                else:
                    certification_active = True
                    continue
            if certification_active and cut_added:
                certification_active = False
            if not cut_added and not separation.robust_feasibility_certified:
                status = (
                    "separation_stalled_duplicate"
                    if duplicate_cut
                    else separation.status
                )
                break
        else:
            status = "iteration_limit"
    finally:
        model.dispose()

    runtime = time.perf_counter() - start
    gap = relative_gap(upper_bound, lower_bound)
    return FairnessBendersResult(
        status=status,
        objective_t=upper_bound,
        robust_minimum_fill_rate=None if upper_bound is None else 1.0 - upper_bound,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        gap=gap,
        runtime=runtime,
        iterations=len(log),
        cuts=cuts,
        cuts_with_cost_component=cost_component_cuts,
        cuts_with_fairness_component=fairness_component_cuts,
        joint_cost_fairness_cuts=joint_component_cuts,
        baseline_cost=budget.baseline_cost,
        rho=budget.rho,
        cost_budget=budget.budget,
        y_values=best_y,
        x_values=best_x,
        master_runtime=master_runtime,
        separation_runtime=separation_runtime,
        separation_patterns_seen=patterns_seen,
        iteration_log=log,
        metadata={
            "precision_policy": precision.precision_policy,
            "master_error_budget_ratio": precision.master_error_budget_ratio,
            "subproblem_error_budget_ratio": precision.subproblem_error_budget_ratio,
            "core_point_baseline_policy": "core_point",
            "core_point_applied_to_fairness_farkas_cuts": False,
            "core_point_fairness_reason": "not_validated_for_y_x_T_farkas_cut_family",
            "same_recourse_satisfies_cost_and_fairness": True,
            "cost_and_fairness_worst_separately_identified_in_post_evaluation": True,
            "separation_mode": "candidate_milp_plus_fixed_scenario_lp_certificate",
            "separation_incumbent_role": "candidate_scenario_only",
            "fixed_scenario_certificate_required_for_cut": True,
            "secondary_cut_enabled": False,
        },
    )


def load_development_config(path: str | Path) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Development config must contain a YAML mapping.")
    return loaded


def development_run_plan(config: dict[str, Any]) -> dict[str, Any]:
    seeds = [int(value) for value in config.get("random_seeds", [])]
    rhos = [float(value) for value in config.get("fairness_development", {}).get("rho_grid", [])]
    sizes = [str(value) for value in config.get("instance_sizes", [])]
    baseline = len(seeds) * len(sizes)
    frontier = baseline * len(rhos)
    baseline_limit = float(config.get("baseline_time_limit", config.get("time_limit", 0.0)))
    fairness_limit = float(config.get("fairness_time_limit", config.get("time_limit", 0.0)))
    scenario_counts = {
        size: sum(
            math.comb(INSTANCE_SIZES[size]["num_products"] * INSTANCE_SIZES[size]["num_regions"], k)
            for k in range(int(config.get("gamma_target", 2)) + 1)
        )
        for size in sizes
    }
    return {
        "experiment_name": config.get("experiment_name"),
        "protocol_phase": config.get("protocol_phase"),
        "instance_sizes": sizes,
        "seeds": seeds,
        "rho_grid": rhos,
        "method": config.get("variants", [None])[0],
        "baseline_run_count": baseline,
        "fairness_frontier_run_count": frontier,
        "total_computational_run_count": baseline + frontier,
        "output_dir": config.get("output_dir"),
        "scenario_count_by_size": scenario_counts,
        "theoretical_serial_upper_bound_seconds": baseline * baseline_limit + frontier * fairness_limit,
        "automatic_parallelism_enabled": False,
        "instances_generated": False,
        "solver_called": False,
    }


def _baseline_method_config(config: dict[str, Any], instance_size: str, seed: int) -> tuple[str, dict[str, Any]]:
    resolved = _apply_selected_parameters(config)
    base = _base_config(resolved, instance_size, seed)
    variant = dict(resolved.get("variant_settings", {}).get("joint_v1_core_point_strengthened", {}))
    method, _flags, method_config = _apply_variant_config(base, "proposed_adaptive_benders", variant)
    return method, method_config


def _record_path(output_dir: Path, run_key: str) -> Path:
    return output_dir / "runs" / run_key / "run.json"


def _write_record(output_dir: Path, run_key: str, payload: dict[str, Any]) -> None:
    atomic_write_json(_record_path(output_dir, run_key), payload)


def _validate_resume_record_identity(
    record: dict[str, Any] | None,
    *,
    config_hash: str,
    commit: str,
    run_key: str,
) -> None:
    if record is None:
        return
    if record.get("run_key") != run_key:
        raise ValueError(f"Run-key mismatch while resuming {run_key}.")
    if record.get("config_sha256") != config_hash:
        raise ValueError(f"Config identity mismatch while resuming {run_key}.")
    if record.get("git_commit") != commit:
        raise ValueError(f"Git-commit identity mismatch while resuming {run_key}.")


FAIRNESS_DEVELOPMENT_MANIFEST_SCHEMA_VERSION = 1


def _certified_baseline_anchor(
    baseline_record: dict[str, Any],
    *,
    baseline_run_key: str,
    config_hash: str,
    commit: str,
    candidate_config_sha256: str,
    tolerance: float,
) -> dict[str, Any]:
    """Freeze the exact certified V3 conservative UB used by every rho."""
    result = baseline_record.get("result", {})
    upper_bound = result.get("upper_bound")
    gap = result.get("gap")
    solved = (
        baseline_record.get("solved_to_tolerance") is True
        and result.get("status") == "optimal"
        and result.get("valid_UB") is True
        and gap is not None
        and float(gap) <= float(tolerance)
        and upper_bound is not None
        and math.isfinite(float(upper_bound))
    )
    if not solved:
        raise RuntimeError(
            f"Frozen V3 baseline lacks a certified feasible robust upper bound for {baseline_run_key}."
        )
    value = float(upper_bound)
    anchor = {
        "source": "solve_result.upper_bound",
        "value": value,
        "value_hex": value.hex(),
        "baseline_run_key": baseline_run_key,
        "base_git_commit": commit,
        "base_config_sha256": config_hash,
        "candidate_config_sha256": str(candidate_config_sha256).upper(),
        "valid_UB": True,
        "baseline_status": "optimal",
        "baseline_final_gap": float(gap),
    }
    anchor["anchor_sha256"] = config_sha256(anchor)
    return anchor


def _validate_frontier_anchor_identity(
    record: dict[str, Any] | None,
    *,
    anchor: dict[str, Any],
    baseline_run_key: str,
    rho: float,
) -> None:
    if record is None:
        return
    if record.get("baseline_run_key") != baseline_run_key:
        raise ValueError("Baseline run-key identity mismatch while resuming fairness frontier.")
    if record.get("baseline_anchor_sha256") != anchor["anchor_sha256"]:
        raise ValueError("Certified C_anchor identity mismatch while resuming fairness frontier.")
    if float(record.get("rho")) != float(rho):
        raise ValueError("Rho identity mismatch while resuming fairness frontier.")


def _write_fairness_development_manifest(
    output_dir: Path,
    *,
    config: dict[str, Any],
    config_hash: str,
    commit: str,
    run_keys: list[str],
    anchors: dict[str, dict[str, Any]],
) -> None:
    previous = read_json(output_dir / "fairness_development_manifest.json") or {}
    tasks: list[dict[str, Any]] = []
    for run_key in run_keys:
        record = load_run_record(output_dir, run_key)
        tasks.append(
            {
                "run_key": run_key,
                "state": None if record is None else record.get("state"),
                "success": False if record is None else bool(record.get("success")),
                "solved_to_tolerance": (
                    False if record is None else bool(record.get("solved_to_tolerance"))
                ),
                "baseline_run_key": None if record is None else record.get("baseline_run_key"),
                "baseline_anchor_sha256": (
                    None if record is None else record.get("baseline_anchor_sha256")
                ),
                "rho": None if record is None else record.get("rho"),
                "certification_status": (
                    None if record is None else record.get("certification_status")
                ),
            }
        )
    payload = {
        "schema_version": FAIRNESS_DEVELOPMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "protocol_phase": config["protocol_phase"],
        "config_sha256": config_hash,
        "git_commit": commit,
        "candidate_config_sha256": str(config["candidate_config_sha256"]).upper(),
        "baseline_anchor_source": "solve_result.upper_bound",
        "run_keys": run_keys,
        "baseline_anchors": anchors,
        "tasks": tasks,
        "completed_count": sum(task["state"] == "complete" for task in tasks),
        "solved_count": sum(task["solved_to_tolerance"] for task in tasks),
        "pending_count": sum(task["state"] != "complete" for task in tasks),
        "created_at": previous.get("created_at", utc_now_iso()),
        "updated_at": utc_now_iso(),
    }
    atomic_write_json(output_dir / "fairness_development_manifest.json", payload)


def _write_fairness_result_tables(output_dir: Path, run_keys: list[str]) -> None:
    """Rebuild deterministic result/summary tables from atomic run records."""
    rows: list[dict[str, Any]] = []
    for run_key in run_keys:
        record = load_run_record(output_dir, run_key)
        if record is None:
            continue
        result = record.get("result", {})
        rows.append(
            {
                "run_key": run_key,
                "task_type": record.get("task_type"),
                "instance_size": record.get("instance_size"),
                "seed": record.get("seed"),
                "method": record.get("method"),
                "rho": record.get("rho"),
                "baseline_run_key": record.get("baseline_run_key"),
                "baseline_anchor_sha256": record.get("baseline_anchor_sha256"),
                "baseline_cost": record.get("baseline_cost"),
                "cost_budget": result.get("cost_budget"),
                "status": result.get("status"),
                "certification_status": record.get("certification_status"),
                "solved_to_tolerance": record.get("solved_to_tolerance"),
                "objective_t": result.get("objective_t"),
                "lower_bound": result.get("lower_bound"),
                "upper_bound": result.get("upper_bound"),
                "gap": result.get("gap"),
                "runtime": result.get("runtime"),
                "penalized_runtime_par2": result.get("penalized_runtime_par2"),
                "iterations": result.get("iterations"),
                "cuts": result.get("cuts"),
            }
        )
    rows.sort(key=lambda row: (int(row["seed"]), row["task_type"] != "baseline", -1.0 if row["rho"] is None else float(row["rho"])))
    fields = list(rows[0]) if rows else [
        "run_key", "task_type", "instance_size", "seed", "method", "rho",
        "status", "certification_status", "solved_to_tolerance",
    ]
    atomic_write_csv(output_dir / "results.csv", rows, fields)
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["task_type"], row["rho"]), []).append(row)
    summary = []
    for (task_type, rho), group in sorted(
        groups.items(), key=lambda item: (item[0][0] != "baseline", -1.0 if item[0][1] is None else float(item[0][1]))
    ):
        summary.append(
            {
                "task_type": task_type,
                "rho": rho,
                "attempt_count": len(group),
                "solved_count": sum(bool(row["solved_to_tolerance"]) for row in group),
                "optimal_count": sum(row["status"] == "optimal" for row in group),
                "time_limit_count": sum(row["status"] == "time_limit" for row in group),
                "uncertified_count": sum(
                    str(row["certification_status"] or "").startswith("uncertified_")
                    for row in group
                ),
                "invalid_post_evaluation_count": sum(
                    row["certification_status"] == "invalid_post_evaluation"
                    for row in group
                ),
                "infeasible_count": sum(row["status"] == "infeasible" for row in group),
            }
        )
    atomic_write_csv(
        output_dir / "summary.csv",
        summary,
        [
            "task_type", "rho", "attempt_count", "solved_count",
            "optimal_count", "time_limit_count", "uncertified_count",
            "invalid_post_evaluation_count", "infeasible_count",
        ],
    )


def _validate_development_manifest_identity(
    manifest: dict[str, Any] | None,
    *,
    config: dict[str, Any],
    config_hash: str,
    commit: str,
    run_keys: list[str],
) -> None:
    if manifest is None:
        return
    expected = {
        "schema_version": FAIRNESS_DEVELOPMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "protocol_phase": config["protocol_phase"],
        "config_sha256": config_hash,
        "git_commit": commit,
        "candidate_config_sha256": str(config["candidate_config_sha256"]).upper(),
        "baseline_anchor_source": "solve_result.upper_bound",
        "run_keys": run_keys,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(
            "Fairness development manifest identity mismatch: " + ", ".join(mismatches)
        )


def _run_fairness_development_locked(
    config: dict[str, Any], *, resume: bool = False, overwrite: bool = False
) -> Path:
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit(Path(__file__).resolve().parents[1])
    cfg_hash = config_sha256(config)
    seeds = [int(value) for value in config["random_seeds"]]
    size = str(config["instance_sizes"][0])
    rhos = [float(value) for value in config["fairness_development"]["rho_grid"]]
    run_keys: list[str] = []
    for seed in seeds:
        run_keys.append(stable_run_key(
            experiment_name=str(config["experiment_name"]), sensitivity_axis="stage",
            sensitivity_value="baseline", instance_size=size, seed=seed,
            variant_name="joint_v1_core_point_strengthened",
        ))
        for rho in rhos:
            run_keys.append(stable_run_key(
                experiment_name=str(config["experiment_name"]), sensitivity_axis="rho",
                sensitivity_value=rho, instance_size=size, seed=seed,
                variant_name="robust_regional_fairness",
            ))
    existing_manifest = read_json(output_dir / "fairness_development_manifest.json")
    if (
        not overwrite
        and (output_dir / "fairness_development_manifest.json").exists()
        and existing_manifest is None
    ):
        raise ValueError("Existing fairness development manifest is unreadable or corrupt.")
    if not overwrite:
        _validate_development_manifest_identity(
            existing_manifest,
            config=config,
            config_hash=cfg_hash,
            commit=commit,
            run_keys=run_keys,
        )
    atomic_write_yaml(output_dir / "resolved_config.yaml", config)
    anchors: dict[str, dict[str, Any]] = {}
    if existing_manifest is not None:
        anchors = dict(existing_manifest.get("baseline_anchors", {}))
    _write_fairness_development_manifest(
        output_dir,
        config=config,
        config_hash=cfg_hash,
        commit=commit,
        run_keys=run_keys,
        anchors=anchors,
    )
    skipped = 0
    for seed in seeds:
        instance = generate_instance(_base_config(config, size, seed), seed=seed)
        atomic_write_json(output_dir / "instances" / f"{instance.name}.json", instance.to_dict())
        baseline_key = stable_run_key(
            experiment_name=str(config["experiment_name"]), sensitivity_axis="stage",
            sensitivity_value="baseline", instance_size=size, seed=seed,
            variant_name="joint_v1_core_point_strengthened",
        )
        baseline_record = load_run_record(output_dir, baseline_key)
        if not overwrite:
            _validate_resume_record_identity(
                baseline_record, config_hash=cfg_hash, commit=commit, run_key=baseline_key
            )
        action = decide_run_action(baseline_record, resume=resume, overwrite=overwrite)
        if action == "skip_success":
            skipped += 1
        elif action.startswith("run"):
            write_run_state(output_dir, baseline_key, state="running")
            method, method_config = _baseline_method_config(config, size, seed)
            result = solve_benders(method_config, instance, method)
            solved = result.status == "optimal" and result.gap is not None and result.gap <= float(config["tol"])
            baseline_payload = result.summary_dict()
            baseline_payload["iteration_log"] = result.iteration_log
            baseline_record = {
                "run_key": baseline_key,
                "task_type": "baseline",
                "instance_size": size,
                "seed": seed,
                "method": "joint_v1_core_point_strengthened",
                "state": "complete",
                "success": solved,
                "solved_to_tolerance": solved,
                "git_commit": commit,
                "config_sha256": cfg_hash,
                "created_at": utc_now_iso(),
                "result": baseline_payload,
            }
            _write_record(output_dir, baseline_key, baseline_record)
            write_run_state(
                output_dir,
                baseline_key,
                state="complete",
                details={"success": solved, "solved_to_tolerance": solved},
            )
        if not baseline_record or not baseline_record.get("solved_to_tolerance"):
            raise RuntimeError(f"Frozen V3 baseline did not solve to tolerance for seed {seed}.")
        anchor = _certified_baseline_anchor(
            baseline_record,
            baseline_run_key=baseline_key,
            config_hash=cfg_hash,
            commit=commit,
            candidate_config_sha256=str(config["candidate_config_sha256"]),
            tolerance=float(config["tol"]),
        )
        anchors[str(seed)] = anchor
        if baseline_record.get("certified_cost_anchor") != anchor:
            baseline_record["certified_cost_anchor"] = anchor
            _write_record(output_dir, baseline_key, baseline_record)
        _write_fairness_development_manifest(
            output_dir,
            config=config,
            config_hash=cfg_hash,
            commit=commit,
            run_keys=run_keys,
            anchors=anchors,
        )
        _write_fairness_result_tables(output_dir, run_keys)
        c_anchor = float(anchor["value"])
        for rho in rhos:
            key = stable_run_key(
                experiment_name=str(config["experiment_name"]), sensitivity_axis="rho",
                sensitivity_value=rho, instance_size=size, seed=seed,
                variant_name="robust_regional_fairness",
            )
            existing = load_run_record(output_dir, key)
            if not overwrite:
                _validate_resume_record_identity(
                    existing, config_hash=cfg_hash, commit=commit, run_key=key
                )
                _validate_frontier_anchor_identity(
                    existing,
                    anchor=anchor,
                    baseline_run_key=baseline_key,
                    rho=rho,
                )
            action = decide_run_action(existing, resume=resume, overwrite=overwrite)
            if action == "skip_success":
                skipped += 1
                continue
            if not action.startswith("run"):
                continue
            write_run_state(output_dir, key, state="running")
            algorithm = load_development_config(str(config["candidate_parameters_must_be_fixed_from"]))["algorithm"]
            result = solve_fairness_benders(
                instance,
                baseline_cost=c_anchor,
                rho=rho,
                gamma=int(config["gamma_target"]),
                algorithm_config=algorithm,
                max_iterations=int(config["max_iterations"]),
                time_limit=float(config["fairness_time_limit"]),
                tol=float(config["tol"]),
                feasibility_tolerance=float(config["fairness_development"]["feasibility_tolerance"]),
                output_flag=bool(config.get("output_flag", False)),
            )
            algorithm_solved = (
                result.status == "optimal"
                and result.gap is not None
                and result.gap <= float(config["tol"])
            )
            payload = result.to_dict()
            evaluation_valid = False
            if (
                algorithm_solved
                and result.y_values is not None
                and result.x_values is not None
                and result.objective_t is not None
            ):
                evaluation = evaluate_fairness_solution(
                    instance,
                    y_values=result.y_values,
                    x_values=result.x_values,
                    t_value=result.objective_t,
                    baseline_cost=c_anchor,
                    rho=rho,
                    gamma=int(config["gamma_target"]),
                    max_scenarios=int(config["max_scenarios"]),
                    per_scenario_time_limit=float(
                        config["fairness_development"].get(
                            "post_evaluation_time_limit_per_scenario", 30.0
                        )
                    ),
                    tolerance=float(
                        config["fairness_development"]["feasibility_tolerance"]
                    ),
                    output_flag=bool(config.get("output_flag", False)),
                )
                payload["post_evaluation"] = evaluation.to_dict()
                payload["post_evaluation_runtime_excluded_from_algorithm_runtime"] = True
                evaluation_valid = (
                    evaluation.valid and evaluation.objective_t_consistent is not False
                )
            solved = algorithm_solved and evaluation_valid
            certification_status = (
                "certified_optimal"
                if solved
                else (
                    "invalid_post_evaluation"
                    if algorithm_solved
                    else f"uncertified_{result.status}"
                )
            )
            payload["penalized_runtime_par2"] = penalized_runtime_par2(
                solved_to_tolerance=solved,
                runtime=result.runtime,
                time_limit=float(config["fairness_time_limit"]),
            )
            _write_record(output_dir, key, {
                "run_key": key,
                "task_type": "fairness_frontier",
                "instance_size": size,
                "seed": seed,
                "method": "robust_regional_fairness",
                "state": "complete",
                "success": solved,
                "solved_to_tolerance": solved,
                "certification_status": certification_status,
                "git_commit": commit,
                "config_sha256": cfg_hash,
                "baseline_run_key": baseline_key,
                "baseline_anchor_sha256": anchor["anchor_sha256"],
                "certified_cost_anchor": anchor,
                "baseline_cost": c_anchor,
                "rho": rho,
                "created_at": utc_now_iso(),
                "result": payload,
            })
            write_run_state(
                output_dir,
                key,
                state="complete",
                details={
                    "success": solved,
                    "solved_to_tolerance": solved,
                    "certification_status": certification_status,
                },
            )
            _write_fairness_development_manifest(
                output_dir,
                config=config,
                config_hash=cfg_hash,
                commit=commit,
                run_keys=run_keys,
                anchors=anchors,
            )
            _write_fairness_result_tables(output_dir, run_keys)
        update_run_manifest(
            output_dir=output_dir,
            run_keys=run_keys,
            config_hash=cfg_hash,
            commit=commit,
            skipped_run_count=skipped,
        )
    return output_dir / "run_manifest.json"


def run_fairness_development(
    config: dict[str, Any], *, resume: bool = False, overwrite: bool = False
) -> Path:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    from .fairness_development_audit import audit_fairness_development

    audit = audit_fairness_development(
        config_overrides={str(config["experiment_name"]): config},
        allow_existing_output=True,
    )
    failed = [
        check["check"]
        for check in audit["checks"]
        if check.get("required", True) and not check["passed"]
    ]
    if failed:
        raise ValueError(f"Fairness development protocol audit failed: {', '.join(failed)}")
    output_dir = Path(str(config["output_dir"]))
    with SingleWriterLock(output_dir / ".fairness_development.lock", resume=resume):
        return _run_fairness_development_locked(
            config, resume=resume, overwrite=overwrite
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust regional service fairness development runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_development_config(args.config)
    if args.dry_run:
        from .fairness_development_audit import audit_fairness_development

        report = development_run_plan(config)
        audit = audit_fairness_development()
        report["protocol_audit_errors"] = [
            check["check"] for check in audit["checks"] if check.get("required", True) and not check["passed"]
        ]
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    manifest = run_fairness_development(config, resume=args.resume, overwrite=args.overwrite)
    print(manifest)


if __name__ == "__main__":
    main()
