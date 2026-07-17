from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .experiment_protocol import (
    ProtocolRunSpec,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    config_sha256,
    decide_run_action,
    git_commit,
    load_run_record,
    theoretical_maximum_hours,
    update_run_manifest,
    utc_now_iso,
    write_run_state,
)
from .experiment_suite import (
    RESULT_FIELDS,
    SUMMARY_FIELDS,
    _apply_selected_parameters,
    _apply_variant_config,
    _base_config,
    _failure_row,
    _result_row,
    _solve_experiment_method,
    _summary_rows,
    _variant_specs,
    _write_csv,
    _write_iteration_log,
)
from .instance import generate_instance
from .managerial_evaluation import (
    evaluate_managerial_solution,
    invalid_managerial_evaluation,
)


MANAGERIAL_RESULT_FIELDS = [
    "sensitivity_axis",
    "sensitivity_value",
    "baseline_value",
    *RESULT_FIELDS,
    "opened_warehouses",
    "total_inventory",
    "inventory_by_product",
    "inventory_by_warehouse",
    "fixed_opening_cost",
    "worst_case_recourse_cost",
    "transport_cost",
    "shortage_cost",
    "service_violation_cost",
    "total_worst_case_demand",
    "shortage_by_product",
    "service_violation_by_product",
    "realized_fill_rate",
    "worst_case_active_deviations",
    "worst_case_demand_values",
    "managerial_evaluation_status",
    "managerial_evaluation_runtime",
    "managerial_metrics_valid",
    "managerial_evaluation_error",
    "full_resolved_config",
]


def managerial_run_specs(config: dict[str, Any]) -> list[ProtocolRunSpec]:
    exp_name = str(config.get("experiment_name", "managerial_sensitivity_joint_v1"))
    seeds = [int(seed) for seed in config.get("random_seeds", [])]
    sizes = [str(size) for size in config.get("instance_sizes", [])]
    variants = _variant_specs(config)
    axes = config.get("sensitivity_axes", {})
    specs: list[ProtocolRunSpec] = []
    for axis, axis_config in axes.items():
        baseline = axis_config["baseline_value"]
        for value in axis_config["values"]:
            for seed in seeds:
                for size_name in sizes:
                    for variant_name, _method, _variant in variants:
                        specs.append(
                            ProtocolRunSpec(
                                experiment_name=exp_name,
                                sensitivity_axis=str(axis),
                                sensitivity_value=value,
                                baseline_value=baseline,
                                instance_size=size_name,
                                seed=seed,
                                variant_name=variant_name,
                            )
                        )
    return specs


def managerial_run_config(
    config: dict[str, Any], spec: ProtocolRunSpec
) -> dict[str, Any]:
    baseline = config["baseline"]
    base = deepcopy(config)
    base["gamma_target"] = int(baseline["gamma_target"])
    base["gamma_schedule"] = [int(baseline["gamma_target"])]
    base["budget_factor"] = float(baseline["budget_factor"])
    base["capacity_factor"] = float(baseline["capacity_factor"])
    service_level = float(baseline["service_level"])
    if spec.sensitivity_axis == "service_level":
        service_level = float(spec.sensitivity_value)
    run_config = _base_config(
        exp_cfg=base,
        size_name=spec.instance_size,
        seed=spec.seed,
        alpha=service_level,
    )
    if spec.sensitivity_axis == "gamma_target":
        gamma_target = int(spec.sensitivity_value)
        run_config["robust"]["gamma_target"] = gamma_target
        run_config["robust"]["gamma_schedule"] = [gamma_target]
    elif spec.sensitivity_axis == "budget_factor":
        run_config["instance"]["budget_factor"] = float(spec.sensitivity_value)
    elif spec.sensitivity_axis == "capacity_factor":
        run_config["instance"]["capacity_factor"] = float(spec.sensitivity_value)
    elif spec.sensitivity_axis != "service_level":
        raise ValueError(f"Unsupported sensitivity axis: {spec.sensitivity_axis}")
    run_config["gamma_continuation_enabled"] = False
    run_config["robust"]["gamma_schedule"] = [run_config["robust"]["gamma_target"]]
    return run_config


def managerial_dry_run_report(config: dict[str, Any]) -> dict[str, Any]:
    resolved = _apply_selected_parameters(config)
    specs = managerial_run_specs(resolved)
    counts: dict[str, int] = {}
    for spec in specs:
        axis = str(spec.sensitivity_axis)
        counts[axis] = counts.get(axis, 0) + 1
    audit_errors: list[str] = []
    try:
        from .extended_experiment_audit import audit_protocols

        audit = audit_protocols()
        audit_errors = [
            str(check["check"])
            for check in audit["checks"]
            if check.get("required", True) and not check.get("passed", False)
        ]
    except Exception as exc:  # noqa: BLE001 - dry-run reports audit failures.
        audit_errors = [f"audit_execution_failed: {exc}"]
    time_limit = float(resolved.get("time_limit", 0.0))
    return {
        "experiment_name": resolved.get("experiment_name"),
        "total_run_count": len(specs),
        "run_count_by_axis": counts,
        "seeds": sorted({spec.seed for spec in specs}),
        "instance_sizes": sorted({spec.instance_size for spec in specs}),
        "methods": [name for name, _method, _variant in _variant_specs(resolved)],
        "output_dir": resolved.get("output_dir"),
        "time_limit_seconds": time_limit,
        "managerial_evaluation_time_limit_seconds": float(
            resolved.get("managerial_evaluation_time_limit", 300.0)
        ),
        "theoretical_maximum_seconds": len(specs) * time_limit,
        "theoretical_maximum_hours": theoretical_maximum_hours(len(specs), time_limit),
        "serial_upper_bound_not_runtime_prediction": True,
        "automatic_parallelism_enabled": False,
        "protocol_audit_errors": audit_errors,
    }


def run_managerial_sensitivity_suite(
    config: dict[str, Any],
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    config = _apply_selected_parameters(config)
    exp_name = str(config["experiment_name"])
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = atomic_write_yaml(output_dir / "resolved_config.yaml", config)
    specs = managerial_run_specs(config)
    run_keys = [spec.run_key for spec in specs]
    config_hash = config_sha256(config)
    commit = git_commit(Path.cwd())
    skipped_run_count = 0
    results: list[dict[str, Any]] = []
    variants = _variant_specs(config)
    if len(variants) != 1:
        raise ValueError("Managerial sensitivity requires exactly one frozen variant.")
    variant_name, method, variant = variants[0]
    manifest_path = update_run_manifest(
        output_dir=output_dir,
        run_keys=run_keys,
        config_hash=config_hash,
        commit=commit,
        skipped_run_count=0,
    )

    for spec in specs:
        existing = load_run_record(output_dir, spec.run_key)
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

        run_key = spec.run_key
        run_dir = output_dir / "runs" / run_key
        run_config = managerial_run_config(config, spec)
        _solver_method, _flags, method_resolved_config = _apply_variant_config(
            deepcopy(run_config), method, variant
        )
        full_resolved_config = {
            "protocol": config,
            "run": {
                "run_key": run_key,
                "sensitivity_axis": spec.sensitivity_axis,
                "sensitivity_value": spec.sensitivity_value,
                "baseline_value": spec.baseline_value,
                "seed": spec.seed,
                "instance_size": spec.instance_size,
                "variant_name": spec.variant_name,
            },
            "solver_config": method_resolved_config,
        }
        run_config_hash = config_sha256(full_resolved_config)
        atomic_write_yaml(run_dir / "resolved_config.yaml", full_resolved_config)
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
        instance = generate_instance(run_config, seed=spec.seed)
        instance_path = atomic_write_json(run_dir / "instance.json", instance.to_dict())
        try:
            result, flags = _solve_experiment_method(
                deepcopy(run_config), instance, method, variant
            )
            row = _result_row(
                exp_name,
                spec.instance_size,
                spec.seed,
                method,
                variant_name,
                result,
                flags,
                instance,
                instance_path,
                time_limit=float(run_config["benders"]["time_limit"]),
                solve_tolerance=float(run_config["benders"]["tol"]),
            )
            if bool(config.get("save_iteration_log", False)):
                row["iteration_log_path"] = str(
                    _write_iteration_log(
                        output_dir,
                        exp_name,
                        instance.name,
                        spec.seed,
                        method,
                        variant_name,
                        result.iteration_log,
                        run_key=run_key,
                    )
                )
            managerial = evaluate_managerial_solution(
                instance,
                best_y_values=result.metadata.get("best_y_values"),
                best_x_values=result.metadata.get("best_x_values"),
                gamma_target=int(run_config["robust"]["gamma_target"]),
                time_limit=float(config.get("managerial_evaluation_time_limit", 300.0)),
                output_flag=bool(run_config["benders"].get("output_flag", False)),
            )
            atomic_write_text(run_dir / "error.txt", "")
            algorithm_success = row["status"] not in {"failed", "skipped"}
        except Exception as exc:  # noqa: BLE001 - one failed run must not stop the suite.
            flags = {
                "adaptive_gap_enabled": False,
                "gamma_continuation_enabled": False,
                "cut_selection_enabled": False,
            }
            row = _failure_row(
                exp_name,
                spec.instance_size,
                spec.seed,
                method,
                variant_name,
                flags,
                instance,
                instance_path,
                exc,
                time_limit=float(run_config["benders"]["time_limit"]),
                solve_tolerance=float(run_config["benders"]["tol"]),
            )
            managerial = invalid_managerial_evaluation(
                "algorithm_failed",
                "Managerial evaluation was not run because the algorithm run failed.",
                0.0,
            )
            algorithm_success = False
            atomic_write_text(run_dir / "error.txt", f"{type(exc).__name__}: {exc}\n")
            if bool(config.get("save_iteration_log", False)):
                row["iteration_log_path"] = str(
                    _write_iteration_log(
                        output_dir,
                        exp_name,
                        instance.name,
                        spec.seed,
                        method,
                        variant_name,
                        [],
                        run_key=run_key,
                    )
                )

        managerial_dict = managerial.to_dict()
        row.update(managerial_dict)
        row.update(
            {
                "sensitivity_axis": spec.sensitivity_axis,
                "sensitivity_value": spec.sensitivity_value,
                "baseline_value": spec.baseline_value,
                "gamma_target": run_config["robust"]["gamma_target"],
                "gamma_schedule": run_config["robust"]["gamma_schedule"],
                "total_shortage": managerial.total_shortage,
                "service_violation": managerial.service_violation,
                "inventory_cost": managerial.inventory_cost,
                "run_key": run_key,
                "config_sha256": run_config_hash,
                "git_commit": commit,
                "full_resolved_config": full_resolved_config,
            }
        )
        atomic_write_json(run_dir / "managerial_evaluation.json", managerial_dict)
        results.append(row)
        success = algorithm_success and managerial.managerial_metrics_valid
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
                "managerial_metrics_valid": managerial.managerial_metrics_valid,
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
    _write_csv(results_path, results, MANAGERIAL_RESULT_FIELDS)
    _write_csv(summary_path, _summary_rows(results), SUMMARY_FIELDS)
    return {
        "results": results_path,
        "summary": summary_path,
        "resolved_config": resolved_config_path,
        "run_manifest": manifest_path,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen one-factor managerial sensitivity protocol."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(managerial_dry_run_report(config), ensure_ascii=False, indent=2))
        return
    outputs = run_managerial_sensitivity_suite(
        config,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {key: str(value) for key, value in outputs.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
