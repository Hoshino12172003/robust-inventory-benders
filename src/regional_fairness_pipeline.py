from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
from typing import Any, Callable, Iterable, Mapping

from .config import load_config
from .experiment_protocol import (
    atomic_write_csv,
    atomic_write_json,
    config_sha256,
    file_sha256,
    git_commit,
    read_json,
    utc_now_iso,
)
from .experiment_suite import (
    _apply_selected_parameters,
    experiment_run_specs,
    run_experiment_suite,
)
from .instance import InventoryInstance, load_instance
from .regional_fairness_diagnostic import (
    DEGENERACY_REDUCTION_THRESHOLD,
    MATERIAL_GAP_THRESHOLD,
    NO_MATERIAL_MEDIAN_THRESHOLD,
    REGION_SCENARIO_FIELDS,
    INSTANCE_SUMMARY_FIELDS,
    STRUCTURAL_MEDIAN_THRESHOLD,
    deviation_pattern_payload,
    deviation_pattern_sha256,
    first_stage_x_sha256,
    solve_default_and_fair_best_recourse,
    stable_scenario_key,
    summarize_regional_service,
)
from .scenarios import DemandScenario, enumerate_budget_scenarios_with_metadata


DIAGNOSTIC_MANIFEST_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_INDEX_SCHEMA_VERSION = 1
EXPECTED_METHOD = "joint_v1_core_point_strengthened"


@dataclass(frozen=True)
class PipelineDependencies:
    base_runner: Callable[..., Mapping[str, Path]] = run_experiment_suite
    scenario_enumerator: Callable[..., Any] = enumerate_budget_scenarios_with_metadata
    scenario_solver: Callable[..., Any] = solve_default_and_fair_best_recourse
    fault_injector: Callable[[str, Mapping[str, Any]], None] | None = None


class DiagnosticIdentityError(RuntimeError):
    pass


class DiagnosticCheckpointError(RuntimeError):
    pass


class DiagnosticLockError(RuntimeError):
    pass


def _inject(dependencies: PipelineDependencies, event: str, **context: Any) -> None:
    if dependencies.fault_injector is not None:
        dependencies.fault_injector(event, context)


def _json_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def diagnostic_run_key(identity: Mapping[str, Any]) -> str:
    return f"regional_fairness_{_canonical_sha256(dict(identity))[:24]}"


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class SingleWriterLock:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.resume = resume
        self.acquired = False

    def __enter__(self) -> "SingleWriterLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_at": utc_now_iso(),
        }
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                existing = read_json(self.path)
                existing_pid = int((existing or {}).get("pid", -1))
                if not self.resume or _process_alive(existing_pid):
                    raise DiagnosticLockError(
                        f"Diagnostic output is locked by pid {existing_pid}."
                    )
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
            self.acquired = True
            return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.acquired = False


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticCheckpointError(f"Invalid JSON checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DiagnosticCheckpointError(f"Checkpoint {path} must contain an object.")
    return value


def _base_file_identity(output_dir: Path) -> dict[str, str]:
    identity: dict[str, str] = {}
    for filename in ("results.csv", "summary.csv", "run_manifest.json"):
        path = output_dir / filename
        if not path.is_file():
            raise DiagnosticIdentityError(f"Missing required base output: {path}")
        identity[filename] = file_sha256(path).lower()
    return identity


def _resolve_instance_path(repo_root: Path, output_dir: Path, value: Any) -> Path:
    candidate = Path(str(value))
    options = [candidate, repo_root / candidate, output_dir / candidate]
    for option in options:
        if option.is_file():
            return option.resolve()
    raise DiagnosticIdentityError(f"Base instance file is missing: {value}")


def validate_complete_base_results(
    *,
    repo_root: Path,
    output_dir: Path,
    resolved_config: dict[str, Any],
    current_commit: str,
) -> dict[str, Any]:
    specs = experiment_run_specs(resolved_config)
    if len(specs) != 10:
        raise DiagnosticIdentityError("The frozen diagnostic requires exactly 10 base runs.")
    expected_keys = [spec.run_key for spec in specs]
    manifest = read_json(output_dir / "run_manifest.json")
    if manifest is None:
        raise DiagnosticIdentityError("Base run_manifest.json is missing or invalid.")
    expected_config_hash = config_sha256(resolved_config)
    if manifest.get("config_sha256") != expected_config_hash:
        raise DiagnosticIdentityError("Base manifest resolved-config identity does not match.")
    if (
        int(manifest.get("expected_run_count", -1)) != 10
        or int(manifest.get("completed_run_count", -1)) != 10
        or int(manifest.get("solved_run_count", -1)) != 10
        or int(manifest.get("failed_run_count", -1)) != 0
        or int(manifest.get("remaining_run_count", -1)) != 0
    ):
        raise DiagnosticIdentityError("Base experiment is not complete and solved to tolerance.")

    run_records: list[dict[str, Any]] = []
    for spec in specs:
        record_path = output_dir / "runs" / spec.run_key / "run.json"
        record = _strict_json(record_path)
        result = record.get("result")
        if (
            record.get("run_key") != spec.run_key
            or record.get("state") != "complete"
            or record.get("success") is not True
            or record.get("solved_to_tolerance") is not True
            or not isinstance(result, dict)
        ):
            raise DiagnosticIdentityError(f"Base run is incomplete or invalid: {spec.run_key}")
        if (
            result.get("run_key") != spec.run_key
            or result.get("variant_name") != EXPECTED_METHOD
            or int(result.get("seed", -1)) != spec.seed
            or result.get("instance_size") != spec.instance_size
            or result.get("git_commit") != current_commit
            or record.get("git_commit") != current_commit
        ):
            raise DiagnosticIdentityError(f"Base run identity drift: {spec.run_key}")
        best_x = result.get("best_x_values")
        if not isinstance(best_x, list) or not best_x:
            raise DiagnosticIdentityError(f"Base run lacks best_x_values: {spec.run_key}")
        instance_path = _resolve_instance_path(
            repo_root, output_dir, result.get("instance_path")
        )
        run_records.append(
            {
                "base_run_key": spec.run_key,
                "seed": spec.seed,
                "scale": spec.instance_size,
                "method": EXPECTED_METHOD,
                "base_git_commit": current_commit,
                "base_config_sha256": str(result.get("config_sha256", "")),
                "instance_path": str(instance_path),
                "instance_sha256": file_sha256(instance_path).lower(),
                "instance_name": str(result.get("instance_name", instance_path.stem)),
                "best_x_values": best_x,
                "best_x_sha256": first_stage_x_sha256(best_x),
            }
        )

    with (output_dir / "results.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    csv_keys = [str(row.get("run_key", "")) for row in rows]
    if len(csv_keys) != 10 or len(set(csv_keys)) != 10 or set(csv_keys) != set(expected_keys):
        raise DiagnosticIdentityError("Base results.csv run keys are incomplete or duplicated.")
    return {
        "resolved_config_sha256": expected_config_hash,
        "base_git_commit": current_commit,
        "base_run_keys": expected_keys,
        "base_files": _base_file_identity(output_dir),
        "run_records": sorted(run_records, key=lambda item: (item["seed"], item["base_run_key"])),
    }


def validate_partial_base_identity(
    *,
    output_dir: Path,
    resolved_config: dict[str, Any],
    current_commit: str,
) -> None:
    """Reject foreign partial results before delegating recovery to experiment_suite."""
    expected_keys = {spec.run_key for spec in experiment_run_specs(resolved_config)}
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = _strict_json(manifest_path)
        if manifest.get("config_sha256") != config_sha256(resolved_config):
            raise DiagnosticIdentityError("Partial base manifest config identity mismatch.")
        if manifest.get("git_commit") not in (None, current_commit):
            raise DiagnosticIdentityError("Partial base manifest commit identity mismatch.")
    runs_dir = output_dir / "runs"
    if runs_dir.exists():
        for run_file in runs_dir.glob("*/run.json"):
            record = _strict_json(run_file)
            run_key = str(record.get("run_key", ""))
            if run_key not in expected_keys:
                raise DiagnosticIdentityError(f"Unexpected partial base run key: {run_key}")
            if record.get("git_commit") not in (None, current_commit):
                raise DiagnosticIdentityError(f"Partial base run commit mismatch: {run_key}")
    results_path = output_dir / "results.csv"
    if results_path.exists():
        with results_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        keys = [str(row.get("run_key", "")) for row in rows]
        if len(keys) != len(set(keys)) or not set(keys).issubset(expected_keys):
            raise DiagnosticIdentityError("Partial base results contain duplicate or foreign run keys.")


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "diagnostic_run_manifest.json"


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now_iso()
    atomic_write_json(_manifest_path(output_dir), manifest)


def _validate_resume_identity(manifest: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != DIAGNOSTIC_MANIFEST_SCHEMA_VERSION:
        raise DiagnosticIdentityError("Diagnostic manifest schema does not match.")
    if manifest.get("identity") != dict(identity):
        raise DiagnosticIdentityError("Diagnostic identity mismatch; resume is refused.")


def _checkpoint_path(output_dir: Path, base_run_key: str, chunk_index: int) -> Path:
    # The full stable base key remains inside every checkpoint, while its hash keeps
    # Windows paths safely below legacy MAX_PATH limits.
    base_directory = f"base_{hashlib.sha256(base_run_key.encode('utf-8')).hexdigest()[:20]}"
    return output_dir / "checkpoint" / base_directory / f"chunk_{chunk_index:05d}.json"


def _index_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint" / "index.json"


def _scenario_record(
    *,
    instance: InventoryInstance,
    scenario: DemandScenario,
    scenario_index: int,
    best_x_values: list[list[float]],
    diagnostic_config: Mapping[str, Any],
    dependencies: PipelineDependencies,
) -> dict[str, Any]:
    pattern = deviation_pattern_payload(instance, scenario)
    pattern_hash = deviation_pattern_sha256(pattern)
    scenario_key = stable_scenario_key(instance, scenario, scenario_index)
    absolute_tolerance = float(diagnostic_config["cost_absolute_tolerance"])
    relative_tolerance = float(diagnostic_config["cost_relative_tolerance"])
    metric_tolerance = float(diagnostic_config["metric_tolerance"])
    try:
        default, fair = dependencies.scenario_solver(
            instance,
            scenario,
            best_x_values,
            cost_absolute_tolerance=absolute_tolerance,
            cost_relative_tolerance=relative_tolerance,
            metric_tolerance=metric_tolerance,
            time_limit=float(diagnostic_config["recourse_time_limit"]),
            output_flag=False,
        )
        default_metrics = summarize_regional_service(
            [list(row) for row in scenario.demand],
            default.shortage_values,
            transport_cost_by_region=default.transport_cost_by_region,
            allocated_units_by_region=default.allocated_units_by_region,
            reachable_warehouse_count=instance.num_warehouses,
            metric_tolerance=metric_tolerance,
        )
        fair_metrics = summarize_regional_service(
            [list(row) for row in scenario.demand],
            fair.shortage_values,
            transport_cost_by_region=fair.transport_cost_by_region,
            allocated_units_by_region=fair.allocated_units_by_region,
            reachable_warehouse_count=instance.num_warehouses,
            metric_tolerance=metric_tolerance,
        )
        if default.status != "optimal" or fair.status != "optimal":
            raise RuntimeError("Both recourse variants must be optimal.")
        if default.first_stage_x_sha256 != fair.first_stage_x_sha256:
            raise RuntimeError("Recourse variants did not use the same first-stage solution.")
        if not default.constraints_satisfied or not fair.constraints_satisfied:
            raise RuntimeError("Recourse feasibility audit failed.")
        if not fair.cost_cap_satisfied:
            raise RuntimeError("Fair-best recourse exceeded the frozen cost tolerance.")
        if float(fair_metrics["fill_rate_gap"] or 0.0) > float(
            default_metrics["fill_rate_gap"] or 0.0
        ) + metric_tolerance:
            raise RuntimeError("Fair-best recourse worsened the default fill-rate gap.")
        return {
            "scenario_key": scenario_key,
            "scenario_index": scenario_index,
            "scenario_id": scenario.name,
            "gamma_usage": scenario.gamma,
            "deviation_pattern": pattern,
            "deviation_pattern_sha256": pattern_hash,
            "demand": [list(row) for row in scenario.demand],
            "valid": True,
            "invalid_reason": "",
            "default": {
                "status": default.status,
                "objective": default.objective,
                "original_optimal_cost": default.original_optimal_cost,
                "cost_tolerance": default.cost_tolerance,
                "shortage_values": default.shortage_values,
                "metrics": default_metrics,
            },
            "fair_best": {
                "status": fair.status,
                "objective": fair.objective,
                "original_optimal_cost": fair.original_optimal_cost,
                "cost_tolerance": fair.cost_tolerance,
                "shortage_values": fair.shortage_values,
                "metrics": fair_metrics,
            },
        }
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 - failures must be persisted, never hidden.
        return {
            "scenario_key": scenario_key,
            "scenario_index": scenario_index,
            "scenario_id": scenario.name,
            "gamma_usage": scenario.gamma,
            "deviation_pattern": pattern,
            "deviation_pattern_sha256": pattern_hash,
            "demand": [list(row) for row in scenario.demand],
            "valid": False,
            "invalid_reason": f"{type(exc).__name__}: {exc}",
            "default": {"status": "invalid", "objective": None, "metrics": None},
            "fair_best": {"status": "invalid", "objective": None, "metrics": None},
        }


def _validate_checkpoint(
    *,
    checkpoint: Mapping[str, Any],
    diagnostic_key: str,
    base_run_key: str,
    chunk_index: int,
    scenarios: list[DemandScenario],
    instance: InventoryInstance,
    start: int,
    end: int,
) -> None:
    expected_keys = [
        stable_scenario_key(instance, scenarios[index], index) for index in range(start, end)
    ]
    records = checkpoint.get("scenario_records")
    actual_keys = (
        [record.get("scenario_key") for record in records]
        if isinstance(records, list)
        else []
    )
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("diagnostic_run_key") != diagnostic_key
        or checkpoint.get("base_run_key") != base_run_key
        or checkpoint.get("chunk_index") != chunk_index
        or checkpoint.get("scenario_start") != start
        or checkpoint.get("scenario_end_exclusive") != end
        or actual_keys != expected_keys
        or len(actual_keys) != len(set(actual_keys))
    ):
        raise DiagnosticCheckpointError(
            f"Checkpoint identity or scenario order is invalid for {base_run_key} chunk {chunk_index}."
        )


def _write_checkpoint_index(
    *,
    output_dir: Path,
    diagnostic_key: str,
    entries: list[dict[str, Any]],
) -> None:
    atomic_write_json(
        _index_path(output_dir),
        {
            "schema_version": CHECKPOINT_INDEX_SCHEMA_VERSION,
            "diagnostic_run_key": diagnostic_key,
            "entries": sorted(entries, key=lambda item: (item["base_run_key"], item["chunk_index"])),
            "updated_at": utc_now_iso(),
        },
    )


def _load_index_entries(output_dir: Path, diagnostic_key: str) -> dict[tuple[str, int], dict[str, Any]]:
    path = _index_path(output_dir)
    if not path.exists():
        return {}
    index = _strict_json(path)
    if (
        index.get("schema_version") != CHECKPOINT_INDEX_SCHEMA_VERSION
        or index.get("diagnostic_run_key") != diagnostic_key
        or not isinstance(index.get("entries"), list)
    ):
        raise DiagnosticCheckpointError("Checkpoint index identity is invalid.")
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in index["entries"]:
        key = (str(entry.get("base_run_key")), int(entry.get("chunk_index", -1)))
        if key in entries:
            raise DiagnosticCheckpointError("Checkpoint index contains duplicate entries.")
        entries[key] = dict(entry)
    return entries


def _checkpoint_plan(
    run_records: list[dict[str, Any]],
    scenarios_by_run: Mapping[str, list[DemandScenario]],
    chunk_size: int,
) -> list[tuple[dict[str, Any], int, int, int]]:
    plan: list[tuple[dict[str, Any], int, int, int]] = []
    for run in run_records:
        scenarios = scenarios_by_run[run["base_run_key"]]
        for chunk_index, start in enumerate(range(0, len(scenarios), chunk_size)):
            plan.append((run, chunk_index, start, min(len(scenarios), start + chunk_size)))
    return plan


def _manifest_counts(
    plan: list[tuple[dict[str, Any], int, int, int]],
    checkpoints: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    interrupted_count: int = 0,
) -> dict[str, int]:
    completed = sum(bool(value.get("success")) for value in checkpoints.values())
    failed = sum(not bool(value.get("success")) for value in checkpoints.values())
    pending = max(0, len(plan) - len(checkpoints))
    return {
        "expected_chunk_count": len(plan),
        "completed_chunk_count": completed,
        "failed_chunk_count": failed,
        "pending_chunk_count": pending,
        "interrupted_chunk_count": interrupted_count,
        "completed_count": completed,
        "pending_count": pending,
        "failed_count": failed,
        "interrupted_count": interrupted_count,
        "completed_scenario_count": sum(
            len(value.get("scenario_records", [])) for value in checkpoints.values()
        ),
    }


def _aggregate_instance(
    *,
    diagnostic_key: str,
    experiment_name: str,
    resolved_config_sha256: str,
    run: Mapping[str, Any],
    scenario_records: list[dict[str, Any]],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenario_records = sorted(scenario_records, key=lambda item: int(item["scenario_index"]))
    valid_records = [record for record in scenario_records if record.get("valid") is True]
    all_valid = len(valid_records) == len(scenario_records)
    cost_worst = (
        max(valid_records, key=lambda item: float(item["default"]["objective"]))["scenario_key"]
        if valid_records
        else None
    )
    fairness_worst = (
        max(
            valid_records,
            key=lambda item: float(item["fair_best"]["metrics"]["fill_rate_gap"] or 0.0),
        )["scenario_key"]
        if valid_records
        else None
    )
    rows: list[dict[str, Any]] = []
    for record in scenario_records:
        is_nominal = int(record["gamma_usage"]) == 0
        is_cost_worst = record["scenario_key"] == cost_worst
        is_fairness_worst = record["scenario_key"] == fairness_worst
        scenario_type = "nominal" if is_nominal else "budget_extreme"
        if is_cost_worst:
            scenario_type += "|cost_worst"
        if is_fairness_worst:
            scenario_type += "|fairness_worst"
        for variant in ("default", "fair_best"):
            allocation = record[variant]
            metrics = allocation.get("metrics")
            regions = (
                metrics.get("regions", [])
                if isinstance(metrics, dict)
                else [
                    {
                        "region": region,
                        "regional_demand": sum(record["demand"][region]),
                        "regional_shortage": None,
                        "fill_rate": None,
                        "fill_rate_applicable": sum(record["demand"][region]) > 0,
                        "not_applicable_reason": "diagnostic_invalid",
                    }
                    for region in range(len(record["demand"]))
                ]
            )
            for region in regions:
                row = {
                    "diagnostic_run_key": diagnostic_key,
                    "base_run_key": run["base_run_key"],
                    "instance_name": run["instance_name"],
                    "experiment_name": experiment_name,
                    "scale": run["scale"],
                    "method": run["method"],
                    "seed": run["seed"],
                    "base_git_commit": run["base_git_commit"],
                    "base_config_sha256": run["base_config_sha256"],
                    "resolved_config_sha256": resolved_config_sha256,
                    "scenario_key": record["scenario_key"],
                    "scenario_index": record["scenario_index"],
                    "scenario_type": scenario_type,
                    "is_nominal": is_nominal,
                    "is_cost_worst": is_cost_worst,
                    "is_fairness_worst": is_fairness_worst,
                    "deviation_pattern": record["deviation_pattern"],
                    "deviation_pattern_sha256": record["deviation_pattern_sha256"],
                    "region_id": region.get("region"),
                    "recourse_variant": variant,
                    "default_recourse_status": record["default"].get("status"),
                    "fair_best_recourse_status": record["fair_best"].get("status"),
                    "default_recourse_cost": record["default"].get("objective"),
                    "fair_best_recourse_cost": record["fair_best"].get("objective"),
                    "cost_absolute_tolerance": absolute_tolerance,
                    "cost_relative_tolerance": relative_tolerance,
                    "invalid_reason": record.get("invalid_reason", ""),
                    "instance_size": run["scale"],
                    "scenario_id": record["scenario_id"],
                    "scenario_kind": scenario_type,
                    "region": region.get("region"),
                    **region,
                    "recourse_policy": variant,
                    "original_recourse_cost": record["default"].get("objective"),
                    "evaluated_recourse_cost": allocation.get("objective"),
                    "cost_tolerance": allocation.get("cost_tolerance"),
                    "scenario_gamma_usage": record["gamma_usage"],
                    "first_stage_x_sha256": run["best_x_sha256"],
                }
                rows.append({field: row.get(field) for field in REGION_SCENARIO_FIELDS})

    def robust(variant: str, metric: str, function: Callable[[Iterable[float]], float]) -> float | None:
        values = [
            float(record[variant]["metrics"][metric])
            for record in valid_records
            if record[variant]["metrics"].get(metric) is not None
        ]
        return None if not values else float(function(values))

    default_gap = robust("default", "fill_rate_gap", max)
    fair_gap = robust("fair_best", "fill_rate_gap", max)
    nominal = next((item for item in valid_records if int(item["gamma_usage"]) == 0), None)
    cost_record = next((item for item in valid_records if item["scenario_key"] == cost_worst), None)
    summary = {
        "diagnostic_run_key": diagnostic_key,
        "base_run_key": run["base_run_key"],
        "instance_name": run["instance_name"],
        "experiment_name": experiment_name,
        "method": run["method"],
        "base_git_commit": run["base_git_commit"],
        "base_config_sha256": run["base_config_sha256"],
        "resolved_config_sha256": resolved_config_sha256,
        "seed": run["seed"],
        "size": run["scale"],
        "default_WGap": default_gap,
        "fair_best_WGap": fair_gap,
        "default_WMinFR": robust("default", "minimum_fill_rate", min),
        "fair_best_WMinFR": robust("fair_best", "minimum_fill_rate", min),
        "default_WWD": robust("default", "worst_region_deviation", max),
        "fair_best_WWD": robust("fair_best", "worst_region_deviation", max),
        "nominal_gap": None if nominal is None else nominal["fair_best"]["metrics"]["fill_rate_gap"],
        "cost_worst_gap": None if cost_record is None else cost_record["fair_best"]["metrics"]["fill_rate_gap"],
        "cost_worst_scenario": cost_worst,
        "fairness_worst_scenario": fairness_worst,
        "default_minus_fair_best_WGap": (
            None if default_gap is None or fair_gap is None else default_gap - fair_gap
        ),
        "diagnosis_category": (
            "invalid"
            if not all_valid
            else (
                "structural_fairness_gap"
                if float(fair_gap or 0.0) >= MATERIAL_GAP_THRESHOLD
                else "recourse_degeneracy_only"
                if float(default_gap or 0.0) >= MATERIAL_GAP_THRESHOLD
                and float(default_gap or 0.0) - float(fair_gap or 0.0)
                >= DEGENERACY_REDUCTION_THRESHOLD
                else "no_material_fairness_gap"
                if float(fair_gap or 0.0) < NO_MATERIAL_MEDIAN_THRESHOLD
                and float(default_gap or 0.0) < MATERIAL_GAP_THRESHOLD
                else "fairness_diagnostic_inconclusive"
            )
        ),
        "scenario_count": len(scenario_records),
        "first_stage_x_sha256": run["best_x_sha256"],
    }
    return rows, {field: summary.get(field) for field in INSTANCE_SUMMARY_FIELDS}


def _scale_diagnosis(summaries: list[dict[str, Any]], *, all_valid: bool) -> dict[str, Any]:
    fair = [float(row["fair_best_WGap"]) for row in summaries if row.get("fair_best_WGap") is not None]
    default = [float(row["default_WGap"]) for row in summaries if row.get("default_WGap") is not None]
    if not all_valid or len(fair) != 10 or len(default) != 10:
        return {
            "decision": "fairness_diagnostic_invalid",
            "diagnosis_valid": False,
            "reason": "One or more recourse evaluations or diagnostic outputs are invalid.",
            "next_authorized_stage": "none",
        }
    fair_structural = (
        sum(value >= MATERIAL_GAP_THRESHOLD for value in fair) >= 4
        and statistics.median(fair) >= STRUCTURAL_MEDIAN_THRESHOLD
    )
    default_structural = (
        sum(value >= MATERIAL_GAP_THRESHOLD for value in default) >= 4
        and statistics.median(default) >= STRUCTURAL_MEDIAN_THRESHOLD
    )
    reductions = sum(
        float(row["default_WGap"]) - float(row["fair_best_WGap"])
        >= DEGENERACY_REDUCTION_THRESHOLD
        for row in summaries
    )
    no_material = (
        sum(value >= MATERIAL_GAP_THRESHOLD for value in fair) <= 1
        and statistics.median(fair) < NO_MATERIAL_MEDIAN_THRESHOLD
        and not default_structural
    )
    scale_signal = (
        "structural_fairness_gap"
        if fair_structural
        else "recourse_degeneracy_only"
        if default_structural and reductions >= 4
        else "no_material_fairness_gap"
        if no_material
        else "fairness_diagnostic_inconclusive"
    )
    return {
        "decision": "fairness_diagnostic_inconclusive",
        "diagnosis_valid": True,
        "decision_scope": "single_scale",
        "scale_signal": scale_signal,
        "reason": "The frozen overall decision requires both scale outputs and a separate read-only decision audit.",
        "next_authorized_stage": "combined_read_only_fairness_diagnostic_decision_only",
        "fair_best_count_at_least_0_10": sum(value >= MATERIAL_GAP_THRESHOLD for value in fair),
        "fair_best_median_WGap": statistics.median(fair),
        "default_count_at_least_0_10": sum(value >= MATERIAL_GAP_THRESHOLD for value in default),
        "default_median_WGap": statistics.median(default),
        "default_to_fair_best_reduction_count_at_least_0_05": reductions,
    }


def _aggregate_outputs(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    run_records: list[dict[str, Any]],
    checkpoints: Mapping[tuple[str, int], Mapping[str, Any]],
    diagnostic_config: Mapping[str, Any],
    dependencies: PipelineDependencies,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    region_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run in run_records:
        records: list[dict[str, Any]] = []
        for (base_key, _chunk_index), checkpoint in sorted(checkpoints.items()):
            if base_key == run["base_run_key"]:
                records.extend(checkpoint["scenario_records"])
        rows, summary = _aggregate_instance(
            diagnostic_key=manifest["diagnostic_run_key"],
            experiment_name=manifest["experiment_name"],
            resolved_config_sha256=manifest["resolved_config_sha256"],
            run=run,
            scenario_records=records,
            absolute_tolerance=float(diagnostic_config["cost_absolute_tolerance"]),
            relative_tolerance=float(diagnostic_config["cost_relative_tolerance"]),
        )
        region_rows.extend(rows)
        summaries.append(summary)
    region_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            int(row["scenario_index"]),
            str(row["recourse_variant"]),
            int(row["region_id"]),
        )
    )
    summaries.sort(key=lambda row: (str(row["size"]), int(row["seed"])))
    primary_keys = [
        (
            row["diagnostic_run_key"],
            row["base_run_key"],
            row["scenario_key"],
            row["recourse_variant"],
            row["region_id"],
        )
        for row in region_rows
    ]
    if len(primary_keys) != len(set(primary_keys)):
        raise DiagnosticCheckpointError("Final regional output contains duplicate primary keys.")
    if len(summaries) != 10 or len({row["base_run_key"] for row in summaries}) != 10:
        raise DiagnosticCheckpointError("Final instance summary is incomplete or duplicated.")
    expected_scenarios = int(manifest["scenario_count_per_instance"])
    expected_region_rows = 0
    for run in run_records:
        run_checkpoints = [
            checkpoint
            for (base_key, _chunk_index), checkpoint in checkpoints.items()
            if base_key == run["base_run_key"]
        ]
        run_scenarios = sum(len(item["scenario_records"]) for item in run_checkpoints)
        if run_scenarios != expected_scenarios:
            raise DiagnosticCheckpointError(
                f"Base run {run['base_run_key']} has {run_scenarios} scenarios; "
                f"expected {expected_scenarios}."
            )
        for checkpoint in run_checkpoints:
            for record in checkpoint["scenario_records"]:
                expected_region_rows += 2 * len(record["demand"])
    if len(region_rows) != expected_region_rows:
        raise DiagnosticCheckpointError(
            f"Final regional output has {len(region_rows)} rows; "
            f"expected {expected_region_rows}."
        )
    diagnosis = _scale_diagnosis(
        summaries,
        all_valid=all(checkpoint.get("success") for checkpoint in checkpoints.values()),
    )
    diagnosis.update(
        {
            "diagnostic_run_key": manifest["diagnostic_run_key"],
            "experiment_name": manifest["experiment_name"],
            "scale": manifest["instance_size"],
            "instance_count": len(summaries),
            "scenario_count_per_instance": manifest["scenario_count_per_instance"],
        }
    )
    _inject(dependencies, "before_final_aggregation_commit", row_count=len(region_rows))
    atomic_write_csv(
        output_dir / "region_scenario_metrics.csv",
        region_rows,
        REGION_SCENARIO_FIELDS,
        value_encoder=_json_value,
    )
    _inject(dependencies, "after_region_csv_before_diagnosis", row_count=len(region_rows))
    atomic_write_csv(
        output_dir / "instance_summary.csv",
        summaries,
        INSTANCE_SUMMARY_FIELDS,
        value_encoder=_json_value,
    )
    atomic_write_json(output_dir / "diagnosis.json", diagnosis)
    atomic_write_json(
        output_dir / "audit_log.json",
        {
            "diagnostic_run_key": manifest["diagnostic_run_key"],
            "primary_keys_unique": True,
            "instance_keys_unique": True,
            "all_checkpoints_valid": all(
                checkpoint.get("success") for checkpoint in checkpoints.values()
            ),
            "diagnostic_updates_base_results": False,
            "region_row_count": len(region_rows),
            "instance_count": len(summaries),
        },
    )
    return region_rows, summaries, diagnosis


def run_regional_fairness_pipeline(
    config_path: str | Path,
    *,
    resume: bool,
    overwrite: bool = False,
    dependencies: PipelineDependencies | None = None,
    strict_protocol_audit: bool = True,
) -> dict[str, Any]:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    dependencies = dependencies or PipelineDependencies()
    config_file = Path(config_path).resolve()
    repo_root = config_file.parents[2]
    raw_config = load_config(config_file)
    resolved_config = _apply_selected_parameters(raw_config)
    output_dir = repo_root / str(raw_config["output_dir"])
    current_commit = git_commit(repo_root)
    protocol_path = repo_root / "docs/regional_fairness_diagnostic_protocol.md"
    candidate_path = repo_root / "experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml"
    static_audit: dict[str, Any] | None = None
    if strict_protocol_audit:
        from .regional_fairness_diagnostic_audit import audit_regional_fairness_diagnostic

        static_audit = audit_regional_fairness_diagnostic(
            repo_root, require_absent_outputs=False
        )
        if not static_audit["all_required_checks_passed"]:
            raise DiagnosticIdentityError(
                f"Static protocol audit failed: {static_audit['failed_checks']}"
            )
    scale = str(raw_config["instance_sizes"][0])
    identity = {
        "protocol_phase": raw_config.get("protocol_phase"),
        "experiment_name": raw_config.get("experiment_name"),
        "instance_size": scale,
        "output_dir": str(raw_config["output_dir"]),
        "config_path": str(config_file),
        "config_sha256": file_sha256(config_file).lower(),
        "resolved_config_sha256": config_sha256(resolved_config),
        "protocol_document_sha256": file_sha256(protocol_path).lower(),
        "candidate_config_sha256": file_sha256(candidate_path).lower(),
        "diagnostic_code_git_commit": current_commit,
        "seeds": list(raw_config["random_seeds"]),
        "method": EXPECTED_METHOD,
        "checkpoint_scenario_chunk_size": int(
            raw_config["fairness_diagnostic"]["checkpoint_scenario_chunk_size"]
        ),
    }
    diagnostic_key = diagnostic_run_key(identity)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".regional_fairness_diagnostic.lock"
    with SingleWriterLock(lock_path, resume=resume):
        manifest_path = _manifest_path(output_dir)
        existing_manifest = read_json(manifest_path)
        if existing_manifest is not None:
            if not resume and not overwrite:
                raise DiagnosticIdentityError(
                    "A diagnostic manifest already exists; use --resume after verifying identity."
                )
            _validate_resume_identity(existing_manifest, identity)
            manifest = dict(existing_manifest)
            manifest.setdefault("resumed_at", []).append(utc_now_iso())
        else:
            manifest = {
                "schema_version": DIAGNOSTIC_MANIFEST_SCHEMA_VERSION,
                "diagnostic_run_key": diagnostic_key,
                "identity": identity,
                "protocol_phase": raw_config.get("protocol_phase"),
                "experiment_name": raw_config.get("experiment_name"),
                "instance_size": scale,
                "output_directory": str(output_dir),
                "cli_arguments": {
                    "config": str(config_file),
                    "resume": bool(resume),
                    "overwrite": bool(overwrite),
                },
                "config_path": str(config_file),
                "config_sha256": identity["config_sha256"],
                "resolved_config_sha256": identity["resolved_config_sha256"],
                "protocol_document_sha256": identity["protocol_document_sha256"],
                "candidate_config_sha256": identity["candidate_config_sha256"],
                "diagnostic_code_git_commit": current_commit,
                "seed_list": identity["seeds"],
                "method": EXPECTED_METHOD,
                "status": "initializing",
                "started_at": utc_now_iso(),
                "resumed_at": [],
                "completed_at": None,
                "failure_reason": "",
                "final_outputs": {},
                "completed_count": 0,
                "pending_count": 0,
                "failed_count": 0,
                "interrupted_count": 0,
            }
            _write_manifest(output_dir, manifest)
        try:
            locked_base = manifest.get("base_input_identity")
            if locked_base is None:
                base_complete = False
                try:
                    base_identity = validate_complete_base_results(
                        repo_root=repo_root,
                        output_dir=output_dir,
                        resolved_config=resolved_config,
                        current_commit=current_commit,
                    )
                    base_complete = True
                except (DiagnosticIdentityError, DiagnosticCheckpointError):
                    base_identity = None
                if not base_complete:
                    validate_partial_base_identity(
                        output_dir=output_dir,
                        resolved_config=resolved_config,
                        current_commit=current_commit,
                    )
                    manifest["status"] = "base_running"
                    _write_manifest(output_dir, manifest)
                    previous_cwd = Path.cwd()
                    try:
                        os.chdir(repo_root)
                        dependencies.base_runner(
                            raw_config,
                            resume=resume,
                            overwrite=overwrite,
                        )
                    finally:
                        os.chdir(previous_cwd)
                    base_identity = validate_complete_base_results(
                        repo_root=repo_root,
                        output_dir=output_dir,
                        resolved_config=resolved_config,
                        current_commit=current_commit,
                    )
                manifest["base_input_identity"] = {
                    "base_git_commit": base_identity["base_git_commit"],
                    "base_results_sha256": base_identity["base_files"]["results.csv"],
                    "base_summary_sha256": base_identity["base_files"]["summary.csv"],
                    "base_run_manifest_sha256": base_identity["base_files"]["run_manifest.json"],
                    "base_run_keys": base_identity["base_run_keys"],
                }
                manifest["base_git_commit"] = base_identity["base_git_commit"]
                manifest["base_results_files"] = base_identity["base_files"]
                _write_manifest(output_dir, manifest)
            else:
                actual_files = _base_file_identity(output_dir)
                expected_files = {
                    "results.csv": locked_base["base_results_sha256"],
                    "summary.csv": locked_base["base_summary_sha256"],
                    "run_manifest.json": locked_base["base_run_manifest_sha256"],
                }
                if actual_files != expected_files:
                    raise DiagnosticIdentityError(
                        "Base result hashes changed after diagnostic checkpoints were created."
                    )
                base_identity = validate_complete_base_results(
                    repo_root=repo_root,
                    output_dir=output_dir,
                    resolved_config=resolved_config,
                    current_commit=current_commit,
                )
                if base_identity["base_run_keys"] != locked_base["base_run_keys"]:
                    raise DiagnosticIdentityError("Base run keys changed during resume.")

            diagnostic_config = raw_config["fairness_diagnostic"]
            chunk_size = int(diagnostic_config["checkpoint_scenario_chunk_size"])
            if chunk_size <= 0:
                raise ValueError("checkpoint_scenario_chunk_size must be positive.")
            scenarios_by_run: dict[str, list[DemandScenario]] = {}
            instances_by_run: dict[str, InventoryInstance] = {}
            expected_scenario_count: int | None = None
            for run in base_identity["run_records"]:
                instance = load_instance(run["instance_path"])
                if file_sha256(run["instance_path"]).lower() != run["instance_sha256"]:
                    raise DiagnosticIdentityError("Instance identity changed during diagnostic setup.")
                enumeration = dependencies.scenario_enumerator(
                    instance,
                    int(diagnostic_config["gamma"]),
                    max_scenarios=int(diagnostic_config["max_scenarios"]),
                    exact_scenarios=bool(diagnostic_config["exact_scenarios"]),
                )
                if enumeration.scenario_mode != "full" or not enumeration.exact_scenarios:
                    raise DiagnosticIdentityError("Formal fairness diagnosis requires exact full enumeration.")
                scenarios = list(enumeration.scenarios)
                patterns = [scenario.active_units for scenario in scenarios]
                if len(patterns) != len(set(patterns)):
                    raise DiagnosticIdentityError("Enumerated scenarios contain duplicates.")
                if expected_scenario_count is None:
                    expected_scenario_count = len(scenarios)
                elif expected_scenario_count != len(scenarios):
                    raise DiagnosticIdentityError("Scenario counts differ across same-scale instances.")
                scenarios_by_run[run["base_run_key"]] = scenarios
                instances_by_run[run["base_run_key"]] = instance
            manifest["scenario_count_per_instance"] = int(expected_scenario_count or 0)
            manifest["total_scenario_count"] = int(expected_scenario_count or 0) * 10
            manifest["checkpoint_rule"] = {
                "scenario_order": "enumerate_budget_scenarios_with_metadata_order",
                "chunk_size": chunk_size,
                "atomic_commit": "temporary_file_then_os.replace",
                "checkpoint_is_resume_source_of_truth": True,
            }
            plan = _checkpoint_plan(base_identity["run_records"], scenarios_by_run, chunk_size)
            index_entries = _load_index_entries(output_dir, diagnostic_key)
            checkpoints: dict[tuple[str, int], dict[str, Any]] = {}
            planned_keys = {(run["base_run_key"], chunk_index) for run, chunk_index, _start, _end in plan}
            planned_paths = {
                _checkpoint_path(output_dir, run["base_run_key"], chunk_index).resolve()
                for run, chunk_index, _start, _end in plan
            }
            checkpoint_root = output_dir / "checkpoint"
            existing_checkpoint_paths = (
                {path.resolve() for path in checkpoint_root.glob("base_*/chunk_*.json")}
                if checkpoint_root.is_dir()
                else set()
            )
            unexpected_paths = existing_checkpoint_paths - planned_paths
            if unexpected_paths:
                raise DiagnosticCheckpointError(
                    "Unplanned checkpoint files exist: "
                    + ", ".join(str(path) for path in sorted(unexpected_paths))
                )
            if set(index_entries) - planned_keys:
                raise DiagnosticCheckpointError("Checkpoint index contains unplanned chunks.")
            for key, entry in index_entries.items():
                checkpoint_file = output_dir / str(entry["relative_path"])
                if not checkpoint_file.is_file():
                    raise DiagnosticCheckpointError(
                        f"Checkpoint listed in index is missing: {checkpoint_file}"
                    )
                if file_sha256(checkpoint_file).lower() != entry.get("sha256"):
                    raise DiagnosticCheckpointError(
                        f"Checkpoint listed in index is corrupted: {checkpoint_file}"
                    )

            manifest["status"] = "postprocessing"
            _write_manifest(output_dir, manifest)
            rebuilt_entries: list[dict[str, Any]] = []
            for run, chunk_index, start, end in plan:
                base_key = run["base_run_key"]
                key = (base_key, chunk_index)
                path = _checkpoint_path(output_dir, base_key, chunk_index)
                instance = instances_by_run[base_key]
                scenarios = scenarios_by_run[base_key]
                checkpoint: dict[str, Any] | None = None
                if path.exists():
                    checkpoint = _strict_json(path)
                    _validate_checkpoint(
                        checkpoint=checkpoint,
                        diagnostic_key=diagnostic_key,
                        base_run_key=base_key,
                        chunk_index=chunk_index,
                        scenarios=scenarios,
                        instance=instance,
                        start=start,
                        end=end,
                    )
                    if checkpoint.get("success") is True:
                        checkpoints[key] = checkpoint
                elif key in index_entries:
                    raise DiagnosticCheckpointError(f"Indexed checkpoint disappeared: {path}")
                if key not in checkpoints:
                    if checkpoint is not None and not resume:
                        raise DiagnosticCheckpointError(
                            "A failed checkpoint exists; use --resume to retry it."
                        )
                    records: list[dict[str, Any]] = []
                    for scenario_index in range(start, end):
                        record = _scenario_record(
                            instance=instance,
                            scenario=scenarios[scenario_index],
                            scenario_index=scenario_index,
                            best_x_values=run["best_x_values"],
                            diagnostic_config=diagnostic_config,
                            dependencies=dependencies,
                        )
                        records.append(record)
                        _inject(
                            dependencies,
                            "during_chunk",
                            base_run_key=base_key,
                            chunk_index=chunk_index,
                            scenario_index=scenario_index,
                        )
                    checkpoint = {
                        "schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "diagnostic_run_key": diagnostic_key,
                        "base_run_key": base_key,
                        "chunk_index": chunk_index,
                        "scenario_start": start,
                        "scenario_end_exclusive": end,
                        "scenario_records": records,
                        "success": all(record["valid"] for record in records),
                        "created_at": utc_now_iso(),
                    }
                    atomic_write_json(path, checkpoint)
                    checkpoints[key] = checkpoint
                    _inject(
                        dependencies,
                        "after_checkpoint_commit_before_index",
                        base_run_key=base_key,
                        chunk_index=chunk_index,
                    )
                rebuilt_entries.append(
                    {
                        "base_run_key": base_key,
                        "chunk_index": chunk_index,
                        "relative_path": str(path.relative_to(output_dir)).replace("\\", "/"),
                        "sha256": file_sha256(path).lower(),
                        "success": bool(checkpoints[key].get("success")),
                        "scenario_count": end - start,
                    }
                )
                _write_checkpoint_index(
                    output_dir=output_dir,
                    diagnostic_key=diagnostic_key,
                    entries=rebuilt_entries,
                )
                manifest.update(
                    _manifest_counts(
                        plan,
                        checkpoints,
                        interrupted_count=int(manifest.get("interrupted_count", 0)),
                    )
                )
                _write_manifest(output_dir, manifest)

            if len(checkpoints) != len(plan):
                raise DiagnosticCheckpointError("Not all planned chunks have checkpoints.")
            _inject(dependencies, "after_all_chunks_before_aggregation", chunk_count=len(plan))
            _region_rows, _summaries, diagnosis = _aggregate_outputs(
                output_dir=output_dir,
                manifest=manifest,
                run_records=base_identity["run_records"],
                checkpoints=checkpoints,
                diagnostic_config=diagnostic_config,
                dependencies=dependencies,
            )
            final_files = {}
            for filename in (
                "region_scenario_metrics.csv",
                "instance_summary.csv",
                "diagnosis.json",
                "audit_log.json",
                "resolved_config.yaml",
                "checkpoint/index.json",
            ):
                path = output_dir / filename
                if not path.is_file():
                    raise DiagnosticCheckpointError(f"Required final output is missing: {filename}")
                final_files[filename] = file_sha256(path).lower()
            all_chunks_valid = all(checkpoint.get("success") for checkpoint in checkpoints.values())
            manifest.update(
                _manifest_counts(
                    plan,
                    checkpoints,
                    interrupted_count=int(manifest.get("interrupted_count", 0)),
                )
            )
            manifest["final_outputs"] = final_files
            manifest["status"] = "completed" if all_chunks_valid else "failed"
            manifest["completed_at"] = utc_now_iso() if all_chunks_valid else None
            manifest["failure_reason"] = (
                "" if all_chunks_valid else "One or more recourse evaluations were invalid."
            )
            _write_manifest(output_dir, manifest)
            if not all_chunks_valid:
                raise RuntimeError(manifest["failure_reason"])
            return {
                "status": manifest["status"],
                "diagnostic_run_key": diagnostic_key,
                "output_dir": str(output_dir),
                "diagnosis": diagnosis,
                "manifest": str(manifest_path),
            }
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            manifest["failure_reason"] = "KeyboardInterrupt"
            manifest["interrupted_at"] = utc_now_iso()
            interrupted_count = int(manifest.get("interrupted_count", 0)) + 1
            manifest["interrupted_count"] = interrupted_count
            manifest["interrupted_chunk_count"] = interrupted_count
            _write_manifest(output_dir, manifest)
            raise
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["failure_reason"] = f"{type(exc).__name__}: {exc}"
            _write_manifest(output_dir, manifest)
            raise
