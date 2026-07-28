"""Identity-locked runner and dry-run for final fairness remediation stages.

The checked-in L0/L1/M1 configurations are intentionally protocol-only.  This
runner therefore exposes the complete gate and identity preflight now, while
refusing every formal invocation before creating an output directory or an
instance.  A later, separately reviewed authorization may change that external
gate; this implementation stage cannot.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import yaml

from .benders import solve_benders
from .experiment_protocol import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_yaml,
    config_sha256,
    file_sha256,
    git_commit,
    penalized_runtime_par2,
    read_json,
    utc_now_iso,
)
from .fairness_large_final_remediation import (
    CANDIDATE,
    CANDIDATE_SHA256,
    INITIAL_UB_THEOREM_SHA256,
    PROTOCOL_SHA256,
    solve_certified_adaptive_multicut_fair_benders,
)
from .fairness_large_final_remediation_audit import (
    CUT_SCHEMA,
    PATTERN_SCHEMA,
    QUANTIZATION_SCHEMA,
    RELATIVE_VIOLATION_SCHEMA,
    _candidate_paths,
    expand_plan,
)
from .fairness_post_evaluation import checkpointed_fairness_post_evaluation
from .fairness_benders import _baseline_method_config
from .experiment_suite import _base_config
from .instance import InventoryInstance, generate_instance


STAGES = {"L0", "L1", "M1"}
PROHIBITED_STAGES = {"holdout", "S2", "s2", "full-grid", "full_grid", "Attempt4", "attempt4"}
EXPECTED_FILE_SHA256 = {
    "L0": "C18AE2CA1BEA5D222197268462D6BE342553FA88CB06CB060A2D7CED28F24B2E",
    "L1": "08F77E62DCB0252AAA7E5C23B8234A2DCF40D46CC2290E6FB6851F202CE1DE53",
    "M1": "D8AAEAD792369032A47BE230954D0D5CB5EFE3A562FE50BF40D9FA6E10815C19",
}
RECOVERABLE_PHASES = (
    "algorithm_checkpoint",
    "separation_complete",
    "post_evaluation_chunks_complete",
    "aggregation_complete",
    "csv_complete",
)


class RemediationGateError(RuntimeError):
    """A formal stage or identity gate is not satisfied."""


@dataclass(frozen=True)
class RemediationDependencies:
    """Injectable boundaries; production defaults are used only after authorization."""

    generate_instance: Any | None = None
    solve_baseline: Any | None = None
    solve_frontier: Any | None = None
    post_evaluate: Any | None = None
    configure_solver: Any | None = None


_TEST_AUTHORIZATION = object()
SOLVER_PARAMETERS = {"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7}
RESULT_FIELDS = [
    "run_key", "run_directory_id", "stage", "scale", "task_type", "seed", "rho",
    "candidate", "state", "scientific_status", "algorithm_status", "certified_solved",
    "algorithm_runtime", "penalized_runtime_par2", "post_evaluation_wall_runtime",
    "total_wall_runtime", "instance_sha256", "baseline_run_key", "anchor_sha256",
]


def advance_recovery_ledger(
    path: str | Path,
    *,
    identity: dict[str, Any],
    phase_action: Any,
    failure_injector: Any | None = None,
) -> dict[str, Any]:
    """Advance post-algorithm phases atomically and resume without replaying commits."""
    target = Path(path)
    payload = read_json(target)
    if target.exists() and not isinstance(payload, dict):
        raise RemediationGateError("recovery ledger corrupt")
    if payload is None:
        payload = {
            "schema": "fairness_large_remediation_recovery_ledger_v1",
            "identity": deepcopy(identity),
            "completed_phases": [],
        }
    if payload.get("identity") != identity:
        raise RemediationGateError("recovery ledger identity mismatch")
    completed = payload.get("completed_phases")
    if not isinstance(completed, list) or len(completed) != len(set(completed)):
        raise RemediationGateError("recovery ledger phase list corrupt")
    if completed != list(RECOVERABLE_PHASES[: len(completed)]):
        raise RemediationGateError("recovery ledger phase order corrupt")
    for phase in RECOVERABLE_PHASES[len(completed):]:
        phase_action(phase)
        if failure_injector:
            failure_injector(f"before_{phase}_checkpoint", deepcopy(payload))
        committed = deepcopy(payload)
        committed["completed_phases"] = list(completed) + [phase]
        atomic_write_json(target, committed)
        payload = committed
        completed = payload["completed_phases"]
        if failure_injector:
            failure_injector(f"after_{phase}_checkpoint", deepcopy(payload))
    return payload


def load_remediation_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Remediation config must be a YAML mapping.")
    return payload


def validate_frozen_config(path: str | Path, config: dict[str, Any], *, stage: str) -> None:
    if stage not in STAGES:
        if stage in PROHIBITED_STAGES:
            raise RemediationGateError(f"stage {stage} is permanently unauthorized by this runner")
        raise RemediationGateError("stage must be exactly L0, L1, or M1")
    if config.get("stage") != stage:
        raise RemediationGateError("CLI stage does not match the frozen config stage")
    actual = file_sha256(path).upper()
    if actual != EXPECTED_FILE_SHA256[stage]:
        raise RemediationGateError("frozen stage config file SHA256 mismatch")
    if config.get("required_protocol_sha256") != PROTOCOL_SHA256:
        raise RemediationGateError("frozen protocol SHA256 mismatch")
    if config.get("required_candidate_sha256") != CANDIDATE_SHA256:
        raise RemediationGateError("frozen candidate SHA256 mismatch")
    if config.get("candidate") != CANDIDATE:
        raise RemediationGateError("a second remediation candidate is forbidden")
    expected_attempt = 4 if stage == "L0" else 3
    if int(config.get("execution_attempt", -1)) != expected_attempt:
        raise RemediationGateError("remediation execution attempt identity mismatch")
    if config.get("previous_attempt_results_reused") is not False:
        raise RemediationGateError("previous formal results may not be reused")
    if config.get("overwrite_supported") is not False:
        raise RemediationGateError("overwrite must remain unsupported")
    seeds = [int(value) for value in config.get("seeds", [])]
    if any(130 <= value <= 159 or 170 <= value <= 179 for value in seeds):
        raise RemediationGateError("prohibited or holdout seed requested")
    if seeds != ([160] if stage == "L0" else [160, 161, 162]):
        raise RemediationGateError("development seed matrix drift")
    if [float(value) for value in config.get("rho", [])] != ([0.0] if stage == "L0" else [0.0, 0.01]):
        raise RemediationGateError("rho matrix drift")
    locks = set(config.get("manifest_identity_lock", []))
    required_locks = {
        "git", "config_file_sha256", "resolved_config_file_sha256",
        "resolved_config_canonical_sha256", "protocol_sha256", "candidate_sha256",
        "pattern_schema", "cut_schema", "quantization_schema",
        "relative_violation_identity", "adaptive_batch_schedule",
        "committed_cut_state_machine", "checkpoint_selection_state",
        "solver", "instance", "baseline", "anchor",
    }
    if not required_locks.issubset(locks):
        raise RemediationGateError("manifest scientific identity lock is incomplete")


def validate_stage_gate(stage: str, decision: dict[str, Any] | None) -> None:
    if stage == "L0":
        if decision not in (None, {}):
            raise RemediationGateError("L0 does not consume a prior run decision")
        return
    if not isinstance(decision, dict):
        raise RemediationGateError(f"{stage} requires an identity-locked prior-stage decision")
    if stage == "L1":
        required = {
            "decision": "authorize_L1_after_L0",
            "scientific_status": "certified_robust_optimal",
            "post_evaluation_valid": True,
            "scenario_count": 4657,
            "valid_upper_bound": True,
            "valid_lower_bound": True,
            "implementation_error": False,
            "invalid_post_evaluation": False,
        }
    elif stage == "M1":
        required = {
            "decision": "authorize_M1_after_L1",
            "certified_frontier_count": 4,
            "frontier_count": 6,
            "implementation_error_count": 0,
            "invalid_post_evaluation_count": 0,
            "all_successful_post_evaluations_valid": True,
        }
        if int(decision.get("certified_frontier_count", -1)) < 4:
            raise RemediationGateError("L1 certified frontier is below 4/6")
        required.pop("certified_frontier_count")
    else:
        raise RemediationGateError("unknown remediation stage")
    for field, expected in required.items():
        if decision.get(field) != expected:
            raise RemediationGateError(f"prior-stage decision mismatch: {field}")
    if not isinstance(decision.get("identity_sha256"), str) or len(decision["identity_sha256"]) != 64:
        raise RemediationGateError("prior-stage decision identity SHA256 missing")


def execution_identity(
    root: Path, config_path: Path, config: dict[str, Any], *, stage: str,
    decision: dict[str, Any] | None,
) -> dict[str, Any]:
    decision_hash = None
    if decision:
        decision_hash = hashlib.sha256(
            json.dumps(decision, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest().upper()
    resolved_bytes = yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode("utf-8")
    return {
        "schema": "fairness_large_final_remediation_manifest_v1",
        "execution_attempt": int(config["execution_attempt"]),
        "stage": stage,
        "git_commit": git_commit(root),
        "config_file_sha256": file_sha256(config_path).upper(),
        "resolved_config_file_sha256": hashlib.sha256(resolved_bytes).hexdigest().upper(),
        "resolved_config_canonical_sha256": config_sha256(config).upper(),
        "protocol_sha256": PROTOCOL_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "candidate": CANDIDATE,
        "initial_ub_theorem_sha256": INITIAL_UB_THEOREM_SHA256,
        "pattern_schema": PATTERN_SCHEMA,
        "cut_schema": CUT_SCHEMA,
        "relative_violation_schema": RELATIVE_VIOLATION_SCHEMA,
        "quantization_schema": QUANTIZATION_SCHEMA,
        "solver_parameters": deepcopy(SOLVER_PARAMETERS),
        "baseline_time_limit_seconds": float(config["baseline_time_limit_seconds"]),
        "fairness_time_limit_seconds": float(config["algorithm_time_limit_seconds"]),
        "general_time_limit_seconds": float(config["general_time_limit_seconds"]),
        "post_evaluation_identity": deepcopy(config["post_evaluation"]),
        "checkpoint_identity": deepcopy(config["checkpointing"]),
        "scale": str(config["scale"]),
        "seeds": [int(value) for value in config["seeds"]],
        "rhos": [float(value) for value in config["rho"]],
        "relative_violation_identity": {
            "quantum": "1e-9",
            "threshold_bucket": 100_000_000,
            "deadband_buckets": 2,
            "eligibility": "strictly_greater_than_100000002",
            "denominator": "maximum_normalized_violation_bucket_over_unique_certified_violating_candidates",
        },
        "adaptive_batch_schedule": [[1000, 5], [3000, 3], [5000, 2], [None, 1]],
        "stage_decision_sha256": decision_hash,
        "previous_attempt_results_reused": False,
        "prior_attempts": deepcopy(config["prior_attempts"]),
        "formal_run_authorized": bool(config.get("formal_run_authorized")),
    }


def dry_run_remediation(
    config_path: str | Path,
    *,
    stage: str,
    root: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    repository = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    config = load_remediation_config(config_path)
    validate_frozen_config(config_path, config, stage=stage)
    rows = expand_plan(config)
    if len({row["run_key"] for row in rows}) != len(rows):
        raise RemediationGateError("duplicate run key in frozen plan")
    if len({row["run_directory_id"] for row in rows}) != len(rows):
        raise RemediationGateError("short run-directory hash collision")
    output = repository / str(config["output_dir"])
    paths = _candidate_paths(repository, config, rows)
    longest = max(paths, key=lambda value: len(str(value.resolve())))
    maximum = int(config["path_policy"]["maximum_absolute_path_length"])
    if len(str(longest.resolve())) > maximum:
        raise RemediationGateError("Windows path portability limit exceeded")
    return {
        "stage": stage,
        "tasks": len(rows),
        "baseline_count": sum(row["task_type"] == "baseline" for row in rows),
        "frontier_count": sum(row["task_type"] == "frontier" for row in rows),
        "scenario_count": int(config["scenario_count"]),
        "run_key_count": len(rows),
        "run_directory_id_count": len(rows),
        "longest_windows_path": str(longest.resolve()),
        "longest_windows_path_length": len(str(longest.resolve())),
        "instances_generated": False,
        "solver_called": False,
        "output_dir_exists": output.exists(),
        "formal_run_authorized": False,
        "identity": execution_identity(repository, config_path, config, stage=stage, decision=None),
    }


def _run_root(output_dir: Path, row: dict[str, Any]) -> Path:
    return output_dir / "runs" / str(row["run_directory_id"])


def _read_json_strict(path: Path, *, label: str) -> dict[str, Any] | None:
    payload = read_json(path)
    if path.exists() and not isinstance(payload, dict):
        raise RemediationGateError(f"{label} corrupt")
    return payload


def _instance_identity(instance: InventoryInstance) -> str:
    return config_sha256(instance.to_dict()).upper()


def _same_identity(actual: dict[str, Any], expected: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return all(actual.get(field) == expected.get(field) for field in fields)


def _certified_anchor(
    baseline_record: dict[str, Any], *, common_identity: dict[str, Any], tolerance: float,
) -> dict[str, Any]:
    result = baseline_record.get("result")
    if not isinstance(result, dict):
        raise RemediationGateError("certified baseline result missing")
    upper = result.get("upper_bound")
    gap = result.get("gap")
    if not (
        baseline_record.get("scientific_status") == "certified_robust_optimal"
        and baseline_record.get("solved_to_tolerance") is True
        and result.get("status") == "optimal"
        and result.get("valid_UB") is True
        and upper is not None and math.isfinite(float(upper))
        and gap is not None and math.isfinite(float(gap)) and float(gap) <= tolerance
    ):
        raise RemediationGateError("certified baseline anchor unavailable")
    payload = {
        "source": "solve_result.upper_bound",
        "value": float(upper),
        "value_hex": float(upper).hex(),
        "anchor_value_hex": float(upper).hex(),
        "base_git_commit": common_identity["git_commit"],
        "base_config_sha256": common_identity["resolved_config_file_sha256"],
        "candidate_config_sha256": CANDIDATE_SHA256,
        "valid_UB": True,
        "baseline_status": "optimal",
        "baseline_final_gap": float(gap),
        **deepcopy(common_identity),
    }
    payload["anchor_sha256"] = config_sha256(payload).upper()
    return payload


def _classify_frontier(
    result: dict[str, Any], post_evaluation: dict[str, Any] | None, *, tolerance: float,
    expected_scenario_count: int | None = None,
) -> str:
    status = str(result.get("status", "unknown"))
    if status == "interrupted":
        return "interrupted"
    if status == "time_limit":
        return "time_limit_uncertified"
    if status == "infeasible":
        return "infeasible"
    gap = result.get("gap")
    log = result.get("iteration_log")
    final_iteration = log[-1] if isinstance(log, list) and log and isinstance(log[-1], dict) else {}
    separation_bound = final_iteration.get("separation_objective_bound")
    algorithm_certified = (
        status == "optimal"
        and result.get("lower_bound") is not None and math.isfinite(float(result["lower_bound"]))
        and result.get("upper_bound") is not None and math.isfinite(float(result["upper_bound"]))
        and gap is not None and math.isfinite(float(gap)) and float(gap) <= tolerance
        and result.get("metadata", {}).get("full_separation_objective_bound_required") is True
        and final_iteration.get("robust_feasibility_certified") is True
        and final_iteration.get("master_status") == "optimal"
        and separation_bound is not None and math.isfinite(float(separation_bound))
        and float(separation_bound) <= SOLVER_PARAMETERS["FeasibilityTol"]
    )
    if not algorithm_certified:
        return "master_optimal_but_robust_uncertified"
    if not isinstance(post_evaluation, dict) or post_evaluation.get("valid") is not True:
        return "invalid_post_evaluation"
    if post_evaluation.get("objective_t_consistent") is False or post_evaluation.get("errors") not in (None, []):
        return "invalid_post_evaluation"
    if expected_scenario_count is not None and post_evaluation.get("scenario_count") != expected_scenario_count:
        return "invalid_post_evaluation"
    return "certified_robust_optimal"


def _configure_solver_parameters(settings: dict[str, Any]) -> None:
    if settings != SOLVER_PARAMETERS:
        raise RemediationGateError("frozen solver identity mismatch")
    import gurobipy as gp
    for key, value in settings.items():
        gp.setParam(key, value)


def _scale_template(config: dict[str, Any]) -> dict[str, Any]:
    name = (
        "fairness_scalability_development_large.yaml"
        if config["scale"] == "large"
        else "fairness_scalability_development_medium_large.yaml"
    )
    path = Path(__file__).resolve().parents[1] / "experiments" / "configs" / name
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RemediationGateError("frozen instance template is invalid")
    return payload


def _production_generate_instance(config: dict[str, Any], seed: int) -> InventoryInstance:
    template = _scale_template(config)
    return generate_instance(_base_config(template, str(config["scale"]), seed), seed=seed)


def _production_solve_baseline(
    config: dict[str, Any], instance: InventoryInstance, seed: int, solver_parameters: dict[str, Any],
) -> dict[str, Any]:
    _configure_solver_parameters(solver_parameters)
    template = _scale_template(config)
    resolved = deepcopy(template)
    resolved["time_limit"] = float(config["baseline_time_limit_seconds"])
    method, method_config = _baseline_method_config(resolved, str(config["scale"]), seed)
    result = solve_benders(method_config, instance, method)
    payload = result.summary_dict()
    payload["iteration_log"] = result.iteration_log
    return payload


def _production_solve_frontier(
    config: dict[str, Any], instance: InventoryInstance, baseline_record: dict[str, Any],
    anchor: dict[str, Any], expected_identity: dict[str, Any], checkpoint_path: Path,
    solver_parameters: dict[str, Any], row: dict[str, Any],
) -> dict[str, Any]:
    result = solve_certified_adaptive_multicut_fair_benders(
        instance, baseline_record=baseline_record, anchor=anchor,
        expected_identity=expected_identity, solver_parameters=solver_parameters,
        rho=float(row["rho"]), gamma=2, max_iterations=10000,
        time_limit=float(config["algorithm_time_limit_seconds"]), tol=1.0e-4,
        feasibility_tolerance=float(solver_parameters["FeasibilityTol"]), output_flag=False,
        checkpoint_path=checkpoint_path, checkpoint_identity={
            "run_key": row["run_key"], **deepcopy(expected_identity),
        },
    )
    return result.to_dict()


def _production_post_evaluate(
    config: dict[str, Any], instance: InventoryInstance, result: dict[str, Any],
    anchor: dict[str, Any], run_identity: dict[str, Any], root: Path, row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    evaluation, timing = checkpointed_fairness_post_evaluation(
        instance, root=root, run_key=row["run_key"],
        config_sha256_value=run_identity["resolved_config_file_sha256"],
        git_commit=run_identity["git_commit"], baseline_anchor_sha256=anchor["anchor_sha256"],
        y_values=result["y_values"], x_values=result["x_values"],
        t_value=float(result["objective_t"]), baseline_cost=float(anchor["value"]),
        rho=float(row["rho"]), gamma=2,
        max_scenarios=int(config["post_evaluation"]["scenario_count"]),
        per_scenario_time_limit=float(config["post_evaluation"]["time_limit_per_scenario_seconds"]),
        tolerance=float(SOLVER_PARAMETERS["FeasibilityTol"]),
        chunk_size=int(config["post_evaluation"]["checkpoint_chunk_size"]),
        resume_count=0, output_flag=False,
    )
    return evaluation.to_dict(), {
        "post_evaluation_solver_runtime": timing.solver_runtime,
        "post_evaluation_wall_runtime": timing.wall_runtime,
        "aggregation_runtime": timing.aggregation_runtime,
        "checkpoint_io_runtime": timing.checkpoint_io_runtime,
    }


def _production_dependencies() -> RemediationDependencies:
    return RemediationDependencies(
        generate_instance=_production_generate_instance,
        solve_baseline=_production_solve_baseline,
        solve_frontier=_production_solve_frontier,
        post_evaluate=_production_post_evaluate,
        configure_solver=_configure_solver_parameters,
    )


def _aggregate_records(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    result_rows = []
    for row in rows:
        record = _read_json_strict(_run_root(output_dir, row) / "run.json", label="run record")
        if record is None:
            continue
        result = record.get("result", {})
        result_rows.append({
            **{field: record.get(field, "NOT_APPLICABLE") for field in RESULT_FIELDS},
            "algorithm_runtime": result.get("algorithm_runtime", result.get("runtime", 0.0)),
            "penalized_runtime_par2": result.get("penalized_runtime_par2"),
            "post_evaluation_wall_runtime": result.get("post_evaluation_wall_runtime", 0.0),
            "total_wall_runtime": result.get("total_wall_runtime", result.get("runtime", 0.0)),
            "certified_solved": record.get("scientific_status") == "certified_robust_optimal",
        })
    atomic_write_csv(output_dir / "results.csv", result_rows, RESULT_FIELDS)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in result_rows:
        key = (row["scale"], row["task_type"], row["candidate"], row["rho"])
        groups.setdefault(key, []).append(row)
    summary = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        values = groups[key]
        summary.append({
            "scale": key[0], "task_type": key[1], "candidate": key[2], "rho": key[3],
            "run_count": len(values),
            "certified_solved_count": sum(bool(value["certified_solved"]) for value in values),
            "mean_algorithm_runtime": math.fsum(float(value["algorithm_runtime"]) for value in values) / len(values),
            "mean_penalized_runtime_par2": math.fsum(float(value["penalized_runtime_par2"]) for value in values) / len(values),
        })
    summary_fields = ["scale", "task_type", "candidate", "rho", "run_count", "certified_solved_count", "mean_algorithm_runtime", "mean_penalized_runtime_par2"]
    atomic_write_csv(output_dir / "summary.csv", summary, summary_fields)


def _execute_pipeline(
    config_path: Path, config: dict[str, Any], *, stage: str, resume: bool,
    decision: dict[str, Any] | None, output_dir: Path, dependencies: RemediationDependencies,
    failure_injector: Any | None = None,
) -> dict[str, Any]:
    validate_stage_gate(stage, decision)
    rows = expand_plan(config)
    path_config = deepcopy(config)
    path_config["output_dir"] = str(output_dir)
    planned_paths = _candidate_paths(Path.cwd(), path_config, rows)
    maximum_path = max(len(str(path.resolve())) for path in planned_paths)
    if maximum_path > int(config["path_policy"]["maximum_absolute_path_length"]):
        raise RemediationGateError("Windows path portability limit exceeded")
    identity = execution_identity(Path(__file__).resolve().parents[1], config_path, config, stage=stage, decision=decision)
    identity["path_portability"] = {"maximum_absolute_path_length": maximum_path, "validated": True}
    manifest_path = output_dir / "manifest.json"
    existing_manifest = _read_json_strict(manifest_path, label="manifest")
    if output_dir.exists() and existing_manifest is None:
        raise RemediationGateError("existing output has no valid manifest")
    if existing_manifest is not None:
        if not resume or existing_manifest.get("identity") != identity:
            raise RemediationGateError("resume manifest identity mismatch")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_yaml(output_dir / "resolved_config.yaml", config)
        if file_sha256(output_dir / "resolved_config.yaml").upper() != identity["resolved_config_file_sha256"]:
            raise RemediationGateError("resolved config file identity mismatch")
    manifest = existing_manifest or {
        "schema": "fairness_large_final_remediation_manifest_v1",
        "identity": identity,
        "run_key_to_directory_id": {row["run_key"]: row["run_directory_id"] for row in rows},
        "directory_id_to_run_key": {row["run_directory_id"]: row["run_key"] for row in rows},
        "baseline_anchors": {}, "run_identities": {}, "completed_run_count": 0,
        "certified_solved_count": 0, "created_at": utc_now_iso(),
    }
    atomic_write_json(manifest_path, manifest)
    configure = dependencies.configure_solver or _configure_solver_parameters
    configure(deepcopy(SOLVER_PARAMETERS))
    for seed in sorted({int(row["seed"]) for row in rows}):
        instance_path = output_dir / "instances" / f"{seed}.json"
        stored_instance = _read_json_strict(instance_path, label="instance")
        if stored_instance is None:
            if dependencies.generate_instance is None:
                raise RemediationGateError("production instance generator is not configured")
            instance = dependencies.generate_instance(config, seed)
            if not isinstance(instance, InventoryInstance):
                raise RemediationGateError("instance generator returned invalid type")
            atomic_write_json(instance_path, instance.to_dict())
        else:
            instance = InventoryInstance.from_dict(stored_instance)
        instance_sha256 = _instance_identity(instance)
        baseline_row = next(row for row in rows if row["task_type"] == "baseline" and int(row["seed"]) == seed)
        baseline_root = _run_root(output_dir, baseline_row)
        baseline_record = _read_json_strict(baseline_root / "run.json", label="baseline run")
        common = {
            "instance_sha256": instance_sha256, "seed": seed, "scale": str(config["scale"]),
            "git_commit": identity["git_commit"], "config_file_sha256": identity["config_file_sha256"],
            "resolved_config_file_sha256": identity["resolved_config_file_sha256"],
            "candidate_sha256": CANDIDATE_SHA256, "baseline_run_key": baseline_row["run_key"],
        }
        if baseline_record is None:
            if dependencies.solve_baseline is None:
                raise RemediationGateError("production baseline solver is not configured")
            baseline_checkpoint_path = baseline_root / "baseline_checkpoint.json"
            baseline_checkpoint = _read_json_strict(baseline_checkpoint_path, label="baseline checkpoint")
            if baseline_checkpoint is not None:
                if baseline_checkpoint.get("identity") != common or not isinstance(baseline_checkpoint.get("result"), dict):
                    raise RemediationGateError("baseline checkpoint identity mismatch")
                baseline_payload = deepcopy(baseline_checkpoint["result"])
            else:
                baseline_payload = dependencies.solve_baseline(config, instance, seed, deepcopy(SOLVER_PARAMETERS))
                atomic_write_json(baseline_checkpoint_path, {"identity": common, "result": baseline_payload})
                if failure_injector:
                    failure_injector("after_baseline_checkpoint", baseline_row)
            baseline_solved = (
                baseline_payload.get("status") == "optimal" and baseline_payload.get("valid_UB") is True
                and baseline_payload.get("gap") is not None
                and float(baseline_payload["gap"]) <= 1.0e-4
            )
            baseline_runtime = float(baseline_payload.get("runtime", 0.0))
            baseline_payload["algorithm_runtime"] = baseline_runtime
            baseline_payload["post_evaluation_wall_runtime"] = 0.0
            baseline_payload["total_wall_runtime"] = baseline_runtime
            baseline_payload["penalized_runtime_par2"] = penalized_runtime_par2(
                solved_to_tolerance=baseline_solved, runtime=baseline_runtime,
                time_limit=float(config["baseline_time_limit_seconds"]),
            )
            baseline_record = {
                **common, "run_key": baseline_row["run_key"], "run_directory_id": baseline_row["run_directory_id"],
                "stage": stage, "task_type": "baseline", "rho": "NOT_APPLICABLE", "candidate": "baseline",
                "state": "complete", "algorithm_status": baseline_payload.get("status"),
                "scientific_status": "certified_robust_optimal" if baseline_solved else "master_optimal_but_robust_uncertified",
                "solved_to_tolerance": baseline_solved, "result": baseline_payload,
            }
            atomic_write_json(baseline_root / "run.json", baseline_record)
            atomic_write_json(baseline_root / "status.json", {"state": "complete", "scientific_status": baseline_record["scientific_status"]})
        elif not _same_identity(baseline_record, common, tuple(common)):
            raise RemediationGateError("baseline resume identity mismatch")
        anchor = _certified_anchor(baseline_record, common_identity=common, tolerance=1.0e-4)
        manifest["baseline_anchors"][str(seed)] = anchor
        for row in [item for item in rows if item["task_type"] == "frontier" and int(item["seed"]) == seed]:
            run_root = _run_root(output_dir, row)
            record = _read_json_strict(run_root / "run.json", label="frontier run")
            run_identity = {
                **common, "run_key": row["run_key"], "run_directory_id": row["run_directory_id"],
                "rho": float(row["rho"]), "anchor_sha256": anchor["anchor_sha256"],
                "anchor_value_hex": anchor["value_hex"], "checkpoint_schema": "fairness_large_remediation_algorithm_checkpoint_v1",
                "post_evaluation_schema": "fairness_post_evaluation_checkpoint_v1",
            }
            manifest["run_identities"][row["run_key"]] = run_identity
            if record is not None:
                if not _same_identity(record, run_identity, tuple(run_identity)):
                    raise RemediationGateError("frontier resume identity mismatch")
                continue
            if dependencies.solve_frontier is None or dependencies.post_evaluate is None:
                raise RemediationGateError("production frontier pipeline is not configured")
            if failure_injector:
                failure_injector("before_frontier", row)
            expected_ub_identity = {
                **{key: common[key] for key in (
                    "instance_sha256", "seed", "scale", "git_commit", "config_file_sha256",
                    "resolved_config_file_sha256", "candidate_sha256", "baseline_run_key",
                )},
                "anchor_value_hex": anchor["value_hex"], "anchor_sha256": anchor["anchor_sha256"],
            }
            algorithm_start = time.perf_counter()
            try:
                result = dependencies.solve_frontier(
                    config, instance, baseline_record, anchor, expected_ub_identity,
                    run_root / "algorithm_checkpoint.json", deepcopy(SOLVER_PARAMETERS), row,
                )
                algorithm_runtime = float(result.get("runtime", time.perf_counter() - algorithm_start))
                algorithm_certified = (
                    result.get("status") == "optimal" and result.get("gap") is not None
                    and float(result["gap"]) <= 1.0e-4
                )
                evaluation = None
                timing = {"post_evaluation_solver_runtime": 0.0, "post_evaluation_wall_runtime": 0.0, "aggregation_runtime": 0.0, "checkpoint_io_runtime": 0.0}
                if algorithm_certified:
                    evaluation, timing = dependencies.post_evaluate(
                        config, instance, result, anchor, run_identity, run_root / "post_evaluation", row,
                    )
            except KeyboardInterrupt:
                atomic_write_json(run_root / "status.json", {
                    "state": "interrupted", "scientific_status": "interrupted",
                    "algorithm_status": "interrupted",
                })
                raise
            except Exception as exc:
                failure = {
                    **run_identity, "stage": stage, "task_type": "frontier", "candidate": CANDIDATE,
                    "state": "complete", "algorithm_status": "exception",
                    "scientific_status": "implementation_error", "solved_to_tolerance": False,
                    "failure_reason": str(exc), "result": {
                        "status": "exception", "algorithm_runtime": time.perf_counter() - algorithm_start,
                        "penalized_runtime_par2": 2.0 * float(config["algorithm_time_limit_seconds"]),
                    },
                }
                atomic_write_json(run_root / "run.json", failure)
                atomic_write_json(run_root / "status.json", {
                    "state": "complete", "scientific_status": "implementation_error",
                    "algorithm_status": "exception",
                })
                atomic_write_json(manifest_path, manifest)
                _aggregate_records(output_dir, rows)
                raise
            scientific = _classify_frontier(
                result, evaluation, tolerance=1.0e-4,
                expected_scenario_count=int(config["post_evaluation"]["scenario_count"]),
            )
            result.update(timing)
            result["post_evaluation"] = evaluation
            result["algorithm_runtime"] = algorithm_runtime
            result["post_evaluation_wall_runtime"] = float(timing.get("post_evaluation_wall_runtime", 0.0))
            result["total_wall_runtime"] = algorithm_runtime + result["post_evaluation_wall_runtime"] + float(timing.get("aggregation_runtime", 0.0)) + float(timing.get("checkpoint_io_runtime", 0.0))
            result["penalized_runtime_par2"] = penalized_runtime_par2(
                solved_to_tolerance=scientific == "certified_robust_optimal",
                runtime=algorithm_runtime, time_limit=float(config["algorithm_time_limit_seconds"]),
            )
            record = {
                **run_identity, "stage": stage, "task_type": "frontier", "candidate": CANDIDATE,
                "state": "complete", "algorithm_status": result.get("status"),
                "scientific_status": scientific, "solved_to_tolerance": scientific == "certified_robust_optimal",
                "result": result,
            }
            atomic_write_json(run_root / "run.json", record)
            atomic_write_json(run_root / "status.json", {"state": "complete", "scientific_status": scientific, "algorithm_status": result.get("status")})
            if failure_injector:
                failure_injector("after_frontier_record", row)
        atomic_write_json(manifest_path, manifest)
    _aggregate_records(output_dir, rows)
    records = [
        _read_json_strict(_run_root(output_dir, row) / "run.json", label="run record") for row in rows
    ]
    manifest["completed_run_count"] = sum(record is not None and record.get("state") == "complete" for record in records)
    manifest["certified_solved_count"] = sum(record is not None and record.get("scientific_status") == "certified_robust_optimal" for record in records)
    manifest["updated_at"] = utc_now_iso()
    atomic_write_json(manifest_path, manifest)
    return manifest


def run_remediation_stage(
    config_path: str | Path,
    *,
    stage: str,
    resume: bool,
    dry_run: bool,
    decision: dict[str, Any] | None = None,
    dependencies: RemediationDependencies | None = None,
    output_dir_override: str | Path | None = None,
    test_authorization: object | None = None,
    failure_injector: Any | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_remediation_config(config_path)
    validate_frozen_config(config_path, config, stage=stage)
    if dry_run:
        return dry_run_remediation(config_path, stage=stage)
    # This test happens before gate parsing, mkdir, instance generation, module
    # imports that create models, or solver parameter configuration.
    is_test = test_authorization is _TEST_AUTHORIZATION
    if not is_test and (
        config.get("formal_run_authorized") is not True
        or config.get("authorization") != "formal_execution_authorized"
    ):
        raise RemediationGateError("formal_run_not_authorized")
    if is_test and (dependencies is None or output_dir_override is None):
        raise RemediationGateError("test execution requires explicit dependencies and isolated output")
    output_dir = (
        Path(output_dir_override).resolve()
        if output_dir_override is not None
        else (Path(__file__).resolve().parents[1] / str(config["output_dir"])).resolve()
    )
    return _execute_pipeline(
        config_path, config, stage=stage, resume=resume, decision=decision,
        output_dir=output_dir, dependencies=dependencies or _production_dependencies(),
        failure_injector=failure_injector,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final fairness remediation runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_remediation_stage(
        args.config,
        stage=args.stage,
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
