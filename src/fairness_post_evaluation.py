from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from .experiment_protocol import (
    atomic_write_json,
    config_sha256,
    file_sha256,
    read_json,
    utc_now_iso,
)
from .instance import InventoryInstance
from .robust_regional_fairness import (
    FAIRNESS_METRIC_TOLERANCE,
    FairnessScenarioPolicy,
    FairnessSolutionEvaluation,
    constraint_acceptance_evidence,
    fairness_cost_budget,
    feasibility_acceptance_threshold,
    first_stage_cost_value,
    solve_scenario_policy_with_shared_caps,
)
from .scenarios import DemandScenario, enumerate_budget_scenarios_with_metadata


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_INDEX_SCHEMA_VERSION = 1
POST_EVALUATION_SCHEMA_VERSION = 1


class PostEvaluationCheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class PostEvaluationTiming:
    solver_runtime: float
    wall_runtime: float
    aggregation_runtime: float
    checkpoint_io_runtime: float
    resume_count: int


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scenario_identity(scenario: DemandScenario, index: int) -> dict[str, Any]:
    pattern = [
        {"region": int(region), "product": int(product), "value": 1}
        for region, product in scenario.active_units
    ]
    payload = {
        "scenario_index": int(index),
        "scenario_name": str(scenario.name),
        "deviation_pattern": pattern,
    }
    return {
        **payload,
        "deviation_pattern_sha256": _canonical_sha256(pattern),
        "scenario_key": _canonical_sha256(payload),
    }


def _checkpoint_path(root: Path, chunk_index: int) -> Path:
    return root / "checkpoint" / f"chunk_{chunk_index:05d}.json"


def _index_path(root: Path) -> Path:
    return root / "checkpoint" / "index.json"


def _output_path(root: Path) -> Path:
    return root / "post_evaluation.json"


def _inject(
    failure_injector: Callable[[str, Mapping[str, Any]], None] | None,
    stage: str,
    context: Mapping[str, Any],
) -> None:
    if failure_injector is not None:
        failure_injector(stage, context)


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    expected_scenarios: list[dict[str, Any]],
    chunk_index: int,
    start: int,
    end: int,
) -> None:
    records = checkpoint.get("records")
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("identity_sha256") != _canonical_sha256(identity)
        or checkpoint.get("chunk_index") != chunk_index
        or checkpoint.get("scenario_start") != start
        or checkpoint.get("scenario_end_exclusive") != end
        or not isinstance(records, list)
        or len(records) != end - start
        or [record.get("scenario_key") for record in records]
        != [item["scenario_key"] for item in expected_scenarios[start:end]]
    ):
        raise PostEvaluationCheckpointError(
            f"Post-evaluation checkpoint {chunk_index} has invalid identity or scenario order."
        )


def _load_checkpoints(
    root: Path,
    *,
    identity: Mapping[str, Any],
    scenario_identities: list[dict[str, Any]],
    chunk_size: int,
) -> dict[int, dict[str, Any]]:
    checkpoint_dir = root / "checkpoint"
    planned = {
        index: (start, min(len(scenario_identities), start + chunk_size))
        for index, start in enumerate(range(0, len(scenario_identities), chunk_size))
    }
    index_file = _index_path(root)
    index_payload = read_json(index_file)
    if index_file.exists() and index_payload is None:
        raise PostEvaluationCheckpointError("Post-evaluation checkpoint index is corrupt.")
    indexed: dict[int, dict[str, Any]] = {}
    if index_payload is not None:
        if (
            index_payload.get("schema_version") != CHECKPOINT_INDEX_SCHEMA_VERSION
            or index_payload.get("identity_sha256") != _canonical_sha256(identity)
        ):
            raise PostEvaluationCheckpointError("Post-evaluation checkpoint index identity drift.")
        for entry in index_payload.get("chunks", []):
            chunk_index = int(entry["chunk_index"])
            if chunk_index in indexed:
                raise PostEvaluationCheckpointError("Duplicate checkpoint index entry.")
            if entry.get("relative_path") != _checkpoint_path(
                root, chunk_index
            ).relative_to(root).as_posix():
                raise PostEvaluationCheckpointError("Checkpoint index path drift.")
            indexed[chunk_index] = dict(entry)

    discovered = (
        list(checkpoint_dir.glob("chunk_*.json")) if checkpoint_dir.is_dir() else []
    )
    checkpoints: dict[int, dict[str, Any]] = {}
    for path in discovered:
        try:
            chunk_index = int(path.stem.split("_")[-1])
        except ValueError as exc:
            raise PostEvaluationCheckpointError(
                f"Unexpected checkpoint filename: {path.name}"
            ) from exc
        if chunk_index not in planned:
            raise PostEvaluationCheckpointError(f"Unplanned checkpoint: {path.name}")
        checkpoint = read_json(path)
        if checkpoint is None:
            raise PostEvaluationCheckpointError(f"Corrupt checkpoint: {path.name}")
        start, end = planned[chunk_index]
        _validate_checkpoint(
            checkpoint,
            identity=identity,
            expected_scenarios=scenario_identities,
            chunk_index=chunk_index,
            start=start,
            end=end,
        )
        if chunk_index in indexed and file_sha256(path) != indexed[chunk_index]["sha256"]:
            raise PostEvaluationCheckpointError(f"Checkpoint hash mismatch: {path.name}")
        checkpoints[chunk_index] = checkpoint
    missing = set(indexed) - set(checkpoints)
    if missing:
        raise PostEvaluationCheckpointError(
            f"Indexed checkpoints are missing: {sorted(missing)}"
        )
    return checkpoints


def _write_index(
    root: Path,
    *,
    identity: Mapping[str, Any],
    checkpoints: Mapping[int, Mapping[str, Any]],
    checkpoint_io_runtime: float,
) -> float:
    started = time.perf_counter()
    entries = []
    for chunk_index in sorted(checkpoints):
        path = _checkpoint_path(root, chunk_index)
        entries.append(
            {
                "chunk_index": chunk_index,
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
                "scenario_count": len(checkpoints[chunk_index]["records"]),
            }
        )
    atomic_write_json(
        _index_path(root),
        {
            "schema_version": CHECKPOINT_INDEX_SCHEMA_VERSION,
            "identity_sha256": _canonical_sha256(identity),
            "chunks": entries,
            "checkpoint_io_runtime": checkpoint_io_runtime,
            "updated_at": utc_now_iso(),
        },
    )
    return time.perf_counter() - started


def _aggregate(
    instance: InventoryInstance,
    *,
    records: list[Mapping[str, Any]],
    y_values: list[float],
    x_values: list[list[float]],
    t_value: float,
    baseline_cost: float,
    rho: float,
    tolerance: float,
    wall_runtime: float,
) -> FairnessSolutionEvaluation:
    policies = [
        FairnessScenarioPolicy(**record["policy"])
        for record in records
        if record.get("policy") is not None
    ]
    errors = [
        str(record["error"]) for record in records if record.get("error") is not None
    ]
    evidence = [
        item
        for record in records
        for item in record.get("acceptance_evidence", [])
    ]
    acceptance_threshold, floating_point_slack = feasibility_acceptance_threshold(
        tolerance
    )
    inventory_by_warehouse = [
        math.fsum(float(x_values[i][j]) for j in instance.J) for i in instance.I
    ]
    inventory_by_product = [
        math.fsum(float(x_values[i][j]) for i in instance.I) for j in instance.J
    ]
    common = {
        "scenario_count": len(records),
        "opened_warehouses": sum(float(value) >= 0.5 for value in y_values),
        "total_inventory": float(math.fsum(inventory_by_warehouse)),
        "inventory_by_warehouse": [float(value) for value in inventory_by_warehouse],
        "inventory_by_product": [float(value) for value in inventory_by_product],
        "runtime": float(wall_runtime),
        "feasibility_tolerance": float(tolerance),
        "acceptance_threshold": acceptance_threshold,
        "floating_point_slack": floating_point_slack,
        "acceptance_evidence": evidence,
    }
    if errors or len(policies) != len(records):
        return FairnessSolutionEvaluation(
            valid=False,
            actual_robust_cost=None,
            actual_price_of_fairness=None,
            wgap=None,
            wminfr=None,
            realized_worst_shortage_rate=None,
            objective_t_consistent=None,
            wwd=None,
            minimum_weighted_mean_fill_rate=None,
            cost_worst_scenario=None,
            fairness_worst_scenario=None,
            errors=errors,
            **common,
        )
    budget = fairness_cost_budget(baseline_cost, rho)
    first_cost = first_stage_cost_value(instance, y_values, x_values)
    worst_recourse = max(policy.recourse_cost for policy in policies)
    actual_cost = math.fsum((first_cost, worst_recourse))
    actual_price = (
        0.0
        if budget.baseline_cost <= FAIRNESS_METRIC_TOLERANCE
        else actual_cost / budget.baseline_cost - 1.0
    )
    minima = [
        policy.minimum_fill_rate
        for policy in policies
        if policy.minimum_fill_rate is not None
    ]
    gaps = [policy.fill_rate_gap for policy in policies if policy.fill_rate_gap is not None]
    deviations = [
        policy.worst_region_deviation
        for policy in policies
        if policy.worst_region_deviation is not None
    ]
    means = [
        policy.weighted_mean_fill_rate
        for policy in policies
        if policy.weighted_mean_fill_rate is not None
    ]
    cost_worst = max(policies, key=lambda policy: policy.recourse_cost).scenario_name
    fairness_worst = min(
        (policy for policy in policies if policy.minimum_fill_rate is not None),
        key=lambda policy: float(policy.minimum_fill_rate),
        default=None,
    )
    wminfr = None if not minima else float(min(minima))
    realized = None if wminfr is None else 1.0 - wminfr
    objective_consistent = None
    if realized is not None:
        objective_evidence = constraint_acceptance_evidence(
            lhs=realized,
            rhs=float(t_value),
            tolerance=tolerance,
            constraint_type="objective_t_consistency",
            scenario_id=None if fairness_worst is None else fairness_worst.scenario_name,
            region_id=None,
        )
        evidence.append(objective_evidence)
        objective_consistent = bool(objective_evidence["accepted"])
    return FairnessSolutionEvaluation(
        valid=True,
        actual_robust_cost=float(actual_cost),
        actual_price_of_fairness=float(actual_price),
        wgap=None if not gaps else float(max(gaps)),
        wminfr=wminfr,
        realized_worst_shortage_rate=realized,
        objective_t_consistent=objective_consistent,
        wwd=None if not deviations else float(max(deviations)),
        minimum_weighted_mean_fill_rate=None if not means else float(min(means)),
        cost_worst_scenario=cost_worst,
        fairness_worst_scenario=(
            None if fairness_worst is None else fairness_worst.scenario_name
        ),
        errors=[],
        **common,
    )


def checkpointed_fairness_post_evaluation(
    instance: InventoryInstance,
    *,
    root: Path,
    run_key: str,
    config_sha256_value: str,
    git_commit: str,
    baseline_anchor_sha256: str,
    y_values: list[float],
    x_values: list[list[float]],
    t_value: float,
    baseline_cost: float,
    rho: float,
    gamma: int,
    max_scenarios: int,
    per_scenario_time_limit: float,
    tolerance: float,
    chunk_size: int,
    resume_count: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    failure_injector: Callable[[str, Mapping[str, Any]], None] | None = None,
    scenario_enumerator: Callable[..., Any] = enumerate_budget_scenarios_with_metadata,
    scenario_solver: Callable[..., FairnessScenarioPolicy] = solve_scenario_policy_with_shared_caps,
    output_flag: bool = False,
) -> tuple[FairnessSolutionEvaluation, PostEvaluationTiming]:
    if chunk_size <= 0:
        raise ValueError("post-evaluation checkpoint chunk size must be positive.")
    enumeration = scenario_enumerator(
        instance, gamma, max_scenarios=max_scenarios, exact_scenarios=True
    )
    scenarios = list(enumeration.scenarios)
    scenario_identities = [
        _scenario_identity(scenario, index) for index, scenario in enumerate(scenarios)
    ]
    if len({item["scenario_key"] for item in scenario_identities}) != len(scenarios):
        raise PostEvaluationCheckpointError("Duplicate deterministic scenario key.")
    solution_identity = {
        "y_values": y_values,
        "x_values": x_values,
        "t_value": float(t_value),
        "baseline_cost": float(baseline_cost),
        "rho": float(rho),
    }
    identity = {
        "schema_version": POST_EVALUATION_SCHEMA_VERSION,
        "execution_attempt": 4,
        "run_key": run_key,
        "config_sha256": config_sha256_value,
        "git_commit": git_commit,
        "baseline_anchor_sha256": baseline_anchor_sha256,
        "solution_sha256": config_sha256(solution_identity),
        "scenario_count": len(scenarios),
        "scenario_sequence_sha256": _canonical_sha256(scenario_identities),
        "chunk_size": int(chunk_size),
        "per_scenario_time_limit": float(per_scenario_time_limit),
    }
    root.mkdir(parents=True, exist_ok=True)
    final_path = _output_path(root)
    checkpoints = _load_checkpoints(
        root,
        identity=identity,
        scenario_identities=scenario_identities,
        chunk_size=chunk_size,
    )
    total_chunks = math.ceil(len(scenarios) / chunk_size)
    if final_path.exists():
        final = read_json(final_path)
        if (
            final is None
            or final.get("identity_sha256") != _canonical_sha256(identity)
            or len(checkpoints) != total_chunks
        ):
            raise PostEvaluationCheckpointError(
                "Final post-evaluation identity or checkpoint coverage drift."
            )
        evaluation = FairnessSolutionEvaluation(**final["evaluation"])
        return evaluation, PostEvaluationTiming(**final["timing"])
    existing_index = read_json(_index_path(root)) or {}
    checkpoint_io = float(existing_index.get("checkpoint_io_runtime", 0.0))
    if progress_callback is not None:
        completed_scenarios = sum(
            len(item["records"]) for item in checkpoints.values()
        )
        progress_callback(
            {
                "phase": "post_evaluation",
                "post_evaluation_total_scenarios": len(scenarios),
                "post_evaluation_completed_scenarios": completed_scenarios,
                "post_evaluation_pending_scenarios": len(scenarios)
                - completed_scenarios,
                "post_evaluation_failed_scenarios": sum(
                    record["error"] is not None
                    for item in checkpoints.values()
                    for record in item["records"]
                ),
                "current_chunk": (
                    None if not checkpoints else max(checkpoints)
                ),
                "total_chunks": math.ceil(len(scenarios) / chunk_size),
                "heartbeat_at": utc_now_iso(),
                "resume_count": int(resume_count),
            }
        )
    _inject(failure_injector, "before_first_chunk", {"run_key": run_key})
    first_cost = first_stage_cost_value(instance, y_values, x_values)
    budget = fairness_cost_budget(baseline_cost, rho)
    for chunk_index, start in enumerate(range(0, len(scenarios), chunk_size)):
        end = min(len(scenarios), start + chunk_size)
        if chunk_index in checkpoints:
            continue
        records: list[dict[str, Any]] = []
        chunk_started = time.perf_counter()
        for scenario_index in range(start, end):
            scenario = scenarios[scenario_index]
            scenario_identity = scenario_identities[scenario_index]
            acceptance_evidence: list[dict[str, Any]] = []
            policy_payload: dict[str, Any] | None = None
            error: str | None = None
            try:
                policy = scenario_solver(
                    instance,
                    scenario,
                    y_values=y_values,
                    x_values=x_values,
                    t_value=t_value,
                    cost_budget_value=budget.budget,
                    feasibility_tolerance=tolerance,
                    time_limit=per_scenario_time_limit,
                    output_flag=output_flag,
                )
                cost_evidence = constraint_acceptance_evidence(
                    lhs=math.fsum((first_cost, policy.recourse_cost)),
                    rhs=budget.budget,
                    tolerance=tolerance,
                    constraint_type="robust_cost_budget",
                    scenario_id=scenario.name,
                    region_id=None,
                )
                acceptance_evidence.append(cost_evidence)
                if not cost_evidence["accepted"]:
                    raise RuntimeError(
                        "Recovered policy exceeds the shared robust cost budget."
                    )
                for region_id, (shortage, demand) in enumerate(
                    zip(policy.regional_shortage, policy.regional_demand)
                ):
                    if float(demand) <= FAIRNESS_METRIC_TOLERANCE:
                        continue
                    fairness_evidence = constraint_acceptance_evidence(
                        lhs=float(shortage),
                        rhs=float(t_value) * float(demand),
                        tolerance=tolerance,
                        constraint_type="regional_shortage_rate_cap",
                        scenario_id=scenario.name,
                        region_id=region_id,
                    )
                    acceptance_evidence.append(fairness_evidence)
                    if not fairness_evidence["accepted"]:
                        raise RuntimeError(
                            "Recovered policy violates the regional max-shortage-rate cap."
                        )
                policy_payload = asdict(policy)
            except Exception as exc:  # noqa: BLE001 - failure must remain explicit.
                error = f"{scenario.name}: {type(exc).__name__}: {exc}"
            records.append(
                {
                    **scenario_identity,
                    "policy": policy_payload,
                    "acceptance_evidence": acceptance_evidence,
                    "error": error,
                }
            )
            _inject(
                failure_injector,
                "after_scenario",
                {"chunk_index": chunk_index, "scenario_index": scenario_index},
            )
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "identity_sha256": _canonical_sha256(identity),
            "chunk_index": chunk_index,
            "scenario_start": start,
            "scenario_end_exclusive": end,
            "records": records,
            "wall_runtime": time.perf_counter() - chunk_started,
            "solver_runtime": math.fsum(
                float(record["policy"].get("solver_runtime", 0.0))
                for record in records
                if record["policy"] is not None
            ),
            "completed_at": utc_now_iso(),
        }
        io_started = time.perf_counter()
        atomic_write_json(_checkpoint_path(root, chunk_index), checkpoint)
        checkpoint_io += time.perf_counter() - io_started
        checkpoints[chunk_index] = checkpoint
        _inject(
            failure_injector,
            "after_chunk_commit_before_index",
            {"chunk_index": chunk_index},
        )
        checkpoint_io += _write_index(
            root,
            identity=identity,
            checkpoints=checkpoints,
            checkpoint_io_runtime=checkpoint_io,
        )
        completed_scenarios = sum(len(item["records"]) for item in checkpoints.values())
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "post_evaluation",
                    "post_evaluation_total_scenarios": len(scenarios),
                    "post_evaluation_completed_scenarios": completed_scenarios,
                    "post_evaluation_pending_scenarios": len(scenarios)
                    - completed_scenarios,
                    "post_evaluation_failed_scenarios": sum(
                        record["error"] is not None
                        for item in checkpoints.values()
                        for record in item["records"]
                    ),
                    "current_chunk": chunk_index,
                    "total_chunks": total_chunks,
                    "heartbeat_at": utc_now_iso(),
                    "resume_count": int(resume_count),
                }
            )
    _inject(failure_injector, "after_last_chunk_before_aggregation", {})
    if len(checkpoints) != total_chunks:
        raise PostEvaluationCheckpointError("Not all post-evaluation chunks completed.")
    all_records = [
        record
        for chunk_index in sorted(checkpoints)
        for record in checkpoints[chunk_index]["records"]
    ]
    if len(all_records) != len(scenarios) or len(
        {record["scenario_key"] for record in all_records}
    ) != len(scenarios):
        raise PostEvaluationCheckpointError(
            "Post-evaluation scenarios are missing or duplicated."
        )
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "aggregation",
                "post_evaluation_total_scenarios": len(scenarios),
                "post_evaluation_completed_scenarios": len(scenarios),
                "post_evaluation_pending_scenarios": 0,
                "post_evaluation_failed_scenarios": sum(
                    record["error"] is not None for record in all_records
                ),
                "current_chunk": total_chunks - 1,
                "total_chunks": total_chunks,
                "heartbeat_at": utc_now_iso(),
                "resume_count": int(resume_count),
            }
        )
    aggregation_started = time.perf_counter()
    evaluation = _aggregate(
        instance,
        records=all_records,
        y_values=y_values,
        x_values=x_values,
        t_value=t_value,
        baseline_cost=baseline_cost,
        rho=rho,
        tolerance=tolerance,
        wall_runtime=math.fsum(
            float(checkpoint["wall_runtime"]) for checkpoint in checkpoints.values()
        ),
    )
    aggregation_runtime = time.perf_counter() - aggregation_started
    timing = PostEvaluationTiming(
        solver_runtime=math.fsum(
            float(checkpoint["solver_runtime"]) for checkpoint in checkpoints.values()
        ),
        wall_runtime=math.fsum(
            float(checkpoint["wall_runtime"]) for checkpoint in checkpoints.values()
        ),
        aggregation_runtime=aggregation_runtime,
        checkpoint_io_runtime=checkpoint_io,
        resume_count=int(resume_count),
    )
    _inject(failure_injector, "before_final_output", {})
    atomic_write_json(
        final_path,
        {
            "schema_version": POST_EVALUATION_SCHEMA_VERSION,
            "identity": identity,
            "identity_sha256": _canonical_sha256(identity),
            "evaluation": evaluation.to_dict(),
            "timing": asdict(timing),
            "completed_at": utc_now_iso(),
        },
    )
    _inject(failure_injector, "after_final_output", {})
    return evaluation, timing
