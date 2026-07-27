from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import gurobipy as gp
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
    stable_run_key,
    utc_now_iso,
)
from .experiment_suite import _base_config
from .fairness_benders import (
    _baseline_method_config,
    _certified_baseline_anchor,
    fairness_frontier_overall_status,
    solve_fairness_benders,
)
from .fairness_post_evaluation import checkpointed_fairness_post_evaluation
from .fairness_scalability import SCALABILITY_CANDIDATES
from .fairness_scalability_results_audit import (
    CANONICALIZATION,
    RESULT_FIELDS,
    aggregate_records,
    resolved_config_file_bytes,
)
from .instance import InventoryInstance, generate_instance
from .regional_fairness_pipeline import SingleWriterLock


SCALABILITY_MANIFEST_SCHEMA_VERSION = 2
SCALABILITY_EXECUTION_ATTEMPT = 2
RUN_DIRECTORY_HASH_HEX_LENGTH = 24
WINDOWS_PORTABLE_PATH_LIMIT = 220
PRIOR_ATTEMPTS = (
    {
        "attempt": 1,
        "stage": "scalability_s1_medium_large",
        "git_commit": "22ce2d63a4ad8cea021bf2b6cbe60273c0c2919c",
        "status": "execution_incomplete",
        "scientifically_usable_for_candidate_selection": False,
        "results_reused": False,
        "seeds_accessed": [160],
        "failure_class": "windows_path_length_pipeline_defect",
    },
)
STAGES = ("s1", "s2", "full-grid")
STAGE_ORDER = {name: index for index, name in enumerate(STAGES)}
S1_SEEDS = [160, 161, 162]
S2_SEEDS = list(range(160, 170))
SCREEN_RHOS = [0.0, 0.01]
FULL_GRID_ADDITIONAL_RHOS = [0.025, 0.05, 0.10]
PUBLIC_STATUSES = (
    "certified_robust_optimal",
    "master_optimal_but_robust_uncertified",
    "time_limit_uncertified",
    "infeasible",
    "invalid_post_evaluation",
    "implementation_error",
    "interrupted",
    "iteration_limit_uncertified",
    "numerical_uncertified",
    "unknown_uncertified",
)


@dataclass(frozen=True)
class ScalabilityRunSpec:
    run_key: str
    introduced_stage: str
    task_type: str
    scale: str
    seed: int
    rho: float | None
    candidate: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScalabilityDependencies:
    generate_instance: Callable[..., InventoryInstance] = generate_instance
    solve_baseline: Callable[..., dict[str, Any]] | None = None
    solve_frontier: Callable[..., dict[str, Any]] | None = None
    post_evaluate: Callable[..., tuple[dict[str, Any], dict[str, float]]] | None = None
    configure_solver: Callable[[Mapping[str, Any]], None] | None = None


def run_directory_id(run_key: str) -> str:
    """Return the stable physical directory id for a canonical scientific key."""
    digest = hashlib.sha256(str(run_key).encode("utf-8")).hexdigest()
    return f"r_{digest[:RUN_DIRECTORY_HASH_HEX_LENGTH]}"


def build_run_directory_mapping(
    run_keys: list[str] | tuple[str, ...],
    *,
    directory_id_factory: Callable[[str], str] = run_directory_id,
) -> tuple[dict[str, str], dict[str, str]]:
    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for key in run_keys:
        directory_id = str(directory_id_factory(key))
        if not directory_id.startswith("r_") or len(directory_id) != 2 + RUN_DIRECTORY_HASH_HEX_LENGTH:
            raise ValueError("Invalid scalability run directory id.")
        previous = reverse.get(directory_id)
        if previous is not None and previous != key:
            raise ValueError(
                f"Scalability run-directory hash collision: {directory_id} maps to multiple run keys."
            )
        if key in forward and forward[key] != directory_id:
            raise ValueError(f"Scalability run key has inconsistent directory mapping: {key}")
        forward[key] = directory_id
        reverse[directory_id] = key
    if len(forward) != len(run_keys):
        raise ValueError("Duplicate canonical scalability run key.")
    return forward, reverse


def _run_directory(output_dir: Path, run_key: str) -> Path:
    return output_dir / "runs" / run_directory_id(run_key)


def _atomic_temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def path_portability_report(
    output_dir: Path,
    specs: list[ScalabilityRunSpec],
    *,
    scenario_count: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Enumerate every frozen output path, including atomic temporary names."""
    if scenario_count <= 0 or chunk_size <= 0:
        raise ValueError("Scenario count and checkpoint chunk size must be positive.")
    forward, reverse = build_run_directory_mapping([spec.run_key for spec in specs])
    root = output_dir.resolve(strict=False)
    candidates: list[tuple[str, Path]] = []

    def add(path_type: str, path: Path, *, atomic: bool = False) -> None:
        candidates.append((path_type, path))
        if atomic:
            candidates.append((f"{path_type}_atomic_tmp", _atomic_temporary_path(path)))

    for name, path_type in (
        ("scalability_development_manifest.json", "scalability_manifest"),
        ("run_manifest.json", "run_manifest"),
        ("resolved_config.yaml", "resolved_config"),
        ("results.csv", "results_csv"),
        ("summary.csv", "summary_csv"),
        ("audit_log.json", "audit_log"),
    ):
        add(path_type, root / name, atomic=True)
    for seed in sorted({spec.seed for spec in specs}):
        add("instance", root / "instances" / f"{seed}.json", atomic=True)
    total_chunks = math.ceil(scenario_count / chunk_size)
    last_chunk = max(0, total_chunks - 1)
    for spec in specs:
        run_root = root / "runs" / forward[spec.run_key]
        add("run_json", run_root / "run.json", atomic=True)
        add("status_json", run_root / "status.json", atomic=True)
        if spec.task_type == "frontier":
            add("algorithm_checkpoint", run_root / "algorithm_checkpoint.json", atomic=True)
            post_root = run_root / "post_evaluation"
            add(
                "post_evaluation_chunk",
                post_root / "checkpoint" / f"chunk_{last_chunk:05d}.json",
                atomic=True,
            )
            add("post_evaluation_index", post_root / "checkpoint" / "index.json", atomic=True)
            add("post_evaluation_final", post_root / "post_evaluation.json", atomic=True)
    longest_type, longest_path = max(candidates, key=lambda item: len(str(item[1])))
    maximum = len(str(longest_path))
    return {
        "windows_portable_path_limit": WINDOWS_PORTABLE_PATH_LIMIT,
        "max_absolute_path_length": maximum,
        "longest_path_type": longest_type,
        "longest_path": str(longest_path),
        "windows_portability_check": maximum <= WINDOWS_PORTABLE_PATH_LIMIT,
        "run_key_to_directory_id": forward,
        "directory_id_to_run_key": reverse,
        "atomic_temporary_paths_checked": True,
    }


def assert_windows_portable_paths(report: Mapping[str, Any]) -> None:
    if report.get("windows_portability_check") is not True:
        raise ValueError(
            "Scalability output path is not Windows-portable: "
            f"{report.get('max_absolute_path_length')} > {WINDOWS_PORTABLE_PATH_LIMIT} "
            f"for {report.get('longest_path_type')}: {report.get('longest_path')}"
        )


def _stage_introduced(seed: int) -> str:
    return "s1" if int(seed) in S1_SEEDS else "s2"


def _run_key(
    config: Mapping[str, Any], *, task_type: str, seed: int, rho: float | None, candidate: str
) -> str:
    scale = str(config["instance_sizes"][0])
    introduced_stage = (
        "full-grid"
        if task_type == "frontier" and float(rho) in FULL_GRID_ADDITIONAL_RHOS
        else _stage_introduced(seed)
    )
    stage_value = "baseline" if task_type == "baseline" else f"rho={float(rho):.12g}"
    return stable_run_key(
        experiment_name=str(config["experiment_name"]),
        sensitivity_axis=f"scalability:{introduced_stage}:{task_type}",
        sensitivity_value=stage_value,
        instance_size=scale,
        seed=int(seed),
        variant_name=str(candidate),
    )


def cumulative_run_plan(
    config: Mapping[str, Any], stage: str, *, selected_candidate: str | None = None
) -> list[ScalabilityRunSpec]:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}")
    scale = str(config["instance_sizes"][0])
    candidates = tuple(str(value) for value in config["scalability_candidates"])
    if candidates != SCALABILITY_CANDIDATES:
        raise ValueError("Scalability candidate identity drift.")
    seeds = S1_SEEDS if stage == "s1" else S2_SEEDS
    specs: list[ScalabilityRunSpec] = []
    for seed in seeds:
        introduced = _stage_introduced(seed)
        specs.append(
            ScalabilityRunSpec(
                run_key=_run_key(
                    config,
                    task_type="baseline",
                    seed=seed,
                    rho=None,
                    candidate="joint_v1_core_point_strengthened",
                ),
                introduced_stage=introduced,
                task_type="baseline",
                scale=scale,
                seed=seed,
                rho=None,
                candidate="joint_v1_core_point_strengthened",
            )
        )
        for rho in SCREEN_RHOS:
            for candidate in candidates:
                specs.append(
                    ScalabilityRunSpec(
                        run_key=_run_key(
                            config,
                            task_type="frontier",
                            seed=seed,
                            rho=rho,
                            candidate=candidate,
                        ),
                        introduced_stage=introduced,
                        task_type="frontier",
                        scale=scale,
                        seed=seed,
                        rho=rho,
                        candidate=candidate,
                    )
                )
    if stage == "full-grid":
        if selected_candidate not in candidates:
            raise ValueError("Full-grid requires exactly one frozen candidate.")
        for seed in S2_SEEDS:
            for rho in FULL_GRID_ADDITIONAL_RHOS:
                specs.append(
                    ScalabilityRunSpec(
                        run_key=_run_key(
                            config,
                            task_type="frontier",
                            seed=seed,
                            rho=rho,
                            candidate=str(selected_candidate),
                        ),
                        introduced_stage="full-grid",
                        task_type="frontier",
                        scale=scale,
                        seed=seed,
                        rho=rho,
                        candidate=str(selected_candidate),
                    )
                )
    keys = [spec.run_key for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate scalability run key.")
    expected = {"s1": 27, "s2": 90, "full-grid": 120}[stage]
    if len(specs) != expected:
        raise ValueError(f"{stage} plan must contain {expected} unique tasks.")
    return specs


def stage_new_specs(specs: list[ScalabilityRunSpec], stage: str) -> list[ScalabilityRunSpec]:
    return [spec for spec in specs if spec.introduced_stage == stage]


def _strict_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping.")
    return value


def validate_stage_decision(
    stage: str, decision_path: str | Path | None
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if stage == "s1":
        if decision_path is not None:
            raise ValueError("S1 must not consume a prior-stage decision.")
        return None, None, None
    if decision_path is None:
        raise ValueError(f"{stage} requires a frozen prior-stage decision file.")
    path = Path(decision_path)
    decision = _strict_yaml(path)
    digest = file_sha256(path).upper()
    if stage == "s2":
        if (
            decision.get("decision") != "s1_pass"
            or decision.get("next_authorized_stage") != "s2"
            or decision.get("mathematical_and_certification_correctness") is not True
        ):
            raise ValueError("S2 requires a correctness-approved frozen S1 decision.")
        return decision, digest, None
    if (
        decision.get("decision") != "s2_candidate_selected"
        or decision.get("next_authorized_stage") != "full-grid"
        or decision.get("mathematical_and_certification_correctness") is not True
    ):
        raise ValueError("Full-grid requires a correctness-approved frozen S2 decision.")
    scale_results = decision.get("scale_results", {})
    for scale in ("medium_large", "large"):
        result = scale_results.get(scale, {})
        if int(result.get("certified_solved_count", -1)) < 16 or int(
            result.get("denominator", -1)
        ) != 20:
            raise ValueError("Full-grid requires at least 16/20 certified runs per scale.")
    selected = decision.get("selected_candidate")
    if selected not in SCALABILITY_CANDIDATES:
        raise ValueError("S2 decision must freeze exactly one known candidate.")
    if decision.get("selection_order") != [
        "mathematical_and_certification_correctness",
        "certified_solved_count_descending",
        "par2_ascending",
        "separation_runtime_ascending",
        "total_wall_runtime_ascending",
    ]:
        raise ValueError("S2 candidate was not selected by the preregistered order.")
    candidate_path = Path(str(decision.get("selected_candidate_config", "")))
    expected_hash = str(decision.get("selected_candidate_config_sha256", "")).upper()
    if not candidate_path.is_file() or file_sha256(candidate_path).upper() != expected_hash:
        raise ValueError("Frozen selected scalability candidate identity mismatch.")
    selected_payload = _strict_yaml(candidate_path)
    if selected_payload.get("selected_candidate") != selected:
        raise ValueError("Selected candidate config disagrees with S2 decision.")
    return decision, digest, str(selected)


def solver_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(config.get("gurobi_parameters", {}))
    expected = {"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7}
    if value != expected:
        raise ValueError("Frozen Gurobi settings drifted.")
    return expected


def post_evaluation_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(config.get("post_evaluation", {}))
    expected = {
        "enabled": True,
        "exact_scenarios": True,
        "max_scenarios": 5000,
        "time_limit_per_scenario": 30.0,
        "checkpoint_chunk_size": 25,
        "feasibility_tolerance": 1.0e-7,
    }
    if value != expected:
        raise ValueError("Frozen post-evaluation settings drifted.")
    return expected


def validate_runtime_config(config: Mapping[str, Any]) -> None:
    """Reject scientific or execution-identity drift before any artifact is made."""
    if int(config.get("execution_attempt", -1)) != SCALABILITY_EXECUTION_ATTEMPT:
        raise ValueError("Frozen scalability execution attempt drifted.")
    if config.get("previous_attempt_results_reused") is not False:
        raise ValueError("Prior scalability attempt results must not be reused.")
    if config.get("prior_attempts") != [dict(item) for item in PRIOR_ATTEMPTS]:
        raise ValueError("Frozen scalability prior-attempt history drifted.")
    if list(config.get("development_seeds", [])) != S2_SEEDS:
        raise ValueError("Frozen scalability development seeds drifted.")
    if list(config.get("s1_seeds", [])) != S1_SEEDS:
        raise ValueError("Frozen S1 seeds drifted.")
    if [float(v) for v in config.get("s1_rho_grid", [])] != SCREEN_RHOS:
        raise ValueError("Frozen S1 rho grid drifted.")
    if [float(v) for v in config.get("s2_rho_grid", [])] != SCREEN_RHOS:
        raise ValueError("Frozen S2 rho grid drifted.")
    if [float(v) for v in config.get("rho_grid", [])] != [
        *SCREEN_RHOS,
        *FULL_GRID_ADDITIONAL_RHOS,
    ]:
        raise ValueError("Frozen full rho grid drifted.")
    if tuple(config.get("scalability_candidates", [])) != SCALABILITY_CANDIDATES:
        raise ValueError("Frozen scalability candidates drifted.")
    if any(float(config.get(key, math.nan)) != 1800.0 for key in (
        "time_limit", "baseline_time_limit", "fairness_time_limit"
    )):
        raise ValueError("Frozen algorithm time limits drifted.")
    if (
        int(config.get("gamma_target", -1)) != 2
        or list(config.get("gamma_schedule", [])) != [2]
        or config.get("gamma_continuation_enabled") is not False
    ):
        raise ValueError("Frozen uncertainty budget drifted.")
    if float(config.get("tol", math.nan)) != 1.0e-4:
        raise ValueError("Frozen optimality tolerance drifted.")
    solver_identity(config)
    post_evaluation_identity(config)
    path_config = dict(config.get("path_portability", {}))
    if path_config != {
        "windows_max_absolute_path_length": WINDOWS_PORTABLE_PATH_LIMIT,
        "run_directory_hash_hex_length": RUN_DIRECTORY_HASH_HEX_LENGTH,
        "atomic_temporary_suffix": ".tmp",
    }:
        raise ValueError("Frozen scalability path-portability settings drifted.")


def execution_identity(
    config: Mapping[str, Any], *, config_path: Path, stage: str, commit: str,
    decision_sha256: str | None, selected_candidate: str | None,
) -> dict[str, Any]:
    validate_runtime_config(config)
    protocol_path = Path(str(config["protocol_document"]))
    candidate_path = Path(str(config["candidate_parameters_must_be_fixed_from"]))
    return {
        "schema_version": SCALABILITY_MANIFEST_SCHEMA_VERSION,
        "execution_attempt": SCALABILITY_EXECUTION_ATTEMPT,
        "prior_attempts": [dict(item) for item in PRIOR_ATTEMPTS],
        "previous_attempt_results_reused": False,
        "experiment_name": str(config["experiment_name"]),
        "scale": str(config["instance_sizes"][0]),
        "git_commit": commit,
        "config_path": config_path.as_posix(),
        "config_file_sha256": file_sha256(config_path).upper(),
        "resolved_config_file_sha256": hashlib.sha256(
            resolved_config_file_bytes(config)
        ).hexdigest().upper(),
        "resolved_config_canonical_sha256": config_sha256(dict(config)).upper(),
        "resolved_config_canonicalization": CANONICALIZATION,
        "protocol_path": protocol_path.as_posix(),
        "protocol_sha256": file_sha256(protocol_path).upper(),
        "candidate_config_sha256": file_sha256(candidate_path).upper(),
        "candidate_definitions": list(SCALABILITY_CANDIDATES),
        "gurobi_parameters": solver_identity(config),
        "baseline_time_limit": float(config["baseline_time_limit"]),
        "fairness_time_limit": float(config["fairness_time_limit"]),
        "post_evaluation": post_evaluation_identity(config),
        "par2": {"multiplier": 2, "basis": "algorithm_runtime"},
        "runtime_semantics": dict(config["runtime_semantics"]),
        "requested_stage": stage,
        "prior_stage_decision_sha256": decision_sha256,
        "selected_candidate": selected_candidate,
    }


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "scalability_development_manifest.json"


def _run_manifest_path(output_dir: Path) -> Path:
    return output_dir / "run_manifest.json"


def _record_path(output_dir: Path, run_key: str) -> Path:
    return _run_directory(output_dir, run_key) / "run.json"


def _status_path(output_dir: Path, run_key: str) -> Path:
    return _run_directory(output_dir, run_key) / "status.json"


def _algorithm_checkpoint_path(output_dir: Path, run_key: str) -> Path:
    return _run_directory(output_dir, run_key) / "algorithm_checkpoint.json"


def _write_scalability_run_state(
    output_dir: Path,
    run_key: str,
    *,
    state: str,
    details: Mapping[str, Any] | None = None,
) -> Path:
    _run_directory(output_dir, run_key).mkdir(parents=True, exist_ok=True)
    payload = {
        "run_key": run_key,
        "run_directory_id": run_directory_id(run_key),
        "state": state,
        "updated_at": utc_now_iso(),
    }
    payload.update(dict(details or {}))
    return atomic_write_json(_status_path(output_dir, run_key), payload)


def _read_record(output_dir: Path, run_key: str) -> dict[str, Any] | None:
    path = _record_path(output_dir, run_key)
    value = read_json(path)
    if path.exists() and value is None:
        raise ValueError(f"Corrupt run record: {run_key}")
    return value


def _write_record(output_dir: Path, spec: ScalabilityRunSpec, record: dict[str, Any]) -> None:
    record = {**record, "run_key": spec.run_key, "run_directory_id": run_directory_id(spec.run_key)}
    atomic_write_json(_record_path(output_dir, spec.run_key), record)
    _write_scalability_run_state(
        output_dir,
        spec.run_key,
        state=str(record["state"]),
        details={
            "scientific_status": record["scientific_status"],
            "algorithm_status": record.get("algorithm_status"),
            "solved_to_tolerance": record["scientific_status"] == "certified_robust_optimal",
            "task_type": spec.task_type,
            "seed": spec.seed,
            "rho": spec.rho,
            "candidate": spec.candidate,
        },
    )


def _refresh_manifest(
    output_dir: Path, *, identity: dict[str, Any], specs: list[ScalabilityRunSpec],
    new_specs: list[ScalabilityRunSpec], anchors: Mapping[str, Any], resume_count: int,
) -> dict[str, Any]:
    records = [_read_record(output_dir, spec.run_key) for spec in specs]
    completed = sum(record is not None and record.get("state") == "complete" for record in records)
    solved = sum(record is not None and record.get("scientific_status") == "certified_robust_optimal" for record in records)
    failed = sum(
        record is not None
        and record.get("scientific_status") in {"implementation_error", "invalid_post_evaluation", "interrupted"}
        for record in records
    )
    previous = read_json(_manifest_path(output_dir)) or {}
    run_key_to_directory_id, directory_id_to_run_key = build_run_directory_mapping(
        [spec.run_key for spec in specs]
    )
    payload = {
        **identity,
        "authorized_cumulative_stage": identity["requested_stage"],
        "expected_run_count": len(specs),
        "new_run_count": len(new_specs),
        "completed_run_count": completed,
        "pending_run_count": len(specs) - completed,
        "failed_run_count": failed,
        "solved_run_count": solved,
        "seeds": sorted({spec.seed for spec in specs}),
        "rhos": sorted({spec.rho for spec in specs if spec.rho is not None}),
        "candidates": list(SCALABILITY_CANDIDATES),
        "allowed_candidate_differences": {
            "single_cut": {"persistent": False, "cache": False, "pool": False, "max_cuts": 1},
            "persistent_separation": {"persistent": True, "cache": False, "pool": False, "max_cuts": 1},
            "persistent_certified_cache": {"persistent": True, "cache": True, "pool": False, "max_cuts": 1},
            "persistent_certified_cache_batch5": {"persistent": True, "cache": True, "pool": True, "max_cuts": 5},
        },
        "run_specs": [spec.to_dict() for spec in specs],
        "run_key_to_directory_id": run_key_to_directory_id,
        "directory_id_to_run_key": directory_id_to_run_key,
        "baseline_anchors": dict(anchors),
        "public_scientific_statuses": list(PUBLIC_STATUSES),
        "resume_count": int(resume_count),
        "created_at": previous.get("created_at", utc_now_iso()),
        "updated_at": utc_now_iso(),
    }
    atomic_write_json(_manifest_path(output_dir), payload)
    atomic_write_json(
        _run_manifest_path(output_dir),
        {
            key: payload[key]
            for key in (
                "schema_version",
                "execution_attempt",
                "experiment_name",
                "scale",
                "git_commit",
                "config_file_sha256",
                "resolved_config_file_sha256",
                "resolved_config_canonical_sha256",
                "resolved_config_canonicalization",
                "protocol_sha256",
                "candidate_config_sha256",
                "authorized_cumulative_stage",
                "expected_run_count",
                "new_run_count",
                "completed_run_count",
                "pending_run_count",
                "failed_run_count",
                "solved_run_count",
                "resume_count",
                "created_at",
                "updated_at",
            )
        }
        | {
            "previous_attempt_results_reused": False,
            "prior_attempts": [dict(item) for item in PRIOR_ATTEMPTS],
            "run_key_to_directory_id": run_key_to_directory_id,
            "directory_id_to_run_key": directory_id_to_run_key,
            "path_portability_report": identity["path_portability_report"],
        },
    )
    return payload


def _aggregate(output_dir: Path, specs: list[ScalabilityRunSpec]) -> None:
    start = time.perf_counter()
    records = [record for spec in specs if (record := _read_record(output_dir, spec.run_key))]
    rows, summary = aggregate_records(
        records,
        [spec.to_dict() for spec in specs],
        time_limit=1800.0,
    )
    atomic_write_csv(output_dir / "results.csv", rows, RESULT_FIELDS)
    atomic_write_csv(output_dir / "summary.csv", summary, list(summary[0]) if summary else [])
    atomic_write_json(
        output_dir / "audit_log.json",
        {
            "aggregation_runtime": time.perf_counter() - start,
            "unique_run_keys": len(rows) == len({row["run_key"] for row in rows}),
            "record_count": len(rows),
            "updated_at": utc_now_iso(),
        },
    )


def _production_baseline(
    config: dict[str, Any], instance: InventoryInstance, *, scale: str, seed: int
) -> dict[str, Any]:
    resolved = deepcopy(config)
    # Explicit wiring; all frozen configs currently set both values to 1800.
    resolved["time_limit"] = float(config["baseline_time_limit"])
    method, method_config = _baseline_method_config(resolved, scale, seed)
    result = solve_benders(method_config, instance, method)
    payload = result.summary_dict()
    payload["iteration_log"] = result.iteration_log
    return payload


def _production_frontier(
    config: dict[str, Any], instance: InventoryInstance, *, anchor: float,
    rho: float, candidate: str,
) -> dict[str, Any]:
    candidate_config = _strict_yaml(Path(str(config["candidate_parameters_must_be_fixed_from"])))
    algorithm = deepcopy(candidate_config["algorithm"])
    algorithm["fairness_scalability_strategy"] = candidate
    result = solve_fairness_benders(
        instance,
        baseline_cost=anchor,
        rho=rho,
        gamma=int(config["gamma_target"]),
        algorithm_config=algorithm,
        max_iterations=int(config["max_iterations"]),
        time_limit=float(config["fairness_time_limit"]),
        tol=float(config["tol"]),
        feasibility_tolerance=float(config["post_evaluation"]["feasibility_tolerance"]),
        output_flag=False,
    )
    return result.to_dict()


def _production_post_evaluate(
    config: dict[str, Any], instance: InventoryInstance, *, output_dir: Path,
    spec: ScalabilityRunSpec, payload: dict[str, Any], anchor: dict[str, Any],
    identity: dict[str, Any], resume_count: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    run_root = _run_directory(output_dir, spec.run_key)
    post_root = run_root / "post_evaluation"
    checkpoint_root = post_root / "checkpoint"
    run_root.mkdir(parents=True, exist_ok=True)
    post_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    evaluation, timing = checkpointed_fairness_post_evaluation(
        instance,
        root=post_root,
        run_key=spec.run_key,
        config_sha256_value=identity["resolved_config_canonical_sha256"],
        git_commit=identity["git_commit"],
        baseline_anchor_sha256=anchor["anchor_sha256"],
        y_values=payload["y_values"],
        x_values=payload["x_values"],
        t_value=float(payload["objective_t"]),
        baseline_cost=float(anchor["value"]),
        rho=float(spec.rho),
        gamma=int(config["gamma_target"]),
        max_scenarios=int(config["post_evaluation"]["max_scenarios"]),
        per_scenario_time_limit=float(config["post_evaluation"]["time_limit_per_scenario"]),
        tolerance=float(config["post_evaluation"]["feasibility_tolerance"]),
        chunk_size=int(config["post_evaluation"]["checkpoint_chunk_size"]),
        resume_count=resume_count,
        output_flag=False,
    )
    return evaluation.to_dict(), {
        "post_evaluation_solver_runtime": timing.solver_runtime,
        "post_evaluation_wall_runtime": timing.wall_runtime,
        "aggregation_runtime": timing.aggregation_runtime,
        "checkpoint_io_runtime": timing.checkpoint_io_runtime,
    }


def _configure_gurobi(settings: Mapping[str, Any]) -> None:
    gp.setParam("Threads", int(settings["Threads"]))
    gp.setParam("Seed", int(settings["Seed"]))
    gp.setParam("FeasibilityTol", float(settings["FeasibilityTol"]))


def _scientific_status(
    payload: Mapping[str, Any], post_valid: bool | None, *, tolerance: float
) -> str:
    status = str(payload.get("status", "unknown"))
    gap = payload.get("gap")
    algorithm_solved = status == "optimal" and gap is not None and float(gap) <= tolerance
    if status == "infeasible":
        return "infeasible"
    mapped = fairness_frontier_overall_status(
        algorithm_status=status,
        algorithm_solved=algorithm_solved,
        post_evaluation_attempted=post_valid is not None,
        post_evaluation_valid=bool(post_valid),
    )
    return mapped if mapped in PUBLIC_STATUSES else "master_optimal_but_robust_uncertified"


def run_scalability_stage(
    config: dict[str, Any], *, config_path: Path, stage: str, resume: bool,
    decision_path: str | Path | None = None,
    dependencies: ScalabilityDependencies | None = None,
    failure_injector: Callable[[str, ScalabilityRunSpec], None] | None = None,
) -> Path:
    if not resume:
        raise ValueError("Formal scalability execution requires --resume; --overwrite is unsupported.")
    validate_runtime_config(config)
    _decision, decision_sha, selected = validate_stage_decision(stage, decision_path)
    specs = cumulative_run_plan(config, stage, selected_candidate=selected)
    if len({spec.run_key for spec in specs}) != len(specs):
        raise ValueError("Duplicate scalability run key in executable plan.")
    new_specs = stage_new_specs(specs, stage)
    output_dir = Path(str(config["output_dir"]))
    current_commit = git_commit(Path(__file__).resolve().parents[1])
    identity = execution_identity(
        config,
        config_path=config_path,
        stage=stage,
        commit=current_commit,
        decision_sha256=decision_sha,
        selected_candidate=selected,
    )
    scenario_count = int(config["post_evaluation"]["max_scenarios"])
    # The exact frozen counts are smaller than max_scenarios. Using the cap is
    # conservative for filename width and does not enumerate or generate an instance.
    portability_report = path_portability_report(
        output_dir,
        specs,
        scenario_count=scenario_count,
        chunk_size=int(config["post_evaluation"]["checkpoint_chunk_size"]),
    )
    assert_windows_portable_paths(portability_report)
    identity["path_portability_report"] = portability_report
    existing = read_json(_manifest_path(output_dir))
    if output_dir.exists() and existing is None:
        raise ValueError("Existing output lacks a valid scalability identity manifest.")
    if existing is not None:
        for field in (
            "schema_version", "experiment_name", "scale", "git_commit",
            "config_file_sha256", "resolved_config_file_sha256",
            "resolved_config_canonical_sha256", "resolved_config_canonicalization",
            "protocol_sha256",
            "candidate_config_sha256", "candidate_definitions", "gurobi_parameters",
            "baseline_time_limit", "fairness_time_limit", "post_evaluation", "par2",
            "execution_attempt", "prior_attempts", "previous_attempt_results_reused",
        ):
            if existing.get(field) != identity.get(field):
                raise ValueError(f"Scalability resume identity mismatch: {field}")
        previous_stage = str(existing.get("authorized_cumulative_stage"))
        prior_mapping = existing.get("run_key_to_directory_id")
        prior_reverse = existing.get("directory_id_to_run_key")
        prior_keys = [str(value.get("run_key")) for value in existing.get("run_specs", [])]
        expected_prior_mapping, expected_prior_reverse = build_run_directory_mapping(prior_keys)
        if prior_mapping != expected_prior_mapping or prior_reverse != expected_prior_reverse:
            raise ValueError("Scalability run-directory mapping identity mismatch.")
        if previous_stage == stage:
            for field in (
                "requested_stage",
                "prior_stage_decision_sha256",
                "selected_candidate",
            ):
                if existing.get(field) != identity.get(field):
                    raise ValueError(f"Scalability stage identity mismatch: {field}")
            if existing.get("path_portability_report") != portability_report:
                raise ValueError("Scalability path-portability identity mismatch.")
        if STAGE_ORDER[stage] < STAGE_ORDER[previous_stage]:
            raise ValueError("Cannot resume an earlier stage over a later-stage output.")
        if STAGE_ORDER[stage] > STAGE_ORDER[previous_stage] + 1:
            raise ValueError("Scalability stages cannot be skipped.")
        if STAGE_ORDER[stage] > STAGE_ORDER[previous_stage]:
            prior_specs = [ScalabilityRunSpec(**value) for value in existing.get("run_specs", [])]
            if (
                len(prior_specs) != int(existing.get("expected_run_count", -1))
                or any(
                    (record := _read_record(output_dir, spec.run_key)) is None
                    or record.get("state") != "complete"
                    for spec in prior_specs
                )
            ):
                raise ValueError("Prior scalability stage is not completely and atomically recorded.")
    elif stage != "s1":
        raise ValueError("S2/full-grid require the cumulative prior-stage output.")
    deps = dependencies or ScalabilityDependencies()
    baseline_solver = deps.solve_baseline or _production_baseline
    frontier_solver = deps.solve_frontier or _production_frontier
    post_solver = deps.post_evaluate or _production_post_evaluate
    # The frozen target itself must remain nonexistent until identity-locked
    # initialization; only its neutral container is prepared for the writer lock.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.scalability.lock"
    with SingleWriterLock(lock_path, resume=True):
        if not output_dir.exists():
            output_dir.mkdir(parents=False, exist_ok=False)
            atomic_write_yaml(output_dir / "resolved_config.yaml", config)
        resume_count = int((existing or {}).get("resume_count", 0)) + int(existing is not None)
        anchors = dict((existing or {}).get("baseline_anchors", {}))
        _refresh_manifest(
            output_dir, identity=identity, specs=specs, new_specs=new_specs,
            anchors=anchors, resume_count=resume_count,
        )
        (deps.configure_solver or _configure_gurobi)(identity["gurobi_parameters"])
        for seed in sorted({spec.seed for spec in specs}):
            instance_path = output_dir / "instances" / f"{seed}.json"
            if instance_path.exists():
                instance_payload = read_json(instance_path)
                if instance_payload is None:
                    raise ValueError("Stored instance is corrupt.")
                instance = InventoryInstance.from_dict(instance_payload)
            else:
                instance = deps.generate_instance(_base_config(config, str(config["instance_sizes"][0]), seed), seed=seed)
                atomic_write_json(instance_path, instance.to_dict())
            instance_sha256 = file_sha256(instance_path).upper()
            baseline_spec = next(spec for spec in specs if spec.seed == seed and spec.task_type == "baseline")
            baseline_record = _read_record(output_dir, baseline_spec.run_key)
            if baseline_record is not None and (
                baseline_record.get("git_commit") != current_commit
                or baseline_record.get("config_sha256") != identity["resolved_config_canonical_sha256"]
                or baseline_record.get("instance_sha256") != instance_sha256
            ):
                raise ValueError("Completed baseline identity mismatch.")
            if baseline_record is None:
                try:
                    _write_scalability_run_state(
                        output_dir,
                        baseline_spec.run_key,
                        state="running",
                        details={"phase": "baseline"},
                    )
                    if failure_injector:
                        failure_injector("before_baseline", baseline_spec)
                    payload = baseline_solver(config, instance, scale=baseline_spec.scale, seed=seed)
                    solved = payload.get("status") == "optimal" and payload.get("valid_UB") is True and payload.get("gap") is not None and float(payload["gap"]) <= float(config["tol"])
                    scientific = "certified_robust_optimal" if solved else "master_optimal_but_robust_uncertified"
                    baseline_record = {
                        "run_key": baseline_spec.run_key,
                        "state": "complete",
                        "task_type": "baseline",
                        "seed": seed,
                        "candidate": baseline_spec.candidate,
                        "scientific_status": scientific,
                        "solved_to_tolerance": scientific == "certified_robust_optimal",
                        "algorithm_status": payload.get("status"),
                        "git_commit": current_commit,
                        "config_sha256": identity["resolved_config_canonical_sha256"],
                        "instance_sha256": instance_sha256,
                        "result": payload,
                    }
                    _write_record(output_dir, baseline_spec, baseline_record)
                except KeyboardInterrupt:
                    _write_scalability_run_state(
                        output_dir,
                        baseline_spec.run_key,
                        state="interrupted",
                        details={"scientific_status": "interrupted"},
                    )
                    raise
                except Exception as exc:
                    _write_record(output_dir, baseline_spec, {
                        "run_key": baseline_spec.run_key, "state": "complete", "task_type": "baseline",
                        "seed": seed, "candidate": baseline_spec.candidate,
                        "scientific_status": "implementation_error", "algorithm_status": "exception",
                        "git_commit": current_commit, "config_sha256": identity["resolved_config_canonical_sha256"],
                        "instance_sha256": instance_sha256,
                        "failure_reason": str(exc), "result": {},
                    })
                    raise
            if baseline_record.get("scientific_status") != "certified_robust_optimal":
                for blocked in [
                    item for item in specs
                    if item.seed == seed and item.task_type == "frontier"
                ]:
                    if _read_record(output_dir, blocked.run_key) is None:
                        _write_record(output_dir, blocked, {
                            "run_key": blocked.run_key,
                            "state": "complete",
                            "task_type": "frontier",
                            "seed": seed,
                            "rho": blocked.rho,
                            "candidate": blocked.candidate,
                            "scientific_status": "implementation_error",
                            "algorithm_status": "baseline_uncertified",
                            "git_commit": current_commit,
                            "config_sha256": identity["resolved_config_canonical_sha256"],
                            "instance_sha256": instance_sha256,
                            "baseline_run_key": baseline_spec.run_key,
                            "failure_reason": "certified_baseline_anchor_unavailable",
                            "result": {},
                        })
                _refresh_manifest(
                    output_dir,
                    identity=identity,
                    specs=specs,
                    new_specs=new_specs,
                    anchors=anchors,
                    resume_count=resume_count,
                )
                continue
            certified_anchor = _certified_baseline_anchor(
                baseline_record,
                baseline_run_key=baseline_spec.run_key,
                config_hash=identity["resolved_config_canonical_sha256"],
                commit=current_commit,
                candidate_config_sha256=identity["candidate_config_sha256"],
                tolerance=float(config["tol"]),
            )
            if str(seed) in anchors and anchors[str(seed)] != certified_anchor:
                raise ValueError("Stored baseline anchor does not match its certified baseline.")
            anchors[str(seed)] = certified_anchor
            anchor = certified_anchor
            for spec in [item for item in specs if item.seed == seed and item.task_type == "frontier"]:
                record = _read_record(output_dir, spec.run_key)
                if record is not None and record.get("state") == "complete":
                    if (
                        record.get("baseline_run_key") != baseline_spec.run_key
                        or record.get("anchor_sha256") != anchor["anchor_sha256"]
                        or record.get("git_commit") != current_commit
                        or record.get("config_sha256") != identity["resolved_config_canonical_sha256"]
                        or record.get("instance_sha256") != instance_sha256
                    ):
                        raise ValueError("Completed frontier anchor identity mismatch.")
                    continue
                checkpoint_path = _algorithm_checkpoint_path(output_dir, spec.run_key)
                checkpoint = read_json(checkpoint_path)
                if checkpoint_path.exists() and checkpoint is None:
                    raise ValueError("Algorithm checkpoint is corrupt.")
                checkpoint_identity = {
                    "run_key": spec.run_key,
                    "git_commit": current_commit,
                    "config_sha256": identity["resolved_config_canonical_sha256"],
                    "anchor_sha256": anchor["anchor_sha256"],
                    "instance_sha256": instance_sha256,
                    "candidate": spec.candidate,
                    "rho": spec.rho,
                }
                if checkpoint is not None and (
                    checkpoint.get("identity") != checkpoint_identity
                    or not isinstance(checkpoint.get("result"), dict)
                ):
                    raise ValueError("Algorithm checkpoint identity mismatch.")
                try:
                    _write_scalability_run_state(
                        output_dir,
                        spec.run_key,
                        state="running",
                        details={"phase": "algorithm", "candidate": spec.candidate},
                    )
                    if checkpoint is None:
                        if failure_injector:
                            failure_injector("before_frontier", spec)
                        payload = frontier_solver(
                            config,
                            instance,
                            anchor=float(anchor["value"]),
                            rho=float(spec.rho),
                            candidate=spec.candidate,
                        )
                        atomic_write_json(checkpoint_path, {"identity": checkpoint_identity, "result": payload})
                    else:
                        payload = dict(checkpoint["result"])
                    algorithm_runtime = float(payload.get("runtime", 0.0))
                    algorithm_solved = payload.get("status") == "optimal" and payload.get("gap") is not None and float(payload["gap"]) <= float(config["tol"])
                    post_valid: bool | None = None
                    timing = {"post_evaluation_solver_runtime": 0.0, "post_evaluation_wall_runtime": 0.0, "aggregation_runtime": 0.0, "checkpoint_io_runtime": 0.0}
                    if algorithm_solved:
                        evaluation, timing = post_solver(
                            config,
                            instance,
                            output_dir=output_dir,
                            spec=spec,
                            payload=payload,
                            anchor=anchor,
                            identity=identity,
                            resume_count=resume_count,
                        )
                        payload["post_evaluation"] = evaluation
                        post_valid = bool(evaluation.get("valid")) and evaluation.get("objective_t_consistent") is not False
                    scientific = _scientific_status(
                        payload, post_valid, tolerance=float(config["tol"])
                    )
                    payload.update(timing)
                    payload["algorithm_runtime"] = algorithm_runtime
                    payload["total_wall_runtime"] = math.fsum(
                        (
                            algorithm_runtime,
                            float(timing["post_evaluation_wall_runtime"]),
                            float(timing["aggregation_runtime"]),
                            float(timing["checkpoint_io_runtime"]),
                        )
                    )
                    payload["post_evaluation_runtime_excluded_from_algorithm_runtime"] = True
                    payload["penalized_runtime_par2"] = penalized_runtime_par2(
                        solved_to_tolerance=scientific == "certified_robust_optimal",
                        runtime=algorithm_runtime,
                        time_limit=float(config["fairness_time_limit"]),
                    )
                    record = {
                        "run_key": spec.run_key,
                        "state": "complete",
                        "task_type": "frontier",
                        "seed": seed,
                        "rho": spec.rho,
                        "candidate": spec.candidate,
                        "scientific_status": scientific,
                        "algorithm_status": payload.get("status"),
                        "git_commit": current_commit,
                        "config_sha256": identity["resolved_config_canonical_sha256"],
                        "instance_sha256": instance_sha256,
                        "baseline_run_key": baseline_spec.run_key,
                        "anchor_sha256": anchor["anchor_sha256"],
                        "anchor_value_hex": anchor["value_hex"],
                        "result": payload,
                    }
                    _write_record(output_dir, spec, record)
                    if failure_injector:
                        failure_injector("after_run_record", spec)
                except KeyboardInterrupt:
                    committed = _read_record(output_dir, spec.run_key)
                    if committed is None or committed.get("state") != "complete":
                        _write_scalability_run_state(
                            output_dir,
                            spec.run_key,
                            state="interrupted",
                            details={"scientific_status": "interrupted"},
                        )
                    _refresh_manifest(output_dir, identity=identity, specs=specs, new_specs=new_specs, anchors=anchors, resume_count=resume_count)
                    raise
                except Exception as exc:
                    _write_record(output_dir, spec, {
                        "run_key": spec.run_key, "state": "complete", "task_type": "frontier",
                        "seed": seed, "rho": spec.rho, "candidate": spec.candidate,
                        "scientific_status": "implementation_error", "algorithm_status": "exception",
                        "git_commit": current_commit, "config_sha256": identity["resolved_config_canonical_sha256"],
                        "instance_sha256": instance_sha256,
                        "baseline_run_key": baseline_spec.run_key, "anchor_sha256": anchor["anchor_sha256"],
                        "failure_reason": str(exc), "result": {},
                    })
                    _refresh_manifest(output_dir, identity=identity, specs=specs, new_specs=new_specs, anchors=anchors, resume_count=resume_count)
                    raise
            _refresh_manifest(output_dir, identity=identity, specs=specs, new_specs=new_specs, anchors=anchors, resume_count=resume_count)
            _aggregate(output_dir, specs)
        _refresh_manifest(output_dir, identity=identity, specs=specs, new_specs=new_specs, anchors=anchors, resume_count=resume_count)
        _aggregate(output_dir, specs)
    return _manifest_path(output_dir)
