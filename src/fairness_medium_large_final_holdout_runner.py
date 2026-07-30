from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import gurobipy as gp
import yaml

from .experiment_protocol import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_yaml,
    config_sha256,
    file_sha256,
    git_commit,
    read_json,
    utc_now_iso,
)
from .experiment_suite import _base_config
from .fairness_benders import _certified_baseline_anchor, fairness_frontier_overall_status
from .fairness_scalability_results_audit import (
    CANONICALIZATION,
    RESULT_FIELDS,
    aggregate_records,
    project_record,
    resolved_config_file_bytes,
)
from .fairness_scalability_runner import (
    ScalabilityRunSpec,
    _production_baseline,
    _production_frontier,
    _production_post_evaluate,
    build_run_directory_mapping,
    run_directory_id,
)
from .instance import InventoryInstance, generate_instance
from .regional_fairness_pipeline import SingleWriterLock


FINAL_HOLDOUT_SCHEMA_VERSION = 1
FINAL_HOLDOUT_EXECUTION_ATTEMPT = 1
FINAL_HOLDOUT_STAGE = "medium_large_final_holdout"
HOLDOUT_SEEDS = tuple(range(170, 180))
PROHIBITED_SEEDS = tuple(range(130, 160))
RHOS = (0.0, 0.01, 0.025, 0.05, 0.10)
CANDIDATES = ("single_cut", "persistent_certified_cache_batch5")
BASELINE_CANDIDATE = "joint_v1_core_point_strengthened"
SCENARIO_COUNT = 1831
WINDOWS_PORTABLE_PATH_LIMIT = 220
EXPECTED_CANDIDATE_SHA256 = "7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6"
# Replaced after the protocol and authorized config are frozen in this PR.
EXPECTED_PROTOCOL_SHA256 = "6406975DFA9B637C99E322772EBFF3521444EA33F0F1E7028941FD66CF86FC27"
EXPECTED_FILE_SHA256 = "5DE72B46EF1E45A41A56A929601461A090FDC0991BC0D130591EEA757B57903E"
PUBLIC_STATUSES = (
    "certified_robust_optimal",
    "master_optimal_but_robust_uncertified",
    "time_limit_uncertified",
    "separation_stalled_duplicate",
    "infeasible",
    "invalid_post_evaluation",
    "implementation_error",
    "interrupted",
    "iteration_limit_uncertified",
    "numerical_uncertified",
    "unknown_uncertified",
)
FAIL_CLOSED_STATUSES = {
    "invalid_post_evaluation",
    "implementation_error",
    "interrupted",
}
HOLDOUT_RESULT_FIELDS = [
    *RESULT_FIELDS,
    "final_gap",
    "certified_cuts",
    "cost_budget",
    "certified_robust_cost",
    "worst_regional_shortage_rate",
]


@dataclass(frozen=True)
class HoldoutDependencies:
    generate_instance: Callable[..., InventoryInstance] = generate_instance
    solve_baseline: Callable[..., dict[str, Any]] | None = None
    solve_frontier: Callable[..., dict[str, Any]] | None = None
    post_evaluate: Callable[..., tuple[dict[str, Any], dict[str, float]]] | None = None
    configure_solver: Callable[[Mapping[str, Any]], None] | None = None


def _strict_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping.")
    return value


def _canonical_run_key(*, task_type: str, seed: int, rho: float | None, candidate: str) -> str:
    payload = {
        "candidate": str(candidate),
        "execution_attempt": FINAL_HOLDOUT_EXECUTION_ATTEMPT,
        "experiment": FINAL_HOLDOUT_STAGE,
        "rho": "NOT_APPLICABLE" if rho is None else format(float(rho), ".12g"),
        "scale": "medium_large",
        "seed": int(seed),
        "task_type": str(task_type),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def final_holdout_run_plan() -> list[ScalabilityRunSpec]:
    specs: list[ScalabilityRunSpec] = []
    for seed in HOLDOUT_SEEDS:
        specs.append(
            ScalabilityRunSpec(
                run_key=_canonical_run_key(
                    task_type="baseline", seed=seed, rho=None, candidate=BASELINE_CANDIDATE
                ),
                introduced_stage=FINAL_HOLDOUT_STAGE,
                task_type="baseline",
                scale="medium_large",
                seed=seed,
                rho=None,
                candidate=BASELINE_CANDIDATE,
            )
        )
        for rho in RHOS:
            for candidate in CANDIDATES:
                specs.append(
                    ScalabilityRunSpec(
                        run_key=_canonical_run_key(
                            task_type="frontier", seed=seed, rho=rho, candidate=candidate
                        ),
                        introduced_stage=FINAL_HOLDOUT_STAGE,
                        task_type="frontier",
                        scale="medium_large",
                        seed=seed,
                        rho=rho,
                        candidate=candidate,
                    )
                )
    keys = [spec.run_key for spec in specs]
    scientific = [
        (spec.scale, spec.task_type, spec.seed, spec.rho, spec.candidate) for spec in specs
    ]
    if len(specs) != 110 or len(keys) != len(set(keys)) or len(scientific) != len(set(scientific)):
        raise ValueError("Final holdout plan must contain exactly 110 unique scientific tasks.")
    build_run_directory_mapping(keys)
    return specs


def _atomic_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def path_portability_report(output_dir: Path, specs: Sequence[ScalabilityRunSpec]) -> dict[str, Any]:
    root = output_dir.resolve(strict=False)
    forward, reverse = build_run_directory_mapping([spec.run_key for spec in specs])
    paths: list[tuple[str, Path]] = []

    def add(kind: str, path: Path, *, atomic: bool = True) -> None:
        paths.append((kind, path))
        if atomic:
            paths.append((f"{kind}_atomic_tmp", _atomic_path(path)))

    for filename in (
        "manifest.json",
        "run_manifest.json",
        "resolved_config.yaml",
        "results.csv",
        "summary.csv",
        "paired_comparison.csv",
        "cost_fairness_frontier.csv",
        "paired_statistics.json",
        "audit_log.json",
    ):
        add(filename, root / filename)
    for seed in HOLDOUT_SEEDS:
        add("instance", root / "instances" / f"{seed}.json")
    last_chunk = math.ceil(SCENARIO_COUNT / 25) - 1
    for spec in specs:
        run_root = root / "runs" / forward[spec.run_key]
        add("run_json", run_root / "run.json")
        add("status_json", run_root / "status.json")
        if spec.task_type == "baseline":
            add("baseline_checkpoint", run_root / "baseline_checkpoint.json")
        else:
            add("algorithm_checkpoint", run_root / "algorithm_checkpoint.json")
            post_root = run_root / "post_evaluation"
            add("post_evaluation_chunk", post_root / "checkpoint" / f"chunk_{last_chunk:05d}.json")
            add("post_evaluation_index", post_root / "checkpoint" / "index.json")
            add("post_evaluation_final", post_root / "post_evaluation.json")
    kind, path = max(paths, key=lambda item: len(str(item[1])))
    length = len(str(path))
    return {
        "windows_portable_path_limit": WINDOWS_PORTABLE_PATH_LIMIT,
        "max_absolute_path_length": length,
        "longest_path_type": kind,
        "longest_path": str(path),
        "windows_portability_check": length <= WINDOWS_PORTABLE_PATH_LIMIT,
        "atomic_temporary_paths_checked": True,
        "run_key_to_directory_id": forward,
        "directory_id_to_run_key": reverse,
    }


def solver_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7}
    if dict(config.get("gurobi_parameters", {})) != expected:
        raise ValueError("Final holdout Gurobi identity drifted.")
    return expected


def _validate_prerequisite_evidence(config: Mapping[str, Any]) -> None:
    evidence = config.get("prerequisite_evidence")
    expected = {
        "large_attempt5_audit": {
            "sha256": "3A3BEC3BE024972DB93EE60BAB2A7BD7C1D4BB4C7CDD4C64033DAC43A05D734B",
            "required_decision": "stop_final_large_remediation",
        },
        "holdout_seed_access_audit": {
            "sha256": "BF6060E8974C9F8E8D6C2C64A6592E433316CF49DD848073F30F05013602F6A3",
            "required_decision": "holdout_seed_set_pristine",
        },
    }
    if not isinstance(evidence, Mapping) or set(evidence) != set(expected):
        raise ValueError("Final holdout prerequisite evidence identity drifted.")
    repo_root = Path(__file__).resolve().parents[1]
    for name, identity in expected.items():
        supplied = evidence.get(name)
        if not isinstance(supplied, Mapping):
            raise ValueError(f"Missing prerequisite evidence: {name}.")
        if str(supplied.get("sha256", "")).upper() != identity["sha256"]:
            raise ValueError(f"Prerequisite evidence SHA field drifted: {name}.")
        if supplied.get("required_decision") != identity["required_decision"]:
            raise ValueError(f"Prerequisite evidence decision field drifted: {name}.")
        relative = Path(str(supplied.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe prerequisite evidence path: {name}.")
        target = repo_root / relative
        if not target.is_file() or file_sha256(target).upper() != identity["sha256"]:
            raise ValueError(f"Prerequisite evidence file identity mismatch: {name}.")
        payload = read_json(target)
        decision_payload = payload.get("decision")
        decision_name = (
            decision_payload.get("decision")
            if isinstance(decision_payload, Mapping)
            else decision_payload
        )
        if decision_name != identity["required_decision"]:
            raise ValueError(f"Prerequisite evidence decision mismatch: {name}.")
        if name == "holdout_seed_access_audit" and payload.get("holdout_seed_accessed") is not False:
            raise ValueError("Holdout seed non-access evidence is not valid.")
        if name == "large_attempt5_audit":
            if not isinstance(decision_payload, Mapping):
                raise ValueError("Large Attempt 5 stop decision evidence is malformed.")
            if decision_payload.get("additional_large_runs_authorized") is not False:
                raise ValueError("Large Attempt 5 stop decision drifted.")
            if decision_payload.get("large_frontier_certified") is not False:
                raise ValueError("Large Attempt 5 scientific status drifted.")


def validate_runtime_config(config: Mapping[str, Any], *, config_path: Path | None = None) -> None:
    if config.get("authorization") != "formal_execution_authorized":
        raise ValueError("Final holdout authorization is not formal_execution_authorized.")
    if config.get("formal_run_authorized") is not True:
        raise ValueError("Final holdout formal_run_authorized must be true.")
    if int(config.get("execution_attempt", -1)) != FINAL_HOLDOUT_EXECUTION_ATTEMPT:
        raise ValueError("Final holdout execution attempt drifted.")
    if config.get("previous_attempt_results_reused") is not False:
        raise ValueError("Final holdout must not reuse previous results.")
    _validate_prerequisite_evidence(config)
    if config.get("instance_sizes") != ["medium_large"]:
        raise ValueError("Final holdout scale drifted.")
    if tuple(config.get("holdout_seeds", [])) != HOLDOUT_SEEDS:
        raise ValueError("Final holdout seeds drifted.")
    if tuple(config.get("prohibited_seeds", [])) != PROHIBITED_SEEDS:
        raise ValueError("Prohibited seed identity drifted.")
    if tuple(float(value) for value in config.get("rho_grid", [])) != RHOS:
        raise ValueError("Final holdout rho grid drifted.")
    if tuple(config.get("candidates", [])) != CANDIDATES:
        raise ValueError("Final holdout candidate set drifted.")
    expected_settings = {
        "single_cut": {
            "fairness_scalability_strategy": "single_cut",
            "persistent_separation_enabled": False,
            "certified_scenario_cache_enabled": False,
            "separation_solution_pool_enabled": False,
            "max_cuts_per_iteration": 1,
        },
        "persistent_certified_cache_batch5": {
            "fairness_scalability_strategy": "persistent_certified_cache_batch5",
            "persistent_separation_enabled": True,
            "certified_scenario_cache_enabled": True,
            "cache_payload": "deviation_pattern_only",
            "current_point_fixed_scenario_recertification_required": True,
            "separation_solution_pool_enabled": True,
            "max_cuts_per_iteration": 5,
        },
    }
    if config.get("candidate_settings") != expected_settings:
        raise ValueError("Final holdout candidate definitions drifted.")
    if any(float(config.get(key, math.nan)) != 1800.0 for key in (
        "time_limit", "baseline_time_limit", "fairness_time_limit"
    )):
        raise ValueError("Final holdout algorithm time limits drifted.")
    if int(config.get("gamma_target", -1)) != 2 or config.get("gamma_schedule") != [2]:
        raise ValueError("Final holdout uncertainty budget drifted.")
    if float(config.get("tol", math.nan)) != 1.0e-4:
        raise ValueError("Final holdout optimality tolerance drifted.")
    solver_identity(config)
    post = config.get("post_evaluation")
    if post != {
        "enabled": True,
        "exact_scenarios": True,
        "scenario_count": SCENARIO_COUNT,
        "max_scenarios": 5000,
        "time_limit_per_scenario": 30.0,
        "checkpoint_chunk_size": 25,
        "feasibility_tolerance": 1.0e-7,
    }:
        raise ValueError("Final holdout post-evaluation identity drifted.")
    if config.get("runtime_semantics") != {
        "par2_multiplier": 2,
        "par2_basis": "algorithm_runtime",
        "algorithm_runtime_excludes_post_evaluation": True,
        "total_wall_runtime_includes": [
            "algorithm", "post_evaluation", "aggregation", "checkpoint_io"
        ],
    }:
        raise ValueError("Final holdout runtime semantics drifted.")
    stats = config.get("statistical_analysis")
    if stats != {
        "independent_unit": "seed",
        "paired_unit_per_rho": "seed",
        "pairs_per_rho": 10,
        "cluster_bootstrap_unit": "seed",
        "cluster_contents": "all_rhos_and_both_candidates",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260728,
        "confidence_level": 0.95,
        "overall_rule": "mean_within_seed_over_rho_then_seed_cluster_bootstrap",
        "wilcoxon_per_rho": True,
        "wilcoxon_minimum_nonzero_pairs": 6,
        "multiple_testing_correction": "Holm",
        "failed_run_par2_seconds": 3600.0,
        "seed_rho_tasks_are_independent": False,
    }:
        raise ValueError("Final holdout statistical identity drifted.")
    candidate_path = Path(str(config.get("candidate_parameters_must_be_fixed_from", "")))
    protocol_path = Path(str(config.get("protocol_document", "")))
    if not candidate_path.is_file() or file_sha256(candidate_path).upper() != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("Frozen candidate file identity mismatch.")
    if not protocol_path.is_file() or file_sha256(protocol_path).upper() != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Final holdout protocol identity mismatch.")
    if str(config.get("candidate_config_sha256", "")).upper() != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("Candidate SHA field drifted.")
    if str(config.get("protocol_sha256", "")).upper() != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Protocol SHA field drifted.")
    if config_path is not None and file_sha256(config_path).upper() != EXPECTED_FILE_SHA256:
        raise ValueError("Final holdout authorized config file identity mismatch.")


def execution_identity(config: Mapping[str, Any], *, config_path: Path, commit: str) -> dict[str, Any]:
    validate_runtime_config(config, config_path=config_path)
    return {
        "schema_version": FINAL_HOLDOUT_SCHEMA_VERSION,
        "execution_attempt": FINAL_HOLDOUT_EXECUTION_ATTEMPT,
        "stage": FINAL_HOLDOUT_STAGE,
        "previous_attempt_results_reused": False,
        "prerequisite_evidence": deepcopy(config["prerequisite_evidence"]),
        "prior_attempts": list(config.get("prior_attempts", [])),
        "experiment_name": str(config["experiment_name"]),
        "scale": "medium_large",
        "git_commit": commit,
        "config_path": config_path.as_posix(),
        "config_file_sha256": file_sha256(config_path).upper(),
        "resolved_config_file_sha256": hashlib.sha256(
            resolved_config_file_bytes(config)
        ).hexdigest().upper(),
        "resolved_config_canonical_sha256": config_sha256(dict(config)).upper(),
        "resolved_config_canonicalization": CANONICALIZATION,
        "protocol_path": str(config["protocol_document"]),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "candidate_config_sha256": EXPECTED_CANDIDATE_SHA256,
        "candidate_definitions": list(CANDIDATES),
        "gurobi_parameters": solver_identity(config),
        "baseline_time_limit": 1800.0,
        "fairness_time_limit": 1800.0,
        "post_evaluation": dict(config["post_evaluation"]),
        "runtime_semantics": dict(config["runtime_semantics"]),
        "statistical_analysis": dict(config["statistical_analysis"]),
        "authorization": "formal_execution_authorized",
        "formal_run_authorized": True,
        "seeds": list(HOLDOUT_SEEDS),
        "rhos": list(RHOS),
        "candidates": list(CANDIDATES),
    }


def dry_run_report(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    validate_runtime_config(config, config_path=config_path)
    specs = final_holdout_run_plan()
    output_dir = Path(str(config["output_dir"]))
    portability = path_portability_report(output_dir, specs)
    if not portability["windows_portability_check"]:
        raise ValueError("Final holdout path portability check failed.")
    return {
        "stage": FINAL_HOLDOUT_STAGE,
        "scale": "medium_large",
        "baseline_count": 10,
        "frontier_count": 100,
        "total": 110,
        "unique_run_keys": len({spec.run_key for spec in specs}),
        "duplicate_run_keys": 110 - len({spec.run_key for spec in specs}),
        "scenario_count": SCENARIO_COUNT,
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "rhos": list(RHOS),
        "candidates": list(CANDIDATES),
        "instances_generated": False,
        "solver_called": False,
        "output_dir": str(output_dir),
        "output_dir_exists": output_dir.exists(),
        "windows_portability_check": portability["windows_portability_check"],
        "max_absolute_path_length": portability["max_absolute_path_length"],
        "longest_path": portability["longest_path"],
        "formal_run_authorized_in_config": True,
    }


def _run_root(output_dir: Path, run_key: str) -> Path:
    return output_dir / "runs" / run_directory_id(run_key)


def _record_path(output_dir: Path, run_key: str) -> Path:
    return _run_root(output_dir, run_key) / "run.json"


def _status_path(output_dir: Path, run_key: str) -> Path:
    return _run_root(output_dir, run_key) / "status.json"


def _checkpoint_path(output_dir: Path, spec: ScalabilityRunSpec) -> Path:
    filename = "baseline_checkpoint.json" if spec.task_type == "baseline" else "algorithm_checkpoint.json"
    return _run_root(output_dir, spec.run_key) / filename


def _read_record(output_dir: Path, run_key: str) -> dict[str, Any] | None:
    path = _record_path(output_dir, run_key)
    record = read_json(path)
    if path.exists() and record is None:
        raise ValueError(f"Corrupt final holdout run record: {run_key}")
    return record


def _write_status(
    output_dir: Path, spec: ScalabilityRunSpec, *, state: str,
    scientific_status: str | None = None, algorithm_status: str | None = None,
    phase: str | None = None,
) -> None:
    payload = {
        "run_key": spec.run_key,
        "run_directory_id": run_directory_id(spec.run_key),
        "state": state,
        "task_type": spec.task_type,
        "seed": spec.seed,
        "rho": spec.rho,
        "candidate": spec.candidate,
        "updated_at": utc_now_iso(),
    }
    if scientific_status is not None:
        payload["scientific_status"] = scientific_status
    if algorithm_status is not None:
        payload["algorithm_status"] = algorithm_status
    if phase is not None:
        payload["phase"] = phase
    atomic_write_json(_status_path(output_dir, spec.run_key), payload)


def _write_record(output_dir: Path, spec: ScalabilityRunSpec, record: Mapping[str, Any]) -> None:
    payload = {
        **dict(record),
        "run_key": spec.run_key,
        "run_directory_id": run_directory_id(spec.run_key),
    }
    atomic_write_json(_record_path(output_dir, spec.run_key), payload)
    _write_status(
        output_dir,
        spec,
        state=str(payload["state"]),
        scientific_status=str(payload["scientific_status"]),
        algorithm_status=str(payload.get("algorithm_status", "unknown")),
        phase="completed",
    )


def _strict_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite binary64 value.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite binary64 value.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    return result


def validate_production_baseline_payload(
    instance: InventoryInstance, payload: Mapping[str, Any]
) -> dict[str, Any]:
    y = payload.get("best_y_values")
    x = payload.get("best_x_values")
    if not isinstance(y, list) or len(y) != instance.num_warehouses:
        raise ValueError("invalid_production_baseline_best_y_values_shape")
    if not isinstance(x, list) or len(x) != instance.num_warehouses:
        raise ValueError("invalid_production_baseline_best_x_values_shape")
    parsed_y = [_strict_float(value, field=f"best_y_values[{i}]") for i, value in enumerate(y)]
    parsed_x: list[list[float]] = []
    for i, row in enumerate(x):
        if not isinstance(row, list) or len(row) != instance.num_products:
            raise ValueError("invalid_production_baseline_best_x_values_shape")
        parsed_x.append(
            [_strict_float(value, field=f"best_x_values[{i}][{j}]") for j, value in enumerate(row)]
        )
    return {
        "schema": "production_baseline_best_y_vector_best_x_matrix_v1",
        "warehouse_count": instance.num_warehouses,
        "product_count": instance.num_products,
        "warehouse_index_order": list(range(instance.num_warehouses)),
        "product_index_order": list(range(instance.num_products)),
        "best_y_values_sha256": hashlib.sha256(
            json.dumps(parsed_y, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest().upper(),
        "best_x_values_sha256": hashlib.sha256(
            json.dumps(parsed_x, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest().upper(),
    }


def _scientific_status(payload: Mapping[str, Any], post_valid: bool | None, *, tolerance: float) -> str:
    status = str(payload.get("status", "unknown"))
    gap = payload.get("gap")
    algorithm_solved = (
        status == "optimal"
        and gap is not None
        and math.isfinite(float(gap))
        and float(gap) <= float(tolerance)
    )
    if status == "separation_stalled_duplicate":
        return "separation_stalled_duplicate"
    if status == "infeasible":
        return "infeasible"
    mapped = fairness_frontier_overall_status(
        algorithm_status=status,
        algorithm_solved=algorithm_solved,
        post_evaluation_attempted=post_valid is not None,
        post_evaluation_valid=bool(post_valid),
    )
    return mapped if mapped in PUBLIC_STATUSES else "master_optimal_but_robust_uncertified"


def _configure_gurobi(settings: Mapping[str, Any]) -> None:
    gp.setParam("Threads", int(settings["Threads"]))
    gp.setParam("Seed", int(settings["Seed"]))
    gp.setParam("FeasibilityTol", float(settings["FeasibilityTol"]))


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def _validate_completed_status(output_dir: Path, spec: ScalabilityRunSpec, record: Mapping[str, Any]) -> None:
    path = _status_path(output_dir, spec.run_key)
    status = read_json(path)
    if path.exists() and status is None:
        raise ValueError("Final holdout status record is corrupt.")
    if (
        status is None
        or status.get("run_key") != spec.run_key
        or status.get("run_directory_id") != run_directory_id(spec.run_key)
        or status.get("state") != record.get("state")
        or status.get("scientific_status") != record.get("scientific_status")
        or status.get("algorithm_status") != record.get("algorithm_status")
    ):
        raise ValueError("Final holdout run/status identity mismatch.")


def _validate_post_evaluation_artifacts(
    output_dir: Path, spec: ScalabilityRunSpec, record: Mapping[str, Any]
) -> None:
    post_root = _run_root(output_dir, spec.run_key) / "post_evaluation"
    index_path = post_root / "checkpoint" / "index.json"
    final_path = post_root / "post_evaluation.json"
    index = read_json(index_path)
    final = read_json(final_path)
    if index is None or final is None:
        raise ValueError("Certified final holdout post-evaluation artifact is missing or corrupt.")
    chunks = index.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != math.ceil(SCENARIO_COUNT / 25):
        raise ValueError("Final holdout post-evaluation checkpoint index is incomplete.")
    scenario_total = 0
    seen: set[int] = set()
    for entry in chunks:
        if not isinstance(entry, dict):
            raise ValueError("Final holdout post-evaluation checkpoint entry is invalid.")
        chunk_index = int(entry.get("chunk_index", -1))
        relative = entry.get("relative_path")
        if chunk_index in seen or not isinstance(relative, str):
            raise ValueError("Final holdout post-evaluation checkpoint identity is duplicated.")
        seen.add(chunk_index)
        chunk_path = post_root / relative
        if not chunk_path.is_file() or file_sha256(chunk_path).lower() != str(entry.get("sha256", "")).lower():
            raise ValueError("Final holdout post-evaluation checkpoint hash mismatch.")
        scenario_total += int(entry.get("scenario_count", -1))
    if seen != set(range(len(chunks))) or scenario_total != SCENARIO_COUNT:
        raise ValueError("Final holdout post-evaluation scenario coverage mismatch.")
    recorded = (record.get("result") or {}).get("post_evaluation")
    if (
        final != recorded
        or final.get("valid") is not True
        or int(final.get("scenario_count", -1)) != SCENARIO_COUNT
        or final.get("objective_t_consistent") is False
        or final.get("errors")
    ):
        raise ValueError("Final holdout post-evaluation final identity mismatch.")


def _refresh_manifest(
    output_dir: Path, *, identity: Mapping[str, Any], specs: Sequence[ScalabilityRunSpec],
    anchors: Mapping[str, Any], resume_count: int, portability: Mapping[str, Any],
) -> dict[str, Any]:
    records = [_read_record(output_dir, spec.run_key) for spec in specs]
    completed = sum(record is not None and record.get("state") == "complete" for record in records)
    certified = sum(
        record is not None and record.get("scientific_status") == "certified_robust_optimal"
        for record in records
    )
    failed = sum(
        record is not None and record.get("scientific_status") in FAIL_CLOSED_STATUSES
        for record in records
    )
    payload = {
        **dict(identity),
        "created_at": (read_json(_manifest_path(output_dir)) or {}).get("created_at", utc_now_iso()),
        "updated_at": utc_now_iso(),
        "resume_count": resume_count,
        "expected_run_count": len(specs),
        "completed_run_count": completed,
        "pending_run_count": len(specs) - completed,
        "certified_solved_count": certified,
        "failed_run_count": failed,
        "baseline_anchors": dict(anchors),
        "run_specs": [spec.to_dict() for spec in specs],
        "run_key_to_directory_id": portability["run_key_to_directory_id"],
        "directory_id_to_run_key": portability["directory_id_to_run_key"],
        "path_portability_report": dict(portability),
    }
    atomic_write_json(_manifest_path(output_dir), payload)
    atomic_write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": FINAL_HOLDOUT_SCHEMA_VERSION,
            "execution_attempt": FINAL_HOLDOUT_EXECUTION_ATTEMPT,
            "git_commit": identity["git_commit"],
            "config_file_sha256": identity["config_file_sha256"],
            "run_specs": [spec.to_dict() for spec in specs],
            "run_key_to_directory_id": portability["run_key_to_directory_id"],
            "directory_id_to_run_key": portability["directory_id_to_run_key"],
        },
    )
    return payload


def _result_rows(
    records: Sequence[Mapping[str, Any]], specs: Sequence[ScalabilityRunSpec]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_rows, summary = aggregate_records(
        records,
        [spec.to_dict() for spec in specs],
        time_limit=1800.0,
    )
    by_key = {str(record["run_key"]): record for record in records}
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        record = by_key[str(row["run_key"])]
        result = dict(record.get("result") or {})
        post = result.get("post_evaluation") or {}
        certified = row["scientific_status"] == "certified_robust_optimal"
        rows.append(
            {
                **row,
                "final_gap": result.get("gap", "NOT_APPLICABLE"),
                "certified_cuts": result.get("cuts", 0),
                "cost_budget": result.get("cost_budget", "NOT_APPLICABLE"),
                "certified_robust_cost": (
                    post.get("actual_robust_cost", "NOT_APPLICABLE")
                    if certified else "NOT_APPLICABLE"
                ),
                "worst_regional_shortage_rate": (
                    post.get("realized_worst_shortage_rate", "NOT_APPLICABLE")
                    if certified else "NOT_APPLICABLE"
                ),
            }
        )
    by_group: dict[tuple[str, str, Any], list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault((str(row["task_type"]), str(row["candidate"]), row["rho"]), []).append(row)
    for item in summary:
        group = by_group.get((str(item["task_type"]), str(item["candidate"]), item["rho"]), [])
        item["status_separation_stalled_duplicate_count"] = sum(
            row["scientific_status"] == "separation_stalled_duplicate" for row in group
        )
        item["mean_final_gap"] = statistics.fmean(
            float(row["final_gap"]) for row in group
            if row["final_gap"] != "NOT_APPLICABLE"
        ) if group else 0.0
        item["mean_certified_cuts"] = statistics.fmean(float(row["certified_cuts"]) for row in group) if group else 0.0
    return rows, summary


def _paired_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frontier = {
        (int(row["seed"]), float(row["rho"]), str(row["candidate"])): row
        for row in rows if row["task_type"] == "frontier"
    }
    result: list[dict[str, Any]] = []
    for seed in HOLDOUT_SEEDS:
        for rho in RHOS:
            single = frontier.get((seed, rho, "single_cut"))
            batch = frontier.get((seed, rho, "persistent_certified_cache_batch5"))
            if single is None or batch is None:
                continue
            single_cert = single["scientific_status"] == "certified_robust_optimal"
            batch_cert = batch["scientific_status"] == "certified_robust_optimal"
            single_par2 = float(single["penalized_runtime_par2"])
            batch_par2 = float(batch["penalized_runtime_par2"])
            common = single_cert and batch_cert
            result.append(
                {
                    "seed": seed,
                    "rho": rho,
                    "single_cut_scientific_status": single["scientific_status"],
                    "batch5_scientific_status": batch["scientific_status"],
                    "single_cut_certified": single_cert,
                    "batch5_certified": batch_cert,
                    "single_cut_par2": single_par2,
                    "batch5_par2": batch_par2,
                    "paired_par2_difference_batch5_minus_single": batch_par2 - single_par2,
                    "common_certified": common,
                    "common_certified_algorithm_runtime_difference": (
                        float(batch["algorithm_runtime"]) - float(single["algorithm_runtime"])
                        if common else "NOT_APPLICABLE"
                    ),
                    "single_cut_objective_t": single["objective_t"] if common else "NOT_APPLICABLE",
                    "batch5_objective_t": batch["objective_t"] if common else "NOT_APPLICABLE",
                }
            )
    return result


def _round_half_even_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("Bootstrap denominator must be positive.")
    quotient, remainder = divmod(abs(int(numerator)), int(denominator))
    doubled = 2 * remainder
    rounded = quotient + int(doubled > denominator or (doubled == denominator and quotient % 2 == 1))
    return -rounded if numerator < 0 else rounded


def _bootstrap_interval(values_by_seed: Mapping[int, float], *, replicates: int, seed: int) -> dict[str, Any]:
    seeds = sorted(values_by_seed)
    if not seeds:
        return {"estimate": None, "lower": None, "upper": None, "clusters": 0}
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        sample = [values_by_seed[rng.choice(seeds)] for _ in seeds]
        draws.append(statistics.fmean(sample))
    draws.sort()
    lower_index = max(0, _round_half_even_ratio(25 * (replicates - 1), 1000))
    upper_index = min(replicates - 1, _round_half_even_ratio(975 * (replicates - 1), 1000))
    return {
        "estimate": statistics.fmean(values_by_seed.values()),
        "lower": draws[lower_index],
        "upper": draws[upper_index],
        "clusters": len(seeds),
        "replicates": replicates,
        "cluster_unit": "seed",
    }


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for index, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (m - index) * float(value)))
        adjusted[key] = running
    return adjusted


def paired_statistics(pairs: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    stats = dict(config["statistical_analysis"])
    if len(pairs) not in {0, 50}:
        return {
            "status": "incomplete_preregistered_matrix",
            "pair_count": len(pairs),
            "independent_unit": "seed",
        }
    if not pairs:
        return {"status": "no_completed_pairs", "pair_count": 0, "independent_unit": "seed"}
    scientific_units = [(int(row["seed"]), float(row["rho"])) for row in pairs]
    expected_units = {(seed, rho) for seed in HOLDOUT_SEEDS for rho in RHOS}
    if len(set(scientific_units)) != len(scientific_units) or set(scientific_units) != expected_units:
        raise ValueError("Paired comparison must contain exactly one pair per seed and rho.")
    replicates = int(stats["bootstrap_replicates"])
    bootstrap_seed = int(stats["bootstrap_seed"])
    per_rho: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except ImportError:  # pragma: no cover - environment evidence reports this explicitly.
        wilcoxon = None
    for rho_index, rho in enumerate(RHOS):
        subset = [row for row in pairs if float(row["rho"]) == rho]
        values = {int(row["seed"]): float(row["paired_par2_difference_batch5_minus_single"]) for row in subset}
        if set(values) != set(HOLDOUT_SEEDS) or len(values) != 10:
            raise ValueError("Each rho must contain exactly ten independent seed pairs.")
        nonzero = sum(value != 0.0 for value in values.values())
        key = format(rho, ".12g")
        item = {
            "pair_count": len(values),
            "bootstrap": _bootstrap_interval(
                values, replicates=replicates, seed=bootstrap_seed + rho_index
            ),
            "wilcoxon_condition": f"at_least_{stats['wilcoxon_minimum_nonzero_pairs']}_nonzero_seed_pairs",
            "wilcoxon_nonzero_pairs": nonzero,
            "wilcoxon_reported": False,
            "wilcoxon_raw_p": None,
            "wilcoxon_holm_p": None,
        }
        if wilcoxon is not None and nonzero >= int(stats["wilcoxon_minimum_nonzero_pairs"]):
            test = wilcoxon(list(values.values()), zero_method="wilcox", alternative="two-sided")
            item["wilcoxon_reported"] = True
            item["wilcoxon_raw_p"] = float(test.pvalue)
            raw_p[key] = float(test.pvalue)
        per_rho[key] = item
    adjusted = _holm_adjust(raw_p)
    for key, value in adjusted.items():
        per_rho[key]["wilcoxon_holm_p"] = value
    per_seed: dict[int, list[float]] = {seed: [] for seed in HOLDOUT_SEEDS}
    for row in pairs:
        per_seed[int(row["seed"])].append(
            float(row["paired_par2_difference_batch5_minus_single"])
        )
    if any(len(values) != len(RHOS) for values in per_seed.values()):
        raise ValueError("Overall comparison requires all five rho values within every seed cluster.")
    seed_means = {seed: statistics.fmean(values) for seed, values in per_seed.items()}
    return {
        "status": "complete",
        "independent_unit": "seed",
        "seed_rho_tasks_treated_as_independent": False,
        "certification": {
            "single_cut": {
                "certified_count": sum(bool(row["single_cut_certified"]) for row in pairs),
                "denominator": 50,
                "rate": sum(bool(row["single_cut_certified"]) for row in pairs) / 50.0,
            },
            "persistent_certified_cache_batch5": {
                "certified_count": sum(bool(row["batch5_certified"]) for row in pairs),
                "denominator": 50,
                "rate": sum(bool(row["batch5_certified"]) for row in pairs) / 50.0,
            },
        },
        "per_rho": per_rho,
        "overall": {
            "within_seed_aggregation": "arithmetic_mean_over_five_rhos",
            "bootstrap": _bootstrap_interval(
                seed_means, replicates=replicates, seed=bootstrap_seed + 100
            ),
        },
        "multiple_testing_correction": "Holm",
        "disclosure_independent_of_significance": True,
    }


def _cost_fairness_rows(
    rows: Sequence[Mapping[str, Any]], anchors: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    certified_t = {
        (int(row["seed"]), float(row["rho"]), str(row["candidate"])): float(row["objective_t"])
        for row in rows
        if row["task_type"] == "frontier"
        and row["scientific_status"] == "certified_robust_optimal"
    }
    for row in rows:
        if row["task_type"] != "frontier":
            continue
        seed = int(row["seed"])
        rho = float(row["rho"])
        anchor = anchors[str(seed)]
        certified = row["scientific_status"] == "certified_robust_optimal"
        result.append(
            {
                "seed": seed,
                "rho": rho,
                "candidate": row["candidate"],
                "scientific_status": row["scientific_status"],
                "certified_c_anchor": anchor["value"],
                "anchor_value_hex": anchor["value_hex"],
                "anchor_sha256": anchor["anchor_sha256"],
                "b_rho": (1.0 + rho) * float(anchor["value"]),
                "certified_t": row["objective_t"] if certified else "NOT_APPLICABLE",
                "certified_robust_cost": row["certified_robust_cost"] if certified else "NOT_APPLICABLE",
                "worst_regional_shortage_rate": (
                    row["worst_regional_shortage_rate"] if certified else "NOT_APPLICABLE"
                ),
                "fairness_improvement_vs_rho0": (
                    certified_t[(seed, 0.0, str(row["candidate"]))]
                    - certified_t[(seed, rho, str(row["candidate"]))]
                    if certified
                    and (seed, 0.0, str(row["candidate"])) in certified_t
                    else "NOT_APPLICABLE"
                ),
                "budget_increment_vs_baseline": rho * float(anchor["value"]),
            }
        )
    return result


def _aggregate(
    output_dir: Path, specs: Sequence[ScalabilityRunSpec], anchors: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    start = time.perf_counter()
    records = [record for spec in specs if (record := _read_record(output_dir, spec.run_key))]
    rows, summary = _result_rows(records, specs)
    pairs = _paired_rows(rows)
    frontier = _cost_fairness_rows(rows, anchors)
    atomic_write_csv(output_dir / "results.csv", rows, HOLDOUT_RESULT_FIELDS)
    atomic_write_csv(output_dir / "summary.csv", summary, list(summary[0]) if summary else [])
    atomic_write_csv(
        output_dir / "paired_comparison.csv", pairs, list(pairs[0]) if pairs else ["seed", "rho"]
    )
    atomic_write_csv(
        output_dir / "cost_fairness_frontier.csv",
        frontier,
        list(frontier[0]) if frontier else ["seed", "rho", "candidate"],
    )
    atomic_write_json(output_dir / "paired_statistics.json", paired_statistics(pairs, config))
    atomic_write_json(
        output_dir / "audit_log.json",
        {
            "record_count": len(rows),
            "unique_run_keys": len(rows) == len({row["run_key"] for row in rows}),
            "paired_count": len(pairs),
            "independent_unit": "seed",
            "aggregation_runtime": time.perf_counter() - start,
            "updated_at": utc_now_iso(),
        },
    )


def _validate_existing_identity(existing: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    for field in (
        "schema_version", "execution_attempt", "stage", "previous_attempt_results_reused",
        "experiment_name", "scale", "git_commit", "config_file_sha256",
        "resolved_config_file_sha256", "resolved_config_canonical_sha256",
        "resolved_config_canonicalization", "protocol_sha256", "candidate_config_sha256",
        "candidate_definitions", "gurobi_parameters", "baseline_time_limit",
        "fairness_time_limit", "post_evaluation", "runtime_semantics",
        "statistical_analysis", "seeds", "rhos", "candidates",
    ):
        if existing.get(field) != identity.get(field):
            raise ValueError(f"Final holdout resume identity mismatch: {field}")


def run_final_holdout(
    config: dict[str, Any], *, config_path: Path, resume: bool,
    dependencies: HoldoutDependencies | None = None,
    test_authorization: bool = False,
    failure_injector: Callable[[str, ScalabilityRunSpec], None] | None = None,
) -> Path:
    if not resume:
        raise ValueError("Final holdout execution requires --resume; --overwrite is unsupported.")
    validate_runtime_config(config, config_path=config_path)
    if dependencies is not None and not test_authorization:
        raise ValueError("Dependency substitution requires explicit test_authorization.")
    production_execution = dependencies is None
    specs = final_holdout_run_plan()
    output_dir = Path(str(config["output_dir"]))
    portability = path_portability_report(output_dir, specs)
    if not portability["windows_portability_check"]:
        raise ValueError("Final holdout Windows path portability check failed.")
    current_commit = git_commit(Path(__file__).resolve().parents[1])
    identity = execution_identity(config, config_path=config_path, commit=current_commit)
    existing = read_json(_manifest_path(output_dir))
    if output_dir.exists() and existing is None:
        raise ValueError("Existing final holdout output lacks a valid identity manifest.")
    if existing is not None:
        _validate_existing_identity(existing, identity)
        if existing.get("path_portability_report") != portability:
            raise ValueError("Final holdout path identity drifted.")
        resolved_path = output_dir / "resolved_config.yaml"
        if (
            not resolved_path.is_file()
            or file_sha256(resolved_path).upper() != identity["resolved_config_file_sha256"]
        ):
            raise ValueError("Final holdout resolved config is missing, corrupt, or drifted.")
    deps = dependencies or HoldoutDependencies()
    baseline_solver = deps.solve_baseline or _production_baseline
    frontier_solver = deps.solve_frontier or _production_frontier
    post_solver = deps.post_evaluate or _production_post_evaluate
    configure = deps.configure_solver or _configure_gurobi
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.final_holdout.lock"
    with SingleWriterLock(lock_path, resume=True):
        if not output_dir.exists():
            output_dir.mkdir(parents=False, exist_ok=False)
            atomic_write_yaml(output_dir / "resolved_config.yaml", config)
        resume_count = int((existing or {}).get("resume_count", 0)) + int(existing is not None)
        anchors = dict((existing or {}).get("baseline_anchors", {}))
        _refresh_manifest(
            output_dir, identity=identity, specs=specs, anchors=anchors,
            resume_count=resume_count, portability=portability,
        )
        configure(identity["gurobi_parameters"])
        for seed in HOLDOUT_SEEDS:
            instance_path = output_dir / "instances" / f"{seed}.json"
            if instance_path.exists():
                instance_payload = read_json(instance_path)
                if instance_payload is None:
                    raise ValueError("Stored final holdout instance is corrupt.")
                instance = InventoryInstance.from_dict(instance_payload)
            else:
                instance = deps.generate_instance(_base_config(config, "medium_large", seed), seed=seed)
                atomic_write_json(instance_path, instance.to_dict())
            instance_sha = file_sha256(instance_path).upper()
            baseline_spec = next(
                spec for spec in specs if spec.seed == seed and spec.task_type == "baseline"
            )
            baseline_record = _read_record(output_dir, baseline_spec.run_key)
            if baseline_record is not None:
                if (
                    baseline_record.get("git_commit") != current_commit
                    or baseline_record.get("config_sha256") != identity["resolved_config_canonical_sha256"]
                    or baseline_record.get("instance_sha256") != instance_sha
                ):
                    raise ValueError("Completed final holdout baseline identity mismatch.")
                _validate_completed_status(output_dir, baseline_spec, baseline_record)
                validate_production_baseline_payload(instance, baseline_record.get("result") or {})
                checkpoint = read_json(_checkpoint_path(output_dir, baseline_spec))
                expected_checkpoint_identity = {
                    "run_key": baseline_spec.run_key,
                    "git_commit": current_commit,
                    "config_sha256": identity["resolved_config_canonical_sha256"],
                    "instance_sha256": instance_sha,
                }
                if (
                    checkpoint is None
                    or checkpoint.get("identity") != expected_checkpoint_identity
                    or checkpoint.get("result") != baseline_record.get("result")
                ):
                    raise ValueError("Completed final holdout baseline checkpoint mismatch.")
            else:
                checkpoint_path = _checkpoint_path(output_dir, baseline_spec)
                checkpoint = read_json(checkpoint_path)
                if checkpoint_path.exists() and checkpoint is None:
                    raise ValueError("Final holdout baseline checkpoint is corrupt.")
                checkpoint_identity = {
                    "run_key": baseline_spec.run_key,
                    "git_commit": current_commit,
                    "config_sha256": identity["resolved_config_canonical_sha256"],
                    "instance_sha256": instance_sha,
                }
                if checkpoint is not None and checkpoint.get("identity") != checkpoint_identity:
                    raise ValueError("Final holdout baseline checkpoint identity mismatch.")
                try:
                    _write_status(output_dir, baseline_spec, state="running", phase="baseline")
                    if checkpoint is None:
                        if failure_injector:
                            failure_injector("before_baseline", baseline_spec)
                        payload = baseline_solver(config, instance, scale="medium_large", seed=seed)
                        schema_evidence = validate_production_baseline_payload(instance, payload)
                        atomic_write_json(
                            checkpoint_path,
                            {"identity": checkpoint_identity, "result": payload, "schema_evidence": schema_evidence},
                        )
                    else:
                        payload = dict(checkpoint.get("result") or {})
                        schema_evidence = validate_production_baseline_payload(instance, payload)
                    solved = (
                        payload.get("status") == "optimal"
                        and payload.get("valid_UB") is True
                        and payload.get("gap") is not None
                        and float(payload["gap"]) <= float(config["tol"])
                    )
                    scientific = "certified_robust_optimal" if solved else "master_optimal_but_robust_uncertified"
                    baseline_record = {
                        "state": "complete",
                        "task_type": "baseline",
                        "scale": "medium_large",
                        "seed": seed,
                        "candidate": BASELINE_CANDIDATE,
                        "scientific_status": scientific,
                        "algorithm_status": payload.get("status"),
                        "solved_to_tolerance": solved,
                        "git_commit": current_commit,
                        "config_sha256": identity["resolved_config_canonical_sha256"],
                        "config_file_sha256": identity["config_file_sha256"],
                        "instance_sha256": instance_sha,
                        "baseline_solution_schema_evidence": schema_evidence,
                        "result": payload,
                    }
                    _write_record(output_dir, baseline_spec, baseline_record)
                except KeyboardInterrupt:
                    _write_status(output_dir, baseline_spec, state="interrupted", scientific_status="interrupted")
                    raise
                except Exception as exc:
                    _write_record(
                        output_dir,
                        baseline_spec,
                        {
                            "state": "complete", "task_type": "baseline", "scale": "medium_large",
                            "seed": seed, "candidate": BASELINE_CANDIDATE,
                            "scientific_status": "implementation_error", "algorithm_status": "exception",
                            "git_commit": current_commit,
                            "config_sha256": identity["resolved_config_canonical_sha256"],
                            "config_file_sha256": identity["config_file_sha256"],
                            "instance_sha256": instance_sha, "failure_reason": str(exc), "result": {},
                        },
                    )
                    _refresh_manifest(
                        output_dir, identity=identity, specs=specs, anchors=anchors,
                        resume_count=resume_count, portability=portability,
                    )
                    raise
            if baseline_record.get("scientific_status") != "certified_robust_optimal":
                raise RuntimeError(f"Certified baseline unavailable for holdout seed {seed}.")
            anchor = _certified_baseline_anchor(
                baseline_record,
                baseline_run_key=baseline_spec.run_key,
                config_hash=identity["resolved_config_canonical_sha256"],
                commit=current_commit,
                candidate_config_sha256=identity["candidate_config_sha256"],
                tolerance=float(config["tol"]),
            )
            if str(seed) in anchors and anchors[str(seed)] != anchor:
                raise ValueError("Stored final holdout anchor identity mismatch.")
            anchors[str(seed)] = anchor
            for spec in [item for item in specs if item.seed == seed and item.task_type == "frontier"]:
                record = _read_record(output_dir, spec.run_key)
                if record is not None and record.get("state") == "complete":
                    if (
                        record.get("git_commit") != current_commit
                        or record.get("config_sha256") != identity["resolved_config_canonical_sha256"]
                        or record.get("instance_sha256") != instance_sha
                        or record.get("baseline_run_key") != baseline_spec.run_key
                        or record.get("anchor_sha256") != anchor["anchor_sha256"]
                    ):
                        raise ValueError("Completed final holdout frontier identity mismatch.")
                    _validate_completed_status(output_dir, spec, record)
                    checkpoint = read_json(_checkpoint_path(output_dir, spec))
                    expected_checkpoint_identity = {
                        "run_key": spec.run_key,
                        "git_commit": current_commit,
                        "config_sha256": identity["resolved_config_canonical_sha256"],
                        "instance_sha256": instance_sha,
                        "baseline_run_key": baseline_spec.run_key,
                        "anchor_sha256": anchor["anchor_sha256"],
                        "candidate": spec.candidate,
                        "rho": spec.rho,
                    }
                    if (
                        checkpoint is None
                        or checkpoint.get("identity") != expected_checkpoint_identity
                        or not isinstance(checkpoint.get("result"), dict)
                    ):
                        raise ValueError("Completed final holdout algorithm checkpoint mismatch.")
                    if (
                        production_execution
                        and record.get("scientific_status") == "certified_robust_optimal"
                    ):
                        _validate_post_evaluation_artifacts(output_dir, spec, record)
                    continue
                checkpoint_path = _checkpoint_path(output_dir, spec)
                checkpoint = read_json(checkpoint_path)
                if checkpoint_path.exists() and checkpoint is None:
                    raise ValueError("Final holdout algorithm checkpoint is corrupt.")
                checkpoint_identity = {
                    "run_key": spec.run_key,
                    "git_commit": current_commit,
                    "config_sha256": identity["resolved_config_canonical_sha256"],
                    "instance_sha256": instance_sha,
                    "baseline_run_key": baseline_spec.run_key,
                    "anchor_sha256": anchor["anchor_sha256"],
                    "candidate": spec.candidate,
                    "rho": spec.rho,
                }
                if checkpoint is not None and checkpoint.get("identity") != checkpoint_identity:
                    raise ValueError("Final holdout algorithm checkpoint identity mismatch.")
                try:
                    _write_status(output_dir, spec, state="running", phase="algorithm")
                    if checkpoint is None:
                        if failure_injector:
                            failure_injector("before_frontier", spec)
                        payload = frontier_solver(
                            config, instance, anchor=float(anchor["value"]),
                            rho=float(spec.rho), candidate=spec.candidate,
                        )
                        atomic_write_json(checkpoint_path, {"identity": checkpoint_identity, "result": payload})
                    else:
                        payload = dict(checkpoint.get("result") or {})
                    status = str(payload.get("status", "unknown"))
                    gap = payload.get("gap")
                    algorithm_solved = (
                        status == "optimal" and gap is not None and float(gap) <= float(config["tol"])
                    )
                    post_valid: bool | None = None
                    if algorithm_solved:
                        _write_status(output_dir, spec, state="running", phase="post_evaluation")
                        post, timing = post_solver(
                            config, instance, output_dir=output_dir, spec=spec,
                            payload=payload, anchor=anchor, identity=identity,
                            resume_count=resume_count,
                        )
                        if int(post.get("scenario_count", -1)) != SCENARIO_COUNT:
                            post["valid"] = False
                            post.setdefault("errors", []).append("scenario_count_identity_mismatch")
                        post_valid = (
                            post.get("valid") is True
                            and post.get("objective_t_consistent") is not False
                            and not post.get("errors")
                        )
                        payload["post_evaluation"] = post
                        payload.update(timing)
                    payload.setdefault("post_evaluation_solver_runtime", 0.0)
                    payload.setdefault("post_evaluation_wall_runtime", 0.0)
                    payload.setdefault("aggregation_runtime", 0.0)
                    payload.setdefault("checkpoint_io_runtime", 0.0)
                    algorithm_runtime = float(payload.get("algorithm_runtime", payload.get("runtime", 0.0)))
                    scientific = _scientific_status(payload, post_valid, tolerance=float(config["tol"]))
                    solved = scientific == "certified_robust_optimal"
                    payload["algorithm_runtime"] = algorithm_runtime
                    payload["penalized_runtime_par2"] = algorithm_runtime if solved else 3600.0
                    payload["total_wall_runtime"] = math.fsum(
                        (
                            algorithm_runtime,
                            float(payload["post_evaluation_wall_runtime"]),
                            float(payload["aggregation_runtime"]),
                            float(payload["checkpoint_io_runtime"]),
                        )
                    )
                    payload["post_evaluation_runtime_excluded_from_algorithm_runtime"] = True
                    record = {
                        "state": "complete", "task_type": "frontier", "scale": "medium_large",
                        "seed": seed, "rho": spec.rho, "candidate": spec.candidate,
                        "scientific_status": scientific, "algorithm_status": status,
                        "solved_to_tolerance": solved, "git_commit": current_commit,
                        "config_sha256": identity["resolved_config_canonical_sha256"],
                        "config_file_sha256": identity["config_file_sha256"],
                        "instance_sha256": instance_sha, "baseline_run_key": baseline_spec.run_key,
                        "anchor_value_hex": anchor["value_hex"],
                        "anchor_sha256": anchor["anchor_sha256"], "result": payload,
                    }
                    _write_record(output_dir, spec, record)
                    if scientific == "invalid_post_evaluation":
                        raise RuntimeError("invalid_post_evaluation")
                except KeyboardInterrupt:
                    _write_status(output_dir, spec, state="interrupted", scientific_status="interrupted")
                    raise
                except Exception as exc:
                    existing_record = _read_record(output_dir, spec.run_key)
                    if existing_record is None:
                        _write_record(
                            output_dir,
                            spec,
                            {
                                "state": "complete", "task_type": "frontier", "scale": "medium_large",
                                "seed": seed, "rho": spec.rho, "candidate": spec.candidate,
                                "scientific_status": "implementation_error", "algorithm_status": "exception",
                                "git_commit": current_commit,
                                "config_sha256": identity["resolved_config_canonical_sha256"],
                                "config_file_sha256": identity["config_file_sha256"],
                                "instance_sha256": instance_sha, "baseline_run_key": baseline_spec.run_key,
                                "anchor_value_hex": anchor["value_hex"],
                                "anchor_sha256": anchor["anchor_sha256"],
                                "failure_reason": str(exc), "result": {},
                            },
                        )
                    _refresh_manifest(
                        output_dir, identity=identity, specs=specs, anchors=anchors,
                        resume_count=resume_count, portability=portability,
                    )
                    _aggregate(output_dir, specs, anchors, config)
                    raise
                _refresh_manifest(
                    output_dir, identity=identity, specs=specs, anchors=anchors,
                    resume_count=resume_count, portability=portability,
                )
                _aggregate(output_dir, specs, anchors, config)
        _refresh_manifest(
            output_dir, identity=identity, specs=specs, anchors=anchors,
            resume_count=resume_count, portability=portability,
        )
        _aggregate(output_dir, specs, anchors, config)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Final medium-large fairness holdout runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=[FINAL_HOLDOUT_STAGE])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    if argv is not None and "--overwrite" in argv:
        parser.error("--overwrite is unsupported")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = _strict_yaml(config_path)
    if args.dry_run:
        print(json.dumps(dry_run_report(config, config_path=config_path), indent=2, sort_keys=True))
        return 0
    run_final_holdout(config, config_path=config_path, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
