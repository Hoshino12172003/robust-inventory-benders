from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Iterable

import yaml

from .experiment_protocol import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_yaml,
    canonical_json_sha256,
    file_sha256,
    penalized_runtime_par2,
)


STAGE = "GAMMA_SENSITIVITY"
SCHEMA = "fairness_hybrid_gamma_sensitivity_manifest_v2"
CHECKPOINT_SCHEMA = "fairness_hybrid_gamma_sensitivity_algorithm_checkpoint_v2"
POST_SCHEMA = "fairness_hybrid_gamma_sensitivity_post_evaluation_v2"
AUTHORIZATION_SCHEMA = "fairness_hybrid_gamma_sensitivity_authorization_v2"
BASELINE_CHECKPOINT_SCHEMA = "fairness_hybrid_gamma_sensitivity_baseline_checkpoint_v2"
EXECUTION_ATTEMPT = 2
BASE_COMMIT = "72288df0f628e499616b5132daf1e89b2467dce5"
CANDIDATE = "certified_hybrid_scenario_benders_fairness"
CANDIDATE_SHA256 = "8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0"
EXPECTED_CONFIG_SHA256 = "ED492C925A32E751882FBA685120A87530DE852D6F3F7C5E374479300AE15F68"
EXPECTED_PROTOCOL_SHA256 = "098287216AF8A9917F6488E8BAF662F62A115D10F98B307C7CB66032C7452375"
SEEDS = [180, 181, 182, 183, 184]
GAMMAS = [0, 1, 2]
RHO = 0.025
SOLVER_PARAMETERS = {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}
SCALES = {
    "medium_large": {
        "num_regions": 10,
        "num_products": 6,
        "demand_components": 60,
        "scenario_counts": {0: 1, 1: 61, 2: 1831},
        "output_dir": "experiments/results_fh_gamma/ml_a2",
    },
    "large": {
        "num_regions": 12,
        "num_products": 8,
        "demand_components": 96,
        "scenario_counts": {0: 1, 1: 97, 2: 4657},
        "output_dir": "experiments/results_fh_gamma/lg_a2",
    },
}
FORBIDDEN_REUSE_PARTS = (
    "results_fairness_hybrid_final_holdout",
    "fairness_hybrid_ccg_benders_d1",
    "fairness_hybrid_ccg_benders_d2",
    "experiments/results_fh_gamma/ml_a1",
    "experiments/results_fh_gamma/lg_a1",
)
ATTEMPT1_OUTPUT_DIRS = (
    "experiments/results_fh_gamma/ml_a1",
    "experiments/results_fh_gamma/lg_a1",
)


class ProtocolGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class GammaDependencies:
    generate_instance: Callable[[dict[str, Any], int], Any]
    serialize_instance: Callable[[Any], dict[str, Any]]
    deserialize_instance: Callable[[dict[str, Any]], Any]
    solve_baseline: Callable[..., dict[str, Any]]
    make_anchor: Callable[..., dict[str, Any]]
    solve_frontier: Callable[..., dict[str, Any]]
    post_evaluate: Callable[..., tuple[dict[str, Any], dict[str, float]]]
    configure_solver: Callable[[dict[str, Any]], None]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_value(value: Any) -> str:
    return canonical_json_sha256(value)


def instance_archive_payload(
    frozen_identity: dict[str, Any], serialized_instance: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(serialized_instance, dict):
        raise ProtocolGateError("instance serializer returned invalid payload")
    canonical_sha = sha256_value(serialized_instance)
    return {
        "identity": {**frozen_identity, "instance_canonical_sha256": canonical_sha},
        "instance": serialized_instance,
    }


def validate_instance_archive(
    archive: dict[str, Any], frozen_identity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    if not isinstance(archive, dict):
        raise ProtocolGateError("Gamma-specific instance archive is corrupt")
    serialized = archive.get("instance")
    stored_identity = archive.get("identity")
    if not isinstance(serialized, dict) or not isinstance(stored_identity, dict):
        raise ProtocolGateError("Gamma-specific instance archive is corrupt")
    canonical_sha = sha256_value(serialized)
    expected_identity = {**frozen_identity, "instance_canonical_sha256": canonical_sha}
    if stored_identity != expected_identity:
        raise ProtocolGateError("Gamma-specific instance identity mismatch")
    return serialized, stored_identity, canonical_sha, sha256_value(stored_identity)


def record_manifest_identity(
    manifest: dict[str, Any], section: str, key: str, value: dict[str, Any],
) -> None:
    target = manifest.get(section)
    if not isinstance(target, dict):
        raise ProtocolGateError(f"manifest {section} is corrupt")
    existing = target.get(key)
    if existing is not None and existing != value:
        raise ProtocolGateError(f"manifest {section} identity mismatch")
    target[key] = deepcopy(value)


def run_directory_id(run_key: str) -> str:
    return "r_" + hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:24]


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolGateError("Gamma sensitivity config must be a mapping")
    return value


def scenario_count(scale: str, gamma: int) -> int:
    if scale not in SCALES:
        raise ProtocolGateError(f"unsupported scale: {scale}")
    if type(gamma) is not int or gamma not in GAMMAS:
        raise ProtocolGateError(f"Gamma must be one of {GAMMAS}; Gamma 3 and 4 are forbidden")
    return int(SCALES[scale]["scenario_counts"][gamma])


def _frozen_scale(scale: str) -> dict[str, Any]:
    value = SCALES[scale]
    return {
        "num_regions": value["num_regions"],
        "num_products": value["num_products"],
        "demand_components": value["demand_components"],
        "scenario_counts": value["scenario_counts"],
        "baseline_count": 15,
        "frontier_count": 15,
        "total_tasks": 30,
        "output_dir": value["output_dir"],
    }


def validate_config(path: str | Path, config: dict[str, Any]) -> None:
    source = Path(path)
    if EXPECTED_CONFIG_SHA256 != "TO_BE_FROZEN" and file_sha256(source).upper() != EXPECTED_CONFIG_SHA256:
        raise ProtocolGateError("Gamma sensitivity config SHA mismatch")
    expected = {
        "stage": STAGE,
        "authorization": "protocol_only_pre_run_audit",
        "formal_run_authorized": False,
        "next_authorized_stage": "fairness_hybrid_gamma_sensitivity_pre_run_audit_only",
        "schema_version": 2,
        "execution_attempt": EXECUTION_ATTEMPT,
        "previous_attempt_results_reused": False,
        "formal_worktree_root": r"E:\rfgs",
        "base_commit": BASE_COMMIT,
        "candidate": CANDIDATE,
        "required_candidate_sha256": CANDIDATE_SHA256,
        "seeds": SEEDS,
        "gamma": GAMMAS,
        "rho": [RHO],
        "baseline_rho_identity": "NOT_APPLICABLE",
        "baseline_count": 30,
        "frontier_count": 30,
        "total_tasks": 60,
        "solver_identity": SOLVER_PARAMETERS,
        "tol": 1e-4,
        "baseline_time_limit_seconds": 1800,
        "algorithm_time_limit_seconds": 1800,
        "general_time_limit_seconds": 1800,
        "max_iterations": 10000,
        "post_evaluation": {
            "time_limit_per_scenario_seconds": 30,
            "checkpoint_chunk_size": 25,
            "pipeline_generation": 4,
        },
        "runtime_semantics": {"par2_multiplier": 2, "par2_basis": "algorithm_runtime"},
        "resume": True,
        "overwrite_supported": False,
        "forbidden_reuse_families": ["FINAL_HOLDOUT", "D1", "D2"],
        "forbidden_prior_attempt_output_dirs": list(ATTEMPT1_OUTPUT_DIRS),
        "forbidden_gamma": [3, 4],
        "forbidden_additional_rho": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ProtocolGateError(f"Gamma sensitivity identity drifted: {key}")
    if set(config.get("scales", {})) != set(SCALES):
        raise ProtocolGateError("Gamma sensitivity scale set drifted")
    for scale in SCALES:
        if config["scales"][scale] != _frozen_scale(scale):
            raise ProtocolGateError(f"Gamma sensitivity scale identity drifted: {scale}")
    root = Path(__file__).resolve().parents[1]
    protocol = root / str(config["protocol_document"])
    candidate = root / str(config["candidate_definition"])
    actual_protocol = file_sha256(protocol).upper()
    if config["required_protocol_sha256"] != actual_protocol:
        raise ProtocolGateError("Gamma sensitivity protocol SHA mismatch")
    if EXPECTED_PROTOCOL_SHA256 != "TO_BE_FROZEN" and actual_protocol != EXPECTED_PROTOCOL_SHA256:
        raise ProtocolGateError("runner protocol SHA identity drifted")
    if file_sha256(candidate).upper() != CANDIDATE_SHA256:
        raise ProtocolGateError("candidate SHA mismatch")
    stats = config.get("statistics", {})
    if stats.get("independent_unit") != "seed" or stats.get("paired_axis") != "gamma_within_scale_and_seed":
        raise ProtocolGateError("Gamma sensitivity statistical unit drifted")


def _rho_text(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def expand_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scale in SCALES:
        for seed in SEEDS:
            for gamma in GAMMAS:
                for task_type, candidate, rho in (
                    ("baseline", "baseline", "NOT_APPLICABLE"),
                    ("frontier", CANDIDATE, _rho_text(RHO)),
                ):
                    identity = {
                        "candidate": candidate,
                        "execution_attempt": EXECUTION_ATTEMPT,
                        "gamma": gamma,
                        "rho": rho,
                        "scale": scale,
                        "seed": seed,
                        "stage": STAGE,
                        "task_type": task_type,
                    }
                    key = canonical_json(identity)
                    rows.append({**identity, "run_key": key, "run_directory_id": run_directory_id(key)})
    keys = [row["run_key"] for row in rows]
    directories = [row["run_directory_id"] for row in rows]
    if len(rows) != 60 or len(set(keys)) != 60 or len(set(directories)) != 60:
        raise ProtocolGateError("run plan duplicate or short-directory collision")
    return rows


def paired_baseline(plan: Iterable[dict[str, Any]], frontier: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row for row in plan
        if row["task_type"] == "baseline"
        and (row["scale"], row["seed"], row["gamma"], row["execution_attempt"])
        == (frontier["scale"], frontier["seed"], frontier["gamma"], frontier["execution_attempt"])
    ]
    if len(matches) != 1:
        raise ProtocolGateError("frontier does not have exactly one Gamma-specific baseline")
    return matches[0]


def reject_reuse_path(path: str | Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if any(part.lower() in normalized for part in FORBIDDEN_REUSE_PARTS):
        raise ProtocolGateError("Final Holdout, D1, and D2 artifacts may not be reused")


def run_identity(row: dict[str, Any], config_path: str | Path, *, git_commit_value: str) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config_path, config)
    baseline = paired_baseline(expand_plan(), row) if row["task_type"] == "frontier" else row
    return {
        "schema": SCHEMA,
        "stage": STAGE,
        "execution_attempt": EXECUTION_ATTEMPT,
        "previous_attempt_results_reused": False,
        "git_commit": git_commit_value,
        "base_commit": BASE_COMMIT,
        "config_file_sha256": file_sha256(config_path).upper(),
        "protocol_sha256": config["required_protocol_sha256"],
        "candidate": row["candidate"],
        "candidate_sha256": CANDIDATE_SHA256,
        "solver_parameters": SOLVER_PARAMETERS,
        "scale": row["scale"],
        "seed": row["seed"],
        "gamma": row["gamma"],
        "rho": row["rho"],
        "task_type": row["task_type"],
        "run_key": row["run_key"],
        "run_directory_id": row["run_directory_id"],
        "baseline_run_key": baseline["run_key"],
        "scenario_count": scenario_count(row["scale"], row["gamma"]),
    }


def bind_data_identities(
    identity: dict[str, Any], *, instance_canonical_sha256: str,
    instance_identity_sha256: str, baseline: dict[str, Any], anchor: dict[str, Any]
) -> dict[str, Any]:
    if identity["task_type"] != "frontier":
        raise ProtocolGateError("data identity binding applies to frontier runs")
    expected = {
        "scale": identity["scale"], "seed": identity["seed"], "gamma": identity["gamma"],
        "execution_attempt": identity["execution_attempt"],
        "instance_sha256": instance_canonical_sha256,
        "instance_canonical_sha256": instance_canonical_sha256,
        "instance_identity_sha256": instance_identity_sha256,
    }
    for key, value in expected.items():
        if baseline.get(key) != value:
            raise ProtocolGateError(f"Gamma-specific baseline identity mismatch: {key}")
    if baseline.get("run_key") != identity["baseline_run_key"]:
        raise ProtocolGateError("Gamma-specific baseline run key mismatch")
    if anchor.get("baseline_run_key") != identity["baseline_run_key"]:
        raise ProtocolGateError("anchor is not bound to the Gamma-specific baseline")
    for key, value in expected.items():
        if anchor.get(key) != value:
            raise ProtocolGateError(f"Gamma-specific anchor identity mismatch: {key}")
    for name, value in (
        ("instance_canonical_sha256", instance_canonical_sha256),
        ("instance_identity_sha256", instance_identity_sha256),
        ("anchor_sha256", anchor.get("anchor_sha256")),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ProtocolGateError(f"invalid {name}")
    return {
        **identity,
        "instance_sha256": instance_canonical_sha256,
        "instance_canonical_sha256": instance_canonical_sha256,
        "instance_identity_sha256": instance_identity_sha256,
        "anchor_sha256": anchor["anchor_sha256"],
        "anchor_value_hex": anchor.get("anchor_value_hex"),
    }


def _planned_paths(root: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> list[tuple[str, Path]]:
    planned: list[tuple[str, Path]] = []
    chunk_size = int(config["post_evaluation"]["checkpoint_chunk_size"])
    for scale in SCALES:
        output = root / SCALES[scale]["output_dir"]
        for kind, relative in (
            ("manifest", "manifest.json"), ("manifest_tmp", ".manifest.json.tmp"),
            ("resolved_config", "resolved_config.yaml"), ("resolved_config_tmp", ".resolved_config.yaml.tmp"),
            ("results", "results.csv"), ("results_tmp", ".results.csv.tmp"),
            ("summary", "summary.csv"), ("summary_tmp", ".summary.csv.tmp"),
            ("audit_log", "audit.log"), ("audit_log_tmp", ".audit.log.tmp"),
        ):
            planned.append((f"{scale}_{kind}", (output / relative).resolve()))
    for row in rows:
        output = root / SCALES[row["scale"]]["output_dir"]
        run = output / "runs" / row["run_directory_id"]
        instance = output / "instances" / f"s{row['seed']}_g{row['gamma']}.json"
        planned.append(("instance", instance.resolve()))
        planned.append(("instance_tmp", instance.with_name(f".{instance.name}.tmp").resolve()))
        for kind, relative in (
            ("run", "run.json"),
            ("run_tmp", ".run.json.tmp"),
            ("status", "status.json"),
            ("status_tmp", ".status.json.tmp"),
            ("baseline_checkpoint", "baseline_checkpoint.json"),
            ("baseline_checkpoint_tmp", ".baseline_checkpoint.json.tmp"),
            ("algorithm_checkpoint", "algorithm_checkpoint.json"),
            ("algorithm_checkpoint_tmp", ".algorithm_checkpoint.json.tmp"),
            ("post_final", "post_evaluation/final.json"),
            ("post_final_tmp", "post_evaluation/.final.json.tmp"),
            ("post_index", "post_evaluation/checkpoint/index.json"),
            ("post_index_tmp", "post_evaluation/checkpoint/.index.json.tmp"),
            ("post_chunk", f"post_evaluation/checkpoint/chunk_{math.ceil(scenario_count(row['scale'], row['gamma']) / chunk_size) - 1:05d}.json"),
            ("post_chunk_tmp", f"post_evaluation/checkpoint/.chunk_{math.ceil(scenario_count(row['scale'], row['gamma']) / chunk_size) - 1:05d}.json.tmp"),
        ):
            planned.append((kind, (run / relative).resolve()))
    return planned


def dry_run(
    config_path: str | Path, *, worktree_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config_path, config)
    repository_root = Path(__file__).resolve().parents[1]
    root = Path(worktree_root).resolve() if worktree_root is not None else repository_root
    rows = expand_plan()
    paths = _planned_paths(root, rows, config)
    longest_type, longest = max(paths, key=lambda item: len(str(item[1])))
    outputs = [repository_root / SCALES[scale]["output_dir"] for scale in SCALES]
    per_scale = {
        scale: {
            "baseline": sum(row["scale"] == scale and row["task_type"] == "baseline" for row in rows),
            "frontier": sum(row["scale"] == scale and row["task_type"] == "frontier" for row in rows),
            "total": sum(row["scale"] == scale for row in rows),
            "scenario_counts": SCALES[scale]["scenario_counts"],
        }
        for scale in SCALES
    }
    post_scenarios = {
        scale: len(SEEDS) * sum(SCALES[scale]["scenario_counts"].values()) for scale in SCALES
    }
    return {
        "stage": STAGE,
        "scales": list(SCALES),
        "seeds": SEEDS,
        "gamma": GAMMAS,
        "rho": [RHO],
        "candidate": CANDIDATE,
        "baseline": 30,
        "frontier": 30,
        "total": 60,
        "unique_run_keys": len({row["run_key"] for row in rows}),
        "unique_short_directory_ids": len({row["run_directory_id"] for row in rows}),
        "duplicate_or_collision_count": 0,
        "by_scale": per_scale,
        "algorithm_solver_limit_seconds": 60 * 1800,
        "algorithm_solver_limit_hours": 30,
        "post_evaluation_scenarios": {**post_scenarios, "total": sum(post_scenarios.values())},
        "post_evaluation_solver_limit_seconds": sum(post_scenarios.values()) * 30,
        "post_evaluation_solver_limit_hours_approx": 277,
        "solver_limit_envelopes_are_not_wall_time_predictions": True,
        "longest_windows_absolute_path": str(longest),
        "planned_worktree_root": str(root),
        "longest_windows_path_type": longest_type,
        "longest_windows_path_length": len(str(longest)),
        "windows_path_check": len(str(longest)) < 220,
        "instances_generated": False,
        "solver_called": False,
        "output_dir_exists": any(path.exists() for path in outputs),
        "formal_run_authorized": False,
        "next_authorized_stage": config["next_authorized_stage"],
    }


def manifest_payload(config_path: str | Path, scale: str, *, git_commit_value: str) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config_path, config)
    rows = [row for row in expand_plan() if row["scale"] == scale]
    forward = {row["run_key"]: row["run_directory_id"] for row in rows}
    reverse = {value: key for key, value in forward.items()}
    if len(forward) != 30 or len(reverse) != 30:
        raise ProtocolGateError("manifest run-key mapping collision")
    return {
        "schema": SCHEMA,
        "identity": {
            "stage": STAGE,
            "scale": scale,
            "execution_attempt": EXECUTION_ATTEMPT,
            "git_commit": git_commit_value,
            "base_commit": BASE_COMMIT,
            "config_file_sha256": file_sha256(config_path).upper(),
            "protocol_sha256": config["required_protocol_sha256"],
            "candidate_sha256": CANDIDATE_SHA256,
            "solver_parameters": SOLVER_PARAMETERS,
            "seeds": SEEDS,
            "gamma": GAMMAS,
            "rho": [RHO],
            "previous_attempt_results_reused": False,
        },
        "run_key_to_directory_id": forward,
        "directory_id_to_run_key": reverse,
        "instance_identities": {},
        "baseline_anchors": {},
        "run_identities": {},
    }


def read_json_strict(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolGateError(f"corrupt checkpoint or JSON: {source}") from exc
    if not isinstance(value, dict):
        raise ProtocolGateError(f"JSON object required: {source}")
    return value


def validate_resume_manifest(path: str | Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    source = Path(path)
    manifest = read_json_strict(source)
    if manifest is None:
        if source.parent.exists() and any(source.parent.iterdir()):
            raise ProtocolGateError("existing output lacks a valid manifest")
        return None
    if manifest != expected:
        raise ProtocolGateError("strict resume manifest identity mismatch")
    return manifest


def write_identity_file_once(path: str | Path, value: dict[str, Any], *, label: str) -> None:
    existing = read_json_strict(path)
    if existing is not None:
        if existing != value:
            raise ProtocolGateError(f"{label} already exists with different identity or content")
        return
    atomic_write_json(path, value)


def write_run_status(path: str | Path, identity: dict[str, Any], *, state: str, scientific_status: str) -> None:
    if state not in {"pending", "running", "interrupted", "complete"}:
        raise ProtocolGateError("invalid run status state")
    atomic_write_json(path, {"identity": identity, "state": state, "scientific_status": scientific_status})


def validate_status_file(path: str | Path, identity: dict[str, Any], record: dict[str, Any] | None) -> None:
    status = read_json_strict(path)
    if status is None:
        if record is not None:
            raise ProtocolGateError("committed run is missing status.json")
        return
    if status.get("identity") != identity or status.get("state") not in {"pending", "running", "interrupted", "complete"}:
        raise ProtocolGateError("run status identity or state mismatch")
    if record is not None and (
        status.get("state") != "complete" or status.get("scientific_status") != record.get("scientific_status")
    ):
        raise ProtocolGateError("committed run/status state contradiction")


def validate_run_record(record: dict[str, Any], identity: dict[str, Any]) -> str:
    for key, value in identity.items():
        if record.get(key) != value:
            raise ProtocolGateError(f"run identity mismatch: {key}")
    if record.get("state") == "complete":
        return "skip_committed_result"
    return "resume_incomplete"


def algorithm_checkpoint(identity: dict[str, Any], iterations: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_shas: list[str] = []
    cut_shas: list[str] = []
    previous_iteration = 0
    for item in iterations:
        number = int(item.get("iteration", -1))
        if number != previous_iteration + 1:
            raise ProtocolGateError("algorithm checkpoint iteration order drift")
        previous_iteration = number
        new_scenarios = list(item.get("new_scenario_sha256", []))
        new_cuts = list(item.get("new_cut_sha256", []))
        if len(new_scenarios) > 1 or len(new_cuts) > 1:
            raise ProtocolGateError("at most one scenario and one cut may be added per iteration")
        if set(scenario_shas).intersection(new_scenarios) or set(cut_shas).intersection(new_cuts):
            raise ProtocolGateError("append-only checkpoint contains duplicate SHA")
        scenario_shas.extend(new_scenarios)
        cut_shas.extend(new_cuts)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "identity": identity,
        "iterations": iterations,
        "scenario_sha256_append_only": scenario_shas,
        "cut_sha256_append_only": cut_shas,
    }


def validate_algorithm_checkpoint(value: dict[str, Any], identity: dict[str, Any]) -> None:
    if value.get("schema") != CHECKPOINT_SCHEMA or value.get("identity") != identity:
        raise ProtocolGateError("algorithm checkpoint identity mismatch")
    rebuilt = algorithm_checkpoint(identity, list(value.get("iterations", [])))
    if rebuilt != value:
        raise ProtocolGateError("algorithm checkpoint append-only ledger mismatch")


def write_algorithm_checkpoint(path: str | Path, value: dict[str, Any], identity: dict[str, Any]) -> None:
    validate_algorithm_checkpoint(value, identity)
    atomic_write_json(path, value)


def post_chunk(run_identity_value: dict[str, Any], chunk_index: int, start: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    if chunk_index < 0 or start < 0 or not records:
        raise ProtocolGateError("invalid post-evaluation chunk range")
    core = {
        "schema": POST_SCHEMA,
        "run_identity": run_identity_value,
        "chunk_index": chunk_index,
        "scenario_start": start,
        "scenario_count": len(records),
        "scenario_end_exclusive": start + len(records),
        "records": records,
    }
    return {**core, "chunk_sha256": sha256_value(core)}


def validate_post_evaluation_chunks(chunks: list[dict[str, Any]], identity: dict[str, Any], expected_count: int) -> dict[str, Any]:
    progress = resume_post_evaluation_chunks(chunks, identity, expected_count)
    if progress["next_scenario_index"] != expected_count:
        raise ProtocolGateError("post-evaluation scenario total mismatch")
    return {
        "schema": POST_SCHEMA,
        "run_identity": identity,
        "chunk_count": progress["next_chunk_index"],
        "scenario_count": progress["next_scenario_index"],
        "chunk_sha256": progress["chunk_sha256"],
        "complete": True,
    }


def resume_post_evaluation_chunks(
    chunks: list[dict[str, Any]], identity: dict[str, Any], expected_count: int
) -> dict[str, Any]:
    cumulative = 0
    hashes: list[str] = []
    for index, chunk in enumerate(chunks):
        sha = chunk.get("chunk_sha256")
        core = {key: value for key, value in chunk.items() if key != "chunk_sha256"}
        if sha != sha256_value(core):
            raise ProtocolGateError("post-evaluation chunk SHA mismatch")
        if chunk.get("schema") != POST_SCHEMA or chunk.get("run_identity") != identity:
            raise ProtocolGateError("post-evaluation chunk identity mismatch")
        if chunk.get("chunk_index") != index or chunk.get("scenario_start") != cumulative:
            raise ProtocolGateError("post-evaluation chunk order or cumulative count mismatch")
        if chunk.get("scenario_count") != len(chunk.get("records", [])):
            raise ProtocolGateError("post-evaluation chunk record count mismatch")
        cumulative = int(chunk["scenario_end_exclusive"])
        hashes.append(str(sha))
    if cumulative > expected_count:
        raise ProtocolGateError("post-evaluation scenario total exceeds frozen count")
    return {
        "next_chunk_index": len(chunks),
        "next_scenario_index": cumulative,
        "chunk_sha256": hashes,
        "complete": cumulative == expected_count,
    }


def write_post_chunk(path: str | Path, value: dict[str, Any]) -> None:
    write_identity_file_once(path, value, label="post-evaluation chunk")


def write_post_index(path: str | Path, value: dict[str, Any]) -> None:
    if value.get("schema") != POST_SCHEMA or value.get("complete") is not True:
        raise ProtocolGateError("invalid post-evaluation final index")
    write_identity_file_once(path, value, label="post-evaluation final index")


def classify_scientific_status(result: dict[str, Any]) -> str:
    try:
        separation_bound = float(result.get("final_exact_separation_objective_bound", math.nan))
    except (TypeError, ValueError):
        separation_bound = math.nan
    certified = (
        result.get("algorithm_status") == "optimal"
        and result.get("robust_feasibility_certified") is True
        and result.get("final_exact_separation_performed") is True
        and result.get("final_exact_separation_status") == "optimal"
        and math.isfinite(separation_bound)
        and separation_bound <= 1e-4
        and result.get("post_evaluation_valid") is True
    )
    return "certified_robust_optimal" if certified else "robust_uncertified"


def par2(scientific_status: str, algorithm_runtime: float, limit: float = 1800.0) -> float:
    return penalized_runtime_par2(
        solved_to_tolerance=scientific_status == "certified_robust_optimal",
        runtime=algorithm_runtime,
        time_limit=limit,
    )


RESULT_FIELDS = [
    "run_key", "run_directory_id", "stage", "execution_attempt", "git_commit", "config_file_sha256",
    "protocol_sha256", "candidate_sha256", "scale", "task_type", "seed", "gamma", "rho",
    "candidate", "instance_sha256", "instance_canonical_sha256", "instance_identity_sha256",
    "baseline_run_key", "anchor_sha256", "state", "algorithm_status", "scientific_status",
    "algorithm_runtime", "master_runtime", "separation_runtime",
    "post_evaluation_wall_runtime", "total_wall_runtime", "penalized_runtime_par2", "baseline_robust_cost",
    "cost_budget", "actual_robust_cost", "actual_price_of_fairness", "objective_t",
    "robust_minimum_fill_rate", "wminfr", "minimum_weighted_mean_fill_rate", "inventory",
    "opened_warehouses", "iterations", "scenario_block_count", "certified_farkas_cut_count",
]


def validate_result_row(row: dict[str, Any]) -> None:
    missing = [field for field in RESULT_FIELDS if field not in row]
    if missing:
        raise ProtocolGateError(f"results CSV fields missing: {missing}")
    if row["stage"] != STAGE or int(row["execution_attempt"]) != EXECUTION_ATTEMPT:
        raise ProtocolGateError("results CSV stage or attempt identity drift")
    if int(row["gamma"]) not in GAMMAS or int(row["seed"]) not in SEEDS or row["scale"] not in SCALES:
        raise ProtocolGateError("results CSV matrix identity drift")
    if row["run_directory_id"] != run_directory_id(str(row["run_key"])):
        raise ProtocolGateError("results CSV run-key directory mapping drift")
    planned = next((item for item in expand_plan() if item["run_key"] == row["run_key"]), None)
    if planned is None or any(row[name] != planned[name] for name in ("scale", "task_type", "seed", "gamma", "rho", "candidate")):
        raise ProtocolGateError("results CSV run identity is not in the frozen plan")
    if row["task_type"] == "frontier" and row["baseline_run_key"] != paired_baseline(expand_plan(), planned)["run_key"]:
        raise ProtocolGateError("results CSV baseline identity drift")
    for name in ("algorithm_runtime", "post_evaluation_wall_runtime", "total_wall_runtime", "penalized_runtime_par2"):
        try:
            value = float(row[name])
        except (TypeError, ValueError) as exc:
            raise ProtocolGateError(f"results CSV non-finite field: {name}") from exc
        if not math.isfinite(value):
            raise ProtocolGateError(f"results CSV non-finite field: {name}")
    if row["task_type"] == "frontier":
        for name in (
            "objective_t", "robust_minimum_fill_rate", "wminfr", "minimum_weighted_mean_fill_rate",
            "actual_robust_cost", "actual_price_of_fairness", "inventory", "opened_warehouses",
            "iterations", "scenario_block_count", "certified_farkas_cut_count",
        ):
            try:
                value = float(row[name])
            except (TypeError, ValueError) as exc:
                raise ProtocolGateError(f"results CSV non-finite field: {name}") from exc
            if not math.isfinite(value):
                raise ProtocolGateError(f"results CSV non-finite field: {name}")
        if not math.isclose(float(row["robust_minimum_fill_rate"]), 1.0 - float(row["objective_t"]), abs_tol=1e-12):
            raise ProtocolGateError("robust_minimum_fill_rate must equal 1-T")


def write_results(path: str | Path, rows: list[dict[str, Any]]) -> None:
    keys = [str(row.get("run_key")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ProtocolGateError("duplicate results CSV run key")
    for row in rows:
        validate_result_row(row)
    atomic_write_csv(path, rows, RESULT_FIELDS)


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        validate_result_row(row)
    summaries: list[dict[str, Any]] = []
    for scale in SCALES:
        scale_rows = [row for row in rows if row["scale"] == scale]
        summaries.append({
            "scale": scale,
            "expected_runs": 30,
            "completed_runs": sum(row.get("state") == "complete" for row in scale_rows),
            "baseline_runs": sum(row["task_type"] == "baseline" for row in scale_rows),
            "frontier_runs": sum(row["task_type"] == "frontier" for row in scale_rows),
            "certified_frontiers": sum(
                row["task_type"] == "frontier" and row["scientific_status"] == "certified_robust_optimal"
                for row in scale_rows
            ),
        })
    return summaries


def write_summary(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fields = ["scale", "expected_runs", "completed_runs", "baseline_runs", "frontier_runs", "certified_frontiers"]
    atomic_write_csv(path, summary_rows(rows), fields)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def validate_authorization(path: str | Path, config_path: str | Path, root: Path) -> dict[str, Any]:
    authorization = read_json_strict(path)
    if authorization is None:
        raise ProtocolGateError("formal_run_not_authorized: reviewed authorization file is required")
    expected = {
        "schema": AUTHORIZATION_SCHEMA,
        "schema_version": 2,
        "stage": STAGE,
        "formal_run_authorized": True,
        "execution_attempt": EXECUTION_ATTEMPT,
        "config_file_sha256": file_sha256(config_path).upper(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "candidate": CANDIDATE,
        "scales": list(SCALES),
        "seeds": SEEDS,
        "gamma": GAMMAS,
        "rho": [RHO],
        "output_directories": [value["output_dir"] for value in SCALES.values()],
        "previous_attempt_results_reused": False,
        "next_authorized_stage": "fairness_hybrid_gamma_sensitivity_attempt2_formal_run_only",
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise ProtocolGateError(f"formal_run_not_authorized: authorization identity mismatch: {key}")
    protocol_commit = authorization.get("protocol_merge_commit")
    if not isinstance(protocol_commit, str) or len(protocol_commit) != 40:
        raise ProtocolGateError("formal_run_not_authorized: invalid protocol merge commit")
    tracked = _git(root, "ls-files", "--error-unmatch", str(Path(path).resolve().relative_to(root)).replace("\\", "/"))
    if tracked.returncode:
        raise ProtocolGateError("formal_run_not_authorized: authorization file must be Git tracked")
    ancestor = _git(root, "merge-base", "--is-ancestor", protocol_commit, "HEAD")
    if ancestor.returncode:
        raise ProtocolGateError("formal_run_not_authorized: protocol merge commit is not an ancestor of HEAD")
    return authorization


def formal_git_gate(root: Path) -> str:
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    head = _git(root, "rev-parse", "HEAD")
    main = _git(root, "rev-parse", "origin/main")
    symbolic = _git(root, "symbolic-ref", "-q", "HEAD")
    if status.returncode or status.stdout.strip():
        raise ProtocolGateError("formal_run_not_authorized: worktree is not clean")
    if head.returncode or main.returncode or head.stdout.strip() != main.stdout.strip():
        raise ProtocolGateError("formal_run_not_authorized: HEAD is not current origin/main")
    if symbolic.returncode == 0:
        raise ProtocolGateError("formal_run_not_authorized: formal worktree must be detached")
    return head.stdout.strip()


def pre_run_seed_gate(root: Path) -> dict[str, Any]:
    from .fairness_hybrid_gamma_sensitivity_audit import audit_repository_seed_access

    report = audit_repository_seed_access(root, excluded_untracked_roots=ATTEMPT1_OUTPUT_DIRS)
    prefixes = tuple(value["output_dir"].replace("\\", "/") + "/" for value in SCALES.values())
    unexpected = []
    for category in ("generated_instance_evidence", "solved_run_evidence", "formal_result_access_evidence"):
        unexpected.extend(item for item in report[category] if not str(item["path"]).replace("\\", "/").startswith(prefixes))
    if unexpected:
        raise ProtocolGateError(f"formal_run_not_authorized: reserved seed access evidence: {unexpected}")
    return report


def _scale_config(config: dict[str, Any], scale: str, gamma: int) -> dict[str, Any]:
    value = deepcopy(config)
    value.update({"scale": scale, "gamma_value": gamma, "scenario_count": scenario_count(scale, gamma)})
    return value


def gamma_baseline_template(template: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gamma = int(config["gamma_value"])
    if gamma not in GAMMAS:
        raise ProtocolGateError("baseline Gamma is outside the frozen sensitivity matrix")
    resolved = deepcopy(template)
    resolved.update({
        "gamma_target": gamma,
        "gamma_schedule": [gamma],
        "gamma_continuation_enabled": False,
        "exact_scenarios": True,
        "max_scenarios": scenario_count(str(config["scale"]), gamma),
        "time_limit": float(config["baseline_time_limit_seconds"]),
        "baseline_time_limit": float(config["baseline_time_limit_seconds"]),
    })
    return resolved


def production_dependencies() -> GammaDependencies:
    # Deliberately lazy: dry-run and every authorization gate complete before importing solver code.
    from .fairness_hybrid_ccg_benders import (
        initial_upper_bound_expected_identity,
        solve_certified_hybrid_scenario_benders_fairness,
    )
    from .fairness_hybrid_ccg_benders_runner import _hybrid_certified_anchor
    from .fairness_large_final_remediation_runner import (
        _configure_solver_parameters,
        _production_generate_instance,
        _scale_template,
    )
    from .benders import solve_benders
    from .experiment_suite import _apply_selected_parameters, _apply_variant_config, _base_config
    from .fairness_post_evaluation import checkpointed_fairness_post_evaluation
    from .instance import InventoryInstance

    def solve_baseline(
        config: dict[str, Any], instance: Any, seed: int, solver: dict[str, Any],
    ) -> dict[str, Any]:
        _configure_solver_parameters(solver)
        selected = _apply_selected_parameters(_scale_template(config))
        resolved = gamma_baseline_template(selected, config)
        base = _base_config(resolved, str(config["scale"]), seed)
        variant = dict(resolved.get("variant_settings", {}).get("joint_v1_core_point_strengthened", {}))
        method, _flags, method_config = _apply_variant_config(base, "proposed_adaptive_benders", variant)
        result = solve_benders(method_config, instance, method)
        payload = result.summary_dict()
        payload["iteration_log"] = result.iteration_log
        payload["gamma"] = int(config["gamma_value"])
        return payload

    def solve_frontier(
        config: dict[str, Any], instance: Any, baseline: dict[str, Any], anchor: dict[str, Any],
        common: dict[str, Any], checkpoint: Path, solver: dict[str, Any], row: dict[str, Any],
    ) -> dict[str, Any]:
        expected_identity = initial_upper_bound_expected_identity(common, anchor)
        expected_identity.update({
            "instance_canonical_sha256": common["instance_canonical_sha256"],
            "gamma": common["gamma"],
            "execution_attempt": common["execution_attempt"],
        })
        result = solve_certified_hybrid_scenario_benders_fairness(
            instance, baseline_record=baseline, anchor=anchor,
            expected_identity=expected_identity,
            solver_parameters=solver, rho=RHO, gamma=int(row["gamma"]),
            max_iterations=int(config["max_iterations"]),
            time_limit=float(config["algorithm_time_limit_seconds"]), tol=float(config["tol"]),
            feasibility_tolerance=float(solver["FeasibilityTol"]), checkpoint_path=checkpoint,
            checkpoint_identity={"run_key": row["run_key"], **deepcopy(common)},
            execution_protocol_sha256=EXPECTED_PROTOCOL_SHA256, output_flag=False,
        )
        return result.to_dict()

    def post_evaluate(
        config: dict[str, Any], instance: Any, result: dict[str, Any], anchor: dict[str, Any],
        identity: dict[str, Any], post_root: Path, row: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        evaluation, timing = checkpointed_fairness_post_evaluation(
            instance, root=post_root, run_key=row["run_key"],
            config_sha256_value=identity["config_file_sha256"], git_commit=identity["git_commit"],
            baseline_anchor_sha256=anchor["anchor_sha256"], y_values=result["y_values"],
            x_values=result["x_values"], t_value=float(result["objective_t"]),
            baseline_cost=float(anchor["value"]), rho=RHO, gamma=int(row["gamma"]),
            max_scenarios=scenario_count(str(row["scale"]), int(row["gamma"])),
            per_scenario_time_limit=float(config["post_evaluation"]["time_limit_per_scenario_seconds"]),
            tolerance=1e-7, chunk_size=int(config["post_evaluation"]["checkpoint_chunk_size"]),
            resume_count=0, output_flag=False, run_execution_attempt=EXECUTION_ATTEMPT,
            post_evaluation_pipeline_generation=int(config["post_evaluation"]["pipeline_generation"]),
        )
        return evaluation.to_dict(), {
            "post_evaluation_solver_runtime": timing.solver_runtime,
            "post_evaluation_wall_runtime": timing.wall_runtime,
            "aggregation_runtime": timing.aggregation_runtime,
            "checkpoint_io_runtime": timing.checkpoint_io_runtime,
        }

    return GammaDependencies(
        generate_instance=_production_generate_instance,
        serialize_instance=lambda value: value.to_dict(),
        deserialize_instance=InventoryInstance.from_dict,
        solve_baseline=solve_baseline,
        make_anchor=_hybrid_certified_anchor,
        solve_frontier=solve_frontier,
        post_evaluate=post_evaluate,
        configure_solver=_configure_solver_parameters,
    )


def _frontier_status(result: dict[str, Any], evaluation: dict[str, Any] | None, expected_count: int, tol: float) -> str:
    log = result.get("iteration_log")
    final = log[-1] if isinstance(log, list) and log and isinstance(log[-1], dict) else {}
    bound = final.get("separation_objective_bound")
    certified = (
        result.get("status") == "optimal" and result.get("gap") is not None
        and math.isfinite(float(result["gap"])) and float(result["gap"]) <= tol
        and result.get("metadata", {}).get("full_separation_objective_bound_required") is True
        and final.get("final_exact_separation_performed") is True
        and final.get("robust_feasibility_certified") is True and final.get("master_status") == "optimal"
        and bound is not None and math.isfinite(float(bound)) and float(bound) <= SOLVER_PARAMETERS["FeasibilityTol"]
    )
    if not certified:
        return "robust_uncertified"
    if not isinstance(evaluation, dict) or evaluation.get("valid") is not True:
        return "invalid_post_evaluation"
    if evaluation.get("errors") not in (None, []) or evaluation.get("objective_t_consistent") is False:
        return "invalid_post_evaluation"
    if evaluation.get("scenario_count") != expected_count:
        return "invalid_post_evaluation"
    return "certified_robust_optimal"


def _result_projection(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result", {})
    post = result.get("post_evaluation") or {}
    frontier = record["task_type"] == "frontier"
    zeros = 0.0
    row = {field: "NOT_APPLICABLE" for field in RESULT_FIELDS}
    row.update({field: record.get(field, row[field]) for field in RESULT_FIELDS})
    row.update({
        "algorithm_runtime": float(result.get("algorithm_runtime", result.get("runtime", 0.0))),
        "master_runtime": float(result.get("master_runtime", 0.0)),
        "separation_runtime": float(result.get("separation_runtime", 0.0)),
        "post_evaluation_wall_runtime": float(result.get("post_evaluation_wall_runtime", 0.0)),
        "total_wall_runtime": float(result.get("total_wall_runtime", result.get("runtime", 0.0))),
        "penalized_runtime_par2": float(result.get("penalized_runtime_par2", 0.0)),
        "baseline_robust_cost": float(record.get("baseline_robust_cost", result.get("upper_bound", 0.0))),
        "cost_budget": float(record.get("cost_budget", 0.0)),
        "actual_robust_cost": float(post.get("actual_robust_cost", zeros)),
        "actual_price_of_fairness": float(post.get("actual_price_of_fairness", zeros)),
        "objective_t": float(result.get("objective_t", zeros)),
        "robust_minimum_fill_rate": float(result.get("robust_minimum_fill_rate", zeros)),
        "wminfr": float(post.get("wminfr", zeros)),
        "minimum_weighted_mean_fill_rate": float(post.get("minimum_weighted_mean_fill_rate", zeros)),
        "inventory": float(sum(float(value) for value in result.get("x_values", {}).values())) if frontier else zeros,
        "opened_warehouses": int(sum(float(value) >= 0.5 for value in result.get("y_values", {}).values())) if frontier else 0,
        "iterations": int(result.get("iterations", len(result.get("iteration_log", [])))) if frontier else 0,
        "scenario_block_count": int(result.get("metadata", {}).get("committed_scenario_count", 0)) if frontier else 0,
        "certified_farkas_cut_count": int(result.get("cuts", 0)) if frontier else 0,
    })
    return row


def aggregate_output(output: Path, rows: list[dict[str, Any]], *, require_complete: bool = False) -> list[dict[str, Any]]:
    result_rows: list[dict[str, Any]] = []
    for planned in rows:
        record = read_json_strict(output / "runs" / planned["run_directory_id"] / "run.json")
        if record is not None:
            result_rows.append(_result_projection(record))
    keys = [str(row["run_key"]) for row in result_rows]
    if len(keys) != len(set(keys)):
        raise ProtocolGateError("duplicate results CSV run key")
    expected = {row["run_key"] for row in rows}
    if not set(keys).issubset(expected) or (require_complete and set(keys) != expected):
        raise ProtocolGateError("results CSV does not exactly match the frozen run plan")
    write_results(output / "results.csv", result_rows)
    write_summary(output / "summary.csv", result_rows)
    return result_rows


def _baseline_checkpoint(identity: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {"schema": BASELINE_CHECKPOINT_SCHEMA, "identity": identity, "result": result}


def _load_baseline_checkpoint(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    value = read_json_strict(path)
    if value is None:
        return None
    if value.get("schema") != BASELINE_CHECKPOINT_SCHEMA or value.get("identity") != identity:
        raise ProtocolGateError("baseline checkpoint identity mismatch")
    if not isinstance(value.get("result"), dict):
        raise ProtocolGateError("baseline checkpoint result missing")
    return value["result"]


def _run_scale(
    config_path: Path, config: dict[str, Any], scale: str, rows: list[dict[str, Any]], deps: GammaDependencies,
    git_commit_value: str, *, failure_injector: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = root / SCALES[scale]["output_dir"]
    expected_manifest = manifest_payload(config_path, scale, git_commit_value=git_commit_value)
    manifest_path = output / "manifest.json"
    existing = read_json_strict(manifest_path)
    if output.exists() and existing is None:
        raise ProtocolGateError(f"existing {scale} output lacks a valid manifest")
    if existing is not None:
        for key in ("schema", "identity", "run_key_to_directory_id", "directory_id_to_run_key"):
            if existing.get(key) != expected_manifest[key]:
                raise ProtocolGateError(f"strict resume {scale} manifest identity mismatch")
        resolved = output / "resolved_config.yaml"
        if not resolved.exists() or load_config(resolved) != config:
            raise ProtocolGateError(f"strict resume {scale} resolved config mismatch")
        manifest = existing
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_yaml(output / "resolved_config.yaml", config)
        manifest = expected_manifest
        atomic_write_json(manifest_path, manifest)
    audit_identity = {
        "schema": "fairness_hybrid_gamma_sensitivity_audit_log_v2",
        "identity": expected_manifest["identity"],
        "formal_worktree_root": str(root),
        "solver_limit_envelopes_are_not_wall_time_predictions": True,
    }
    existing_audit = read_json_strict(output / "audit.log")
    if existing_audit is not None and existing_audit.get("identity") != audit_identity["identity"]:
        raise ProtocolGateError(f"strict resume {scale} audit log identity mismatch")
    atomic_write_json(output / "audit.log", existing_audit or audit_identity)

    completed: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for gamma in GAMMAS:
            cell = [row for row in rows if row["seed"] == seed and row["gamma"] == gamma]
            baseline_row = next(row for row in cell if row["task_type"] == "baseline")
            frontier_row = next(row for row in cell if row["task_type"] == "frontier")
            scale_config = _scale_config(config, scale, gamma)
            instance_path = output / "instances" / f"s{seed}_g{gamma}.json"
            stored_archive = read_json_strict(instance_path)
            frozen_instance_identity = {
                "stage": STAGE, "scale": scale, "seed": seed, "gamma": gamma,
                "execution_attempt": EXECUTION_ATTEMPT, "git_commit": git_commit_value,
                "config_file_sha256": file_sha256(config_path).upper(),
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            }
            if stored_archive is None:
                instance = deps.generate_instance(scale_config, seed)
                serialized = deps.serialize_instance(instance)
                archive = instance_archive_payload(frozen_instance_identity, serialized)
                serialized, instance_identity, instance_canonical_sha, instance_identity_sha = (
                    validate_instance_archive(archive, frozen_instance_identity)
                )
                atomic_write_json(instance_path, archive)
                instance_archive_file_sha = file_sha256(instance_path).upper()
            else:
                serialized, instance_identity, instance_canonical_sha, instance_identity_sha = (
                    validate_instance_archive(stored_archive, frozen_instance_identity)
                )
                instance = deps.deserialize_instance(serialized)
                instance_archive_file_sha = file_sha256(instance_path).upper()
            restored_instance_sha = sha256_value(deps.serialize_instance(instance))
            if restored_instance_sha != instance_canonical_sha:
                raise ProtocolGateError("Gamma-specific instance canonical round-trip mismatch")
            baseline_identity = run_identity(baseline_row, config_path, git_commit_value=git_commit_value)
            baseline_identity.update({
                "instance_sha256": instance_canonical_sha,
                "instance_canonical_sha256": instance_canonical_sha,
                "instance_identity_sha256": instance_identity_sha,
            })
            baseline_root = output / "runs" / baseline_row["run_directory_id"]
            baseline_record = read_json_strict(baseline_root / "run.json")
            validate_status_file(baseline_root / "status.json", baseline_identity, baseline_record)
            if baseline_record is None:
                checkpoint_path = baseline_root / "baseline_checkpoint.json"
                payload = _load_baseline_checkpoint(checkpoint_path, baseline_identity)
                if payload is None:
                    write_run_status(baseline_root / "status.json", baseline_identity, state="running", scientific_status="pending")
                    payload = deps.solve_baseline(scale_config, instance, seed, deepcopy(SOLVER_PARAMETERS))
                    atomic_write_json(checkpoint_path, _baseline_checkpoint(baseline_identity, payload))
                    if failure_injector:
                        failure_injector("after_baseline_checkpoint", deepcopy(baseline_row))
                solved = (
                    payload.get("status") == "optimal" and payload.get("valid_UB") is True
                    and math.isfinite(float(payload.get("upper_bound", math.nan)))
                    and math.isfinite(float(payload.get("gap", math.nan))) and float(payload["gap"]) <= float(config["tol"])
                )
                runtime = float(payload.get("runtime", 0.0))
                payload.update({
                    "algorithm_runtime": runtime, "post_evaluation_wall_runtime": 0.0,
                    "total_wall_runtime": runtime,
                    "penalized_runtime_par2": par2("certified_robust_optimal" if solved else "robust_uncertified", runtime),
                })
                baseline_record = {
                    **baseline_identity, "candidate": "baseline", "state": "complete",
                    "algorithm_status": payload.get("status"),
                    "scientific_status": "certified_robust_optimal" if solved else "robust_uncertified",
                    "solved_to_tolerance": solved, "result": payload,
                }
                atomic_write_json(baseline_root / "run.json", baseline_record)
                write_run_status(baseline_root / "status.json", baseline_identity, state="complete", scientific_status=baseline_record["scientific_status"])
            else:
                validate_run_record(baseline_record, baseline_identity)
            completed[baseline_row["run_key"]] = baseline_record
            common = {
                "instance_sha256": instance_canonical_sha,
                "instance_canonical_sha256": instance_canonical_sha,
                "instance_identity_sha256": instance_identity_sha,
                "seed": seed, "gamma": gamma, "scale": scale,
                "stage": STAGE, "execution_attempt": EXECUTION_ATTEMPT, "git_commit": git_commit_value,
                "config_file_sha256": file_sha256(config_path).upper(),
                "resolved_config_file_sha256": file_sha256(config_path).upper(),
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256, "candidate_sha256": CANDIDATE_SHA256,
                "baseline_run_key": baseline_row["run_key"],
            }
            anchor = deps.make_anchor(baseline_record, common_identity=common, tolerance=float(config["tol"]))
            manifest_instance_identity = {
                **instance_identity, "instance_identity_sha256": instance_identity_sha,
                "instance_archive_file_sha256": instance_archive_file_sha,
            }
            cell_key = f"s{seed}_g{gamma}"
            record_manifest_identity(
                manifest, "instance_identities", cell_key, manifest_instance_identity,
            )
            record_manifest_identity(manifest, "baseline_anchors", cell_key, anchor)

            frontier_identity = bind_data_identities(
                run_identity(frontier_row, config_path, git_commit_value=git_commit_value),
                instance_canonical_sha256=instance_canonical_sha,
                instance_identity_sha256=instance_identity_sha,
                baseline=baseline_record, anchor=anchor,
            )
            frontier_identity.update({
                "post_evaluation_pipeline_generation": int(config["post_evaluation"]["pipeline_generation"]),
                "run_execution_attempt": EXECUTION_ATTEMPT,
            })
            record_manifest_identity(
                manifest, "run_identities", frontier_row["run_key"], frontier_identity,
            )
            atomic_write_json(manifest_path, manifest)
            frontier_root = output / "runs" / frontier_row["run_directory_id"]
            frontier_record = read_json_strict(frontier_root / "run.json")
            validate_status_file(frontier_root / "status.json", frontier_identity, frontier_record)
            if frontier_record is not None:
                validate_run_record(frontier_record, frontier_identity)
                completed[frontier_row["run_key"]] = frontier_record
                continue
            write_run_status(frontier_root / "status.json", frontier_identity, state="running", scientific_status="pending")
            started = time.perf_counter()
            result = deps.solve_frontier(
                scale_config, instance, baseline_record, anchor, common,
                frontier_root / "algorithm_checkpoint.json", deepcopy(SOLVER_PARAMETERS), frontier_row,
            )
            algorithm_runtime = float(result.get("runtime", time.perf_counter() - started))
            log = result.get("iteration_log")
            final = log[-1] if isinstance(log, list) and log else {}
            algorithm_certified = (
                result.get("status") == "optimal" and result.get("metadata", {}).get("robust_feasibility_certified") is True
                and final.get("final_exact_separation_performed") is True
            )
            evaluation = None
            timing = {"post_evaluation_wall_runtime": 0.0, "aggregation_runtime": 0.0, "checkpoint_io_runtime": 0.0}
            if algorithm_certified:
                evaluation, timing = deps.post_evaluate(
                    scale_config, instance, result, anchor, frontier_identity, frontier_root / "post_evaluation", frontier_row,
                )
            scientific = _frontier_status(result, evaluation, scenario_count(scale, gamma), float(config["tol"]))
            result.update(timing)
            result.update({
                "post_evaluation": evaluation, "algorithm_runtime": algorithm_runtime,
                "post_evaluation_wall_runtime": float(timing.get("post_evaluation_wall_runtime", 0.0)),
                "total_wall_runtime": algorithm_runtime + float(timing.get("post_evaluation_wall_runtime", 0.0))
                    + float(timing.get("aggregation_runtime", 0.0)) + float(timing.get("checkpoint_io_runtime", 0.0)),
                "penalized_runtime_par2": par2(scientific, algorithm_runtime),
            })
            frontier_record = {
                **frontier_identity, "candidate": CANDIDATE, "state": "complete",
                "algorithm_status": result.get("status"), "scientific_status": scientific,
                "solved_to_tolerance": scientific == "certified_robust_optimal",
                "baseline_robust_cost": float(anchor["value"]),
                "cost_budget": (1.0 + RHO) * float(anchor["value"]), "result": result,
            }
            atomic_write_json(frontier_root / "run.json", frontier_record)
            write_run_status(frontier_root / "status.json", frontier_identity, state="complete", scientific_status=scientific)
            completed[frontier_row["run_key"]] = frontier_record
            aggregate_output(output, rows)
            if scientific == "invalid_post_evaluation":
                raise ProtocolGateError("Gamma sensitivity stopped fail closed: invalid post-evaluation")

    aggregate_output(output, rows, require_complete=True)
    manifest.update({
        "completed_run_count": len(completed),
        "baseline_certified_count": sum(r["task_type"] == "baseline" and r["scientific_status"] == "certified_robust_optimal" for r in completed.values()),
        "frontier_certified_count": sum(r["task_type"] == "frontier" and r["scientific_status"] == "certified_robust_optimal" for r in completed.values()),
    })
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(output / "audit.log", {
        **audit_identity, "completed_run_count": manifest["completed_run_count"],
        "baseline_certified_count": manifest["baseline_certified_count"],
        "frontier_certified_count": manifest["frontier_certified_count"],
    })
    return manifest


def run_sensitivity(
    config_path: str | Path, *, resume: bool, authorization_file: str | Path | None = None,
    dependencies: GammaDependencies | None = None, test_authorization: bool = False,
    test_git_root: Path | None = None,
    failure_injector: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not resume:
        raise ProtocolGateError("GAMMA_SENSITIVITY requires strict --resume; --overwrite is unsupported")
    path = Path(config_path)
    config = load_config(path)
    validate_config(path, config)
    if dependencies is not None and not test_authorization:
        raise ProtocolGateError("dependency substitution requires test_authorization")
    if test_git_root is not None and not test_authorization:
        raise ProtocolGateError("test Git root requires test_authorization")
    root = Path(__file__).resolve().parents[1]
    if test_authorization:
        git_commit_value = formal_git_gate(test_git_root) if test_git_root is not None else "T" * 40
    else:
        if authorization_file is None:
            raise ProtocolGateError("formal_run_not_authorized: reviewed authorization file is required")
        git_commit_value = formal_git_gate(root)
        validate_authorization(authorization_file, path, root)
        pre_run_seed_gate(root)
    rows = expand_plan()
    paths = _planned_paths(root, rows, config)
    maximum = max(len(str(value)) for _, value in paths)
    if maximum > 220:
        raise ProtocolGateError(f"Windows path portability limit exceeded: {maximum}")
    for scale in SCALES:
        output = root / SCALES[scale]["output_dir"]
        manifest = read_json_strict(output / "manifest.json")
        if output.exists() and manifest is None:
            raise ProtocolGateError(f"existing {scale} output lacks a valid manifest")
        if manifest is not None:
            expected = manifest_payload(path, scale, git_commit_value=git_commit_value)
            for key in ("schema", "identity", "run_key_to_directory_id", "directory_id_to_run_key"):
                if manifest.get(key) != expected[key]:
                    raise ProtocolGateError(f"strict resume {scale} manifest identity mismatch")
            resolved = output / "resolved_config.yaml"
            if not resolved.exists() or load_config(resolved) != config:
                raise ProtocolGateError(f"strict resume {scale} resolved config mismatch")
    deps = dependencies or production_dependencies()
    deps.configure_solver(deepcopy(SOLVER_PARAMETERS))
    manifests = {}
    for scale in SCALES:
        manifests[scale] = _run_scale(
            path, config, scale, [row for row in rows if row["scale"] == scale], deps,
            git_commit_value, failure_injector=failure_injector,
        )
    return {
        "stage": STAGE, "completed_run_count": sum(m["completed_run_count"] for m in manifests.values()),
        "baseline_certified_count": sum(m["baseline_certified_count"] for m in manifests.values()),
        "frontier_certified_count": sum(m["frontier_certified_count"] for m in manifests.values()),
        "manifests": manifests,
    }


def formal_run(config_path: str | Path, *, resume: bool, authorization_file: str | Path | None = None) -> dict[str, Any]:
    return run_sensitivity(config_path, resume=resume, authorization_file=authorization_file)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorization-file", type=Path)
    args = parser.parse_args(argv)
    if args.stage != STAGE:
        raise ProtocolGateError("only GAMMA_SENSITIVITY is accepted by this runner")
    if args.dry_run:
        print(json.dumps(dry_run(args.config), indent=2, sort_keys=True))
        return 0
    print(json.dumps(formal_run(args.config, resume=args.resume, authorization_file=args.authorization_file), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
