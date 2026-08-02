from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml

from .experiment_protocol import atomic_write_csv, atomic_write_json, file_sha256, penalized_runtime_par2


STAGE = "GAMMA_SENSITIVITY"
SCHEMA = "fairness_hybrid_gamma_sensitivity_manifest_v1"
CHECKPOINT_SCHEMA = "fairness_hybrid_gamma_sensitivity_algorithm_checkpoint_v1"
POST_SCHEMA = "fairness_hybrid_gamma_sensitivity_post_evaluation_v1"
EXECUTION_ATTEMPT = 1
BASE_COMMIT = "827b1373702972ae780231899afe17cf6eff0d53"
CANDIDATE = "certified_hybrid_scenario_benders_fairness"
CANDIDATE_SHA256 = "8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0"
EXPECTED_CONFIG_SHA256 = "F09074EA83E91514A0D1DFBF49F31EA24D2AABD9C94530B5F6A7E5504DA4585C"
EXPECTED_PROTOCOL_SHA256 = "D8CEA1249E92E9594D8308F5617D8767F23FFF7472644012DD4FD031CC7EF245"
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
        "output_dir": "experiments/results_fh_gamma/ml_a1",
    },
    "large": {
        "num_regions": 12,
        "num_products": 8,
        "demand_components": 96,
        "scenario_counts": {0: 1, 1: 97, 2: 4657},
        "output_dir": "experiments/results_fh_gamma/lg_a1",
    },
}
FORBIDDEN_REUSE_PARTS = (
    "results_fairness_hybrid_final_holdout",
    "fairness_hybrid_ccg_benders_d1",
    "fairness_hybrid_ccg_benders_d2",
)


class ProtocolGateError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


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
        "schema_version": 1,
        "execution_attempt": EXECUTION_ATTEMPT,
        "previous_attempt_results_reused": False,
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
    identity: dict[str, Any], *, instance_sha256: str, baseline: dict[str, Any], anchor: dict[str, Any]
) -> dict[str, Any]:
    if identity["task_type"] != "frontier":
        raise ProtocolGateError("data identity binding applies to frontier runs")
    for key in ("scale", "seed", "gamma", "execution_attempt"):
        if baseline.get(key) != identity[key]:
            raise ProtocolGateError(f"Gamma-specific baseline identity mismatch: {key}")
    if baseline.get("run_key") != identity["baseline_run_key"]:
        raise ProtocolGateError("Gamma-specific baseline run key mismatch")
    if anchor.get("baseline_run_key") != identity["baseline_run_key"]:
        raise ProtocolGateError("anchor is not bound to the Gamma-specific baseline")
    for name, value in (("instance_sha256", instance_sha256), ("anchor_sha256", anchor.get("anchor_sha256"))):
        if not isinstance(value, str) or len(value) != 64:
            raise ProtocolGateError(f"invalid {name}")
    return {
        **identity,
        "instance_sha256": instance_sha256,
        "anchor_sha256": anchor["anchor_sha256"],
        "anchor_value_hex": anchor.get("anchor_value_hex"),
    }


def _planned_paths(root: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> list[tuple[str, Path]]:
    planned: list[tuple[str, Path]] = []
    chunk_size = int(config["post_evaluation"]["checkpoint_chunk_size"])
    for row in rows:
        output = root / SCALES[row["scale"]]["output_dir"]
        run = output / "runs" / row["run_directory_id"]
        instance = output / "instances" / f"s{row['seed']}_g{row['gamma']}.json"
        planned.append(("instance", instance.resolve()))
        for kind, relative in (
            ("run", "run.json"),
            ("status", "status.json"),
            ("algorithm_checkpoint", "algorithm_checkpoint.json"),
            ("post_index", "post/checkpoint/index.json"),
            ("post_chunk_tmp", f"post/checkpoint/.chunk_{math.ceil(scenario_count(row['scale'], row['gamma']) / chunk_size) - 1:05d}.json.tmp"),
        ):
            planned.append((kind, (run / relative).resolve()))
    return planned


def dry_run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config_path, config)
    root = Path(__file__).resolve().parents[1]
    rows = expand_plan()
    paths = _planned_paths(root, rows, config)
    longest_type, longest = max(paths, key=lambda item: len(str(item[1])))
    outputs = [root / SCALES[scale]["output_dir"] for scale in SCALES]
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
    "candidate", "instance_sha256", "baseline_run_key", "anchor_sha256", "state", "algorithm_status", "scientific_status",
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


def formal_run(config_path: str | Path, *, resume: bool) -> None:
    if not resume:
        raise ProtocolGateError("GAMMA_SENSITIVITY requires strict --resume; --overwrite is unsupported")
    config = load_config(config_path)
    validate_config(config_path, config)
    if config["formal_run_authorized"] is not True:
        raise ProtocolGateError("formal_run_not_authorized: only pre-run audit is authorized")
    raise ProtocolGateError("formal_run_not_authorized: production execution requires a separate reviewed authorization")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.stage != STAGE:
        raise ProtocolGateError("only GAMMA_SENSITIVITY is accepted by this runner")
    if args.dry_run:
        print(json.dumps(dry_run(args.config), indent=2, sort_keys=True))
        return 0
    formal_run(args.config, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
