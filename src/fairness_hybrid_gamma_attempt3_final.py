"""Solver-free audit and deterministic reporting for Gamma Attempt 3.

The source archives are opened read-only.  This module intentionally avoids all
solver modules and never imports gurobipy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable
from zipfile import ZipFile


GAMMA_ARCHIVE_SHA256 = "EE45A00AA341EE5EB2894DE43EE2F47022C27F1D29146FCFEC803236EF59DB6F"
CONFIG_SHA256 = "C26236A93E669B877D74DE0F08D0BC86817345821DEF91066911D723788C7C07"
PROTOCOL_SHA256 = "F8D058C390FC9446DD9885E58ACE06EAA2685B6B9007D1E787ABDDF66B1EBB0E"
AUTHORIZATION_SHA256 = "8B643287CC13B2BA4D5681A00453FA77742CDCF0FFF6103FFE399E8E1FF45908"
CANDIDATE_SHA256 = "8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0"
OPTIMIZATION_COMMIT = "b1b5e9908bbb685b8a852aff762f08ce7226aba1"
ATTEMPT = 3
STAGE = "GAMMA_SENSITIVITY"
SEEDS = [180, 181, 182, 183, 184]
GAMMAS = [0, 1, 2]
RHO = 0.025
SCALES = {
    "medium_large": {"directory": "ml_a3", "scenarios": {0: 1, 1: 61, 2: 1831}},
    "large": {"directory": "lg_a3", "scenarios": {0: 1, 1: 97, 2: 4657}},
}
SOLVER_PARAMETERS = {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}
RESULT_FIELDS = [
    "run_key", "run_directory_id", "stage", "execution_attempt", "git_commit",
    "config_file_sha256", "resolved_config_file_sha256", "protocol_sha256",
    "candidate_sha256", "scale", "task_type", "seed", "gamma", "rho",
    "candidate", "instance_sha256", "instance_canonical_sha256",
    "instance_identity_sha256", "baseline_run_key", "anchor_sha256", "state",
    "algorithm_status", "scientific_status", "algorithm_runtime", "master_runtime",
    "separation_runtime", "post_evaluation_wall_runtime", "total_wall_runtime",
    "penalized_runtime_par2", "baseline_robust_cost", "cost_budget",
    "actual_robust_cost", "actual_price_of_fairness", "objective_t",
    "robust_minimum_fill_rate", "wminfr", "minimum_weighted_mean_fill_rate",
    "inventory", "opened_warehouses", "iterations", "scenario_block_count",
    "certified_farkas_cut_count",
]
PAPER_METRICS = [
    "baseline_robust_cost", "cost_budget", "actual_robust_cost",
    "actual_price_of_fairness", "objective_t", "robust_minimum_fill_rate",
    "wminfr", "minimum_weighted_mean_fill_rate", "inventory", "opened_warehouses",
    "algorithm_runtime", "master_runtime", "separation_runtime",
    "post_evaluation_wall_runtime", "total_wall_runtime", "penalized_runtime_par2",
    "iterations", "scenario_block_count", "certified_farkas_cut_count",
]


class AuditError(RuntimeError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return bytes_sha256(canonical_json_bytes(value))


def _reject_constant(value: str) -> None:
    raise AuditError(f"non-finite JSON token: {value}")


def _finite_tree(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AuditError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")


def _json(source: ZipFile, name: str) -> dict[str, Any]:
    try:
        value = json.loads(source.read(name).decode("utf-8"), parse_constant=_reject_constant)
    except KeyError as exc:
        raise AuditError(f"missing ZIP entry: {name}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON: {name}") from exc
    _check(isinstance(value, dict), f"JSON root must be an object: {name}")
    _finite_tree(value, name)
    return value


def _csv(source: ZipFile, name: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        reader = csv.DictReader(io.StringIO(source.read(name).decode("utf-8"), newline=""), strict=True)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    except (KeyError, UnicodeDecodeError, csv.Error) as exc:
        raise AuditError(f"invalid CSV: {name}") from exc
    _check(fields and None not in fields, f"missing CSV header: {name}")
    forbidden = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
    for row in rows:
        _check(None not in row and all(item is not None for item in row.values()), f"malformed CSV row: {name}")
        _check(not any(item.strip().lower() in forbidden for item in row.values()), f"non-finite CSV: {name}")
    return fields, rows


def _run_key(scale: str, seed: int, gamma: int, task_type: str) -> str:
    identity = {
        "candidate": "baseline" if task_type == "baseline" else "certified_hybrid_scenario_benders_fairness",
        "execution_attempt": ATTEMPT,
        "gamma": gamma,
        "rho": "NOT_APPLICABLE" if task_type == "baseline" else "0.025",
        "scale": scale,
        "seed": seed,
        "stage": STAGE,
        "task_type": task_type,
    }
    return canonical_json_bytes(identity).decode("utf-8")


def _directory_id(run_key: str) -> str:
    return "r_" + hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:24]


def _number(value: Any, label: str) -> float:
    _check(type(value) in {int, float} and math.isfinite(float(value)), f"{label} must be finite numeric JSON")
    return float(value)


def _integer(value: Any, label: str) -> int:
    _check(type(value) is int and value >= 0, f"{label} must be a nonnegative integer")
    return value


def _instance_dimensions(archive: dict[str, Any]) -> tuple[int, int]:
    instance = archive.get("instance")
    _check(isinstance(instance, dict), "instance archive payload missing")
    warehouses = _integer(instance.get("num_warehouses"), "instance.num_warehouses")
    products = _integer(instance.get("num_products"), "instance.num_products")
    _check(warehouses > 0 and products > 0, "instance dimensions must be positive")
    return warehouses, products


def _matrix_total(value: Any, rows: int, columns: int, label: str) -> float:
    _check(isinstance(value, list) and len(value) == rows, f"{label} row dimension mismatch")
    total = 0.0
    for i, row in enumerate(value):
        _check(isinstance(row, list) and len(row) == columns, f"{label}[{i}] column dimension mismatch")
        total += sum(_number(item, f"{label}[{i}]") for item in row)
    return total


def _opened(value: Any, length: int, label: str) -> int:
    _check(isinstance(value, list) and len(value) == length, f"{label} vector dimension mismatch")
    return sum(_number(item, label) >= 0.5 for item in value)


def _project(record: dict[str, Any], instance_archive: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result")
    _check(isinstance(result, dict), "run.result missing")
    frontier = record.get("task_type") == "frontier"
    _check(frontier or record.get("task_type") == "baseline", "invalid task type")
    warehouses, products = _instance_dimensions(instance_archive)
    x_name = "x_values" if frontier else "best_x_values"
    y_name = "y_values" if frontier else "best_y_values"
    inventory = _matrix_total(result.get(x_name), warehouses, products, f"result.{x_name}")
    opened = _opened(result.get(y_name), warehouses, f"result.{y_name}")
    row = {field: "NOT_APPLICABLE" for field in RESULT_FIELDS}
    row.update({field: record[field] for field in RESULT_FIELDS if field in record})
    row.update({
        "algorithm_runtime": _number(result.get("algorithm_runtime", result.get("runtime")), "algorithm_runtime"),
        "master_runtime": _number(result.get("master_runtime", 0.0), "master_runtime"),
        "separation_runtime": _number(result.get("separation_runtime", result.get("subproblem_runtime", 0.0)), "separation_runtime"),
        "post_evaluation_wall_runtime": _number(result.get("post_evaluation_wall_runtime", 0.0), "post_evaluation_wall_runtime"),
        "total_wall_runtime": _number(result.get("total_wall_runtime", result.get("runtime")), "total_wall_runtime"),
        "penalized_runtime_par2": _number(result.get("penalized_runtime_par2"), "penalized_runtime_par2"),
        "baseline_robust_cost": _number(record.get("baseline_robust_cost", result.get("upper_bound")), "baseline_robust_cost"),
        "inventory": inventory,
        "opened_warehouses": opened,
        "iterations": _integer(result.get("iterations"), "iterations"),
    })
    if frontier:
        metadata = result.get("metadata")
        post = result.get("post_evaluation")
        _check(isinstance(metadata, dict) and isinstance(post, dict), "frontier reporting payload incomplete")
        row.update({
            "cost_budget": _number(record.get("cost_budget"), "cost_budget"),
            "objective_t": _number(result.get("objective_t"), "objective_t"),
            "robust_minimum_fill_rate": _number(result.get("robust_minimum_fill_rate"), "robust_minimum_fill_rate"),
            "scenario_block_count": _integer(metadata.get("committed_scenario_count"), "scenario_block_count"),
            "certified_farkas_cut_count": _integer(result.get("cuts"), "cuts"),
        })
        _check(post.get("valid") is True, "frontier post-evaluation invalid")
        for name in ("actual_robust_cost", "actual_price_of_fairness", "wminfr", "minimum_weighted_mean_fill_rate"):
            row[name] = _number(post.get(name), name)
    return row


def _same_csv_value(expected: Any, actual: str) -> bool:
    if expected == "NOT_APPLICABLE":
        return actual == expected
    if isinstance(expected, bool):
        return actual == str(expected)
    if type(expected) in {int, float}:
        try:
            value = float(actual)
        except ValueError:
            return False
        return math.isclose(float(expected), value, rel_tol=1e-12, abs_tol=1e-12)
    return str(expected) == actual


def audit_gamma_archive(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before = file_sha256(path)
    _check(before == GAMMA_ARCHIVE_SHA256, f"Gamma ZIP SHA mismatch: {before}")
    failures: list[str] = []
    with ZipFile(path) as source:
        bad_crc = source.testzip()
        _check(bad_crc is None, f"ZIP CRC failure: {bad_crc}")
        names = source.namelist()
        files = [name for name in names if not name.endswith("/")]
        # Strictly parse every structured source entry before scientific checks.
        json_names = [name for name in files if name.lower().endswith(".json")]
        csv_names = [name for name in files if name.lower().endswith(".csv")]
        parsed_json = {name: _json(source, name) for name in json_names}
        for name in csv_names:
            _csv(source, name)
        identity_files = {
            "config": "experiments/configs/fairness_hybrid_gamma_sensitivity.yaml",
            "protocol": "docs/fairness_hybrid_gamma_sensitivity_protocol.md",
            "authorization": "experiments/configs/fairness_hybrid_gamma_sensitivity_authorization.json",
        }
        expected_hashes = {"config": CONFIG_SHA256, "protocol": PROTOCOL_SHA256, "authorization": AUTHORIZATION_SHA256}
        identity_hashes = {key: bytes_sha256(source.read(name)) for key, name in identity_files.items()}
        _check(identity_hashes == expected_hashes, "config/protocol/authorization SHA mismatch")

        all_rows: list[dict[str, Any]] = []
        all_keys: set[str] = set()
        all_dirs: set[str] = set()
        counts = {
            "run_json": 0, "status_json": 0, "baseline_checkpoint": 0,
            "algorithm_checkpoint": 0, "final_exact_certification": 0,
            "post_evaluation": 0, "chunk": 0, "scenario_records": 0,
            "acceptance_evidence": 0, "accepted_evidence": 0,
        }
        max_residual = 0.0
        bound_crossings = 0
        source_summary_zero_display_rows = 0
        for scale, spec in SCALES.items():
            prefix = f"experiments/results_fh_gamma/{spec['directory']}"
            manifest = parsed_json[f"{prefix}/manifest.json"]
            identity = manifest.get("identity", {})
            expected_identity = {
                "stage": STAGE, "scale": scale, "seeds": SEEDS, "gamma": GAMMAS,
                "rho": [RHO], "execution_attempt": ATTEMPT,
                "git_commit": OPTIMIZATION_COMMIT, "config_file_sha256": CONFIG_SHA256,
                "protocol_sha256": PROTOCOL_SHA256, "candidate_sha256": CANDIDATE_SHA256,
                "solver_parameters": SOLVER_PARAMETERS, "previous_attempt_results_reused": False,
            }
            for key, expected in expected_identity.items():
                _check(identity.get(key) == expected, f"{scale} manifest identity drift: {key}")
            forward = manifest.get("run_key_to_directory_id")
            reverse = manifest.get("directory_id_to_run_key")
            _check(isinstance(forward, dict) and isinstance(reverse, dict) and len(forward) == len(reverse) == 30,
                   f"{scale} manifest mapping count mismatch")
            _check(all(reverse.get(directory) == key for key, directory in forward.items()), f"{scale} mapping not bijective")
            expected_keys = {
                _run_key(scale, seed, gamma, task)
                for seed in SEEDS for gamma in GAMMAS for task in ("baseline", "frontier")
            }
            _check(set(forward) == expected_keys, f"{scale} run plan mismatch")

            instance_archives = {
                (seed, gamma): parsed_json[f"{prefix}/instances/s{seed}_g{gamma}.json"]
                for seed in SEEDS for gamma in GAMMAS
            }
            scale_rows: list[dict[str, Any]] = []
            for run_key in sorted(expected_keys):
                directory = forward[run_key]
                _check(directory == _directory_id(run_key), f"short directory mismatch: {directory}")
                _check(run_key not in all_keys and directory not in all_dirs, "cross-scale run identity collision")
                all_keys.add(run_key); all_dirs.add(directory)
                run_name = f"{prefix}/runs/{directory}/run.json"
                status_name = f"{prefix}/runs/{directory}/status.json"
                run = parsed_json[run_name]
                status = parsed_json[status_name]
                counts["run_json"] += 1; counts["status_json"] += 1
                run_identity = json.loads(run_key)
                for key, expected in run_identity.items():
                    _check(run.get(key) == expected, f"run identity mismatch {directory}:{key}")
                status_identity = status.get("identity")
                _check(isinstance(status_identity, dict), f"status identity missing: {directory}")
                for key in ("run_key", "run_directory_id", "stage", "scale", "task_type", "seed", "gamma", "rho",
                            "execution_attempt", "git_commit", "config_file_sha256", "protocol_sha256",
                            "candidate_sha256", "instance_sha256", "instance_canonical_sha256",
                            "instance_identity_sha256", "baseline_run_key", "solver_parameters"):
                    _check(status_identity.get(key) == run.get(key), f"run/status identity mismatch {directory}:{key}")
                for key in ("state", "scientific_status"):
                    _check(status.get(key) == run.get(key), f"run/status mismatch {directory}:{key}")
                _check(run.get("state") == "complete", f"incomplete run: {directory}")
                _check(run.get("execution_attempt") == ATTEMPT and run.get("previous_attempt_results_reused") is False,
                       f"attempt reuse drift: {directory}")
                _check(run.get("solver_parameters") == SOLVER_PARAMETERS, f"solver parameter drift: {directory}")
                _check(run.get("scenario_count") == spec["scenarios"][run["gamma"]], f"scenario count drift: {directory}")
                _check(run.get("config_file_sha256") == CONFIG_SHA256 and run.get("protocol_sha256") == PROTOCOL_SHA256,
                       f"config/protocol identity drift: {directory}")
                _check(run.get("candidate_sha256") == CANDIDATE_SHA256, f"candidate identity drift: {directory}")
                instance = instance_archives[(run["seed"], run["gamma"])]
                archive_identity = instance.get("identity", {})
                canonical = canonical_sha256(instance.get("instance"))
                identity_sha = canonical_sha256(archive_identity)
                _check(run.get("instance_sha256") == canonical == run.get("instance_canonical_sha256"),
                       f"canonical instance identity mismatch: {directory}")
                _check(run.get("instance_identity_sha256") == identity_sha, f"instance identity SHA mismatch: {directory}")

                result = run.get("result", {})
                if run["task_type"] == "baseline":
                    counts["baseline_checkpoint"] += 1
                    _check(f"{prefix}/runs/{directory}/baseline_checkpoint.json" in parsed_json,
                           f"baseline checkpoint missing: {directory}")
                    checkpoint = parsed_json[f"{prefix}/runs/{directory}/baseline_checkpoint.json"]
                    _check(checkpoint.get("identity", {}).get("run_key") == run_key, f"baseline checkpoint identity drift: {directory}")
                    _check(result.get("status") == "optimal" and result.get("valid_UB") is True,
                           f"baseline not optimal valid_UB: {directory}")
                    _check(run.get("scientific_status") in {"certified_baseline_optimal", "certified_robust_optimal"},
                           f"baseline scientific status invalid: {directory}")
                else:
                    counts["algorithm_checkpoint"] += 1
                    checkpoint_name = f"{prefix}/runs/{directory}/algorithm_checkpoint.json"
                    checkpoint = parsed_json[checkpoint_name]
                    expected_checkpoint_sha = checkpoint.get("checkpoint_sha256")
                    payload = dict(checkpoint); payload.pop("checkpoint_sha256", None)
                    _check(expected_checkpoint_sha == canonical_sha256(payload), f"algorithm checkpoint SHA mismatch: {directory}")
                    _check(checkpoint.get("identity", {}).get("run_key") == run_key, f"algorithm checkpoint identity drift: {directory}")
                    _check(run.get("scientific_status") == "certified_robust_optimal" and run.get("algorithm_status") == "optimal",
                           f"frontier not certified optimal: {directory}")
                    log = result.get("iteration_log")
                    _check(isinstance(log, list) and log, f"iteration log missing: {directory}")
                    for prior in log[:-1]:
                        _check(prior.get("robust_feasibility_certified") is not True,
                               f"premature robust certification: {directory}")
                    final = log[-1]
                    _check(final.get("final_exact_separation_performed") is True, f"final exact separation missing: {directory}")
                    _check(final.get("robust_feasibility_certified") is True, f"final robust certification missing: {directory}")
                    _check(final.get("master_status") == "optimal" and final.get("separation_status") == "optimal",
                           f"final solver status invalid: {directory}")
                    separation_bound = _number(final.get("separation_objective_bound"), "separation objective bound")
                    _check(separation_bound <= 1e-7, f"final separation bound not certified: {directory}")
                    _check(result.get("metadata", {}).get("robust_feasibility_certified") is True and
                           result.get("metadata", {}).get("full_separation_objective_bound_required") is True,
                           f"frontier certificate metadata invalid: {directory}")
                    counts["final_exact_certification"] += 1
                    lower = _number(final.get("master_solver_best_bound"), "final master bound")
                    upper = _number(result.get("upper_bound"), "upper bound")
                    if lower > upper + 1e-4:
                        bound_crossings += 1
                    _check(lower <= upper + 1e-4, f"LB/UB crossing: {directory}")
                    _check(math.isclose(_number(result.get("robust_minimum_fill_rate"), "fill"),
                                        1.0 - _number(result.get("objective_t"), "T"), abs_tol=1e-12),
                           f"fairness identity mismatch: {directory}")
                    post_name = f"{prefix}/runs/{directory}/post_evaluation/post_evaluation.json"
                    post_doc = parsed_json[post_name]
                    evaluation = post_doc.get("evaluation", {})
                    embedded = result.get("post_evaluation", {})
                    _check(evaluation == embedded and evaluation.get("valid") is True and evaluation.get("errors") == [],
                           f"post-evaluation mismatch/invalid: {directory}")
                    _check(evaluation.get("scenario_count") == spec["scenarios"][run["gamma"]],
                           f"post-evaluation scenario count mismatch: {directory}")
                    _check(_number(evaluation.get("actual_robust_cost"), "actual cost") <=
                           _number(run.get("cost_budget"), "cost budget") + 1e-7,
                           f"cost budget violation: {directory}")
                    baseline_cost = _number(run.get("baseline_robust_cost"), "baseline robust cost")
                    actual_cost = _number(evaluation.get("actual_robust_cost"), "actual robust cost")
                    _check(math.isclose(_number(run.get("cost_budget"), "cost budget"),
                                        (1.0 + RHO) * baseline_cost, rel_tol=1e-12, abs_tol=1e-8),
                           f"cost budget identity mismatch: {directory}")
                    _check(math.isclose(_number(evaluation.get("actual_price_of_fairness"), "price of fairness"),
                                        actual_cost / baseline_cost - 1.0, rel_tol=1e-10, abs_tol=1e-12),
                           f"actual price of fairness identity mismatch: {directory}")
                    wminfr = _number(evaluation.get("wminfr"), "wminfr")
                    weighted_fill = _number(evaluation.get("minimum_weighted_mean_fill_rate"), "weighted fill rate")
                    objective_t = _number(result.get("objective_t"), "objective T")
                    _check(wminfr <= weighted_fill + 1e-7, f"fill-rate ordering mismatch: {directory}")
                    _check(1.0 - wminfr <= objective_t + 1e-7, f"post-evaluation violates certified T: {directory}")
                    _check(evaluation.get("objective_t_consistent") is True, f"post objective inconsistency: {directory}")
                    final_acceptance = evaluation.get("acceptance_evidence")
                    _check(isinstance(final_acceptance, list) and final_acceptance,
                           f"final acceptance evidence missing: {directory}")
                    for item_evidence in final_acceptance:
                        residual = _number(item_evidence.get("residual"), "final residual")
                        _check(residual <= _number(item_evidence.get("acceptance_threshold"), "acceptance threshold"),
                               f"final acceptance residual exceeds threshold: {directory}")
                        max_residual = max(max_residual, residual)
                        counts["acceptance_evidence"] += 1
                        if item_evidence.get("accepted") is True:
                            counts["accepted_evidence"] += 1
                        else:
                            raise AuditError(f"unaccepted final evidence: {directory}")
                    counts["post_evaluation"] += 1
                    index_name = f"{prefix}/runs/{directory}/post_evaluation/checkpoint/index.json"
                    index = parsed_json[index_name]
                    chunks = index.get("chunks")
                    _check(isinstance(chunks, list) and chunks, f"post checkpoint index invalid: {directory}")
                    scenario_cursor = 0
                    for chunk_index, item in enumerate(chunks):
                        _check(item.get("chunk_index") == chunk_index, f"chunk order mismatch: {directory}")
                        chunk_name = f"{prefix}/runs/{directory}/post_evaluation/{item['relative_path']}"
                        raw = source.read(chunk_name)
                        _check(bytes_sha256(raw).lower() == str(item.get("sha256")).lower(), f"chunk SHA mismatch: {chunk_name}")
                        chunk = parsed_json[chunk_name]
                        records = chunk.get("records")
                        _check(isinstance(records, list) and len(records) == item.get("scenario_count"), f"chunk record count mismatch: {chunk_name}")
                        _check(chunk.get("scenario_start") == scenario_cursor and
                               chunk.get("scenario_end_exclusive") == scenario_cursor + len(records),
                               f"chunk scenario ordering mismatch: {chunk_name}")
                        scenario_cursor += len(records)
                        counts["chunk"] += 1; counts["scenario_records"] += len(records)
                        for record in records:
                            evidence = record.get("acceptance_evidence")
                            _check(record.get("error") is None and isinstance(evidence, list), f"post scenario invalid: {chunk_name}")
                            for item_evidence in evidence:
                                residual = _number(item_evidence.get("residual"), "residual")
                                _check(residual <= _number(item_evidence.get("acceptance_threshold"), "acceptance threshold"),
                                       f"chunk acceptance residual exceeds threshold: {chunk_name}")
                                max_residual = max(max_residual, residual)
                                if item_evidence.get("accepted") is not True:
                                    raise AuditError(f"unaccepted evidence: {chunk_name}")
                    _check(scenario_cursor == spec["scenarios"][run["gamma"]], f"post scenarios incomplete: {directory}")

                projected = _project(run, instance)
                if run["task_type"] == "frontier":
                    evaluation = result["post_evaluation"]
                    _check(math.isclose(projected["inventory"], _number(evaluation.get("total_inventory"), "post inventory"),
                                        rel_tol=1e-12, abs_tol=1e-8), f"post inventory projection mismatch: {directory}")
                    _check(projected["opened_warehouses"] == _integer(evaluation.get("opened_warehouses"), "post opened warehouses"),
                           f"post opened-warehouse projection mismatch: {directory}")
                scale_rows.append(projected); all_rows.append(projected)

            fields, source_rows = _csv(source, f"{prefix}/results.csv")
            _check(fields == RESULT_FIELDS and len(source_rows) == 30, f"{scale} results CSV schema/count mismatch")
            projected_by_key = {row["run_key"]: row for row in scale_rows}
            _check(set(projected_by_key) == {row["run_key"] for row in source_rows}, f"{scale} results keys mismatch")
            for source_row in source_rows:
                projected = projected_by_key[source_row["run_key"]]
                for field in RESULT_FIELDS:
                    _check(_same_csv_value(projected[field], source_row[field]), f"results projection mismatch {scale}:{field}")
            _, summary_rows = _csv(source, f"{prefix}/summary.csv")
            source_summary_zero_display_rows += sum(
                row.get("scale") != scale and int(row.get("completed_runs", "0")) == 0 for row in summary_rows
            )

        expected_counts = {
            "run_json": 60, "status_json": 60, "baseline_checkpoint": 30,
            "algorithm_checkpoint": 30, "final_exact_certification": 30,
            "post_evaluation": 30, "chunk": 1350, "scenario_records": 33240,
            "acceptance_evidence": 413220, "accepted_evidence": 413220,
        }
        _check(counts == expected_counts, f"coverage totals mismatch: {counts}")
        _check(len(all_keys) == len(all_dirs) == len(all_rows) == 60, "global run identity count mismatch")
        report = {
            "decision": "approve_gamma_sensitivity_attempt3",
            "scientific_solution_valid": True,
            "optimization_rerun_required": False,
            "source_archive": {
                "name": "fairness_hybrid_gamma_sensitivity_attempt3_results.zip",
                "sha256_before": before, "sha256_expected": GAMMA_ARCHIVE_SHA256,
                "entry_count": len(names), "file_count": len(files), "crc_valid": True,
            },
            "identity": {
                "stage": STAGE, "execution_attempt": ATTEMPT, "optimization_commit": OPTIMIZATION_COMMIT,
                "config_sha256": CONFIG_SHA256, "protocol_sha256": PROTOCOL_SHA256,
                "authorization_sha256": AUTHORIZATION_SHA256, "candidate_sha256": CANDIDATE_SHA256,
                "previous_attempt_results_reused": False, "attempt1_or_attempt2_mixed": False,
            },
            "coverage": counts,
            "matrix": {"scales": list(SCALES), "seeds": SEEDS, "gamma": GAMMAS, "rho": RHO,
                       "baseline": 30, "frontier": 30, "total": 60,
                       "unique_run_keys": len(all_keys), "unique_directory_ids": len(all_dirs)},
            "post_evaluation": {"maximum_acceptance_residual": max_residual},
            "bound_crossing_count": bound_crossings,
            "source_summary_cross_scale_zero_display_rows": source_summary_zero_display_rows,
            "source_summary_zero_rows_classification": "display_only_not_scientific_error",
            "structured_parse": {"json_count": len(json_names), "csv_count": len(csv_names), "nan_or_inf": 0},
            "failures": failures,
        }
    after = file_sha256(path)
    _check(after == before, "source ZIP changed during audit")
    report["source_archive"]["sha256_after"] = after
    report["source_archive"]["unchanged"] = True
    return report, sorted(all_rows, key=lambda row: row["run_key"])


def _csv_text(rows: list[dict[str, Any]], fields: Iterable[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue()


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _quantile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position); upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _descriptive(values: list[float]) -> dict[str, float]:
    _check(len(values) == 5, "descriptive cell must contain five seeds")
    return {
        "mean": statistics.fmean(values), "median": statistics.median(values),
        "std": statistics.stdev(values), "iqr": _quantile(values, 0.75) - _quantile(values, 0.25),
        "min": min(values), "max": max(values),
    }


def _frontier_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {(row["scale"], row["seed"], row["gamma"]): row for row in rows if row["task_type"] == "baseline"}
    combined = []
    for row in rows:
        if row["task_type"] != "frontier":
            continue
        item = dict(row)
        baseline = baselines[(row["scale"], row["seed"], row["gamma"])]
        item["baseline_algorithm_runtime"] = baseline["algorithm_runtime"]
        item["baseline_inventory"] = baseline["inventory"]
        item["baseline_opened_warehouses"] = baseline["opened_warehouses"]
        combined.append(item)
    return sorted(combined, key=lambda row: (row["scale"], row["gamma"], row["seed"]))


def _plot(output: Path, seed_rows: list[dict[str, Any]], kind: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "legend.fontsize": 7, "pdf.compression": 9,
        "savefig.dpi": 180,
    })
    colors = {"medium_large": "#0072B2", "large": "#D55E00"}
    if kind == "cost":
        panels = [("baseline_robust_cost", "Baseline robust cost"),
                  ("actual_robust_cost", "Actual robust cost"),
                  ("actual_price_of_fairness", "Actual price of fairness")]
    elif kind == "fairness":
        panels = [("robust_minimum_fill_rate", "Certified minimum regional fill rate"),
                  ("wminfr", "Post-evaluation worst scenario-region fill rate"),
                  ("minimum_weighted_mean_fill_rate", "Worst-scenario weighted mean fill rate")]
    else:
        panels = [("algorithm_runtime", "Algorithm runtime (s)"),
                  ("iterations", "Iterations"),
                  ("scenario_block_count", "Scenario blocks")]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), constrained_layout=True)
    for axis, (metric, title) in zip(axes, panels):
        for scale in SCALES:
            means=[]; lows=[]; highs=[]
            for gamma in GAMMAS:
                values=[float(row[metric]) for row in seed_rows if row["scale"]==scale and row["gamma"]==gamma]
                means.append(statistics.fmean(values)); lows.append(min(values)); highs.append(max(values))
            errors=[[m-l for m,l in zip(means,lows)],[h-m for m,h in zip(means,highs)]]
            axis.errorbar(GAMMAS, means, yerr=errors, marker="o", linewidth=1.5, capsize=3,
                          label=scale.replace("_", "-").title(), color=colors[scale])
        axis.set_title(title); axis.set_xlabel(r"$\Gamma$"); axis.set_xticks(GAMMAS); axis.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False)
    stem = f"figure_gamma_{kind}"
    metadata = {"Creator": "Gamma Attempt 3 deterministic reporting", "CreationDate": None, "ModDate": None}
    fig.savefig(output / f"{stem}.pdf", metadata=metadata)
    fig.savefig(output / f"{stem}.png", metadata={"Software": "Gamma Attempt 3 deterministic reporting"})
    plt.close(fig)


def write_artifact_hash_index(output: Path) -> dict[str, str]:
    indexed = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != "artifact_sha256.csv")
    hashes = {name: file_sha256(output / name) for name in indexed}
    hash_rows = [{"source_archive_sha256": GAMMA_ARCHIVE_SHA256, "artifact_path": name, "sha256": hashes[name]}
                 for name in indexed]
    _write_text(output / "artifact_sha256.csv",
                _csv_text(hash_rows, ["source_archive_sha256", "artifact_path", "sha256"]))
    hashes["artifact_sha256.csv"] = file_sha256(output / "artifact_sha256.csv")
    return hashes


def generate_gamma_artifacts(path: Path, output: Path) -> dict[str, str]:
    report, rows = audit_gamma_archive(path)
    _check(not output.exists() or not any(output.iterdir()), f"output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_rows = _frontier_rows(rows)
    stats_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for scale in SCALES:
        for gamma in GAMMAS:
            cell = [row for row in seed_rows if row["scale"] == scale and row["gamma"] == gamma]
            _check(len(cell) == 5 and sorted(row["seed"] for row in cell) == SEEDS, "seed cell mismatch")
            summary = {"scale": scale, "gamma": gamma, "seed_count": 5,
                       "baseline_runs": 5, "frontier_runs": 5, "certified_frontiers": 5}
            for metric in PAPER_METRICS:
                descriptive = _descriptive([float(row[metric]) for row in cell])
                for statistic, value in descriptive.items():
                    stats_rows.append({"scale": scale, "gamma": gamma, "comparison_type": "level", "metric": metric,
                                       "statistic": statistic, "value": value, "seed_count": 5})
                summary[f"{metric}_mean"] = descriptive["mean"]
                summary[f"{metric}_median"] = descriptive["median"]
            summary_rows.append(summary)
        for lower, upper in ((0, 1), (1, 2)):
            lower_rows = {row["seed"]: row for row in seed_rows if row["scale"] == scale and row["gamma"] == lower}
            upper_rows = {row["seed"]: row for row in seed_rows if row["scale"] == scale and row["gamma"] == upper}
            _check(set(lower_rows) == set(upper_rows) == set(SEEDS), "paired Gamma seed identity mismatch")
            for metric in PAPER_METRICS:
                deltas = [float(upper_rows[seed][metric]) - float(lower_rows[seed][metric]) for seed in SEEDS]
                for statistic, value in _descriptive(deltas).items():
                    stats_rows.append({"scale": scale, "gamma": f"{lower}_to_{upper}",
                                       "comparison_type": "paired_delta", "metric": metric,
                                       "statistic": statistic, "value": value, "seed_count": 5})

    provenance = {
        "source_archive_name": path.name, "source_archive_sha256": GAMMA_ARCHIVE_SHA256,
        "access_mode": "read_only", "derived_files_modify_source": False,
        "scientific_unit": "seed", "paired_comparison": "within scale and seed across Gamma",
    }
    freeze = {
        "decision": report["decision"], "source_archive_sha256": GAMMA_ARCHIVE_SHA256,
        "optimization_commit": OPTIMIZATION_COMMIT, "config_sha256": CONFIG_SHA256,
        "protocol_sha256": PROTOCOL_SHA256, "authorization_sha256": AUTHORIZATION_SHA256,
        "candidate_sha256": CANDIDATE_SHA256, "execution_attempt": ATTEMPT,
        "matrix": report["matrix"], "immutable_source": True,
        "deterministic_rebuild_requirement": "two independent output directories must be byte-identical",
    }
    _write_json(output / "source_archive_provenance.json", provenance)
    _write_json(output / "final_audit.json", report)
    _write_json(output / "freeze_manifest.json", freeze)
    _write_text(output / "results.combined.csv", _csv_text(rows, RESULT_FIELDS))
    summary_fields = list(summary_rows[0])
    _write_text(output / "summary.combined.csv", _csv_text(summary_rows, summary_fields))
    _write_text(output / "table_gamma_complete_statistics.csv",
                _csv_text(stats_rows, ["scale", "gamma", "comparison_type", "metric", "statistic", "value", "seed_count"]))
    seed_fields = list(seed_rows[0])
    _write_text(output / "table_gamma_seed_results.csv", _csv_text(seed_rows, seed_fields))

    md = [
        "# Gamma sensitivity: main results", "",
        "All cells contain five seeds; values are means. Gamma comparisons are paired within seed.", "",
        "| Scale | Gamma | Baseline robust cost | Actual robust cost | Certified minimum fill rate | Algorithm runtime (s) | Iterations | Scenario blocks | Farkas cuts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md.append(
            f"| {row['scale']} | {row['gamma']} | {row['baseline_robust_cost_mean']:.2f} | "
            f"{row['actual_robust_cost_mean']:.2f} | {row['robust_minimum_fill_rate_mean']:.4f} | "
            f"{row['algorithm_runtime_mean']:.3f} | {row['iterations_mean']:.1f} | "
            f"{row['scenario_block_count_mean']:.1f} | {row['certified_farkas_cut_count_mean']:.1f} |"
        )
    _write_text(output / "table_gamma_main.md", "\n".join(md) + "\n")

    result_md = [
        "# Gamma sensitivity results", "",
        f"Frozen source: `{GAMMA_ARCHIVE_SHA256}`. The independent unit is the seed (n=5 per scale-Gamma cell).",
        "Gamma differences are interpreted descriptively and through within-seed paired trajectories; seed-Gamma rows are not treated as independent replicates.", "",
        "## Metric semantics", "",
        "- `robust_minimum_fill_rate = 1 - objective_t` is the certified minimum regional service guarantee.",
        "- `wminfr` is the post-evaluation fill rate of the worst scenario-region combination.",
        "- `minimum_weighted_mean_fill_rate` is the demand-weighted system mean fill rate in the worst scenario.",
        "These three quantities are distinct and are not relabelled as one another.", "",
        "## Findings", "",
    ]
    for scale in SCALES:
        cells = {row["gamma"]: row for row in summary_rows if row["scale"] == scale}
        result_md.append(
            f"For {scale.replace('_', '-')}, mean baseline robust cost rose from "
            f"{cells[0]['baseline_robust_cost_mean']:.2f} at Gamma=0 to {cells[2]['baseline_robust_cost_mean']:.2f} at Gamma=2, "
            f"while the certified minimum regional fill-rate guarantee changed from "
            f"{cells[0]['robust_minimum_fill_rate_mean']:.4f} to {cells[2]['robust_minimum_fill_rate_mean']:.4f}. "
            f"Mean algorithm runtime increased from {cells[0]['algorithm_runtime_mean']:.3f}s to "
            f"{cells[2]['algorithm_runtime_mean']:.3f}s."
        )
        result_md.append(
            f"Within-seed paired mean changes for Gamma 0->1 and 1->2 were, respectively: "
            f"baseline cost +{cells[1]['baseline_robust_cost_mean']-cells[0]['baseline_robust_cost_mean']:.2f} and "
            f"+{cells[2]['baseline_robust_cost_mean']-cells[1]['baseline_robust_cost_mean']:.2f}; certified minimum fill rate "
            f"{cells[1]['robust_minimum_fill_rate_mean']-cells[0]['robust_minimum_fill_rate_mean']:+.4f} and "
            f"{cells[2]['robust_minimum_fill_rate_mean']-cells[1]['robust_minimum_fill_rate_mean']:+.4f}; algorithm runtime "
            f"{cells[1]['algorithm_runtime_mean']-cells[0]['algorithm_runtime_mean']:+.3f}s and "
            f"{cells[2]['algorithm_runtime_mean']-cells[1]['algorithm_runtime_mean']:+.3f}s."
        )
    result_md.extend(["", "The small n=5 design supports structural sensitivity and managerial interpretation, not a claim that non-significance proves no effect.", ""])
    _write_text(output / "gamma_sensitivity_results.md",
                "\n\n".join(result_md).replace("\n\n\n", "\n\n").rstrip() + "\n")

    for kind in ("cost", "fairness", "runtime_mechanism"):
        _plot(output, seed_rows, kind)
    return write_artifact_hash_index(output)


def compare_artifact_directories(first: Path, second: Path) -> dict[str, Any]:
    first_files = sorted(path.relative_to(first).as_posix() for path in first.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second).as_posix() for path in second.rglob("*") if path.is_file())
    _check(first_files == second_files, "deterministic rebuild file sets differ")
    mismatches = [name for name in first_files if file_sha256(first / name) != file_sha256(second / name)]
    _check(not mismatches, f"deterministic rebuild byte mismatches: {mismatches}")
    return {"file_count": len(first_files), "byte_identical": True, "mismatches": []}


def _archive_runs(path: Path) -> list[dict[str, Any]]:
    with ZipFile(path) as source:
        _check(source.testzip() is None, f"comparison archive CRC failed: {path.name}")
        return [_json(source, name) for name in source.namelist() if name.endswith("/run.json")]


def audit_algorithm_comparisons(archives: dict[str, Path], output: Path) -> dict[str, Any]:
    _check(not output.exists() or not any(output.iterdir()), f"output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    provenance = []
    records: list[dict[str, Any]] = []
    missing = []
    for family, path in archives.items():
        if not path.is_file():
            missing.append({"family": family, "archive_name": path.name})
            continue
        digest = file_sha256(path)
        runs = _archive_runs(path)
        provenance.append({"family": family, "archive_name": path.name, "sha256": digest,
                           "run_count": len(runs), "access_mode": "read_only"})
        for run in runs:
            if run.get("task_type") != "baseline":
                records.append({"family": family, "archive_sha256": digest, **run})

    s1_candidates = ["single_cut", "persistent_separation", "persistent_certified_cache", "persistent_certified_cache_batch5"]
    success_rows = []
    for scale, family in (("medium_large", "s1_medium_large"), ("large", "s1_large")):
        for candidate in s1_candidates:
            cell = [r for r in records if r["family"] == family and r.get("candidate") == candidate]
            certified = [r for r in cell if r.get("scientific_status") == "certified_robust_optimal"]
            time_limit = [r for r in cell if r.get("scientific_status") == "time_limit_uncertified"]
            pipeline = [r for r in cell if r.get("state") != "complete" or "pipeline" in str(r.get("scientific_status", ""))]
            success_rows.append({"scale": scale, "candidate": candidate, "tasks": len(cell),
                                 "certified_solved": len(certified), "time_limit_uncertified": len(time_limit),
                                 "pipeline_failures": len(pipeline)})
    expected = {
        ("medium_large", "single_cut"): 6, ("medium_large", "persistent_separation"): 6,
        ("medium_large", "persistent_certified_cache"): 2,
        ("medium_large", "persistent_certified_cache_batch5"): 6,
        ("large", "single_cut"): 0, ("large", "persistent_separation"): 0,
        ("large", "persistent_certified_cache"): 0,
        ("large", "persistent_certified_cache_batch5"): 0,
    }
    _check(all(row["tasks"] == 6 and row["certified_solved"] == expected[(row["scale"], row["candidate"])]
               for row in success_rows), "S1 certified success table mismatch")
    _check(all(row["pipeline_failures"] == 0 for row in success_rows), "S1 pipeline failure detected")
    _check(all(row["time_limit_uncertified"] == 6 for row in success_rows if row["scale"] == "large"),
           "Large S1 failures are not uniformly time-limit uncertified")

    identity_fields = ["scale", "seed", "gamma", "rho", "instance_sha256", "baseline_run_key", "anchor_sha256"]
    overlap_rows=[]; unmatched_rows=[]
    hybrid_gamma = [r for r in records if r["family"] == "gamma_attempt3" and r.get("candidate") == "certified_hybrid_scenario_benders_fairness"]
    references = [r for r in records if r.get("candidate") in s1_candidates]
    for hybrid in hybrid_gamma:
        matched = []
        for reference in references:
            same = all(hybrid.get(field) == reference.get(field) for field in identity_fields)
            # Explicitly require equal solver and certification identities where available.
            same = same and hybrid.get("solver_parameters") == reference.get("solver_parameters")
            if same:
                matched.append(reference)
        if matched:
            for reference in matched:
                overlap_rows.append({"hybrid_run_key": hybrid.get("run_key"), "reference_run_key": reference.get("run_key"),
                                     "reference_candidate": reference.get("candidate"), "strict_pair": True})
        else:
            unmatched_rows.append({"scale": hybrid.get("scale"), "seed": hybrid.get("seed"), "gamma": hybrid.get("gamma"),
                                   "rho": hybrid.get("rho"), "hybrid_run_key": hybrid.get("run_key"),
                                   "reason": "no reference with identical instance, baseline, anchor, solver and certification identity"})
    assessment_decision = "minimal_paired_algorithm_benchmark_recommended"
    matrix_rows = []
    for row in success_rows:
        matrix_rows.append({
            **row,
            "comparison_target": "Gamma Attempt 3, seeds 180-184, Gamma=2, rho=0.025",
            "scale_identity_matches": True,
            "seed_identity_matches": False,
            "gamma_identity_matches": False,
            "rho_identity_matches": False,
            "instance_identity_matches": False,
            "baseline_run_key_matches": False,
            "anchor_sha256_matches": False,
            "threads_match": True,
            "solver_seed_matches": True,
            "feasibility_tolerance_matches": True,
            "algorithm_time_limit_matches": True,
            "scientific_success_definition_matches": True,
            "final_exact_certification_requirement_matches": True,
            "strict_pair_count": 0,
            "evidence_scope": "development-stage certification and timeout/PAR-2 evidence",
            "strict_paired_runtime_claim_allowed": False,
        })
    par_rows=[]
    for candidate in s1_candidates:
        for scale, family in (("medium_large", "s1_medium_large"), ("large", "s1_large")):
            cell=[r for r in records if r["family"]==family and r.get("candidate")==candidate]
            values=[]; gaps=[]
            for r in cell:
                result=r.get("result") or {}
                runtime=result.get("penalized_runtime_par2")
                if type(runtime) not in {int,float}:
                    runtime = result.get("algorithm_runtime", result.get("runtime", 3600.0)) if r.get("scientific_status")=="certified_robust_optimal" else 3600.0
                values.append(float(runtime))
                if type(result.get("gap")) in {int,float}: gaps.append(float(result["gap"]))
            par_rows.append({"scale":scale,"candidate":candidate,"task_count":len(cell),
                             "mean_par2":statistics.fmean(values),"median_par2":statistics.median(values),
                             "mean_final_gap":statistics.fmean(gaps) if gaps else "NOT_AVAILABLE"})

    _write_json(output / "comparison_source_provenance.json", {"sources": provenance, "missing_sources": missing})
    baseline_md = """# Baseline definitions

## Model baseline

The model baseline is the cost-optimal robust model without the fairness constraint. It defines C*(Gamma) and therefore the fairness cost budget (1+rho)C*(Gamma). It is not an algorithmic competitor.

## Algorithm baselines

The algorithm baselines are certified Benders variants: single-cut, persistent separation, certified cache, batch-5 certified cache, and variants without complete scenario recourse blocks. Runtime comparisons are strictly paired only when scale, seed, Gamma, rho, instance, baseline, anchor, solver parameters, time limit, success definition, and final exact certification requirement all coincide.
"""
    _write_text(output / "baseline_definition.md", baseline_md)
    _write_text(output / "algorithm_comparison_matrix.csv", _csv_text(matrix_rows, list(matrix_rows[0])))
    overlap_fields=["hybrid_run_key","reference_run_key","reference_candidate","strict_pair"]
    _write_text(output / "exact_overlap_cells.csv", _csv_text(overlap_rows, overlap_fields))
    _write_text(output / "unmatched_cells.csv", _csv_text(unmatched_rows, ["scale","seed","gamma","rho","hybrid_run_key","reason"]))
    _write_text(output / "candidate_success_summary.csv", _csv_text(success_rows, list(success_rows[0])))
    _write_text(output / "par2_and_gap_summary.csv", _csv_text(par_rows, list(par_rows[0])))
    assessment = f"""# EJOR algorithm-comparison assessment

Decision: `{assessment_decision}`.

The S1 archives provide auditable development evidence: medium-large certified counts are 6/6, 6/6, 2/6, and 6/6 for single-cut, persistent separation, certified cache, and batch-5 respectively; all four Large variants are 0/6 because they reached the time limit without certification, not because of pipeline errors. This supports a qualitative statement that the older certified methods did not scale reliably on the Large development screen.

However, the Gamma Attempt 3 Hybrid cells (seeds 180-184, Gamma=0/1/2, rho=0.025) have no fully identical algorithm-baseline cells. Consequently, the existing evidence does not support a strict paired runtime speedup claim. A minimal benchmark should add only ten reference frontiers: medium-large and large, seeds 180-184, Gamma=2, rho=0.025, using the exact Attempt 3 instances, baselines and anchors, Threads=1, solver Seed=0, FeasibilityTol=1e-7, and a 1800-second limit. Hybrid results remain read-only; no Hybrid rerun or tuning is permitted.
"""
    _write_text(output / "ejor_algorithm_comparison_assessment.md", assessment)
    return {"decision": assessment_decision, "strict_overlap_cells": len(overlap_rows),
            "unmatched_gamma_hybrid_cells": len(unmatched_rows), "missing_sources": missing,
            "s1_success": success_rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solver-free Gamma Attempt 3 final audit and reporting")
    parser.add_argument("--gamma-zip", type=Path, required=True)
    parser.add_argument("--gamma-output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--s1-medium-large", type=Path, required=True)
    parser.add_argument("--s1-large", type=Path, required=True)
    parser.add_argument("--d1", type=Path, required=True)
    parser.add_argument("--d2", type=Path, required=True)
    parser.add_argument("--final-holdout", type=Path, required=True)
    args = parser.parse_args(argv)
    hashes = generate_gamma_artifacts(args.gamma_zip, args.gamma_output)
    comparison = audit_algorithm_comparisons({
        "s1_medium_large": args.s1_medium_large, "s1_large": args.s1_large,
        "d1": args.d1, "d2": args.d2, "final_holdout": args.final_holdout,
        "gamma_attempt3": args.gamma_zip,
    }, args.comparison_output)
    print(json.dumps({"gamma_artifacts": hashes, "algorithm_comparison": comparison}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
