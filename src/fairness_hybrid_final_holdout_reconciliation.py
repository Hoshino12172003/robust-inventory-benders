"""Solver-free reconciliation of the formal Hybrid final-holdout archive.

The source ZIP is opened read-only.  This module never imports Gurobi, never
generates an instance, and writes only deterministic derived reports.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import random
import statistics
import subprocess
from typing import Any, Iterable
from zipfile import ZipFile

import yaml


ARCHIVE_SHA256 = "BD253A4DE967143B8CD88E7DEADCB821D6062208CB71664B900F9BD8EBDF3839"
RUN_COMMIT = "3e4d5dcda0fda9e616af99f1cd6ba5bb70d5465a"
CANDIDATE = "certified_hybrid_scenario_benders_fairness"
CANDIDATE_SHA256 = "8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0"
PROTOCOL_SHA256 = "BC01396163EE9E9CD7AB2F6CBCD682B33BE107DDAED5FADEB0CE96290B1AA931"
CONFIG_SHA256 = "E0C3A5312520BA7220D3D61675D1794422AA4CAD2AB4866016CA0DF515A96E16"
RESOLVED_CONFIG_SHA256 = "1A4B4AA80A7FDDC5CEDA228CB2D607707F4781242E492D8BE908CC222BFE072C"
D2_DECISION_SHA256 = "A43D3A6E9B74C19996AD6C8F3CFF3543462AFAA99A691C79781E1EABFEFC666D"
ROOT = "results_fairness_hybrid_final_holdout"
SEEDS = list(range(170, 180))
RHOS = [0.0, 0.01, 0.025, 0.05, 0.10]
TOLERANCE = 1.0e-4
SCALES = {
    "medium_large": {"directory": "ml_a1", "regions": 10, "products": 6, "scenarios": 1831, "chunks": 74},
    "large": {"directory": "lg_a1", "regions": 12, "products": 8, "scenarios": 4657, "chunks": 187},
}
RESULT_FIELDS = [
    "run_key", "run_directory_id", "stage", "scale", "task_type", "seed", "rho",
    "candidate", "state", "scientific_status", "algorithm_status", "certified_solved",
    "algorithm_runtime", "penalized_runtime_par2", "post_evaluation_wall_runtime",
    "total_wall_runtime", "instance_sha256", "baseline_run_key", "anchor_sha256",
]


class ReconciliationError(RuntimeError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ReconciliationError(message)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _canonical_sha(value: Any, *, upper: bool = False) -> str:
    digest = hashlib.sha256(_canonical_json(value)).hexdigest()
    return digest.upper() if upper else digest


def _yaml_sha(value: dict[str, Any], *, upper: bool = False) -> str:
    payload = yaml.safe_dump(value, sort_keys=True, allow_unicode=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return digest.upper() if upper else digest


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _strict_constant(token: str) -> None:
    raise ReconciliationError(f"non-finite JSON token: {token}")


def _assert_finite(value: Any, label: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReconciliationError(f"non-finite number at {label}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")


def _json(source: ZipFile, name: str) -> dict[str, Any]:
    try:
        value = json.loads(source.read(name).decode("utf-8"), parse_constant=_strict_constant)
    except KeyError as exc:
        raise ReconciliationError(f"missing ZIP entry: {name}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReconciliationError(f"invalid JSON: {name}") from exc
    _check(isinstance(value, dict), f"JSON root is not an object: {name}")
    _assert_finite(value, name)
    return value


def _csv(source: ZipFile, name: str) -> list[dict[str, str]]:
    try:
        text = source.read(name).decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text, newline=""), strict=True))
    except (KeyError, UnicodeDecodeError, csv.Error) as exc:
        raise ReconciliationError(f"invalid CSV: {name}") from exc
    _check(bool(rows), f"empty CSV: {name}")
    forbidden = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
    for row in rows:
        _check(None not in row and all(value is not None for value in row.values()), f"malformed CSV row: {name}")
        _check(not any(value.strip().lower() in forbidden for value in row.values()), f"non-finite CSV value: {name}")
    return rows


def _run_directory_id(run_key: str) -> str:
    return "r_" + hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:24]


def _expected_key(scale: str, seed: int, task_type: str, rho: float | None = None) -> str:
    value = {
        "candidate": "baseline" if task_type == "baseline" else CANDIDATE,
        "execution_attempt": 1,
        "rho": "NOT_APPLICABLE" if rho is None else f"{rho:.3f}".rstrip("0").rstrip("."),
        "scale": scale,
        "seed": seed,
        "stage": "FINAL_HOLDOUT",
        "task_type": task_type,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def corrected_gap(upper_bound: float, reported_lower_bound: float) -> float:
    return max(0.0, upper_bound - reported_lower_bound) / max(1.0, abs(upper_bound))


def reconcile_bounds(result: dict[str, Any], tolerance: float = TOLERANCE) -> dict[str, Any]:
    log = result.get("iteration_log")
    _check(isinstance(log, list) and log, "frontier iteration log missing")
    final = log[-1]
    historical = float(result["lower_bound"])
    current = float(final["master_solver_best_bound"])
    upper = float(result["upper_bound"])
    crossing = historical - upper
    reported = current
    gap = corrected_gap(upper, reported)
    return {
        "historical_recorded_lower_bound": historical,
        "final_master_solver_best_bound": current,
        "reported_certification_lower_bound": reported,
        "upper_bound": upper,
        "raw_bound_crossing": crossing,
        "bound_crossing_within_tolerance": crossing > 0.0 and crossing <= tolerance,
        "reported_gap": gap,
        "original_recorded_gap": float(result["gap"]),
        "final_exact_separation_objective_bound": float(final["separation_objective_bound"]),
    }


def _checkpoint_valid(checkpoint: dict[str, Any]) -> bool:
    payload = deepcopy(checkpoint)
    digest = payload.pop("checkpoint_sha256", None)
    return digest == _canonical_sha(payload, upper=True)


def _instance_initial_scenarios(instance: dict[str, Any], instance_sha: str) -> list[str]:
    regions = int(instance["num_regions"])
    products = int(instance["num_products"])
    order = [[r, j] for r in range(regions) for j in range(products)]
    active_sets: list[set[tuple[int, int]]] = [set()]
    for r in range(regions):
        ranked = sorted(
            ((r, j) for j in range(products)),
            key=lambda item: (-float(instance["demand_deviation"][item[0]][item[1]]), item),
        )
        active_sets.append(set(ranked[:2]))
    result = []
    for active in active_sets:
        demand_hex = []
        for r in range(regions):
            row = []
            for j in range(products):
                value = float(instance["base_demand"][r][j])
                if (r, j) in active:
                    value += float(instance["demand_deviation"][r][j])
                row.append(value.hex())
            demand_hex.append(row)
        payload = {
            "schema": "fairness_hybrid_scenario_v1",
            "instance_sha256": instance_sha,
            "component_order": order,
            "values": [1 if tuple(item) in active else 0 for item in order],
            "demand_hex": demand_hex,
        }
        result.append(_canonical_sha(payload, upper=True))
    return result


def _project_original(record: dict[str, Any]) -> dict[str, Any]:
    result = record["result"]
    row = {field: record.get(field, "NOT_APPLICABLE") for field in RESULT_FIELDS}
    row.update(
        {
            "algorithm_runtime": result.get("algorithm_runtime", result.get("runtime", 0.0)),
            "penalized_runtime_par2": result.get("penalized_runtime_par2"),
            "post_evaluation_wall_runtime": result.get("post_evaluation_wall_runtime", 0.0),
            "total_wall_runtime": result.get("total_wall_runtime", result.get("runtime", 0.0)),
            "certified_solved": record.get("scientific_status") == "certified_robust_optimal",
        }
    )
    return row


def _as_csv_string(value: Any) -> str:
    return str(value)


def _check_source(repo_root: Path) -> dict[str, Any]:
    expected_files = {
        "experiments/configs/fairness_hybrid_final_cross_scale_holdout.yaml": CONFIG_SHA256,
        "docs/fairness_hybrid_final_cross_scale_holdout_protocol.md": PROTOCOL_SHA256,
        "experiments/configs/certified_hybrid_scenario_benders_fairness_d1_candidate.yaml": CANDIDATE_SHA256,
        "analysis/fairness_hybrid_ccg_benders_d2_decision/decision.json": D2_DECISION_SHA256,
    }
    for name, digest in expected_files.items():
        _check(_file_sha(repo_root / name) == digest, f"run-commit source identity mismatch: {name}")
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{RUN_COMMIT}^{{commit}}"],
        check=False, capture_output=True, text=True,
    )
    _check(commit.returncode == 0, "formal run commit is unavailable")
    source_path = repo_root / "src/fairness_hybrid_ccg_benders.py"
    source = source_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tokens = {
        "master_created_once_before_loop": "model, y, x, t = _build_master(instance, output_flag)",
        "master_optimized_in_loop": "model.optimize()",
        "solver_bound_read": "master_bound = float(model.ObjBound)",
        "historical_max_update": "lower_bound = master_bound if lower_bound is None else max(float(lower_bound), master_bound)",
        "scenario_append": "add_complete_scenario_block(model, instance, scenario, x, t",
        "certified_cut_append": "add_canonical_cut_payload(model, variables, candidate.canonical_cut_payload",
        "complete_scenario_cost_block": "first_stage + transport + shortage + service <= float(cost_budget)",
        "complete_scenario_fairness_block": "u[r, j] for j in instance.J) <= t * regional_demand",
        "gap_function_call": "gap = relative_gap(upper_bound, lower_bound)",
    }
    evidence = {}
    for label, token in tokens.items():
        _check(token in source, f"run-commit source formula missing: {label}")
        evidence[label] = {"file": "src/fairness_hybrid_ccg_benders.py", "line": next(i for i, line in enumerate(lines, 1) if token in line)}
    gap_source = (repo_root / "src/fairness_benders.py").read_text(encoding="utf-8")
    gap_token = "return max(0.0, (float(upper_bound) - float(lower_bound)) / max(1.0, abs(float(upper_bound))))"
    _check(gap_token in gap_source, "run-commit gap truncation formula missing")
    evidence["gap_negative_truncation"] = {
        "file": "src/fairness_benders.py",
        "line": next(i for i, line in enumerate(gap_source.splitlines(), 1) if gap_token in line),
    }
    return {
        "run_commit": RUN_COMMIT,
        "verified_source_files": expected_files,
        "algorithm_source_sha256": _file_sha(source_path),
        "master_is_persistent_and_not_rebuilt_per_iteration": source.index(tokens["master_created_once_before_loop"]) < source.index("for iteration in range"),
        "evidence": evidence,
    }


def _audit_frontier(
    source: ZipFile,
    root: str,
    record: dict[str, Any],
    checkpoint: dict[str, Any],
    instance: dict[str, Any],
    expected_initial: list[str],
    scale_info: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    result = record["result"]
    metadata = result["metadata"]
    _check(_checkpoint_valid(checkpoint), f"algorithm checkpoint hash mismatch: {record['run_key']}")
    _check(checkpoint["schema"] == "fairness_hybrid_ccg_benders_checkpoint_v1", "algorithm checkpoint schema mismatch")
    identity = checkpoint["identity"]
    _check(identity["run_key"] == record["run_key"], "checkpoint run key mismatch")
    _check(identity["candidate_sha256"] == CANDIDATE_SHA256, "checkpoint candidate mismatch")
    _check(identity["protocol_sha256"] == PROTOCOL_SHA256, "checkpoint protocol mismatch")
    state = checkpoint["state"]
    scenarios = state["committed_scenario_sha256_values"]
    scenario_payloads = state["scenario_payloads_by_sha256"]
    cuts = state["committed_farkas_cut_sha256_values"]
    cut_payloads = state["cut_payloads_by_sha256"]
    _check(len(scenarios) == len(set(scenarios)) and set(scenarios) == set(scenario_payloads), "scenario checkpoint replacement/duplication")
    _check(len(cuts) == len(set(cuts)) and set(cuts) == set(cut_payloads), "cut checkpoint replacement/duplication")
    _check(scenarios[: len(expected_initial)] == expected_initial, "initial scenario identity/order mismatch")
    _check(metadata["initial_scenario_count"] == len(expected_initial), "initial scenario count mismatch")
    for digest, payload in scenario_payloads.items():
        _check(_canonical_sha(payload, upper=True) == digest, "scenario payload SHA mismatch")
    for digest, payload in cut_payloads.items():
        _check(_canonical_sha(payload, upper=True) == digest, "Farkas cut payload SHA mismatch")
        _check(payload.get("schema") == "fairness_farkas_cut_v1", "non-certified cut schema")
        _check(payload.get("farkas_normalization") == "nonnegative_multipliers_sum_to_one_v1", "Farkas normalization mismatch")
    log = result["iteration_log"]
    _check(log == state["iteration_log"], "checkpoint/run iteration log mismatch")
    _check(int(result["iterations"]) == len(log) == int(state["iteration"]), "iteration count mismatch")
    previous_count = len(expected_initial)
    committed_scenarios = []
    committed_cuts = []
    previous_historical: float | None = None
    for index, item in enumerate(log):
        _check(int(item["iteration"]) == index + 1, "iteration order mismatch")
        master = float(item["master_solver_best_bound"])
        expected_historical = master if previous_historical is None else max(previous_historical, master)
        _check(float(item["lower_bound"]) == expected_historical, "historical lower-bound ledger formula mismatch")
        previous_historical = expected_historical
        scenario = item["committed_scenario_sha256"]
        cut = item["committed_farkas_cut_sha256"]
        _check((scenario is None) == (cut is None), "scenario/cut commit pairing mismatch")
        if scenario is not None:
            committed_scenarios.append(scenario)
            committed_cuts.append(cut)
            previous_count += 1
        _check(int(item["scenario_count"]) == previous_count, "scenario append-only count mismatch")
        if index < len(log) - 1:
            _check(item["final_exact_separation_performed"] is False, "premature final exact separation")
            _check(item["robust_feasibility_certified"] is False, "premature robust certification")
    _check(scenarios[len(expected_initial):] == committed_scenarios, "scenario append ledger mismatch")
    _check(cuts == committed_cuts, "cut append ledger mismatch")
    _check(int(result["cuts"]) == len(cuts), "reported cut count mismatch")
    _check(metadata["committed_scenario_count"] == len(scenarios), "reported scenario count mismatch")
    _check(metadata["committed_scenario_sha256_values"] == scenarios, "run/checkpoint scenario SHA projection mismatch")
    _check(metadata["committed_farkas_cut_sha256_values"] == cuts, "run/checkpoint cut SHA projection mismatch")
    final = log[-1]
    _check(final["final_exact_separation_performed"] is True, "final exact separation missing")
    _check(final["robust_feasibility_certified"] is True, "final robust certification missing")
    _check(final["master_status"] == "optimal" and final["separation_status"] == "optimal", "final solver status not optimal")
    _check(math.isfinite(float(final["separation_objective_bound"])) and float(final["separation_objective_bound"]) <= 1.0e-7, "illegal final separation objective bound")
    _check(state["final_certification_state"] == "complete_exact_certified", "checkpoint final certificate state mismatch")
    _check(metadata["robust_feasibility_certified"] is True, "metadata robust certificate missing")
    _check(metadata["scenario_master_lower_bound_valid"] is True, "scenario-master bound validity flag missing")
    _check(result["status"] == "optimal", "frontier algorithm status mismatch")
    _check(record["scientific_status"] == "certified_robust_optimal", "frontier scientific status mismatch")
    _check(record["solved_to_tolerance"] is True, "frontier solved flag mismatch")
    bounds = reconcile_bounds(result)
    _check(bounds["reported_certification_lower_bound"] <= bounds["upper_bound"] + TOLERANCE, "reconciled LB exceeds UB tolerance")
    _check(abs(bounds["final_master_solver_best_bound"] - float(final["master_incumbent_objective"])) <= 1.0e-9, "final master bound/incumbent mismatch")
    _check(abs(bounds["upper_bound"] - float(final["master_incumbent_objective"])) <= 1.0e-9, "final incumbent/UB mismatch")
    _check(abs(float(result["objective_t"]) - bounds["upper_bound"]) <= 1.0e-12, "objective T/UB mismatch")
    _check(float(result["penalized_runtime_par2"]) == float(result["algorithm_runtime"]), "solved PAR-2 mismatch")

    final_post = _json(source, f"{root}/post_evaluation/post_evaluation.json")
    index = _json(source, f"{root}/post_evaluation/checkpoint/index.json")
    evaluation = result["post_evaluation"]
    _check(final_post["evaluation"] == evaluation, "run/final post-evaluation mismatch")
    post_identity = final_post["identity"]
    _check(final_post["identity_sha256"] == _canonical_sha(post_identity), "post-evaluation identity SHA mismatch")
    _check(index["identity_sha256"] == final_post["identity_sha256"], "post index identity mismatch")
    _check(post_identity["run_key"] == record["run_key"], "post run key mismatch")
    _check(post_identity["git_commit"] == RUN_COMMIT and post_identity["config_sha256"] == RESOLVED_CONFIG_SHA256, "post source identity mismatch")
    _check(post_identity["baseline_anchor_sha256"] == record["anchor_sha256"], "post anchor mismatch")
    _check(post_identity["run_execution_attempt"] == record["post_evaluation_run_execution_attempt"] == 1, "post run attempt mismatch")
    _check(post_identity["post_evaluation_pipeline_generation"] == record["post_evaluation_pipeline_generation"] == 4, "post pipeline generation mismatch")
    _check(post_identity["scenario_count"] == scale_info["scenarios"], "post scenario identity mismatch")
    _check(post_identity["chunk_size"] == 25 and post_identity["per_scenario_time_limit"] == 30.0, "post checkpoint parameters mismatch")
    solution_identity = {
        "y_values": result["y_values"], "x_values": result["x_values"],
        "t_value": float(result["objective_t"]), "baseline_cost": float(result["baseline_cost"]),
        "rho": float(result["rho"]),
    }
    _check(post_identity["solution_sha256"] == _yaml_sha(solution_identity), "post solution identity mismatch")
    _check(evaluation["valid"] is True and evaluation["errors"] == [], "invalid post-evaluation")
    _check(evaluation["objective_t_consistent"] is True, "post objective T inconsistent")
    _check(evaluation["scenario_count"] == scale_info["scenarios"], "post evaluation scenario count mismatch")
    _check(isinstance(evaluation["acceptance_evidence"], list) and evaluation["acceptance_evidence"], "post acceptance evidence missing")
    _check(all(item.get("accepted") is True for item in evaluation["acceptance_evidence"]), "post acceptance evidence failed")
    chunks = index["chunks"]
    _check(len(chunks) == scale_info["chunks"], "post chunk count mismatch")
    _check([item["chunk_index"] for item in chunks] == list(range(scale_info["chunks"])), "post chunk order mismatch")
    scenario_identities = []
    cursor = 0
    evidence_count = 0
    for entry in chunks:
        expected_relative = f"checkpoint/chunk_{entry['chunk_index']:05d}.json"
        _check(entry["relative_path"] == expected_relative, "post chunk path mismatch")
        name = f"{root}/post_evaluation/{expected_relative}"
        raw = source.read(name)
        _check(hashlib.sha256(raw).hexdigest() == entry["sha256"], "post chunk SHA mismatch")
        try:
            chunk = json.loads(raw.decode("utf-8"), parse_constant=_strict_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReconciliationError(f"invalid post chunk JSON: {name}") from exc
        _assert_finite(chunk, name)
        _check(chunk["identity_sha256"] == final_post["identity_sha256"], "chunk identity mismatch")
        _check(chunk["chunk_index"] == entry["chunk_index"] and chunk["scenario_start"] == cursor, "chunk index/start mismatch")
        records = chunk["records"]
        _check(len(records) == entry["scenario_count"], "chunk record count mismatch")
        for item in records:
            _check(item["scenario_index"] == cursor, "post scenario order mismatch")
            pattern = item["deviation_pattern"]
            payload = {"scenario_index": cursor, "scenario_name": item["scenario_name"], "deviation_pattern": pattern}
            _check(item["deviation_pattern_sha256"] == _canonical_sha(pattern), "post deviation SHA mismatch")
            _check(item["scenario_key"] == _canonical_sha(payload), "post scenario key mismatch")
            _check(item["error"] is None and isinstance(item["policy"], dict), "post scenario solve error")
            evidence = item["acceptance_evidence"]
            _check(isinstance(evidence, list) and evidence, "scenario acceptance evidence missing")
            _check(all(value.get("accepted") is True for value in evidence), "scenario acceptance evidence failed")
            _check(any(value.get("constraint_type") == "robust_cost_budget" for value in evidence), "scenario cost-budget evidence missing")
            evidence_count += len(evidence)
            scenario_identities.append({**payload, "deviation_pattern_sha256": item["deviation_pattern_sha256"], "scenario_key": item["scenario_key"]})
            cursor += 1
        _check(chunk["scenario_end_exclusive"] == cursor, "chunk end mismatch")
    _check(cursor == scale_info["scenarios"], "post chunk scenario total mismatch")
    _check(post_identity["scenario_sequence_sha256"] == _canonical_sha(scenario_identities), "post scenario sequence SHA mismatch")
    return bounds, {"chunks": len(chunks), "scenarios": cursor, "acceptance_evidence": evidence_count}


def audit_archive(archive: str | Path, repo_root: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    archive_path = Path(archive)
    repo = Path(repo_root)
    before = _file_sha(archive_path)
    _check(before == ARCHIVE_SHA256, "formal archive SHA256 mismatch")
    source_evidence = _check_source(repo)
    records: list[dict[str, Any]] = []
    scale_reports = {}
    total_chunks = 0
    total_scenarios = 0
    total_acceptance = 0
    json_count = 0
    csv_count = 0
    with ZipFile(archive_path, "r") as source:
        entries = source.infolist()
        _check(len(entries) == 13640, "ZIP entry count mismatch")
        _check(sum(not entry.is_dir() for entry in entries) == 13638, "ZIP file count mismatch")
        _check(sum(entry.is_dir() for entry in entries) == 2, "ZIP directory count mismatch")
        _check(source.testzip() is None, "ZIP CRC failure")
        normalized = [PurePosixPath(entry.filename).as_posix().rstrip("/") for entry in entries]
        _check(len(normalized) == len(set(normalized)), "duplicate normalized ZIP entry")
        actual_files = {name for name, entry in zip(normalized, entries) if not entry.is_dir()}
        expected_files: set[str] = set()
        for scale, info in SCALES.items():
            prefix = f"{ROOT}/{info['directory']}"
            expected_files.update(f"{prefix}/{name}" for name in ("manifest.json", "resolved_config.yaml", "results.csv", "summary.csv"))
            manifest = _json(source, f"{prefix}/manifest.json")
            json_count += 1
            resolved_raw = source.read(f"{prefix}/resolved_config.yaml")
            _check(hashlib.sha256(resolved_raw).hexdigest().upper() == RESOLVED_CONFIG_SHA256, "resolved config file SHA mismatch")
            resolved = yaml.safe_load(resolved_raw.decode("utf-8"))
            _check(isinstance(resolved, dict), "resolved config is not a mapping")
            results_rows = _csv(source, f"{prefix}/results.csv")
            summary_rows = _csv(source, f"{prefix}/summary.csv")
            csv_count += 2
            identity = manifest["identity"]
            expected_identity = {
                "candidate": CANDIDATE, "candidate_sha256": CANDIDATE_SHA256,
                "config_file_sha256": CONFIG_SHA256, "d2_decision_sha256": D2_DECISION_SHA256,
                "execution_attempt": 1, "git_commit": RUN_COMMIT,
                "previous_attempt_results_reused": False, "protocol_sha256": PROTOCOL_SHA256,
                "resolved_config_file_sha256": RESOLVED_CONFIG_SHA256,
                "rhos": RHOS, "scale": scale, "scenario_count": info["scenarios"],
                "seeds": SEEDS, "stage": "FINAL_HOLDOUT",
            }
            for key, value in expected_identity.items():
                _check(identity.get(key) == value, f"manifest identity mismatch: {scale}/{key}")
            _check(manifest["completed_run_count"] == manifest["certified_solved_count"] == 60, "manifest total count mismatch")
            _check(manifest["baseline_certified_count"] == 10 and manifest["frontier_certified_count"] == 50, "manifest task count mismatch")
            forward = manifest["run_key_to_directory_id"]
            reverse = manifest["directory_id_to_run_key"]
            _check(len(forward) == len(reverse) == 60 and {value: key for key, value in forward.items()} == reverse, "manifest run mapping mismatch")
            expected_keys = {_expected_key(scale, seed, "baseline") for seed in SEEDS}
            expected_keys |= {_expected_key(scale, seed, "frontier", rho) for seed in SEEDS for rho in RHOS}
            _check(set(forward) == expected_keys, f"task matrix mismatch: {scale}")
            instances = {}
            initial_by_seed = {}
            for seed in SEEDS:
                name = f"{prefix}/instances/{seed}.json"
                expected_files.add(name)
                instance = _json(source, name)
                json_count += 1
                _check(instance["num_regions"] == info["regions"] and instance["num_products"] == info["products"], "instance scale mismatch")
                _check(instance["name"].endswith(f"seed{seed}"), "instance seed identity mismatch")
                instance_sha = _yaml_sha(instance, upper=True)
                scenario_instance_sha = _canonical_sha(instance, upper=True)
                instances[seed] = (instance, instance_sha)
                initial_by_seed[seed] = _instance_initial_scenarios(instance, scenario_instance_sha)
            scale_records = []
            bounds_rows = []
            for run_key, directory_id in sorted(forward.items()):
                parsed = json.loads(run_key, parse_constant=_strict_constant)
                _check(directory_id == _run_directory_id(run_key), "run short-directory mapping mismatch")
                root = f"{prefix}/runs/{directory_id}"
                for suffix in ("run.json", "status.json"):
                    expected_files.add(f"{root}/{suffix}")
                record = _json(source, f"{root}/run.json")
                status = _json(source, f"{root}/status.json")
                json_count += 2
                seed = int(parsed["seed"])
                instance, instance_sha = instances[seed]
                _check(record["run_key"] == run_key and record["run_directory_id"] == directory_id, "run identity mapping mismatch")
                _check(record["state"] == status["state"] == "complete", "run/status state mismatch")
                _check(record["scientific_status"] == status["scientific_status"], "run/status scientific status mismatch")
                _check(record["algorithm_status"] == status["algorithm_status"], "run/status algorithm status mismatch")
                for key, value in {"stage": "FINAL_HOLDOUT", "scale": scale, "seed": seed, "execution_attempt": 1,
                                   "git_commit": RUN_COMMIT, "config_file_sha256": CONFIG_SHA256,
                                   "resolved_config_file_sha256": RESOLVED_CONFIG_SHA256,
                                   "protocol_sha256": PROTOCOL_SHA256, "candidate_sha256": CANDIDATE_SHA256,
                                   "instance_sha256": instance_sha}.items():
                    _check(record.get(key) == value, f"run identity mismatch: {key}")
                _check(record["baseline_run_key"] == _expected_key(scale, seed, "baseline"), "baseline run-key anchor mismatch")
                result = record["result"]
                _check(record["scientific_status"] == "certified_robust_optimal", "uncertified run in formal archive")
                _check(record["algorithm_status"] == result["status"] == "optimal", "algorithm status projection mismatch")
                if parsed["task_type"] == "baseline":
                    name = f"{root}/baseline_checkpoint.json"
                    expected_files.add(name)
                    checkpoint = _json(source, name)
                    json_count += 1
                    _check(_checkpoint_valid(checkpoint), "baseline checkpoint hash mismatch")
                    _check(checkpoint["result"] == {key: value for key, value in result.items() if key not in {"algorithm_runtime", "post_evaluation_wall_runtime", "total_wall_runtime", "penalized_runtime_par2"}}, "baseline checkpoint/run payload mismatch")
                    _check(result["valid_UB"] is True and float(result["gap"]) <= TOLERANCE, "baseline invalid UB/gap")
                    _check(float(result["penalized_runtime_par2"]) == float(result["algorithm_runtime"]), "baseline PAR-2 mismatch")
                    anchor = manifest["baseline_anchors"][str(seed)]
                    unhashed_anchor = {key: value for key, value in anchor.items() if key != "anchor_sha256"}
                    _check(anchor["anchor_sha256"] == _yaml_sha(unhashed_anchor, upper=True), "baseline anchor SHA mismatch")
                    _check(anchor["value"] == result["upper_bound"] and anchor["value_hex"] == float(result["upper_bound"]).hex(), "baseline anchor value mismatch")
                    projection = {"record": record, "bounds": None, "post": None}
                else:
                    _check(record["candidate"] == CANDIDATE and float(record["rho"]) in RHOS, "frontier identity mismatch")
                    _check(record["anchor_sha256"] == manifest["baseline_anchors"][str(seed)]["anchor_sha256"], "frontier anchor mismatch")
                    _check(manifest["run_identities"][run_key] == {key: value for key, value in record.items() if key not in {"task_type", "candidate", "state", "algorithm_status", "scientific_status", "solved_to_tolerance", "result"}}, "manifest/run identity projection mismatch")
                    name = f"{root}/algorithm_checkpoint.json"
                    expected_files.add(name)
                    checkpoint = _json(source, name)
                    json_count += 1
                    for suffix in ("post_evaluation/post_evaluation.json", "post_evaluation/checkpoint/index.json"):
                        expected_files.add(f"{root}/{suffix}")
                    for index in range(info["chunks"]):
                        expected_files.add(f"{root}/post_evaluation/checkpoint/chunk_{index:05d}.json")
                    bounds, post_counts = _audit_frontier(source, root, record, checkpoint, instance, initial_by_seed[seed], info)
                    json_count += 2 + info["chunks"]
                    total_chunks += post_counts["chunks"]
                    total_scenarios += post_counts["scenarios"]
                    total_acceptance += post_counts["acceptance_evidence"]
                    bounds_rows.append({"run_key": run_key, "run_directory_id": directory_id, "scale": scale, "seed": seed, "rho": float(record["rho"]), **bounds})
                    post = result["post_evaluation"]
                    projection = {
                        "record": record,
                        "bounds": bounds,
                        "post": {
                            key: post.get(key) for key in (
                                "minimum_weighted_mean_fill_rate", "actual_robust_cost", "actual_price_of_fairness",
                                "wgap", "wminfr", "wwd", "realized_worst_shortage_rate", "scenario_count",
                            )
                        },
                    }
                original_projection = _project_original(record)
                projection["original_projection"] = original_projection
                scale_records.append(projection)
                records.append(projection)
            _check(len(results_rows) == len(scale_records) == 60, "results.csv row count mismatch")
            by_key = {item["record"]["run_key"]: item for item in scale_records}
            for row in results_rows:
                _check(row["run_key"] in by_key, "results.csv unknown run")
                expected = by_key[row["run_key"]]["original_projection"]
                _check(list(row) == RESULT_FIELDS, "results.csv field order mismatch")
                _check(all(row[key] == _as_csv_string(expected[key]) for key in RESULT_FIELDS), "results.csv projection mismatch")
            expected_summary = _summary_rows([item["original_projection"] for item in scale_records])
            _check(summary_rows == [{key: str(value) for key, value in row.items()} for row in expected_summary], "summary.csv projection mismatch")
            scale_reports[scale] = {
                "runs": 60, "baselines": 10, "frontiers": 50,
                "bound_crossings": sum(item["raw_bound_crossing"] > 0.0 for item in bounds_rows),
            }
        _check(actual_files == expected_files, "ZIP entry inventory has missing or unexpected files")
    after = _file_sha(archive_path)
    _check(after == before, "formal archive changed during audit")
    crossings = [item for projection in records if projection["bounds"] for item in [projection["bounds"]] if item["raw_bound_crossing"] > 0.0]
    _check(len(crossings) == 1, "expected unique bound crossing was not unique")
    crossing_projection = next(item for item in records if item["record"]["run_directory_id"] == "r_45f40e77afd919415d895390")
    _check(crossing_projection["bounds"]["raw_bound_crossing"] > 0.0, "known anomaly not classified as crossing")
    audit = {
        "status": "pass",
        "read_only": True,
        "gurobi_called": False,
        "archive_sha256_before": before,
        "archive_sha256_after": after,
        "crc_valid": True,
        "entry_count": 13640,
        "file_count": 13638,
        "directory_count": 2,
        "json_file_count": json_count,
        "csv_file_count": csv_count,
        "run_count": len(records),
        "baseline_count": sum(item["record"]["task_type"] == "baseline" for item in records),
        "frontier_count": sum(item["record"]["task_type"] == "frontier" for item in records),
        "exact_certification_count": sum(item["bounds"] is not None for item in records),
        "post_evaluation_index_count": 100,
        "post_evaluation_chunk_count": total_chunks,
        "post_evaluation_scenario_record_count": total_scenarios,
        "post_evaluation_acceptance_evidence_count": total_acceptance,
        "chunk_sha_error_count": 0,
        "bound_crossing_count": len(crossings),
        "scales": scale_reports,
        "source_code_evidence": source_evidence,
    }
    _check((len(records), total_chunks, total_scenarios) == (120, 13050, 324400), "formal aggregate count mismatch")
    return audit, records


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["scale"], row["task_type"], row["candidate"], row["rho"]), []).append(row)
    result = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        values = groups[key]
        result.append(
            {
                "scale": key[0], "task_type": key[1], "candidate": key[2], "rho": key[3],
                "run_count": len(values),
                "certified_solved_count": sum(bool(item["certified_solved"]) for item in values),
                "mean_algorithm_runtime": math.fsum(float(item["algorithm_runtime"]) for item in values) / len(values),
                "mean_penalized_runtime_par2": math.fsum(float(item["penalized_runtime_par2"]) for item in values) / len(values),
            }
        )
    return result


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _stats(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    return {
        "mean": math.fsum(data) / len(data), "median": statistics.median(data),
        "standard_deviation": statistics.stdev(data) if len(data) > 1 else 0.0,
        "iqr": _percentile(data, 0.75) - _percentile(data, 0.25),
        "min": min(data), "max": max(data),
    }


def _permutation_pvalue(differences: list[float]) -> float:
    observed = abs(math.fsum(differences) / len(differences))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        values.append(abs(math.fsum(sign * value for sign, value in zip(signs, differences)) / len(differences)))
    return sum(value + 1.0e-15 >= observed for value in values) / len(values)


def _holm(pvalues: list[float]) -> list[float]:
    ordered = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(pvalues) - rank) * value))
        adjusted[index] = running
    return adjusted


def _bootstrap_ci(seed_values: dict[int, float], *, draws: int = 10000) -> list[float]:
    rng = random.Random(20260730)
    seeds = sorted(seed_values)
    means = []
    for _ in range(draws):
        sample = [rng.choice(seeds) for _ in seeds]
        means.append(math.fsum(seed_values[seed] for seed in sample) / len(sample))
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _paper_metrics(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frontiers = [item for item in records if item["bounds"] is not None]
    full_rows = []
    metric_names = [
        "objective_t", "robust_minimum_fill_rate", "minimum_weighted_mean_fill_rate",
        "actual_robust_cost", "algorithm_runtime", "post_evaluation_wall_runtime",
        "total_wall_runtime", "iterations", "scenario_count", "cut_count",
    ]
    for item in frontiers:
        record = item["record"]
        result = record["result"]
        post = item["post"]
        full_rows.append(
            {
                "scale": record["scale"], "seed": record["seed"], "rho": float(record["rho"]),
                "scientific_status": record["scientific_status"],
                "objective_t": float(result["objective_t"]),
                "robust_minimum_fill_rate": float(result["robust_minimum_fill_rate"]),
                "minimum_weighted_mean_fill_rate": float(post["minimum_weighted_mean_fill_rate"]),
                "actual_robust_cost": float(post["actual_robust_cost"]),
                "actual_price_of_fairness": float(post["actual_price_of_fairness"]),
                "wgap": float(post["wgap"]), "wminfr": float(post["wminfr"]), "wwd": float(post["wwd"]),
                "algorithm_runtime": float(result["algorithm_runtime"]),
                "post_evaluation_wall_runtime": float(result["post_evaluation_wall_runtime"]),
                "total_wall_runtime": float(result["total_wall_runtime"]),
                "penalized_runtime_par2": float(result["penalized_runtime_par2"]),
                "iterations": int(result["iterations"]),
                "scenario_count": int(result["metadata"]["committed_scenario_count"]),
                "cut_count": int(result["cuts"]),
            }
        )
    groups = []
    for scale in SCALES:
        for rho in RHOS:
            rows = [row for row in full_rows if row["scale"] == scale and row["rho"] == rho]
            _check(len(rows) == 10 and sorted(row["seed"] for row in rows) == SEEDS, "paper group is not ten paired seeds")
            groups.append(
                {"scale": scale, "rho": rho, "seed_count": 10, "certified_count": 10,
                 "metrics": {name: _stats(row[name] for row in rows) for name in metric_names}}
            )
    per_rho = []
    raw_pvalues = []
    for rho in RHOS:
        differences = []
        seed_map = {}
        for seed in SEEDS:
            large = next(row for row in full_rows if row["scale"] == "large" and row["rho"] == rho and row["seed"] == seed)
            medium = next(row for row in full_rows if row["scale"] == "medium_large" and row["rho"] == rho and row["seed"] == seed)
            difference = large["objective_t"] - medium["objective_t"]
            differences.append(difference)
            seed_map[seed] = difference
        pvalue = _permutation_pvalue(differences)
        raw_pvalues.append(pvalue)
        per_rho.append(
            {"rho": rho, "paired_seed_count": 10, "large_minus_medium_large_objective_t": _stats(differences),
             "cluster_bootstrap_95_percent_ci": _bootstrap_ci(seed_map), "paired_permutation_pvalue_raw": pvalue}
        )
    for row, adjusted in zip(per_rho, _holm(raw_pvalues)):
        row["paired_permutation_pvalue_holm"] = adjusted
    overall_by_seed = {}
    for seed in SEEDS:
        large = math.fsum(row["objective_t"] for row in full_rows if row["scale"] == "large" and row["seed"] == seed) / len(RHOS)
        medium = math.fsum(row["objective_t"] for row in full_rows if row["scale"] == "medium_large" and row["seed"] == seed) / len(RHOS)
        overall_by_seed[seed] = large - medium
    paper = {
        "scope": "FINAL_HOLDOUT_only_D1_D2_excluded",
        "independent_unit": "seed",
        "seed_cluster_count": 10,
        "frontier_count": 100,
        "certified_count": 100,
        "par2_rule": {"basis": "algorithm_runtime", "unsolved_seconds": 3600.0},
        "runtime_fields_distinct": ["algorithm_runtime", "post_evaluation_wall_runtime", "total_wall_runtime"],
        "scale_rho_summaries": groups,
        "cross_scale_per_rho_paired": per_rho,
        "cross_scale_overall_seed_aggregated": {
            "paired_seed_count": 10, "rho_aggregated_within_seed_first": True,
            "large_minus_medium_large_objective_t": _stats(overall_by_seed.values()),
            "cluster_bootstrap_95_percent_ci": _bootstrap_ci(overall_by_seed),
            "paired_permutation_pvalue": _permutation_pvalue(list(overall_by_seed.values())),
        },
        "complete_seed_results": full_rows,
    }
    return paper, full_rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    columns = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(audit: dict[str, Any], records: list[dict[str, Any]], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    bound_rows = []
    corrected_rows = []
    for item in sorted(records, key=lambda value: (value["record"]["scale"], value["record"]["seed"], value["record"]["task_type"], str(value["record"]["rho"]))):
        record = item["record"]
        result = record["result"]
        base = dict(item["original_projection"])
        if item["bounds"] is None:
            extras = {
                "historical_recorded_lower_bound": result.get("lower_bound", "NOT_APPLICABLE"),
                "final_master_solver_best_bound": "NOT_APPLICABLE", "reported_certification_lower_bound": result.get("lower_bound", "NOT_APPLICABLE"),
                "upper_bound": result.get("upper_bound"), "raw_bound_crossing": "NOT_APPLICABLE",
                "bound_crossing_within_tolerance": False, "reported_gap": result.get("gap"),
                "final_exact_separation_objective_bound": "NOT_APPLICABLE",
            }
        else:
            extras = {key: value for key, value in item["bounds"].items() if key != "original_recorded_gap"}
            bound_rows.append({"run_key": record["run_key"], "run_directory_id": record["run_directory_id"], "scale": record["scale"], "seed": record["seed"], "rho": record["rho"], **item["bounds"]})
        corrected_rows.append(
            {**base, **extras, "objective_t": result.get("objective_t", "NOT_APPLICABLE"),
             "robust_minimum_fill_rate": result.get("robust_minimum_fill_rate", "NOT_APPLICABLE"),
             "iterations": result.get("iterations", "NOT_APPLICABLE"), "cut_count": result.get("cuts", "NOT_APPLICABLE"),
             "scenario_count": result.get("metadata", {}).get("committed_scenario_count", "NOT_APPLICABLE"),
             "actual_robust_cost": (item["post"] or {}).get("actual_robust_cost", "NOT_APPLICABLE"),
             "minimum_weighted_mean_fill_rate": (item["post"] or {}).get("minimum_weighted_mean_fill_rate", "NOT_APPLICABLE")}
        )
    corrected_summary = _summary_rows(corrected_rows)
    paper, full_rows = _paper_metrics(records)
    cross_rows = []
    for row in paper["cross_scale_per_rho_paired"]:
        cross_rows.append(
            {"scope": "per_rho", "rho": row["rho"], "paired_seed_count": 10,
             "mean_large_minus_medium_large_objective_t": row["large_minus_medium_large_objective_t"]["mean"],
             "bootstrap_ci_low": row["cluster_bootstrap_95_percent_ci"][0], "bootstrap_ci_high": row["cluster_bootstrap_95_percent_ci"][1],
             "paired_permutation_pvalue_raw": row["paired_permutation_pvalue_raw"], "paired_permutation_pvalue_holm": row["paired_permutation_pvalue_holm"]}
        )
    overall = paper["cross_scale_overall_seed_aggregated"]
    cross_rows.append(
        {"scope": "overall_rho_aggregated_within_seed", "rho": "ALL", "paired_seed_count": 10,
         "mean_large_minus_medium_large_objective_t": overall["large_minus_medium_large_objective_t"]["mean"],
         "bootstrap_ci_low": overall["cluster_bootstrap_95_percent_ci"][0], "bootstrap_ci_high": overall["cluster_bootstrap_95_percent_ci"][1],
         "paired_permutation_pvalue_raw": overall["paired_permutation_pvalue"], "paired_permutation_pvalue_holm": "NOT_APPLICABLE"}
    )
    anomaly = next(row for row in bound_rows if row["run_directory_id"] == "r_45f40e77afd919415d895390")
    reconciliation = {
        "classification": "reporting_and_lower_bound_ledger_semantics_error_within_frozen_numerical_tolerance",
        "global_tolerance": TOLERANCE,
        "known_anomaly": anomaly,
        "frontier_bounds": bound_rows,
        "mathematical_basis": {
            "final_master_is_a_relaxation": "It contains complete recourse blocks for an append-only subset of uncertainty scenarios, so its optimal solver bound is a lower bound for the full robust minimization problem.",
            "final_incumbent_is_an_upper_bound": "Complete exact separation returned an optimal nonpositive objective bound and certified the same incumbent robust feasible.",
            "optimality_sandwich": "The final master solver best bound equals the final master incumbent and UB; exact separation makes that incumbent feasible for the full robust problem.",
            "historical_max_role": "The historical max is retained only as trajectory/ledger evidence and is not used as the paper certification lower bound.",
            "reported_gap_formula": "max(0, UB - reported_certification_lower_bound) / max(1, abs(UB))",
        },
    }
    field_reconciliation = {
        "source_fields_immutable": True,
        "historical_recorded_lower_bound": "Original result.lower_bound; historical max ledger, preserved verbatim.",
        "final_master_solver_best_bound": "Last iteration master_solver_best_bound from the original run.",
        "reported_certification_lower_bound": "Final current master solver best bound used only in derived reporting.",
        "upper_bound": "Original certified robust incumbent/objective T.",
        "raw_bound_crossing": "historical_recorded_lower_bound - upper_bound; never hidden by gap clipping.",
        "bound_crossing_within_tolerance": "True only for a positive crossing no larger than 1e-4.",
        "reported_gap": "Recomputed from reported_certification_lower_bound using the frozen denominator.",
        "final_exact_separation_objective_bound": "Last complete exact-separation objective bound.",
    }
    decision = {
        "decision": "approve_final_holdout_after_reporting_reconciliation",
        "scientific_solution_valid": True,
        "optimization_rerun_required": False,
        "source_archive_modified": False,
        "formal_results_modified": False,
        "gurobi_called": False,
        "certified_runs": "120/120",
        "exact_frontier_certificates": "100/100",
        "post_evaluation_chunks": "13050/13050",
    }
    provenance = {
        "archive_filename": "fairness_hybrid_final_holdout_results.zip",
        "archive_sha256": audit["archive_sha256_before"], "archive_sha256_after_audit": audit["archive_sha256_after"],
        "entry_count": audit["entry_count"], "file_count": audit["file_count"], "directory_count": audit["directory_count"],
        "crc_valid": True, "read_only_audit": True, "original_archive_committed_to_git": False,
        "run_commit": RUN_COMMIT,
    }
    decision_md = f"""# Hybrid final holdout reconciliation decision

```yaml
decision: approve_final_holdout_after_reporting_reconciliation
scientific_solution_valid: true
optimization_rerun_required: false
```

The formal ZIP passed 120/120 run checks, 100/100 exact-certification checks, and 13,050/13,050 post-evaluation chunk SHA checks. The archive SHA remained `{audit['archive_sha256_before']}` before and after the read-only audit.

The sole crossing is `large`, seed 172, rho 0.10 (`r_45f40e77afd919415d895390`). The implementation stores `max(historical_lower_bound, current_master_ObjBound)` while the persistent master receives additional scenario blocks. Gurobi's numerically solved objective decreased by {anomaly['raw_bound_crossing']:.17g}, within the frozen 1e-4 tolerance. The original gap function clipped the resulting negative numerator to zero without separately recording the crossing.

Scientific validity does not rely on that historical maximum. In the final iteration, the current persistent scenario master was optimal and its solver best bound equaled both its incumbent and the reported UB ({anomaly['final_master_solver_best_bound']:.17g}). Complete exact separation was optimal with objective bound {anomaly['final_exact_separation_objective_bound']:.17g} and certified that same incumbent robust feasible. Thus the current master bound supplies the relaxation lower bound and exact separation supplies full robust feasibility; together they prove final optimality under the frozen protocol tolerance.

The derived reports therefore preserve `historical_recorded_lower_bound` as a trajectory field and use `final_master_solver_best_bound` as `reported_certification_lower_bound`. No run, checkpoint, instance, post-evaluation artifact, or source ZIP was changed, and no optimization or Gurobi call was made.
"""
    files: dict[str, Any] = {
        "source_archive_provenance.json": provenance,
        "final_holdout_audit.json": audit,
        "bound_reconciliation.json": reconciliation,
        "field_reconciliation.json": field_reconciliation,
        "paper_metrics.json": paper,
        "decision.json": decision,
    }
    for name, value in files.items():
        _write_json(output / name, value)
    _write_csv(output / "results.corrected.csv", corrected_rows)
    _write_csv(output / "summary.corrected.csv", corrected_summary)
    _write_csv(output / "cross_scale_summary.csv", cross_rows)
    (output / "decision.md").write_text(decision_md, encoding="utf-8", newline="\n")
    artifact_names = sorted([*files, "results.corrected.csv", "summary.corrected.csv", "cross_scale_summary.csv", "decision.md"])
    artifacts = {name: _file_sha(output / name) for name in artifact_names}
    _write_csv(output / "artifact_sha256.csv", [{"relative_path": name, "sha256": digest} for name, digest in artifacts.items()])
    return {**artifacts, "artifact_sha256.csv": _file_sha(output / "artifact_sha256.csv")}


def reconcile(archive: str | Path, repo_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    audit, records = audit_archive(archive, repo_root)
    artifacts = write_reports(audit, records, output_dir)
    return {"decision": "approve_final_holdout_after_reporting_reconciliation", "audit": audit, "artifacts": artifacts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = reconcile(args.archive, args.repo_root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
