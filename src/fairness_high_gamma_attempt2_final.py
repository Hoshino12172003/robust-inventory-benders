"""Read-only audit and deterministic paper reporting for High-Gamma Attempt 2.

This module only reads the frozen ZIP.  It intentionally does not import any
solver module and never imports gurobipy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile


ARCHIVE_SHA256 = "4F8CEC3F9EAF69B053AE3DBAE6C29D5AAEF7DAAD87C49A798039DBCF9FADD783"
RUN_COMMIT = "797caafd12c006e85bc3394b01905bbfb137b0a9"
CONFIG_SHA256 = "A377D5B040FED160B323B58D42D9FFD1DE57E52F6C64D2050D6667E47DCA9334"
PROTOCOL_SHA256 = "4C76B5C7A02E245174BE02B6FCEBBCD744EB6B684A1F0CA71D05964EB1F1A32F"
AUTHORIZATION_SHA256 = "262272FEAC62579BCF1FBD60A83381A1B4727162E6A33AC84EA1120967433F3A"
HYBRID_SHA256 = "8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0"
DIRECT_SHA256 = "4A0CA29C6367858A96D62F9B30DC45BBA969CBF0679E758EBB7229AFD999DFAF"
STAGE = "HIGH_GAMMA_EXTERNAL_BENCHMARK"
ATTEMPT = 2
SEEDS = [185, 186, 187, 188, 189]
GAMMAS = [2, 3, 4]
RHO = 0.025
SCENARIOS = {2: 211, 3: 1351, 4: 6196}
SOLVER_PARAMETERS = {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}
TOLERANCE = 1e-4
GENERATION_SCHEMA = "fairness_high_gamma_attempt2_final_reporting_v1"


class AuditError(RuntimeError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return bytes_sha256(canonical_json_bytes(value))


def _reject_constant(token: str) -> None:
    raise AuditError(f"non-finite JSON token: {token}")


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float):
        _check(math.isfinite(value), f"non-finite value: {label}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")


def _member(name: str) -> str:
    return name if name.startswith("./") else "./" + name


def _json(archive: ZipFile, name: str) -> dict[str, Any]:
    member = _member(name)
    try:
        value = json.loads(
            archive.read(member).decode("utf-8"), parse_constant=_reject_constant,
        )
    except KeyError as exc:
        raise AuditError(f"missing ZIP entry: {member}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON: {member}") from exc
    _check(isinstance(value, dict), f"JSON root is not an object: {member}")
    _finite_tree(value, member)
    return value


def _csv(archive: ZipFile, name: str) -> tuple[list[str], list[dict[str, str]]]:
    member = _member(name)
    try:
        reader = csv.DictReader(
            io.StringIO(archive.read(member).decode("utf-8"), newline=""),
            strict=True,
        )
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    except (KeyError, UnicodeDecodeError, csv.Error) as exc:
        raise AuditError(f"invalid CSV: {member}") from exc
    _check(fields and None not in fields, f"missing CSV header: {member}")
    forbidden = {
        "nan", "+nan", "-nan", "inf", "+inf", "-inf",
        "infinity", "+infinity", "-infinity",
    }
    for row in rows:
        _check(
            None not in row and all(value is not None for value in row.values()),
            f"malformed CSV row: {member}",
        )
        _check(
            not any(value.strip().lower() in forbidden for value in row.values()),
            f"non-finite CSV value: {member}",
        )
    return fields, rows


def _validate_frozen_yaml(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError("resolved config is not UTF-8") from exc
    _check(text.endswith("\n") and "\t" not in text, "resolved config formatting invalid")
    stack: list[int] = []
    for number, line in enumerate(text.splitlines(), 1):
        _check(line.rstrip() == line, f"resolved config trailing whitespace at line {number}")
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        _check(indent % 2 == 0, f"resolved config indentation invalid at line {number}")
        content = line.strip()
        if content.startswith("- "):
            _check(
                stack and indent in {stack[-1], stack[-1] - 2},
                f"orphan YAML list item at line {number}",
            )
        else:
            _check(":" in content and content.split(":", 1)[0], f"malformed YAML mapping at line {number}")
            while stack and indent < stack[-1]:
                stack.pop()
            if content.endswith(":"):
                stack.append(indent + 2)
    required = (
        "stage: HIGH_GAMMA_EXTERNAL_BENCHMARK",
        "execution_attempt: 2",
        "previous_attempt_results_reused: false",
        "scale: small",
        "rho: 0.025",
        "baseline_count: 15",
        "hybrid_frontier_count: 15",
        "direct_extensive_frontier_count: 15",
        "total_tasks: 45",
        "Threads: 1",
        "Seed: 0",
        "FeasibilityTol: 1.0e-07",
        "BendersStrategy_direct: 0",
        "algorithm_time_limit_seconds: 1800",
        "checkpoint_chunk_size: 25",
        "par2_basis: algorithm_runtime",
        "uncertified_seconds: 3600",
        "overwrite_supported: false",
    )
    for item in required:
        _check(item in text, f"resolved config missing frozen field: {item}")
    return text


def _number(value: Any, label: str) -> float:
    _check(type(value) in {int, float}, f"{label} must be numeric")
    result = float(value)
    _check(math.isfinite(result), f"{label} must be finite")
    return result


def _integer(value: Any, label: str) -> int:
    _check(type(value) is int and value >= 0, f"{label} must be a nonnegative integer")
    return value


def _directory_id(run_key: str) -> str:
    return "r_" + hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:24]


def _candidate(task_type: str) -> str:
    return {
        "baseline": "baseline",
        "hybrid_frontier": "certified_hybrid_scenario_benders_fairness",
        "direct_extensive_frontier": "gurobi_direct_extensive_form",
    }[task_type]


def _run_key(seed: int, gamma: int, task_type: str) -> str:
    return canonical_json_bytes({
        "candidate": _candidate(task_type),
        "execution_attempt": ATTEMPT,
        "gamma": gamma,
        "rho": "NOT_APPLICABLE" if task_type == "baseline" else "0.025",
        "scale": "small",
        "seed": seed,
        "stage": STAGE,
        "task_type": task_type,
    }).decode("utf-8")


def _stats(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    _check(ordered and all(math.isfinite(value) for value in ordered), "invalid statistics input")
    midpoint = len(ordered) // 2
    lower = ordered[:midpoint]
    upper = ordered[midpoint + (len(ordered) % 2):]
    q1 = statistics.median(lower) if lower else ordered[0]
    q3 = statistics.median(upper) if upper else ordered[-1]
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "sample_std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "iqr": q3 - q1,
        "min": ordered[0],
        "max": ordered[-1],
    }


def _validate_checkpoint(checkpoint: dict[str, Any], run: dict[str, Any]) -> None:
    checksum = checkpoint.get("checkpoint_sha256")
    unsigned = dict(checkpoint)
    unsigned.pop("checkpoint_sha256", None)
    _check(checksum == canonical_sha256(unsigned), "algorithm checkpoint SHA mismatch")
    identity = checkpoint.get("identity")
    _check(isinstance(identity, dict), "checkpoint identity missing")
    for field in (
        "run_key", "run_directory_id", "stage", "scale", "task_type", "seed",
        "gamma", "rho", "execution_attempt", "git_commit", "config_file_sha256",
        "resolved_config_file_sha256", "protocol_sha256", "candidate_sha256",
        "instance_sha256", "instance_canonical_sha256",
        "instance_archive_file_sha256", "instance_identity_sha256",
        "baseline_run_key", "solver_parameters", "requested_gamma",
        "gamma_target", "active_gamma", "gamma_schedule", "scenario_count",
    ):
        if field in identity or field in run:
            _check(identity.get(field) == run.get(field), f"checkpoint identity drift: {field}")
    checkpoint_result = checkpoint.get("result")
    run_result = run.get("result")
    _check(isinstance(checkpoint_result, dict) and isinstance(run_result, dict), "checkpoint result missing")
    for key, value in checkpoint_result.items():
        _check(run_result.get(key) == value, f"checkpoint/run scientific result mismatch: {key}")


def _validate_anchor(anchor: dict[str, Any], baseline: dict[str, Any]) -> None:
    checksum = anchor.get("anchor_sha256")
    _check(
        isinstance(checksum, str) and len(checksum) == 64
        and all(character in "0123456789ABCDEF" for character in checksum),
        "anchor SHA format mismatch",
    )
    result = baseline["result"]
    _check(anchor["value"] == result["upper_bound"], "anchor value/baseline UB mismatch")
    _check(anchor["value_hex"] == float(anchor["value"]).hex(), "anchor float hex mismatch")
    _check(anchor["anchor_value_hex"] == anchor["value_hex"], "anchor hex aliases mismatch")
    for field in (
        "seed", "gamma", "scale", "execution_attempt", "baseline_run_key",
        "instance_canonical_sha256", "instance_archive_file_sha256",
        "instance_identity_sha256", "requested_gamma", "gamma_target",
        "active_gamma", "gamma_schedule", "scenario_count",
    ):
        expected = baseline["run_key"] if field == "baseline_run_key" else baseline.get(field)
        _check(anchor.get(field) == expected, f"anchor identity drift: {field}")


def _validate_hybrid(result: dict[str, Any]) -> dict[str, Any]:
    _check(result.get("status") == "optimal", "Hybrid status is not optimal")
    lower = _number(result.get("lower_bound"), "Hybrid LB")
    upper = _number(result.get("upper_bound"), "Hybrid UB")
    _check(lower <= upper + TOLERANCE, "Hybrid LB/UB crossing exceeds tolerance")
    log = result.get("iteration_log")
    metadata = result.get("metadata")
    _check(isinstance(log, list) and log, "Hybrid iteration log missing")
    _check(isinstance(metadata, dict), "Hybrid metadata missing")
    scenarios = metadata.get("committed_scenario_sha256_values")
    cuts = metadata.get("committed_farkas_cut_sha256_values")
    _check(isinstance(scenarios, list) and isinstance(cuts, list), "Hybrid commitment SHA ledger missing")
    _check(len(scenarios) == len(set(scenarios)), "duplicate committed scenario SHA")
    _check(len(cuts) == len(set(cuts)), "duplicate committed cut SHA")
    _check(metadata.get("committed_scenario_count") == len(scenarios), "committed scenario count mismatch")
    _check(result.get("cuts") == len(cuts), "committed cut count mismatch")
    scenario_counts = [_integer(item.get("scenario_count"), "iteration scenario_count") for item in log]
    _check(
        all(0 <= right - left <= 1 for left, right in zip(scenario_counts, scenario_counts[1:])),
        "committed scenario ledger is not append-only",
    )
    committed_scenarios = [
        item["committed_scenario_sha256"] for item in log
        if item.get("committed_scenario_sha256") is not None
    ]
    committed_cuts = [
        item["committed_farkas_cut_sha256"] for item in log
        if item.get("committed_farkas_cut_sha256") is not None
    ]
    _check(len(committed_scenarios) == len(set(committed_scenarios)), "scenario replacement/recommit detected")
    _check(len(committed_cuts) == len(set(committed_cuts)), "cut replacement/recommit detected")
    _check(set(committed_scenarios).issubset(scenarios), "iteration scenario SHA absent from final ledger")
    _check(set(committed_cuts).issubset(cuts), "iteration cut SHA absent from final ledger")
    final = log[-1]
    _check(final.get("final_exact_separation_performed") is True, "final exact separation missing")
    _check(final.get("separation_status") == "optimal", "final separation status is not optimal")
    _check(final.get("robust_feasibility_certified") is True, "final robust certificate missing")
    objective_bound = _number(final.get("separation_objective_bound"), "final separation objective bound")
    _check(objective_bound <= TOLERANCE, "final separation objective bound exceeds tolerance")
    _check(metadata.get("robust_feasibility_certified") is True, "metadata robust certificate missing")
    _check(
        all(item.get("robust_feasibility_certified") is not True for item in log[:-1]),
        "premature robust certification in iteration history",
    )
    counters = result.get("reporting_counters")
    _check(isinstance(counters, dict), "Hybrid reporting counters missing")
    _check(
        counters.get("unique_committed_scenario_blocks") == len(scenarios)
        and counters.get("committed_farkas_cuts") == len(cuts),
        "Hybrid counter/ledger mismatch",
    )
    return {
        "final_separation_objective_bound": objective_bound,
        "scenario_blocks": len(scenarios),
        "cuts": len(cuts),
        **{key: _integer(counters.get(key), key) for key in (
            "candidate_pool_maximum_size", "eviction_count",
            "rediscovered_evicted_scenario_count", "duplicate_proposal_count",
            "maximum_consecutive_non_improving_iterations",
        )},
    }


def _artifact_tree_sha256(archive: ZipFile, prefix: str) -> str:
    normalized = _member(prefix).rstrip("/") + "/"
    entries = []
    for name in sorted(
        item.filename for item in archive.infolist()
        if not item.is_dir() and item.filename.startswith(normalized)
    ):
        entries.append({
            "path": name[len(normalized):],
            "sha256": bytes_sha256(archive.read(name)),
        })
    return canonical_sha256(entries)


def _validate_evaluation_tree(
    archive: ZipFile,
    root: str,
    *,
    run: dict[str, Any],
    expected_count: int,
    baseline_lifting: bool,
) -> dict[str, Any]:
    index = _json(archive, f"{root}/checkpoint/index.json")
    final = _json(archive, f"{root}/post_evaluation.json")
    identity = final.get("identity")
    _check(isinstance(identity, dict), "post-evaluation identity missing")
    if "identity" in index:
        _check(index.get("identity") == identity, "post-evaluation index/final identity mismatch")
    _check(
        str(index.get("identity_sha256")).upper() == canonical_sha256(identity)
        and str(final.get("identity_sha256")).upper() == canonical_sha256(identity),
        "post-evaluation identity SHA mismatch",
    )
    _check(identity.get("run_key") == run["run_key"], "post-evaluation run key mismatch")
    _check(identity.get("scenario_count") == expected_count, "post-evaluation scenario identity mismatch")
    _check(identity.get("run_execution_attempt") == ATTEMPT, "post-evaluation attempt mismatch")
    chunks = index.get("chunks")
    _check(isinstance(chunks, list), "post-evaluation chunk index missing")
    expected_chunks = math.ceil(expected_count / 25)
    _check(len(chunks) == expected_chunks, "post-evaluation chunk count mismatch")
    records_seen = 0
    evidence_seen = 0
    maximum_residual = 0.0
    scenario_names: list[str] = []
    all_evidence: list[dict[str, Any]] = []
    for expected_index, item in enumerate(chunks):
        _check(item.get("chunk_index") == expected_index, "post-evaluation chunk order mismatch")
        relative = item.get("relative_path")
        _check(relative == f"checkpoint/chunk_{expected_index:05d}.json", "post-evaluation chunk path mismatch")
        name = _member(f"{root}/{relative}")
        raw = archive.read(name)
        _check(bytes_sha256(raw) == str(item.get("sha256")).upper(), "post-evaluation chunk SHA mismatch")
        chunk = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        _finite_tree(chunk, name)
        _check(chunk.get("identity_sha256") == index.get("identity_sha256"), "chunk identity SHA mismatch")
        records = chunk.get("records")
        _check(isinstance(records, list), "chunk records missing")
        _check(chunk.get("scenario_start") == records_seen, "chunk scenario_start mismatch")
        _check(chunk.get("scenario_end_exclusive") == records_seen + len(records), "chunk scenario_end mismatch")
        _check(item.get("scenario_count") == len(records), "chunk index scenario count mismatch")
        for record in records:
            _check(record.get("scenario_index") == records_seen, "scenario sequence index mismatch")
            _check(record.get("error") is None, "post-evaluation scenario error")
            scenario_name = record.get("scenario_name")
            _check(isinstance(scenario_name, str) and scenario_name, "empty scenario name")
            scenario_names.append(scenario_name)
            evidence = record.get("acceptance_evidence")
            _check(isinstance(evidence, list) and len(evidence) == 6, "scenario acceptance evidence count mismatch")
            for proof in evidence:
                _check(proof.get("accepted") is True, "rejected acceptance evidence")
                residual = _number(proof.get("residual"), "acceptance residual")
                threshold = _number(proof.get("acceptance_threshold"), "acceptance threshold")
                _check(residual <= threshold, "acceptance residual exceeds threshold")
                if proof.get("constraint_type") == "robust_cost_budget":
                    _check(
                        _number(proof.get("lhs"), "cost lhs")
                        <= _number(proof.get("rhs"), "cost rhs") + threshold,
                        "scenario cost budget violation",
                    )
                maximum_residual = max(maximum_residual, residual)
                all_evidence.append(proof)
            evidence_seen += len(evidence)
            records_seen += 1
    _check(records_seen == expected_count, "post-evaluation scenario total mismatch")
    _check(len(scenario_names) == len(set(scenario_names)), "duplicate post-evaluation scenario")
    evaluation = final.get("evaluation")
    _check(isinstance(evaluation, dict), "post-evaluation final payload missing")
    _check(
        evaluation.get("valid") is True
        and evaluation.get("errors") == []
        and evaluation.get("objective_t_consistent") is True,
        "post-evaluation final is invalid",
    )
    _check(evaluation.get("scenario_count") == expected_count, "final scenario count mismatch")
    final_evidence = evaluation.get("acceptance_evidence")
    _check(
        isinstance(final_evidence, list) and len(final_evidence) == evidence_seen + 1,
        f"chunk/final acceptance evidence count mismatch: {evidence_seen}/"
        f"{len(final_evidence) if isinstance(final_evidence, list) else 'invalid'}",
    )
    for proof in final_evidence:
        _check(proof.get("accepted") is True, "final acceptance evidence rejected")
        final_residual = _number(proof.get("residual"), "final acceptance residual")
        _check(
            final_residual <= _number(proof.get("acceptance_threshold"), "final acceptance threshold"),
            "final acceptance residual exceeds threshold",
        )
        maximum_residual = max(maximum_residual, final_residual)
    objective_t = 1.0 if baseline_lifting else _number(run["result"].get("objective_t"), "objective T")
    if not baseline_lifting:
        _check(
            math.isclose(
                _number(evaluation.get("realized_worst_shortage_rate"), "realized worst shortage"),
                objective_t,
                rel_tol=0,
                abs_tol=1e-7,
            ),
            "objective T/post-evaluation mismatch",
        )
        _check(
            math.isclose(
                _number(run["result"].get("robust_minimum_fill_rate"), "certified fill rate"),
                1.0 - objective_t,
                rel_tol=0,
                abs_tol=1e-12,
            ),
            "certified fill-rate identity mismatch",
        )
    cost_budget = (
        (1.0 + RHO) * _number(run["result"].get("upper_bound"), "baseline UB")
        if baseline_lifting
        else _number(run.get("cost_budget"), "cost budget")
    )
    _check(
        _number(evaluation.get("actual_robust_cost"), "actual robust cost")
        <= cost_budget + 1e-7,
        "post-evaluation cost budget violation",
    )
    return {
        "chunk_count": len(chunks),
        "scenario_count": records_seen,
        "acceptance_evidence_count": len(final_evidence),
        "maximum_acceptance_residual": maximum_residual,
        "evaluation": evaluation,
        "artifact_tree_sha256": _artifact_tree_sha256(archive, root),
    }


def _result_projection(run: dict[str, Any]) -> dict[str, Any]:
    result = run["result"]
    post = result.get("post_evaluation") or {}
    counters = result.get("reporting_counters") or {}
    return {
        "run_key": run["run_key"],
        "run_directory_id": run["run_directory_id"],
        "stage": run["stage"],
        "execution_attempt": run["execution_attempt"],
        "git_commit": run["git_commit"],
        "config_file_sha256": run["config_file_sha256"],
        "protocol_sha256": run["protocol_sha256"],
        "scale": run["scale"],
        "seed": run["seed"],
        "gamma": run["gamma"],
        "rho": run["rho"],
        "task_type": run["task_type"],
        "candidate": run["candidate"],
        "candidate_sha256": run["candidate_sha256"],
        "instance_canonical_sha256": run["instance_canonical_sha256"],
        "instance_archive_file_sha256": run["instance_archive_file_sha256"],
        "baseline_run_key": run["baseline_run_key"],
        "anchor_sha256": run.get("anchor_sha256", "NOT_APPLICABLE"),
        "scientific_status": run["scientific_status"],
        "algorithm_status": run["algorithm_status"],
        "algorithm_runtime": result["algorithm_runtime"],
        "penalized_runtime_par2": result["penalized_runtime_par2"],
        "post_evaluation_wall_runtime": result.get("post_evaluation_wall_runtime", 0.0),
        "total_wall_runtime": result["total_wall_runtime"],
        "lower_bound": result.get("lower_bound", "NOT_APPLICABLE"),
        "upper_bound": result.get("upper_bound", "NOT_APPLICABLE"),
        "gap": result.get("gap", "NOT_APPLICABLE"),
        "objective_t": result.get("objective_t", "NOT_APPLICABLE"),
        "robust_minimum_fill_rate": result.get("robust_minimum_fill_rate", "NOT_APPLICABLE"),
        "actual_robust_cost": post.get("actual_robust_cost", "NOT_APPLICABLE"),
        "iterations": result.get("iterations", "NOT_APPLICABLE"),
        "certified_cuts": result.get("cuts", "NOT_APPLICABLE"),
        "committed_scenario_blocks": (result.get("metadata") or {}).get("committed_scenario_count", 0),
        "master_runtime": result.get("master_runtime", "NOT_APPLICABLE"),
        "separation_runtime": result.get("separation_runtime", "NOT_APPLICABLE"),
        "model_build_runtime": result.get("model_build_runtime", "NOT_APPLICABLE"),
        "optimize_runtime": result.get("optimize_runtime", "NOT_APPLICABLE"),
        "rows": result.get("rows", "NOT_APPLICABLE"),
        "columns": result.get("columns", "NOT_APPLICABLE"),
        "binaries": result.get("binaries", "NOT_APPLICABLE"),
        "continuous_variables": result.get("continuous_variables", "NOT_APPLICABLE"),
        "nonzeros": result.get("nonzeros", "NOT_APPLICABLE"),
        "incumbent": result.get("incumbent", "NOT_APPLICABLE"),
        "objective_bound": result.get("objective_bound", "NOT_APPLICABLE"),
        "candidate_pool_maximum_size": counters.get("candidate_pool_maximum_size", "NOT_APPLICABLE"),
        "proposal_evictions": counters.get("eviction_count", "NOT_APPLICABLE"),
        "rediscovered_evicted_scenarios": counters.get("rediscovered_evicted_scenario_count", "NOT_APPLICABLE"),
        "duplicate_proposals": counters.get("duplicate_proposal_count", "NOT_APPLICABLE"),
        "maximum_consecutive_non_improving_iterations": counters.get(
            "maximum_consecutive_non_improving_iterations", "NOT_APPLICABLE",
        ),
    }


def _same_csv_value(expected: Any, actual: str) -> bool:
    if expected is None:
        return actual == ""
    if expected == "NOT_APPLICABLE":
        return actual == expected
    if isinstance(expected, bool):
        return actual == str(expected)
    if type(expected) in {int, float}:
        try:
            return math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12)
        except ValueError:
            return False
    return str(expected) == actual


def audit_archive(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    before = file_sha256(path)
    _check(before == ARCHIVE_SHA256, f"ZIP SHA mismatch: {before}")
    with ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        _check(len(names) == len(set(names)), "duplicate ZIP member")
        _check(archive.testzip() is None, "ZIP CRC failure")
        files = [name for name in names if not name.endswith("/")]
        for root in (
            "manifest.json", "run_manifest.json", "resolved_config.yaml",
            "results.csv", "summary.csv", "paired_comparison.csv",
            "high_gamma_stability.csv", "model_size_summary.csv",
        ):
            _check(_member(root) in names, f"missing root artifact: {root}")
        manifest = _json(archive, "manifest.json")
        _check(_json(archive, "run_manifest.json") == manifest, "manifest/run_manifest mismatch")
        resolved_raw = archive.read(_member("resolved_config.yaml"))
        _validate_frozen_yaml(resolved_raw)
        resolved_archive_sha256 = bytes_sha256(resolved_raw)
        _, source_results = _csv(archive, "results.csv")
        _csv(archive, "summary.csv")
        _, source_paired = _csv(archive, "paired_comparison.csv")
        _csv(archive, "high_gamma_stability.csv")
        _csv(archive, "model_size_summary.csv")
        identity = manifest.get("identity")
        _check(isinstance(identity, dict), "manifest identity missing")
        expected_identity = {
            "stage": STAGE,
            "scale": "small",
            "seeds": SEEDS,
            "gamma": GAMMAS,
            "rho": RHO,
            "execution_attempt": ATTEMPT,
            "git_commit": RUN_COMMIT,
            "config_file_sha256": CONFIG_SHA256,
            "protocol_sha256": PROTOCOL_SHA256,
            "solver_parameters": SOLVER_PARAMETERS,
            "previous_attempt_results_reused": False,
        }
        for key, expected in expected_identity.items():
            _check(identity.get(key) == expected, f"manifest identity drift: {key}")
        auth_value = identity.get("authorization_file_sha256", identity.get("authorization_sha256"))
        _check(auth_value in (None, AUTHORIZATION_SHA256), "authorization SHA identity mismatch")
        _check(manifest.get("completed_run_count") == 45, "manifest completed count mismatch")
        forward = manifest.get("run_key_to_directory_id")
        reverse = manifest.get("directory_id_to_run_key")
        _check(isinstance(forward, dict) and isinstance(reverse, dict), "manifest mappings missing")
        expected_keys = {
            _run_key(seed, gamma, task)
            for seed in SEEDS for gamma in GAMMAS
            for task in ("baseline", "hybrid_frontier", "direct_extensive_frontier")
        }
        _check(set(forward) == expected_keys and len(reverse) == 45, "45-run plan mismatch")
        _check(
            all(reverse.get(directory) == key and directory == _directory_id(key)
                for key, directory in forward.items()),
            "run key/directory mapping mismatch",
        )
        runs: dict[str, dict[str, Any]] = {}
        checkpoints = 0
        internal_checkpoints = 0
        bound_crossings: list[dict[str, Any]] = []
        for run_key in sorted(expected_keys):
            directory = forward[run_key]
            run = _json(archive, f"runs/{directory}/run.json")
            status = _json(archive, f"runs/{directory}/status.json")
            checkpoint = _json(archive, f"runs/{directory}/algorithm_checkpoint.json")
            checkpoints += 1
            _check(run.get("run_key") == run_key and run.get("run_directory_id") == directory, "run identity mismatch")
            key_identity = json.loads(run_key)
            _check(all(run.get(key) == value for key, value in key_identity.items()), "run key projection mismatch")
            _check(run.get("state") == "complete" and status.get("state") == "complete", "incomplete task")
            _check(status.get("scientific_status") == run.get("scientific_status"), "run/status scientific mismatch")
            _check(status.get("identity", {}).get("run_key") == run_key, "status identity mismatch")
            _check(run.get("execution_attempt") == ATTEMPT and run.get("previous_attempt_results_reused") is False, "attempt reuse drift")
            _check(run.get("git_commit") == RUN_COMMIT, "run Git identity mismatch")
            _check(run.get("config_file_sha256") == CONFIG_SHA256, "run config identity mismatch")
            _check(run.get("protocol_sha256") == PROTOCOL_SHA256, "run protocol identity mismatch")
            _check(run.get("solver_parameters") == SOLVER_PARAMETERS, "run solver identity mismatch")
            _check(run.get("scenario_count") == SCENARIOS[run["gamma"]], "run scenario count mismatch")
            expected_sha = HYBRID_SHA256 if run["task_type"] != "direct_extensive_frontier" else DIRECT_SHA256
            _check(run.get("candidate_sha256") == expected_sha, "candidate SHA mismatch")
            instance_name = f"instances/s{run['seed']}_g{run['gamma']}.json"
            raw_instance = archive.read(_member(instance_name))
            instance_archive = json.loads(raw_instance.decode("utf-8"), parse_constant=_reject_constant)
            _finite_tree(instance_archive, instance_name)
            canonical = canonical_sha256(instance_archive.get("instance"))
            identity_sha = canonical_sha256(instance_archive.get("identity"))
            _check(run.get("instance_archive_file_sha256") == bytes_sha256(raw_instance), "instance file SHA mismatch")
            _check(run.get("instance_canonical_sha256") == canonical == run.get("instance_sha256"), "instance canonical SHA mismatch")
            _check(run.get("instance_identity_sha256") == identity_sha, "instance identity SHA mismatch")
            _validate_checkpoint(checkpoint, run)
            lower, upper = run["result"].get("lower_bound"), run["result"].get("upper_bound")
            if type(lower) in {int, float} and type(upper) in {int, float} and float(lower) > float(upper):
                bound_crossings.append({
                    "run_key": run_key,
                    "task_type": run["task_type"],
                    "seed": run["seed"],
                    "gamma": run["gamma"],
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "crossing": float(lower) - float(upper),
                })
                _check(float(lower) <= float(upper) + TOLERANCE, "LB/UB crossing exceeds tolerance")
            if run["task_type"] == "hybrid_frontier":
                hidden = _json(archive, f"runs/{directory}/hybrid_internal_checkpoint.json")
                internal_checkpoints += 1
                _check(isinstance(hidden, dict), "Hybrid internal checkpoint invalid")
            runs[run_key] = run
        baseline_checks = 0
        lifting_chunks = lifting_scenarios = lifting_evidence = 0
        frontier_chunks = frontier_scenarios = frontier_evidence = 0
        maximum_residual = 0.0
        hybrid_details: dict[tuple[int, int], dict[str, Any]] = {}
        paired: list[dict[str, Any]] = []
        anchors = manifest.get("baseline_anchors")
        _check(isinstance(anchors, dict) and len(anchors) == 15, "baseline anchor mapping mismatch")
        for seed in SEEDS:
            for gamma in GAMMAS:
                baseline = runs[_run_key(seed, gamma, "baseline")]
                hybrid = runs[_run_key(seed, gamma, "hybrid_frontier")]
                direct = runs[_run_key(seed, gamma, "direct_extensive_frontier")]
                result = baseline["result"]
                for source in (baseline, result):
                    _check(
                        source.get("requested_gamma")
                        == source.get("gamma_target")
                        == source.get("active_gamma")
                        == gamma,
                        "baseline internal Gamma identity mismatch",
                    )
                    _check(source.get("gamma_schedule") == [gamma], "baseline Gamma schedule mismatch")
                    _check(source.get("scenario_count") == SCENARIOS[gamma], "baseline scenario count mismatch")
                _check(result.get("max_scenarios") == SCENARIOS[gamma], "baseline max_scenarios mismatch")
                _check(
                    result.get("status") == "optimal"
                    and result.get("valid_UB") is True
                    and _number(result.get("gap"), "baseline gap") <= TOLERANCE,
                    "baseline is not certified within tolerance",
                )
                anchor = anchors[f"s{seed}_g{gamma}"]
                _validate_anchor(anchor, baseline)
                lifting = _validate_evaluation_tree(
                    archive,
                    f"runs/{baseline['run_directory_id']}/baseline_t1_lifting",
                    run=baseline,
                    expected_count=SCENARIOS[gamma],
                    baseline_lifting=True,
                )
                lifting_summary = baseline.get("baseline_t1_lifting")
                _check(isinstance(lifting_summary, dict), "baseline lifting summary missing")
                _check(lifting_summary.get("acceptance_evidence") == lifting["evaluation"]["acceptance_evidence"], "baseline lifting evidence mismatch")
                _check(
                    lifting_summary.get("artifact_tree_sha256") == lifting["artifact_tree_sha256"],
                    "baseline lifting artifact tree SHA mismatch",
                )
                baseline_checks += 1
                lifting_chunks += lifting["chunk_count"]
                lifting_scenarios += lifting["scenario_count"]
                lifting_evidence += lifting["acceptance_evidence_count"]
                maximum_residual = max(maximum_residual, lifting["maximum_acceptance_residual"])
                for frontier in (hybrid, direct):
                    for field in (
                        "instance_sha256", "instance_canonical_sha256",
                        "instance_archive_file_sha256", "instance_identity_sha256",
                        "baseline_run_key", "anchor_sha256", "anchor_value_hex",
                        "cost_budget", "solver_parameters",
                    ):
                        expected = (
                            baseline["run_key"] if field == "baseline_run_key"
                            else anchor["anchor_sha256"] if field == "anchor_sha256"
                            else anchor["anchor_value_hex"] if field == "anchor_value_hex"
                            else (1.0 + RHO) * anchor["value"] if field == "cost_budget"
                            else baseline.get(field)
                        )
                        _check(frontier.get(field) == expected, f"paired identity mismatch: {field}")
                details = _validate_hybrid(hybrid["result"])
                hybrid_details[(seed, gamma)] = details
                _check(hybrid["scientific_status"] == "certified_robust_optimal", "Hybrid scientific status mismatch")
                hybrid_post = _validate_evaluation_tree(
                    archive,
                    f"runs/{hybrid['run_directory_id']}/post_evaluation",
                    run=hybrid,
                    expected_count=SCENARIOS[gamma],
                    baseline_lifting=False,
                )
                _check(
                    hybrid["result"].get("post_evaluation_artifact_tree_sha256")
                    == hybrid_post["artifact_tree_sha256"],
                    "Hybrid post-evaluation artifact tree SHA mismatch",
                )
                frontier_chunks += hybrid_post["chunk_count"]
                frontier_scenarios += hybrid_post["scenario_count"]
                frontier_evidence += hybrid_post["acceptance_evidence_count"]
                maximum_residual = max(maximum_residual, hybrid_post["maximum_acceptance_residual"])
                direct_result = direct["result"]
                _check(
                    direct_result.get("complete_model_built") is True
                    and direct_result.get("benders_strategy") == 0
                    and direct_result.get("resource_failure") is False,
                    "Direct deterministic equivalent identity mismatch",
                )
                build = _number(direct_result.get("model_build_runtime"), "Direct build runtime")
                optimize = _number(direct_result.get("optimize_runtime"), "Direct optimize runtime")
                runtime = _number(direct_result.get("algorithm_runtime"), "Direct algorithm runtime")
                _check(math.isclose(runtime, build + optimize, rel_tol=0, abs_tol=1e-9), "Direct runtime decomposition mismatch")
                if gamma in (2, 3):
                    _check(
                        direct_result.get("status") == "optimal"
                        and direct["scientific_status"] == "certified_robust_optimal"
                        and _number(direct_result.get("gap"), "Direct gap") <= TOLERANCE,
                        "Direct Gamma 2/3 certification mismatch",
                    )
                    direct_post = _validate_evaluation_tree(
                        archive,
                        f"runs/{direct['run_directory_id']}/post_evaluation",
                        run=direct,
                        expected_count=SCENARIOS[gamma],
                        baseline_lifting=False,
                    )
                    _check(
                        direct["result"].get("post_evaluation_artifact_tree_sha256")
                        == direct_post["artifact_tree_sha256"],
                        "Direct post-evaluation artifact tree SHA mismatch",
                    )
                    frontier_chunks += direct_post["chunk_count"]
                    frontier_scenarios += direct_post["scenario_count"]
                    frontier_evidence += direct_post["acceptance_evidence_count"]
                    maximum_residual = max(maximum_residual, direct_post["maximum_acceptance_residual"])
                else:
                    _check(
                        direct_result.get("status") == "time_limit"
                        and direct["scientific_status"] == "time_limit_uncertified"
                        and direct_result.get("incumbent") is None
                        and direct_result.get("upper_bound") is None
                        and direct_result.get("objective_t") is None
                        and direct_result.get("gap") is None
                        and _number(direct_result.get("penalized_runtime_par2"), "Direct PAR-2") == 3600.0,
                        "Direct Gamma 4 timeout classification mismatch",
                    )
                    _check(
                        not any(name.startswith(_member(f"runs/{direct['run_directory_id']}/post_evaluation/")) for name in names),
                        "uncertified Direct Gamma 4 has post-evaluation artifacts",
                    )
                both = gamma in (2, 3)
                hybrid_cost = hybrid_post["evaluation"]["actual_robust_cost"]
                direct_cost = (
                    direct["result"]["post_evaluation"]["actual_robust_cost"] if both else "NOT_APPLICABLE"
                )
                paired.append({
                    "seed": seed,
                    "gamma": gamma,
                    "scenario_count": SCENARIOS[gamma],
                    "instance_canonical_sha256": baseline["instance_canonical_sha256"],
                    "baseline_run_key": baseline["run_key"],
                    "anchor_sha256": anchor["anchor_sha256"],
                    "C_anchor": anchor["value"],
                    "cost_budget": (1.0 + RHO) * anchor["value"],
                    "hybrid_status": hybrid["scientific_status"],
                    "direct_status": direct["scientific_status"],
                    "hybrid_algorithm_runtime": hybrid["result"]["algorithm_runtime"],
                    "direct_algorithm_runtime": direct_result["algorithm_runtime"],
                    "hybrid_par2": hybrid["result"]["penalized_runtime_par2"],
                    "direct_par2": direct_result["penalized_runtime_par2"],
                    "hybrid_objective_t": hybrid["result"]["objective_t"],
                    "direct_objective_t": direct_result.get("objective_t", "NOT_APPLICABLE"),
                    "hybrid_actual_robust_cost": hybrid_cost,
                    "direct_actual_robust_cost": direct_cost,
                    "runtime_ratio_direct_over_hybrid": direct_result["algorithm_runtime"] / hybrid["result"]["algorithm_runtime"],
                    "par2_ratio_direct_over_hybrid": direct_result["penalized_runtime_par2"] / hybrid["result"]["penalized_runtime_par2"],
                    "objective_t_difference_direct_minus_hybrid": (
                        direct_result["objective_t"] - hybrid["result"]["objective_t"] if both else "NOT_APPLICABLE"
                    ),
                    "actual_robust_cost_difference_direct_minus_hybrid": (
                        direct_cost - hybrid_cost if both else "NOT_APPLICABLE"
                    ),
                    "certification_agreement": hybrid["scientific_status"] == direct["scientific_status"],
                    "hybrid_iterations": hybrid["result"]["iterations"],
                    "hybrid_certified_cuts": hybrid["result"]["cuts"],
                    "hybrid_committed_scenario_blocks": details["scenario_blocks"],
                    "hybrid_proposal_evictions": details["eviction_count"],
                    "hybrid_rediscovered_evicted_scenarios": details["rediscovered_evicted_scenario_count"],
                    "hybrid_duplicate_proposals": details["duplicate_proposal_count"],
                    "hybrid_final_exact_certified": True,
                    "direct_rows": direct_result["rows"],
                    "direct_columns": direct_result["columns"],
                    "direct_binaries": direct_result["binaries"],
                    "direct_continuous_variables": direct_result["continuous_variables"],
                    "direct_nonzeros": direct_result["nonzeros"],
                    "direct_has_incumbent": direct_result.get("incumbent") is not None,
                })
        rows = [_result_projection(runs[key]) for key in sorted(runs)]
        _check(len(source_results) == len(rows) == 45, "source results row count mismatch")
        source_by_key = {row["run_key"]: row for row in source_results}
        for row in rows:
            source = source_by_key.get(row["run_key"])
            _check(source is not None, "source results run missing")
            for field, expected in row.items():
                if field in source:
                    _check(_same_csv_value(expected, source[field]), f"source results projection mismatch: {field}")
        _check(len(source_paired) == len(paired) == 15, "source paired comparison count mismatch")
        source_pair_by_cell = {(int(row["seed"]), int(row["gamma"])): row for row in source_paired}
        pair_field_map = {
            "hybrid_status": "hybrid_status",
            "direct_status": "direct_status",
            "hybrid_algorithm_runtime": "hybrid_runtime",
            "direct_algorithm_runtime": "direct_runtime",
            "hybrid_par2": "hybrid_par2",
            "direct_par2": "direct_par2",
            "runtime_ratio_direct_over_hybrid": "runtime_ratio_direct_over_hybrid",
            "par2_ratio_direct_over_hybrid": "par2_ratio_direct_over_hybrid",
            "objective_t_difference_direct_minus_hybrid": "objective_difference_direct_minus_hybrid",
            "actual_robust_cost_difference_direct_minus_hybrid": "cost_difference_direct_minus_hybrid",
            "baseline_run_key": "baseline_run_key",
            "anchor_sha256": "anchor_sha256",
            "direct_rows": "direct_rows",
            "direct_columns": "direct_columns",
            "direct_nonzeros": "direct_nonzeros",
            "certification_agreement": "certification_agreement",
        }
        for pair in paired:
            source = source_pair_by_cell[(pair["seed"], pair["gamma"])]
            for field, source_field in pair_field_map.items():
                _check(
                    _same_csv_value(pair[field], source[source_field]),
                    f"source paired projection mismatch: {field}",
                )
        audit = {
            "generation_schema": GENERATION_SCHEMA,
            "decision": "approve_high_gamma_external_benchmark_attempt2",
            "scientific_solution_valid": True,
            "optimization_rerun_required": False,
            "hybrid_high_gamma_stability_supported_on_small_scale": True,
            "external_direct_solver_benchmark_valid": True,
            "archive": {
                "sha256_before": before,
                "entries": len(infos),
                "files": len(files),
                "directories": len(infos) - len(files),
                "duplicate_members": 0,
                "crc_errors": 0,
                "uncompressed_bytes": sum(info.file_size for info in infos),
            },
            "identity": {
                "git_commit": RUN_COMMIT,
                "config_sha256": CONFIG_SHA256,
                "protocol_sha256": PROTOCOL_SHA256,
                "authorization_sha256": AUTHORIZATION_SHA256,
                "hybrid_candidate_sha256": HYBRID_SHA256,
                "direct_candidate_sha256": DIRECT_SHA256,
                "execution_attempt": ATTEMPT,
                "previous_attempt_results_reused": False,
                "resolved_config_archive_file_sha256": resolved_archive_sha256,
            },
            "coverage": {
                "tasks": len(runs),
                "baselines": baseline_checks,
                "hybrid_frontiers": 15,
                "direct_frontiers": 15,
                "algorithm_checkpoints": checkpoints,
                "hybrid_internal_checkpoints": internal_checkpoints,
                "baseline_lifting_chunks": lifting_chunks,
                "frontier_post_evaluation_chunks": frontier_chunks,
                "all_chunks": lifting_chunks + frontier_chunks,
                "baseline_lifting_scenarios": lifting_scenarios,
                "frontier_post_evaluation_scenarios": frontier_scenarios,
                "all_scenario_records": lifting_scenarios + frontier_scenarios,
                "baseline_lifting_acceptance_evidence": lifting_evidence,
                "frontier_acceptance_evidence": frontier_evidence,
                "all_acceptance_evidence": lifting_evidence + frontier_evidence,
            },
            "certification": {
                "baseline": 15,
                "hybrid": {str(gamma): 5 for gamma in GAMMAS},
                "direct": {"2": 5, "3": 5, "4": 0},
                "direct_gamma4_time_limit": 5,
                "direct_gamma4_with_incumbent": 0,
                "bound_crossings_within_tolerance": len(bound_crossings),
                "bound_crossing_details": bound_crossings,
                "maximum_acceptance_residual": maximum_residual,
            },
            "anchor_maximum_relative_gap": max(
                (run["result"]["upper_bound"] - run["result"]["lower_bound"])
                / max(1.0, abs(run["result"]["upper_bound"]))
                for run in runs.values() if run["task_type"] == "baseline"
            ),
            "gurobipy_imported": False,
            "solver_called": False,
        }
    after = file_sha256(path)
    _check(after == before, "source ZIP changed during audit")
    audit["archive"]["sha256_after"] = after
    return audit, rows, paired


PROVENANCE_FIELDS = {
    "source_zip_sha256": ARCHIVE_SHA256,
    "run_git_commit": RUN_COMMIT,
    "config_sha256": CONFIG_SHA256,
    "protocol_sha256": PROTOCOL_SHA256,
    "generation_schema": GENERATION_SCHEMA,
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _check(rows, f"cannot write empty CSV: {path.name}")
    enriched = [{**PROVENANCE_FIELDS, **row} for row in rows]
    fields = list(dict.fromkeys(key for row in enriched for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(enriched)


def _front_matter() -> str:
    return (
        "---\n"
        f"source_zip_sha256: {ARCHIVE_SHA256}\n"
        f"run_git_commit: {RUN_COMMIT}\n"
        f"config_sha256: {CONFIG_SHA256}\n"
        f"protocol_sha256: {PROTOCOL_SHA256}\n"
        f"generation_schema: {GENERATION_SCHEMA}\n"
        "---\n\n"
    )


def _task_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for gamma in GAMMAS:
        for task in ("baseline", "hybrid_frontier", "direct_extensive_frontier"):
            chosen = [row for row in rows if row["gamma"] == gamma and row["task_type"] == task]
            runtime = _stats(row["algorithm_runtime"] for row in chosen)
            par2 = _stats(row["penalized_runtime_par2"] for row in chosen)
            item = {
                "gamma": gamma,
                "task_type": task,
                "planned": len(chosen),
                "completed": len(chosen),
                "certified": sum(row["scientific_status"] == "certified_robust_optimal" for row in chosen),
                "certification_rate": sum(row["scientific_status"] == "certified_robust_optimal" for row in chosen) / len(chosen),
            }
            item.update({f"algorithm_runtime_{key}": value for key, value in runtime.items()})
            item.update({f"par2_{key}": value for key, value in par2.items()})
            for metric in (
                "iterations", "certified_cuts", "committed_scenario_blocks",
                "proposal_evictions", "rediscovered_evicted_scenarios",
                "duplicate_proposals",
            ):
                numeric = [
                    float(row[metric]) for row in chosen
                    if type(row.get(metric)) in {int, float}
                ]
                if numeric:
                    item.update({f"{metric}_{key}": value for key, value in _stats(numeric).items()})
            output.append(item)
    return output


def _pair_summary(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for gamma in GAMMAS:
        chosen = [row for row in paired if row["gamma"] == gamma]
        hybrid_runtimes = [float(row["hybrid_algorithm_runtime"]) for row in chosen]
        direct_runtimes = [float(row["direct_algorithm_runtime"]) for row in chosen]
        ratios = [float(row["runtime_ratio_direct_over_hybrid"]) for row in chosen]
        par2_ratios = [float(row["par2_ratio_direct_over_hybrid"]) for row in chosen]
        row = {
            "gamma": gamma,
            "scenario_count": SCENARIOS[gamma],
            "planned_pairs": len(chosen),
            "hybrid_certified": sum(item["hybrid_status"] == "certified_robust_optimal" for item in chosen),
            "direct_certified": sum(item["direct_status"] == "certified_robust_optimal" for item in chosen),
            "certification_agreement_count": sum(item["certification_agreement"] for item in chosen),
            "direct_timeout_count": sum(item["direct_status"] == "time_limit_uncertified" for item in chosen),
            "direct_incumbent_count": sum(item["direct_has_incumbent"] for item in chosen),
            "runtime_ratio_of_means": statistics.fmean(direct_runtimes) / statistics.fmean(hybrid_runtimes),
            "paired_runtime_ratio_mean": statistics.fmean(ratios),
            "paired_runtime_ratio_median": statistics.median(ratios),
            "paired_par2_ratio_mean": statistics.fmean(par2_ratios),
            "paired_par2_ratio_median": statistics.median(par2_ratios),
        }
        row.update({f"hybrid_runtime_{key}": value for key, value in _stats(hybrid_runtimes).items()})
        row.update({f"direct_runtime_{key}": value for key, value in _stats(direct_runtimes).items()})
        row.update({f"hybrid_par2_{key}": value for key, value in _stats(float(item["hybrid_par2"]) for item in chosen).items()})
        row.update({f"direct_par2_{key}": value for key, value in _stats(float(item["direct_par2"]) for item in chosen).items()})
        both = [item for item in chosen if item["hybrid_status"] == item["direct_status"] == "certified_robust_optimal"]
        if both:
            row["objective_t_difference_max_abs"] = max(
                abs(float(item["objective_t_difference_direct_minus_hybrid"])) for item in both
            )
            row["actual_robust_cost_difference_max_abs"] = max(
                abs(float(item["actual_robust_cost_difference_direct_minus_hybrid"])) for item in both
            )
        else:
            row["objective_t_difference_max_abs"] = "NOT_APPLICABLE"
            row["actual_robust_cost_difference_max_abs"] = "NOT_APPLICABLE"
        output.append(row)
    return output


def _anchor_rows(rows: list[dict[str, Any]], paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_by_cell = {(row["seed"], row["gamma"]): row for row in paired}
    output = []
    for row in rows:
        if row["task_type"] != "baseline":
            continue
        anchor = pair_by_cell[(row["seed"], row["gamma"])]
        lower = float(row["lower_bound"])
        upper = float(row["upper_bound"])
        relative = (upper - lower) / max(1.0, abs(upper))
        output.append({
            "seed": row["seed"],
            "gamma": row["gamma"],
            "scenario_count": SCENARIOS[int(row["gamma"])],
            "baseline_run_key": row["run_key"],
            "baseline_lower_bound": lower,
            "baseline_upper_bound": upper,
            "baseline_relative_certified_gap": relative,
            "C_anchor": anchor["C_anchor"],
            "cost_budget": anchor["cost_budget"],
            "anchor_sha256": anchor["anchor_sha256"],
        })
    return sorted(output, key=lambda item: (item["gamma"], item["seed"]))


def _model_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    previous = None
    for gamma in GAMMAS:
        chosen = [row for row in paired if row["gamma"] == gamma]
        item: dict[str, Any] = {"gamma": gamma, "scenario_count": SCENARIOS[gamma]}
        for metric in ("direct_rows", "direct_columns", "direct_binaries", "direct_continuous_variables", "direct_nonzeros"):
            values = [float(row[metric]) for row in chosen]
            item.update({f"{metric}_{key}": value for key, value in _stats(values).items()})
        if previous is None:
            item.update({
                "scenario_growth_from_previous": "NOT_APPLICABLE",
                "columns_growth_from_previous": "NOT_APPLICABLE",
                "nonzeros_growth_from_previous": "NOT_APPLICABLE",
            })
        else:
            item.update({
                "scenario_growth_from_previous": item["scenario_count"] / previous["scenario_count"],
                "columns_growth_from_previous": item["direct_columns_mean"] / previous["direct_columns_mean"],
                "nonzeros_growth_from_previous": item["direct_nonzeros_mean"] / previous["direct_nonzeros_mean"],
            })
        output.append(item)
        previous = item
    return output


def _pool_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for gamma in GAMMAS:
        chosen = [row for row in paired if row["gamma"] == gamma]
        item = {
            "gamma": gamma,
            "scenario_count": SCENARIOS[gamma],
            "hybrid_certified": sum(row["hybrid_status"] == "certified_robust_optimal" for row in chosen),
            "hybrid_certification_rate": sum(row["hybrid_status"] == "certified_robust_optimal" for row in chosen) / len(chosen),
            "all_final_exact_certified": all(row["hybrid_final_exact_certified"] for row in chosen),
        }
        for metric in (
            "hybrid_iterations", "hybrid_certified_cuts",
            "hybrid_committed_scenario_blocks", "hybrid_proposal_evictions",
            "hybrid_rediscovered_evicted_scenarios", "hybrid_duplicate_proposals",
        ):
            item.update({f"{metric}_{key}": value for key, value in _stats(float(row[metric]) for row in chosen).items()})
        output.append(item)
    return output


def _write_markdown_tables(
    output: Path,
    pair_summary: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
) -> None:
    main = [
        "# High-Gamma stress test and external solver benchmark",
        "",
        "| Gamma | Scenarios | Hybrid certified | Hybrid mean runtime (s) | Direct certified | Direct mean runtime (s) | Direct mean PAR-2 (s) | Ratio of means | Median paired ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pair_summary:
        main.append(
            f"| {row['gamma']} | {row['scenario_count']} | {row['hybrid_certified']}/5 | "
            f"{row['hybrid_runtime_mean']:.3f} | {row['direct_certified']}/5 | "
            f"{row['direct_runtime_mean']:.3f} | {row['direct_par2_mean']:.3f} | "
            f"{row['runtime_ratio_of_means']:.2f} | {row['paired_runtime_ratio_median']:.2f} |"
        )
    (output / "table_high_gamma_main.md").write_text(
        _front_matter() + "\n".join(main) + "\n", encoding="utf-8", newline="\n",
    )
    model = [
        "# Direct deterministic equivalent model size",
        "",
        "| Gamma | Scenarios | Mean rows | Mean columns | Mean binaries | Mean continuous | Mean nonzeros | Column growth | Nonzero growth |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        cg = row["columns_growth_from_previous"]
        ng = row["nonzeros_growth_from_previous"]
        model.append(
            f"| {row['gamma']} | {row['scenario_count']} | {row['direct_rows_mean']:.0f} | "
            f"{row['direct_columns_mean']:.0f} | {row['direct_binaries_mean']:.0f} | "
            f"{row['direct_continuous_variables_mean']:.0f} | {row['direct_nonzeros_mean']:.0f} | "
            f"{cg if isinstance(cg, str) else f'{cg:.2f}x'} | {ng if isinstance(ng, str) else f'{ng:.2f}x'} |"
        )
    (output / "table_external_solver_model_size.md").write_text(
        _front_matter() + "\n".join(model) + "\n", encoding="utf-8", newline="\n",
    )
    pool = [
        "# Rolling proposal pool stability",
        "",
        "| Gamma | Hybrid certified | Mean iterations | Mean cuts | Mean blocks | Mean evictions | Mean rediscoveries | Mean duplicates | Final exact |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in pool_rows:
        pool.append(
            f"| {row['gamma']} | {row['hybrid_certified']}/5 | {row['hybrid_iterations_mean']:.1f} | "
            f"{row['hybrid_certified_cuts_mean']:.1f} | {row['hybrid_committed_scenario_blocks_mean']:.1f} | "
            f"{row['hybrid_proposal_evictions_mean']:.1f} | {row['hybrid_rediscovered_evicted_scenarios_mean']:.1f} | "
            f"{row['hybrid_duplicate_proposals_mean']:.1f} | yes |"
        )
    (output / "table_high_gamma_pool_stability.md").write_text(
        _front_matter() + "\n".join(pool) + "\n", encoding="utf-8", newline="\n",
    )
    anchors = [
        "# Certified anchor quality",
        "",
        "| Seed | Gamma | Baseline LB | C_anchor (certified UB) | Relative gap | Cost budget |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in anchor_rows:
        anchors.append(
            f"| {row['seed']} | {row['gamma']} | {row['baseline_lower_bound']:.6f} | "
            f"{row['C_anchor']:.6f} | {row['baseline_relative_certified_gap']:.8f} | {row['cost_budget']:.6f} |"
        )
    (output / "table_anchor_quality.md").write_text(
        _front_matter() + "\n".join(anchors) + "\n", encoding="utf-8", newline="\n",
    )


def _plot(output: Path, pair_summary: list[dict[str, Any]], model_rows: list[dict[str, Any]], pool_rows: list[dict[str, Any]]) -> None:
    from PIL import Image, PngImagePlugin
    from reportlab.graphics import renderPDF
    from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen import canvas

    width, height = 480, 310
    x0, y0, chart_width, chart_height = 62, 48, 360, 205
    gammas = [2, 3, 4]
    x_positions = {gamma: x0 + index * chart_width / 2 for index, gamma in enumerate(gammas)}

    def label(drawing: Drawing, x: float, y: float, value: str, size: float = 8, anchor: str = "middle") -> None:
        drawing.add(String(x, y, value, fontName="Helvetica", fontSize=size, textAnchor=anchor, fillColor=HexColor("#222222")))

    def axes(drawing: Drawing, ylabel: str) -> None:
        drawing.add(Line(x0, y0, x0, y0 + chart_height, strokeColor=HexColor("#333333"), strokeWidth=.7))
        drawing.add(Line(x0, y0, x0 + chart_width, y0, strokeColor=HexColor("#333333"), strokeWidth=.7))
        for gamma in gammas:
            x = x_positions[gamma]
            drawing.add(Line(x, y0, x, y0 - 4, strokeColor=HexColor("#333333"), strokeWidth=.5))
            label(drawing, x, y0 - 16, str(gamma))
        label(drawing, x0 + chart_width / 2, 12, "Gamma")
        label(drawing, x0, y0 + chart_height + 10, ylabel, 8, "start")

    def log_map(value: float, minimum: float, maximum: float) -> float:
        return y0 + (math.log10(value) - math.log10(minimum)) / (math.log10(maximum) - math.log10(minimum)) * chart_height

    def add_log_grid(drawing: Drawing, minimum: float, maximum: float, ticks: list[float]) -> None:
        for tick in ticks:
            y = log_map(tick, minimum, maximum)
            drawing.add(Line(x0, y, x0 + chart_width, y, strokeColor=HexColor("#DDDDDD"), strokeWidth=.45))
            label(drawing, x0 - 7, y - 3, f"{tick:g}", 7, "end")

    def add_line(drawing: Drawing, values: list[float], minimum: float, maximum: float, color: str, marker: str, dash: list[int] | None = None) -> None:
        points = [(x_positions[gamma], log_map(value, minimum, maximum)) for gamma, value in zip(gammas, values)]
        for left, right in zip(points, points[1:]):
            drawing.add(Line(*left, *right, strokeColor=HexColor(color), strokeWidth=1.35, strokeDashArray=dash))
        for x, y in points:
            if marker == "circle":
                drawing.add(Circle(x, y, 3.1, fillColor=HexColor(color), strokeColor=HexColor(color)))
            elif marker == "square":
                drawing.add(Rect(x - 3, y - 3, 6, 6, fillColor=HexColor(color), strokeColor=HexColor(color)))
            else:
                drawing.add(Line(x - 3, y - 3, x + 3, y + 3, strokeColor=HexColor(color), strokeWidth=1))
                drawing.add(Line(x - 3, y + 3, x + 3, y - 3, strokeColor=HexColor(color), strokeWidth=1))

    def legend(drawing: Drawing, entries: list[tuple[str, str]], y: float = 282) -> None:
        x = 65
        for text_value, color in entries:
            drawing.add(Line(x, y, x + 16, y, strokeColor=HexColor(color), strokeWidth=1.4))
            label(drawing, x + 20, y - 3, text_value, 7, "start")
            x += 125

    def save(drawing: Drawing, stem: str) -> None:
        pdf_path = output / f"{stem}.pdf"
        pdf_canvas = canvas.Canvas(str(pdf_path), pagesize=(width, height), pageCompression=1, invariant=1)
        pdf_canvas.setCreator(GENERATION_SCHEMA)
        pdf_canvas.setSubject(ARCHIVE_SHA256)
        pdf_canvas.setKeywords(f"{RUN_COMMIT} {CONFIG_SHA256} {PROTOCOL_SHA256}")
        renderPDF.draw(drawing, pdf_canvas, 0, 0)
        pdf_canvas.showPage()
        pdf_canvas.save()
        raw_prefix = output / f".{stem}.raw"
        raw_png = output / f".{stem}.raw.png"
        tool = shutil.which("pdftoppm")
        _check(tool is not None, "pdftoppm is required to render deterministic PNG figures")
        if Path(tool).suffix.lower() in {".cmd", ".bat"}:
            native_tool = (
                Path(tool).parents[2] / "native" / "poppler" / "Library" / "bin" / "pdftoppm.exe"
            )
            _check(native_tool.is_file(), "native pdftoppm executable missing")
            tool = str(native_tool)
        command = [tool, "-singlefile", "-png", "-r", "220", str(pdf_path), str(raw_prefix)]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        _check(completed.returncode == 0 and raw_png.is_file(), f"pdftoppm failed: {completed.stderr}")
        image = Image.open(raw_png)
        metadata = PngImagePlugin.PngInfo()
        for key, value in {
            "Software": GENERATION_SCHEMA,
            "SourceZIPSHA256": ARCHIVE_SHA256,
            "RunGitCommit": RUN_COMMIT,
            "ConfigSHA256": CONFIG_SHA256,
            "ProtocolSHA256": PROTOCOL_SHA256,
        }.items():
            metadata.add_text(key, value)
        image.save(output / f"{stem}.png", pnginfo=metadata, optimize=False)
        raw_png.unlink()

    runtime = Drawing(width, height)
    axes(runtime, "Runtime / PAR-2 (s, log scale)")
    add_log_grid(runtime, .1, 10000, [.1, 1, 10, 100, 1000, 10000])
    hybrid = [row["hybrid_runtime_mean"] for row in pair_summary]
    direct = [row["direct_runtime_mean"] for row in pair_summary]
    direct_par2 = [row["direct_par2_mean"] for row in pair_summary]
    add_line(runtime, hybrid, .1, 10000, "#3B6F8A", "circle")
    add_line(runtime, direct, .1, 10000, "#777777", "square", [4, 3])
    add_line(runtime, direct_par2, .1, 10000, "#222222", "cross", [1, 3])
    legend(runtime, [("Hybrid raw", "#3B6F8A"), ("Direct raw", "#777777"), ("Direct PAR-2", "#222222")])
    timeout_y = log_map(direct_par2[-1], .1, 10000)
    runtime.add(Circle(x_positions[4], timeout_y, 6, fillColor=None, strokeColor=HexColor("#222222"), strokeWidth=.9))
    label(runtime, x_positions[4] - 8, timeout_y - 18, "5/5 timeout; no incumbent", 7, "end")
    save(runtime, "figure_high_gamma_runtime")

    growth = Drawing(width, height)
    axes(growth, "Count (log scale)")
    add_log_grid(growth, 100, 10000000, [100, 1000, 10000, 100000, 1000000, 10000000])
    add_line(growth, [row["scenario_count"] for row in model_rows], 100, 10000000, "#3B6F8A", "circle")
    add_line(growth, [row["direct_columns_mean"] for row in model_rows], 100, 10000000, "#777777", "square")
    add_line(growth, [row["direct_nonzeros_mean"] for row in model_rows], 100, 10000000, "#222222", "cross")
    legend(growth, [("Scenarios", "#3B6F8A"), ("Columns", "#777777"), ("Nonzeros", "#222222")])
    save(growth, "figure_external_model_growth")

    stability = Drawing(width, height)
    axes(stability, "Mean count across five seeds")
    maximum = max(
        row[metric]
        for row in pool_rows
        for metric in (
            "hybrid_proposal_evictions_mean",
            "hybrid_rediscovered_evicted_scenarios_mean",
            "hybrid_duplicate_proposals_mean",
        )
    )
    ceiling = max(1.0, maximum * 1.25)
    for fraction in (0, .25, .5, .75, 1):
        y = y0 + fraction * chart_height
        stability.add(Line(x0, y, x0 + chart_width, y, strokeColor=HexColor("#DDDDDD"), strokeWidth=.45))
        label(stability, x0 - 7, y - 3, f"{fraction * ceiling:.0f}", 7, "end")
    bar_width = 19
    colors = ["#9AB6C5", "#777777", "#333333"]
    metrics = [
        "hybrid_proposal_evictions_mean",
        "hybrid_rediscovered_evicted_scenarios_mean",
        "hybrid_duplicate_proposals_mean",
    ]
    for gamma, row in zip(gammas, pool_rows):
        center = x_positions[gamma]
        for index, (metric, color) in enumerate(zip(metrics, colors)):
            value = row[metric]
            bar_height = value / ceiling * chart_height
            stability.add(Rect(center + (index - 1) * bar_width - bar_width / 2, y0, bar_width - 2, bar_height, fillColor=HexColor(color), strokeColor=None))
        local_maximum = max(row[metric] for metric in metrics)
        label(stability, center, min(y0 + chart_height - 9, y0 + local_maximum / ceiling * chart_height + 8), "5/5 certified", 7)
    legend(stability, [("Evictions", colors[0]), ("Rediscoveries", colors[1]), ("Duplicates", colors[2])])
    save(stability, "figure_pool_stability")


def _write_prose(output: Path, audit: dict[str, Any], pair_summary: list[dict[str, Any]], pool_rows: list[dict[str, Any]]) -> None:
    by_gamma = {row["gamma"]: row for row in pair_summary}
    maximum_gap = audit["anchor_maximum_relative_gap"]
    zh = f"""# 高 Gamma 压力测试与外部通用求解器基准

## 实验定位

本实验是在 4 个仓库、4 个产品和 5 个区域的小规模实例上进行的压力测试，不进入 Final Holdout 主样本，也不替代 Gamma=0/1/2 的跨规模敏感性分析。需求不确定维度为20，完整预算场景数由 Gamma=2 的211个增长至 Gamma=3 的1,351个和 Gamma=4 的6,196个。比较对象统一称为“基于直接确定性等价模型的通用求解器基准”（general-purpose solver benchmark based on the direct deterministic equivalent formulation）。

## 认证与计算结果

Hybrid 在三个 Gamma 水平的15个单元上均获得 `certified_robust_optimal`，且全部完成最终精确分离和完整 post-evaluation。Direct 在 Gamma=2和3各认证5/5；Gamma=4的5个单元均达到1800秒时限、没有 incumbent，因而只能按 time-limit/PAR-2 计入，不能报告 Direct 的目标值或最优性 gap。Gamma=2和3中双方均认证单元的最大绝对 T 差分别为 {by_gamma[2]['objective_t_difference_max_abs']:.3e} 和 {by_gamma[3]['objective_t_difference_max_abs']:.3e}。

## Rolling proposal pool 与 master

rolling pool 只限制候选场景 proposal 的记忆，不删除已经提交到 master 的场景块。其不变量为

$$
S_{{k+1}} = S_k \\cup \\{{\\hat s_k\\}}.
$$

Gamma=4 的平均 eviction、rediscovery 和 duplicate proposal 数分别为 {pool_rows[2]['hybrid_proposal_evictions_mean']:.1f}、{pool_rows[2]['hybrid_rediscovered_evicted_scenarios_mean']:.1f} 和 {pool_rows[2]['hybrid_duplicate_proposals_mean']:.1f}；15个 Hybrid 任务的 committed blocks/cuts 账本均为 append-only，并以 final exact separation 完成认证。在这一小规模压力测试中，没有观察到由候选池淘汰导致的循环或认证失败。该经验结果不能外推为任意 Gamma 的理论或计算可扩展性证明，也不代表 Hybrid 优于所有先进算法。

## 成本锚点解释

每个单元使用对应 Gamma 的认证 baseline 上界作为 `C_anchor`。严格表述是 $C^* \\in [LB,C_{{anchor}}]$，而不是 $C_{{anchor}}=C^*$。15个锚点的最大相对认证 gap 为 {maximum_gap:.8f}，小于冻结容差 $10^{{-4}}$，相对于 $\\rho=0.025$ 很小，因而不足以解释主要公平变化。

## 结论边界

结论仅限于本次 five-seed、小规模、Gamma=2/3/4 的预注册实验。Direct Gamma=4 的结果是“无 incumbent 的 time limit”，不是认证解；Gurobi 也不代表全部行业算法。
"""
    en = f"""# High-Gamma stress test and external solver benchmark

## Scope and motivation

We conducted a controlled stress test on instances with 4 warehouses, 4 products, and 5 regions. This experiment is separate from the Final Holdout and from the cross-scale Gamma=0/1/2 sensitivity analysis. With 20 uncertain demand components, the complete scenario set grows from 211 at Gamma=2 to 1,351 at Gamma=3 and 6,196 at Gamma=4. The external comparator is described as a general-purpose solver benchmark based on the direct deterministic equivalent formulation.

## Certification and computational evidence

Hybrid certified all 15 cells and completed final exact separation and exhaustive post-evaluation in every cell. The direct formulation certified 5/5 cells at Gamma=2 and 5/5 at Gamma=3. At Gamma=4, all five direct runs reached the 1,800 s time limit without an incumbent. These cells therefore enter the analysis through their time-limit classification and 3,600 s PAR-2 penalty; no direct objective value or optimality gap is reported. Among jointly certified cells, the maximum absolute differences in T were {by_gamma[2]['objective_t_difference_max_abs']:.3e} at Gamma=2 and {by_gamma[3]['objective_t_difference_max_abs']:.3e} at Gamma=3.

## Rolling proposals versus committed master blocks

The rolling pool limits only proposal memory. It does not delete scenario blocks already committed to the master:

$$
S_{{k+1}} = S_k \\cup \\{{\\hat s_k\\}}.
$$

At Gamma=4, the mean counts of proposal evictions, rediscoveries, and duplicate proposals were {pool_rows[2]['hybrid_proposal_evictions_mean']:.1f}, {pool_rows[2]['hybrid_rediscovered_evicted_scenarios_mean']:.1f}, and {pool_rows[2]['hybrid_duplicate_proposals_mean']:.1f}, respectively. All committed scenario and certified-cut ledgers were append-only, and every Hybrid cell ended with exact certification. Thus, in this small-scale stress test, we observed no cycling or certification failure caused by proposal-pool eviction. This empirical result is not a proof of computational scalability for arbitrary Gamma, nor a claim that Hybrid dominates all state-of-the-art methods.

## Anchor quality and interpretation

For each seed-Gamma cell, `C_anchor` is the certified upper bound from the matching robust-cost baseline. The valid statement is $C^* \\in [LB,C_{{anchor}}]$, not $C_{{anchor}}=C^*$. The maximum relative certified anchor gap was {maximum_gap:.8f}, below the frozen $10^{{-4}}$ tolerance and small relative to $\\rho=0.025$.

## Boundary of the conclusion

The evidence is limited to five pre-registered seeds and the stated small-scale Gamma=2/3/4 design. A Gamma=4 direct time limit without an incumbent is not a certified solution, and Gurobi is not used as a proxy for every industrial algorithm.
"""
    reviewer = f"""# Response to reviewer comments: high Gamma and external benchmark

## A. Dependence on very small Gamma and possible cycling of the rolling window

We added a pre-registered small-scale stress test with Gamma=2, 3, and 4, corresponding to 211, 1,351, and 6,196 complete scenarios. Hybrid certified all 15 cells. We also clarified the mechanism: the rolling pool governs only candidate-proposal memory, whereas committed master blocks satisfy $S_{{k+1}}=S_k \\cup \\{{\\hat s_k\\}}$. The audit verifies append-only scenario and cut SHA ledgers, reports evictions, rediscoveries, and duplicates, and confirms final exact separation in every cell. Our revised claim is deliberately limited: in the 4-warehouse by 4-product by 5-region stress test, we observed no cycling or certification failure caused by pool eviction. We do not claim validity for arbitrary Gamma or equate finite theoretical convergence with practical scalability.

## B. Missing external general-purpose solver benchmark

We added a general-purpose solver benchmark based on the direct deterministic equivalent formulation, with shared first-stage variables and explicit recourse blocks for every scenario. Gurobi's built-in Benders strategy was disabled. The direct formulation certified all Gamma=2 and Gamma=3 cells. At Gamma=4, all five runs reached the 1,800 s limit without an incumbent as model size grew sharply; these cells are retained in the pre-registered comparison with a 3,600 s PAR-2 penalty. We do not report fabricated objectives or gaps for those runs and do not claim that Gurobi represents all industrial algorithms.

## C. Conservatism of C_anchor and the economic meaning of rho

We now state the anchor relationship as $C^* \\in [LB,C_{{anchor}}]$. The anchor is a certified upper bound, not an exact optimum. Across all 15 baseline cells, the maximum certified relative gap was {maximum_gap:.8f}, below $10^{{-4}}$. This residual conservatism is small compared with $\\rho=0.025$; it is reported explicitly in the anchor-quality table and is not used to overstate economic precision.
"""
    (output / "high_gamma_external_benchmark_results_zh.md").write_text(
        _front_matter() + zh, encoding="utf-8", newline="\n",
    )
    (output / "high_gamma_external_benchmark_results_en.md").write_text(
        _front_matter() + en, encoding="utf-8", newline="\n",
    )
    (output / "reviewer_response_high_gamma_and_external_solver_en.md").write_text(
        _front_matter() + reviewer, encoding="utf-8", newline="\n",
    )
    (output / "experimental_results_high_gamma_addendum.md").write_text(
        _front_matter()
        + "# High-Gamma stress test and external solver benchmark\n\n"
        + "This independent subsection is intended for insertion after the frozen Final Holdout and Gamma-sensitivity sections. "
        + "It uses only the High-Gamma Attempt 2 sample and does not alter or pool the Final Holdout observations.\n\n"
        + en.split("## Scope and motivation", 1)[1],
        encoding="utf-8",
        newline="\n",
    )


def _artifact_index(output: Path) -> None:
    rows = []
    for path in sorted(item for item in output.iterdir() if item.name != "artifact_sha256.csv"):
        rows.append({"relative_path": path.name, "sha256": file_sha256(path)})
    _write_csv(output / "artifact_sha256.csv", rows)


def generate_reports(archive_path: Path, output: Path) -> dict[str, Any]:
    audit, rows, paired = audit_archive(archive_path)
    output.mkdir(parents=True, exist_ok=False)
    task_summary = _task_summary(rows)
    pair_summary = _pair_summary(paired)
    anchor_rows = _anchor_rows(rows, paired)
    model_rows = _model_rows(paired)
    pool_rows = _pool_rows(paired)
    audit["task_summary"] = task_summary
    audit["paired_summary"] = pair_summary
    audit["archive"]["sha256_after_generation"] = file_sha256(archive_path)
    _check(audit["archive"]["sha256_after_generation"] == ARCHIVE_SHA256, "ZIP changed during report generation")
    decision = {
        **PROVENANCE_FIELDS,
        "decision": audit["decision"],
        "scientific_solution_valid": True,
        "optimization_rerun_required": False,
        "hybrid_high_gamma_stability_supported_on_small_scale": True,
        "external_direct_solver_benchmark_valid": True,
    }
    provenance = {
        **PROVENANCE_FIELDS,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "hybrid_candidate_sha256": HYBRID_SHA256,
        "direct_candidate_sha256": DIRECT_SHA256,
        "execution_attempt": ATTEMPT,
        "previous_attempt_results_reused": False,
        "archive_read_only": True,
    }
    freeze = {
        **provenance,
        "decision": audit["decision"],
        "frozen_matrix": {
            "scale": ["small"],
            "seeds": SEEDS,
            "gamma": GAMMAS,
            "rho": RHO,
            "baseline_tasks": 15,
            "hybrid_frontier_tasks": 15,
            "direct_frontier_tasks": 15,
            "total_tasks": 45,
        },
        "scope": {
            "final_holdout_remains_primary": True,
            "gamma_attempt3_remains_cross_scale_sensitivity": True,
            "attempt2_role": [
                "high-Gamma small-scale stress test",
                "external general-purpose solver benchmark",
                "rolling-pool mechanism evidence",
            ],
            "mixed_into_final_holdout_statistics": False,
        },
    }
    _write_json(output / "decision.json", decision)
    _write_json(output / "source_archive_provenance.json", provenance)
    _write_json(output / "freeze_manifest.json", freeze)
    _write_json(output / "final_audit.json", {**PROVENANCE_FIELDS, **audit})
    (output / "decision.md").write_text(
        _front_matter()
        + "# Final decision\n\n"
        + "`approve_high_gamma_external_benchmark_attempt2`\n\n"
        + "- Scientific solution valid: true\n"
        + "- Optimization rerun required: false\n"
        + "- Hybrid high-Gamma stability supported on small scale: true\n"
        + "- External direct solver benchmark valid: true\n"
        + "- Scope: five seeds, small scale, Gamma=2/3/4 only.\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_csv(output / "results.complete.csv", rows)
    _write_csv(output / "summary_by_gamma.csv", task_summary)
    _write_csv(output / "paired_results.complete.csv", paired)
    _write_csv(output / "table_high_gamma_main.csv", pair_summary)
    _write_csv(output / "table_external_solver_model_size.csv", model_rows)
    _write_csv(output / "table_high_gamma_pool_stability.csv", pool_rows)
    _write_csv(output / "table_high_gamma_complete_seed_results.csv", paired)
    _write_csv(output / "table_anchor_quality.csv", anchor_rows)
    _write_markdown_tables(output, pair_summary, model_rows, pool_rows, anchor_rows)
    _plot(output, pair_summary, model_rows, pool_rows)
    _write_prose(output, audit, pair_summary, pool_rows)
    _artifact_index(output)
    return {
        "decision": audit["decision"],
        "tasks": len(rows),
        "pairs": len(paired),
        "archive_sha256": ARCHIVE_SHA256,
        "gurobipy_imported": False,
        "solver_called": False,
    }


def compare_directories(left: Path, right: Path) -> None:
    left_files = sorted(path.relative_to(left) for path in left.rglob("*") if path.is_file())
    right_files = sorted(path.relative_to(right) for path in right.rglob("*") if path.is_file())
    _check(left_files == right_files, "deterministic rebuild file sets differ")
    for relative in left_files:
        _check(
            (left / relative).read_bytes() == (right / relative).read_bytes(),
            f"deterministic rebuild differs: {relative}",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate_reports(args.archive, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
