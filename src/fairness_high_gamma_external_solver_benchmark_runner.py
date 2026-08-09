"""Strict runner for the preregistered high-Gamma external benchmark."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import yaml

from .experiment_protocol import atomic_write_csv, atomic_write_json, atomic_write_yaml, file_sha256


STAGE = "HIGH_GAMMA_EXTERNAL_BENCHMARK"
SCHEMA = "fairness_high_gamma_external_solver_benchmark_v1"
AUTH_SCHEMA = "fairness_high_gamma_external_solver_benchmark_authorization_v1"
ATTEMPT = 1
SEEDS = [185, 186, 187, 188, 189]
GAMMAS = [2, 3, 4]
RHO = 0.025
SCENARIOS = {2: 211, 3: 1351, 4: 6196}
SOLVER_PARAMETERS = {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}
HYBRID = "certified_hybrid_scenario_benders_fairness"
HYBRID_SHA256 = "8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0"
DIRECT = "gurobi_direct_extensive_form"
EXPECTED_PROTOCOL_SHA256 = "6CC3132C03D0991AA8757FD531A4FB700F50260BEC40E854EA235A47DB94C470"
EXPECTED_CONFIG_SHA256 = "CB43D37667DB704E2D80B6C7DCA64603D462E6370D998342979F62D7C5265E95"
EXPECTED_DIRECT_SHA256 = "4A0CA29C6367858A96D62F9B30DC45BBA969CBF0679E758EBB7229AFD999DFAF"
AUTH_RELATIVE_PATH = "experiments/configs/fairness_high_gamma_external_solver_benchmark_authorization.json"
OUTPUT_RELATIVE_PATH = "experiments/results_fh_ext/hg1"
SEED_AUDIT_RELATIVE_PATH = "analysis/fairness_high_gamma_external_solver_benchmark_protocol/seed_access_audit.json"


class HighGammaGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class Dependencies:
    generate_instance: Callable[[dict[str, Any], int], Any]
    serialize_instance: Callable[[Any], dict[str, Any]]
    deserialize_instance: Callable[[dict[str, Any]], Any]
    solve_baseline: Callable[[dict[str, Any], Any, int, dict[str, Any]], dict[str, Any]]
    make_anchor: Callable[[dict[str, Any], dict[str, Any], float], dict[str, Any]]
    solve_hybrid: Callable[[dict[str, Any], Any, dict[str, Any], dict[str, Any], dict[str, Any], Path, dict[str, Any], dict[str, Any]], dict[str, Any]]
    solve_direct: Callable[[dict[str, Any], Any, dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]
    post_evaluate: Callable[[dict[str, Any], Any, dict[str, Any], dict[str, Any], dict[str, Any], Path, dict[str, Any]], tuple[dict[str, Any], dict[str, float]]]
    configure_solver: Callable[[dict[str, Any]], None]


def _check(value: bool, message: str) -> None:
    if not value:
        raise HighGammaGateError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _check(isinstance(value, dict), f"YAML root must be mapping: {path}")
    return value


def read_json(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HighGammaGateError(f"corrupt JSON: {source}") from exc
    _check(isinstance(value, dict), f"JSON root must be object: {source}")
    return value


def run_directory_id(run_key: str) -> str:
    return "r_" + hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:24]


def scenario_count(gamma: int) -> int:
    _check(type(gamma) is int and gamma in GAMMAS, "Gamma must be exactly 2, 3, or 4")
    expected = sum(math.comb(20, k) for k in range(gamma + 1))
    _check(expected == SCENARIOS[gamma], "frozen scenario-count formula mismatch")
    return expected


def _rho_text() -> str:
    return "0.025"


def expand_plan() -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        for gamma in GAMMAS:
            for task_type, candidate, rho in (
                ("baseline", "baseline", "NOT_APPLICABLE"),
                ("hybrid_frontier", HYBRID, _rho_text()),
                ("direct_extensive_frontier", DIRECT, _rho_text()),
            ):
                identity = {"candidate": candidate, "execution_attempt": ATTEMPT, "gamma": gamma,
                            "rho": rho, "scale": "small", "seed": seed, "stage": STAGE,
                            "task_type": task_type}
                run_key = canonical_json(identity)
                rows.append({**identity, "run_key": run_key, "run_directory_id": run_directory_id(run_key)})
    keys = [row["run_key"] for row in rows]
    directories = [row["run_directory_id"] for row in rows]
    _check(len(keys) == len(set(keys)) == 45 and len(directories) == len(set(directories)) == 45,
           "run plan duplicate or short-directory collision")
    return rows


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_config(path: str | Path, config: dict[str, Any]) -> None:
    source = Path(path)
    actual_sha = file_sha256(source).upper()
    if EXPECTED_CONFIG_SHA256 != "TO_BE_FROZEN":
        _check(actual_sha == EXPECTED_CONFIG_SHA256, "config SHA mismatch")
    expected = {
        "experiment_name": "fairness_high_gamma_external_solver_benchmark",
        "stage": STAGE, "schema_version": 1, "execution_attempt": ATTEMPT,
        "previous_attempt_results_reused": False, "formal_run_authorized": False,
        "formal_worktree_root": r"E:\rfext1", "output_dir": OUTPUT_RELATIVE_PATH,
        "hybrid_candidate": HYBRID, "direct_candidate": DIRECT, "scale": "small",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "hybrid_candidate_sha256": HYBRID_SHA256,
        "direct_candidate_sha256": EXPECTED_DIRECT_SHA256,
        "seeds": SEEDS, "gamma": GAMMAS, "rho": RHO,
        "scenario_counts": SCENARIOS, "baseline_count": 15,
        "hybrid_frontier_count": 15, "direct_extensive_frontier_count": 15,
        "total_tasks": 45, "tol": 1e-4, "baseline_time_limit_seconds": 1800,
        "algorithm_time_limit_seconds": 1800, "max_iterations": 10000,
        "resume": True, "overwrite_supported": False,
    }
    for key, value in expected.items():
        _check(config.get(key) == value, f"config identity drift: {key}")
    _check(config.get("instance") == {"num_warehouses": 4, "num_products": 4, "num_regions": 5,
                                      "demand_components": 20,
                                      "generator_template": "experiments/configs/fairness_scalability_development_medium_large.yaml",
                                      "preserve_generator_distributions": True}, "small instance identity drift")
    _check(config.get("solver_identity") == {**SOLVER_PARAMETERS, "BendersStrategy_direct": 0},
           "solver identity drift")
    _check(config.get("post_evaluation") == {"time_limit_per_scenario_seconds": 30,
                                              "checkpoint_chunk_size": 25,
                                              "pipeline_generation": 1}, "post-evaluation identity drift")
    root = _root()
    protocol_sha = file_sha256(root / config["protocol_document"]).upper()
    direct_sha = file_sha256(root / config["direct_candidate_definition"]).upper()
    _check(file_sha256(root / config["hybrid_candidate_definition"]).upper() == HYBRID_SHA256,
           "Hybrid candidate SHA drift")
    if EXPECTED_PROTOCOL_SHA256 != "TO_BE_FROZEN":
        _check(protocol_sha == EXPECTED_PROTOCOL_SHA256, "protocol SHA mismatch")
    if EXPECTED_DIRECT_SHA256 != "TO_BE_FROZEN":
        _check(direct_sha == EXPECTED_DIRECT_SHA256, "direct candidate SHA mismatch")
    for gamma in GAMMAS:
        scenario_count(gamma)


def _planned_paths(root: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> list[tuple[str, Path]]:
    output = root / config["output_dir"]
    paths = [("manifest", output / "manifest.json"), ("manifest_tmp", output / ".manifest.json.tmp"),
             ("run_manifest", output / "run_manifest.json"),
             ("run_manifest_tmp", output / ".run_manifest.json.tmp"),
             ("resolved", output / "resolved_config.yaml"),
             ("resolved_tmp", output / ".resolved_config.yaml.tmp"),
             ("results", output / "results.csv"), ("results_tmp", output / ".results.csv.tmp"),
             ("summary", output / "summary.csv"), ("summary_tmp", output / ".summary.csv.tmp"),
             ("paired", output / "paired_comparison.csv"),
             ("paired_tmp", output / ".paired_comparison.csv.tmp"),
             ("model_size", output / "model_size_summary.csv"),
             ("model_size_tmp", output / ".model_size_summary.csv.tmp"),
             ("stability", output / "high_gamma_stability.csv"),
             ("stability_tmp", output / ".high_gamma_stability.csv.tmp"),
             ("audit", output / "audit.log"), ("audit_tmp", output / ".audit.log.tmp")]
    for row in rows:
        run = output / "runs" / row["run_directory_id"]
        paths.extend([("status", run / "status.json"), ("status_tmp", run / ".status.json.tmp"),
                      ("run", run / "run.json"), ("run_tmp", run / ".run.json.tmp"),
                      ("algorithm_checkpoint", run / "algorithm_checkpoint.json"),
                      ("algorithm_checkpoint_tmp", run / ".algorithm_checkpoint.json.tmp")])
        if row["task_type"] == "hybrid_frontier":
            paths.extend([("hybrid_internal_checkpoint", run / "hybrid_internal_checkpoint.json"),
                          ("hybrid_internal_checkpoint_tmp", run / ".hybrid_internal_checkpoint.json.tmp")])
        if row["task_type"] != "baseline":
            post = run / "post_evaluation"
            last = math.ceil(scenario_count(row["gamma"]) / 25) - 1
            paths.extend([("post_final", post / "post_evaluation.json"),
                          ("post_final_tmp", post / ".post_evaluation.json.tmp"),
                          ("post_index", post / "checkpoint" / "index.json"),
                          ("post_index_tmp", post / "checkpoint" / ".index.json.tmp")])
            for chunk in range(last + 1):
                paths.extend([("post_chunk", post / "checkpoint" / f"chunk_{chunk:05d}.json"),
                              ("post_chunk_tmp", post / "checkpoint" / f".chunk_{chunk:05d}.json.tmp")])
    for seed in SEEDS:
        for gamma in GAMMAS:
            instance = output / "instances" / f"s{seed}_g{gamma}.json"
            paths.extend([("instance", instance), ("instance_tmp", instance.parent / f".{instance.name}.tmp")])
    return paths


def dry_run(config_path: str | Path, *, root_override: Path | None = None) -> dict[str, Any]:
    before = "gurobipy" in sys.modules
    config = load_yaml(config_path)
    validate_config(config_path, config)
    rows = expand_plan()
    root = (root_override or _root()).resolve()
    paths = _planned_paths(root, rows, config)
    longest_type, longest = max(paths, key=lambda item: len(str(item[1])))
    return {
        "stage": STAGE, "scale": ["small"], "seeds": SEEDS, "gamma": GAMMAS, "rho": RHO,
        "baselines": 15, "hybrid_frontiers": 15, "direct_extensive_frontiers": 15,
        "total": 45, "scenario_counts": SCENARIOS,
        "unique_run_keys": len({row["run_key"] for row in rows}),
        "unique_directory_ids": len({row["run_directory_id"] for row in rows}),
        "directory_collisions": 0, "instances_generated": False, "solver_called": False,
        "gurobipy_imported": ("gurobipy" in sys.modules) and not before,
        "output_dir_exists": (root / config["output_dir"]).exists(),
        "longest_windows_path": str(longest), "longest_windows_path_length": len(str(longest)),
        "longest_windows_path_type": longest_type,
        "algorithm_solver_limit_envelope_seconds": 45 * 1800,
        "post_evaluation_solver_limit_envelope_seconds": 2 * 5 * sum(SCENARIOS.values()) * 30,
        "solver_limit_envelopes_are_not_wall_time_predictions": True,
    }


def _source_tree_identity(root: Path) -> str:
    process = subprocess.run(["git", "ls-files", "-s"], cwd=root, text=True, capture_output=True, check=True)
    lines = []
    for line in process.stdout.splitlines():
        metadata, path = line.split("\t", 1)
        mode, digest, _stage = metadata.split()
        if path != AUTH_RELATIVE_PATH:
            lines.append(f"{mode} {digest} {path}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest().upper()


def _git_gate(root: Path, config: dict[str, Any]) -> str:
    _check(root.resolve() == Path(config["formal_worktree_root"]).resolve(), "formal_worktree_root mismatch")
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
                            text=True, capture_output=True, check=True).stdout.strip()
    _check(not status, f"formal Git worktree is dirty: {status}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                          capture_output=True, check=True).stdout.strip()


def validate_authorization(path: str | Path, config_path: str | Path, root: Path,
                           config: dict[str, Any]) -> dict[str, Any]:
    auth = read_json(path)
    _check(auth is not None, "authorization file missing")
    expected = {
        "schema": AUTH_SCHEMA, "schema_version": 1, "formal_run_authorized": True,
        "stage": STAGE, "execution_attempt": ATTEMPT, "scales": ["small"],
        "seeds": SEEDS, "gamma": GAMMAS, "rho": RHO,
        "baseline_count": 15, "hybrid_frontier_count": 15,
        "direct_extensive_frontier_count": 15, "total_tasks": 45,
        "output_directory": OUTPUT_RELATIVE_PATH, "formal_worktree_root": r"E:\rfext1",
        "previous_benchmark_results_reused": False,
    }
    for key, value in expected.items():
        _check(auth.get(key) == value, f"authorization drift: {key}")
    _check(auth.get("config_sha256") == file_sha256(config_path).upper() and
           auth.get("protocol_sha256") == file_sha256(root / config["protocol_document"]).upper() and
           auth.get("hybrid_candidate_sha256") == HYBRID_SHA256 and
           auth.get("direct_candidate_sha256") == file_sha256(root / config["direct_candidate_definition"]).upper(),
           "authorization SHA identity mismatch")
    _check(auth.get("authorized_source_tree_sha256") == _source_tree_identity(root),
           "authorization source tree mismatch")
    basis = auth.get("authorization_basis_commit")
    _check(isinstance(basis, str) and len(basis) == 40 and
           all(character in "0123456789abcdef" for character in basis),
           "authorization basis commit identity invalid")
    prohibited = auth.get("prohibited_scope", {})
    _check(all(prohibited.get(key) is True for key in (
        "other_stages", "other_scales", "other_seeds", "gamma_0_or_1", "other_gamma",
        "other_rho", "new_hybrid_candidate", "overwrite", "selective_rerun", "prior_result_reuse")),
        "authorization prohibited scope incomplete")
    return auth


def seed_access_gate(root: Path) -> dict[str, Any]:
    audit = read_json(root / SEED_AUDIT_RELATIVE_PATH)
    _check(audit is not None and audit.get("candidate_seeds") == SEEDS and
           audit.get("formal_instance_or_solve_access_evidence_count") == 0 and
           audit.get("decision") == "seed_access_clear", "seed-access evidence is not clear")
    return audit


def _identity(row: dict[str, Any], config_path: Path, commit: str) -> dict[str, Any]:
    candidate_sha = EXPECTED_DIRECT_SHA256 if row["task_type"] == "direct_extensive_frontier" else HYBRID_SHA256
    return {**row, "schema": SCHEMA, "git_commit": commit,
            "config_file_sha256": file_sha256(config_path).upper(),
            "resolved_config_file_sha256": file_sha256(config_path).upper(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256, "candidate_sha256": candidate_sha,
            "solver_parameters": deepcopy(SOLVER_PARAMETERS),
            "previous_attempt_results_reused": False, "scenario_count": scenario_count(row["gamma"]),
            "baseline_run_key": row["run_key"] if row["task_type"] == "baseline" else None}


def _instance_payload(identity: dict[str, Any], serialized: dict[str, Any]) -> dict[str, Any]:
    canonical_sha = sha256_value(serialized)
    frozen = {key: identity[key] for key in ("stage", "scale", "seed", "gamma", "execution_attempt",
                                               "git_commit", "config_file_sha256", "protocol_sha256")}
    return {"identity": {**frozen, "instance_canonical_sha256": canonical_sha}, "instance": serialized}


def _validate_instance(payload: dict[str, Any], identity: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    _check(isinstance(payload, dict) and isinstance(payload.get("instance"), dict) and
           isinstance(payload.get("identity"), dict), "instance archive corrupt")
    serialized = payload["instance"]
    canonical_sha = sha256_value(serialized)
    frozen = {key: identity[key] for key in ("stage", "scale", "seed", "gamma", "execution_attempt",
                                               "git_commit", "config_file_sha256", "protocol_sha256")}
    _check(payload["identity"] == {**frozen, "instance_canonical_sha256": canonical_sha},
           "instance scientific identity mismatch")
    return serialized, canonical_sha, sha256_value(payload["identity"])


def _status(path: Path, identity: dict[str, Any], state: str, scientific: str) -> None:
    atomic_write_json(path, {"schema": SCHEMA, "identity": identity, "state": state,
                             "scientific_status": scientific})


def _validate_status(path: Path, identity: dict[str, Any], record: dict[str, Any] | None) -> None:
    status = read_json(path)
    if status is None:
        _check(record is None, "completed run lacks status")
        return
    _check(status.get("identity") == identity, "status identity mismatch")
    if record is not None:
        _check(status.get("state") == record.get("state") and
               status.get("scientific_status") == record.get("scientific_status"),
               "run/status state mismatch")


def _algorithm_checkpoint(path: Path, identity: dict[str, Any], method: str,
                          result: dict[str, Any]) -> None:
    payload = {"schema": SCHEMA, "identity": identity, "method": method, "result": result}
    payload["checkpoint_sha256"] = sha256_value(payload)
    atomic_write_json(path, payload)


def _load_checkpoint(path: Path, identity: dict[str, Any], method: str) -> dict[str, Any] | None:
    value = read_json(path)
    if value is None:
        return None
    checksum = value.get("checkpoint_sha256")
    unsigned = dict(value); unsigned.pop("checkpoint_sha256", None)
    _check(checksum == sha256_value(unsigned) and value.get("identity") == identity and
           value.get("method") == method and isinstance(value.get("result"), dict),
           "algorithm checkpoint corrupt or identity drifted")
    return value["result"]


def _validate_completed_checkpoint(record: dict[str, Any], checkpoint: dict[str, Any] | None) -> None:
    _check(checkpoint is not None and isinstance(record.get("result"), dict),
           "completed run lacks algorithm checkpoint")
    stored = record["result"]
    _check(all(key in stored and stored[key] == value for key, value in checkpoint.items()),
           "run and algorithm checkpoint disagree")


def _finite_number(value: Any, field: str) -> float:
    _check(type(value) in {int, float} and math.isfinite(float(value)), f"{field} must be finite numeric")
    return float(value)


def _solution_values(result: dict[str, Any], task_type: str) -> tuple[list[float], list[list[float]]]:
    y_name = "best_y_values" if task_type == "baseline" else "y_values"
    x_name = "best_x_values" if task_type == "baseline" else "x_values"
    y_values = result.get(y_name)
    x_values = result.get(x_name)
    _check(isinstance(y_values, list) and len(y_values) == 4, f"{y_name} must follow frozen warehouse order")
    _check(isinstance(x_values, list) and len(x_values) == 4, f"{x_name} must have four warehouse rows")
    y = [_finite_number(value, f"{y_name}[{i}]") for i, value in enumerate(y_values)]
    matrix = []
    for i, row in enumerate(x_values):
        _check(isinstance(row, list) and len(row) == 4, f"{x_name}[{i}] must follow frozen product order")
        matrix.append([_finite_number(value, f"{x_name}[{i}][{j}]") for j, value in enumerate(row)])
    return y, matrix


def _validate_solver_result(result: dict[str, Any], task_type: str, gamma: int) -> None:
    _check(isinstance(result, dict) and isinstance(result.get("status"), str), "solver result schema invalid")
    runtime_name = "runtime" if task_type == "baseline" else "algorithm_runtime"
    _finite_number(result.get(runtime_name, result.get("runtime")), runtime_name)
    if task_type == "baseline":
        _check(result.get("valid_UB") is True, "baseline valid_UB missing")
        for field in ("lower_bound", "upper_bound", "gap"):
            _finite_number(result.get(field), field)
        _solution_values(result, task_type)
        return
    if task_type == "direct_extensive_frontier":
        _check(result.get("scenario_count") == scenario_count(gamma), "direct scenario count mismatch")
        _check(result.get("benders_strategy") == 0, "direct BendersStrategy identity mismatch")
        for field in ("model_build_runtime", "optimize_runtime", "algorithm_runtime"):
            _finite_number(result.get(field), field)
    if result.get("status") == "optimal":
        for field in ("lower_bound", "upper_bound", "gap", "objective_t", "robust_minimum_fill_rate"):
            _finite_number(result.get(field), field)
        _solution_values(result, task_type)


def _par2(scientific: str, runtime: float) -> float:
    return float(runtime) if scientific == "certified_robust_optimal" else 3600.0


def _valid_post(evaluation: dict[str, Any] | None, count: int) -> bool:
    return bool(isinstance(evaluation, dict) and evaluation.get("valid") is True and
                evaluation.get("errors") == [] and evaluation.get("objective_t_consistent") is True and
                evaluation.get("scenario_count") == count)


def _hybrid_counters(result: dict[str, Any]) -> dict[str, int]:
    log = result.get("iteration_log") if isinstance(result.get("iteration_log"), list) else []
    maximum = max((int(item.get("pool_candidate_count", 0)) for item in log), default=0)
    evictions = sum(int(item.get("evicted_proposal_count", 0)) for item in log)
    duplicates = sum(int(item.get("duplicate_proposal_count", 0)) for item in log)
    rediscovered = sum(int(item.get("rediscovered_evicted_scenario_count", 0)) for item in log)
    consecutive = maximum_run = 0
    previous = None
    for item in log:
        bound = item.get("lower_bound")
        improving = type(bound) in {int, float} and (previous is None or float(bound) > previous + 1e-12)
        consecutive = 0 if improving else consecutive + 1
        maximum_run = max(maximum_run, consecutive)
        if type(bound) in {int, float}:
            previous = max(float(bound), previous if previous is not None else float(bound))
    metadata = result.get("metadata") or {}
    return {"candidate_pool_maximum_size": maximum, "eviction_count": evictions,
            "rediscovered_evicted_scenario_count": rediscovered,
            "duplicate_proposal_count": duplicates,
            "unique_committed_scenario_blocks": int(metadata.get("committed_scenario_count", 0)),
            "committed_farkas_cuts": int(result.get("cuts", 0)),
            "maximum_consecutive_non_improving_iterations": maximum_run}


def _frontier_scientific(task_type: str, result: dict[str, Any], evaluation: dict[str, Any] | None,
                         count: int, tol: float) -> str:
    if task_type == "hybrid_frontier":
        log = result.get("iteration_log") if isinstance(result.get("iteration_log"), list) else []
        final = log[-1] if log else {}
        certified = (result.get("status") == "optimal" and
                     (result.get("metadata") or {}).get("robust_feasibility_certified") is True and
                     final.get("final_exact_separation_performed") is True and
                     final.get("robust_feasibility_certified") is True and
                     type(final.get("separation_objective_bound")) in {int, float} and
                     float(final["separation_objective_bound"]) <= tol)
    else:
        certified = (result.get("status") == "optimal" and result.get("complete_model_built") is True and
                     result.get("resource_failure") is False and type(result.get("lower_bound")) in {int, float} and
                     type(result.get("upper_bound")) in {int, float} and type(result.get("gap")) in {int, float} and
                     float(result["gap"]) <= tol)
    if not certified:
        return "time_limit_uncertified" if "time_limit" in str(result.get("status")) else "robust_uncertified"
    return "certified_robust_optimal" if _valid_post(evaluation, count) else "invalid_post_evaluation"


def production_dependencies() -> Dependencies:
    # Deliberately lazy. Every formal authorization gate runs before this function.
    from .benders import solve_benders
    from .experiment_suite import _apply_selected_parameters, _apply_variant_config, _base_config
    from .fairness_high_gamma_external_solver_benchmark import solve_gurobi_direct_extensive_form
    from .fairness_hybrid_ccg_benders import initial_upper_bound_expected_identity, solve_certified_hybrid_scenario_benders_fairness
    from .fairness_hybrid_ccg_benders_runner import _hybrid_certified_anchor
    from .fairness_large_final_remediation_runner import _configure_solver_parameters
    from .fairness_post_evaluation import checkpointed_fairness_post_evaluation
    from .instance import InventoryInstance, generate_instance

    def template(config: dict[str, Any], gamma: int) -> dict[str, Any]:
        value = load_yaml(_root() / config["instance"]["generator_template"])
        value["instance_overrides"] = {"num_warehouses": 4, "num_products": 4, "num_regions": 5}
        value.update({"gamma_target": gamma, "gamma_schedule": [gamma],
                      "gamma_continuation_enabled": False, "exact_scenarios": True,
                      "max_scenarios": scenario_count(gamma), "time_limit": 1800,
                      "baseline_time_limit": 1800})
        return value

    def generate(config: dict[str, Any], seed: int) -> Any:
        return generate_instance(_base_config(template(config, int(config["gamma_value"])), "small", seed), seed=seed)

    def baseline(config: dict[str, Any], instance: Any, seed: int, solver: dict[str, Any]) -> dict[str, Any]:
        _configure_solver_parameters(solver)
        selected = _apply_selected_parameters(template(config, int(config["gamma_value"])))
        base = _base_config(selected, "small", seed)
        variant = dict(selected.get("variant_settings", {}).get("joint_v1_core_point_strengthened", {}))
        method, _flags, method_config = _apply_variant_config(base, "proposed_adaptive_benders", variant)
        result = solve_benders(method_config, instance, method)
        payload = result.summary_dict(); payload["iteration_log"] = result.iteration_log
        payload["gamma"] = int(config["gamma_value"])
        return payload

    def anchor(record: dict[str, Any], common: dict[str, Any], tolerance: float) -> dict[str, Any]:
        return _hybrid_certified_anchor(record, common_identity=common, tolerance=tolerance)

    def hybrid(config: dict[str, Any], instance: Any, baseline_record: dict[str, Any], anchor_value: dict[str, Any],
               common: dict[str, Any], checkpoint: Path, solver: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        expected = initial_upper_bound_expected_identity(common, anchor_value)
        expected.update({"instance_canonical_sha256": common["instance_canonical_sha256"],
                         "gamma": common["gamma"], "execution_attempt": ATTEMPT})
        result = solve_certified_hybrid_scenario_benders_fairness(
            instance, baseline_record=baseline_record, anchor=anchor_value, expected_identity=expected,
            solver_parameters=solver, rho=RHO, gamma=int(row["gamma"]), max_iterations=10000,
            time_limit=1800, tol=1e-4, feasibility_tolerance=1e-7,
            checkpoint_path=checkpoint, checkpoint_identity={"run_key": row["run_key"], **deepcopy(common)},
            execution_protocol_sha256=EXPECTED_PROTOCOL_SHA256, output_flag=False)
        payload = result.to_dict(); payload["reporting_counters"] = _hybrid_counters(payload)
        return payload

    def direct(config: dict[str, Any], instance: Any, baseline_record: dict[str, Any],
               anchor_value: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        return solve_gurobi_direct_extensive_form(
            instance, baseline_cost=float(anchor_value["value"]), rho=RHO, gamma=int(row["gamma"]),
            expected_scenario_count=scenario_count(int(row["gamma"])),
            solver_parameters=deepcopy(SOLVER_PARAMETERS), time_limit=1800, output_flag=False).to_dict()

    def post(config: dict[str, Any], instance: Any, result: dict[str, Any], anchor_value: dict[str, Any],
             identity: dict[str, Any], post_root: Path, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
        evaluation, timing = checkpointed_fairness_post_evaluation(
            instance, root=post_root, run_key=row["run_key"],
            config_sha256_value=identity["config_file_sha256"], git_commit=identity["git_commit"],
            baseline_anchor_sha256=anchor_value["anchor_sha256"], y_values=result["y_values"],
            x_values=result["x_values"], t_value=float(result["objective_t"]),
            baseline_cost=float(anchor_value["value"]), rho=RHO, gamma=int(row["gamma"]),
            max_scenarios=scenario_count(int(row["gamma"])), per_scenario_time_limit=30,
            tolerance=1e-7, chunk_size=25, resume_count=0, output_flag=False,
            run_execution_attempt=ATTEMPT, post_evaluation_pipeline_generation=1)
        return evaluation.to_dict(), {"post_evaluation_solver_runtime": timing.solver_runtime,
                                      "post_evaluation_wall_runtime": timing.wall_runtime,
                                      "aggregation_runtime": timing.aggregation_runtime,
                                      "checkpoint_io_runtime": timing.checkpoint_io_runtime}

    return Dependencies(generate, lambda value: value.to_dict(), InventoryInstance.from_dict,
                        baseline, anchor, hybrid, direct, post, _configure_solver_parameters)


def _manifest(config_path: Path, commit: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    forward = {row["run_key"]: row["run_directory_id"] for row in rows}
    return {"schema": SCHEMA, "identity": {"stage": STAGE, "execution_attempt": ATTEMPT,
            "git_commit": commit, "config_file_sha256": file_sha256(config_path).upper(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256, "hybrid_candidate_sha256": HYBRID_SHA256,
            "direct_candidate_sha256": EXPECTED_DIRECT_SHA256, "scale": "small", "seeds": SEEDS,
            "gamma": GAMMAS, "rho": RHO, "solver_parameters": SOLVER_PARAMETERS,
            "previous_attempt_results_reused": False}, "run_key_to_directory_id": forward,
            "directory_id_to_run_key": {value: key for key, value in forward.items()},
            "instance_identities": {}, "baseline_anchors": {}, "run_identities": {}}


def _write_record(output: Path, row: dict[str, Any], identity: dict[str, Any], result: dict[str, Any],
                  scientific: str, anchor: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime = float(result.get("algorithm_runtime", result.get("runtime", 0.0)))
    result.update({"algorithm_runtime": runtime,
                   "penalized_runtime_par2": _par2(scientific, runtime),
                   "post_evaluation_wall_runtime": float(result.get("post_evaluation_wall_runtime", 0.0)),
                   "total_wall_runtime": runtime + float(result.get("post_evaluation_wall_runtime", 0.0)) +
                   float(result.get("aggregation_runtime", 0.0)) + float(result.get("checkpoint_io_runtime", 0.0))})
    record = {**identity, "state": "complete", "algorithm_status": result.get("status"),
              "scientific_status": scientific, "solved_to_tolerance": scientific == "certified_robust_optimal",
              "result": result}
    if anchor is not None:
        record.update({"baseline_run_key": anchor["baseline_run_key"], "anchor_sha256": anchor["anchor_sha256"],
                       "anchor_value_hex": anchor["value_hex"], "baseline_robust_cost": anchor["value"],
                       "cost_budget": (1.0 + RHO) * float(anchor["value"])})
    run_root = output / "runs" / row["run_directory_id"]
    atomic_write_json(run_root / "run.json", record)
    _status(run_root / "status.json", identity, "complete", scientific)
    return record


def _result_row(record: dict[str, Any]) -> dict[str, Any]:
    result = record["result"]; post = result.get("post_evaluation") or {}; counters = result.get("reporting_counters") or {}
    return {"run_key": record["run_key"], "run_directory_id": record["run_directory_id"],
            "scale": record["scale"], "seed": record["seed"], "gamma": record["gamma"],
            "rho": record["rho"], "task_type": record["task_type"], "candidate": record["candidate"],
            "scientific_status": record["scientific_status"], "algorithm_runtime": result["algorithm_runtime"],
            "penalized_runtime_par2": result["penalized_runtime_par2"],
            "post_evaluation_wall_runtime": result["post_evaluation_wall_runtime"],
            "total_wall_runtime": result["total_wall_runtime"], "lower_bound": result.get("lower_bound", "NOT_APPLICABLE"),
            "upper_bound": result.get("upper_bound", "NOT_APPLICABLE"), "gap": result.get("gap", "NOT_APPLICABLE"),
            "objective_t": result.get("objective_t", "NOT_APPLICABLE"),
            "actual_robust_cost": post.get("actual_robust_cost", "NOT_APPLICABLE"),
            "iterations": result.get("iterations", "NOT_APPLICABLE"), "cuts": result.get("cuts", "NOT_APPLICABLE"),
            "master_runtime": result.get("master_runtime", "NOT_APPLICABLE"),
            "separation_runtime": result.get("separation_runtime", "NOT_APPLICABLE"),
            "scenario_blocks": (result.get("metadata") or {}).get("committed_scenario_count", 0),
            "model_build_runtime": result.get("model_build_runtime", "NOT_APPLICABLE"),
            "optimize_runtime": result.get("optimize_runtime", "NOT_APPLICABLE"),
            "rows": result.get("rows", "NOT_APPLICABLE"), "columns": result.get("columns", "NOT_APPLICABLE"),
            "binaries": result.get("binaries", "NOT_APPLICABLE"),
            "continuous_variables": result.get("continuous_variables", "NOT_APPLICABLE"),
            "nonzeros": result.get("nonzeros", "NOT_APPLICABLE"),
            "incumbent": result.get("incumbent", "NOT_APPLICABLE"),
            "objective_bound": result.get("objective_bound", "NOT_APPLICABLE"),
            "candidate_pool_maximum_size": counters.get("candidate_pool_maximum_size", "NOT_APPLICABLE"),
            "eviction_count": counters.get("eviction_count", "NOT_APPLICABLE"),
            "rediscovered_evicted_scenario_count": counters.get("rediscovered_evicted_scenario_count", "NOT_APPLICABLE"),
            "duplicate_proposal_count": counters.get("duplicate_proposal_count", "NOT_APPLICABLE"),
            "maximum_consecutive_non_improving_iterations": counters.get("maximum_consecutive_non_improving_iterations", "NOT_APPLICABLE"),
            "instance_canonical_sha256": record["instance_canonical_sha256"],
            "instance_archive_file_sha256": record["instance_archive_file_sha256"],
            "baseline_run_key": record.get("baseline_run_key", record["run_key"]),
            "anchor_sha256": record.get("anchor_sha256", "NOT_APPLICABLE")}


def _describe(values: list[float], prefix: str) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    _check(ordered and all(math.isfinite(value) for value in ordered), f"empty or nonfinite statistic: {prefix}")
    midpoint = len(ordered) // 2
    lower = ordered[:midpoint]
    upper = ordered[midpoint + (len(ordered) % 2):]
    q1 = statistics.median(lower) if lower else ordered[0]
    q3 = statistics.median(upper) if upper else ordered[-1]
    return {
        f"{prefix}_mean": statistics.fmean(ordered),
        f"{prefix}_median": statistics.median(ordered),
        f"{prefix}_std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        f"{prefix}_iqr": q3 - q1,
        f"{prefix}_min": ordered[0],
        f"{prefix}_max": ordered[-1],
    }


def _field_union(rows: list[dict[str, Any]], fallback: str) -> list[str]:
    return list(dict.fromkeys(key for row in rows for key in row)) if rows else [fallback]


def aggregate(output: Path, rows: list[dict[str, Any]], require_complete: bool = False) -> list[dict[str, Any]]:
    records = []
    by_key = {}
    for row in rows:
        record = read_json(output / "runs" / row["run_directory_id"] / "run.json")
        if record is not None:
            _check(record.get("run_key") == row["run_key"], "aggregation run identity mismatch")
            records.append(_result_row(record)); by_key[row["run_key"]] = record
    if require_complete:
        _check(len(records) == 45, "aggregation requires all 45 tasks")
    atomic_write_csv(output / "results.csv", records, list(records[0]) if records else ["run_key"])
    summary = []
    for task in ("baseline", "hybrid_frontier", "direct_extensive_frontier"):
        for gamma in GAMMAS + ["ALL"]:
            selected = [item for item in records if item["task_type"] == task and (gamma == "ALL" or item["gamma"] == gamma)]
            if not selected:
                continue
            if gamma == "ALL":
                by_seed_values = []
                for seed in sorted({item["seed"] for item in selected}):
                    seed_rows = [item for item in selected if item["seed"] == seed]
                    by_seed_values.append({
                        "algorithm_runtime": statistics.fmean(float(item["algorithm_runtime"]) for item in seed_rows),
                        "par2": statistics.fmean(float(item["penalized_runtime_par2"]) for item in seed_rows),
                    })
                runtime_values = [item["algorithm_runtime"] for item in by_seed_values]
                par2_values = [item["par2"] for item in by_seed_values]
            else:
                runtime_values = [float(item["algorithm_runtime"]) for item in selected]
                par2_values = [float(item["penalized_runtime_par2"]) for item in selected]
            certified = sum(item["scientific_status"] == "certified_robust_optimal" for item in selected)
            summary.append({"task_type": task, "gamma": gamma, "planned_tasks": len(selected), "completed_tasks": len(selected),
                            "independent_seed_count": len({item["seed"] for item in selected}),
                            "certified": certified, "certification_rate": certified / len(selected),
                            **_describe(runtime_values, "algorithm_runtime"),
                            **_describe(par2_values, "par2")})
    atomic_write_csv(output / "summary.csv", summary, _field_union(summary, "task_type"))
    paired = []
    if len(by_key) == 45:
        for seed in SEEDS:
            for gamma in GAMMAS:
                baseline = next(record for record in by_key.values() if record["seed"] == seed and record["gamma"] == gamma and record["task_type"] == "baseline")
                hybrid = next(record for record in by_key.values() if record["seed"] == seed and record["gamma"] == gamma and record["task_type"] == "hybrid_frontier")
                direct = next(record for record in by_key.values() if record["seed"] == seed and record["gamma"] == gamma and record["task_type"] == "direct_extensive_frontier")
                hr, dr = hybrid["result"], direct["result"]
                both = hybrid["scientific_status"] == direct["scientific_status"] == "certified_robust_optimal"
                paired.append({"seed": seed, "gamma": gamma,
                               "instance_sha256": hybrid["instance_canonical_sha256"],
                               "baseline_run_key": baseline["run_key"], "anchor_sha256": hybrid["anchor_sha256"],
                               "hybrid_status": hybrid["scientific_status"], "direct_status": direct["scientific_status"],
                               "hybrid_runtime": hr["algorithm_runtime"], "direct_runtime": dr["algorithm_runtime"],
                               "hybrid_par2": hr["penalized_runtime_par2"], "direct_par2": dr["penalized_runtime_par2"],
                               "runtime_difference_direct_minus_hybrid": dr["algorithm_runtime"] - hr["algorithm_runtime"],
                               "runtime_ratio_direct_over_hybrid": dr["algorithm_runtime"] / hr["algorithm_runtime"] if hr["algorithm_runtime"] > 0 else "NOT_APPLICABLE",
                               "par2_difference_direct_minus_hybrid": dr["penalized_runtime_par2"] - hr["penalized_runtime_par2"],
                               "par2_ratio_direct_over_hybrid": dr["penalized_runtime_par2"] / hr["penalized_runtime_par2"] if hr["penalized_runtime_par2"] > 0 else "NOT_APPLICABLE",
                               "objective_difference_direct_minus_hybrid": dr["objective_t"] - hr["objective_t"] if both else "NOT_APPLICABLE",
                               "cost_difference_direct_minus_hybrid": (dr["post_evaluation"]["actual_robust_cost"] - hr["post_evaluation"]["actual_robust_cost"]) if both else "NOT_APPLICABLE",
                               "hybrid_lower_bound": hr.get("lower_bound"), "hybrid_upper_bound": hr.get("upper_bound"), "hybrid_gap": hr.get("gap"),
                               "direct_lower_bound": dr.get("lower_bound"), "direct_upper_bound": dr.get("upper_bound"), "direct_gap": dr.get("gap"),
                               "hybrid_iterations": hr.get("iterations"), "hybrid_blocks": (hr.get("metadata") or {}).get("committed_scenario_count"),
                               "hybrid_cuts": hr.get("cuts"), "direct_rows": dr.get("rows"),
                               "direct_columns": dr.get("columns"), "direct_nonzeros": dr.get("nonzeros"),
                               "certification_agreement": hybrid["scientific_status"] == direct["scientific_status"]})
    atomic_write_csv(output / "paired_comparison.csv", paired, list(paired[0]) if paired else ["seed"])
    if paired:
        for gamma in GAMMAS + ["ALL"]:
            chosen = [item for item in paired if gamma == "ALL" or item["gamma"] == gamma]
            if gamma == "ALL":
                runtime_values = [statistics.fmean(
                    float(item["runtime_difference_direct_minus_hybrid"])
                    for item in chosen if item["seed"] == seed) for seed in SEEDS]
                par2_values = [statistics.fmean(
                    float(item["par2_difference_direct_minus_hybrid"])
                    for item in chosen if item["seed"] == seed) for seed in SEEDS]
            else:
                runtime_values = [float(item["runtime_difference_direct_minus_hybrid"]) for item in chosen]
                par2_values = [float(item["par2_difference_direct_minus_hybrid"]) for item in chosen]
            summary.append({"task_type": "paired_direct_minus_hybrid", "gamma": gamma,
                            "planned_tasks": len(chosen), "completed_tasks": len(chosen),
                            "independent_seed_count": len({item["seed"] for item in chosen}),
                            "certified": sum(item["hybrid_status"] == item["direct_status"] == "certified_robust_optimal" for item in chosen),
                            "certification_rate": sum(item["hybrid_status"] == item["direct_status"] == "certified_robust_optimal" for item in chosen) / len(chosen),
                            **_describe(runtime_values, "runtime_difference"),
                            **_describe(par2_values, "par2_difference")})
        atomic_write_csv(output / "summary.csv", summary, _field_union(summary, "task_type"))
    model_rows = [item for item in records if item["task_type"] == "direct_extensive_frontier"]
    stability_rows = []
    if paired:
        for seed in SEEDS:
            chosen = [item for item in paired if item["seed"] == seed]
            stability_rows.append({"seed": seed, "gamma_cells": 3,
                                   "mean_hybrid_runtime": statistics.fmean(float(item["hybrid_runtime"]) for item in chosen),
                                   "mean_direct_runtime": statistics.fmean(float(item["direct_runtime"]) for item in chosen),
                                   "mean_hybrid_par2": statistics.fmean(float(item["hybrid_par2"]) for item in chosen),
                                   "mean_direct_par2": statistics.fmean(float(item["direct_par2"]) for item in chosen),
                                   "hybrid_certified_cells": sum(item["hybrid_status"] == "certified_robust_optimal" for item in chosen),
                                   "direct_certified_cells": sum(item["direct_status"] == "certified_robust_optimal" for item in chosen)})
    atomic_write_csv(output / "model_size_summary.csv", model_rows, list(model_rows[0]) if model_rows else ["run_key"])
    atomic_write_csv(output / "high_gamma_stability.csv", stability_rows, list(stability_rows[0]) if stability_rows else ["run_key"])
    return records


def _deferred_checkpoint(solve: Callable[[], dict[str, Any]], checkpoint: Path,
                         identity: dict[str, Any], method: str, task_type: str, gamma: int) -> dict[str, Any]:
    interrupted = False
    old = signal.getsignal(signal.SIGINT)
    def handler(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, handler)
    try:
        result = solve()
        _check(isinstance(result, dict), "solver returned invalid payload")
        _validate_solver_result(result, task_type, gamma)
        _algorithm_checkpoint(checkpoint, identity, method, result)
    finally:
        signal.signal(signal.SIGINT, old)
    if interrupted:
        raise KeyboardInterrupt
    return result


def execute(config_path: Path, config: dict[str, Any], rows: list[dict[str, Any]], output: Path,
            commit: str, deps: Dependencies) -> dict[str, Any]:
    expected_manifest = _manifest(config_path, commit, rows)
    existing = read_json(output / "manifest.json")
    if output.exists():
        _check(existing is not None, "existing output lacks valid manifest")
        for key in ("schema", "identity", "run_key_to_directory_id", "directory_id_to_run_key"):
            _check(existing.get(key) == expected_manifest[key], f"strict resume manifest mismatch: {key}")
        _check(load_yaml(output / "resolved_config.yaml") == config, "resolved config mismatch")
        _check(read_json(output / "run_manifest.json") == existing, "run manifest mismatch")
        manifest = existing
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_yaml(output / "resolved_config.yaml", config)
        manifest = expected_manifest
        atomic_write_json(output / "manifest.json", manifest)
        atomic_write_json(output / "run_manifest.json", expected_manifest)
    for seed in SEEDS:
        for gamma in GAMMAS:
            cell_rows = [row for row in rows if row["seed"] == seed and row["gamma"] == gamma]
            baseline_row = next(row for row in cell_rows if row["task_type"] == "baseline")
            base_identity = _identity(baseline_row, config_path, commit)
            instance_path = output / "instances" / f"s{seed}_g{gamma}.json"
            stored = read_json(instance_path)
            cell_config = {**config, "gamma_value": gamma}
            if stored is None:
                instance = deps.generate_instance(cell_config, seed)
                serialized = deps.serialize_instance(instance)
                payload = _instance_payload(base_identity, serialized)
                atomic_write_json(instance_path, payload)
                file_sha = file_sha256(instance_path).upper()
            else:
                serialized, _canonical, _identity_sha = _validate_instance(stored, base_identity)
                instance = deps.deserialize_instance(serialized)
                file_sha = file_sha256(instance_path).upper()
            payload = read_json(instance_path); _check(payload is not None, "instance archive missing")
            serialized, canonical_sha, instance_identity_sha = _validate_instance(payload, base_identity)
            _check(sha256_value(deps.serialize_instance(instance)) == canonical_sha, "instance round-trip mismatch")
            baseline_identity = {**base_identity, "instance_sha256": canonical_sha,
                                 "instance_canonical_sha256": canonical_sha,
                                 "instance_identity_sha256": instance_identity_sha,
                                 "instance_archive_file_sha256": file_sha}
            base_root = output / "runs" / baseline_row["run_directory_id"]
            base_record = read_json(base_root / "run.json")
            _validate_status(base_root / "status.json", baseline_identity, base_record)
            if base_record is None:
                checkpoint = _load_checkpoint(base_root / "algorithm_checkpoint.json", baseline_identity, "baseline")
                if checkpoint is None:
                    _status(base_root / "status.json", baseline_identity, "running", "pending")
                    checkpoint = _deferred_checkpoint(
                        lambda: deps.solve_baseline(cell_config, instance, seed, deepcopy(SOLVER_PARAMETERS)),
                        base_root / "algorithm_checkpoint.json", baseline_identity, "baseline", "baseline", gamma)
                solved = checkpoint.get("status") == "optimal" and checkpoint.get("valid_UB") is True and type(checkpoint.get("gap")) in {int, float} and float(checkpoint["gap"]) <= 1e-4
                _check(solved, f"baseline failed for seed {seed} Gamma {gamma}; formal run stops fail closed")
                base_record = _write_record(output, baseline_row, baseline_identity, checkpoint,
                                            "certified_robust_optimal")
            else:
                _check(base_record.get("scientific_status") == "certified_robust_optimal", "stored baseline is not certified")
                _validate_completed_checkpoint(
                    base_record,
                    _load_checkpoint(base_root / "algorithm_checkpoint.json", baseline_identity, "baseline"),
                )
            common = {"instance_sha256": canonical_sha, "instance_canonical_sha256": canonical_sha,
                      "instance_identity_sha256": instance_identity_sha, "instance_archive_file_sha256": file_sha,
                      "seed": seed, "gamma": gamma, "scale": "small", "stage": STAGE,
                      "execution_attempt": ATTEMPT, "git_commit": commit,
                      "config_file_sha256": file_sha256(config_path).upper(),
                      "resolved_config_file_sha256": file_sha256(config_path).upper(),
                      "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                      "candidate_sha256": HYBRID_SHA256, "baseline_run_key": baseline_row["run_key"]}
            anchor = deps.make_anchor(base_record, common, 1e-4)
            cell_key = f"s{seed}_g{gamma}"
            manifest["instance_identities"].setdefault(cell_key, payload["identity"])
            _check(manifest["instance_identities"][cell_key] == payload["identity"], "manifest instance drift")
            manifest["baseline_anchors"].setdefault(cell_key, anchor)
            _check(manifest["baseline_anchors"][cell_key] == anchor, "manifest anchor drift")
            for row in [item for item in cell_rows if item["task_type"] != "baseline"]:
                identity = {**_identity(row, config_path, commit), **common,
                            "candidate_sha256": HYBRID_SHA256 if row["task_type"] == "hybrid_frontier" else EXPECTED_DIRECT_SHA256,
                            "baseline_run_key": baseline_row["run_key"], "anchor_sha256": anchor["anchor_sha256"],
                            "anchor_value_hex": anchor["value_hex"]}
                manifest["run_identities"].setdefault(row["run_key"], identity)
                _check(manifest["run_identities"][row["run_key"]] == identity, "manifest run identity drift")
                atomic_write_json(output / "manifest.json", manifest)
                atomic_write_json(output / "run_manifest.json", manifest)
                run_root = output / "runs" / row["run_directory_id"]
                record = read_json(run_root / "run.json")
                _validate_status(run_root / "status.json", identity, record)
                if record is not None:
                    _check(all(record.get(key) == value for key, value in identity.items()),
                           "stored frontier identity mismatch")
                    method = "hybrid" if row["task_type"] == "hybrid_frontier" else "direct_extensive_form"
                    _validate_completed_checkpoint(
                        record,
                        _load_checkpoint(run_root / "algorithm_checkpoint.json", identity, method),
                    )
                    continue
                _status(run_root / "status.json", identity, "running", "pending")
                method = "hybrid" if row["task_type"] == "hybrid_frontier" else "direct_extensive_form"
                checkpoint_path = run_root / "algorithm_checkpoint.json"
                result = _load_checkpoint(checkpoint_path, identity, method)
                if result is None:
                    if row["task_type"] == "hybrid_frontier":
                        hybrid_checkpoint = run_root / "hybrid_internal_checkpoint.json"
                        solve = lambda: deps.solve_hybrid(cell_config, instance, base_record, anchor, common,
                                                          hybrid_checkpoint, deepcopy(SOLVER_PARAMETERS), row)
                    else:
                        solve = lambda: deps.solve_direct(cell_config, instance, base_record, anchor, row)
                    result = _deferred_checkpoint(solve, checkpoint_path, identity, method,
                                                  row["task_type"], gamma)
                algorithm_status = _frontier_scientific(row["task_type"], result,
                                                        {"valid": True, "errors": [],
                                                         "objective_t_consistent": True,
                                                         "scenario_count": scenario_count(gamma)},
                                                        scenario_count(gamma), 1e-4)
                algorithm_certified = algorithm_status == "certified_robust_optimal"
                evaluation = None; timing = {"post_evaluation_wall_runtime": 0.0, "post_evaluation_solver_runtime": 0.0,
                                             "aggregation_runtime": 0.0, "checkpoint_io_runtime": 0.0}
                if algorithm_certified:
                    evaluation, timing = deps.post_evaluate(cell_config, instance, result, anchor, identity,
                                                            run_root / "post_evaluation", row)
                scientific = _frontier_scientific(row["task_type"], result, evaluation,
                                                  scenario_count(gamma), 1e-4)
                result.update(timing); result["post_evaluation"] = evaluation
                record = _write_record(output, row, identity, result, scientific, anchor)
                aggregate(output, rows)
                if scientific == "invalid_post_evaluation":
                    raise HighGammaGateError("invalid post-evaluation; formal run stops fail closed")
    aggregate(output, rows, require_complete=True)
    manifest["completed_run_count"] = 45
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_json(output / "run_manifest.json", manifest)
    atomic_write_json(output / "audit.log", {"schema": SCHEMA, "identity": manifest["identity"],
                                              "completed_run_count": 45})
    return manifest


def formal_run(config_path: str | Path, *, resume: bool, authorization_file: str | Path | None,
               dependencies: Dependencies | None = None, test_authorization: bool = False,
               test_root: Path | None = None) -> dict[str, Any]:
    _check(resume, "strict --resume is required; --overwrite is unsupported")
    path = Path(config_path)
    config = load_yaml(path); validate_config(path, config)
    root = (test_root or _root()).resolve()
    if test_authorization:
        commit = "T" * 40
    else:
        _check(dependencies is None, "formal dependency substitution forbidden")
        commit = _git_gate(root, config)
        _check(authorization_file is not None, "formal run authorization file required")
        validate_authorization(authorization_file, path, root, config)
        seed_access_gate(root)
    paths = _planned_paths(root, expand_plan(), config)
    _check(max(len(str(item[1])) for item in paths) <= 220, "Windows path limit exceeded")
    output = root / config["output_dir"]
    if not output.exists():
        _check(not any((root / part).exists() for part in ("experiments/results_fh_ext/hg0",)), "prior output reuse forbidden")
    deps = dependencies or production_dependencies()
    deps.configure_solver(deepcopy(SOLVER_PARAMETERS))
    return execute(path, config, expand_plan(), output, commit, deps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _check(args.stage == STAGE, "wrong stage")
    if args.dry_run:
        print(json.dumps(dry_run(args.config), indent=2, sort_keys=True))
        return 0
    result = formal_run(args.config, resume=args.resume, authorization_file=args.authorization_file)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
