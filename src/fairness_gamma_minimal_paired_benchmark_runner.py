from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable
import zipfile

import yaml

from .experiment_protocol import atomic_write_csv, atomic_write_json, atomic_write_yaml, file_sha256


STAGE = "FAIRNESS_GAMMA_MINIMAL_PAIRED_BENCHMARK"
SCALES = ("medium_large", "large")
SEEDS = tuple(range(180, 185))
GAMMA = 2
RHO = 0.025
CANDIDATE = "certified_single_cut_without_complete_scenario_blocks"
ATTEMPT = 1
SOURCE_ZIP_SHA256 = "EE45A00AA341EE5EB2894DE43EE2F47022C27F1D29146FCFEC803236EF59DB6F"
SOURCE_COMMIT = "b1b5e9908bbb685b8a852aff762f08ce7226aba1"
SOURCE_CONFIG_SHA256 = "C26236A93E669B877D74DE0F08D0BC86817345821DEF91066911D723788C7C07"
SOURCE_PROTOCOL_SHA256 = "F8D058C390FC9446DD9885E58ACE06EAA2685B6B9007D1E787ABDDF66B1EBB0E"
SOURCE_CANDIDATE_SHA256 = "8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0"
RESULT_FIELDS = (
    "run_key", "run_directory_id", "scale", "seed", "gamma", "rho", "candidate",
    "scientific_status", "algorithm_runtime", "master_runtime", "separation_runtime",
    "post_evaluation_wall_runtime", "total_wall_runtime", "penalized_runtime_par2",
    "final_gap", "iterations", "scenario_blocks", "certified_farkas_cuts", "objective_t",
    "actual_robust_cost", "instance_sha256", "baseline_run_key", "anchor_sha256",
)


class BenchmarkGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class Dependencies:
    solve_reference: Callable[..., dict[str, Any]]
    post_evaluate: Callable[..., tuple[dict[str, Any], dict[str, float]]]
    deserialize_instance: Callable[[dict[str, Any]], Any]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def run_key(scale: str, seed: int) -> str:
    return canonical_json({
        "candidate": CANDIDATE, "execution_attempt": ATTEMPT, "gamma": GAMMA,
        "rho": "0.025", "scale": scale, "seed": seed, "stage": STAGE,
        "task_type": "reference_frontier",
    })


def run_directory_id(key: str) -> str:
    return "r_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _source_prefix(scale: str) -> str:
    if scale not in SCALES:
        raise BenchmarkGateError("invalid source scale")
    return f"experiments/results_fh_gamma/{'ml_a3' if scale == 'medium_large' else 'lg_a3'}"


def _source_hybrid_key(scale: str, seed: int) -> str:
    return canonical_json({
        "candidate": "certified_hybrid_scenario_benders_fairness", "execution_attempt": 3,
        "gamma": 2, "rho": "0.025", "scale": scale, "seed": seed,
        "stage": "GAMMA_SENSITIVITY", "task_type": "frontier",
    })


def expand_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scale in SCALES:
        for seed in SEEDS:
            key = run_key(scale, seed)
            rows.append({
                "scale": scale, "seed": seed, "gamma": GAMMA, "rho": "0.025",
                "candidate": CANDIDATE, "task_type": "reference_frontier",
                "run_key": key, "run_directory_id": run_directory_id(key),
            })
    keys = [row["run_key"] for row in rows]
    dirs = [row["run_directory_id"] for row in rows]
    if len(rows) != 10 or len(keys) != len(set(keys)) or len(dirs) != len(set(dirs)):
        raise BenchmarkGateError("benchmark plan duplicate or directory collision")
    return rows


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkGateError(f"{path} must contain a YAML object")
    return value


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkGateError(f"{path} must contain a JSON object")
    return value


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    if result.returncode:
        raise BenchmarkGateError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def validate_config(config_path: str | Path, config: dict[str, Any]) -> None:
    expected = {
        "stage": STAGE, "execution_attempt": ATTEMPT, "previous_benchmark_results_reused": False,
        "scales": list(SCALES), "seeds": list(SEEDS), "gamma": GAMMA, "rho": RHO,
        "reference_candidate": CANDIDATE, "reference_frontier_count": 10,
        "baseline_new_count": 0, "hybrid_new_count": 0, "overwrite_supported": False,
        "algorithm_time_limit_seconds": 1800, "checkpoint_chunk_size": 25,
    }
    for field, wanted in expected.items():
        if config.get(field) != wanted:
            raise BenchmarkGateError(f"config {field} is outside the frozen protocol")
    if config.get("solver_identity") != {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}:
        raise BenchmarkGateError("solver identity drift")
    if config.get("post_evaluation_time_limit_per_scenario_seconds") != 30:
        raise BenchmarkGateError("post-evaluation time limit drift")
    if config.get("par2") != {"basis": "algorithm_runtime", "multiplier": 2}:
        raise BenchmarkGateError("PAR-2 identity drift")
    root = Path(__file__).resolve().parents[1]
    checks = {
        "protocol_sha256": file_sha256(root / config["protocol_document"]).upper(),
        "candidate_sha256": file_sha256(root / config["candidate_definition"]).upper(),
        "source_catalog_sha256": file_sha256(root / config["source_catalog"]).upper(),
    }
    for field, actual in checks.items():
        if config.get(field) != actual:
            raise BenchmarkGateError(f"config {field} mismatch")


def load_catalog(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    value = load_json(root / config["source_catalog"])
    if value.get("source_zip_sha256") != SOURCE_ZIP_SHA256:
        raise BenchmarkGateError("source catalog ZIP SHA mismatch")
    cells = value.get("cells")
    if not isinstance(cells, list) or len(cells) != 10:
        raise BenchmarkGateError("source catalog must contain ten cells")
    planned = {(r["scale"], r["seed"]): r for r in expand_plan()}
    seen: set[tuple[str, int]] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise BenchmarkGateError("invalid source catalog cell")
        key = (cell.get("scale"), cell.get("seed"))
        if key not in planned or key in seen:
            raise BenchmarkGateError("source catalog cell mismatch")
        seen.add(key)
        for field in (
            "source_hybrid_run_key", "instance_canonical_sha256", "instance_file_sha256",
            "baseline_run_key", "anchor_sha256", "anchor_value_hex",
        ):
            if not isinstance(cell.get(field), str) or not cell[field]:
                raise BenchmarkGateError(f"source catalog {field} is missing")
        if cell.get("source_hybrid_scientific_status") != "certified_robust_optimal":
            raise BenchmarkGateError("source Hybrid is not certified")
        if cell.get("baseline_scientific_status") != "certified_robust_optimal":
            raise BenchmarkGateError("source baseline is not optimal")
    return sorted(cells, key=lambda row: (SCALES.index(row["scale"]), row["seed"]))


def _planned_paths(root: Path, rows: list[dict[str, Any]], chunk_size: int) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for name in ("manifest.json", "run_manifest.json", "resolved_config.yaml", "results.csv", "summary.csv", "paired_comparison.csv", "audit_log.json"):
        paths.extend(((name, root / name), (name + "_tmp", root / ("." + name + ".tmp"))))
    for row in rows:
        rr = root / "runs" / row["run_directory_id"]
        for name in ("run.json", "status.json", "algorithm_checkpoint.json"):
            paths.extend(((name, rr / name), (name + "_tmp", rr / ("." + name + ".tmp"))))
        count = 1831 if row["scale"] == "medium_large" else 4657
        chunk = math.ceil(count / chunk_size) - 1
        for name in ("post_evaluation.json", "checkpoint/index.json", f"checkpoint/chunk_{chunk:05d}.json"):
            path = rr / "post_evaluation" / name
            paths.extend(((name, path), (name + "_tmp", path.with_name("." + path.name + ".tmp"))))
    return paths


def verify_source_metadata(source_zip: str | Path, cells: list[dict[str, Any]]) -> int:
    source_zip = Path(source_zip)
    if file_sha256(source_zip).upper() != SOURCE_ZIP_SHA256:
        raise BenchmarkGateError("source ZIP SHA mismatch during dry-run")
    with zipfile.ZipFile(source_zip) as archive:
        for cell in cells:
            prefix = _source_prefix(cell["scale"])
            manifest = json.loads(archive.read(f"{prefix}/manifest.json"))
            mapping = manifest.get("run_key_to_directory_id", {})
            hdir = mapping.get(cell["source_hybrid_run_key"])
            bdir = mapping.get(cell["baseline_run_key"])
            if hdir != cell["source_hybrid_directory_id"] or not isinstance(bdir, str):
                raise BenchmarkGateError("source dry-run mapping mismatch")
            hybrid = json.loads(archive.read(f"{prefix}/runs/{hdir}/run.json"))
            baseline = json.loads(archive.read(f"{prefix}/runs/{bdir}/run.json"))
            if (
                hybrid.get("scientific_status") != "certified_robust_optimal"
                or hybrid.get("instance_canonical_sha256") != cell["instance_canonical_sha256"]
                or hybrid.get("baseline_run_key") != cell["baseline_run_key"]
                or hybrid.get("anchor_sha256") != cell["anchor_sha256"]
                or baseline.get("scientific_status") != "certified_robust_optimal"
                or float(baseline.get("result", {}).get("upper_bound")).hex() != cell["anchor_value_hex"]
            ):
                raise BenchmarkGateError("source dry-run scientific identity mismatch")
    return len(cells)


def dry_run(config_path: str | Path, *, formal_root: str | Path | None = None) -> dict[str, Any]:
    gurobi_before = "gurobipy" in sys.modules
    config = load_yaml(config_path)
    validate_config(config_path, config)
    cells = load_catalog(config)
    verified_metadata = verify_source_metadata(config["source_zip"], cells)
    rows = expand_plan()
    root = Path(formal_root or config["formal_worktree_root"])
    output = root / config["output_relative_path"]
    path_type, longest = max(_planned_paths(output, rows, config["checkpoint_chunk_size"]), key=lambda x: len(str(x[1])))
    report = {
        "stage": STAGE, "reference_frontier": 10, "baseline_new": 0, "hybrid_new": 0,
        "scales": list(SCALES), "seeds": list(SEEDS), "gamma": GAMMA, "rho": RHO,
        "source_pairing_cells_verified": len(cells), "unique_run_keys": len({r["run_key"] for r in rows}),
        "unique_directory_ids": len({r["run_directory_id"] for r in rows}), "directory_collisions": 0,
        "instances_generated": False, "solver_called": False,
        "gurobipy_imported": "gurobipy" in sys.modules,
        "gurobipy_imported_by_dry_run": "gurobipy" in sys.modules and not gurobi_before,
        "output_dir_exists": output.exists(),
        "source_metadata_cells_verified": verified_metadata,
        "source_instance_payloads_read": False, "longest_windows_path": str(longest),
        "longest_windows_path_length": len(str(longest)), "longest_windows_path_type": path_type,
    }
    if report["longest_windows_path_length"] >= 220:
        raise BenchmarkGateError("Windows path length is not below 220")
    if report["gurobipy_imported_by_dry_run"]:
        raise BenchmarkGateError("dry-run imported gurobipy")
    return report


def _strict_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise BenchmarkGateError(f"{field} must be a finite JSON number")
    return float(value)


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkGateError(f"{field} must be a nonnegative integer")
    return value


def validate_solution_payload(result: dict[str, Any], serialized_instance: dict[str, Any], *, baseline: bool = False) -> None:
    warehouses = _strict_int(serialized_instance.get("num_warehouses"), "instance.num_warehouses")
    products = _strict_int(serialized_instance.get("num_products"), "instance.num_products")
    x_name, y_name = ("best_x_values", "best_y_values") if baseline else ("x_values", "y_values")
    x_values, y_values = result.get(x_name), result.get(y_name)
    if not isinstance(x_values, list) or len(x_values) != warehouses:
        raise BenchmarkGateError(f"{x_name} must follow frozen warehouse order")
    for i, row in enumerate(x_values):
        if not isinstance(row, list) or len(row) != products:
            raise BenchmarkGateError(f"{x_name}[{i}] must follow frozen product order")
        for j, value in enumerate(row):
            _strict_number(value, f"{x_name}[{i}][{j}]")
    if not isinstance(y_values, list) or len(y_values) != warehouses:
        raise BenchmarkGateError(f"{y_name} must follow frozen warehouse order")
    for i, value in enumerate(y_values):
        _strict_number(value, f"{y_name}[{i}]")


def _final_certificate(result: dict[str, Any]) -> bool:
    log = result.get("iteration_log")
    final = log[-1] if isinstance(log, list) and log and isinstance(log[-1], dict) else {}
    bound = final.get("separation_objective_bound")
    return bool(
        result.get("status") == "optimal"
        and isinstance(result.get("gap"), (int, float)) and not isinstance(result.get("gap"), bool)
        and math.isfinite(float(result["gap"])) and float(result["gap"]) <= 1e-4
        and final.get("certification_active") is True
        and final.get("robust_feasibility_certified") is True
        and final.get("master_status") in {2, "optimal"}
        and final.get("separation_status") == "optimal"
        and final.get("separation_requested_mip_gap") == 0.0
        and isinstance(bound, (int, float)) and not isinstance(bound, bool)
        and math.isfinite(float(bound)) and float(bound) <= 1e-7
    )


def _solve_and_checkpoint_with_deferred_ctrl_c(
    solve: Callable[[], dict[str, Any]], checkpoint_path: Path,
    identity: dict[str, Any], serialized: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    interrupted = False
    previous = signal.getsignal(signal.SIGINT)

    def defer(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, defer)
    try:
        result = solve()
        validate_solution_payload(result, serialized)
        atomic_write_json(checkpoint_path, {"identity": identity, "result": result})
    finally:
        signal.signal(signal.SIGINT, previous)
    return result, interrupted


def classify_status(result: dict[str, Any], post: dict[str, Any] | None, expected_scenarios: int) -> str:
    if not _final_certificate(result):
        status = str(result.get("status", "unknown"))
        return "time_limit_uncertified" if status == "time_limit" else "master_optimal_but_robust_uncertified"
    if not isinstance(post, dict) or post.get("valid") is not True or post.get("errors") != []:
        return "invalid_post_evaluation"
    if post.get("objective_t_consistent") is not True or post.get("scenario_count") != expected_scenarios:
        return "invalid_post_evaluation"
    return "certified_robust_optimal"


def _identity(config_path: Path, config: dict[str, Any], row: dict[str, Any], cell: dict[str, Any], commit: str) -> dict[str, Any]:
    return {
        "schema": "fairness_gamma_minimal_paired_benchmark_run_v1", "stage": STAGE,
        "execution_attempt": ATTEMPT, "previous_benchmark_results_reused": False,
        "git_commit": commit, "config_file_sha256": file_sha256(config_path).upper(),
        "protocol_sha256": config["protocol_sha256"], "candidate_sha256": config["candidate_sha256"],
        "source_zip_sha256": SOURCE_ZIP_SHA256, "run_key": row["run_key"],
        "run_directory_id": row["run_directory_id"], "task_type": row["task_type"],
        "candidate": CANDIDATE, "scale": row["scale"], "seed": row["seed"],
        "gamma": GAMMA, "rho": "0.025", "solver_parameters": config["solver_identity"],
        "instance_canonical_sha256": cell["instance_canonical_sha256"],
        "instance_file_sha256": cell["instance_file_sha256"],
        "baseline_run_key": cell["baseline_run_key"], "anchor_sha256": cell["anchor_sha256"],
        "source_hybrid_run_key": cell["source_hybrid_run_key"],
    }


def _read_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkGateError(f"corrupt JSON checkpoint: {path}") from exc


def _write_status(path: Path, identity: dict[str, Any], state: str, phase: str, scientific: str) -> None:
    atomic_write_json(path, {"identity": identity, "state": state, "phase": phase, "scientific_status": scientific})


def _load_source_instance(source_zip: Path, cell: dict[str, Any]) -> dict[str, Any]:
    if file_sha256(source_zip).upper() != SOURCE_ZIP_SHA256:
        raise BenchmarkGateError("source ZIP SHA mismatch before instance read")
    with zipfile.ZipFile(source_zip) as archive:
        try:
            raw = archive.read(cell["instance_member"])
        except KeyError as exc:
            raise BenchmarkGateError("source instance member is missing") from exc
    if hashlib.sha256(raw).hexdigest().upper() != cell["instance_file_sha256"]:
        raise BenchmarkGateError("source instance file SHA mismatch")
    archive_payload = json.loads(raw)
    serialized = archive_payload.get("instance")
    stored_identity = archive_payload.get("identity")
    if not isinstance(serialized, dict) or not isinstance(stored_identity, dict):
        raise BenchmarkGateError("source instance archive schema mismatch")
    if sha256_value(serialized) != cell["instance_canonical_sha256"]:
        raise BenchmarkGateError("source instance canonical SHA mismatch")
    if sha256_value(stored_identity) != cell["instance_identity_sha256"]:
        raise BenchmarkGateError("source instance scientific identity mismatch")
    if (stored_identity.get("scale"), stored_identity.get("seed"), stored_identity.get("gamma")) != (cell["scale"], cell["seed"], 2):
        raise BenchmarkGateError("source instance scale/seed/Gamma mismatch")
    return serialized


def production_dependencies() -> Dependencies:
    # Solver imports are intentionally isolated behind all authorization and source gates.
    from .fairness_benders import solve_fairness_benders
    from .fairness_large_final_remediation_runner import _configure_solver_parameters
    from .fairness_post_evaluation import checkpointed_fairness_post_evaluation
    from .instance import InventoryInstance

    def solve_reference(config: dict[str, Any], instance: Any, cell: dict[str, Any]) -> dict[str, Any]:
        _configure_solver_parameters(config["solver_identity"])
        algorithm = deepcopy(load_yaml(Path(__file__).resolve().parents[1] / config["algorithm_template"])["algorithm"])
        algorithm.update({
            "fairness_scalability_strategy": "single_cut", "max_cuts_per_iteration": 1,
            "persistent_separation_enabled": False, "certified_scenario_cache_enabled": False,
            "separation_solution_pool_enabled": False, "complete_scenario_recourse_blocks_enabled": False,
        })
        result = solve_fairness_benders(
            instance, baseline_cost=float(cell["anchor_value"]), rho=RHO, gamma=GAMMA,
            algorithm_config=algorithm, max_iterations=int(config["max_iterations"]),
            time_limit=float(config["algorithm_time_limit_seconds"]), tol=float(config["tol"]),
            feasibility_tolerance=float(config["solver_identity"]["FeasibilityTol"]), output_flag=False,
        )
        return result.to_dict()

    def post_evaluate(config: dict[str, Any], instance: Any, result: dict[str, Any], cell: dict[str, Any], identity: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, float]]:
        evaluation, timing = checkpointed_fairness_post_evaluation(
            instance, root=root, run_key=identity["run_key"],
            config_sha256_value=identity["config_file_sha256"], git_commit=identity["git_commit"],
            baseline_anchor_sha256=identity["anchor_sha256"], y_values=result["y_values"],
            x_values=result["x_values"], t_value=float(result["objective_t"]),
            baseline_cost=float(cell["anchor_value"]), rho=RHO, gamma=GAMMA,
            max_scenarios=1831 if identity["scale"] == "medium_large" else 4657,
            per_scenario_time_limit=float(config["post_evaluation_time_limit_per_scenario_seconds"]),
            tolerance=float(config["solver_identity"]["FeasibilityTol"]),
            chunk_size=int(config["checkpoint_chunk_size"]), resume_count=0, output_flag=False,
            run_execution_attempt=ATTEMPT,
            post_evaluation_pipeline_generation=int(config["post_evaluation_pipeline_generation"]),
        )
        return evaluation.to_dict(), {
            "post_evaluation_solver_runtime": timing.solver_runtime,
            "post_evaluation_wall_runtime": timing.wall_runtime,
            "aggregation_runtime": timing.aggregation_runtime,
            "checkpoint_io_runtime": timing.checkpoint_io_runtime,
        }

    return Dependencies(solve_reference, post_evaluate, InventoryInstance.from_dict)


def _result_row(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result")
    if not isinstance(result, dict):
        raise BenchmarkGateError("run result missing")
    post = result.get("post_evaluation")
    algorithm_runtime = _strict_number(result.get("algorithm_runtime", result.get("runtime")), "algorithm_runtime")
    scientific = str(record.get("scientific_status"))
    par2 = algorithm_runtime if scientific == "certified_robust_optimal" else 3600.0
    logs = result.get("iteration_log")
    logs = logs if isinstance(logs, list) else []
    master_runtime = math.fsum(_strict_number(row.get("master_time", 0.0), "master_time") for row in logs if isinstance(row, dict))
    separation_runtime = math.fsum(_strict_number(row.get("subproblem_time", 0.0), "subproblem_time") for row in logs if isinstance(row, dict))
    post_wall = _strict_number(result.get("post_evaluation_wall_runtime", 0.0), "post_evaluation_wall_runtime")
    return {
        "run_key": record["run_key"], "run_directory_id": record["run_directory_id"],
        "scale": record["scale"], "seed": record["seed"], "gamma": GAMMA, "rho": "0.025",
        "candidate": CANDIDATE, "scientific_status": scientific,
        "algorithm_runtime": algorithm_runtime, "master_runtime": master_runtime,
        "separation_runtime": separation_runtime, "post_evaluation_wall_runtime": post_wall,
        "total_wall_runtime": algorithm_runtime + post_wall,
        "penalized_runtime_par2": par2, "final_gap": result.get("gap", "NOT_APPLICABLE"),
        "iterations": _strict_int(result.get("iterations", len(logs)), "iterations"),
        "scenario_blocks": 0, "certified_farkas_cuts": _strict_int(result.get("cuts", 0), "cuts"),
        "objective_t": result.get("objective_t", "NOT_APPLICABLE") if scientific == "certified_robust_optimal" else "NOT_APPLICABLE",
        "actual_robust_cost": post.get("actual_robust_cost", "NOT_APPLICABLE") if isinstance(post, dict) and post.get("valid") is True else "NOT_APPLICABLE",
        "instance_sha256": record["instance_canonical_sha256"],
        "baseline_run_key": record["baseline_run_key"], "anchor_sha256": record["anchor_sha256"],
    }


def _paired_row(reference: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    both = reference["scientific_status"] == "certified_robust_optimal" and cell["source_hybrid_scientific_status"] == "certified_robust_optimal"
    reference_runtime = float(reference["algorithm_runtime"])
    hybrid_runtime = float(cell["source_hybrid_algorithm_runtime"])
    return {
        "scale": reference["scale"], "seed": reference["seed"], "gamma": GAMMA, "rho": "0.025",
        "instance_sha256": reference["instance_sha256"], "baseline_run_key": reference["baseline_run_key"],
        "anchor_sha256": reference["anchor_sha256"],
        "hybrid_scientific_status": cell["source_hybrid_scientific_status"],
        "reference_scientific_status": reference["scientific_status"],
        "hybrid_algorithm_runtime": hybrid_runtime, "reference_algorithm_runtime": reference_runtime,
        "hybrid_par2": cell["source_hybrid_par2"], "reference_par2": reference["penalized_runtime_par2"],
        "runtime_difference_reference_minus_hybrid": reference_runtime - hybrid_runtime if both else "NOT_APPLICABLE",
        "runtime_ratio_reference_over_hybrid": reference_runtime / hybrid_runtime if both else "NOT_APPLICABLE",
        "hybrid_iterations": cell["source_hybrid_iterations"], "reference_iterations": reference["iterations"],
        "hybrid_scenario_blocks": cell["source_hybrid_scenario_blocks"], "reference_scenario_blocks": reference["scenario_blocks"],
        "hybrid_certified_farkas_cuts": cell["source_hybrid_certified_farkas_cuts"],
        "reference_certified_farkas_cuts": reference["certified_farkas_cuts"],
        "objective_t_difference_reference_minus_hybrid": float(reference["objective_t"]) - float(cell["source_hybrid_objective_t"]) if both else "NOT_APPLICABLE",
        "cost_difference_reference_minus_hybrid": float(reference["actual_robust_cost"]) - float(cell["source_hybrid_actual_robust_cost"]) if both else "NOT_APPLICABLE",
        "certification_agreement": reference["scientific_status"] == cell["source_hybrid_scientific_status"],
    }


def aggregate(output: Path, rows: list[dict[str, Any]], cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        record = _read_optional(output / "runs" / row["run_directory_id"] / "run.json")
        if record is not None:
            results.append(_result_row(record))
    results.sort(key=lambda x: (SCALES.index(x["scale"]), x["seed"]))
    atomic_write_csv(output / "results.csv", results, list(RESULT_FIELDS))
    summaries: list[dict[str, Any]] = []
    for scale in SCALES:
        subset = [row for row in results if row["scale"] == scale]
        solved = [row for row in subset if row["scientific_status"] == "certified_robust_optimal"]
        summaries.append({
            "scale": scale, "planned": 5, "completed": len(subset), "certified_solved": len(solved),
            "certified_solved_rate": len(solved) / 5,
            "mean_algorithm_runtime": math.fsum(float(r["algorithm_runtime"]) for r in subset) / len(subset) if subset else "NOT_APPLICABLE",
            "mean_par2": math.fsum(float(r["penalized_runtime_par2"]) for r in subset) / len(subset) if subset else "NOT_APPLICABLE",
        })
    atomic_write_csv(output / "summary.csv", summaries, list(summaries[0]))
    cell_map = {(cell["scale"], cell["seed"]): cell for cell in cells}
    paired = [_paired_row(row, cell_map[(row["scale"], row["seed"])]) for row in results]
    atomic_write_csv(output / "paired_comparison.csv", paired, list(paired[0]) if paired else [])
    atomic_write_json(output / "audit_log.json", {
        "planned": 10, "completed": len(results),
        "certified_solved": sum(r["scientific_status"] == "certified_robust_optimal" for r in results),
        "unique_run_keys": len(results) == len({r["run_key"] for r in results}),
        "source_zip_sha256": SOURCE_ZIP_SHA256,
    })
    return results


def execute_plan(config_path: Path, config: dict[str, Any], cells: list[dict[str, Any]], output: Path, *, commit: str, dependencies: Dependencies, source_zip: Path, source_instance_loader: Callable[[Path, dict[str, Any]], dict[str, Any]] = _load_source_instance) -> dict[str, Any]:
    rows = expand_plan()
    cell_map = {(cell["scale"], cell["seed"]): cell for cell in cells}
    mapping = {row["run_key"]: row["run_directory_id"] for row in rows}
    reverse = {value: key for key, value in mapping.items()}
    manifest_identity = {
        "schema": "fairness_gamma_minimal_paired_benchmark_manifest_v1", "stage": STAGE,
        "execution_attempt": ATTEMPT, "git_commit": commit,
        "config_file_sha256": file_sha256(config_path).upper(), "protocol_sha256": config["protocol_sha256"],
        "candidate_sha256": config["candidate_sha256"], "source_zip_sha256": SOURCE_ZIP_SHA256,
        "previous_benchmark_results_reused": False, "solver_parameters": config["solver_identity"],
    }
    existing = _read_optional(output / "manifest.json") if output.exists() else None
    if output.exists() and existing is None:
        raise BenchmarkGateError("existing output lacks a valid manifest")
    if existing is not None and (
        existing.get("identity") != manifest_identity
        or existing.get("run_key_to_directory_id") != mapping
        or existing.get("directory_id_to_run_key") != reverse
    ):
        raise BenchmarkGateError("resume manifest or run-directory mapping mismatch")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"identity": manifest_identity, "run_key_to_directory_id": mapping, "directory_id_to_run_key": reverse}
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_json(output / "run_manifest.json", {**manifest, "source_cells": cells})
    atomic_write_yaml(output / "resolved_config.yaml", config)
    for row in rows:
        cell = cell_map[(row["scale"], row["seed"])]
        identity = _identity(config_path, config, row, cell, commit)
        run_root = output / "runs" / row["run_directory_id"]
        run_path, status_path = run_root / "run.json", run_root / "status.json"
        checkpoint_path = run_root / "algorithm_checkpoint.json"
        record = _read_optional(run_path)
        status = _read_optional(status_path)
        if record is not None:
            if record.get("identity") != identity or record.get("state") != "complete":
                raise BenchmarkGateError("completed run identity mismatch")
            if status is None or status.get("identity") != identity or status.get("state") != "complete":
                raise BenchmarkGateError("completed status identity mismatch")
            checkpoint = _read_optional(checkpoint_path)
            if checkpoint is None or checkpoint.get("identity") != identity:
                raise BenchmarkGateError("completed algorithm checkpoint missing or mismatched")
            continue
        if status is not None and status.get("identity") != identity:
            raise BenchmarkGateError("resume status identity mismatch")
        checkpoint = _read_optional(checkpoint_path)
        if checkpoint is not None and (checkpoint.get("identity") != identity or not isinstance(checkpoint.get("result"), dict)):
            raise BenchmarkGateError("algorithm checkpoint identity mismatch")
        run_root.mkdir(parents=True, exist_ok=True)
        try:
            if checkpoint is None:
                _write_status(status_path, identity, "running", "algorithm", "not_yet_certified")
                serialized = source_instance_loader(source_zip, cell)
                result, ctrl_c_deferred = _solve_and_checkpoint_with_deferred_ctrl_c(
                    lambda: dependencies.solve_reference(config, dependencies.deserialize_instance(serialized), cell),
                    checkpoint_path, identity, serialized,
                )
                if ctrl_c_deferred:
                    raise KeyboardInterrupt
            else:
                result = checkpoint["result"]
                serialized = source_instance_loader(source_zip, cell)
                validate_solution_payload(result, serialized)
            post: dict[str, Any] | None = None
            timings = {"post_evaluation_wall_runtime": 0.0}
            if _final_certificate(result):
                _write_status(status_path, identity, "running", "post_evaluation", "not_yet_certified")
                post, timings = dependencies.post_evaluate(
                    config, dependencies.deserialize_instance(serialized), result, cell, identity,
                    run_root / "post_evaluation",
                )
            expected = 1831 if row["scale"] == "medium_large" else 4657
            scientific = classify_status(result, post, expected)
            result = deepcopy(result)
            result["post_evaluation"] = post
            result.update(timings)
            result["algorithm_runtime"] = _strict_number(result.get("runtime", result.get("algorithm_runtime")), "runtime")
            result["penalized_runtime_par2"] = result["algorithm_runtime"] if scientific == "certified_robust_optimal" else 3600.0
            record = {**identity, "identity": identity, "state": "complete", "scientific_status": scientific, "result": result}
            atomic_write_json(run_path, record)
            _write_status(status_path, identity, "complete", "complete", scientific)
            aggregate(output, rows, cells)
        except KeyboardInterrupt:
            _write_status(status_path, identity, "interrupted", "unknown", "not_yet_certified")
            raise
    results = aggregate(output, rows, cells)
    if len(results) != 10:
        raise BenchmarkGateError("final aggregation is incomplete")
    if file_sha256(source_zip).upper() != SOURCE_ZIP_SHA256:
        raise BenchmarkGateError("source ZIP changed during execution")
    return {"output": str(output), "completed": 10, "certified_solved": sum(r["scientific_status"] == "certified_robust_optimal" for r in results)}


def validate_authorization(path: str | Path, config_path: Path, config: dict[str, Any], root: Path) -> tuple[dict[str, Any], str]:
    authorization = load_json(path)
    expected = {
        "schema_version": 1, "formal_run_authorized": True, "stage": STAGE,
        "source_zip_sha256": SOURCE_ZIP_SHA256, "reference_candidate": CANDIDATE,
        "scales": list(SCALES), "seeds": list(SEEDS), "gamma": [2], "rho": [0.025],
        "reference_frontier_count": 10, "baseline_new_count": 0, "hybrid_new_count": 0,
        "execution_attempt": ATTEMPT, "previous_benchmark_results_reused": False,
        "formal_worktree_root": str(config["formal_worktree_root"]),
        "output_relative_path": config["output_relative_path"],
        "config_sha256": file_sha256(config_path).upper(),
        "protocol_sha256": config["protocol_sha256"], "candidate_sha256": config["candidate_sha256"],
    }
    for field, wanted in expected.items():
        if authorization.get(field) != wanted:
            raise BenchmarkGateError(f"authorization {field} mismatch")
    if authorization.get("forbidden") != {
        "new_baseline": True, "hybrid_rerun": True, "gamma_0_or_1": True,
        "other_rho": True, "other_seed": True, "full_grid": True,
        "selective_rerun": True, "mathematical_model_change": True,
    }:
        raise BenchmarkGateError("authorization forbidden scope mismatch")
    if root.resolve() != Path(config["formal_worktree_root"]).resolve():
        raise BenchmarkGateError("formal worktree root mismatch")
    return authorization, file_sha256(path).upper()


def formal_git_gate(root: Path, authorization: dict[str, Any]) -> str:
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise BenchmarkGateError("formal worktree has tracked or unrelated untracked changes")
    implementation = authorization.get("authorized_implementation_commit")
    auth_path = authorization.get("authorization_only_path")
    if not isinstance(implementation, str) or len(implementation) != 40 or not isinstance(auth_path, str):
        raise BenchmarkGateError("authorization implementation identity missing")
    changed = [line for line in _git(root, "diff", "--name-only", implementation, commit).splitlines() if line]
    if changed != [auth_path]:
        raise BenchmarkGateError("current commit is not an authorization-only successor")
    output_root = str(authorization["output_relative_path"]).split("/a1", 1)[0] + "/"
    ignored = subprocess.run(["git", "check-ignore", "-q", output_root], cwd=root, check=False)
    if ignored.returncode != 0:
        raise BenchmarkGateError("formal output root is not ignored")
    return commit


def formal_run(config_path: str | Path, *, resume: bool, authorization_file: str | Path | None) -> dict[str, Any]:
    if not resume:
        raise BenchmarkGateError("formal benchmark requires --resume; --overwrite is unsupported")
    if authorization_file is None:
        raise BenchmarkGateError("authorization file is required")
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    validate_config(config_path, config)
    root = Path(__file__).resolve().parents[1]
    authorization, authorization_sha = validate_authorization(authorization_file, config_path, config, root)
    commit = formal_git_gate(root, authorization)
    source_zip = Path(config["source_zip"])
    if file_sha256(source_zip).upper() != SOURCE_ZIP_SHA256:
        raise BenchmarkGateError("source ZIP SHA mismatch")
    cells = load_catalog(config)
    output = root / config["output_relative_path"]
    if output.exists() and not (output / "manifest.json").exists():
        raise BenchmarkGateError("existing output is not a resumable benchmark root")
    dependencies = production_dependencies()
    result = execute_plan(config_path, config, cells, output, commit=commit, dependencies=dependencies, source_zip=source_zip)
    result["authorization_sha256"] = authorization_sha
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorization-file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.stage != STAGE:
        raise BenchmarkGateError("stage is outside the frozen authorization")
    if args.dry_run:
        print(json.dumps(dry_run(args.config), indent=2, sort_keys=True))
        return 0
    print(json.dumps(formal_run(args.config, resume=args.resume, authorization_file=args.authorization_file), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
