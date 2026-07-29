from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

from .experiment_protocol import atomic_write_json, atomic_write_yaml, config_sha256, file_sha256, git_commit, penalized_runtime_par2, utc_now_iso
from .fairness_hybrid_ccg_benders import (
    CANDIDATE,
    CANDIDATE_SHA256,
    PROTOCOL_SHA256,
    initial_scenario_plan_identity,
    solve_certified_hybrid_scenario_benders_fairness,
)
from .fairness_large_final_remediation_runner import (
    SOLVER_PARAMETERS,
    RemediationGateError,
    _aggregate_records,
    _classify_frontier,
    _configure_solver_parameters,
    _instance_identity,
    _production_generate_instance,
    _production_post_evaluate,
    _production_solve_baseline,
    _read_json_strict,
    _run_root,
    _same_identity,
    _scale_template,
)
from .fairness_scalability_results_audit import resolved_config_file_bytes
from .fairness_scalability_runner import run_directory_id
from .experiment_suite import _base_config
from .instance import InventoryInstance


STAGE = "D1"
SCHEMA = "fairness_hybrid_ccg_benders_manifest_v1"
EXECUTION_ATTEMPT = 1
EXPECTED_CONFIG_SHA256 = "95514DD43167583CCE8D09A2C9491FF8892A0ACE4B800DC9B0F8CD879B5C7156"
OUTPUT_RELATIVE = "experiments/results_fairness_hybrid_ccg_benders/development_d1_large_seed160_rho0"
ATTEMPT5_SHA256 = "09B41862A5BFED724EDBEC1E64996B54AA878119F5C0DEDFE5B10126B2525A98"
ATTEMPT5_DECISION_SHA256 = "23E2835EC803A7D14367A6EAE246E31A116982213BB0BD2EEB3C6F41F798FD53"
ATTEMPT5_PROVENANCE_SHA256 = "841151684AEECF50B486E42C70DF90229E30998CC84F956B72E8F655E84BE11B"
RESOLVED_CONFIG_CANONICALIZATION = "PyYAML safe_dump(sort_keys=True, allow_unicode=True), UTF-8"


@dataclass
class HybridDependencies:
    generate_instance: Callable[[dict[str, Any], int], InventoryInstance]
    solve_baseline: Callable[..., dict[str, Any]]
    solve_frontier: Callable[..., dict[str, Any]]
    post_evaluate: Callable[..., tuple[dict[str, Any], dict[str, float]]]
    configure_solver: Callable[[dict[str, Any]], None]


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RemediationGateError("D1 config must be a mapping")
    return payload


def _validate_config(path: Path, config: dict[str, Any]) -> None:
    if file_sha256(path).upper() != EXPECTED_CONFIG_SHA256:
        raise RemediationGateError("D1 authorized config SHA mismatch")
    expected = {
        "stage": STAGE,
        "authorization": "formal_execution_authorized",
        "formal_run_authorized": True,
        "schema_version": 1,
        "execution_attempt": EXECUTION_ATTEMPT,
        "previous_attempt_results_reused": False,
        "prior_large_attempt5_archive_sha256": ATTEMPT5_SHA256,
        "prior_large_attempt5_decision": "stop_final_large_remediation",
        "candidate": CANDIDATE,
        "scale": "large",
        "seeds": [160],
        "rho": [0.0],
        "baseline_count": 1,
        "frontier_count": 1,
        "total_tasks": 2,
        "scenario_count": 4657,
        "output_dir": OUTPUT_RELATIVE,
        "baseline_time_limit_seconds": 1800,
        "algorithm_time_limit_seconds": 1800,
        "general_time_limit_seconds": 1800,
        "solver_identity": SOLVER_PARAMETERS,
        "gamma": 2,
        "tol": 1.0e-4,
        "post_evaluation": {"time_limit_per_scenario_seconds": 30, "checkpoint_chunk_size": 25, "scenario_count": 4657},
        "runtime_semantics": {"par2_multiplier": 2, "par2_basis": "algorithm_runtime"},
        "resume": True,
        "overwrite_supported": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RemediationGateError(f"D1 identity drifted: {key}")
    root = Path(__file__).resolve().parents[1]
    protocol = root / str(config["protocol_document"])
    candidate = root / str(config["candidate_definition"])
    if file_sha256(protocol).upper() != PROTOCOL_SHA256 or config.get("required_protocol_sha256") != PROTOCOL_SHA256:
        raise RemediationGateError("D1 protocol identity mismatch")
    if file_sha256(candidate).upper() != CANDIDATE_SHA256 or config.get("required_candidate_sha256") != CANDIDATE_SHA256:
        raise RemediationGateError("D1 candidate identity mismatch")
    decision_path = root / "analysis/fairness_hybrid_ccg_benders_d1_freeze/large_attempt5_stop_decision.json"
    provenance_path = root / "analysis/fairness_hybrid_ccg_benders_d1_freeze/source_archive_provenance.json"
    if file_sha256(decision_path).upper() != ATTEMPT5_DECISION_SHA256:
        raise RemediationGateError("Attempt 5 stop decision identity mismatch")
    if file_sha256(provenance_path).upper() != ATTEMPT5_PROVENANCE_SHA256:
        raise RemediationGateError("Attempt 5 provenance identity mismatch")
    decision = _read_json_strict(decision_path, label="Attempt 5 stop decision")
    provenance = _read_json_strict(provenance_path, label="Attempt 5 provenance")
    if decision is None or decision.get("decision") != "stop_final_large_remediation":
        raise RemediationGateError("Attempt 5 stop decision missing")
    if provenance is None or provenance.get("archive_sha256") != ATTEMPT5_SHA256:
        raise RemediationGateError("Attempt 5 archive provenance mismatch")


def expand_d1_plan() -> list[dict[str, Any]]:
    rows = []
    for task_type, candidate, rho in (("baseline", "baseline", "NOT_APPLICABLE"), ("frontier", CANDIDATE, "0.00")):
        key = json.dumps({
            "candidate": candidate, "execution_attempt": EXECUTION_ATTEMPT,
            "rho": rho, "scale": "large", "seed": 160, "task_type": task_type,
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        rows.append({
            "stage": STAGE, "scale": "large", "seed": 160, "rho": rho,
            "candidate": candidate, "task_type": task_type, "run_key": key,
            "run_directory_id": run_directory_id(key),
        })
    return rows


def _planned_paths(root: Path, output: Path, rows: list[dict[str, Any]]) -> list[Path]:
    paths = [output / name for name in (
        "manifest.json", ".manifest.json.tmp", "resolved_config.yaml", ".resolved_config.yaml.tmp",
        "results.csv", ".results.csv.tmp", "summary.csv", ".summary.csv.tmp", "audit.log",
        "instances/160.json", "instances/.160.json.tmp",
    )]
    for row in rows:
        run = output / "runs" / row["run_directory_id"]
        paths.extend(run / name for name in (
            "run.json", ".run.json.tmp", "status.json", ".status.json.tmp",
            "algorithm_checkpoint.json", ".algorithm_checkpoint.json.tmp",
            "post_evaluation/index.json", "post_evaluation/.index.json.tmp",
            "post_evaluation/checkpoint/chunk_00186.json", "post_evaluation/checkpoint/.chunk_00186.json.tmp",
        ))
    return [(root / path).resolve() if not path.is_absolute() else path.resolve() for path in paths]


def dry_run(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = load_config(path)
    _validate_config(path, config)
    rows = expand_d1_plan()
    root = Path(__file__).resolve().parents[1]
    output = root / str(config["output_dir"])
    paths = _planned_paths(root, Path(str(config["output_dir"])), rows)
    longest = max(paths, key=lambda item: len(str(item)))
    template = _base_config(_scale_template(config), "large", 160)
    plan = initial_scenario_plan_identity(
        num_regions=int(template["instance"]["num_regions"]),
        num_products=int(template["instance"]["num_products"]),
        gamma=int(config["gamma"]),
    )
    return {
        "stage": STAGE,
        "baseline": 1,
        "frontier": 1,
        "total": 2,
        "scale": "large",
        "seed": 160,
        "rho": 0.0,
        "candidate": CANDIDATE,
        "uncertainty_scenarios": 4657,
        **plan,
        "instances_generated": False,
        "solver_called": False,
        "output_dir_exists": output.exists(),
        "windows_path_check": len(str(longest)) < 220,
        "longest_path": str(longest),
        "longest_path_length": len(str(longest)),
    }


def _identity(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    resolved = resolved_config_file_bytes(config)
    return {
        "schema": SCHEMA,
        "execution_attempt": EXECUTION_ATTEMPT,
        "stage": STAGE,
        "git_commit": git_commit(root),
        "config_file_sha256": file_sha256(config_path).upper(),
        "resolved_config_file_sha256": hashlib.sha256(resolved).hexdigest().upper(),
        "resolved_config_canonical_sha256": config_sha256(config).upper(),
        "resolved_config_canonicalization": RESOLVED_CONFIG_CANONICALIZATION,
        "protocol_sha256": PROTOCOL_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "candidate": CANDIDATE,
        "solver_parameters": deepcopy(SOLVER_PARAMETERS),
        "scale": "large", "seeds": [160], "rhos": [0.0],
        "previous_attempt_results_reused": False,
        "prior_large_attempt5_archive_sha256": ATTEMPT5_SHA256,
    }


def _baseline_checkpoint_payload(identity: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    body = {"schema": "fairness_hybrid_baseline_checkpoint_v1", "identity": deepcopy(identity), "result": deepcopy(result)}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    body["checkpoint_sha256"] = hashlib.sha256(encoded).hexdigest().upper()
    return body


def _validate_baseline_checkpoint(payload: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "fairness_hybrid_baseline_checkpoint_v1" or payload.get("identity") != identity:
        raise RemediationGateError("D1 baseline checkpoint identity mismatch")
    unhashed = {key: deepcopy(value) for key, value in payload.items() if key != "checkpoint_sha256"}
    encoded = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if payload.get("checkpoint_sha256") != hashlib.sha256(encoded).hexdigest().upper():
        raise RemediationGateError("D1 baseline checkpoint hash mismatch")
    if not isinstance(payload.get("result"), dict):
        raise RemediationGateError("D1 baseline checkpoint result missing")
    return deepcopy(payload["result"])


def _hybrid_certified_anchor(
    baseline_record: dict[str, Any], *, common_identity: dict[str, Any], tolerance: float,
) -> dict[str, Any]:
    result = baseline_record.get("result")
    upper = None if not isinstance(result, dict) else result.get("upper_bound")
    gap = None if not isinstance(result, dict) else result.get("gap")
    if not (
        isinstance(result, dict)
        and baseline_record.get("scientific_status") == "certified_robust_optimal"
        and baseline_record.get("solved_to_tolerance") is True
        and result.get("status") == "optimal" and result.get("valid_UB") is True
        and upper is not None and math.isfinite(float(upper))
        and gap is not None and math.isfinite(float(gap)) and float(gap) <= tolerance
    ):
        raise RemediationGateError("certified D1 baseline anchor unavailable")
    payload = {
        "source": "solve_result.upper_bound", "value": float(upper),
        "value_hex": float(upper).hex(), "anchor_value_hex": float(upper).hex(),
        "base_git_commit": common_identity["git_commit"],
        "base_config_sha256": common_identity["resolved_config_file_sha256"],
        "candidate_config_sha256": CANDIDATE_SHA256,
        "valid_UB": True, "baseline_status": "optimal", "baseline_final_gap": float(gap),
        **deepcopy(common_identity),
    }
    payload["anchor_sha256"] = config_sha256(payload).upper()
    return payload


def _production_frontier(
    config: dict[str, Any], instance: InventoryInstance, baseline: dict[str, Any], anchor: dict[str, Any],
    expected: dict[str, Any], checkpoint: Path, solver: dict[str, Any], row: dict[str, Any],
) -> dict[str, Any]:
    result = solve_certified_hybrid_scenario_benders_fairness(
        instance, baseline_record=baseline, anchor=anchor, expected_identity=expected,
        solver_parameters=solver, rho=0.0, gamma=2,
        max_iterations=int(config["max_iterations"]), time_limit=1800.0, tol=1.0e-4,
        feasibility_tolerance=1.0e-7, checkpoint_path=checkpoint,
        checkpoint_identity={"run_key": row["run_key"]}, output_flag=False,
    )
    return result.to_dict()


def production_dependencies() -> HybridDependencies:
    return HybridDependencies(
        generate_instance=_production_generate_instance,
        solve_baseline=_production_solve_baseline,
        solve_frontier=_production_frontier,
        post_evaluate=_production_post_evaluate,
        configure_solver=_configure_solver_parameters,
    )


def run_d1(
    config_path: str | Path, *, resume: bool, dependencies: HybridDependencies | None = None,
    test_authorization: bool = False, failure_injector: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not resume:
        raise RemediationGateError("D1 requires --resume; --overwrite is unsupported")
    path = Path(config_path)
    config = load_config(path)
    _validate_config(path, config)
    if dependencies is not None and not test_authorization:
        raise RemediationGateError("dependency substitution requires test_authorization")
    deps = dependencies or production_dependencies()
    root = Path(__file__).resolve().parents[1]
    output = root / str(config["output_dir"])
    rows = expand_d1_plan()
    paths = _planned_paths(root, Path(str(config["output_dir"])), rows)
    if max(len(str(item)) for item in paths) >= 220:
        raise RemediationGateError("D1 Windows path preflight failed")
    identity = _identity(path, config)
    manifest_path = output / "manifest.json"
    existing = _read_json_strict(manifest_path, label="D1 manifest")
    if output.exists() and existing is None:
        raise RemediationGateError("existing D1 output lacks a valid manifest")
    if existing is not None and existing.get("identity") != identity:
        raise RemediationGateError("D1 resume identity mismatch")
    if existing is None:
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_yaml(output / "resolved_config.yaml", config)
    resolved_path = output / "resolved_config.yaml"
    if not resolved_path.is_file() or file_sha256(resolved_path).upper() != identity["resolved_config_file_sha256"]:
        raise RemediationGateError("resolved config file identity mismatch")
    manifest = existing or {
        "schema": SCHEMA, "identity": identity,
        "run_key_to_directory_id": {row["run_key"]: row["run_directory_id"] for row in rows},
        "directory_id_to_run_key": {row["run_directory_id"]: row["run_key"] for row in rows},
        "baseline_anchors": {}, "run_identities": {}, "created_at": utc_now_iso(),
    }
    atomic_write_json(manifest_path, manifest)
    deps.configure_solver(deepcopy(SOLVER_PARAMETERS))
    instance_path = output / "instances/160.json"
    stored = _read_json_strict(instance_path, label="D1 instance")
    if stored is None:
        instance = deps.generate_instance(config, 160)
        atomic_write_json(instance_path, instance.to_dict())
    else:
        instance = InventoryInstance.from_dict(stored)
    instance_sha = _instance_identity(instance)
    baseline_row, frontier_row = rows
    baseline_root = _run_root(output, baseline_row)
    baseline = _read_json_strict(baseline_root / "run.json", label="D1 baseline")
    common = {
        "instance_sha256": instance_sha, "seed": 160, "scale": "large",
        "git_commit": identity["git_commit"], "config_file_sha256": identity["config_file_sha256"],
        "resolved_config_file_sha256": identity["resolved_config_file_sha256"],
        "candidate_sha256": CANDIDATE_SHA256, "baseline_run_key": baseline_row["run_key"],
    }
    if baseline is None:
        baseline_checkpoint_path = baseline_root / "baseline_checkpoint.json"
        baseline_checkpoint = _read_json_strict(baseline_checkpoint_path, label="D1 baseline checkpoint")
        if baseline_checkpoint is not None:
            payload = _validate_baseline_checkpoint(baseline_checkpoint, common)
        else:
            payload = deps.solve_baseline(config, instance, 160, deepcopy(SOLVER_PARAMETERS))
            atomic_write_json(baseline_checkpoint_path, _baseline_checkpoint_payload(common, payload))
            if failure_injector:
                failure_injector("after_baseline_checkpoint", deepcopy(baseline_row))
        solved = payload.get("status") == "optimal" and payload.get("valid_UB") is True and float(payload.get("gap", math.inf)) <= 1.0e-4
        runtime = float(payload.get("runtime", 0.0))
        payload.update({"algorithm_runtime": runtime, "post_evaluation_wall_runtime": 0.0, "total_wall_runtime": runtime,
                        "penalized_runtime_par2": penalized_runtime_par2(solved_to_tolerance=solved, runtime=runtime, time_limit=1800.0)})
        baseline = {**common, "run_key": baseline_row["run_key"], "run_directory_id": baseline_row["run_directory_id"],
                    "stage": STAGE, "task_type": "baseline", "rho": "NOT_APPLICABLE", "candidate": "baseline",
                    "state": "complete", "algorithm_status": payload.get("status"),
                    "scientific_status": "certified_robust_optimal" if solved else "master_optimal_but_robust_uncertified",
                    "solved_to_tolerance": solved, "result": payload}
        atomic_write_json(baseline_root / "run.json", baseline)
        atomic_write_json(baseline_root / "status.json", {"state": "complete", "scientific_status": baseline["scientific_status"]})
    elif not _same_identity(baseline, common, tuple(common)):
        raise RemediationGateError("D1 baseline resume identity mismatch")
    anchor = _hybrid_certified_anchor(baseline, common_identity=common, tolerance=1.0e-4)
    manifest["baseline_anchors"]["160"] = anchor
    frontier_root = _run_root(output, frontier_row)
    record = _read_json_strict(frontier_root / "run.json", label="D1 frontier")
    run_identity = {**common, "run_key": frontier_row["run_key"], "run_directory_id": frontier_row["run_directory_id"],
                    "rho": 0.0, "anchor_sha256": anchor["anchor_sha256"], "anchor_value_hex": anchor["value_hex"],
                    "checkpoint_schema": "fairness_hybrid_ccg_benders_checkpoint_v1", "post_evaluation_schema": "fairness_post_evaluation_checkpoint_v1"}
    manifest["run_identities"][frontier_row["run_key"]] = run_identity
    if record is None:
        expected = {**{key: common[key] for key in common}, "anchor_value_hex": anchor["value_hex"], "anchor_sha256": anchor["anchor_sha256"]}
        tick = time.perf_counter()
        try:
            result = deps.solve_frontier(config, instance, baseline, anchor, expected, frontier_root / "algorithm_checkpoint.json", deepcopy(SOLVER_PARAMETERS), frontier_row)
        except KeyboardInterrupt:
            atomic_write_json(frontier_root / "status.json", {"state": "interrupted", "scientific_status": "interrupted", "algorithm_status": "interrupted"})
            raise
        except Exception as exc:
            failure = {**run_identity, "stage": STAGE, "task_type": "frontier", "candidate": CANDIDATE,
                       "state": "complete", "algorithm_status": "exception", "scientific_status": "implementation_error",
                       "solved_to_tolerance": False, "failure_reason": str(exc),
                       "result": {"status": "exception", "algorithm_runtime": time.perf_counter() - tick,
                                  "penalized_runtime_par2": 3600.0}}
            atomic_write_json(frontier_root / "run.json", failure)
            atomic_write_json(frontier_root / "status.json", {"state": "complete", "scientific_status": "implementation_error", "algorithm_status": "exception"})
            atomic_write_json(manifest_path, manifest)
            _aggregate_records(output, rows)
            raise
        algorithm_runtime = float(result.get("runtime", time.perf_counter() - tick))
        algorithm_certified = result.get("status") == "optimal" and result.get("gap") is not None and float(result["gap"]) <= 1.0e-4 and result.get("metadata", {}).get("robust_feasibility_certified") is True
        evaluation = None
        timing = {"post_evaluation_solver_runtime": 0.0, "post_evaluation_wall_runtime": 0.0, "aggregation_runtime": 0.0, "checkpoint_io_runtime": 0.0}
        if algorithm_certified:
            evaluation, timing = deps.post_evaluate(config, instance, result, anchor, run_identity, frontier_root / "post_evaluation", frontier_row)
        scientific = _classify_frontier(result, evaluation, tolerance=1.0e-4, expected_scenario_count=4657)
        result.update(timing)
        result.update({"post_evaluation": evaluation, "algorithm_runtime": algorithm_runtime,
                       "post_evaluation_wall_runtime": float(timing.get("post_evaluation_wall_runtime", 0.0)),
                       "penalized_runtime_par2": penalized_runtime_par2(solved_to_tolerance=scientific == "certified_robust_optimal", runtime=algorithm_runtime, time_limit=1800.0)})
        result["total_wall_runtime"] = algorithm_runtime + result["post_evaluation_wall_runtime"] + float(timing.get("aggregation_runtime", 0.0)) + float(timing.get("checkpoint_io_runtime", 0.0))
        record = {**run_identity, "stage": STAGE, "task_type": "frontier", "candidate": CANDIDATE,
                  "state": "complete", "algorithm_status": result.get("status"), "scientific_status": scientific,
                  "solved_to_tolerance": scientific == "certified_robust_optimal", "result": result}
        atomic_write_json(frontier_root / "run.json", record)
        atomic_write_json(frontier_root / "status.json", {"state": "complete", "scientific_status": scientific, "algorithm_status": result.get("status")})
    elif not _same_identity(record, run_identity, tuple(run_identity)):
        raise RemediationGateError("D1 frontier resume identity mismatch")
    atomic_write_json(manifest_path, manifest)
    _aggregate_records(output, rows)
    manifest["completed_run_count"] = 2
    manifest["certified_solved_count"] = sum(item.get("scientific_status") == "certified_robust_optimal" for item in (baseline, record))
    manifest["updated_at"] = utc_now_iso()
    atomic_write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage != STAGE:
        raise RemediationGateError("only D1 is authorized")
    if args.dry_run:
        print(json.dumps(dry_run(args.config), indent=2, sort_keys=True))
        return 0
    manifest = run_d1(args.config, resume=args.resume)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
