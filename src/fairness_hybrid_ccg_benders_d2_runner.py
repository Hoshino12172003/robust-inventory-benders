from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from .experiment_protocol import (
    atomic_write_json,
    atomic_write_yaml,
    config_sha256,
    file_sha256,
    git_commit,
    penalized_runtime_par2,
    utc_now_iso,
)
from .fairness_hybrid_ccg_benders import (
    CANDIDATE,
    CANDIDATE_SHA256,
    initial_upper_bound_expected_identity,
    initial_scenario_plan_identity,
    solve_certified_hybrid_scenario_benders_fairness,
)
from .fairness_hybrid_ccg_benders_runner import (
    HybridDependencies,
    _baseline_checkpoint_payload,
    _hybrid_certified_anchor,
    _validate_baseline_checkpoint,
    load_config,
)
from .fairness_large_final_remediation_runner import (
    SOLVER_PARAMETERS,
    RemediationGateError,
    _aggregate_records,
    _classify_frontier,
    _configure_solver_parameters,
    _instance_identity,
    _production_generate_instance,
    _production_solve_baseline,
    _read_json_strict,
    _run_root,
    _same_identity,
    _scale_template,
)
from .fairness_post_evaluation import checkpointed_fairness_post_evaluation
from .fairness_scalability_results_audit import resolved_config_file_bytes
from .fairness_scalability_runner import run_directory_id
from .experiment_suite import _base_config
from .instance import InventoryInstance


STAGE = "D2"
SCHEMA = "fairness_hybrid_ccg_benders_manifest_v2"
EXECUTION_ATTEMPT = 3
EXPECTED_CONFIG_SHA256 = "ED8F145A9ACAA1AC799DBBDE2BAEBF1A35F2F614FE41B0F73EF8F278690EF63A"
PROTOCOL_SHA256 = "A1D1655F4D66B79ADB9AF28E69F8E04D50F0EAEFB8577F645080D5713D1426BC"
D1_DECISION_SHA256 = "1F7101CB722C4A4E6974C2D8597F4ED37BF89C2072FA039502EC69E834C7F17E"
D1_ARCHIVE_SHA256 = "7E89115E3BE325C9A37C31D28D32EA80EEA95F09528DBC8AEDA833EF0129A4A9"
OUTPUT_RELATIVE = "experiments/results_fairness_hybrid_ccg_benders/controlled_d2_a3_large_s160_162_r0_001_010"
SEEDS = [160, 161, 162]
RHOS = [0.0, 0.01, 0.10]
RESOLVED_CONFIG_CANONICALIZATION = "PyYAML safe_dump(sort_keys=True, allow_unicode=True), UTF-8"


def _validate_config(path: Path, config: dict[str, Any]) -> None:
    if file_sha256(path).upper() != EXPECTED_CONFIG_SHA256:
        raise RemediationGateError("D2 authorized config SHA mismatch")
    expected = {
        "stage": STAGE,
        "authorization": "formal_execution_authorized_after_merge",
        "formal_run_authorized": True,
        "schema_version": 2,
        "execution_attempt": EXECUTION_ATTEMPT,
        "previous_attempt_results_reused": False,
        "d1_decision": "approve_for_d2_controlled_large_expansion",
        "required_d1_decision_sha256": D1_DECISION_SHA256,
        "prior_d1_archive_sha256": D1_ARCHIVE_SHA256,
        "required_protocol_sha256": PROTOCOL_SHA256,
        "required_candidate_sha256": CANDIDATE_SHA256,
        "candidate": CANDIDATE,
        "scale": "large",
        "seeds": SEEDS,
        "rho": RHOS,
        "baseline_count": 3,
        "frontier_count": 9,
        "total_tasks": 12,
        "scenario_count": 4657,
        "output_dir": OUTPUT_RELATIVE,
        "baseline_time_limit_seconds": 1800,
        "algorithm_time_limit_seconds": 1800,
        "general_time_limit_seconds": 1800,
        "solver_identity": SOLVER_PARAMETERS,
        "gamma": 2,
        "tol": 1.0e-4,
        "post_evaluation": {
            "time_limit_per_scenario_seconds": 30,
            "checkpoint_chunk_size": 25,
            "scenario_count": 4657,
            "pipeline_generation": 4,
        },
        "runtime_semantics": {"par2_multiplier": 2, "par2_basis": "algorithm_runtime"},
        "resume": True,
        "overwrite_supported": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RemediationGateError(f"D2 identity drifted: {key}")
    root = Path(__file__).resolve().parents[1]
    protocol = root / str(config["protocol_document"])
    candidate = root / str(config["candidate_definition"])
    decision = root / str(config["d1_decision_file"])
    if file_sha256(protocol).upper() != PROTOCOL_SHA256:
        raise RemediationGateError("D2 protocol identity mismatch")
    if file_sha256(candidate).upper() != CANDIDATE_SHA256:
        raise RemediationGateError("D2 candidate identity mismatch")
    if file_sha256(decision).upper() != D1_DECISION_SHA256:
        raise RemediationGateError("D1 decision identity mismatch")
    payload = _read_json_strict(decision, label="D1 decision")
    if payload is None or payload.get("decision") != "approve_for_d2_controlled_large_expansion":
        raise RemediationGateError("D1 decision does not authorize D2")


def expand_d2_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        tasks = [("baseline", "baseline", "NOT_APPLICABLE")]
        tasks.extend(("frontier", CANDIDATE, f"{rho:.2f}") for rho in RHOS)
        for task_type, candidate, rho in tasks:
            key = json.dumps(
                {
                    "candidate": candidate,
                    "execution_attempt": EXECUTION_ATTEMPT,
                    "rho": rho,
                    "scale": "large",
                    "seed": seed,
                    "task_type": task_type,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            rows.append(
                {
                    "stage": STAGE,
                    "scale": "large",
                    "seed": seed,
                    "rho": rho,
                    "candidate": candidate,
                    "task_type": task_type,
                    "run_key": key,
                    "run_directory_id": run_directory_id(key),
                }
            )
    return rows


def _planned_paths(root: Path, output: Path, rows: list[dict[str, Any]]) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for name in (
        "manifest.json", ".manifest.json.tmp", "resolved_config.yaml",
        ".resolved_config.yaml.tmp", "results.csv", ".results.csv.tmp",
        "summary.csv", ".summary.csv.tmp", "audit.log",
    ):
        paths.append(("root_artifact", output / name))
    for seed in SEEDS:
        paths.extend(
            [("instance", output / f"instances/{seed}.json"),
             ("instance_tmp", output / f"instances/.{seed}.json.tmp")]
        )
    for row in rows:
        run = output / "runs" / row["run_directory_id"]
        for kind, name in (
            ("run", "run.json"), ("run_tmp", ".run.json.tmp"),
            ("status", "status.json"), ("status_tmp", ".status.json.tmp"),
            ("algorithm_checkpoint", "algorithm_checkpoint.json"),
            ("algorithm_checkpoint_tmp", ".algorithm_checkpoint.json.tmp"),
            ("post_index", "post_evaluation/checkpoint/index.json"),
            ("post_index_tmp", "post_evaluation/checkpoint/.index.json.tmp"),
            ("post_chunk", "post_evaluation/checkpoint/chunk_00186.json"),
            ("post_chunk_tmp", "post_evaluation/checkpoint/.chunk_00186.json.tmp"),
        ):
            paths.append((kind, run / name))
    return [
        (kind, (root / path).resolve() if not path.is_absolute() else path.resolve())
        for kind, path in paths
    ]


def dry_run(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = load_config(path)
    _validate_config(path, config)
    rows = expand_d2_plan()
    root = Path(__file__).resolve().parents[1]
    output = root / str(config["output_dir"])
    planned = _planned_paths(root, output, rows)
    kind, longest = max(planned, key=lambda item: len(str(item[1])))
    template = _base_config(_scale_template(config), "large", 160)
    initial = initial_scenario_plan_identity(
        num_regions=int(template["instance"]["num_regions"]),
        num_products=int(template["instance"]["num_products"]),
        gamma=2,
    )
    return {
        "stage": STAGE,
        "scale": "large",
        "seeds": SEEDS,
        "rho": RHOS,
        "candidate": CANDIDATE,
        "baseline": 3,
        "frontier": 9,
        "total": 12,
        "unique_run_keys": len({row["run_key"] for row in rows}),
        "duplicate_run_keys": len(rows) - len({row["run_key"] for row in rows}),
        "uncertainty_scenarios": 4657,
        **initial,
        "instances_generated": False,
        "solver_called": False,
        "output_dir_exists": output.exists(),
        "windows_path_check": len(str(longest)) < 220,
        "longest_path_type": kind,
        "longest_path": str(longest),
        "longest_path_length": len(str(longest)),
        "baseline_solver_limit_envelope_seconds": 5400,
        "frontier_algorithm_limit_envelope_seconds": 16200,
        "combined_algorithm_limit_envelope_seconds": 21600,
        "post_evaluation_solver_limit_envelope_seconds": 1257390,
    }


def _git_output(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _formal_git_gate(root: Path) -> None:
    status = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    head = _git_output(root, "rev-parse", "HEAD")
    main = _git_output(root, "rev-parse", "origin/main")
    symbolic = _git_output(root, "symbolic-ref", "-q", "HEAD")
    if status.returncode or status.stdout.strip():
        raise RemediationGateError("formal_run_not_authorized: worktree is not clean")
    if head.returncode or main.returncode or head.stdout.strip() != main.stdout.strip():
        raise RemediationGateError("formal_run_not_authorized: HEAD is not current origin/main")
    if symbolic.returncode == 0:
        raise RemediationGateError("formal_run_not_authorized: worktree must be detached")


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
        "d1_decision_sha256": D1_DECISION_SHA256,
        "candidate": CANDIDATE,
        "solver_parameters": deepcopy(SOLVER_PARAMETERS),
        "scale": "large",
        "seeds": SEEDS,
        "rhos": RHOS,
        "previous_attempt_results_reused": False,
    }


def _production_frontier(
    config: dict[str, Any], instance: InventoryInstance, baseline: dict[str, Any],
    anchor: dict[str, Any], expected: dict[str, Any], checkpoint: Path,
    solver: dict[str, Any], row: dict[str, Any],
) -> dict[str, Any]:
    result = solve_certified_hybrid_scenario_benders_fairness(
        instance,
        baseline_record=baseline,
        anchor=anchor,
        expected_identity=expected,
        solver_parameters=solver,
        rho=float(row["rho"]),
        gamma=2,
        max_iterations=int(config["max_iterations"]),
        time_limit=float(config["algorithm_time_limit_seconds"]),
        tol=float(config["tol"]),
        feasibility_tolerance=float(SOLVER_PARAMETERS["FeasibilityTol"]),
        checkpoint_path=checkpoint,
        checkpoint_identity={"run_key": row["run_key"]},
        execution_protocol_sha256=PROTOCOL_SHA256,
        output_flag=False,
    )
    return result.to_dict()


def _production_post_evaluate_d2(
    config: dict[str, Any], instance: InventoryInstance, result: dict[str, Any],
    anchor: dict[str, Any], run_identity: dict[str, Any], root: Path,
    row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    evaluation, timing = checkpointed_fairness_post_evaluation(
        instance,
        root=root,
        run_key=row["run_key"],
        config_sha256_value=run_identity["resolved_config_file_sha256"],
        git_commit=run_identity["git_commit"],
        baseline_anchor_sha256=anchor["anchor_sha256"],
        y_values=result["y_values"],
        x_values=result["x_values"],
        t_value=float(result["objective_t"]),
        baseline_cost=float(anchor["value"]),
        rho=float(row["rho"]),
        gamma=2,
        max_scenarios=4657,
        per_scenario_time_limit=30.0,
        tolerance=1e-7,
        chunk_size=25,
        resume_count=0,
        output_flag=False,
        run_execution_attempt=EXECUTION_ATTEMPT,
        post_evaluation_pipeline_generation=4,
    )
    return evaluation.to_dict(), {
        "post_evaluation_solver_runtime": timing.solver_runtime,
        "post_evaluation_wall_runtime": timing.wall_runtime,
        "aggregation_runtime": timing.aggregation_runtime,
        "checkpoint_io_runtime": timing.checkpoint_io_runtime,
    }


def production_dependencies() -> HybridDependencies:
    return HybridDependencies(
        generate_instance=_production_generate_instance,
        solve_baseline=_production_solve_baseline,
        solve_frontier=_production_frontier,
        post_evaluate=_production_post_evaluate_d2,
        configure_solver=_configure_solver_parameters,
    )


def _write_status(root: Path, *, state: str, scientific: str, algorithm: str) -> None:
    atomic_write_json(
        root / "status.json",
        {"state": state, "scientific_status": scientific, "algorithm_status": algorithm},
    )


def run_d2(
    config_path: str | Path,
    *,
    resume: bool,
    dependencies: HybridDependencies | None = None,
    test_authorization: bool = False,
    failure_injector: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not resume:
        raise RemediationGateError("D2 requires --resume; --overwrite is unsupported")
    path = Path(config_path)
    config = load_config(path)
    _validate_config(path, config)
    if dependencies is not None and not test_authorization:
        raise RemediationGateError("dependency substitution requires test_authorization")
    root = Path(__file__).resolve().parents[1]
    if not test_authorization:
        _formal_git_gate(root)
    deps = dependencies or production_dependencies()
    output = root / str(config["output_dir"])
    rows = expand_d2_plan()
    planned = _planned_paths(root, output, rows)
    if max(len(str(path_value)) for _, path_value in planned) >= 220:
        raise RemediationGateError("D2 Windows path preflight failed")
    identity = _identity(path, config)
    manifest_path = output / "manifest.json"
    existing = _read_json_strict(manifest_path, label="D2 manifest")
    if output.exists() and existing is None:
        raise RemediationGateError("existing D2 output lacks a valid manifest")
    if existing is not None and existing.get("identity") != identity:
        raise RemediationGateError("D2 resume identity mismatch")
    if existing is None:
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_yaml(output / "resolved_config.yaml", config)
    if file_sha256(output / "resolved_config.yaml").upper() != identity["resolved_config_file_sha256"]:
        raise RemediationGateError("D2 resolved config identity mismatch")
    manifest = existing or {
        "schema": SCHEMA,
        "identity": identity,
        "run_key_to_directory_id": {row["run_key"]: row["run_directory_id"] for row in rows},
        "directory_id_to_run_key": {row["run_directory_id"]: row["run_key"] for row in rows},
        "baseline_anchors": {},
        "run_identities": {},
        "created_at": utc_now_iso(),
    }
    atomic_write_json(manifest_path, manifest)
    deps.configure_solver(deepcopy(SOLVER_PARAMETERS))

    records_by_key: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        instance_path = output / f"instances/{seed}.json"
        stored = _read_json_strict(instance_path, label=f"D2 instance {seed}")
        if stored is None:
            instance = deps.generate_instance(config, seed)
            atomic_write_json(instance_path, instance.to_dict())
        else:
            instance = InventoryInstance.from_dict(stored)
        instance_sha = _instance_identity(instance)
        seed_rows = [row for row in rows if row["seed"] == seed]
        baseline_row = next(row for row in seed_rows if row["task_type"] == "baseline")
        baseline_root = _run_root(output, baseline_row)
        common = {
            "instance_sha256": instance_sha,
            "seed": seed,
            "scale": "large",
            "stage": STAGE,
            "execution_attempt": EXECUTION_ATTEMPT,
            "git_commit": identity["git_commit"],
            "config_file_sha256": identity["config_file_sha256"],
            "resolved_config_file_sha256": identity["resolved_config_file_sha256"],
            "resolved_config_canonical_sha256": identity["resolved_config_canonical_sha256"],
            "protocol_sha256": PROTOCOL_SHA256,
            "candidate_sha256": CANDIDATE_SHA256,
            "baseline_run_key": baseline_row["run_key"],
        }
        baseline = _read_json_strict(baseline_root / "run.json", label=f"D2 baseline {seed}")
        if baseline is None:
            checkpoint_path = baseline_root / "baseline_checkpoint.json"
            checkpoint = _read_json_strict(checkpoint_path, label=f"D2 baseline checkpoint {seed}")
            if checkpoint is not None:
                payload = _validate_baseline_checkpoint(checkpoint, common)
            else:
                _write_status(baseline_root, state="running", scientific="pending", algorithm="running")
                payload = deps.solve_baseline(config, instance, seed, deepcopy(SOLVER_PARAMETERS))
                atomic_write_json(checkpoint_path, _baseline_checkpoint_payload(common, payload))
                if failure_injector:
                    failure_injector("after_baseline_checkpoint", deepcopy(baseline_row))
            solved = (
                payload.get("status") == "optimal"
                and payload.get("valid_UB") is True
                and math.isfinite(float(payload.get("upper_bound", math.nan)))
                and float(payload.get("gap", math.inf)) <= 1e-4
            )
            runtime = float(payload.get("runtime", 0.0))
            payload.update(
                {
                    "algorithm_runtime": runtime,
                    "post_evaluation_wall_runtime": 0.0,
                    "total_wall_runtime": runtime,
                    "penalized_runtime_par2": penalized_runtime_par2(
                        solved_to_tolerance=solved, runtime=runtime, time_limit=1800.0
                    ),
                }
            )
            baseline = {
                **common,
                "run_key": baseline_row["run_key"],
                "run_directory_id": baseline_row["run_directory_id"],
                "stage": STAGE,
                "task_type": "baseline",
                "rho": "NOT_APPLICABLE",
                "candidate": "baseline",
                "state": "complete",
                "algorithm_status": payload.get("status"),
                "scientific_status": "certified_robust_optimal" if solved else "master_optimal_but_robust_uncertified",
                "solved_to_tolerance": solved,
                "result": payload,
            }
            atomic_write_json(baseline_root / "run.json", baseline)
            _write_status(
                baseline_root, state="complete", scientific=baseline["scientific_status"],
                algorithm=str(payload.get("status")),
            )
        elif not _same_identity(baseline, common, tuple(common)):
            raise RemediationGateError("D2 baseline resume identity mismatch")
        records_by_key[baseline["run_key"]] = baseline
        anchor = _hybrid_certified_anchor(baseline, common_identity=common, tolerance=1e-4)
        manifest["baseline_anchors"][str(seed)] = anchor
        atomic_write_json(manifest_path, manifest)

        for row in [value for value in seed_rows if value["task_type"] == "frontier"]:
            run_root = _run_root(output, row)
            run_identity = {
                **common,
                "run_key": row["run_key"],
                "run_directory_id": row["run_directory_id"],
                "rho": float(row["rho"]),
                "anchor_sha256": anchor["anchor_sha256"],
                "anchor_value_hex": anchor["value_hex"],
                "checkpoint_schema": "fairness_hybrid_ccg_benders_checkpoint_v1",
                "post_evaluation_schema": "fairness_post_evaluation_checkpoint_v1",
                "post_evaluation_run_execution_attempt": EXECUTION_ATTEMPT,
                "post_evaluation_pipeline_generation": 4,
            }
            manifest["run_identities"][row["run_key"]] = run_identity
            atomic_write_json(manifest_path, manifest)
            record = _read_json_strict(run_root / "run.json", label=f"D2 frontier {seed}/{row['rho']}")
            if record is not None:
                if not _same_identity(record, run_identity, tuple(run_identity)):
                    raise RemediationGateError("D2 frontier resume identity mismatch")
                records_by_key[record["run_key"]] = record
                continue
            expected = initial_upper_bound_expected_identity(common, anchor)
            _write_status(run_root, state="running", scientific="pending", algorithm="running")
            started = time.perf_counter()
            try:
                result = deps.solve_frontier(
                    config, instance, baseline, anchor, expected,
                    run_root / "algorithm_checkpoint.json", deepcopy(SOLVER_PARAMETERS), row,
                )
            except KeyboardInterrupt:
                _write_status(run_root, state="interrupted", scientific="interrupted", algorithm="interrupted")
                raise
            except Exception as exc:
                failure = {
                    **run_identity, "stage": STAGE, "task_type": "frontier",
                    "candidate": CANDIDATE, "state": "complete",
                    "algorithm_status": "exception", "scientific_status": "implementation_error",
                    "solved_to_tolerance": False, "failure_reason": str(exc),
                    "result": {"status": "exception", "algorithm_runtime": time.perf_counter() - started,
                               "penalized_runtime_par2": 3600.0},
                }
                atomic_write_json(run_root / "run.json", failure)
                _write_status(run_root, state="complete", scientific="implementation_error", algorithm="exception")
                _aggregate_records(output, rows)
                raise
            algorithm_runtime = float(result.get("runtime", time.perf_counter() - started))
            algorithm_certified = (
                result.get("status") == "optimal"
                and result.get("gap") is not None
                and float(result["gap"]) <= 1e-4
                and result.get("metadata", {}).get("robust_feasibility_certified") is True
                and bool(result.get("iteration_log"))
                and result["iteration_log"][-1].get("final_exact_separation_performed") is True
            )
            evaluation = None
            timing = {
                "post_evaluation_solver_runtime": 0.0,
                "post_evaluation_wall_runtime": 0.0,
                "aggregation_runtime": 0.0,
                "checkpoint_io_runtime": 0.0,
            }
            if algorithm_certified:
                evaluation, timing = deps.post_evaluate(
                    config, instance, result, anchor, run_identity,
                    run_root / "post_evaluation", row,
                )
            scientific = _classify_frontier(
                result, evaluation, tolerance=1e-4, expected_scenario_count=4657
            )
            result.update(timing)
            result.update(
                {
                    "post_evaluation": evaluation,
                    "algorithm_runtime": algorithm_runtime,
                    "post_evaluation_wall_runtime": float(timing["post_evaluation_wall_runtime"]),
                    "penalized_runtime_par2": penalized_runtime_par2(
                        solved_to_tolerance=scientific == "certified_robust_optimal",
                        runtime=algorithm_runtime,
                        time_limit=1800.0,
                    ),
                }
            )
            result["total_wall_runtime"] = (
                algorithm_runtime
                + float(timing["post_evaluation_wall_runtime"])
                + float(timing["aggregation_runtime"])
                + float(timing["checkpoint_io_runtime"])
            )
            record = {
                **run_identity, "stage": STAGE, "task_type": "frontier",
                "candidate": CANDIDATE, "state": "complete",
                "algorithm_status": result.get("status"), "scientific_status": scientific,
                "solved_to_tolerance": scientific == "certified_robust_optimal", "result": result,
            }
            atomic_write_json(run_root / "run.json", record)
            _write_status(run_root, state="complete", scientific=scientific, algorithm=str(result.get("status")))
            records_by_key[record["run_key"]] = record
            _aggregate_records(output, rows)
            if scientific in {"invalid_post_evaluation", "implementation_error"}:
                raise RemediationGateError(f"D2 batch stopped fail closed: {scientific}")

    _aggregate_records(output, rows)
    completed = list(records_by_key.values())
    manifest["completed_run_count"] = len(completed)
    manifest["certified_solved_count"] = sum(
        record.get("scientific_status") == "certified_robust_optimal" for record in completed
    )
    baseline_certified = sum(
        record.get("task_type") == "baseline"
        and record.get("scientific_status") == "certified_robust_optimal"
        for record in completed
    )
    frontier_certified = sum(
        record.get("task_type") == "frontier"
        and record.get("scientific_status") == "certified_robust_optimal"
        for record in completed
    )
    manifest["d2_gate"] = {
        "baseline_certified": baseline_certified,
        "frontier_certified": frontier_certified,
        "passed": baseline_certified == 3 and frontier_certified == 9 and len(completed) == 12,
        "selective_rerun_authorized": False,
        "final_holdout_or_full_grid_authorized": baseline_certified == 3 and frontier_certified == 9 and len(completed) == 12,
    }
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
        raise RemediationGateError("only D2 is authorized by the D2 runner")
    if args.dry_run:
        print(json.dumps(dry_run(args.config), indent=2, sort_keys=True))
        return 0
    manifest = run_d2(args.config, resume=args.resume)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
