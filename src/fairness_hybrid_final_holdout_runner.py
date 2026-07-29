from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

import yaml

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
    initial_scenario_plan_identity,
    initial_upper_bound_expected_identity,
    solve_certified_hybrid_scenario_benders_fairness,
)
from .fairness_hybrid_ccg_benders_runner import (
    HybridDependencies,
    _baseline_checkpoint_payload,
    _hybrid_certified_anchor,
    _validate_baseline_checkpoint,
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
)
from .fairness_post_evaluation import checkpointed_fairness_post_evaluation
from .fairness_scalability_results_audit import resolved_config_file_bytes
from .fairness_scalability_runner import run_directory_id
from .instance import InventoryInstance


STAGE = "FINAL_HOLDOUT"
SCHEMA = "fairness_hybrid_final_holdout_manifest_v1"
EXECUTION_ATTEMPT = 1
EXPECTED_CONFIG_SHA256 = "E0C3A5312520BA7220D3D61675D1794422AA4CAD2AB4866016CA0DF515A96E16"
PROTOCOL_SHA256 = "BC01396163EE9E9CD7AB2F6CBCD682B33BE107DDAED5FADEB0CE96290B1AA931"
D2_DECISION_SHA256 = "A43D3A6E9B74C19996AD6C8F3CFF3543462AFAA99A691C79781E1EABFEFC666D"
RESOLVED_CONFIG_CANONICALIZATION = "PyYAML safe_dump(sort_keys=True, allow_unicode=True), UTF-8"
SEEDS = list(range(170, 180))
RHOS = [0.0, 0.01, 0.025, 0.05, 0.10]
SCALES = {
    "medium_large": {"num_regions": 10, "num_products": 6, "scenario_count": 1831, "output_dir": "experiments/results_fairness_hybrid_final_holdout/ml_a1"},
    "large": {"num_regions": 12, "num_products": 8, "scenario_count": 4657, "output_dir": "experiments/results_fairness_hybrid_final_holdout/lg_a1"},
}


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RemediationGateError("final holdout config must be a mapping")
    return value


def validate_config(path: str | Path, config: dict[str, Any]) -> None:
    source = Path(path)
    if file_sha256(source).upper() != EXPECTED_CONFIG_SHA256:
        raise RemediationGateError("final holdout config SHA mismatch")
    expected = {
        "stage": STAGE,
        "authorization": "formal_execution_authorized_after_merge",
        "formal_run_authorized": True,
        "schema_version": 1,
        "execution_attempt": EXECUTION_ATTEMPT,
        "previous_attempt_results_reused": False,
        "d2_decision": "approve_final_cross_scale_holdout_protocol",
        "required_d2_decision_sha256": D2_DECISION_SHA256,
        "required_protocol_sha256": PROTOCOL_SHA256,
        "required_candidate_sha256": CANDIDATE_SHA256,
        "candidate": CANDIDATE,
        "seeds": SEEDS,
        "rho": RHOS,
        "baseline_count": 20,
        "frontier_count": 100,
        "total_tasks": 120,
        "solver_identity": {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7},
        "gamma": 2,
        "tol": 1e-4,
        "baseline_time_limit_seconds": 1800,
        "algorithm_time_limit_seconds": 1800,
        "general_time_limit_seconds": 1800,
        "max_iterations": 10000,
        "post_evaluation": {"time_limit_per_scenario_seconds": 30, "checkpoint_chunk_size": 25},
        "runtime_semantics": {"par2_multiplier": 2, "par2_basis": "algorithm_runtime"},
        "resume": True,
        "overwrite_supported": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RemediationGateError(f"final holdout identity drifted: {key}")
    root = Path(__file__).resolve().parents[1]
    if file_sha256(root / config["protocol_document"]).upper() != PROTOCOL_SHA256:
        raise RemediationGateError("final holdout protocol SHA mismatch")
    if file_sha256(root / config["candidate_definition"]).upper() != CANDIDATE_SHA256:
        raise RemediationGateError("final holdout candidate SHA mismatch")
    decision_path = root / config["d2_decision_file"]
    if file_sha256(decision_path).upper() != D2_DECISION_SHA256:
        raise RemediationGateError("D2 decision SHA mismatch")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != "approve_final_cross_scale_holdout_protocol":
        raise RemediationGateError("D2 decision does not authorize holdout protocol")
    for scale, frozen in SCALES.items():
        configured = config.get("scales", {}).get(scale, {})
        if configured != {**frozen, "baseline_count": 10, "frontier_count": 50, "total_tasks": 60}:
            raise RemediationGateError(f"final holdout scale identity drifted: {scale}")
    statistics = config.get("statistics", {})
    if statistics.get("independent_unit") != "seed" or statistics.get("prohibit_seed_rho_independence") is not True:
        raise RemediationGateError("final holdout statistical unit drifted")


def expand_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scale in SCALES:
        for seed in SEEDS:
            tasks = [("baseline", "baseline", "NOT_APPLICABLE")]
            tasks.extend(("frontier", CANDIDATE, f"{rho:.3f}".rstrip("0").rstrip(".")) for rho in RHOS)
            for task_type, candidate, rho in tasks:
                key = json.dumps(
                    {
                        "candidate": candidate,
                        "execution_attempt": EXECUTION_ATTEMPT,
                        "rho": rho,
                        "scale": scale,
                        "seed": seed,
                        "stage": STAGE,
                        "task_type": task_type,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                rows.append(
                    {
                        "scale": scale,
                        "seed": seed,
                        "rho": rho,
                        "candidate": candidate,
                        "task_type": task_type,
                        "run_key": key,
                        "run_directory_id": run_directory_id(key),
                    }
                )
    return rows


def _planned_paths(root: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for scale, frozen in SCALES.items():
        output = root / frozen["output_dir"]
        for name in ("manifest.json", ".manifest.json.tmp", "resolved_config.yaml", "results.csv", "summary.csv", "audit.log"):
            paths.append((f"{scale}_root", output / name))
        for seed in SEEDS:
            paths.append((f"{scale}_instance", output / f"instances/{seed}.json"))
        chunk_index = math_ceil(frozen["scenario_count"], int(config["post_evaluation"]["checkpoint_chunk_size"])) - 1
        for row in (item for item in rows if item["scale"] == scale):
            run = output / "runs" / row["run_directory_id"]
            for kind, name in (
                ("run", "run.json"),
                ("run_tmp", ".run.json.tmp"),
                ("status", "status.json"),
                ("algorithm_checkpoint", "algorithm_checkpoint.json"),
                ("post_index", "post_evaluation/checkpoint/index.json"),
                ("post_chunk_tmp", f"post_evaluation/checkpoint/.chunk_{chunk_index:05d}.json.tmp"),
            ):
                paths.append((f"{scale}_{kind}", run / name))
    return [(kind, path.resolve()) for kind, path in paths]


def math_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def structured_seed_access_evidence(root: Path) -> list[str]:
    evidence: list[str] = []
    excluded = {".git", ".pytest_cache", "__pycache__"}
    for path in root.rglob("*.json"):
        if any(part in excluded for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        if any(relative.startswith(f"{frozen['output_dir']}/") for frozen in SCALES.values()):
            continue
        if relative.startswith("analysis/fairness_hybrid_ccg_benders_d2_decision/"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            value = json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if _contains_access_identity(value):
            evidence.append(relative)
    for scale in SCALES:
        output = root / SCALES[scale]["output_dir"]
        for seed in SEEDS:
            if (output / f"instances/{seed}.json").exists():
                evidence.append((output / f"instances/{seed}.json").relative_to(root).as_posix())
    return sorted(set(evidence))


def _contains_access_identity(value: Any) -> bool:
    if isinstance(value, dict):
        seed = value.get("seed")
        identity_keys = {"run_key", "instance_sha256", "algorithm_status", "checkpoint_sha256", "best_x_values"}
        if isinstance(seed, int) and seed in SEEDS and identity_keys.intersection(value):
            return True
        return any(_contains_access_identity(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_access_identity(item) for item in value)
    return False


def dry_run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config_path, config)
    root = Path(__file__).resolve().parents[1]
    rows = expand_plan()
    paths = _planned_paths(root, config, rows)
    longest_kind, longest = max(paths, key=lambda item: len(str(item[1])))
    by_scale = {}
    for scale, frozen in SCALES.items():
        scale_rows = [row for row in rows if row["scale"] == scale]
        by_scale[scale] = {
            "baseline": sum(row["task_type"] == "baseline" for row in scale_rows),
            "frontier": sum(row["task_type"] == "frontier" for row in scale_rows),
            "total": len(scale_rows),
            "scenario_count": frozen["scenario_count"],
            "output_dir_exists": (root / frozen["output_dir"]).exists(),
            **initial_scenario_plan_identity(
                num_regions=frozen["num_regions"], num_products=frozen["num_products"], gamma=2
            ),
        }
    access = structured_seed_access_evidence(root)
    return {
        "stage": STAGE,
        "candidate": CANDIDATE,
        "seeds": SEEDS,
        "rho": RHOS,
        "baseline": 20,
        "frontier": 100,
        "total": len(rows),
        "unique_run_keys": len({row["run_key"] for row in rows}),
        "duplicate_run_keys": len(rows) - len({row["run_key"] for row in rows}),
        "by_scale": by_scale,
        "independent_unit": "seed",
        "seed_cluster_count": 10,
        "reserved_seed_access_evidence": access,
        "reserved_seed_access_audit_passed": not access,
        "instances_generated": False,
        "solver_called": False,
        "output_dirs_exist": any(value["output_dir_exists"] for value in by_scale.values()),
        "windows_path_check": len(str(longest)) < 220,
        "longest_path_type": longest_kind,
        "longest_path": longest.relative_to(root).as_posix(),
        "longest_path_length": len(str(longest)),
        "formal_run_authorized": True,
    }


def write_protocol_evidence(config_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = dry_run(config_path)
    rows = expand_plan()
    plan_path = output / "frozen_run_plan.csv"
    with plan_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "dry_run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    seed_audit = {
        "access_evidence": report["reserved_seed_access_evidence"],
        "audit_passed": report["reserved_seed_access_audit_passed"],
        "declaration_only_is_not_access": True,
        "formal_run_authorized": True,
        "pre_run_reaudit_required": True,
        "reserved_seeds": SEEDS,
    }
    (output / "seed_access_audit.json").write_text(
        json.dumps(seed_audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result = {}
    for name in ("dry_run.json", "frozen_run_plan.csv", "seed_access_audit.json"):
        result[name] = file_sha256(output / name).upper()
    return result


def _git_output(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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


def _scale_config(config: dict[str, Any], scale: str) -> dict[str, Any]:
    value = deepcopy(config)
    value.update(
        {
            "scale": scale,
            "scenario_count": SCALES[scale]["scenario_count"],
            "output_dir": SCALES[scale]["output_dir"],
        }
    )
    return value


def _identity(config_path: Path, config: dict[str, Any], scale: str) -> dict[str, Any]:
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
        "d2_decision_sha256": D2_DECISION_SHA256,
        "candidate": CANDIDATE,
        "solver_parameters": deepcopy(SOLVER_PARAMETERS),
        "scale": scale,
        "seeds": SEEDS,
        "rhos": RHOS,
        "scenario_count": SCALES[scale]["scenario_count"],
        "previous_attempt_results_reused": False,
    }


def _production_frontier(
    config: dict[str, Any],
    instance: InventoryInstance,
    baseline: dict[str, Any],
    anchor: dict[str, Any],
    expected: dict[str, Any],
    checkpoint: Path,
    solver: dict[str, Any],
    row: dict[str, Any],
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
        checkpoint_identity={"run_key": row["run_key"], **deepcopy(expected)},
        execution_protocol_sha256=PROTOCOL_SHA256,
        output_flag=False,
    )
    return result.to_dict()


def _production_post_evaluate(
    config: dict[str, Any],
    instance: InventoryInstance,
    result: dict[str, Any],
    anchor: dict[str, Any],
    run_identity: dict[str, Any],
    root: Path,
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
        max_scenarios=int(config["scenario_count"]),
        per_scenario_time_limit=float(config["post_evaluation"]["time_limit_per_scenario_seconds"]),
        tolerance=1e-7,
        chunk_size=int(config["post_evaluation"]["checkpoint_chunk_size"]),
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
        post_evaluate=_production_post_evaluate,
        configure_solver=_configure_solver_parameters,
    )


def _write_status(root: Path, *, state: str, scientific: str, algorithm: str) -> None:
    atomic_write_json(
        root / "status.json",
        {"state": state, "scientific_status": scientific, "algorithm_status": algorithm},
    )


def _run_scale(
    config_path: Path,
    config: dict[str, Any],
    scale: str,
    rows: list[dict[str, Any]],
    deps: HybridDependencies,
    existing: dict[str, Any] | None,
    *,
    failure_injector: Callable[[str, dict[str, Any]], None] | None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    scale_config = _scale_config(config, scale)
    output = root / str(scale_config["output_dir"])
    identity = _identity(config_path, config, scale)
    manifest_path = output / "manifest.json"
    if existing is None:
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_yaml(output / "resolved_config.yaml", config)
    if file_sha256(output / "resolved_config.yaml").upper() != identity["resolved_config_file_sha256"]:
        raise RemediationGateError(f"final holdout {scale} resolved config identity mismatch")
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
    records_by_key: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        instance_path = output / f"instances/{seed}.json"
        stored = _read_json_strict(instance_path, label=f"final holdout {scale} instance {seed}")
        if stored is None:
            instance = deps.generate_instance(scale_config, seed)
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
            "scale": scale,
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
        baseline = _read_json_strict(baseline_root / "run.json", label=f"final holdout {scale} baseline {seed}")
        if baseline is None:
            checkpoint_path = baseline_root / "baseline_checkpoint.json"
            checkpoint = _read_json_strict(checkpoint_path, label=f"final holdout {scale} baseline checkpoint {seed}")
            if checkpoint is not None:
                payload = _validate_baseline_checkpoint(checkpoint, common)
            else:
                _write_status(baseline_root, state="running", scientific="pending", algorithm="running")
                payload = deps.solve_baseline(scale_config, instance, seed, deepcopy(SOLVER_PARAMETERS))
                atomic_write_json(checkpoint_path, _baseline_checkpoint_payload(common, payload))
                if failure_injector:
                    failure_injector("after_baseline_checkpoint", deepcopy(baseline_row))
            solved = (
                payload.get("status") == "optimal"
                and payload.get("valid_UB") is True
                and math.isfinite(float(payload.get("upper_bound", math.nan)))
                and float(payload.get("gap", math.inf)) <= float(config["tol"])
            )
            runtime = float(payload.get("runtime", 0.0))
            payload.update(
                {
                    "algorithm_runtime": runtime,
                    "post_evaluation_wall_runtime": 0.0,
                    "total_wall_runtime": runtime,
                    "penalized_runtime_par2": penalized_runtime_par2(
                        solved_to_tolerance=solved,
                        runtime=runtime,
                        time_limit=float(config["baseline_time_limit_seconds"]),
                    ),
                }
            )
            baseline = {
                **common,
                "run_key": baseline_row["run_key"],
                "run_directory_id": baseline_row["run_directory_id"],
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
                baseline_root,
                state="complete",
                scientific=baseline["scientific_status"],
                algorithm=str(payload.get("status")),
            )
        elif not _same_identity(baseline, common, tuple(common)):
            raise RemediationGateError(f"final holdout {scale} baseline resume identity mismatch")
        records_by_key[baseline["run_key"]] = baseline
        anchor = _hybrid_certified_anchor(baseline, common_identity=common, tolerance=float(config["tol"]))
        manifest["baseline_anchors"][str(seed)] = anchor
        atomic_write_json(manifest_path, manifest)

        for row in (value for value in seed_rows if value["task_type"] == "frontier"):
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
            record = _read_json_strict(run_root / "run.json", label=f"final holdout {scale} frontier {seed}/{row['rho']}")
            if record is not None:
                if not _same_identity(record, run_identity, tuple(run_identity)):
                    raise RemediationGateError(f"final holdout {scale} frontier resume identity mismatch")
                records_by_key[record["run_key"]] = record
                continue
            expected = initial_upper_bound_expected_identity(common, anchor)
            _write_status(run_root, state="running", scientific="pending", algorithm="running")
            started = time.perf_counter()
            try:
                result = deps.solve_frontier(
                    scale_config,
                    instance,
                    baseline,
                    anchor,
                    expected,
                    run_root / "algorithm_checkpoint.json",
                    deepcopy(SOLVER_PARAMETERS),
                    row,
                )
            except KeyboardInterrupt:
                _write_status(run_root, state="interrupted", scientific="interrupted", algorithm="interrupted")
                raise
            except Exception as exc:
                failure = {
                    **run_identity,
                    "task_type": "frontier",
                    "candidate": CANDIDATE,
                    "state": "complete",
                    "algorithm_status": "exception",
                    "scientific_status": "implementation_error",
                    "solved_to_tolerance": False,
                    "failure_reason": str(exc),
                    "result": {
                        "status": "exception",
                        "algorithm_runtime": time.perf_counter() - started,
                        "penalized_runtime_par2": 3600.0,
                    },
                }
                atomic_write_json(run_root / "run.json", failure)
                _write_status(run_root, state="complete", scientific="implementation_error", algorithm="exception")
                _aggregate_records(output, rows)
                raise
            algorithm_runtime = float(result.get("runtime", time.perf_counter() - started))
            algorithm_certified = (
                result.get("status") == "optimal"
                and result.get("gap") is not None
                and float(result["gap"]) <= float(config["tol"])
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
                    scale_config,
                    instance,
                    result,
                    anchor,
                    run_identity,
                    run_root / "post_evaluation",
                    row,
                )
            scientific = _classify_frontier(
                result,
                evaluation,
                tolerance=float(config["tol"]),
                expected_scenario_count=int(scale_config["scenario_count"]),
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
                        time_limit=float(config["algorithm_time_limit_seconds"]),
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
                **run_identity,
                "task_type": "frontier",
                "candidate": CANDIDATE,
                "state": "complete",
                "algorithm_status": result.get("status"),
                "scientific_status": scientific,
                "solved_to_tolerance": scientific == "certified_robust_optimal",
                "result": result,
            }
            atomic_write_json(run_root / "run.json", record)
            _write_status(run_root, state="complete", scientific=scientific, algorithm=str(result.get("status")))
            records_by_key[record["run_key"]] = record
            _aggregate_records(output, rows)
            if scientific in {"invalid_post_evaluation", "implementation_error"}:
                raise RemediationGateError(f"final holdout batch stopped fail closed: {scientific}")

    _aggregate_records(output, rows)
    completed = list(records_by_key.values())
    manifest["completed_run_count"] = len(completed)
    manifest["certified_solved_count"] = sum(
        record.get("scientific_status") == "certified_robust_optimal" for record in completed
    )
    manifest["baseline_certified_count"] = sum(
        record.get("task_type") == "baseline"
        and record.get("scientific_status") == "certified_robust_optimal"
        for record in completed
    )
    manifest["frontier_certified_count"] = sum(
        record.get("task_type") == "frontier"
        and record.get("scientific_status") == "certified_robust_optimal"
        for record in completed
    )
    manifest["updated_at"] = utc_now_iso()
    atomic_write_json(manifest_path, manifest)
    return manifest


def run_holdout(
    config_path: str | Path,
    *,
    resume: bool,
    dependencies: HybridDependencies | None = None,
    test_authorization: bool = False,
    failure_injector: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not resume:
        raise RemediationGateError("FINAL_HOLDOUT requires --resume; --overwrite is unsupported")
    path = Path(config_path)
    config = load_config(path)
    validate_config(path, config)
    if dependencies is not None and not test_authorization:
        raise RemediationGateError("dependency substitution requires test_authorization")
    root = Path(__file__).resolve().parents[1]
    if not test_authorization:
        _formal_git_gate(root)
        access = structured_seed_access_evidence(root)
        if access:
            raise RemediationGateError(f"formal_run_not_authorized: reserved seed access evidence: {access}")
    rows = expand_plan()
    planned = _planned_paths(root, config, rows)
    if max(len(str(value)) for _, value in planned) >= 220:
        raise RemediationGateError("final holdout Windows path preflight failed")
    existing_by_scale: dict[str, dict[str, Any] | None] = {}
    for scale in SCALES:
        output = root / SCALES[scale]["output_dir"]
        manifest = _read_json_strict(output / "manifest.json", label=f"final holdout {scale} manifest")
        if output.exists() and manifest is None:
            raise RemediationGateError(f"existing final holdout {scale} output lacks a valid manifest")
        expected = _identity(path, config, scale)
        if manifest is not None and manifest.get("identity") != expected:
            raise RemediationGateError(f"final holdout {scale} resume identity mismatch")
        if manifest is not None and file_sha256(output / "resolved_config.yaml").upper() != expected["resolved_config_file_sha256"]:
            raise RemediationGateError(f"final holdout {scale} resolved config identity mismatch")
        existing_by_scale[scale] = manifest
    deps = dependencies or production_dependencies()
    deps.configure_solver(deepcopy(SOLVER_PARAMETERS))
    manifests = {}
    for scale in SCALES:
        scale_rows = [row for row in rows if row["scale"] == scale]
        manifests[scale] = _run_scale(
            path,
            config,
            scale,
            scale_rows,
            deps,
            existing_by_scale[scale],
            failure_injector=failure_injector,
        )
    return {
        "stage": STAGE,
        "manifests": manifests,
        "completed_run_count": sum(value["completed_run_count"] for value in manifests.values()),
        "certified_solved_count": sum(value["certified_solved_count"] for value in manifests.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.stage != STAGE:
        raise RemediationGateError("only FINAL_HOLDOUT is authorized by this runner")
    if args.dry_run:
        report: dict[str, Any] = {"dry_run": dry_run(args.config)}
        if args.evidence_output is not None:
            report["artifacts"] = write_protocol_evidence(args.config, args.evidence_output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    report = run_holdout(args.config, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
