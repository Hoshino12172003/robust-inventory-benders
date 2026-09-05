"""Strictly paired cross-scale direct deterministic-equivalent benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable
import zipfile

import yaml

from .experiment_protocol import atomic_write_csv, atomic_write_json, atomic_write_yaml, file_sha256


STAGE = "GUROBI_DIRECT_CROSS_SCALE_PAIRED"
SCHEMA = "gurobi_direct_cross_scale_paired_v1"
SCALES = ("medium_large", "large")
SEEDS = tuple(range(180, 185))
GAMMA = 2
RHO = 0.025
SOURCE_SHA256 = "EE45A00AA341EE5EB2894DE43EE2F47022C27F1D29146FCFEC803236EF59DB6F"
EXPECTED_SCENARIOS = {"medium_large": 1831, "large": 4657}


class DirectBenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class Dependencies:
    solve_direct: Callable[..., dict[str, Any]]
    post_evaluate: Callable[..., tuple[dict[str, Any], dict[str, float]]]
    deserialize_instance: Callable[[dict[str, Any]], Any]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DirectBenchmarkError("config must be a YAML object")
    return value


def load_catalog(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    value = json.loads((root / config["source_catalog"]).read_text(encoding="utf-8"))
    cells = value.get("cells")
    if value.get("source_zip_sha256") != SOURCE_SHA256 or not isinstance(cells, list):
        raise DirectBenchmarkError("source catalog identity mismatch")
    selected = [cell for cell in cells if cell.get("scale") in SCALES and cell.get("seed") in SEEDS]
    selected.sort(key=lambda cell: (SCALES.index(cell["scale"]), cell["seed"]))
    expected = {(scale, seed) for scale in SCALES for seed in SEEDS}
    if {(cell.get("scale"), cell.get("seed")) for cell in selected} != expected or len(selected) != 10:
        raise DirectBenchmarkError("source catalog does not contain the exact ten paired cells")
    for cell in selected:
        if cell.get("gamma") != GAMMA or cell.get("rho") != "0.025":
            raise DirectBenchmarkError("source cell Gamma/rho mismatch")
        if cell.get("source_hybrid_scientific_status") != "certified_robust_optimal":
            raise DirectBenchmarkError("paired Hybrid result is not certified")
        if cell.get("baseline_scientific_status") != "certified_robust_optimal":
            raise DirectBenchmarkError("paired baseline is not certified")
    return selected


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1, "stage": STAGE, "source_zip_sha256": SOURCE_SHA256,
        "scales": list(SCALES), "seeds": list(SEEDS), "gamma": GAMMA, "rho": RHO,
        "algorithm_time_limit_seconds": 1800,
        "post_evaluation_time_limit_per_scenario_seconds": 30,
        "checkpoint_chunk_size": 25, "par2_seconds": 3600,
        "resume_required": True, "hybrid_rerun": False, "baseline_rerun": False,
    }
    for name, wanted in expected.items():
        if config.get(name) != wanted:
            raise DirectBenchmarkError(f"config {name} drift")
    if config.get("solver_identity") != {
        "Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7, "BendersStrategy": 0,
    }:
        raise DirectBenchmarkError("solver identity drift")


def run_key(scale: str, seed: int) -> str:
    return canonical_json({
        "candidate": "gurobi_direct_extensive_form", "gamma": GAMMA, "rho": "0.025",
        "scale": scale, "seed": seed, "stage": STAGE, "task_type": "direct_frontier",
    })


def expand_plan() -> list[dict[str, Any]]:
    rows = []
    for scale in SCALES:
        for seed in SEEDS:
            key = run_key(scale, seed)
            rows.append({"scale": scale, "seed": seed, "run_key": key,
                         "run_directory_id": "r_" + hashlib.sha256(key.encode()).hexdigest()[:24]})
    if len({row["run_key"] for row in rows}) != 10 or len({row["run_directory_id"] for row in rows}) != 10:
        raise DirectBenchmarkError("run identity collision")
    return rows


def _source_prefix(scale: str) -> str:
    return f"experiments/results_fh_gamma/{'ml_a3' if scale == 'medium_large' else 'lg_a3'}"


def verify_source_metadata(source: Path, cells: list[dict[str, Any]]) -> None:
    if file_sha256(source).upper() != SOURCE_SHA256:
        raise DirectBenchmarkError("source ZIP SHA mismatch")
    with zipfile.ZipFile(source) as archive:
        for cell in cells:
            manifest = json.loads(archive.read(f"{_source_prefix(cell['scale'])}/manifest.json"))
            directory = manifest["run_key_to_directory_id"].get(cell["source_hybrid_run_key"])
            if directory != cell["source_hybrid_directory_id"]:
                raise DirectBenchmarkError("source Hybrid directory mapping drift")
            hybrid = json.loads(archive.read(f"{_source_prefix(cell['scale'])}/runs/{directory}/run.json"))
            if hybrid.get("scientific_status") != "certified_robust_optimal":
                raise DirectBenchmarkError("source Hybrid status drift")
            if hybrid.get("instance_canonical_sha256") != cell["instance_canonical_sha256"]:
                raise DirectBenchmarkError("source Hybrid instance identity drift")
            baseline_directory = manifest["run_key_to_directory_id"].get(cell["baseline_run_key"])
            if not isinstance(baseline_directory, str):
                raise DirectBenchmarkError("source baseline directory mapping drift")
            baseline = json.loads(archive.read(
                f"{_source_prefix(cell['scale'])}/runs/{baseline_directory}/run.json"
            ))
            upper = baseline.get("result", {}).get("upper_bound")
            if baseline.get("scientific_status") != "certified_robust_optimal":
                raise DirectBenchmarkError("source baseline status drift")
            if not isinstance(upper, (int, float)) or float(upper).hex() != cell["anchor_value_hex"]:
                raise DirectBenchmarkError("source anchor value drift")


def _load_instance(source: Path, cell: dict[str, Any]) -> dict[str, Any]:
    with zipfile.ZipFile(source) as archive:
        raw = archive.read(cell["instance_member"])
    if hashlib.sha256(raw).hexdigest().upper() != cell["instance_file_sha256"]:
        raise DirectBenchmarkError("instance file SHA mismatch")
    payload = json.loads(raw)
    instance = payload.get("instance")
    if not isinstance(instance, dict) or sha256_value(instance) != cell["instance_canonical_sha256"]:
        raise DirectBenchmarkError("instance canonical identity mismatch")
    return instance


def dry_run(config_path: str | Path) -> dict[str, Any]:
    before = "gurobipy" in sys.modules
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(config_path); validate_config(config)
    cells = load_catalog(root, config)
    source = Path(config["source_zip"])
    verify_source_metadata(source, cells)
    output = Path(config["output_dir"])
    report = {
        "stage": STAGE, "planned_direct_runs": 10, "hybrid_reruns": 0,
        "baseline_reruns": 0, "scales": list(SCALES), "seeds": list(SEEDS),
        "gamma": GAMMA, "rho": RHO, "source_cells_verified": len(cells),
        "source_zip_sha256": file_sha256(source).upper(), "solver_called": False,
        "gurobipy_imported_by_dry_run": "gurobipy" in sys.modules and not before,
        "output_exists": output.exists(), "estimated_worst_case_seconds": 18000,
    }
    if report["gurobipy_imported_by_dry_run"]:
        raise DirectBenchmarkError("dry-run imported gurobipy")
    return report


def production_dependencies() -> Dependencies:
    from .fairness_high_gamma_external_solver_benchmark import solve_gurobi_direct_extensive_form
    from .fairness_post_evaluation import checkpointed_fairness_post_evaluation
    from .instance import InventoryInstance

    def solve(config: dict[str, Any], instance: Any, cell: dict[str, Any]) -> dict[str, Any]:
        solver = dict(config["solver_identity"]); solver.pop("BendersStrategy")
        return solve_gurobi_direct_extensive_form(
            instance, baseline_cost=float(cell["anchor_value"]), rho=RHO, gamma=GAMMA,
            expected_scenario_count=EXPECTED_SCENARIOS[cell["scale"]],
            solver_parameters=solver, time_limit=float(config["algorithm_time_limit_seconds"]),
            output_flag=True,
        ).to_dict()

    def post(config: dict[str, Any], instance: Any, result: dict[str, Any], cell: dict[str, Any],
             identity: dict[str, Any], post_root: Path) -> tuple[dict[str, Any], dict[str, float]]:
        evaluation, timing = checkpointed_fairness_post_evaluation(
            instance, root=post_root, run_key=identity["run_key"],
            config_sha256_value=identity["config_sha256"], git_commit=identity["git_commit"],
            baseline_anchor_sha256=cell["anchor_sha256"], y_values=result["y_values"],
            x_values=result["x_values"], t_value=float(result["objective_t"]),
            baseline_cost=float(cell["anchor_value"]), rho=RHO, gamma=GAMMA,
            max_scenarios=EXPECTED_SCENARIOS[cell["scale"]],
            per_scenario_time_limit=float(config["post_evaluation_time_limit_per_scenario_seconds"]),
            tolerance=float(config["solver_identity"]["FeasibilityTol"]),
            chunk_size=int(config["checkpoint_chunk_size"]), resume_count=0,
            output_flag=False, run_execution_attempt=1, post_evaluation_pipeline_generation=4,
        )
        return evaluation.to_dict(), asdict(timing)

    return Dependencies(solve, post, InventoryInstance.from_dict)


def _identity(config_path: Path, commit: str, row: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "git_commit": commit, "config_sha256": file_sha256(config_path).upper(),
        "run_key": row["run_key"], "run_directory_id": row["run_directory_id"],
        "scale": row["scale"], "seed": row["seed"], "gamma": GAMMA, "rho": "0.025",
        "instance_canonical_sha256": cell["instance_canonical_sha256"],
        "baseline_run_key": cell["baseline_run_key"], "anchor_sha256": cell["anchor_sha256"],
        "source_hybrid_run_key": cell["source_hybrid_run_key"],
    }


def _scientific_status(result: dict[str, Any], post: dict[str, Any] | None, expected: int) -> str:
    exact = (
        result.get("status") == "optimal" and result.get("complete_model_built") is True
        and isinstance(result.get("gap"), (int, float)) and float(result["gap"]) <= 1e-4
        and isinstance(result.get("objective_t"), (int, float))
    )
    if not exact:
        return "time_limit_uncertified" if "time_limit" in str(result.get("status")) else str(result.get("status"))
    if not isinstance(post, dict) or post.get("valid") is not True or post.get("errors") != []:
        return "invalid_post_evaluation"
    if post.get("scenario_count") != expected or post.get("objective_t_consistent") is not True:
        return "invalid_post_evaluation"
    return "certified_robust_optimal"


def _aggregate(output: Path, plan: list[dict[str, Any]], cells: list[dict[str, Any]], par2: float) -> None:
    cell_map = {(cell["scale"], cell["seed"]): cell for cell in cells}
    rows = []
    for item in plan:
        path = output / "runs" / item["run_directory_id"] / "run.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8")); result = record["result"]
        cell = cell_map[(item["scale"], item["seed"])]
        certified = record["scientific_status"] == "certified_robust_optimal"
        runtime = float(result["algorithm_runtime"])
        rows.append({
            "scale": item["scale"], "seed": item["seed"], "scenario_count": EXPECTED_SCENARIOS[item["scale"]],
            "direct_status": record["scientific_status"], "direct_runtime_seconds": runtime,
            "direct_par2_seconds": runtime if certified else par2,
            "hybrid_status": cell["source_hybrid_scientific_status"],
            "hybrid_runtime_seconds": cell["source_hybrid_algorithm_runtime"],
            "runtime_ratio_direct_over_hybrid": runtime / float(cell["source_hybrid_algorithm_runtime"]) if certified else "NOT_APPLICABLE",
            "direct_objective_t": result.get("objective_t") if certified else "NOT_APPLICABLE",
            "hybrid_objective_t": cell["source_hybrid_objective_t"],
            "objective_abs_difference": abs(float(result["objective_t"]) - float(cell["source_hybrid_objective_t"])) if certified else "NOT_APPLICABLE",
            "direct_rows": result.get("rows", 0), "direct_columns": result.get("columns", 0),
            "direct_nonzeros": result.get("nonzeros", 0), "direct_node_count": result.get("node_count"),
            "instance_sha256": cell["instance_canonical_sha256"], "anchor_sha256": cell["anchor_sha256"],
        })
    fields = list(rows[0]) if rows else []
    atomic_write_csv(output / "paired_results.csv", rows, fields)
    summaries = []
    for scale in SCALES:
        subset = [row for row in rows if row["scale"] == scale]
        summaries.append({
            "scale": scale, "planned": 5, "completed": len(subset),
            "direct_certified": sum(row["direct_status"] == "certified_robust_optimal" for row in subset),
            "mean_direct_runtime_seconds": math.fsum(float(row["direct_runtime_seconds"]) for row in subset) / len(subset) if subset else "NOT_APPLICABLE",
            "mean_direct_par2_seconds": math.fsum(float(row["direct_par2_seconds"]) for row in subset) / len(subset) if subset else "NOT_APPLICABLE",
            "mean_hybrid_runtime_seconds": math.fsum(float(row["hybrid_runtime_seconds"]) for row in subset) / len(subset) if subset else "NOT_APPLICABLE",
        })
    atomic_write_csv(output / "summary.csv", summaries, list(summaries[0]))


def execute(config_path: str | Path, *, resume: bool,
            dependencies: Dependencies | None = None) -> dict[str, Any]:
    if not resume:
        raise DirectBenchmarkError("execution requires --resume")
    config_path = Path(config_path).resolve(); root = Path(__file__).resolve().parents[1]
    config = load_yaml(config_path); validate_config(config)
    git_status = __import__("subprocess").run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if git_status:
        raise DirectBenchmarkError("formal execution requires a clean committed worktree")
    source = Path(config["source_zip"]); cells = load_catalog(root, config)
    verify_source_metadata(source, cells)
    plan = expand_plan(); cell_map = {(cell["scale"], cell["seed"]): cell for cell in cells}
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    commit = __import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    manifest = {"schema": SCHEMA, "git_commit": commit, "config_sha256": file_sha256(config_path).upper(),
                "source_zip_sha256": SOURCE_SHA256, "plan": plan}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise DirectBenchmarkError("resume manifest identity mismatch")
    atomic_write_json(manifest_path, manifest); atomic_write_yaml(output / "resolved_config.yaml", config)
    deps = dependencies or production_dependencies()
    for index, row in enumerate(plan, start=1):
        cell = cell_map[(row["scale"], row["seed"])]
        identity = _identity(config_path, commit, row, cell)
        run_root = output / "runs" / row["run_directory_id"]
        run_path = run_root / "run.json"; status_path = run_root / "status.json"
        checkpoint_path = run_root / "direct_checkpoint.json"
        if run_path.exists():
            record = json.loads(run_path.read_text(encoding="utf-8"))
            if record.get("identity") != identity or record.get("state") != "complete":
                raise DirectBenchmarkError("completed run identity mismatch")
            if not status_path.exists() or not checkpoint_path.exists():
                raise DirectBenchmarkError("completed run lacks status or Direct checkpoint")
            status_record = json.loads(status_path.read_text(encoding="utf-8"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if status_record.get("identity") != identity or checkpoint.get("identity") != identity:
                raise DirectBenchmarkError("completed auxiliary identity mismatch")
            continue
        run_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(status_path, {"identity": identity, "state": "running"})
        print(f"[{index}/10] Direct {row['scale']} seed={row['seed']} scenarios={EXPECTED_SCENARIOS[row['scale']]}", flush=True)
        instance_payload = _load_instance(source, cell)
        instance = deps.deserialize_instance(instance_payload)
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("identity") != identity or not isinstance(checkpoint.get("result"), dict):
                raise DirectBenchmarkError("Direct checkpoint identity mismatch")
            result = checkpoint["result"]
        else:
            result = deps.solve_direct(config, instance, cell)
            atomic_write_json(checkpoint_path, {"identity": identity, "result": result})
        post = None; timing: dict[str, float] = {}
        if result.get("status") == "optimal" and result.get("complete_model_built") is True:
            post, timing = deps.post_evaluate(config, instance, result, cell, identity, run_root / "post_evaluation")
        status = _scientific_status(result, post, EXPECTED_SCENARIOS[row["scale"]])
        result = {**result, "post_evaluation": post, "post_evaluation_timing": timing}
        record = {"identity": identity, "state": "complete", "scientific_status": status, "result": result}
        atomic_write_json(run_path, record)
        atomic_write_json(status_path, {"identity": identity, "state": "complete", "scientific_status": status})
        _aggregate(output, plan, cells, float(config["par2_seconds"]))
        print(f"[{index}/10] saved status={status} runtime={result['algorithm_runtime']:.3f}s", flush=True)
    _aggregate(output, plan, cells, float(config["par2_seconds"]))
    completed = len(list((output / "runs").glob("*/run.json")))
    return {"completed": completed, "output": str(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--stage", required=True)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.stage != STAGE:
        raise DirectBenchmarkError("stage mismatch")
    result = dry_run(args.config) if args.dry_run else execute(args.config, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
