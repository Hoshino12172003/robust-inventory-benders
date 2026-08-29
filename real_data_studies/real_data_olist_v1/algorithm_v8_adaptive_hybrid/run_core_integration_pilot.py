from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
REPO_ROOT = STUDY_ROOT.parents[1]
SCRIPTS = STUDY_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_factorized_finite_comparison as finite  # noqa: E402
from src.experiment_protocol import config_sha256  # noqa: E402
from src.fairness_hybrid_ccg_benders import (  # noqa: E402
    CANDIDATE_SHA256,
    HYBRID_V8_CANDIDATE_SHA256,
    solve_certified_hybrid_scenario_benders_fairness,
)
from src.instance import generate_instance, load_instance, save_instance  # noqa: E402
from src.scenarios import DemandScenario, enumerate_budget_scenarios  # noqa: E402


CONFIG = REPO_ROOT / "experiments/configs/hybrid_v8_core_integration_pilot.json"
PROTOCOL = REPO_ROOT / "docs/hybrid_v8_core_integration_pilot_protocol.md"
CORE_SOURCE = REPO_ROOT / "src/fairness_hybrid_ccg_benders.py"


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scenario_payload(scenario: DemandScenario) -> dict:
    return {
        "name": scenario.name,
        "active_units": [list(value) for value in scenario.active_units],
        "demand": [list(row) for row in scenario.demand],
    }


def load_scenarios(path: Path) -> list[DemandScenario]:
    return [
        DemandScenario(
            name=row["name"],
            active_units=tuple(tuple(int(value) for value in cell) for cell in row["active_units"]),
            demand=tuple(tuple(float(value) for value in line) for line in row["demand"]),
        )
        for row in json.loads(path.read_text(encoding="utf-8"))
    ]


def prepare() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = REPO_ROOT / config["output_dir"]
    if root.exists():
        raise RuntimeError("core pilot output already exists; preparation is append-only")
    files = []
    for seed in config["seeds"]:
        case = root / "cases" / f"seed_{seed}"
        instance = generate_instance(config, seed=int(seed))
        instance_path = save_instance(instance, case / "instance.json")
        scenarios = enumerate_budget_scenarios(
            instance, int(config["gamma"]),
            max_scenarios=int(config["expected_scenario_count"]), exact_scenarios=True,
        )
        if len(scenarios) != int(config["expected_scenario_count"]):
            raise RuntimeError(f"seed {seed}: unexpected finite scenario count")
        scenarios_path = case / "scenarios.json"
        save_json(scenarios_path, [scenario_payload(value) for value in scenarios])
        for path in (instance_path, scenarios_path):
            files.append({
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            })
    save_json(root / "input_freeze.json", {
        "status": "inputs_frozen_before_optimization",
        "config_sha256": sha256(CONFIG),
        "protocol_sha256": sha256(PROTOCOL),
        "runner_sha256": sha256(Path(__file__)),
        "core_source_sha256": sha256(CORE_SOURCE),
        "file_count": len(files),
        "files": files,
    })


def verify_inputs(root: Path) -> None:
    manifest = json.loads((root / "input_freeze.json").read_text(encoding="utf-8"))
    expected = {
        "config_sha256": sha256(CONFIG),
        "protocol_sha256": sha256(PROTOCOL),
        "runner_sha256": sha256(Path(__file__)),
        "core_source_sha256": sha256(CORE_SOURCE),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"core pilot {key} changed after input freeze")
    for row in manifest["files"]:
        if sha256(root / row["path"]) != row["sha256"]:
            raise RuntimeError(f"core pilot input changed: {row['path']}")


def evidence(instance, anchor: dict, seed: int, candidate_sha: str, manifest: dict):
    value = float(anchor["objective"])
    identity = {
        "instance_sha256": config_sha256(instance.to_dict()).upper(),
        "seed": int(seed),
        "scale": "core_integration_I3_J3_R4",
        "git_commit": manifest["core_source_sha256"][:40],
        "config_file_sha256": manifest["config_sha256"].upper(),
        "resolved_config_file_sha256": manifest["config_sha256"].upper(),
        "candidate_sha256": candidate_sha,
        "baseline_run_key": f"finite-cost-anchor-seed-{seed}",
    }
    record = {
        "run_key": identity["baseline_run_key"],
        **identity,
        "solved_to_tolerance": True,
        "scientific_status": "certified_robust_optimal",
        "result": {
            "status": "optimal",
            "valid_UB": True,
            "gap": 0.0,
            "upper_bound": value,
            "best_y_values": deepcopy(anchor["y_values"]),
            "best_x_values": deepcopy(anchor["x_values"]),
        },
    }
    anchor_record = {
        "source": "finite_scenario_cost_anchor.objective",
        "value": value,
        "value_hex": value.hex(),
        "baseline_run_key": record["run_key"],
        "base_git_commit": identity["git_commit"],
        "base_config_sha256": identity["config_file_sha256"],
        "candidate_config_sha256": candidate_sha,
        "valid_UB": True,
        "baseline_status": "optimal",
        "baseline_final_gap": 0.0,
        **identity,
    }
    anchor_record["anchor_value_hex"] = value.hex()
    anchor_record["anchor_sha256"] = config_sha256(anchor_record)
    expected = {
        **identity,
        "anchor_value_hex": anchor_record["value_hex"],
        "anchor_sha256": anchor_record["anchor_sha256"],
    }
    return record, anchor_record, expected


def solve_core(instance, anchor, seed, candidate_sha, policy, config, manifest):
    record, anchor_record, expected = evidence(instance, anchor, seed, candidate_sha, manifest)
    result = solve_certified_hybrid_scenario_benders_fairness(
        instance,
        baseline_record=record,
        anchor=anchor_record,
        expected_identity=expected,
        solver_parameters={"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7},
        rho=float(config["rho"]), gamma=int(config["gamma"]),
        max_iterations=int(config["max_iterations"]),
        time_limit=float(config["time_limit_seconds_per_method"]),
        tol=float(config["objective_abs_tolerance"]),
        algorithm_policy=policy,
        execution_protocol_sha256=manifest["protocol_sha256"].upper(),
    )
    return {
        "status": result.status,
        "objective_t": result.objective_t,
        "runtime": result.runtime,
        "iterations": result.iterations,
        "scenario_blocks": result.metadata["committed_scenario_count"],
        "farkas_cuts": result.cuts,
        "certified": result.metadata["robust_feasibility_certified"],
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "metadata": result.metadata,
        "iteration_log": result.iteration_log,
    }


def run() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = REPO_ROOT / config["output_dir"]
    verify_inputs(root)
    manifest = json.loads((root / "input_freeze.json").read_text(encoding="utf-8"))
    orders = (("v8", "legacy"), ("legacy", "v8"), ("v8", "legacy"))
    for position, seed in enumerate(config["seeds"]):
        output = root / "results" / f"seed_{seed}.json"
        if output.exists():
            raise RuntimeError(f"result already exists: {output}")
        case = root / "cases" / f"seed_{seed}"
        instance = load_instance(case / "instance.json")
        scenarios = load_scenarios(case / "scenarios.json")
        finite.RHO = float(config["rho"])
        anchor = finite.solve_cost_anchor(instance, scenarios)
        direct = finite.solve_direct_fairness(instance, scenarios, anchor)
        methods = {}
        for method in orders[position]:
            methods[method] = solve_core(
                instance, anchor, int(seed),
                HYBRID_V8_CANDIDATE_SHA256 if method == "v8" else CANDIDATE_SHA256,
                method, config, manifest,
            )
        save_json(output, {
            "seed": seed,
            "run_order": list(orders[position]),
            "anchor": anchor,
            "direct": direct,
            "methods": methods,
        })
        print(json.dumps({
            "seed": seed,
            "direct_t": direct["objective_t"],
            "v8_t": methods["v8"]["objective_t"],
            "legacy_t": methods["legacy"]["objective_t"],
            "v8_runtime": methods["v8"]["runtime"],
            "legacy_runtime": methods["legacy"]["runtime"],
        }), flush=True)


def summarize() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = REPO_ROOT / config["output_dir"]
    verify_inputs(root)
    rows = []
    for seed in config["seeds"]:
        payload = json.loads((root / "results" / f"seed_{seed}.json").read_text(encoding="utf-8"))
        direct_t = float(payload["direct"]["objective_t"])
        v8 = payload["methods"]["v8"]
        legacy = payload["methods"]["legacy"]
        rows.append({
            "seed": seed,
            "v8_runtime": v8["runtime"],
            "legacy_runtime": legacy["runtime"],
            "v8_certified": v8["certified"],
            "legacy_certified": legacy["certified"],
            "v8_direct_abs_error": abs(float(v8["objective_t"]) - direct_t),
            "legacy_direct_abs_error": abs(float(legacy["objective_t"]) - direct_t),
            "v8_iterations": v8["iterations"],
            "legacy_iterations": legacy["iterations"],
            "v8_scenario_blocks": v8["scenario_blocks"],
            "legacy_scenario_blocks": legacy["scenario_blocks"],
            "v8_farkas_cuts": v8["farkas_cuts"],
            "legacy_farkas_cuts": legacy["farkas_cuts"],
        })
    summary = {
        "status": "core_integration_pilot_complete",
        "cases": len(rows),
        "v8_certified": sum(row["v8_certified"] for row in rows),
        "legacy_certified": sum(row["legacy_certified"] for row in rows),
        "v8_wins_vs_legacy": sum(row["v8_runtime"] < row["legacy_runtime"] for row in rows),
        "v8_geometric_speedup_vs_legacy": math.exp(statistics.mean(
            math.log(row["legacy_runtime"] / row["v8_runtime"]) for row in rows
        )),
        "v8_max_direct_abs_error": max(row["v8_direct_abs_error"] for row in rows),
        "legacy_max_direct_abs_error": max(row["legacy_direct_abs_error"] for row in rows),
        "claim_limit": "development core-integration pilot; not a formal holdout",
        "rows": rows,
    }
    save_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "run", "summarize"), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.stage]()


if __name__ == "__main__":
    main()
