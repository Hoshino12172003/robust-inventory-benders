from __future__ import annotations

import argparse
import csv
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
BATCH4_ROOT = STUDY_ROOT / "algorithm_v7_batch4_sentinel" / "ablation"
for path in (REPO_ROOT, SCRIPTS, BATCH4_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_factorized_finite_comparison as finite  # noqa: E402
from run_batch4_ccg import solve as solve_batch4  # noqa: E402
from run_hybrid_v8 import solve as solve_v8  # noqa: E402
from src.instance import generate_instance, load_instance, save_instance  # noqa: E402
from src.scenarios import DemandScenario, enumerate_budget_scenarios  # noqa: E402


CONFIG = REPO_ROOT / "experiments" / "configs" / "hybrid_v8_portable_pilot.json"


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
    files = []
    for seed in config["seeds"]:
        case = root / "cases" / f"seed_{seed}"
        instance = generate_instance(config, seed=int(seed))
        instance_path = save_instance(instance, case / "instance.json")
        scenarios = enumerate_budget_scenarios(
            instance,
            int(config["gamma"]),
            max_scenarios=int(config["expected_scenario_count"]),
            exact_scenarios=True,
        )
        if len(scenarios) != int(config["expected_scenario_count"]):
            raise RuntimeError(f"seed {seed}: expected 79 scenarios, found {len(scenarios)}")
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
        "candidate_source_sha256": sha256(HERE / "run_hybrid_v8.py"),
        "file_count": len(files),
        "files": files,
    })


def verify_inputs(root: Path) -> None:
    manifest = json.loads((root / "input_freeze.json").read_text(encoding="utf-8"))
    if manifest["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("pilot config changed after input freeze")
    if manifest["candidate_source_sha256"] != sha256(HERE / "run_hybrid_v8.py"):
        raise RuntimeError("Hybrid v8 candidate changed after input freeze")
    for row in manifest["files"]:
        if sha256(root / row["path"]) != row["sha256"]:
            raise RuntimeError(f"pilot input changed: {row['path']}")


def run() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = REPO_ROOT / config["output_dir"]
    verify_inputs(root)
    finite.RHO = float(config["rho"])
    orders = (
        ("hybrid_v8", "pure_ccg", "batch4_ccg"),
        ("pure_ccg", "batch4_ccg", "hybrid_v8"),
        ("batch4_ccg", "hybrid_v8", "pure_ccg"),
    )
    for position, seed in enumerate(config["seeds"]):
        case = root / "cases" / f"seed_{seed}"
        instance = load_instance(case / "instance.json")
        scenarios = load_scenarios(case / "scenarios.json")
        anchor = finite.solve_cost_anchor(instance, scenarios)
        direct = finite.solve_direct_fairness(instance, scenarios, anchor)
        results = {}
        for method in orders[position % len(orders)]:
            if method == "hybrid_v8":
                results[method] = solve_v8(instance, scenarios, anchor, selection_mode="max")
            elif method == "batch4_ccg":
                results[method] = solve_batch4(instance, scenarios, anchor)
            else:
                results[method] = finite.solve_fairness(instance, scenarios, anchor, "pure_ccg")
        save_json(root / "results" / f"seed_{seed}.json", {
            "seed": seed,
            "run_order": list(orders[position % len(orders)]),
            "anchor": anchor,
            "direct": direct,
            "results": results,
        })
        print(json.dumps({
            "seed": seed,
            "direct_t": direct["objective_t"],
            "v8_t": results["hybrid_v8"]["objective_t"],
            "v8_runtime": results["hybrid_v8"]["runtime"],
            "pure_runtime": results["pure_ccg"]["runtime"],
            "batch4_runtime": results["batch4_ccg"]["runtime"],
            "v8_cuts": results["hybrid_v8"]["cuts_added_total"],
        }), flush=True)


def summarize() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = REPO_ROOT / config["output_dir"]
    rows = []
    for path in sorted((root / "results").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        direct = payload["direct"]
        v8 = payload["results"]["hybrid_v8"]
        pure = payload["results"]["pure_ccg"]
        batch4 = payload["results"]["batch4_ccg"]
        rows.append({
            "seed": payload["seed"],
            "run_order": "->".join(payload["run_order"]),
            "v8_runtime": v8["runtime"],
            "pure_runtime": pure["runtime"],
            "batch4_runtime": batch4["runtime"],
            "v8_certified": v8["robust_feasibility_certified"],
            "pure_certified": pure["robust_feasibility_certified"],
            "batch4_certified": batch4["robust_feasibility_certified"],
            "v8_direct_abs_error": abs(v8["objective_t"] - direct["objective_t"]),
            "pure_direct_abs_error": abs(pure["objective_t"] - direct["objective_t"]),
            "batch4_direct_abs_error": abs(batch4["objective_t"] - direct["objective_t"]),
            "v8_iterations": v8["iterations"],
            "pure_iterations": pure["iterations"],
            "batch4_iterations": batch4["iterations"],
            "v8_blocks": v8["blocked_scenarios"],
            "pure_blocks": pure["blocked_scenarios"],
            "batch4_blocks": batch4["blocked_scenarios"],
            "v8_cuts": v8["cuts_added_total"],
        })
    if len(rows) != len(config["seeds"]):
        raise RuntimeError("pilot result matrix is incomplete")
    summary = {
        "status": "portable_pilot_complete",
        "cases": len(rows),
        "certification": {
            "hybrid_v8": sum(row["v8_certified"] for row in rows),
            "pure_ccg": sum(row["pure_certified"] for row in rows),
            "batch4_ccg": sum(row["batch4_certified"] for row in rows),
        },
        "v8_wins_vs_pure": sum(row["v8_runtime"] < row["pure_runtime"] for row in rows),
        "v8_wins_vs_batch4": sum(row["v8_runtime"] < row["batch4_runtime"] for row in rows),
        "v8_geometric_speedup_vs_pure": math.exp(statistics.mean(math.log(row["pure_runtime"] / row["v8_runtime"]) for row in rows)),
        "v8_geometric_speedup_vs_batch4": math.exp(statistics.mean(math.log(row["batch4_runtime"] / row["v8_runtime"]) for row in rows)),
        "v8_max_direct_abs_error": max(row["v8_direct_abs_error"] for row in rows),
        "v8_total_farkas_cuts": sum(row["v8_cuts"] for row in rows),
        "claim_limit": "development portability pilot; not an independent paper holdout",
    }
    save_json(root / "summary.json", summary)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "run", "summarize"), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.stage]()


if __name__ == "__main__":
    main()
