from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
REPO_ROOT = STUDY_ROOT.parents[1]
SCRIPTS = STUDY_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_factorized_finite_comparison as baseline  # noqa: E402
from run_hybrid_v4 import ROOT, load_scenarios, solve as solve_v4  # noqa: E402
from src.instance import load_instance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", choices=["20", "30"], default="20")
    args = parser.parse_args()
    baseline.RHO = 0.0001
    instance = load_instance(ROOT / "instances" / f"city_hubs_{args.cities}.json")
    scenarios = load_scenarios(instance)
    anchor_name = "cost_anchor.json" if args.cities == "20" else "cost_anchor_city30.json"
    anchor = json.loads((ROOT / "results" / anchor_name).read_text(encoding="utf-8"))
    trials = []
    for repeat in range(1, 4):
        order = ("v4", "pure_ccg") if repeat % 2 else ("pure_ccg", "v4")
        for method in order:
            if method == "v4":
                result = solve_v4(instance, scenarios, anchor, 4)
            else:
                result = baseline.solve_fairness(instance, scenarios, anchor, "pure_ccg")
            trials.append({
                "repeat": repeat,
                "order": list(order),
                "method": method,
                "status": result["status"],
                "runtime": result["runtime"],
                "objective_t": result["objective_t"],
                "iterations": result["iterations"],
                "master_runtime": result["master_runtime"],
                "separation_runtime": result["separation_runtime"],
                "blocked_scenarios": result["blocked_scenarios"],
                "cuts": result["cuts"],
                "certified": result["robust_feasibility_certified"],
            })
    payload = {
        "purpose": f"runtime stability audit on the {args.cities}-city instance",
        "policy": "frozen v4 p4 versus unchanged pure CCG implementation",
        "test_split_used": False,
        "trials": trials,
    }
    output = HERE / "results" / f"paired_repeats_city{args.cities}_dev.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(trials, indent=2))


if __name__ == "__main__":
    main()
