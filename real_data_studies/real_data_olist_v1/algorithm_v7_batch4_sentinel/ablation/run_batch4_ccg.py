from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
V6_ROOT = STUDY_ROOT / "algorithm_v6_ephemeral_probes"
for path in (REPO_ROOT, V6_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_hybrid_v6 as v6  # noqa: E402
from src.instance import load_instance  # noqa: E402


def solve(instance, scenarios, anchor: dict) -> dict:
    original_hits = v6.PROMOTION_HITS
    v6.PROMOTION_HITS = 1
    try:
        result = v6.solve(instance, scenarios, anchor)
    finally:
        v6.PROMOTION_HITS = original_hits
    result["candidate"] = "batch4_ccg_no_farkas_cuts"
    result["policy"] = {
        "initial_blocks": list(v6.INITIAL_BLOCKS),
        "max_scenario_blocks_per_iteration": v6.MAX_PROMOTIONS_PER_ITERATION,
        "promotion_hits": 1,
        "farkas_cuts_per_iteration": 0,
    }
    return result


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", choices=["20", "30"], required=True)
    args = parser.parse_args()
    instance = load_instance(v6.ROOT / "instances" / f"city_hubs_{args.cities}.json")
    scenarios = v6.load_scenarios(instance)
    anchor_name = "cost_anchor.json" if args.cities == "20" else "cost_anchor_city30.json"
    anchor = json.loads((v6.ROOT / "results" / anchor_name).read_text(encoding="utf-8"))
    result = solve(instance, scenarios, anchor)
    save(HERE / "results" / f"batch4_city{args.cities}_dev.json", result)
    print(json.dumps({
        key: result[key]
        for key in (
            "status",
            "runtime",
            "objective_t",
            "iterations",
            "master_runtime",
            "separation_runtime",
            "blocked_scenarios",
            "active_probe_cuts",
            "robust_feasibility_certified",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
