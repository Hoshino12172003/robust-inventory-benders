from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
REPO_ROOT = STUDY_ROOT.parents[1]
V5_ROOT = STUDY_ROOT / "algorithm_v5_cut_cleanup"
for path in (REPO_ROOT, V5_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_hybrid_v5 as v5  # noqa: E402
from src.instance import load_instance  # noqa: E402


OUTPUT = HERE / "results"


def solve(instance, scenarios, anchor: dict) -> dict:
    original = (
        v5.PROMOTION_HITS,
        v5.MAX_CUTS_PER_ITERATION,
        v5.MAX_PROMOTIONS_PER_ITERATION,
    )
    v5.PROMOTION_HITS = 1
    v5.MAX_CUTS_PER_ITERATION = 1
    v5.MAX_PROMOTIONS_PER_ITERATION = 4
    try:
        result = v5.solve(instance, scenarios, anchor)
    finally:
        (
            v5.PROMOTION_HITS,
            v5.MAX_CUTS_PER_ITERATION,
            v5.MAX_PROMOTIONS_PER_ITERATION,
        ) = original
    result["candidate"] = "hybrid_v7_batch4_plus_one_sentinel_cut"
    result["policy"]["direct_first_round_promotion"] = True
    result["policy"]["sentinel_cut_definition"] = "most violated non-promoted scenario"
    return result


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", choices=["20", "30"], required=True)
    args = parser.parse_args()
    instance = load_instance(v5.ROOT / "instances" / f"city_hubs_{args.cities}.json")
    scenarios = v5.load_scenarios(instance)
    anchor_name = "cost_anchor.json" if args.cities == "20" else "cost_anchor_city30.json"
    anchor = json.loads((v5.ROOT / "results" / anchor_name).read_text(encoding="utf-8"))
    result = solve(instance, scenarios, anchor)
    save(OUTPUT / f"hybrid_v7_city{args.cities}_dev.json", result)
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
            "active_cuts",
            "cuts_added_total",
            "cuts_removed_total",
            "robust_feasibility_certified",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
