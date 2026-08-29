from __future__ import annotations

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


CANDIDATES = (
    (3, 1),
    (3, 4),
    (3, 10),
    (2, 4),
    (2, 10),
    (1, 10),
)


def solve_candidate(instance, scenarios, anchor: dict, blocks: int, cuts: int) -> dict:
    original = (
        v5.PROMOTION_HITS,
        v5.MAX_CUTS_PER_ITERATION,
        v5.MAX_PROMOTIONS_PER_ITERATION,
    )
    v5.PROMOTION_HITS = 1
    v5.MAX_CUTS_PER_ITERATION = cuts
    v5.MAX_PROMOTIONS_PER_ITERATION = blocks
    try:
        result = v5.solve(instance, scenarios, anchor)
    finally:
        (
            v5.PROMOTION_HITS,
            v5.MAX_CUTS_PER_ITERATION,
            v5.MAX_PROMOTIONS_PER_ITERATION,
        ) = original
    result["candidate"] = f"hybrid_v8_screen_b{blocks}_c{cuts}"
    result["screening_policy"] = {
        "scenario_blocks_per_iteration": blocks,
        "farkas_cuts_per_iteration": cuts,
        "promotion_hits": 1,
    }
    return result


def main() -> None:
    output = HERE / "screening"
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for cities in ("20", "30"):
        instance = load_instance(v5.ROOT / "instances" / f"city_hubs_{cities}.json")
        scenarios = v5.load_scenarios(instance)
        anchor_name = "cost_anchor.json" if cities == "20" else "cost_anchor_city30.json"
        anchor = json.loads((v5.ROOT / "results" / anchor_name).read_text(encoding="utf-8"))
        for blocks, cuts in CANDIDATES:
            result = solve_candidate(instance, scenarios, anchor, blocks, cuts)
            path = output / f"city{cities}_b{blocks}_c{cuts}.json"
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            row = {
                "cities": int(cities),
                "blocks_per_iteration": blocks,
                "cuts_per_iteration": cuts,
                "status": result["status"],
                "runtime": result["runtime"],
                "objective_t": result["objective_t"],
                "iterations": result["iterations"],
                "master_runtime": result["master_runtime"],
                "separation_runtime": result["separation_runtime"],
                "blocked_scenarios": result["blocked_scenarios"],
                "cuts_added_total": result["cuts_added_total"],
                "certified": result["robust_feasibility_certified"],
            }
            summary.append(row)
            print(json.dumps(row), flush=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
