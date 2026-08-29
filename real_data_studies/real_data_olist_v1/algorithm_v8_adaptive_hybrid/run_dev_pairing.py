from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
REPO_ROOT = STUDY_ROOT.parents[1]
V5_ROOT = STUDY_ROOT / "algorithm_v5_cut_cleanup"
BATCH4_ROOT = STUDY_ROOT / "algorithm_v7_batch4_sentinel" / "ablation"
for path in (REPO_ROOT, V5_ROOT, BATCH4_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_hybrid_v5 as v5  # noqa: E402
from run_batch4_ccg import solve as solve_batch4  # noqa: E402
from run_hybrid_v8 import solve as solve_v8  # noqa: E402
from src.instance import load_instance  # noqa: E402


def main() -> None:
    output = HERE / "dev_pairing"
    output.mkdir(parents=True, exist_ok=True)
    pairs = []
    for cities in ("20", "30"):
        instance = load_instance(v5.ROOT / "instances" / f"city_hubs_{cities}.json")
        scenarios = v5.load_scenarios(instance)
        anchor_name = "cost_anchor.json" if cities == "20" else "cost_anchor_city30.json"
        anchor = json.loads((v5.ROOT / "results" / anchor_name).read_text(encoding="utf-8"))
        for repetition in range(6):
            order = ("v8", "batch4") if repetition % 2 == 0 else ("batch4", "v8")
            results = {}
            for method in order:
                results[method] = (
                    solve_v8(instance, scenarios, anchor, selection_mode="max")
                    if method == "v8"
                    else solve_batch4(instance, scenarios, anchor)
                )
            pair = {
                "cities": int(cities),
                "repetition": repetition + 1,
                "order": list(order),
                "results": results,
            }
            pairs.append(pair)
            (output / f"city{cities}_rep{repetition + 1}.json").write_text(
                json.dumps(pair, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(json.dumps({
                "cities": int(cities),
                "repetition": repetition + 1,
                "v8_runtime": results["v8"]["runtime"],
                "batch4_runtime": results["batch4"]["runtime"],
            }), flush=True)
    summary = []
    for cities in (20, 30):
        selected = [pair for pair in pairs if pair["cities"] == cities]
        v8_times = [pair["results"]["v8"]["runtime"] for pair in selected]
        batch_times = [pair["results"]["batch4"]["runtime"] for pair in selected]
        summary.append({
            "cities": cities,
            "pairs": len(selected),
            "v8_median_runtime": statistics.median(v8_times),
            "batch4_median_runtime": statistics.median(batch_times),
            "v8_wins": sum(v8 < batch for v8, batch in zip(v8_times, batch_times)),
            "max_objective_difference": max(
                abs(pair["results"]["v8"]["objective_t"] - pair["results"]["batch4"]["objective_t"])
                for pair in selected
            ),
        })
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
