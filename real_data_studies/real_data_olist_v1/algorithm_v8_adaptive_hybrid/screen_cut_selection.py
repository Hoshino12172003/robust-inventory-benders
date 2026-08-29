from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
REPO_ROOT = STUDY_ROOT.parents[1]
V5_ROOT = STUDY_ROOT / "algorithm_v5_cut_cleanup"
for path in (REPO_ROOT, V5_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_hybrid_v5 as v5  # noqa: E402
from run_hybrid_v8 import solve  # noqa: E402
from src.instance import load_instance  # noqa: E402


def main() -> None:
    output = HERE / "cut_selection_screening"
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for cities in ("20", "30"):
        instance = load_instance(v5.ROOT / "instances" / f"city_hubs_{cities}.json")
        scenarios = v5.load_scenarios(instance)
        anchor_name = "cost_anchor.json" if cities == "20" else "cost_anchor_city30.json"
        anchor = json.loads((v5.ROOT / "results" / anchor_name).read_text(encoding="utf-8"))
        for mode in ("l2", "max", "t"):
            result = solve(instance, scenarios, anchor, selection_mode=mode)
            (output / f"city{cities}_{mode}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            row = {
                "cities": int(cities),
                "selection_mode": mode,
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
