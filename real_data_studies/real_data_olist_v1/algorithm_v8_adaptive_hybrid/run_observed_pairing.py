from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
REPO_ROOT = STUDY_ROOT.parents[1]
HOLDOUT = STUDY_ROOT / "algorithm_v4_holdout_validation"
BATCH4_ROOT = STUDY_ROOT / "algorithm_v7_batch4_sentinel" / "ablation"
for path in (REPO_ROOT, HOLDOUT, BATCH4_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_batch4_ccg import solve as solve_batch4  # noqa: E402
from run_hybrid_v8 import solve as solve_v8  # noqa: E402
from run_validation import load_scenarios  # noqa: E402
from src.instance import load_instance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append", dest="case_ids")
    args = parser.parse_args()
    selected = set(args.case_ids or [])
    catalog = json.loads((HOLDOUT / "case_catalog.json").read_text(encoding="utf-8"))
    output = HERE / "observed_pairing"
    output.mkdir(parents=True, exist_ok=True)
    for position, case in enumerate(catalog):
        if selected and case["case_id"] not in selected:
            continue
        case_dir = HOLDOUT / "cases" / case["case_id"]
        instance = load_instance(case_dir / "instance.json")
        scenarios = load_scenarios(case_dir / "scenarios.json")
        anchor = json.loads((case_dir / "cost_anchor.json").read_text(encoding="utf-8"))
        order = ("v8", "batch4") if position % 2 == 0 else ("batch4", "v8")
        results = {}
        for method in order:
            results[method] = (
                solve_v8(instance, scenarios, anchor, selection_mode="max")
                if method == "v8"
                else solve_batch4(instance, scenarios, anchor)
            )
        payload = {"case_id": case["case_id"], "order": list(order), "results": results}
        (output / f"{case['case_id']}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps({
            "case_id": case["case_id"],
            "v8_runtime": results["v8"]["runtime"],
            "batch4_runtime": results["batch4"]["runtime"],
            "v8_certified": results["v8"]["robust_feasibility_certified"],
            "batch4_certified": results["batch4"]["robust_feasibility_certified"],
            "objective_difference": abs(results["v8"]["objective_t"] - results["batch4"]["objective_t"]),
        }), flush=True)


if __name__ == "__main__":
    main()
