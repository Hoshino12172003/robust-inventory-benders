from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
REPO_ROOT = STUDY_ROOT.parents[1]
SCRIPTS = STUDY_ROOT / "scripts"
HOLDOUT = STUDY_ROOT / "algorithm_v4_holdout_validation"
for path in (REPO_ROOT, SCRIPTS, HOLDOUT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_factorized_finite_comparison as finite  # noqa: E402
from run_hybrid_v8 import solve as solve_v8  # noqa: E402
from run_validation import load_scenarios  # noqa: E402
from src.instance import load_instance  # noqa: E402


def main() -> None:
    finite.RHO = 0.0001
    catalog = json.loads((HOLDOUT / "case_catalog.json").read_text(encoding="utf-8"))
    output = HERE / "v8_vs_pure_results"
    output.mkdir(parents=True, exist_ok=True)
    for position, case in enumerate(catalog):
        case_dir = HOLDOUT / "cases" / case["case_id"]
        instance = load_instance(case_dir / "instance.json")
        scenarios = load_scenarios(case_dir / "scenarios.json")
        anchor = json.loads((case_dir / "cost_anchor.json").read_text(encoding="utf-8"))
        order = ("v8", "pure_ccg") if position % 2 == 0 else ("pure_ccg", "v8")
        results = {}
        for method in order:
            results[method] = (
                solve_v8(instance, scenarios, anchor, selection_mode="max")
                if method == "v8"
                else finite.solve_fairness(instance, scenarios, anchor, "pure_ccg")
            )
        payload = {"case_id": case["case_id"], "run_order": list(order), "results": results}
        (output / f"{case['case_id']}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps({
            "case_id": case["case_id"],
            "v8_runtime": results["v8"]["runtime"],
            "pure_runtime": results["pure_ccg"]["runtime"],
            "v8_certified": results["v8"]["robust_feasibility_certified"],
            "pure_certified": results["pure_ccg"]["robust_feasibility_certified"],
            "v8_cuts": results["v8"]["cuts_added_total"],
            "objective_difference": abs(results["v8"]["objective_t"] - results["pure_ccg"]["objective_t"]),
        }), flush=True)


if __name__ == "__main__":
    main()
