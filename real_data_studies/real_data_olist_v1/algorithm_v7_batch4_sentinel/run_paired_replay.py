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
from run_hybrid_v7 import solve as solve_v7  # noqa: E402
from run_validation import load_scenarios  # noqa: E402
from src.instance import load_instance  # noqa: E402


def main() -> None:
    finite.RHO = 0.0001
    catalog = json.loads((HOLDOUT / "case_catalog.json").read_text(encoding="utf-8"))
    output = HERE / "paired_replay"
    output.mkdir(parents=True, exist_ok=True)
    for case in catalog:
        case_dir = HOLDOUT / "cases" / case["case_id"]
        instance = load_instance(case_dir / "instance.json")
        scenarios = load_scenarios(case_dir / "scenarios.json")
        anchor = json.loads((case_dir / "cost_anchor.json").read_text(encoding="utf-8"))
        payload = {
            "case_id": case["case_id"],
            "run_order": ["hybrid_v7" if item == "hybrid_v4" else item for item in case["run_order"]],
            "results": {},
        }
        for method in case["run_order"]:
            if method == "hybrid_v4":
                result = solve_v7(instance, scenarios, anchor)
                output_name = "hybrid_v7"
            else:
                result = finite.solve_fairness(instance, scenarios, anchor, "pure_ccg")
                output_name = "pure_ccg"
            payload["results"][output_name] = result
        (output / f"{case['case_id']}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"completed {case['case_id']}", flush=True)


if __name__ == "__main__":
    main()
