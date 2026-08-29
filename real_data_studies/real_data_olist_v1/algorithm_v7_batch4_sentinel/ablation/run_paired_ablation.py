from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
HOLDOUT = STUDY_ROOT / "algorithm_v4_holdout_validation"
V7_ROOT = STUDY_ROOT / "algorithm_v7_batch4_sentinel"
for path in (REPO_ROOT, HOLDOUT, V7_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_batch4_ccg import solve as solve_batch4  # noqa: E402
from run_hybrid_v7 import solve as solve_v7  # noqa: E402
from run_validation import load_scenarios  # noqa: E402
from src.instance import load_instance  # noqa: E402


def main() -> None:
    catalog = json.loads((HOLDOUT / "case_catalog.json").read_text(encoding="utf-8"))
    output = HERE / "paired_results"
    output.mkdir(parents=True, exist_ok=True)
    for position, case in enumerate(catalog):
        case_dir = HOLDOUT / "cases" / case["case_id"]
        instance = load_instance(case_dir / "instance.json")
        scenarios = load_scenarios(case_dir / "scenarios.json")
        anchor = json.loads((case_dir / "cost_anchor.json").read_text(encoding="utf-8"))
        order = ["hybrid_v7", "batch4_ccg"] if position % 2 == 0 else ["batch4_ccg", "hybrid_v7"]
        payload = {"case_id": case["case_id"], "run_order": order, "results": {}}
        for method in order:
            payload["results"][method] = (
                solve_v7(instance, scenarios, anchor)
                if method == "hybrid_v7"
                else solve_batch4(instance, scenarios, anchor)
            )
        (output / f"{case['case_id']}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"completed {case['case_id']}", flush=True)


if __name__ == "__main__":
    main()
