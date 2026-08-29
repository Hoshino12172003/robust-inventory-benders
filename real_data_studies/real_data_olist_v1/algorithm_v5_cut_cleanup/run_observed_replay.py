from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
REPO_ROOT = STUDY_ROOT.parents[1]
HOLDOUT = STUDY_ROOT / "algorithm_v4_holdout_validation"
for path in (REPO_ROOT, HOLDOUT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_hybrid_v5 import solve  # noqa: E402
from run_validation import load_scenarios  # noqa: E402
from src.instance import load_instance  # noqa: E402


def main() -> None:
    catalog = json.loads((HOLDOUT / "case_catalog.json").read_text(encoding="utf-8"))
    output = HERE / "observed_replay"
    output.mkdir(parents=True, exist_ok=True)
    for case in catalog:
        case_dir = HOLDOUT / "cases" / case["case_id"]
        instance = load_instance(case_dir / "instance.json")
        scenarios = load_scenarios(case_dir / "scenarios.json")
        anchor = json.loads((case_dir / "cost_anchor.json").read_text(encoding="utf-8"))
        result = solve(instance, scenarios, anchor)
        (output / f"{case['case_id']}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"completed {case['case_id']}", flush=True)


if __name__ == "__main__":
    main()
