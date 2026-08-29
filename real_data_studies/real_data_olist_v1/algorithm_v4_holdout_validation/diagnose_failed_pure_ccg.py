from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
REPO_ROOT = STUDY_ROOT.parents[1]
SCRIPTS = STUDY_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_factorized_finite_comparison as finite  # noqa: E402
from run_validation import load_scenarios  # noqa: E402
from src.instance import load_instance  # noqa: E402


def main() -> None:
    finite.RHO = 0.0001
    case = HERE / "cases" / "week_2018-05-21"
    instance = load_instance(case / "instance.json")
    scenarios = load_scenarios(case / "scenarios.json")
    anchor = json.loads((case / "cost_anchor.json").read_text(encoding="utf-8"))
    repeats = [finite.solve_fairness(instance, scenarios, anchor, "pure_ccg") for _ in range(3)]
    payload = {
        "status": "post-hoc diagnostic; does not replace the preregistered primary run",
        "case_id": "week_2018-05-21",
        "method": "unchanged pure_ccg",
        "solver_parameters_unchanged": True,
        "repeats": repeats,
    }
    output = HERE / "diagnostics" / "pure_ccg_week_2018-05-21_repeats.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps([
        {
            "status": row["status"],
            "runtime": row["runtime"],
            "objective_t": row["objective_t"],
            "iterations": row["iterations"],
            "certified": row["robust_feasibility_certified"],
        }
        for row in repeats
    ], indent=2))


if __name__ == "__main__":
    main()
