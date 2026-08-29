from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
REPO_ROOT = STUDY_ROOT.parents[1]
SCRIPTS = STUDY_ROOT / "scripts"
V4_ROOT = STUDY_ROOT / "algorithm_v4_delayed_promotion"
for path in (REPO_ROOT, SCRIPTS, V4_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_factorized_finite_comparison as finite  # noqa: E402
from run_hybrid_v4 import solve as solve_v4  # noqa: E402
from src.instance import InventoryInstance, load_instance, save_instance  # noqa: E402
from src.scenarios import DemandScenario  # noqa: E402


SELECTED_INDICES = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15)
GAMMA = 3
RHO = 0.0001
CASES = HERE / "cases"


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_scenarios(instance, membership: pd.DataFrame) -> list[DemandScenario]:
    factor_cells = {
        factor: tuple(
            sorted(
                (r, j)
                for r in instance.R
                for j in instance.J
                if int(membership.iloc[r * instance.num_products + j]["factor"]) == factor
            )
        )
        for factor in range(1, 7)
    }
    scenarios = []
    for size in range(GAMMA + 1):
        for active in combinations(range(1, 7), size):
            cells = tuple(sorted({cell for factor in active for cell in factor_cells[factor]}))
            demand = [list(row) for row in instance.base_demand]
            for r, j in cells:
                demand[r][j] += float(instance.demand_deviation[r][j])
            name = "nominal" if not active else "factors_" + "_".join(map(str, active))
            scenarios.append(DemandScenario(
                name=name,
                active_units=cells,
                demand=tuple(tuple(float(value) for value in row) for row in demand),
            ))
    if len(scenarios) != 42:
        raise RuntimeError(f"expected 42 scenarios, found {len(scenarios)}")
    return scenarios


def prepare() -> None:
    weekly = pd.read_csv(STUDY_ROOT / "processed" / "weekly_demand.csv")
    test = weekly.loc[weekly["split"].eq("test")].copy()
    weeks = sorted(test["week_start"].unique())
    if len(weeks) != 17:
        raise RuntimeError(f"expected 17 test weeks, found {len(weeks)}")
    design = json.loads(
        (STUDY_ROOT / "factorized_olist_v3" / "configs" / "factor_design.json").read_text(encoding="utf-8")
    )
    products = design["products"]
    regions = design["regions"]
    membership = pd.read_csv(
        STUDY_ROOT / "factorized_olist_v3" / "processed" / "factor_membership.csv"
    ).set_index(["region", "product"])
    template = load_instance(
        STUDY_ROOT / "factorized_olist_v3" / "instances" / "city_hubs_20.json"
    )
    catalog = []
    for position, index in enumerate(SELECTED_INDICES):
        week = weeks[index]
        observed = test.loc[test["week_start"].eq(week)].set_index(["region", "product"])
        base_demand = [
            [float(observed.loc[(region, product), "demand_units"]) for product in products]
            for region in regions
        ]
        deviation = [
            [float(membership.loc[(region, product), "upward_scale"]) for product in products]
            for region in regions
        ]
        payload = template.to_dict()
        payload["name"] = f"olist_factor_holdout_{week}"
        payload["base_demand"] = base_demand
        payload["demand_deviation"] = deviation
        instance = InventoryInstance.from_dict(payload)
        case_id = f"week_{week}"
        case_dir = CASES / case_id
        save_instance(instance, case_dir / "instance.json")
        scenarios = build_scenarios(instance, membership.reset_index())
        save_json(case_dir / "scenarios.json", [
            {
                "name": scenario.name,
                "active_units": [list(cell) for cell in scenario.active_units],
                "demand": [list(row) for row in scenario.demand],
            }
            for scenario in scenarios
        ])
        catalog.append({
            "case_id": case_id,
            "week_start": week,
            "test_week_index": index,
            "run_order": ["hybrid_v4", "pure_ccg"] if position % 2 == 0 else ["pure_ccg", "hybrid_v4"],
            "observed_total_demand": sum(sum(row) for row in base_demand),
            "scenario_count": len(scenarios),
        })
    save_json(HERE / "case_catalog.json", catalog)
    generated_inputs = []
    for case in catalog:
        for filename in ("instance.json", "scenarios.json"):
            path = CASES / case["case_id"] / filename
            generated_inputs.append({
                "path": str(path.relative_to(HERE)).replace("\\", "/"),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            })
    save_json(HERE / "generated_input_freeze.json", {
        "status": "generated inputs frozen before any optimization run",
        "file_count": len(generated_inputs),
        "files": generated_inputs,
    })


def load_scenarios(path: Path) -> list[DemandScenario]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [DemandScenario(
        name=row["name"],
        active_units=tuple(tuple(int(value) for value in cell) for cell in row["active_units"]),
        demand=tuple(tuple(float(value) for value in line) for line in row["demand"]),
    ) for row in raw]


def run() -> None:
    finite.RHO = RHO
    catalog = json.loads((HERE / "case_catalog.json").read_text(encoding="utf-8"))
    for case in catalog:
        case_dir = CASES / case["case_id"]
        instance = load_instance(case_dir / "instance.json")
        scenarios = load_scenarios(case_dir / "scenarios.json")
        anchor_path = case_dir / "cost_anchor.json"
        anchor = finite.solve_cost_anchor(instance, scenarios)
        save_json(anchor_path, anchor)
        for method in case["run_order"]:
            if method == "hybrid_v4":
                result = solve_v4(instance, scenarios, anchor, 4)
            else:
                result = finite.solve_fairness(instance, scenarios, anchor, "pure_ccg")
            save_json(case_dir / f"{method}.json", result)
        direct = finite.solve_direct_fairness(instance, scenarios, anchor)
        save_json(case_dir / "direct.json", direct)
        print(f"completed {case['case_id']}")


def summarize() -> None:
    catalog = json.loads((HERE / "case_catalog.json").read_text(encoding="utf-8"))
    rows = []
    for case in catalog:
        case_dir = CASES / case["case_id"]
        hybrid = json.loads((case_dir / "hybrid_v4.json").read_text(encoding="utf-8"))
        pure = json.loads((case_dir / "pure_ccg.json").read_text(encoding="utf-8"))
        direct = json.loads((case_dir / "direct.json").read_text(encoding="utf-8"))
        rows.append({
            **case,
            "hybrid_status": hybrid["status"],
            "pure_status": pure["status"],
            "direct_status": direct["status"],
            "hybrid_runtime": hybrid["runtime"],
            "pure_runtime": pure["runtime"],
            "direct_runtime": direct["runtime"],
            "hybrid_t": hybrid["objective_t"],
            "pure_t": pure["objective_t"],
            "direct_t": direct["objective_t"],
            "hybrid_iterations": hybrid["iterations"],
            "pure_iterations": pure["iterations"],
            "hybrid_blocks": hybrid["blocked_scenarios"],
            "pure_blocks": pure["blocked_scenarios"],
            "hybrid_cuts": hybrid["cuts"],
            "hybrid_certified": hybrid["robust_feasibility_certified"],
            "pure_certified": pure["robust_feasibility_certified"],
        })
    save_json(HERE / "validation_summary.json", rows)
    pd.DataFrame(rows).to_csv(HERE / "validation_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "run", "summarize"], required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.stage]()


if __name__ == "__main__":
    main()
