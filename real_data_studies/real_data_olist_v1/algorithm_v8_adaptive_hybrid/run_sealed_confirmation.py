from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
REPO_ROOT = STUDY_ROOT.parents[1]
SCRIPTS = STUDY_ROOT / "scripts"
HOLDOUT = STUDY_ROOT / "algorithm_v4_holdout_validation"
BATCH4_ROOT = STUDY_ROOT / "algorithm_v7_batch4_sentinel" / "ablation"
for path in (REPO_ROOT, SCRIPTS, HOLDOUT, BATCH4_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_factorized_finite_comparison as finite  # noqa: E402
from run_batch4_ccg import solve as solve_batch4  # noqa: E402
from run_hybrid_v8 import solve as solve_v8  # noqa: E402
from run_validation import build_scenarios, load_scenarios  # noqa: E402
from src.instance import InventoryInstance, load_instance, save_instance  # noqa: E402


UNUSED_INDICES = (0, 4, 8, 12, 16)
CASES = HERE / "sealed_confirmation" / "cases"
RESULTS = HERE / "sealed_confirmation" / "results"


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    template = load_instance(STUDY_ROOT / "factorized_olist_v3" / "instances" / "city_hubs_20.json")
    catalog = []
    for position, index in enumerate(UNUSED_INDICES):
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
        payload["name"] = f"olist_factor_sealed_{week}"
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
        orders = (
            ["v8", "pure_ccg", "batch4"],
            ["pure_ccg", "batch4", "v8"],
            ["batch4", "v8", "pure_ccg"],
        )
        catalog.append({
            "case_id": case_id,
            "week_start": week,
            "test_week_index": index,
            "run_order": orders[position % len(orders)],
            "scenario_count": len(scenarios),
        })
    save_json(HERE / "sealed_confirmation" / "case_catalog.json", catalog)
    files = []
    for case in catalog:
        for filename in ("instance.json", "scenarios.json"):
            path = CASES / case["case_id"] / filename
            files.append({
                "path": str(path.relative_to(HERE)).replace("\\", "/"),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            })
    save_json(HERE / "sealed_confirmation" / "input_freeze.json", {
        "status": "inputs_frozen_before_optimization",
        "file_count": len(files),
        "files": files,
    })


def run() -> None:
    finite.RHO = 0.0001
    catalog = json.loads((HERE / "sealed_confirmation" / "case_catalog.json").read_text(encoding="utf-8"))
    for case in catalog:
        case_dir = CASES / case["case_id"]
        instance = load_instance(case_dir / "instance.json")
        scenarios = load_scenarios(case_dir / "scenarios.json")
        anchor = finite.solve_cost_anchor(instance, scenarios)
        direct = finite.solve_direct_fairness(instance, scenarios, anchor)
        results = {}
        for method in case["run_order"]:
            if method == "v8":
                results[method] = solve_v8(instance, scenarios, anchor, selection_mode="max")
            elif method == "batch4":
                results[method] = solve_batch4(instance, scenarios, anchor)
            else:
                results[method] = finite.solve_fairness(instance, scenarios, anchor, "pure_ccg")
        save_json(RESULTS / f"{case['case_id']}.json", {
            "case_id": case["case_id"],
            "run_order": case["run_order"],
            "anchor": anchor,
            "direct": direct,
            "results": results,
        })
        print(json.dumps({
            "case_id": case["case_id"],
            "v8_runtime": results["v8"]["runtime"],
            "pure_runtime": results["pure_ccg"]["runtime"],
            "batch4_runtime": results["batch4"]["runtime"],
            "v8_certified": results["v8"]["robust_feasibility_certified"],
            "pure_certified": results["pure_ccg"]["robust_feasibility_certified"],
            "batch4_certified": results["batch4"]["robust_feasibility_certified"],
            "v8_cuts": results["v8"]["cuts_added_total"],
        }), flush=True)


def summarize() -> None:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        v8 = payload["results"]["v8"]
        pure = payload["results"]["pure_ccg"]
        batch4 = payload["results"]["batch4"]
        direct = payload["direct"]
        rows.append({
            "case_id": payload["case_id"],
            "run_order": "->".join(payload["run_order"]),
            "v8_runtime": v8["runtime"],
            "pure_runtime": pure["runtime"],
            "batch4_runtime": batch4["runtime"],
            "v8_certified": v8["robust_feasibility_certified"],
            "pure_certified": pure["robust_feasibility_certified"],
            "batch4_certified": batch4["robust_feasibility_certified"],
            "v8_cuts": v8["cuts_added_total"],
            "v8_iterations": v8["iterations"],
            "pure_iterations": pure["iterations"],
            "batch4_iterations": batch4["iterations"],
            "v8_direct_abs_error": abs(v8["objective_t"] - direct["objective_t"]),
            "pure_direct_abs_error": abs(pure["objective_t"] - direct["objective_t"]) if pure["robust_feasibility_certified"] else None,
            "batch4_direct_abs_error": abs(batch4["objective_t"] - direct["objective_t"]),
        })
    joint_pure = [row for row in rows if row["v8_certified"] and row["pure_certified"]]
    joint_batch = [row for row in rows if row["v8_certified"] and row["batch4_certified"]]

    def paired_summary(selected, other_key: str) -> dict:
        ratios = [row[other_key] / row["v8_runtime"] for row in selected]
        return {
            "jointly_certified": len(selected),
            "v8_wins": sum(row["v8_runtime"] < row[other_key] for row in selected),
            "v8_losses": sum(row["v8_runtime"] > row[other_key] for row in selected),
            "v8_mean_runtime": statistics.mean(row["v8_runtime"] for row in selected),
            "other_mean_runtime": statistics.mean(row[other_key] for row in selected),
            "geometric_mean_speedup": math.exp(statistics.mean(math.log(value) for value in ratios)),
        }

    summary = {
        "status": "sealed_confirmation_complete",
        "cases": len(rows),
        "certification": {
            "v8": sum(row["v8_certified"] for row in rows),
            "pure_ccg": sum(row["pure_certified"] for row in rows),
            "batch4": sum(row["batch4_certified"] for row in rows),
        },
        "v8_vs_pure_ccg": paired_summary(joint_pure, "pure_runtime"),
        "v8_vs_batch4": paired_summary(joint_batch, "batch4_runtime"),
        "v8_cases_with_farkas_cuts": sum(row["v8_cuts"] > 0 for row in rows),
        "v8_total_farkas_cuts": sum(row["v8_cuts"] for row in rows),
        "v8_max_direct_objective_abs_error": max(row["v8_direct_abs_error"] for row in rows),
    }
    save_json(HERE / "sealed_confirmation" / "summary.json", summary)
    with (HERE / "sealed_confirmation" / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "run", "summarize"), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.stage]()


if __name__ == "__main__":
    main()
