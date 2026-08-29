from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import sys

import gurobipy as gp
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
STUDY_ROOT = REPO_ROOT / "real_data_studies/real_data_olist_v1"
SCRIPTS = STUDY_ROOT / "scripts"
V8_ROOT = STUDY_ROOT / "algorithm_v8_adaptive_hybrid"
BATCH4_ROOT = STUDY_ROOT / "algorithm_v7_batch4_sentinel/ablation"
for path in (REPO_ROOT, SCRIPTS, V8_ROOT, BATCH4_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_batch4_ccg as batch4  # noqa: E402
import run_factorized_finite_comparison as finite  # noqa: E402
import run_hybrid_v8 as hybrid_v8  # noqa: E402
from src.instance import load_instance  # noqa: E402
from src.scenarios import DemandScenario  # noqa: E402


CONFIG = REPO_ROOT / "experiments/configs/hybrid_v8_m5_external_holdout.json"
INPUT_ROOT = HERE / "hybrid_v8_formal_holdout"
RESULTS = INPUT_ROOT / "results"
METHODS = ("hybrid_v8", "pure_ccg", "batch4_ccg", "direct")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    temporary.replace(path)


def method_order(case_index: int) -> tuple[str, ...]:
    offset = case_index % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def verify_inputs() -> tuple[dict, list[dict]]:
    config = load_config()
    freeze = json.loads((INPUT_ROOT / "input_freeze.json").read_text(encoding="utf-8"))
    if freeze["config_sha256"].lower() != sha256(CONFIG):
        raise RuntimeError("M5 configuration changed after input freeze")
    candidate = REPO_ROOT / config["candidate"]["source"]
    if config["candidate"]["source_sha256"].lower() != sha256(candidate):
        raise RuntimeError("Hybrid v8 candidate changed after preregistration")
    for row in freeze["files"]:
        path = INPUT_ROOT / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"M5 processed input changed: {row['path']}")
    catalog = json.loads((INPUT_ROOT / "case_catalog.json").read_text(encoding="utf-8"))
    if len(catalog) != int(config["temporal_split"]["expected_holdout_cases"]):
        raise RuntimeError("M5 formal case count drifted")
    for position, case in enumerate(catalog):
        if int(case["case_index"]) != position:
            raise RuntimeError("M5 case order drifted")
        scenario_rows = json.loads(
            (INPUT_ROOT / "cases" / case["case_id"] / "scenarios.json").read_text(encoding="utf-8")
        )
        if len(scenario_rows) != int(config["mapping"]["expected_scenario_count"]):
            raise RuntimeError(f"M5 scenario count drifted: {case['case_id']}")
    return config, catalog


def load_scenarios(path: Path) -> list[DemandScenario]:
    membership = pd.read_csv(INPUT_ROOT / "factor_membership.csv")
    stores = sorted(membership["store_id"].unique().tolist())
    departments = sorted(membership["dept_id"].unique().tolist())
    lookup = {
        (row.store_id, row.dept_id): int(row.factor)
        for row in membership.itertuples(index=False)
    }
    factor_cells = {
        factor: tuple(
            (r, j)
            for r, store in enumerate(stores)
            for j, department in enumerate(departments)
            if lookup[(store, department)] == factor
        )
        for factor in range(1, 7)
    }
    scenarios = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        cells = tuple(sorted({cell for factor in row["active_factors"] for cell in factor_cells[int(factor)]}))
        scenarios.append(DemandScenario(
            name=row["scenario_id"],
            active_units=cells,
            demand=tuple(tuple(float(value) for value in line) for line in row["demand"]),
        ))
    return scenarios


def configure_solver_modules(config: dict) -> None:
    seconds = float(config["solver"]["time_limit_seconds"])
    rho = float(config["calibrated_parameters"]["rho"])
    feasibility = float(config["solver"]["FeasibilityTol"])
    finite.TIME_LIMIT = seconds
    finite.RHO = rho
    finite.TOL = float(config["solver"]["final_MIPGap"])
    finite.FEAS_TOL = feasibility
    hybrid_v8.v5.TIME_LIMIT = seconds
    hybrid_v8.v5.FROZEN_RHO = rho
    hybrid_v8.v5.FEAS_TOL = feasibility
    batch4.v6.TIME_LIMIT = seconds
    batch4.v6.FROZEN_RHO = rho
    batch4.v6.FEAS_TOL = feasibility


def environment_snapshot(config: dict) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    return {
        "schema": "m5_formal_holdout_environment_v1",
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "gurobi_version": list(gp.gurobi.version()),
        "config_sha256": sha256(CONFIG),
        "input_freeze_sha256": sha256(INPUT_ROOT / "input_freeze.json"),
        "runner_sha256": sha256(Path(__file__)),
        "hybrid_v8_sha256": sha256(V8_ROOT / "run_hybrid_v8.py"),
        "pure_ccg_sha256": sha256(SCRIPTS / "run_factorized_finite_comparison.py"),
        "batch4_sha256": sha256(BATCH4_ROOT / "run_batch4_ccg.py"),
        "solver_controls": config["solver"],
        "method_orders": [list(method_order(index)) for index in range(4)],
    }


def solve_method(method: str, instance, scenarios: list[DemandScenario], anchor: dict) -> dict:
    if method == "hybrid_v8":
        return hybrid_v8.solve(instance, scenarios, anchor, selection_mode="max")
    if method == "pure_ccg":
        return finite.solve_fairness(instance, scenarios, anchor, "pure_ccg")
    if method == "batch4_ccg":
        return batch4.solve(instance, scenarios, anchor)
    if method == "direct":
        return finite.solve_direct_fairness(instance, scenarios, anchor)
    raise ValueError(f"unknown formal method: {method}")


def run_case(case: dict) -> None:
    case_id = case["case_id"]
    input_case = INPUT_ROOT / "cases" / case_id
    result_case = RESULTS / case_id
    instance = load_instance(input_case / "instance.json")
    scenarios = load_scenarios(input_case / "scenarios.json")
    anchor_path = result_case / "cost_anchor.json"
    if anchor_path.exists():
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    else:
        anchor = finite.solve_cost_anchor(instance, scenarios)
        save_json(anchor_path, anchor)
    order = method_order(int(case["case_index"]))
    plan = {
        "case_id": case_id,
        "case_index": case["case_index"],
        "method_order": list(order),
        "scenario_count": len(scenarios),
        "input_instance_sha256": sha256(input_case / "instance.json"),
        "input_scenarios_sha256": sha256(input_case / "scenarios.json"),
    }
    plan_path = result_case / "run_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError(f"formal run plan changed: {case_id}")
    else:
        save_json(plan_path, plan)
    for method in order:
        output = result_case / f"{method}.json"
        if output.exists():
            continue
        result = solve_method(method, instance, scenarios, anchor)
        save_json(output, result)
        print(json.dumps({
            "case_id": case_id,
            "method": method,
            "status": result["status"],
            "runtime": result["runtime"],
        }), flush=True)


def certified(method: str, result: dict) -> bool:
    if method == "direct":
        return result["status"] == "optimal" and result.get("gap") is not None and result["gap"] <= 1.0e-12
    return bool(result["robust_feasibility_certified"])


def par2(result: dict, is_certified: bool, penalty: float) -> float:
    return float(result["runtime"]) if is_certified else penalty


def exact_sign_test(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if not trials:
        return None
    tail = sum(math.comb(trials, value) for value in range(min(wins, losses) + 1)) / (2 ** trials)
    return min(1.0, 2.0 * tail)


def paired_summary(rows: list[dict], other: str, penalty: float) -> dict:
    hybrid_key = "hybrid_v8_par2"
    other_key = f"{other}_par2"
    wins = sum(row[hybrid_key] < row[other_key] for row in rows)
    losses = sum(row[hybrid_key] > row[other_key] for row in rows)
    ties = len(rows) - wins - losses
    joint = [row for row in rows if row["hybrid_v8_certified"] and row[f"{other}_certified"]]
    ratios = [row[f"{other}_runtime"] / row["hybrid_v8_runtime"] for row in joint]
    return {
        "cases": len(rows),
        "hybrid_wins_on_par2": wins,
        "hybrid_losses_on_par2": losses,
        "ties_on_par2": ties,
        "exact_two_sided_sign_test_p": exact_sign_test(wins, losses),
        "jointly_certified": len(joint),
        "hybrid_mean_raw_runtime_joint": statistics.mean(row["hybrid_v8_runtime"] for row in joint) if joint else None,
        "other_mean_raw_runtime_joint": statistics.mean(row[f"{other}_runtime"] for row in joint) if joint else None,
        "geometric_mean_other_over_hybrid_runtime_joint": (
            math.exp(statistics.mean(math.log(value) for value in ratios)) if ratios else None
        ),
        "par2_penalty_seconds": penalty,
    }


def summarize(config: dict, catalog: list[dict]) -> None:
    penalty = float(config["solver"]["PAR2_penalty_seconds"])
    rows = []
    for case in catalog:
        case_id = case["case_id"]
        values = {}
        for method in METHODS:
            path = RESULTS / case_id / f"{method}.json"
            if not path.is_file():
                raise RuntimeError(f"formal result matrix incomplete: {case_id}/{method}")
            values[method] = json.loads(path.read_text(encoding="utf-8"))
        row = {"case_id": case_id, "method_order": "->".join(method_order(int(case["case_index"])))}
        for method, result in values.items():
            is_certified = certified(method, result)
            row[f"{method}_runtime"] = float(result["runtime"])
            row[f"{method}_certified"] = is_certified
            row[f"{method}_par2"] = par2(result, is_certified, penalty)
        direct_ok = certified("direct", values["direct"])
        direct_t = values["direct"].get("objective_t")
        for method in ("hybrid_v8", "pure_ccg", "batch4_ccg"):
            objective = values[method].get("objective_t")
            row[f"{method}_direct_abs_error"] = (
                abs(float(objective) - float(direct_t))
                if certified(method, values[method]) and direct_ok and objective is not None and direct_t is not None
                else None
            )
        row["hybrid_v8_iterations"] = values["hybrid_v8"]["iterations"]
        row["pure_ccg_iterations"] = values["pure_ccg"]["iterations"]
        row["batch4_ccg_iterations"] = values["batch4_ccg"]["iterations"]
        row["hybrid_v8_blocks"] = values["hybrid_v8"]["blocked_scenarios"]
        row["pure_ccg_blocks"] = values["pure_ccg"]["blocked_scenarios"]
        row["batch4_ccg_blocks"] = values["batch4_ccg"]["blocked_scenarios"]
        row["hybrid_v8_farkas_cuts"] = values["hybrid_v8"]["cuts_added_total"]
        rows.append(row)
    objective_errors = [
        row[f"{method}_direct_abs_error"]
        for row in rows
        for method in ("hybrid_v8", "pure_ccg", "batch4_ccg")
        if row[f"{method}_direct_abs_error"] is not None
    ]
    objective_tolerance = float(config["objective_abs_tolerance"])
    summary = {
        "schema": "m5_formal_holdout_summary_v1",
        "status": "complete_all_preregistered_cases",
        "cases": len(rows),
        "certification": {
            method: sum(row[f"{method}_certified"] for row in rows)
            for method in METHODS
        },
        "hybrid_v8_vs_pure_ccg": paired_summary(rows, "pure_ccg", penalty),
        "hybrid_v8_vs_batch4_ccg": paired_summary(rows, "batch4_ccg", penalty),
        "direct_objective_check": {
            "absolute_tolerance": objective_tolerance,
            "certified_pairs_checked": len(objective_errors),
            "pairs_within_tolerance": sum(value <= objective_tolerance for value in objective_errors),
            "maximum_absolute_error": max(objective_errors) if objective_errors else None,
        },
        "hybrid_v8_total_farkas_cuts": sum(row["hybrid_v8_farkas_cuts"] for row in rows),
        "hybrid_v8_cases_with_farkas_cuts": sum(row["hybrid_v8_farkas_cuts"] > 0 for row in rows),
    }
    save_json(RESULTS / "summary.json", summary)
    with (RESULTS / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("verify", "run", "summarize"), required=True)
    parser.add_argument("--case", type=int, choices=range(1, 13))
    args = parser.parse_args()
    config, catalog = verify_inputs()
    configure_solver_modules(config)
    if args.stage == "verify":
        print(json.dumps(environment_snapshot(config), indent=2))
        return
    if args.stage == "run":
        if not (RESULTS / "environment.json").exists():
            save_json(RESULTS / "environment.json", environment_snapshot(config))
        selected = catalog if args.case is None else [catalog[args.case - 1]]
        for case in selected:
            run_case(case)
        return
    summarize(config, catalog)


if __name__ == "__main__":
    main()
