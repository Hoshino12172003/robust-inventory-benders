from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Literal

import gurobipy as gp
from gurobipy import GRB
import yaml

from .experiment_suite import INSTANCE_SIZES
from .instance import InventoryInstance, generate_instance
from .scenarios import DemandScenario, enumerate_budget_scenarios
from .status import gurobi_status_name


Mode = Literal["k1", "k2", "full"]
ROOT = Path(__file__).resolve().parents[1]


def build_eligibility(instance: InventoryInstance, mode: Mode) -> dict[int, tuple[int, ...]]:
    if mode == "full":
        return {r: tuple(instance.I) for r in instance.R}
    if mode not in ("k1", "k2"):
        raise ValueError(f"unsupported mode: {mode}")
    count = 1 if mode == "k1" else min(2, instance.num_warehouses)
    result = {}
    for r in instance.R:
        ranked = sorted(
            instance.I,
            key=lambda i: (
                math.fsum(instance.transport_cost[i][r][j] for j in instance.J)
                / instance.num_products,
                i,
            ),
        )
        result[r] = tuple(ranked[:count])
    return result


def _first_stage(instance, y, x):
    return gp.quicksum(instance.fixed_cost[i] * y[i] for i in instance.I) + gp.quicksum(
        instance.inventory_cost[i][j] * x[i, j] for i in instance.I for j in instance.J
    )


def _base_model(instance, *, fixed_y=None, fixed_x=None, settings: dict[str, Any]):
    model = gp.Model("fulfillment_flexibility")
    model.Params.OutputFlag = 0
    model.Params.Threads = int(settings["threads"])
    model.Params.Seed = int(settings["solver_seed"])
    model.Params.MIPGap = float(settings["mip_gap"])
    model.Params.FeasibilityTol = float(settings["feasibility_tolerance"])
    model.Params.TimeLimit = float(settings["time_limit"])
    if fixed_y is None:
        y = model.addVars(instance.I, vtype=GRB.BINARY, name="y")
        x = model.addVars(instance.I, instance.J, lb=0.0, name="x")
        for i in instance.I:
            model.addConstr(
                gp.quicksum(instance.volume[j] * x[i, j] for j in instance.J)
                <= instance.capacity[i] * y[i]
            )
            for j in instance.J:
                model.addConstr(x[i, j] <= instance.inventory_ub[i][j] * y[i])
        first = _first_stage(instance, y, x)
        model.addConstr(first <= instance.budget)
    else:
        y = {i: float(fixed_y[i]) for i in instance.I}
        x = {(i, j): float(fixed_x[i][j]) for i in instance.I for j in instance.J}
        first = _first_stage(instance, y, x)
    return model, y, x, first


def _add_recourse(model, instance, scenario, index, x, eligibility):
    q = model.addVars(instance.I, instance.R, instance.J, lb=0.0, name=f"q[{index}]")
    u = model.addVars(instance.R, instance.J, lb=0.0, name=f"u[{index}]")
    e = model.addVars(instance.J, lb=0.0, name=f"e[{index}]")
    for r in instance.R:
        eligible = set(eligibility[r])
        for j in instance.J:
            model.addConstr(gp.quicksum(q[i, r, j] for i in instance.I) + u[r, j] >= scenario.demand[r][j])
            for i in instance.I:
                if i not in eligible:
                    model.addConstr(q[i, r, j] == 0.0)
    for i in instance.I:
        for j in instance.J:
            model.addConstr(gp.quicksum(q[i, r, j] for r in instance.R) <= x[i, j])
    for j in instance.J:
        model.addConstr(
            gp.quicksum(u[r, j] for r in instance.R) - e[j]
            <= (1.0 - instance.service_level[j])
            * math.fsum(scenario.demand[r][j] for r in instance.R)
        )
    transport = gp.quicksum(
        instance.transport_cost[i][r][j] * q[i, r, j]
        for i in instance.I for r in instance.R for j in instance.J
    )
    shortage = gp.quicksum(
        instance.shortage_penalty[r][j] * u[r, j] for r in instance.R for j in instance.J
    )
    service = gp.quicksum(instance.service_penalty[j] * e[j] for j in instance.J)
    return {"q": q, "u": u, "e": e, "transport": transport, "shortage": shortage,
            "service": service, "cost": transport + shortage + service}


def _metrics(instance, scenarios, recourse, y, x, first_value, eligibility, objective_t):
    tol = 1.0e-7
    policies = []
    regional = {r: {"worst_shortage_rate": 0.0, "total_inbound": 0.0, "used": set()} for r in instance.R}
    for s, scenario in enumerate(scenarios):
        block = recourse[s]
        region_rates = []
        total_shortage = 0.0
        total_demand = 0.0
        for r in instance.R:
            demand = math.fsum(scenario.demand[r][j] for j in instance.J)
            shortage = math.fsum(block["u"][r, j].X for j in instance.J)
            rate = 0.0 if demand <= tol else shortage / demand
            region_rates.append(rate)
            total_shortage += shortage
            total_demand += demand
            regional[r]["worst_shortage_rate"] = max(regional[r]["worst_shortage_rate"], rate)
            for i in instance.I:
                inbound = math.fsum(block["q"][i, r, j].X for j in instance.J)
                regional[r]["total_inbound"] += inbound
                if inbound > tol:
                    regional[r]["used"].add(i)
        policies.append({
            "scenario": scenario.name,
            "recourse_cost": float(block["cost"].getValue()),
            "transportation_cost": float(block["transport"].getValue()),
            "shortage_cost": float(block["shortage"].getValue()),
            "service_violation_cost": float(block["service"].getValue()),
            "worst_shortage_rate": max(region_rates),
            "weighted_fill_rate": 1.0 - (total_shortage / total_demand if total_demand > tol else 0.0),
        })
    worst_cost = max(policies, key=lambda row: row["recourse_cost"])
    facility_cost = math.fsum(instance.fixed_cost[i] * float(y[i].X if hasattr(y[i], "X") else y[i]) for i in instance.I)
    inventory_cost = math.fsum(
        instance.inventory_cost[i][j] * float(x[i, j].X if hasattr(x[i, j], "X") else x[i, j])
        for i in instance.I for j in instance.J
    )
    return {
        "first_stage_cost": first_value,
        "warehouse_opening_cost": facility_cost,
        "inventory_cost": inventory_cost,
        "actual_robust_cost": first_value + worst_cost["recourse_cost"],
        "transportation_cost": worst_cost["transportation_cost"],
        "shortage_cost": worst_cost["shortage_cost"],
        "service_violation_cost": worst_cost["service_violation_cost"],
        "worst_region_shortage_rate": max(row["worst_shortage_rate"] for row in policies),
        "minimum_regional_fill_rate": 1.0 - max(row["worst_shortage_rate"] for row in policies),
        "minimum_weighted_fill_rate": min(row["weighted_fill_rate"] for row in policies),
        "number_opened_warehouses": sum(float(y[i].X if hasattr(y[i], "X") else y[i]) >= 0.5 for i in instance.I),
        "active_warehouse_region_arcs": sum(len(regional[r]["used"]) for r in instance.R),
        "objective_t_consistency_error": abs(max(row["worst_shortage_rate"] for row in policies) - objective_t),
        "regional": [
            {"region": r, "worst_shortage_rate": regional[r]["worst_shortage_rate"],
             "fill_rate": 1.0 - regional[r]["worst_shortage_rate"],
             "number_eligible_warehouses": len(eligibility[r]),
             "number_actually_used_warehouses": len(regional[r]["used"]),
             "total_inbound_shipment": regional[r]["total_inbound"]}
            for r in instance.R
        ],
    }


def solve_cost_anchor(instance, scenarios, mode: Mode, settings, *, fixed_y=None, fixed_x=None):
    start = time.perf_counter()
    eligibility = build_eligibility(instance, mode)
    model, y, x, first = _base_model(
        instance, fixed_y=fixed_y, fixed_x=fixed_x, settings=settings
    )
    theta = model.addVar(lb=0.0, name="theta")
    recourse = []
    for s, scenario in enumerate(scenarios):
        block = _add_recourse(model, instance, scenario, s, x, eligibility)
        model.addConstr(theta >= block["cost"])
        recourse.append(block)
    model.setObjective(first + theta, GRB.MINIMIZE)
    model.optimize()
    status = gurobi_status_name(model.Status)
    result = {"status": status, "certified": model.Status == GRB.OPTIMAL,
              "runtime": time.perf_counter() - start, "objective": None,
              "bound": None, "y_values": None, "x_values": None}
    if model.SolCount:
        result.update({"objective": float(model.ObjVal), "bound": float(model.ObjBound),
                       "y_values": [float(y[i].X if hasattr(y[i], "X") else y[i]) for i in instance.I],
                       "x_values": [[float(x[i, j].X if hasattr(x[i, j], "X") else x[i, j]) for j in instance.J] for i in instance.I]})
    model.dispose()
    return result


def solve_service(instance, scenarios, mode: Mode, baseline_cost: float, rho: float, settings,
                  *, fixed_y=None, fixed_x=None):
    start = time.perf_counter()
    eligibility = build_eligibility(instance, mode)
    model, y, x, first = _base_model(instance, fixed_y=fixed_y, fixed_x=fixed_x, settings=settings)
    t = model.addVar(lb=0.0, ub=1.0, name="T")
    budget = (1.0 + rho) * baseline_cost
    recourse = []
    for s, scenario in enumerate(scenarios):
        block = _add_recourse(model, instance, scenario, s, x, eligibility)
        model.addConstr(first + block["cost"] <= budget)
        for r in instance.R:
            demand = math.fsum(scenario.demand[r][j] for j in instance.J)
            model.addConstr(gp.quicksum(block["u"][r, j] for j in instance.J) <= t * demand)
        recourse.append(block)
    model.setObjective(t, GRB.MINIMIZE)
    model.optimize()
    status = gurobi_status_name(model.Status)
    result = {"status": status, "certified": model.Status == GRB.OPTIMAL,
              "runtime": time.perf_counter() - start, "objective_t": None, "bound": None,
              "cost_budget": budget, "y_values": None, "x_values": None, "metrics": None}
    if model.SolCount:
        t_value = float(t.X)
        y_values = [float(y[i].X if hasattr(y[i], "X") else y[i]) for i in instance.I]
        x_values = [[float(x[i, j].X if hasattr(x[i, j], "X") else x[i, j]) for j in instance.J] for i in instance.I]
        result.update({"objective_t": t_value, "bound": float(model.ObjBound),
                       "y_values": y_values, "x_values": x_values,
                       "metrics": _metrics(instance, scenarios, recourse, y, x,
                                           float(first.getValue()), eligibility, t_value)})
    model.dispose()
    return result


def _config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _instance(scale: str, seed: int) -> InventoryInstance:
    return generate_instance({"seed": seed, "instance": {**INSTANCE_SIZES[scale],
        "budget_factor": 0.68, "capacity_factor": 1.25}, "robust": {"gamma_target": 2}}, seed=seed)


def _write_json(path: Path, payload):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run(config_path: Path):
    cfg = _config(config_path)
    output = ROOT / cfg["output_dir"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    settings = {key: cfg[key] for key in ("time_limit", "mip_gap", "feasibility_tolerance", "threads", "solver_seed")}
    for scale in cfg["scales"]:
        for seed in cfg["seeds"]:
            instance = _instance(scale, seed)
            scenarios = enumerate_budget_scenarios(instance, cfg["gamma"], max_scenarios=cfg["max_scenarios"], exact_scenarios=True)
            anchors = {mode: solve_cost_anchor(instance, scenarios, mode, settings) for mode in cfg["modes"]}
            optimized = {mode: solve_service(instance, scenarios, mode, anchors[mode]["objective"], cfg["rho"], settings)
                         for mode in cfg["modes"]}
            full = optimized["full"]
            fixed_anchors = {
                mode: solve_cost_anchor(
                    instance, scenarios, mode, settings,
                    fixed_y=full["y_values"], fixed_x=full["x_values"]
                )
                for mode in cfg["modes"]
            }
            fixed_common_anchor = max(value["objective"] for value in fixed_anchors.values())
            fixed = {mode: solve_service(instance, scenarios, mode, fixed_common_anchor, cfg["rho"], settings,
                                         fixed_y=full["y_values"], fixed_x=full["x_values"])
                     for mode in cfg["modes"]}
            _write_json(output / "raw" / f"{scale}_seed{seed}.json", {
                "scale": scale, "seed": seed, "gamma": cfg["gamma"], "rho": cfg["rho"],
                "scenario_count": len(scenarios), "anchors": anchors,
                "reoptimized": optimized, "fixed_first_stage_anchors": fixed_anchors,
                "fixed_first_stage_common_anchor": fixed_common_anchor,
                "fixed_first_stage": fixed})


def summarize(config_path: Path):
    cfg = _config(config_path); output = ROOT / cfg["output_dir"]
    summary_rows, region_rows = [], []
    for path in sorted((output / "raw").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for evaluation in ("reoptimized", "fixed_first_stage"):
            for mode in cfg["modes"]:
                result = payload[evaluation][mode]; metrics = result["metrics"]
                row = {"scale": payload["scale"], "seed": payload["seed"], "evaluation": evaluation,
                       "mode": mode, "rho": payload["rho"], "scenario_count": payload["scenario_count"],
                       "certified": result["certified"], "runtime": result["runtime"],
                       "objective_t": result["objective_t"], "objective_bound": result["bound"], **metrics}
                summary_rows.append(row)
                for region in metrics["regional"]:
                    region_rows.append({"scale": payload["scale"], "seed": payload["seed"],
                                        "evaluation": evaluation, "mode": mode, **region})
    paired = []
    for scale in cfg["scales"]:
        for seed in cfg["seeds"]:
            for evaluation in ("reoptimized", "fixed_first_stage"):
                rows = {row["mode"]: row for row in summary_rows if row["scale"] == scale and row["seed"] == seed and row["evaluation"] == evaluation}
                for left, right in (("k1", "k2"), ("k2", "full"), ("k1", "full")):
                    t_left, t_right = rows[left]["objective_t"], rows[right]["objective_t"]
                    paired.append({"scale": scale, "seed": seed, "evaluation": evaluation,
                                   "comparison": f"{left}_vs_{right}",
                                   "absolute_shortage_reduction": t_left - t_right,
                                   "relative_shortage_reduction": None if abs(t_left) <= 1e-12 else (t_left - t_right) / t_left,
                                   "relative_cost_change": (rows[right]["actual_robust_cost"] - rows[left]["actual_robust_cost"]) / rows[left]["actual_robust_cost"]})
    def write_csv(name, rows):
        with (output / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_csv("summary.csv", [{k: v for k, v in row.items() if k != "regional"} for row in summary_rows])
    write_csv("regional_diagnostic.csv", region_rows); write_csv("paired_comparisons.csv", paired)
    primary = [row for row in paired if row["evaluation"] == "reoptimized" and row["comparison"] == "k1_vs_full"]
    relative = [row["relative_shortage_reduction"] for row in primary if row["relative_shortage_reduction"] is not None]
    positive = sum(row["absolute_shortage_reduction"] > 1e-7 for row in primary)
    median = statistics.median(relative) if relative else None
    if len(primary) < len(cfg["scales"]) * len(cfg["seeds"]): recommendation = "inconclusive"
    elif positive >= cfg["screening"]["majority_required"] and median is not None and median >= cfg["screening"]["strong_median_relative_reduction"]: recommendation = "proceed_to_formal_flexibility_experiment"
    elif median is not None and median < cfg["screening"]["negligible_median_relative_reduction"]: recommendation = "do_not_proceed"
    else: recommendation = "inconclusive"
    result = {"recommendation": recommendation, "positive_seed_count": positive,
              "pooled_median_relative_reduction": median, "seed_count": len(primary)}
    _write_json(output / "diagnostic_decision.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("run", "summarize", "all"), required=True); args = parser.parse_args()
    if args.stage in ("run", "all"): run(args.config)
    if args.stage in ("summarize", "all"): print(json.dumps(summarize(args.config), indent=2))


if __name__ == "__main__": main()
