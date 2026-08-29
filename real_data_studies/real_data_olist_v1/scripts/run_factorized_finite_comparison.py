from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

from src.fairness_benders import _add_fairness_cut, _build_master, _cut_key
from src.fairness_hybrid_ccg_benders import add_complete_scenario_block, scenario_sha256
from src.instance import load_instance
from src.robust_regional_fairness import (
    _first_stage_expression,
    certify_fixed_scenario_fairness_feasibility,
    fairness_cut_from_ray,
)
from src.scenarios import DemandScenario
from src.status import gurobi_status_name


STUDY_ROOT = Path(__file__).resolve().parents[1]
ROOT = STUDY_ROOT / "factorized_olist_v3"
TIME_LIMIT = 900.0
RHO = 0.01
TOL = 1.0e-4
FEAS_TOL = 1.0e-7


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_scenarios(instance) -> list[DemandScenario]:
    raw = json.loads((ROOT / "instances" / "factor_scenarios.json").read_text(encoding="utf-8"))
    membership = pd.read_csv(ROOT / "processed" / "factor_membership.csv")
    factor_cells = {
        factor: tuple(
            sorted(
                (instance.R.start + row_index, instance.J.start + col_index)
                for row_index, region in enumerate(
                    json.loads((ROOT / "configs" / "factor_design.json").read_text(encoding="utf-8"))["regions"]
                )
                for col_index, product in enumerate(
                    json.loads((ROOT / "configs" / "factor_design.json").read_text(encoding="utf-8"))["products"]
                )
                if int(membership.loc[
                    membership["region"].eq(region) & membership["product"].eq(product), "factor"
                ].iloc[0]) == factor
            )
        )
        for factor in range(1, 7)
    }
    scenarios = []
    for row in raw:
        cells = tuple(sorted({cell for factor in row["active_factors"] for cell in factor_cells[int(factor)]}))
        scenarios.append(DemandScenario(
            name=row["scenario_id"],
            active_units=cells,
            demand=tuple(tuple(float(value) for value in line) for line in row["demand"]),
        ))
    return scenarios


def solve_cost_anchor(instance, scenarios: list[DemandScenario]) -> dict:
    start = time.perf_counter()
    model = gp.Model("factorized_cost_anchor")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.Seed = 0
    model.Params.MIPGap = TOL
    model.Params.TimeLimit = TIME_LIMIT
    y = model.addVars(instance.I, vtype=GRB.BINARY, name="y")
    x = model.addVars(instance.I, instance.J, lb=0.0, name="x")
    theta = model.addVar(lb=0.0, name="theta")
    first_stage = _first_stage_expression(instance, y, x)
    for i in instance.I:
        model.addConstr(gp.quicksum(instance.volume[j] * x[i, j] for j in instance.J) <= instance.capacity[i] * y[i])
        for j in instance.J:
            model.addConstr(x[i, j] <= instance.inventory_ub[i][j] * y[i])
    model.addConstr(first_stage <= instance.budget)
    for s, scenario in enumerate(scenarios):
        q = model.addVars(instance.I, instance.R, instance.J, lb=0.0, name=f"q[{s}]")
        u = model.addVars(instance.R, instance.J, lb=0.0, name=f"u[{s}]")
        e = model.addVars(instance.J, lb=0.0, name=f"e[{s}]")
        for r in instance.R:
            for j in instance.J:
                model.addConstr(gp.quicksum(q[i, r, j] for i in instance.I) + u[r, j] >= scenario.demand[r][j])
        for i in instance.I:
            for j in instance.J:
                model.addConstr(gp.quicksum(q[i, r, j] for r in instance.R) <= x[i, j])
        for j in instance.J:
            model.addConstr(
                gp.quicksum(u[r, j] for r in instance.R) - e[j]
                <= (1.0 - instance.service_level[j]) * sum(scenario.demand[r][j] for r in instance.R)
            )
        recourse = (
            gp.quicksum(instance.transport_cost[i][r][j] * q[i, r, j] for i in instance.I for r in instance.R for j in instance.J)
            + gp.quicksum(instance.shortage_penalty[r][j] * u[r, j] for r in instance.R for j in instance.J)
            + gp.quicksum(instance.service_penalty[j] * e[j] for j in instance.J)
        )
        model.addConstr(theta >= recourse)
    model.setObjective(first_stage + theta, GRB.MINIMIZE)
    model.optimize()
    payload = {
        "status": gurobi_status_name(model.Status),
        "runtime": time.perf_counter() - start,
        "objective": float(model.ObjVal) if model.SolCount else None,
        "lower_bound": float(model.ObjBound) if model.SolCount else None,
        "gap": float(model.MIPGap) if model.SolCount else None,
        "y_values": [float(y[i].X) for i in instance.I] if model.SolCount else [],
        "x_values": [[float(x[i, j].X) for j in instance.J] for i in instance.I] if model.SolCount else [],
        "scenario_count": len(scenarios),
    }
    model.dispose()
    if payload["status"] != "optimal" or payload["gap"] is None or payload["gap"] > TOL:
        raise RuntimeError(f"factorized cost anchor not certified: {payload['status']}, gap={payload['gap']}")
    return payload


def solve_fairness(
    instance,
    scenarios: list[DemandScenario],
    anchor: dict,
    method: str,
    *,
    promotion_hits: int = 2,
) -> dict:
    start = time.perf_counter()
    budget = (1.0 + RHO) * float(anchor["objective"])
    model, y, x, t = _build_master(instance, False)
    model.Params.Threads = 1
    model.Params.Seed = 0
    model.Params.FeasibilityTol = FEAS_TOL
    first_stage = _first_stage_expression(instance, y, x)
    initial_indices = {index for index, scenario in enumerate(scenarios) if len(scenario.name.split("_")) <= 2}
    # Explicit definition avoids relying on names for the nominal and six singleton factors.
    initial_indices = {0, 1, 2, 3, 4, 5, 6}
    blocked = set(initial_indices)
    for index in sorted(blocked):
        add_complete_scenario_block(
            model, instance, scenarios[index], x, t,
            first_stage=first_stage, cost_budget=budget,
            scenario_sha=scenario_sha256(instance, scenarios[index]),
        )
    for i in instance.I:
        y[i].Start = float(anchor["y_values"][i])
        for j in instance.J:
            x[i, j].Start = float(anchor["x_values"][i][j])
    t.Start = 1.0
    cut_keys = set()
    violation_seen = {index: 0 for index in range(len(scenarios))}
    master_runtime = 0.0
    separation_runtime = 0.0
    log = []
    status = "iteration_limit"
    best_t = 1.0
    for iteration in range(1, 501):
        remaining = TIME_LIMIT - (time.perf_counter() - start)
        if remaining <= 0:
            status = "time_limit"
            break
        model.Params.TimeLimit = remaining
        model.Params.MIPGap = 0.0
        tick = time.perf_counter()
        model.optimize()
        master_runtime += time.perf_counter() - tick
        if model.SolCount <= 0:
            status = gurobi_status_name(model.Status)
            break
        cy = [float(y[i].X) for i in instance.I]
        cx = [[float(x[i, j].X) for j in instance.J] for i in instance.I]
        ct = float(t.X)
        violations = []
        scan_indices = [index for index in range(len(scenarios)) if index not in blocked]
        tick = time.perf_counter()
        for position, index in enumerate(scan_indices):
            remaining = TIME_LIMIT - (time.perf_counter() - start)
            if remaining <= 0:
                break
            scenario = scenarios[index]
            certificate = certify_fixed_scenario_fairness_feasibility(
                instance,
                y_values=cy, x_values=cx, t_value=ct,
                cost_budget_value=budget,
                demand_values=[list(row) for row in scenario.demand],
                time_limit=remaining,
                feasibility_tolerance=FEAS_TOL,
                output_flag=False,
            )
            if certificate.infeasibility_certified and certificate.ray is not None:
                cut = fairness_cut_from_ray(
                    instance,
                    cost_budget_value=budget,
                    demand_values=[list(row) for row in scenario.demand],
                    ray=certificate.ray,
                    active_deviations=[{"region": r, "product": j} for r, j in scenario.active_units],
                )
                violation = -float(cut.value(cy, cx, ct))
                if violation > FEAS_TOL:
                    violations.append((violation, index, cut))
            elif not certificate.primal_feasible:
                status = f"uncertified_{certificate.primal_status}"
                break
        separation_runtime += time.perf_counter() - tick
        if status.startswith("uncertified_"):
            break
        violations.sort(key=lambda item: (-item[0], item[1]))
        actions = []
        if not violations:
            best_t = ct
            status = "optimal"
            log.append({"iteration": iteration, "t": ct, "violations": 0, "actions": []})
            break
        if method == "pure_ccg":
            _, index, _ = violations[0]
            add_complete_scenario_block(
                model, instance, scenarios[index], x, t,
                first_stage=first_stage, cost_budget=budget,
                scenario_sha=scenario_sha256(instance, scenarios[index]),
            )
            blocked.add(index)
            actions.append(f"block:{index}")
        else:
            for _, index, cut in violations[:10]:
                violation_seen[index] += 1
                if violation_seen[index] >= int(promotion_hits) and index not in blocked:
                    add_complete_scenario_block(
                        model, instance, scenarios[index], x, t,
                        first_stage=first_stage, cost_budget=budget,
                        scenario_sha=scenario_sha256(instance, scenarios[index]),
                    )
                    blocked.add(index)
                    actions.append(f"block:{index}")
                    continue
                key = _cut_key(cut)
                if key not in cut_keys:
                    _add_fairness_cut(model, y, x, t, cut, len(cut_keys))
                    cut_keys.add(key)
                    actions.append(f"cut:{index}")
        log.append({
            "iteration": iteration, "t": ct, "violations": len(violations),
            "maximum_violation": violations[0][0], "actions": actions,
            "blocked_scenarios": len(blocked), "cuts": len(cut_keys),
        })
    result = {
        "method": method, "status": status, "runtime": time.perf_counter() - start,
        "objective_t": best_t, "robust_minimum_fill_rate": 1.0 - best_t,
        "iterations": len(log), "master_runtime": master_runtime,
        "separation_runtime": separation_runtime, "blocked_scenarios": len(blocked),
        "cuts": len(cut_keys), "robust_feasibility_certified": status == "optimal",
        "cost_budget": budget, "promotion_hits": int(promotion_hits), "iteration_log": log,
    }
    model.dispose()
    return result


def solve_direct_fairness(instance, scenarios: list[DemandScenario], anchor: dict) -> dict:
    start = time.perf_counter()
    budget = (1.0 + RHO) * float(anchor["objective"])
    model, y, x, t = _build_master(instance, False)
    model.Params.Threads = 1
    model.Params.Seed = 0
    model.Params.FeasibilityTol = FEAS_TOL
    model.Params.MIPGap = 0.0
    model.Params.TimeLimit = TIME_LIMIT
    first_stage = _first_stage_expression(instance, y, x)
    for index, scenario in enumerate(scenarios):
        add_complete_scenario_block(
            model, instance, scenario, x, t,
            first_stage=first_stage, cost_budget=budget,
            scenario_sha=scenario_sha256(instance, scenario),
        )
    model.optimize()
    result = {
        "method": "direct_finite_scenarios",
        "status": gurobi_status_name(model.Status),
        "runtime": time.perf_counter() - start,
        "objective_t": float(t.X) if model.SolCount else None,
        "objective_bound": float(model.ObjBound) if model.SolCount else None,
        "gap": float(model.MIPGap) if model.SolCount else None,
        "scenario_count": len(scenarios),
    }
    model.dispose()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["anchor", "compare", "all"], default="all")
    args = parser.parse_args()
    instance = load_instance(ROOT / "instances" / "city_hubs_20.json")
    scenarios = load_scenarios(instance)
    anchor_path = ROOT / "results" / "cost_anchor.json"
    if args.stage in {"anchor", "all"} and not anchor_path.exists():
        save(anchor_path, solve_cost_anchor(instance, scenarios))
    if args.stage in {"compare", "all"}:
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
        for method in ("hybrid_factor", "pure_ccg"):
            path = ROOT / "results" / f"{method}.json"
            if not path.exists():
                save(path, solve_fairness(instance, scenarios, anchor, method))
    print(json.dumps({"stage": args.stage, "scenario_count": len(scenarios)}, indent=2))


if __name__ == "__main__":
    main()
