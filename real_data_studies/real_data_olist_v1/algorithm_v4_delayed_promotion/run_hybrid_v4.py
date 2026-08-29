from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
SCRIPTS = STUDY_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_factorized_finite_comparison import (  # noqa: E402
    FEAS_TOL,
    TIME_LIMIT,
    _add_fairness_cut,
    _build_master,
    _cut_key,
    _first_stage_expression,
    add_complete_scenario_block,
    certify_fixed_scenario_fairness_feasibility,
    fairness_cut_from_ray,
    load_scenarios,
    scenario_sha256,
)
from src.instance import load_instance  # noqa: E402
from src.status import gurobi_status_name  # noqa: E402


ROOT = STUDY_ROOT / "factorized_olist_v3"
OUTPUT = Path(__file__).resolve().parent / "results"
FROZEN_RHO = 0.0001
INITIAL_BLOCKS = tuple(range(7))
MAX_CUTS_PER_ITERATION = 10
PROMOTION_HITS = 2


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def solve(instance, scenarios, anchor: dict, max_promotions: int) -> dict:
    start = time.perf_counter()
    budget = (1.0 + FROZEN_RHO) * float(anchor["objective"])
    model, y, x, t = _build_master(instance, False)
    model.Params.Threads = 1
    model.Params.Seed = 0
    model.Params.FeasibilityTol = FEAS_TOL
    first_stage = _first_stage_expression(instance, y, x)
    blocked = set(INITIAL_BLOCKS)
    for index in INITIAL_BLOCKS:
        add_complete_scenario_block(
            model,
            instance,
            scenarios[index],
            x,
            t,
            first_stage=first_stage,
            cost_budget=budget,
            scenario_sha=scenario_sha256(instance, scenarios[index]),
        )
    for i in instance.I:
        y[i].Start = float(anchor["y_values"][i])
        for j in instance.J:
            x[i, j].Start = float(anchor["x_values"][i][j])
    t.Start = 1.0

    cut_keys = set()
    violation_hits = {index: 0 for index in range(len(scenarios))}
    log = []
    master_runtime = 0.0
    separation_runtime = 0.0
    status = "iteration_limit"
    best_t = 1.0
    best_y = []
    best_x = []

    for iteration in range(1, 501):
        remaining = TIME_LIMIT - (time.perf_counter() - start)
        if remaining <= 0.0:
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
        tick = time.perf_counter()
        for index, scenario in enumerate(scenarios):
            if index in blocked:
                continue
            remaining = TIME_LIMIT - (time.perf_counter() - start)
            if remaining <= 0.0:
                status = "time_limit"
                break
            certificate = certify_fixed_scenario_fairness_feasibility(
                instance,
                y_values=cy,
                x_values=cx,
                t_value=ct,
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
                    active_deviations=[
                        {"region": r, "product": j} for r, j in scenario.active_units
                    ],
                )
                violation = -float(cut.value(cy, cx, ct))
                if violation > FEAS_TOL:
                    violations.append((violation, index, cut))
            elif not certificate.primal_feasible:
                status = f"uncertified_{certificate.primal_status}"
                break
        separation_runtime += time.perf_counter() - tick
        if status == "time_limit" or status.startswith("uncertified_"):
            break

        violations.sort(key=lambda item: (-item[0], item[1]))
        if not violations:
            best_t = ct
            best_y = cy
            best_x = cx
            status = "optimal"
            log.append({
                "iteration": iteration,
                "t": ct,
                "violations": 0,
                "actions": [],
                "blocked_scenarios": len(blocked),
                "cuts": len(cut_keys),
            })
            break

        for _, index, _ in violations:
            violation_hits[index] += 1

        actions = []
        promoted = 0
        for _, index, _ in violations:
            if violation_hits[index] < PROMOTION_HITS:
                continue
            add_complete_scenario_block(
                model,
                instance,
                scenarios[index],
                x,
                t,
                first_stage=first_stage,
                cost_budget=budget,
                scenario_sha=scenario_sha256(instance, scenarios[index]),
            )
            blocked.add(index)
            actions.append(f"block:{index}")
            promoted += 1
            if promoted >= max_promotions:
                break

        added_cuts = 0
        for _, index, cut in violations:
            if index in blocked:
                continue
            key = _cut_key(cut)
            if key in cut_keys:
                continue
            _add_fairness_cut(model, y, x, t, cut, len(cut_keys))
            cut_keys.add(key)
            actions.append(f"cut:{index}")
            added_cuts += 1
            if added_cuts >= MAX_CUTS_PER_ITERATION:
                break

        log.append({
            "iteration": iteration,
            "t": ct,
            "violations": len(violations),
            "maximum_violation": violations[0][0],
            "actions": actions,
            "blocked_scenarios": len(blocked),
            "cuts": len(cut_keys),
        })

    result = {
        "candidate": "hybrid_v4_bounded_promotion",
        "status": status,
        "runtime": time.perf_counter() - start,
        "objective_t": best_t,
        "robust_minimum_fill_rate": 1.0 - best_t,
        "iterations": len(log),
        "master_runtime": master_runtime,
        "separation_runtime": separation_runtime,
        "blocked_scenarios": len(blocked),
        "cuts": len(cut_keys),
        "robust_feasibility_certified": status == "optimal",
        "cost_budget": budget,
        "rho": FROZEN_RHO,
        "policy": {
            "initial_blocks": list(INITIAL_BLOCKS),
            "max_cuts_per_iteration": MAX_CUTS_PER_ITERATION,
            "promotion_hits": PROMOTION_HITS,
            "max_promotions_per_iteration": max_promotions,
        },
        "solver_parameters": {
            "Threads": 1,
            "Seed": 0,
            "FeasibilityTol": FEAS_TOL,
            "MIPGap": 0.0,
        },
        "y_values": best_y,
        "x_values": best_x,
        "iteration_log": log,
    }
    model.dispose()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", choices=["20", "30"], required=True)
    parser.add_argument("--max-promotions", choices=[1, 2, 4], type=int, required=True)
    args = parser.parse_args()
    instance = load_instance(ROOT / "instances" / f"city_hubs_{args.cities}.json")
    scenarios = load_scenarios(instance)
    anchor_name = "cost_anchor.json" if args.cities == "20" else "cost_anchor_city30.json"
    anchor = json.loads((ROOT / "results" / anchor_name).read_text(encoding="utf-8"))
    result = solve(instance, scenarios, anchor, args.max_promotions)
    output_path = OUTPUT / f"hybrid_v4_city{args.cities}_p{args.max_promotions}_dev.json"
    save(output_path, result)
    print(json.dumps({
        key: result[key]
        for key in (
            "status",
            "runtime",
            "objective_t",
            "iterations",
            "master_runtime",
            "separation_runtime",
            "blocked_scenarios",
            "cuts",
            "robust_feasibility_certified",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
