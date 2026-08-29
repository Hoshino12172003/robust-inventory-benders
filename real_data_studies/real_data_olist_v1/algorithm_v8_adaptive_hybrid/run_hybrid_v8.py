from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
REPO_ROOT = STUDY_ROOT.parents[1]
V5_ROOT = STUDY_ROOT / "algorithm_v5_cut_cleanup"
for path in (REPO_ROOT, V5_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_hybrid_v5 as v5  # noqa: E402
from src.status import gurobi_status_name  # noqa: E402


MAX_BLOCKS_PER_ITERATION = 4
MIN_NORMALIZED_CUT_EFFICACY = 0.10


def _coefficient_scale(cut, mode: str) -> float:
    coefficients = (
        [cut.constant, cut.t_coefficient]
        + list(cut.y_coefficients)
        + [value for row in cut.x_coefficients for value in row]
    )
    if mode == "l2":
        norm = math.sqrt(sum(value * value for value in coefficients))
    elif mode == "max":
        norm = max(abs(value) for value in coefficients)
    elif mode == "t":
        norm = abs(cut.t_coefficient)
    else:
        raise ValueError(f"unknown cut scaling mode: {mode}")
    return 1.0 / max(norm, 1.0e-12)


def _scaled_cut(cut, scale: float):
    return replace(
        cut,
        constant=cut.constant * scale,
        y_coefficients=[value * scale for value in cut.y_coefficients],
        x_coefficients=[[value * scale for value in row] for row in cut.x_coefficients],
        t_coefficient=cut.t_coefficient * scale,
    )


def solve(instance, scenarios, anchor: dict, *, selection_mode: str = "l2") -> dict:
    start = time.perf_counter()
    budget = (1.0 + v5.FROZEN_RHO) * float(anchor["objective"])
    model, y, x, t = v5._build_master(instance, False)
    model.Params.Threads = 1
    model.Params.Seed = 0
    model.Params.FeasibilityTol = v5.FEAS_TOL
    first_stage = v5._first_stage_expression(instance, y, x)
    blocked = set(v5.INITIAL_BLOCKS)
    for index in v5.INITIAL_BLOCKS:
        v5.add_complete_scenario_block(
            model,
            instance,
            scenarios[index],
            x,
            t,
            first_stage=first_stage,
            cost_budget=budget,
            scenario_sha=v5.scenario_sha256(instance, scenarios[index]),
        )
    for i in instance.I:
        y[i].Start = float(anchor["y_values"][i])
        for j in instance.J:
            x[i, j].Start = float(anchor["x_values"][i][j])
    t.Start = 1.0

    active_cut_keys = set()
    cuts_by_scenario = {index: [] for index in range(len(scenarios))}
    cut_serial = 0
    cuts_added_total = 0
    cuts_removed_total = 0
    master_runtime = 0.0
    separation_runtime = 0.0
    log = []
    status = "iteration_limit"
    best_t = 1.0
    best_y = []
    best_x = []

    for iteration in range(1, 501):
        remaining = v5.TIME_LIMIT - (time.perf_counter() - start)
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
            remaining = v5.TIME_LIMIT - (time.perf_counter() - start)
            if remaining <= 0.0:
                status = "time_limit"
                break
            certificate = v5.certify_fixed_scenario_fairness_feasibility(
                instance,
                y_values=cy,
                x_values=cx,
                t_value=ct,
                cost_budget_value=budget,
                demand_values=[list(row) for row in scenario.demand],
                time_limit=remaining,
                feasibility_tolerance=v5.FEAS_TOL,
                output_flag=False,
            )
            if certificate.infeasibility_certified and certificate.ray is not None:
                cut = v5.fairness_cut_from_ray(
                    instance,
                    cost_budget_value=budget,
                    demand_values=[list(row) for row in scenario.demand],
                    ray=certificate.ray,
                    active_deviations=[
                        {"region": r, "product": j} for r, j in scenario.active_units
                    ],
                )
                violation = -float(cut.value(cy, cx, ct))
                if violation > v5.FEAS_TOL:
                    scale = _coefficient_scale(cut, selection_mode)
                    violations.append((violation, violation * scale, index, cut, scale))
            elif not certificate.primal_feasible:
                status = f"uncertified_{certificate.primal_status}"
                break
        separation_runtime += time.perf_counter() - tick
        if status == "time_limit" or status.startswith("uncertified_"):
            break

        violations.sort(key=lambda item: (-item[0], item[2]))
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
                "active_cuts": len(active_cut_keys),
            })
            break

        actions = []
        promoted = set()
        for _, _, index, _, _ in violations[:MAX_BLOCKS_PER_ITERATION]:
            v5.add_complete_scenario_block(
                model,
                instance,
                scenarios[index],
                x,
                t,
                first_stage=first_stage,
                cost_budget=budget,
                scenario_sha=v5.scenario_sha256(instance, scenarios[index]),
            )
            blocked.add(index)
            promoted.add(index)
            removed_here = 0
            for key, constraint in cuts_by_scenario[index]:
                model.remove(constraint)
                active_cut_keys.discard(key)
                removed_here += 1
            cuts_by_scenario[index].clear()
            cuts_removed_total += removed_here
            actions.append(f"block:{index};remove_cuts:{removed_here}")

        remaining_violations = [item for item in violations if item[2] not in promoted]
        if remaining_violations:
            _, efficacy, index, cut, scale = max(
                remaining_violations, key=lambda item: (item[1], -item[2])
            )
            if efficacy >= MIN_NORMALIZED_CUT_EFFICACY:
                normalized_cut = _scaled_cut(cut, scale)
                key = v5._cut_key(normalized_cut)
                if key not in active_cut_keys:
                    constraint = v5.add_tracked_cut(
                        model, y, x, t, normalized_cut, cut_serial
                    )
                    cut_serial += 1
                    cuts_added_total += 1
                    active_cut_keys.add(key)
                    cuts_by_scenario[index].append((key, constraint))
                    actions.append(
                        f"normalized_{selection_mode}_cut:{index};efficacy:{efficacy}"
                    )
            else:
                actions.append(
                    f"skip_low_efficacy_cut:{index};efficacy:{efficacy}"
                )

        log.append({
            "iteration": iteration,
            "t": ct,
            "violations": len(violations),
            "maximum_violation": violations[0][0],
            "actions": actions,
            "blocked_scenarios": len(blocked),
            "active_cuts": len(active_cut_keys),
            "cuts_removed_total": cuts_removed_total,
        })

    result = {
        "candidate": f"hybrid_v8_batch4_normalized_{selection_mode}_cut",
        "status": status,
        "runtime": time.perf_counter() - start,
        "objective_t": best_t,
        "robust_minimum_fill_rate": 1.0 - best_t,
        "iterations": len(log),
        "master_runtime": master_runtime,
        "separation_runtime": separation_runtime,
        "blocked_scenarios": len(blocked),
        "active_cuts": len(active_cut_keys),
        "cuts_added_total": cuts_added_total,
        "cuts_removed_total": cuts_removed_total,
        "robust_feasibility_certified": status == "optimal",
        "cost_budget": budget,
        "rho": v5.FROZEN_RHO,
        "policy": {
            "initial_blocks": list(v5.INITIAL_BLOCKS),
            "max_blocks_per_iteration": MAX_BLOCKS_PER_ITERATION,
            "cut_selection": f"maximum_{selection_mode}_normalized_efficacy",
            "normalized_farkas_cuts_per_iteration": 1,
            "minimum_normalized_cut_efficacy": MIN_NORMALIZED_CUT_EFFICACY,
            "remove_promoted_scenario_cuts": True,
        },
        "solver_parameters": {
            "Threads": 1,
            "Seed": 0,
            "FeasibilityTol": v5.FEAS_TOL,
            "MIPGap": 0.0,
        },
        "y_values": best_y,
        "x_values": best_x,
        "iteration_log": log,
    }
    model.dispose()
    return result
