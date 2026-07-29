from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import gurobipy as gp
from gurobipy import GRB

from .experiment_protocol import atomic_write_json, read_json
from .fairness_benders import FairnessBendersResult, _build_master, relative_gap
from .fairness_large_final_remediation import (
    CUT_SCHEMA,
    CertifiedAdaptiveCut,
    CertifiedAdaptiveSeparator,
    RemediationIdentityError,
    _master_variables,
    add_canonical_cut_payload,
    canonical_json_bytes,
    construct_initial_t1_upper_bound,
)
from .instance import InventoryInstance
from .robust_regional_fairness import (
    FAIRNESS_FEASIBILITY_TOLERANCE,
    FAIRNESS_METRIC_TOLERANCE,
    _first_stage_expression,
    _recourse_expressions,
    fairness_cost_budget,
    gurobi_status_name,
)
from .scenarios import DemandScenario, _scenario_from_units


CANDIDATE = "certified_hybrid_scenario_benders_fairness"
SCENARIO_SCHEMA = "fairness_hybrid_scenario_v1"
CHECKPOINT_SCHEMA = "fairness_hybrid_ccg_benders_checkpoint_v1"
PROTOCOL_SHA256 = "C1F608E6ABD1D0EE27A106BD28EE098A26FF262F987033C1BD9DDFB53E3EF750"
CANDIDATE_SHA256 = "8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0"


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RemediationIdentityError(f"{label} must be finite binary64")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RemediationIdentityError(f"{label} must be finite binary64") from exc
    if not math.isfinite(parsed):
        raise RemediationIdentityError(f"{label} must be finite binary64")
    return 0.0 if parsed == 0.0 else parsed


def _instance_sha256(instance: InventoryInstance) -> str:
    return hashlib.sha256(canonical_json_bytes(instance.to_dict())).hexdigest().upper()


def canonical_scenario_payload(instance: InventoryInstance, scenario: DemandScenario) -> dict[str, Any]:
    component_order = [[int(r), int(j)] for r in instance.R for j in instance.J]
    active = {(int(r), int(j)) for r, j in scenario.active_units}
    if len(active) != len(scenario.active_units) or not active.issubset({tuple(item) for item in component_order}):
        raise RemediationIdentityError("invalid scenario active component identity")
    if len(scenario.demand) != instance.num_regions or any(
        len(row) != instance.num_products for row in scenario.demand
    ):
        raise RemediationIdentityError("invalid scenario demand dimensions")
    demand_hex = [
        [_finite_float(scenario.demand[r][j], "scenario demand").hex() for j in instance.J]
        for r in instance.R
    ]
    values = [1 if tuple(item) in active else 0 for item in component_order]
    return {
        "schema": SCENARIO_SCHEMA,
        "instance_sha256": _instance_sha256(instance),
        "component_order": component_order,
        "values": values,
        "demand_hex": demand_hex,
    }


def scenario_sha256(instance: InventoryInstance, scenario: DemandScenario) -> str:
    return hashlib.sha256(canonical_json_bytes(canonical_scenario_payload(instance, scenario))).hexdigest().upper()


def scenario_from_payload(payload: Mapping[str, Any]) -> DemandScenario:
    if payload.get("schema") != SCENARIO_SCHEMA:
        raise RemediationIdentityError("scenario schema mismatch")
    order = payload.get("component_order")
    values = payload.get("values")
    demand = payload.get("demand_hex")
    if not isinstance(order, list) or not isinstance(values, list) or len(order) != len(values):
        raise RemediationIdentityError("scenario component identity corrupt")
    if any(value not in (0, 1) or isinstance(value, bool) for value in values):
        raise RemediationIdentityError("scenario activity must be binary")
    active = tuple((int(item[0]), int(item[1])) for item, value in zip(order, values) if value == 1)
    if list(active) != sorted(active) or len(set(active)) != len(active):
        raise RemediationIdentityError("scenario component order drifted")
    if not isinstance(demand, list):
        raise RemediationIdentityError("scenario demand payload corrupt")
    rows = tuple(tuple(_finite_float(float.fromhex(value), "scenario demand") for value in row) for row in demand)
    return DemandScenario(name="canonical_" + hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()[:16], active_units=active, demand=rows)


def initial_scenarios(instance: InventoryInstance, gamma: int) -> list[DemandScenario]:
    gamma_value = max(0, int(gamma))
    scenarios = [_scenario_from_units(instance, ())]
    for r in instance.R:
        ordered = sorted(
            ((r, j) for j in instance.J),
            key=lambda item: (-_finite_float(instance.demand_deviation[item[0]][item[1]], "demand deviation"), item),
        )
        scenarios.append(_scenario_from_units(instance, ordered[: min(gamma_value, len(ordered))]))
    unique: dict[str, DemandScenario] = {}
    for scenario in scenarios:
        unique.setdefault(scenario_sha256(instance, scenario), scenario)
    return [unique[digest] for digest in unique]


def initial_scenario_plan_identity(*, num_regions: int, num_products: int, gamma: int) -> dict[str, Any]:
    payload = {
        "schema": "fairness_hybrid_initial_scenario_plan_v1",
        "num_regions": int(num_regions),
        "num_products": int(num_products),
        "gamma": int(gamma),
        "rule": "nominal_then_regional_stress_descending_deviation_canonical_tie",
    }
    return {
        "initial_scenario_count": 1 + int(num_regions),
        "initial_scenario_plan_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper(),
    }


def add_complete_scenario_block(
    model: gp.Model,
    instance: InventoryInstance,
    scenario: DemandScenario,
    x: Any,
    t: Any,
    *,
    first_stage: gp.LinExpr,
    cost_budget: float,
    scenario_sha: str,
) -> None:
    prefix = scenario_sha[:16]
    _expected = scenario_sha256(instance, scenario)
    if _expected != scenario_sha:
        raise RemediationIdentityError("scenario SHA does not match complete block payload")
    _q, u, _e, transport, shortage, service = _recourse_expressions(
        model, instance, scenario, x, prefix=prefix
    )
    model.addConstr(first_stage + transport + shortage + service <= float(cost_budget), name=f"hybrid_cost_cap[{prefix}]")
    for r in instance.R:
        regional_demand = math.fsum(float(scenario.demand[r][j]) for j in instance.J)
        if regional_demand > FAIRNESS_METRIC_TOLERANCE:
            model.addConstr(
                gp.quicksum(u[r, j] for j in instance.J) <= t * regional_demand,
                name=f"hybrid_regional_service[{prefix},{r}]",
            )


def select_one_new_scenario(
    instance: InventoryInstance,
    candidates: list[CertifiedAdaptiveCut],
    committed_scenario_sha256: set[str],
) -> tuple[CertifiedAdaptiveCut, DemandScenario, str] | None:
    eligible: list[tuple[CertifiedAdaptiveCut, DemandScenario, str]] = []
    for candidate in candidates:
        active = tuple(sorted((int(item["region"]), int(item["product"])) for item in candidate.cut.active_deviations))
        scenario = _scenario_from_units(instance, active)
        digest = scenario_sha256(instance, scenario)
        if digest not in committed_scenario_sha256:
            eligible.append((candidate, scenario, digest))
    if not eligible:
        return None
    eligible.sort(key=lambda item: (-item[0].normalized_violation_bucket, item[2], item[0].cut_sha256))
    return eligible[0]


def _checkpoint_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest().upper()


def _incumbent_identity(best_y: list[float], best_x: list[list[float]], upper_bound: float) -> str:
    payload = {
        "schema": "fairness_hybrid_incumbent_v1",
        "best_y_hex": [float(value).hex() for value in best_y],
        "best_x_hex": [[float(value).hex() for value in row] for row in best_x],
        "upper_bound_hex": float(upper_bound).hex(),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()


def _write_checkpoint(path: Path, identity: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    body = {"schema": CHECKPOINT_SCHEMA, "identity": deepcopy(dict(identity)), "state": deepcopy(dict(state))}
    body["checkpoint_sha256"] = _checkpoint_hash(body)
    atomic_write_json(path, body)


def _load_checkpoint(path: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RemediationIdentityError("hybrid checkpoint corrupt")
    digest = payload.pop("checkpoint_sha256", None)
    if digest != _checkpoint_hash(payload):
        raise RemediationIdentityError("hybrid checkpoint hash mismatch")
    if payload.get("identity") != dict(identity) or not isinstance(payload.get("state"), dict):
        raise RemediationIdentityError("hybrid checkpoint identity mismatch")
    return payload["state"]


def solve_certified_hybrid_scenario_benders_fairness(
    instance: InventoryInstance,
    *,
    baseline_record: dict[str, Any],
    anchor: dict[str, Any],
    expected_identity: dict[str, Any],
    solver_parameters: dict[str, Any],
    rho: float,
    gamma: int = 2,
    max_iterations: int = 10000,
    time_limit: float = 1800.0,
    tol: float = 1.0e-4,
    feasibility_tolerance: float = FAIRNESS_FEASIBILITY_TOLERANCE,
    output_flag: bool = False,
    checkpoint_path: str | Path | None = None,
    checkpoint_identity: dict[str, Any] | None = None,
    failure_injector: Callable[[str, dict[str, Any]], None] | None = None,
) -> FairnessBendersResult:
    start = time.perf_counter()
    initial = construct_initial_t1_upper_bound(
        instance, baseline_record=baseline_record, anchor=anchor, rho=rho,
        tolerance=tol, expected_identity=expected_identity,
        expected_candidate_sha256=CANDIDATE_SHA256,
    )
    if solver_parameters != {"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7}:
        raise RemediationIdentityError("frozen solver identity mismatch")
    gp.setParam("Threads", 1)
    gp.setParam("Seed", 0)
    gp.setParam("FeasibilityTol", 1.0e-7)
    budget = fairness_cost_budget(float(anchor["value"]), rho)
    model, y, x, t = _build_master(instance, output_flag)
    model.Params.Threads = 1
    model.Params.Seed = 0
    model.Params.FeasibilityTol = 1.0e-7
    variables = _master_variables(instance, y, x, t)
    first_stage = _first_stage_expression(instance, y, x)
    identity = {
        "candidate": CANDIDATE,
        "candidate_sha256": CANDIDATE_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "scenario_schema": SCENARIO_SCHEMA,
        "rho_hex": float(rho).hex(),
        "anchor_sha256": str(anchor["anchor_sha256"]).upper(),
        "run_identity": deepcopy(expected_identity),
        "solver_parameters": deepcopy(solver_parameters),
        **deepcopy(checkpoint_identity or {}),
    }
    initial_set = initial_scenarios(instance, gamma)
    scenario_payloads = {scenario_sha256(instance, scenario): canonical_scenario_payload(instance, scenario) for scenario in initial_set}
    scenario_order = list(scenario_payloads)
    cut_payloads: dict[str, dict[str, Any]] = {}
    cut_order: list[str] = []
    lower_bound: float | None = None
    upper_bound = 1.0
    best_y = list(initial.y_values)
    best_x = deepcopy(initial.x_values)
    iteration_start = 1
    log: list[dict[str, Any]] = []
    checkpoint = None if checkpoint_path is None else _load_checkpoint(Path(checkpoint_path), identity)
    if checkpoint is not None:
        scenario_order = list(checkpoint["committed_scenario_sha256_values"])
        scenario_payloads = deepcopy(checkpoint["scenario_payloads_by_sha256"])
        cut_order = list(checkpoint["committed_farkas_cut_sha256_values"])
        cut_payloads = deepcopy(checkpoint["cut_payloads_by_sha256"])
        if len(set(scenario_order)) != len(scenario_order) or set(scenario_order) != set(scenario_payloads):
            raise RemediationIdentityError("checkpoint scenario order or payload drifted")
        if len(set(cut_order)) != len(cut_order) or set(cut_order) != set(cut_payloads):
            raise RemediationIdentityError("checkpoint cut order or payload drifted")
        lower_bound = checkpoint.get("lower_bound")
        upper_bound = float(checkpoint["upper_bound"])
        best_y = [float(value) for value in checkpoint["best_y"]]
        best_x = [[float(value) for value in row] for row in checkpoint["best_x"]]
        if checkpoint.get("incumbent_identity_sha256") != _incumbent_identity(best_y, best_x, upper_bound):
            raise RemediationIdentityError("checkpoint incumbent identity drifted")
        log = list(checkpoint["iteration_log"])
        iteration_start = int(checkpoint["iteration"]) + 1
    for digest in scenario_order:
        scenario = scenario_from_payload(scenario_payloads[digest])
        if scenario_sha256(instance, scenario) != digest:
            raise RemediationIdentityError("checkpoint scenario SHA drifted")
        add_complete_scenario_block(model, instance, scenario, x, t, first_stage=first_stage, cost_budget=budget.budget, scenario_sha=digest)
    for index, digest in enumerate(cut_order):
        add_canonical_cut_payload(model, variables, cut_payloads[digest], index=index)
    for i in instance.I:
        y[i].Start = initial.y_values[i]
        for j in instance.J:
            x[i, j].Start = initial.x_values[i][j]
    t.Start = 1.0
    separator = CertifiedAdaptiveSeparator(instance, gamma=gamma, feasibility_tolerance=feasibility_tolerance, output_flag=output_flag)
    status = "iteration_limit"
    master_runtime = 0.0
    separation_runtime = 0.0
    final_certified = False
    try:
        for iteration in range(iteration_start, int(max_iterations) + 1):
            remaining = float(time_limit) - (time.perf_counter() - start)
            if remaining <= 0.0:
                status = "time_limit"
                break
            model.Params.MIPGap = 0.0
            model.Params.TimeLimit = max(1.0e-3, remaining)
            if failure_injector:
                failure_injector("before_master", {"iteration": iteration})
            tick = time.perf_counter()
            model.optimize()
            master_runtime += time.perf_counter() - tick
            if model.SolCount <= 0:
                status = "infeasible" if model.Status == GRB.INFEASIBLE else gurobi_status_name(model.Status)
                break
            master_bound = float(model.ObjBound)
            lower_bound = master_bound if lower_bound is None else max(float(lower_bound), master_bound)
            candidate_t = float(t.X)
            candidate_y = [float(y[i].X) for i in instance.I]
            candidate_x = [[float(x[i, j].X) for j in instance.J] for i in instance.I]
            remaining = float(time_limit) - (time.perf_counter() - start)
            tick = time.perf_counter()
            separated = separator.separate(
                y_values=candidate_y, x_values=candidate_x, t_value=candidate_t,
                cost_budget_value=budget.budget, mip_gap=0.05, time_limit=max(1.0e-3, remaining),
                final_certification=False,
            )
            separation_runtime += time.perf_counter() - tick
            full = separated.full_separation
            chosen = select_one_new_scenario(instance, separated.candidates, set(scenario_order))
            final_separation_performed = False
            if chosen is None:
                remaining = float(time_limit) - (time.perf_counter() - start)
                if remaining <= 0.0:
                    status = "time_limit"
                    break
                tick = time.perf_counter()
                separated = separator.separate(
                    y_values=candidate_y, x_values=candidate_x, t_value=candidate_t,
                    cost_budget_value=budget.budget, mip_gap=0.0, time_limit=remaining,
                    final_certification=True,
                )
                separation_runtime += time.perf_counter() - tick
                full = separated.full_separation
                chosen = select_one_new_scenario(instance, separated.candidates, set(scenario_order))
                final_separation_performed = True
            if final_separation_performed and full.robust_feasibility_certified:
                upper_bound = min(upper_bound, candidate_t)
                best_y, best_x = candidate_y, candidate_x
                final_certified = True
            committed_scenario = None
            committed_cut = None
            if chosen is not None:
                candidate, scenario, digest = chosen
                if failure_injector:
                    failure_injector("before_scenario_commit", {"scenario_sha256": digest})
                add_complete_scenario_block(model, instance, scenario, x, t, first_stage=first_stage, cost_budget=budget.budget, scenario_sha=digest)
                scenario_payloads[digest] = canonical_scenario_payload(instance, scenario)
                scenario_order.append(digest)
                committed_scenario = digest
                if candidate.cut_sha256 not in cut_payloads:
                    add_canonical_cut_payload(model, variables, candidate.canonical_cut_payload, index=len(cut_order))
                    cut_payloads[candidate.cut_sha256] = deepcopy(candidate.canonical_cut_payload)
                    cut_order.append(candidate.cut_sha256)
                    committed_cut = candidate.cut_sha256
                final_certified = False
            gap = relative_gap(upper_bound, lower_bound)
            entry = {
                "iteration": iteration,
                "master_status": gurobi_status_name(model.Status),
                "master_solver_best_bound": master_bound,
                "master_incumbent_objective": candidate_t,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "gap": gap,
                "separation_status": full.status,
                "separation_objective_bound": full.objective_bound,
                "final_exact_separation_performed": final_separation_performed,
                "robust_feasibility_certified": full.robust_feasibility_certified,
                "committed_scenario_sha256": committed_scenario,
                "committed_farkas_cut_sha256": committed_cut,
                "scenario_count": len(scenario_order),
            }
            log.append(entry)
            state = {
                "iteration": iteration,
                "committed_scenario_sha256_values": list(scenario_order),
                "scenario_payloads_by_sha256": deepcopy(scenario_payloads),
                "committed_farkas_cut_sha256_values": list(cut_order),
                "cut_payloads_by_sha256": deepcopy(cut_payloads),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "gap": gap,
                "master_solver_best_bound": master_bound,
                "best_y": best_y,
                "best_x": best_x,
                "incumbent_identity_sha256": _incumbent_identity(best_y, best_x, upper_bound),
                "final_certification_state": "complete_exact_certified" if final_certified else "not_certified",
                "iteration_log": log,
            }
            if checkpoint_path is not None:
                _write_checkpoint(Path(checkpoint_path), identity, state)
            if failure_injector:
                failure_injector("after_scenario_commit_checkpoint", deepcopy(state))
            if final_certified and gap is not None and gap <= float(tol) and model.Status == GRB.OPTIMAL:
                status = "optimal"
                break
            if chosen is None and not final_certified:
                status = full.status if full.status not in {"optimal", "unknown"} else "separation_stalled_duplicate"
                break
        else:
            status = "iteration_limit"
    finally:
        separator.dispose()
        model.dispose()
    runtime = time.perf_counter() - start
    return FairnessBendersResult(
        status=status,
        objective_t=upper_bound,
        robust_minimum_fill_rate=1.0 - upper_bound,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        gap=relative_gap(upper_bound, lower_bound),
        runtime=runtime,
        iterations=len(log),
        cuts=len(cut_order),
        cuts_with_cost_component=0,
        cuts_with_fairness_component=0,
        joint_cost_fairness_cuts=len(cut_order),
        baseline_cost=float(anchor["value"]),
        rho=float(rho),
        cost_budget=budget.budget,
        y_values=best_y,
        x_values=best_x,
        master_runtime=master_runtime,
        separation_runtime=separation_runtime,
        separation_patterns_seen=[],
        iteration_log=log,
        metadata={
            "candidate": CANDIDATE,
            "protocol_sha256": PROTOCOL_SHA256,
            "candidate_sha256": CANDIDATE_SHA256,
            "initial_robust_upper_bound": initial.evidence,
            "initial_scenario_count": len(initial_set),
            "committed_scenario_count": len(scenario_order),
            "committed_scenario_sha256_values": scenario_order,
            "committed_farkas_cut_sha256_values": cut_order,
            "scenario_master_lower_bound_valid": True,
            "full_separation_objective_bound_required": True,
            "robust_feasibility_certified": final_certified,
            "runtime_driven_scientific_branching": False,
        },
    )
