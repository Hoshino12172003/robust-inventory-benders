from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Iterable

import gurobipy as gp
from gurobipy import GRB

from .instance import InventoryInstance
from .monolithic import first_stage_cost_expr
from .robust_dual_subproblem import discretize_robust_pattern, solve_robust_dual_subproblem
from .scenarios import DemandScenario
from .status import gurobi_status_name


Pattern = tuple[tuple[int, int], ...]


@dataclass
class CCGResult:
    status: str
    certified: bool
    lower_bound: float | None
    upper_bound: float | None
    gap: float | None
    runtime: float
    iterations: int
    scenarios_added: int
    initial_scenario_count: int
    inherited_scenario_count: int
    duplicate_scenarios_removed: int
    final_active_scenario_count: int
    incumbent_reused: bool
    master_initialization_runtime: float
    master_runtime: float
    adversarial_runtime: float
    first_iteration_lower_bound: float | None
    first_iteration_upper_bound: float | None
    time_to_best_incumbent: float | None
    time_to_certification: float | None
    best_y_values: list[float] | None
    best_x_values: list[list[float]] | None
    scenario_patterns: list[Pattern]
    iteration_log: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["scenario_patterns"] = [
            [[r, j] for r, j in pattern] for pattern in self.scenario_patterns
        ]
        return value


def relative_gap(upper_bound: float | None, lower_bound: float | None) -> float | None:
    if upper_bound is None or lower_bound is None:
        return None
    return max(0.0, float(upper_bound) - float(lower_bound)) / max(
        1.0, abs(float(upper_bound))
    )


def canonical_pattern(units: Iterable[tuple[int, int]]) -> Pattern:
    return tuple(sorted({(int(r), int(j)) for r, j in units}))


def scenario_from_pattern(instance: InventoryInstance, pattern: Pattern) -> DemandScenario:
    active = set(pattern)
    expected = {(r, j) for r in instance.R for j in instance.J}
    if not active.issubset(expected):
        raise ValueError("Scenario pattern contains an unknown demand component.")
    demand = tuple(
        tuple(
            float(instance.base_demand[r][j])
            + (float(instance.demand_deviation[r][j]) if (r, j) in active else 0.0)
            for j in instance.J
        )
        for r in instance.R
    )
    label = "nominal" if not pattern else "shock_" + "_".join(f"{r}-{j}" for r, j in pattern)
    return DemandScenario(name=label, active_units=pattern, demand=demand)


def deduplicate_patterns(patterns: Iterable[Iterable[tuple[int, int]]]) -> tuple[list[Pattern], int]:
    unique: list[Pattern] = []
    seen: set[Pattern] = set()
    duplicates = 0
    for raw in patterns:
        pattern = canonical_pattern(raw)
        if pattern in seen:
            duplicates += 1
        else:
            seen.add(pattern)
            unique.append(pattern)
    return unique, duplicates


class RestrictedScenarioMaster:
    def __init__(self, instance: InventoryInstance, *, output_flag: bool = False) -> None:
        self.instance = instance
        self.model = gp.Model("robust_inventory_ccg_master")
        self.model.Params.OutputFlag = 1 if output_flag else 0
        self.y = self.model.addVars(instance.I, vtype=GRB.BINARY, name="y")
        self.x = self.model.addVars(instance.I, instance.J, lb=0.0, name="x")
        self.theta = self.model.addVar(lb=0.0, name="theta")
        for i in instance.I:
            self.model.addConstr(
                gp.quicksum(instance.volume[j] * self.x[i, j] for j in instance.J)
                <= instance.capacity[i] * self.y[i],
                name=f"capacity[{i}]",
            )
            for j in instance.J:
                self.model.addConstr(
                    self.x[i, j] <= instance.inventory_ub[i][j] * self.y[i],
                    name=f"logic[{i},{j}]",
                )
        self.first_stage = first_stage_cost_expr(instance, self.y, self.x)
        self.model.addConstr(self.first_stage <= instance.budget, name="budget")
        self.model.setObjective(self.first_stage + self.theta, GRB.MINIMIZE)
        self.patterns: list[Pattern] = []
        self._pattern_set: set[Pattern] = set()

    def add_scenario(self, pattern: Pattern) -> bool:
        pattern = canonical_pattern(pattern)
        if pattern in self._pattern_set:
            return False
        scenario = scenario_from_pattern(self.instance, pattern)
        index = len(self.patterns)
        q = self.model.addVars(
            self.instance.I,
            self.instance.R,
            self.instance.J,
            lb=0.0,
            name=f"q_{index}",
        )
        u = self.model.addVars(
            self.instance.R, self.instance.J, lb=0.0, name=f"u_{index}"
        )
        e = self.model.addVars(self.instance.J, lb=0.0, name=f"e_{index}")
        for r in self.instance.R:
            for j in self.instance.J:
                self.model.addConstr(
                    gp.quicksum(q[i, r, j] for i in self.instance.I) + u[r, j]
                    >= scenario.demand[r][j],
                    name=f"demand[{index},{r},{j}]",
                )
        for i in self.instance.I:
            for j in self.instance.J:
                self.model.addConstr(
                    gp.quicksum(q[i, r, j] for r in self.instance.R) <= self.x[i, j],
                    name=f"supply[{index},{i},{j}]",
                )
        for j in self.instance.J:
            self.model.addConstr(
                gp.quicksum(u[r, j] for r in self.instance.R) - e[j]
                <= (1.0 - self.instance.service_level[j])
                * sum(scenario.demand[r][j] for r in self.instance.R),
                name=f"service[{index},{j}]",
            )
        scenario_cost = (
            gp.quicksum(
                self.instance.transport_cost[i][r][j] * q[i, r, j]
                for i in self.instance.I
                for r in self.instance.R
                for j in self.instance.J
            )
            + gp.quicksum(
                self.instance.shortage_penalty[r][j] * u[r, j]
                for r in self.instance.R
                for j in self.instance.J
            )
            + gp.quicksum(
                self.instance.service_penalty[j] * e[j] for j in self.instance.J
            )
        )
        self.model.addConstr(
            self.theta >= scenario_cost, name=f"robust_theta[{index}]"
        )
        self.patterns.append(pattern)
        self._pattern_set.add(pattern)
        return True

    def apply_start(self, y_values: list[float], x_values: list[list[float]]) -> None:
        if len(y_values) != self.instance.num_warehouses or len(x_values) != self.instance.num_warehouses:
            raise ValueError("MIP start dimensions do not match the instance.")
        for i in self.instance.I:
            if len(x_values[i]) != self.instance.num_products:
                raise ValueError("MIP start dimensions do not match the instance.")
            self.y[i].Start = float(y_values[i])
            for j in self.instance.J:
                self.x[i, j].Start = float(x_values[i][j])

    def dispose(self) -> None:
        self.model.dispose()


def solve_ccg(
    instance: InventoryInstance,
    *,
    gamma: int,
    time_limit: float,
    tolerance: float,
    inherited_patterns: Iterable[Iterable[tuple[int, int]]] = (),
    initial_y: list[float] | None = None,
    initial_x: list[list[float]] | None = None,
    initial_upper_bound: float | None = None,
    output_flag: bool = False,
    max_iterations: int = 20_000,
) -> CCGResult:
    if gamma < 0 or tolerance <= 0.0 or time_limit <= 0.0:
        raise ValueError("Gamma, tolerance, or time limit is invalid.")
    if (initial_y is None) != (initial_x is None):
        raise ValueError("Both y and x are required for an incumbent MIP start.")
    start = time.perf_counter()
    inherited, inherited_duplicates = deduplicate_patterns(inherited_patterns)
    initial_patterns, nominal_duplicates = deduplicate_patterns([(), *inherited])
    build_start = time.perf_counter()
    master = RestrictedScenarioMaster(instance, output_flag=output_flag)
    try:
        for pattern in initial_patterns:
            master.add_scenario(pattern)
        if initial_y is not None and initial_x is not None:
            master.apply_start(initial_y, initial_x)
        master.model.update()
        initialization_runtime = time.perf_counter() - build_start
        lower_bound: float | None = None
        upper_bound = (
            float(initial_upper_bound)
            if initial_upper_bound is not None and math.isfinite(float(initial_upper_bound))
            else None
        )
        best_y = list(initial_y) if initial_y is not None else None
        best_x = [list(row) for row in initial_x] if initial_x is not None else None
        best_incumbent_time = 0.0 if upper_bound is not None else None
        master_runtime = 0.0
        adversarial_runtime = 0.0
        scenarios_added = 0
        log: list[dict[str, Any]] = []
        certified = False
        status = "iteration_limit"
        certification_time: float | None = None

        for iteration in range(1, max_iterations + 1):
            remaining = time_limit - (time.perf_counter() - start)
            if remaining <= 0.0:
                status = "time_limit"
                break
            master.model.Params.MIPGap = 0.0
            master.model.Params.TimeLimit = max(1.0e-3, remaining)
            tick = time.perf_counter()
            master.model.optimize()
            iteration_master_runtime = time.perf_counter() - tick
            master_runtime += iteration_master_runtime
            master_status = gurobi_status_name(master.model.Status)
            if master.model.SolCount <= 0:
                status = master_status
                break
            master_bound = float(master.model.ObjBound)
            lower_bound = master_bound if lower_bound is None else max(lower_bound, master_bound)
            candidate_y = [float(master.y[i].X) for i in instance.I]
            candidate_x = [
                [float(master.x[i, j].X) for j in instance.J] for i in instance.I
            ]
            x_values = {
                (i, j): candidate_x[i][j] for i in instance.I for j in instance.J
            }
            first_stage_value = float(master.first_stage.getValue())
            theta_value = float(master.theta.X)

            remaining = time_limit - (time.perf_counter() - start)
            if remaining <= 0.0:
                status = "time_limit"
                break
            tick = time.perf_counter()
            separated = solve_robust_dual_subproblem(
                instance,
                x_values,
                gamma,
                time_limit=max(1.0e-3, remaining),
                mip_gap=0.0,
                output_flag=output_flag,
            )
            iteration_adversarial_runtime = time.perf_counter() - tick
            adversarial_runtime += iteration_adversarial_runtime
            if not separated.has_incumbent or separated.objective is None:
                status = separated.status
                break

            valid_candidate_upper = None
            if separated.objective_bound is not None and math.isfinite(separated.objective_bound):
                valid_candidate_upper = first_stage_value + float(separated.objective_bound)
                if upper_bound is None or valid_candidate_upper < upper_bound:
                    upper_bound = valid_candidate_upper
                    best_y = candidate_y
                    best_x = candidate_x
                    best_incumbent_time = time.perf_counter() - start
            pattern_map = discretize_robust_pattern(instance, separated.z_values)
            pattern = (
                canonical_pattern(key for key, value in pattern_map.items() if value)
                if pattern_map is not None
                else None
            )
            violation = float(separated.objective) - theta_value
            gap = relative_gap(upper_bound, lower_bound)
            added = False
            if pattern is not None and pattern not in master._pattern_set and violation > tolerance:
                added = master.add_scenario(pattern)
                scenarios_added += int(added)
            exact_separation = separated.status == "optimal" and separated.mip_gap is not None and separated.mip_gap <= 1.0e-12
            certified = bool(
                exact_separation
                and master.model.Status == GRB.OPTIMAL
                and gap is not None
                and gap <= tolerance
            )
            elapsed = time.perf_counter() - start
            log.append(
                {
                    "iteration": iteration,
                    "elapsed_time": elapsed,
                    "master_status": master_status,
                    "master_runtime": iteration_master_runtime,
                    "master_lower_bound": master_bound,
                    "master_incumbent_objective": float(master.model.ObjVal),
                    "adversarial_status": separated.status,
                    "adversarial_runtime": iteration_adversarial_runtime,
                    "adversarial_objective": separated.objective,
                    "adversarial_objective_bound": separated.objective_bound,
                    "adversarial_mip_gap": separated.mip_gap,
                    "exact_adversarial_separation": exact_separation,
                    "theta": theta_value,
                    "scenario_violation": violation,
                    "scenario_added": added,
                    "active_scenario_count": len(master.patterns),
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "gap": gap,
                }
            )
            if certified:
                status = "optimal"
                certification_time = elapsed
                break
            if not added:
                status = (
                    "time_limit"
                    if elapsed >= time_limit or separated.status == "time_limit"
                    else "separation_stalled_duplicate"
                )
                break
        else:
            status = "iteration_limit"

        runtime = time.perf_counter() - start
        return CCGResult(
            status=status,
            certified=certified,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            gap=relative_gap(upper_bound, lower_bound),
            runtime=runtime,
            iterations=len(log),
            scenarios_added=scenarios_added,
            initial_scenario_count=len(initial_patterns),
            inherited_scenario_count=len(inherited),
            duplicate_scenarios_removed=inherited_duplicates + nominal_duplicates,
            final_active_scenario_count=len(master.patterns),
            incumbent_reused=initial_y is not None,
            master_initialization_runtime=initialization_runtime,
            master_runtime=master_runtime,
            adversarial_runtime=adversarial_runtime,
            first_iteration_lower_bound=log[0]["lower_bound"] if log else None,
            first_iteration_upper_bound=log[0]["upper_bound"] if log else upper_bound,
            time_to_best_incumbent=best_incumbent_time,
            time_to_certification=certification_time,
            best_y_values=best_y,
            best_x_values=best_x,
            scenario_patterns=list(master.patterns),
            iteration_log=log,
        )
    finally:
        master.dispose()
