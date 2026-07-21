from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any, Iterable

import gurobipy as gp
from gurobipy import GRB

from .instance import InventoryInstance
from .regional_fairness_diagnostic import summarize_regional_service
from .scenarios import DemandScenario, enumerate_budget_scenarios_with_metadata
from .status import gurobi_status_name


FAIRNESS_COST_ABSOLUTE_TOLERANCE = 1.0e-6
FAIRNESS_COST_RELATIVE_TOLERANCE = 1.0e-6
FAIRNESS_FEASIBILITY_TOLERANCE = 1.0e-7
FAIRNESS_METRIC_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class FairnessBudget:
    baseline_cost: float
    rho: float
    budget: float


@dataclass(frozen=True)
class FairnessFarkasRay:
    demand: list[list[float]]
    supply: list[list[float]]
    service: list[float]
    cost: float
    regional_fairness: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FairnessFeasibilityCut:
    constant: float
    y_coefficients: list[float]
    x_coefficients: list[list[float]]
    t_coefficient: float
    active_deviations: list[dict[str, int]]
    demand_values: list[list[float]]
    ray: FairnessFarkasRay

    def value(
        self,
        y_values: list[float],
        x_values: list[list[float]],
        t_value: float,
    ) -> float:
        return float(
            self.constant
            + sum(self.y_coefficients[i] * float(y_values[i]) for i in range(len(y_values)))
            + sum(
                self.x_coefficients[i][j] * float(x_values[i][j])
                for i in range(len(x_values))
                for j in range(len(x_values[i]))
            )
            + self.t_coefficient * float(t_value)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FairnessSeparationResult:
    status: str
    has_incumbent: bool
    objective: float | None
    objective_bound: float | None
    mip_gap: float | None
    runtime: float
    requested_mip_gap: float
    robust_feasibility_certified: bool
    cut: FairnessFeasibilityCut | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FairnessScenarioPolicy:
    scenario_name: str
    active_deviations: list[dict[str, int]]
    recourse_cost: float
    transport_cost: float
    shortage_cost: float
    service_violation_cost: float
    regional_shortage: list[float]
    regional_demand: list[float]
    fill_rates: list[float | None]
    minimum_fill_rate: float | None
    fill_rate_gap: float | None
    worst_region_deviation: float | None
    weighted_mean_fill_rate: float | None


@dataclass(frozen=True)
class FairnessExtensiveFormResult:
    status: str
    objective_t: float | None
    robust_minimum_fill_rate: float | None
    baseline_cost: float
    rho: float
    cost_budget: float
    actual_robust_cost: float | None
    actual_price_of_fairness: float | None
    y_values: list[float] | None
    x_values: list[list[float]] | None
    scenario_policies: list[FairnessScenarioPolicy] = field(default_factory=list)
    cost_worst_scenario: str | None = None
    fairness_worst_scenario: str | None = None
    wgap: float | None = None
    wwd: float | None = None
    weighted_mean_fill_rate: float | None = None
    runtime: float = 0.0
    lexicographic_cost_stage_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FairnessSolutionEvaluation:
    valid: bool
    scenario_count: int
    actual_robust_cost: float | None
    actual_price_of_fairness: float | None
    wgap: float | None
    wminfr: float | None
    wwd: float | None
    minimum_weighted_mean_fill_rate: float | None
    cost_worst_scenario: str | None
    fairness_worst_scenario: str | None
    opened_warehouses: int
    total_inventory: float
    inventory_by_warehouse: list[float]
    inventory_by_product: list[float]
    runtime: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fairness_cost_budget(baseline_cost: float, rho: float) -> FairnessBudget:
    baseline = float(baseline_cost)
    ratio = float(rho)
    if not math.isfinite(baseline) or baseline < 0.0:
        raise ValueError("baseline_cost must be finite and nonnegative.")
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("rho must be finite and nonnegative.")
    return FairnessBudget(baseline, ratio, (1.0 + ratio) * baseline)


def cost_tolerance(
    value: float,
    *,
    absolute_tolerance: float = FAIRNESS_COST_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = FAIRNESS_COST_RELATIVE_TOLERANCE,
) -> float:
    absolute = float(absolute_tolerance)
    relative = float(relative_tolerance)
    if not all(math.isfinite(item) and item >= 0.0 for item in (absolute, relative)):
        raise ValueError("Cost tolerances must be finite and nonnegative.")
    return absolute + relative * max(1.0, abs(float(value)))


def first_stage_cost_value(
    instance: InventoryInstance,
    y_values: list[float],
    x_values: list[list[float]],
) -> float:
    return float(
        sum(instance.fixed_cost[i] * float(y_values[i]) for i in instance.I)
        + sum(
            instance.inventory_cost[i][j] * float(x_values[i][j])
            for i in instance.I
            for j in instance.J
        )
    )


def scenario_demand(instance: InventoryInstance, active: Iterable[tuple[int, int]]) -> list[list[float]]:
    demand = [[float(value) for value in row] for row in instance.base_demand]
    for r, j in active:
        demand[int(r)][int(j)] += float(instance.demand_deviation[int(r)][int(j)])
    return demand


def _first_stage_expression(instance: InventoryInstance, y: Any, x: Any) -> gp.LinExpr:
    return gp.quicksum(instance.fixed_cost[i] * y[i] for i in instance.I) + gp.quicksum(
        instance.inventory_cost[i][j] * x[i, j] for i in instance.I for j in instance.J
    )


def _recourse_expressions(
    model: gp.Model,
    instance: InventoryInstance,
    scenario: DemandScenario,
    x: Any,
    *,
    prefix: str,
) -> tuple[Any, Any, Any, gp.LinExpr, gp.LinExpr, gp.LinExpr]:
    q = model.addVars(instance.I, instance.R, instance.J, lb=0.0, name=f"q_{prefix}")
    u = model.addVars(instance.R, instance.J, lb=0.0, name=f"u_{prefix}")
    e = model.addVars(instance.J, lb=0.0, name=f"e_{prefix}")
    for r in instance.R:
        for j in instance.J:
            model.addConstr(
                gp.quicksum(q[i, r, j] for i in instance.I) + u[r, j] >= scenario.demand[r][j],
                name=f"demand[{prefix},{r},{j}]",
            )
    for i in instance.I:
        for j in instance.J:
            model.addConstr(
                gp.quicksum(q[i, r, j] for r in instance.R) <= x[i, j],
                name=f"supply[{prefix},{i},{j}]",
            )
    for j in instance.J:
        model.addConstr(
            gp.quicksum(u[r, j] for r in instance.R) - e[j]
            <= (1.0 - instance.service_level[j])
            * sum(scenario.demand[r][j] for r in instance.R),
            name=f"service[{prefix},{j}]",
        )
    transport = gp.quicksum(
        instance.transport_cost[i][r][j] * q[i, r, j]
        for i in instance.I
        for r in instance.R
        for j in instance.J
    )
    shortage = gp.quicksum(
        instance.shortage_penalty[r][j] * u[r, j] for r in instance.R for j in instance.J
    )
    service = gp.quicksum(instance.service_penalty[j] * e[j] for j in instance.J)
    return q, u, e, transport, shortage, service


def _policy_from_solution(
    instance: InventoryInstance,
    scenario: DemandScenario,
    q: Any,
    u: Any,
    e: Any,
) -> FairnessScenarioPolicy:
    demand = [[float(value) for value in row] for row in scenario.demand]
    shortages = [[float(u[r, j].X) for j in instance.J] for r in instance.R]
    transport = sum(
        instance.transport_cost[i][r][j] * float(q[i, r, j].X)
        for i in instance.I
        for r in instance.R
        for j in instance.J
    )
    shortage_cost = sum(
        instance.shortage_penalty[r][j] * shortages[r][j]
        for r in instance.R
        for j in instance.J
    )
    service_cost = sum(instance.service_penalty[j] * float(e[j].X) for j in instance.J)
    metrics = summarize_regional_service(demand, shortages, metric_tolerance=FAIRNESS_METRIC_TOLERANCE)
    return FairnessScenarioPolicy(
        scenario_name=scenario.name,
        active_deviations=[{"region": r, "product": j} for r, j in scenario.active_units],
        recourse_cost=float(transport + shortage_cost + service_cost),
        transport_cost=float(transport),
        shortage_cost=float(shortage_cost),
        service_violation_cost=float(service_cost),
        regional_shortage=[float(row["regional_shortage"]) for row in metrics["regions"]],
        regional_demand=[float(row["regional_demand"]) for row in metrics["regions"]],
        fill_rates=[None if row["fill_rate"] is None else float(row["fill_rate"]) for row in metrics["regions"]],
        minimum_fill_rate=None if metrics["minimum_fill_rate"] is None else float(metrics["minimum_fill_rate"]),
        fill_rate_gap=None if metrics["fill_rate_gap"] is None else float(metrics["fill_rate_gap"]),
        worst_region_deviation=(
            None if metrics["worst_region_deviation"] is None else float(metrics["worst_region_deviation"])
        ),
        weighted_mean_fill_rate=(
            None if metrics["weighted_mean_fill_rate"] is None else float(metrics["weighted_mean_fill_rate"])
        ),
    )


def solve_fairness_extensive_form(
    instance: InventoryInstance,
    *,
    baseline_cost: float,
    rho: float,
    gamma: int,
    max_scenarios: int = 5000,
    time_limit: float = 120.0,
    mip_gap: float = 0.0,
    lexicographic_cost_stage: bool = True,
    t_tolerance: float = 1.0e-7,
    output_flag: bool = False,
) -> FairnessExtensiveFormResult:
    """Exact extensive form used only as a small-instance correctness oracle.

    The same scenario-specific recourse variables satisfy the original recourse,
    the total-cost cap, and the regional service constraints.  No solution from
    a separate cost or fairness recourse problem is spliced into this policy.
    """
    budget = fairness_cost_budget(baseline_cost, rho)
    enumeration = enumerate_budget_scenarios_with_metadata(
        instance, gamma, max_scenarios=max_scenarios, exact_scenarios=True
    )
    start = time.perf_counter()
    model = gp.Model("robust_regional_fairness_extensive_form")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.TimeLimit = max(1.0e-3, float(time_limit))
    model.Params.MIPGap = max(0.0, float(mip_gap))
    y = model.addVars(instance.I, vtype=GRB.BINARY, name="y")
    x = model.addVars(instance.I, instance.J, lb=0.0, name="x")
    t = model.addVar(lb=0.0, ub=1.0, name="T")
    for i in instance.I:
        model.addConstr(
            gp.quicksum(instance.volume[j] * x[i, j] for j in instance.J)
            <= instance.capacity[i] * y[i],
            name=f"capacity[{i}]",
        )
        for j in instance.J:
            model.addConstr(x[i, j] <= instance.inventory_ub[i][j] * y[i], name=f"logic[{i},{j}]")
    first_stage = _first_stage_expression(instance, y, x)
    model.addConstr(first_stage <= instance.budget, name="first_stage_budget")

    recourse: list[tuple[DemandScenario, Any, Any, Any, gp.LinExpr]] = []
    robust_recourse = model.addVar(lb=0.0, name="robust_recourse_cost")
    for index, scenario in enumerate(enumeration.scenarios):
        q, u, e, transport, shortage, service = _recourse_expressions(
            model, instance, scenario, x, prefix=str(index)
        )
        scenario_cost = transport + shortage + service
        model.addConstr(first_stage + scenario_cost <= budget.budget, name=f"cost_cap[{index}]")
        model.addConstr(robust_recourse >= scenario_cost, name=f"robust_cost[{index}]")
        for r in instance.R:
            regional_demand = sum(scenario.demand[r][j] for j in instance.J)
            if regional_demand > FAIRNESS_METRIC_TOLERANCE:
                model.addConstr(
                    gp.quicksum(u[r, j] for j in instance.J) <= t * regional_demand,
                    name=f"regional_service[{index},{r}]",
                )
        recourse.append((scenario, q, u, e, scenario_cost))

    model.setObjective(t, GRB.MINIMIZE)
    model.optimize()
    status = gurobi_status_name(model.Status)
    if model.Status != GRB.OPTIMAL:
        runtime = time.perf_counter() - start
        model.dispose()
        return FairnessExtensiveFormResult(
            status=status,
            objective_t=None,
            robust_minimum_fill_rate=None,
            baseline_cost=budget.baseline_cost,
            rho=budget.rho,
            cost_budget=budget.budget,
            actual_robust_cost=None,
            actual_price_of_fairness=None,
            y_values=None,
            x_values=None,
            runtime=runtime,
            lexicographic_cost_stage_used=False,
        )
    t_star = float(t.X)
    lex_used = False
    if lexicographic_cost_stage:
        model.addConstr(t <= t_star + float(t_tolerance), name="lexicographic_T_cap")
        model.setObjective(first_stage + robust_recourse, GRB.MINIMIZE)
        remaining = float(time_limit) - (time.perf_counter() - start)
        model.Params.TimeLimit = max(1.0e-3, remaining)
        model.optimize()
        status = gurobi_status_name(model.Status)
        if model.Status != GRB.OPTIMAL:
            runtime = time.perf_counter() - start
            model.dispose()
            return FairnessExtensiveFormResult(
                status=f"lexicographic_{status}",
                objective_t=None,
                robust_minimum_fill_rate=None,
                baseline_cost=budget.baseline_cost,
                rho=budget.rho,
                cost_budget=budget.budget,
                actual_robust_cost=None,
                actual_price_of_fairness=None,
                y_values=None,
                x_values=None,
                runtime=runtime,
                lexicographic_cost_stage_used=True,
            )
        lex_used = True

    y_values = [float(y[i].X) for i in instance.I]
    x_values = [[float(x[i, j].X) for j in instance.J] for i in instance.I]
    policies = [_policy_from_solution(instance, scenario, q, u, e) for scenario, q, u, e, _ in recourse]
    first_value = first_stage_cost_value(instance, y_values, x_values)
    worst_recourse = max(policy.recourse_cost for policy in policies)
    actual_cost = first_value + worst_recourse
    price = 0.0 if budget.baseline_cost <= FAIRNESS_METRIC_TOLERANCE else actual_cost / budget.baseline_cost - 1.0
    cost_worst = max(policies, key=lambda policy: policy.recourse_cost).scenario_name
    fairness_worst = max(
        policies,
        key=lambda policy: -math.inf if policy.minimum_fill_rate is None else 1.0 - policy.minimum_fill_rate,
    ).scenario_name
    wgap_values = [policy.fill_rate_gap for policy in policies if policy.fill_rate_gap is not None]
    wwd_values = [policy.worst_region_deviation for policy in policies if policy.worst_region_deviation is not None]
    mean_values = [
        policy.weighted_mean_fill_rate for policy in policies if policy.weighted_mean_fill_rate is not None
    ]
    result = FairnessExtensiveFormResult(
        status="optimal",
        objective_t=t_star,
        robust_minimum_fill_rate=1.0 - t_star,
        baseline_cost=budget.baseline_cost,
        rho=budget.rho,
        cost_budget=budget.budget,
        actual_robust_cost=float(actual_cost),
        actual_price_of_fairness=float(price),
        y_values=y_values,
        x_values=x_values,
        scenario_policies=policies,
        cost_worst_scenario=cost_worst,
        fairness_worst_scenario=fairness_worst,
        wgap=None if not wgap_values else float(max(wgap_values)),
        wwd=None if not wwd_values else float(max(wwd_values)),
        weighted_mean_fill_rate=None if not mean_values else float(min(mean_values)),
        runtime=time.perf_counter() - start,
        lexicographic_cost_stage_used=lex_used,
    )
    model.dispose()
    return result


def fairness_cut_from_ray(
    instance: InventoryInstance,
    *,
    cost_budget_value: float,
    demand_values: list[list[float]],
    ray: FairnessFarkasRay,
    active_deviations: list[dict[str, int]],
) -> FairnessFeasibilityCut:
    regional_demand = [sum(float(demand_values[r][j]) for j in instance.J) for r in instance.R]
    service_rhs = [
        (1.0 - instance.service_level[j])
        * sum(float(demand_values[r][j]) for r in instance.R)
        for j in instance.J
    ]
    constant = (
        sum(service_rhs[j] * ray.service[j] for j in instance.J)
        + float(cost_budget_value) * ray.cost
        - sum(float(demand_values[r][j]) * ray.demand[r][j] for r in instance.R for j in instance.J)
    )
    return FairnessFeasibilityCut(
        constant=float(constant),
        y_coefficients=[-ray.cost * instance.fixed_cost[i] for i in instance.I],
        x_coefficients=[
            [ray.supply[i][j] - ray.cost * instance.inventory_cost[i][j] for j in instance.J]
            for i in instance.I
        ],
        t_coefficient=float(
            sum(regional_demand[r] * ray.regional_fairness[r] for r in instance.R)
        ),
        active_deviations=[dict(value) for value in active_deviations],
        demand_values=[[float(value) for value in row] for row in demand_values],
        ray=ray,
    )


def validate_farkas_ray(
    instance: InventoryInstance,
    ray: FairnessFarkasRay,
    *,
    tolerance: float = FAIRNESS_FEASIBILITY_TOLERANCE,
) -> bool:
    if (
        len(ray.demand) != instance.num_regions
        or any(len(row) != instance.num_products for row in ray.demand)
        or len(ray.supply) != instance.num_warehouses
        or any(len(row) != instance.num_products for row in ray.supply)
        or len(ray.service) != instance.num_products
        or len(ray.regional_fairness) != instance.num_regions
    ):
        return False
    values = (
        [value for row in ray.demand for value in row]
        + [value for row in ray.supply for value in row]
        + list(ray.service)
        + [ray.cost]
        + list(ray.regional_fairness)
    )
    tol = float(tolerance)
    if any(not math.isfinite(float(value)) or float(value) < -tol for value in values):
        return False
    if sum(float(value) for value in values) > 1.0 + tol:
        return False
    for i in instance.I:
        for r in instance.R:
            for j in instance.J:
                if (
                    -ray.demand[r][j]
                    + ray.supply[i][j]
                    + instance.transport_cost[i][r][j] * ray.cost
                    < -tol
                ):
                    return False
    for r in instance.R:
        for j in instance.J:
            if (
                -ray.demand[r][j]
                + ray.service[j]
                + instance.shortage_penalty[r][j] * ray.cost
                + ray.regional_fairness[r]
                < -tol
            ):
                return False
    return all(
        -ray.service[j] + instance.service_penalty[j] * ray.cost >= -tol
        for j in instance.J
    )


def _add_binary_product(model: gp.Model, binary: Any, continuous: Any, name: str) -> Any:
    product = model.addVar(lb=0.0, ub=1.0, name=name)
    model.addConstr(product <= binary, name=f"{name}_binary_ub")
    model.addConstr(product <= continuous, name=f"{name}_continuous_ub")
    model.addConstr(product >= continuous - (1.0 - binary), name=f"{name}_lower")
    return product


def separate_robust_fairness_feasibility(
    instance: InventoryInstance,
    *,
    y_values: list[float],
    x_values: list[list[float]],
    t_value: float,
    cost_budget_value: float,
    gamma: int,
    mip_gap: float = 0.0,
    time_limit: float = 120.0,
    feasibility_tolerance: float = FAIRNESS_FEASIBILITY_TOLERANCE,
    output_flag: bool = False,
) -> FairnessSeparationResult:
    """Maximize a normalized Farkas violation over the Gamma uncertainty set.

    For fixed demand, recourse feasibility is ``A v <= b(x,y,T,d), v>=0``.
    A normalized ray ``pi>=0, A'pi>=0`` yields the valid master inequality
    ``b(x,y,T,d)'pi >= 0``.  Binary-continuous McCormick products combine this
    ray with the budgeted demand pattern without enumerating every scenario.
    """
    start = time.perf_counter()
    if len(y_values) != instance.num_warehouses:
        raise ValueError("y_values has the wrong shape.")
    if len(x_values) != instance.num_warehouses or any(
        len(row) != instance.num_products for row in x_values
    ):
        raise ValueError("x_values has the wrong shape.")
    if not math.isfinite(float(t_value)) or float(t_value) < 0.0:
        raise ValueError("t_value must be finite and nonnegative.")

    model = gp.Model("robust_regional_fairness_separation")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.TimeLimit = max(1.0e-3, float(time_limit))
    model.Params.MIPGap = max(0.0, float(mip_gap))
    z = model.addVars(instance.R, instance.J, vtype=GRB.BINARY, name="z")
    a = model.addVars(instance.R, instance.J, lb=0.0, ub=1.0, name="pi_demand")
    b = model.addVars(instance.I, instance.J, lb=0.0, ub=1.0, name="pi_supply")
    c = model.addVars(instance.J, lb=0.0, ub=1.0, name="pi_service")
    k = model.addVar(lb=0.0, ub=1.0, name="pi_cost")
    ell = model.addVars(instance.R, lb=0.0, ub=1.0, name="pi_fairness")
    model.addConstr(gp.quicksum(z[r, j] for r in instance.R for j in instance.J) <= int(gamma), name="gamma")
    model.addConstr(
        gp.quicksum(a[r, j] for r in instance.R for j in instance.J)
        + gp.quicksum(b[i, j] for i in instance.I for j in instance.J)
        + gp.quicksum(c[j] for j in instance.J)
        + k
        + gp.quicksum(ell[r] for r in instance.R)
        <= 1.0,
        name="ray_normalization",
    )
    for i in instance.I:
        for r in instance.R:
            for j in instance.J:
                model.addConstr(
                    -a[r, j] + b[i, j] + instance.transport_cost[i][r][j] * k >= 0.0,
                    name=f"dual_q[{i},{r},{j}]",
                )
    for r in instance.R:
        for j in instance.J:
            model.addConstr(
                -a[r, j]
                + c[j]
                + instance.shortage_penalty[r][j] * k
                + ell[r]
                >= 0.0,
                name=f"dual_u[{r},{j}]",
            )
    for j in instance.J:
        model.addConstr(-c[j] + instance.service_penalty[j] * k >= 0.0, name=f"dual_e[{j}]")

    za = {(r, j): _add_binary_product(model, z[r, j], a[r, j], f"za[{r},{j}]") for r in instance.R for j in instance.J}
    zc = {(r, j): _add_binary_product(model, z[r, j], c[j], f"zc[{r},{j}]") for r in instance.R for j in instance.J}
    zl = {(r, j): _add_binary_product(model, z[r, j], ell[r], f"zl[{r},{j}]") for r in instance.R for j in instance.J}
    first_stage = first_stage_cost_value(instance, y_values, x_values)
    objective = gp.quicksum(
        instance.base_demand[r][j] * a[r, j]
        + instance.demand_deviation[r][j] * za[r, j]
        for r in instance.R
        for j in instance.J
    )
    objective -= gp.quicksum(float(x_values[i][j]) * b[i, j] for i in instance.I for j in instance.J)
    objective -= gp.quicksum(
        (1.0 - instance.service_level[j])
        * (
            sum(instance.base_demand[r][j] for r in instance.R) * c[j]
            + gp.quicksum(instance.demand_deviation[r][j] * zc[r, j] for r in instance.R)
        )
        for j in instance.J
    )
    objective -= (float(cost_budget_value) - first_stage) * k
    objective -= float(t_value) * gp.quicksum(
        sum(instance.base_demand[r][j] for j in instance.J) * ell[r]
        + gp.quicksum(instance.demand_deviation[r][j] * zl[r, j] for j in instance.J)
        for r in instance.R
    )
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    status = gurobi_status_name(model.Status)
    has_incumbent = model.SolCount > 0
    objective_value = float(model.ObjVal) if has_incumbent else None
    objective_bound = float(model.ObjBound) if model.Status not in {GRB.INFEASIBLE, GRB.UNBOUNDED} else None
    mip_gap_value = float(model.MIPGap) if has_incumbent and model.IsMIP else None
    certified = objective_bound is not None and objective_bound <= float(feasibility_tolerance)
    cut: FairnessFeasibilityCut | None = None
    if has_incumbent and objective_value is not None and objective_value > float(feasibility_tolerance):
        active = [
            (r, j) for r in instance.R for j in instance.J if float(z[r, j].X) >= 0.5
        ]
        demand_values = scenario_demand(instance, active)
        ray = FairnessFarkasRay(
            demand=[[float(a[r, j].X) for j in instance.J] for r in instance.R],
            supply=[[float(b[i, j].X) for j in instance.J] for i in instance.I],
            service=[float(c[j].X) for j in instance.J],
            cost=float(k.X),
            regional_fairness=[float(ell[r].X) for r in instance.R],
        )
        if not validate_farkas_ray(instance, ray, tolerance=10.0 * float(feasibility_tolerance)):
            model.dispose()
            raise RuntimeError("Separation incumbent did not define a valid normalized Farkas ray.")
        cut = fairness_cut_from_ray(
            instance,
            cost_budget_value=float(cost_budget_value),
            demand_values=demand_values,
            ray=ray,
            active_deviations=[{"region": r, "product": j} for r, j in active],
        )
    result = FairnessSeparationResult(
        status=status,
        has_incumbent=has_incumbent,
        objective=objective_value,
        objective_bound=objective_bound,
        mip_gap=mip_gap_value,
        runtime=time.perf_counter() - start,
        requested_mip_gap=float(mip_gap),
        robust_feasibility_certified=certified,
        cut=cut,
    )
    model.dispose()
    return result


def solve_scenario_policy_with_shared_caps(
    instance: InventoryInstance,
    scenario: DemandScenario,
    *,
    y_values: list[float],
    x_values: list[list[float]],
    t_value: float,
    cost_budget_value: float,
    feasibility_tolerance: float = 0.0,
    time_limit: float = 30.0,
    output_flag: bool = False,
) -> FairnessScenarioPolicy:
    """Recover one policy satisfying the cost and fairness caps simultaneously."""
    model = gp.Model(f"fairness_policy_{scenario.name}")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.Method = 1
    model.Params.TimeLimit = max(1.0e-3, float(time_limit))
    fixed_x = {(i, j): float(x_values[i][j]) for i in instance.I for j in instance.J}
    q, u, e, transport, shortage, service = _recourse_expressions(
        model, instance, scenario, fixed_x, prefix="fixed"
    )
    recourse_cost = transport + shortage + service
    remaining = float(cost_budget_value) - first_stage_cost_value(instance, y_values, x_values)
    model.addConstr(
        recourse_cost <= remaining + float(feasibility_tolerance),
        name="shared_cost_cap",
    )
    for r in instance.R:
        demand = sum(scenario.demand[r][j] for j in instance.J)
        if demand > FAIRNESS_METRIC_TOLERANCE:
            model.addConstr(
                gp.quicksum(u[r, j] for j in instance.J)
                <= (float(t_value) + float(feasibility_tolerance)) * demand,
                name=f"shared_fairness_cap[{r}]",
            )
    model.setObjective(recourse_cost, GRB.MINIMIZE)
    model.optimize()
    status = gurobi_status_name(model.Status)
    if model.Status != GRB.OPTIMAL:
        model.dispose()
        raise RuntimeError(f"Shared-cap recourse ended with status {status}.")
    policy = _policy_from_solution(instance, scenario, q, u, e)
    model.dispose()
    return policy


def evaluate_fairness_solution(
    instance: InventoryInstance,
    *,
    y_values: list[float],
    x_values: list[list[float]],
    t_value: float,
    baseline_cost: float,
    rho: float,
    gamma: int,
    max_scenarios: int = 5000,
    per_scenario_time_limit: float = 30.0,
    tolerance: float = FAIRNESS_FEASIBILITY_TOLERANCE,
    output_flag: bool = False,
) -> FairnessSolutionEvaluation:
    """Post-evaluate shared-cap policies without changing algorithm bounds."""
    start = time.perf_counter()
    budget = fairness_cost_budget(baseline_cost, rho)
    enumeration = enumerate_budget_scenarios_with_metadata(
        instance, gamma, max_scenarios=max_scenarios, exact_scenarios=True
    )
    policies: list[FairnessScenarioPolicy] = []
    errors: list[str] = []
    first_cost = first_stage_cost_value(instance, y_values, x_values)
    for scenario in enumeration.scenarios:
        try:
            policy = solve_scenario_policy_with_shared_caps(
                instance,
                scenario,
                y_values=y_values,
                x_values=x_values,
                t_value=t_value,
                cost_budget_value=budget.budget,
                feasibility_tolerance=tolerance,
                time_limit=per_scenario_time_limit,
                output_flag=output_flag,
            )
            if first_cost + policy.recourse_cost > budget.budget + float(tolerance):
                raise RuntimeError("Recovered policy exceeds the shared robust cost budget.")
            if (
                policy.minimum_fill_rate is not None
                and policy.minimum_fill_rate < 1.0 - t_value - float(tolerance)
            ):
                raise RuntimeError("Recovered policy violates the regional max-shortage-rate cap.")
            policies.append(policy)
        except Exception as exc:  # noqa: BLE001 - invalid evaluation is explicit.
            errors.append(f"{scenario.name}: {type(exc).__name__}: {exc}")
    inventory_by_warehouse = [sum(float(x_values[i][j]) for j in instance.J) for i in instance.I]
    inventory_by_product = [sum(float(x_values[i][j]) for i in instance.I) for j in instance.J]
    common = {
        "scenario_count": len(enumeration.scenarios),
        "opened_warehouses": sum(float(value) >= 0.5 for value in y_values),
        "total_inventory": float(sum(inventory_by_warehouse)),
        "inventory_by_warehouse": [float(value) for value in inventory_by_warehouse],
        "inventory_by_product": [float(value) for value in inventory_by_product],
        "runtime": time.perf_counter() - start,
    }
    if errors or len(policies) != len(enumeration.scenarios):
        return FairnessSolutionEvaluation(
            valid=False,
            actual_robust_cost=None,
            actual_price_of_fairness=None,
            wgap=None,
            wminfr=None,
            wwd=None,
            minimum_weighted_mean_fill_rate=None,
            cost_worst_scenario=None,
            fairness_worst_scenario=None,
            errors=errors,
            **common,
        )
    worst_recourse = max(policy.recourse_cost for policy in policies)
    actual_cost = first_cost + worst_recourse
    actual_price = (
        0.0
        if budget.baseline_cost <= FAIRNESS_METRIC_TOLERANCE
        else actual_cost / budget.baseline_cost - 1.0
    )
    applicable_minimum = [policy.minimum_fill_rate for policy in policies if policy.minimum_fill_rate is not None]
    gaps = [policy.fill_rate_gap for policy in policies if policy.fill_rate_gap is not None]
    deviations = [policy.worst_region_deviation for policy in policies if policy.worst_region_deviation is not None]
    means = [policy.weighted_mean_fill_rate for policy in policies if policy.weighted_mean_fill_rate is not None]
    cost_worst = max(policies, key=lambda policy: policy.recourse_cost).scenario_name
    fairness_worst = min(
        (policy for policy in policies if policy.minimum_fill_rate is not None),
        key=lambda policy: float(policy.minimum_fill_rate),
        default=None,
    )
    return FairnessSolutionEvaluation(
        valid=True,
        actual_robust_cost=float(actual_cost),
        actual_price_of_fairness=float(actual_price),
        wgap=None if not gaps else float(max(gaps)),
        wminfr=None if not applicable_minimum else float(min(applicable_minimum)),
        wwd=None if not deviations else float(max(deviations)),
        minimum_weighted_mean_fill_rate=None if not means else float(min(means)),
        cost_worst_scenario=cost_worst,
        fairness_worst_scenario=None if fairness_worst is None else fairness_worst.scenario_name,
        errors=[],
        **common,
    )
