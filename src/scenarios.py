from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterable

from .instance import InventoryInstance


@dataclass(frozen=True)
class DemandScenario:
    name: str
    active_units: tuple[tuple[int, int], ...]
    demand: tuple[tuple[float, ...], ...]

    @property
    def gamma(self) -> int:
        return len(self.active_units)


def _all_units(instance: InventoryInstance) -> list[tuple[int, int]]:
    return [(r, j) for r in instance.R for j in instance.J]


def _scenario_from_units(instance: InventoryInstance, units: Iterable[tuple[int, int]]) -> DemandScenario:
    active = tuple(sorted(units))
    demand = [row[:] for row in instance.base_demand]
    for r, j in active:
        demand[r][j] += instance.demand_deviation[r][j]
    suffix = "base" if not active else "_".join(f"r{r}j{j}" for r, j in active)
    return DemandScenario(
        name=f"g{len(active)}_{suffix}",
        active_units=active,
        demand=tuple(tuple(float(v) for v in row) for row in demand),
    )


def count_budget_scenarios(instance: InventoryInstance, gamma: int) -> int:
    n = instance.num_regions * instance.num_products
    gamma = min(max(0, int(gamma)), n)
    return sum(comb(n, k) for k in range(gamma + 1))


def enumerate_budget_scenarios(
    instance: InventoryInstance,
    gamma: int,
    max_scenarios: int = 5000,
    exact_scenarios: bool = True,
) -> list[DemandScenario]:
    units = _all_units(instance)
    gamma = min(max(0, int(gamma)), len(units))
    total = count_budget_scenarios(instance, gamma)
    if total <= max_scenarios:
        scenarios: list[DemandScenario] = []
        for k in range(gamma + 1):
            for active in combinations(units, k):
                scenarios.append(_scenario_from_units(instance, active))
        return scenarios
    if exact_scenarios:
        raise ValueError("Exact scenario enumeration exceeds max_scenarios.")
    return candidate_budget_scenarios(instance, gamma, max_scenarios)


def scenario_metadata(
    instance: InventoryInstance,
    gamma: int,
    max_scenarios: int,
    exact_scenarios: bool,
    num_scenarios_used: int,
) -> dict[str, int | bool | str]:
    total = count_budget_scenarios(instance, gamma)
    return {
        "scenario_mode": "full" if total <= max_scenarios else "candidate",
        "exact_scenarios": exact_scenarios,
        "num_scenarios_used": num_scenarios_used,
        "num_scenarios_total_estimated": total,
        "max_scenarios": max_scenarios,
    }


def candidate_budget_scenarios(
    instance: InventoryInstance,
    gamma: int,
    max_scenarios: int,
) -> list[DemandScenario]:
    units = _all_units(instance)
    ranked = sorted(
        units,
        key=lambda u: instance.demand_deviation[u[0]][u[1]] * max(instance.shortage_penalty[u[0]][u[1]], 1.0),
        reverse=True,
    )
    scenarios = [_scenario_from_units(instance, [])]
    for k in range(1, min(gamma, len(ranked)) + 1):
        scenarios.append(_scenario_from_units(instance, ranked[:k]))
    for unit in ranked:
        scenarios.append(_scenario_from_units(instance, [unit]))
        if len(scenarios) >= max_scenarios:
            break
    seen: set[tuple[tuple[int, int], ...]] = set()
    unique = []
    for scenario in scenarios:
        if scenario.active_units not in seen:
            unique.append(scenario)
            seen.add(scenario.active_units)
    return unique[:max_scenarios]
