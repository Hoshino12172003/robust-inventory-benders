from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InventoryInstance:
    name: str
    num_warehouses: int
    num_products: int
    num_regions: int
    fixed_cost: list[float]
    inventory_cost: list[list[float]]
    capacity: list[float]
    volume: list[float]
    budget: float
    transport_cost: list[list[list[float]]]
    shortage_penalty: list[list[float]]
    service_penalty: list[float]
    service_level: list[float]
    base_demand: list[list[float]]
    demand_deviation: list[list[float]]
    inventory_ub: list[list[float]]

    @property
    def I(self) -> range:
        return range(self.num_warehouses)

    @property
    def J(self) -> range:
        return range(self.num_products)

    @property
    def R(self) -> range:
        return range(self.num_regions)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InventoryInstance":
        return cls(**data)


def generate_instance(config: dict[str, Any], seed: int | None = None) -> InventoryInstance:
    instance_cfg = config.get("instance", {})
    rng = np.random.default_rng(config.get("seed", 42) if seed is None else seed)

    n_i = int(instance_cfg.get("num_warehouses", 3))
    n_j = int(instance_cfg.get("num_products", 2))
    n_r = int(instance_cfg.get("num_regions", 4))
    budget_factor = float(instance_cfg.get("budget_factor", 0.58))

    base_demand = rng.integers(35, 95, size=(n_r, n_j)).astype(float)
    demand_deviation = np.maximum(5.0, np.round(base_demand * rng.uniform(0.15, 0.45, size=(n_r, n_j)), 2))
    total_product_demand = base_demand.sum(axis=0)

    fixed_cost = rng.integers(120, 280, size=n_i).astype(float)
    inventory_cost = rng.uniform(1.4, 4.2, size=(n_i, n_j)).round(2)
    volume = rng.uniform(0.8, 1.8, size=n_j).round(2)
    capacity = rng.uniform(0.55, 0.9, size=n_i)
    capacity = np.round(capacity / capacity.sum() * float(total_product_demand.sum()) * 1.18, 2)
    inventory_ub = np.zeros((n_i, n_j))
    for i in range(n_i):
        for j in range(n_j):
            inventory_ub[i, j] = round(min(capacity[i] / volume[j], total_product_demand[j] * 1.35), 2)

    transport_cost = rng.uniform(0.6, 5.5, size=(n_i, n_r, n_j)).round(2)
    shortage_penalty = rng.uniform(10.0, 20.0, size=(n_r, n_j)).round(2)
    service_penalty = (shortage_penalty.max(axis=0) * rng.uniform(1.5, 2.2, size=n_j)).round(2)
    service_level = rng.uniform(0.82, 0.94, size=n_j).round(2)

    full_open_cost = float(fixed_cost.sum() + (inventory_cost * inventory_ub * 0.45).sum())
    budget = round(full_open_cost * budget_factor, 2)

    return InventoryInstance(
        name=f"inventory_I{n_i}_J{n_j}_R{n_r}_seed{seed if seed is not None else config.get('seed', 42)}",
        num_warehouses=n_i,
        num_products=n_j,
        num_regions=n_r,
        fixed_cost=fixed_cost.round(2).tolist(),
        inventory_cost=inventory_cost.tolist(),
        capacity=capacity.tolist(),
        volume=volume.tolist(),
        budget=budget,
        transport_cost=transport_cost.tolist(),
        shortage_penalty=shortage_penalty.tolist(),
        service_penalty=service_penalty.tolist(),
        service_level=service_level.tolist(),
        base_demand=base_demand.tolist(),
        demand_deviation=demand_deviation.tolist(),
        inventory_ub=inventory_ub.tolist(),
    )


def save_instance(instance: InventoryInstance, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(instance.to_dict(), indent=2), encoding="utf-8")
    return target


def load_instance(path: str | Path) -> InventoryInstance:
    return InventoryInstance.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
