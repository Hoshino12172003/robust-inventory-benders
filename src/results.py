from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SolveResult:
    method: str
    status: str
    objective: float | None
    lower_bound: float | None
    upper_bound: float | None
    gap: float | None
    runtime: float
    iterations: int = 0
    cuts: int = 0
    master_runtime: float = 0.0
    subproblem_runtime: float = 0.0
    robust_cost: float | None = None
    first_stage_cost: float | None = None
    gamma_target: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    iteration_log: list[dict[str, Any]] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "status": self.status,
            "objective": self.objective,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "gap": self.gap,
            "runtime": self.runtime,
            "iterations": self.iterations,
            "cuts": self.cuts,
            "master_runtime": self.master_runtime,
            "subproblem_runtime": self.subproblem_runtime,
            "robust_cost": self.robust_cost,
            "first_stage_cost": self.first_stage_cost,
            "gamma_target": self.gamma_target,
            **self.metadata,
        }
