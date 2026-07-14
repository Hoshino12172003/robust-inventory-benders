from __future__ import annotations

from typing import Any

from gurobipy import GRB


def gurobi_status_name(status: int) -> str:
    status = int(status)
    if status == GRB.OPTIMAL:
        return "optimal"
    if status == GRB.TIME_LIMIT:
        return "time_limit"
    if status == GRB.SUBOPTIMAL:
        return "suboptimal"
    if status == GRB.INFEASIBLE:
        return "infeasible"
    if status == GRB.UNBOUNDED:
        return "unbounded"
    return f"gurobi_status_{status}"


def normalize_run_status(status: Any) -> str:
    if isinstance(status, bool):
        return str(status).lower()
    if isinstance(status, int):
        return gurobi_status_name(status)

    normalized = str(status).strip().lower()
    if normalized in {"9", "gurobi_status_9", "grb.time_limit"}:
        return "time_limit"
    return normalized
