from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Literal

import yaml

from .experiment_protocol import atomic_write_json, canonical_json_sha256, file_sha256
from .experiment_suite import INSTANCE_SIZES
from .instance import InventoryInstance, generate_instance
from .scenarios import enumerate_budget_scenarios


Mode = Literal["k1", "k2", "full"]
ROOT = Path(__file__).resolve().parents[1]
FORMAL_SEEDS = tuple(range(230, 240))
FALLBACK_SEEDS = tuple(range(230, 235))
MODES: tuple[Mode, ...] = ("k1", "k2", "full")
GAMMA = 2
RHO = 0.025
RESULT_ROOT = Path("experiments/results_fulfillment_flexibility/formal")
FORMAL_PATH_MARKERS = (
    "fulfillment_flexibility_formal",
    "results_fulfillment_flexibility/formal",
)


class FormalProtocolError(RuntimeError):
    pass


class FormalOptimizationProhibited(FormalProtocolError):
    pass


def _git_commit(root: Path = ROOT) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FormalProtocolError(f"invalid configuration: {path}")
    return payload


def build_eligibility(instance: InventoryInstance, mode: Mode) -> dict[int, tuple[int, ...]]:
    """Return candidate sources ranked only by arithmetic mean transport cost."""
    if mode == "full":
        return {r: tuple(instance.I) for r in instance.R}
    if mode not in ("k1", "k2"):
        raise ValueError(f"unsupported fulfillment mode: {mode}")
    count = 1 if mode == "k1" else min(2, instance.num_warehouses)
    eligibility: dict[int, tuple[int, ...]] = {}
    for r in instance.R:
        ranked = sorted(
            instance.I,
            key=lambda i: (
                math.fsum(instance.transport_cost[i][r][j] for j in instance.J)
                / instance.num_products,
                i,
            ),
        )
        eligibility[r] = tuple(ranked[:count])
    return eligibility


def eligibility_identity() -> str:
    return hashlib.sha256(inspect.getsource(build_eligibility).encode("utf-8")).hexdigest().upper()


def validate_config(config: dict[str, Any], *, expected_scale: str | None = None) -> None:
    scale = str(config.get("scale", ""))
    if expected_scale is not None and scale != expected_scale:
        raise FormalProtocolError("scale/config mismatch")
    if scale not in {"medium_large", "large"}:
        raise FormalProtocolError("formal scale must be medium_large or large")
    if config.get("stage") != "FORMAL_PROTOCOL_ONLY":
        raise FormalProtocolError("formal stage drifted")
    if config.get("formal_run_authorized") is not False:
        raise FormalProtocolError("protocol PR must prohibit formal optimization")
    if tuple(config.get("seeds", ())) != FORMAL_SEEDS:
        raise FormalProtocolError("formal seed list drifted")
    if tuple(config.get("fallback_seeds", ())) != FALLBACK_SEEDS:
        raise FormalProtocolError("fallback seed list drifted")
    if tuple(config.get("modes", ())) != MODES:
        raise FormalProtocolError("fulfillment modes drifted")
    if int(config.get("gamma", -1)) != GAMMA or float(config.get("rho", -1.0)) != RHO:
        raise FormalProtocolError("Gamma or rho drifted")
    if Path(config.get("output_dir", "")) != RESULT_ROOT:
        raise FormalProtocolError("formal output root drifted")
    expected = INSTANCE_SIZES[scale]
    dimensions = config.get("instance", {})
    for key, value in expected.items():
        if int(dimensions.get(key, -1)) != int(value):
            raise FormalProtocolError(f"{scale} dimension drifted: {key}")
    if config.get("overwrite_supported") is not False:
        raise FormalProtocolError("formal overwrite must remain disabled")
    if config.get("solver_backend") != "exact_finite_scenario_extensive_form":
        raise FormalProtocolError("unreviewed formal solver backend")


def _walk_seed_values(value: Any, path: tuple[str, ...] = ()) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (str(key),)
            if "seed" in str(key).lower() and isinstance(child, (int, list, tuple)):
                values = [child] if isinstance(child, int) else child
                for item in values:
                    if isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 999:
                        found.append((int(item), ".".join(child_path)))
            found.extend(_walk_seed_values(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_seed_values(child, path + (str(index),)))
    return found


def seed_nonreuse_audit(root: Path = ROOT) -> dict[str, Any]:
    structured_hits: dict[int, set[str]] = {}
    scanned = 0
    for directory in ("configs", "experiments/configs", "analysis", "real_data_studies"):
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
                continue
            relative = path.relative_to(root).as_posix()
            if any(marker in relative for marker in FORMAL_PATH_MARKERS):
                continue
            try:
                value = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, yaml.YAMLError):
                continue
            scanned += 1
            for seed, key_path in _walk_seed_values(value):
                structured_hits.setdefault(seed, set()).add(f"{relative}:{key_path}")

    explicit_text_hits: dict[int, set[str]] = {}
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.splitlines()
    for relative in tracked:
        normalized = relative.replace("\\", "/")
        if any(marker in normalized for marker in FORMAL_PATH_MARKERS):
            continue
        path = root / relative
        if path.suffix.lower() not in {".py", ".md", ".json", ".yaml", ".yml", ".csv", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        patterns = (
            re.compile(r"(?i)\bseed\s*[:=]\s*(\d{1,3})(?!\d)"),
            re.compile(r"(?i)\bseed[_-](\d{1,3})(?!\d)"),
            re.compile(r"(?i)\bseed\s+(\d{1,3})(?!\d)"),
        )
        for pattern in patterns:
            for match in pattern.finditer(text):
                seed = int(match.group(1))
                explicit_text_hits.setdefault(seed, set()).add(normalized)
        for match in re.finditer(r"(?i)\bseeds\s*[:=]\s*\[([^\]\n]*)\]", text):
            for token in re.findall(r"(?<!\d)\d{1,3}(?!\d)", match.group(1)):
                explicit_text_hits.setdefault(int(token), set()).add(normalized)

    prior_hits = {seed: sorted(paths) for seed, paths in sorted(structured_hits.items())}
    main_used = sorted(prior_hits)
    external_development = {
        "source": "Draft PR #79 development-only diagnostic",
        "commit": "d7a0cbf7be5a3f1a7f0b6ab8758e0e6281da51fd",
        "seeds": [190, 191, 192],
    }
    conflicts = {
        str(seed): sorted(set(prior_hits.get(seed, ())) | explicit_text_hits.get(seed, set()))
        for seed in FORMAL_SEEDS
        if prior_hits.get(seed) or explicit_text_hits.get(seed)
    }
    archive_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".zip", ".tar", ".gz"}
    )
    return {
        "schema": "fulfillment_flexibility_formal_seed_audit_v1",
        "audited_git_commit": _git_commit(root),
        "structured_files_scanned": scanned,
        "tracked_text_files_scanned": len(tracked),
        "archive_files_present": archive_files,
        "main_structured_seed_minimum": min(main_used) if main_used else None,
        "main_structured_seed_maximum": max(main_used) if main_used else None,
        "main_structured_seed_values": main_used,
        "explicit_tracked_text_seed_values": sorted(explicit_text_hits),
        "explicit_tracked_text_seed_sources": {
            str(seed): sorted(paths) for seed, paths in sorted(explicit_text_hits.items())
        },
        "external_development_reservation": external_development,
        "formal_seeds": list(FORMAL_SEEDS),
        "fallback_seeds": list(FALLBACK_SEEDS),
        "formal_seed_conflicts": conflicts,
        "formal_seeds_untouched": not conflicts,
        "evidence_boundary": (
            "Structured repository metadata and candidate-specific tracked-text search; "
            "real-data temporal units are not synthetic seeds."
        ),
    }


def scenario_count(num_regions: int, num_products: int, gamma: int = GAMMA) -> int:
    components = int(num_regions) * int(num_products)
    return sum(math.comb(components, order) for order in range(int(gamma) + 1))


def model_size_estimate(config: dict[str, Any], mode: Mode) -> dict[str, int]:
    dimensions = config["instance"]
    i_count = int(dimensions["num_warehouses"])
    j_count = int(dimensions["num_products"])
    r_count = int(dimensions["num_regions"])
    scenarios = scenario_count(r_count, j_count, int(config["gamma"]))
    eligible_count = i_count if mode == "full" else (1 if mode == "k1" else min(2, i_count))
    recourse_columns = eligible_count * r_count * j_count + r_count * j_count + j_count
    first_columns = i_count + i_count * j_count + 1
    common_rows = r_count * j_count + i_count * j_count + j_count
    base_rows = i_count + i_count * j_count + 1
    return {
        "scenario_count": scenarios,
        "columns": first_columns + scenarios * recourse_columns,
        "anchor_rows": base_rows + scenarios * (common_rows + 1),
        "service_rows": base_rows + scenarios * (common_rows + 1 + r_count),
        "omitted_ineligible_arc_variables_per_scenario": (
            i_count - eligible_count
        ) * r_count * j_count,
    }


def task_matrix(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for config in configs:
        for seed in config["seeds"]:
            for phase in (
                "reoptimized_cost_anchor",
                "reoptimized_service",
                "fixed_first_stage_cost_anchor",
                "fixed_first_stage_service",
            ):
                for mode in MODES:
                    tasks.append({
                        "scale": config["scale"],
                        "seed": int(seed),
                        "phase": phase,
                        "mode": mode,
                    })
    return tasks


def static_audit(config_paths: list[Path], root: Path = ROOT) -> dict[str, Any]:
    configs = [load_config(path) for path in config_paths]
    for config in configs:
        validate_config(config)
    seed_audit = seed_nonreuse_audit(root)
    tasks = task_matrix(configs)
    limit = max(float(config["solver"]["time_limit_seconds"]) for config in configs)
    estimates = {
        config["scale"]: {mode: model_size_estimate(config, mode) for mode in MODES}
        for config in configs
    }
    ram = None
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            ram = int(completed.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            ram = None
    disk = __import__("shutil").disk_usage(root)
    minimum_ram = max(int(config["fallback_rule"]["triggers"]["minimum_ram_bytes"]) for config in configs)
    minimum_disk = max(int(config["fallback_rule"]["triggers"]["minimum_free_disk_bytes"]) for config in configs)
    sequential_hours = len(tasks) * limit / 3600.0
    hardware_safe = ram is not None and ram >= minimum_ram and disk.free >= minimum_disk
    return {
        "schema": "fulfillment_flexibility_formal_static_audit_v1",
        "audited_git_commit": _git_commit(root),
        "config_sha256": {path.name: file_sha256(path).upper() for path in config_paths},
        "protocol_sha256": file_sha256(root / "docs/fulfillment_flexibility_formal_protocol.md").upper()
        if (root / "docs/fulfillment_flexibility_formal_protocol.md").exists() else None,
        "seed_audit_pass": seed_audit["formal_seeds_untouched"],
        "formal_instance_count": len(configs) * len(FORMAL_SEEDS),
        "optimization_task_count": len(tasks),
        "worst_case_sequential_hours": sequential_hours,
        "model_size_estimates": estimates,
        "host_total_ram_bytes": ram,
        "host_free_disk_bytes": disk.free,
        "primary_hardware_gate_pass": hardware_safe,
        "fallback_rule_frozen": all(
            config["fallback_rule"]["frozen_before_optimization"] is True for config in configs
        ),
        "formal_optimization_authorized": False,
        "decision": "protocol_ready_but_formal_optimization_prohibited",
        "scientific_concern": (
            "The exact large extensive form has about 4.06 million columns. "
            "Current-host execution is blocked unless the hardware gate passes or a separately "
            "reviewed eligibility-aware scalable solver is introduced without changing the model."
        ),
    }


def protocol_identity(config_path: Path, root: Path = ROOT) -> dict[str, Any]:
    return {
        "source_commit": _git_commit(root),
        "protocol_sha256": file_sha256(root / "docs/fulfillment_flexibility_formal_protocol.md").upper(),
        "config_sha256": file_sha256(config_path).upper(),
        "runner_sha256": file_sha256(Path(__file__)).upper(),
        "eligibility_sha256": eligibility_identity(),
        "gamma": GAMMA,
        "rho_hex": float(RHO).hex(),
        "formal_seeds": list(FORMAL_SEEDS),
    }


def assert_formal_execution_gate(config_path: Path, root: Path = ROOT) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config)
    if config.get("formal_run_authorized") is not True:
        raise FormalOptimizationProhibited(
            "FORMAL OPTIMIZATION PROHIBITED UNTIL PROTOCOL REVIEW AND AUTHORIZATION"
        )
    authorization_path = root / str(config["authorization_file"])
    if not authorization_path.exists():
        raise FormalOptimizationProhibited("reviewed formal authorization file is absent")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    expected = protocol_identity(config_path, root)
    if authorization.get("formal_optimization_authorized") is not True:
        raise FormalOptimizationProhibited("formal authorization is false")
    if authorization.get("identity") != expected:
        raise FormalOptimizationProhibited("formal authorization identity drifted")
    return expected


def dry_run(config_paths: list[Path], root: Path = ROOT) -> dict[str, Any]:
    configs = [load_config(path) for path in config_paths]
    for config in configs:
        validate_config(config)
    audit = seed_nonreuse_audit(root)
    if not audit["formal_seeds_untouched"]:
        raise FormalProtocolError("formal seed collision")
    tasks = task_matrix(configs)
    gate_failures = []
    for path in config_paths:
        try:
            assert_formal_execution_gate(path, root)
        except FormalOptimizationProhibited as exc:
            gate_failures.append({"config": path.name, "reason": str(exc)})
    if len(gate_failures) != len(config_paths):
        raise FormalProtocolError("dry-run expected every formal execution gate to remain closed")
    return {
        "schema": "fulfillment_flexibility_formal_dry_run_v1",
        "source_commit": _git_commit(root),
        "formal_instances": [
            {"scale": config["scale"], "seed": seed}
            for config in configs for seed in config["seeds"]
        ],
        "task_count": len(tasks),
        "tasks_by_scale": {
            config["scale"]: sum(task["scale"] == config["scale"] for task in tasks)
            for config in configs
        },
        "gamma": GAMMA,
        "rho": RHO,
        "modes": list(MODES),
        "seed_audit_pass": True,
        "formal_output_exists": (root / RESULT_ROOT).exists(),
        "formal_solver_imported_or_called": False,
        "gate_failures": gate_failures,
        "status": "pass_protocol_only_no_formal_optimization",
    }


def _solver_settings(config: dict[str, Any]) -> dict[str, Any]:
    solver = config["solver"]
    return {
        "time_limit": float(solver["time_limit_seconds"]),
        "mip_gap": float(solver["mip_gap"]),
        "feasibility_tolerance": float(solver["feasibility_tolerance"]),
        "threads": int(solver["threads"]),
        "solver_seed": int(solver["seed"]),
    }


def _first_stage(instance: InventoryInstance, y: Any, x: Any, gp: Any) -> Any:
    return gp.quicksum(instance.fixed_cost[i] * y[i] for i in instance.I) + gp.quicksum(
        instance.inventory_cost[i][j] * x[i, j] for i in instance.I for j in instance.J
    )


def _base_model(
    instance: InventoryInstance,
    *,
    fixed_y: list[float] | None,
    fixed_x: list[list[float]] | None,
    settings: dict[str, Any],
) -> tuple[Any, Any, Any, Any, Any]:
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("formal_fulfillment_flexibility")
    model.Params.OutputFlag = 0
    model.Params.Threads = settings["threads"]
    model.Params.Seed = settings["solver_seed"]
    model.Params.MIPGap = settings["mip_gap"]
    model.Params.FeasibilityTol = settings["feasibility_tolerance"]
    model.Params.TimeLimit = settings["time_limit"]
    if fixed_y is None:
        y = model.addVars(instance.I, vtype=GRB.BINARY, name="y")
        x = model.addVars(instance.I, instance.J, lb=0.0, name="x")
        for i in instance.I:
            model.addConstr(
                gp.quicksum(instance.volume[j] * x[i, j] for j in instance.J)
                <= instance.capacity[i] * y[i]
            )
            for j in instance.J:
                model.addConstr(x[i, j] <= instance.inventory_ub[i][j] * y[i])
        first = _first_stage(instance, y, x, gp)
        model.addConstr(first <= instance.budget)
    else:
        if fixed_x is None:
            raise ValueError("fixed_x is required with fixed_y")
        y = {i: float(fixed_y[i]) for i in instance.I}
        x = {(i, j): float(fixed_x[i][j]) for i in instance.I for j in instance.J}
        first = _first_stage(instance, y, x, gp)
    return model, y, x, first, gp


def _add_recourse(
    model: Any,
    instance: InventoryInstance,
    scenario: Any,
    index: int,
    x: Any,
    eligibility: dict[int, tuple[int, ...]],
    gp: Any,
) -> dict[str, Any]:
    q_indices = [
        (i, r, j)
        for i in instance.I
        for r in instance.R
        if i in eligibility[r]
        for j in instance.J
    ]
    q = model.addVars(q_indices, lb=0.0, name=f"q[{index}]")
    u = model.addVars(instance.R, instance.J, lb=0.0, name=f"u[{index}]")
    e = model.addVars(instance.J, lb=0.0, name=f"e[{index}]")
    for r in instance.R:
        for j in instance.J:
            model.addConstr(
                gp.quicksum(q[i, r, j] for i in eligibility[r]) + u[r, j]
                >= scenario.demand[r][j]
            )
    for i in instance.I:
        for j in instance.J:
            model.addConstr(
                gp.quicksum(q[i, r, j] for r in instance.R if i in eligibility[r])
                <= x[i, j]
            )
    for j in instance.J:
        model.addConstr(
            gp.quicksum(u[r, j] for r in instance.R) - e[j]
            <= (1.0 - instance.service_level[j])
            * math.fsum(scenario.demand[r][j] for r in instance.R)
        )
    transport = gp.quicksum(
        instance.transport_cost[i][r][j] * q[i, r, j]
        for i, r, j in q_indices
    )
    shortage = gp.quicksum(
        instance.shortage_penalty[r][j] * u[r, j]
        for r in instance.R for j in instance.J
    )
    violation = gp.quicksum(instance.service_penalty[j] * e[j] for j in instance.J)
    return {
        "q": q,
        "u": u,
        "e": e,
        "transport": transport,
        "shortage": shortage,
        "violation": violation,
        "cost": transport + shortage + violation,
    }


def _values(instance: InventoryInstance, y: Any, x: Any) -> tuple[list[float], list[list[float]]]:
    y_values = [float(y[i].X if hasattr(y[i], "X") else y[i]) for i in instance.I]
    x_values = [
        [float(x[i, j].X if hasattr(x[i, j], "X") else x[i, j]) for j in instance.J]
        for i in instance.I
    ]
    return y_values, x_values


def solve_cost_anchor(
    instance: InventoryInstance,
    scenarios: list[Any],
    mode: Mode,
    settings: dict[str, Any],
    *,
    fixed_y: list[float] | None = None,
    fixed_x: list[list[float]] | None = None,
) -> dict[str, Any]:
    from gurobipy import GRB
    from .status import gurobi_status_name

    start = time.perf_counter()
    eligibility = build_eligibility(instance, mode)
    model, y, x, first, gp = _base_model(
        instance, fixed_y=fixed_y, fixed_x=fixed_x, settings=settings
    )
    theta = model.addVar(lb=0.0, name="theta")
    for index, scenario in enumerate(scenarios):
        block = _add_recourse(model, instance, scenario, index, x, eligibility, gp)
        model.addConstr(theta >= block["cost"])
    model.setObjective(first + theta, GRB.MINIMIZE)
    model.optimize()
    result: dict[str, Any] = {
        "status": gurobi_status_name(model.Status),
        "certified": model.Status == GRB.OPTIMAL,
        "runtime": time.perf_counter() - start,
        "objective": None,
        "bound": None,
        "gap": None,
        "y_values": None,
        "x_values": None,
        "scenario_count": len(scenarios),
    }
    if model.SolCount:
        y_values, x_values = _values(instance, y, x)
        result.update({
            "objective": float(model.ObjVal),
            "bound": float(model.ObjBound),
            "gap": abs(float(model.ObjVal) - float(model.ObjBound)),
            "y_values": y_values,
            "x_values": x_values,
        })
    model.dispose()
    return result


def _service_metrics(
    instance: InventoryInstance,
    scenarios: list[Any],
    recourse: list[dict[str, Any]],
    y_values: list[float],
    x_values: list[list[float]],
    first_stage_cost: float,
    eligibility: dict[int, tuple[int, ...]],
    objective_t: float,
) -> dict[str, Any]:
    tol = 1.0e-7
    policies: list[dict[str, Any]] = []
    region_worst = {r: {"rate": -1.0, "scenario": None, "used": 0} for r in instance.R}
    total_region_scenarios = 0
    multi_source_region_scenarios = 0
    concentration_values: list[float] = []
    coexisting_shortage_unused_count = 0
    inaccessible_inventory_diagnostic = 0.0
    def q_value(block: dict[str, Any], i: int, r: int, j: int) -> float:
        variable = block["q"].get((i, r, j))
        return 0.0 if variable is None else float(variable.X)

    for index, scenario in enumerate(scenarios):
        block = recourse[index]
        region_rates: list[float] = []
        total_shortage = 0.0
        total_demand = 0.0
        unused = {
            (i, j): max(
                0.0,
                x_values[i][j]
                - math.fsum(q_value(block, i, r, j) for r in instance.R),
            )
            for i in instance.I for j in instance.J
        }
        for r in instance.R:
            demand = math.fsum(float(scenario.demand[r][j]) for j in instance.J)
            shortage = math.fsum(float(block["u"][r, j].X) for j in instance.J)
            rate = 0.0 if demand <= tol else shortage / demand
            region_rates.append(rate)
            total_shortage += shortage
            total_demand += demand
            inbound = [
                math.fsum(q_value(block, i, r, j) for j in instance.J)
                for i in instance.I
            ]
            used = sum(value > tol for value in inbound)
            total_region_scenarios += 1
            multi_source_region_scenarios += int(used > 1)
            inbound_total = math.fsum(inbound)
            if inbound_total > tol:
                concentration_values.append(
                    math.fsum((value / inbound_total) ** 2 for value in inbound if value > tol)
                )
            if rate > region_worst[r]["rate"]:
                region_worst[r] = {"rate": rate, "scenario": scenario.name, "used": used}
            eligible = set(eligibility[r])
            for j in instance.J:
                shortage_rj = float(block["u"][r, j].X)
                if shortage_rj <= tol:
                    continue
                unused_elsewhere = math.fsum(unused[i, j] for i in instance.I)
                if unused_elsewhere > tol:
                    coexisting_shortage_unused_count += 1
                inaccessible = math.fsum(unused[i, j] for i in instance.I if i not in eligible)
                inaccessible_inventory_diagnostic += min(shortage_rj, inaccessible)
        policies.append({
            "scenario": scenario.name,
            "recourse_cost": float(block["cost"].getValue()),
            "transportation_cost": float(block["transport"].getValue()),
            "shortage_cost": float(block["shortage"].getValue()),
            "service_violation_cost": float(block["violation"].getValue()),
            "worst_region_shortage_rate": max(region_rates),
            "weighted_fill_rate": 1.0 - (total_shortage / total_demand if total_demand > tol else 0.0),
            "total_shortage": total_shortage,
        })
    cost_worst = max(policies, key=lambda row: row["recourse_cost"])
    t_recomputed = max(row["worst_region_shortage_rate"] for row in policies)
    facility = math.fsum(instance.fixed_cost[i] * y_values[i] for i in instance.I)
    inventory = math.fsum(
        instance.inventory_cost[i][j] * x_values[i][j]
        for i in instance.I for j in instance.J
    )
    physical = facility + inventory + cost_worst["transportation_cost"]
    failure = cost_worst["shortage_cost"] + cost_worst["service_violation_cost"]
    worst_region = max(region_worst, key=lambda r: region_worst[r]["rate"])
    return {
        "first_stage_cost": first_stage_cost,
        "warehouse_opening_cost": facility,
        "inventory_cost": inventory,
        "transportation_cost": cost_worst["transportation_cost"],
        "shortage_cost": cost_worst["shortage_cost"],
        "service_violation_cost": cost_worst["service_violation_cost"],
        "physical_fulfillment_expenditure": physical,
        "failure_related_expenditure": failure,
        "actual_robust_cost": first_stage_cost + cost_worst["recourse_cost"],
        "worst_region_shortage_rate": t_recomputed,
        "minimum_regional_fill_rate": 1.0 - t_recomputed,
        "minimum_weighted_fill_rate": min(row["weighted_fill_rate"] for row in policies),
        "maximum_total_shortage": max(row["total_shortage"] for row in policies),
        "number_opened_warehouses": sum(value >= 0.5 for value in y_values),
        "total_inventory": math.fsum(math.fsum(row) for row in x_values),
        "multi_source_region_scenario_share": (
            multi_source_region_scenarios / total_region_scenarios if total_region_scenarios else 0.0
        ),
        "mean_shipment_concentration_hhi": (
            math.fsum(concentration_values) / len(concentration_values)
            if concentration_values else None
        ),
        "shortage_unused_inventory_coexistence_count": coexisting_shortage_unused_count,
        "inaccessible_unused_inventory_diagnostic": inaccessible_inventory_diagnostic,
        "worst_shortage_region": int(worst_region),
        "worst_shortage_region_scenario": region_worst[worst_region]["scenario"],
        "objective_t_consistency_error": abs(t_recomputed - objective_t),
        "scenario_count": len(scenarios),
        "regional": [
            {
                "region": int(r),
                "worst_shortage_rate": region_worst[r]["rate"],
                "fill_rate": 1.0 - region_worst[r]["rate"],
                "number_eligible_warehouses": len(eligibility[r]),
                "number_actually_used_warehouses_in_worst_scenario": region_worst[r]["used"],
                "worst_scenario": region_worst[r]["scenario"],
            }
            for r in instance.R
        ],
    }


def solve_service(
    instance: InventoryInstance,
    scenarios: list[Any],
    mode: Mode,
    anchor: float,
    rho: float,
    settings: dict[str, Any],
    *,
    fixed_y: list[float] | None = None,
    fixed_x: list[list[float]] | None = None,
) -> dict[str, Any]:
    from gurobipy import GRB
    from .status import gurobi_status_name

    start = time.perf_counter()
    eligibility = build_eligibility(instance, mode)
    model, y, x, first, gp = _base_model(
        instance, fixed_y=fixed_y, fixed_x=fixed_x, settings=settings
    )
    t = model.addVar(lb=0.0, ub=1.0, name="T")
    budget = (1.0 + float(rho)) * float(anchor)
    recourse: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        block = _add_recourse(model, instance, scenario, index, x, eligibility, gp)
        model.addConstr(first + block["cost"] <= budget)
        for r in instance.R:
            demand = math.fsum(float(scenario.demand[r][j]) for j in instance.J)
            model.addConstr(gp.quicksum(block["u"][r, j] for j in instance.J) <= t * demand)
        recourse.append(block)
    model.setObjective(t, GRB.MINIMIZE)
    model.optimize()
    result: dict[str, Any] = {
        "status": gurobi_status_name(model.Status),
        "certified": False,
        "runtime": time.perf_counter() - start,
        "objective_t": None,
        "incumbent": None,
        "lower_bound": None,
        "objective_bound_gap": None,
        "cost_budget": budget,
        "y_values": None,
        "x_values": None,
        "metrics": None,
        "scenario_count": len(scenarios),
        "scenario_complete": True,
    }
    if model.SolCount:
        objective = float(t.X)
        y_values, x_values = _values(instance, y, x)
        first_value = math.fsum(instance.fixed_cost[i] * y_values[i] for i in instance.I) + math.fsum(
            instance.inventory_cost[i][j] * x_values[i][j]
            for i in instance.I for j in instance.J
        )
        metrics = _service_metrics(
            instance, scenarios, recourse, y_values, x_values, first_value, eligibility, objective
        )
        bound = float(model.ObjBound)
        cap_tol = 1.0e-6 + 1.0e-6 * max(1.0, abs(budget))
        certified = (
            model.Status == GRB.OPTIMAL
            and abs(objective - bound) <= 1.0e-4
            and metrics["objective_t_consistency_error"] <= 1.0e-7
            and metrics["actual_robust_cost"] <= budget + cap_tol
        )
        result.update({
            "certified": certified,
            "objective_t": objective,
            "incumbent": objective,
            "lower_bound": bound,
            "objective_bound_gap": abs(objective - bound),
            "y_values": y_values,
            "x_values": x_values,
            "metrics": metrics,
            "cost_cap_residual": metrics["actual_robust_cost"] - budget,
        })
    model.dispose()
    return result


def _first_stage_feasibility(
    instance: InventoryInstance,
    y_values: list[float],
    x_values: list[list[float]],
    tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    maximum_violation = 0.0
    for i in instance.I:
        capacity_lhs = math.fsum(instance.volume[j] * x_values[i][j] for j in instance.J)
        maximum_violation = max(
            maximum_violation,
            capacity_lhs - instance.capacity[i] * y_values[i],
        )
        for j in instance.J:
            maximum_violation = max(
                maximum_violation,
                x_values[i][j] - instance.inventory_ub[i][j] * y_values[i],
                -x_values[i][j],
            )
    first = math.fsum(instance.fixed_cost[i] * y_values[i] for i in instance.I) + math.fsum(
        instance.inventory_cost[i][j] * x_values[i][j]
        for i in instance.I for j in instance.J
    )
    maximum_violation = max(maximum_violation, first - instance.budget)
    return {
        "feasible": maximum_violation <= tolerance,
        "maximum_violation": max(0.0, maximum_violation),
        "first_stage_cost": first,
    }


def full_mode_regression() -> dict[str, Any]:
    """Compare full mode with existing unrestricted oracles on old tiny seeds."""
    from .monolithic import solve_monolithic
    from .robust_regional_fairness import solve_fairness_extensive_form

    cases: list[dict[str, Any]] = []
    settings = {
        "time_limit": 120.0,
        "mip_gap": 0.0,
        "feasibility_tolerance": 1.0e-7,
        "threads": 1,
        "solver_seed": 0,
    }
    for seed in (0, 1):
        instance = generate_instance(
            {
                "seed": seed,
                "instance": {
                    **INSTANCE_SIZES["very_small"],
                    "budget_factor": 0.68,
                    "capacity_factor": 1.25,
                },
                "robust": {"gamma_target": 1},
            },
            seed=seed,
        )
        scenarios = enumerate_budget_scenarios(
            instance, 1, max_scenarios=5000, exact_scenarios=True
        )
        formal_anchor = solve_cost_anchor(instance, scenarios, "full", settings)
        original_anchor = solve_monolithic(
            {
                "robust": {"gamma_target": 1, "max_scenarios": 5000},
                "benders": {"time_limit": 120.0, "output_flag": False},
            },
            instance,
        )
        if not formal_anchor["certified"] or original_anchor.status != "optimal":
            raise FormalProtocolError("full-mode cost regression did not certify")
        formal_service = solve_service(
            instance, scenarios, "full", float(formal_anchor["objective"]), RHO, settings
        )
        original_service = solve_fairness_extensive_form(
            instance,
            baseline_cost=float(formal_anchor["objective"]),
            rho=RHO,
            gamma=1,
            max_scenarios=5000,
            time_limit=120.0,
            mip_gap=0.0,
            lexicographic_cost_stage=False,
            output_flag=False,
        )
        if not formal_service["certified"] or original_service.status != "optimal":
            raise FormalProtocolError("full-mode service regression did not certify")
        original_y = original_service.y_values or []
        original_x = original_service.x_values or []
        formal_flat = [value for row in formal_service["x_values"] for value in row]
        original_flat = [value for row in original_x for value in row]
        case: dict[str, Any] = {
            "seed": seed,
            "scale": "very_small_nonformal_regression",
            "gamma": 1,
            "scenario_count": len(scenarios),
            "cost_objective_difference": abs(
                float(formal_anchor["objective"]) - float(original_anchor.objective)
            ),
            "T_difference": abs(
                float(formal_service["objective_t"]) - float(original_service.objective_t)
            ),
            "formal_robust_cost": formal_service["metrics"]["actual_robust_cost"],
            "original_robust_cost": original_service.actual_robust_cost,
            "robust_cost_difference": abs(
                float(formal_service["metrics"]["actual_robust_cost"])
                - float(original_service.actual_robust_cost)
            ),
            "warehouse_opening_hamming": sum(
                abs(a - b) > 1.0e-7
                for a, b in zip(formal_service["y_values"], original_y)
            ),
            "inventory_l1_difference": math.fsum(
                abs(a - b) for a, b in zip(formal_flat, original_flat)
            ),
            "formal_first_stage_feasibility": _first_stage_feasibility(
                instance, formal_service["y_values"], formal_service["x_values"]
            ),
            "original_first_stage_feasibility": _first_stage_feasibility(
                instance, original_y, original_x
            ),
            "formal_cost_cap_feasible": (
                formal_service["metrics"]["actual_robust_cost"]
                <= formal_service["cost_budget"]
                + 1.0e-6
                + 1.0e-6 * max(1.0, abs(formal_service["cost_budget"]))
            ),
            "original_cost_cap_feasible": (
                float(original_service.actual_robust_cost)
                <= float(original_service.cost_budget)
                + 1.0e-6
                + 1.0e-6 * max(1.0, abs(float(original_service.cost_budget)))
            ),
        }
        case["pass"] = (
            case["cost_objective_difference"] <= 1.0e-7
            and case["T_difference"] <= 1.0e-7
            and case["formal_first_stage_feasibility"]["feasible"]
            and case["original_first_stage_feasibility"]["feasible"]
            and case["formal_cost_cap_feasible"]
            and case["original_cost_cap_feasible"]
        )
        cases.append(case)
    return {
        "schema": "fulfillment_flexibility_full_mode_regression_v1",
        "formal_seed_accessed": False,
        "cases": cases,
        "status": "pass" if all(case["pass"] for case in cases) else "fail",
        "interpretation": (
            "Objective and feasible-region equivalence are required. Opening, inventory, and "
            "robust-cost differences are recorded because T-only service optima need not be unique."
        ),
    }


def _formal_instance(config: dict[str, Any], seed: int) -> InventoryInstance:
    return generate_instance(
        {
            "seed": int(seed),
            "instance": dict(config["instance"]),
            "robust": {"gamma_target": int(config["gamma"])},
        },
        seed=int(seed),
    )


def execute_formal_config(config_path: Path, root: Path = ROOT) -> None:
    identity = assert_formal_execution_gate(config_path, root)
    config = load_config(config_path)
    scale = config["scale"]
    scale_output = root / RESULT_ROOT / scale
    if scale_output.exists():
        raise FileExistsError(f"refusing to overwrite existing formal output: {scale_output}")
    settings = _solver_settings(config)
    manifest = {
        "schema": "fulfillment_flexibility_formal_manifest_v1",
        "identity": identity,
        "scale": scale,
        "seeds": list(config["seeds"]),
        "modes": list(MODES),
        "gamma": GAMMA,
        "rho": RHO,
        "model_source_sha256": {
            "runner": file_sha256(Path(__file__)).upper(),
            "instance": file_sha256(root / "src/instance.py").upper(),
            "scenarios": file_sha256(root / "src/scenarios.py").upper(),
        },
    }
    atomic_write_json(scale_output / "manifest.json", manifest)
    for seed in config["seeds"]:
        instance = _formal_instance(config, int(seed))
        scenarios = enumerate_budget_scenarios(
            instance,
            GAMMA,
            max_scenarios=int(config["max_scenarios"]),
            exact_scenarios=True,
        )
        if len(scenarios) != int(config["expected_scenario_count"]):
            raise FormalProtocolError("scenario count drifted")
        anchors = {mode: solve_cost_anchor(instance, scenarios, mode, settings) for mode in MODES}
        if not all(result["certified"] for result in anchors.values()):
            raise FormalProtocolError("uncertified re-optimized cost anchor")
        reoptimized = {
            mode: solve_service(
                instance, scenarios, mode, anchors[mode]["objective"], RHO, settings
            )
            for mode in MODES
        }
        full = reoptimized["full"]
        if not all(result["certified"] for result in reoptimized.values()):
            raise FormalProtocolError("uncertified re-optimized service result")
        fixed_anchors = {
            mode: solve_cost_anchor(
                instance,
                scenarios,
                mode,
                settings,
                fixed_y=full["y_values"],
                fixed_x=full["x_values"],
            )
            for mode in MODES
        }
        if not all(result["certified"] for result in fixed_anchors.values()):
            raise FormalProtocolError("uncertified fixed-first-stage cost anchor")
        common_anchor = max(float(result["objective"]) for result in fixed_anchors.values())
        fixed = {
            mode: solve_service(
                instance,
                scenarios,
                mode,
                common_anchor,
                RHO,
                settings,
                fixed_y=full["y_values"],
                fixed_x=full["x_values"],
            )
            for mode in MODES
        }
        payload = {
            "schema": "fulfillment_flexibility_formal_instance_v1",
            "identity": {
                **identity,
                "scale": scale,
                "seed": int(seed),
                "instance_sha256": canonical_json_sha256(instance.to_dict()),
                "first_stage_identity": canonical_json_sha256({
                    "y": full["y_values"], "x": full["x_values"]
                }),
                "common_budget_identity": canonical_json_sha256({
                    "anchor": common_anchor, "rho": RHO
                }),
            },
            "scenario_count": len(scenarios),
            "anchors": anchors,
            "reoptimized": reoptimized,
            "fixed_first_stage_anchors": fixed_anchors,
            "fixed_first_stage_common_anchor": common_anchor,
            "fixed_first_stage": fixed,
        }
        atomic_write_json(scale_output / "raw" / f"seed_{seed}.json", payload)
        if not all(result["certified"] for result in fixed.values()):
            raise FormalProtocolError("uncertified fixed-first-stage result; formal run stopped")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        required=True,
        help="Repeat for medium_large and large configurations.",
    )
    parser.add_argument(
        "--stage",
        choices=("seed-audit", "static-audit", "dry-run", "full-regression", "run"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "seed-audit":
        payload = seed_nonreuse_audit(ROOT)
    elif args.stage == "static-audit":
        payload = static_audit(args.config, ROOT)
    elif args.stage == "dry-run":
        payload = dry_run(args.config, ROOT)
    elif args.stage == "full-regression":
        payload = full_mode_regression()
    else:
        if len(args.config) != 1:
            raise FormalProtocolError("formal run accepts exactly one scale config")
        execute_formal_config(args.config[0], ROOT)
        return
    if args.output is None:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
