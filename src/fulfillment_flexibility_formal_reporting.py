from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable

from .experiment_protocol import atomic_write_csv, atomic_write_json, canonical_json_sha256, file_sha256
from .fulfillment_flexibility_formal_runner import (
    FORMAL_SEEDS,
    MODES,
    RHO,
    RESULT_ROOT,
    ROOT,
    _anchor_identity,
    protocol_identity,
    solver_environment_identity,
)


class FormalReportingError(RuntimeError):
    pass


def _finite(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FormalReportingError(f"{label} is not numeric") from exc
    if not math.isfinite(parsed):
        raise FormalReportingError(f"{label} is not finite")
    return parsed


def _validate_manifest_family(manifests: dict[str, dict[str, Any]]) -> None:
    if set(manifests) != {"medium_large", "large"}:
        raise FormalReportingError("both scale manifests are required")
    shared_keys = (
        "source_commit", "protocol_sha256", "runner_sha256", "reporting_sha256",
        "eligibility_sha256", "solver_parameter_sha256", "instance_generator_sha256",
        "scenario_generator_sha256", "strict_solver_environment_sha256",
        "gurobi_major_minor", "gamma", "rho_hex", "formal_seeds",
    )
    reference = manifests["medium_large"]["identity"]
    for scale, manifest in manifests.items():
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            raise FormalReportingError(f"missing manifest identity: {scale}")
        for key in shared_keys:
            if identity.get(key) != reference.get(key):
                raise FormalReportingError(f"mixed formal manifest identity: {key}")
    for scale, manifest in manifests.items():
        identity = manifest["identity"]
        if manifest.get("scale") != scale or manifest.get("seeds") != list(FORMAL_SEEDS):
            raise FormalReportingError("manifest scale or seed set drifted")
        config_path = ROOT / f"experiments/configs/fulfillment_flexibility_formal_{scale}.yaml"
        expected_identity = protocol_identity(config_path, ROOT)
        for name, value in expected_identity.items():
            if identity.get(name) != value:
                raise FormalReportingError(f"formal current identity drifted: {name}")
        expected_model_sources = {
            "runner": file_sha256(ROOT / "src/fulfillment_flexibility_formal_runner.py").upper(),
            "instance": file_sha256(ROOT / "src/instance.py").upper(),
            "scenarios": file_sha256(ROOT / "src/scenarios.py").upper(),
        }
        if manifest.get("model_source_sha256") != expected_model_sources:
            raise FormalReportingError("formal model-source identity drifted")
        solver_environment = manifest.get("solver_environment")
        if not isinstance(solver_environment, dict):
            raise FormalReportingError("formal solver environment metadata missing")
        if solver_environment.get("strict") != solver_environment_identity()["strict"]:
            raise FormalReportingError("formal strict solver environment drifted")


def _result_sha256(result: dict[str, Any]) -> str:
    return canonical_json_sha256(result)


def _require_bound_result(
    raw_result: Any,
    checkpoint: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    checkpoint_result = checkpoint.get("result")
    if not isinstance(checkpoint_result, dict):
        raise FormalReportingError(f"checkpoint result missing: {label}")
    if checkpoint_result.get("certified") is not True or checkpoint_result.get("status") != "optimal":
        raise FormalReportingError(f"checkpoint result is not certified optimal: {label}")
    checkpoint_sha256 = checkpoint.get("result_sha256")
    if checkpoint_sha256 != _result_sha256(checkpoint_result):
        raise FormalReportingError(f"checkpoint result hash mismatch: {label}")
    if not isinstance(raw_result, dict) or _result_sha256(raw_result) != checkpoint_sha256:
        raise FormalReportingError(f"raw/checkpoint result mismatch: {label}")
    return checkpoint_result


def _validate_anchor_certificate(result: dict[str, Any], label: str) -> None:
    objective = _finite(result.get("objective"), f"{label} objective")
    bound = _finite(result.get("bound"), f"{label} bound")
    reported_gap = _finite(result.get("gap"), f"{label} gap")
    recomputed_gap = abs(objective - bound)
    if abs(reported_gap - recomputed_gap) > 1.0e-12:
        raise FormalReportingError(f"anchor gap recomputation mismatch: {label}")
    if recomputed_gap > 1.0e-4 or _finite(
        result.get("formal_anchor_mip_gap"), f"{label} MIPGap"
    ) != 0.0:
        raise FormalReportingError(f"anchor certification rule violated: {label}")


def _validate_service_certificate(result: dict[str, Any], label: str) -> None:
    incumbent = _finite(result.get("incumbent"), f"{label} incumbent")
    lower_bound = _finite(result.get("lower_bound"), f"{label} lower bound")
    reported_gap = _finite(result.get("objective_bound_gap"), f"{label} gap")
    recomputed_gap = abs(incumbent - lower_bound)
    if abs(reported_gap - recomputed_gap) > 1.0e-12 or recomputed_gap > 1.0e-4:
        raise FormalReportingError(f"service certification rule violated: {label}")


def _validate_task_source_identity(
    task_identity: dict[str, Any],
    manifest_identity: dict[str, Any],
    instance_sha256: str,
    scenario_sha256: str,
) -> None:
    if task_identity.get("scenario_sha256") != scenario_sha256:
        raise FormalReportingError("task scenario identity mismatch")
    if task_identity.get("instance_sha256") != instance_sha256:
        raise FormalReportingError("task instance identity mismatch")
    for source_name in (
        "solver_parameter_sha256", "instance_generator_sha256",
        "scenario_generator_sha256", "eligibility_sha256",
        "strict_solver_environment_sha256",
    ):
        if task_identity.get(source_name) != manifest_identity.get(source_name):
            raise FormalReportingError(f"task source identity mismatch: {source_name}")


def _validate_common_budget_chain(
    payload: dict[str, Any],
    checkpoint_identities: dict[str, dict[str, Any]],
    checkpoints: dict[str, dict[str, Any]],
    common_budget_sha256: str,
) -> tuple[float, float]:
    fixed_anchor_values = [
        _finite(
            checkpoints[f"fixed_first_stage_cost_anchor_{mode}"]["result"].get("objective"),
            f"fixed anchor {mode}",
        )
        for mode in MODES
    ]
    recomputed_common_anchor = max(fixed_anchor_values)
    reported_common_anchor = _finite(
        payload.get("fixed_first_stage_common_anchor"), "fixed common anchor"
    )
    if abs(reported_common_anchor - recomputed_common_anchor) > 1.0e-7:
        raise FormalReportingError("fixed common anchor mismatch")
    recomputed_common_budget = (1.0 + RHO) * recomputed_common_anchor
    recomputed_common_budget_sha256 = canonical_json_sha256({
        "anchor": recomputed_common_anchor,
        "rho_hex": float(RHO).hex(),
    })
    if common_budget_sha256 != recomputed_common_budget_sha256:
        raise FormalReportingError("fixed common-budget hash mismatch")
    for mode in MODES:
        fixed_service_identity = checkpoint_identities[f"fixed_first_stage_service_{mode}"]
        if fixed_service_identity.get("common_budget_sha256") != recomputed_common_budget_sha256:
            raise FormalReportingError("fixed service references a different common budget")
        service_budget = _finite(
            checkpoints[f"fixed_first_stage_service_{mode}"]["result"].get("cost_budget"),
            f"fixed service budget {mode}",
        )
        if abs(service_budget - recomputed_common_budget) > 1.0e-7:
            raise FormalReportingError("fixed service common budget mismatch")
    return recomputed_common_anchor, recomputed_common_budget


def _load_records(result_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    expected = {(scale, seed) for scale in ("medium_large", "large") for seed in FORMAL_SEEDS}
    observed: set[tuple[str, int]] = set()
    manifests: dict[str, dict[str, Any]] = {}
    for scale in ("medium_large", "large"):
        manifest_path = result_root / scale / "manifest.json"
        try:
            manifests[scale] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FormalReportingError(f"missing or corrupt formal manifest: {manifest_path}") from exc
    _validate_manifest_family(manifests)
    for scale, seed in sorted(expected):
        path = result_root / scale / "raw" / f"seed_{seed}.json"
        if not path.exists():
            raise FormalReportingError(f"missing formal result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = payload.get("identity", {})
        manifest_identity = manifests[scale]["identity"]
        for name, value in manifest_identity.items():
            if identity.get(name) != value:
                raise FormalReportingError(f"raw/manifest identity mismatch: {name}")
        key = (str(identity.get("scale")), int(identity.get("seed", -1)))
        if key != (scale, seed) or key in observed:
            raise FormalReportingError("formal scale-seed identity mismatch or duplicate")
        observed.add(key)
        if int(payload.get("scenario_count", -1)) <= 0:
            raise FormalReportingError("scenario coverage is missing")
        scenario_sha256 = identity.get("scenario_sha256")
        instance_sha256 = identity.get("instance_sha256")
        first_stage_sha256 = identity.get("first_stage_sha256")
        common_budget_sha256 = identity.get("common_budget_sha256")
        if not all(isinstance(value, str) and value for value in (
            scenario_sha256, instance_sha256, first_stage_sha256, common_budget_sha256
        )):
            raise FormalReportingError(
                "instance, scenario, first-stage, or common-budget identity missing"
            )
        checkpoint_identities = payload.get("task_checkpoint_identities")
        expected_task_names = {
            f"{phase}_{mode}"
            for phase in (
                "reoptimized_cost_anchor", "reoptimized_service",
                "fixed_first_stage_cost_anchor", "fixed_first_stage_service",
            )
            for mode in MODES
        }
        if not isinstance(checkpoint_identities, dict) or set(checkpoint_identities) != expected_task_names:
            raise FormalReportingError("task checkpoint identity matrix is incomplete")
        checkpoint_root = result_root / scale / "checkpoints" / f"seed_{seed}"
        checkpoints: dict[str, dict[str, Any]] = {}
        for task_name, task_identity in checkpoint_identities.items():
            checkpoint_path = checkpoint_root / f"{task_name}.json"
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise FormalReportingError(f"missing or corrupt task checkpoint: {checkpoint_path}") from exc
            if checkpoint.get("identity") != task_identity:
                raise FormalReportingError("task checkpoint identity mismatch")
            checkpoints[task_name] = checkpoint
            _validate_task_source_identity(
                task_identity, manifest_identity, instance_sha256, scenario_sha256
            )
        bound_reoptimized_services: dict[str, dict[str, Any]] = {}
        for mode in MODES:
            reoptimized_anchor = _require_bound_result(
                payload["anchors"][mode],
                checkpoints[f"reoptimized_cost_anchor_{mode}"],
                f"reoptimized_cost_anchor_{mode}",
            )
            _validate_anchor_certificate(
                reoptimized_anchor, f"reoptimized_cost_anchor_{mode}"
            )
            expected_anchor = _anchor_identity(reoptimized_anchor)
            if checkpoint_identities[f"reoptimized_service_{mode}"].get(
                "anchor_sha256"
            ) != expected_anchor:
                raise FormalReportingError("re-optimized anchor identity mismatch")
            reoptimized_service = _require_bound_result(
                payload["reoptimized"][mode],
                checkpoints[f"reoptimized_service_{mode}"],
                f"reoptimized_service_{mode}",
            )
            _validate_service_certificate(
                reoptimized_service, f"reoptimized_service_{mode}"
            )
            bound_reoptimized_services[mode] = reoptimized_service
            expected_mode_budget = (1.0 + RHO) * _finite(
                reoptimized_anchor.get("objective"), "re-optimized anchor"
            )
            if abs(_finite(reoptimized_service.get("cost_budget"), "re-optimized budget") - expected_mode_budget) > 1.0e-7:
                raise FormalReportingError("re-optimized service budget mismatch")
            fixed_anchor_identity = checkpoint_identities[
                f"fixed_first_stage_cost_anchor_{mode}"
            ]
            if fixed_anchor_identity.get("first_stage_sha256") != first_stage_sha256:
                raise FormalReportingError("fixed first-stage identity mismatch")
            fixed_service_identity = checkpoint_identities[f"fixed_first_stage_service_{mode}"]
            if fixed_service_identity.get("first_stage_sha256") != first_stage_sha256:
                raise FormalReportingError("fixed service first-stage identity mismatch")
            if fixed_service_identity.get("common_budget_sha256") != common_budget_sha256:
                raise FormalReportingError("fixed common-budget identity mismatch")
            fixed_anchor = _require_bound_result(
                payload["fixed_first_stage_anchors"][mode],
                checkpoints[f"fixed_first_stage_cost_anchor_{mode}"],
                f"fixed_first_stage_cost_anchor_{mode}",
            )
            _validate_anchor_certificate(
                fixed_anchor, f"fixed_first_stage_cost_anchor_{mode}"
            )
            fixed_service = _require_bound_result(
                payload["fixed_first_stage"][mode],
                checkpoints[f"fixed_first_stage_service_{mode}"],
                f"fixed_first_stage_service_{mode}",
            )
            _validate_service_certificate(
                fixed_service, f"fixed_first_stage_service_{mode}"
            )
        full_result = bound_reoptimized_services["full"]
        recomputed_first_stage_sha256 = canonical_json_sha256({
            "y": full_result.get("y_values"),
            "x": full_result.get("x_values"),
        })
        if first_stage_sha256 != recomputed_first_stage_sha256:
            raise FormalReportingError("full first-stage identity mismatch")
        for mode in MODES:
            for task_name in (
                f"fixed_first_stage_cost_anchor_{mode}",
                f"fixed_first_stage_service_{mode}",
            ):
                result = checkpoints[task_name]["result"]
                result_first_stage_sha256 = canonical_json_sha256({
                    "y": result.get("y_values"), "x": result.get("x_values")
                })
                if result_first_stage_sha256 != recomputed_first_stage_sha256:
                    raise FormalReportingError("fixed task first-stage values drifted")
        _validate_common_budget_chain(
            payload, checkpoint_identities, checkpoints, common_budget_sha256
        )
        for evaluation in ("reoptimized", "fixed_first_stage"):
            section = payload.get(evaluation, {})
            if set(section) != set(MODES):
                raise FormalReportingError(f"incomplete modes for {key} {evaluation}")
            for mode in MODES:
                result = section[mode]
                if result.get("certified") is not True or result.get("status") != "optimal":
                    raise FormalReportingError(f"uncertified main result: {key} {evaluation} {mode}")
                metrics = result.get("metrics")
                if not isinstance(metrics, dict):
                    raise FormalReportingError("formal metrics missing")
                if int(result.get("scenario_count", -1)) != int(payload["scenario_count"]):
                    raise FormalReportingError("incomplete scenario coverage")
                if result.get("scenario_complete") is not True:
                    raise FormalReportingError("scenario completeness flag is false")
                if _finite(metrics.get("objective_t_consistency_error"), "T error") > 1.0e-7:
                    raise FormalReportingError("objective/recomputed T inconsistency")
                budget = _finite(result.get("cost_budget"), "cost budget")
                cost = _finite(metrics.get("actual_robust_cost"), "robust cost")
                if cost > budget + 1.0e-6 + 1.0e-6 * max(1.0, abs(budget)):
                    raise FormalReportingError("cost-cap violation")
        records.append(payload)
    if observed != expected:
        raise FormalReportingError("formal instance matrix is incomplete")
    return records


def exact_two_sided_sign_test(effects: Iterable[float], tolerance: float = 1.0e-12) -> dict[str, Any]:
    values = [float(value) for value in effects]
    wins = sum(value > tolerance for value in values)
    losses = sum(value < -tolerance for value in values)
    ties = len(values) - wins - losses
    n = wins + losses
    if n == 0:
        p_value = 1.0
    else:
        tail = min(wins, losses)
        probability = sum(math.comb(n, index) for index in range(tail + 1)) / (2 ** n)
        p_value = min(1.0, 2.0 * probability)
    return {"wins": wins, "losses": losses, "ties": ties, "non_ties": n, "p_value": p_value}


def bootstrap_mean_ci(
    effects: list[float], *, resamples: int = 10000, seed: int = 20260901
) -> tuple[float, float]:
    if not effects:
        raise FormalReportingError("bootstrap requires at least one paired effect")
    rng = random.Random(int(seed))
    n = len(effects)
    draws = sorted(
        statistics.fmean(effects[rng.randrange(n)] for _ in range(n))
        for _ in range(int(resamples))
    )
    lower = draws[max(0, math.floor(0.025 * (len(draws) - 1)))]
    upper = draws[min(len(draws) - 1, math.ceil(0.975 * (len(draws) - 1)))]
    return float(lower), float(upper)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def _relative(delta: float, denominator: float) -> float | None:
    return None if abs(denominator) <= 1.0e-12 else delta / denominator


def _inventory_flat(result: dict[str, Any]) -> list[float]:
    return [float(value) for row in result["x_values"] for value in row]


def _hamming(left: list[float], right: list[float]) -> int:
    return sum(abs(float(a) - float(b)) > 1.0e-7 for a, b in zip(left, right))


def _normalized_l1(left: list[float], right: list[float]) -> float:
    denominator = math.fsum(abs(value) for value in left)
    distance = math.fsum(abs(a - b) for a, b in zip(left, right))
    return distance if denominator <= 1.0e-12 else distance / denominator


def _inventory_concentration(values: list[float]) -> float | None:
    total = math.fsum(values)
    return None if total <= 1.0e-12 else math.fsum((value / total) ** 2 for value in values)


def build_tables(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    marginal: list[dict[str, Any]] = []
    cost: list[dict[str, Any]] = []
    spatial: list[dict[str, Any]] = []
    first_stage: list[dict[str, Any]] = []
    active_arcs: list[dict[str, Any]] = []
    warehouse_product: list[dict[str, Any]] = []
    region_product: list[dict[str, Any]] = []
    regional: list[dict[str, Any]] = []
    for payload in records:
        scale = payload["identity"]["scale"]
        seed = int(payload["identity"]["seed"])
        for evaluation in ("reoptimized", "fixed_first_stage"):
            by_mode = payload[evaluation]
            worst_regions = {mode: by_mode[mode]["metrics"]["worst_shortage_region"] for mode in MODES}
            for mode in MODES:
                result = by_mode[mode]
                metrics = result["metrics"]
                summary.append({
                    "scale": scale,
                    "seed": seed,
                    "evaluation": evaluation,
                    "mode": mode,
                    "T": result["objective_t"],
                    "lower_bound": result["lower_bound"],
                    "objective_bound_gap": result["objective_bound_gap"],
                    "recomputed_T_error": metrics["objective_t_consistency_error"],
                    "robust_total_cost": metrics["actual_robust_cost"],
                    "cost_budget": result["cost_budget"],
                    "runtime": result["runtime"],
                    "certified": result["certified"],
                    "interpretation": "confirmatory_T_based",
                })
                cost.append({
                    "scale": scale,
                    "seed": seed,
                    "evaluation": evaluation,
                    "mode": mode,
                    "opening": metrics["warehouse_opening_cost"],
                    "inventory": metrics["inventory_cost"],
                    "transportation": metrics["transportation_cost"],
                    "physical_fulfillment": metrics["physical_fulfillment_expenditure"],
                    "shortage": metrics["shortage_cost"],
                    "service_violation": metrics["service_violation_cost"],
                    "failure_related": metrics["failure_related_expenditure"],
                    "robust_total": metrics["actual_robust_cost"],
                    "interpretation": "exploratory_nonunique_T_optimum",
                })
                spatial.append({
                    "scale": scale,
                    "seed": seed,
                    "evaluation": evaluation,
                    "mode": mode,
                    "eligible_warehouses_per_region": (
                        len(result["y_values"]) if mode == "full" else (1 if mode == "k1" else 2)
                    ),
                    "multi_source_region_scenario_share": metrics["multi_source_region_scenario_share"],
                    "mean_actually_used_warehouses_per_region_scenario": metrics[
                        "mean_actually_used_warehouses_per_region_scenario"
                    ],
                    "maximum_actually_used_warehouses_per_region_scenario": metrics[
                        "maximum_actually_used_warehouses_per_region_scenario"
                    ],
                    "active_warehouse_region_arc_count": metrics[
                        "active_warehouse_region_arc_count"
                    ],
                    "shipment_concentration_hhi": metrics["mean_shipment_concentration_hhi"],
                    "shortage_unused_inventory_coexistence_count": metrics[
                        "shortage_unused_inventory_coexistence_count"
                    ],
                    "inaccessible_unused_inventory_diagnostic": metrics[
                        "inaccessible_unused_inventory_diagnostic"
                    ],
                    "worst_shortage_region": metrics["worst_shortage_region"],
                    "worst_region_changed_from_k1": worst_regions[mode] != worst_regions["k1"],
                    "interpretation": "exploratory_nonunique_T_optimum",
                })
                for row in metrics["active_warehouse_region_arcs"]:
                    active_arcs.append({
                        "scale": scale, "seed": seed, "evaluation": evaluation, "mode": mode,
                        **row, "active_in_at_least_one_scenario": True,
                        "interpretation": "exploratory_nonunique_T_optimum",
                    })
                for row in metrics["maximum_unused_inventory_by_warehouse_product"]:
                    warehouse_product.append({
                        "scale": scale, "seed": seed, "evaluation": evaluation, "mode": mode,
                        **row, "interpretation": "exploratory_nonunique_T_optimum",
                    })
                for row in metrics["maximum_shortage_by_region_product"]:
                    region_product.append({
                        "scale": scale, "seed": seed, "evaluation": evaluation, "mode": mode,
                        **row, "interpretation": "exploratory_nonunique_T_optimum",
                    })
                for row in metrics["regional"]:
                    regional.append({
                        "scale": scale, "seed": seed, "evaluation": evaluation, "mode": mode,
                        **row, "interpretation": "exploratory_nonunique_T_optimum",
                    })
            t1 = _finite(by_mode["k1"]["objective_t"], "T_k1")
            t2 = _finite(by_mode["k2"]["objective_t"], "T_k2")
            tf = _finite(by_mode["full"]["objective_t"], "T_full")
            d12, d2f, d1f = t1 - t2, t2 - tf, t1 - tf
            marginal.append({
                "scale": scale,
                "seed": seed,
                "evaluation": evaluation,
                "delta_12": d12,
                "delta_2F": d2f,
                "delta_1F": d1f,
                "relative_12": _relative(d12, t1),
                "relative_1F": _relative(d1f, t1),
                "capture_k2": None if abs(d1f) <= 1.0e-12 else d12 / d1f,
                "diminishing_returns": d12 > d2f,
                "interpretation": "confirmatory_T_based",
            })
            for left, right in (("k1", "k2"), ("k2", "full"), ("k1", "full")):
                left_t = _finite(by_mode[left]["objective_t"], "left T")
                right_t = _finite(by_mode[right]["objective_t"], "right T")
                left_cost = _finite(by_mode[left]["metrics"]["actual_robust_cost"], "left cost")
                right_cost = _finite(by_mode[right]["metrics"]["actual_robust_cost"], "right cost")
                paired.append({
                    "scale": scale,
                    "seed": seed,
                    "evaluation": evaluation,
                    "comparison": f"{left}_vs_{right}",
                    "absolute_T_reduction": left_t - right_t,
                    "relative_T_reduction": _relative(left_t - right_t, left_t),
                    "relative_robust_cost_change": _relative(right_cost - left_cost, left_cost),
                    "T_effect_interpretation": "confirmatory_T_based",
                    "cost_interpretation": "exploratory_nonunique_T_optimum",
                })
        optimized = payload["reoptimized"]
        for mode in MODES:
            result = optimized[mode]
            inventory = _inventory_flat(result)
            warehouses = [math.fsum(row) for row in result["x_values"]]
            first_stage.append({
                "scale": scale,
                "seed": seed,
                "mode": mode,
                "opened_warehouses": result["metrics"]["number_opened_warehouses"],
                "warehouse_opening_vector": json.dumps(result["y_values"], separators=(",", ":")),
                "total_inventory": result["metrics"]["total_inventory"],
                "inventory_vector": json.dumps(result["x_values"], separators=(",", ":")),
                "inventory_concentration_hhi": _inventory_concentration(warehouses),
                "hamming_from_k1": _hamming(optimized["k1"]["y_values"], result["y_values"]),
                "hamming_from_full": _hamming(optimized["full"]["y_values"], result["y_values"]),
                "normalized_inventory_l1_from_k1": _normalized_l1(
                    _inventory_flat(optimized["k1"]), inventory
                ),
                "normalized_inventory_l1_from_full": _normalized_l1(
                    _inventory_flat(optimized["full"]), inventory
                ),
                "interpretation": "exploratory_nonunique_T_optimum",
            })
    return {
        "summary": summary,
        "paired_comparisons": paired,
        "fixed_first_stage": [row for row in summary if row["evaluation"] == "fixed_first_stage"],
        "marginal_flexibility": marginal,
        "cost_decomposition": cost,
        "spatial_mismatch_diagnostic": spatial,
        "first_stage_diagnostic": first_stage,
        "active_arc_diagnostic": active_arcs,
        "warehouse_product_diagnostic": warehouse_product,
        "region_product_diagnostic": region_product,
        "regional_diagnostic": regional,
    }


def _comparison_statistics(rows: list[dict[str, Any]], comparison: str) -> dict[str, Any]:
    selected = [row for row in rows if row["comparison"] == comparison]
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in selected:
        seed = int(row["seed"])
        scale = str(row["scale"])
        if scale in by_seed.setdefault(seed, {}):
            raise FormalReportingError("duplicate scale within a seed cluster")
        by_seed[seed][scale] = row
    if set(by_seed) != set(FORMAL_SEEDS) or any(
        set(cluster) != {"medium_large", "large"} for cluster in by_seed.values()
    ):
        raise FormalReportingError("pooled seed-cluster matrix is incomplete")
    cluster_effects = [
        statistics.fmean(
            _finite(by_seed[seed][scale]["absolute_T_reduction"], "cluster effect")
            for scale in ("medium_large", "large")
        )
        for seed in FORMAL_SEEDS
    ]
    cluster_relative = [
        statistics.fmean(
            _finite(by_seed[seed][scale]["relative_T_reduction"], "cluster relative effect")
            for scale in ("medium_large", "large")
        )
        for seed in FORMAL_SEEDS
        if all(by_seed[seed][scale]["relative_T_reduction"] is not None for scale in (
            "medium_large", "large"
        ))
    ]
    sign = exact_two_sided_sign_test(cluster_effects)
    lower, upper = bootstrap_mean_ci(cluster_effects)
    scale_specific: dict[str, Any] = {}
    for scale in ("medium_large", "large"):
        effects = [
            _finite(by_seed[seed][scale]["absolute_T_reduction"], "scale effect")
            for seed in FORMAL_SEEDS
        ]
        relative = [
            _finite(by_seed[seed][scale]["relative_T_reduction"], "scale relative effect")
            for seed in FORMAL_SEEDS
            if by_seed[seed][scale]["relative_T_reduction"] is not None
        ]
        scale_lower, scale_upper = bootstrap_mean_ci(effects)
        scale_specific[scale] = {
            **exact_two_sided_sign_test(effects),
            "mean": statistics.fmean(effects),
            "median": statistics.median(effects),
            "mean_relative": statistics.fmean(relative) if relative else None,
            "median_relative": statistics.median(relative) if relative else None,
            "bootstrap_mean_ci_95": [scale_lower, scale_upper],
            "independent_seed_count": len(effects),
        }
    return {
        **sign,
        "mean_effect": statistics.fmean(cluster_effects),
        "median_effect": statistics.median(cluster_effects),
        "mean_relative_effect": statistics.fmean(cluster_relative) if cluster_relative else None,
        "median_relative_effect": statistics.median(cluster_relative) if cluster_relative else None,
        "bootstrap_mean_ci_95": [lower, upper],
        "scale_specific": scale_specific,
        "pooled_independent_unit": "synthetic_seed_cluster",
        "pooled_independent_cluster_count": len(cluster_effects),
        "pooled_cluster_effect": "mean_medium_large_and_large_effect_within_seed",
        "cluster_effects_by_seed": {
            str(seed): cluster_effects[index] for index, seed in enumerate(FORMAL_SEEDS)
        },
    }


def formal_decision(tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    fixed = [
        row for row in tables["paired_comparisons"]
        if row["evaluation"] == "fixed_first_stage"
    ]
    primary = _comparison_statistics(fixed, "k1_vs_full")
    secondary = {
        comparison: _comparison_statistics(fixed, comparison)
        for comparison in ("k1_vs_k2", "k2_vs_full")
    }
    adjusted = holm_adjust({name: value["p_value"] for name, value in secondary.items()})
    reoptimized_rows = [
        row for row in tables["paired_comparisons"] if row["evaluation"] == "reoptimized"
    ]
    reoptimized = {
        comparison: _comparison_statistics(reoptimized_rows, comparison)
        for comparison in ("k1_vs_k2", "k2_vs_full", "k1_vs_full")
    }
    reoptimized_adjusted = holm_adjust({
        name: value["p_value"]
        for name, value in reoptimized.items()
        if name in {"k1_vs_k2", "k2_vs_full"}
    })
    scale_positive = all(value["median"] > 0.0 for value in primary["scale_specific"].values())
    relative = primary["median_relative_effect"]
    lower, upper = primary["bootstrap_mean_ci_95"]
    if (
        primary["p_value"] < 0.05
        and lower > 0.0
        and scale_positive
        and primary["wins"] >= 8
        and relative is not None
        and relative >= 0.10
    ):
        classification = "confirm_strong_fulfillment_flexibility_mechanism"
    elif lower > 0.0 and scale_positive and primary["wins"] >= 6 and relative is not None and relative > 0.0:
        classification = "confirm_moderate_fulfillment_flexibility_mechanism"
    elif upper <= 0.0 or primary["median_effect"] <= 0.0:
        classification = "do_not_confirm_fulfillment_flexibility_mechanism"
    else:
        classification = "mixed_or_scale_dependent_fulfillment_flexibility"
    capture = [
        row["capture_k2"] for row in tables["marginal_flexibility"]
        if row["evaluation"] == "fixed_first_stage" and row["capture_k2"] is not None
    ]
    diminishing = [
        row for row in tables["marginal_flexibility"]
        if row["evaluation"] == "fixed_first_stage" and row["diminishing_returns"]
    ]
    return {
        "schema": "fulfillment_flexibility_formal_decision_v1",
        "classification": classification,
        "classification_rule_frozen_before_results": True,
        "primary_fixed_first_stage_k1_vs_full": primary,
        "secondary_fixed_first_stage": secondary,
        "secondary_holm_adjusted_p_values": adjusted,
        "reoptimized_system_effects": reoptimized,
        "reoptimized_secondary_holm_adjusted_p_values": reoptimized_adjusted,
        "reoptimized_interpretation": (
            "network adaptation plus inventory adaptation plus recourse flexibility"
        ),
        "fixed_first_stage_interpretation": (
            "recourse-set expansion under common y, x, scenarios, Gamma, and absolute cost allowance; "
            "conditional on the certified full-mode T-optimal first-stage configuration selected "
            "by the frozen solution procedure"
        ),
        "median_capture_k2_untruncated": statistics.median(capture) if capture else None,
        "diminishing_returns_instance_count": len(diminishing),
        "formal_scale_seed_instance_count": 20,
        "formal_independent_seed_cluster_count": 10,
        "certification_complete": True,
    }


def write_formal_reports(result_root: Path) -> dict[str, Any]:
    records = _load_records(result_root)
    tables = build_tables(records)
    mapping = {
        "summary": "summary.csv",
        "paired_comparisons": "paired_comparisons.csv",
        "fixed_first_stage": "fixed_first_stage.csv",
        "marginal_flexibility": "marginal_flexibility.csv",
        "cost_decomposition": "cost_decomposition.csv",
        "spatial_mismatch_diagnostic": "spatial_mismatch_diagnostic.csv",
        "first_stage_diagnostic": "first_stage_diagnostic.csv",
        "active_arc_diagnostic": "active_arc_diagnostic.csv",
        "warehouse_product_diagnostic": "warehouse_product_diagnostic.csv",
        "region_product_diagnostic": "region_product_diagnostic.csv",
        "regional_diagnostic": "regional_diagnostic.csv",
    }
    for key, filename in mapping.items():
        rows = tables[key]
        if not rows:
            raise FormalReportingError(f"empty formal table: {key}")
        target = result_root / filename
        if target.exists():
            raise FileExistsError(f"refusing to overwrite formal report: {target}")
        atomic_write_csv(target, rows, list(rows[0]))
    decision = formal_decision(tables)
    decision_path = result_root / "formal_decision.json"
    if decision_path.exists():
        raise FileExistsError(f"refusing to overwrite formal decision: {decision_path}")
    atomic_write_json(decision_path, decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=ROOT / RESULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(write_formal_reports(args.result_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
