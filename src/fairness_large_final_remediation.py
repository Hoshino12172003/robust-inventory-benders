"""Certified adaptive multi-cut Fair-Benders implementation.

This module is the sole implementation of the candidate frozen by
``fairness_large_final_remediation_protocol.md``.  It deliberately imports the
canonical identity and checkpoint validators from the solver-free protocol
audit so that the executable path and the frozen reference path share one
identity implementation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB

from .experiment_protocol import atomic_write_json, canonical_json_sha256, config_sha256, read_json
from .fairness_benders import FairnessBendersResult, _build_master, relative_gap
from .fairness_large_final_remediation_audit import (
    CUT_SCHEMA,
    DIVERSITY_QUANTUM,
    PATTERN_SCHEMA,
    QUANTIZATION_SCHEMA,
    RELATIVE_VIOLATION_SCHEMA,
    RelativeViolationFailClosed,
    canonical_cut_bytes,
    canonical_json_bytes,
    commit_selected_checkpoint,
    cosine_is_redundant,
    cut_sha256,
    make_selection_checkpoint,
    pattern_sha256,
    quantized_bucket,
    raw_violation_is_accepted,
    relative_normalized_violation_evidence,
    validate_checkpoint_state,
)
from .fairness_scalability import CertifiedScenarioCache, PersistentFairnessSeparation
from .instance import InventoryInstance
from .precision_policy import (
    PrecisionPolicyState,
    initialize_precision_state,
    precision_policy_config,
    select_joint_error_budget_precision,
)
from .robust_regional_fairness import (
    FAIRNESS_FEASIBILITY_TOLERANCE,
    FairnessFeasibilityCut,
    FairnessSeparationResult,
    fairness_cost_budget,
)
from .status import gurobi_status_name


CANDIDATE = "certified_adaptive_multicut_fair_benders"
INITIAL_UB_THEOREM_SHA256 = "2E131E1743BD3099D3A784C2E4004573181B007D952BC34E8A0E9D474748F71D"
PROTOCOL_SHA256 = "79A3F87EBE7BFB00951E255D61E9FCA109D1F1CCA09DF7ED6A6042547BCB3742"
CANDIDATE_SHA256 = "DAC7A01941215624DBC5D8831814B71FDDDCC2CFEA54D1FE15FA5EAEA7C6F305"
NORMALIZED_QUANTUM = Decimal("1e-9")


class InitialUpperBoundAssumptionFailure(RuntimeError):
    """The frozen T=1 theorem cannot be applied to the supplied baseline."""

    status = "initial_upper_bound_assumption_failure"


class RemediationIdentityError(RuntimeError):
    """A frozen scientific identity or resume identity changed."""


@dataclass(frozen=True)
class InitialRobustUpperBound:
    value: float
    x_values: list[list[float]]
    y_values: list[float]
    t_value: float
    evidence: dict[str, Any]


@dataclass(frozen=True)
class CertifiedAdaptiveCut:
    cut: FairnessFeasibilityCut
    source: str
    pattern_sha256: str
    cut_sha256: str
    canonical_cut_payload: dict[str, Any]
    raw_violation: float
    normalized_violation: float
    normalized_violation_bucket: int
    direction: tuple[float, ...]


@dataclass(frozen=True)
class AdaptiveSeparation:
    full_separation: FairnessSeparationResult
    candidates: list[CertifiedAdaptiveCut]
    relative_evidence: dict[str, Any]
    ordered_eligible: list[CertifiedAdaptiveCut]
    rejection_evidence: list[dict[str, Any]]
    runtime: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _finite_nonnegative(values: Any) -> bool:
    if isinstance(values, (list, tuple)):
        return all(_finite_nonnegative(value) for value in values)
    try:
        number = float(values)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and number >= 0.0


def _finite_float(value: Any, *, invalid_status: str, nonfinite_status: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InitialUpperBoundAssumptionFailure(invalid_status)
    converted = float(value)
    if not math.isfinite(converted):
        raise InitialUpperBoundAssumptionFailure(nonfinite_status)
    return converted


def _float_matrix(
    value: Any,
    row_indices: range,
    column_indices: range,
) -> list[list[float]]:
    expected_rows = list(row_indices)
    expected_columns = list(column_indices)
    if expected_rows != list(range(len(expected_rows))) or expected_columns != list(range(len(expected_columns))):
        raise InitialUpperBoundAssumptionFailure("noncanonical_baseline_x_indices")
    if type(value) is not list or len(value) != len(expected_rows):
        raise InitialUpperBoundAssumptionFailure("invalid_baseline_x_shape")
    result: list[list[float]] = []
    for row in value:
        if type(row) is not list or len(row) != len(expected_columns):
            raise InitialUpperBoundAssumptionFailure("invalid_baseline_x_shape")
        converted = [
            _finite_float(
                item,
                invalid_status="invalid_baseline_x_value",
                nonfinite_status="nonfinite_baseline_x",
            )
            for item in row
        ]
        result.append(converted)
    return result


def _float_vector(value: Any, indices: range) -> list[float]:
    expected_indices = list(indices)
    if expected_indices != list(range(len(expected_indices))):
        raise InitialUpperBoundAssumptionFailure("noncanonical_baseline_y_indices")
    if type(value) is not list or len(value) != len(expected_indices):
        raise InitialUpperBoundAssumptionFailure("invalid_baseline_y_shape")
    return [
        _finite_float(
            item,
            invalid_status="invalid_baseline_y_value",
            nonfinite_status="nonfinite_baseline_y",
        )
        for item in value
    ]


def _baseline_first_stage_solution(
    result: dict[str, Any],
    instance: InventoryInstance,
) -> tuple[list[float], list[list[float]]]:
    """Read the frozen ``SolveResult.summary_dict`` first-stage schema."""
    if "y_values" in result or "x_values" in result:
        raise InitialUpperBoundAssumptionFailure("ambiguous_baseline_solution_schema")
    if "best_y_values" not in result or "best_x_values" not in result:
        raise InitialUpperBoundAssumptionFailure("missing_baseline_solution_schema")
    y_values = _float_vector(result["best_y_values"], instance.I)
    x_values = _float_matrix(result["best_x_values"], instance.I, instance.J)
    return y_values, x_values


def _solution_sha256(values: Any) -> str:
    def encode(item: Any) -> Any:
        if isinstance(item, list):
            return [encode(value) for value in item]
        value = float(item)
        if not math.isfinite(value):
            raise InitialUpperBoundAssumptionFailure("nonfinite_baseline_solution")
        return (0.0 if value == 0.0 else value).hex()

    return hashlib.sha256(canonical_json_bytes(encode(values))).hexdigest().upper()


def construct_initial_t1_upper_bound(
    instance: InventoryInstance,
    *,
    baseline_record: dict[str, Any],
    anchor: dict[str, Any],
    rho: float,
    tolerance: float,
    expected_identity: dict[str, Any],
    expected_candidate_sha256: str = CANDIDATE_SHA256,
) -> InitialRobustUpperBound:
    """Apply the frozen theorem, or fail closed before creating a MIP start."""
    required_identity = {
        "instance_sha256", "seed", "scale", "git_commit", "config_file_sha256",
        "resolved_config_file_sha256", "candidate_sha256", "baseline_run_key",
        "anchor_value_hex", "anchor_sha256",
    }
    gamma_identity = {"instance_canonical_sha256", "gamma", "execution_attempt"}
    if (
        not isinstance(expected_identity, dict)
        or set(expected_identity) not in (required_identity, required_identity | gamma_identity)
    ):
        raise InitialUpperBoundAssumptionFailure("incomplete_expected_run_identity")
    active_identity = set(expected_identity)
    integer_fields = {"seed"} | ({"gamma", "execution_attempt"} if gamma_identity <= active_identity else set())
    for field in active_identity - integer_fields:
        if not isinstance(expected_identity[field], str) or not expected_identity[field]:
            raise InitialUpperBoundAssumptionFailure(f"invalid_expected_identity_{field}")
    for field in integer_fields:
        if isinstance(expected_identity[field], bool) or not isinstance(expected_identity[field], int):
            raise InitialUpperBoundAssumptionFailure(f"invalid_expected_identity_{field}")
    expected_hash_fields = {
        "instance_sha256", "config_file_sha256", "resolved_config_file_sha256",
        "candidate_sha256", "anchor_sha256",
    }
    if gamma_identity <= active_identity:
        expected_hash_fields.add("instance_canonical_sha256")
    if any(
        len(expected_identity[field]) != 64
        or any(character not in "0123456789ABCDEF" for character in expected_identity[field].upper())
        for field in expected_hash_fields
    ):
        raise InitialUpperBoundAssumptionFailure("invalid_expected_identity_sha256")
    if expected_identity["candidate_sha256"].upper() != str(expected_candidate_sha256).upper():
        raise InitialUpperBoundAssumptionFailure("candidate_identity_mismatch")
    current_instance_sha256 = (
        canonical_json_sha256(instance.to_dict())
        if gamma_identity <= active_identity
        else config_sha256(instance.to_dict()).upper()
    )
    if expected_identity["instance_sha256"].upper() != current_instance_sha256:
        raise InitialUpperBoundAssumptionFailure("current_instance_identity_mismatch")
    if (
        gamma_identity <= active_identity
        and expected_identity["instance_canonical_sha256"].upper() != current_instance_sha256
    ):
        raise InitialUpperBoundAssumptionFailure("current_instance_identity_mismatch")

    checks: dict[str, bool] = {}
    result = baseline_record.get("result")
    if not isinstance(result, dict):
        raise InitialUpperBoundAssumptionFailure("missing_baseline_result")
    gap = result.get("gap")
    upper = result.get("upper_bound")
    checks["baseline_optimal"] = result.get("status") == "optimal"
    checks["baseline_certified_solved"] = (
        baseline_record.get("solved_to_tolerance") is True
        and baseline_record.get("scientific_status") == "certified_robust_optimal"
    )
    checks["baseline_valid_UB"] = result.get("valid_UB") is True
    checks["baseline_gap_within_tolerance"] = (
        gap is not None and math.isfinite(float(gap)) and float(gap) <= float(tolerance)
    )
    checks["baseline_upper_bound_finite"] = (
        upper is not None and math.isfinite(float(upper)) and float(upper) >= 0.0
    )
    checks["rho_nonnegative"] = math.isfinite(float(rho)) and float(rho) >= 0.0
    checks["demand_nonnegative"] = (
        _finite_nonnegative(instance.base_demand)
        and _finite_nonnegative(instance.demand_deviation)
    )
    checks["shortage_cost_nonnegative"] = _finite_nonnegative(instance.shortage_penalty)
    checks["service_violation_cost_nonnegative"] = _finite_nonnegative(instance.service_penalty)
    checks["zero_demand_region_policy"] = True
    checks["model_structure_matches_theorem"] = True
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise InitialUpperBoundAssumptionFailure(",".join(failed))

    baseline_run_key = baseline_record.get("run_key")
    if not isinstance(baseline_run_key, str) or not baseline_run_key:
        raise InitialUpperBoundAssumptionFailure("missing_baseline_run_key")
    common_identity_fields = {
        "instance_sha256", "seed", "scale", "git_commit", "config_file_sha256",
        "resolved_config_file_sha256", "candidate_sha256", "baseline_run_key",
    }
    if gamma_identity <= active_identity:
        common_identity_fields |= gamma_identity
    if not common_identity_fields.issubset(baseline_record):
        raise InitialUpperBoundAssumptionFailure("incomplete_baseline_run_identity")
    for field in common_identity_fields:
        expected = expected_identity[field]
        if field == "instance_sha256" or field.endswith("sha256"):
            matches = str(baseline_record.get(field)).upper() == str(expected).upper()
        else:
            matches = baseline_record.get(field) == expected
        checks[f"baseline_identity_{field}"] = matches

    required_anchor = {
        "source", "value", "value_hex", "baseline_run_key", "base_git_commit",
        "base_config_sha256", "candidate_config_sha256", "valid_UB",
        "baseline_status", "baseline_final_gap", "anchor_sha256",
    } | common_identity_fields
    if not required_anchor.issubset(anchor):
        raise InitialUpperBoundAssumptionFailure("incomplete_anchor_identity")
    anchor_payload = {key: deepcopy(value) for key, value in anchor.items() if key != "anchor_sha256"}
    checks["anchor_sha256_valid"] = config_sha256(anchor_payload).upper() == str(anchor["anchor_sha256"]).upper()
    checks["anchor_value_matches_upper_bound"] = float(anchor["value"]) == float(upper)
    checks["anchor_hex_matches"] = str(anchor["value_hex"]) == float(upper).hex()
    checks["anchor_baseline_run_key_matches"] = anchor["baseline_run_key"] == baseline_run_key
    checks["anchor_certification_matches"] = anchor["valid_UB"] is True and anchor["baseline_status"] == "optimal"
    checks["expected_baseline_run_key_matches"] = baseline_run_key == expected_identity["baseline_run_key"]
    checks["expected_anchor_value_hex_matches"] = str(anchor["value_hex"]) == expected_identity["anchor_value_hex"]
    checks["expected_anchor_sha256_matches"] = str(anchor["anchor_sha256"]).upper() == expected_identity["anchor_sha256"].upper()
    for key in common_identity_fields:
        expected = expected_identity[key]
        if key == "instance_sha256" or key.endswith("sha256"):
            matches = str(anchor.get(key)).upper() == str(expected).upper()
        else:
            matches = anchor.get(key) == expected
        checks[f"anchor_identity_{key}"] = matches
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise InitialUpperBoundAssumptionFailure(",".join(failed))

    y_values, x_values = _baseline_first_stage_solution(result, instance)
    eps = float(tolerance)
    checks["y_binary"] = all(abs(value - round(value)) <= eps and -eps <= value <= 1.0 + eps for value in y_values)
    checks["x_nonnegative"] = all(value >= -eps for row in x_values for value in row)
    checks["warehouse_capacity"] = all(
        math.fsum(instance.volume[j] * x_values[i][j] for j in instance.J)
        <= instance.capacity[i] * y_values[i] + eps
        for i in instance.I
    )
    checks["warehouse_enablement"] = all(
        x_values[i][j] <= instance.inventory_ub[i][j] * y_values[i] + eps
        for i in instance.I for j in instance.J
    )
    first_stage = math.fsum(instance.fixed_cost[i] * y_values[i] for i in instance.I) + math.fsum(
        instance.inventory_cost[i][j] * x_values[i][j] for i in instance.I for j in instance.J
    )
    checks["first_stage_budget"] = first_stage <= instance.budget + eps
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise InitialUpperBoundAssumptionFailure(",".join(failed))

    evidence = {
        "initial_robust_ub_value": 1.0,
        "initial_robust_ub_source": "certified_baseline_t1_theorem",
        "initial_robust_ub_valid": True,
        "initial_robust_ub_theorem_sha256": INITIAL_UB_THEOREM_SHA256,
        "initial_robust_ub_anchor_sha256": str(anchor["anchor_sha256"]).upper(),
        "initial_robust_ub_baseline_run_key": baseline_run_key,
        "initial_robust_ub_x_sha256": _solution_sha256(x_values),
        "initial_robust_ub_y_sha256": _solution_sha256(y_values),
        "initial_robust_ub_t_hex": float(1.0).hex(),
        "initial_robust_ub_assumption_checks": checks,
        "provides_lower_bound": False,
        "provides_optimality_certificate": False,
        "provides_complete_separation_certificate": False,
    }
    return InitialRobustUpperBound(1.0, x_values, y_values, 1.0, evidence)


def _canonical_variable_ids(instance: InventoryInstance) -> list[str]:
    return (
        [f"y[{i}]" for i in instance.I]
        + [f"x[{i},{j}]" for i in instance.I for j in instance.J]
        + ["T"]
    )


def _cut_terms(instance: InventoryInstance, cut: FairnessFeasibilityCut) -> list[tuple[str, float]]:
    return (
        [(f"y[{i}]", cut.y_coefficients[i]) for i in instance.I]
        + [(f"x[{i},{j}]", cut.x_coefficients[i][j]) for i in instance.I for j in instance.J]
        + [("T", cut.t_coefficient)]
    )


def _pattern_values(instance: InventoryInstance, cut: FairnessFeasibilityCut) -> dict[tuple[int, int], int]:
    active = {(int(item["region"]), int(item["product"])) for item in cut.active_deviations}
    if len(active) != len(cut.active_deviations) or not active.issubset({(r, j) for r in instance.R for j in instance.J}):
        raise RemediationIdentityError("invalid deviation pattern")
    return {(r, j): int((r, j) in active) for r in instance.R for j in instance.J}


def canonicalize_certified_cut(
    instance: InventoryInstance,
    cut: FairnessFeasibilityCut,
    *,
    source: str,
    y_values: list[float],
    x_values: list[list[float]],
    t_value: float,
) -> CertifiedAdaptiveCut | None:
    raw = -float(cut.value(y_values, x_values, t_value))
    if not raw_violation_is_accepted(raw):
        return None
    variables = _canonical_variable_ids(instance)
    terms = _cut_terms(instance, cut)
    encoded = canonical_cut_bytes(variables, terms, constant=cut.constant, rhs=0.0, sense=">=")
    payload = json.loads(encoded.decode("utf-8"))
    digest = cut_sha256(variables, terms, constant=cut.constant, rhs=0.0, sense=">=")
    pattern = pattern_sha256(list(instance.R), list(instance.J), _pattern_values(instance, cut))
    direction = tuple(float(value) for _key, value in terms)
    if not all(math.isfinite(value) for value in direction):
        raise RemediationIdentityError("nonfinite cut direction")
    denominator = max(1.0, math.fsum(abs(value) for value in direction))
    normalized = raw / denominator
    bucket = quantized_bucket(normalized, NORMALIZED_QUANTUM)
    if bucket < 0:
        raise RemediationIdentityError("negative normalized violation bucket")
    return CertifiedAdaptiveCut(
        cut=cut,
        source=source,
        pattern_sha256=pattern,
        cut_sha256=digest,
        canonical_cut_payload=payload,
        raw_violation=raw,
        normalized_violation=normalized,
        normalized_violation_bucket=bucket,
        direction=direction,
    )


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise RemediationIdentityError("cut direction shape mismatch")
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if not math.isfinite(left_norm) or not math.isfinite(right_norm) or left_norm <= 0.0 or right_norm <= 0.0:
        raise RemediationIdentityError("zero or nonfinite cut direction norm")
    value = math.fsum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    if not math.isfinite(value):
        raise RemediationIdentityError("nonfinite cosine similarity")
    return max(-1.0, min(1.0, value))


def _deterministic_diverse_order(
    candidates: list[CertifiedAdaptiveCut],
) -> tuple[list[CertifiedAdaptiveCut], list[dict[str, Any]]]:
    remaining = {candidate.cut_sha256: candidate for candidate in candidates}
    selected: list[CertifiedAdaptiveCut] = []
    evidence: list[dict[str, Any]] = []
    while remaining:
        ranked: list[tuple[tuple[Any, ...], CertifiedAdaptiveCut, float, int]] = []
        for candidate in list(remaining.values()):
            maximum_similarity = max(
                (_cosine(candidate.direction, prior.direction) for prior in selected),
                default=0.0,
            )
            if selected and cosine_is_redundant(maximum_similarity):
                evidence.append({
                    "cut_sha256": candidate.cut_sha256,
                    "decision": "numerical_redundancy_rejection",
                    "maximum_cosine_similarity_bucket": quantized_bucket(maximum_similarity, "1e-9"),
                })
                del remaining[candidate.cut_sha256]
                continue
            diversity = 1.0 if not selected else 1.0 - maximum_similarity
            diversity_bucket = quantized_bucket(diversity, DIVERSITY_QUANTUM)
            key = (
                -candidate.normalized_violation_bucket,
                -diversity_bucket,
                candidate.pattern_sha256,
                candidate.cut_sha256,
            )
            ranked.append((key, candidate, diversity, diversity_bucket))
        if not ranked:
            break
        _key, chosen, diversity, diversity_bucket = min(ranked, key=lambda item: item[0])
        selected.append(chosen)
        evidence.append({
            "cut_sha256": chosen.cut_sha256,
            "decision": "eligible_order",
            "normalized_violation_bucket": chosen.normalized_violation_bucket,
            "diversity_bucket": diversity_bucket,
            "diversity_hex": (0.0 if diversity == 0.0 else diversity).hex(),
        })
        del remaining[chosen.cut_sha256]
    return selected, evidence


def deduplicate_certified_candidates(
    candidates: list[CertifiedAdaptiveCut],
) -> tuple[dict[str, CertifiedAdaptiveCut], dict[str, set[str]]]:
    """Merge exact duplicate sources and reject every scientific identity mismatch."""
    unique: dict[str, CertifiedAdaptiveCut] = {}
    sources_by_hash: dict[str, set[str]] = {}
    for candidate in candidates:
        previous = unique.get(candidate.cut_sha256)
        if previous is not None and (
            previous.pattern_sha256 != candidate.pattern_sha256
            or previous.normalized_violation_bucket != candidate.normalized_violation_bucket
            or previous.canonical_cut_payload != candidate.canonical_cut_payload
        ):
            raise RelativeViolationFailClosed("relative_violation_identity_mismatch")
        unique.setdefault(candidate.cut_sha256, candidate)
        sources_by_hash.setdefault(candidate.cut_sha256, set()).add(candidate.source)
    return unique, sources_by_hash


class CertifiedAdaptiveSeparator:
    """Persistent separator plus pattern-only cache and deterministic selection."""

    def __init__(
        self,
        instance: InventoryInstance,
        *,
        gamma: int,
        feasibility_tolerance: float = FAIRNESS_FEASIBILITY_TOLERANCE,
        output_flag: bool = False,
    ) -> None:
        self.instance = instance
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.output_flag = bool(output_flag)
        self.cache = CertifiedScenarioCache()
        self.persistent = PersistentFairnessSeparation(
            instance,
            gamma=int(gamma),
            feasibility_tolerance=feasibility_tolerance,
            output_flag=output_flag,
        )

    def separate(
        self,
        *,
        y_values: list[float],
        x_values: list[list[float]],
        t_value: float,
        cost_budget_value: float,
        mip_gap: float,
        time_limit: float,
        final_certification: bool,
    ) -> AdaptiveSeparation:
        start = time.perf_counter()
        cache_batch = None
        if self.cache.size:
            cache_batch = self.cache.certify_current_point(
                self.instance,
                y_values=y_values,
                x_values=x_values,
                t_value=t_value,
                cost_budget_value=cost_budget_value,
                time_limit=time_limit,
                feasibility_tolerance=self.feasibility_tolerance,
                max_cuts=max(1, self.cache.size),
                output_flag=self.output_flag,
            )
        remaining = float(time_limit) - (time.perf_counter() - start)
        if remaining <= 0.0:
            raise RemediationIdentityError("full separation time unavailable after cache recertification")
        full = self.persistent.separate(
            y_values=y_values,
            x_values=x_values,
            t_value=t_value,
            cost_budget_value=cost_budget_value,
            mip_gap=mip_gap,
            time_limit=remaining,
            max_cuts=1 if final_certification else 5,
            use_solution_pool=not final_certification,
            output_flag=self.output_flag,
        )
        source_cuts: list[tuple[str, FairnessFeasibilityCut]] = []
        if cache_batch is not None:
            source_cuts.extend(("pattern_cache", cut) for cut in cache_batch.cuts)
        full_cuts = full.cuts or ([full.cut] if full.cut is not None else [])
        for index, cut in enumerate(full_cuts):
            source_cuts.append((
                "primary_full_separation_incumbent" if index == 0 else "solution_pool",
                cut,
            ))
            self.cache.add(cut.active_deviations)

        certified_candidates: list[CertifiedAdaptiveCut] = []
        relative_inputs: list[dict[str, Any]] = []
        for source, cut in source_cuts:
            candidate = canonicalize_certified_cut(
                self.instance,
                cut,
                source=source,
                y_values=y_values,
                x_values=x_values,
                t_value=t_value,
            )
            if candidate is None:
                continue
            certified_candidates.append(candidate)
        unique, sources_by_hash = deduplicate_certified_candidates(certified_candidates)
        for digest in sorted(unique):
            candidate = unique[digest]
            for source in sorted(sources_by_hash[digest]):
                relative_inputs.append({
                    "source": source,
                    "certified_current_point": True,
                    "strictly_violating": True,
                    "canonical_cut_identity_valid": True,
                    "cut_sha256": digest,
                    "pattern_sha256": candidate.pattern_sha256,
                    "normalized_violation_bucket": candidate.normalized_violation_bucket,
                })
        relative = relative_normalized_violation_evidence(relative_inputs)
        eligible_hashes = {
            item["cut_sha256"]
            for item in relative["relative_violation_candidate_evidence"]
            if item["relative_violation_eligible"]
        }
        eligible = [unique[digest] for digest in sorted(eligible_hashes)]
        ordered, rejection = _deterministic_diverse_order(eligible)
        if full.robust_feasibility_certified and ordered:
            raise RemediationIdentityError("full separation certificate contradicts certified violating cuts")
        return AdaptiveSeparation(
            full_separation=full,
            candidates=[unique[digest] for digest in sorted(unique)],
            relative_evidence=relative,
            ordered_eligible=ordered,
            rejection_evidence=rejection,
            runtime=time.perf_counter() - start,
            metadata={
                "cache_candidate_count": 0 if cache_batch is None else cache_batch.candidate_count,
                "cache_hit_count": 0 if cache_batch is None else cache_batch.hit_count,
                "certified_cached_cut_count": 0 if cache_batch is None else cache_batch.certified_cut_count,
                "pool_candidate_count": full.pool_candidate_count,
                "duplicate_pattern_count": full.duplicate_pattern_count,
            },
        )

    def dispose(self) -> None:
        self.persistent.dispose()


def _master_variables(instance: InventoryInstance, y: Any, x: Any, t: Any) -> dict[str, Any]:
    return (
        {f"y[{i}]": y[i] for i in instance.I}
        | {f"x[{i},{j}]": x[i, j] for i in instance.I for j in instance.J}
        | {"T": t}
    )


def add_canonical_cut_payload(
    model: gp.Model,
    variables: dict[str, Any],
    payload: dict[str, Any],
    *,
    index: int,
) -> None:
    if payload.get("schema") != CUT_SCHEMA or payload.get("sense") != ">=" or payload.get("rhs") != float(0.0).hex():
        raise RemediationIdentityError("unsupported canonical cut payload")
    terms = payload.get("terms")
    if not isinstance(terms, list) or [item[0] for item in terms] != list(variables):
        raise RemediationIdentityError("canonical cut variable order mismatch")
    coefficients = [(identifier, float.fromhex(encoded)) for identifier, encoded in terms]
    constant = float.fromhex(payload["constant"])
    model.addConstr(
        constant + gp.quicksum(coefficient * variables[identifier] for identifier, coefficient in coefficients)
        >= 0.0,
        name=f"certified_adaptive_fairness_cut[{index}]",
    )


def _checkpoint_payload(
    *, identity: dict[str, Any], selection: dict[str, Any], algorithm_state: dict[str, Any],
) -> dict[str, Any]:
    validated = validate_checkpoint_state(selection)
    payload = {
        "schema": "fairness_large_remediation_algorithm_checkpoint_v1",
        "identity": deepcopy(identity),
        "selection": validated,
        "selection_sha256": hashlib.sha256(canonical_json_bytes(validated)).hexdigest().upper(),
        "algorithm_state": deepcopy(algorithm_state),
    }
    payload["checkpoint_identity_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()
    return payload


def _load_checkpoint(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    payload = read_json(path)
    if not path.exists():
        return None
    if not isinstance(payload, dict):
        raise RemediationIdentityError("algorithm checkpoint is corrupt")
    digest = payload.get("checkpoint_identity_sha256")
    unhashed = {key: deepcopy(value) for key, value in payload.items() if key != "checkpoint_identity_sha256"}
    if digest != hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest().upper():
        raise RemediationIdentityError("algorithm checkpoint identity hash mismatch")
    if payload.get("identity") != identity:
        raise RemediationIdentityError("algorithm checkpoint run identity mismatch")
    selection = validate_checkpoint_state(payload.get("selection", {}))
    expected = hashlib.sha256(canonical_json_bytes(selection)).hexdigest().upper()
    if payload.get("selection_sha256") != expected:
        raise RemediationIdentityError("selection checkpoint identity mismatch")
    if not isinstance(payload.get("algorithm_state"), dict):
        raise RemediationIdentityError("algorithm checkpoint state missing")
    return payload


def solve_certified_adaptive_multicut_fair_benders(
    instance: InventoryInstance,
    *,
    baseline_record: dict[str, Any],
    anchor: dict[str, Any],
    expected_identity: dict[str, Any],
    solver_parameters: dict[str, Any],
    rho: float,
    gamma: int = 2,
    algorithm_config: dict[str, Any] | None = None,
    max_iterations: int = 10000,
    time_limit: float = 1800.0,
    tol: float = 1.0e-4,
    feasibility_tolerance: float = FAIRNESS_FEASIBILITY_TOLERANCE,
    output_flag: bool = False,
    checkpoint_path: str | Path | None = None,
    checkpoint_identity: dict[str, Any] | None = None,
    failure_injector: Callable[[str, dict[str, Any]], None] | None = None,
) -> FairnessBendersResult:
    """Solve the sole remediation candidate with atomic cut-state recovery."""
    start = time.perf_counter()
    initial = construct_initial_t1_upper_bound(
        instance,
        baseline_record=baseline_record,
        anchor=anchor,
        rho=rho,
        tolerance=tol,
        expected_identity=expected_identity,
    )
    if solver_parameters != {"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7}:
        raise RemediationIdentityError("frozen solver identity mismatch")
    gp.setParam("Threads", 1)
    gp.setParam("Seed", 0)
    gp.setParam("FeasibilityTol", 1.0e-7)
    budget = fairness_cost_budget(float(anchor["value"]), rho)
    cfg = deepcopy(algorithm_config or {})
    precision = precision_policy_config(
        cfg,
        fixed_master_gap=float(cfg.get("fixed_master_mip_gap", 0.02)),
        fixed_subproblem_gap=float(cfg.get("fixed_subproblem_mip_gap", 0.05)),
        legacy_subproblem_gaps=[0.05, 0.0001],
    )
    if precision.precision_policy != "joint_error_budget":
        raise RemediationIdentityError("precision_policy must remain joint_error_budget")
    precision_state = initialize_precision_state(precision)
    model, y, x, t = _build_master(instance, output_flag)
    model.Params.Threads = 1
    model.Params.Seed = 0
    model.Params.FeasibilityTol = 1.0e-7
    variables = _master_variables(instance, y, x, t)
    for i in instance.I:
        y[i].Start = initial.y_values[i]
        for j in instance.J:
            x[i, j].Start = initial.x_values[i][j]
    t.Start = 1.0
    separator = CertifiedAdaptiveSeparator(
        instance,
        gamma=gamma,
        feasibility_tolerance=feasibility_tolerance,
        output_flag=output_flag,
    )
    identity = {
        "candidate": CANDIDATE,
        "candidate_sha256": CANDIDATE_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "initial_ub_theorem_sha256": INITIAL_UB_THEOREM_SHA256,
        "pattern_schema": PATTERN_SCHEMA,
        "cut_schema": CUT_SCHEMA,
        "relative_violation_schema": RELATIVE_VIOLATION_SCHEMA,
        "quantization_schema": QUANTIZATION_SCHEMA,
        "rho_hex": float(rho).hex(),
        "anchor_sha256": str(anchor["anchor_sha256"]).upper(),
        "run_identity": deepcopy(expected_identity),
        "solver_parameters": deepcopy(solver_parameters),
        **deepcopy(checkpoint_identity or {}),
    }
    checkpoint = None if checkpoint_path is None else _load_checkpoint(Path(checkpoint_path), identity)
    upper_bound = 1.0
    lower_bound: float | None = None
    best_y = list(initial.y_values)
    best_x = deepcopy(initial.x_values)
    certification_active = False
    log: list[dict[str, Any]] = []
    committed: list[str] = []
    payloads: dict[str, Any] = {}
    iteration_start = 1
    stall_counter = 0
    if checkpoint is not None:
        selection = checkpoint["selection"]
        state = checkpoint["algorithm_state"]
        upper_bound = float(state["upper_bound"])
        lower_bound = None if state.get("lower_bound") is None else float(state["lower_bound"])
        best_y = [float(value) for value in state["best_y"]]
        best_x = [[float(value) for value in row] for row in state["best_x"]]
        certification_active = bool(state["certification_active"])
        log = list(state.get("iteration_log", []))
        stall_counter = int(selection["stall_counter"])
        precision_payload = state.get("precision_state")
        if not isinstance(precision_payload, dict):
            raise RemediationIdentityError("checkpoint precision state missing")
        precision_state = PrecisionPolicyState(
            previous_master_gap=float(precision_payload["previous_master_gap"]),
            previous_subproblem_gap=float(precision_payload["previous_subproblem_gap"]),
        )
        payloads = deepcopy(selection["canonical_cut_payloads_by_sha256"])
        committed = list(selection["committed_master_cut_sha256_values"])
        for index, digest in enumerate(committed):
            add_canonical_cut_payload(model, variables, payloads[digest], index=index)
        if selection["cut_commit_state"] == "selection_complete_not_committed":
            for digest in selection["selected_cut_sha256_values"]:
                add_canonical_cut_payload(model, variables, payloads[digest], index=len(committed))
                committed.append(digest)
            committed_state = commit_selected_checkpoint(selection)
            committed = list(committed_state["committed_master_cut_sha256_values"])
            if checkpoint_path is not None:
                atomic_write_json(Path(checkpoint_path), _checkpoint_payload(
                    identity=identity, selection=committed_state, algorithm_state=state,
                ))
        iteration_start = int(selection["iteration"]) + 1

    status = "iteration_limit"
    master_runtime = 0.0
    separation_runtime = 0.0
    try:
        for iteration in range(iteration_start, int(max_iterations) + 1):
            remaining = float(time_limit) - (time.perf_counter() - start)
            if remaining <= 0.0:
                status = "time_limit"
                break
            decision = select_joint_error_budget_precision(
                precision,
                precision_state,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
                update_state=not certification_active,
            )
            precision_state = decision.next_state
            master_gap = 0.0 if certification_active else decision.master_selected_gap
            subproblem_gap = 0.0 if certification_active else decision.subproblem_selected_gap
            model.Params.MIPGap = master_gap
            model.Params.TimeLimit = max(1.0e-3, remaining)
            if failure_injector:
                failure_injector("before_master", {"iteration": iteration})
            master_start = time.perf_counter()
            model.optimize()
            master_elapsed = time.perf_counter() - master_start
            master_runtime += master_elapsed
            master_status = gurobi_status_name(model.Status)
            if model.SolCount <= 0:
                status = "infeasible" if model.Status == GRB.INFEASIBLE else master_status
                break
            candidate_lb = float(model.ObjBound)
            previous_lb = lower_bound
            lower_bound = candidate_lb if lower_bound is None else max(lower_bound, candidate_lb)
            improvement = 0.0 if previous_lb is None else max(0.0, lower_bound - previous_lb)
            improvement_bucket = quantized_bucket(improvement, "1e-9")
            stall_counter = stall_counter + 1 if improvement_bucket == 0 else 0
            candidate_t = float(t.X)
            candidate_y = [float(y[i].X) for i in instance.I]
            candidate_x = [[float(x[i, j].X) for j in instance.J] for i in instance.I]
            remaining = float(time_limit) - (time.perf_counter() - start)
            if remaining <= 0.0:
                status = "time_limit"
                break
            separated = separator.separate(
                y_values=candidate_y,
                x_values=candidate_x,
                t_value=candidate_t,
                cost_budget_value=budget.budget,
                mip_gap=subproblem_gap,
                time_limit=remaining,
                final_certification=certification_active,
            )
            if failure_injector:
                failure_injector("after_separation", {"iteration": iteration})
            separation_runtime += separated.runtime
            full = separated.full_separation
            if full.robust_feasibility_certified:
                if candidate_t < upper_bound:
                    upper_bound = candidate_t
                    best_y = candidate_y
                    best_x = candidate_x
            gap = relative_gap(upper_bound, lower_bound)
            all_payloads = deepcopy(payloads)
            all_payloads.update({candidate.cut_sha256: candidate.canonical_cut_payload for candidate in separated.candidates})
            ordered_hashes = [candidate.cut_sha256 for candidate in separated.ordered_eligible if candidate.cut_sha256 not in committed]
            selection = make_selection_checkpoint(
                iteration=iteration,
                committed_hashes=committed,
                eligible_ordered_hashes=ordered_hashes,
                canonical_cut_payloads_by_sha256=all_payloads,
                final_certification=certification_active,
                pattern_hashes=[candidate.pattern_sha256 for candidate in separated.candidates],
                quantized_lb_improvement_state=improvement_bucket,
                stall_counter=stall_counter,
                duplicate_and_redundancy_decisions=separated.rejection_evidence,
                relative_violation_evidence=separated.relative_evidence,
            )
            selected_hashes = list(selection["selected_cut_sha256_values"])
            next_certification_active = certification_active
            if upper_bound is not None and gap is not None and gap <= float(tol) and not certification_active:
                next_certification_active = True
            if certification_active and selected_hashes:
                next_certification_active = False
            iteration_entry = {
                "iteration": iteration,
                "master_status": master_status,
                "master_requested_mip_gap": master_gap,
                "separation_status": full.status,
                "separation_requested_mip_gap": subproblem_gap,
                "separation_objective": full.objective,
                "separation_objective_bound": full.objective_bound,
                "robust_feasibility_certified": full.robust_feasibility_certified,
                "relative_violation_status": separated.relative_evidence["relative_violation_status"],
                "relative_violation_denominator_bucket": separated.relative_evidence["relative_violation_denominator_bucket"],
                "relative_violation_candidate_evidence": separated.relative_evidence["relative_violation_candidate_evidence"],
                "selected_cut_sha256_values": selected_hashes,
                "committed_master_cut_sha256_values_before_iteration_commit": list(committed),
                "cuts_per_iteration": len(selected_hashes),
                "candidate_t": candidate_t,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "global_gap": gap,
                "certification_active": certification_active,
                "master_runtime": master_elapsed,
                "separation_runtime": separated.runtime,
                **separated.metadata,
            }
            log.append(iteration_entry)
            algorithm_state = {
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "best_y": best_y,
                "best_x": best_x,
                "certification_active": next_certification_active,
                "precision_state": asdict(precision_state),
                "iteration_log": log,
            }
            if failure_injector:
                failure_injector("before_selection_checkpoint", selection)
            if checkpoint_path is not None:
                atomic_write_json(Path(checkpoint_path), _checkpoint_payload(
                    identity=identity, selection=selection, algorithm_state=algorithm_state,
                ))
            if failure_injector:
                failure_injector("after_selection_checkpoint", selection)
            for offset, digest in enumerate(selected_hashes, start=1):
                add_canonical_cut_payload(model, variables, all_payloads[digest], index=len(committed) + offset - 1)
                if failure_injector:
                    failure_injector("memory_cut_add", {"index": offset, "cut_sha256": digest})
            if selected_hashes:
                committed_state = commit_selected_checkpoint(selection)
                if failure_injector:
                    failure_injector("before_commit_checkpoint", committed_state)
                if checkpoint_path is not None:
                    atomic_write_json(Path(checkpoint_path), _checkpoint_payload(
                        identity=identity, selection=committed_state, algorithm_state=algorithm_state,
                    ))
                committed = list(committed_state["committed_master_cut_sha256_values"])
                payloads = all_payloads
                if failure_injector:
                    failure_injector("after_commit_checkpoint", committed_state)
            if upper_bound is not None and gap is not None and gap <= float(tol):
                if certification_active:
                    if full.robust_feasibility_certified and model.Status == GRB.OPTIMAL and not selected_hashes:
                        status = "optimal"
                        break
                else:
                    certification_active = next_certification_active
                    continue
            if certification_active and selected_hashes:
                certification_active = next_certification_active
            if not selected_hashes and not full.robust_feasibility_certified:
                status = full.status if full.status not in {"optimal", "unknown"} else "separation_stalled_duplicate"
                break
        else:
            status = "iteration_limit"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        separator.dispose()
        model.dispose()

    runtime = time.perf_counter() - start
    gap = relative_gap(upper_bound, lower_bound)
    return FairnessBendersResult(
        status=status,
        objective_t=upper_bound,
        robust_minimum_fill_rate=None if upper_bound is None else 1.0 - upper_bound,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        gap=gap,
        runtime=runtime,
        iterations=len(log),
        cuts=len(committed),
        cuts_with_cost_component=0,
        cuts_with_fairness_component=0,
        joint_cost_fairness_cuts=0,
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
            "total_committed_unique_master_cuts": len(committed),
            "selected_committed_atomic_state_machine": True,
            "full_separation_objective_bound_required": True,
            "runtime_driven_scientific_branching": False,
        },
    )
