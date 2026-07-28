"""Solver-free static audit and dry-run for the final fairness remediation protocol."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import unicodedata

import yaml

BASE = "d6ae40c0bbe3d9af7c67cb7e71fb8ad45e64b033"
CANDIDATE = "certified_adaptive_multicut_fair_benders"
FREEZE_DECISION_SHA = "7EBFD4F22C5AF2B26E630722FB3F8D17E83FAEE9AA58248BDA32A386FFDE29B2"
FREEZE_INDEX_SHA = "BC0781818CA2DD1F5964512BEEF5438CAD8181BE5FF8B0A992DE716E98DF2358"
FREEZE_DOC_SHA = "AA95348D1677CBB714F88812B56C78E94E8450E533D3F37E2AA8EB5616311C9B"
CONFIGS = (
    "experiments/configs/fairness_large_final_remediation_pilot.yaml",
    "experiments/configs/fairness_large_final_remediation_large_s1.yaml",
    "experiments/configs/fairness_large_final_remediation_medium_large_s1.yaml",
)
PATTERN_SCHEMA = "fairness_deviation_pattern_v1"
CUT_SCHEMA = "fairness_farkas_cut_v1"
FARKAS_NORMALIZATION = "nonnegative_multipliers_sum_to_one_v1"
RAW_VIOLATION_TOLERANCE = Decimal("1e-7")
RAW_VIOLATION_QUANTUM = Decimal("1e-9")
NORMALIZED_VIOLATION_QUANTUM = Decimal("1e-9")
COSINE_SIMILARITY_QUANTUM = Decimal("1e-9")
RELATIVE_VIOLATION_QUANTUM = Decimal("1e-9")
DIVERSITY_QUANTUM = Decimal("1e-9")
THRESHOLD_DEADBAND_BUCKETS = 2
RELATIVE_VIOLATION_THRESHOLD = Decimal("0.10")
COSINE_REDUNDANCY_THRESHOLD = Decimal("0.98")
QUANTIZATION_SCHEMA = "binary64_integer_ratio_round_half_even_v2"
RELATIVE_VIOLATION_SCHEMA = "relative_normalized_violation_v1"
RELATIVE_VIOLATION_SCALE = 10**9
RELATIVE_VIOLATION_THRESHOLD_BUCKET = 100_000_000
RELATIVE_VIOLATION_ELIGIBILITY_FLOOR = 100_000_002
RELATIVE_VIOLATION_DENOMINATOR_RULE = (
    "maximum_normalized_violation_bucket_over_unique_certified_violating_candidates"
)
RELATIVE_CANDIDATE_SOURCES = {
    "primary_full_separation_incumbent", "pattern_cache", "solution_pool",
}
CUT_COMMIT_STATES = {
    "no_selection", "selection_complete_not_committed", "commit_complete",
}


class QuantizationFailClosed(ValueError):
    """A classified numeric-identity failure that must stop the scientific path."""

    def __init__(self, status: str):
        self.status = status
        super().__init__(status)


class CommitInterrupted(RuntimeError):
    """Simulated volatile interruption; no partially committed state is durable."""


class CheckpointWriteFailure(RuntimeError):
    """Atomic commit-checkpoint failure; the process must stop and rebuild."""


class RelativeViolationFailClosed(ValueError):
    """A classified relative-violation identity or candidate-set failure."""

    def __init__(self, status: str):
        self.status = status
        super().__init__(status)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_id(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value))


def _canonical_float_hex(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Canonical cut values must be finite IEEE-754 binary64 values.")
    if number == 0.0:
        number = 0.0
    return number.hex()


def canonical_pattern_bytes(
    region_ids: list[Any] | tuple[Any, ...],
    product_ids: list[Any] | tuple[Any, ...],
    values_by_component: dict[tuple[Any, Any], Any],
) -> bytes:
    regions = [_canonical_id(value) for value in region_ids]
    products = [_canonical_id(value) for value in product_ids]
    if len(regions) != len(set(regions)) or len(products) != len(set(products)):
        raise ValueError("Canonical instance region/product IDs must be unique after NFC normalization.")
    normalized_values = {
        (_canonical_id(region), _canonical_id(product)): value
        for (region, product), value in values_by_component.items()
    }
    order = [(region, product) for region in regions for product in products]
    if len(normalized_values) != len(values_by_component) or set(normalized_values) != set(order):
        raise ValueError("Pattern components must exactly match the formal instance R-major/J-minor order.")
    values: list[int] = []
    for component in order:
        value = normalized_values[component]
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            raise ValueError("Pattern values must be integer 0 or 1.")
        values.append(value)
    payload = {
        "schema": PATTERN_SCHEMA,
        "component_order": [[region, product] for region, product in order],
        "values": values,
    }
    return canonical_json_bytes(payload)


def pattern_sha256(region_ids: list[Any], product_ids: list[Any], values: dict[tuple[Any, Any], Any]) -> str:
    return hashlib.sha256(canonical_pattern_bytes(region_ids, product_ids, values)).hexdigest().upper()


def canonical_cut_bytes(
    variable_ids: list[Any] | tuple[Any, ...],
    terms: dict[Any, Any] | list[tuple[Any, Any]],
    *,
    constant: Any,
    rhs: Any,
    sense: str,
) -> bytes:
    canonical_variables = [_canonical_id(value) for value in variable_ids]
    if len(canonical_variables) != len(set(canonical_variables)):
        raise ValueError("Canonical master variable IDs must be unique after NFC normalization.")
    items = list(terms.items()) if isinstance(terms, dict) else list(terms)
    term_map: dict[str, Any] = {}
    for identifier, coefficient in items:
        key = _canonical_id(identifier)
        if key in term_map:
            raise ValueError("Duplicate canonical master variable ID in cut terms.")
        term_map[key] = coefficient
    if set(term_map) != set(canonical_variables):
        raise ValueError("Cut terms must exactly match canonical master variable IDs.")
    if sense not in {">=", "<=", "="}:
        raise ValueError("Unsupported cut sense.")
    payload = {
        "schema": CUT_SCHEMA,
        "farkas_normalization": FARKAS_NORMALIZATION,
        "sense": sense,
        "terms": [[key, _canonical_float_hex(term_map[key])] for key in canonical_variables],
        "constant": _canonical_float_hex(constant),
        "rhs": _canonical_float_hex(rhs),
    }
    return canonical_json_bytes(payload)


def cut_sha256(variable_ids: list[Any], terms: dict[Any, Any] | list[tuple[Any, Any]], **identity: Any) -> str:
    return hashlib.sha256(canonical_cut_bytes(variable_ids, terms, **identity)).hexdigest().upper()


def _binary64(value: Any) -> float:
    if isinstance(value, bool):
        raise QuantizationFailClosed("invalid_binary64_input")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuantizationFailClosed("invalid_binary64_input") from exc
    if not math.isfinite(number):
        raise QuantizationFailClosed("nonfinite_numeric_identity")
    return 0.0 if number == 0.0 else number


def _positive_quantum_ratio(quantum: Any) -> tuple[int, int]:
    try:
        if isinstance(quantum, bool):
            raise ValueError
        if isinstance(quantum, Decimal):
            if not quantum.is_finite() or quantum <= 0:
                raise ValueError
            numerator, denominator = quantum.as_integer_ratio()
        elif isinstance(quantum, int):
            if quantum <= 0:
                raise ValueError
            numerator, denominator = quantum, 1
        elif isinstance(quantum, str):
            parsed = Decimal(quantum)
            if not parsed.is_finite() or parsed <= 0:
                raise ValueError
            numerator, denominator = parsed.as_integer_ratio()
        else:
            raise ValueError
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise QuantizationFailClosed("invalid_quantum") from exc
    return numerator, denominator


def quantized_bucket(value: Any, quantum: Any) -> int:
    """Exact RoundHalfEven(binary64 / rational quantum) using arbitrary-size ints."""
    number = _binary64(value)
    quantum_numerator, quantum_denominator = _positive_quantum_ratio(quantum)
    numerator, denominator = number.as_integer_ratio()
    scaled = abs(numerator) * quantum_denominator
    divisor = denominator * quantum_numerator
    quotient, remainder = divmod(scaled, divisor)
    doubled = 2 * remainder
    if doubled > divisor or (doubled == divisor and quotient % 2 == 1):
        quotient += 1
    return -quotient if numerator < 0 else quotient


def raw_violation_is_accepted(value: Any) -> bool:
    bucket = quantized_bucket(value, RAW_VIOLATION_QUANTUM)
    threshold = quantized_bucket(float(RAW_VIOLATION_TOLERANCE), RAW_VIOLATION_QUANTUM)
    return bucket > threshold + THRESHOLD_DEADBAND_BUCKETS


def relative_violation_bucket_is_eligible(bucket: int) -> bool:
    if isinstance(bucket, bool) or not isinstance(bucket, int) or bucket < 0:
        raise RelativeViolationFailClosed("invalid_relative_violation_bucket")
    return bucket > RELATIVE_VIOLATION_ELIGIBILITY_FLOOR


def round_half_even_nonnegative_ratio(numerator: int, denominator: int) -> int:
    """Round a nonnegative integer ratio exactly, without floating-point division."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (numerator, denominator)):
        raise RelativeViolationFailClosed("invalid_relative_violation_integer_ratio")
    if numerator < 0 or denominator <= 0:
        raise RelativeViolationFailClosed("invalid_relative_violation_integer_ratio")
    quotient, remainder = divmod(numerator, denominator)
    doubled = 2 * remainder
    if doubled > denominator or (doubled == denominator and quotient % 2 == 1):
        quotient += 1
    return quotient


def relative_normalized_violation_evidence(
    candidates: list[dict[str, Any]], *, schema: str = RELATIVE_VIOLATION_SCHEMA,
) -> dict[str, Any]:
    """Build the order-independent relative-eligibility evidence for one iteration."""
    if schema != RELATIVE_VIOLATION_SCHEMA:
        raise RelativeViolationFailClosed("relative_violation_identity_mismatch")
    if not isinstance(candidates, list):
        raise RelativeViolationFailClosed("invalid_relative_candidate_set")
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RelativeViolationFailClosed("invalid_relative_candidate_set")
        source = candidate.get("source")
        if source not in RELATIVE_CANDIDATE_SOURCES:
            raise RelativeViolationFailClosed("invalid_relative_candidate_source")
        if candidate.get("certified_current_point") is not True or candidate.get("strictly_violating") is not True:
            continue
        cut_hash = candidate.get("cut_sha256")
        pattern_hash = candidate.get("pattern_sha256")
        normalized = candidate.get("normalized_violation_bucket")
        if not isinstance(cut_hash, str) or re.fullmatch(r"[0-9A-F]{64}", cut_hash) is None:
            raise RelativeViolationFailClosed("invalid_relative_candidate_identity")
        if not isinstance(pattern_hash, str) or re.fullmatch(r"[0-9A-F]{64}", pattern_hash) is None:
            raise RelativeViolationFailClosed("invalid_relative_candidate_identity")
        if isinstance(normalized, bool) or not isinstance(normalized, int) or normalized < 0:
            raise RelativeViolationFailClosed("invalid_normalized_violation_bucket")
        if candidate.get("canonical_cut_identity_valid") is not True:
            raise RelativeViolationFailClosed("invalid_relative_candidate_identity")
        previous = unique.get(cut_hash)
        if previous is None:
            unique[cut_hash] = {
                "cut_sha256": cut_hash,
                "pattern_sha256": pattern_hash,
                "sources": {source},
                "normalized_violation_bucket": normalized,
            }
        else:
            if (previous["pattern_sha256"] != pattern_hash
                    or previous["normalized_violation_bucket"] != normalized):
                raise RelativeViolationFailClosed("relative_violation_identity_mismatch")
            previous["sources"].add(source)
    ordered = [unique[key] for key in sorted(unique)]
    common = {
        "relative_violation_schema": RELATIVE_VIOLATION_SCHEMA,
        "relative_violation_quantum": "1e-9",
        "relative_violation_threshold_bucket": RELATIVE_VIOLATION_THRESHOLD_BUCKET,
        "relative_violation_deadband_buckets": THRESHOLD_DEADBAND_BUCKETS,
        "relative_violation_denominator_rule": RELATIVE_VIOLATION_DENOMINATOR_RULE,
    }
    if not ordered:
        return {
            **common,
            "relative_violation_denominator_bucket": None,
            "relative_violation_candidate_evidence": [],
            "relative_violation_status": "no_certified_violating_candidates",
            "relative_violation_eligible_count": 0,
        }
    maximum = max(item["normalized_violation_bucket"] for item in ordered)
    if maximum == 0:
        evidence = [
            {
                **item,
                "sources": sorted(item["sources"]),
                "relative_violation_bucket": None,
                "relative_violation_eligible": False,
                "relative_violation_status": "no_positive_quantized_normalized_violation",
            }
            for item in ordered
        ]
        return {
            **common,
            "relative_violation_denominator_bucket": 0,
            "relative_violation_candidate_evidence": evidence,
            "relative_violation_status": "no_positive_quantized_normalized_violation",
            "relative_violation_eligible_count": 0,
        }
    evidence = []
    for item in ordered:
        relative = round_half_even_nonnegative_ratio(
            item["normalized_violation_bucket"] * RELATIVE_VIOLATION_SCALE, maximum
        )
        eligible = relative > RELATIVE_VIOLATION_ELIGIBILITY_FLOOR
        evidence.append(
            {
                **item,
                "sources": sorted(item["sources"]),
                "relative_violation_bucket": relative,
                "relative_violation_eligible": eligible,
                "relative_violation_status": "relative_violation_computed",
            }
        )
    return {
        **common,
        "relative_violation_denominator_bucket": maximum,
        "relative_violation_candidate_evidence": evidence,
        "relative_violation_status": "relative_violation_computed",
        "relative_violation_eligible_count": sum(item["relative_violation_eligible"] for item in evidence),
    }


def cosine_is_redundant(value: Any) -> bool:
    bucket = quantized_bucket(value, COSINE_SIMILARITY_QUANTUM)
    threshold = quantized_bucket(float(COSINE_REDUNDANCY_THRESHOLD), COSINE_SIMILARITY_QUANTUM)
    return bucket >= threshold - THRESHOLD_DEADBAND_BUCKETS


def candidate_order_key(
    *, normalized_violation: Any, diversity: Any,
    pattern_hash: str, cut_hash: str,
) -> tuple[Any, ...]:
    return (
        -quantized_bucket(normalized_violation, NORMALIZED_VIOLATION_QUANTUM),
        -quantized_bucket(diversity, DIVERSITY_QUANTUM),
        str(pattern_hash).upper(),
        str(cut_hash).upper(),
    )


def adaptive_batch_size(
    total_committed_unique_master_cuts: int,
    *,
    final_certification: bool = False,
    reporting_runtime: dict[str, Any] | None = None,
) -> int:
    del reporting_runtime
    if (isinstance(total_committed_unique_master_cuts, bool)
            or not isinstance(total_committed_unique_master_cuts, int)
            or total_committed_unique_master_cuts < 0):
        raise ValueError("total_committed_unique_master_cuts must be a nonnegative integer.")
    if final_certification:
        return 1
    if total_committed_unique_master_cuts < 1000:
        return 5
    if total_committed_unique_master_cuts < 3000:
        return 3
    if total_committed_unique_master_cuts < 5000:
        return 2
    return 1


def adaptive_batch_segment(total_committed_unique_master_cuts: int, *, final_certification: bool = False) -> str:
    size = adaptive_batch_size(total_committed_unique_master_cuts, final_certification=final_certification)
    if final_certification:
        return "final_certification_batch_1"
    if total_committed_unique_master_cuts < 1000:
        return "cuts_0_999_batch_5"
    if total_committed_unique_master_cuts < 3000:
        return "cuts_1000_2999_batch_3"
    if total_committed_unique_master_cuts < 5000:
        return "cuts_3000_4999_batch_2"
    assert size == 1
    return "cuts_5000_plus_batch_1"


def selected_cut_count(eligible_count: int, committed_count: int, *, final_certification: bool = False) -> int:
    if isinstance(eligible_count, bool) or not isinstance(eligible_count, int) or eligible_count < 0:
        raise ValueError("eligible_count must be a nonnegative integer.")
    return min(eligible_count, adaptive_batch_size(committed_count, final_certification=final_certification))


CHECKPOINT_SELECTION_FIELDS = (
    "iteration", "total_committed_unique_master_cuts", "final_certification",
    "current_batch_schedule_segment", "quantized_lb_improvement_state", "stall_counter",
    "pattern_sha256_values", "candidate_ordering", "duplicate_and_redundancy_decisions",
    "selected_cut_sha256_values", "committed_master_cut_sha256_values",
    "committed_cut_count_before_iteration", "committed_cut_count_after_iteration",
    "cut_commit_state", "canonical_cut_payloads_by_sha256", "pattern_schema_version",
    "cut_schema_version", "quantization_schema", "quantization_identity",
    "relative_violation_schema", "relative_violation_quantum",
    "relative_violation_threshold_bucket", "relative_violation_deadband_buckets",
    "relative_violation_denominator_rule", "relative_violation_denominator_bucket",
    "relative_violation_candidate_evidence", "relative_violation_status",
    "relative_violation_eligible_count",
)


def _valid_sha_list(values: Any) -> bool:
    return isinstance(values, list) and all(
        isinstance(value, str) and re.fullmatch(r"[0-9A-F]{64}", value) for value in values
    )


def _validate_cut_payload_map(payloads: Any, required_hashes: set[str]) -> None:
    if not isinstance(payloads, dict) or not required_hashes.issubset(payloads):
        raise ValueError("Selected and committed cuts must have reconstructable canonical payloads.")
    for digest in required_hashes:
        payload = payloads[digest]
        actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()
        if actual != digest or payload.get("schema") != CUT_SCHEMA:
            raise ValueError("Canonical cut payload does not match its cut SHA or schema.")


def _validate_relative_checkpoint_evidence(payload: dict[str, Any]) -> set[str]:
    evidence = payload["relative_violation_candidate_evidence"]
    if not isinstance(evidence, list):
        raise RelativeViolationFailClosed("relative_violation_identity_mismatch")
    candidates: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("sources"), list):
            raise RelativeViolationFailClosed("relative_violation_identity_mismatch")
        for source in item["sources"]:
            candidates.append(
                {
                    "source": source,
                    "certified_current_point": True,
                    "strictly_violating": True,
                    "canonical_cut_identity_valid": True,
                    "cut_sha256": item.get("cut_sha256"),
                    "pattern_sha256": item.get("pattern_sha256"),
                    "normalized_violation_bucket": item.get("normalized_violation_bucket"),
                }
            )
    expected = relative_normalized_violation_evidence(
        candidates, schema=payload["relative_violation_schema"]
    )
    fields = (
        "relative_violation_schema", "relative_violation_quantum",
        "relative_violation_threshold_bucket", "relative_violation_deadband_buckets",
        "relative_violation_denominator_rule", "relative_violation_denominator_bucket",
        "relative_violation_candidate_evidence", "relative_violation_status",
        "relative_violation_eligible_count",
    )
    if any(payload[field] != expected[field] for field in fields):
        raise RelativeViolationFailClosed("relative_violation_identity_mismatch")
    return {
        item["cut_sha256"] for item in evidence if item["relative_violation_eligible"]
    }


def validate_checkpoint_state(state: dict[str, Any]) -> dict[str, Any]:
    if any(field not in state for field in CHECKPOINT_SELECTION_FIELDS):
        raise ValueError("Checkpoint is missing frozen adaptive-selection or commit state.")
    payload = {field: deepcopy(state[field]) for field in CHECKPOINT_SELECTION_FIELDS}
    if payload["quantization_schema"] != QUANTIZATION_SCHEMA:
        raise QuantizationFailClosed("quantization_schema_mismatch")
    if payload["pattern_schema_version"] != PATTERN_SCHEMA or payload["cut_schema_version"] != CUT_SCHEMA:
        raise ValueError("Checkpoint canonicalization schema mismatch.")
    for field in (
        "iteration", "total_committed_unique_master_cuts", "quantized_lb_improvement_state",
        "stall_counter", "committed_cut_count_before_iteration", "committed_cut_count_after_iteration",
    ):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Checkpoint {field} must be a nonnegative integer.")
    if not isinstance(payload["final_certification"], bool):
        raise ValueError("Checkpoint final_certification must be boolean.")
    for field in (
        "pattern_sha256_values", "candidate_ordering", "selected_cut_sha256_values",
        "committed_master_cut_sha256_values",
    ):
        if not _valid_sha_list(payload[field]):
            raise ValueError(f"Checkpoint {field} must contain uppercase SHA256 values.")
    candidates = payload["candidate_ordering"]
    selected = payload["selected_cut_sha256_values"]
    committed = payload["committed_master_cut_sha256_values"]
    if any(len(values) != len(set(values)) for values in (candidates, selected, committed)):
        raise ValueError("Candidate, selected and committed cut SHA lists must each be unique.")
    before = payload["committed_cut_count_before_iteration"]
    after = payload["committed_cut_count_after_iteration"]
    state_name = payload["cut_commit_state"]
    if state_name not in CUT_COMMIT_STATES:
        raise ValueError("Illegal cut commit state.")
    if payload["current_batch_schedule_segment"] != adaptive_batch_segment(
        before, final_certification=payload["final_certification"]
    ):
        raise ValueError("Checkpoint batch schedule segment is inconsistent with start-of-iteration C_k.")
    expected_selected_count = selected_cut_count(
        len(candidates), before, final_certification=payload["final_certification"]
    )
    if selected != candidates[:expected_selected_count] or set(candidates).intersection(committed[:before]):
        raise ValueError("Selected cuts must be exactly the frozen eligible-order prefix and new to the master.")
    if payload["quantization_identity"] != {
        "quantum_numerator": 1, "quantum_denominator": 10**9, "deadband_buckets": 2,
    }:
        raise QuantizationFailClosed("quantization_schema_mismatch")
    relative_eligible_hashes = _validate_relative_checkpoint_evidence(payload)
    if not set(candidates).issubset(relative_eligible_hashes):
        raise RelativeViolationFailClosed("relative_violation_identity_mismatch")
    if state_name in {"no_selection", "selection_complete_not_committed"}:
        if before != after or after != len(committed) or payload["total_committed_unique_master_cuts"] != before:
            raise ValueError("Uncommitted checkpoint counts must equal the committed list length C_k.")
        if set(selected).intersection(committed):
            raise ValueError("Selected uncommitted cuts must be new relative to committed cuts.")
        if state_name == "no_selection" and selected:
            raise ValueError("A no-selection checkpoint cannot contain selected cuts.")
        if state_name == "selection_complete_not_committed" and not selected:
            raise ValueError("A selection checkpoint must contain selected cuts.")
    else:
        if after != len(committed) or payload["total_committed_unique_master_cuts"] != after:
            raise ValueError("Committed checkpoint after-count must equal the committed list length.")
        if after != before + len(selected) or committed[before:] != selected:
            raise ValueError("Committed cuts must append the selected list exactly once in frozen order.")
    _validate_cut_payload_map(payload["canonical_cut_payloads_by_sha256"], set(selected) | set(committed))
    return payload


def checkpoint_selection_signature(state: dict[str, Any]) -> str:
    payload = validate_checkpoint_state(state)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()


def make_selection_checkpoint(
    *, iteration: int, committed_hashes: list[str], eligible_ordered_hashes: list[str],
    canonical_cut_payloads_by_sha256: dict[str, Any], final_certification: bool = False,
    pattern_hashes: list[str] | None = None, quantized_lb_improvement_state: int = 0,
    stall_counter: int = 0, duplicate_and_redundancy_decisions: list[Any] | None = None,
    relative_violation_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if relative_violation_evidence is None:
        raise RelativeViolationFailClosed("missing_relative_violation_evidence")
    if len(eligible_ordered_hashes) != len(set(eligible_ordered_hashes)):
        raise ValueError("Eligible cut hashes must be unique before selection.")
    if set(eligible_ordered_hashes).intersection(committed_hashes):
        raise ValueError("Eligible cuts cannot already be committed.")
    before = len(committed_hashes)
    selected = eligible_ordered_hashes[:selected_cut_count(
        len(eligible_ordered_hashes), before, final_certification=final_certification
    )]
    relative_fields = {
        key: deepcopy(relative_violation_evidence[key])
        for key in (
            "relative_violation_schema", "relative_violation_quantum",
            "relative_violation_threshold_bucket", "relative_violation_deadband_buckets",
            "relative_violation_denominator_rule", "relative_violation_denominator_bucket",
            "relative_violation_candidate_evidence", "relative_violation_status",
            "relative_violation_eligible_count",
        )
    }
    state = {
        "iteration": iteration,
        "total_committed_unique_master_cuts": before,
        "final_certification": final_certification,
        "current_batch_schedule_segment": adaptive_batch_segment(before, final_certification=final_certification),
        "quantized_lb_improvement_state": quantized_lb_improvement_state,
        "stall_counter": stall_counter,
        "pattern_sha256_values": list(pattern_hashes or []),
        "candidate_ordering": list(eligible_ordered_hashes),
        "duplicate_and_redundancy_decisions": list(duplicate_and_redundancy_decisions or []),
        "selected_cut_sha256_values": selected,
        "committed_master_cut_sha256_values": list(committed_hashes),
        "committed_cut_count_before_iteration": before,
        "committed_cut_count_after_iteration": before,
        "cut_commit_state": "selection_complete_not_committed" if selected else "no_selection",
        "canonical_cut_payloads_by_sha256": deepcopy(canonical_cut_payloads_by_sha256),
        "pattern_schema_version": PATTERN_SCHEMA,
        "cut_schema_version": CUT_SCHEMA,
        "quantization_schema": QUANTIZATION_SCHEMA,
        "quantization_identity": {"quantum_numerator": 1, "quantum_denominator": 10**9, "deadband_buckets": 2},
        **relative_fields,
    }
    validate_checkpoint_state(state)
    return state


def commit_selected_checkpoint(
    state: dict[str, Any], *, interrupt_after_memory_adds: int | None = None,
    checkpoint_write_success: bool = True,
) -> dict[str, Any]:
    persisted = validate_checkpoint_state(state)
    if persisted["cut_commit_state"] != "selection_complete_not_committed":
        raise ValueError("Only a persisted selection checkpoint can be committed.")
    volatile_committed = list(persisted["committed_master_cut_sha256_values"])
    for index, digest in enumerate(persisted["selected_cut_sha256_values"], start=1):
        _validate_cut_payload_map(persisted["canonical_cut_payloads_by_sha256"], {digest})
        if digest in volatile_committed:
            raise ValueError("A cut SHA cannot be added to the master twice.")
        volatile_committed.append(digest)
        if interrupt_after_memory_adds == index:
            raise CommitInterrupted("volatile master additions are not durable scientific state")
    if not checkpoint_write_success:
        raise CheckpointWriteFailure("atomic commit checkpoint failed; stop and rebuild")
    committed = deepcopy(persisted)
    committed["committed_master_cut_sha256_values"] = volatile_committed
    committed["committed_cut_count_after_iteration"] = len(volatile_committed)
    committed["total_committed_unique_master_cuts"] = len(volatile_committed)
    committed["cut_commit_state"] = "commit_complete"
    validate_checkpoint_state(committed)
    return committed


def resume_action(state: dict[str, Any]) -> str:
    persisted = validate_checkpoint_state(state)
    return {
        "no_selection": "resolve_from_complete_separation",
        "selection_complete_not_committed": "rebuild_committed_master_then_recommit_same_selected_hashes_once",
        "commit_complete": "rebuild_committed_master_then_advance_to_next_iteration",
    }[persisted["cut_commit_state"]]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def canonical_run_key(config: dict[str, Any], task: str, seed: int, rho: float | None) -> str:
    identity = {
        "candidate": "baseline" if task == "baseline" else CANDIDATE,
        "execution_attempt": config["execution_attempt"],
        "rho": "NOT_APPLICABLE" if rho is None else format(rho, ".2f"),
        "scale": config["scale"],
        "seed": seed,
        "task_type": task,
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def expand_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        key = canonical_run_key(config, "baseline", seed, None)
        rows.append({"task_type": "baseline", "seed": seed, "rho": None, "run_key": key})
        for rho in config["rho"]:
            key = canonical_run_key(config, "frontier", seed, float(rho))
            rows.append({"task_type": "frontier", "seed": seed, "rho": float(rho), "run_key": key})
    for row in rows:
        row["run_directory_id"] = "r_" + hashlib.sha256(row["run_key"].encode("utf-8")).hexdigest()[:24]
    return rows


def _candidate_paths(root: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> list[Path]:
    output = root / config["output_dir"]
    paths = [output / "manifest.json.tmp", output / "results.csv.tmp", output / "summary.csv.tmp", output / "audit.json.tmp"]
    chunk_count = (int(config["scenario_count"]) + int(config["post_evaluation"]["checkpoint_chunk_size"]) - 1) // int(config["post_evaluation"]["checkpoint_chunk_size"])
    for row in rows:
        run_dir = output / row["run_directory_id"]
        paths.extend((run_dir / "run.json.tmp", run_dir / "status.json.tmp", run_dir / "algorithm_checkpoint.json.tmp"))
        if row["task_type"] == "frontier":
            paths.extend((run_dir / "post_evaluation" / "index.json.tmp", run_dir / "post_evaluation" / f"chunk_{chunk_count - 1:04d}.json.tmp"))
    return paths


def audit(root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    freeze = root / "analysis/fairness_scalability_s1_attempt2_cross_scale_freeze"
    decision_path = freeze / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    check("freeze decision SHA", sha256(decision_path) == FREEZE_DECISION_SHA, sha256(decision_path))
    check("freeze decision", decision["decision"] == "no_existing_candidate_passes_cross_scale_s1")
    check("freeze medium-large", decision["medium_large_certified_frontier"] == "20/24")
    check("freeze large", decision["large_certified_frontier"] == "0/24")
    check("freeze gates", not decision["original_s2_authorized"] and not decision["full_grid_authorized"] and not decision["attempt4_authorized"])
    index_path = freeze / "artifact_sha256.csv"
    check("freeze artifact index SHA", sha256(index_path) == FREEZE_INDEX_SHA, sha256(index_path))
    with index_path.open(newline="", encoding="utf-8-sig") as stream:
        indexed = list(csv.DictReader(stream))
    check("freeze artifact index coverage", len(indexed) == 7)
    for row in indexed:
        check(f"freeze artifact {row['artifact_path']}", sha256(freeze / row["artifact_path"]) == row["sha256"])
    freeze_doc = root / "docs/audits/fairness_scalability_s1_attempt2_cross_scale_decision.md"
    check("freeze decision document SHA", sha256(freeze_doc) == FREEZE_DOC_SHA, sha256(freeze_doc))

    candidate_path = root / "experiments/configs/fairness_large_final_remediation_candidate.yaml"
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    check("one candidate", candidate["candidate_name"] == CANDIDATE)
    check("no cut deletion", candidate["forbidden"]["cut_deletion"] is True)
    check("current point recertification", candidate["certification"]["fixed_scenario_recertification_at_current_point"] is True)
    check("complete separation bound", candidate["certification"]["full_separation_objective_bound_required"] is True)
    check("adaptive cut bounds", candidate["adaptive_multicut"]["minimum_cuts_when_a_certified_violation_exists"] == 1 and candidate["adaptive_multicut"]["maximum_certified_cuts_per_iteration"] == 5)
    canonical = candidate["canonical_identity"]
    check("canonical pattern schema", canonical["pattern"]["schema"] == PATTERN_SCHEMA)
    check("canonical cut schema", canonical["cut"]["schema"] == CUT_SCHEMA)
    check("canonical JSON", canonical["json_serialization"] == {
        "sort_keys": True, "separators": [",", ":"], "ensure_ascii": False,
        "allow_nan": False, "encoding": "utf-8", "trailing_newline": False,
    })
    check("float hex and signed zero", canonical["cut"]["coefficient_encoding"] == "python_float_hex_binary64" and canonical["cut"]["signed_zero"] == "canonical_positive_zero")
    check("nonfinite fails closed", canonical["cut"]["nonfinite"] == "fail_closed" and candidate["numeric_selection"]["nonfinite"] == "fail_closed")
    numeric = candidate["numeric_selection"]
    check("exact integer-ratio quantization", numeric["rounding_mode"] == "ROUND_HALF_EVEN" and numeric["implementation"] == "binary64_as_integer_ratio_arbitrary_precision_integer" and numeric["quantization_schema"] == QUANTIZATION_SCHEMA and all(float(numeric[name]) == 1e-9 for name in (
        "raw_violation_quantum", "normalized_violation_quantum", "relative_violation_quantum",
        "cosine_similarity_quantum", "diversity_quantum", "lb_improvement_quantum",
    )))
    check("deadband", numeric["threshold_deadband_buckets"] == 2 and numeric["violation_deadband_policy"] == "reject_or_do_not_promote" and numeric["cosine_deadband_policy"] == "redundant")
    relative = candidate["relative_normalized_violation"]
    check("relative violation schema and candidate union", relative["schema"] == RELATIVE_VIOLATION_SCHEMA and relative["source_union_before_denominator"] is True and set(relative["allowed_sources"]) == RELATIVE_CANDIDATE_SOURCES)
    check("relative violation denominator", relative["denominator_rule"] == RELATIVE_VIOLATION_DENOMINATOR_RULE and relative["exact_cut_sha256_deduplication_before_denominator"] is True and relative["uncertified_candidates_in_denominator"] == "forbidden")
    check("relative violation integer formula", relative["ratio_formula"] == "RoundHalfEven(N_i*1000000000/N_max)_with_arbitrary_precision_integers" and relative["binary64_or_decimal_context_division"] == "forbidden")
    check("relative violation threshold identity", relative["threshold_bucket"] == RELATIVE_VIOLATION_THRESHOLD_BUCKET and relative["deadband_buckets"] == 2 and relative["eligibility_rule"] == "relative_violation_bucket_strictly_greater_than_100000002")
    check("relative violation empty policies", relative["empty_set_status"] == "no_certified_violating_candidates" and relative["zero_denominator_status"] == "no_positive_quantized_normalized_violation" and relative["empty_or_zero_denominator_robust_certification"] == "forbidden")
    check("relative eligibility before similarity", relative["similarity_and_diversity_filtering"] == "after_relative_eligibility_without_denominator_recomputation")
    check("stable tie break", candidate["adaptive_multicut"]["selection_order"] == [
        "quantized_normalized_violation_descending", "quantized_diversity_descending",
        "pattern_sha256_ascending", "cut_sha256_ascending",
    ])
    expected_schedule = [
        {"condition": "final_certification", "maximum_batch_size": 1},
        {"if_total_committed_unique_master_cuts_lt": 1000, "maximum_batch_size": 5},
        {"if_total_committed_unique_master_cuts_lt": 3000, "maximum_batch_size": 3},
        {"if_total_committed_unique_master_cuts_lt": 5000, "maximum_batch_size": 2},
        {"otherwise": True, "maximum_batch_size": 1},
    ]
    multicut = candidate["adaptive_multicut"]
    check("discrete batch schedule", multicut["adaptive_batch_schedule"] == expected_schedule)
    check("start-of-iteration committed count", multicut["schedule_count_field"] == "total_committed_unique_master_cuts" and multicut["schedule_count_timing"] == "start_of_iteration_before_selection")
    check("eligible min batch formula", multicut["selected_count_formula"] == "A_k=min(E_k,B(C_k))")
    check("committed count exclusions", set(multicut["committed_count_excludes"]) == {
        "certified_but_not_selected", "selected_but_not_committed", "duplicate",
        "numerical_redundancy_rejection", "invalid_or_uncertified", "pool_candidate", "cache_candidate",
    })
    check("runtime branching forbidden", multicut["runtime_fields_role"] == "reporting_only_never_scientific_branching" and not any("runtime" in json.dumps(row).lower() for row in multicut["adaptive_batch_schedule"]))
    required_checkpoint_state = {
        "iteration", "total_committed_unique_master_cuts", "final_certification", "current_batch_schedule_segment",
        "quantized_lb_improvement_state", "stall_counter", "pattern_sha256_values",
        "candidate_ordering", "duplicate_and_redundancy_decisions", "selected_cut_sha256_values",
        "committed_master_cut_sha256_values", "committed_cut_count_before_iteration",
        "committed_cut_count_after_iteration", "cut_commit_state", "canonical_cut_payloads_by_sha256",
        "pattern_schema_version", "cut_schema_version", "quantization_schema", "quantization_identity",
        "relative_violation_schema", "relative_violation_quantum",
        "relative_violation_threshold_bucket", "relative_violation_deadband_buckets",
        "relative_violation_denominator_rule", "relative_violation_denominator_bucket",
        "relative_violation_candidate_evidence", "relative_violation_status",
        "relative_violation_eligible_count",
    }
    check("checkpointed adaptive state", set(candidate["checkpointed_adaptive_state"]) == required_checkpoint_state)
    check("selected committed state machine", candidate["cut_commit_state_machine"]["states"] == [
        "no_selection", "selection_complete_not_committed", "commit_complete",
    ] and candidate["cut_commit_state_machine"]["committed_list"] == "append_only_unique_in_successful_commit_order")
    check("resume duplicate prevention", candidate["cut_commit_state_machine"]["resume_duplicate_prevention"] == "rebuild_from_persisted_committed_list_and_never_add_existing_sha")
    check("append-only atomic commit", candidate["cut_commit_state_machine"]["partial_commit_checkpoint"] == "forbidden" and candidate["cut_commit_state_machine"]["memory_add_without_commit_checkpoint"] == "volatile_and_scientifically_invisible")
    check("quantization failure states", set(numeric["fail_closed_states"]) == {
        "nonfinite_numeric_identity", "invalid_quantum", "invalid_binary64_input", "quantization_schema_mismatch",
    })
    check("binary64 extremes", quantized_bucket(float.fromhex("0x1.fffffffffffffp+1023"), Decimal("1e-9")) > 0 and quantized_bucket(float.fromhex("0x0.0000000000001p-1022"), Decimal("1e-9")) == 0)
    check("relative integer ratio reference", round_half_even_nonnegative_ratio(100_000_002, 1) == 100_000_002 and relative_violation_bucket_is_eligible(100_000_003) and not relative_violation_bucket_is_eligible(100_000_002))
    source_text = (root / "src/fairness_large_final_remediation_audit.py").read_text(encoding="utf-8")
    check("integer ratio implementation", ".as_integer_ratio()" in source_text and "divmod(scaled, divisor)" in source_text)
    check("Decimal context forbidden", "Decimal." + "from_float" not in source_text and "local" + "context" not in source_text and ".quan" + "tize(" not in source_text)
    check("relative division is integer-only", "divmod(numerator, denominator)" in source_text and "def relative_violation_" + "is_eligible" not in source_text)
    check("canonical reference behavior", _canonical_float_hex(-0.0) == _canonical_float_hex(0.0) and adaptive_batch_size(999) == 5 and adaptive_batch_size(1000) == 3 and adaptive_batch_size(5000) == 1)
    test_text = (root / "tests/test_fairness_large_final_remediation_protocol.py").read_text(encoding="utf-8")
    check("canonical boundary tests present", all(name in test_text for name in (
        "test_pattern_hash_canonicalization_and_validation", "test_cut_hash_canonicalization_and_binary64_validation",
        "test_numeric_nextafter_deadbands_and_half_even", "test_batch_schedule_boundaries_and_runtime_independence",
        "test_resume_selection_state_is_discrete_and_reproducible", "test_binary64_extremes_use_exact_integer_arithmetic",
        "test_selected_committed_interrupt_recovery_state_machine", "test_batch_count_uses_only_start_committed_unique_cuts",
        "test_relative_violation_single_multiple_and_order_independence",
        "test_relative_violation_empty_zero_duplicate_and_uncertified",
        "test_relative_violation_integer_boundaries_and_half_even",
        "test_relative_violation_resume_identity_and_similarity_independence",
        "test_relative_violation_config_identity_sha_locks",
    )))

    all_rows: dict[str, list[dict[str, Any]]] = {}
    longest = Path()
    longest_length = -1
    outputs_absent = True
    for rel in CONFIGS:
        path = root / rel
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        rows = expand_plan(config)
        all_rows[config["stage"]] = rows
        if config["stage"] == "L0":
            check("L0 formal authorization", config["authorization"] == "formal_execution_authorized" and config["formal_run_authorized"] is True)
        else:
            check(f"{config['stage']} remains unauthorized", config["authorization"] == "protocol_only_no_formal_execution" and config["formal_run_authorized"] is False)
        check(f"{config['stage']} identity", config["base_commit"] == BASE and config["schema_version"] == 3 and config["execution_attempt"] == 3 and config["previous_attempt_results_reused"] is False)
        check(f"{config['stage']} unique plan", len({r["run_key"] for r in rows}) == len(rows))
        check(f"{config['stage']} arithmetic", len(rows) == config["total_tasks"] and sum(r["task_type"] == "baseline" for r in rows) == config["baseline_count"] and sum(r["task_type"] == "frontier" for r in rows) == config["frontier_count"])
        check(f"{config['stage']} solver identity", config["solver_identity"] == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7})
        check(f"{config['stage']} time identity", config["algorithm_time_limit_seconds"] == config["baseline_time_limit_seconds"] == config["general_time_limit_seconds"] == 1800)
        check(f"{config['stage']} resume no overwrite", config["resume"] is True and config["overwrite_supported"] is False)
        check(f"{config['stage']} adaptive identity lock", {
            "pattern_schema", "cut_schema", "canonical_json_rule", "float_encoding_rule",
            "quantization_parameters", "quantization_schema", "adaptive_batch_schedule",
            "committed_cut_state_machine", "checkpoint_selection_state", "relative_violation_identity",
        }.issubset(set(config["manifest_identity_lock"])))
        check(f"{config['stage']} erratum SHA lock", config["required_protocol_sha256"] == sha256(root / "docs/fairness_large_final_remediation_protocol.md") and config["required_candidate_sha256"] == sha256(candidate_path))
        output = root / config["output_dir"]
        outputs_absent = outputs_absent and not output.exists()
        for item in _candidate_paths(root, config, rows):
            length = len(str(item.resolve()))
            if length > longest_length:
                longest, longest_length = item.resolve(), length
    check("L0 plan", len(all_rows["L0"]) == 2)
    check("L1 cumulative plan", len(all_rows["L1"]) == 9)
    l0_keys = {row["run_key"] for row in all_rows["L0"]}
    l1_keys = {row["run_key"] for row in all_rows["L1"]}
    check("L0 identities are an L1 subset", l0_keys < l1_keys)
    check("L1 adds seven tasks after L0", len(l1_keys - l0_keys) == 7)
    check("M1 plan", len(all_rows["M1"]) == 9)
    check("cross-scale maximum", len(all_rows["L1"]) + len(all_rows["M1"]) == 18)
    accessed_seeds = {row["seed"] for rows in all_rows.values() for row in rows}
    check("prohibited seeds untouched", not accessed_seeds.intersection(range(130, 160)))
    check("reserved holdout seeds untouched", not accessed_seeds.intersection(range(170, 180)))
    check("output isolation", outputs_absent)
    check("Windows path portability", longest_length <= 220, {"path": str(longest), "length": longest_length})

    envelopes = {
        "L0": 2 * 1800 + 1 * 4657 * 30,
        "L1": 9 * 1800 + 6 * 4657 * 30,
        "M1": 9 * 1800 + 6 * 1831 * 30,
    }
    result = {
        "decision": "approve_for_implementation" if all(c["passed"] for c in checks) else "mathematical_or_protocol_blocker",
        "next_authorized_stage": "fairness_large_final_remediation_implementation_only" if all(c["passed"] for c in checks) else "stop_and_report",
        "formal_run_authorized": {"L0": True, "L1": False, "M1": False},
        "initial_T1_UB_proved": True,
        "checks_passed": sum(c["passed"] for c in checks),
        "checks_total": len(checks),
        "checks": checks,
        "dry_run": {
            "S0": [
                "single_region", "symmetric_regions", "clear_fairness_gap", "rho_0", "rho_0.01",
                "gamma_0", "gamma_2", "zero_demand_region", "infeasible_cost_budget",
                "valid_baseline_t1_ub", "invalid_t1_ub_counterexample", "cache_current_point_recertification",
                "multiple_independently_certified_cuts", "tie_breaking", "near_parallel_cuts",
                "adaptive_budget_5_to_1", "benders_extensive_form_equivalence",
            ],
            "L0_tasks": 2,
            "L1_cumulative_tasks": 9,
            "M1_tasks": 9,
            "cross_scale_cumulative_tasks": 18,
            "scenarios": {"medium_large": 1831, "large": 4657},
            "algorithm_time_envelope_seconds": {"L0": 3600, "L1": 16200, "M1": 16200},
            "post_evaluation_envelope_seconds": {"L0": 139710, "L1": 838260, "M1": 329580},
            "total_wall_time_upper_accounting_seconds": envelopes,
            "longest_windows_path": str(longest),
            "longest_windows_path_length": longest_length,
            "instances_generated": False,
            "solver_called": False,
            "output_dir_exists": not outputs_absent,
        },
        "artifact_sha256": {
            "protocol": sha256(root / "docs/fairness_large_final_remediation_protocol.md"),
            "initial_upper_bound_proof": sha256(root / "docs/robust_regional_fairness_initial_upper_bound.md"),
            "candidate": sha256(candidate_path),
            **{Path(rel).name: sha256(root / rel) for rel in CONFIGS},
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = audit(args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["decision"] == "approve_for_implementation" else 1


if __name__ == "__main__":
    raise SystemExit(main())
