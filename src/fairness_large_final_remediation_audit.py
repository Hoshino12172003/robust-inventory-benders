"""Solver-free static audit and dry-run for the final fairness remediation protocol."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
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


def quantized_bucket(value: Any, quantum: Decimal) -> int:
    if not isinstance(quantum, Decimal) or not quantum.is_finite() or quantum <= 0:
        raise ValueError("Selection quantum must be a finite positive Decimal.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Selection metrics must be finite IEEE-754 binary64 values.")
    if number == 0.0:
        number = 0.0
    with localcontext() as context:
        context.prec = 80
        return int((Decimal.from_float(number) / quantum).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def raw_violation_is_accepted(value: Any) -> bool:
    bucket = quantized_bucket(value, RAW_VIOLATION_QUANTUM)
    threshold = quantized_bucket(float(RAW_VIOLATION_TOLERANCE), RAW_VIOLATION_QUANTUM)
    return bucket > threshold + THRESHOLD_DEADBAND_BUCKETS


def relative_violation_is_eligible(value: Any) -> bool:
    bucket = quantized_bucket(value, RELATIVE_VIOLATION_QUANTUM)
    threshold = quantized_bucket(float(RELATIVE_VIOLATION_THRESHOLD), RELATIVE_VIOLATION_QUANTUM)
    return bucket > threshold + THRESHOLD_DEADBAND_BUCKETS


def cosine_is_redundant(value: Any) -> bool:
    bucket = quantized_bucket(value, COSINE_SIMILARITY_QUANTUM)
    threshold = quantized_bucket(float(COSINE_REDUNDANCY_THRESHOLD), COSINE_SIMILARITY_QUANTUM)
    return bucket >= threshold - THRESHOLD_DEADBAND_BUCKETS


def candidate_order_key(
    *, normalized_violation: Any, raw_violation: Any, diversity: Any,
    pattern_hash: str, cut_hash: str,
) -> tuple[Any, ...]:
    return (
        -quantized_bucket(normalized_violation, NORMALIZED_VIOLATION_QUANTUM),
        -quantized_bucket(diversity, DIVERSITY_QUANTUM),
        -quantized_bucket(raw_violation, RAW_VIOLATION_QUANTUM),
        str(pattern_hash).upper(),
        str(cut_hash).upper(),
    )


def adaptive_batch_size(
    total_certified_cuts: int,
    *,
    final_certification: bool = False,
    reporting_runtime: dict[str, Any] | None = None,
) -> int:
    del reporting_runtime
    if isinstance(total_certified_cuts, bool) or not isinstance(total_certified_cuts, int) or total_certified_cuts < 0:
        raise ValueError("total_certified_cuts must be a nonnegative integer.")
    if final_certification:
        return 1
    if total_certified_cuts < 1000:
        return 5
    if total_certified_cuts < 3000:
        return 3
    if total_certified_cuts < 5000:
        return 2
    return 1


def adaptive_batch_segment(total_certified_cuts: int, *, final_certification: bool = False) -> str:
    size = adaptive_batch_size(total_certified_cuts, final_certification=final_certification)
    if final_certification:
        return "final_certification_batch_1"
    if total_certified_cuts < 1000:
        return "cuts_0_999_batch_5"
    if total_certified_cuts < 3000:
        return "cuts_1000_2999_batch_3"
    if total_certified_cuts < 5000:
        return "cuts_3000_4999_batch_2"
    assert size == 1
    return "cuts_5000_plus_batch_1"


CHECKPOINT_SELECTION_FIELDS = (
    "iteration", "total_certified_cuts", "final_certification", "current_batch_schedule_segment",
    "quantized_lb_improvement_state", "stall_counter", "pattern_sha256_values", "cut_sha256_values",
    "candidate_ordering", "duplicate_and_redundancy_decisions", "pattern_schema_version",
    "cut_schema_version", "quantization_schema",
)


def checkpoint_selection_signature(state: dict[str, Any]) -> str:
    if any(field not in state for field in CHECKPOINT_SELECTION_FIELDS):
        raise ValueError("Checkpoint is missing frozen adaptive-selection state.")
    payload = {field: state[field] for field in CHECKPOINT_SELECTION_FIELDS}
    expected_segment = adaptive_batch_segment(
        payload["total_certified_cuts"], final_certification=payload["final_certification"]
    )
    if payload["current_batch_schedule_segment"] != expected_segment:
        raise ValueError("Checkpoint batch schedule segment is inconsistent with discrete state.")
    if payload["pattern_schema_version"] != PATTERN_SCHEMA or payload["cut_schema_version"] != CUT_SCHEMA:
        raise ValueError("Checkpoint canonicalization schema mismatch.")
    for field in ("iteration", "total_certified_cuts", "quantized_lb_improvement_state", "stall_counter"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Checkpoint {field} must be a nonnegative integer.")
    if not isinstance(payload["final_certification"], bool):
        raise ValueError("Checkpoint final_certification must be boolean.")
    for field in ("pattern_sha256_values", "cut_sha256_values"):
        if not isinstance(payload[field], list) or not all(re.fullmatch(r"[0-9A-F]{64}", str(value)) for value in payload[field]):
            raise ValueError(f"Checkpoint {field} must contain uppercase SHA256 values.")
    if payload["quantization_schema"] != "decimal_bucket_round_half_even_v1":
        raise ValueError("Checkpoint quantization schema mismatch.")
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()


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
    check("quantization", numeric["decimal_rounding_mode"] == "ROUND_HALF_EVEN" and all(float(numeric[name]) == 1e-9 for name in (
        "raw_violation_quantum", "normalized_violation_quantum", "relative_violation_quantum",
        "cosine_similarity_quantum", "diversity_quantum", "lb_improvement_quantum",
    )))
    check("deadband", numeric["threshold_deadband_buckets"] == 2 and numeric["violation_deadband_policy"] == "reject_or_do_not_promote" and numeric["cosine_deadband_policy"] == "redundant")
    check("stable tie break", candidate["adaptive_multicut"]["selection_order"] == [
        "quantized_normalized_violation_descending", "quantized_diversity_descending",
        "quantized_raw_violation_descending", "pattern_sha256_ascending", "cut_sha256_ascending",
    ])
    expected_schedule = [
        {"condition": "final_certification", "maximum_batch_size": 1},
        {"if_total_certified_cuts_lt": 1000, "maximum_batch_size": 5},
        {"if_total_certified_cuts_lt": 3000, "maximum_batch_size": 3},
        {"if_total_certified_cuts_lt": 5000, "maximum_batch_size": 2},
        {"otherwise": True, "maximum_batch_size": 1},
    ]
    multicut = candidate["adaptive_multicut"]
    check("discrete batch schedule", multicut["adaptive_batch_schedule"] == expected_schedule)
    check("runtime branching forbidden", multicut["runtime_fields_role"] == "reporting_only_never_scientific_branching" and not any("runtime" in json.dumps(row).lower() for row in multicut["adaptive_batch_schedule"]))
    required_checkpoint_state = {
        "iteration", "total_certified_cuts", "final_certification", "current_batch_schedule_segment",
        "quantized_lb_improvement_state", "stall_counter", "pattern_sha256_values",
        "cut_sha256_values", "candidate_ordering", "duplicate_and_redundancy_decisions",
        "pattern_schema_version", "cut_schema_version", "quantization_schema",
    }
    check("checkpointed adaptive state", set(candidate["checkpointed_adaptive_state"]) == required_checkpoint_state)
    check("canonical reference behavior", _canonical_float_hex(-0.0) == _canonical_float_hex(0.0) and adaptive_batch_size(999) == 5 and adaptive_batch_size(1000) == 3 and adaptive_batch_size(5000) == 1)
    test_text = (root / "tests/test_fairness_large_final_remediation_protocol.py").read_text(encoding="utf-8")
    check("canonical boundary tests present", all(name in test_text for name in (
        "test_pattern_hash_canonicalization_and_validation", "test_cut_hash_canonicalization_and_binary64_validation",
        "test_numeric_nextafter_deadbands_and_half_even", "test_batch_schedule_boundaries_and_runtime_independence",
        "test_resume_selection_state_is_discrete_and_reproducible",
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
        check(f"{config['stage']} protocol-only", config["authorization"] == "protocol_only_no_formal_execution" and config["formal_run_authorized"] is False)
        check(f"{config['stage']} identity", config["base_commit"] == BASE and config["schema_version"] == 3 and config["execution_attempt"] == 3 and config["previous_attempt_results_reused"] is False)
        check(f"{config['stage']} unique plan", len({r["run_key"] for r in rows}) == len(rows))
        check(f"{config['stage']} arithmetic", len(rows) == config["total_tasks"] and sum(r["task_type"] == "baseline" for r in rows) == config["baseline_count"] and sum(r["task_type"] == "frontier" for r in rows) == config["frontier_count"])
        check(f"{config['stage']} solver identity", config["solver_identity"] == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7})
        check(f"{config['stage']} time identity", config["algorithm_time_limit_seconds"] == config["baseline_time_limit_seconds"] == config["general_time_limit_seconds"] == 1800)
        check(f"{config['stage']} resume no overwrite", config["resume"] is True and config["overwrite_supported"] is False)
        check(f"{config['stage']} adaptive identity lock", {
            "pattern_schema", "cut_schema", "canonical_json_rule", "float_encoding_rule",
            "quantization_parameters", "adaptive_batch_schedule",
        }.issubset(set(config["manifest_identity_lock"])))
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
        "formal_run_authorized": False,
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
