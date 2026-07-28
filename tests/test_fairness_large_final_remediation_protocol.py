import hashlib
import json
import math
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
import yaml

from src.fairness_large_final_remediation_audit import (
    CANDIDATE,
    CONFIGS,
    CUT_SCHEMA,
    QUANTIZATION_SCHEMA,
    CheckpointWriteFailure,
    CommitInterrupted,
    QuantizationFailClosed,
    RelativeViolationFailClosed,
    PATTERN_SCHEMA,
    adaptive_batch_segment,
    adaptive_batch_size,
    audit,
    candidate_order_key,
    canonical_cut_bytes,
    canonical_json_bytes,
    canonical_pattern_bytes,
    checkpoint_selection_signature,
    commit_selected_checkpoint,
    cosine_is_redundant,
    cut_sha256,
    expand_plan,
    pattern_sha256,
    quantized_bucket,
    raw_violation_is_accepted,
    relative_normalized_violation_evidence,
    relative_violation_bucket_is_eligible,
    round_half_even_nonnegative_ratio,
    make_selection_checkpoint,
    resume_action,
    selected_cut_count,
    validate_checkpoint_state,
)


ROOT = Path(__file__).resolve().parents[1]


def cut_payloads(count: int, *, offset: int = 0):
    result = {}
    for index in range(offset, offset + count):
        payload = json.loads(canonical_cut_bytes(
            ["y[0]", "T"], {"y[0]": float(index + 1), "T": -1.0},
            constant=0.0, rhs=float(index), sense=">=",
        ))
        digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()
        result[digest] = payload
    return result


def relative_evidence_for_hashes(hashes, buckets=None):
    bucket_values = list(buckets or [10**9] * len(hashes))
    return relative_normalized_violation_evidence([
        {
            "source": "primary_full_separation_incumbent",
            "certified_current_point": True,
            "strictly_violating": True,
            "canonical_cut_identity_valid": True,
            "cut_sha256": digest,
            "pattern_sha256": hashlib.sha256(f"pattern-{index}".encode()).hexdigest().upper(),
            "normalized_violation_bucket": bucket_values[index],
        }
        for index, digest in enumerate(hashes)
    ])


def load(rel: str):
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def test_solver_free_protocol_audit_passes_and_is_deterministic():
    first = audit(ROOT)
    second = audit(ROOT)
    assert first == second
    assert first["decision"] == "approve_for_implementation"
    assert first["checks_passed"] == first["checks_total"]
    assert first["dry_run"]["solver_called"] is False
    assert first["dry_run"]["instances_generated"] is False
    assert first["dry_run"]["output_dir_exists"] is False


def test_stage_arithmetic_identity_and_short_directories():
    expected = {"L0": (1, 1, 2), "L1": (3, 6, 9), "M1": (3, 6, 9)}
    for rel in CONFIGS:
        config = load(rel)
        rows = expand_plan(config)
        baseline, frontier, total = expected[config["stage"]]
        assert sum(r["task_type"] == "baseline" for r in rows) == baseline
        assert sum(r["task_type"] == "frontier" for r in rows) == frontier
        assert len(rows) == total
        assert len({r["run_key"] for r in rows}) == total
        for row in rows:
            expected_id = "r_" + hashlib.sha256(row["run_key"].encode("utf-8")).hexdigest()[:24]
            assert row["run_directory_id"] == expected_id


def test_only_one_candidate_and_certification_invariants():
    candidate = load("experiments/configs/fairness_large_final_remediation_candidate.yaml")
    assert candidate["candidate_name"] == CANDIDATE
    assert candidate["forbidden"] == {
        "cut_deletion": True,
        "old_ray_reuse": True,
        "old_cut_reuse": True,
        "old_violation_reuse": True,
        "old_certification_reuse": True,
    }
    assert candidate["certification"]["fixed_scenario_recertification_at_current_point"] is True
    assert candidate["certification"]["full_separation_objective_bound_required"] is True
    assert candidate["adaptive_multicut"]["minimum_cuts_when_a_certified_violation_exists"] == 1
    assert candidate["adaptive_multicut"]["maximum_certified_cuts_per_iteration"] == 5


def test_gate_state_machine_and_seed_isolation():
    l0, l1, m1 = (load(rel) for rel in CONFIGS)
    assert l0["gate_requires"] == "implementation_and_pre_run_audit_approved"
    assert l1["gate_requires"] == "L0_certified_robust_optimal_with_valid_post_evaluation"
    assert l1["minimum_certified_frontier"] == 4
    assert m1["gate_requires"] == "L1_certified_frontier_at_least_4_of_6"
    accessed = set(l0["seeds"] + l1["seeds"] + m1["seeds"])
    assert accessed == {160, 161, 162}
    assert not accessed.intersection(range(130, 160))
    assert not accessed.intersection(range(170, 180))


def test_l0_hotfix_attempt_is_identity_isolated_from_frozen_l1():
    l0, l1 = (load(rel) for rel in CONFIGS[:2])
    l0_keys = {row["run_key"] for row in expand_plan(l0)}
    l1_keys = {row["run_key"] for row in expand_plan(l1)}
    assert l0_keys.isdisjoint(l1_keys)
    assert len(l0_keys) == 2
    assert len(l1_keys) == 9
    assert l1["incremental_tasks_after_l0"] == 7
    assert l1["l0_evidence_referenced_read_only"] is True


def test_initial_upper_bound_document_freezes_fail_closed_semantics():
    text = (ROOT / "docs/robust_regional_fairness_initial_upper_bound.md").read_text(encoding="utf-8").lower()
    for phrase in ("u'[r,j] = min", "valid objective upper bound", "not a lower bound", "fail-closed", "nonnegative"):
        assert phrase in text


def test_production_solver_files_are_not_part_of_protocol_artifacts():
    allowed = {Path(rel).name for rel in CONFIGS} | {
        "fairness_large_final_remediation_candidate.yaml",
        "fairness_large_final_remediation_protocol.md",
        "robust_regional_fairness_initial_upper_bound.md",
        "fairness_large_final_remediation_audit.py",
        "test_fairness_large_final_remediation_protocol.py",
    }
    assert "fairness_benders.py" not in allowed
    assert "fairness_scalability.py" not in allowed
    assert "robust_regional_fairness.py" not in allowed


def test_pattern_hash_canonicalization_and_validation():
    regions = ["区域é", "r1"]
    products = ["j0", "品类β"]
    forward = {("区域é", "j0"): 0, ("区域é", "品类β"): 1, ("r1", "j0"): 1, ("r1", "品类β"): 0}
    reverse = dict(reversed(list(forward.items())))
    assert pattern_sha256(regions, products, forward) == pattern_sha256(regions, products, reverse)
    decomposed = {("区域e\u0301", "j0"): 0, ("区域e\u0301", "品类β"): 1, ("r1", "j0"): 1, ("r1", "品类β"): 0}
    assert pattern_sha256(["区域e\u0301", "r1"], products, decomposed) == pattern_sha256(regions, products, forward)
    encoded = canonical_pattern_bytes(regions, products, forward)
    assert b"\r" not in encoded and b"\n" not in encoded and b": " not in encoded
    assert json.loads(encoded)["component_order"] == [["区域é", "j0"], ["区域é", "品类β"], ["r1", "j0"], ["r1", "品类β"]]
    pretty_windows = json.dumps(json.loads(encoded), ensure_ascii=False, indent=2).replace("\n", "\r\n")
    assert canonical_json_bytes(json.loads(pretty_windows)) == encoded
    assert pattern_sha256(list(reversed(regions)), products, forward) != pattern_sha256(regions, products, forward)
    with pytest.raises(ValueError):
        pattern_sha256(regions, products, {**forward, ("extra", "j0"): 1})
    with pytest.raises(ValueError):
        pattern_sha256(regions, products, {**forward, ("区域é", "j0"): 2})
    with pytest.raises(ValueError):
        pattern_sha256(regions, products, {**forward, ("区域é", "j0"): True})


def test_cut_hash_canonicalization_and_binary64_validation():
    variables = ["y[0]", "x[0,0]", "T"]
    terms = [("y[0]", 0.125), ("x[0,0]", -0.0), ("T", -1.0)]
    identity = {"constant": 0.5, "rhs": 0.0, "sense": ">="}
    digest = cut_sha256(variables, terms, **identity)
    assert digest == cut_sha256(variables, list(reversed(terms)), **identity)
    positive_zero = [("y[0]", 0.125), ("x[0,0]", 0.0), ("T", -1.0)]
    assert digest == cut_sha256(variables, positive_zero, **identity)
    payload = json.loads(canonical_cut_bytes(variables, terms, **identity))
    for _identifier, encoded in payload["terms"]:
        assert float.fromhex(encoded).hex() == encoded
    assert payload["constant"] == float.hex(0.5)
    assert digest != cut_sha256(["y[0]", "x[0,1]", "T"], [("y[0]", 0.125), ("x[0,1]", 0.0), ("T", -1.0)], **identity)
    assert digest != cut_sha256(variables, terms, constant=0.5, rhs=1.0, sense=">=")
    assert digest != cut_sha256(variables, terms, constant=0.5, rhs=0.0, sense="<=")
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            cut_sha256(variables, [("y[0]", bad), ("x[0,0]", 0.0), ("T", -1.0)], **identity)
        with pytest.raises(ValueError):
            cut_sha256(variables, terms, constant=bad, rhs=0.0, sense=">=")
        with pytest.raises(ValueError):
            cut_sha256(variables, terms, constant=0.0, rhs=bad, sense=">=")


def test_numeric_nextafter_deadbands_and_half_even():
    assert not relative_violation_bucket_is_eligible(100_000_000)
    assert not relative_violation_bucket_is_eligible(100_000_002)
    assert relative_violation_bucket_is_eligible(100_000_003)
    cosine = 0.98
    assert cosine_is_redundant(math.nextafter(cosine, -math.inf))
    assert cosine_is_redundant(cosine)
    assert cosine_is_redundant(math.nextafter(cosine, math.inf))
    assert cosine_is_redundant(cosine - 2.0e-9)
    assert not cosine_is_redundant(cosine - 3.1e-9)
    raw = 1.0e-7
    assert not raw_violation_is_accepted(math.nextafter(raw, -math.inf))
    assert not raw_violation_is_accepted(raw)
    assert not raw_violation_is_accepted(math.nextafter(raw, math.inf))
    assert raw_violation_is_accepted(raw + 3.1e-9)
    assert quantized_bucket(0.5, Decimal("1")) == 0
    assert quantized_bucket(1.5, Decimal("1")) == 2
    assert quantized_bucket(-0.0, Decimal("1e-9")) == quantized_bucket(0.0, Decimal("1e-9")) == 0
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(QuantizationFailClosed, match="nonfinite_numeric_identity"):
            quantized_bucket(bad, Decimal("1e-9"))


def _integer_reference_bucket(value: float, quantum_numerator: int = 1, quantum_denominator: int = 10**9) -> int:
    numerator, denominator = value.as_integer_ratio()
    quotient, remainder = divmod(abs(numerator) * quantum_denominator, denominator * quantum_numerator)
    divisor = denominator * quantum_numerator
    if 2 * remainder > divisor or (2 * remainder == divisor and quotient % 2):
        quotient += 1
    return -quotient if numerator < 0 else quotient


def test_binary64_extremes_use_exact_integer_arithmetic():
    smallest_subnormal = math.nextafter(0.0, 1.0)
    values = [sys.float_info.max, -sys.float_info.max, sys.float_info.min, smallest_subnormal, 0.0, -0.0, 1.5e-9, -1.5e-9]
    for value in values:
        assert quantized_bucket(value, Decimal("1e-9")) == _integer_reference_bucket(value)
    assert quantized_bucket(0.5, Decimal("1")) == 0
    assert quantized_bucket(1.5, Decimal("1")) == 2
    assert quantized_bucket(-0.5, Decimal("1")) == 0
    assert quantized_bucket(-1.5, Decimal("1")) == -2
    with localcontext() as context:
        context.prec = 1
        low_precision = quantized_bucket(sys.float_info.max, Decimal("1e-9"))
    with localcontext() as context:
        context.prec = 999
        high_precision = quantized_bucket(sys.float_info.max, Decimal("1e-9"))
    assert low_precision == high_precision
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(QuantizationFailClosed) as error:
            quantized_bucket(value, Decimal("1e-9"))
        assert error.value.status == "nonfinite_numeric_identity"
    for quantum in (Decimal("0"), Decimal("-1"), Decimal("NaN"), 1e-9):
        with pytest.raises(QuantizationFailClosed) as error:
            quantized_bucket(1.0, quantum)
        assert error.value.status == "invalid_quantum"
    with pytest.raises(QuantizationFailClosed) as error:
        quantized_bucket(object(), Decimal("1e-9"))
    assert error.value.status == "invalid_binary64_input"


def test_quantized_tie_break_is_hash_stable():
    common = {"normalized_violation": 0.2, "diversity": 0.5}
    first = candidate_order_key(**common, pattern_hash="A" * 64, cut_hash="F" * 64)
    second = candidate_order_key(**common, pattern_hash="B" * 64, cut_hash="0" * 64)
    assert first < second
    same_pattern_first = candidate_order_key(**common, pattern_hash="A" * 64, cut_hash="0" * 64)
    assert same_pattern_first < first
    same_bucket = candidate_order_key(
        normalized_violation=0.2 + math.ulp(0.2),
        diversity=0.5, pattern_hash="A" * 64, cut_hash="F" * 64,
    )
    assert same_bucket == first


def _relative_candidate(cut_digit, pattern_digit, bucket, *, source="solution_pool", certified=True, violating=True):
    return {
        "source": source,
        "certified_current_point": certified,
        "strictly_violating": violating,
        "canonical_cut_identity_valid": True,
        "cut_sha256": cut_digit * 64,
        "pattern_sha256": pattern_digit * 64,
        "normalized_violation_bucket": bucket,
    }


def test_relative_violation_single_multiple_and_order_independence():
    single = relative_normalized_violation_evidence([_relative_candidate("A", "1", 7)])
    assert single["relative_violation_denominator_bucket"] == 7
    assert single["relative_violation_candidate_evidence"][0]["relative_violation_bucket"] == 10**9
    assert single["relative_violation_eligible_count"] == 1
    candidates = [
        _relative_candidate("A", "1", 10),
        _relative_candidate("B", "2", 5, source="pattern_cache"),
        _relative_candidate("C", "3", 1, source="primary_full_separation_incumbent"),
    ]
    forward = relative_normalized_violation_evidence(candidates)
    reverse = relative_normalized_violation_evidence(list(reversed(candidates)))
    assert forward == reverse
    assert forward["relative_violation_denominator_bucket"] == 10
    assert [item["relative_violation_bucket"] for item in forward["relative_violation_candidate_evidence"]] == [10**9, 500_000_000, 100_000_000]
    assert forward["relative_violation_eligible_count"] == 2


def test_relative_violation_empty_zero_duplicate_and_uncertified():
    empty = relative_normalized_violation_evidence([])
    assert empty["relative_violation_status"] == "no_certified_violating_candidates"
    assert empty["relative_violation_denominator_bucket"] is None
    assert empty["relative_violation_eligible_count"] == 0
    zero = relative_normalized_violation_evidence([_relative_candidate("A", "1", 0)])
    assert zero["relative_violation_status"] == "no_positive_quantized_normalized_violation"
    assert zero["relative_violation_denominator_bucket"] == 0
    assert zero["relative_violation_candidate_evidence"][0]["relative_violation_bucket"] is None
    duplicate = _relative_candidate("B", "2", 11, source="pattern_cache")
    duplicate_pool = {**duplicate, "source": "solution_pool"}
    uncertified = _relative_candidate("C", "3", 10**100, certified=False)
    evidence = relative_normalized_violation_evidence([duplicate, uncertified, duplicate_pool])
    assert evidence["relative_violation_denominator_bucket"] == 11
    assert len(evidence["relative_violation_candidate_evidence"]) == 1
    assert evidence["relative_violation_candidate_evidence"][0]["sources"] == ["pattern_cache", "solution_pool"]
    with pytest.raises(RelativeViolationFailClosed, match="relative_violation_identity_mismatch"):
        relative_normalized_violation_evidence([duplicate, {**duplicate_pool, "normalized_violation_bucket": 12}])


def test_relative_violation_integer_boundaries_and_half_even():
    assert round_half_even_nonnegative_ratio(1, 2) == 0
    assert round_half_even_nonnegative_ratio(3, 2) == 2
    assert round_half_even_nonnegative_ratio(5, 2) == 2
    assert round_half_even_nonnegative_ratio(7, 2) == 4
    huge = 10**500 + 1
    assert round_half_even_nonnegative_ratio(huge * 10**200, 10**200) == huge
    candidates = [
        _relative_candidate("A", "1", 10**9),
        _relative_candidate("B", "2", 100_000_000),
        _relative_candidate("C", "3", 100_000_002),
        _relative_candidate("D", "4", 100_000_003),
    ]
    evidence = relative_normalized_violation_evidence(candidates)
    by_hash = {item["cut_sha256"]: item for item in evidence["relative_violation_candidate_evidence"]}
    assert by_hash["B" * 64]["relative_violation_bucket"] == 100_000_000
    assert by_hash["C" * 64]["relative_violation_bucket"] == 100_000_002
    assert by_hash["D" * 64]["relative_violation_bucket"] == 100_000_003
    assert not relative_violation_bucket_is_eligible(100_000_000)
    assert not relative_violation_bucket_is_eligible(100_000_002)
    assert relative_violation_bucket_is_eligible(100_000_003)
    denominator_one = relative_normalized_violation_evidence([_relative_candidate("E", "5", 1)])
    assert denominator_one["relative_violation_candidate_evidence"][0]["relative_violation_bucket"] == 10**9


def test_relative_violation_resume_identity_and_similarity_independence():
    payloads = cut_payloads(3, offset=3000)
    hashes = list(payloads)
    candidates = [
        _relative_candidate(digest[0], str(index + 1), bucket)
        for index, (digest, bucket) in enumerate(zip(hashes, (10**9, 500_000_000, 100_000_003)))
    ]
    # Preserve the actual canonical cut hashes rather than the helper's repeated digit.
    for candidate, digest in zip(candidates, hashes):
        candidate["cut_sha256"] = digest
    relative = relative_normalized_violation_evidence(candidates)
    eligible = [item["cut_sha256"] for item in relative["relative_violation_candidate_evidence"] if item["relative_violation_eligible"]]
    similarity_filtered = eligible[::2]
    state = make_selection_checkpoint(
        iteration=22, committed_hashes=[], eligible_ordered_hashes=similarity_filtered,
        canonical_cut_payloads_by_sha256=payloads,
        relative_violation_evidence=relative,
    )
    assert state["relative_violation_denominator_bucket"] == 10**9
    assert checkpoint_selection_signature(state) == checkpoint_selection_signature({**state, "runtime": 999})
    assert checkpoint_selection_signature(state) == checkpoint_selection_signature(state.copy())
    corrupted = json.loads(json.dumps(state))
    corrupted["relative_violation_candidate_evidence"][0]["relative_violation_bucket"] += 1
    with pytest.raises(RelativeViolationFailClosed, match="relative_violation_identity_mismatch"):
        validate_checkpoint_state(corrupted)
    with pytest.raises(RelativeViolationFailClosed, match="relative_violation_identity_mismatch"):
        validate_checkpoint_state({**state, "relative_violation_schema": "drift"})


def test_relative_violation_config_identity_sha_locks():
    protocol_hash = hashlib.sha256((ROOT / "docs/fairness_large_final_remediation_protocol.md").read_bytes()).hexdigest().upper()
    candidate_hash = hashlib.sha256((ROOT / "experiments/configs/fairness_large_final_remediation_candidate.yaml").read_bytes()).hexdigest().upper()
    for rel in CONFIGS:
        config = load(rel)
        assert config["required_protocol_sha256"] == protocol_hash
        assert config["required_candidate_sha256"] == candidate_hash
        assert "relative_violation_identity" in config["manifest_identity_lock"]
    with pytest.raises(RelativeViolationFailClosed, match="invalid_normalized_violation_bucket"):
        relative_normalized_violation_evidence([_relative_candidate("A", "1", math.nan)])


def test_batch_schedule_boundaries_and_runtime_independence():
    expected = {999: 5, 1000: 3, 1001: 3, 2999: 3, 3000: 2, 3001: 2, 4999: 2, 5000: 1, 5001: 1}
    for cuts, size in expected.items():
        assert adaptive_batch_size(cuts) == size
        assert adaptive_batch_size(cuts, reporting_runtime={"master": math.ulp(1.0), "cpu": "A"}) == size
        assert adaptive_batch_size(cuts, reporting_runtime={"master": 1e300, "cpu": "B"}) == size
    assert adaptive_batch_size(0, final_certification=True) == 1


def test_batch_count_uses_only_start_committed_unique_cuts():
    for committed_count, expected_batch in ((999, 5), (1000, 3), (2999, 3), (3000, 2), (4999, 2), (5000, 1)):
        assert adaptive_batch_size(committed_count) == expected_batch
        for eligible in (0, 1, max(0, expected_batch - 1), expected_batch, expected_batch + 1):
            assert selected_cut_count(eligible, committed_count) == min(eligible, expected_batch)
    assert selected_cut_count(99, 0, final_certification=True) == 1

    committed_payloads = cut_payloads(999)
    eligible_payload = cut_payloads(1, offset=2000)
    state = make_selection_checkpoint(
        iteration=5, committed_hashes=list(committed_payloads), eligible_ordered_hashes=list(eligible_payload),
        canonical_cut_payloads_by_sha256={**committed_payloads, **eligible_payload},
        duplicate_and_redundancy_decisions=[
            {"source": "certified_not_selected", "counted": False}, {"source": "duplicate", "counted": False},
            {"source": "redundancy_rejection", "counted": False}, {"source": "pool_candidate", "counted": False},
            {"source": "cache_candidate", "counted": False},
        ],
        relative_violation_evidence=relative_evidence_for_hashes(list(eligible_payload)),
    )
    assert state["total_committed_unique_master_cuts"] == 999
    assert adaptive_batch_size(state["committed_cut_count_before_iteration"]) == 5
    committed = commit_selected_checkpoint(state)
    assert committed["total_committed_unique_master_cuts"] == 1000
    assert adaptive_batch_size(committed["total_committed_unique_master_cuts"]) == 3

    empty = make_selection_checkpoint(
        iteration=6, committed_hashes=list(committed_payloads), eligible_ordered_hashes=[],
        canonical_cut_payloads_by_sha256=committed_payloads,
        relative_violation_evidence=relative_normalized_violation_evidence([]),
    )
    assert empty["cut_commit_state"] == "no_selection"
    assert empty["selected_cut_sha256_values"] == []
    assert empty["committed_cut_count_before_iteration"] == empty["committed_cut_count_after_iteration"] == 999


def test_resume_selection_state_is_discrete_and_reproducible():
    payloads = cut_payloads(4)
    state = make_selection_checkpoint(
        iteration=17, committed_hashes=list(payloads)[:1], eligible_ordered_hashes=list(payloads)[1:],
        canonical_cut_payloads_by_sha256=payloads, pattern_hashes=["A" * 64],
        quantized_lb_improvement_state=12, stall_counter=3,
        relative_violation_evidence=relative_evidence_for_hashes(list(payloads)[1:]),
    )
    before = checkpoint_selection_signature({**state, "runtime": {"master": 1.0}, "machine": "A"})
    after = checkpoint_selection_signature({**state, "runtime": {"master": 999.0}, "machine": "B"})
    assert before == after
    assert adaptive_batch_size(state["total_committed_unique_master_cuts"], reporting_runtime={"cpu": "A"}) == 5
    assert adaptive_batch_size(state["total_committed_unique_master_cuts"], reporting_runtime={"cpu": "B"}) == 5
    with pytest.raises(ValueError):
        checkpoint_selection_signature({**state, "current_batch_schedule_segment": "wrong"})
    with pytest.raises(QuantizationFailClosed) as error:
        checkpoint_selection_signature({**state, "quantization_schema": "drift"})
    assert error.value.status == "quantization_schema_mismatch"


def test_selected_committed_interrupt_recovery_state_machine():
    payloads = cut_payloads(5)
    hashes = list(payloads)
    selection = make_selection_checkpoint(
        iteration=9, committed_hashes=hashes[:2], eligible_ordered_hashes=hashes[2:],
        canonical_cut_payloads_by_sha256=payloads,
        relative_violation_evidence=relative_evidence_for_hashes(hashes[2:]),
    )
    assert selection["cut_commit_state"] == "selection_complete_not_committed"
    assert resume_action(selection) == "rebuild_committed_master_then_recommit_same_selected_hashes_once"
    selection_signature = checkpoint_selection_signature(selection)

    # An interruption before selection simply reruns the same deterministic selection.
    rerun = make_selection_checkpoint(
        iteration=9, committed_hashes=hashes[:2], eligible_ordered_hashes=hashes[2:],
        canonical_cut_payloads_by_sha256=payloads,
        relative_violation_evidence=relative_evidence_for_hashes(hashes[2:]),
    )
    assert checkpoint_selection_signature(rerun) == selection_signature

    # Partial and complete volatile adds are invisible until the atomic commit checkpoint exists.
    for point in (1, len(selection["selected_cut_sha256_values"])):
        with pytest.raises(CommitInterrupted):
            commit_selected_checkpoint(selection, interrupt_after_memory_adds=point)
        assert checkpoint_selection_signature(selection) == selection_signature
    with pytest.raises(CheckpointWriteFailure):
        commit_selected_checkpoint(selection, checkpoint_write_success=False)
    assert checkpoint_selection_signature(selection) == selection_signature

    committed = commit_selected_checkpoint(selection)
    assert committed["cut_commit_state"] == "commit_complete"
    assert committed["committed_master_cut_sha256_values"] == hashes
    assert committed["committed_cut_count_before_iteration"] == 2
    assert committed["committed_cut_count_after_iteration"] == 5
    assert resume_action(committed) == "rebuild_committed_master_then_advance_to_next_iteration"
    assert commit_selected_checkpoint(selection) == committed
    with pytest.raises(ValueError):
        commit_selected_checkpoint(committed)

    corruptions = [
        {**selection, "committed_master_cut_sha256_values": hashes[:2] + [hashes[0]]},
        {**selection, "committed_cut_count_before_iteration": 99},
        {**selection, "cut_commit_state": "partially_committed"},
        {**committed, "committed_master_cut_sha256_values": hashes + [hashes[-1]]},
    ]
    for corrupted in corruptions:
        with pytest.raises(ValueError):
            validate_checkpoint_state(corrupted)
