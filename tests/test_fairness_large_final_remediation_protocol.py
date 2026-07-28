import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from src.fairness_large_final_remediation_audit import (
    CANDIDATE,
    CONFIGS,
    CUT_SCHEMA,
    PATTERN_SCHEMA,
    adaptive_batch_segment,
    adaptive_batch_size,
    audit,
    candidate_order_key,
    canonical_cut_bytes,
    canonical_json_bytes,
    canonical_pattern_bytes,
    checkpoint_selection_signature,
    cosine_is_redundant,
    cut_sha256,
    expand_plan,
    pattern_sha256,
    quantized_bucket,
    raw_violation_is_accepted,
    relative_violation_is_eligible,
)


ROOT = Path(__file__).resolve().parents[1]


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


def test_l0_is_a_stable_identity_subset_of_cumulative_l1():
    l0, l1 = (load(rel) for rel in CONFIGS[:2])
    l0_keys = {row["run_key"] for row in expand_plan(l0)}
    l1_keys = {row["run_key"] for row in expand_plan(l1)}
    assert l0_keys < l1_keys
    assert len(l1_keys - l0_keys) == 7
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
    relative = 0.10
    assert not relative_violation_is_eligible(math.nextafter(relative, -math.inf))
    assert not relative_violation_is_eligible(relative)
    assert not relative_violation_is_eligible(math.nextafter(relative, math.inf))
    assert not relative_violation_is_eligible(relative + 2.0e-9)
    assert relative_violation_is_eligible(relative + 3.1e-9)
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
        with pytest.raises(ValueError):
            quantized_bucket(bad, Decimal("1e-9"))


def test_quantized_tie_break_is_hash_stable():
    common = {"normalized_violation": 0.2, "raw_violation": 0.3, "diversity": 0.5}
    first = candidate_order_key(**common, pattern_hash="A" * 64, cut_hash="F" * 64)
    second = candidate_order_key(**common, pattern_hash="B" * 64, cut_hash="0" * 64)
    assert first < second
    same_pattern_first = candidate_order_key(**common, pattern_hash="A" * 64, cut_hash="0" * 64)
    assert same_pattern_first < first
    same_bucket = candidate_order_key(
        normalized_violation=0.2 + math.ulp(0.2), raw_violation=0.3,
        diversity=0.5, pattern_hash="A" * 64, cut_hash="F" * 64,
    )
    assert same_bucket == first


def test_batch_schedule_boundaries_and_runtime_independence():
    expected = {999: 5, 1000: 3, 1001: 3, 2999: 3, 3000: 2, 3001: 2, 4999: 2, 5000: 1, 5001: 1}
    for cuts, size in expected.items():
        assert adaptive_batch_size(cuts) == size
        assert adaptive_batch_size(cuts, reporting_runtime={"master": math.ulp(1.0), "cpu": "A"}) == size
        assert adaptive_batch_size(cuts, reporting_runtime={"master": 1e300, "cpu": "B"}) == size
    assert adaptive_batch_size(0, final_certification=True) == 1


def test_resume_selection_state_is_discrete_and_reproducible():
    state = {
        "iteration": 17,
        "total_certified_cuts": 1000,
        "final_certification": False,
        "current_batch_schedule_segment": adaptive_batch_segment(1000),
        "quantized_lb_improvement_state": 12,
        "stall_counter": 3,
        "pattern_sha256_values": ["A" * 64],
        "cut_sha256_values": ["B" * 64],
        "candidate_ordering": [[12, 8, 7, "A" * 64, "B" * 64]],
        "duplicate_and_redundancy_decisions": [{"cut": "B" * 64, "decision": "selected"}],
        "pattern_schema_version": PATTERN_SCHEMA,
        "cut_schema_version": CUT_SCHEMA,
        "quantization_schema": "decimal_bucket_round_half_even_v1",
    }
    before = checkpoint_selection_signature({**state, "runtime": {"master": 1.0}, "machine": "A"})
    after = checkpoint_selection_signature({**state, "runtime": {"master": 999.0}, "machine": "B"})
    assert before == after
    assert adaptive_batch_size(state["total_certified_cuts"], reporting_runtime={"cpu": "A"}) == 3
    assert adaptive_batch_size(state["total_certified_cuts"], reporting_runtime={"cpu": "B"}) == 3
    with pytest.raises(ValueError):
        checkpoint_selection_signature({**state, "current_batch_schedule_segment": "wrong"})
