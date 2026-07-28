import hashlib
from pathlib import Path

import yaml

from src.fairness_large_final_remediation_audit import CANDIDATE, CONFIGS, audit, expand_plan


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
