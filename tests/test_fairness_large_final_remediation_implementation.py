from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from src.experiment_protocol import config_sha256
from src.fairness_large_final_remediation import (
    CANDIDATE,
    InitialUpperBoundAssumptionFailure,
    construct_initial_t1_upper_bound,
    solve_certified_adaptive_multicut_fair_benders,
)
from src.fairness_large_final_remediation_audit import (
    relative_normalized_violation_evidence,
)
from src.fairness_large_final_remediation_runner import (
    RECOVERABLE_PHASES,
    RemediationGateError,
    advance_recovery_ledger,
    dry_run_remediation,
    run_remediation_stage,
    validate_stage_gate,
)
from src.robust_regional_fairness import solve_fairness_extensive_form
from tests.test_fairness_benders_against_extensive_form import FROZEN_PRECISION
from tests.test_robust_regional_fairness import tiny_instance


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "L0": ROOT / "experiments/configs/fairness_large_final_remediation_pilot.yaml",
    "L1": ROOT / "experiments/configs/fairness_large_final_remediation_large_s1.yaml",
    "M1": ROOT / "experiments/configs/fairness_large_final_remediation_medium_large_s1.yaml",
}


def baseline_evidence(instance, *, gamma: int, upper: float | None = None):
    value = float(30.0 if gamma == 0 else 36.0) if upper is None else float(upper)
    inventory = 5.0 if gamma == 0 else 6.0
    record = {
        "run_key": f"manual-certified-baseline-gamma-{gamma}",
        "solved_to_tolerance": True,
        "scientific_status": "certified_robust_optimal",
        "result": {
            "status": "optimal",
            "valid_UB": True,
            "gap": 0.0,
            "upper_bound": value,
            "y_values": [1.0],
            "x_values": [[inventory for _ in instance.J]],
        },
    }
    anchor = {
        "source": "solve_result.upper_bound",
        "value": value,
        "value_hex": value.hex(),
        "baseline_run_key": record["run_key"],
        "base_git_commit": "hand-built-s0",
        "base_config_sha256": "A" * 64,
        "candidate_config_sha256": "B" * 64,
        "valid_UB": True,
        "baseline_status": "optimal",
        "baseline_final_gap": 0.0,
    }
    anchor["anchor_sha256"] = config_sha256(anchor)
    return record, anchor


def relative_candidate(digit: str, bucket: int, *, source="solution_pool", certified=True):
    return {
        "source": source,
        "certified_current_point": certified,
        "strictly_violating": True,
        "canonical_cut_identity_valid": True,
        "cut_sha256": digit * 64,
        "pattern_sha256": str((int(digit, 16) + 1) % 10) * 64,
        "normalized_violation_bucket": bucket,
    }


def test_initial_t1_upper_bound_has_complete_auditable_evidence():
    instance = tiny_instance()
    record, anchor = baseline_evidence(instance, gamma=0)
    initial = construct_initial_t1_upper_bound(
        instance, baseline_record=record, anchor=anchor, rho=0.0, tolerance=1e-7,
    )
    assert initial.value == initial.t_value == 1.0
    assert initial.x_values == [[5.0]]
    evidence = initial.evidence
    assert evidence["initial_robust_ub_valid"] is True
    assert evidence["provides_lower_bound"] is False
    assert evidence["provides_optimality_certificate"] is False
    assert evidence["provides_complete_separation_certificate"] is False
    assert all(evidence["initial_robust_ub_assumption_checks"].values())
    assert len(evidence["initial_robust_ub_x_sha256"]) == 64
    assert len(evidence["initial_robust_ub_y_sha256"]) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record, anchor, instance: record["result"].update(valid_UB=False),
        lambda record, anchor, instance: record["result"].update(status="time_limit"),
        lambda record, anchor, instance: record["result"].update(gap=1.0),
        lambda record, anchor, instance: anchor.update(value_hex="drift"),
        lambda record, anchor, instance: instance.shortage_penalty[0].__setitem__(0, -1.0),
    ],
)
def test_initial_t1_upper_bound_assumptions_fail_closed(mutation):
    instance = tiny_instance()
    record, anchor = baseline_evidence(instance, gamma=0)
    mutation(record, anchor, instance)
    with pytest.raises(InitialUpperBoundAssumptionFailure) as error:
        construct_initial_t1_upper_bound(
            instance, baseline_record=record, anchor=anchor, rho=0.0, tolerance=1e-7,
        )
    assert error.value.status == "initial_upper_bound_assumption_failure"


def test_relative_union_denominator_integer_formula_and_zero_policies():
    duplicate = relative_candidate("B", 5, source="pattern_cache")
    payload = [
        relative_candidate("A", 10, source="primary_full_separation_incumbent"),
        duplicate,
        {**duplicate, "source": "solution_pool"},
        relative_candidate("C", 1),
        relative_candidate("D", 10**300, certified=False),
    ]
    forward = relative_normalized_violation_evidence(payload)
    assert forward == relative_normalized_violation_evidence(list(reversed(payload)))
    assert forward["relative_violation_denominator_bucket"] == 10
    assert [row["relative_violation_bucket"] for row in forward["relative_violation_candidate_evidence"]] == [10**9, 500_000_000, 100_000_000]
    assert forward["relative_violation_eligible_count"] == 2
    assert relative_normalized_violation_evidence([])["relative_violation_status"] == "no_certified_violating_candidates"
    zero = relative_normalized_violation_evidence([relative_candidate("A", 0)])
    assert zero["relative_violation_status"] == "no_positive_quantized_normalized_violation"
    assert zero["relative_violation_eligible_count"] == 0


@pytest.mark.parametrize("gamma,rho,baseline", [(0, 0.0, 30.0), (0, 0.01, 30.0), (2, 0.0, 36.0), (2, 0.01, 36.0)])
def test_s0_adaptive_solver_matches_extensive_form(gamma, rho, baseline):
    instance = tiny_instance()
    record, anchor = baseline_evidence(instance, gamma=gamma, upper=baseline)
    extensive = solve_fairness_extensive_form(
        instance, baseline_cost=baseline, rho=rho, gamma=gamma,
        max_scenarios=10, mip_gap=0.0,
    )
    result = solve_certified_adaptive_multicut_fair_benders(
        instance,
        baseline_record=record,
        anchor=anchor,
        rho=rho,
        gamma=gamma,
        algorithm_config=FROZEN_PRECISION,
        max_iterations=100,
        time_limit=60.0,
        tol=1e-6,
    )
    assert extensive.status == "optimal"
    assert result.status == "optimal"
    assert result.objective_t == pytest.approx(extensive.objective_t, abs=2e-6)
    assert result.lower_bound <= extensive.objective_t + 2e-6
    assert result.metadata["candidate"] == CANDIDATE
    assert result.metadata["full_separation_objective_bound_required"] is True
    assert result.iteration_log[-1]["robust_feasibility_certified"] is True


def test_s0_zero_demand_region_and_initial_ub():
    instance = tiny_instance(zero_second_demand=True)
    record, anchor = baseline_evidence(instance, gamma=0, upper=25.0)
    result = solve_certified_adaptive_multicut_fair_benders(
        instance, baseline_record=record, anchor=anchor, rho=0.0, gamma=0,
        algorithm_config=FROZEN_PRECISION, max_iterations=100, time_limit=60.0, tol=1e-6,
    )
    assert result.status == "optimal"
    assert result.metadata["initial_robust_upper_bound"]["initial_robust_ub_assumption_checks"]["zero_demand_region_policy"] is True


def test_s0_selection_checkpoint_interrupt_resumes_to_clean_result(tmp_path):
    instance = tiny_instance()
    record, anchor = baseline_evidence(instance, gamma=0)
    clean = solve_certified_adaptive_multicut_fair_benders(
        instance, baseline_record=record, anchor=anchor, rho=0.01, gamma=0,
        algorithm_config=FROZEN_PRECISION, max_iterations=100, time_limit=60.0, tol=1e-6,
    )
    checkpoint = tmp_path / "algorithm_checkpoint.json"
    injected = {"raised": False}

    def interrupt(point, _payload):
        if point == "memory_cut_add" and not injected["raised"]:
            injected["raised"] = True
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        solve_certified_adaptive_multicut_fair_benders(
            instance, baseline_record=record, anchor=anchor, rho=0.01, gamma=0,
            algorithm_config=FROZEN_PRECISION, max_iterations=100, time_limit=60.0, tol=1e-6,
            checkpoint_path=checkpoint, checkpoint_identity={"run_key": "s0-resume"},
            failure_injector=interrupt,
        )
    persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert persisted["selection"]["cut_commit_state"] == "selection_complete_not_committed"
    resumed = solve_certified_adaptive_multicut_fair_benders(
        instance, baseline_record=record, anchor=anchor, rho=0.01, gamma=0,
        algorithm_config=FROZEN_PRECISION, max_iterations=100, time_limit=60.0, tol=1e-6,
        checkpoint_path=checkpoint, checkpoint_identity={"run_key": "s0-resume"},
    )
    assert resumed.status == clean.status == "optimal"
    assert resumed.objective_t == pytest.approx(clean.objective_t, abs=1e-8)
    assert resumed.lower_bound == pytest.approx(clean.lower_bound, abs=1e-8)
    assert resumed.cuts == clean.cuts
    final_checkpoint = json.loads(checkpoint.read_text(encoding="utf-8"))
    committed = final_checkpoint["selection"]["committed_master_cut_sha256_values"]
    assert len(committed) == len(set(committed))


@pytest.mark.parametrize(
    "fault_point",
    ["before_master", "after_separation", "before_selection_checkpoint",
     "after_selection_checkpoint", "before_commit_checkpoint", "after_commit_checkpoint"],
)
def test_s0_fault_points_resume_to_same_scientific_result(tmp_path, fault_point):
    instance = tiny_instance()
    record, anchor = baseline_evidence(instance, gamma=0)
    clean = solve_certified_adaptive_multicut_fair_benders(
        instance, baseline_record=record, anchor=anchor, rho=0.0, gamma=0,
        algorithm_config=FROZEN_PRECISION, max_iterations=100, time_limit=60.0, tol=1e-6,
    )
    checkpoint = tmp_path / f"{fault_point}.json"
    injected = {"raised": False}

    def interrupt(point, _payload):
        if point == fault_point and not injected["raised"]:
            injected["raised"] = True
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        solve_certified_adaptive_multicut_fair_benders(
            instance, baseline_record=record, anchor=anchor, rho=0.0, gamma=0,
            algorithm_config=FROZEN_PRECISION, max_iterations=100, time_limit=60.0, tol=1e-6,
            checkpoint_path=checkpoint, checkpoint_identity={"run_key": fault_point},
            failure_injector=interrupt,
        )
    resumed = solve_certified_adaptive_multicut_fair_benders(
        instance, baseline_record=record, anchor=anchor, rho=0.0, gamma=0,
        algorithm_config=FROZEN_PRECISION, max_iterations=100, time_limit=60.0, tol=1e-6,
        checkpoint_path=checkpoint, checkpoint_identity={"run_key": fault_point},
    )
    assert resumed.status == clean.status == "optimal"
    assert resumed.objective_t == pytest.approx(clean.objective_t, abs=1e-8)
    assert resumed.lower_bound == pytest.approx(clean.lower_bound, abs=1e-8)
    assert resumed.cuts == clean.cuts


def test_recovery_ledger_faults_resume_without_replaying_committed_phases(tmp_path):
    ledger = tmp_path / "pipeline.json"
    calls: list[str] = []
    failed = {"done": False}

    def action(phase):
        calls.append(phase)

    def fail(point, _payload):
        if point == "before_aggregation_complete_checkpoint" and not failed["done"]:
            failed["done"] = True
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        advance_recovery_ledger(ledger, identity={"run_key": "x"}, phase_action=action, failure_injector=fail)
    persisted = json.loads(ledger.read_text(encoding="utf-8"))
    assert persisted["completed_phases"] == list(RECOVERABLE_PHASES[:3])
    resumed_calls: list[str] = []
    final = advance_recovery_ledger(
        ledger, identity={"run_key": "x"}, phase_action=resumed_calls.append,
    )
    assert resumed_calls == list(RECOVERABLE_PHASES[3:])
    assert final["completed_phases"] == list(RECOVERABLE_PHASES)
    with pytest.raises(RemediationGateError, match="identity"):
        advance_recovery_ledger(ledger, identity={"run_key": "drift"}, phase_action=lambda _phase: None)


def test_runner_dry_runs_are_read_only_and_formal_execution_is_refused():
    expected = {"L0": (2, 1, 1), "L1": (9, 3, 6), "M1": (9, 3, 6)}
    for stage, path in CONFIGS.items():
        config = json.loads(json.dumps(__import__("yaml").safe_load(path.read_text(encoding="utf-8"))))
        output = ROOT / config["output_dir"]
        assert not output.exists()
        report = dry_run_remediation(path, stage=stage, root=ROOT)
        assert (report["tasks"], report["baseline_count"], report["frontier_count"]) == expected[stage]
        assert report["solver_called"] is report["instances_generated"] is False
        assert report["output_dir_exists"] is False
        assert report["longest_windows_path_length"] <= 220
        with pytest.raises(RemediationGateError, match="formal_run_not_authorized"):
            run_remediation_stage(path, stage=stage, resume=False, dry_run=False)
        assert not output.exists()


def test_stage_gates_fail_closed_and_no_holdout_is_available():
    validate_stage_gate("L0", None)
    l0 = {
        "decision": "authorize_L1_after_L0", "scientific_status": "certified_robust_optimal",
        "post_evaluation_valid": True, "scenario_count": 4657, "valid_upper_bound": True,
        "valid_lower_bound": True, "implementation_error": False,
        "invalid_post_evaluation": False, "identity_sha256": "A" * 64,
    }
    validate_stage_gate("L1", l0)
    with pytest.raises(RemediationGateError):
        validate_stage_gate("L1", {**l0, "scientific_status": "time_limit_uncertified"})
    l1 = {
        "decision": "authorize_M1_after_L1", "certified_frontier_count": 4,
        "frontier_count": 6, "implementation_error_count": 0,
        "invalid_post_evaluation_count": 0,
        "all_successful_post_evaluations_valid": True, "identity_sha256": "B" * 64,
    }
    validate_stage_gate("M1", l1)
    with pytest.raises(RemediationGateError):
        validate_stage_gate("M1", {**l1, "certified_frontier_count": 3})
