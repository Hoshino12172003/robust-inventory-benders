from __future__ import annotations

from copy import deepcopy
import json

import pytest

from src.separation_instrumentation import (
    InstrumentationError,
    PHASES,
    SeparationInstrumentation,
    canonical_json_bytes,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        return self.value

    def advance(self, amount: int) -> None:
        self.value += amount


def complete_call(observer: SeparationInstrumentation, clock: FakeClock, *,
                  iteration: int = 1, call_index: int = 0,
                  final_exact: bool = False) -> str:
    call_id = observer.begin_call(
        run_key="mock_run", iteration=iteration,
        separation_call_index=call_index,
        final_exact_certification=final_exact,
    )
    assert call_id is not None
    clock.advance(3)
    for index, phase in enumerate(PHASES, start=1):
        with observer.phase(call_id, phase):
            clock.advance(index)
        clock.advance(1)
    observer.diagnostic(call_id, "separation_milp_node_count", None,
                        missing_reason="mock_attribute_unavailable")
    observer.finish_call(call_id)
    return call_id


def test_exact_phase_timing_and_conservation() -> None:
    clock = FakeClock()
    observer = SeparationInstrumentation(enabled=True, clock_ns=clock)
    complete_call(observer, clock)
    observer.commit_iteration(1)
    record = observer.committed[0]
    assert [record[name] for name in PHASES] == list(range(1, len(PHASES) + 1))
    assert record["separation_total_ns"] == sum(range(1, 11)) + 13
    assert record["separation_unclassified_ns"] == 13
    assert sum(record[name] for name in PHASES) + record["separation_unclassified_ns"] == record["separation_total_ns"]


def test_nested_timer_is_rejected() -> None:
    observer = SeparationInstrumentation(enabled=True, clock_ns=FakeClock())
    call_id = observer.begin_call(run_key="r", iteration=1,
                                  separation_call_index=0,
                                  final_exact_certification=False)
    with pytest.raises(InstrumentationError, match="nested"):
        with observer.phase(call_id, PHASES[0]):
            with observer.phase(call_id, PHASES[1]):
                pass


def test_persistent_model_setup_is_separate_and_conservative() -> None:
    observer = SeparationInstrumentation(enabled=True, clock_ns=FakeClock())
    observer.record_persistent_model_setup(run_key="r", start_ns=10, end_ns=25)
    record = observer.committed[0]
    assert record["call_role"] == "persistent_model_setup"
    assert record["separation_model_prepare_ns"] == 15
    assert record["separation_total_ns"] == 15
    assert record["separation_unclassified_ns"] == 0


def test_disabled_observer_is_noop_and_emits_no_payload() -> None:
    observer = SeparationInstrumentation(enabled=False, clock_ns=FakeClock())
    scientific = {"status": "optimal", "objective": 0.25, "cuts": ["A", "B"]}
    before = deepcopy(scientific)
    call_id = observer.begin_call(run_key="r", iteration=1,
                                  separation_call_index=0,
                                  final_exact_certification=False)
    with observer.phase(call_id, PHASES[0]):
        scientific["cuts"] = sorted(scientific["cuts"])
    observer.finish_call(call_id)
    observer.commit_iteration(1)
    assert scientific == before
    assert observer.checkpoint_payload() == {}
    assert observer.committed == []


def _mock_scientific_pipeline(enabled: bool) -> tuple[dict[str, object], dict[str, object]]:
    clock = FakeClock()
    observer = SeparationInstrumentation(enabled=enabled, clock_ns=clock)
    call_id = observer.begin_call(run_key="mock", iteration=2,
                                  separation_call_index=0,
                                  final_exact_certification=False)
    candidates = [("b", 2), ("a", 1), ("a", 1)]
    with observer.phase(call_id, "candidate_identity_dedup_ns"):
        unique = sorted(set(candidates))
        clock.advance(2)
    with observer.phase(call_id, "deterministic_candidate_selection_ns"):
        selected = unique[0]
        clock.advance(3)
    scientific = {
        "status": "cut_added", "objective": 0.5, "lower_bound": 0.4,
        "upper_bound": 0.6, "gap": 0.2,
        "selected_ids": [selected[0]], "cut_ids": [item[0] for item in unique],
        "scenario_block_order": [item[1] for item in unique],
    }
    observer.finish_call(call_id)
    observer.commit_iteration(2)
    return scientific, observer.checkpoint_payload()


def test_on_off_mock_pipeline_has_identical_scientific_outputs() -> None:
    disabled, disabled_payload = _mock_scientific_pipeline(False)
    enabled, enabled_payload = _mock_scientific_pipeline(True)
    assert disabled == enabled
    assert disabled_payload == {}
    assert enabled_payload["committed_records"][0]["state"] == "committed"


@pytest.mark.parametrize("failure_point", [
    "prepare_after_optimize_before", "milp_after_pool_before", "fixed_primal_mid",
    "farkas_after_selection_before", "selection_after_master_commit_before",
    "master_commit_after_checkpoint_before", "final_exact_after_state_commit_before",
])
def test_fault_injection_discards_uncommitted_attempt(failure_point: str) -> None:
    clock = FakeClock()
    observer = SeparationInstrumentation(enabled=True, clock_ns=clock)
    call_id = observer.begin_call(
        run_key="fault", iteration=3, separation_call_index=0,
        final_exact_certification=failure_point.startswith("final_exact"),
    )
    phase_by_point = {
        "prepare_after_optimize_before": "separation_model_prepare_ns",
        "milp_after_pool_before": "separation_milp_optimize_ns",
        "fixed_primal_mid": "fixed_primal_certification_ns",
        "farkas_after_selection_before": "farkas_certification_ns",
        "selection_after_master_commit_before": "deterministic_candidate_selection_ns",
        "master_commit_after_checkpoint_before": "candidate_identity_dedup_ns",
        "final_exact_after_state_commit_before": "final_exact_optimize_ns",
    }
    with observer.phase(call_id, phase_by_point[failure_point]):
        clock.advance(1)
    payload = observer.checkpoint_payload()
    resumed = SeparationInstrumentation.from_checkpoint(payload, clock_ns=clock)
    assert resumed.committed == []
    assert len(resumed.discarded) == 1
    assert resumed.discarded[0]["state"] == "discarded_incomplete"


def test_resume_counts_committed_call_exactly_once_and_is_byte_stable() -> None:
    clock = FakeClock()
    observer = SeparationInstrumentation(enabled=True, clock_ns=clock)
    call_id = complete_call(observer, clock)
    observer.increment(call_id, "fixed_primal_calls", 2)
    observer.commit_iteration(1)
    first = observer.checkpoint_payload()
    resumed = SeparationInstrumentation.from_checkpoint(first, clock_ns=clock)
    second = resumed.checkpoint_payload()
    assert first == second
    assert sum(r["fixed_primal_calls"] for r in resumed.committed) == 2
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_crash_after_in_memory_commit_before_atomic_write_uses_last_durable_state() -> None:
    clock = FakeClock()
    durable = SeparationInstrumentation(enabled=True, clock_ns=clock).checkpoint_payload()
    attempt = SeparationInstrumentation.from_checkpoint(durable, clock_ns=clock)
    complete_call(attempt, clock, iteration=4)
    attempt.commit_iteration(4)
    assert len(attempt.committed) == 1
    resumed = SeparationInstrumentation.from_checkpoint(durable, clock_ns=clock)
    assert resumed.committed == []
    assert resumed.last_committed_iteration == 0


def test_checkpoint_rejects_schema_corruption_and_duplicate_identity() -> None:
    clock = FakeClock()
    observer = SeparationInstrumentation(enabled=True, clock_ns=clock)
    complete_call(observer, clock)
    observer.commit_iteration(1)
    corrupted = observer.checkpoint_payload()
    corrupted["instrumentation_schema_version"] = "wrong"
    with pytest.raises(InstrumentationError, match="schema drift"):
        SeparationInstrumentation.from_checkpoint(corrupted, clock_ns=clock)
    with pytest.raises(InstrumentationError, match="duplicate"):
        observer.begin_call(run_key="mock_run", iteration=1,
                            separation_call_index=0,
                            final_exact_certification=False)


def test_nonfinite_solver_diagnostic_is_nullable_not_json_nan() -> None:
    clock = FakeClock()
    observer = SeparationInstrumentation(enabled=True, clock_ns=clock)
    call_id = observer.begin_call(run_key="r", iteration=1,
                                  separation_call_index=0,
                                  final_exact_certification=False)
    observer.diagnostic(call_id, "separation_milp_gap", float("nan"))
    observer.finish_call(call_id)
    observer.commit_iteration(1)
    record = observer.committed[0]
    assert record["separation_milp_gap"] is None
    assert record["separation_milp_gap_missing_reason"] == "nonfinite_solver_attribute"
    json.loads(canonical_json_bytes(observer.checkpoint_payload()))
