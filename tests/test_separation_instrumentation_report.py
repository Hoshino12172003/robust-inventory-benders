from __future__ import annotations

from pathlib import Path

from src.separation_instrumentation import PHASES, SeparationInstrumentation
from src.separation_instrumentation_audit import audit_ledger, audit_sources
from src.separation_instrumentation_report import aggregate_instrumentation


class Clock:
    value = 0

    def __call__(self) -> int:
        return self.value


def ledger(run_key: str, final_exact: bool) -> dict[str, object]:
    clock = Clock()
    observer = SeparationInstrumentation(enabled=True, clock_ns=clock)
    call_id = observer.begin_call(run_key=run_key, iteration=1,
                                  separation_call_index=int(final_exact),
                                  final_exact_certification=final_exact)
    clock.value = 7
    observer.finish_call(call_id)
    observer.commit_iteration(1)
    return observer.checkpoint_payload()


def test_aggregator_is_deterministic_and_separates_final_exact() -> None:
    runs = [
        {"run_key": "b", "algorithm_runtime_ns": 100, "instrumentation": ledger("b", True)},
        {"run_key": "a", "algorithm_runtime_ns": 200, "instrumentation": ledger("a", False)},
    ]
    first = aggregate_instrumentation(runs)
    second = aggregate_instrumentation(reversed(runs))
    assert first == second
    summaries = {row["call_kind"]: row for row in first["phase_summaries"]}
    assert summaries["ordinary"]["separation_total_ns"] == 7
    assert summaries["final_exact"]["separation_total_ns"] == 7
    assert summaries["all"]["conservation_error_ns"] == 0
    assert summaries["all"]["separation_unclassified_ns_share_algorithm"] == 14 / 300
    assert summaries["all"]["separation_milp_node_count_missing_count"] == 2


def test_solver_free_static_and_ledger_audits_pass() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = ledger("audit", False)
    assert audit_ledger(payload)["passed"]
    result = audit_sources(root)
    assert result["passed"], result


def test_every_phase_is_present_even_when_zero() -> None:
    payload = ledger("fields", False)
    record = payload["committed_records"][0]
    assert all(name in record for name in PHASES)
