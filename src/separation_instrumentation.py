"""Behavior-neutral, solver-independent CAMS-CCG separation instrumentation."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import time
from typing import Any, Callable, Iterator, Mapping


SCHEMA_VERSION = "cams_ccg_separation_instrumentation_v1"
CHECKPOINT_SCHEMA_VERSION = "cams_ccg_separation_instrumentation_checkpoint_v1"
PHASES = (
    "separation_model_prepare_ns", "separation_milp_optimize_ns",
    "solution_pool_extract_ns", "cache_candidate_processing_ns",
    "fixed_primal_certification_ns", "farkas_certification_ns",
    "candidate_identity_dedup_ns", "deterministic_candidate_selection_ns",
    "final_exact_prepare_ns", "final_exact_optimize_ns",
)
COUNTERS = (
    "separation_milp_optimize_calls", "pool_patterns_extracted",
    "cache_patterns_considered", "cache_hits", "cache_misses",
    "fixed_primal_calls", "fixed_primal_optimal", "fixed_primal_infeasible",
    "fixed_primal_other", "farkas_calls", "farkas_certified", "farkas_failed",
    "candidate_patterns_before_dedup", "candidate_patterns_after_dedup",
    "certified_candidates", "selected_scenarios", "final_exact_calls",
)
NULLABLE_DIAGNOSTICS = (
    "separation_milp_node_count", "separation_milp_solution_count",
    "separation_milp_status", "separation_milp_incumbent",
    "separation_milp_objective_bound", "separation_milp_gap",
    "final_exact_node_count", "final_exact_status", "final_exact_objective_bound",
)


class InstrumentationError(ValueError):
    pass


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def identity_sha256(run_key: str, iteration: int, call_index: int, final_exact: bool) -> str:
    payload = {"final_exact_certification": bool(final_exact), "iteration": int(iteration),
               "run_key": str(run_key), "separation_call_index": int(call_index)}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()


class SeparationInstrumentation:
    """Transactional call ledger. Timings never feed an algorithm decision."""

    def __init__(self, *, enabled: bool = False,
                 clock_ns: Callable[[], int] = time.perf_counter_ns) -> None:
        self.enabled = bool(enabled)
        self._clock_ns = clock_ns
        self.committed: list[dict[str, Any]] = []
        self.discarded: list[dict[str, Any]] = []
        self.pending: dict[str, dict[str, Any]] = {}
        self._active_phase: dict[str, str] = {}
        self.last_committed_iteration = 0

    def begin_call(self, *, run_key: str, iteration: int, separation_call_index: int,
                   final_exact_certification: bool) -> str | None:
        if not self.enabled:
            return None
        call_id = identity_sha256(run_key, iteration, separation_call_index,
                                  final_exact_certification)
        if call_id in self.pending or any(r["call_id"] == call_id for r in self.committed):
            raise InstrumentationError("duplicate separation call identity")
        start = int(self._clock_ns())
        if start < 0:
            raise InstrumentationError("monotonic clock returned a negative value")
        record: dict[str, Any] = {
            "instrumentation_schema_version": SCHEMA_VERSION,
            "call_id": call_id, "run_key": str(run_key), "iteration": int(iteration),
            "separation_call_index": int(separation_call_index),
            "final_exact_certification": bool(final_exact_certification),
            "call_role": "final_exact" if final_exact_certification else "ordinary",
            "start_monotonic_ns": start, "end_monotonic_ns": None,
            "separation_total_ns": None, "separation_unclassified_ns": None,
            "state": "pending", "incomplete_reason": None,
        }
        record.update({phase: 0 for phase in PHASES})
        record.update({counter: 0 for counter in COUNTERS})
        record.update({key: None for key in NULLABLE_DIAGNOSTICS})
        record.update({f"{key}_missing_reason": "not_reported" for key in NULLABLE_DIAGNOSTICS})
        self.pending[call_id] = record
        return call_id

    def record_persistent_model_setup(self, *, run_key: str, start_ns: int,
                                      end_ns: int) -> None:
        """Record pre-separation persistent-model construction without backdating a call."""
        if not self.enabled:
            return
        if type(start_ns) is not int or type(end_ns) is not int or end_ns < start_ns:
            raise InstrumentationError("invalid persistent model setup duration")
        prior = sum(record.get("call_role") == "persistent_model_setup"
                    for record in self.committed)
        call_id = self.begin_call(
            run_key=run_key, iteration=0, separation_call_index=-(prior + 1),
            final_exact_certification=False,
        )
        assert call_id is not None
        record = self.pending[call_id]
        record["call_role"] = "persistent_model_setup"
        record["start_monotonic_ns"] = start_ns
        record["end_monotonic_ns"] = end_ns
        record["separation_model_prepare_ns"] = end_ns - start_ns
        record["separation_total_ns"] = end_ns - start_ns
        record["separation_unclassified_ns"] = 0
        record["state"] = "finished_pending_iteration_commit"
        self.commit_iteration(0)

    @contextmanager
    def phase(self, call_id: str | None, name: str) -> Iterator[None]:
        if call_id is None:
            yield
            return
        if name not in PHASES:
            raise InstrumentationError(f"unknown phase: {name}")
        if call_id not in self.pending:
            raise InstrumentationError("phase has no pending call")
        if call_id in self._active_phase:
            raise InstrumentationError("nested instrumentation phase")
        self._active_phase[call_id] = name
        start = int(self._clock_ns())
        try:
            yield
        finally:
            end = int(self._clock_ns())
            elapsed = end - start
            self._active_phase.pop(call_id, None)
            if elapsed < 0:
                raise InstrumentationError("non-monotonic phase clock")
            self.pending[call_id][name] += elapsed

    def increment(self, call_id: str | None, name: str, amount: int = 1) -> None:
        if call_id is None:
            return
        if name not in COUNTERS or type(amount) is not int or amount < 0:
            raise InstrumentationError("invalid instrumentation counter")
        self.pending[call_id][name] += amount

    def add_duration(self, call_id: str | None, name: str, duration_ns: int) -> None:
        if call_id is None:
            return
        if name not in PHASES or type(duration_ns) is not int or duration_ns < 0:
            raise InstrumentationError("invalid instrumentation duration")
        self.pending[call_id][name] += duration_ns

    def diagnostic(self, call_id: str | None, name: str, value: Any,
                   *, missing_reason: str | None = None) -> None:
        if call_id is None:
            return
        if name not in NULLABLE_DIAGNOSTICS:
            raise InstrumentationError("invalid solver diagnostic")
        if isinstance(value, float) and not (value == value and abs(value) != float("inf")):
            value = None
            missing_reason = missing_reason or "nonfinite_solver_attribute"
        self.pending[call_id][name] = value
        self.pending[call_id][f"{name}_missing_reason"] = (
            None if value is not None else (missing_reason or "solver_attribute_unavailable")
        )

    def add_numeric_diagnostic(self, call_id: str | None, name: str,
                               value: Any) -> None:
        if call_id is None:
            return
        if name not in NULLABLE_DIAGNOSTICS:
            raise InstrumentationError("invalid solver diagnostic")
        if value is None:
            self.diagnostic(call_id, name, None)
            return
        if not isinstance(value, (int, float)):
            raise InstrumentationError("non-numeric cumulative solver diagnostic")
        previous = self.pending[call_id][name]
        self.diagnostic(call_id, name, value + (0 if previous is None else previous))

    def finish_call(self, call_id: str | None) -> None:
        if call_id is None:
            return
        if call_id in self._active_phase:
            raise InstrumentationError("cannot finish inside a phase")
        record = self.pending[call_id]
        end = int(self._clock_ns())
        total = end - int(record["start_monotonic_ns"])
        classified = sum(int(record[name]) for name in PHASES)
        unclassified = total - classified
        if unclassified < 0:
            raise InstrumentationError("classified phase time exceeds separation total")
        record["end_monotonic_ns"] = end
        record["separation_total_ns"] = total
        record["separation_unclassified_ns"] = unclassified
        record["state"] = "finished_pending_iteration_commit"

    def commit_iteration(self, iteration: int) -> None:
        if not self.enabled:
            return
        ids = [call_id for call_id, record in self.pending.items()
               if record["iteration"] == int(iteration)]
        if any(self.pending[call_id]["state"] != "finished_pending_iteration_commit" for call_id in ids):
            raise InstrumentationError("iteration contains incomplete call")
        for call_id in ids:
            record = self.pending.pop(call_id)
            record["state"] = "committed"
            self.committed.append(record)
        self.last_committed_iteration = max(self.last_committed_iteration, int(iteration))

    def discard_pending(self, reason: str) -> None:
        if not self.enabled:
            return
        for call_id in list(self.pending):
            record = self.pending.pop(call_id)
            record["state"] = "discarded_incomplete"
            record["incomplete_reason"] = str(reason)
            self.discarded.append(record)
        self._active_phase.clear()

    def checkpoint_payload(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        cumulative = {name: sum(int(r[name]) for r in self.committed)
                      for name in PHASES + ("separation_unclassified_ns",) + COUNTERS}
        payload = {
            "instrumentation_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "committed_call_ids": [r["call_id"] for r in self.committed],
            "pending_call_id": next(iter(self.pending), None),
            "cumulative_committed_counters": cumulative,
            "last_committed_iteration": self.last_committed_iteration,
            "committed_records": deepcopy(self.committed),
            "discarded_records": deepcopy(self.discarded),
            "pending_records": deepcopy(list(self.pending.values())),
        }
        identity_payload = deepcopy(payload)
        payload["identity_sha256"] = hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest().upper()
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any], *,
                        clock_ns: Callable[[], int] = time.perf_counter_ns) -> "SeparationInstrumentation":
        if payload.get("instrumentation_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise InstrumentationError("instrumentation checkpoint schema drift")
        supplied = payload.get("identity_sha256")
        check = dict(payload); check.pop("identity_sha256", None)
        expected = hashlib.sha256(canonical_json_bytes(check)).hexdigest().upper()
        if supplied != expected:
            raise InstrumentationError("instrumentation checkpoint identity mismatch")
        obj = cls(enabled=True, clock_ns=clock_ns)
        obj.committed = deepcopy(list(payload.get("committed_records", [])))
        obj.discarded = deepcopy(list(payload.get("discarded_records", [])))
        for record in deepcopy(list(payload.get("pending_records", []))):
            record["state"] = "discarded_incomplete"
            record["incomplete_reason"] = "discarded_pending_attempt_on_resume"
            obj.discarded.append(record)
        ids = [r["call_id"] for r in obj.committed]
        if len(ids) != len(set(ids)) or ids != list(payload.get("committed_call_ids", [])):
            raise InstrumentationError("committed call identity drift")
        for record in obj.committed:
            if record.get("instrumentation_schema_version") != SCHEMA_VERSION:
                raise InstrumentationError("committed record schema drift")
            if record.get("state") != "committed":
                raise InstrumentationError("committed record state drift")
            expected_call_id = identity_sha256(
                str(record.get("run_key")), int(record.get("iteration")),
                int(record.get("separation_call_index")),
                bool(record.get("final_exact_certification")),
            )
            if record.get("call_id") != expected_call_id:
                raise InstrumentationError("committed call identity payload drift")
            values = [record.get(name) for name in PHASES + (
                "separation_unclassified_ns", "separation_total_ns",
            ) + COUNTERS]
            if any(type(value) is not int or value < 0 for value in values):
                raise InstrumentationError("committed record numeric drift")
            if record["separation_total_ns"] != (
                sum(record[name] for name in PHASES)
                + record["separation_unclassified_ns"]
            ):
                raise InstrumentationError("committed record timing conservation drift")
        expected_cumulative = {
            name: sum(int(record[name]) for record in obj.committed)
            for name in PHASES + ("separation_unclassified_ns",) + COUNTERS
        }
        if payload.get("cumulative_committed_counters") != expected_cumulative:
            raise InstrumentationError("cumulative committed counters drift")
        obj.last_committed_iteration = int(payload.get("last_committed_iteration", 0))
        expected_last = max((int(record["iteration"]) for record in obj.committed), default=0)
        if obj.last_committed_iteration != expected_last:
            raise InstrumentationError("last committed iteration drift")
        # Pending attempts are intentionally discarded on resume; never counted twice.
        return obj
