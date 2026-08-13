"""Deterministic solver-free aggregation of separation instrumentation ledgers."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping

from .separation_instrumentation import (
    COUNTERS, NULLABLE_DIAGNOSTICS, PHASES, SCHEMA_VERSION,
)


def aggregate_instrumentation(
    runs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    algorithm_runtime_ns = 0
    discarded = 0
    for run in sorted(runs, key=lambda item: str(item["run_key"])):
        algorithm_runtime_ns += int(run["algorithm_runtime_ns"])
        ledger = run["instrumentation"]
        discarded += len(ledger.get("discarded_records", []))
        for record in ledger.get("committed_records", []):
            row = deepcopy(record)
            row["source_run_key"] = str(run["run_key"])
            details.append(row)
    details.sort(key=lambda row: (
        row["source_run_key"], int(row["iteration"]),
        int(row["separation_call_index"]), row["call_id"],
    ))
    groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in details:
        kind = str(row.get("call_role", (
            "final_exact" if row["final_exact_certification"] else "ordinary"
        )))
        for name in PHASES + ("separation_unclassified_ns", "separation_total_ns") + COUNTERS:
            groups[kind][name] += int(row[name])
            groups["all"][name] += int(row[name])
    summaries: list[dict[str, Any]] = []
    for kind in ("persistent_model_setup", "ordinary", "final_exact", "all"):
        values = groups[kind]
        total_sep = int(values["separation_total_ns"])
        row: dict[str, Any] = {
            "call_kind": kind,
            "algorithm_runtime_ns": algorithm_runtime_ns,
            "separation_total_ns": total_sep,
            "conservation_error_ns": total_sep - sum(
                int(values[name]) for name in PHASES + ("separation_unclassified_ns",)
            ),
        }
        for name in PHASES + ("separation_unclassified_ns",) + COUNTERS:
            value = int(values[name])
            row[name] = value
            if name.endswith("_ns"):
                row[f"{name}_seconds"] = value / 1_000_000_000
                row[f"{name}_share_algorithm"] = (
                    value / algorithm_runtime_ns if algorithm_runtime_ns else None
                )
                row[f"{name}_share_separation"] = value / total_sep if total_sep else None
        matching = [
            item for item in details
            if kind == "all" or item.get("call_role", (
                "final_exact" if item["final_exact_certification"] else "ordinary"
            )) == kind
        ]
        for name in NULLABLE_DIAGNOSTICS:
            present = [item[name] for item in matching if item[name] is not None]
            row[f"{name}_reported_count"] = len(present)
            row[f"{name}_missing_count"] = len(matching) - len(present)
            if name.endswith("_node_count"):
                row[f"{name}_sum"] = sum(float(value) for value in present)
        summaries.append(row)
    return {
        "schema_version": "cams_ccg_separation_instrumentation_report_v1",
        "source_schema_version": SCHEMA_VERSION,
        "call_details": details,
        "phase_summaries": summaries,
        "committed_record_count": len(details),
        "discarded_incomplete_record_count": discarded,
    }
