"""Solver-free static and ledger audit for behavior-neutral instrumentation."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .separation_instrumentation import COUNTERS, PHASES, SCHEMA_VERSION


PRODUCTION_PATHS = (
    "src/fairness_hybrid_ccg_benders.py",
    "src/fairness_large_final_remediation.py",
    "src/fairness_scalability.py",
    "src/robust_regional_fairness.py",
)


def audit_ledger(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = list(payload.get("committed_records", []))
    ids = [record.get("call_id") for record in records]
    required = set(PHASES) | set(COUNTERS) | {
        "separation_total_ns", "separation_unclassified_ns", "state",
        "instrumentation_schema_version",
    }
    checks = {
        "fields_complete": all(required <= set(record) for record in records),
        "nonnegative_integer_times": all(
            type(record.get(name)) is int and record[name] >= 0
            for record in records for name in PHASES + (
                "separation_total_ns", "separation_unclassified_ns",
            )
        ),
        "unique_call_identity": len(ids) == len(set(ids)),
        "committed_records_unique": len(ids) == len(set(ids)),
        "timing_conservation": all(
            record["separation_total_ns"]
            == sum(record[name] for name in PHASES)
               + record["separation_unclassified_ns"]
            for record in records
        ),
        "schema_matches": all(
            record.get("instrumentation_schema_version") == SCHEMA_VERSION
            for record in records
        ),
    }
    return {**checks, "passed": all(checks.values())}


def _solver_parameter_signatures(text: str) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                rendered = ast.unparse(target)
                if ".Params." in rendered:
                    result.add((rendered, ast.unparse(node.value)))
    return result


def _behavioral_call_counts(text: str) -> dict[str, int]:
    counts = {name: 0 for name in (
        "optimize", "certify_fixed_scenario_fairness_feasibility",
        "_pool_patterns", "select_one_new_scenario", "_deterministic_diverse_order",
    )}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func).rsplit(".", 1)[-1]
            if name in counts:
                counts[name] += 1
    return counts


def audit_sources(root: Path) -> dict[str, Any]:
    current_sources = {
        relative: (root / relative).read_text(encoding="utf-8")
        for relative in PRODUCTION_PATHS
    }
    measured_names = set(PHASES) | set(COUNTERS) | {
        "separation_total_ns", "separation_unclassified_ns", "perf_counter_ns",
    }
    unexpected_conditions: list[str] = []
    for relative, source in current_sources.items():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.If, ast.While)):
                condition = ast.unparse(node.test)
                if any(name in condition for name in measured_names):
                    unexpected_conditions.append(f"{relative}: {condition}")

    parameter_matches_base = True
    behavioral_counts: dict[str, dict[str, dict[str, int]]] = {}
    base_readable = True
    for relative, source in current_sources.items():
        completed = subprocess.run(
            ["git", "show", f"origin/main:{relative}"], cwd=root,
            capture_output=True, text=True, encoding="utf-8", check=False,
        )
        if completed.returncode != 0:
            base_readable = False
            parameter_matches_base = False
            break
        if _solver_parameter_signatures(source) != _solver_parameter_signatures(completed.stdout):
            parameter_matches_base = False
        behavioral_counts[relative] = {
            "origin_main": _behavioral_call_counts(completed.stdout),
            "instrumented_source": _behavioral_call_counts(source),
        }

    hybrid_source = current_sources[PRODUCTION_PATHS[0]]
    checks = {
        "instrumentation_not_in_scientific_conditions": not unexpected_conditions,
        "disabled_has_no_instrumentation_output": (
            "if not separation_instrumentation_enabled:" in hybrid_source
            and "observer = None" in hybrid_source
            and '**({"separation_instrumentation": observer.checkpoint_payload()}' in hybrid_source
            and "if observer is not None else {}" in hybrid_source
        ),
        "uses_perf_counter_ns": "perf_counter_ns" in (
            root / "src/separation_instrumentation.py"
        ).read_text(encoding="utf-8"),
        "origin_main_readable_for_parameter_audit": base_readable,
        "does_not_change_solver_parameter_expressions": parameter_matches_base,
        "production_instrumentation_default_false": (
            "separation_instrumentation_enabled: bool = False" in hybrid_source
        ),
        "enabled_checkpoint_identity_binds_schema": (
            'identity["separation_instrumentation_schema"] = SCHEMA_VERSION'
            in hybrid_source
        ),
    }
    return {**checks, "unexpected_conditions": unexpected_conditions,
            "behavioral_call_counts_for_review": behavioral_counts,
            "passed": all(checks.values())}
