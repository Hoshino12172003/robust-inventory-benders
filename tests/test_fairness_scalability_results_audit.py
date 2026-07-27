from __future__ import annotations

import json

import yaml

from src.fairness_scalability_results_audit import (
    CANONICALIZATION,
    FIELD_PROJECTIONS,
    aggregate_records,
    canonical_config_bytes,
    resolved_config_file_bytes,
)


def _records_and_specs():
    specs = [
        {
            "run_key": "baseline", "introduced_stage": "s1", "task_type": "baseline",
            "scale": "medium_large", "seed": 160, "rho": None,
            "candidate": "joint_v1_core_point_strengthened",
        },
        {
            "run_key": "frontier", "introduced_stage": "s1", "task_type": "frontier",
            "scale": "medium_large", "seed": 160, "rho": 0.0,
            "candidate": "persistent_certified_cache_batch5",
        },
    ]
    iteration_log = [
        {
            "separation_model_build_runtime": 1.0, "separation_optimize_runtime": 2.0,
            "cache_candidate_count": 3, "cache_hit_count": 2,
            "certified_cached_cut_count": 2, "pool_candidate_count": 4,
            "certified_batch_cut_count": 3, "duplicate_pattern_count": 1,
            "cuts_per_iteration": 3,
        },
        {
            "separation_model_build_runtime": 0.5, "separation_optimize_runtime": 1.5,
            "cache_candidate_count": 2, "cache_hit_count": 1,
            "certified_cached_cut_count": 1, "pool_candidate_count": 3,
            "certified_batch_cut_count": 2, "duplicate_pattern_count": 0,
            "cuts_per_iteration": 2,
        },
    ]
    metadata = {
        "separation_model_build_runtime": 1.5, "separation_optimize_runtime": 3.5,
        "cache_candidate_count": 5, "cache_hit_count": 3,
        "certified_cached_cut_count": 3, "pool_candidate_count": 7,
        "certified_batch_cut_count": 5, "duplicate_pattern_count": 1,
        "cuts_per_iteration": 2.5, "total_iterations": 2,
    }
    records = [
        {
            "run_key": "baseline", "task_type": "baseline", "seed": 160,
            "candidate": "joint_v1_core_point_strengthened", "state": "complete",
            "scientific_status": "certified_robust_optimal", "algorithm_status": "optimal",
            "result": {"runtime": 10.0, "master_runtime": 4.0, "iterations": 2, "cuts": 2},
        },
        {
            "run_key": "frontier", "task_type": "frontier", "seed": 160, "rho": 0.0,
            "candidate": "persistent_certified_cache_batch5", "state": "complete",
            "scientific_status": "unknown_uncertified",
            "algorithm_status": "separation_stalled_duplicate",
            "result": {
                "runtime": 20.0, "algorithm_runtime": 20.0, "total_wall_runtime": 21.0,
                "separation_runtime": 5.5, "master_runtime": 6.0, "cuts": 5,
                "iteration_log": iteration_log, "metadata": metadata,
            },
        },
    ]
    return records, specs


def test_projection_fields_are_nonempty_and_have_one_declared_semantic() -> None:
    records, specs = _records_and_specs()
    rows, _ = aggregate_records(
        records, specs, source_archive_sha256="A" * 64, time_limit=1800.0
    )
    assert len(rows) == 2
    required = {
        "separation_runtime", *FIELD_PROJECTIONS, "master_runtime", "total_iterations",
        "cuts", "algorithm_runtime", "penalized_runtime_par2", "total_wall_runtime",
    }
    assert all(row[field] not in (None, "") for row in rows for field in required)
    frontier = next(row for row in rows if row["task_type"] == "frontier")
    assert frontier["separation_model_build_runtime"] == 1.5
    assert frontier["cache_candidate_count"] == 5.0
    assert frontier["cuts_per_iteration"] == 2.5
    baseline = next(row for row in rows if row["task_type"] == "baseline")
    assert all(baseline[field] == 0.0 for field in FIELD_PROJECTIONS)
    assert {value["aggregation"] for value in FIELD_PROJECTIONS.values()} == {
        "task_metadata_total_verified_against_iteration_log_sum",
        "task_metadata_mean_verified_against_iteration_log_mean",
    }


def test_summary_does_not_count_uncertified_as_solved_and_reports_runtime_statistics() -> None:
    records, specs = _records_and_specs()
    _, summary = aggregate_records(records, specs, source_archive_sha256="B" * 64)
    frontier = next(row for row in summary if row["task_type"] == "frontier")
    assert frontier["run_count"] == 1
    assert frontier["certified_solved_count"] == 0
    assert frontier["solved_rate"] == 0.0
    assert frontier["status_unknown_uncertified_count"] == 1
    assert frontier["mean_algorithm_runtime"] == 20.0
    assert frontier["median_algorithm_runtime"] == 20.0
    assert frontier["mean_penalized_runtime_par2"] == 3600.0
    assert frontier["total_certified_batch_cut_count"] == 5.0


def test_file_and_canonical_config_hash_inputs_are_explicit_and_distinct() -> None:
    config = yaml.safe_load("z: 1\na: 2\n")
    assert resolved_config_file_bytes(config) != canonical_config_bytes(config)
    assert CANONICALIZATION == "PyYAML safe_dump(sort_keys=True, allow_unicode=True), UTF-8"


def test_reaggregation_is_byte_deterministic() -> None:
    records, specs = _records_and_specs()
    first = aggregate_records(records, specs, source_archive_sha256="C" * 64)
    second = aggregate_records(list(reversed(records)), specs, source_archive_sha256="C" * 64)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
