from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.fairness_scalability_s1_cross_scale_audit import (
    CrossScaleAuditError,
    EXPECTED_CANDIDATES,
    EXPERIMENT_MATRIX_SUMMARY_FIELDS,
    LARGE_ARCHIVE_SHA256,
    MEDIUM_LARGE_ARCHIVE_SHA256,
    NOT_APPLICABLE,
    build_decision,
    candidate_summary,
    experiment_matrix_summary,
    file_sha256,
    run_directory_id,
    validate_run_matrix,
    write_freeze_artifacts,
)


def _audit(scale: str, certified: int) -> dict:
    counts = {
        candidate: (2 if scale == "medium_large" and candidate != "persistent_certified_cache" else 0)
        for candidate in EXPECTED_CANDIDATES
    }
    if scale == "medium_large":
        counts["persistent_certified_cache"] = 2
        counts["single_cut"] = 6
        counts["persistent_separation"] = 6
        counts["persistent_certified_cache_batch5"] = 6
    return {
        "audit_status": "passed",
        "archive": {
            "sha256_before_audit": (
                MEDIUM_LARGE_ARCHIVE_SHA256 if scale == "medium_large" else LARGE_ARCHIVE_SHA256
            ),
            "sha256_after_audit": (
                MEDIUM_LARGE_ARCHIVE_SHA256 if scale == "medium_large" else LARGE_ARCHIVE_SHA256
            ),
            "size_bytes": 1,
        },
        "scientific_status": {
            "frontier_certified_solved_count": certified,
            "candidate_certified_counts": counts,
        },
    }


def _matrix() -> list[dict]:
    rows: list[dict] = []
    certified_counts = {
        "medium_large": {
            "single_cut": 6,
            "persistent_separation": 6,
            "persistent_certified_cache": 2,
            "persistent_certified_cache_batch5": 6,
        },
        "large": {candidate: 0 for candidate in EXPECTED_CANDIDATES},
    }
    for scale in ("medium_large", "large"):
        for seed in (160, 161, 162):
            rows.append(
                {
                    "medium_large_source_archive_sha256": MEDIUM_LARGE_ARCHIVE_SHA256,
                    "large_source_archive_sha256": LARGE_ARCHIVE_SHA256,
                    "source_archive_sha256": (
                        MEDIUM_LARGE_ARCHIVE_SHA256
                        if scale == "medium_large"
                        else LARGE_ARCHIVE_SHA256
                    ),
                    "scale": scale,
                    "task_type": "baseline",
                    "seed": seed,
                    "rho": NOT_APPLICABLE,
                    "candidate": "joint_v1_core_point_strengthened",
                    "run_key": f"{scale}-baseline-{seed}",
                    "baseline_run_key": NOT_APPLICABLE,
                    "anchor_value_hex": NOT_APPLICABLE,
                    "anchor_sha256": NOT_APPLICABLE,
                    "state": "complete",
                    "scientific_status": "certified_robust_optimal",
                    "certified_solved": "true",
                    "algorithm_runtime": 1.0,
                    "penalized_runtime_par2": 1.0,
                    "separation_runtime": 0.0,
                    "master_runtime": 0.0,
                    "total_wall_runtime": 1.0,
                    "iterations": 0,
                    "cuts": 0,
                }
            )
        for candidate in EXPECTED_CANDIDATES:
            certified_remaining = certified_counts[scale][candidate]
            for seed in (160, 161, 162):
                for rho in (0.0, 0.01):
                    certified = certified_remaining > 0
                    certified_remaining -= int(certified)
                    rows.append(
                        {
                            "medium_large_source_archive_sha256": MEDIUM_LARGE_ARCHIVE_SHA256,
                            "large_source_archive_sha256": LARGE_ARCHIVE_SHA256,
                            "source_archive_sha256": (
                                MEDIUM_LARGE_ARCHIVE_SHA256
                                if scale == "medium_large"
                                else LARGE_ARCHIVE_SHA256
                            ),
                            "scale": scale,
                            "task_type": "frontier",
                            "seed": seed,
                            "rho": rho,
                            "candidate": candidate,
                            "run_key": f"{scale}-{candidate}-{seed}-{rho}",
                            "baseline_run_key": f"{scale}-baseline-{seed}",
                            "anchor_value_hex": "0x1.0p+0",
                            "anchor_sha256": "A" * 64,
                            "state": "complete",
                            "scientific_status": (
                                "certified_robust_optimal"
                                if certified
                                else "time_limit_uncertified"
                            ),
                            "certified_solved": "true" if certified else "false",
                            "algorithm_runtime": 10.0,
                            "penalized_runtime_par2": 10.0 if certified else 3600.0,
                            "separation_runtime": 7.0,
                            "master_runtime": 2.0,
                            "total_wall_runtime": 10.0,
                            "iterations": 3,
                            "cuts": 4,
                        }
                    )
    return rows


def test_run_directory_id_is_the_frozen_short_hash() -> None:
    key = "canonical-run-key"
    value = run_directory_id(key)
    assert value.startswith("r_")
    assert len(value) == 26
    assert value == run_directory_id(key)


def test_candidate_summary_excludes_objective_and_uses_par2() -> None:
    rows = []
    for candidate in EXPECTED_CANDIDATES:
        rows.append(
            {
                "task_type": "frontier",
                "candidate": candidate,
                "scientific_status": "time_limit_uncertified",
                "algorithm_runtime": 10.0,
                "penalized_runtime_par2": 3600.0,
                "separation_runtime": 7.0,
                "master_runtime": 2.0,
                "post_evaluation_solver_runtime": 0.0,
                "post_evaluation_wall_runtime": 0.0,
                "total_wall_runtime": 10.0,
                "iterations": 3,
                "cuts": 4,
                "cache_candidate_count": 0,
                "cache_hit_count": 0,
                "certified_cached_cut_count": 0,
                "pool_candidate_count": 0,
                "certified_batch_cut_count": 0,
                "duplicate_pattern_count": 0,
                "cuts_per_iteration": 1.0,
                "objective_t": 123.0,
            }
        )
    summary = candidate_summary("large", rows)
    assert len(summary) == 4
    assert all(item["certified_solved_count"] == 0 for item in summary)
    assert all(item["mean_penalized_runtime_par2"] == 3600.0 for item in summary)
    assert all("objective_t" not in item for item in summary)


def test_decision_is_fail_closed_and_authorizes_only_protocol() -> None:
    decision = build_decision(_audit("medium_large", 20), _audit("large", 0))
    assert decision["decision"] == "no_existing_candidate_passes_cross_scale_s1"
    assert decision["existing_candidate_selected"] is None
    assert decision["original_s2_authorized"] is False
    assert decision["full_grid_authorized"] is False
    assert decision["attempt4_authorized"] is False
    assert decision["next_authorized_stage"] == "fairness_large_final_remediation_protocol_only"
    assert decision["remediation_requirements"]["maximum_new_algorithm_candidates"] == 1


def test_decision_rejects_unexpected_large_certification() -> None:
    with pytest.raises(CrossScaleAuditError, match="decision facts"):
        build_decision(_audit("medium_large", 20), _audit("large", 1))


def test_frozen_matrix_coverage_and_summary_are_complete() -> None:
    matrix = _matrix()
    validation = validate_run_matrix(matrix)
    assert validation["row_count"] == 54
    assert validation["unique_run_key_count"] == 54
    assert validation["cross_scale"] == {"baseline": 6, "frontier": 48, "total": 54}
    summary = experiment_matrix_summary(matrix)
    assert set(EXPERIMENT_MATRIX_SUMMARY_FIELDS) <= set(summary[0])
    cross = {
        row["candidate"]: row
        for row in summary
        if row["scale"] == "ALL_SCALES"
        and row["task_type"] == "frontier"
        and row["candidate"] in EXPECTED_CANDIDATES
    }
    assert cross["single_cut"]["certified_count"] == 6
    assert cross["persistent_certified_cache"]["certified_count"] == 2
    assert all(row["planned_count"] == 12 for row in cross.values())


def test_freeze_artifacts_are_byte_deterministic_and_record_both_sources(
    tmp_path: Path,
) -> None:
    medium = _audit("medium_large", 20)
    large = _audit("large", 0)
    rows = []
    for scale in ("medium_large", "large"):
        for candidate in EXPECTED_CANDIDATES:
            row = {
                "medium_large_source_archive_sha256": MEDIUM_LARGE_ARCHIVE_SHA256,
                "large_source_archive_sha256": LARGE_ARCHIVE_SHA256,
                "scale": scale,
                "candidate": candidate,
            }
            rows.append(row)
    first = tmp_path / "first"
    second = tmp_path / "second"
    matrix = _matrix()
    hashes_first = write_freeze_artifacts(first, medium, large, rows, matrix)
    hashes_second = write_freeze_artifacts(
        second, medium, large, list(reversed(rows)), list(reversed(matrix))
    )
    assert hashes_first == hashes_second
    # Production passes a canonically ordered summary; normalize the synthetic order here.
    write_rows = sorted(rows, key=lambda item: (item["scale"], item["candidate"]))
    third = tmp_path / "third"
    fourth = tmp_path / "fourth"
    hashes_third = write_freeze_artifacts(third, medium, large, write_rows, matrix)
    hashes_fourth = write_freeze_artifacts(fourth, medium, large, write_rows, matrix)
    assert hashes_third == hashes_fourth
    assert {path.name for path in third.iterdir()} == {
        "decision.json", "source_archive_provenance.json", "medium_large_audit.json",
        "large_audit.json", "cross_scale_candidate_summary.csv", "frozen_run_matrix.csv",
        "experiment_matrix_summary.csv", "artifact_sha256.csv",
    }
    with (third / "artifact_sha256.csv").open(encoding="utf-8", newline="") as handle:
        artifact_rows = list(csv.DictReader(handle))
    assert len(artifact_rows) == 7
    assert all(row["medium_large_source_archive_sha256"] == MEDIUM_LARGE_ARCHIVE_SHA256 for row in artifact_rows)
    assert all(row["large_source_archive_sha256"] == LARGE_ARCHIVE_SHA256 for row in artifact_rows)
    assert json.loads((third / "decision.json").read_text(encoding="utf-8"))["data_integrity_valid"] is True
    assert file_sha256(third / "artifact_sha256.csv") == hashes_third["artifact_sha256.csv"]
