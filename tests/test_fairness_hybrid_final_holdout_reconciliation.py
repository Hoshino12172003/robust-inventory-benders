from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from src.fairness_hybrid_final_holdout_reconciliation import (
    ARCHIVE_SHA256,
    TOLERANCE,
    audit_archive,
    corrected_gap,
    reconcile_bounds,
    write_reports,
)


def test_bound_crossing_is_explicit_and_final_bound_is_distinct() -> None:
    result = {
        "lower_bound": 0.20034335381719642,
        "upper_bound": 0.20032265941243410,
        "gap": 0.0,
        "iteration_log": [
            {
                "master_solver_best_bound": 0.20032265941243410,
                "separation_objective_bound": -0.0,
            }
        ],
    }
    reconciled = reconcile_bounds(result)
    assert reconciled["raw_bound_crossing"] == pytest.approx(2.069440476232e-5)
    assert reconciled["bound_crossing_within_tolerance"] is True
    assert reconciled["historical_recorded_lower_bound"] != reconciled["reported_certification_lower_bound"]
    assert reconciled["reported_certification_lower_bound"] == reconciled["upper_bound"]
    assert reconciled["reported_gap"] == 0.0


def test_gap_clipping_cannot_hide_unrecorded_crossing() -> None:
    upper = 0.2
    historical = 0.20002
    assert corrected_gap(upper, historical) == 0.0
    raw_crossing = historical - upper
    assert raw_crossing > 0.0
    assert raw_crossing <= TOLERANCE


def test_corrected_gap_uses_frozen_denominator() -> None:
    assert corrected_gap(0.75, 0.70) == pytest.approx(0.05)
    assert corrected_gap(2.0, 1.5) == pytest.approx(0.25)


def test_formal_archive_full_audit_and_deterministic_reports(tmp_path: Path) -> None:
    archive_value = os.environ.get("FINAL_HOLDOUT_ARCHIVE")
    if not archive_value:
        pytest.skip("set FINAL_HOLDOUT_ARCHIVE to run the read-only formal archive audit")
    archive = Path(archive_value)
    before = hashlib.sha256(archive.read_bytes()).hexdigest().upper()
    audit, records = audit_archive(archive, Path(__file__).resolve().parents[1])
    assert audit["status"] == "pass"
    assert audit["run_count"] == 120
    assert audit["baseline_count"] == 20
    assert audit["frontier_count"] == audit["exact_certification_count"] == 100
    assert audit["post_evaluation_index_count"] == 100
    assert audit["post_evaluation_chunk_count"] == 13050
    assert audit["post_evaluation_scenario_record_count"] == 324400
    assert audit["chunk_sha_error_count"] == 0
    assert audit["bound_crossing_count"] == 1
    assert audit["archive_sha256_before"] == audit["archive_sha256_after"] == ARCHIVE_SHA256
    assert before == hashlib.sha256(archive.read_bytes()).hexdigest().upper() == ARCHIVE_SHA256

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_hashes = write_reports(audit, records, first)
    second_hashes = write_reports(audit, records, second)
    assert first_hashes == second_hashes
    assert {path.name for path in first.iterdir()} == {path.name for path in second.iterdir()}
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()

    results_header = (first / "results.corrected.csv").read_text(encoding="utf-8").splitlines()[0]
    for field in (
        "historical_recorded_lower_bound",
        "final_master_solver_best_bound",
        "reported_certification_lower_bound",
        "raw_bound_crossing",
        "bound_crossing_within_tolerance",
        "reported_gap",
        "final_exact_separation_objective_bound",
    ):
        assert field in results_header
