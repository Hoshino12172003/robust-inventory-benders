from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest

from src.fairness_high_gamma_attempt2_final import (
    ARCHIVE_SHA256,
    GAMMAS,
    SEEDS,
    _directory_id,
    _run_key,
    _stats,
    audit_archive,
    compare_directories,
    generate_reports,
)


ARCHIVE = Path(
    r"E:\论文代码\fairness_high_gamma_external_solver_benchmark_attempt2_results.zip"
)


def test_frozen_plan_has_45_unique_keys_and_short_directories() -> None:
    keys = [
        _run_key(seed, gamma, task)
        for seed in SEEDS
        for gamma in GAMMAS
        for task in ("baseline", "hybrid_frontier", "direct_extensive_frontier")
    ]
    assert len(keys) == len(set(keys)) == 45
    assert len({_directory_id(key) for key in keys}) == 45
    assert all(json.loads(key)["execution_attempt"] == 2 for key in keys)


def test_descriptive_statistics_use_sample_standard_deviation_and_seed_iqr() -> None:
    result = _stats([1, 2, 3, 4, 5])
    assert result == {
        "mean": 3.0,
        "median": 3.0,
        "sample_std": pytest.approx(math.sqrt(2.5)),
        "iqr": 3.0,
        "min": 1.0,
        "max": 5.0,
    }


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="formal read-only archive is not mounted")
def test_formal_archive_solver_free_audit_and_deterministic_reporting(tmp_path: Path) -> None:
    before = ARCHIVE.read_bytes()
    audit, rows, paired = audit_archive(ARCHIVE)
    assert audit["decision"] == "approve_high_gamma_external_benchmark_attempt2"
    assert audit["archive"]["sha256_before"] == ARCHIVE_SHA256
    assert audit["coverage"]["tasks"] == 45
    assert audit["coverage"]["all_chunks"] == 3440
    assert audit["coverage"]["all_scenario_records"] == 85390
    assert audit["coverage"]["all_acceptance_evidence"] == 512380
    assert audit["certification"]["hybrid"] == {"2": 5, "3": 5, "4": 5}
    assert audit["certification"]["direct"] == {"2": 5, "3": 5, "4": 0}
    assert audit["certification"]["direct_gamma4_with_incumbent"] == 0
    assert len(rows) == 45
    assert len(paired) == 15
    assert all(
        row["direct_objective_t"] is None
        and row["direct_actual_robust_cost"] == "NOT_APPLICABLE"
        and row["direct_has_incumbent"] is False
        for row in paired
        if row["gamma"] == 4
    )
    left, right = tmp_path / "left", tmp_path / "right"
    generate_reports(ARCHIVE, left)
    generate_reports(ARCHIVE, right)
    compare_directories(left, right)
    assert ARCHIVE.read_bytes() == before
    assert "gurobipy" not in sys.modules
