from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.fairness_benchmark_anchor_reporting import (
    ReportingAuditError,
    _run_directory,
    _stats,
    _vector_l1,
    audit_benchmark,
)


def test_canonical_short_directory() -> None:
    key = '{"candidate":"x","seed":1}'
    assert _run_directory(key) == "r_426c1ee1f7b53c37a7ad89b4"


def test_statistics_use_sample_standard_deviation_and_linear_quartiles() -> None:
    result = _stats([1, 2, 3, 4, 5])
    assert result == {"mean": 3.0, "median": 3.0, "std": pytest.approx(math.sqrt(2.5)),
                      "iqr": 2.0, "min": 1.0, "max": 5.0}


def test_strict_nested_vector_l1() -> None:
    assert _vector_l1([[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [3.5, 4.0]], "x") == 3.5
    with pytest.raises(ReportingAuditError):
        _vector_l1([[1.0]], [[1.0, 2.0]], "x")
    with pytest.raises(ReportingAuditError):
        _vector_l1([[True]], [[1.0]], "x")


def test_formal_benchmark_archive_if_mounted() -> None:
    benchmark = Path(r"E:\论文代码\fairness_gamma_minimal_paired_benchmark_a2_results.zip")
    gamma = Path(r"E:\论文代码\fairness_hybrid_gamma_sensitivity_attempt3_results.zip")
    if not benchmark.is_file() or not gamma.is_file():
        pytest.skip("formal read-only archives are not mounted")
    audit, rows, paired = audit_benchmark(benchmark, gamma)
    assert audit["coverage"]["certified"] == 3
    assert audit["coverage"]["chunks"] == 222
    assert audit["coverage"]["acceptance_evidence"] == 60426
    assert audit["maximum_acceptance_residual"] == pytest.approx(1.1314114090055227e-9)
    assert len(rows) == len(paired) == 10
    assert all(row["master_runtime"] > 0 and row["separation_runtime"] > 0 for row in rows)
    successful = [row for row in paired if row["certification_agreement"]]
    assert len(successful) == 3
    assert max(abs(row["objective_t_difference_reference_minus_hybrid"]) for row in successful) < 1e-4
