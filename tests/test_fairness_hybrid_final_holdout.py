from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.fairness_hybrid_final_holdout_audit import (
    D2_ARCHIVE_SHA256,
    audit_d2_archive,
    write_freeze_evidence,
)
from src.fairness_hybrid_final_holdout_static_audit import audit as static_audit
import src.fairness_hybrid_final_holdout_runner as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/fairness_hybrid_final_cross_scale_holdout.yaml"
D2_ARCHIVE = Path(r"E:\论文代码\fairness_hybrid_ccg_benders_d2_a3_large_results.zip")


def test_final_holdout_plan_is_exact_and_unique() -> None:
    rows = runner.expand_plan()
    assert len(rows) == len({row["run_key"] for row in rows}) == 120
    assert {row["scale"] for row in rows} == {"medium_large", "large"}
    for scale in ("medium_large", "large"):
        scale_rows = [row for row in rows if row["scale"] == scale]
        assert sum(row["task_type"] == "baseline" for row in scale_rows) == 10
        assert sum(row["task_type"] == "frontier" for row in scale_rows) == 50
        for seed in range(170, 180):
            seed_rows = [row for row in scale_rows if row["seed"] == seed]
            assert sum(row["task_type"] == "baseline" for row in seed_rows) == 1
            assert sum(row["task_type"] == "frontier" for row in seed_rows) == 5


def test_dry_run_is_solver_free_side_effect_free_and_portable() -> None:
    outputs = [ROOT / value["output_dir"] for value in runner.SCALES.values()]
    assert not any(path.exists() for path in outputs)
    report = runner.dry_run(CONFIG)
    assert report["baseline"] == 20 and report["frontier"] == 100 and report["total"] == 120
    assert report["instances_generated"] is report["solver_called"] is False
    assert report["output_dirs_exist"] is False
    assert report["windows_path_check"] is True and report["longest_path_length"] < 220
    assert report["reserved_seed_access_audit_passed"] is True
    assert report["independent_unit"] == "seed" and report["seed_cluster_count"] == 10
    assert not any(path.exists() for path in outputs)


def test_formal_execution_fails_before_any_output() -> None:
    outputs = [ROOT / value["output_dir"] for value in runner.SCALES.values()]
    with pytest.raises(Exception, match="formal_run_not_authorized"):
        runner.run_holdout(CONFIG)
    assert not any(path.exists() for path in outputs)


@pytest.mark.parametrize("stage", ["D1", "D2", "L0", "L1", "M1", "S2", "full-grid"])
def test_other_stages_cannot_borrow_holdout_protocol(stage: str) -> None:
    with pytest.raises(Exception, match="only FINAL_HOLDOUT"):
        runner.main(["--config", str(CONFIG), "--stage", stage, "--dry-run"])


def test_seed_access_audit_distinguishes_declaration_from_evidence(tmp_path: Path) -> None:
    declaration = tmp_path / "declaration.json"
    declaration.write_text(json.dumps({"reserved_seeds": list(range(170, 180))}), encoding="utf-8")
    assert runner.structured_seed_access_evidence(tmp_path) == []
    evidence = tmp_path / "run.json"
    evidence.write_text(json.dumps({"seed": 170, "run_key": "formal-run"}), encoding="utf-8")
    assert runner.structured_seed_access_evidence(tmp_path) == ["run.json"]


def test_statistics_freeze_seed_as_independent_unit() -> None:
    config = runner.load_config(CONFIG)
    statistics = config["statistics"]
    assert statistics["independent_unit"] == "seed"
    assert statistics["cluster_members"] == ["scale", "rho"]
    assert statistics["cluster_bootstrap_resamples_seed"] is True
    assert statistics["per_rho_cross_scale_pair_count"] == 10
    assert statistics["multiple_testing"] == "Holm_if_five_rho_tests"
    assert statistics["prohibit_seed_rho_independence"] is True


def test_protocol_evidence_rebuild_is_deterministic(tmp_path: Path) -> None:
    first = runner.write_protocol_evidence(CONFIG, tmp_path / "first")
    second = runner.write_protocol_evidence(CONFIG, tmp_path / "second")
    assert first == second
    assert len((tmp_path / "first/frozen_run_plan.csv").read_text(encoding="utf-8").splitlines()) == 121


@pytest.mark.skipif(not D2_ARCHIVE.exists(), reason="formal D2 archive is not available")
def test_real_d2_archive_and_deterministic_freeze(tmp_path: Path) -> None:
    before = D2_ARCHIVE.read_bytes()
    result = audit_d2_archive(D2_ARCHIVE)
    assert result["status"] == "pass"
    assert result["source_archive_sha256"] == D2_ARCHIVE_SHA256
    assert result["baseline_certified_count"] == 3
    assert result["frontier_certified_count"] == 9
    assert result["post_evaluation_chunk_count"] == 1683
    assert result["post_evaluation_scenario_total"] == 9 * 4657
    first = write_freeze_evidence(result, tmp_path / "first")
    second = write_freeze_evidence(result, tmp_path / "second")
    assert first == second
    assert D2_ARCHIVE.read_bytes() == before


def test_committed_d2_decision_is_locked_to_archive() -> None:
    decision = json.loads(
        (ROOT / "analysis/fairness_hybrid_ccg_benders_d2_decision/decision.json").read_text(encoding="utf-8")
    )
    assert decision["decision"] == "approve_final_cross_scale_holdout_protocol"
    assert decision["source_archive_sha256"] == D2_ARCHIVE_SHA256
    assert decision["algorithm_development_closed"] is True
    assert decision["final_holdout_formal_run_authorized"] is False


def test_static_audit_passes() -> None:
    result = static_audit(ROOT)
    assert result["status"] == "pass" and result["passed"] == result["total"]
