from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import yaml

from src.experiment_protocol import atomic_write_json, file_sha256
import src.fairness_hybrid_gamma_sensitivity_audit as audit
import src.fairness_hybrid_gamma_sensitivity_runner as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/fairness_hybrid_gamma_sensitivity.yaml"


def test_exact_plan_has_sixty_unique_scientific_identities() -> None:
    plan = runner.expand_plan()
    assert len(plan) == 60
    assert len({row["run_key"] for row in plan}) == 60
    assert len({row["run_directory_id"] for row in plan}) == 60
    assert all(row["run_directory_id"].startswith("r_") and len(row["run_directory_id"]) == 26 for row in plan)
    assert sum(row["task_type"] == "baseline" for row in plan) == 30
    assert sum(row["task_type"] == "frontier" for row in plan) == 30
    for scale in runner.SCALES:
        rows = [row for row in plan if row["scale"] == scale]
        assert len(rows) == 30
        for seed in runner.SEEDS:
            for gamma in runner.GAMMAS:
                cell = [row for row in rows if row["seed"] == seed and row["gamma"] == gamma]
                assert {row["task_type"] for row in cell} == {"baseline", "frontier"}


def test_gamma_specific_baseline_and_anchor_pairing() -> None:
    plan = runner.expand_plan()
    frontiers = [row for row in plan if row["task_type"] == "frontier"]
    baseline_keys = set()
    for frontier in frontiers:
        baseline = runner.paired_baseline(plan, frontier)
        assert (baseline["scale"], baseline["seed"], baseline["gamma"]) == (
            frontier["scale"], frontier["seed"], frontier["gamma"]
        )
        assert baseline["rho"] == "NOT_APPLICABLE"
        baseline_keys.add(baseline["run_key"])
    assert len(baseline_keys) == 30


@pytest.mark.parametrize(
    ("scale", "expected"),
    [("medium_large", [1, 61, 1831]), ("large", [1, 97, 4657])],
)
def test_exact_gamma_scenario_counts(scale: str, expected: list[int]) -> None:
    assert [runner.scenario_count(scale, gamma) for gamma in runner.GAMMAS] == expected
    for forbidden in (3, 4):
        with pytest.raises(runner.ProtocolGateError, match="forbidden"):
            runner.scenario_count(scale, forbidden)


def test_config_gate_rejects_seed_gamma_rho_scale_and_authorization_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runner.load_config(CONFIG)
    mutations = {
        "seeds": [180],
        "gamma": [0, 1, 2, 3],
        "rho": [0.01],
        "formal_run_authorized": True,
    }
    for key, value in mutations.items():
        config = {**original, key: value}
        path = tmp_path / f"{key}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        monkeypatch.setattr(runner, "EXPECTED_CONFIG_SHA256", file_sha256(path).upper())
        with pytest.raises(runner.ProtocolGateError, match="identity drifted"):
            runner.validate_config(path, config)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["scales"].pop("large")
    path = tmp_path / "scale.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(runner, "EXPECTED_CONFIG_SHA256", file_sha256(path).upper())
    with pytest.raises(runner.ProtocolGateError, match="scale set"):
        runner.validate_config(path, config)


def test_dry_run_has_zero_side_effect_and_exact_envelopes() -> None:
    outputs = [ROOT / item["output_dir"] for item in runner.SCALES.values()]
    assert not any(path.exists() for path in outputs)
    report = runner.dry_run(CONFIG)
    assert report["scales"] == ["medium_large", "large"]
    assert report["seeds"] == [180, 181, 182, 183, 184]
    assert report["gamma"] == [0, 1, 2] and report["rho"] == [0.025]
    assert (report["baseline"], report["frontier"], report["total"]) == (30, 30, 60)
    assert report["algorithm_solver_limit_seconds"] == 108000
    assert report["post_evaluation_scenarios"] == {"medium_large": 9465, "large": 23775, "total": 33240}
    assert report["post_evaluation_solver_limit_seconds"] == 997200
    assert report["instances_generated"] is report["solver_called"] is report["output_dir_exists"] is False
    assert report["windows_path_check"] is True and report["longest_windows_path_length"] < 220
    assert not any(path.exists() for path in outputs)


def test_manifest_and_run_identity_bind_gamma_config_protocol_git_and_solver() -> None:
    plan = runner.expand_plan()
    frontier = next(row for row in plan if row["task_type"] == "frontier" and row["gamma"] == 1)
    identity = runner.run_identity(frontier, CONFIG, git_commit_value="a" * 40)
    assert identity["gamma"] == 1 and identity["rho"] == "0.025"
    assert identity["baseline_run_key"] == runner.paired_baseline(plan, frontier)["run_key"]
    assert identity["config_file_sha256"] == runner.EXPECTED_CONFIG_SHA256
    assert identity["protocol_sha256"] == runner.EXPECTED_PROTOCOL_SHA256
    assert identity["solver_parameters"] == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}
    manifest = runner.manifest_payload(CONFIG, "large", git_commit_value="a" * 40)
    assert len(manifest["run_key_to_directory_id"]) == len(manifest["directory_id_to_run_key"]) == 30
    for key, directory in manifest["run_key_to_directory_id"].items():
        assert manifest["directory_id_to_run_key"][directory] == key
    baseline_row = runner.paired_baseline(plan, frontier)
    baseline = {**baseline_row, "run_key": baseline_row["run_key"], "execution_attempt": 1}
    bound = runner.bind_data_identities(
        identity,
        instance_sha256="I" * 64,
        baseline=baseline,
        anchor={"baseline_run_key": baseline_row["run_key"], "anchor_sha256": "A" * 64, "anchor_value_hex": "0x1p+0"},
    )
    assert bound["instance_sha256"] == "I" * 64 and bound["anchor_sha256"] == "A" * 64
    with pytest.raises(runner.ProtocolGateError, match="gamma"):
        runner.bind_data_identities(
            identity,
            instance_sha256="I" * 64,
            baseline={**baseline, "gamma": 2},
            anchor={"baseline_run_key": baseline_row["run_key"], "anchor_sha256": "A" * 64},
        )


def test_strict_resume_rejects_corruption_identity_drift_and_repetition(tmp_path: Path) -> None:
    expected = runner.manifest_payload(CONFIG, "medium_large", git_commit_value="b" * 40)
    path = tmp_path / "manifest.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(runner.ProtocolGateError, match="corrupt"):
        runner.validate_resume_manifest(path, expected)
    atomic_write_json(path, {**expected, "identity": {**expected["identity"], "git_commit": "c" * 40}})
    with pytest.raises(runner.ProtocolGateError, match="identity mismatch"):
        runner.validate_resume_manifest(path, expected)
    atomic_write_json(path, expected)
    assert runner.validate_resume_manifest(path, expected) == expected
    row = next(item for item in runner.expand_plan() if item["task_type"] == "frontier")
    identity = runner.run_identity(row, CONFIG, git_commit_value="b" * 40)
    assert runner.validate_run_record({**identity, "state": "complete"}, identity) == "skip_committed_result"
    with pytest.raises(runner.ProtocolGateError, match="gamma"):
        runner.validate_run_record({**identity, "state": "running", "gamma": 2}, identity)


def test_algorithm_checkpoint_is_append_only_and_one_addition_per_iteration() -> None:
    identity = {"run_key": "x", "gamma": 1}
    iterations = [
        {"iteration": 1, "new_scenario_sha256": ["S1"], "new_cut_sha256": ["C1"]},
        {"iteration": 2, "new_scenario_sha256": [], "new_cut_sha256": []},
        {"iteration": 3, "new_scenario_sha256": ["S2"], "new_cut_sha256": ["C2"]},
    ]
    checkpoint = runner.algorithm_checkpoint(identity, iterations)
    runner.validate_algorithm_checkpoint(checkpoint, identity)
    with pytest.raises(runner.ProtocolGateError, match="at most one"):
        runner.algorithm_checkpoint(identity, [{"iteration": 1, "new_scenario_sha256": ["A", "B"]}])
    with pytest.raises(runner.ProtocolGateError, match="duplicate"):
        runner.algorithm_checkpoint(identity, [
            {"iteration": 1, "new_scenario_sha256": ["A"]},
            {"iteration": 2, "new_scenario_sha256": ["A"]},
        ])
    broken = {**checkpoint, "scenario_sha256_append_only": ["S2"]}
    with pytest.raises(runner.ProtocolGateError, match="ledger"):
        runner.validate_algorithm_checkpoint(broken, identity)


def test_atomic_manifest_status_and_checkpoint_writes(tmp_path: Path) -> None:
    manifest = runner.manifest_payload(CONFIG, "medium_large", git_commit_value="d" * 40)
    manifest_path = tmp_path / "manifest.json"
    runner.write_identity_file_once(manifest_path, manifest, label="manifest")
    runner.write_identity_file_once(manifest_path, manifest, label="manifest")
    assert not (tmp_path / ".manifest.json.tmp").exists()
    with pytest.raises(runner.ProtocolGateError, match="different identity"):
        runner.write_identity_file_once(manifest_path, {**manifest, "schema": "drift"}, label="manifest")
    identity = {"run_key": "x", "gamma": 1}
    checkpoint = runner.algorithm_checkpoint(identity, [{"iteration": 1, "new_scenario_sha256": [], "new_cut_sha256": []}])
    runner.write_algorithm_checkpoint(tmp_path / "algorithm_checkpoint.json", checkpoint, identity)
    runner.write_run_status(tmp_path / "status.json", identity, state="running", scientific_status="pending")
    assert not any(path.name.endswith(".tmp") for path in tmp_path.iterdir())


def test_post_evaluation_chunk_resume_checks_order_count_identity_and_sha(tmp_path: Path) -> None:
    identity = {"run_key": "x", "gamma": 0}
    chunks = [
        runner.post_chunk(identity, 0, 0, [{"scenario": 0}, {"scenario": 1}]),
        runner.post_chunk(identity, 1, 2, [{"scenario": 2}]),
    ]
    index = runner.validate_post_evaluation_chunks(chunks, identity, 3)
    assert index["complete"] is True and index["scenario_count"] == 3
    progress = runner.resume_post_evaluation_chunks(chunks[:1], identity, 3)
    assert progress == {
        "next_chunk_index": 1,
        "next_scenario_index": 2,
        "chunk_sha256": [chunks[0]["chunk_sha256"]],
        "complete": False,
    }
    chunk_path = tmp_path / "chunk_00000.json"
    runner.write_post_chunk(chunk_path, chunks[0])
    runner.write_post_chunk(chunk_path, chunks[0])
    runner.write_post_index(tmp_path / "index.json", index)
    assert not any(path.name.endswith(".tmp") for path in tmp_path.iterdir())
    corrupt = json.loads(json.dumps(chunks))
    corrupt[0]["records"][0]["scenario"] = 99
    with pytest.raises(runner.ProtocolGateError, match="SHA"):
        runner.validate_post_evaluation_chunks(corrupt, identity, 3)
    with pytest.raises(runner.ProtocolGateError, match="total"):
        runner.validate_post_evaluation_chunks(chunks, identity, 4)
    reversed_chunks = list(reversed(chunks))
    with pytest.raises(runner.ProtocolGateError, match="order"):
        runner.validate_post_evaluation_chunks(reversed_chunks, identity, 3)


def test_forbidden_final_holdout_d1_d2_reuse() -> None:
    for path in (
        "experiments/results_fairness_hybrid_final_holdout/ml_a1/instances/170.json",
        "analysis/fairness_hybrid_ccg_benders_d1/run.json",
        "analysis/fairness_hybrid_ccg_benders_d2/checkpoint.json",
    ):
        with pytest.raises(runner.ProtocolGateError, match="may not be reused"):
            runner.reject_reuse_path(path)
    runner.reject_reuse_path("experiments/results_fh_gamma/ml_a1/run.json")


def _valid_result_row() -> dict[str, object]:
    row = {field: 0 for field in runner.RESULT_FIELDS}
    planned = next(item for item in runner.expand_plan() if item["task_type"] == "frontier")
    baseline = runner.paired_baseline(runner.expand_plan(), planned)
    row.update({
        **planned,
        "execution_attempt": 1, "git_commit": "G" * 40,
        "config_file_sha256": runner.EXPECTED_CONFIG_SHA256,
        "protocol_sha256": runner.EXPECTED_PROTOCOL_SHA256,
        "candidate_sha256": runner.CANDIDATE_SHA256,
        "instance_sha256": "I" * 64, "baseline_run_key": baseline["run_key"], "anchor_sha256": "A" * 64,
        "state": "complete", "algorithm_status": "optimal",
        "scientific_status": "certified_robust_optimal", "objective_t": 0.2,
        "robust_minimum_fill_rate": 0.8, "wminfr": 0.8,
        "minimum_weighted_mean_fill_rate": 0.9,
    })
    return row


def test_csv_fields_and_three_fill_rate_semantics(tmp_path: Path) -> None:
    row = _valid_result_row()
    runner.validate_result_row(row)
    runner.write_results(tmp_path / "results.csv", [row])
    runner.write_summary(tmp_path / "summary.csv", [row])
    assert not (tmp_path / ".results.csv.tmp").exists()
    summary = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "medium_large,30,1,0,1,1" in summary
    missing = dict(row)
    missing.pop("wminfr")
    with pytest.raises(runner.ProtocolGateError, match="fields missing"):
        runner.validate_result_row(missing)
    crossing = dict(row)
    crossing["robust_minimum_fill_rate"] = crossing["wminfr"]
    crossing["objective_t"] = 0.3
    with pytest.raises(runner.ProtocolGateError, match="must equal 1-T"):
        runner.validate_result_row(crossing)
    protocol = (ROOT / "docs/fairness_hybrid_gamma_sensitivity_protocol.md").read_text(encoding="utf-8")
    assert "robust_minimum_fill_rate = 1 - T" in protocol
    assert "`wminfr`: exact post-evaluation minimum" in protocol
    assert "system demand-weighted mean fill rate" in protocol


def test_scientific_status_and_par2_use_algorithm_runtime() -> None:
    certified = {
        "algorithm_status": "optimal", "robust_feasibility_certified": True,
        "final_exact_separation_performed": True, "final_exact_separation_status": "optimal",
        "final_exact_separation_objective_bound": -0.0, "post_evaluation_valid": True,
    }
    assert runner.classify_scientific_status(certified) == "certified_robust_optimal"
    assert runner.par2("certified_robust_optimal", 12.5) == 12.5
    for key in ("robust_feasibility_certified", "final_exact_separation_performed", "post_evaluation_valid"):
        invalid = {**certified, key: False}
        assert runner.classify_scientific_status(invalid) == "robust_uncertified"
        assert runner.par2("robust_uncertified", 12.5) == 3600.0
    assert runner.classify_scientific_status({**certified, "final_exact_separation_objective_bound": 0.01}) == "robust_uncertified"


def test_formal_entrypoint_fails_before_outputs_or_solver() -> None:
    outputs = [ROOT / value["output_dir"] for value in runner.SCALES.values()]
    assert not any(path.exists() for path in outputs)
    with pytest.raises(runner.ProtocolGateError, match="strict --resume"):
        runner.formal_run(CONFIG, resume=False)
    with pytest.raises(runner.ProtocolGateError, match="only pre-run audit"):
        runner.formal_run(CONFIG, resume=True)
    assert not any(path.exists() for path in outputs)


@pytest.mark.parametrize("stage", ["FINAL_HOLDOUT", "D1", "D2", "full-grid"])
def test_wrong_stage_is_rejected(stage: str) -> None:
    with pytest.raises(runner.ProtocolGateError, match="only GAMMA_SENSITIVITY"):
        runner.main(["--config", str(CONFIG), "--stage", stage, "--dry-run"])


def test_overwrite_option_is_not_supported() -> None:
    with pytest.raises(SystemExit):
        runner.main(["--config", str(CONFIG), "--stage", runner.STAGE, "--overwrite"])


def test_seed_access_audit_distinguishes_registration_and_actual_access(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    config_dir = tmp_path / "experiments/configs"
    config_dir.mkdir(parents=True)
    declaration = config_dir / "registered.yaml"
    declaration.write_text("seeds: [180, 181, 182, 183, 184]\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    report = audit.audit_repository_seed_access(tmp_path)
    assert report["audit_passed"] is True
    instance = tmp_path / "instances/s180_g0.json"
    instance.parent.mkdir()
    instance.write_text(json.dumps({"seed": 180, "instance_sha256": "A" * 64}), encoding="utf-8")
    report = audit.audit_repository_seed_access(tmp_path)
    assert report["audit_passed"] is False
    assert report["generated_instance_evidence"]


def test_protocol_and_config_hashes_are_frozen() -> None:
    assert file_sha256(CONFIG).upper() == runner.EXPECTED_CONFIG_SHA256
    assert file_sha256(ROOT / "docs/fairness_hybrid_gamma_sensitivity_protocol.md").upper() == runner.EXPECTED_PROTOCOL_SHA256


def test_static_audit_passes_without_gurobi() -> None:
    result = audit.static_audit(ROOT, None)
    assert result["status"] == "pass" and result["passed"] == result["total"]
    assert result["seed_access_audit"]["audit_passed"] is True
