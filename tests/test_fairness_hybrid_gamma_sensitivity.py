from __future__ import annotations

import json
from copy import deepcopy
import math
from pathlib import Path
import subprocess

import pytest
import yaml

from src.experiment_protocol import atomic_write_json, file_sha256
import src.fairness_hybrid_gamma_sensitivity_audit as audit
import src.fairness_hybrid_gamma_sensitivity_runner as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/fairness_hybrid_gamma_sensitivity.yaml"


def _detached_git_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "gamma-review@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Gamma Review"], check=True)
    (path / ".gitignore").write_text("/experiments/results_fh_gamma/\n", encoding="utf-8")
    (path / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", ".gitignore", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True)
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(path), "update-ref", "refs/remotes/origin/main", head], check=True)
    subprocess.run(["git", "-C", str(path), "switch", "--detach", "-q", head], check=True)
    return path


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


@pytest.mark.parametrize("scale", ["medium_large", "large"])
@pytest.mark.parametrize("gamma", [0, 1, 2])
def test_baseline_solver_template_is_gamma_specific(scale: str, gamma: int) -> None:
    config = runner._scale_config(runner.load_config(CONFIG), scale, gamma)
    resolved = runner.gamma_baseline_template({"gamma_target": 99, "gamma_schedule": [99]}, config)
    assert resolved["gamma_target"] == gamma
    assert resolved["gamma_schedule"] == [gamma]
    assert resolved["gamma_continuation_enabled"] is False
    assert resolved["exact_scenarios"] is True
    assert resolved["max_scenarios"] == runner.scenario_count(scale, gamma)
    assert resolved["baseline_time_limit"] == resolved["time_limit"] == 1800.0


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


def test_path_plan_covers_every_persistent_and_atomic_artifact() -> None:
    config = runner.load_config(CONFIG)
    outputs = [ROOT / item["output_dir"] for item in runner.SCALES.values()]
    paths = runner._planned_paths(ROOT, runner.expand_plan(), config)
    kinds = {kind for kind, _path in paths}
    assert len(paths) >= 560
    assert {
        "medium_large_manifest_tmp", "large_results_tmp", "baseline_checkpoint",
        "baseline_checkpoint_tmp", "algorithm_checkpoint_tmp", "post_final_tmp",
        "post_index_tmp", "post_chunk", "post_chunk_tmp", "instance_tmp",
    }.issubset(kinds)
    report = runner.dry_run(CONFIG, worktree_root=Path(config["formal_worktree_root"]))
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


def test_dry_run_uses_the_operational_worktree_root(tmp_path: Path) -> None:
    operational_root = Path(r"E:\rfgs2") if Path.cwd().drive else tmp_path / "rfgs2"
    report = runner.dry_run(CONFIG, worktree_root=operational_root)
    resolved = operational_root.resolve()
    assert report["planned_worktree_root"] == str(resolved)
    assert report["longest_windows_absolute_path"].startswith(str(resolved))
    assert report["windows_path_check"] is True
    if Path.cwd().drive:
        assert report["longest_windows_path_length"] == 124


def test_formal_run_does_not_treat_worktree_path_as_scientific_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OperationalRootAccepted(RuntimeError):
        pass

    monkeypatch.setattr(runner, "formal_git_gate", lambda _root: "F" * 40)
    monkeypatch.setattr(
        runner, "validate_authorization",
        lambda *_args: {"optimization_commit": "O" * 40, "reporting_hotfix_commit": "R" * 40},
    )
    monkeypatch.setattr(
        runner, "pre_run_seed_gate", lambda _root: {"actual_access_evidence_count": 0},
    )
    monkeypatch.setattr(
        runner, "_planned_paths", lambda *_args: [("short", Path(r"E:\rfgs2\x"))],
    )
    monkeypatch.setattr(
        runner, "production_dependencies",
        lambda: (_ for _ in ()).throw(OperationalRootAccepted()),
    )
    with pytest.raises(OperationalRootAccepted):
        runner.run_sensitivity(
            CONFIG, resume=True, authorization_file=Path("authorization.json"),
        )


def test_authorization_locks_reporting_only_successor_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "fairness_hybrid_gamma_sensitivity.yaml"
    config.write_bytes(CONFIG.read_bytes())
    authorization = json.loads(
        (ROOT / "experiments/configs/fairness_hybrid_gamma_sensitivity_authorization.json").read_text(encoding="utf-8"),
    )
    authorization.update({
        "optimization_core_commit": runner.OPTIMIZATION_CORE_COMMIT,
        "optimization_commit": runner.REPORTING_HOTFIX_BASE_COMMIT,
        "reporting_hotfix_base_commit": runner.REPORTING_HOTFIX_BASE_COMMIT,
        "reporting_hotfix_commit": "a" * 40,
        "reporting_only_changed_files": list(runner.REPORTING_ONLY_CHANGED_FILES),
    })
    authorization_path = tmp_path / "authorization.json"
    atomic_write_json(authorization_path, authorization)

    def accepted_git(_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        stdout = ""
        if args[:2] == ("diff", "--name-only"):
            if args[2] == f"{runner.REPORTING_HOTFIX_BASE_COMMIT}..{'a' * 40}":
                stdout = "\n".join(runner.REPORTING_ONLY_CHANGED_FILES) + "\n"
            else:
                stdout = runner.AUTHORIZATION_RELATIVE_PATH + "\n"
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(runner, "_git", accepted_git)
    assert runner.validate_authorization(authorization_path, config, tmp_path)["reporting_hotfix_commit"] == "a" * 40
    authorization["reporting_only_changed_files"] = ["src/benders.py"]
    atomic_write_json(authorization_path, authorization)
    with pytest.raises(runner.ProtocolGateError, match="reporting_only_changed_files"):
        runner.validate_authorization(authorization_path, config, tmp_path)


def test_manifest_and_run_identity_bind_gamma_config_protocol_git_and_solver() -> None:
    plan = runner.expand_plan()
    frontier = next(row for row in plan if row["task_type"] == "frontier" and row["gamma"] == 1)
    identity = runner.run_identity(frontier, CONFIG, git_commit_value="a" * 40)
    assert identity["gamma"] == 1 and identity["rho"] == "0.025"
    assert identity["baseline_run_key"] == runner.paired_baseline(plan, frontier)["run_key"]
    assert identity["config_file_sha256"] == runner.EXPECTED_CONFIG_SHA256
    assert identity["resolved_config_file_sha256"] == runner.EXPECTED_CONFIG_SHA256
    assert identity["protocol_sha256"] == runner.EXPECTED_PROTOCOL_SHA256
    assert identity["solver_parameters"] == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}
    manifest = runner.manifest_payload(CONFIG, "large", git_commit_value="a" * 40)
    assert len(manifest["run_key_to_directory_id"]) == len(manifest["directory_id_to_run_key"]) == 30
    for key, directory in manifest["run_key_to_directory_id"].items():
        assert manifest["directory_id_to_run_key"][directory] == key
    baseline_row = runner.paired_baseline(plan, frontier)
    instance_sha = "I" * 64
    identity_sha = "D" * 64
    shared = {
        "scale": identity["scale"], "seed": identity["seed"], "gamma": identity["gamma"],
        "execution_attempt": runner.EXECUTION_ATTEMPT,
        "instance_sha256": instance_sha,
        "instance_canonical_sha256": instance_sha,
        "instance_identity_sha256": identity_sha,
    }
    baseline = {**baseline_row, **shared, "run_key": baseline_row["run_key"]}
    anchor = {
        **shared, "baseline_run_key": baseline_row["run_key"],
        "anchor_sha256": "A" * 64, "anchor_value_hex": "0x1p+0",
    }
    bound = runner.bind_data_identities(
        identity,
        instance_canonical_sha256=instance_sha,
        instance_identity_sha256=identity_sha,
        baseline=baseline,
        anchor=anchor,
    )
    assert bound["instance_canonical_sha256"] == instance_sha and bound["anchor_sha256"] == "A" * 64
    with pytest.raises(runner.ProtocolGateError, match="gamma"):
        runner.bind_data_identities(
            identity,
            instance_canonical_sha256=instance_sha,
            instance_identity_sha256=identity_sha,
            baseline={**baseline, "gamma": 2},
            anchor=anchor,
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
    with pytest.raises(runner.ProtocolGateError, match="identity or state"):
        runner.validate_status_file(tmp_path / "status.json", {"run_key": "drift", "gamma": 1}, None)


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
        "experiments/results_fh_gamma/ml_a1/run.json",
        "experiments/results_fh_gamma/lg_a1/run.json",
        "experiments/results_fh_gamma/ml_a2/run.json",
        "experiments/results_fh_gamma/lg_a2/run.json",
    ):
        with pytest.raises(runner.ProtocolGateError, match="may not be reused"):
            runner.reject_reuse_path(path)
    runner.reject_reuse_path("experiments/results_fh_gamma/ml_a3/run.json")


def _valid_result_row() -> dict[str, object]:
    row = {field: 0 for field in runner.RESULT_FIELDS}
    planned = next(item for item in runner.expand_plan() if item["task_type"] == "frontier")
    baseline = runner.paired_baseline(runner.expand_plan(), planned)
    row.update({
        **planned,
        "execution_attempt": runner.EXECUTION_ATTEMPT, "git_commit": "G" * 40,
        "config_file_sha256": runner.EXPECTED_CONFIG_SHA256,
        "resolved_config_file_sha256": runner.EXPECTED_CONFIG_SHA256,
        "protocol_sha256": runner.EXPECTED_PROTOCOL_SHA256,
        "candidate_sha256": runner.CANDIDATE_SHA256,
        "instance_sha256": "A" * 64, "instance_canonical_sha256": "A" * 64,
        "instance_identity_sha256": "D" * 64,
        "baseline_run_key": baseline["run_key"], "anchor_sha256": "A" * 64,
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
    with pytest.raises(runner.ProtocolGateError, match="resolved_config_file_sha256"):
        runner.validate_result_row({**row, "resolved_config_file_sha256": "missing"})
    with pytest.raises(runner.ProtocolGateError, match="canonical instance identity drift"):
        runner.validate_result_row({**row, "instance_canonical_sha256": "C" * 64})
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


def test_frontier_status_requires_complete_certificate_and_post_evaluation() -> None:
    result = {
        "status": "optimal", "gap": 0.0,
        "metadata": {
            "robust_feasibility_certified": True,
            "full_separation_objective_bound_required": True,
        },
        "iteration_log": [{
            "final_exact_separation_performed": True,
            "robust_feasibility_certified": True,
            "master_status": "optimal", "separation_objective_bound": -0.0,
        }],
    }
    post = {"valid": True, "errors": [], "objective_t_consistent": True, "scenario_count": 61}
    assert runner._frontier_status(result, post, 61, 1e-4) == "certified_robust_optimal"
    drifted = deepcopy(result)
    drifted["metadata"]["robust_feasibility_certified"] = False
    assert runner._frontier_status(drifted, post, 61, 1e-4) == "robust_uncertified"
    assert runner._frontier_status(result, {**post, "errors": None}, 61, 1e-4) == "invalid_post_evaluation"
    assert runner._frontier_status(result, {**post, "objective_t_consistent": None}, 61, 1e-4) == "invalid_post_evaluation"


def test_formal_entrypoint_fails_before_outputs_or_solver() -> None:
    outputs = [ROOT / value["output_dir"] for value in runner.SCALES.values()]
    assert not any(path.exists() for path in outputs)
    with pytest.raises(runner.ProtocolGateError, match="strict --resume"):
        runner.formal_run(CONFIG, resume=False)
    with pytest.raises(runner.ProtocolGateError, match="reviewed authorization file"):
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


def test_seed_access_audit_detects_nonstructured_paths_and_zip_members(tmp_path: Path) -> None:
    import zipfile

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    artifact = tmp_path / "experiments/results/run_seed-181.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"opaque")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    report = audit.audit_repository_seed_access(tmp_path)
    assert report["audit_passed"] is False
    assert any(item["location"] == "path" for item in report["solved_run_evidence"])
    archive = tmp_path / "evidence.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("runs/s182_g1/checkpoint.bin", b"opaque")
    zip_report = audit.audit_zip_seed_access(archive)
    assert zip_report["audit_passed"] is False
    assert zip_report["target_seed_records"][0]["location"] == "path"


def _fake_dependencies(calls: dict[str, int]) -> runner.GammaDependencies:
    def generate(config: dict[str, object], seed: int) -> dict[str, object]:
        calls["generate"] += 1
        warehouses = 2 if config["scale"] == "medium_large" else 3
        products = 2 if config["scale"] == "medium_large" else 3
        return {
            "seed": seed, "gamma": config["gamma_value"], "scale": config["scale"],
            "num_warehouses": warehouses, "num_products": products,
        }

    def baseline(config: dict[str, object], instance: object, seed: int, solver: dict[str, object]) -> dict[str, object]:
        calls["baseline"] += 1
        payload = dict(instance)
        warehouses = int(payload["num_warehouses"])
        products = int(payload["num_products"])
        return {
            "status": "optimal", "valid_UB": True,
            "upper_bound": 100.0 + int(config["gamma_value"]), "gap": 0.0,
            "runtime": 1.0, "master_runtime": 0.6, "subproblem_runtime": 0.4,
            "iterations": 1,
            "best_x_values": [[1.0 for _ in range(products)] for _ in range(warehouses)],
            "best_y_values": [1.0 for _ in range(warehouses)],
        }

    def anchor(record: dict[str, object], *, common_identity: dict[str, object], tolerance: float) -> dict[str, object]:
        value = float(record["result"]["upper_bound"])
        payload = {**common_identity, "baseline_run_key": record["run_key"], "value": value, "value_hex": value.hex()}
        return {**payload, "anchor_sha256": runner.sha256_value(payload), "anchor_value_hex": value.hex()}

    def frontier(
        config: dict[str, object], instance: object, baseline_record: dict[str, object], anchor_value: dict[str, object],
        common: dict[str, object], checkpoint: Path, solver: dict[str, object], row: dict[str, object],
    ) -> dict[str, object]:
        calls["frontier"] += 1
        atomic_write_json(checkpoint, runner.algorithm_checkpoint({"run_key": row["run_key"]}, []))
        gamma = int(row["gamma"])
        payload = dict(instance)
        warehouses = int(payload["num_warehouses"])
        products = int(payload["num_products"])
        return {
            "status": "optimal", "gap": 0.0, "lower_bound": 100.0, "upper_bound": 100.0,
            "runtime": 2.0, "objective_t": 0.2 + gamma * 0.01,
            "robust_minimum_fill_rate": 0.8 - gamma * 0.01,
            "master_runtime": 1.25, "separation_runtime": 0.75,
            "x_values": [[10.0 + gamma for _ in range(products)] for _ in range(warehouses)],
            "y_values": [1.0 for _ in range(warehouses)], "iterations": 1, "cuts": gamma,
            "metadata": {"robust_feasibility_certified": True, "full_separation_objective_bound_required": True, "committed_scenario_count": gamma + 1},
            "iteration_log": [{"final_exact_separation_performed": True, "robust_feasibility_certified": True, "master_status": "optimal", "separation_objective_bound": -0.0}],
        }

    def post(
        config: dict[str, object], instance: object, result: dict[str, object], anchor_value: dict[str, object],
        identity: dict[str, object], post_root: Path, row: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, float]]:
        calls["post"] += 1
        count = runner.scenario_count(str(row["scale"]), int(row["gamma"]))
        atomic_write_json(post_root / "final.json", {"run_key": row["run_key"], "scenario_count": count, "valid": True})
        return ({
            "valid": True, "errors": [], "objective_t_consistent": True, "scenario_count": count,
            "actual_robust_cost": float(anchor_value["value"]), "actual_price_of_fairness": 0.0,
            "wminfr": float(result["robust_minimum_fill_rate"]), "minimum_weighted_mean_fill_rate": 0.9,
        }, {"post_evaluation_solver_runtime": 0.2, "post_evaluation_wall_runtime": 0.3, "aggregation_runtime": 0.1, "checkpoint_io_runtime": 0.05})

    def configure(settings: dict[str, object]) -> None:
        calls["configure"] += 1
        assert settings == runner.SOLVER_PARAMETERS

    return runner.GammaDependencies(generate, lambda value: value, lambda value: value, baseline, anchor, frontier, post, configure)


def test_solver_free_full_sixty_run_pipeline_and_second_resume_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_root = _detached_git_repo(tmp_path / "g")
    calls = {name: 0 for name in ("generate", "baseline", "frontier", "post", "configure")}
    scales = deepcopy(runner.SCALES)
    for scale, short in (("medium_large", "m"), ("large", "l")):
        scales[scale]["output_dir"] = str(git_root / "experiments/results_fh_gamma" / short)
    monkeypatch.setattr(runner, "SCALES", scales)
    monkeypatch.setattr(runner, "validate_config", lambda path, config: None)
    deps = _fake_dependencies(calls)
    report = runner.run_sensitivity(
        CONFIG, resume=True, dependencies=deps, test_authorization=True, test_git_root=git_root,
    )
    assert report["completed_run_count"] == 60
    assert report["baseline_certified_count"] == report["frontier_certified_count"] == 30
    assert calls == {"generate": 30, "baseline": 30, "frontier": 30, "post": 30, "configure": 1}
    first_bytes = {}
    for scale in scales:
        output = Path(scales[scale]["output_dir"])
        results = list(__import__("csv").DictReader((output / "results.csv").open(encoding="utf-8")))
        assert len(results) == 30 and len({row["run_key"] for row in results}) == 30
        assert len(list((output / "instances").glob("*.json"))) == 15
        run_files = list((output / "runs").glob("*/run.json"))
        status_files = list((output / "runs").glob("*/status.json"))
        assert len(run_files) == len(status_files) == 30
        assert all(json.loads(path.read_text(encoding="utf-8"))["state"] == "complete" for path in status_files)
        first_bytes[scale] = {
            "results": (output / "results.csv").read_bytes(),
            "summary": (output / "summary.csv").read_bytes(),
        }
    optimization_commit = subprocess.run(
        ["git", "-C", str(git_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    successor = {
        "optimization_commit": optimization_commit,
        "reporting_hotfix_commit": "a" * 40,
        "authorization_head_commit": "b" * 40,
        "changed_files": list(runner.REPORTING_ONLY_CHANGED_FILES),
    }
    config = runner.load_config(CONFIG)
    plan = runner.expand_plan()
    for scale in scales:
        runner._run_scale(
            CONFIG, config, scale, [row for row in plan if row["scale"] == scale], deps,
            optimization_commit, reporting_successor=successor,
        )
        manifest = json.loads((Path(scales[scale]["output_dir"]) / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["identity"]["git_commit"] == optimization_commit
        assert manifest["reporting_successor"] == successor
    assert calls == {"generate": 30, "baseline": 30, "frontier": 30, "post": 30, "configure": 1}
    status = subprocess.run(
        ["git", "-C", str(git_root), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    )
    assert status.stdout == ""
    runner.run_sensitivity(
        CONFIG, resume=True, dependencies=deps, test_authorization=True, test_git_root=git_root,
    )
    assert calls == {"generate": 30, "baseline": 30, "frontier": 30, "post": 30, "configure": 2}
    for scale in scales:
        output = Path(scales[scale]["output_dir"])
        assert (output / "results.csv").read_bytes() == first_bytes[scale]["results"]
        assert (output / "summary.csv").read_bytes() == first_bytes[scale]["summary"]


def _persisted_projection_record(task_type: str) -> tuple[dict[str, object], dict[str, object]]:
    instance = {"num_warehouses": 2, "num_products": 3}
    result = {
        "status": "optimal", "runtime": 2.0, "algorithm_runtime": 2.0,
        "master_runtime": 1.25, "separation_runtime": 0.75,
        "post_evaluation_wall_runtime": 0.3, "total_wall_runtime": 2.45,
        "penalized_runtime_par2": 2.0, "iterations": 2,
    }
    if task_type == "baseline":
        result.update({
            "valid_UB": True, "upper_bound": 100.0,
            "best_x_values": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "best_y_values": [1.0, 0.0],
        })
    else:
        result.update({
            "objective_t": 0.2, "robust_minimum_fill_rate": 0.8,
            "x_values": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "y_values": [1.0, 0.0], "cuts": 2,
            "metadata": {"committed_scenario_count": 3},
            "post_evaluation": {
                "valid": True, "errors": [], "objective_t_consistent": True,
                "actual_robust_cost": 101.0, "actual_price_of_fairness": 0.01,
                "wminfr": 0.8, "minimum_weighted_mean_fill_rate": 0.9,
            },
        })
    record = {
        "task_type": task_type, "scientific_status": "certified_robust_optimal",
        "baseline_robust_cost": 100.0, "cost_budget": 102.5, "result": result,
    }
    return record, instance


@pytest.mark.parametrize("task_type", ["baseline", "frontier"])
def test_production_json_round_trip_projects_strict_first_stage_schema(
    tmp_path: Path, task_type: str,
) -> None:
    record, instance = _persisted_projection_record(task_type)
    path = tmp_path / f"{task_type}.json"
    atomic_write_json(path, record)
    loaded = runner.read_json_strict(path)
    assert loaded is not None
    row = runner._result_projection(loaded, instance)
    assert row["inventory"] == 21.0
    assert row["opened_warehouses"] == 1
    assert row["iterations"] == 2
    if task_type == "baseline":
        assert row["objective_t"] == "NOT_APPLICABLE"
        assert row["baseline_robust_cost"] == 100.0
    else:
        assert row["objective_t"] == 0.2
        assert row["wminfr"] == 0.8


@pytest.mark.parametrize(
    "value",
    [
        [[1.0, 2.0, 3.0]],
        [[1.0, 2.0], [3.0, 4.0, 5.0]],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[True, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [["1", 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[math.nan, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[math.inf, 2.0, 3.0], [4.0, 5.0, 6.0]],
    ],
)
def test_reporting_matrix_rejects_shape_and_nonfinite_drift(value: object) -> None:
    with pytest.raises(runner.ProtocolGateError, match="reporting field"):
        runner._strict_numeric_matrix(value, 2, 3, "result.x_values")


@pytest.mark.parametrize("value", [[1.0], [1.0, 0.0, 1.0], [True, 0.0], ["1", 0.0], [math.nan, 0.0]])
def test_reporting_vector_rejects_shape_and_type_drift(value: object) -> None:
    with pytest.raises(runner.ProtocolGateError, match="reporting field"):
        runner._strict_numeric_vector(value, 2, "result.y_values")


def test_runner_to_production_frontier_identity_chain_for_all_thirty_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.fairness_hybrid_ccg_benders as hybrid
    from src.fairness_large_final_remediation import construct_initial_t1_upper_bound
    from src.instance import InventoryInstance
    from tests.test_fairness_large_final_remediation_implementation import tiny_instance

    git_root = _detached_git_repo(tmp_path / "g")
    calls = {name: 0 for name in ("generate", "baseline", "frontier", "post", "configure")}
    scales = deepcopy(runner.SCALES)
    for scale, short in (("medium_large", "m"), ("large", "l")):
        scales[scale]["output_dir"] = str(git_root / "experiments/results_fh_gamma" / short)
    monkeypatch.setattr(runner, "SCALES", scales)
    monkeypatch.setattr(runner, "validate_config", lambda path, config: None)

    class Result:
        def __init__(self, gamma: int, warehouses: int, products: int) -> None:
            self.gamma = gamma
            self.warehouses = warehouses
            self.products = products

        def to_dict(self) -> dict[str, object]:
            return {
                "status": "optimal", "gap": 0.0, "lower_bound": 0.2, "upper_bound": 0.2,
                "runtime": 0.1, "objective_t": 0.2,
                "robust_minimum_fill_rate": 0.8,
                "x_values": [[0.0 for _ in range(self.products)] for _ in range(self.warehouses)],
                "y_values": [1.0 for _ in range(self.warehouses)],
                "master_runtime": 0.05, "separation_runtime": 0.05,
                "iterations": 1, "cuts": self.gamma,
                "metadata": {
                    "robust_feasibility_certified": True,
                    "full_separation_objective_bound_required": True,
                    "committed_scenario_count": self.gamma + 1,
                },
                "iteration_log": [{
                    "final_exact_separation_performed": True,
                    "robust_feasibility_certified": True,
                    "master_status": "optimal", "separation_objective_bound": -0.0,
                }],
            }

    def fake_hybrid_solver(instance: InventoryInstance, **kwargs: object) -> Result:
        calls["frontier"] += 1
        expected = kwargs["expected_identity"]
        baseline_record = kwargs["baseline_record"]
        anchor = kwargs["anchor"]
        assert baseline_record["resolved_config_file_sha256"] == expected["resolved_config_file_sha256"]
        construct_initial_t1_upper_bound(
            instance, baseline_record=baseline_record, anchor=anchor,
            rho=float(kwargs["rho"]), tolerance=float(kwargs["tol"]),
            expected_identity=expected,
            expected_candidate_sha256=runner.CANDIDATE_SHA256,
        )
        return Result(int(kwargs["gamma"]), len(instance.I), len(instance.J))

    monkeypatch.setattr(hybrid, "solve_certified_hybrid_scenario_benders_fairness", fake_hybrid_solver)
    production = runner.production_dependencies()

    def generate(_config: dict[str, object], _seed: int) -> InventoryInstance:
        calls["generate"] += 1
        return tiny_instance()

    def baseline(_config: dict[str, object], instance: InventoryInstance, _seed: int, _solver: dict[str, object]) -> dict[str, object]:
        calls["baseline"] += 1
        return {
            "status": "optimal", "valid_UB": True, "upper_bound": 30.0,
            "gap": 0.0, "runtime": 0.1, "master_runtime": 0.06,
            "subproblem_runtime": 0.04, "iterations": 1,
            "best_y_values": [1.0 for _ in instance.I],
            "best_x_values": [[0.0 for _ in instance.J] for _ in instance.I],
        }

    def post(
        _config: dict[str, object], _instance: InventoryInstance, result: dict[str, object],
        anchor: dict[str, object], _identity: dict[str, object], post_root: Path,
        row: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, float]]:
        calls["post"] += 1
        count = runner.scenario_count(str(row["scale"]), int(row["gamma"]))
        atomic_write_json(post_root / "final.json", {"valid": True, "scenario_count": count})
        return ({
            "valid": True, "errors": [], "objective_t_consistent": True,
            "scenario_count": count, "actual_robust_cost": float(anchor["value"]),
            "actual_price_of_fairness": 0.0,
            "wminfr": float(result["robust_minimum_fill_rate"]),
            "minimum_weighted_mean_fill_rate": 0.9,
        }, {"post_evaluation_solver_runtime": 0.0, "post_evaluation_wall_runtime": 0.0,
            "aggregation_runtime": 0.0, "checkpoint_io_runtime": 0.0})

    dependencies = runner.GammaDependencies(
        generate_instance=generate,
        serialize_instance=lambda value: value.to_dict(),
        deserialize_instance=InventoryInstance.from_dict,
        solve_baseline=baseline,
        make_anchor=production.make_anchor,
        solve_frontier=production.solve_frontier,
        post_evaluate=post,
        configure_solver=lambda _settings: calls.__setitem__("configure", calls["configure"] + 1),
    )
    report = runner.run_sensitivity(
        CONFIG, resume=True, dependencies=dependencies,
        test_authorization=True, test_git_root=git_root,
    )
    assert report["completed_run_count"] == 60
    assert calls == {"generate": 30, "baseline": 30, "frontier": 30, "post": 30, "configure": 1}


def test_real_git_gate_ignores_only_frozen_output_root_and_rejects_other_dirt(tmp_path: Path) -> None:
    git_root = _detached_git_repo(tmp_path / "g")
    runner.formal_git_gate(git_root)
    ignored = git_root / "experiments/results_fh_gamma/ml_a3/manifest.json"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("{}\n", encoding="utf-8")
    check = subprocess.run(
        ["git", "-C", str(git_root), "check-ignore", "-v", str(ignored)],
        check=True, capture_output=True, text=True,
    )
    assert "/experiments/results_fh_gamma/" in check.stdout
    runner.formal_git_gate(git_root)
    unrelated = git_root / "unrelated.tmp"
    unrelated.write_text("not allowed\n", encoding="utf-8")
    with pytest.raises(runner.ProtocolGateError, match="worktree is not clean"):
        runner.formal_git_gate(git_root)
    unrelated.unlink()
    (git_root / "tracked.txt").write_text("modified\n", encoding="utf-8")
    with pytest.raises(runner.ProtocolGateError, match="worktree is not clean"):
        runner.formal_git_gate(git_root)


def test_repository_ignore_rule_is_exact() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert lines.count("/experiments/results_fh_gamma/") == 1
    assert "experiments/results_fh_gamma*" not in lines


def test_baseline_checkpoint_failure_resumes_without_repeating_solve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_root = _detached_git_repo(tmp_path / "g")
    calls = {name: 0 for name in ("generate", "baseline", "frontier", "post", "configure")}
    scales = deepcopy(runner.SCALES)
    for scale, short in (("medium_large", "m"), ("large", "l")):
        scales[scale]["output_dir"] = str(git_root / "experiments/results_fh_gamma" / short)
    monkeypatch.setattr(runner, "SCALES", scales)
    monkeypatch.setattr(runner, "validate_config", lambda path, config: None)
    injected = {"done": False}
    def fail_once(point: str, row: dict[str, object]) -> None:
        if not injected["done"]:
            injected["done"] = True
            raise KeyboardInterrupt
    deps = _fake_dependencies(calls)
    with pytest.raises(KeyboardInterrupt):
        runner.run_sensitivity(
            CONFIG, resume=True, dependencies=deps, test_authorization=True,
            test_git_root=git_root, failure_injector=fail_once,
        )
    assert calls["baseline"] == 1 and calls["frontier"] == 0
    checkpoints = list((git_root / "experiments/results_fh_gamma").rglob("baseline_checkpoint.json"))
    assert len(checkpoints) == 1
    original_checkpoint = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    corrupt_checkpoint = deepcopy(original_checkpoint)
    corrupt_checkpoint["identity"]["gamma"] = 99
    atomic_write_json(checkpoints[0], corrupt_checkpoint)
    with pytest.raises(runner.ProtocolGateError, match="baseline checkpoint identity mismatch"):
        runner.run_sensitivity(
            CONFIG, resume=True, dependencies=deps, test_authorization=True, test_git_root=git_root,
        )
    assert calls["baseline"] == 1 and calls["frontier"] == 0
    atomic_write_json(checkpoints[0], original_checkpoint)
    runner.run_sensitivity(
        CONFIG, resume=True, dependencies=deps, test_authorization=True, test_git_root=git_root,
    )
    assert calls["baseline"] == 30 and calls["frontier"] == 30


def test_corrupt_existing_output_fails_before_solver_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_root = _detached_git_repo(tmp_path / "g")
    calls = {name: 0 for name in ("generate", "baseline", "frontier", "post", "configure")}
    scales = deepcopy(runner.SCALES)
    for scale, short in (("medium_large", "m"), ("large", "l")):
        scales[scale]["output_dir"] = str(git_root / "experiments/results_fh_gamma" / short)
    monkeypatch.setattr(runner, "SCALES", scales)
    monkeypatch.setattr(runner, "validate_config", lambda path, config: None)
    output = Path(scales["medium_large"]["output_dir"])
    atomic_write_json(output / "manifest.json", {"schema": "drift"})
    with pytest.raises(runner.ProtocolGateError, match="manifest identity mismatch"):
        runner.run_sensitivity(
            CONFIG, resume=True, dependencies=_fake_dependencies(calls), test_authorization=True,
            test_git_root=git_root,
        )
    assert calls == {"generate": 0, "baseline": 0, "frontier": 0, "post": 0, "configure": 0}


def test_results_writer_rejects_duplicate_run_keys(tmp_path: Path) -> None:
    row = _valid_result_row()
    with pytest.raises(runner.ProtocolGateError, match="duplicate"):
        runner.write_results(tmp_path / "results.csv", [row, dict(row)])
    with pytest.raises(runner.ProtocolGateError, match="run-key directory"):
        runner.write_results(tmp_path / "results.csv", [{**row, "run_directory_id": "r_bad"}])


def test_protocol_and_config_hashes_are_frozen() -> None:
    assert file_sha256(CONFIG).upper() == runner.EXPECTED_CONFIG_SHA256
    assert file_sha256(ROOT / "docs/fairness_hybrid_gamma_sensitivity_protocol.md").upper() == runner.EXPECTED_PROTOCOL_SHA256


def test_static_audit_passes_without_gurobi() -> None:
    result = audit.static_audit(ROOT, None)
    assert result["status"] == "pass" and result["passed"] == result["total"]
    assert result["seed_access_audit"]["audit_passed"] is True
