from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.experiment_protocol import atomic_write_json, file_sha256
from src.fairness_hybrid_ccg_benders import initial_upper_bound_expected_identity
from src.fairness_hybrid_ccg_benders_runner import _hybrid_certified_anchor
import src.fairness_hybrid_gamma_sensitivity_runner as runner
from src.fairness_large_final_remediation import (
    InitialUpperBoundAssumptionFailure,
    construct_initial_t1_upper_bound,
)
from src.fairness_large_final_remediation_runner import _production_generate_instance


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/fairness_hybrid_gamma_sensitivity.yaml"


def _production_identity_bundle(scale: str, gamma: int, tmp_path: Path) -> dict[str, object]:
    config = runner._scale_config(runner.load_config(CONFIG), scale, gamma)
    instance = _production_generate_instance(config, 991)
    serialized = instance.to_dict()
    frozen = {
        "stage": runner.STAGE, "scale": scale, "seed": 991, "gamma": gamma,
        "execution_attempt": runner.EXECUTION_ATTEMPT, "git_commit": "G" * 40,
        "config_file_sha256": runner.EXPECTED_CONFIG_SHA256,
        "protocol_sha256": runner.EXPECTED_PROTOCOL_SHA256,
    }
    archive = runner.instance_archive_payload(frozen, serialized)
    archive_path = tmp_path / f"{scale}_g{gamma}.json"
    atomic_write_json(archive_path, archive)
    restored, instance_identity, canonical_sha, identity_sha = runner.validate_instance_archive(archive, frozen)
    assert restored == serialized
    archive_file_sha = file_sha256(archive_path).upper()
    baseline_run_key = f"production-schema::{scale}::991::{gamma}::baseline::attempt3"
    common = {
        "instance_sha256": canonical_sha,
        "instance_canonical_sha256": canonical_sha,
        "instance_identity_sha256": identity_sha,
        "seed": 991, "gamma": gamma, "scale": scale, "stage": runner.STAGE,
        "execution_attempt": runner.EXECUTION_ATTEMPT, "git_commit": "G" * 40,
        "config_file_sha256": runner.EXPECTED_CONFIG_SHA256,
        "resolved_config_file_sha256": runner.EXPECTED_CONFIG_SHA256,
        "protocol_sha256": runner.EXPECTED_PROTOCOL_SHA256,
        "candidate_sha256": runner.CANDIDATE_SHA256,
        "baseline_run_key": baseline_run_key,
    }
    result = {
        "status": "optimal", "valid_UB": True, "upper_bound": 1.0, "gap": 0.0,
        "best_y_values": [1.0 for _ in instance.I],
        "best_x_values": [[0.0 for _ in instance.J] for _ in instance.I],
    }
    baseline = {
        **common, "run_key": baseline_run_key, "scientific_status": "certified_robust_optimal",
        "solved_to_tolerance": True, "result": result,
    }
    anchor = _hybrid_certified_anchor(baseline, common_identity=common, tolerance=1e-4)
    expected = initial_upper_bound_expected_identity(common, anchor)
    expected.update({
        "instance_canonical_sha256": canonical_sha,
        "gamma": gamma,
        "execution_attempt": runner.EXECUTION_ATTEMPT,
    })
    manifest_identity = {
        **instance_identity, "instance_identity_sha256": identity_sha,
        "instance_archive_file_sha256": archive_file_sha,
    }
    return {
        "instance": instance, "archive": archive, "frozen": frozen,
        "canonical_sha": canonical_sha, "archive_file_sha": archive_file_sha,
        "identity_sha": identity_sha, "common": common, "baseline": baseline,
        "anchor": anchor, "expected": expected, "manifest_identity": manifest_identity,
    }


@pytest.mark.parametrize("scale", ["medium_large", "large"])
@pytest.mark.parametrize("gamma", [0, 1, 2])
def test_production_archive_baseline_anchor_identity_contract(
    scale: str, gamma: int, tmp_path: Path,
) -> None:
    bundle = _production_identity_bundle(scale, gamma, tmp_path)
    proof = construct_initial_t1_upper_bound(
        bundle["instance"], baseline_record=bundle["baseline"], anchor=bundle["anchor"],
        rho=runner.RHO, tolerance=1e-4, expected_identity=bundle["expected"],
        expected_candidate_sha256=runner.CANDIDATE_SHA256,
    )
    assert proof.value == 1.0 and proof.evidence["initial_robust_ub_valid"] is True
    restored = runner.validate_instance_archive(bundle["archive"], bundle["frozen"])
    assert restored[2] == bundle["canonical_sha"] and restored[3] == bundle["identity_sha"]


def test_production_identity_rejects_demand_gamma_and_hash_domain_drift(tmp_path: Path) -> None:
    bundle = _production_identity_bundle("medium_large", 1, tmp_path)
    changed = deepcopy(bundle["instance"])
    changed.base_demand[0][0] += 1.0
    with pytest.raises(InitialUpperBoundAssumptionFailure, match="current_instance_identity_mismatch"):
        construct_initial_t1_upper_bound(
            changed, baseline_record=bundle["baseline"], anchor=bundle["anchor"],
            rho=runner.RHO, tolerance=1e-4, expected_identity=bundle["expected"],
            expected_candidate_sha256=runner.CANDIDATE_SHA256,
        )
    gamma_drift = deepcopy(bundle["expected"])
    gamma_drift["gamma"] = 2
    with pytest.raises(InitialUpperBoundAssumptionFailure, match="identity_gamma"):
        construct_initial_t1_upper_bound(
            bundle["instance"], baseline_record=bundle["baseline"], anchor=bundle["anchor"],
            rho=runner.RHO, tolerance=1e-4, expected_identity=gamma_drift,
            expected_candidate_sha256=runner.CANDIDATE_SHA256,
        )
    swapped = deepcopy(bundle["archive"])
    swapped["identity"]["instance_canonical_sha256"] = bundle["archive_file_sha"]
    with pytest.raises(runner.ProtocolGateError, match="instance identity mismatch"):
        runner.validate_instance_archive(swapped, bundle["frozen"])


def test_production_identity_rejects_baseline_anchor_and_manifest_drift(tmp_path: Path) -> None:
    bundle = _production_identity_bundle("large", 2, tmp_path)
    frontier_identity = {
        "task_type": "frontier", "scale": "large", "seed": 991, "gamma": 2,
        "execution_attempt": runner.EXECUTION_ATTEMPT,
        "baseline_run_key": bundle["baseline"]["run_key"],
    }
    runner.bind_data_identities(
        frontier_identity,
        instance_canonical_sha256=bundle["canonical_sha"],
        instance_identity_sha256=bundle["identity_sha"],
        baseline=bundle["baseline"], anchor=bundle["anchor"],
    )
    for label, baseline, anchor in (
        ("baseline", {**bundle["baseline"], "instance_canonical_sha256": "B" * 64}, bundle["anchor"]),
        ("anchor", bundle["baseline"], {**bundle["anchor"], "gamma": 1}),
    ):
        with pytest.raises(runner.ProtocolGateError, match=label):
            runner.bind_data_identities(
                frontier_identity,
                instance_canonical_sha256=bundle["canonical_sha"],
                instance_identity_sha256=bundle["identity_sha"], baseline=baseline, anchor=anchor,
            )
    manifest = {"instance_identities": {}}
    runner.record_manifest_identity(
        manifest, "instance_identities", "s991_g2", bundle["manifest_identity"],
    )
    with pytest.raises(runner.ProtocolGateError, match="manifest instance_identities identity mismatch"):
        runner.record_manifest_identity(
            manifest, "instance_identities", "s991_g2",
            {**bundle["manifest_identity"], "gamma": 1},
        )


def test_production_frontier_adapter_receives_complete_runner_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _production_identity_bundle("medium_large", 0, tmp_path)
    import src.fairness_hybrid_ccg_benders as hybrid

    captured: dict[str, object] = {}

    class Result:
        def to_dict(self) -> dict[str, object]:
            return {"adapter_reached": True}

    def fake_solver(instance: object, **kwargs: object) -> Result:
        expected = kwargs["expected_identity"]
        baseline = kwargs["baseline_record"]
        anchor = kwargs["anchor"]
        assert isinstance(expected, dict) and isinstance(baseline, dict) and isinstance(anchor, dict)
        assert baseline["resolved_config_file_sha256"] == expected["resolved_config_file_sha256"]
        construct_initial_t1_upper_bound(
            instance, baseline_record=baseline, anchor=anchor, rho=runner.RHO,
            tolerance=1e-4, expected_identity=expected,
            expected_candidate_sha256=runner.CANDIDATE_SHA256,
        )
        captured["expected"] = expected
        return Result()

    monkeypatch.setattr(hybrid, "solve_certified_hybrid_scenario_benders_fairness", fake_solver)
    dependencies = runner.production_dependencies()
    result = dependencies.solve_frontier(
        runner._scale_config(runner.load_config(CONFIG), "medium_large", 0),
        bundle["instance"], bundle["baseline"], bundle["anchor"], bundle["common"],
        tmp_path / "algorithm_checkpoint.json", runner.SOLVER_PARAMETERS,
        {"gamma": 0, "run_key": "production-adapter"},
    )
    assert result == {"adapter_reached": True}
    assert captured["expected"]["execution_attempt"] == runner.EXECUTION_ATTEMPT
