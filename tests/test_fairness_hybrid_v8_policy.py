from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.fairness_hybrid_ccg_benders import (
    HYBRID_V8_CANDIDATE_SHA256,
    max_coefficient_normalized_cut,
    scenario_sha256,
    select_new_scenarios,
    select_v8_sentinel_cut,
    solve_certified_hybrid_scenario_benders_fairness,
)
from src.experiment_protocol import config_sha256
from src.fairness_large_final_remediation import CUT_SCHEMA
from src.scenarios import _scenario_from_units
from tests.test_robust_regional_fairness import tiny_instance


def _candidate(*, region: int, raw: float, cut_sha: str, coefficient: float = 2.0):
    cut = SimpleNamespace(active_deviations=[{"region": region, "product": 0}])
    payload = {
        "schema": CUT_SCHEMA,
        "sense": ">=",
        "rhs": float(0.0).hex(),
        "constant": float(-raw).hex(),
        "terms": [["T", float(coefficient).hex()]],
    }
    return SimpleNamespace(
        cut=cut,
        raw_violation=raw,
        cut_sha256=cut_sha,
        pattern_sha256=f"pattern-{region}",
        canonical_cut_payload=payload,
    )


def test_v8_selects_up_to_four_distinct_uncommitted_blocks_by_raw_violation() -> None:
    instance = tiny_instance(regions=6, products=1)
    candidates = [
        _candidate(region=region, raw=float(region + 1), cut_sha=str(region) * 64)
        for region in range(6)
    ]
    committed = {
        scenario_sha256(instance, _scenario_from_units(instance, ((5, 0),)))
    }
    selected = select_new_scenarios(instance, candidates, committed, limit=4)
    assert len(selected) == 4
    assert [item[1].active_units for item in selected] == [
        ((4, 0),), ((3, 0),), ((2, 0),), ((1, 0),)
    ]


def test_v8_block_selection_deduplicates_scenarios() -> None:
    instance = tiny_instance(regions=2, products=1)
    weaker = _candidate(region=0, raw=1.0, cut_sha="B" * 64)
    stronger = _candidate(region=0, raw=2.0, cut_sha="A" * 64)
    selected = select_new_scenarios(instance, [weaker, stronger], set(), limit=4)
    assert len(selected) == 1
    assert selected[0][0] is stronger
    with pytest.raises(ValueError, match="nonnegative integer"):
        select_new_scenarios(instance, [], set(), limit=True)


def test_v8_cut_is_scaled_by_maximum_absolute_coefficient() -> None:
    candidate = _candidate(region=0, raw=0.4, cut_sha="A" * 64, coefficient=2.0)
    payload, digest, efficacy = max_coefficient_normalized_cut(candidate)
    assert float.fromhex(payload["terms"][0][1]) == pytest.approx(1.0)
    assert float.fromhex(payload["constant"]) == pytest.approx(-0.2)
    assert efficacy == pytest.approx(0.2)
    assert len(digest) == 64


def test_v8_sentinel_cut_excludes_promoted_candidates_and_enforces_threshold() -> None:
    promoted = _candidate(region=0, raw=1.0, cut_sha="A" * 64, coefficient=2.0)
    sentinel = _candidate(region=1, raw=0.4, cut_sha="B" * 64, coefficient=2.0)
    selected = select_v8_sentinel_cut(
        [promoted, sentinel], {promoted.pattern_sha256}, set(), minimum_efficacy=0.10
    )
    assert selected is not None
    assert selected[0] is sentinel
    assert selected[3] == pytest.approx(0.2)
    assert select_v8_sentinel_cut(
        [sentinel], set(), set(), minimum_efficacy=0.21
    ) is None
    assert select_v8_sentinel_cut(
        [sentinel], set(), {selected[2]}, minimum_efficacy=0.10
    ) is None


def test_v8_core_solver_reaches_exact_certification() -> None:
    from tests.test_fairness_large_final_remediation_implementation import (
        baseline_evidence,
        upper_bound_identity,
    )

    instance = tiny_instance()
    record, anchor = baseline_evidence(instance, gamma=2)
    record["candidate_sha256"] = HYBRID_V8_CANDIDATE_SHA256
    anchor["candidate_sha256"] = HYBRID_V8_CANDIDATE_SHA256
    anchor["anchor_sha256"] = config_sha256({
        key: value for key, value in anchor.items() if key != "anchor_sha256"
    })
    result = solve_certified_hybrid_scenario_benders_fairness(
        instance,
        baseline_record=record,
        anchor=anchor,
        expected_identity=upper_bound_identity(instance, record, anchor),
        solver_parameters={"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7},
        rho=0.0,
        gamma=2,
        max_iterations=20,
        time_limit=30.0,
        tol=1.0e-6,
        algorithm_policy="v8",
    )
    assert result.status == "optimal"
    assert result.metadata["algorithm_policy"] == "v8"
    assert result.metadata["robust_feasibility_certified"] is True
    assert result.lower_bound == pytest.approx(result.upper_bound, abs=1.0e-6)
    assert all(
        len(entry["committed_scenario_sha256_values"]) <= 4
        and len(entry["committed_farkas_cut_sha256_values"]) <= 1
        for entry in result.iteration_log
    )
