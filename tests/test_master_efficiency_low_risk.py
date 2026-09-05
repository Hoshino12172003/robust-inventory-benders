from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

import src.benders as benders_module
from src.benders import solve_benders
from src.instance import generate_instance
from src.master_efficiency import ExactCutRegistry


def _cut(constant: float, beta0: float, beta1: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        constant=constant,
        x_coefficients={(0, 0): beta0, (1, 0): beta1},
    )


def _paired_config(seed: int, enabled: bool) -> dict:
    return {
        "seed": seed,
        "instance": {
            "num_warehouses": 2,
            "num_products": 1,
            "num_regions": 2,
            "budget_factor": 0.72,
        },
        "robust": {
            "gamma_target": 1,
            "gamma_schedule": [1],
            "max_scenarios": 50,
            "exact_scenarios": True,
        },
        "algorithm": {
            "subproblem_mode": "robust_dual_milp",
            "cut_selection_enabled": False,
            "cut_violation_tol": 1.0e-8,
            "master_efficiency_low_risk": enabled,
            "duplicate_cut_tolerance": 1.0e-10,
            "final_certification_enabled": True,
            "final_certification_no_cut_patience": 2,
        },
        "benders": {
            "max_iterations": 80,
            "tol": 1.0e-4,
            "initial_mip_gap": 0.0,
            "final_mip_gap": 0.0,
            "time_limit": 60.0,
            "output_flag": False,
        },
    }


def test_exact_duplicate_registry_is_strict_and_near_duplicate_is_diagnostic_only() -> None:
    registry = ExactCutRegistry(((0, 0), (1, 0)), tolerance=1.0e-10)
    original = registry.canonicalize(_cut(4.0, 3.0, -2.0))
    registry.add(original)

    within_tolerance = registry.canonicalize(
        _cut(4.0 + 0.5e-10, 3.0 - 0.5e-10, -2.0)
    )
    near_but_distinct = registry.canonicalize(_cut(4.0 + 2.0e-10, 3.0, -2.0))

    assert registry.is_duplicate(within_tolerance) is True
    assert registry.is_duplicate(near_but_distinct) is False
    assert registry.nearest_cosine_similarity(near_but_distinct) == pytest.approx(1.0)


def test_nonfinite_cut_is_rejected_before_master_insertion() -> None:
    registry = ExactCutRegistry(((0, 0), (1, 0)), tolerance=1.0e-10)
    with pytest.raises(ValueError, match="non-finite"):
        registry.canonicalize(_cut(float("nan"), 1.0))


@pytest.mark.parametrize("seed", [17, 18])
def test_low_risk_mode_matches_certified_legacy_objective(seed: int) -> None:
    legacy_config = _paired_config(seed, enabled=False)
    improved_config = deepcopy(legacy_config)
    improved_config["algorithm"]["master_efficiency_low_risk"] = True
    instance = generate_instance(legacy_config, seed=seed)

    legacy = solve_benders(legacy_config, instance, "standard_benders")
    improved = solve_benders(improved_config, instance, "standard_benders")

    assert legacy.status == improved.status == "optimal"
    assert legacy.gap is not None and legacy.gap <= legacy_config["benders"]["tol"]
    assert improved.gap is not None and improved.gap <= improved_config["benders"]["tol"]
    assert improved.objective == pytest.approx(legacy.objective, abs=1.0e-6)
    assert improved.lower_bound == pytest.approx(legacy.lower_bound, abs=1.0e-6)
    assert improved.upper_bound == pytest.approx(legacy.upper_bound, abs=1.0e-6)
    assert improved.metadata["persistent_master"] is True
    assert improved.metadata["duplicate_cut_filtering"] == "strict_tolerance"
    assert improved.iteration_log[0]["warm_start_attempted"] is False
    assert all(row["warm_start_attempted"] for row in improved.iteration_log[1:])
    required = {
        "accumulated_cut_count",
        "master_time",
        "realized_master_gap",
        "master_node_count",
        "master_status_name",
        "lower_bound",
        "upper_bound",
        "global_gap",
        "cut_violation",
        "normalized_cut_violation",
        "cut_added",
        "exact_duplicates_rejected_this_iteration",
        "worst_case_pattern_id",
        "strengthened_cut_attempted",
        "strengthened_cut_accepted",
        "core_point_gain",
        "warm_start_attempted",
        "nearest_cut_similarity",
    }
    assert all(required <= row.keys() for row in improved.iteration_log)


def test_master_is_built_once_and_reoptimized_persistently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _paired_config(19, enabled=True)
    instance = generate_instance(config, seed=19)
    original_build = benders_module._build_master
    build_count = 0

    def counted_build(*args: object, **kwargs: object):
        nonlocal build_count
        build_count += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(benders_module, "_build_master", counted_build)
    result = solve_benders(config, instance, "standard_benders")

    assert result.status == "optimal"
    assert result.iterations > 1
    assert build_count == 1
