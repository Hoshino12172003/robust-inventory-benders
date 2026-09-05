from __future__ import annotations

from copy import deepcopy

import pytest

from src.benders import solve_benders
from src.ccg_tail_handoff import (
    canonical_pattern,
    deduplicate_patterns,
    scenario_from_pattern,
    solve_ccg,
)
from src.instance import generate_instance
from src.monolithic import solve_monolithic


def _config() -> dict:
    return {
        "instance": {
            "num_warehouses": 2,
            "num_products": 2,
            "num_regions": 2,
            "budget_factor": 0.8,
        },
        "robust": {
            "gamma_target": 2,
            "gamma_schedule": [2],
            "max_scenarios": 20,
            "exact_scenarios": True,
        },
        "algorithm": {
            "subproblem_mode": "robust_dual_milp",
            "final_certification_enabled": True,
            "final_certification_no_cut_patience": 2,
        },
        "benders": {
            "max_iterations": 200,
            "tol": 1e-8,
            "initial_mip_gap": 0.0,
            "final_mip_gap": 0.0,
            "time_limit": 30,
            "output_flag": False,
        },
    }


def test_pattern_canonicalization_deduplication_and_demand() -> None:
    config = _config()
    instance = generate_instance(config, seed=21)
    pattern = canonical_pattern([(1, 0), (0, 1), (1, 0)])
    assert pattern == ((0, 1), (1, 0))
    unique, duplicates = deduplicate_patterns([pattern, reversed(pattern), ()])
    assert unique == [pattern, ()]
    assert duplicates == 1
    scenario = scenario_from_pattern(instance, pattern)
    assert scenario.demand[0][1] == pytest.approx(
        instance.base_demand[0][1] + instance.demand_deviation[0][1]
    )
    assert scenario.demand[0][0] == pytest.approx(instance.base_demand[0][0])


@pytest.mark.parametrize("seed", [31, 32, 33])
def test_benders_ccg_and_handoff_match_exact_monolithic(seed: int) -> None:
    config = _config()
    instance = generate_instance(config, seed=seed)
    exact = solve_monolithic(config, instance)
    benders = solve_benders(config, instance, "standard_benders")
    pure = solve_ccg(instance, gamma=2, time_limit=30, tolerance=1e-8)

    partial_config = deepcopy(config)
    partial_config["benders"]["max_iterations"] = 1
    partial_config["diagnostics"] = {"record_adversarial_patterns": True}
    partial = solve_benders(partial_config, instance, "standard_benders")
    patterns = [
        row["adversarial_pattern"]
        for row in partial.iteration_log
        if row.get("adversarial_pattern") is not None
    ]
    hybrid = solve_ccg(
        instance,
        gamma=2,
        time_limit=30,
        tolerance=1e-8,
        inherited_patterns=patterns,
        initial_y=partial.metadata["best_y_values"],
        initial_x=partial.metadata["best_x_values"],
        initial_upper_bound=partial.upper_bound,
    )

    assert exact.status == "optimal"
    assert benders.status == "optimal"
    assert pure.certified is True
    assert hybrid.certified is True
    for objective in (benders.upper_bound, pure.upper_bound, hybrid.upper_bound):
        assert objective == pytest.approx(exact.objective, abs=1e-6)
    assert pure.lower_bound <= exact.objective + 1e-6
    assert hybrid.lower_bound <= exact.objective + 1e-6
    assert hybrid.incumbent_reused is True
    assert hybrid.inherited_scenario_count >= 1
