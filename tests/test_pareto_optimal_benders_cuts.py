from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.benders import solve_benders
from src.cut_strengthening import (
    cut_strengthening_config,
    pareto_cut_acceptance,
    relative_interior_core_point,
)
from src.instance import InventoryInstance, generate_instance
from src.robust_dual_subproblem import (
    solve_pareto_optimal_mw_cut,
    solve_robust_dual_subproblem,
)


def _config(seed: int = 301, policy: str = "pareto_optimal_mw") -> dict:
    return {
        "seed": seed,
        "instance": {
            "num_warehouses": 2,
            "num_products": 2,
            "num_regions": 2,
            "budget_factor": 0.8,
        },
        "robust": {"gamma_target": 1, "gamma_schedule": [1]},
        "algorithm": {
            "cut_strengthening_policy": policy,
            "subproblem_mode": "robust_dual_milp",
            "cut_selection_enabled": False,
            "final_certification_enabled": True,
            "final_certification_no_cut_patience": 2,
            "core_point_update_weight": 0.5,
            "core_point_min_distance": 1.0e-9,
            "core_point_stage1_time_limit": 2.0,
            "core_point_stage2_time_limit": 2.0,
            "core_point_min_remaining_time": 5.0,
            "core_point_min_global_gap": 0.0,
            "core_point_current_abs_tol": 1.0e-7,
            "core_point_current_rel_tol": 1.0e-8,
            "core_point_min_normalized_improvement": 1.0e-7,
        },
        "benders": {
            "max_iterations": 100,
            "tol": 1.0e-4,
            "initial_mip_gap": 0.0,
            "final_mip_gap": 0.0,
            "time_limit": 60.0,
            "output_flag": False,
        },
    }


def _sample_x(instance: InventoryInstance, scale: float) -> dict[tuple[int, int], float]:
    return {
        (i, j): min(
            instance.inventory_ub[i][j],
            scale * sum(instance.base_demand[r][j] for r in instance.R),
        )
        for i in instance.I
        for j in instance.J
    }


def test_pareto_policy_uses_a_strict_feasible_core_point() -> None:
    config = _config()
    instance = generate_instance(config, seed=301)
    core = relative_interior_core_point(instance)
    base = {
        (i, j): min(
            0.5 * instance.inventory_ub[i][j],
            0.5 * instance.capacity[i] / (instance.num_products * instance.volume[j]),
        )
        for i in instance.I
        for j in instance.J
    }
    epsilon = core[0, 0] / base[0, 0]

    assert 0.0 < epsilon < 1.0
    assert all(0.0 < core[i, j] < instance.inventory_ub[i][j] * epsilon for i in instance.I for j in instance.J)
    assert all(
        sum(instance.volume[j] * core[i, j] for j in instance.J)
        < instance.capacity[i] * epsilon
        for i in instance.I
    )
    first_stage = sum(instance.fixed_cost[i] * epsilon for i in instance.I) + sum(
        instance.inventory_cost[i][j] * core[i, j]
        for i in instance.I
        for j in instance.J
    )
    assert first_stage < instance.budget


def test_mw_cut_is_dual_feasible_globally_valid_and_tight_at_current_point() -> None:
    config = _config(seed=302)
    instance = generate_instance(config, seed=302)
    current = _sample_x(instance, 0.22)
    core = relative_interior_core_point(instance)
    ordinary = solve_robust_dual_subproblem(instance, current, gamma=1, mip_gap=0.0)
    result = solve_pareto_optimal_mw_cut(
        instance,
        current,
        core,
        ordinary,
        stage1_time_limit=5.0,
        stage2_time_limit=5.0,
        remaining_global_time=15.0,
    )

    assert result.stage1_status == result.stage2_status == "optimal"
    assert result.strengthened_cut is not None
    assert result.dual_feasible
    assert result.auxiliary_bound_used_for_ub is False
    assert result.strengthened_value_at_current == pytest.approx(
        result.stage1_objective, abs=1.0e-6
    )
    assert result.stage1_objective == pytest.approx(ordinary.objective, abs=1.0e-6)
    assert result.strengthened_value_at_core >= result.original_value_at_core - 1.0e-6

    rng = np.random.default_rng(302)
    for _ in range(4):
        test_x = {
            (i, j): float(rng.uniform(0.0, instance.inventory_ub[i][j]))
            for i in instance.I
            for j in instance.J
        }
        exact = solve_robust_dual_subproblem(instance, test_x, gamma=1, mip_gap=0.0)
        assert result.strengthened_cut.cut_value(test_x) <= exact.objective + 1.0e-5


def test_pareto_acceptance_requires_current_point_tightness() -> None:
    accepted = pareto_cut_acceptance(
        stage1_optimal=True,
        stage2_optimal=True,
        dual_feasible=True,
        pareto_value_at_current=100.0,
        current_optimal_value=100.0,
        pareto_value_at_core=110.0,
        ordinary_value_at_core=100.0,
        tightness_tolerance=1.0e-7,
        minimum_normalized_improvement=1.0e-7,
        duplicate=False,
        original_primary_violated=True,
        certification_active=False,
    )
    not_tight = pareto_cut_acceptance(
        stage1_optimal=True,
        stage2_optimal=True,
        dual_feasible=True,
        pareto_value_at_current=99.999,
        current_optimal_value=100.0,
        pareto_value_at_core=110.0,
        ordinary_value_at_core=100.0,
        tightness_tolerance=1.0e-7,
        minimum_normalized_improvement=1.0e-7,
        duplicate=False,
        original_primary_violated=True,
        certification_active=False,
    )
    assert accepted.accepted
    assert not not_tight.accepted
    assert not_tight.fallback_reason == "current_point_not_tight"


def test_pareto_mode_preserves_certified_objective_and_emits_diagnostics() -> None:
    pareto_config = _config(seed=303)
    current_config = deepcopy(pareto_config)
    current_config["algorithm"]["cut_strengthening_policy"] = "core_point"
    instance = generate_instance(pareto_config, seed=303)

    current = solve_benders(current_config, instance, "standard_benders")
    pareto = solve_benders(pareto_config, instance, "standard_benders")

    assert current.status == pareto.status == "optimal"
    assert current.objective == pytest.approx(pareto.objective, abs=1.0e-6)
    assert pareto.lower_bound <= pareto.objective + 1.0e-6
    assert pareto.upper_bound == pytest.approx(current.upper_bound, abs=1.0e-6)
    required = {
        "ordinary_cut_rhs_at_current",
        "pareto_cut_rhs_at_current",
        "ordinary_cut_value_at_core",
        "pareto_cut_value_at_core",
        "strengthening_gain",
        "strengthened_coefficient_norm",
        "accepted_cut_type",
        "lb_gain_after_previous_cut",
        "master_time",
        "cuts_added_total",
    }
    assert all(required <= row.keys() for row in pareto.iteration_log)
    assert any(row["core_point_attempted"] for row in pareto.iteration_log)


def test_pareto_mode_rejects_zero_core_weight() -> None:
    algorithm = _config()["algorithm"]
    algorithm["core_point_update_weight"] = 0.0
    with pytest.raises(ValueError, match="positive core-point update weight"):
        cut_strengthening_config(algorithm)
