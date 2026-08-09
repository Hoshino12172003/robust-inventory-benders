from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.experiment_protocol import config_sha256
from src.fairness_high_gamma_external_solver_benchmark import solve_gurobi_direct_extensive_form
from src.fairness_hybrid_ccg_benders import CANDIDATE_SHA256, solve_certified_hybrid_scenario_benders_fairness
from src.instance import InventoryInstance
from src.robust_regional_fairness import evaluate_fairness_solution
from tests.test_fairness_large_final_remediation_implementation import baseline_evidence, upper_bound_identity


def _s0_instance(variant: int):
    return InventoryInstance(
        name=f"high_gamma_s0_{variant}", num_warehouses=1, num_products=5, num_regions=1,
        fixed_cost=[0.0], inventory_cost=[[4.0 + 0.1 * j for j in range(5)]],
        capacity=[22.0 + variant], volume=[1.0] * 5, budget=100.0,
        transport_cost=[[[0.1 * (j + 1) for j in range(5)]]],
        shortage_penalty=[[2.0 + 0.3 * variant + 0.2 * j for j in range(5)]],
        service_penalty=[40.0 + j for j in range(5)], service_level=[0.0] * 5,
        base_demand=[[4.0 + 0.2 * j for j in range(5)]],
        demand_deviation=[[0.5 + 0.2 * variant + 0.1 * j for j in range(5)]],
        inventory_ub=[[8.0] * 5],
    )


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize("gamma", [0, 1, 2, 3, 4])
def test_s0_hybrid_matches_direct_complete_model(tmp_path: Path, variant: int, gamma: int) -> None:
    instance = _s0_instance(variant)
    anchor_value = 300.0
    record, anchor = baseline_evidence(instance, gamma=gamma, upper=anchor_value)
    record["candidate_sha256"] = CANDIDATE_SHA256
    record["result"]["best_x_values"] = [[4.0] * 5]
    anchor["candidate_sha256"] = CANDIDATE_SHA256
    anchor["anchor_sha256"] = config_sha256({key: value for key, value in anchor.items() if key != "anchor_sha256"})
    expected = upper_bound_identity(instance, record, anchor)
    count = sum(math.comb(5, k) for k in range(gamma + 1))
    hybrid = solve_certified_hybrid_scenario_benders_fairness(
        instance, baseline_record=record, anchor=anchor, expected_identity=expected,
        solver_parameters={"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7},
        rho=0.025, gamma=gamma, max_iterations=100, time_limit=60.0, tol=1e-7,
        feasibility_tolerance=1e-7, checkpoint_path=tmp_path / f"h_{variant}_{gamma}.json",
        checkpoint_identity={"run_key": f"s0-{variant}-{gamma}"},
        execution_protocol_sha256="A" * 64, output_flag=False,
    )
    direct = solve_gurobi_direct_extensive_form(
        instance, baseline_cost=anchor_value, rho=0.025, gamma=gamma,
        expected_scenario_count=count,
        solver_parameters={"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7},
        time_limit=60.0, output_flag=False,
    )
    assert hybrid.status == direct.status == "optimal", direct.resource_failure_detail
    assert hybrid.objective_t == pytest.approx(direct.objective_t, abs=1e-7)
    assert hybrid.lower_bound <= direct.objective_t + 1e-7
    assert hybrid.upper_bound >= direct.objective_t - 1e-7
    assert direct.lower_bound <= hybrid.objective_t + 1e-7
    assert direct.upper_bound >= hybrid.objective_t - 1e-7
    evaluations = []
    for result in (hybrid.to_dict(), direct.to_dict()):
        evaluation = evaluate_fairness_solution(
            instance, y_values=result["y_values"], x_values=result["x_values"],
            t_value=result["objective_t"], baseline_cost=anchor_value, rho=0.025,
            gamma=gamma, max_scenarios=count, per_scenario_time_limit=10.0,
            tolerance=1e-7, output_flag=False,
        )
        assert evaluation.valid and evaluation.objective_t_consistent
        assert evaluation.actual_robust_cost <= (1.0 + 0.025) * anchor_value + 1e-7
        evaluations.append(evaluation)
    assert evaluations[0].wminfr >= 1.0 - hybrid.objective_t - 1e-7
    assert evaluations[1].wminfr >= 1.0 - direct.objective_t - 1e-7
