from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.fairness_post_evaluation import (
    PostEvaluationCheckpointError,
    checkpointed_fairness_post_evaluation,
)
from src.instance import InventoryInstance
from src.robust_regional_fairness import FairnessScenarioPolicy
from src.scenarios import DemandScenario


def tiny_instance() -> InventoryInstance:
    return InventoryInstance(
        name="post_evaluation_checkpoint_fixture",
        num_warehouses=1,
        num_products=1,
        num_regions=2,
        fixed_cost=[0.0],
        inventory_cost=[[1.0]],
        capacity=[20.0],
        volume=[1.0],
        budget=20.0,
        transport_cost=[[[0.0], [0.0]]],
        shortage_penalty=[[1.0], [1.0]],
        service_penalty=[1.0],
        service_level=[0.0],
        base_demand=[[1.0], [1.0]],
        demand_deviation=[[1.0], [1.0]],
        inventory_ub=[[20.0]],
    )


def scenarios(count: int = 5) -> list[DemandScenario]:
    return [
        DemandScenario(
            name=f"scenario_{index}",
            active_units=((index % 2, 0),) if index else (),
            demand=((1.0 + int(index % 2 == 0),), (1.0 + int(index % 2 == 1),)),
        )
        for index in range(count)
    ]


class FakeSolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, _instance, scenario, **_kwargs) -> FairnessScenarioPolicy:
        self.calls.append(scenario.name)
        demand = [sum(row) for row in scenario.demand]
        return FairnessScenarioPolicy(
            scenario_name=scenario.name,
            active_deviations=[
                {"region": r, "product": j} for r, j in scenario.active_units
            ],
            recourse_cost=0.0,
            transport_cost=0.0,
            shortage_cost=0.0,
            service_violation_cost=0.0,
            regional_shortage=[0.0, 0.0],
            regional_demand=demand,
            fill_rates=[1.0, 1.0],
            minimum_fill_rate=1.0,
            fill_rate_gap=0.0,
            worst_region_deviation=0.0,
            weighted_mean_fill_rate=1.0,
            solver_runtime=0.01,
        )


def run(
    root: Path,
    *,
    solver: FakeSolver,
    failure_injector=None,
    commit: str = "attempt4",
):
    items = scenarios()
    return checkpointed_fairness_post_evaluation(
        tiny_instance(),
        root=root,
        run_key="synthetic-frontier",
        config_sha256_value="config",
        git_commit=commit,
        baseline_anchor_sha256="anchor",
        y_values=[1.0],
        x_values=[[10.0]],
        t_value=0.0,
        baseline_cost=10.0,
        rho=0.0,
        gamma=2,
        max_scenarios=100,
        per_scenario_time_limit=30.0,
        tolerance=1.0e-7,
        chunk_size=2,
        resume_count=0,
        failure_injector=failure_injector,
        scenario_enumerator=lambda *_args, **_kwargs: SimpleNamespace(
            scenarios=items
        ),
        scenario_solver=solver,
    )


@pytest.mark.parametrize(
    "stage",
    [
        "before_first_chunk",
        "after_last_chunk_before_aggregation",
        "before_final_output",
        "after_final_output",
    ],
)
def test_interrupt_boundaries_resume_to_clean_result(
    tmp_path: Path, stage: str
) -> None:
    clean_solver = FakeSolver()
    clean, _ = run(tmp_path / "clean", solver=clean_solver)
    interrupted_solver = FakeSolver()
    fired = False

    def interrupt(current: str, _context) -> None:
        nonlocal fired
        if current == stage and not fired:
            fired = True
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run(
            tmp_path / f"resume-{stage}",
            solver=interrupted_solver,
            failure_injector=interrupt,
        )
    resumed, timing = run(
        tmp_path / f"resume-{stage}", solver=interrupted_solver
    )
    assert resumed.to_dict() == replace(
        clean, runtime=resumed.runtime
    ).to_dict()
    assert timing.solver_runtime == pytest.approx(0.05)
    index = json.loads(
        (
            tmp_path
            / f"resume-{stage}"
            / "checkpoint"
            / "index.json"
        ).read_text()
    )
    assert sum(item["scenario_count"] for item in index["chunks"]) == 5


def test_mid_chunk_interrupt_recomputes_only_uncommitted_chunk(tmp_path: Path) -> None:
    solver = FakeSolver()
    fired = False

    def interrupt(stage: str, context) -> None:
        nonlocal fired
        if stage == "after_scenario" and context["scenario_index"] == 3 and not fired:
            fired = True
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "run", solver=solver, failure_injector=interrupt)
    calls_before_resume = list(solver.calls)
    assert calls_before_resume == ["scenario_0", "scenario_1", "scenario_2", "scenario_3"]
    result, _ = run(tmp_path / "run", solver=solver)
    assert result.valid
    assert solver.calls.count("scenario_0") == 1
    assert solver.calls.count("scenario_1") == 1
    assert solver.calls.count("scenario_2") == 2
    assert solver.calls.count("scenario_3") == 2
    assert solver.calls.count("scenario_4") == 1


def test_committed_chunk_before_index_is_resume_source_of_truth(
    tmp_path: Path,
) -> None:
    solver = FakeSolver()
    fired = False

    def interrupt(stage: str, context) -> None:
        nonlocal fired
        if (
            stage == "after_chunk_commit_before_index"
            and context["chunk_index"] == 0
            and not fired
        ):
            fired = True
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "run", solver=solver, failure_injector=interrupt)
    assert solver.calls == ["scenario_0", "scenario_1"]
    result, _ = run(tmp_path / "run", solver=solver)
    assert result.valid
    assert solver.calls.count("scenario_0") == 1
    assert solver.calls.count("scenario_1") == 1


def test_completed_resume_is_idempotent_and_has_unique_scenarios(
    tmp_path: Path,
) -> None:
    solver = FakeSolver()
    first, timing = run(tmp_path / "run", solver=solver)
    first_calls = list(solver.calls)
    second, second_timing = run(tmp_path / "run", solver=solver)
    assert first.to_dict() == second.to_dict()
    assert timing == second_timing
    assert solver.calls == first_calls
    keys: list[str] = []
    for path in (tmp_path / "run" / "checkpoint").glob("chunk_*.json"):
        keys.extend(record["scenario_key"] for record in json.loads(path.read_text())["records"])
    assert len(keys) == len(set(keys)) == 5


def test_checkpoint_identity_drift_and_corruption_fail_closed(
    tmp_path: Path,
) -> None:
    solver = FakeSolver()
    run(tmp_path / "run", solver=solver)
    (tmp_path / "run" / "post_evaluation.json").unlink()
    with pytest.raises(PostEvaluationCheckpointError, match="identity drift"):
        run(tmp_path / "run", solver=solver, commit="different")
    chunk = tmp_path / "run" / "checkpoint" / "chunk_00000.json"
    chunk.write_text("{broken", encoding="utf-8")
    with pytest.raises(PostEvaluationCheckpointError, match="Corrupt checkpoint"):
        run(tmp_path / "run", solver=solver)


def test_runtime_fields_keep_algorithm_and_post_evaluation_separate() -> None:
    policy = FakeSolver()(tiny_instance(), scenarios(1)[0])
    assert policy.solver_runtime == 0.01
