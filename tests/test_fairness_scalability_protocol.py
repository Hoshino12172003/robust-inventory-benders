from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.fairness_scalability_audit import CONFIGS, audit_fairness_scalability
from src.fairness_scalability_suite import dry_run_report, load_config, main


def configs() -> dict[str, dict]:
    return {size: yaml.safe_load(path.read_text(encoding="utf-8")) for size, path in CONFIGS.items()}


def test_frozen_protocol_audit_passes() -> None:
    report = audit_fairness_scalability()
    assert report["status"] == "passed", [c for c in report["checks"] if not c["passed"]]


@pytest.mark.parametrize("size,scenarios", [("medium_large", 1831), ("large", 4657)])
def test_dry_run_is_static_and_has_frozen_counts(size: str, scenarios: int) -> None:
    report = dry_run_report(load_config(CONFIGS[size]))
    assert report["instances_generated"] is False
    assert report["solver_called"] is False
    assert report["scenario_count"] == scenarios
    assert report["s1"]["total_tasks"] == 27
    assert report["s2_cumulative"]["total_tasks"] == 90
    assert report["complete_staged_unique_tasks"] == 120
    assert report["output_dir_exists"] is False


def test_cli_refuses_formal_execution() -> None:
    with pytest.raises(SystemExit):
        main(["--config", str(CONFIGS["medium_large"])])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["medium_large"].update(development_seeds=list(range(150, 160))),
        lambda c: c["medium_large"]["scalability_candidates"].append("fifth_candidate"),
        lambda c: c["large"].update(fairness_time_limit=1801),
        lambda c: c["large"]["full_grid_gate"].update(minimum_certified_solved_rate=0.75),
        lambda c: c["large"]["certification"].update(old_cut_reuse_allowed=True),
        lambda c: c["large"]["candidate_settings"]["persistent_certified_cache_batch5"].update(max_cuts_per_iteration=6),
    ],
)
def test_audit_rejects_protocol_drift(mutation) -> None:
    values = deepcopy(configs())
    mutation(values)
    assert audit_fairness_scalability(config_overrides=values)["status"] == "failed"
