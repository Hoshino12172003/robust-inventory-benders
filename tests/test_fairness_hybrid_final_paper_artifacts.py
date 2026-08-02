from __future__ import annotations

import csv
import json
from pathlib import Path

from src.fairness_hybrid_final_paper_artifacts import (
    MERGE_COMMIT,
    PAPER_METRICS_SHA256,
    RESULTS_SHA256,
    ZIP_SHA256,
    generate,
)


def test_final_paper_artifacts_are_complete_and_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert generate(root, first) == generate(root, second)
    assert {path.name for path in first.iterdir()} == {path.name for path in second.iterdir()}
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()

    freeze = json.loads((first / "freeze_manifest.json").read_text(encoding="utf-8"))
    assert freeze["source_zip_sha256"] == ZIP_SHA256
    assert freeze["merge_commit"] == MERGE_COMMIT
    assert freeze["results_corrected"]["sha256"] == RESULTS_SHA256
    assert freeze["paper_metrics"]["sha256"] == PAPER_METRICS_SHA256

    with (first / "table_all_descriptive_statistics.csv").open(encoding="utf-8", newline="") as handle:
        descriptive = list(csv.DictReader(handle))
    with (first / "table_complete_seed_results.csv").open(encoding="utf-8", newline="") as handle:
        complete = list(csv.DictReader(handle))
    assert len(descriptive) == 2 * 5 * 15
    assert len(complete) == 100
    assert {row["seed"] for row in complete} == {str(seed) for seed in range(170, 180)}
    assert all((first / name).stat().st_size > 1000 for name in (
        "figure_fairness_cost_tradeoff.png", "figure_runtime_scalability.png", "figure_algorithm_structure.png",
    ))
