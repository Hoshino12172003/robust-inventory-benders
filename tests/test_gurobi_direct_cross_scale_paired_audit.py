from pathlib import Path

import pytest

from src.gurobi_direct_cross_scale_paired_audit import audit


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/gurobi_direct_cross_scale_paired.yaml"


@pytest.mark.skipif(
    not Path(r"E:\论文代码\gurobi_direct_cross_scale_paired_results\paired_results.csv").is_file(),
    reason="formal Direct results are not mounted",
)
def test_formal_results_pass_solver_free_final_audit() -> None:
    result = audit(CONFIG)
    assert result["decision"] == "paired_direct_benchmark_complete"
    assert result["completed_runs"] == result["unique_scale_seed_cells"] == 10
    assert result["source_zip_unchanged"] is True
    assert result["hybrid_reruns"] == result["baseline_reruns"] == 0
