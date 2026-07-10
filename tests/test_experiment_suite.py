from __future__ import annotations

import csv
from pathlib import Path

import yaml

from src.experiment_suite import run_experiment_suite


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tiny_experiment_config(tmp_path: Path) -> dict:
    return {
        "experiment_name": "test_experiment_suite",
        "output_dir": str(tmp_path / "suite"),
        "random_seeds": [0],
        "instance_sizes": ["very_small"],
        "methods": ["standard_benders", "proposed_adaptive_benders"],
        "time_limit": 30,
        "mip_gap": 0.05,
        "final_mip_gap": 1e-5,
        "max_iterations": 20,
        "gamma_target": 1,
        "gamma_schedule": [0, 1],
        "max_scenarios": 50,
        "exact_scenarios": True,
        "subproblem_mode": "robust_dual_milp",
        "budget_factor": 0.8,
        "capacity_factor": 1.3,
        "delta_cut": 0.0,
    }


def test_tiny_experiment_suite_runs(tmp_path: Path) -> None:
    outputs = run_experiment_suite(tiny_experiment_config(tmp_path))
    rows = _read_csv(outputs["results"])

    assert outputs["results"].exists()
    assert len(rows) >= 2
    required_fields = {
        "experiment_name",
        "instance_name",
        "instance_size",
        "seed",
        "method",
        "status",
        "objective",
        "runtime",
        "cuts_added_total",
        "instance_path",
    }
    assert required_fields.issubset(rows[0].keys())


def test_failed_method_does_not_stop_experiment(tmp_path: Path) -> None:
    config = tiny_experiment_config(tmp_path)
    config["methods"] = ["standard_benders", "unsupported_method"]

    outputs = run_experiment_suite(config)
    rows = _read_csv(outputs["results"])

    statuses_by_method = {row["method"]: row["status"] for row in rows}
    assert statuses_by_method["unsupported_method"] == "failed"
    assert statuses_by_method["standard_benders"] in {"optimal", "iteration_limit", "time_limit"}


def test_summary_csv_is_generated(tmp_path: Path) -> None:
    outputs = run_experiment_suite(tiny_experiment_config(tmp_path))
    rows = _read_csv(outputs["summary"])

    assert outputs["summary"].exists()
    assert rows
    assert {"method", "mean_runtime", "num_success"}.issubset(rows[0].keys())


def test_ablation_variant_mapping(tmp_path: Path) -> None:
    config = tiny_experiment_config(tmp_path)
    config["methods"] = None
    config["variants"] = [
        "full",
        "no_adaptive_gap",
        "no_gamma_continuation",
        "no_cut_selection",
        "standard",
    ]
    config["variant_settings"] = {
        "full": {
            "adaptive_gap_enabled": True,
            "gamma_continuation_enabled": True,
            "cut_selection_enabled": True,
        },
        "no_adaptive_gap": {
            "adaptive_gap_enabled": False,
            "gamma_continuation_enabled": True,
            "cut_selection_enabled": True,
        },
        "no_gamma_continuation": {
            "adaptive_gap_enabled": True,
            "gamma_continuation_enabled": False,
            "cut_selection_enabled": True,
        },
        "no_cut_selection": {
            "adaptive_gap_enabled": True,
            "gamma_continuation_enabled": True,
            "cut_selection_enabled": False,
        },
        "standard": {
            "adaptive_gap_enabled": False,
            "gamma_continuation_enabled": False,
            "cut_selection_enabled": False,
        },
    }

    outputs = run_experiment_suite(config)
    rows = {row["variant_name"]: row for row in _read_csv(outputs["results"])}

    assert rows["full"]["adaptive_gap_enabled"] == "True"
    assert rows["full"]["gamma_continuation_enabled"] == "True"
    assert rows["full"]["cut_selection_enabled"] == "True"
    assert rows["no_adaptive_gap"]["adaptive_gap_enabled"] == "False"
    assert rows["no_gamma_continuation"]["gamma_continuation_enabled"] == "False"
    assert rows["no_cut_selection"]["cut_selection_enabled"] == "False"
    assert rows["standard"]["adaptive_gap_enabled"] == "False"
    assert rows["standard"]["gamma_continuation_enabled"] == "False"


def test_formal_experiment_configs_exist_and_parse() -> None:
    config_dir = Path("experiments/configs")
    paths = sorted(config_dir.glob("*.yaml"))

    assert paths
    configs = {path.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths}
    for config in configs.values():
        assert config["experiment_name"]
        assert config["output_dir"]
        assert config["random_seeds"]

    assert "medium" in configs["baseline_comparison.yaml"]["instance_sizes"]
    assert "medium" in configs["ablation_study.yaml"]["instance_sizes"]
    assert "large" in configs["scalability.yaml"]["instance_sizes"]
