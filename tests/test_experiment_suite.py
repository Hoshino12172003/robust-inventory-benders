from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

import src.experiment_suite as experiment_suite_module
from src.experiment_suite import (
    _apply_variant_config,
    _base_config,
    _validate_relative_threshold_config,
    run_experiment_suite,
)
from src.results import SolveResult


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


def selected_algorithm_parameters(**overrides: object) -> dict:
    selected = {
        "selection_status": "selected",
        "cut_selection_mode": "relative",
        "adaptive_subproblem_gap_enabled": True,
        "adaptive_secondary_cut_selection_enabled": True,
        "secondary_cut_warmup_cuts": 40,
        "secondary_cut_master_time_share_trigger": 0.30,
        "secondary_cut_recent_master_time_trigger": 0.40,
        "relative_cut_threshold": 0.0007,
        "cut_violation_tol": 2.0e-8,
        "final_exact_gap": 0.02,
        "cut_stall_patience": 3,
        "subproblem_gap_schedule": [
            {"global_gap_above": 0.1, "mip_gap": 0.07},
            {"global_gap_above": 0.0, "mip_gap": 0.0002},
        ],
        "max_cuts_per_iteration": 2,
        "subproblem_time_budget_per_iteration": None,
    }
    selected.update(overrides)
    return selected


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
        "first_stage_cost",
        "inventory_cost",
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
    assert {
        "method",
        "mean_runtime",
        "num_completed",
        "completed_rate",
        "num_solved",
        "solved_rate",
        "num_success",
        "success_rate",
    }.issubset(rows[0].keys())


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
    assert rows["no_gamma_continuation"]["gamma_schedule"] == rows["no_gamma_continuation"]["gamma_target"]
    assert rows["no_cut_selection"]["cut_selection_enabled"] == "False"
    assert rows["standard"]["adaptive_gap_enabled"] == "False"
    assert rows["standard"]["gamma_continuation_enabled"] == "False"
    assert rows["standard"]["cut_selection_enabled"] == "False"


def test_correctness_summary_missing_references_are_not_ok(tmp_path: Path) -> None:
    config = tiny_experiment_config(tmp_path)
    config["experiment_name"] = "small_correctness"
    config["methods"] = ["standard_benders", "proposed_adaptive_benders"]

    outputs = run_experiment_suite(config)
    correctness_path = outputs["output_dir"] / "correctness_summary.csv"
    rows = _read_csv(correctness_path)

    assert rows
    assert rows[0]["status"] == "missing_monolithic"
    assert "abs_diff_monolithic_vs_proposed" in rows[0]
    assert "rel_diff_scenario_vs_proposed" in rows[0]


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
    assert configs["diagnostic_medium.yaml"]["save_iteration_log"] is True
    assert configs["screen_relative_cut.yaml"]["random_seeds"] == [0, 1, 2]
    assert configs["final_evaluation_template.yaml"]["random_seeds"] == list(range(10, 20))
    selected = configs["selected_algorithm_parameters.yaml"]
    assert selected["selection_status"] == "pending_parameter_screens"
    for field in (
        "adaptive_secondary_cut_selection_enabled",
        "secondary_cut_warmup_cuts",
        "secondary_cut_master_time_share_trigger",
        "secondary_cut_recent_master_time_trigger",
    ):
        assert selected[field] is None
    secondary = configs["screen_secondary_cut_selection.yaml"]
    assert secondary["random_seeds"] == [0, 1, 2]
    assert secondary["instance_sizes"] == ["medium"]
    assert secondary["time_limit"] == 60
    assert secondary["max_iterations"] == 2000
    assert secondary["variants"] == [
        "k1_single_cut",
        "k2_all_cuts",
        "k2_secondary_static",
        "k2_secondary_adaptive",
    ]
    settings = secondary["variant_settings"]
    assert settings["k1_single_cut"]["max_cuts_per_iteration"] == 1
    assert settings["k1_single_cut"]["cut_selection_enabled"] is False
    assert settings["k2_all_cuts"]["max_cuts_per_iteration"] == 2
    assert settings["k2_all_cuts"]["cut_selection_enabled"] is False
    assert settings["k2_secondary_static"]["cut_selection_mode"] == "relative"
    assert (
        settings["k2_secondary_static"]["adaptive_secondary_cut_selection_enabled"]
        is False
    )
    assert (
        settings["k2_secondary_adaptive"]["adaptive_secondary_cut_selection_enabled"]
        is True
    )


def test_round2_tuning_configs_exist_and_parse() -> None:
    config_dir = Path("experiments/configs")
    wide = yaml.safe_load(
        (config_dir / "screen_relative_cut_wide.yaml").read_text(encoding="utf-8")
    )
    master_gamma = yaml.safe_load(
        (config_dir / "screen_master_gamma.yaml").read_text(encoding="utf-8")
    )
    confirm = yaml.safe_load(
        (config_dir / "confirm_equal_time_medium.yaml").read_text(encoding="utf-8")
    )

    for config in (wide, master_gamma, confirm):
        assert config["random_seeds"] == [0, 1, 2]
        assert config["instance_sizes"] == ["medium"]
        assert config["save_iteration_log"] is True

    thresholds = [
        wide["variant_settings"][name]["relative_cut_threshold"]
        for name in wide["variants"]
    ]
    assert thresholds == [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    assert wide["adaptive_subproblem_gap_enabled"] is True
    assert wide["max_cuts_per_iteration"] == 2
    assert wide["max_iterations"] == 300
    assert wide["time_limit"] == 180

    staged = master_gamma["variant_settings"]["staged_gamma"]["gamma_schedule"]
    assert staged[:10] == [0] * 10
    assert staged[10:30] == [1] * 20
    assert staged[30:] == [2]
    assert master_gamma["relative_cut_threshold"] is None

    assert confirm["methods"] == ["standard_benders", "proposed_adaptive_benders"]
    assert confirm["time_limit"] == 60
    assert confirm["max_iterations"] == 2000
    assert confirm["parameters_must_be_fixed_from"].endswith(
        "selected_algorithm_parameters.yaml"
    )

    _validate_relative_threshold_config(wide)


def test_round2_staged_gamma_variant_is_applied() -> None:
    exp_config = {
        "gamma_target": 2,
        "gamma_schedule": [0, 1, 2],
        "subproblem_mode": "robust_dual_milp",
        "mip_gap": 0.05,
        "final_mip_gap": 0.0001,
    }
    config = _base_config(exp_config, "very_small", seed=0)
    staged = [0] * 10 + [1] * 20 + [2]
    _, flags, resolved = _apply_variant_config(
        config,
        "proposed_adaptive_benders",
        {
            "adaptive_gap_enabled": True,
            "gamma_continuation_enabled": True,
            "cut_selection_enabled": True,
            "gamma_schedule": staged,
        },
    )

    assert flags["gamma_continuation_enabled"] is True
    assert resolved["robust"]["gamma_schedule"] == staged


def test_screen_master_gamma_requires_selected_relative_threshold(tmp_path: Path) -> None:
    config_path = Path("experiments/configs/screen_master_gamma.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["output_dir"] = str(tmp_path / "must_not_be_created")

    with pytest.raises(
        ValueError,
        match=(
            "relative_cut_threshold must be selected before running "
            r"screen_master_gamma\.yaml\. Run screen_relative_cut_wide\.yaml first\."
        ),
    ):
        run_experiment_suite(config)

    assert not Path(config["output_dir"]).exists()


def test_relative_variant_missing_threshold_is_rejected() -> None:
    config = {
        "cut_selection_mode": "relative",
        "variants": ["complete", "missing"],
        "variant_settings": {
            "complete": {"relative_cut_threshold": 0.1},
            "missing": {},
        },
    }

    with pytest.raises(
        ValueError,
        match=(
            "relative_cut_threshold must be selected before running "
            r"screen_master_gamma\.yaml\. Run screen_relative_cut_wide\.yaml first\."
        ),
    ):
        _validate_relative_threshold_config(config)


def test_iteration_logs_and_time_to_gap_fields_are_written(tmp_path: Path) -> None:
    config = tiny_experiment_config(tmp_path)
    config["save_iteration_log"] = True
    outputs = run_experiment_suite(config)
    rows = _read_csv(outputs["results"])
    assert rows
    assert rows[0]["iteration_log_path"]
    log_path = Path(rows[0]["iteration_log_path"])
    assert log_path.exists()
    log_rows = _read_csv(log_path)
    assert log_rows
    for field in (
        "requested_master_mip_gap",
        "achieved_master_mip_gap",
        "subproblem_requested_mip_gap",
        "normalized_cut_violation",
        "forced_cut_added",
        "secondary_cut_decisions",
        "secondary_active_threshold",
    ):
        assert field in log_rows[0]
    for field in ("reached_gap_5pct", "time_to_gap_1pct", "subproblem_time_share"):
        assert field in rows[0]


def test_selected_parameters_are_applied_and_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_path = tmp_path / "selected.yaml"
    selected = selected_algorithm_parameters()
    selected_path.write_text(yaml.safe_dump(selected, sort_keys=False), encoding="utf-8")
    config = tiny_experiment_config(tmp_path)
    config["methods"] = [
        "standard_benders",
        "static_inexact_benders",
        "proposed_adaptive_benders",
    ]
    config["parameters_must_be_fixed_from"] = str(selected_path)

    def fake_solve_benders(config: dict, _instance: object, method: str) -> SolveResult:
        algorithm = config["algorithm"]
        robust = config["robust"]
        return SolveResult(
            method=method,
            status="optimal",
            objective=1.0,
            lower_bound=1.0,
            upper_bound=1.0,
            gap=0.0,
            runtime=0.01,
            iterations=1,
            cuts=1,
            gamma_target=int(robust["gamma_target"]),
            metadata={
                "cut_selection_enabled": algorithm["cut_selection_enabled"],
                "cut_selection_mode": algorithm["cut_selection_mode"],
                "relative_cut_threshold": algorithm["relative_cut_threshold"],
                "cut_violation_tol": algorithm["cut_violation_tol"],
                "final_exact_gap": algorithm["final_exact_gap"],
                "cut_stall_patience": algorithm["cut_stall_patience"],
                "adaptive_subproblem_gap_enabled": algorithm[
                    "adaptive_subproblem_gap_enabled"
                ],
                "adaptive_secondary_cut_selection_enabled": algorithm[
                    "adaptive_secondary_cut_selection_enabled"
                ],
                "secondary_cut_warmup_cuts": algorithm["secondary_cut_warmup_cuts"],
                "secondary_cut_master_time_share_trigger": algorithm[
                    "secondary_cut_master_time_share_trigger"
                ],
                "secondary_cut_recent_master_time_trigger": algorithm[
                    "secondary_cut_recent_master_time_trigger"
                ],
                "subproblem_gap_schedule": algorithm["subproblem_gap_schedule"],
                "max_cuts_per_iteration": algorithm["max_cuts_per_iteration"],
                "gamma_schedule": ",".join(
                    str(value) for value in robust["gamma_schedule"]
                ),
            },
        )

    monkeypatch.setattr(experiment_suite_module, "solve_benders", fake_solve_benders)

    outputs = run_experiment_suite(config)
    resolved = yaml.safe_load(outputs["resolved_config"].read_text(encoding="utf-8"))
    for field, value in selected.items():
        if field != "selection_status":
            assert resolved[field] == value

    rows = {row["method"]: row for row in _read_csv(outputs["results"])}
    proposed = rows["proposed_adaptive_benders"]
    assert proposed["cut_selection_mode"] == "relative"
    assert proposed["adaptive_subproblem_gap_enabled"] == "True"
    assert float(proposed["relative_cut_threshold"]) == pytest.approx(0.0007)
    assert float(proposed["cut_violation_tol"]) == pytest.approx(2.0e-8)
    assert float(proposed["final_exact_gap"]) == pytest.approx(0.02)
    assert int(proposed["cut_stall_patience"]) == 3
    assert int(proposed["max_cuts_per_iteration"]) == 2
    assert proposed["adaptive_secondary_cut_selection_enabled"] == "True"
    assert int(proposed["secondary_cut_warmup_cuts"]) == 40
    assert float(proposed["secondary_cut_master_time_share_trigger"]) == pytest.approx(
        0.30
    )
    assert float(proposed["secondary_cut_recent_master_time_trigger"]) == pytest.approx(
        0.40
    )
    assert "0.07" in proposed["subproblem_gap_schedule"]

    standard = rows["standard_benders"]
    assert standard["cut_selection_enabled"] == "False"
    assert standard["adaptive_subproblem_gap_enabled"] == "False"
    assert standard["adaptive_secondary_cut_selection_enabled"] == "False"
    assert int(standard["max_cuts_per_iteration"]) == 1
    assert standard["gamma_schedule"] == standard["gamma_target"]
    assert "1e-05" in standard["subproblem_gap_schedule"]

    static = rows["static_inexact_benders"]
    assert static["cut_selection_enabled"] == "False"
    assert static["adaptive_subproblem_gap_enabled"] == "False"
    assert static["adaptive_secondary_cut_selection_enabled"] == "False"
    assert int(static["max_cuts_per_iteration"]) == 1
    assert static["gamma_schedule"] == static["gamma_target"]


@pytest.mark.parametrize("method", ["standard_benders", "static_inexact_benders"])
def test_baselines_disable_adaptive_secondary_selection(
    tmp_path: Path,
    method: str,
) -> None:
    experiment_config = tiny_experiment_config(tmp_path)
    experiment_config.update(selected_algorithm_parameters())
    base_config = _base_config(experiment_config, "very_small", seed=0)

    _, _, method_config = _apply_variant_config(base_config, method, {})

    assert method_config["algorithm"]["cut_selection_enabled"] is False
    assert (
        method_config["algorithm"]["adaptive_secondary_cut_selection_enabled"]
        is False
    )
    assert method_config["algorithm"]["max_cuts_per_iteration"] == 1


def test_selected_parameters_reject_missing_values(tmp_path: Path) -> None:
    selected_path = tmp_path / "selected_missing.yaml"
    selected_path.write_text(
        yaml.safe_dump({"selection_status": "selected", "relative_cut_threshold": 0.001}),
        encoding="utf-8",
    )
    config = tiny_experiment_config(tmp_path)
    config["parameters_must_be_fixed_from"] = str(selected_path)
    with pytest.raises(ValueError, match="Selected algorithm parameters are missing"):
        run_experiment_suite(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "adaptive_secondary_cut_selection_enabled",
            "true",
            "adaptive_secondary_cut_selection_enabled must be true or false",
        ),
        (
            "secondary_cut_warmup_cuts",
            0,
            "secondary_cut_warmup_cuts must be a positive integer",
        ),
        (
            "secondary_cut_master_time_share_trigger",
            0.0,
            "secondary_cut_master_time_share_trigger must be a positive finite value",
        ),
        (
            "secondary_cut_recent_master_time_trigger",
            float("inf"),
            "secondary_cut_recent_master_time_trigger must be a positive finite value",
        ),
    ],
)
def test_selected_secondary_policy_rejects_invalid_values(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    selected_path = tmp_path / "selected_invalid.yaml"
    selected_path.write_text(
        yaml.safe_dump(selected_algorithm_parameters(**{field: value}), sort_keys=False),
        encoding="utf-8",
    )
    config = tiny_experiment_config(tmp_path)
    config["parameters_must_be_fixed_from"] = str(selected_path)

    with pytest.raises(ValueError, match=message):
        run_experiment_suite(config)
