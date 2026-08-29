from __future__ import annotations

import json

from real_data_studies.m5_external_v1 import run_formal_holdout as formal


def test_method_order_balances_all_four_methods() -> None:
    orders = [formal.method_order(index) for index in range(4)]
    assert orders == [
        ("hybrid_v8", "pure_ccg", "batch4_ccg", "direct"),
        ("pure_ccg", "batch4_ccg", "direct", "hybrid_v8"),
        ("batch4_ccg", "direct", "hybrid_v8", "pure_ccg"),
        ("direct", "hybrid_v8", "pure_ccg", "batch4_ccg"),
    ]
    assert all(sorted(order) == sorted(formal.METHODS) for order in orders)


def test_frozen_m5_scenarios_map_active_factors_to_cells() -> None:
    _config, catalog = formal.verify_inputs()
    case = catalog[0]
    path = formal.INPUT_ROOT / "cases" / case["case_id"] / "scenarios.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenarios = formal.load_scenarios(path)
    assert len(scenarios) == 22
    assert scenarios[0].name == "nominal"
    assert scenarios[0].active_units == ()
    assert len(scenarios[1].active_units) > 0
    assert scenarios[-1].name == raw[-1]["scenario_id"]
    assert len(scenarios[-1].active_units) == 20


def test_sign_test_and_par2_use_preregistered_rules() -> None:
    assert formal.exact_sign_test(6, 0) == 0.03125
    assert formal.exact_sign_test(1, 1) == 1.0
    assert formal.exact_sign_test(0, 0) is None
    assert formal.par2({"runtime": 12.5}, True, 3600.0) == 12.5
    assert formal.par2({"runtime": 12.5}, False, 3600.0) == 3600.0
