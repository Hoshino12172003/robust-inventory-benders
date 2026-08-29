from __future__ import annotations

import numpy as np
import pandas as pd

from real_data_studies.m5_external_v1.prepare_m5_inputs import (
    calibrate_factor_deviations,
    factor_scenarios,
    learn_factors,
)


def synthetic_daily(cells: int = 8, days: int = 42) -> pd.DataFrame:
    values = np.ones((cells, days), dtype=float)
    for cell in range(cells):
        values[cell, cell % days] = 5.0 + cell
        values[cell, (cell + 8) % days] = 3.0 + cell
    return pd.DataFrame(values, columns=[f"d_{value}" for value in range(1, days + 1)])


def test_factor_learning_and_calibration_are_deterministic() -> None:
    daily = synthetic_daily()
    first = learn_factors(daily, training_end=28, factor_count=2)
    second = learn_factors(daily, training_end=28, factor_count=2)
    assert np.array_equal(first[0], second[0])
    assert np.allclose(first[1], second[1])
    deviations, scores = calibrate_factor_deviations(
        daily, first[0], first[1], calibration_start=29, calibration_end=42,
        factor_count=2, quantile=0.9,
    )
    assert deviations.shape == (2,)
    assert scores.shape == (2, 2)
    assert np.all(deviations >= 0.0)


def test_gamma_two_over_six_factors_produces_twenty_two_scenarios() -> None:
    nominal = np.ones((2, 3), dtype=float)
    membership = np.arange(6, dtype=int)
    deviations = np.full(6, 0.25)
    scenarios = factor_scenarios(nominal, membership, deviations, gamma=2)
    assert len(scenarios) == 22
    assert scenarios[0]["scenario_id"] == "nominal"
    assert scenarios[-1]["active_factors"] == [5, 6]
