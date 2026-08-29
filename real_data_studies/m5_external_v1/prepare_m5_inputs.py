from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.instance import InventoryInstance, save_instance


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/configs/hybrid_v8_m5_external_holdout.json"
RAW_MANIFEST = Path(__file__).resolve().parent / "raw_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")


def verify_raw(raw_directory: Path) -> dict:
    manifest = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = raw_directory / row["file"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"raw M5 identity mismatch: {row['file']}")
    return manifest


def aggregate_daily(sales: pd.DataFrame, day_columns: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    stores = sorted(sales["store_id"].unique().tolist())
    departments = sorted(sales["dept_id"].unique().tolist())
    grouped = sales.groupby(["store_id", "dept_id"], sort=True)[day_columns].sum()
    expected = pd.MultiIndex.from_product([stores, departments], names=["store_id", "dept_id"])
    grouped = grouped.reindex(expected, fill_value=0)
    if grouped.shape != (len(stores) * len(departments), len(day_columns)):
        raise RuntimeError("store-department aggregation shape drifted")
    return grouped, stores, departments


def learn_factors(
    daily: pd.DataFrame, *, training_end: int, factor_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    training = daily[[f"d_{value}" for value in range(1, training_end + 1)]].to_numpy(dtype=float).T
    scale = np.maximum(training.mean(axis=0), 1.0e-9)
    residual = np.maximum(training / scale - 1.0, 0.0)
    _u, singular_values, vt = np.linalg.svd(residual, full_matrices=False)
    loadings = np.abs(vt[:factor_count].T)
    membership = np.argmax(loadings, axis=1)
    counts = np.bincount(membership, minlength=factor_count)
    if np.any(counts == 0):
        raise RuntimeError(f"empty frozen uncertainty factor: counts={counts.tolist()}")
    return membership, scale, singular_values


def calibrate_factor_deviations(
    daily: pd.DataFrame,
    membership: np.ndarray,
    scale: np.ndarray,
    *,
    calibration_start: int,
    calibration_end: int,
    factor_count: int,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    columns = [f"d_{value}" for value in range(calibration_start, calibration_end + 1)]
    values = daily[columns].to_numpy(dtype=float).T
    if values.shape[0] % 7:
        raise RuntimeError("calibration interval is not a whole number of weeks")
    weekly = values.reshape(values.shape[0] // 7, 7, values.shape[1]).sum(axis=1)
    weekly_scale = 7.0 * scale
    positive = np.maximum(weekly / weekly_scale - 1.0, 0.0)
    scores = np.column_stack([
        positive[:, membership == factor].max(axis=1)
        for factor in range(factor_count)
    ])
    deviations = np.quantile(scores, quantile, axis=0, method="linear")
    return deviations, scores


def factor_scenarios(
    nominal: np.ndarray,
    membership: np.ndarray,
    factor_deviation: np.ndarray,
    *,
    gamma: int,
) -> list[dict]:
    factor_count = len(factor_deviation)
    scenarios = []
    for size in range(gamma + 1):
        for active in itertools.combinations(range(factor_count), size):
            active_set = set(active)
            demand = nominal.copy()
            for cell, factor in enumerate(membership):
                if int(factor) in active_set:
                    demand.flat[cell] += nominal.flat[cell] * factor_deviation[int(factor)]
            scenarios.append({
                "scenario_id": "nominal" if not active else "factors_" + "_".join(str(value + 1) for value in active),
                "active_factors": [value + 1 for value in active],
                "demand": demand.tolist(),
            })
    return scenarios


def build_instance(
    nominal: np.ndarray,
    daily: pd.DataFrame,
    stores: list[str],
    departments: list[str],
    state_by_store: dict[str, str],
    *,
    name: str,
    config: dict,
) -> InventoryInstance:
    parameters = config["calibrated_parameters"]
    training_end = int(config["temporal_split"]["factor_estimation"][1])
    training = daily[[f"d_{value}" for value in range(1, training_end + 1)]].to_numpy(dtype=float)
    usable = training[:, : (training.shape[1] // 7) * 7]
    weekly = usable.reshape(usable.shape[0], -1, 7).sum(axis=2)
    peak_total = float(weekly.sum(axis=0).max())
    peak_department = weekly.reshape(len(stores), len(departments), -1).sum(axis=0).max(axis=1)
    locations = len(stores)
    products = len(departments)
    fixed = [100.0 + 10.0 * index for index in range(locations)]
    inventory_cost = [[1.0] * products for _ in stores]
    capacity = [
        parameters["capacity_multiplier_of_training_peak_week"] * peak_total / locations
        for _ in stores
    ]
    inventory_ub = [
        [1.35 * float(peak_department[j]) / locations for j in range(products)]
        for _ in stores
    ]
    full_open_cost = sum(fixed) + 0.45 * sum(sum(row) for row in inventory_ub)
    budget = parameters["capital_budget_multiplier"] * full_open_cost
    transport = [
        [
            [
                parameters["same_state_transfer_cost"]
                if state_by_store[facility] == state_by_store[region]
                else parameters["cross_state_transfer_cost"]
                for _ in departments
            ]
            for region in stores
        ]
        for facility in stores
    ]
    return InventoryInstance(
        name=name,
        num_warehouses=locations,
        num_products=products,
        num_regions=locations,
        fixed_cost=fixed,
        inventory_cost=inventory_cost,
        capacity=capacity,
        volume=[1.0] * products,
        budget=budget,
        transport_cost=transport,
        shortage_penalty=[[parameters["shortage_penalty_multiplier"]] * products for _ in stores],
        service_penalty=[parameters["service_violation_penalty_multiplier"]] * products,
        service_level=[parameters["service_target"]] * products,
        base_demand=nominal.tolist(),
        demand_deviation=[[0.0] * products for _ in stores],
        inventory_ub=inventory_ub,
    )


def prepare(raw_directory: Path, output_directory: Path) -> None:
    if output_directory.exists():
        raise RuntimeError("M5 processed output already exists; preparation is append-only")
    raw_manifest = verify_raw(raw_directory)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    sales = pd.read_csv(raw_directory / "sales_train_evaluation.csv")
    calendar = pd.read_csv(raw_directory / "calendar.csv", usecols=["d", "date"])
    day_columns = [f"d_{value}" for value in range(1, 1942)]
    daily, stores, departments = aggregate_daily(sales, day_columns)
    if len(stores) != 10 or len(departments) != 7:
        raise RuntimeError("official M5 store or department identity drifted")
    state_by_store_rows = sales[["store_id", "state_id"]].drop_duplicates()
    if state_by_store_rows["store_id"].duplicated().any():
        raise RuntimeError("store maps to multiple states")
    state_by_store = dict(zip(state_by_store_rows["store_id"], state_by_store_rows["state_id"]))
    split = config["temporal_split"]
    factor_count = int(config["mapping"]["factor_count"])
    membership, scale, singular_values = learn_factors(
        daily, training_end=int(split["factor_estimation"][1]), factor_count=factor_count
    )
    factor_deviation, calibration_scores = calibrate_factor_deviations(
        daily, membership, scale,
        calibration_start=int(split["deviation_calibration"][0]),
        calibration_end=int(split["deviation_calibration"][1]),
        factor_count=factor_count,
        quantile=float(config["mapping"]["deviation_quantile"]),
    )
    output_directory.mkdir(parents=True)
    membership_rows = []
    for cell, (store, department) in enumerate(daily.index):
        membership_rows.append({
            "store_id": store,
            "dept_id": department,
            "factor": int(membership[cell]) + 1,
            "training_daily_scale": float(scale[cell]),
        })
    pd.DataFrame(membership_rows).to_csv(output_directory / "factor_membership.csv", index=False)
    pd.DataFrame(calibration_scores, columns=[f"factor_{value + 1}" for value in range(factor_count)]).to_csv(
        output_directory / "calibration_factor_scores.csv", index=False
    )
    save_json(output_directory / "factor_calibration.json", {
        "factor_deviation_fraction": factor_deviation.tolist(),
        "singular_values_first_six": singular_values[:factor_count].tolist(),
        "membership_counts": np.bincount(membership, minlength=factor_count).tolist(),
    })
    dates = dict(zip(calendar["d"], calendar["date"]))
    holdout_start, holdout_end = map(int, split["formal_holdout"])
    catalog = []
    for case_index, first_day in enumerate(range(holdout_start, holdout_end + 1, 7)):
        columns = [f"d_{value}" for value in range(first_day, first_day + 7)]
        nominal = daily[columns].to_numpy(dtype=float).sum(axis=1).reshape(len(stores), len(departments))
        case_id = f"m5_week_{case_index + 1:02d}_d{first_day}_d{first_day + 6}"
        case = output_directory / "cases" / case_id
        instance = build_instance(
            nominal, daily, stores, departments, state_by_store,
            name=case_id, config=config,
        )
        instance_dict = instance.to_dict()
        instance_dict["demand_deviation"] = (
            nominal * factor_deviation[membership].reshape(len(stores), len(departments))
        ).tolist()
        save_instance(InventoryInstance.from_dict(instance_dict), case / "instance.json")
        scenarios = factor_scenarios(
            nominal, membership, factor_deviation, gamma=int(config["mapping"]["gamma"])
        )
        if len(scenarios) != int(config["mapping"]["expected_scenario_count"]):
            raise RuntimeError("factor scenario count drifted")
        save_json(case / "scenarios.json", scenarios)
        catalog.append({
            "case_id": case_id,
            "case_index": case_index,
            "first_day": first_day,
            "last_day": first_day + 6,
            "first_date": dates[f"d_{first_day}"],
            "last_date": dates[f"d_{first_day + 6}"],
            "scenario_count": len(scenarios),
        })
    if len(catalog) != int(split["expected_holdout_cases"]):
        raise RuntimeError("holdout case count drifted")
    save_json(output_directory / "case_catalog.json", catalog)
    files = []
    for path in sorted(value for value in output_directory.rglob("*") if value.is_file()):
        files.append({
            "path": path.relative_to(output_directory).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        })
    save_json(output_directory / "input_freeze.json", {
        "schema": "m5_hybrid_v8_processed_input_freeze_v1",
        "status": "processed_inputs_frozen_before_optimization",
        "raw_manifest_sha256": sha256(RAW_MANIFEST),
        "config_sha256": sha256(CONFIG),
        "processor_sha256": sha256(Path(__file__)),
        "raw_files": raw_manifest["files"],
        "file_count": len(files),
        "files": files,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    prepare(args.raw_directory, args.output_directory)


if __name__ == "__main__":
    main()
