from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
REPO_ROOT = STUDY_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coordination_model import solve_policy  # noqa: E402
from src.instance import load_instance  # noqa: E402
from src.scenarios import DemandScenario  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_new_json(path: Path, payload) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _source_cases() -> list[Path]:
    roots = (
        STUDY_ROOT / "algorithm_v4_holdout_validation" / "cases",
        STUDY_ROOT / "algorithm_v8_adaptive_hybrid" / "sealed_confirmation" / "cases",
    )
    cases = []
    for root in roots:
        cases.extend(path for path in root.glob("week_*") if path.is_dir())
    return sorted(cases, key=lambda path: path.name)


def prepare() -> None:
    cases = _source_cases()
    if len(cases) != 17 or len({path.name for path in cases}) != 17:
        raise RuntimeError(f"expected 17 unique Olist weekly cases, found {len(cases)}")
    catalog = []
    for case in cases:
        files = []
        for filename in ("instance.json", "scenarios.json"):
            path = case / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(
                {
                    "role": filename.removesuffix(".json"),
                    "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        catalog.append({"case_id": case.name, "files": files})
    _write_new_json(
        HERE / "inputs" / "input_catalog.json",
        {
            "status": "source_inputs_frozen_before_formal_optimization",
            "case_count": len(catalog),
            "file_count": sum(len(case["files"]) for case in catalog),
            "cases": catalog,
        },
    )
    print(json.dumps({"prepared_cases": len(catalog), "solver_called": False}))


def _load_and_verify() -> tuple[dict, dict]:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    catalog = json.loads((HERE / "inputs" / "input_catalog.json").read_text(encoding="utf-8"))
    if catalog["case_count"] != 17 or catalog["file_count"] != 34:
        raise RuntimeError("input catalog dimensions do not match the protocol")
    for case in catalog["cases"]:
        for item in case["files"]:
            path = REPO_ROOT / item["path"]
            if _sha256(path) != item["sha256"]:
                raise RuntimeError(f"source input identity mismatch: {path}")
    return config, catalog


def dry_run() -> None:
    config, catalog = _load_and_verify()
    print(
        json.dumps(
            {
                "status": "dry_run_passed",
                "cases": catalog["case_count"],
                "rho_grid": config["rho_grid"],
                "primary_rho": config["primary_rho"],
                "formal_run_authorized": config["formal_run_authorized"],
                "solver_called": False,
            },
            indent=2,
        )
    )


def _load_scenarios(path: Path) -> list[DemandScenario]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        DemandScenario(
            name=row["name"],
            active_units=tuple(tuple(int(value) for value in cell) for cell in row["active_units"]),
            demand=tuple(tuple(float(value) for value in line) for line in row["demand"]),
        )
        for row in payload
    ]


def _formal_authorization() -> dict:
    path = HERE / "authorization.json"
    if not path.is_file():
        raise RuntimeError("formal execution is not authorized; authorization.json is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("formal_run_authorized") is not True:
        raise RuntimeError("formal execution is not authorized")
    return payload


def _case_paths(case: dict) -> tuple[Path, Path]:
    by_role = {item["role"]: REPO_ROOT / item["path"] for item in case["files"]}
    return by_role["instance"], by_role["scenarios"]


def _solver_arguments(config: dict) -> dict:
    return {
        "severe_shortage_threshold": float(config["severe_shortage_threshold"]),
        "time_limit": float(config["time_limit_seconds_per_solve"]),
        "mip_gap": float(config["mip_gap"]),
        "feasibility_tolerance": float(config["feasibility_tolerance"]),
        "threads": int(config["threads"]),
        "solver_seed": int(config["solver_seed"]),
    }


def run_formal() -> None:
    authorization = _formal_authorization()
    config, catalog = _load_and_verify()
    results_dir = HERE / "formal_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    solver_arguments = _solver_arguments(config)
    for case_index, case in enumerate(catalog["cases"]):
        output_path = results_dir / f"{case['case_id']}.json"
        if output_path.exists():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("case_id") != case["case_id"]:
                raise RuntimeError(f"existing result identity mismatch: {output_path}")
            print(json.dumps({"case_id": case["case_id"], "status": "existing_result_retained"}))
            continue
        instance_path, scenarios_path = _case_paths(case)
        instance = load_instance(instance_path)
        scenarios = _load_scenarios(scenarios_path)
        anchors = {}
        anchor_order = list(config["policies"])
        if case_index % 2:
            anchor_order.reverse()
        for policy in anchor_order:
            anchors[policy] = solve_policy(
                instance,
                scenarios,
                policy,
                objective="cost_anchor",
                **solver_arguments,
            )
        if not all(result["certified"] for result in anchors.values()):
            payload = {
                "case_id": case["case_id"],
                "status": "anchor_not_certified",
                "authorization": authorization,
                "anchors": anchors,
                "frontier": [],
            }
            _write_new_json(output_path, payload)
            print(json.dumps({"case_id": case["case_id"], "status": payload["status"]}))
            continue
        flexible_anchor = anchors["flexible_multiwarehouse"]["objective_value"]
        single_anchor = anchors["optimized_single_source"]["objective_value"]
        if flexible_anchor > single_anchor + 1.0e-5:
            raise RuntimeError(
                f"flexible anchor violates feasible-set containment for {case['case_id']}"
            )
        frontier = []
        for rho_index, rho in enumerate(config["rho_grid"]):
            common_budget = (1.0 + float(rho)) * float(single_anchor)
            policies = list(config["policies"])
            if (case_index + rho_index) % 2:
                policies.reverse()
            results = {}
            for policy in policies:
                results[policy] = solve_policy(
                    instance,
                    scenarios,
                    policy,
                    objective="service_protection",
                    cost_budget=common_budget,
                    **solver_arguments,
                )
            if all(result["certified"] for result in results.values()):
                flexible_t = results["flexible_multiwarehouse"]["objective_value"]
                single_t = results["optimized_single_source"]["objective_value"]
                if flexible_t > single_t + 1.0e-5:
                    raise RuntimeError(
                        f"flexible service objective violates feasible-set containment for "
                        f"{case['case_id']} at rho={rho}"
                    )
            frontier.append(
                {
                    "rho": float(rho),
                    "common_cost_budget": common_budget,
                    "run_order": policies,
                    "results": results,
                }
            )
        payload = {
            "case_id": case["case_id"],
            "status": "complete",
            "authorization": authorization,
            "source_files": case["files"],
            "anchor_run_order": anchor_order,
            "anchors": anchors,
            "frontier": frontier,
        }
        _write_new_json(output_path, payload)
        print(json.dumps({"case_id": case["case_id"], "status": "complete"}), flush=True)


def _bootstrap_mean_interval(values: list[float], seed: int = 20260831) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(10000):
        samples.append(statistics.mean(rng.choice(values) for _ in values))
    samples.sort()
    return [samples[249], samples[9749]]


def summarize() -> None:
    config, catalog = _load_and_verify()
    rows = []
    for case in catalog["cases"]:
        path = HERE / "formal_results" / f"{case['case_id']}.json"
        if not path.is_file():
            raise RuntimeError(f"formal result is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for point in payload.get("frontier", []):
            flexible = point["results"]["flexible_multiwarehouse"]
            single = point["results"]["optimized_single_source"]
            row = {
                "case_id": case["case_id"],
                "rho": point["rho"],
                "common_cost_budget": point["common_cost_budget"],
                "flexible_certified": flexible["certified"],
                "single_certified": single["certified"],
                "flexible_runtime_seconds": flexible["runtime_seconds"],
                "single_runtime_seconds": single["runtime_seconds"],
            }
            for prefix, result in (("flexible", flexible), ("single", single)):
                metrics = result.get("metrics", {})
                row[f"{prefix}_worst_shortage_rate"] = result.get("objective_value")
                for key in (
                    "robust_total_cost",
                    "facility_cost",
                    "inventory_cost",
                    "mean_transport_cost",
                    "worst_transport_cost",
                    "demand_weighted_shortage_rate",
                    "severe_region_scenario_count",
                    "severe_region_count",
                    "nonlocal_fulfillment_share",
                    "mean_active_sources_per_region_scenario",
                    "maximum_active_sources_per_region_scenario",
                    "open_facility_count",
                    "total_inventory_units",
                ):
                    row[f"{prefix}_{key}"] = metrics.get(key)
            if flexible["certified"] and single["certified"]:
                row["flexible_minus_single_worst_shortage_rate"] = (
                    flexible["objective_value"] - single["objective_value"]
                )
                row["relative_shortage_reduction"] = (
                    (single["objective_value"] - flexible["objective_value"])
                    / single["objective_value"]
                    if single["objective_value"] > 1.0e-12
                    else 0.0
                )
            else:
                row["flexible_minus_single_worst_shortage_rate"] = None
                row["relative_shortage_reduction"] = None
            rows.append(row)
    primary = [
        row
        for row in rows
        if math.isclose(row["rho"], float(config["primary_rho"]), abs_tol=1.0e-12)
        and row["flexible_certified"]
        and row["single_certified"]
    ]
    differences = [row["flexible_minus_single_worst_shortage_rate"] for row in primary]
    summary = {
        "status": "formal_summary_complete",
        "case_count": len(catalog["cases"]),
        "row_count": len(rows),
        "primary_rho": config["primary_rho"],
        "primary_jointly_certified_weeks": len(primary),
        "primary_flexible_strict_improvements": sum(value < -1.0e-7 for value in differences),
        "primary_equal_within_tolerance": sum(abs(value) <= 1.0e-7 for value in differences),
        "primary_mean_flexible_minus_single_shortage_rate": (
            statistics.mean(differences) if differences else None
        ),
        "primary_median_flexible_minus_single_shortage_rate": (
            statistics.median(differences) if differences else None
        ),
        "primary_mean_difference_bootstrap_95_interval": _bootstrap_mean_interval(differences),
        "claim_boundary": (
            "paired Olist model evidence; mathematical weak dominance is structural, "
            "whereas empirical magnitude and cost composition are case-study findings"
        ),
    }
    analysis_dir = HERE / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "paired_results.csv"
    if csv_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {csv_path}")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_new_json(analysis_dir / "summary.json", summary)
    report = [
        "# 多仓协同价值实验结果",
        "",
        f"- 周数：{summary['case_count']}",
        f"- 主分析 rho：{summary['primary_rho']}",
        f"- 两种策略均完成认证：{summary['primary_jointly_certified_weeks']}",
        f"- Flexible 严格降低最坏区域缺货率：{summary['primary_flexible_strict_improvements']}",
        f"- 容差内相同：{summary['primary_equal_within_tolerance']}",
        f"- 平均差（Flexible - Single）：{summary['primary_mean_flexible_minus_single_shortage_rate']}",
        f"- 中位数差：{summary['primary_median_flexible_minus_single_shortage_rate']}",
        f"- 周级 bootstrap 95% 区间：{summary['primary_mean_difference_bootstrap_95_interval']}",
        "",
        "该结果只能解释为本研究校准网络中的配对模型证据。Flexible 的弱优势来自可行域包含关系；优势幅度、运输代价与设施/库存变化才是实证贡献。",
    ]
    report_path = analysis_dir / "report_zh.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {report_path}")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "dry-run", "run", "summarize"),
        required=True,
    )
    args = parser.parse_args()
    {
        "prepare": prepare,
        "dry-run": dry_run,
        "run": run_formal,
        "summarize": summarize,
    }[args.stage]()


if __name__ == "__main__":
    main()
