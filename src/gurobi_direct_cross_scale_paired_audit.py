"""Solver-free final audit and paper-table export for the paired Direct benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from .experiment_protocol import atomic_write_json, file_sha256
from .gurobi_direct_cross_scale_paired import (
    EXPECTED_SCENARIOS, SEEDS, SCALES, SOURCE_SHA256, expand_plan, load_catalog,
    load_yaml, validate_config,
)


class DirectAuditError(RuntimeError):
    pass


def audit(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(config_path)
    validate_config(config)
    cells = load_catalog(root, config)
    cell_map = {(cell["scale"], cell["seed"]): cell for cell in cells}
    output = Path(config["output_dir"])
    records: list[dict[str, Any]] = []
    for row in expand_plan():
        path = output / "runs" / row["run_directory_id"] / "run.json"
        if not path.is_file():
            raise DirectAuditError(f"missing run: {row['scale']} seed={row['seed']}")
        record = json.loads(path.read_text(encoding="utf-8"))
        result = record.get("result", {})
        cell = cell_map[(row["scale"], row["seed"])]
        identity = record.get("identity", {})
        checks = {
            "complete": record.get("state") == "complete",
            "instance": identity.get("instance_canonical_sha256") == cell["instance_canonical_sha256"],
            "anchor": identity.get("anchor_sha256") == cell["anchor_sha256"],
            "hybrid": identity.get("source_hybrid_run_key") == cell["source_hybrid_run_key"],
            "scenario_count": result.get("scenario_count") == EXPECTED_SCENARIOS[row["scale"]],
            "direct_benders_disabled": result.get("benders_strategy") == 0,
            "runtime_conservation": math.isclose(
                float(result.get("algorithm_runtime")),
                float(result.get("model_build_runtime")) + float(result.get("optimize_runtime")),
                rel_tol=0.0, abs_tol=1e-9,
            ),
        }
        if not all(checks.values()):
            raise DirectAuditError(f"run audit failed {row['scale']} seed={row['seed']}: {checks}")
        records.append({"scale": row["scale"], "seed": row["seed"],
                        "scientific_status": record["scientific_status"], **result})

    with (output / "paired_results.csv").open(newline="", encoding="utf-8") as handle:
        paired = list(csv.DictReader(handle))
    if len(paired) != 10:
        raise DirectAuditError("paired CSV must contain ten rows")
    if any(row["direct_status"] != "certified_robust_optimal" and (
        row["direct_par2_seconds"] != "3600.0"
        or row["direct_objective_t"] != "NOT_APPLICABLE"
        or row["objective_abs_difference"] != "NOT_APPLICABLE"
        or row["runtime_ratio_direct_over_hybrid"] != "NOT_APPLICABLE"
    ) for row in paired):
        raise DirectAuditError("uncertified comparison reporting drift")

    summaries = []
    for scale in SCALES:
        subset = [record for record in records if record["scale"] == scale]
        cell_subset = [cell_map[(scale, seed)] for seed in SEEDS]
        summaries.append({
            "scale": scale,
            "scenario_count": EXPECTED_SCENARIOS[scale],
            "hybrid_certified": sum(cell["source_hybrid_scientific_status"] == "certified_robust_optimal" for cell in cell_subset),
            "direct_certified": sum(record["scientific_status"] == "certified_robust_optimal" for record in subset),
            "hybrid_mean_runtime_seconds": math.fsum(float(cell["source_hybrid_algorithm_runtime"]) for cell in cell_subset) / 5,
            "direct_mean_raw_runtime_seconds": math.fsum(float(record["algorithm_runtime"]) for record in subset) / 5,
            "direct_mean_par2_seconds": math.fsum(
                float(record["algorithm_runtime"]) if record["scientific_status"] == "certified_robust_optimal" else 3600.0
                for record in subset
            ) / 5,
            "direct_incumbent_count": sum(record.get("incumbent") is not None for record in subset),
            "direct_rows": sorted({int(record["rows"]) for record in subset}),
            "direct_columns": sorted({int(record["columns"]) for record in subset}),
            "direct_nonzeros": sorted({int(record["nonzeros"]) for record in subset}),
        })
    report = {
        "schema": "gurobi_direct_cross_scale_paired_final_audit_v1",
        "decision": "paired_direct_benchmark_complete",
        "source_zip_sha256": file_sha256(Path(config["source_zip"])).upper(),
        "source_zip_unchanged": file_sha256(Path(config["source_zip"])).upper() == SOURCE_SHA256,
        "planned_runs": 10, "completed_runs": len(records),
        "unique_scale_seed_cells": len({(record["scale"], record["seed"]) for record in records}),
        "hybrid_reruns": 0, "baseline_reruns": 0,
        "all_direct_solver_statuses": sorted({record["solver_status"] for record in records}),
        "all_direct_scientific_statuses": sorted({record["scientific_status"] for record in records}),
        "summaries": summaries,
    }
    if not (
        report["source_zip_unchanged"] and report["completed_runs"] == 10
        and report["unique_scale_seed_cells"] == 10
    ):
        raise DirectAuditError("final audit completeness failure")
    atomic_write_json(output / "final_audit.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.config), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
