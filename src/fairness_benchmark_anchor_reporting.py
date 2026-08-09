"""Read-only freeze reporting for the paired benchmark and holdout anchors.

The source ZIPs are never extracted or modified.  This module is intentionally
solver-free and does not import any optimization package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable
from zipfile import ZipFile


BENCHMARK_SHA256 = "8EA91B574748779CAD7E1329A6AE8BCF26D4B6B06655A806A10BDC8CB59C902A"
GAMMA_SHA256 = "EE45A00AA341EE5EB2894DE43EE2F47022C27F1D29146FCFEC803236EF59DB6F"
HOLDOUT_SHA256 = "BD253A4DE967143B8CD88E7DEADCB821D6062208CB71664B900F9BD8EBDF3839"
RUN_COMMIT = "60544f22400a5a34de5c23d556c91c88f9b2df69"
PROTOCOL_SHA256 = "478AF6A88D4DF1678F7902C097EBAEC1DAF41984E33B16A0F1B871FF92790E94"
CONFIG_SHA256 = "AEEFB8EC1BF688A7D60B3CB60A3BAF6CAD084FB08821493293B49594663678AC"
CANDIDATE_SHA256 = "8F96E691E080FBC7CBCC3240BDEE26B08AC040FD049BF808178DA8772C08E009"
AUTHORIZATION_SHA256 = "569A37D2F33CC6206AD415C08762D4C0972EDB27E2A8DD58F4F99755657DAEEC"
TOLERANCE = 1e-4
PAR2 = 3600.0


class ReportingAuditError(RuntimeError):
    pass


def _check(value: bool, message: str) -> None:
    if not value:
        raise ReportingAuditError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ReportingAuditError(f"non-finite JSON token: {value}")


def _finite_tree(value: Any, label: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReportingAuditError(f"non-finite value at {label}")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")


def _json(archive: ZipFile, name: str) -> dict[str, Any]:
    try:
        value = json.loads(archive.read(name).decode("utf-8"), parse_constant=_reject_constant)
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportingAuditError(f"invalid or missing JSON: {name}") from exc
    _check(isinstance(value, dict), f"JSON root is not an object: {name}")
    _finite_tree(value, name)
    return value


def _csv(archive: ZipFile, name: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        reader = csv.DictReader(io.StringIO(archive.read(name).decode("utf-8")), strict=True)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    except (KeyError, UnicodeDecodeError, csv.Error) as exc:
        raise ReportingAuditError(f"invalid or missing CSV: {name}") from exc
    _check(fields and all(None not in row and all(v is not None for v in row.values()) for row in rows),
           f"malformed CSV: {name}")
    for row in rows:
        for value in row.values():
            _check(value.strip().lower() not in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"},
                   f"non-finite CSV token: {name}")
    return fields, rows


def _number(value: Any, label: str) -> float:
    _check(type(value) in {int, float} and math.isfinite(float(value)), f"{label} is not finite numeric")
    return float(value)


def _stats(values: Iterable[float]) -> dict[str, float]:
    data = sorted(float(value) for value in values)
    _check(bool(data), "cannot summarize empty data")

    def quantile(p: float) -> float:
        position = (len(data) - 1) * p
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return data[low]
        return data[low] + (data[high] - data[low]) * (position - low)

    q1, q3 = quantile(.25), quantile(.75)
    return {
        "mean": statistics.fmean(data), "median": statistics.median(data),
        "std": statistics.stdev(data) if len(data) > 1 else 0.0,
        "iqr": q3 - q1, "min": min(data), "max": max(data),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    columns = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _run_directory(run_key: str) -> str:
    return "r_" + hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:24]


def _gamma_source_runs(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    _check(file_sha256(path) == GAMMA_SHA256, "Gamma source ZIP SHA mismatch")
    with ZipFile(path) as archive:
        _check(archive.testzip() is None, "Gamma source ZIP CRC failure")
        runs = {
            value["run_key"]: value
            for name in archive.namelist() if name.endswith("/run.json")
            for value in [_json(archive, name)]
        }
        instances = {}
        for name in archive.namelist():
            if "/instances/" not in name or not name.endswith(".json"):
                continue
            raw = archive.read(name)
            payload = _json(archive, name)
            instances[name] = {
                "file_sha256": bytes_sha256(raw),
                "canonical_sha256": bytes_sha256(canonical_bytes(payload.get("instance"))),
            }
    _check(len(runs) == 60, "Gamma source run count mismatch")
    _check(len(instances) == 30, "Gamma source instance count mismatch")
    return runs, instances


def audit_benchmark(benchmark_zip: Path, gamma_zip: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    before = file_sha256(benchmark_zip)
    _check(before == BENCHMARK_SHA256, f"benchmark ZIP SHA mismatch: {before}")
    source_runs, source_instances = _gamma_source_runs(gamma_zip)
    corrected: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    counts = {"runs": 0, "statuses": 0, "checkpoints": 0, "certified": 0,
              "time_limit_uncertified": 0, "post_evaluations": 0, "chunks": 0,
              "scenarios": 0, "acceptance_evidence": 0, "accepted": 0}
    max_residual = 0.0
    with ZipFile(benchmark_zip) as archive:
        _check(archive.testzip() is None, "benchmark ZIP CRC failure")
        files = [name for name in archive.namelist() if not name.endswith("/")]
        for name in files:
            if name.endswith(".json"):
                _json(archive, name)
            elif name.endswith(".csv"):
                _csv(archive, name)
        manifest = _json(archive, "./manifest.json")
        run_manifest = _json(archive, "./run_manifest.json")
        for key in ("identity", "run_key_to_directory_id", "directory_id_to_run_key"):
            _check(manifest.get(key) == run_manifest.get(key), f"manifest/run_manifest mismatch: {key}")
        _check(isinstance(run_manifest.get("source_cells"), list) and
               len(run_manifest["source_cells"]) == 10, "run_manifest source cell count mismatch")
        source_cells = {cell["source_hybrid_run_key"]: cell for cell in run_manifest["source_cells"]}
        identity = manifest.get("identity", {})
        expected = {
            "git_commit": RUN_COMMIT, "protocol_sha256": PROTOCOL_SHA256,
            "config_file_sha256": CONFIG_SHA256, "candidate_sha256": CANDIDATE_SHA256,
            "execution_attempt": 2, "previous_benchmark_results_reused": False,
            "source_zip_sha256": GAMMA_SHA256,
            "solver_parameters": {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7},
        }
        for key, value in expected.items():
            _check(identity.get(key) == value, f"manifest identity mismatch: {key}")
        forward = manifest.get("run_key_to_directory_id", {})
        reverse = manifest.get("directory_id_to_run_key", {})
        _check(len(forward) == len(reverse) == 10, "run mapping count mismatch")
        _check(all(reverse.get(directory) == key and directory == _run_directory(key)
                   for key, directory in forward.items()), "run mapping not canonical bijection")
        _check(len(set(forward.values())) == 10, "short directory collision")

        for run_key, directory in sorted(forward.items()):
            base = f"./runs/{directory}"
            run = _json(archive, f"{base}/run.json")
            status = _json(archive, f"{base}/status.json")
            checkpoint = _json(archive, f"{base}/algorithm_checkpoint.json")
            counts["runs"] += 1; counts["statuses"] += 1; counts["checkpoints"] += 1
            _check(run.get("run_key") == run_key and run.get("run_directory_id") == directory,
                   f"run identity mismatch: {directory}")
            _check(status.get("identity", {}).get("run_key") == run_key,
                   f"status identity mismatch: {directory}")
            _check(checkpoint.get("identity", {}).get("run_key") == run_key,
                   f"checkpoint identity mismatch: {directory}")
            for key in ("scale", "seed", "gamma", "rho", "candidate_sha256", "config_file_sha256",
                        "protocol_sha256", "git_commit", "instance_file_sha256",
                        "instance_canonical_sha256", "baseline_run_key", "anchor_sha256",
                        "source_hybrid_run_key", "execution_attempt"):
                _check(checkpoint["identity"].get(key) == run.get(key),
                       f"checkpoint/run identity mismatch {directory}:{key}")
            _check(run.get("state") == "complete" and status.get("state") == "complete",
                   f"incomplete benchmark run: {directory}")
            _check(run.get("candidate") == "certified_single_cut_without_complete_scenario_blocks",
                   "reference candidate drift")
            _check(run.get("solver_parameters") == expected["solver_parameters"], "solver parameter drift")
            _check(run.get("previous_benchmark_results_reused") is False, "previous results reused")
            result = run.get("result", {})
            scientific = run.get("scientific_status")
            certified = scientific == "certified_robust_optimal"
            timed_out = scientific == "time_limit_uncertified"
            _check(certified or timed_out, f"unexpected scientific status: {scientific}")
            counts["certified" if certified else "time_limit_uncertified"] += 1
            runtime = _number(result.get("algorithm_runtime"), "algorithm runtime")
            par2 = _number(result.get("penalized_runtime_par2"), "PAR-2")
            _check(math.isclose(par2, runtime if certified else PAR2, abs_tol=1e-9), "PAR-2 mismatch")
            if certified:
                log = result.get("iteration_log")
                _check(isinstance(log, list) and log, "certified iteration log missing")
                final = log[-1]
                _check(final.get("certification_active") is True and
                       final.get("robust_feasibility_certified") is True and
                       final.get("separation_status") == "optimal" and
                       _number(final.get("separation_objective_bound"), "final separation bound") <= TOLERANCE,
                       "final exact certification invalid")
                lower = _number(result.get("lower_bound"), "lower bound")
                upper = _number(result.get("upper_bound"), "upper bound")
                crossing = lower - upper
                _check(crossing <= TOLERANCE, "LB/UB crossing exceeds tolerance")
                post_name = f"{base}/post_evaluation/post_evaluation.json"
                post = _json(archive, post_name)
                evaluation = post.get("evaluation", {})
                _check(evaluation.get("valid") is True and evaluation.get("errors") == [] and
                       evaluation.get("objective_t_consistent") is True, "invalid post-evaluation")
                _check(_number(evaluation.get("actual_robust_cost"), "actual robust cost") <=
                       _number(result.get("cost_budget"), "cost budget") + 1e-7,
                       "certified reference violates cost budget")
                counts["post_evaluations"] += 1
                evidence = evaluation.get("acceptance_evidence", [])
                _check(isinstance(evidence, list), "acceptance evidence missing")
                counts["acceptance_evidence"] += len(evidence)
                counts["accepted"] += sum(item.get("accepted") is True for item in evidence)
                max_residual = max(max_residual, max((_number(item.get("residual"), "residual") for item in evidence), default=0.0))
                index = _json(archive, f"{base}/post_evaluation/checkpoint/index.json")
                chunks = index.get("chunks", [])
                _check([item.get("chunk_index") for item in chunks] == list(range(len(chunks))), "chunk order mismatch")
                scenario_sum = 0
                for item in chunks:
                    chunk_name = f"{base}/post_evaluation/{item['relative_path']}"
                    raw = archive.read(chunk_name)
                    _check(hashlib.sha256(raw).hexdigest() == item.get("sha256"), f"chunk SHA mismatch: {chunk_name}")
                    chunk = _json(archive, chunk_name)
                    _check(len(chunk.get("records", [])) == item.get("scenario_count"), "chunk scenario count mismatch")
                    scenario_sum += item["scenario_count"]
                _check(scenario_sum == evaluation.get("scenario_count") == 1831, "post scenario count mismatch")
                counts["chunks"] += len(chunks); counts["scenarios"] += scenario_sum
            else:
                _check(result.get("status") == "time_limit" and result.get("upper_bound") is None and
                       result.get("objective_t") is None and not result.get("post_evaluation"),
                       "uncertified time-limit schema mismatch")
                _check(not any(name.startswith(f"{base}/post_evaluation/") for name in files),
                       "uncertified run has post-evaluation")
                crossing = "NOT_APPLICABLE"

            source = source_runs.get(run.get("source_hybrid_run_key"))
            source_cell = source_cells.get(run.get("source_hybrid_run_key"))
            _check(source is not None and source.get("scientific_status") == "certified_robust_optimal",
                   "source Hybrid run missing or uncertified")
            _check(source_cell is not None, "source pairing cell missing")
            source_instance = source_instances.get(source_cell.get("instance_member"))
            _check(source_instance is not None and
                   source_instance["file_sha256"] == run.get("instance_file_sha256") and
                   source_instance["canonical_sha256"] == run.get("instance_canonical_sha256"),
                   "source instance file/canonical SHA mismatch")
            for key in ("scale", "seed", "gamma", "baseline_run_key", "anchor_sha256"):
                _check(source.get(key) == run.get(key), f"strict pair mismatch: {key}")
            _check(source.get("instance_canonical_sha256") == run.get("instance_canonical_sha256") and
                   source_cell.get("instance_file_sha256") == run.get("instance_file_sha256"),
                   "strict pair instance mismatch")
            hybrid_result = source["result"]
            reference_post = result.get("post_evaluation") or {}
            row = {
                "run_key": run_key, "run_directory_id": directory, "scale": run["scale"],
                "seed": run["seed"], "gamma": run["gamma"], "rho": run["rho"],
                "candidate": run["candidate"], "scientific_status": scientific,
                "algorithm_runtime": runtime,
                "master_runtime": _number(result.get("master_runtime"), "master runtime"),
                "separation_runtime": _number(result.get("separation_runtime"), "separation runtime"),
                "post_evaluation_wall_runtime": _number(result.get("post_evaluation_wall_runtime", 0.0), "post runtime"),
                "total_wall_runtime": _number(result.get("total_wall_runtime", runtime), "total runtime"),
                "penalized_runtime_par2": par2, "final_gap": result.get("gap") if certified else "NOT_APPLICABLE",
                "iterations": result.get("iterations"), "scenario_blocks": 0,
                "certified_farkas_cuts": result.get("cuts"),
                "objective_t": result.get("objective_t") if certified else "NOT_APPLICABLE",
                "actual_robust_cost": reference_post.get("actual_robust_cost", "NOT_APPLICABLE"),
                "instance_file_sha256": run["instance_file_sha256"],
                "instance_canonical_sha256": run["instance_canonical_sha256"],
                "baseline_run_key": run["baseline_run_key"], "anchor_sha256": run["anchor_sha256"],
                "raw_bound_crossing": crossing,
                "bound_crossing_within_tolerance": bool(certified and crossing > 0 and crossing <= TOLERANCE),
            }
            corrected.append(row)
            both = certified
            ref_cost = reference_post.get("actual_robust_cost") if both else None
            hyb_cost = hybrid_result.get("post_evaluation", {}).get("actual_robust_cost") if both else None
            paired.append({
                "scale": run["scale"], "seed": run["seed"], "gamma": 2, "rho": 0.025,
                "instance_sha256": run["instance_canonical_sha256"],
                "baseline_run_key": run["baseline_run_key"], "anchor_sha256": run["anchor_sha256"],
                "hybrid_scientific_status": source["scientific_status"], "reference_scientific_status": scientific,
                "hybrid_algorithm_runtime": hybrid_result["algorithm_runtime"], "reference_algorithm_runtime": runtime,
                "hybrid_par2": hybrid_result["penalized_runtime_par2"], "reference_par2": par2,
                "par2_difference_reference_minus_hybrid": par2 - hybrid_result["penalized_runtime_par2"],
                "par2_ratio_reference_over_hybrid": par2 / hybrid_result["penalized_runtime_par2"],
                "runtime_difference_reference_minus_hybrid": runtime - hybrid_result["algorithm_runtime"] if both else "NOT_APPLICABLE",
                "runtime_ratio_reference_over_hybrid": runtime / hybrid_result["algorithm_runtime"] if both else "NOT_APPLICABLE",
                "hybrid_iterations": hybrid_result["iterations"], "reference_iterations": result["iterations"],
                "hybrid_scenario_blocks": hybrid_result["metadata"]["committed_scenario_count"],
                "reference_scenario_blocks": 0, "hybrid_certified_farkas_cuts": hybrid_result["cuts"],
                "reference_certified_farkas_cuts": result["cuts"],
                "objective_t_difference_reference_minus_hybrid": result["objective_t"] - hybrid_result["objective_t"] if both else "NOT_APPLICABLE",
                "cost_difference_reference_minus_hybrid": ref_cost - hyb_cost if both else "NOT_APPLICABLE",
                "certification_agreement": certified,
            })
    _check(counts == {"runs": 10, "statuses": 10, "checkpoints": 10, "certified": 3,
                      "time_limit_uncertified": 7, "post_evaluations": 3, "chunks": 222,
                      "scenarios": 5493, "acceptance_evidence": 60426, "accepted": 60426},
           f"benchmark coverage mismatch: {counts}")
    after = file_sha256(benchmark_zip)
    _check(after == before, "benchmark source ZIP changed during audit")
    audit = {
        "decision": "approve_minimal_paired_benchmark_after_reporting_reconciliation",
        "archive_sha256_before": before, "archive_sha256_after": after, "crc_valid": True,
        "coverage": counts, "pipeline_failures": 0, "maximum_acceptance_residual": max_residual,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "authorization_provenance_note": "The immutable manifest omits the external authorization SHA. The original config kept formal_run_authorized=false; the separately reviewed authorization file activated the run. This is derived provenance, not an original run field.",
        "reference_algorithm": {
            "candidate": "certified_single_cut_without_complete_scenario_blocks",
            "complete_scenario_recourse_blocks": False, "persistent_separation": False,
            "certified_cache": False, "solution_pool": False, "batch_size": 1,
            "maximum_certified_farkas_cuts_per_iteration": 1,
            "certified_farkas_separation": True, "final_exact_separation": True,
        },
    }
    return audit, sorted(corrected, key=lambda r: (r["scale"], r["seed"])), sorted(paired, key=lambda r: (r["scale"], r["seed"]))


def _summary_rows(corrected: list[dict[str, Any]], paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scale in ("medium_large", "large", "overall"):
        selected = corrected if scale == "overall" else [row for row in corrected if row["scale"] == scale]
        selected_pairs = paired if scale == "overall" else [row for row in paired if row["scale"] == scale]
        certified = sum(row["scientific_status"] == "certified_robust_optimal" for row in selected)
        base = {"scale": scale, "planned": len(selected), "completed": len(selected),
                "certified": certified, "certification_rate": certified / len(selected)}
        for metric in ("algorithm_runtime", "penalized_runtime_par2", "iterations", "certified_farkas_cuts", "scenario_blocks"):
            for stat, value in _stats(row[metric] for row in selected).items():
                base[f"{metric}_{stat}"] = value
        for metric in ("par2_difference_reference_minus_hybrid", "par2_ratio_reference_over_hybrid"):
            for stat, value in _stats(row[metric] for row in selected_pairs).items():
                base[f"{metric}_{stat}"] = value
        successful = [row for row in selected_pairs if row["certification_agreement"]]
        base["both_certified"] = len(successful)
        for metric in ("runtime_difference_reference_minus_hybrid", "runtime_ratio_reference_over_hybrid",
                       "objective_t_difference_reference_minus_hybrid", "cost_difference_reference_minus_hybrid"):
            values = [float(row[metric]) for row in successful]
            for stat, value in (_stats(values) if values else {key: "NOT_APPLICABLE" for key in ("mean", "median", "std", "iqr", "min", "max")}).items():
                base[f"both_certified_{metric}_{stat}"] = value
        rows.append(base)
    return rows


def _plot_benchmark(output: Path, paired: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                                 "pdf.compression": 9, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for index, scale in enumerate(("medium_large", "large")):
        rows = [row for row in paired if row["scale"] == scale]
        axes[index].bar([0, 1], [statistics.fmean(float(row["hybrid_par2"]) for row in rows),
                                 statistics.fmean(float(row["reference_par2"]) for row in rows)],
                        color=["#0072B2", "#D55E00"])
        axes[index].set_xticks([0, 1], ["Hybrid", "Reference"])
        axes[index].set_ylabel("Mean PAR-2 (s)")
        axes[index].set_title(scale.replace("_", "-").title())
        axes[index].grid(axis="y", alpha=.25)
    metadata = {"Creator": "deterministic paired benchmark reporting", "CreationDate": None, "ModDate": None}
    fig.savefig(output / "figure_algorithm_benchmark.pdf", metadata=metadata)
    fig.savefig(output / "figure_algorithm_benchmark.png", metadata={"Software": "deterministic paired benchmark reporting"})
    plt.close(fig)


def write_benchmark_reports(audit: dict[str, Any], corrected: list[dict[str, Any]],
                            paired: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    summary = _summary_rows(corrected, paired)
    successful = [row for row in paired if row["certification_agreement"]]
    max_t_difference = max(abs(float(row["objective_t_difference_reference_minus_hybrid"])) for row in successful)
    seed184 = next(row for row in paired if row["scale"] == "medium_large" and row["seed"] == 184)
    reconciliation = {
        "tolerance": TOLERANCE,
        "medium_large_seed_184": {
            "raw_bound_crossing": next(row["raw_bound_crossing"] for row in corrected if row["scale"] == "medium_large" and row["seed"] == 184),
            "classification": "numerical_reconciliation_within_frozen_tolerance",
            "certification_status_changed": False,
        },
        "both_certified_cells": 3, "maximum_absolute_objective_t_difference": max_t_difference,
        "objective_t_difference_within_tolerance": max_t_difference < TOLERANCE,
        "medium_large_seed_184_cost_difference": seed184["cost_difference_reference_minus_hybrid"],
        "cost_interpretation": "Cost is a feasibility budget, not a secondary optimization objective. Equal T does not imply a unique cost-minimizing fairness solution.",
    }
    decision = {"decision": "approve_minimal_paired_benchmark_after_reporting_reconciliation",
                "scientific_results_valid": True, "optimization_rerun_required": False,
                "source_archives_modified": False, "gurobi_called": False}
    provenance = {
        "benchmark_zip_sha256": BENCHMARK_SHA256, "gamma_source_zip_sha256": GAMMA_SHA256,
        "run_git_commit": RUN_COMMIT, "protocol_sha256": PROTOCOL_SHA256,
        "config_sha256": CONFIG_SHA256, "candidate_sha256": CANDIDATE_SHA256,
        "external_authorization_sha256": AUTHORIZATION_SHA256,
        "authorization_persisted_in_original_manifest": False,
        "original_config_formal_run_authorized": False,
        "execution_attempt": 2, "previous_benchmark_results_reused": False,
    }
    _write_json(output / "decision.json", decision)
    _write_json(output / "source_archive_provenance.json", provenance)
    _write_json(output / "final_audit.json", audit)
    _write_json(output / "numerical_reconciliation.json", reconciliation)
    _write_csv(output / "results.corrected.csv", corrected)
    _write_csv(output / "summary.corrected.csv", summary)
    _write_csv(output / "paired_comparison.corrected.csv", paired)
    table = [{"scale": row["scale"], "hybrid_certified": "5/5" if row["scale"] != "overall" else "10/10",
              "reference_certified": f"{row['certified']}/{row['planned']}",
              "hybrid_mean_par2": statistics.fmean(float(item["hybrid_par2"]) for item in paired if row["scale"] == "overall" or item["scale"] == row["scale"]),
              "reference_mean_par2": row["penalized_runtime_par2_mean"]}
             for row in summary]
    _write_csv(output / "table_algorithm_benchmark.csv", table)
    lines = ["# Minimal paired algorithm benchmark", "",
             "All ten preregistered cells are retained; uncertified reference cells receive the frozen 3600 s PAR-2 penalty.", "",
             "| Scale | Hybrid certified | Reference certified | Hybrid mean PAR-2 (s) | Reference mean PAR-2 (s) |",
             "|---|---:|---:|---:|---:|"]
    for row in table[:2]:
        lines.append(f"| {row['scale']} | {row['hybrid_certified']} | {row['reference_certified']} | {row['hybrid_mean_par2']:.3f} | {row['reference_mean_par2']:.3f} |")
    lines.extend(["", "The three jointly certified cells agree in objective T within the frozen tolerance. Cost is not a secondary objective, so the seed-184 cost difference is not evidence of a scientific contradiction or solution uniqueness.", ""])
    (output / "table_algorithm_benchmark.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (output / "decision.md").write_text(
        "# Decision\n\n`approve_minimal_paired_benchmark_after_reporting_reconciliation`\n\n"
        "The immutable source passed 10/10 completion checks. Hybrid certified 10/10; the reference certified 3/10 and timed out uncertified in 7/10. Corrected master and separation runtimes are derived directly from each immutable run.json. No optimization result was altered.\n",
        encoding="utf-8", newline="\n")
    _plot_benchmark(output, paired)
    _artifact_index(output)


def _vector_l1(left: Any, right: Any, label: str) -> float:
    _check(isinstance(left, list) and isinstance(right, list) and len(left) == len(right), f"{label} shape mismatch")
    total = 0.0
    for index, (a, b) in enumerate(zip(left, right)):
        if isinstance(a, list) or isinstance(b, list):
            total += _vector_l1(a, b, f"{label}[{index}]")
        else:
            total += abs(_number(a, label) - _number(b, label))
    return total


def audit_holdout_anchors(holdout_zip: Path, repo_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from src.fairness_hybrid_final_holdout_reconciliation import audit_archive

    before = file_sha256(holdout_zip)
    _check(before == HOLDOUT_SHA256, "Final Holdout ZIP SHA mismatch")
    full_audit, records = audit_archive(holdout_zip, repo_root)
    run_records = [item["record"] for item in records]
    anchors: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    utilization: list[dict[str, Any]] = []
    with ZipFile(holdout_zip) as archive:
        for scale, directory in (("medium_large", "ml_a1"), ("large", "lg_a1")):
            manifest = _json(archive, f"results_fairness_hybrid_final_holdout/{directory}/manifest.json")
            anchor_map = manifest.get("baseline_anchors", {})
            for seed in range(170, 180):
                baseline = next(run for run in run_records if run["scale"] == scale and run["seed"] == seed and run["task_type"] == "baseline")
                result = baseline["result"]
                anchor = anchor_map[str(seed)]
                lower = _number(result["lower_bound"], "baseline lower bound")
                upper = _number(result["upper_bound"], "baseline upper bound")
                anchor_value = _number(anchor["value"], "anchor")
                _check(math.isclose(upper, anchor_value, rel_tol=0, abs_tol=1e-9), "anchor/UB mismatch")
                gap = (anchor_value - lower) / max(1.0, abs(anchor_value))
                _check(0 <= gap <= TOLERANCE, "baseline anchor gap outside frozen tolerance")
                anchors.append({"scale": scale, "seed": seed, "baseline_run_key": baseline["run_key"],
                                "instance_sha256": baseline["instance_sha256"], "lower_bound": lower,
                                "upper_bound": upper, "C_anchor": anchor_value,
                                "absolute_gap": anchor_value - lower, "relative_gap_g_i": gap,
                                "solver_status": result["status"], "valid_UB": result["valid_UB"],
                                "anchor_sha256": anchor["anchor_sha256"],
                                "anchor_value_hex": anchor["anchor_value_hex"],
                                "rho_0_max_relative_slack_to_unknown_optimum": gap,
                                "rho_001_max_effective_cost_increment": 1.01 / (1.0 - gap) - 1.0})
                frontier = {float(run["rho"]): run for run in run_records if run["scale"] == scale and run["seed"] == seed and run["task_type"] == "frontier"}
                zero, one = frontier[0.0], frontier[0.01]
                rz, ro = zero["result"], one["result"]
                pz, po = rz["post_evaluation"], ro["post_evaluation"]
                cost0, cost1 = _number(pz["actual_robust_cost"], "rho0 cost"), _number(po["actual_robust_cost"], "rho1 cost")
                budget0, budget1 = anchor_value, 1.01 * anchor_value
                y_l1 = _vector_l1(rz["y_values"], ro["y_values"], "y")
                x_l1 = _vector_l1(rz["x_values"], ro["x_values"], "x")
                changes.append({"scale": scale, "seed": seed,
                                "actual_robust_cost_rho_000": cost0, "actual_robust_cost_rho_001": cost1,
                                "actual_robust_cost_change": cost1 - cost0,
                                "robust_minimum_fill_rate_rho_000": rz["robust_minimum_fill_rate"],
                                "robust_minimum_fill_rate_rho_001": ro["robust_minimum_fill_rate"],
                                "robust_minimum_fill_rate_change": ro["robust_minimum_fill_rate"] - rz["robust_minimum_fill_rate"],
                                "wminfr_rho_000": pz["wminfr"], "wminfr_rho_001": po["wminfr"],
                                "wminfr_change": po["wminfr"] - pz["wminfr"],
                                "actual_price_of_fairness_rho_000": pz["actual_price_of_fairness"],
                                "actual_price_of_fairness_rho_001": po["actual_price_of_fairness"],
                                "budget_utilization_rho_000": cost0 / budget0,
                                "budget_utilization_rho_001": cost1 / budget1,
                                "budget_slack_rho_000": budget0 - cost0, "budget_slack_rho_001": budget1 - cost1,
                                "opened_warehouses_rho_000": sum(_number(v, "y") >= .5 for v in rz["y_values"]),
                                "opened_warehouses_rho_001": sum(_number(v, "y") >= .5 for v in ro["y_values"]),
                                "opened_warehouse_change": sum(_number(v, "y") >= .5 for v in ro["y_values"]) - sum(_number(v, "y") >= .5 for v in rz["y_values"]),
                                "y_vector_l1_change": y_l1, "x_inventory_l1_change": x_l1,
                                "discrete_first_stage_configuration_switch": y_l1 > 1e-9})
                for rho, run in ((0.0, zero), (0.01, one)):
                    cost = _number(run["result"]["post_evaluation"]["actual_robust_cost"], "actual cost")
                    budget = (1.0 + rho) * anchor_value
                    utilization.append({"scale": scale, "seed": seed, "rho": rho,
                                        "C_anchor": anchor_value, "cost_budget": budget,
                                        "actual_robust_cost": cost, "budget_utilization_u": cost / budget,
                                        "budget_slack_fraction_s": 1.0 - cost / budget,
                                        "budget_slack_absolute": budget - cost})
    after = file_sha256(holdout_zip)
    _check(after == before, "Final Holdout ZIP changed during audit")
    audit = {"archive_sha256_before": before, "archive_sha256_after": after,
             "full_holdout_audit_status": full_audit["status"], "baseline_count": len(anchors),
             "frontier_pairs_rho_0_to_001": len(changes),
             "all_baseline_gaps_within_1e_4": all(row["relative_gap_g_i"] <= TOLERANCE for row in anchors),
             "interpretation": "The certified anchor error bound is too small to explain the observed service jump from rho=0 to rho=0.01."}
    return audit, anchors, changes, utilization


def _plot_threshold(output: Path, changes: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                                 "pdf.compression": 9, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for axis, scale in zip(axes, ("medium_large", "large")):
        rows = [row for row in changes if row["scale"] == scale]
        axis.bar([row["seed"] for row in rows], [row["robust_minimum_fill_rate_change"] for row in rows], color="#0072B2")
        axis.set_title(scale.replace("_", "-").title()); axis.set_xlabel("Seed")
        axis.set_ylabel("Change in certified minimum fill rate"); axis.grid(axis="y", alpha=.25)
    metadata = {"Creator": "deterministic holdout anchor reporting", "CreationDate": None, "ModDate": None}
    fig.savefig(output / "figure_rho_threshold_seed_changes.pdf", metadata=metadata)
    plt.close(fig)


def write_anchor_reports(audit: dict[str, Any], anchors: list[dict[str, Any]],
                         changes: list[dict[str, Any]], utilization: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for scale in ("medium_large", "large", "overall"):
        rows = anchors if scale == "overall" else [row for row in anchors if row["scale"] == scale]
        item = {"scale": scale, "count": len(rows)}
        for metric in ("absolute_gap", "relative_gap_g_i", "rho_0_max_relative_slack_to_unknown_optimum",
                       "rho_001_max_effective_cost_increment"):
            for stat, value in _stats(row[metric] for row in rows).items():
                item[f"{metric}_{stat}"] = value
        summaries.append(item)
    _write_csv(output / "anchor_quality_by_seed.csv", anchors)
    _write_csv(output / "anchor_quality_summary.csv", summaries)
    _write_csv(output / "rho_000_to_001_seed_changes.csv", changes)
    _write_csv(output / "rho_budget_utilization.csv", utilization)
    _write_json(output / "final_audit.json", audit)
    table_anchor = ["# Anchor quality", "", "| Scale | max relative gap | mean | median | max effective rho=0.01 increment |", "|---|---:|---:|---:|---:|"]
    for row in summaries:
        table_anchor.append(f"| {row['scale']} | {row['relative_gap_g_i_max']:.8f} | {row['relative_gap_g_i_mean']:.8f} | {row['relative_gap_g_i_median']:.8f} | {row['rho_001_max_effective_cost_increment_max']:.8f} |")
    (output / "table_anchor_quality.md").write_text("\n".join(table_anchor) + "\n", encoding="utf-8", newline="\n")
    threshold = ["# rho=0 to rho=0.01 diagnostics", "", "| Scale | Seed | fill-rate change | cost change | y switch | x L1 change |", "|---|---:|---:|---:|---:|---:|"]
    for row in changes:
        threshold.append(f"| {row['scale']} | {row['seed']} | {row['robust_minimum_fill_rate_change']:.6f} | {row['actual_robust_cost_change']:.3f} | {row['discrete_first_stage_configuration_switch']} | {row['x_inventory_l1_change']:.3f} |")
    (output / "table_rho_threshold_diagnostics.md").write_text("\n".join(threshold) + "\n", encoding="utf-8", newline="\n")
    by_scale = {scale: _stats(row["robust_minimum_fill_rate_change"] for row in changes if row["scale"] == scale) for scale in ("medium_large", "large")}
    interpretation = "# Anchor and threshold interpretation\n\nAll 20 baseline anchors are certified within the frozen 1e-4 relative tolerance. The bound g_i limits how far each anchor can lie above the unknown exact robust optimum; the corresponding conservative rho=0.01 effective increment is `(1.01)/(1-g_i)-1`. These bounds are orders of magnitude below the observed service-level changes, so anchor conservatism cannot explain the rho=0 to 0.01 jump.\n\nThe seed-level table reports budget use, continuous inventory movement, and discrete warehouse-opening switches. Conclusions are based on all ten seeds per scale, not selected cases.\n"
    (output / "anchor_interpretation.md").write_text(interpretation, encoding="utf-8", newline="\n")
    _plot_threshold(output, changes)
    _artifact_index(output)


def _artifact_index(output: Path) -> None:
    paths = sorted(path for path in output.iterdir() if path.name != "artifact_sha256.csv")
    _write_csv(output / "artifact_sha256.csv", [{"relative_path": path.name, "sha256": file_sha256(path)} for path in paths])


def generate_all(benchmark_zip: Path, gamma_zip: Path, holdout_zip: Path,
                 repo_root: Path, benchmark_output: Path, anchor_output: Path) -> dict[str, Any]:
    benchmark_audit, corrected, paired = audit_benchmark(benchmark_zip, gamma_zip)
    write_benchmark_reports(benchmark_audit, corrected, paired, benchmark_output)
    anchor_audit, anchors, changes, utilization = audit_holdout_anchors(holdout_zip, repo_root)
    write_anchor_reports(anchor_audit, anchors, changes, utilization, anchor_output)
    return {"benchmark_decision": benchmark_audit["decision"], "holdout_anchor_audit": "pass",
            "gurobipy_imported": False, "solver_called": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-zip", type=Path, required=True)
    parser.add_argument("--gamma-zip", type=Path, required=True)
    parser.add_argument("--holdout-zip", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--anchor-output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate_all(args.benchmark_zip, args.gamma_zip, args.holdout_zip,
                                  args.repo_root, args.benchmark_output, args.anchor_output),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
