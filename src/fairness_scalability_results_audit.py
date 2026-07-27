from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import subprocess
from typing import Any, Iterable, Mapping
import zipfile

import yaml


CERTIFIED_STATUS = "certified_robust_optimal"
SCIENTIFIC_STATUSES = (
    CERTIFIED_STATUS,
    "master_optimal_but_robust_uncertified",
    "time_limit_uncertified",
    "infeasible",
    "invalid_post_evaluation",
    "implementation_error",
    "interrupted",
    "iteration_limit_uncertified",
    "numerical_uncertified",
    "unknown_uncertified",
)
CANONICALIZATION = "PyYAML safe_dump(sort_keys=True, allow_unicode=True), UTF-8"
CORE_SOLVER_FILES = (
    "src/fairness_scalability.py",
    "src/fairness_benders.py",
    "src/robust_regional_fairness.py",
)

# These definitions are deliberately machine-readable. A task-level metadata
# total is checked against the iteration-log sum; the sole mean is checked
# against the iteration-log arithmetic mean. Baseline rows use structural zero
# for fairness-separation fields, which are not applicable to baseline solves.
FIELD_PROJECTIONS: dict[str, dict[str, str]] = {
    "separation_model_build_runtime": {
        "source": "run.json.result.metadata.separation_model_build_runtime",
        "aggregation": "task_metadata_total_verified_against_iteration_log_sum",
    },
    "separation_optimize_runtime": {
        "source": "run.json.result.metadata.separation_optimize_runtime",
        "aggregation": "task_metadata_total_verified_against_iteration_log_sum",
    },
    "cache_candidate_count": {
        "source": "run.json.result.metadata.cache_candidate_count",
        "aggregation": "task_metadata_total_verified_against_iteration_log_sum",
    },
    "cache_hit_count": {
        "source": "run.json.result.metadata.cache_hit_count",
        "aggregation": "task_metadata_total_verified_against_iteration_log_sum",
    },
    "certified_cached_cut_count": {
        "source": "run.json.result.metadata.certified_cached_cut_count",
        "aggregation": "task_metadata_total_verified_against_iteration_log_sum",
    },
    "pool_candidate_count": {
        "source": "run.json.result.metadata.pool_candidate_count",
        "aggregation": "task_metadata_total_verified_against_iteration_log_sum",
    },
    "certified_batch_cut_count": {
        "source": "run.json.result.metadata.certified_batch_cut_count",
        "aggregation": "task_metadata_total_verified_against_iteration_log_sum",
    },
    "duplicate_pattern_count": {
        "source": "run.json.result.metadata.duplicate_pattern_count",
        "aggregation": "task_metadata_total_verified_against_iteration_log_sum",
    },
    "cuts_per_iteration": {
        "source": "run.json.result.metadata.cuts_per_iteration",
        "aggregation": "task_metadata_mean_verified_against_iteration_log_mean",
    },
}

RESULT_FIELDS = [
    "source_archive_sha256", "run_key", "introduced_stage", "task_type", "scale",
    "seed", "rho", "candidate", "baseline_run_key", "anchor_sha256",
    "scientific_status", "algorithm_status", "solved_to_tolerance", "objective_t",
    "separation_runtime", "separation_model_build_runtime",
    "separation_optimize_runtime", "master_runtime", "cache_candidate_count",
    "cache_hit_count", "certified_cached_cut_count", "pool_candidate_count",
    "certified_batch_cut_count", "duplicate_pattern_count", "cuts_per_iteration",
    "total_iterations", "cuts", "algorithm_runtime", "penalized_runtime_par2",
    "post_evaluation_solver_runtime", "post_evaluation_wall_runtime",
    "aggregation_runtime", "checkpoint_io_runtime", "total_wall_runtime",
]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_config_bytes(config: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(config), sort_keys=True, allow_unicode=True
    ).encode("utf-8")


def resolved_config_file_bytes(config: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(config), sort_keys=False, allow_unicode=True
    ).encode("utf-8")


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _iteration_values(iteration_log: Iterable[Mapping[str, Any]], field: str) -> list[float]:
    return [_number(entry.get(field)) for entry in iteration_log]


def _project_metadata_metric(
    result: Mapping[str, Any], field: str, *, task_type: str
) -> float:
    if task_type != "frontier":
        return 0.0
    metadata = result.get("metadata") or {}
    if field in metadata and metadata[field] is not None:
        return _number(metadata[field])
    values = _iteration_values(result.get("iteration_log") or [], field)
    if not values:
        return 0.0
    if FIELD_PROJECTIONS[field]["aggregation"].endswith("iteration_log_mean"):
        return math.fsum(values) / len(values)
    return math.fsum(values)


def project_record(
    record: Mapping[str, Any],
    spec: Mapping[str, Any] | None = None,
    *,
    source_archive_sha256: str = "NOT_APPLICABLE_FUTURE_RUNNER",
    time_limit: float = 1800.0,
) -> dict[str, Any]:
    spec = dict(spec or {})
    result = dict(record.get("result") or {})
    task_type = str(record.get("task_type", spec.get("task_type", "")))
    scientific_status = str(record.get("scientific_status", ""))
    algorithm_runtime = _number(result.get("algorithm_runtime", result.get("runtime")))
    par2 = result.get("penalized_runtime_par2")
    if par2 is None:
        par2 = algorithm_runtime if scientific_status == CERTIFIED_STATUS else 2.0 * time_limit
    total_iterations = result.get("metadata", {}).get(
        "total_iterations", result.get("total_iterations", result.get("iterations", 0))
    )
    row: dict[str, Any] = {
        "source_archive_sha256": source_archive_sha256,
        "run_key": record.get("run_key", spec.get("run_key")),
        "introduced_stage": spec.get("introduced_stage", record.get("introduced_stage")),
        "task_type": task_type,
        "scale": spec.get("scale", record.get("scale")),
        "seed": record.get("seed", spec.get("seed")),
        "rho": record.get("rho", spec.get("rho")),
        "candidate": record.get("candidate", spec.get("candidate")),
        "baseline_run_key": record.get("baseline_run_key") or "NOT_APPLICABLE",
        "anchor_sha256": record.get("anchor_sha256") or "NOT_APPLICABLE",
        "scientific_status": scientific_status,
        "algorithm_status": record.get("algorithm_status") or "unknown",
        "solved_to_tolerance": scientific_status == CERTIFIED_STATUS,
        "objective_t": result.get("objective_t", "NOT_APPLICABLE"),
        "separation_runtime": _number(result.get("separation_runtime")),
        "master_runtime": _number(result.get("master_runtime")),
        "total_iterations": int(_number(total_iterations)),
        "cuts": int(_number(result.get("cuts"))),
        "algorithm_runtime": algorithm_runtime,
        "penalized_runtime_par2": _number(par2),
        "post_evaluation_solver_runtime": _number(result.get("post_evaluation_solver_runtime")),
        "post_evaluation_wall_runtime": _number(result.get("post_evaluation_wall_runtime")),
        "aggregation_runtime": _number(result.get("aggregation_runtime")),
        "checkpoint_io_runtime": _number(result.get("checkpoint_io_runtime")),
        "total_wall_runtime": _number(result.get("total_wall_runtime", result.get("runtime"))),
    }
    for field in FIELD_PROJECTIONS:
        row[field] = _project_metadata_metric(result, field, task_type=task_type)
    return {field: row.get(field, "") for field in RESULT_FIELDS}


def aggregate_records(
    records: Iterable[Mapping[str, Any]],
    specs: Iterable[Mapping[str, Any]],
    *,
    source_archive_sha256: str = "NOT_APPLICABLE_FUTURE_RUNNER",
    time_limit: float = 1800.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec_by_key = {str(spec["run_key"]): dict(spec) for spec in specs}
    rows = [
        project_record(
            record,
            spec_by_key.get(str(record.get("run_key"))),
            source_archive_sha256=source_archive_sha256,
            time_limit=time_limit,
        )
        for record in records
    ]
    rows.sort(
        key=lambda row: (
            int(row["seed"]), row["task_type"] != "baseline",
            -1.0 if row["rho"] is None else float(row["rho"]), str(row["candidate"]),
        )
    )
    grouped: dict[tuple[str, str, float | None], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["task_type"]), str(row["candidate"]), row["rho"]), []
        ).append(row)
    summary: list[dict[str, Any]] = []
    mean_fields = (
        "algorithm_runtime", "penalized_runtime_par2", "separation_runtime",
        "total_wall_runtime", "total_iterations", "cuts", "cache_candidate_count",
        "cache_hit_count", "certified_cached_cut_count", "pool_candidate_count",
        "certified_batch_cut_count", "duplicate_pattern_count", "cuts_per_iteration",
    )
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        solved = sum(row["scientific_status"] == CERTIFIED_STATUS for row in group)
        item: dict[str, Any] = {
            "source_archive_sha256": source_archive_sha256,
            "task_type": key[0], "candidate": key[1], "rho": key[2],
            "run_count": len(group), "certified_solved_count": solved,
            "solved_rate": solved / len(group),
        }
        for status in SCIENTIFIC_STATUSES:
            item[f"status_{status}_count"] = sum(
                row["scientific_status"] == status for row in group
            )
        item["mean_algorithm_runtime"] = statistics.fmean(
            _number(row["algorithm_runtime"]) for row in group
        )
        item["median_algorithm_runtime"] = statistics.median(
            _number(row["algorithm_runtime"]) for row in group
        )
        for field in mean_fields[1:]:
            item[f"mean_{field}"] = statistics.fmean(_number(row[field]) for row in group)
        for field in (
            "cache_candidate_count", "cache_hit_count", "certified_cached_cut_count",
            "pool_candidate_count", "certified_batch_cut_count", "duplicate_pattern_count",
        ):
            item[f"total_{field}"] = math.fsum(_number(row[field]) for row in group)
        summary.append(item)
    return rows, summary


class ResultsSource:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self._zip is not None:
            self.names = sorted(
                name.removeprefix("./") for name in self._zip.namelist() if not name.endswith("/")
            )
        elif self.path.is_dir():
            self.names = sorted(
                child.relative_to(self.path).as_posix()
                for child in self.path.rglob("*") if child.is_file()
            )
        else:
            raise FileNotFoundError(self.path)

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def read_bytes(self, name: str) -> bytes:
        normalized = PurePosixPath(name).as_posix().removeprefix("./")
        if self._zip is not None:
            candidates = (normalized, f"./{normalized}")
            for candidate in candidates:
                try:
                    return self._zip.read(candidate)
                except KeyError:
                    pass
            raise FileNotFoundError(normalized)
        return (self.path / Path(*PurePosixPath(normalized).parts)).read_bytes()

    def read_json(self, name: str) -> dict[str, Any]:
        value = json.loads(self.read_bytes(name))
        if not isinstance(value, dict):
            raise ValueError(f"{name} must contain a JSON object")
        return value

    def find_one(self, suffix: str) -> str:
        matches = [name for name in self.names if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one *{suffix}; found {len(matches)}")
        return matches[0]

    def crc_ok(self) -> bool:
        return self._zip is None or self._zip.testzip() is None


def _projection_checks(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for record in records:
        if record.get("task_type") != "frontier":
            continue
        result = record.get("result") or {}
        metadata = result.get("metadata") or {}
        iteration_log = result.get("iteration_log") or []
        for field, definition in FIELD_PROJECTIONS.items():
            values = _iteration_values(iteration_log, field)
            expected = (
                math.fsum(values) / len(values)
                if definition["aggregation"].endswith("iteration_log_mean") and values
                else math.fsum(values)
            )
            actual = _number(metadata.get(field))
            checks.append({
                "run_key": record.get("run_key"), "field": field,
                "metadata_value": actual, "iteration_log_recomputed_value": expected,
                "passed": math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9),
            })
    return checks


def _checkpoint_audit(source: ResultsSource) -> dict[str, Any]:
    indexes = [name for name in source.names if name.endswith("post_evaluation/checkpoint/index.json")]
    checked = 0
    mismatches: list[str] = []
    for index_name in indexes:
        index = source.read_json(index_name)
        base = index_name.removesuffix("checkpoint/index.json")
        for chunk in index.get("chunks", []):
            chunk_name = f"{base}{chunk['relative_path']}"
            checked += 1
            if _sha256_bytes(source.read_bytes(chunk_name)).lower() != str(chunk["sha256"]).lower():
                mismatches.append(chunk_name)
    return {
        "post_evaluation_index_count": len(indexes),
        "checkpoint_chunk_hash_count": checked,
        "checkpoint_chunk_hash_mismatch_count": len(mismatches),
        "checkpoint_chunk_hashes_valid": not mismatches,
        "mismatches": mismatches,
    }


def _core_solver_audit(repo_root: Path, commit: str) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for relative in CORE_SOLVER_FILES:
        current = (repo_root / relative).read_bytes()
        historical = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        files[relative] = {
            "run_commit_sha256": _sha256_bytes(historical),
            "audited_worktree_sha256": _sha256_bytes(current),
            "zero_difference": current == historical,
        }
    return {
        "run_commit": commit,
        "all_zero_difference": all(value["zero_difference"] for value in files.values()),
        "files": files,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def audit_results(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    expected_zip_sha256: str | None = None,
    source_zip_sha256: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    source_path = Path(source_path)
    if source_path.is_file():
        archive_sha = file_sha256(source_path)
    elif source_zip_sha256:
        archive_sha = source_zip_sha256.upper()
    else:
        raise ValueError("A copied directory requires --source-zip-sha256 provenance")
    if expected_zip_sha256 and archive_sha != expected_zip_sha256.upper():
        raise ValueError(f"Source ZIP SHA256 mismatch: {archive_sha}")

    source = ResultsSource(source_path)
    try:
        manifest_name = source.find_one("scalability_development_manifest.json")
        config_name = source.find_one("resolved_config.yaml")
        manifest = source.read_json(manifest_name)
        config_bytes = source.read_bytes(config_name)
        config = yaml.safe_load(config_bytes)
        run_names = [name for name in source.names if name.endswith("/run.json")]
        records = [source.read_json(name) for name in run_names]
        specs = list(manifest.get("run_specs") or [])
        spec_keys = {str(spec["run_key"]) for spec in specs}
        record_keys = {str(record["run_key"]) for record in records}
        rows, summary = aggregate_records(
            records, specs, source_archive_sha256=archive_sha,
            time_limit=float(manifest.get("fairness_time_limit", 1800.0)),
        )
        projection_checks = _projection_checks(records)
        checkpoint = _checkpoint_audit(source)
        post_names = [name for name in source.names if name.endswith("post_evaluation/post_evaluation.json")]
        post_values = [source.read_json(name).get("evaluation", {}) for name in post_names]
        canonical_sha = _sha256_bytes(canonical_config_bytes(config))
        file_sha = _sha256_bytes(config_bytes)
        old_sha = manifest.get("resolved_config_sha256")

        status_counts = {
            status: sum(record.get("scientific_status") == status for record in records)
            for status in SCIENTIFIC_STATUSES
        }
        algorithm_status_counts: dict[str, int] = {}
        for record in records:
            status = str(record.get("algorithm_status"))
            algorithm_status_counts[status] = algorithm_status_counts.get(status, 0) + 1
        candidate_statistics: dict[str, Any] = {}
        frontier = [row for row in rows if row["task_type"] == "frontier"]
        for candidate in sorted({str(row["candidate"]) for row in frontier}):
            group = [row for row in frontier if row["candidate"] == candidate]
            candidate_statistics[candidate] = {
                "run_count": len(group),
                "certified_solved_count": sum(row["scientific_status"] == CERTIFIED_STATUS for row in group),
                "solved_rate": sum(row["scientific_status"] == CERTIFIED_STATUS for row in group) / len(group),
                "algorithm_status_counts": {
                    status: sum(row["algorithm_status"] == status for row in group)
                    for status in sorted({str(row["algorithm_status"]) for row in group})
                },
            }
        objective_spreads = []
        for seed, rho in sorted({(int(row["seed"]), float(row["rho"])) for row in frontier}):
            values = [
                _number(row["objective_t"]) for row in frontier
                if int(row["seed"]) == seed and float(row["rho"]) == rho
                and row["scientific_status"] == CERTIFIED_STATUS
            ]
            if len(values) >= 2:
                objective_spreads.append({"seed": seed, "rho": rho, "spread": max(values) - min(values)})

        performance: dict[str, Any] = {}
        single = [row for row in frontier if row["candidate"] == "single_cut" and row["scientific_status"] == CERTIFIED_STATUS]
        batch = [row for row in frontier if row["candidate"] == "persistent_certified_cache_batch5" and row["scientific_status"] == CERTIFIED_STATUS]
        for field in ("algorithm_runtime", "total_wall_runtime", "separation_runtime"):
            single_mean = statistics.fmean(_number(row[field]) for row in single)
            batch_mean = statistics.fmean(_number(row[field]) for row in batch)
            performance[field] = {
                "single_cut_mean": single_mean,
                "batch5_mean": batch_mean,
                "batch5_reduction_percent": 100.0 * (1.0 - batch_mean / single_mean),
            }

        core = _core_solver_audit(Path(repo_root), str(manifest["git_commit"])) if repo_root else None
        required_nonempty = [
            "separation_runtime", *FIELD_PROJECTIONS, "master_runtime", "total_iterations",
            "cuts", "algorithm_runtime", "penalized_runtime_par2", "total_wall_runtime",
        ]
        nonempty = all(row[field] not in (None, "") for row in rows for field in required_nonempty)
        report = {
            "audit_status": "passed",
            "source_archive": source_path.name if source_path.is_file() else "copied_directory",
            "source_archive_sha256": archive_sha,
            "source_archive_sha256_matches_expected": expected_zip_sha256 is None or archive_sha == expected_zip_sha256.upper(),
            "zip_crc_valid": source.crc_ok(),
            "record_count": len(records),
            "unique_record_count": len(record_keys),
            "manifest_expected_run_count": manifest.get("expected_run_count"),
            "complete_record_count": sum(record.get("state") == "complete" for record in records),
            "run_specs_match_records": spec_keys == record_keys,
            "baseline_certified_count": sum(record.get("task_type") == "baseline" and record.get("scientific_status") == CERTIFIED_STATUS for record in records),
            "frontier_certified_count": sum(record.get("task_type") == "frontier" and record.get("scientific_status") == CERTIFIED_STATUS for record in records),
            "scientific_status_counts": status_counts,
            "algorithm_status_counts": algorithm_status_counts,
            "required_results_fields_nonempty_27_of_27": nonempty and len(rows) == 27,
            "projection_check_count": len(projection_checks),
            "projection_checks_passed": all(check["passed"] for check in projection_checks),
            "field_projection_semantics": FIELD_PROJECTIONS,
            "config_hashes": {
                "resolved_config_file_sha256": file_sha,
                "resolved_config_canonical_sha256": canonical_sha,
                "resolved_config_canonicalization": CANONICALIZATION,
                "legacy_manifest_resolved_config_sha256": old_sha,
                "legacy_value_equals_canonical": str(old_sha).upper() == canonical_sha,
                "file_and_canonical_distinct": file_sha != canonical_sha,
            },
            "post_evaluation": {
                "record_count": len(post_values),
                "all_valid": all(value.get("valid") is True and value.get("objective_t_consistent") is not False for value in post_values),
                "scenario_counts": sorted({value.get("scenario_count") for value in post_values}),
            },
            "checkpoint_audit": checkpoint,
            "core_solver_files": core,
        }
        gates = [
            report["source_archive_sha256_matches_expected"], report["zip_crc_valid"],
            len(records) == 27, len(record_keys) == 27, report["complete_record_count"] == 27,
            report["run_specs_match_records"], report["baseline_certified_count"] == 3,
            report["frontier_certified_count"] == 20, report["projection_checks_passed"],
            report["required_results_fields_nonempty_27_of_27"],
            report["config_hashes"]["legacy_value_equals_canonical"],
            report["config_hashes"]["file_and_canonical_distinct"],
            report["post_evaluation"]["record_count"] == 20,
            report["post_evaluation"]["all_valid"],
            report["post_evaluation"]["scenario_counts"] == [1831],
            checkpoint["checkpoint_chunk_hash_count"] == 1480,
            checkpoint["checkpoint_chunk_hashes_valid"],
            core is None or core["all_zero_difference"],
        ]
        if not all(gates):
            report["audit_status"] = "failed"

        evidence = {
            "decision": "large_s1_not_authorized",
            "large_s1_authorized": False,
            "next_authorized_stage_at_most": "fairness_scalability_s1_large_pre_run_audit_only",
            "reason": "This reporting-only hotfix validates Medium-large S1; it does not authorize a Large S1 solver run.",
            "source_archive": source_path.name,
            "source_archive_sha256": archive_sha,
            "audit_status": report["audit_status"],
            "candidate_statistics": candidate_statistics,
            "scientific_status_counts": status_counts,
            "algorithm_status_counts": algorithm_status_counts,
            "maximum_successful_candidate_objective_t_spread": max((item["spread"] for item in objective_spreads), default=0.0),
            "successful_candidate_objective_t_spreads": objective_spreads,
            "batch5_vs_single_cut": performance,
            "config_hashes": report["config_hashes"],
            "post_evaluation": report["post_evaluation"],
            "checkpoint_audit": checkpoint,
            "core_solver_files": core,
            "uncertified_runs_are_not_counted_as_success": True,
            "separation_stalled_duplicate_preserved_in_algorithm_status": algorithm_status_counts.get("separation_stalled_duplicate", 0),
        }

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        _write_json(
            output / "source_archive_provenance.json",
            {
                "source_archive": source_path.name if source_path.is_file() else "copied_directory",
                "source_archive_sha256": archive_sha,
                "source_archive_size_bytes": source_path.stat().st_size if source_path.is_file() else None,
                "audit_access_mode": "read_only",
            },
        )
        _write_csv(output / "results.corrected.csv", rows, RESULT_FIELDS)
        summary_fields = list(summary[0]) if summary else []
        _write_csv(output / "summary.corrected.csv", summary, summary_fields)
        _write_json(output / "results_audit.json", report)
        _write_json(
            output / "field_projection_semantics.json",
            {
                "source_archive_sha256": archive_sha,
                "field_projection_semantics": FIELD_PROJECTIONS,
            },
        )
        _write_json(
            output / "field_reconciliation.json",
            {
                "source_archive_sha256": archive_sha,
                "check_count": len(projection_checks),
                "all_passed": all(check["passed"] for check in projection_checks),
                "checks": projection_checks,
            },
        )
        _write_json(output / "decision_evidence.json", evidence)
        markdown = (
            "# Fairness scalability S1 reporting decision evidence\n\n"
            f"- Source ZIP SHA256: `{archive_sha}`\n"
            f"- Audit status: `{report['audit_status']}`\n"
            f"- Complete records: {report['complete_record_count']}/27\n"
            f"- Certified baselines: {report['baseline_certified_count']}/3\n"
            f"- Certified frontier runs: {report['frontier_certified_count']}/24\n"
            f"- Maximum successful-candidate objective T spread: {evidence['maximum_successful_candidate_objective_t_spread']:.17g}\n"
            f"- Checkpoint chunks verified: {checkpoint['checkpoint_chunk_hash_count']}\n"
            f"- Large S1 authorized: no\n"
            f"- Next stage at most: `{evidence['next_authorized_stage_at_most']}`\n"
        )
        (output / "decision_evidence.md").write_text(markdown, encoding="utf-8")
        return report
    finally:
        source.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Fairness Scalability results audit")
    parser.add_argument("--source", required=True, help="Original ZIP or copied result directory")
    parser.add_argument("--output-dir", required=True, help="Separate directory for derived files")
    parser.add_argument("--expected-zip-sha256")
    parser.add_argument("--source-zip-sha256")
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    report = audit_results(
        args.source, args.output_dir,
        expected_zip_sha256=args.expected_zip_sha256,
        source_zip_sha256=args.source_zip_sha256,
        repo_root=args.repo_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["audit_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
