from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import subprocess
from typing import Any, Iterable, Mapping
import zipfile

import yaml

from .fairness_scalability_results_audit import CANONICALIZATION, FIELD_PROJECTIONS


MEDIUM_LARGE_ARCHIVE_SHA256 = (
    "3919E271C40BDC86F5EE7FFA582A06DDD325D5BB4EAD453550ED17A6C62DB751"
)
LARGE_ARCHIVE_SHA256 = (
    "7A3C7BF75B0D6D3228B6AA9AC0E8B8E6799A4B4925949C89DBBCBF311C2D2376"
)
MEDIUM_LARGE_RUN_COMMIT = "29ae09e968a206b1987714317ff7528165372a46"
LARGE_RUN_COMMIT = "ec33a047ecd60f4cb473260f1b3c4078726db776"
V3_CANDIDATE_SHA256 = (
    "7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6"
)
CERTIFIED_STATUS = "certified_robust_optimal"
EXPECTED_SEEDS = (160, 161, 162)
EXPECTED_RHOS = (0.0, 0.01)
EXPECTED_CANDIDATES = (
    "single_cut",
    "persistent_separation",
    "persistent_certified_cache",
    "persistent_certified_cache_batch5",
)
CORE_FILES = (
    "src/fairness_scalability.py",
    "src/fairness_benders.py",
    "src/robust_regional_fairness.py",
    "src/benders.py",
    "src/scenarios.py",
    "experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml",
)
SCALE_SPECS: dict[str, dict[str, Any]] = {
    "medium_large": {
        "archive_name": "fairness_scalability_s1_attempt2_medium_large_results.zip",
        "archive_sha256": MEDIUM_LARGE_ARCHIVE_SHA256,
        "run_commit": MEDIUM_LARGE_RUN_COMMIT,
        "scenario_count": 1831,
        "frontier_certified_count": 20,
        "candidate_certified_counts": {
            "single_cut": 6,
            "persistent_separation": 6,
            "persistent_certified_cache": 2,
            "persistent_certified_cache_batch5": 6,
        },
    },
    "large": {
        "archive_name": "fairness_scalability_s1_attempt2_large_results.zip",
        "archive_sha256": LARGE_ARCHIVE_SHA256,
        "run_commit": LARGE_RUN_COMMIT,
        "scenario_count": 4657,
        "frontier_certified_count": 0,
        "candidate_certified_counts": {candidate: 0 for candidate in EXPECTED_CANDIDATES},
    },
}
SUMMARY_FIELDS = (
    "medium_large_source_archive_sha256",
    "large_source_archive_sha256",
    "scale",
    "candidate",
    "run_count",
    "certified_solved_count",
    "solved_rate",
    "time_limit_uncertified_count",
    "other_uncertified_count",
    "mean_algorithm_runtime",
    "mean_penalized_runtime_par2",
    "mean_separation_runtime",
    "mean_master_runtime",
    "mean_post_evaluation_solver_runtime",
    "mean_post_evaluation_wall_runtime",
    "mean_total_wall_runtime",
    "mean_iterations",
    "mean_cuts",
    "mean_cache_candidate_count",
    "mean_cache_hit_count",
    "mean_certified_cached_cut_count",
    "mean_pool_candidate_count",
    "mean_certified_batch_cut_count",
    "mean_duplicate_pattern_count",
    "mean_cuts_per_iteration",
)
NOT_APPLICABLE = "NOT_APPLICABLE"
RUN_MATRIX_FIELDS = (
    "medium_large_source_archive_sha256",
    "large_source_archive_sha256",
    "scale",
    "execution_attempt",
    "introduced_stage",
    "task_type",
    "seed",
    "rho",
    "candidate",
    "run_key",
    "run_directory_id",
    "git_commit",
    "config_file_sha256",
    "resolved_config_file_sha256",
    "resolved_config_canonical_sha256",
    "protocol_sha256",
    "instance_sha256",
    "baseline_run_key",
    "anchor_value_hex",
    "anchor_sha256",
    "state",
    "scientific_status",
    "algorithm_status",
    "certified_solved",
    "algorithm_runtime",
    "penalized_runtime_par2",
    "separation_runtime",
    "master_runtime",
    "post_evaluation_wall_runtime",
    "total_wall_runtime",
    "iterations",
    "cuts",
    "source_archive_sha256",
)
EXPERIMENT_MATRIX_SUMMARY_FIELDS = (
    "medium_large_source_archive_sha256",
    "large_source_archive_sha256",
    "scale",
    "task_type",
    "candidate",
    "rho",
    "planned_count",
    "completed_count",
    "certified_count",
    "time_limit_uncertified_count",
    "other_uncertified_count",
    "implementation_error_count",
    "invalid_post_evaluation_count",
    "mean_algorithm_runtime",
    "mean_penalized_runtime_par2",
    "mean_separation_runtime",
    "mean_master_runtime",
    "mean_total_wall_runtime",
    "mean_iterations",
    "mean_cuts",
)


class CrossScaleAuditError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _canonical_yaml_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        yaml.safe_dump(dict(value), sort_keys=True, allow_unicode=True).encode("utf-8")
    )


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _close(left: Any, right: Any) -> bool:
    return math.isclose(_number(left), _number(right), rel_tol=1.0e-12, abs_tol=1.0e-9)


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def run_directory_id(run_key: str) -> str:
    return "r_" + hashlib.sha256(str(run_key).encode("utf-8")).hexdigest()[:24]


class ZipResults:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.archive = zipfile.ZipFile(self.path)
        self.entries = list(self.archive.infolist())
        self.files = [item.filename for item in self.entries if not item.is_dir()]
        self.directories = [item.filename for item in self.entries if item.is_dir()]
        self._actual_by_normalized = {
            name.removeprefix("./"): name for name in self.files
        }
        if len(self._actual_by_normalized) != len(self.files):
            raise CrossScaleAuditError(f"Duplicate normalized ZIP member in {self.path.name}")

    @property
    def names(self) -> list[str]:
        return sorted(self._actual_by_normalized)

    def close(self) -> None:
        self.archive.close()

    def read_bytes(self, normalized_name: str) -> bytes:
        return self.archive.read(self._actual_by_normalized[normalized_name])

    def read_json(self, normalized_name: str) -> dict[str, Any]:
        value = json.loads(self.read_bytes(normalized_name))
        if not isinstance(value, dict):
            raise CrossScaleAuditError(f"{normalized_name} is not a JSON object")
        return value

    def find(self, suffix: str) -> list[str]:
        return [name for name in self.names if name.endswith(suffix)]

    def find_one(self, suffix: str) -> str:
        matches = self.find(suffix)
        if len(matches) != 1:
            raise CrossScaleAuditError(
                f"Expected one *{suffix} in {self.path.name}; found {len(matches)}"
            )
        return matches[0]

    def inferred_directories(self) -> set[str]:
        result: set[str] = set()
        for name in self.names:
            parent = PurePosixPath(name).parent
            while str(parent) not in ("", "."):
                result.add(parent.as_posix())
                parent = parent.parent
        return result


def _git_bytes(repo_root: Path, commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout


def _git_blob(repo_root: Path, commit: str, relative: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", f"{commit}:{relative}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def audit_core_equivalence(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    files: dict[str, Any] = {}
    for relative in CORE_FILES:
        medium_bytes = _git_bytes(root, MEDIUM_LARGE_RUN_COMMIT, relative)
        large_bytes = _git_bytes(root, LARGE_RUN_COMMIT, relative)
        files[relative] = {
            "medium_large_blob": _git_blob(root, MEDIUM_LARGE_RUN_COMMIT, relative),
            "large_blob": _git_blob(root, LARGE_RUN_COMMIT, relative),
            "medium_large_sha256": _sha256_bytes(medium_bytes),
            "large_sha256": _sha256_bytes(large_bytes),
            "zero_difference": medium_bytes == large_bytes,
        }
    report = {
        "medium_large_commit": MEDIUM_LARGE_RUN_COMMIT,
        "large_commit": LARGE_RUN_COMMIT,
        "files": files,
        "all_zero_difference": all(item["zero_difference"] for item in files.values()),
    }
    if not report["all_zero_difference"]:
        raise CrossScaleAuditError(
            "Solver-core drift detected; cross-scale candidate comparison is prohibited."
        )
    return report


def _project_record(record: Mapping[str, Any], time_limit: float) -> dict[str, Any]:
    result = dict(record.get("result") or {})
    metadata = dict(result.get("metadata") or {})
    scientific_status = str(record.get("scientific_status", ""))
    algorithm_runtime = _number(result.get("algorithm_runtime", result.get("runtime")))
    expected_par2 = (
        algorithm_runtime if scientific_status == CERTIFIED_STATUS else 2.0 * time_limit
    )
    total_iterations = metadata.get(
        "total_iterations", result.get("total_iterations", result.get("iterations", 0))
    )
    row: dict[str, Any] = {
        "run_key": str(record.get("run_key")),
        "task_type": str(record.get("task_type")),
        "seed": int(record.get("seed")),
        "rho": record.get("rho"),
        "candidate": str(record.get("candidate")),
        "scientific_status": scientific_status,
        "algorithm_status": str(record.get("algorithm_status")),
        "solved_to_tolerance": scientific_status == CERTIFIED_STATUS,
        "separation_runtime": _number(result.get("separation_runtime")),
        "master_runtime": _number(result.get("master_runtime")),
        "iterations": int(_number(result.get("iterations", total_iterations))),
        "total_iterations": int(_number(total_iterations)),
        "cuts": int(_number(result.get("cuts"))),
        "algorithm_runtime": algorithm_runtime,
        "penalized_runtime_par2": expected_par2,
        "post_evaluation_solver_runtime": _number(
            result.get("post_evaluation_solver_runtime")
        ),
        "post_evaluation_wall_runtime": _number(
            result.get("post_evaluation_wall_runtime")
        ),
        "total_wall_runtime": _number(
            result.get("total_wall_runtime", result.get("runtime"))
        ),
    }
    for field in FIELD_PROJECTIONS:
        row[field] = 0.0 if record.get("task_type") != "frontier" else _number(
            metadata.get(field)
        )
    return row


def _projection_reconciliation(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for record in records:
        if record.get("task_type") != "frontier":
            continue
        result = dict(record.get("result") or {})
        metadata = dict(result.get("metadata") or {})
        iteration_log = list(result.get("iteration_log") or [])
        for field, definition in FIELD_PROJECTIONS.items():
            values = [_number(item.get(field)) for item in iteration_log]
            recomputed = (
                math.fsum(values) / len(values)
                if definition["aggregation"].endswith("iteration_log_mean") and values
                else math.fsum(values)
            )
            actual = _number(metadata.get(field))
            checks.append(
                {
                    "run_key": record.get("run_key"),
                    "field": field,
                    "metadata_value": actual,
                    "iteration_log_recomputed_value": recomputed,
                    "passed": _close(actual, recomputed),
                }
            )
    return {
        "check_count": len(checks),
        "passed_count": sum(item["passed"] for item in checks),
        "all_passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


def candidate_summary(
    scale: str, rows: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    frontier = [dict(row) for row in rows if row["task_type"] == "frontier"]
    result: list[dict[str, Any]] = []
    metric_fields = (
        "algorithm_runtime",
        "penalized_runtime_par2",
        "separation_runtime",
        "master_runtime",
        "post_evaluation_solver_runtime",
        "post_evaluation_wall_runtime",
        "total_wall_runtime",
        "iterations",
        "cuts",
        "cache_candidate_count",
        "cache_hit_count",
        "certified_cached_cut_count",
        "pool_candidate_count",
        "certified_batch_cut_count",
        "duplicate_pattern_count",
        "cuts_per_iteration",
    )
    for candidate in EXPECTED_CANDIDATES:
        group = [row for row in frontier if row["candidate"] == candidate]
        certified = sum(row["scientific_status"] == CERTIFIED_STATUS for row in group)
        item: dict[str, Any] = {
            "medium_large_source_archive_sha256": MEDIUM_LARGE_ARCHIVE_SHA256,
            "large_source_archive_sha256": LARGE_ARCHIVE_SHA256,
            "scale": scale,
            "candidate": candidate,
            "run_count": len(group),
            "certified_solved_count": certified,
            "solved_rate": certified / len(group) if group else 0.0,
            "time_limit_uncertified_count": sum(
                row["scientific_status"] == "time_limit_uncertified" for row in group
            ),
            "other_uncertified_count": sum(
                row["scientific_status"]
                not in (CERTIFIED_STATUS, "time_limit_uncertified")
                for row in group
            ),
        }
        for field in metric_fields:
            item[f"mean_{field}"] = statistics.fmean(
                _number(row[field]) for row in group
            )
        result.append(item)
    return result


def build_run_matrix(
    scale: str,
    records: Iterable[Mapping[str, Any]],
    projected_rows: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    resolved_config_file_sha256: str,
    resolved_config_canonical_sha256: str,
) -> list[dict[str, Any]]:
    projected_by_key = {str(row["run_key"]): dict(row) for row in projected_rows}
    specs_by_key = {
        str(item["run_key"]): dict(item) for item in manifest.get("run_specs") or []
    }
    rows: list[dict[str, Any]] = []
    for raw_record in records:
        record = dict(raw_record)
        run_key = str(record["run_key"])
        projected = projected_by_key[run_key]
        spec = specs_by_key[run_key]
        is_baseline = record.get("task_type") == "baseline"
        rows.append(
            {
                "medium_large_source_archive_sha256": MEDIUM_LARGE_ARCHIVE_SHA256,
                "large_source_archive_sha256": LARGE_ARCHIVE_SHA256,
                "scale": scale,
                "execution_attempt": manifest.get("execution_attempt"),
                "introduced_stage": spec.get("introduced_stage"),
                "task_type": record.get("task_type"),
                "seed": record.get("seed"),
                "rho": NOT_APPLICABLE if is_baseline else record.get("rho"),
                "candidate": record.get("candidate"),
                "run_key": run_key,
                "run_directory_id": record.get("run_directory_id"),
                "git_commit": record.get("git_commit"),
                "config_file_sha256": manifest.get("config_file_sha256"),
                "resolved_config_file_sha256": resolved_config_file_sha256,
                "resolved_config_canonical_sha256": resolved_config_canonical_sha256,
                "protocol_sha256": manifest.get("protocol_sha256"),
                "instance_sha256": record.get("instance_sha256"),
                "baseline_run_key": (
                    NOT_APPLICABLE if is_baseline else record.get("baseline_run_key")
                ),
                "anchor_value_hex": (
                    NOT_APPLICABLE if is_baseline else record.get("anchor_value_hex")
                ),
                "anchor_sha256": (
                    NOT_APPLICABLE if is_baseline else record.get("anchor_sha256")
                ),
                "state": record.get("state"),
                "scientific_status": record.get("scientific_status"),
                "algorithm_status": record.get("algorithm_status"),
                "certified_solved": (
                    "true"
                    if record.get("scientific_status") == CERTIFIED_STATUS
                    else "false"
                ),
                "algorithm_runtime": projected["algorithm_runtime"],
                "penalized_runtime_par2": projected["penalized_runtime_par2"],
                "separation_runtime": projected["separation_runtime"],
                "master_runtime": projected["master_runtime"],
                "post_evaluation_wall_runtime": projected[
                    "post_evaluation_wall_runtime"
                ],
                "total_wall_runtime": projected["total_wall_runtime"],
                "iterations": projected["iterations"],
                "cuts": projected["cuts"],
                "source_archive_sha256": SCALE_SPECS[scale]["archive_sha256"],
            }
        )
    return rows


def _run_matrix_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    task_rank = 0 if row.get("task_type") == "baseline" else 1
    rho = -1.0 if row.get("rho") == NOT_APPLICABLE else _number(row.get("rho"))
    return (
        str(row.get("scale")),
        int(row.get("seed", -1)),
        task_rank,
        rho,
        str(row.get("candidate")),
    )


def validate_run_matrix(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = [dict(row) for row in rows]
    failures: list[str] = []
    run_keys = [str(row["run_key"]) for row in matrix]
    identities = [
        (
            row["scale"],
            row["task_type"],
            int(row["seed"]),
            row["rho"],
            row["candidate"],
        )
        for row in matrix
    ]
    _require(len(matrix) == 54, "frozen matrix row count != 54", failures)
    _require(len(set(run_keys)) == 54, "frozen matrix run keys are not unique", failures)
    _require(
        len(set(identities)) == 54,
        "frozen matrix scientific identities are not unique",
        failures,
    )
    _require(
        all(
            row.get("medium_large_source_archive_sha256")
            == MEDIUM_LARGE_ARCHIVE_SHA256
            and row.get("large_source_archive_sha256") == LARGE_ARCHIVE_SHA256
            and row.get("source_archive_sha256")
            == SCALE_SPECS[str(row["scale"])]["archive_sha256"]
            for row in matrix
        ),
        "frozen matrix source archive provenance mismatch",
        failures,
    )
    expected_certified = {
        "medium_large": SCALE_SPECS["medium_large"]["candidate_certified_counts"],
        "large": SCALE_SPECS["large"]["candidate_certified_counts"],
    }
    for scale in ("medium_large", "large"):
        group = [row for row in matrix if row["scale"] == scale]
        baseline = [row for row in group if row["task_type"] == "baseline"]
        frontier = [row for row in group if row["task_type"] == "frontier"]
        _require(len(group) == 27, f"{scale} matrix total != 27", failures)
        _require(len(baseline) == 3, f"{scale} matrix baseline != 3", failures)
        _require(len(frontier) == 24, f"{scale} matrix frontier != 24", failures)
        _require(
            all(
                row["rho"] == NOT_APPLICABLE
                and row["baseline_run_key"] == NOT_APPLICABLE
                and row["anchor_value_hex"] == NOT_APPLICABLE
                and row["anchor_sha256"] == NOT_APPLICABLE
                for row in baseline
            ),
            f"{scale} baseline NOT_APPLICABLE fields are ambiguous",
            failures,
        )
        for candidate in EXPECTED_CANDIDATES:
            candidate_rows = [row for row in frontier if row["candidate"] == candidate]
            _require(
                len(candidate_rows) == 6,
                f"{scale}/{candidate} coverage != 6",
                failures,
            )
            _require(
                sum(row["certified_solved"] == "true" for row in candidate_rows)
                == expected_certified[scale][candidate],
                f"{scale}/{candidate} certified count mismatch",
                failures,
            )
    for candidate in EXPECTED_CANDIDATES:
        cross = [
            row
            for row in matrix
            if row["task_type"] == "frontier" and row["candidate"] == candidate
        ]
        _require(len(cross) == 12, f"cross-scale/{candidate} coverage != 12", failures)
    _require(
        sum(row["task_type"] == "baseline" for row in matrix) == 6,
        "cross-scale baseline count != 6",
        failures,
    )
    _require(
        sum(row["task_type"] == "frontier" for row in matrix) == 48,
        "cross-scale frontier count != 48",
        failures,
    )
    if failures:
        raise CrossScaleAuditError("Frozen run matrix failed: " + "; ".join(failures))
    return {
        "row_count": len(matrix),
        "unique_run_key_count": len(set(run_keys)),
        "unique_scientific_identity_count": len(set(identities)),
        "medium_large": {"baseline": 3, "frontier": 24, "total": 27},
        "large": {"baseline": 3, "frontier": 24, "total": 27},
        "cross_scale": {"baseline": 6, "frontier": 48, "total": 54},
        "candidate_cross_scale_run_count": {
            candidate: 12 for candidate in EXPECTED_CANDIDATES
        },
        "all_passed": True,
    }


def experiment_matrix_summary(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    matrix = [dict(row) for row in rows]
    metric_fields = (
        "algorithm_runtime",
        "penalized_runtime_par2",
        "separation_runtime",
        "master_runtime",
        "total_wall_runtime",
        "iterations",
        "cuts",
    )

    def summarize(
        scale: str,
        task_type: str,
        candidate: str,
        rho: str | float,
        group: list[dict[str, Any]],
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "medium_large_source_archive_sha256": MEDIUM_LARGE_ARCHIVE_SHA256,
            "large_source_archive_sha256": LARGE_ARCHIVE_SHA256,
            "scale": scale,
            "task_type": task_type,
            "candidate": candidate,
            "rho": rho,
            "planned_count": len(group),
            "completed_count": sum(row["state"] == "complete" for row in group),
            "certified_count": sum(row["certified_solved"] == "true" for row in group),
            "time_limit_uncertified_count": sum(
                row["scientific_status"] == "time_limit_uncertified" for row in group
            ),
            "other_uncertified_count": sum(
                row["scientific_status"]
                not in (
                    CERTIFIED_STATUS,
                    "time_limit_uncertified",
                    "implementation_error",
                    "invalid_post_evaluation",
                )
                for row in group
            ),
            "implementation_error_count": sum(
                row["scientific_status"] == "implementation_error" for row in group
            ),
            "invalid_post_evaluation_count": sum(
                row["scientific_status"] == "invalid_post_evaluation" for row in group
            ),
        }
        for field in metric_fields:
            item[f"mean_{field}"] = (
                statistics.fmean(_number(row[field]) for row in group)
                if group
                else 0.0
            )
        return item

    result: list[dict[str, Any]] = []
    for scale in sorted({str(row["scale"]) for row in matrix}):
        scale_rows = [row for row in matrix if row["scale"] == scale]
        baseline = [row for row in scale_rows if row["task_type"] == "baseline"]
        frontier = [row for row in scale_rows if row["task_type"] == "frontier"]
        baseline_candidate = str(baseline[0]["candidate"])
        result.append(
            summarize(scale, "baseline", baseline_candidate, NOT_APPLICABLE, baseline)
        )
        for candidate in EXPECTED_CANDIDATES:
            candidate_rows = [row for row in frontier if row["candidate"] == candidate]
            for rho in EXPECTED_RHOS:
                group = [row for row in candidate_rows if _close(row["rho"], rho)]
                result.append(summarize(scale, "frontier", candidate, rho, group))
            result.append(
                summarize(scale, "frontier", candidate, "ALL_RHOS", candidate_rows)
            )
        result.extend(
            (
                summarize(scale, "baseline", "ALL_CANDIDATES", "ALL_RHOS", baseline),
                summarize(scale, "frontier", "ALL_CANDIDATES", "ALL_RHOS", frontier),
                summarize(scale, "ALL_TASK_TYPES", "ALL_CANDIDATES", "ALL_RHOS", scale_rows),
            )
        )
    baseline_all = [row for row in matrix if row["task_type"] == "baseline"]
    frontier_all = [row for row in matrix if row["task_type"] == "frontier"]
    for candidate in EXPECTED_CANDIDATES:
        group = [row for row in frontier_all if row["candidate"] == candidate]
        result.append(
            summarize("ALL_SCALES", "frontier", candidate, "ALL_RHOS", group)
        )
    result.extend(
        (
            summarize("ALL_SCALES", "baseline", "ALL_CANDIDATES", "ALL_RHOS", baseline_all),
            summarize("ALL_SCALES", "frontier", "ALL_CANDIDATES", "ALL_RHOS", frontier_all),
            summarize("ALL_SCALES", "ALL_TASK_TYPES", "ALL_CANDIDATES", "ALL_RHOS", matrix),
        )
    )
    result.sort(
        key=lambda row: (
            str(row["scale"]),
            str(row["task_type"]),
            str(row["candidate"]),
            str(row["rho"]),
        )
    )
    return result


def _member_directory_id(member_name: str) -> str:
    parts = PurePosixPath(member_name).parts
    try:
        index = parts.index("runs")
    except ValueError as exc:
        raise CrossScaleAuditError(f"Run artifact is outside runs/: {member_name}") from exc
    if index + 1 >= len(parts):
        raise CrossScaleAuditError(f"Run artifact lacks directory id: {member_name}")
    return parts[index + 1]


def _parseability_audit(source: ZipResults) -> dict[str, Any]:
    json_names = [name for name in source.names if name.endswith(".json")]
    csv_names = [name for name in source.names if name.endswith(".csv")]
    failures: list[str] = []
    for name in json_names:
        try:
            json.loads(source.read_bytes(name))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append(f"{name}: {exc}")
    csv_row_counts: dict[str, int] = {}
    for name in csv_names:
        try:
            rows = list(
                csv.reader(io.StringIO(source.read_bytes(name).decode("utf-8")))
            )
            csv_row_counts[name] = max(0, len(rows) - 1)
        except (UnicodeDecodeError, csv.Error) as exc:
            failures.append(f"{name}: {exc}")
    return {
        "json_file_count": len(json_names),
        "csv_file_count": len(csv_names),
        "csv_data_row_counts": csv_row_counts,
        "parse_failure_count": len(failures),
        "failures": failures,
        "all_parseable": not failures,
    }


def _source_csv_audit(
    source: ZipResults, scale: str, projected_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    results_name = source.find_one("results.csv")
    summary_name = source.find_one("summary.csv")
    results = list(
        csv.DictReader(io.StringIO(source.read_bytes(results_name).decode("utf-8")))
    )
    summary = list(
        csv.DictReader(io.StringIO(source.read_bytes(summary_name).decode("utf-8")))
    )
    projected_by_key = {row["run_key"]: row for row in projected_rows}
    identity_failures: list[str] = []
    for row in results:
        expected = projected_by_key.get(str(row.get("run_key")))
        if expected is None:
            identity_failures.append(f"Unknown source results run key: {row.get('run_key')}")
            continue
        for field in ("scientific_status", "algorithm_status"):
            if str(row.get(field)) != str(expected[field]):
                identity_failures.append(f"{row.get('run_key')}:{field}")
        solved = str(row.get("solved_to_tolerance", "")).lower() == "true"
        if solved != bool(expected["solved_to_tolerance"]):
            identity_failures.append(f"{row.get('run_key')}:solved_to_tolerance")
    return {
        "results_row_count": len(results),
        "summary_row_count": len(summary),
        "status_identity_matches_rebuild": not identity_failures,
        "status_identity_failures": identity_failures,
        "reporting_schema": (
            "legacy_pre_hotfix_incomplete_not_used_for_freeze"
            if scale == "medium_large"
            else "reporting_hotfix_schema"
        ),
        "freeze_metrics_rebuilt_from_run_json": True,
    }


def _post_evaluation_audit(
    source: ZipResults,
    records_by_key: Mapping[str, Mapping[str, Any]],
    directory_to_key: Mapping[str, str],
    expected_scenario_count: int,
) -> dict[str, Any]:
    post_names = source.find("post_evaluation/post_evaluation.json")
    index_names = source.find("post_evaluation/checkpoint/index.json")
    failures: list[str] = []
    chunk_hash_count = 0
    acceptance_evidence_count = 0
    audited_run_keys: list[str] = []
    all_scenario_keys: dict[str, int] = {}
    for post_name in post_names:
        directory_id = _member_directory_id(post_name)
        run_key = directory_to_key.get(directory_id)
        if run_key is None or run_key not in records_by_key:
            failures.append(f"Unmapped post-evaluation directory: {directory_id}")
            continue
        record = records_by_key[run_key]
        audited_run_keys.append(run_key)
        payload = source.read_json(post_name)
        evaluation = dict(payload.get("evaluation") or {})
        _require(record.get("scientific_status") == CERTIFIED_STATUS, f"Post-evaluation attached to uncertified run: {run_key}", failures)
        _require(evaluation.get("valid") is True, f"Invalid post-evaluation: {run_key}", failures)
        _require(evaluation.get("scenario_count") == expected_scenario_count, f"Wrong post scenario count: {run_key}", failures)
        _require(not evaluation.get("errors"), f"Post-evaluation errors: {run_key}", failures)
        _require(evaluation.get("objective_t_consistent") is True, f"Objective T mismatch: {run_key}", failures)
        inline = (record.get("result") or {}).get("post_evaluation")
        _require(inline == evaluation, f"Inline/final post-evaluation mismatch: {run_key}", failures)
        evidence = list(evaluation.get("acceptance_evidence") or [])
        acceptance_evidence_count += len(evidence)
        for item in evidence:
            if item.get("accepted") is not True or _number(item.get("residual")) > _number(item.get("acceptance_threshold")):
                failures.append(f"Acceptance evidence failed: {run_key}")
                break

        base = post_name.removesuffix("post_evaluation.json")
        index_name = f"{base}checkpoint/index.json"
        if index_name not in source.names:
            failures.append(f"Missing post-evaluation index: {run_key}")
            continue
        index = source.read_json(index_name)
        chunks = list(index.get("chunks") or [])
        _require(
            len(chunks) == math.ceil(expected_scenario_count / 25),
            f"Checkpoint chunk count mismatch: {run_key}",
            failures,
        )
        scenario_indices: list[int] = []
        scenario_keys: list[str] = []
        for expected_chunk_index, chunk in enumerate(chunks):
            if int(chunk.get("chunk_index", -1)) != expected_chunk_index:
                failures.append(f"Chunk order mismatch: {run_key}")
            chunk_name = f"{base}{chunk.get('relative_path')}"
            if chunk_name not in source.names:
                failures.append(f"Missing chunk: {chunk_name}")
                continue
            chunk_bytes = source.read_bytes(chunk_name)
            chunk_hash_count += 1
            if _sha256_bytes(chunk_bytes).lower() != str(chunk.get("sha256")).lower():
                failures.append(f"Chunk SHA mismatch: {chunk_name}")
            chunk_payload = json.loads(chunk_bytes)
            if int(chunk_payload.get("chunk_index", -1)) != expected_chunk_index:
                failures.append(f"Chunk payload index mismatch: {chunk_name}")
            chunk_records = list(chunk_payload.get("records") or [])
            if len(chunk_records) != int(chunk.get("scenario_count", -1)):
                failures.append(f"Chunk scenario count mismatch: {chunk_name}")
            scenario_indices.extend(int(item.get("scenario_index", -1)) for item in chunk_records)
            scenario_keys.extend(str(item.get("scenario_key")) for item in chunk_records)
        _require(scenario_indices == list(range(expected_scenario_count)), f"Scenario order mismatch: {run_key}", failures)
        _require(len(set(scenario_keys)) == expected_scenario_count, f"Scenario-key uniqueness mismatch: {run_key}", failures)
        all_scenario_keys[run_key] = len(set(scenario_keys))
    certified_keys = {
        key for key, record in records_by_key.items()
        if record.get("task_type") == "frontier" and record.get("scientific_status") == CERTIFIED_STATUS
    }
    _require(set(audited_run_keys) == certified_keys, "Post-evaluation run set differs from certified frontier set", failures)
    _require(len(index_names) == len(post_names), "Post-evaluation index count mismatch", failures)
    return {
        "post_evaluation_count": len(post_names),
        "checkpoint_index_count": len(index_names),
        "chunk_hash_count": chunk_hash_count,
        "acceptance_evidence_count": acceptance_evidence_count,
        "scenario_key_counts": all_scenario_keys,
        "all_valid": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _algorithm_checkpoint_audit(
    source: ZipResults,
    records_by_key: Mapping[str, Mapping[str, Any]],
    directory_to_key: Mapping[str, str],
) -> dict[str, Any]:
    names = source.find("algorithm_checkpoint.json")
    failures: list[str] = []
    audited: set[str] = set()
    for name in names:
        directory_id = _member_directory_id(name)
        run_key = directory_to_key.get(directory_id)
        if run_key is None:
            failures.append(f"Unmapped algorithm checkpoint: {name}")
            continue
        audited.add(run_key)
        record = records_by_key[run_key]
        checkpoint = source.read_json(name)
        identity = dict(checkpoint.get("identity") or {})
        expected_identity = {
            "run_key": run_key,
            "git_commit": record.get("git_commit"),
            "config_sha256": record.get("config_sha256"),
            "anchor_sha256": record.get("anchor_sha256"),
            "instance_sha256": record.get("instance_sha256"),
            "candidate": record.get("candidate"),
            "rho": record.get("rho"),
        }
        _require(identity == expected_identity, f"Algorithm checkpoint identity mismatch: {run_key}", failures)
        checkpoint_result = dict(checkpoint.get("result") or {})
        run_result = dict(record.get("result") or {})
        _require(
            all(key in run_result and run_result[key] == value for key, value in checkpoint_result.items()),
            f"Algorithm checkpoint result mismatch: {run_key}",
            failures,
        )
    frontier_keys = {
        key for key, record in records_by_key.items() if record.get("task_type") == "frontier"
    }
    _require(audited == frontier_keys, "Algorithm checkpoint set differs from frontier run set", failures)
    return {
        "algorithm_checkpoint_count": len(names),
        "identity_and_result_match": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def audit_archive(
    path: str | Path,
    scale: str,
    repo_root: str | Path,
    source_archives: Mapping[str, str],
    core_equivalence: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    spec = SCALE_SPECS[scale]
    source_path = Path(path)
    before_sha = file_sha256(source_path)
    if before_sha != spec["archive_sha256"]:
        raise CrossScaleAuditError(
            f"{scale} ZIP SHA256 mismatch: {before_sha}"
        )
    source = ZipResults(source_path)
    failures: list[str] = []
    try:
        crc_bad_member = source.archive.testzip()
        parseability = _parseability_audit(source)
        manifest = source.read_json(source.find_one("scalability_development_manifest.json"))
        config_name = source.find_one("resolved_config.yaml")
        config_bytes = source.read_bytes(config_name)
        config = yaml.safe_load(config_bytes)
        if not isinstance(config, dict):
            raise CrossScaleAuditError(f"Resolved config is not a mapping: {scale}")
        resolved_file_sha = _sha256_bytes(config_bytes)
        resolved_canonical_sha = _canonical_yaml_sha256(config)

        config_path = str(manifest.get("config_path"))
        protocol_path = str(manifest.get("protocol_path"))
        candidate_path = str(config.get("candidate_parameters_must_be_fixed_from"))
        commit = str(manifest.get("git_commit"))
        git_config_sha = _sha256_bytes(_git_bytes(Path(repo_root), commit, config_path))
        git_protocol_sha = _sha256_bytes(_git_bytes(Path(repo_root), commit, protocol_path))
        git_candidate_sha = _sha256_bytes(_git_bytes(Path(repo_root), commit, candidate_path))

        _require(manifest.get("schema_version") == 2, "schema_version != 2", failures)
        _require(manifest.get("execution_attempt") == 2, "execution_attempt != 2", failures)
        _require(commit == spec["run_commit"], "run commit mismatch", failures)
        _require(manifest.get("config_file_sha256") == git_config_sha, "config file SHA mismatch", failures)
        _require(manifest.get("protocol_sha256") == git_protocol_sha, "protocol SHA mismatch", failures)
        _require(manifest.get("candidate_config_sha256") == git_candidate_sha == V3_CANDIDATE_SHA256, "candidate SHA mismatch", failures)
        _require(manifest.get("scale") == scale, "scale mismatch", failures)
        _require(config.get("instance_sizes") == [scale], "resolved scale mismatch", failures)
        _require(tuple(manifest.get("seeds") or []) == EXPECTED_SEEDS, "manifest seeds mismatch", failures)
        _require(tuple(float(value) for value in manifest.get("rhos") or []) == EXPECTED_RHOS, "manifest rho mismatch", failures)
        _require(tuple(manifest.get("candidate_definitions") or []) == EXPECTED_CANDIDATES, "candidate definitions mismatch", failures)
        _require(tuple(config.get("scalability_candidates") or []) == EXPECTED_CANDIDATES, "resolved candidate definitions mismatch", failures)
        _require(manifest.get("gurobi_parameters") == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1.0e-7}, "Gurobi identity mismatch", failures)
        _require(all(float(manifest.get(key, math.nan)) == 1800.0 for key in ("baseline_time_limit", "fairness_time_limit")), "manifest time limits mismatch", failures)
        _require(all(float(config.get(key, math.nan)) == 1800.0 for key in ("baseline_time_limit", "fairness_time_limit", "time_limit")), "resolved time limits mismatch", failures)
        _require(manifest.get("post_evaluation") == {
            "enabled": True,
            "exact_scenarios": True,
            "max_scenarios": 5000,
            "time_limit_per_scenario": 30.0,
            "checkpoint_chunk_size": 25,
            "feasibility_tolerance": 1.0e-7,
        }, "post-evaluation identity mismatch", failures)
        _require(manifest.get("par2") == {"multiplier": 2, "basis": "algorithm_runtime"}, "PAR-2 identity mismatch", failures)
        _require(manifest.get("previous_attempt_results_reused") is False, "previous attempt results reused", failures)
        _require(config.get("previous_attempt_results_reused") is False, "resolved previous attempt results reused", failures)
        _require(manifest.get("prior_attempts") == config.get("prior_attempts"), "previous-attempt history mismatch", failures)
        _require(len(manifest.get("prior_attempts") or []) == 1, "previous-attempt history count mismatch", failures)
        if scale == "medium_large":
            _require(str(manifest.get("resolved_config_sha256")).upper() == resolved_canonical_sha, "legacy resolved config canonical SHA mismatch", failures)
        else:
            _require(manifest.get("resolved_config_file_sha256") == resolved_file_sha, "resolved config file SHA mismatch", failures)
            _require(manifest.get("resolved_config_canonical_sha256") == resolved_canonical_sha, "resolved config canonical SHA mismatch", failures)
            _require(manifest.get("resolved_config_canonicalization") == CANONICALIZATION, "resolved config canonicalization mismatch", failures)

        run_names = source.find("/run.json")
        status_names = source.find("/status.json")
        run_entries = [(name, source.read_json(name)) for name in run_names]
        status_entries = [(name, source.read_json(name)) for name in status_names]
        records_by_key = {str(value.get("run_key")): value for _, value in run_entries}
        statuses_by_key = {str(value.get("run_key")): value for _, value in status_entries}
        actual_directory_by_key = {
            str(value.get("run_key")): _member_directory_id(name)
            for name, value in run_entries
        }
        forward = dict(manifest.get("run_key_to_directory_id") or {})
        reverse = dict(manifest.get("directory_id_to_run_key") or {})
        spec_keys = {str(item.get("run_key")) for item in manifest.get("run_specs") or []}
        _require(len(run_names) == 27, "run.json count mismatch", failures)
        _require(len(status_names) == 27, "status.json count mismatch", failures)
        _require(len(records_by_key) == 27, "duplicate run key", failures)
        _require(len(statuses_by_key) == 27, "duplicate status run key", failures)
        _require(set(records_by_key) == set(statuses_by_key) == spec_keys, "run/status/spec key mismatch", failures)
        _require(all(record.get("state") == "complete" for record in records_by_key.values()), "non-complete run record", failures)
        _require(all(status.get("state") == "complete" for status in statuses_by_key.values()), "non-complete status record", failures)
        _require(manifest.get("pending_run_count") == 0, "pending run count is nonzero", failures)
        _require(manifest.get("failed_run_count") == 0, "pipeline failed run count is nonzero", failures)
        _require(manifest.get("completed_run_count") == 27, "manifest complete count mismatch", failures)
        _require(set(forward) == set(records_by_key), "forward mapping key drift", failures)
        _require(set(reverse) == set(forward.values()), "reverse mapping key drift", failures)
        _require(len(set(forward.values())) == 27, "short-directory collision", failures)
        for key, record in records_by_key.items():
            status = statuses_by_key[key]
            expected_directory = run_directory_id(key)
            actual_directory = actual_directory_by_key[key]
            _require(actual_directory == expected_directory, f"physical directory mismatch: {key}", failures)
            _require(record.get("run_directory_id") == expected_directory, f"run directory id mismatch: {key}", failures)
            _require(status.get("run_directory_id") == expected_directory, f"status directory id mismatch: {key}", failures)
            _require(status.get("run_key") == key, f"status canonical key mismatch: {key}", failures)
            _require(forward.get(key) == expected_directory, f"forward mapping mismatch: {key}", failures)
            _require(reverse.get(expected_directory) == key, f"reverse mapping mismatch: {key}", failures)
            for field in ("state", "task_type", "seed", "candidate", "scientific_status", "algorithm_status"):
                _require(status.get(field) == record.get(field), f"run/status {field} mismatch: {key}", failures)

        records = list(records_by_key.values())
        seeds = {int(record.get("seed")) for record in records}
        baseline = [record for record in records if record.get("task_type") == "baseline"]
        frontier = [record for record in records if record.get("task_type") == "frontier"]
        _require(seeds == set(EXPECTED_SEEDS), "record seeds mismatch", failures)
        _require(len(baseline) == 3 and len(frontier) == 24, "baseline/frontier count mismatch", failures)
        _require({float(record.get("rho")) for record in frontier} == set(EXPECTED_RHOS), "frontier rho mismatch", failures)
        for seed in EXPECTED_SEEDS:
            _require(sum(int(record.get("seed")) == seed for record in baseline) == 1, f"baseline count mismatch for seed {seed}", failures)
            _require(sum(int(record.get("seed")) == seed for record in frontier) == 8, f"frontier count mismatch for seed {seed}", failures)
        for candidate in EXPECTED_CANDIDATES:
            _require(sum(record.get("candidate") == candidate for record in frontier) == 6, f"candidate run count mismatch: {candidate}", failures)

        anchors = dict(manifest.get("baseline_anchors") or {})
        for baseline_record in baseline:
            seed = int(baseline_record["seed"])
            result = dict(baseline_record.get("result") or {})
            anchor = dict(anchors.get(str(seed)) or {})
            anchor_without_hash = {key: value for key, value in anchor.items() if key != "anchor_sha256"}
            anchor_hash = _canonical_yaml_sha256(anchor_without_hash).lower()
            tolerance = float(config["tol"])
            _require(baseline_record.get("scientific_status") == CERTIFIED_STATUS, f"baseline not certified: {seed}", failures)
            _require(result.get("status") == "optimal" and result.get("valid_UB") is True, f"baseline invalid UB: {seed}", failures)
            _require(_number(result.get("gap"), math.inf) <= tolerance, f"baseline gap exceeds tolerance: {seed}", failures)
            _require(
                float(anchor.get("value")) == float(result.get("upper_bound")),
                f"anchor upper-bound mismatch: {seed}",
                failures,
            )
            _require(anchor.get("value_hex") == float(anchor.get("value")).hex(), f"anchor hex mismatch: {seed}", failures)
            _require(str(anchor.get("anchor_sha256")).lower() == anchor_hash, f"anchor SHA mismatch: {seed}", failures)
            _require(anchor.get("baseline_run_key") == baseline_record.get("run_key"), f"anchor baseline key mismatch: {seed}", failures)
            seed_frontier = [record for record in frontier if int(record["seed"]) == seed]
            for record in seed_frontier:
                _require(record.get("instance_sha256") == baseline_record.get("instance_sha256"), f"shared instance mismatch: {record['run_key']}", failures)
                _require(record.get("baseline_run_key") == baseline_record.get("run_key"), f"shared baseline key mismatch: {record['run_key']}", failures)
                _require(record.get("anchor_value_hex") == anchor.get("value_hex"), f"shared anchor hex mismatch: {record['run_key']}", failures)
                _require(record.get("anchor_sha256") == anchor.get("anchor_sha256"), f"shared anchor SHA mismatch: {record['run_key']}", failures)
                _require(
                    float((record.get("result") or {}).get("baseline_cost"))
                    == float(anchor.get("value")),
                    f"shared anchor value mismatch: {record['run_key']}",
                    failures,
                )

        baseline_certified = sum(record.get("scientific_status") == CERTIFIED_STATUS for record in baseline)
        frontier_certified = sum(record.get("scientific_status") == CERTIFIED_STATUS for record in frontier)
        candidate_counts = {
            candidate: sum(
                record.get("candidate") == candidate
                and record.get("scientific_status") == CERTIFIED_STATUS
                for record in frontier
            )
            for candidate in EXPECTED_CANDIDATES
        }
        _require(baseline_certified == 3, "baseline certified count mismatch", failures)
        _require(frontier_certified == spec["frontier_certified_count"], "frontier certified count mismatch", failures)
        _require(candidate_counts == spec["candidate_certified_counts"], "candidate certified counts mismatch", failures)
        if scale == "large":
            _require(all(record.get("scientific_status") == "time_limit_uncertified" for record in frontier), "Large frontier status is not uniformly time_limit_uncertified", failures)
            _require(all(record.get("algorithm_status") == "time_limit" for record in frontier), "Large frontier algorithm status is not uniformly time_limit", failures)
        for record in records:
            run_key = str(record["run_key"])
            certified = record.get("scientific_status") == CERTIFIED_STATUS
            _require(
                bool(statuses_by_key[run_key].get("solved_to_tolerance")) == certified,
                f"solved classification mismatch: {run_key}",
                failures,
            )

        checkpoints = _algorithm_checkpoint_audit(source, records_by_key, reverse)
        post = _post_evaluation_audit(
            source, records_by_key, reverse, int(spec["scenario_count"])
        )
        if scale == "medium_large":
            _require(post["post_evaluation_count"] == 20, "Medium-large post-evaluation count mismatch", failures)
            _require(post["chunk_hash_count"] == 1480, "Medium-large checkpoint chunk count mismatch", failures)
        else:
            _require(checkpoints["algorithm_checkpoint_count"] == 24, "Large algorithm checkpoint count mismatch", failures)
            _require(post["post_evaluation_count"] == 0, "Large must not contain successful post-evaluation", failures)
            _require(post["checkpoint_index_count"] == 0, "Large must not contain post-evaluation indexes", failures)
        _require(checkpoints["identity_and_result_match"], "Algorithm checkpoint audit failed", failures)
        _require(post["all_valid"], "Post-evaluation audit failed", failures)

        projected_rows = [_project_record(record, float(config["fairness_time_limit"])) for record in records]
        projected_rows.sort(key=lambda row: row["run_key"])
        reconciliation = _projection_reconciliation(records)
        _require(reconciliation["check_count"] == 216, "projection reconciliation count mismatch", failures)
        _require(reconciliation["all_passed"], "projection reconciliation failed", failures)
        par2_mismatches = []
        par2_stored_count = 0
        par2_derived_baseline_count = 0
        for record, row in zip(sorted(records, key=lambda item: str(item["run_key"])), projected_rows):
            stored = (record.get("result") or {}).get("penalized_runtime_par2")
            if stored is None and record.get("task_type") == "baseline":
                par2_derived_baseline_count += 1
            elif stored is not None:
                par2_stored_count += 1
                if not _close(stored, row["penalized_runtime_par2"]):
                    par2_mismatches.append(str(record["run_key"]))
            else:
                par2_mismatches.append(str(record["run_key"]))
        _require(not par2_mismatches, "PAR-2 mismatch", failures)
        source_csv = _source_csv_audit(source, scale, projected_rows)
        _require(source_csv["results_row_count"] == 27, "source results row count mismatch", failures)
        _require(source_csv["summary_row_count"] == 9, "source summary row count mismatch", failures)
        _require(source_csv["status_identity_matches_rebuild"], "source results status mismatch", failures)
        summaries = candidate_summary(scale, projected_rows)
        _require(len(summaries) == 4, "candidate summary group count mismatch", failures)
        matrix_rows = build_run_matrix(
            scale,
            records,
            projected_rows,
            manifest,
            resolved_file_sha,
            resolved_canonical_sha,
        )
        _require(len(matrix_rows) == 27, "per-scale frozen matrix row count mismatch", failures)

        large_bottleneck = None
        if scale == "large":
            by_candidate = {item["candidate"]: item for item in summaries}
            single = by_candidate["single_cut"]
            persistent = by_candidate["persistent_separation"]
            cache = by_candidate["persistent_certified_cache"]
            batch = by_candidate["persistent_certified_cache_batch5"]
            large_bottleneck = {
                "single_cut_separation_dominates": single["mean_separation_runtime"] > 0.9 * single["mean_algorithm_runtime"],
                "persistent_separation_separation_dominates": persistent["mean_separation_runtime"] > 0.9 * persistent["mean_algorithm_runtime"],
                "persistent_certified_cache_master_dominates": cache["mean_master_runtime"] > 0.8 * cache["mean_algorithm_runtime"] and cache["mean_master_runtime"] > cache["mean_separation_runtime"],
                "batch5_reduces_iterations_vs_cache": batch["mean_iterations"] < cache["mean_iterations"],
                "batch5_increases_master_cuts_vs_cache": batch["mean_cuts"] > cache["mean_cuts"],
                "all_candidates_uncertified_at_1800_seconds": all(item["certified_solved_count"] == 0 for item in summaries),
                "interpretation_scope": "algorithm_development_evidence_only_not_certified_fairness_results",
            }
            _require(all(value is True for key, value in large_bottleneck.items() if key != "interpretation_scope"), "Large bottleneck evidence mismatch", failures)

        after_sha = file_sha256(source_path)
        _require(before_sha == after_sha, "source archive SHA changed during audit", failures)
        _require(crc_bad_member is None, "ZIP CRC failure", failures)
        _require(parseability["all_parseable"], "JSON/CSV parseability failure", failures)
        _require(core_equivalence.get("all_zero_difference") is True, "core equivalence missing", failures)
        report = {
            "audit_status": "passed" if not failures else "failed",
            "evidence_kind": "derived_read_only_cross_scale_audit_not_optimization_output",
            "source_archives": dict(source_archives),
            "scale": scale,
            "archive": {
                "name": source_path.name,
                "sha256_before_audit": before_sha,
                "sha256_after_audit": after_sha,
                "sha256_matches_expected": before_sha == spec["archive_sha256"],
                "crc_valid": crc_bad_member is None,
                "bad_crc_member": crc_bad_member,
                "entry_count": len(source.entries),
                "file_count": len(source.files),
                "explicit_directory_entry_count": len(source.directories),
                "inferred_directory_count": len(source.inferred_directories()),
                "access_mode": "read_only",
            },
            "parseability": parseability,
            "identity": {
                "schema_version": manifest.get("schema_version"),
                "execution_attempt": manifest.get("execution_attempt"),
                "git_commit": commit,
                "config_file_sha256": manifest.get("config_file_sha256"),
                "resolved_config_file_sha256": resolved_file_sha,
                "resolved_config_canonical_sha256": resolved_canonical_sha,
                "resolved_config_canonicalization": CANONICALIZATION,
                "manifest_legacy_resolved_config_sha256": manifest.get("resolved_config_sha256"),
                "protocol_sha256": manifest.get("protocol_sha256"),
                "candidate_config_sha256": manifest.get("candidate_config_sha256"),
                "scale": manifest.get("scale"),
                "seeds": manifest.get("seeds"),
                "rhos": manifest.get("rhos"),
                "candidate_definitions": manifest.get("candidate_definitions"),
                "gurobi_parameters": manifest.get("gurobi_parameters"),
                "baseline_time_limit": manifest.get("baseline_time_limit"),
                "fairness_time_limit": manifest.get("fairness_time_limit"),
                "general_time_limit": config.get("time_limit"),
                "post_evaluation": manifest.get("post_evaluation"),
                "par2": manifest.get("par2"),
                "previous_attempts": manifest.get("prior_attempts"),
                "previous_attempt_results_reused": manifest.get("previous_attempt_results_reused"),
            },
            "core_equivalence": dict(core_equivalence),
            "coverage": {
                "run_json_count": len(run_names),
                "status_json_count": len(status_names),
                "complete_count": sum(record.get("state") == "complete" for record in records),
                "running_count": sum(record.get("state") == "running" for record in records),
                "pending_run_count": manifest.get("pending_run_count"),
                "pipeline_failed_run_count": manifest.get("failed_run_count"),
                "unique_run_key_count": len(records_by_key),
                "unique_directory_id_count": len(set(forward.values())),
                "baseline_count": len(baseline),
                "frontier_count": len(frontier),
                "mapping_rule": 'r_ + sha256(canonical_run_key UTF-8).hexdigest()[:24]',
                "six_way_mapping_consistent": not any("directory" in failure or "mapping" in failure for failure in failures),
            },
            "baseline_and_anchor": {
                "certified_baseline_count": baseline_certified,
                "all_valid_ub": all((record.get("result") or {}).get("valid_UB") is True for record in baseline),
                "all_anchor_checks_passed": not any("anchor" in failure or "baseline" in failure for failure in failures),
                "anchors": anchors,
            },
            "scientific_status": {
                "pipeline_failed_run_count": manifest.get("failed_run_count"),
                "baseline_certified_solved_count": baseline_certified,
                "frontier_certified_solved_count": frontier_certified,
                "frontier_run_count": len(frontier),
                "candidate_certified_counts": candidate_counts,
                "scientific_status_counts": {
                    status: sum(record.get("scientific_status") == status for record in records)
                    for status in sorted({str(record.get("scientific_status")) for record in records})
                },
                "algorithm_status_counts": {
                    status: sum(record.get("algorithm_status") == status for record in records)
                    for status in sorted({str(record.get("algorithm_status")) for record in records})
                },
                "uncertified_values_excluded_from_candidate_selection": True,
            },
            "algorithm_checkpoints": checkpoints,
            "post_evaluation": post,
            "reporting": {
                "results_row_count": len(projected_rows),
                "candidate_summary_group_count": len(summaries),
                "projection_reconciliation": reconciliation,
                "par2_mismatch_count": len(par2_mismatches),
                "par2_stored_frontier_count": par2_stored_count,
                "par2_derived_certified_baseline_count": par2_derived_baseline_count,
                "par2_rule": "certified: algorithm_runtime; uncertified: 2 * 1800 seconds",
                "source_csv": source_csv,
                "projected_rows_sha256": _sha256_bytes(
                    json.dumps(projected_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ),
            },
            "candidate_summary": summaries,
            "large_bottleneck_evidence": large_bottleneck,
            "failure_count": len(failures),
            "failures": failures,
        }
        if failures:
            raise CrossScaleAuditError(
                f"{scale} audit failed: " + "; ".join(failures[:10])
            )
        return report, summaries, matrix_rows
    finally:
        source.close()


def build_decision(
    medium_audit: Mapping[str, Any], large_audit: Mapping[str, Any]
) -> dict[str, Any]:
    medium_certified = int(
        medium_audit["scientific_status"]["frontier_certified_solved_count"]
    )
    large_certified = int(
        large_audit["scientific_status"]["frontier_certified_solved_count"]
    )
    large_counts = dict(
        large_audit["scientific_status"]["candidate_certified_counts"]
    )
    decision = {
        "decision": "no_existing_candidate_passes_cross_scale_s1",
        "data_integrity_valid": True,
        "medium_large_pipeline_valid": medium_audit["audit_status"] == "passed",
        "large_pipeline_valid": large_audit["audit_status"] == "passed",
        "medium_large_certified_frontier": f"{medium_certified}/24",
        "large_certified_frontier": f"{large_certified}/24",
        "large_candidate_certified_counts": {
            candidate: f"{int(large_counts[candidate])}/6"
            for candidate in EXPECTED_CANDIDATES
        },
        "existing_candidate_selected": None,
        "original_s2_authorized": False,
        "full_grid_authorized": False,
        "attempt4_authorized": False,
        "current_results_usable_for_algorithm_development": True,
        "current_large_results_usable_as_certified_fairness_results": False,
        "next_authorized_stage": "fairness_large_final_remediation_protocol_only",
        "s1_screening_complete": True,
        "candidate_passage_may_not_be_manufactured_by": [
            "increasing_time_limit",
            "lowering_certification_threshold",
            "replacing_seed",
            "selective_rerun",
        ],
        "remediation_requirements": {
            "independent_protocol": True,
            "independent_branch": True,
            "independent_output_directory": True,
            "independent_execution_attempt": True,
            "maximum_new_algorithm_candidates": 1,
            "further_candidates_after_failure_authorized": False,
        },
        "evidence_kind": "derived_stage_decision_not_optimization_output",
        "source_archives": {
            "medium_large": MEDIUM_LARGE_ARCHIVE_SHA256,
            "large": LARGE_ARCHIVE_SHA256,
        },
    }
    expected_large = {candidate: "0/6" for candidate in EXPECTED_CANDIDATES}
    if (
        decision["medium_large_certified_frontier"] != "20/24"
        or decision["large_certified_frontier"] != "0/24"
        or decision["large_candidate_certified_counts"] != expected_large
    ):
        raise CrossScaleAuditError("Frozen cross-scale S1 decision facts do not match.")
    return decision


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_freeze_artifacts(
    output_dir: str | Path,
    medium_audit: Mapping[str, Any],
    large_audit: Mapping[str, Any],
    summaries: list[dict[str, Any]],
    run_matrix: list[dict[str, Any]],
) -> dict[str, str]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise CrossScaleAuditError(f"Output directory must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ordered_summaries = sorted(
        summaries,
        key=lambda item: (str(item.get("scale")), str(item.get("candidate"))),
    )
    ordered_run_matrix = sorted(run_matrix, key=_run_matrix_sort_key)
    matrix_validation = validate_run_matrix(ordered_run_matrix)
    matrix_summary = experiment_matrix_summary(ordered_run_matrix)
    decision = build_decision(medium_audit, large_audit)
    provenance = {
        "evidence_kind": "derived_read_only_cross_scale_freeze_not_optimization_output",
        "source_archives": {
            "medium_large": {
                "name": SCALE_SPECS["medium_large"]["archive_name"],
                "sha256": MEDIUM_LARGE_ARCHIVE_SHA256,
                "size_bytes": medium_audit["archive"].get("size_bytes"),
                "access_mode": "read_only",
                "sha256_unchanged": medium_audit["archive"]["sha256_before_audit"]
                == medium_audit["archive"]["sha256_after_audit"],
            },
            "large": {
                "name": SCALE_SPECS["large"]["archive_name"],
                "sha256": LARGE_ARCHIVE_SHA256,
                "size_bytes": large_audit["archive"].get("size_bytes"),
                "access_mode": "read_only",
                "sha256_unchanged": large_audit["archive"]["sha256_before_audit"]
                == large_audit["archive"]["sha256_after_audit"],
            },
        },
        "frozen_experiment_matrix": matrix_validation,
        "derived_artifacts_are_formal_optimization_outputs": False,
    }
    files = {
        "decision.json": decision,
        "source_archive_provenance.json": provenance,
        "medium_large_audit.json": dict(medium_audit),
        "large_audit.json": dict(large_audit),
    }
    for name, value in files.items():
        _write_json(output / name, value)
    _write_csv(
        output / "cross_scale_candidate_summary.csv",
        ordered_summaries,
        SUMMARY_FIELDS,
    )
    _write_csv(output / "frozen_run_matrix.csv", ordered_run_matrix, RUN_MATRIX_FIELDS)
    _write_csv(
        output / "experiment_matrix_summary.csv",
        matrix_summary,
        EXPERIMENT_MATRIX_SUMMARY_FIELDS,
    )
    indexed = [
        "decision.json",
        "source_archive_provenance.json",
        "medium_large_audit.json",
        "large_audit.json",
        "cross_scale_candidate_summary.csv",
        "frozen_run_matrix.csv",
        "experiment_matrix_summary.csv",
    ]
    hashes = {name: file_sha256(output / name) for name in indexed}
    artifact_rows = [
        {
            "medium_large_source_archive_sha256": MEDIUM_LARGE_ARCHIVE_SHA256,
            "large_source_archive_sha256": LARGE_ARCHIVE_SHA256,
            "artifact_path": name,
            "sha256": hashes[name],
            "evidence_kind": "derived_cross_scale_freeze_not_optimization_output",
        }
        for name in indexed
    ]
    _write_csv(
        output / "artifact_sha256.csv",
        artifact_rows,
        (
            "medium_large_source_archive_sha256",
            "large_source_archive_sha256",
            "artifact_path",
            "sha256",
            "evidence_kind",
        ),
    )
    hashes["artifact_sha256.csv"] = file_sha256(output / "artifact_sha256.csv")
    return hashes


def audit_cross_scale(
    medium_large_zip: str | Path,
    large_zip: str | Path,
    output_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    medium_path = Path(medium_large_zip)
    large_path = Path(large_zip)
    # Fail before opening either archive if either immutable identity is absent or wrong.
    for path, expected in (
        (medium_path, MEDIUM_LARGE_ARCHIVE_SHA256),
        (large_path, LARGE_ARCHIVE_SHA256),
    ):
        if not path.is_file():
            raise CrossScaleAuditError(f"Required source archive is missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise CrossScaleAuditError(
                f"Required source archive SHA256 mismatch: {path.name}: {actual}"
            )
    source_archives = {
        "medium_large": MEDIUM_LARGE_ARCHIVE_SHA256,
        "large": LARGE_ARCHIVE_SHA256,
    }
    core = audit_core_equivalence(repo_root)
    medium_audit, medium_summary, medium_matrix = audit_archive(
        medium_path, "medium_large", repo_root, source_archives, core
    )
    medium_audit["archive"]["size_bytes"] = medium_path.stat().st_size
    large_audit, large_summary, large_matrix = audit_archive(
        large_path, "large", repo_root, source_archives, core
    )
    large_audit["archive"]["size_bytes"] = large_path.stat().st_size
    if (
        medium_audit["identity"]["candidate_definitions"]
        != large_audit["identity"]["candidate_definitions"]
    ):
        raise CrossScaleAuditError("Cross-scale candidate definitions differ.")
    hashes = write_freeze_artifacts(
        output_dir,
        medium_audit,
        large_audit,
        [*medium_summary, *large_summary],
        [*medium_matrix, *large_matrix],
    )
    return {
        "decision": build_decision(medium_audit, large_audit),
        "artifact_sha256": hashes,
        "source_archive_sha256": source_archives,
        "audit_status": "passed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Solver-free read-only cross-scale S1 Attempt 2 archive audit"
    )
    parser.add_argument("--medium-large-zip", required=True)
    parser.add_argument("--large-zip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    report = audit_cross_scale(
        args.medium_large_zip,
        args.large_zip,
        args.output_dir,
        args.repo_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
