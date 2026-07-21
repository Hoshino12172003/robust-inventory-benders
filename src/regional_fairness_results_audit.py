from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import statistics
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, TextIO
import zipfile

import yaml

from .experiment_protocol import atomic_write_json, config_sha256, file_sha256, utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_RUN_COMMIT = "ce96c183248044c024f046a9a2bbe29c6f0f6f04"
EXPECTED_METHOD = "joint_v1_core_point_strengthened"
EXPECTED_PROTOCOL_SHA256 = "ec7761d96c1d2a17f96eba90bf4bfb520a9ce6359f938acd7f294a10e7f24a38"
EXPECTED_CANDIDATE_SHA256 = "7e8aaf39de8c100b4ce9b46256a074fbd324b07ddc347d256494ed070d4e0eb6"
EXPECTED_ARCHIVE_SHA256 = {
    "medium_large": "a8011c1b4df7ecae317f7fdceccf9cb0e042db97174cce6a32e96c6fd0070ae8",
    "large": "2d64325b0c40330f54da8644cb90e2bc0607983e79d98810988198b1ebb96ad7",
}
EXPECTED_CONFIG_SHA256 = {
    "medium_large": "04d2ca32c31d7b2d3c9071583c4bc3897740b463d6ad945a8a52554a6317c79c",
    "large": "7a40ff6cfedb02f44d57c999377377b7eb25e406ebe417791ca7a0c22c2fb307",
}
MATERIAL_GAP = 0.10
STRUCTURAL_MEDIAN = 0.05
NO_MATERIAL_MEDIAN = 0.03
DEGENERACY_REDUCTION = 0.05
ABS_TOL = 1.0e-8


@dataclass(frozen=True)
class ScaleExpectation:
    label: str
    instance_size: str
    experiment_name: str
    seeds: tuple[int, ...]
    scenario_count: int
    num_regions: int
    num_products: int
    checkpoint_count: int
    archive_sha256: str | None
    config_sha256: str | None
    config_filename: str | None
    run_commit: str
    protocol_sha256: str
    candidate_sha256: str
    chunk_size: int = 50


FORMAL_EXPECTATIONS = {
    "medium_large": ScaleExpectation(
        label="medium_large",
        instance_size="medium_large",
        experiment_name="regional_fairness_diagnostic_medium_large",
        seeds=tuple(range(110, 120)),
        scenario_count=1831,
        num_regions=10,
        num_products=6,
        checkpoint_count=370,
        archive_sha256=EXPECTED_ARCHIVE_SHA256["medium_large"],
        config_sha256=EXPECTED_CONFIG_SHA256["medium_large"],
        config_filename="regional_fairness_diagnostic_medium_large.yaml",
        run_commit=FORMAL_RUN_COMMIT,
        protocol_sha256=EXPECTED_PROTOCOL_SHA256,
        candidate_sha256=EXPECTED_CANDIDATE_SHA256,
    ),
    "large": ScaleExpectation(
        label="large",
        instance_size="large",
        experiment_name="regional_fairness_diagnostic_large",
        seeds=tuple(range(110, 120)),
        scenario_count=4657,
        num_regions=12,
        num_products=8,
        checkpoint_count=940,
        archive_sha256=EXPECTED_ARCHIVE_SHA256["large"],
        config_sha256=EXPECTED_CONFIG_SHA256["large"],
        config_filename="regional_fairness_diagnostic_large.yaml",
        run_commit=FORMAL_RUN_COMMIT,
        protocol_sha256=EXPECTED_PROTOCOL_SHA256,
        candidate_sha256=EXPECTED_CANDIDATE_SHA256,
    ),
}


class ResultSource:
    """Read a ZIP or extracted result tree without modifying it."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.archive: zipfile.ZipFile | None = None
        if self.path.is_file():
            self.archive = zipfile.ZipFile(self.path, "r")
            self.names = tuple(info.filename for info in self.archive.infolist() if not info.is_dir())
        elif self.path.is_dir():
            self.names = tuple(
                item.relative_to(self.path).as_posix()
                for item in self.path.rglob("*")
                if item.is_file()
            )
        else:
            raise FileNotFoundError(self.path)
        result_names = [name for name in self.names if name == "results.csv" or name.endswith("/results.csv")]
        if len(result_names) != 1:
            raise ValueError(f"Expected one results.csv, found {len(result_names)}")
        parent = str(PurePosixPath(result_names[0]).parent)
        self.root = "" if parent == "." else parent

    def __enter__(self) -> "ResultSource":
        return self

    def __exit__(self, *_args: object) -> None:
        if self.archive is not None:
            self.archive.close()

    def name(self, relative: str) -> str:
        return f"{self.root}/{relative}" if self.root else relative

    def relative_names(self) -> tuple[str, ...]:
        prefix = f"{self.root}/" if self.root else ""
        return tuple(name[len(prefix) :] for name in self.names if name.startswith(prefix))

    def exists(self, relative: str) -> bool:
        return self.name(relative) in self.names

    def open_binary(self, relative: str) -> BinaryIO:
        name = self.name(relative)
        if self.archive is not None:
            return self.archive.open(name, "r")
        return (self.path / name).open("rb")

    def open_text(self, relative: str, *, newline: str | None = None) -> TextIO:
        return io.TextIOWrapper(self.open_binary(relative), encoding="utf-8-sig", newline=newline)

    def read_bytes(self, relative: str) -> bytes:
        with self.open_binary(relative) as source:
            return source.read()

    def json_value(self, relative: str) -> dict[str, Any]:
        value = json.loads(self.read_bytes(relative).decode("utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError(f"{relative} must contain a JSON object")
        return value

    def yaml_value(self, relative: str) -> dict[str, Any]:
        value = yaml.safe_load(self.read_bytes(relative))
        if not isinstance(value, dict):
            raise ValueError(f"{relative} must contain a YAML mapping")
        return value

    def member_sha256(self, relative: str) -> str:
        digest = hashlib.sha256()
        with self.open_binary(relative) as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def archive_sha256(self) -> str | None:
        return file_sha256(self.path).lower() if self.path.is_file() else None

    def crc_error(self) -> str | None:
        return self.archive.testzip() if self.archive is not None else None


def _check(name: str, passed: bool, details: Any = "") -> dict[str, Any]:
    return {"check": name, "required": True, "passed": bool(passed), "details": details}


def _close(left: Any, right: Any, *, tol: float = ABS_TOL) -> bool:
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return _canonical_json(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _first_stage_x_sha256(values: list[list[float]]) -> str:
    payload = json.dumps(
        [[float(value) for value in row] for row in values],
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pattern_sha256(pattern: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(pattern).encode("utf-8")).hexdigest()


def _scenario_key(index: int, pattern: list[dict[str, Any]]) -> str:
    return f"scenario_{index:05d}_{_pattern_sha256(pattern)[:16]}"


def _scenario_units(num_regions: int, num_products: int) -> Iterator[tuple[tuple[int, int], ...]]:
    units = tuple((region, product) for region in range(num_regions) for product in range(num_products))
    yield ()
    yield from itertools.combinations(units, 1)
    yield from itertools.combinations(units, 2)


def _expected_pattern(instance: Mapping[str, Any], active: tuple[tuple[int, int], ...]) -> list[dict[str, Any]]:
    base = instance["base_demand"]
    deviation = instance["demand_deviation"]
    return [
        {
            "region_id": int(region),
            "product_id": int(product),
            "deviation_value": float(deviation[region][product]),
            "base_demand": float(base[region][product]),
            "realized_demand": float(base[region][product] + deviation[region][product]),
        }
        for region, product in active
    ]


def _expected_demand(instance: Mapping[str, Any], active: tuple[tuple[int, int], ...]) -> list[list[float]]:
    demand = [[float(value) for value in row] for row in instance["base_demand"]]
    for region, product in active:
        demand[region][product] += float(instance["demand_deviation"][region][product])
    return demand


def _matrix_close(left: Any, right: Any) -> bool:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    return all(
        isinstance(a, list)
        and isinstance(b, list)
        and len(a) == len(b)
        and all(_close(x, y) for x, y in zip(a, b))
        for a, b in zip(left, right)
    )


def _recompute_metrics(
    demand: list[list[float]],
    shortage: list[list[float]],
    *,
    metric_tolerance: float,
) -> dict[str, Any]:
    if not _matrix_close(shortage, shortage) or len(demand) != len(shortage):
        raise ValueError("Demand/shortage matrix shape mismatch")
    regions: list[dict[str, Any]] = []
    for region, (demand_row, shortage_row) in enumerate(zip(demand, shortage)):
        if len(demand_row) != len(shortage_row):
            raise ValueError("Demand/shortage product dimension mismatch")
        regional_demand = math.fsum(float(value) for value in demand_row)
        regional_shortage = math.fsum(float(value) for value in shortage_row)
        if regional_shortage < -metric_tolerance or regional_shortage > regional_demand + metric_tolerance:
            raise ValueError("Regional shortage violates [0, demand]")
        applicable = regional_demand > metric_tolerance
        fill = None if not applicable else 1.0 - regional_shortage / regional_demand
        if fill is not None and not (-metric_tolerance <= fill <= 1.0 + metric_tolerance):
            raise ValueError("Fill rate outside [0,1]")
        regions.append(
            {
                "region": region,
                "regional_demand": regional_demand,
                "regional_shortage": regional_shortage,
                "shortage_rate": None if not applicable else regional_shortage / regional_demand,
                "fill_rate": fill,
                "fill_rate_applicable": applicable,
                "not_applicable_reason": "" if applicable else "zero_regional_demand",
            }
        )
    applicable_fills = [float(row["fill_rate"]) for row in regions if row["fill_rate_applicable"]]
    total_demand = math.fsum(row["regional_demand"] for row in regions)
    total_shortage = math.fsum(row["regional_shortage"] for row in regions)
    weighted = None if total_demand <= metric_tolerance else 1.0 - total_shortage / total_demand
    gap = None if not applicable_fills else max(applicable_fills) - min(applicable_fills)
    minimum = None if not applicable_fills else min(applicable_fills)
    worst_deviation = (
        None
        if weighted is None or not applicable_fills
        else max(float(weighted) - value for value in applicable_fills)
    )
    return {
        "regions": regions,
        "weighted_mean_fill_rate": weighted,
        "fill_rate_gap": gap,
        "minimum_fill_rate": minimum,
        "worst_region_deviation": worst_deviation,
    }


def _structural(values_by_scale: Mapping[str, list[float]]) -> bool:
    per_scale = any(
        sum(value >= MATERIAL_GAP for value in values) >= 4
        and statistics.median(values) >= STRUCTURAL_MEDIAN
        for values in values_by_scale.values()
    )
    pooled = [value for values in values_by_scale.values() for value in values]
    return per_scale or (
        sum(value >= MATERIAL_GAP for value in pooled) >= 8
        and statistics.median(pooled) >= STRUCTURAL_MEDIAN
    )


def classify_joint_fairness(
    scale_summaries: Mapping[str, Mapping[str, list[float]]],
    *,
    correctness_valid: bool,
) -> dict[str, Any]:
    if not correctness_valid:
        return {
            "decision": "fairness_diagnostic_invalid",
            "diagnosis_valid": False,
            "fairness_gap_source": "invalid_evidence",
            "next_authorized_stage": "none",
        }
    fair = {scale: list(value["fair_best_WGap"]) for scale, value in scale_summaries.items()}
    default = {scale: list(value["default_WGap"]) for scale, value in scale_summaries.items()}
    fair_structural = _structural(fair)
    default_structural = _structural(default)
    reductions_by_scale = {
        scale: sum(
            before - after >= DEGENERACY_REDUCTION
            for before, after in zip(default[scale], fair[scale])
        )
        for scale in fair
    }
    no_material = all(
        sum(value >= MATERIAL_GAP for value in values) <= 1
        and statistics.median(values) < NO_MATERIAL_MEDIAN
        for values in fair.values()
    ) and not default_structural
    if fair_structural:
        decision = "structural_fairness_gap"
        source = "structural_not_recourse_degeneracy"
        next_stage = "fairness_model_development_protocol_only"
    elif default_structural and any(count >= 4 for count in reductions_by_scale.values()):
        decision = "recourse_degeneracy_only"
        source = "cost_optimal_recourse_degeneracy"
        next_stage = "lexicographic_recourse_protocol_only"
    elif no_material:
        decision = "no_material_fairness_gap"
        source = "no_material_regional_service_gap"
        next_stage = "none"
    else:
        decision = "fairness_diagnostic_inconclusive"
        source = "inconclusive"
        next_stage = "none"
    return {
        "decision": decision,
        "diagnosis_valid": True,
        "fairness_gap_source": source,
        "next_authorized_stage": next_stage,
        "retuning_allowed": False,
        "seed_replacement_allowed": False,
        "threshold_revision_allowed": False,
        "reductions_at_least_0_05_by_scale": reductions_by_scale,
    }


def _csv_rows(source: ResultSource, relative: str) -> list[dict[str, str]]:
    with source.open_text(relative, newline="") as handle:
        return list(csv.DictReader(handle))


def _artifact_names(source: ResultSource, prefix: str, suffix: str) -> list[str]:
    return sorted(name for name in source.relative_names() if name.startswith(prefix) and name.endswith(suffix))


def _validate_repository_identity(
    expectation: ScaleExpectation,
    repo_root: Path,
    checks: list[dict[str, Any]],
) -> None:
    if expectation.config_filename is not None and expectation.config_sha256 is not None:
        config_path = repo_root / "experiments/configs" / expectation.config_filename
        actual = file_sha256(config_path).lower() if config_path.is_file() else None
        checks.append(
            _check(
                f"{expectation.label}_repository_config_sha256",
                actual == expectation.config_sha256,
                actual,
            )
        )
    protocol = repo_root / "docs/regional_fairness_diagnostic_protocol.md"
    candidate = repo_root / "experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml"
    checks.extend(
        [
            _check(
                f"{expectation.label}_repository_protocol_sha256",
                protocol.is_file() and file_sha256(protocol).lower() == expectation.protocol_sha256,
                file_sha256(protocol).lower() if protocol.is_file() else None,
            ),
            _check(
                f"{expectation.label}_repository_candidate_sha256",
                candidate.is_file() and file_sha256(candidate).lower() == expectation.candidate_sha256,
                file_sha256(candidate).lower() if candidate.is_file() else None,
            ),
        ]
    )


def _load_base_runs(
    source: ResultSource,
    expectation: ScaleExpectation,
    manifest: Mapping[str, Any],
    checks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = _csv_rows(source, "results.csv")
    run_json_names = _artifact_names(source, "runs/", "/run.json")
    status_names = _artifact_names(source, "runs/", "/status.json")
    error_names = _artifact_names(source, "runs/", "/error.txt")
    resolved_names = _artifact_names(source, "runs/", "/resolved_config.yaml")
    expected_seeds = set(expectation.seeds)
    run_keys = [str(row.get("run_key", "")) for row in rows]
    observed_seeds = {int(row.get("seed", -1)) for row in rows}
    variants = {str(row.get("variant_name", "")) for row in rows}
    checks.extend(
        [
            _check(f"{expectation.label}_ten_base_results", len(rows) == len(expectation.seeds), len(rows)),
            _check(f"{expectation.label}_seeds_exact", observed_seeds == expected_seeds, sorted(observed_seeds)),
            _check(f"{expectation.label}_unique_base_run_keys", len(run_keys) == len(set(run_keys)) == len(expectation.seeds)),
            _check(f"{expectation.label}_only_frozen_method", variants == {EXPECTED_METHOD}, sorted(variants)),
            _check(f"{expectation.label}_run_json_count", len(run_json_names) == len(expectation.seeds), len(run_json_names)),
            _check(f"{expectation.label}_status_json_count", len(status_names) == len(expectation.seeds), len(status_names)),
            _check(f"{expectation.label}_error_file_count", len(error_names) == len(expectation.seeds), len(error_names)),
            _check(
                f"{expectation.label}_per_run_resolved_config_count",
                expectation.config_filename is None
                or len(resolved_names) == len(expectation.seeds),
                len(resolved_names),
            ),
            _check(
                f"{expectation.label}_manifest_base_run_keys",
                set(manifest.get("base_input_identity", {}).get("base_run_keys", [])) == set(run_keys),
            ),
        ]
    )
    if len(rows) != len(expectation.seeds) or observed_seeds != expected_seeds or len(run_keys) != len(set(run_keys)):
        raise ValueError("Base results are incomplete or duplicated")
    records: dict[str, dict[str, Any]] = {}
    instances: dict[str, dict[str, Any]] = {}
    all_valid = True
    for row in rows:
        key = str(row["run_key"])
        record_name = f"runs/{key}/run.json"
        status_name = f"runs/{key}/status.json"
        error_name = f"runs/{key}/error.txt"
        if not all(source.exists(name) for name in (record_name, status_name, error_name)):
            all_valid = False
            continue
        record = source.json_value(record_name)
        status = source.json_value(status_name)
        error_empty = source.read_bytes(error_name).decode("utf-8-sig").strip() == ""
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        seed = int(row["seed"])
        expected_key = (
            f"{expectation.experiment_name}__none__none__{expectation.instance_size}"
            f"__seed_{seed}__{EXPECTED_METHOD}"
        )
        valid = (
            key == expected_key
            and record.get("run_key") == key
            and record.get("state") == "complete"
            and record.get("success") is True
            and record.get("solved_to_tolerance") is True
            and record.get("git_commit") == expectation.run_commit
            and status.get("state") == "complete"
            and status.get("success") is True
            and status.get("solved_to_tolerance") is True
            and error_empty
            and result.get("git_commit") == expectation.run_commit
            and row.get("git_commit") == expectation.run_commit
            and row.get("variant_name") == EXPECTED_METHOD
            and row.get("instance_size") == expectation.instance_size
            and row.get("experiment_name") == expectation.experiment_name
            and _bool(row.get("solved_to_tolerance"))
            and row.get("status") == "optimal"
        )
        best_x = result.get("best_x_values")
        instance_name = str(result.get("instance_name", row.get("instance_name", "")))
        instance_file = f"instances/{instance_name}.json"
        if not isinstance(best_x, list) or not source.exists(instance_file):
            valid = False
        else:
            instance = source.json_value(instance_file)
            valid = valid and (
                instance.get("name") == instance_name
                and int(instance.get("num_regions", -1)) == expectation.num_regions
                and int(instance.get("num_products", -1)) == expectation.num_products
            )
            instances[key] = instance
        if result.get("config_sha256") != row.get("config_sha256") or record.get("config_sha256") != row.get("config_sha256"):
            valid = False
        if expectation.config_filename is not None:
            run_config_name = f"runs/{key}/resolved_config.yaml"
            if not source.exists(run_config_name):
                valid = False
            else:
                run_config = source.yaml_value(run_config_name)
                algorithm = run_config.get("algorithm", {})
                robust = run_config.get("robust", {})
                benders = run_config.get("benders", {})
                instance_config = run_config.get("instance", {})
                valid = valid and (
                    config_sha256(run_config) == row.get("config_sha256")
                    and run_config.get("seed") == seed
                    and instance_config.get("num_regions") == expectation.num_regions
                    and instance_config.get("num_products") == expectation.num_products
                    and robust.get("gamma_target") == 2
                    and robust.get("gamma_schedule") == [2]
                    and robust.get("exact_scenarios") is True
                    and algorithm.get("subproblem_mode") == "robust_dual_milp"
                    and algorithm.get("precision_policy") == "joint_error_budget"
                    and _close(algorithm.get("master_error_budget_ratio"), 0.25)
                    and _close(algorithm.get("subproblem_error_budget_ratio"), 0.50)
                    and algorithm.get("cut_strengthening_policy") == "core_point"
                    and algorithm.get("max_cuts_per_iteration") == 1
                    and algorithm.get("cut_selection_enabled") is False
                    and algorithm.get("adaptive_secondary_generation_enabled") is False
                    and algorithm.get("adaptive_subproblem_gap_enabled") is False
                    and benders.get("tol") == 1.0e-4
                )
        records[key] = {
            "row": row,
            "record": record,
            "result": result,
            "seed": seed,
            "instance_name": instance_name,
            "best_x_values": best_x,
            "first_stage_x_sha256": _first_stage_x_sha256(best_x) if isinstance(best_x, list) else "",
        }
        all_valid = all_valid and valid
    checks.append(_check(f"{expectation.label}_base_runs_complete_and_frozen", all_valid))
    if not all_valid:
        raise ValueError("Base run identity or status failed")
    return rows, records, instances


def _validate_manifest_and_hashes(
    source: ResultSource,
    expectation: ScaleExpectation,
    repo_root: Path,
    checks: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "results.csv",
        "summary.csv",
        "run_manifest.json",
        "resolved_config.yaml",
        "diagnostic_run_manifest.json",
        "diagnosis.json",
        "instance_summary.csv",
        "region_scenario_metrics.csv",
        "audit_log.json",
        "checkpoint/index.json",
    }
    relatives = source.relative_names()
    duplicates = sorted(name for name in set(relatives) if relatives.count(name) > 1)
    archive_sha = source.archive_sha256()
    checks.extend(
        [
            _check(f"{expectation.label}_no_duplicate_archive_members", not duplicates, duplicates[:10]),
            _check(f"{expectation.label}_required_outputs_present", all(source.exists(name) for name in required)),
            _check(
                f"{expectation.label}_archive_sha256",
                expectation.archive_sha256 is None
                or archive_sha is None
                or archive_sha == expectation.archive_sha256,
                archive_sha or "directory input",
            ),
            _check(f"{expectation.label}_archive_crc", source.crc_error() is None),
            _check(
                f"{expectation.label}_no_temporary_or_lock_files",
                not any(name.endswith((".tmp", ".lock")) for name in relatives),
            ),
        ]
    )
    if duplicates or not all(source.exists(name) for name in required):
        raise ValueError("Archive members are duplicated or incomplete")
    manifest = source.json_value("diagnostic_run_manifest.json")
    run_manifest = source.json_value("run_manifest.json")
    identity = manifest.get("identity") if isinstance(manifest.get("identity"), dict) else {}
    count_ok = (
        manifest.get("status") == "completed"
        and int(manifest.get("completed_scenario_count", -1)) == expectation.scenario_count * len(expectation.seeds)
        and int(manifest.get("expected_chunk_count", -1)) == expectation.checkpoint_count
        and int(manifest.get("completed_chunk_count", -1)) == expectation.checkpoint_count
        and int(manifest.get("pending_count", -1)) == 0
        and int(manifest.get("pending_chunk_count", -1)) == 0
        and int(manifest.get("failed_count", -1)) == 0
        and int(manifest.get("failed_chunk_count", -1)) == 0
        and int(manifest.get("interrupted_count", -1)) == 0
        and int(manifest.get("interrupted_chunk_count", -1)) == 0
        and not manifest.get("failure_reason")
    )
    identity_ok = (
        manifest.get("base_git_commit") == expectation.run_commit
        and manifest.get("diagnostic_code_git_commit") == expectation.run_commit
        and manifest.get("config_sha256") == expectation.config_sha256
        and manifest.get("protocol_document_sha256") == expectation.protocol_sha256
        and manifest.get("candidate_config_sha256") == expectation.candidate_sha256
        and identity.get("experiment_name") == expectation.experiment_name
        and identity.get("instance_size") == expectation.instance_size
        and identity.get("method") == EXPECTED_METHOD
        and identity.get("seeds") == list(expectation.seeds)
        and identity.get("protocol_document_sha256") == expectation.protocol_sha256
        and identity.get("candidate_config_sha256") == expectation.candidate_sha256
    )
    run_manifest_ok = (
        run_manifest.get("expected_run_count") == len(expectation.seeds)
        and run_manifest.get("completed_run_count") == len(expectation.seeds)
        and run_manifest.get("solved_run_count") == len(expectation.seeds)
        and run_manifest.get("failed_run_count") == 0
        and run_manifest.get("remaining_run_count") == 0
        and run_manifest.get("git_commit") == expectation.run_commit
    )
    checks.extend(
        [
            _check(f"{expectation.label}_diagnostic_manifest_completed", count_ok),
            _check(f"{expectation.label}_frozen_identity", identity_ok, identity),
            _check(f"{expectation.label}_base_manifest_complete", run_manifest_ok, run_manifest),
        ]
    )
    resolved = source.yaml_value("resolved_config.yaml")
    resolved_hash = config_sha256(resolved)
    frozen_resolved_ok = True
    if expectation.config_filename is not None:
        raw_path = repo_root / "experiments/configs" / expectation.config_filename
        raw_value = yaml.safe_load(raw_path.read_bytes()) if raw_path.is_file() else None
        frozen_resolved_ok = (
            isinstance(raw_value, dict)
            and all(resolved.get(key) == value for key, value in raw_value.items())
            and resolved.get("variants") == [EXPECTED_METHOD]
            and resolved.get("precision_policy") == "joint_error_budget"
            and _close(resolved.get("master_error_budget_ratio"), 0.25)
            and _close(resolved.get("subproblem_error_budget_ratio"), 0.50)
            and resolved.get("cut_strengthening_policy") == "core_point"
            and resolved.get("max_cuts_per_iteration") == 1
            and resolved.get("adaptive_secondary_generation_enabled") is False
            and resolved.get("gamma_target") == 2
            and resolved.get("gamma_schedule") == [2]
        )
    checks.append(
        _check(
            f"{expectation.label}_resolved_config_sha256",
            resolved_hash == manifest.get("resolved_config_sha256")
            == identity.get("resolved_config_sha256")
            == run_manifest.get("config_sha256"),
            resolved_hash,
        )
    )
    checks.append(
        _check(f"{expectation.label}_resolved_config_matches_frozen_protocol", frozen_resolved_ok)
    )
    final_hashes = manifest.get("final_outputs") if isinstance(manifest.get("final_outputs"), dict) else {}
    required_final_hashes = {
        "audit_log.json",
        "checkpoint/index.json",
        "diagnosis.json",
        "instance_summary.csv",
        "region_scenario_metrics.csv",
        "resolved_config.yaml",
    }
    final_mismatches: dict[str, Any] = {}
    for name, expected in final_hashes.items():
        actual = source.member_sha256(name) if source.exists(name) else None
        if actual != expected:
            final_mismatches[name] = {"recorded": expected, "actual": actual}
    base_hashes = manifest.get("base_results_files") if isinstance(manifest.get("base_results_files"), dict) else {}
    required_base_hashes = {"results.csv", "summary.csv", "run_manifest.json"}
    base_mismatches: dict[str, Any] = {}
    for name, expected in base_hashes.items():
        actual = source.member_sha256(name) if source.exists(name) else None
        if actual != expected:
            base_mismatches[name] = {"recorded": expected, "actual": actual}
    checks.extend(
        [
            _check(f"{expectation.label}_all_final_output_hashes", not final_mismatches, final_mismatches),
            _check(f"{expectation.label}_all_base_output_hashes", not base_mismatches, base_mismatches),
            _check(
                f"{expectation.label}_final_hash_set_complete",
                set(final_hashes) == required_final_hashes,
                sorted(final_hashes),
            ),
            _check(
                f"{expectation.label}_base_hash_set_complete",
                set(base_hashes) == required_base_hashes,
                sorted(base_hashes),
            ),
        ]
    )
    _validate_repository_identity(expectation, repo_root, checks)
    if (
        not count_ok
        or not identity_ok
        or not run_manifest_ok
        or final_mismatches
        or base_mismatches
        or set(final_hashes) != required_final_hashes
        or set(base_hashes) != required_base_hashes
    ):
        raise ValueError("Manifest, identity, or recorded output hash failed")
    return manifest, resolved


def _metric_matches(stored: Mapping[str, Any], computed: Mapping[str, Any]) -> bool:
    scalar_fields = (
        "weighted_mean_fill_rate",
        "fill_rate_gap",
        "minimum_fill_rate",
        "worst_region_deviation",
    )
    if not all(_close(stored.get(field), computed.get(field)) for field in scalar_fields):
        return False
    regions = stored.get("regions")
    if not isinstance(regions, list) or len(regions) != len(computed["regions"]):
        return False
    for actual, expected in zip(regions, computed["regions"]):
        if actual.get("region") != expected["region"]:
            return False
        for field in ("regional_demand", "regional_shortage", "shortage_rate", "fill_rate"):
            if expected[field] is None:
                if actual.get(field) is not None:
                    return False
            elif not _close(actual.get(field), expected[field]):
                return False
        if bool(actual.get("fill_rate_applicable")) != expected["fill_rate_applicable"]:
            return False
        if str(actual.get("not_applicable_reason", "")) != expected["not_applicable_reason"]:
            return False
    return True


def _scan_checkpoints(
    source: ResultSource,
    expectation: ScaleExpectation,
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    base_records: Mapping[str, Mapping[str, Any]],
    instances: Mapping[str, Mapping[str, Any]],
    checks: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    index = source.json_value("checkpoint/index.json")
    entries = index.get("entries") if isinstance(index.get("entries"), list) else []
    checkpoint_names = _artifact_names(source, "checkpoint/base_", ".json")
    sorted_entries = sorted(entries, key=lambda item: (str(item.get("base_run_key")), int(item.get("chunk_index", -1))))
    index_order_ok = entries == sorted_entries
    listed_names = [str(item.get("relative_path", "")) for item in entries]
    checks.extend(
        [
            _check(f"{expectation.label}_checkpoint_count", len(entries) == expectation.checkpoint_count, len(entries)),
            _check(f"{expectation.label}_checkpoint_index_order", index_order_ok),
            _check(
                f"{expectation.label}_checkpoint_files_exact",
                len(checkpoint_names) == expectation.checkpoint_count
                and set(checkpoint_names) == set(listed_names),
                {"files": len(checkpoint_names), "index": len(listed_names)},
            ),
        ]
    )
    if len(entries) != expectation.checkpoint_count or set(checkpoint_names) != set(listed_names):
        raise ValueError("Checkpoint index/file coverage mismatch")
    metric_config = resolved.get("fairness_diagnostic") if isinstance(resolved.get("fairness_diagnostic"), dict) else {}
    absolute_tolerance = float(metric_config.get("cost_absolute_tolerance", math.nan))
    relative_tolerance = float(metric_config.get("cost_relative_tolerance", math.nan))
    metric_tolerance = float(metric_config.get("metric_tolerance", math.nan))
    states: dict[str, dict[str, Any]] = {}
    expected_iterators = {
        key: iter(_scenario_units(expectation.num_regions, expectation.num_products))
        for key in base_records
    }
    next_indices = {key: 0 for key in base_records}
    checkpoint_hashes_ok = True
    scenario_integrity_ok = True
    metric_integrity_ok = True
    recourse_integrity_ok = True
    for key, meta in base_records.items():
        states[key] = {
            "seed": meta["seed"],
            "default_WGap": -math.inf,
            "fair_best_WGap": -math.inf,
            "default_WMinFR": math.inf,
            "fair_best_WMinFR": math.inf,
            "default_WWD": -math.inf,
            "fair_best_WWD": -math.inf,
            "nominal_gap": None,
            "cost_worst_scenario": None,
            "cost_worst_cost": -math.inf,
            "cost_worst_gap": None,
            "fairness_worst_scenario": None,
            "fairness_worst_gap": -math.inf,
            "scenario_count": 0,
        }
    for entry in sorted_entries:
        key = str(entry.get("base_run_key"))
        chunk_index = int(entry.get("chunk_index", -1))
        relative = str(entry.get("relative_path", ""))
        if key not in base_records or not source.exists(relative):
            raise ValueError("Checkpoint references unknown base run or missing file")
        actual_hash = source.member_sha256(relative)
        checkpoint_hashes_ok = checkpoint_hashes_ok and actual_hash == entry.get("sha256")
        checkpoint = source.json_value(relative)
        records = checkpoint.get("scenario_records") if isinstance(checkpoint.get("scenario_records"), list) else []
        start = next_indices[key]
        expected_end = min(expectation.scenario_count, start + expectation.chunk_size)
        if (
            checkpoint.get("schema_version") != 1
            or checkpoint.get("diagnostic_run_key") != manifest.get("diagnostic_run_key")
            or checkpoint.get("base_run_key") != key
            or checkpoint.get("chunk_index") != chunk_index
            or checkpoint.get("scenario_start") != start
            or checkpoint.get("scenario_end_exclusive") != expected_end
            or checkpoint.get("success") is not True
            or entry.get("success") is not True
            or len(records) != expected_end - start
        ):
            scenario_integrity_ok = False
        instance = instances[key]
        state = states[key]
        for offset, record in enumerate(records):
            index_value = start + offset
            try:
                active = next(expected_iterators[key])
            except StopIteration as exc:
                raise ValueError("Too many scenarios in checkpoint") from exc
            pattern = _expected_pattern(instance, active)
            demand = _expected_demand(instance, active)
            suffix = "base" if not active else "_".join(f"r{r}j{j}" for r, j in active)
            expected_name = f"g{len(active)}_{suffix}"
            expected_key = _scenario_key(index_value, pattern)
            scenario_ok = (
                record.get("scenario_index") == index_value
                and record.get("scenario_id") == expected_name
                and record.get("scenario_key") == expected_key
                and record.get("gamma_usage") == len(active)
                and record.get("deviation_pattern") == pattern
                and record.get("deviation_pattern_sha256") == _pattern_sha256(pattern)
                and _matrix_close(record.get("demand"), demand)
                and record.get("valid") is True
                and not record.get("invalid_reason")
            )
            scenario_integrity_ok = scenario_integrity_ok and scenario_ok
            default = record.get("default") if isinstance(record.get("default"), dict) else {}
            fair = record.get("fair_best") if isinstance(record.get("fair_best"), dict) else {}
            default_cost = float(default.get("objective", math.nan))
            fair_cost = float(fair.get("objective", math.nan))
            expected_cost_tolerance = absolute_tolerance + relative_tolerance * max(1.0, abs(default_cost))
            recourse_ok = (
                default.get("status") == "optimal"
                and fair.get("status") == "optimal"
                and math.isfinite(default_cost)
                and math.isfinite(fair_cost)
                and _close(default.get("original_optimal_cost"), default_cost)
                and _close(fair.get("original_optimal_cost"), default_cost)
                and _close(default.get("cost_tolerance"), expected_cost_tolerance)
                and _close(fair.get("cost_tolerance"), expected_cost_tolerance)
                and fair_cost <= default_cost + expected_cost_tolerance + ABS_TOL
            )
            recourse_integrity_ok = recourse_integrity_ok and recourse_ok
            for variant_name, allocation in (("default", default), ("fair_best", fair)):
                computed = _recompute_metrics(
                    demand,
                    allocation.get("shortage_values", []),
                    metric_tolerance=metric_tolerance,
                )
                stored = allocation.get("metrics") if isinstance(allocation.get("metrics"), dict) else {}
                matches = _metric_matches(stored, computed)
                metric_integrity_ok = metric_integrity_ok and matches
                gap = float(computed["fill_rate_gap"])
                minimum = float(computed["minimum_fill_rate"])
                worst_deviation = float(computed["worst_region_deviation"])
                state[f"{variant_name}_WGap"] = max(state[f"{variant_name}_WGap"], gap)
                state[f"{variant_name}_WMinFR"] = min(state[f"{variant_name}_WMinFR"], minimum)
                state[f"{variant_name}_WWD"] = max(state[f"{variant_name}_WWD"], worst_deviation)
                if variant_name == "fair_best" and gap > state["fairness_worst_gap"]:
                    state["fairness_worst_gap"] = gap
                    state["fairness_worst_scenario"] = expected_key
            if float(fair["metrics"]["fill_rate_gap"]) > float(default["metrics"]["fill_rate_gap"]) + metric_tolerance:
                recourse_integrity_ok = False
            if not active:
                state["nominal_gap"] = float(fair["metrics"]["fill_rate_gap"])
            if default_cost > state["cost_worst_cost"]:
                state["cost_worst_cost"] = default_cost
                state["cost_worst_scenario"] = expected_key
                state["cost_worst_gap"] = float(fair["metrics"]["fill_rate_gap"])
            state["scenario_count"] += 1
        next_indices[key] = expected_end
    coverage_ok = all(
        next_indices[key] == expectation.scenario_count
        and states[key]["scenario_count"] == expectation.scenario_count
        for key in base_records
    )
    checks.extend(
        [
            _check(f"{expectation.label}_checkpoint_hashes", checkpoint_hashes_ok),
            _check(f"{expectation.label}_exact_scenario_coverage", coverage_ok),
            _check(f"{expectation.label}_deviation_patterns_reconstructable", scenario_integrity_ok),
            _check(f"{expectation.label}_recourse_status_and_cost_tolerance", recourse_integrity_ok),
            _check(f"{expectation.label}_fairness_metrics_independently_recomputed", metric_integrity_ok),
        ]
    )
    if not all((checkpoint_hashes_ok, coverage_ok, scenario_integrity_ok, recourse_integrity_ok, metric_integrity_ok)):
        raise ValueError("Checkpoint scenario, recourse, or metric audit failed")
    return states, sorted_entries


def _expected_region_row(
    *,
    expectation: ScaleExpectation,
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    meta: Mapping[str, Any],
    record: Mapping[str, Any],
    variant: str,
    region: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    is_nominal = int(record["gamma_usage"]) == 0
    is_cost_worst = record["scenario_key"] == state["cost_worst_scenario"]
    is_fairness_worst = record["scenario_key"] == state["fairness_worst_scenario"]
    scenario_type = "nominal" if is_nominal else "budget_extreme"
    if is_cost_worst:
        scenario_type += "|cost_worst"
    if is_fairness_worst:
        scenario_type += "|fairness_worst"
    allocation = record[variant]
    config = resolved["fairness_diagnostic"]
    row = {
        "diagnostic_run_key": manifest["diagnostic_run_key"],
        "base_run_key": meta["row"]["run_key"],
        "instance_name": meta["instance_name"],
        "experiment_name": expectation.experiment_name,
        "scale": expectation.instance_size,
        "method": EXPECTED_METHOD,
        "seed": meta["seed"],
        "base_git_commit": expectation.run_commit,
        "base_config_sha256": meta["result"]["config_sha256"],
        "resolved_config_sha256": manifest["resolved_config_sha256"],
        "scenario_key": record["scenario_key"],
        "scenario_index": record["scenario_index"],
        "scenario_type": scenario_type,
        "is_nominal": is_nominal,
        "is_cost_worst": is_cost_worst,
        "is_fairness_worst": is_fairness_worst,
        "deviation_pattern": record["deviation_pattern"],
        "deviation_pattern_sha256": record["deviation_pattern_sha256"],
        "region_id": region["region"],
        "recourse_variant": variant,
        "default_recourse_status": record["default"]["status"],
        "fair_best_recourse_status": record["fair_best"]["status"],
        "default_recourse_cost": record["default"]["objective"],
        "fair_best_recourse_cost": record["fair_best"]["objective"],
        "cost_absolute_tolerance": config["cost_absolute_tolerance"],
        "cost_relative_tolerance": config["cost_relative_tolerance"],
        "invalid_reason": record.get("invalid_reason", ""),
        "instance_size": expectation.instance_size,
        "scenario_id": record["scenario_id"],
        "scenario_kind": scenario_type,
        "region": region["region"],
        **region,
        "recourse_policy": variant,
        "original_recourse_cost": record["default"]["objective"],
        "evaluated_recourse_cost": allocation["objective"],
        "cost_tolerance": allocation["cost_tolerance"],
        "scenario_gamma_usage": record["gamma_usage"],
        "first_stage_x_sha256": meta["first_stage_x_sha256"],
    }
    return row


def _verify_final_aggregation(
    source: ResultSource,
    expectation: ScaleExpectation,
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    base_records: Mapping[str, Mapping[str, Any]],
    states: Mapping[str, Mapping[str, Any]],
    entries: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    ordered_entries = sorted(
        entries,
        key=lambda item: (
            int(base_records[str(item["base_run_key"])]["seed"]),
            int(item["chunk_index"]),
        ),
    )
    row_count = 0
    exact = True
    unique = True
    previous_key: tuple[str, ...] | None = None
    with source.open_text("region_scenario_metrics.csv", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or [])
        required_trace = {
            "diagnostic_run_key",
            "base_run_key",
            "instance_name",
            "experiment_name",
            "scale",
            "method",
            "seed",
            "base_git_commit",
            "base_config_sha256",
            "resolved_config_sha256",
            "scenario_key",
            "scenario_index",
            "scenario_type",
            "deviation_pattern",
            "deviation_pattern_sha256",
            "region_id",
            "recourse_variant",
            "default_recourse_status",
            "fair_best_recourse_status",
            "default_recourse_cost",
            "fair_best_recourse_cost",
            "cost_absolute_tolerance",
            "cost_relative_tolerance",
            "regional_demand",
            "regional_shortage",
            "fill_rate",
            "invalid_reason",
        }
        trace_ok = required_trace <= set(header) and not any("distance" in field.lower() for field in header)
        for entry in ordered_entries:
            key = str(entry["base_run_key"])
            checkpoint = source.json_value(str(entry["relative_path"]))
            meta = base_records[key]
            state = states[key]
            for record in checkpoint["scenario_records"]:
                for variant in ("default", "fair_best"):
                    for region in record[variant]["metrics"]["regions"]:
                        actual = next(reader, None)
                        if actual is None:
                            exact = False
                            break
                        expected = _expected_region_row(
                            expectation=expectation,
                            manifest=manifest,
                            resolved=resolved,
                            meta=meta,
                            record=record,
                            variant=variant,
                            region=region,
                            state=state,
                        )
                        for field in header:
                            if actual.get(field, "") != _value_text(expected.get(field)):
                                exact = False
                                break
                        primary = (
                            actual["diagnostic_run_key"],
                            actual["base_run_key"],
                            actual["scenario_key"],
                            actual["recourse_variant"],
                            actual["region_id"],
                        )
                        unique = unique and primary != previous_key
                        previous_key = primary
                        row_count += 1
                    if not exact:
                        break
                if not exact:
                    break
            if not exact:
                break
        no_extra = next(reader, None) is None
    expected_rows = expectation.scenario_count * len(expectation.seeds) * 2 * expectation.num_regions
    audit_log = source.json_value("audit_log.json")
    checks.extend(
        [
            _check(f"{expectation.label}_region_output_trace_schema", trace_ok, list(header)),
            _check(f"{expectation.label}_region_primary_keys_unique", unique),
            _check(
                f"{expectation.label}_region_row_count",
                row_count == expected_rows and int(audit_log.get("region_row_count", -1)) == expected_rows,
                {"actual": row_count, "expected": expected_rows},
            ),
            _check(f"{expectation.label}_checkpoint_aggregation_exact", exact and no_extra),
            _check(
                f"{expectation.label}_transport_cost_not_physical_distance",
                trace_ok and "allocated_unit_transport_cost" in header,
            ),
        ]
    )
    if not (trace_ok and unique and row_count == expected_rows and exact and no_extra):
        raise ValueError("Final regional CSV does not exactly match verified checkpoints")


def _scale_report(
    source: ResultSource,
    expectation: ScaleExpectation,
    base_records: Mapping[str, Mapping[str, Any]],
    states: Mapping[str, Mapping[str, Any]],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_rows = _csv_rows(source, "instance_summary.csv")
    if len(summary_rows) != len(expectation.seeds):
        checks.append(_check(f"{expectation.label}_instance_summary_count", False, len(summary_rows)))
        raise ValueError("Instance summary count mismatch")
    by_key = {row["base_run_key"]: row for row in summary_rows}
    summary_exact = len(by_key) == len(summary_rows) and set(by_key) == set(base_records)
    ordered_states = [states[key] for key in sorted(states, key=lambda item: base_records[item]["seed"])]
    for key, state in states.items():
        row = by_key.get(key, {})
        meta = base_records[key]
        exact_fields = (
            row.get("diagnostic_run_key") == source.json_value("diagnostic_run_manifest.json").get("diagnostic_run_key")
            and row.get("base_run_key") == key
            and int(row.get("seed", -1)) == meta["seed"]
            and row.get("size") == expectation.instance_size
            and row.get("method") == EXPECTED_METHOD
            and row.get("base_git_commit") == expectation.run_commit
            and row.get("first_stage_x_sha256") == meta["first_stage_x_sha256"]
            and int(row.get("scenario_count", -1)) == expectation.scenario_count
            and row.get("cost_worst_scenario") == state["cost_worst_scenario"]
            and row.get("fairness_worst_scenario") == state["fairness_worst_scenario"]
        )
        numeric_fields = (
            "default_WGap",
            "fair_best_WGap",
            "default_WMinFR",
            "fair_best_WMinFR",
            "default_WWD",
            "fair_best_WWD",
            "nominal_gap",
            "cost_worst_gap",
        )
        exact_fields = exact_fields and all(_close(row.get(field), state[field]) for field in numeric_fields)
        exact_fields = exact_fields and _close(
            row.get("default_minus_fair_best_WGap"),
            state["default_WGap"] - state["fair_best_WGap"],
        )
        summary_exact = summary_exact and exact_fields
    default_values = [float(state["default_WGap"]) for state in ordered_states]
    fair_values = [float(state["fair_best_WGap"]) for state in ordered_states]
    reductions = [before - after for before, after in zip(default_values, fair_values)]
    scale_signal = (
        "structural_fairness_gap"
        if sum(value >= MATERIAL_GAP for value in fair_values) >= 4
        and statistics.median(fair_values) >= STRUCTURAL_MEDIAN
        else "recourse_degeneracy_only"
        if sum(value >= MATERIAL_GAP for value in default_values) >= 4
        and statistics.median(default_values) >= STRUCTURAL_MEDIAN
        and sum(value >= DEGENERACY_REDUCTION for value in reductions) >= 4
        else "no_material_fairness_gap"
        if sum(value >= MATERIAL_GAP for value in fair_values) <= 1
        and statistics.median(fair_values) < NO_MATERIAL_MEDIAN
        and not (
            sum(value >= MATERIAL_GAP for value in default_values) >= 4
            and statistics.median(default_values) >= STRUCTURAL_MEDIAN
        )
        else "fairness_diagnostic_inconclusive"
    )
    report = {
        "scale": expectation.label,
        "instance_count": len(ordered_states),
        "scenario_count_per_instance": expectation.scenario_count,
        "instance_scenario_count": len(ordered_states) * expectation.scenario_count,
        "default_WGap_by_seed": {str(state["seed"]): state["default_WGap"] for state in ordered_states},
        "fair_best_WGap_by_seed": {str(state["seed"]): state["fair_best_WGap"] for state in ordered_states},
        "default_count_at_least_0_10": sum(value >= MATERIAL_GAP for value in default_values),
        "default_median_WGap": statistics.median(default_values),
        "fair_best_count_at_least_0_10": sum(value >= MATERIAL_GAP for value in fair_values),
        "fair_best_median_WGap": statistics.median(fair_values),
        "default_to_fair_best_reduction_count_at_least_0_05": sum(
            value >= DEGENERACY_REDUCTION for value in reductions
        ),
        "scale_signal": scale_signal,
        "default_WGap": default_values,
        "fair_best_WGap": fair_values,
    }
    diagnosis = source.json_value("diagnosis.json")
    diagnosis_matches = all(
        diagnosis.get(field) == report[field]
        for field in (
            "default_count_at_least_0_10",
            "default_median_WGap",
            "fair_best_count_at_least_0_10",
            "fair_best_median_WGap",
            "default_to_fair_best_reduction_count_at_least_0_05",
            "scale_signal",
        )
    )
    checks.extend(
        [
            _check(f"{expectation.label}_instance_summary_recomputed", summary_exact),
            _check(f"{expectation.label}_diagnosis_recomputed", diagnosis_matches, report),
        ]
    )
    if not summary_exact or not diagnosis_matches:
        raise ValueError("Instance summary or single-scale diagnosis mismatch")
    return report


def _audit_scale(
    input_path: str | Path,
    expectation: ScaleExpectation,
    *,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    report: dict[str, Any] = {"scale": expectation.label}
    try:
        with ResultSource(input_path) as source:
            manifest, resolved = _validate_manifest_and_hashes(
                source, expectation, repo_root, checks
            )
            _rows, base_records, instances = _load_base_runs(
                source, expectation, manifest, checks
            )
            states, entries = _scan_checkpoints(
                source,
                expectation,
                manifest,
                resolved,
                base_records,
                instances,
                checks,
            )
            _verify_final_aggregation(
                source,
                expectation,
                manifest,
                resolved,
                base_records,
                states,
                entries,
                checks,
            )
            report = _scale_report(source, expectation, base_records, states, checks)
    except Exception as exc:  # noqa: BLE001 - every malformed input must become invalid evidence.
        checks.append(
            _check(
                f"{expectation.label}_read_only_audit_completed",
                False,
                f"{type(exc).__name__}: {exc}",
            )
        )
        report["audit_error"] = f"{type(exc).__name__}: {exc}"
    else:
        checks.append(_check(f"{expectation.label}_read_only_audit_completed", True))
    return checks, report


def audit_regional_fairness_results(
    medium_input: str | Path,
    large_input: str | Path,
    *,
    repo_root: str | Path | None = None,
    expectations: Mapping[str, ScaleExpectation] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    frozen = dict(expectations or FORMAL_EXPECTATIONS)
    checks: list[dict[str, Any]] = []
    scale_reports: dict[str, dict[str, Any]] = {}
    for scale, path in (("medium_large", medium_input), ("large", large_input)):
        scale_checks, scale_report = _audit_scale(path, frozen[scale], repo_root=root)
        checks.extend(scale_checks)
        scale_reports[scale] = scale_report
    predecision_valid = all(item["passed"] for item in checks)
    complete_values = all(
        len(scale_reports.get(scale, {}).get("fair_best_WGap", [])) == len(frozen[scale].seeds)
        and len(scale_reports.get(scale, {}).get("default_WGap", [])) == len(frozen[scale].seeds)
        for scale in ("medium_large", "large")
    )
    summaries = {
        scale: {
            "default_WGap": list(scale_reports.get(scale, {}).get("default_WGap", [])),
            "fair_best_WGap": list(scale_reports.get(scale, {}).get("fair_best_WGap", [])),
        }
        for scale in ("medium_large", "large")
    }
    decision = classify_joint_fairness(
        summaries,
        correctness_valid=predecision_valid and complete_values,
    ) if complete_values else classify_joint_fairness({}, correctness_valid=False)
    pooled_fair = [value for scale in summaries.values() for value in scale["fair_best_WGap"]]
    pooled_default = [value for scale in summaries.values() for value in scale["default_WGap"]]
    gates = {
        "medium_large_fair_best_at_least_4_of_10_ge_0_10": (
            len(summaries["medium_large"]["fair_best_WGap"]) == 10
            and sum(value >= MATERIAL_GAP for value in summaries["medium_large"]["fair_best_WGap"]) >= 4
        ),
        "medium_large_fair_best_median_at_least_0_05": (
            len(summaries["medium_large"]["fair_best_WGap"]) == 10
            and statistics.median(summaries["medium_large"]["fair_best_WGap"]) >= STRUCTURAL_MEDIAN
        ),
        "large_fair_best_at_least_4_of_10_ge_0_10": (
            len(summaries["large"]["fair_best_WGap"]) == 10
            and sum(value >= MATERIAL_GAP for value in summaries["large"]["fair_best_WGap"]) >= 4
        ),
        "large_fair_best_median_at_least_0_05": (
            len(summaries["large"]["fair_best_WGap"]) == 10
            and statistics.median(summaries["large"]["fair_best_WGap"]) >= STRUCTURAL_MEDIAN
        ),
        "pooled_fair_best_at_least_8_of_20_ge_0_10": (
            len(pooled_fair) == 20 and sum(value >= MATERIAL_GAP for value in pooled_fair) >= 8
        ),
        "pooled_fair_best_median_at_least_0_05": (
            len(pooled_fair) == 20 and statistics.median(pooled_fair) >= STRUCTURAL_MEDIAN
        ),
    }
    checks.extend(
        [
            _check(
                "cross_scale_seed_labels_match_frozen_expectations",
                set(frozen["medium_large"].seeds) == set(frozen["large"].seeds),
            ),
            _check("no_reserved_fairness_model_seed_used", all(not (set(frozen[scale].seeds) & set(range(120, 160))) for scale in frozen)),
            _check(
                "joint_decision_applies_frozen_thresholds",
                decision["decision"]
                in {
                    "structural_fairness_gap",
                    "recourse_degeneracy_only",
                    "no_material_fairness_gap",
                    "fairness_diagnostic_inconclusive",
                    "fairness_diagnostic_invalid",
                },
                decision,
            ),
        ]
    )
    failed = [item["check"] for item in checks if not item["passed"]]
    all_passed = not failed
    if not all_passed:
        decision = classify_joint_fairness({}, correctness_valid=False)
    return {
        "audit_name": "regional_fairness_diagnostic_results_joint_read_only_audit",
        "created_at": utc_now_iso(),
        "read_only": True,
        "formal_run_commit": FORMAL_RUN_COMMIT,
        "archive_sha256": dict(EXPECTED_ARCHIVE_SHA256),
        "input_config_sha256": dict(EXPECTED_CONFIG_SHA256),
        "protocol_document_sha256": EXPECTED_PROTOCOL_SHA256,
        "candidate_config_sha256": EXPECTED_CANDIDATE_SHA256,
        "all_required_checks_passed": all_passed,
        "required_check_count": len(checks),
        "passed_check_count": sum(item["passed"] for item in checks),
        "failed_checks": failed,
        "scales": scale_reports,
        "pooled": {
            "instance_count": len(pooled_fair),
            "fair_best_count_at_least_0_10": sum(value >= MATERIAL_GAP for value in pooled_fair),
            "fair_best_median_WGap": statistics.median(pooled_fair) if pooled_fair else None,
            "default_count_at_least_0_10": sum(value >= MATERIAL_GAP for value in pooled_default),
            "default_median_WGap": statistics.median(pooled_default) if pooled_default else None,
        },
        "decision_gates": gates,
        **decision,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent read-only joint audit of regional fairness diagnostic evidence."
    )
    parser.add_argument("--medium-input", required=True)
    parser.add_argument("--large-input", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    report = audit_regional_fairness_results(args.medium_input, args.large_input)
    if args.output_json:
        atomic_write_json(args.output_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["decision"] == "fairness_diagnostic_invalid":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
