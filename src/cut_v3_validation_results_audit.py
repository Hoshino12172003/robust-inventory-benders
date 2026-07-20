from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import yaml

from .experiment_protocol import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    utc_now_iso,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_COMMIT = "648556b1956008e93bfc8ac0459cdc3260ab93be"
VALIDATION_SEEDS = tuple(range(80, 90))
SEALED_FINAL_SEEDS = frozenset(range(90, 110))
V1_VARIANT = "proposed_joint_rho025_050"
CORE_VARIANT = "joint_v1_core_point_strengthened"
EXPECTED_VARIANTS = (V1_VARIANT, CORE_VARIANT)
EXPECTED_ARCHIVE_SHA256 = {
    "medium_large": "2c54f54e8fc0eb78326228bb1c3069ebdae6530ddab7e332e8bb24f5cc3d5bcd",
    "large": "91d459de451a99245bbe672d06ab4a0ab00787e576a7b1b5cb7e4973b926c780",
}
EXPECTED_INPUT_CONFIG_SHA256 = {
    "medium_large": "eb7070b8045cfd3fc57b4f7dc906059f8c9ca60d9c0ad58b75cd6e8e98d41007",
    "large": "44106f8a1f12d4caca961439ca4b5eebf8ca263afac567512ce541f4e80ace27",
}
EXPECTED_CANDIDATE_SHA256 = (
    "7e8aaf39de8c100b4ce9b46256a074fbd324b07ddc347d256494ed070d4e0eb6"
)
EXPECTED_SCALE = {
    "medium_large": {
        "config": "cut_strengthened_joint_v3_validation_medium_large.yaml",
        "experiment": "cut_strengthened_joint_v3_validation_medium_large",
        "time_limit": 600.0,
        "max_iterations": 10000,
        "instance": (6, 6, 10),
    },
    "large": {
        "config": "cut_strengthened_joint_v3_validation_large.yaml",
        "experiment": "cut_strengthened_joint_v3_validation_large",
        "time_limit": 1800.0,
        "max_iterations": 20000,
        "instance": (8, 8, 12),
    },
}

ABS_TOL = 1.0e-7
REL_TOL = 1.0e-8


def _check(name: str, passed: bool, details: Any = "") -> dict[str, Any]:
    return {"check": name, "required": True, "passed": bool(passed), "details": details}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return 2**63 - 1


def _close(left: Any, right: Any, *, abs_tol: float = ABS_TOL) -> bool:
    a, b = _float(left), _float(right)
    return math.isfinite(a) and math.isfinite(b) and math.isclose(
        a, b, rel_tol=REL_TOL, abs_tol=abs_tol
    )


def _nondecreasing(values: Iterable[float]) -> bool:
    seq = list(values)
    return all(b + ABS_TOL >= a for a, b in zip(seq, seq[1:]))


def _nonincreasing(values: Iterable[float]) -> bool:
    seq = list(values)
    return all(b <= a + ABS_TOL for a, b in zip(seq, seq[1:]))


class _ResultSource:
    """Read a result tree without extracting or modifying its source."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._zip: zipfile.ZipFile | None = None
        if self.path.is_file():
            self._zip = zipfile.ZipFile(self.path, "r")
            self.names = tuple(
                name for name in self._zip.namelist() if not name.endswith("/")
            )
        elif self.path.is_dir():
            self.names = tuple(
                item.relative_to(self.path).as_posix()
                for item in self.path.rglob("*")
                if item.is_file()
            )
        else:
            raise FileNotFoundError(self.path)
        results = [name for name in self.names if name.endswith("/results.csv") or name == "results.csv"]
        if len(results) != 1:
            raise ValueError(f"Expected exactly one results.csv in {self.path}, found {len(results)}")
        self.root = str(PurePosixPath(results[0]).parent)
        if self.root == ".":
            self.root = ""

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def __enter__(self) -> "_ResultSource":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def is_archive(self) -> bool:
        return self._zip is not None

    def _name(self, relative: str) -> str:
        return f"{self.root}/{relative}" if self.root else relative

    def exists(self, relative: str) -> bool:
        return self._name(relative) in self.names

    def read_bytes(self, relative: str) -> bytes:
        name = self._name(relative)
        if self._zip is not None:
            return self._zip.read(name)
        return (self.path / relative).read_bytes()

    def read_text(self, relative: str) -> str:
        return self.read_bytes(relative).decode("utf-8-sig")

    def csv_rows(self, relative: str) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(self.read_text(relative))))

    def json_value(self, relative: str) -> dict[str, Any]:
        return json.loads(self.read_text(relative))

    def yaml_value(self, relative: str) -> dict[str, Any]:
        value = yaml.safe_load(self.read_bytes(relative))
        return value if isinstance(value, dict) else {}

    def crc_error(self) -> str | None:
        return self._zip.testzip() if self._zip is not None else None

    def sha256(self) -> str | None:
        return file_sha256(self.path).lower() if self.path.is_file() else None


def _expected_run_keys(scale: str) -> set[tuple[int, str]]:
    return {(seed, variant) for seed in VALIDATION_SEEDS for variant in EXPECTED_VARIANTS}


def _resolved_config_ok(
    config: Mapping[str, Any], scale: str, seed: int, variant: str
) -> tuple[bool, list[str]]:
    expected = EXPECTED_SCALE[scale]
    algorithm = config.get("algorithm", {})
    robust = config.get("robust", {})
    benders = config.get("benders", {})
    failures: list[str] = []

    exact = {
        "seed": (config.get("seed"), seed),
        "subproblem_mode": (algorithm.get("subproblem_mode"), "robust_dual_milp"),
        "precision_policy": (algorithm.get("precision_policy"), "joint_error_budget"),
        "gamma_target": (robust.get("gamma_target"), 2),
        "gamma_schedule": (robust.get("gamma_schedule"), [2]),
        "max_iterations": (benders.get("max_iterations"), expected["max_iterations"]),
        "tol": (benders.get("tol"), 1.0e-4),
        "max_cuts_per_iteration": (algorithm.get("max_cuts_per_iteration"), 1),
        "cut_selection_enabled": (algorithm.get("cut_selection_enabled"), False),
        "adaptive_secondary_cut_selection_enabled": (
            algorithm.get("adaptive_secondary_cut_selection_enabled"),
            False,
        ),
        "adaptive_secondary_generation_enabled": (
            algorithm.get("adaptive_secondary_generation_enabled"),
            False,
        ),
        "gamma_continuation_disabled": (config.get("gamma_continuation_enabled", False), False),
        "adaptive_gap_enabled": (config.get("adaptive_gap_enabled"), False),
        "adaptive_subproblem_gap_enabled": (
            algorithm.get("adaptive_subproblem_gap_enabled"),
            False,
        ),
        "monotone_precision_tightening": (
            algorithm.get("monotone_precision_tightening"),
            True,
        ),
        "final_certification_enabled": (
            algorithm.get("final_certification_enabled"),
            True,
        ),
        "cut_strengthening_policy": (
            algorithm.get("cut_strengthening_policy"),
            "none" if variant == V1_VARIANT else "core_point",
        ),
    }
    for name, (actual, wanted) in exact.items():
        if actual != wanted:
            failures.append(f"{name}={actual!r}, expected {wanted!r}")

    numeric = {
        "time_limit": (benders.get("time_limit"), expected["time_limit"]),
        "master_gap_max": (algorithm.get("master_gap_max"), 0.02),
        "master_gap_min": (algorithm.get("master_gap_min"), 0.0001),
        "subproblem_gap_max": (algorithm.get("subproblem_gap_max"), 0.05),
        "subproblem_gap_min": (algorithm.get("subproblem_gap_min"), 0.0001),
        "master_error_budget_ratio": (algorithm.get("master_error_budget_ratio"), 0.25),
        "subproblem_error_budget_ratio": (algorithm.get("subproblem_error_budget_ratio"), 0.50),
        "core_point_update_weight": (algorithm.get("core_point_update_weight"), 0.50),
        "core_point_min_distance": (algorithm.get("core_point_min_distance"), 1.0e-9),
        "core_point_stage1_time_limit": (algorithm.get("core_point_stage1_time_limit"), 2.0),
        "core_point_stage2_time_limit": (algorithm.get("core_point_stage2_time_limit"), 2.0),
        "core_point_min_remaining_time": (algorithm.get("core_point_min_remaining_time"), 10.0),
        "core_point_min_global_gap": (algorithm.get("core_point_min_global_gap"), 5.0e-4),
        "core_point_current_abs_tol": (algorithm.get("core_point_current_abs_tol"), 1.0e-7),
        "core_point_current_rel_tol": (algorithm.get("core_point_current_rel_tol"), 1.0e-8),
        "core_point_min_normalized_improvement": (
            algorithm.get("core_point_min_normalized_improvement"),
            1.0e-7,
        ),
    }
    for name, (actual, wanted) in numeric.items():
        if not _close(actual, wanted):
            failures.append(f"{name}={actual!r}, expected {wanted!r}")
    instance = config.get("instance", {})
    actual_dimensions = (
        instance.get("num_warehouses"),
        instance.get("num_products"),
        instance.get("num_regions"),
    )
    if actual_dimensions != expected["instance"]:
        failures.append(
            f"instance dimensions={actual_dimensions!r}, expected {expected['instance']!r}"
        )
    return not failures, failures


@dataclass
class _ScaleAudit:
    checks: list[dict[str, Any]]
    rows: list[dict[str, str]]
    summary: dict[str, Any]


def _audit_scale(
    scale: str,
    source_path: str | Path,
    expected_archive_sha256: str | None,
) -> _ScaleAudit:
    checks: list[dict[str, Any]] = []
    expected = EXPECTED_SCALE[scale]
    with _ResultSource(source_path) as source:
        actual_sha = source.sha256()
        checks.append(
            _check(
                f"{scale}_archive_sha256",
                expected_archive_sha256 is None
                or actual_sha is None
                or actual_sha == expected_archive_sha256.lower(),
                actual_sha or "directory input (archive hash not applicable)",
            )
        )
        checks.append(_check(f"{scale}_archive_crc", source.crc_error() is None))

        required_top = ("results.csv", "run_manifest.json", "resolved_config.yaml")
        checks.append(
            _check(
                f"{scale}_top_level_files_complete",
                all(source.exists(name) for name in required_top),
            )
        )
        rows = source.csv_rows("results.csv") if source.exists("results.csv") else []
        manifest = source.json_value("run_manifest.json") if source.exists("run_manifest.json") else {}
        top_config = source.yaml_value("resolved_config.yaml") if source.exists("resolved_config.yaml") else {}

        actual_pairs = [(_int(row.get("seed")), str(row.get("variant_name"))) for row in rows]
        pair_set = set(actual_pairs)
        checks.extend(
            [
                _check(f"{scale}_exactly_20_rows", len(rows) == 20, len(rows)),
                _check(
                    f"{scale}_exact_seeds_80_89",
                    {seed for seed, _ in pair_set} == set(VALIDATION_SEEDS),
                    sorted({seed for seed, _ in pair_set}),
                ),
                _check(
                    f"{scale}_sealed_seeds_90_109_absent",
                    {seed for seed, _ in pair_set}.isdisjoint(SEALED_FINAL_SEEDS),
                ),
                _check(
                    f"{scale}_only_v1_and_core",
                    {variant for _, variant in pair_set} == set(EXPECTED_VARIANTS),
                    sorted({variant for _, variant in pair_set}),
                ),
                _check(
                    f"{scale}_no_missing_or_duplicate_pairs",
                    pair_set == _expected_run_keys(scale) and len(actual_pairs) == len(pair_set),
                ),
            ]
        )

        expected_manifest = {
            "expected_run_count": 20,
            "completed_run_count": 20,
            "solved_run_count": 20 if scale == "medium_large" else 18,
            "failed_run_count": 0,
            "skipped_run_count": 0,
            "remaining_run_count": 0,
            "git_commit": VALIDATION_COMMIT,
        }
        manifest_failures = {
            key: manifest.get(key)
            for key, value in expected_manifest.items()
            if manifest.get(key) != value
        }
        checks.append(
            _check(f"{scale}_manifest_complete_and_consistent", not manifest_failures, manifest_failures)
        )

        top_ok = (
            top_config.get("experiment_name") == expected["experiment"]
            and top_config.get("random_seeds") == list(VALIDATION_SEEDS)
            and top_config.get("instance_sizes") == [scale]
            and top_config.get("variants") == list(EXPECTED_VARIANTS)
            and _close(top_config.get("time_limit"), expected["time_limit"])
            and top_config.get("max_iterations") == expected["max_iterations"]
            and _close(top_config.get("tol"), 1.0e-4)
            and top_config.get("gamma_target") == 2
            and top_config.get("gamma_schedule") == [2]
            and top_config.get("formal_inference_allowed") is False
            and top_config.get("protocol_phase") == "validation"
            and str(top_config.get("candidate_config_sha256", "")).lower()
            == EXPECTED_CANDIDATE_SHA256
        )
        checks.append(_check(f"{scale}_top_resolved_protocol_identity", top_ok))

        run_files_ok = True
        resolved_ok = True
        result_identity_ok = True
        status_ok = True
        par2_ok = True
        valid_bounds_ok = True
        monotone_ok = True
        gap_ok = True
        ub_source_ok = True
        core_ok = True
        fallback_ok = True
        no_pseudo_cut_ok = True
        certification_ok = True
        max_one_cut_ok = True
        final_log_ok = True
        errors_empty = True
        secondary_zero = True
        detail_errors: list[str] = []
        logs_by_pair: dict[tuple[int, str], list[dict[str, str]]] = {}

        for row in rows:
            seed = _int(row.get("seed"))
            variant = str(row.get("variant_name"))
            run_key = str(row.get("run_key"))
            pair = (seed, variant)
            relative = f"runs/{run_key}"
            required_run = (
                f"{relative}/run.json",
                f"{relative}/status.json",
                f"{relative}/resolved_config.yaml",
                f"{relative}/error.txt",
            )
            if not all(source.exists(name) for name in required_run):
                run_files_ok = False
                detail_errors.append(f"missing run artifact: {run_key}")
                continue

            run_record = source.json_value(f"{relative}/run.json")
            status_record = source.json_value(f"{relative}/status.json")
            resolved = source.yaml_value(f"{relative}/resolved_config.yaml")
            errors_empty &= source.read_text(f"{relative}/error.txt").strip() == ""
            this_resolved, resolution_errors = _resolved_config_ok(resolved, scale, seed, variant)
            resolved_ok &= this_resolved
            detail_errors.extend(f"{run_key}: {item}" for item in resolution_errors)

            log_name = str(row.get("iteration_log_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
            log_relative = f"iteration_logs/{log_name}"
            if not source.exists(log_relative):
                run_files_ok = False
                detail_errors.append(f"missing iteration log: {run_key}")
                continue
            log = source.csv_rows(log_relative)
            logs_by_pair[pair] = log

            result_identity_ok &= (
                row.get("experiment_name") == expected["experiment"]
                and row.get("instance_size") == scale
                and row.get("git_commit") == VALIDATION_COMMIT
                and run_record.get("git_commit") == VALIDATION_COMMIT
                and run_record.get("run_key") == run_key
                and (run_record.get("result") or {}).get("run_key") == run_key
                and status_record.get("run_key") == run_key
            )

            solved = _bool(row.get("solved_to_tolerance"))
            final_gap = _float(row.get("final_gap"))
            tol = _float(resolved.get("benders", {}).get("tol"))
            state_consistent = (
                status_record.get("state") == "complete"
                and _bool(status_record.get("success"))
                and _bool(status_record.get("solved_to_tolerance")) == solved
                and status_record.get("status") == row.get("status")
                and solved == (math.isfinite(final_gap) and final_gap <= tol + ABS_TOL)
                and ((solved and row.get("status") == "optimal") or (not solved and row.get("status") == "time_limit"))
            )
            status_ok &= state_consistent

            runtime = _float(row.get("runtime"))
            limit = _float(row.get("time_limit"))
            expected_par2 = runtime if solved else 2.0 * limit
            par2_ok &= _close(row.get("penalized_runtime_par2"), expected_par2)
            valid_bounds_ok &= (
                _bool(row.get("valid_UB"))
                and _bool(row.get("ub_uses_subproblem_bound"))
                and math.isfinite(_float(row.get("lower_bound")))
                and math.isfinite(_float(row.get("upper_bound")))
                and _float(row.get("lower_bound")) <= _float(row.get("upper_bound")) + ABS_TOL
            )

            if not log:
                run_files_ok = False
                detail_errors.append(f"empty iteration log: {run_key}")
                continue
            lbs = [_float(item.get("LB")) for item in log]
            ubs = [_float(item.get("UB")) for item in log]
            mp_gaps = [_float(item.get("requested_master_mip_gap")) for item in log]
            sp_gaps = [_float(item.get("subproblem_requested_mip_gap")) for item in log]
            monotone_ok &= (
                all(math.isfinite(value) for value in lbs + ubs + mp_gaps + sp_gaps)
                and _nondecreasing(lbs)
                and _nonincreasing(ubs)
                and _nonincreasing(mp_gaps)
                and _nonincreasing(sp_gaps)
            )

            for item in log:
                lb, ub = _float(item.get("LB")), _float(item.get("UB"))
                expected_gap = max(0.0, (ub - lb) / max(1.0, abs(ub)))
                gap_ok &= _close(item.get("global_gap"), expected_gap)
                ub_source_ok &= (
                    _bool(item.get("target_robust_evaluation_used"))
                    and not _bool(item.get("core_point_auxiliary_bound_used_for_UB"))
                    and not _bool(item.get("v3_secondary_bound_used_for_UB"))
                )
                accepted = _bool(item.get("core_point_cut_accepted"))
                attempted = _bool(item.get("core_point_attempted"))
                if accepted:
                    core_ok &= (
                        attempted
                        and item.get("core_point_stage1_status") == "optimal"
                        and item.get("core_point_stage2_status") == "optimal"
                        and _bool(item.get("core_point_dual_feasible"))
                        and _float(item.get("core_point_strengthened_value_at_current"))
                        + ABS_TOL
                        >= _float(item.get("core_point_current_value_floor"))
                    )
                if attempted and not accepted:
                    fallback_ok &= (
                        bool(str(item.get("core_point_cut_fallback_reason", "")).strip())
                        and _close(
                            item.get("cut_rhs_current"),
                            item.get("core_point_original_value_at_current"),
                            abs_tol=1.0e-5,
                        )
                    )
                if not _bool(item.get("subproblem_has_incumbent")):
                    no_pseudo_cut_ok &= not _bool(item.get("cut_added"))
                if _bool(item.get("final_certification_active")):
                    certification_ok &= (
                        not attempted
                        and not _bool(item.get("v3_secondary_attempted"))
                    )
                max_one_cut_ok &= _int(item.get("cuts_added_this_iteration")) <= 1
                secondary_zero &= (
                    not _bool(item.get("v3_secondary_attempted"))
                    and not _bool(item.get("v3_secondary_cut_added"))
                    and not _bool(item.get("v3_secondary_bound_used_for_UB"))
                )

            last = log[-1]
            final_log_ok &= (
                _close(last.get("LB"), row.get("lower_bound"))
                and _close(last.get("UB"), row.get("upper_bound"))
                and _close(last.get("global_gap"), row.get("final_gap"))
                and _int(last.get("iteration")) == _int(row.get("iterations"))
            )
            secondary_zero &= (
                _int(row.get("v3_secondary_solve_count")) == 0
                and _int(row.get("v3_secondary_cut_added_count")) == 0
                and _int(row.get("v3_secondary_cuts_added")) == 0
            )

        checks.extend(
            [
                _check(f"{scale}_run_artifacts_complete", run_files_ok, detail_errors[:20]),
                _check(f"{scale}_resolved_configs_frozen", resolved_ok, detail_errors[:20]),
                _check(f"{scale}_run_identity_and_commit", result_identity_ok),
                _check(f"{scale}_status_and_solved_semantics", status_ok),
                _check(f"{scale}_par2_correct", par2_ok),
                _check(f"{scale}_valid_bounds", valid_bounds_ok),
                _check(f"{scale}_bounds_and_requested_gaps_monotone", monotone_ok),
                _check(f"{scale}_global_gap_formula", gap_ok),
                _check(f"{scale}_only_original_robust_bound_updates_ub", ub_source_ok),
                _check(f"{scale}_accepted_core_cuts_valid", core_ok),
                _check(f"{scale}_core_failure_falls_back_to_original_cut", fallback_ok),
                _check(f"{scale}_no_pseudo_cut_without_incumbent", no_pseudo_cut_ok),
                _check(f"{scale}_certification_disables_core", certification_ok),
                _check(f"{scale}_at_most_one_cut_per_iteration", max_one_cut_ok),
                _check(f"{scale}_final_log_matches_results", final_log_ok),
                _check(f"{scale}_all_error_files_empty", errors_empty),
                _check(f"{scale}_secondary_component_never_used", secondary_zero),
            ]
        )

    by_pair = {(_int(row["seed"]), row["variant_name"]): row for row in rows}
    intervals_overlap = all(
        max(
            _float(by_pair[(seed, V1_VARIANT)]["lower_bound"]),
            _float(by_pair[(seed, CORE_VARIANT)]["lower_bound"]),
        )
        <= min(
            _float(by_pair[(seed, V1_VARIANT)]["upper_bound"]),
            _float(by_pair[(seed, CORE_VARIANT)]["upper_bound"]),
        )
        + 1.0e-4
        for seed in VALIDATION_SEEDS
        if (seed, V1_VARIANT) in by_pair and (seed, CORE_VARIANT) in by_pair
    ) and len(by_pair) == 20
    checks.append(_check(f"{scale}_paired_final_intervals_overlap", intervals_overlap))

    summary: dict[str, Any] = {"scale": scale, "variants": {}}
    if set(by_pair) == _expected_run_keys(scale):
        for variant in EXPECTED_VARIANTS:
            selected = [by_pair[(seed, variant)] for seed in VALIDATION_SEEDS]
            summary["variants"][variant] = {
                "solved_count": sum(_bool(row["solved_to_tolerance"]) for row in selected),
                "run_count": len(selected),
                "solved_rate": statistics.mean(_bool(row["solved_to_tolerance"]) for row in selected),
                "mean_par2": statistics.mean(_float(row["penalized_runtime_par2"]) for row in selected),
                "mean_runtime": statistics.mean(_float(row["runtime"]) for row in selected),
                "mean_iterations": statistics.mean(_float(row["iterations"]) for row in selected),
                "unsolved_seeds": [
                    _int(row["seed"])
                    for row in selected
                    if not _bool(row["solved_to_tolerance"])
                ],
            }
        v1 = summary["variants"][V1_VARIANT]
        core = summary["variants"][CORE_VARIANT]
        core_rows = [by_pair[(seed, CORE_VARIANT)] for seed in VALIDATION_SEEDS]
        attempts = sum(_float(row["core_point_attempt_count"]) for row in core_rows)
        successes = sum(_float(row["core_point_success_count"]) for row in core_rows)
        core_runtime = sum(_float(row["core_point_total_runtime"]) for row in core_rows)
        total_runtime = sum(_float(row["runtime"]) for row in core_rows)
        core_wins = sum(
            _float(by_pair[(seed, CORE_VARIANT)]["penalized_runtime_par2"])
            <= _float(by_pair[(seed, V1_VARIANT)]["penalized_runtime_par2"]) + ABS_TOL
            for seed in VALIDATION_SEEDS
        )
        iteration_wins = sum(
            _float(by_pair[(seed, CORE_VARIANT)]["iterations"])
            < _float(by_pair[(seed, V1_VARIANT)]["iterations"])
            for seed in VALIDATION_SEEDS
        )
        core_rank_total = 0.0
        v1_rank_total = 0.0
        for seed in VALIDATION_SEEDS:
            core_par2 = _float(by_pair[(seed, CORE_VARIANT)]["penalized_runtime_par2"])
            v1_par2 = _float(by_pair[(seed, V1_VARIANT)]["penalized_runtime_par2"])
            if _close(core_par2, v1_par2):
                core_rank_total += 1.5
                v1_rank_total += 1.5
            elif core_par2 < v1_par2:
                core_rank_total += 1.0
                v1_rank_total += 2.0
            else:
                core_rank_total += 2.0
                v1_rank_total += 1.0
        summary.update(
            {
                "par2_reduction_percent": 100.0 * (v1["mean_par2"] - core["mean_par2"]) / v1["mean_par2"],
                "iteration_reduction_percent": 100.0 * (v1["mean_iterations"] - core["mean_iterations"]) / v1["mean_iterations"],
                "paired_par2_core_not_worse": core_wins,
                "paired_iterations_core_lower": iteration_wins,
                "core_mean_rank": core_rank_total / 10.0,
                "v1_mean_rank": v1_rank_total / 10.0,
                "core_attempt_count": attempts,
                "core_success_count": successes,
                "core_acceptance_rate": successes / attempts if attempts else None,
                "core_total_runtime": core_runtime,
                "core_extra_runtime_share": core_runtime / total_runtime if total_runtime else None,
                "secondary_attempt_count": sum(
                    _float(row["v3_secondary_solve_count"]) for row in core_rows
                ),
            }
        )
    return _ScaleAudit(checks=checks, rows=rows, summary=summary)


def _decision(scales: Mapping[str, dict[str, Any]], correctness: bool) -> tuple[str, dict[str, bool]]:
    if not correctness or any("variants" not in scales[name] for name in EXPECTED_SCALE):
        return "validation_fail", {"correctness_gate": False}
    medium = scales["medium_large"]
    large = scales["large"]
    mv1, mcore = medium["variants"][V1_VARIANT], medium["variants"][CORE_VARIANT]
    lv1, lcore = large["variants"][V1_VARIANT], large["variants"][CORE_VARIANT]
    gates = {
        "correctness_gate": True,
        "medium_solved_rate_noninferior": mcore["solved_rate"] >= mv1["solved_rate"],
        "medium_mean_par2_within_103_percent": mcore["mean_par2"] <= 1.03 * mv1["mean_par2"],
        "large_solved_rate_noninferior": lcore["solved_rate"] >= lv1["solved_rate"],
        "large_mean_par2_reduction_at_least_7_5_percent": large["par2_reduction_percent"] >= 7.5,
        "large_mean_iterations_reduction_at_least_15_percent": large["iteration_reduction_percent"] >= 15.0,
        "large_par2_pairs_at_least_6_of_10": large["paired_par2_core_not_worse"] >= 6,
        "large_iteration_pairs_at_least_6_of_10": large["paired_iterations_core_lower"] >= 6,
        "large_mean_paired_rank_better": (
            large["core_mean_rank"] is not None
            and large["v1_mean_rank"] is not None
            and large["core_mean_rank"] < large["v1_mean_rank"]
        ),
    }
    primary = (
        "medium_solved_rate_noninferior",
        "medium_mean_par2_within_103_percent",
        "large_solved_rate_noninferior",
        "large_mean_par2_reduction_at_least_7_5_percent",
    )
    if not all(gates[name] for name in primary):
        return "validation_fail", gates
    return ("validation_pass" if all(gates.values()) else "validation_inconclusive"), gates


def audit_cut_v3_validation_results(
    medium_input: str | Path,
    large_input: str | Path,
    *,
    repo_root: str | Path | None = None,
    expected_archive_sha256: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Audit frozen validation results without extracting or mutating them."""

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    archive_hashes = EXPECTED_ARCHIVE_SHA256 if expected_archive_sha256 is None else expected_archive_sha256
    checks: list[dict[str, Any]] = []

    config_dir = root / "experiments/configs"
    for scale, expected_hash in EXPECTED_INPUT_CONFIG_SHA256.items():
        path = config_dir / str(EXPECTED_SCALE[scale]["config"])
        actual = file_sha256(path).lower() if path.exists() else "missing"
        checks.append(_check(f"{scale}_raw_input_yaml_sha256", actual == expected_hash, actual))
    candidate_path = config_dir / "selected_cut_strengthened_joint_v3_candidate.yaml"
    candidate_hash = file_sha256(candidate_path).lower() if candidate_path.exists() else "missing"
    checks.append(
        _check("selected_candidate_yaml_sha256", candidate_hash == EXPECTED_CANDIDATE_SHA256, candidate_hash)
    )

    scale_reports: dict[str, dict[str, Any]] = {}
    for scale, source in (("medium_large", medium_input), ("large", large_input)):
        try:
            result = _audit_scale(scale, source, archive_hashes.get(scale))
        except Exception as exc:  # noqa: BLE001 - malformed evidence becomes an audit failure.
            checks.append(
                _check(
                    f"{scale}_evidence_readable",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
            )
            scale_reports[scale] = {"scale": scale, "variants": {}}
        else:
            checks.extend(result.checks)
            scale_reports[scale] = result.summary

    failed_before_decision = [item["check"] for item in checks if not item["passed"]]
    decision, gates = _decision(scale_reports, not failed_before_decision)
    for name, passed in gates.items():
        checks.append(_check(f"decision_{name}", passed))

    failed = [item["check"] for item in checks if not item["passed"]]
    return {
        "audit_name": "cut_strengthened_joint_v3_validation_results",
        "created_at": utc_now_iso(),
        "read_only": True,
        "validation_commit": VALIDATION_COMMIT,
        "formal_inference_allowed": False,
        "decision": decision,
        "selected_candidate": CORE_VARIANT,
        "next_authorized_stage": "final_protocol_only" if decision == "validation_pass" else "v1_fallback",
        "all_required_checks_passed": not failed,
        "required_check_count": len(checks),
        "passed_check_count": sum(item["passed"] for item in checks),
        "failed_checks": failed,
        "archive_sha256": dict(archive_hashes),
        "input_config_sha256": dict(EXPECTED_INPUT_CONFIG_SHA256),
        "candidate_config_sha256": EXPECTED_CANDIDATE_SHA256,
        "scales": scale_reports,
        "decision_gates": gates,
        "checks": checks,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Cut-strengthened Joint V3 validation results audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Candidate: `{report['selected_candidate']}`",
        f"- Checks: {report['passed_check_count']}/{report['required_check_count']} passed",
        "- Formal inference allowed: `false`",
        "",
        "| Scale | Method | Solved | Mean PAR-2 | Mean iterations |",
        "|---|---|---:|---:|---:|",
    ]
    for scale in ("medium_large", "large"):
        for variant in EXPECTED_VARIANTS:
            item = report["scales"][scale]["variants"][variant]
            lines.append(
                f"| {scale} | {variant} | {item['solved_count']}/{item['run_count']} | "
                f"{item['mean_par2']:.12f} | {item['mean_iterations']:.1f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit of frozen V3 validation results.")
    parser.add_argument("--medium-input", required=True, help="Medium-large ZIP or extracted directory")
    parser.add_argument("--large-input", required=True, help="Large ZIP or extracted directory")
    parser.add_argument("--output-dir", help="Optional directory for JSON and Markdown audit reports")
    args = parser.parse_args()
    report = audit_cut_v3_validation_results(args.medium_input, args.large_input)
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / "validation_results_audit.json", report)
        atomic_write_text(output / "validation_results_audit.md", _markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_required_checks_passed"] or report["decision"] != "validation_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
