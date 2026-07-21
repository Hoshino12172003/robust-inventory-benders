from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from .cut_strengthened_v3_final_audit import EXPECTED_FINAL_ANALYSIS
from .cut_v3_validation_results_audit import (
    ABS_TOL,
    CORE_VARIANT,
    EXPECTED_VARIANTS,
    V1_VARIANT,
    _bool,
    _check,
    _close,
    _float,
    _int,
    _nondecreasing,
    _nonincreasing,
    _resolved_config_ok,
)
from .experiment_protocol import (
    atomic_write_json,
    atomic_write_text,
    config_sha256,
    file_sha256,
    utc_now_iso,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_COMMIT = "11020383bfaf49b6f538f672089704f1cdf8b860"
FINAL_SEEDS = {
    "medium_large": tuple(range(90, 100)),
    "large": tuple(range(100, 110)),
}
EXPECTED_ARCHIVE_SHA256 = {
    "medium_large": "1388446bc75e44e8e8afc9e7973f011b14b7172aebc8bb400749c6fe7c1d1e7a",
    "large": "6641dcd67f8bfd6fa15f7580459ad31148bff7df64034e16c1a36f98b78985f4",
}
EXPECTED_INPUT_CONFIG_SHA256 = {
    "medium_large": "1d41a19bb47218f2844c2bdfeadf9b044e8776db944c37989ef8c26feb9c0867",
    "large": "60fdf4a9a642485a46e473a25ddb7502198a84eea927d9a60e670b764f8542f3",
}
EXPECTED_CANDIDATE_SHA256 = (
    "7e8aaf39de8c100b4ce9b46256a074fbd324b07ddc347d256494ed070d4e0eb6"
)
EXPECTED_SCALE = {
    "medium_large": {
        "config": "cut_strengthened_joint_v3_final_medium_large.yaml",
        "experiment": "cut_strengthened_joint_v3_final_medium_large",
        "time_limit": 600.0,
        "max_iterations": 10000,
        "skipped": 0,
    },
    "large": {
        "config": "cut_strengthened_joint_v3_final_large.yaml",
        "experiment": "cut_strengthened_joint_v3_final_large",
        "time_limit": 1800.0,
        "max_iterations": 20000,
        "skipped": 4,
    },
}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_RANDOM_SEED = 20_260_720
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95


class _ResultSource:
    """Read a ZIP or extracted result tree without changing the input."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._zip: zipfile.ZipFile | None = None
        if self.path.is_file():
            self._zip = zipfile.ZipFile(self.path, "r")
            self.names = tuple(name for name in self._zip.namelist() if not name.endswith("/"))
        elif self.path.is_dir():
            self.names = tuple(
                item.relative_to(self.path).as_posix()
                for item in self.path.rglob("*")
                if item.is_file()
            )
        else:
            raise FileNotFoundError(self.path)
        result_names = [
            name for name in self.names if name == "results.csv" or name.endswith("/results.csv")
        ]
        if len(result_names) != 1:
            raise ValueError(f"Expected exactly one results.csv, found {len(result_names)}")
        self.root = str(PurePosixPath(result_names[0]).parent)
        if self.root == ".":
            self.root = ""

    def __enter__(self) -> "_ResultSource":
        return self

    def __exit__(self, *_args: object) -> None:
        if self._zip is not None:
            self._zip.close()

    def _name(self, relative: str) -> str:
        return f"{self.root}/{relative}" if self.root else relative

    def exists(self, relative: str) -> bool:
        return self._name(relative) in self.names

    def read_bytes(self, relative: str) -> bytes:
        name = self._name(relative)
        if self._zip is not None:
            return self._zip.read(name)
        return (self.path / name).read_bytes()

    def read_text(self, relative: str) -> str:
        return self.read_bytes(relative).decode("utf-8-sig")

    def csv_rows(self, relative: str) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(self.read_text(relative))))

    def json_value(self, relative: str) -> dict[str, Any]:
        value = json.loads(self.read_text(relative))
        return value if isinstance(value, dict) else {}

    def yaml_value(self, relative: str) -> dict[str, Any]:
        value = yaml.safe_load(self.read_bytes(relative))
        return value if isinstance(value, dict) else {}

    def crc_error(self) -> str | None:
        return self._zip.testzip() if self._zip is not None else None

    def sha256(self) -> str | None:
        return file_sha256(self.path).lower() if self.path.is_file() else None


def _expected_pairs(scale: str) -> set[tuple[int, str]]:
    return {(seed, variant) for seed in FINAL_SEEDS[scale] for variant in EXPECTED_VARIANTS}


def _artifact_count(source: _ResultSource, suffix: str) -> int:
    prefix = f"{source.root}/" if source.root else ""
    return sum(name.startswith(prefix) and name.endswith(suffix) for name in source.names)


def _iteration_log_count(source: _ResultSource) -> int:
    prefix = f"{source.root}/iteration_logs/" if source.root else "iteration_logs/"
    return sum(name.startswith(prefix) and name.endswith(".csv") for name in source.names)


def _record_matches_row(record: Mapping[str, Any], row: Mapping[str, str]) -> bool:
    result = record.get("result")
    if not isinstance(result, Mapping):
        return False
    exact = ("run_key", "status", "variant_name", "experiment_name", "instance_size", "git_commit")
    numeric = (
        "seed",
        "lower_bound",
        "upper_bound",
        "final_gap",
        "runtime",
        "time_limit",
        "penalized_runtime_par2",
        "iterations",
    )
    return (
        all(str(result.get(key)) == str(row.get(key)) for key in exact)
        and all(_close(result.get(key), row.get(key), abs_tol=1.0e-6) for key in numeric)
        and _bool(result.get("solved_to_tolerance")) == _bool(row.get("solved_to_tolerance"))
        and _bool(result.get("valid_UB")) == _bool(row.get("valid_UB"))
    )


def _all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _paired_bootstrap(differences: list[float]) -> dict[str, Any]:
    if len(differences) != 10 or not _all_finite(differences):
        return {
            "valid": False,
            "mean_paired_difference": None,
            "confidence_interval": None,
        }
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    alpha = 1.0 - BOOTSTRAP_CONFIDENCE_LEVEL
    quantiles = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return {
        "valid": True,
        "estimand": "mean_paired_par2_difference_core_minus_v1",
        "scope": "large_final_only",
        "resamples": BOOTSTRAP_RESAMPLES,
        "analysis_random_seed": BOOTSTRAP_RANDOM_SEED,
        "rng": "numpy.random.default_rng",
        "quantile_method": "linear",
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "mean_paired_difference": float(values.mean()),
        "confidence_interval": [float(quantiles[0]), float(quantiles[1])],
        "auxiliary_only": True,
        "replaces_confirmation_thresholds": False,
    }


def _summary(scale: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    by_pair = {(_int(row.get("seed")), str(row.get("variant_name"))): row for row in rows}
    observed_seeds = sorted({seed for seed, _ in by_pair})
    if set(by_pair) != _expected_pairs(scale) or len(rows) != len(by_pair):
        return {"scale": scale, "observed_seeds": observed_seeds, "variants": {}}
    summary: dict[str, Any] = {
        "scale": scale,
        "observed_seeds": observed_seeds,
        "variants": {},
    }
    for variant in EXPECTED_VARIANTS:
        selected = [by_pair[(seed, variant)] for seed in FINAL_SEEDS[scale]]
        summary["variants"][variant] = {
            "solved_count": sum(_bool(row.get("solved_to_tolerance")) for row in selected),
            "run_count": len(selected),
            "solved_rate": statistics.mean(_bool(row.get("solved_to_tolerance")) for row in selected),
            "mean_par2": statistics.mean(_float(row.get("penalized_runtime_par2")) for row in selected),
            "mean_runtime": statistics.mean(_float(row.get("runtime")) for row in selected),
            "mean_iterations": statistics.mean(_float(row.get("iterations")) for row in selected),
            "mean_master_time": statistics.mean(_float(row.get("master_time")) for row in selected),
            "mean_subproblem_time": statistics.mean(_float(row.get("subproblem_time")) for row in selected),
            "mean_final_gap": statistics.mean(_float(row.get("final_gap")) for row in selected),
        }
    v1 = summary["variants"][V1_VARIANT]
    core = summary["variants"][CORE_VARIANT]
    core_rows = [by_pair[(seed, CORE_VARIANT)] for seed in FINAL_SEEDS[scale]]
    attempts = sum(_float(row.get("core_point_attempt_count")) for row in core_rows)
    accepted = sum(_float(row.get("core_point_success_count")) for row in core_rows)
    extra_runtime = sum(_float(row.get("core_point_total_runtime")) for row in core_rows)
    runtime = sum(_float(row.get("runtime")) for row in core_rows)
    par2_wins = 0
    iteration_wins = 0
    core_rank = 0.0
    v1_rank = 0.0
    differences: list[float] = []
    for seed in FINAL_SEEDS[scale]:
        core_row, v1_row = by_pair[(seed, CORE_VARIANT)], by_pair[(seed, V1_VARIANT)]
        core_par2 = _float(core_row.get("penalized_runtime_par2"))
        v1_par2 = _float(v1_row.get("penalized_runtime_par2"))
        differences.append(core_par2 - v1_par2)
        par2_wins += core_par2 <= v1_par2 + ABS_TOL
        iteration_wins += _float(core_row.get("iterations")) < _float(v1_row.get("iterations"))
        if _close(core_par2, v1_par2):
            core_rank += 1.5
            v1_rank += 1.5
        elif core_par2 < v1_par2:
            core_rank += 1.0
            v1_rank += 2.0
        else:
            core_rank += 2.0
            v1_rank += 1.0
    summary.update(
        {
            "par2_reduction_percent": 100.0 * (v1["mean_par2"] - core["mean_par2"]) / v1["mean_par2"],
            "iteration_reduction_percent": 100.0 * (v1["mean_iterations"] - core["mean_iterations"]) / v1["mean_iterations"],
            "paired_par2_core_not_worse": par2_wins,
            "paired_iterations_core_lower": iteration_wins,
            "core_mean_rank": core_rank / 10.0,
            "v1_mean_rank": v1_rank / 10.0,
            "core_attempt_count": int(attempts),
            "core_success_count": int(accepted),
            "core_acceptance_rate": accepted / attempts if attempts else None,
            "core_total_runtime": extra_runtime,
            "core_extra_runtime_share": extra_runtime / runtime if runtime else None,
            "secondary_attempt_count": int(sum(_float(row.get("v3_secondary_trigger_count")) for row in core_rows)),
            "paired_par2_differences_core_minus_v1": differences,
        }
    )
    return summary


def _audit_scale(
    scale: str,
    input_path: str | Path,
    expected_archive_sha256: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    expected = EXPECTED_SCALE[scale]
    with _ResultSource(input_path) as source:
        archive_sha = source.sha256()
        checks.append(
            _check(
                f"{scale}_archive_sha256",
                expected_archive_sha256 is None or archive_sha is None or archive_sha == expected_archive_sha256.lower(),
                archive_sha or "directory input",
            )
        )
        checks.append(_check(f"{scale}_archive_crc", source.crc_error() is None))
        required_top = ("results.csv", "run_manifest.json", "resolved_config.yaml")
        checks.append(_check(f"{scale}_top_level_files_complete", all(source.exists(x) for x in required_top)))
        rows = source.csv_rows("results.csv") if source.exists("results.csv") else []
        manifest = source.json_value("run_manifest.json") if source.exists("run_manifest.json") else {}
        top_config = source.yaml_value("resolved_config.yaml") if source.exists("resolved_config.yaml") else {}
        pairs = [(_int(row.get("seed")), str(row.get("variant_name"))) for row in rows]
        pair_set = set(pairs)
        seeds = {seed for seed, _ in pair_set}
        variants = {variant for _, variant in pair_set}
        checks.extend(
            [
                _check(f"{scale}_exactly_20_rows", len(rows) == 20, len(rows)),
                _check(f"{scale}_exact_final_seeds", seeds == set(FINAL_SEEDS[scale]), sorted(seeds)),
                _check(f"{scale}_only_v1_and_core", variants == set(EXPECTED_VARIANTS), sorted(variants)),
                _check(f"{scale}_no_missing_or_duplicate_pairs", pair_set == _expected_pairs(scale) and len(pairs) == len(pair_set)),
                _check(f"{scale}_exactly_20_run_json", _artifact_count(source, "/run.json") == 20, _artifact_count(source, "/run.json")),
                _check(f"{scale}_exactly_20_status_json", _artifact_count(source, "/status.json") == 20, _artifact_count(source, "/status.json")),
                _check(f"{scale}_exactly_20_error_files", _artifact_count(source, "/error.txt") == 20, _artifact_count(source, "/error.txt")),
                _check(f"{scale}_exactly_20_iteration_logs", _iteration_log_count(source) == 20, _iteration_log_count(source)),
            ]
        )
        expected_manifest = {
            "expected_run_count": 20,
            "completed_run_count": 20,
            "solved_run_count": 20,
            "failed_run_count": 0,
            "remaining_run_count": 0,
            "skipped_run_count": expected["skipped"],
            "git_commit": FINAL_COMMIT,
        }
        manifest_failures = {
            key: manifest.get(key)
            for key, value in expected_manifest.items()
            if manifest.get(key) != value
        }
        checks.append(_check(f"{scale}_manifest_complete_and_consistent", not manifest_failures, manifest_failures))
        checks.append(
            _check(
                f"{scale}_manifest_canonical_config_identity",
                manifest.get("config_sha256") == config_sha256(top_config),
                {
                    "manifest": manifest.get("config_sha256"),
                    "recomputed": config_sha256(top_config),
                },
            )
        )
        top_ok = (
            top_config.get("experiment_name") == expected["experiment"]
            and top_config.get("random_seeds") == list(FINAL_SEEDS[scale])
            and top_config.get("instance_sizes") == [scale]
            and top_config.get("variants") == list(EXPECTED_VARIANTS)
            and _close(top_config.get("time_limit"), expected["time_limit"])
            and top_config.get("max_iterations") == expected["max_iterations"]
            and _close(top_config.get("tol"), 1.0e-4)
            and top_config.get("gamma_target") == 2
            and top_config.get("gamma_schedule") == [2]
            and top_config.get("formal_inference_allowed") is True
            and top_config.get("protocol_phase") == "final"
            and top_config.get("final_analysis") == EXPECTED_FINAL_ANALYSIS
            and str(top_config.get("candidate_config_sha256", "")).lower() == EXPECTED_CANDIDATE_SHA256
        )
        checks.append(_check(f"{scale}_top_resolved_protocol_identity", top_ok))

        flags = {
            "artifacts": True,
            "resolved": True,
            "identity": True,
            "record_consistency": True,
            "status": True,
            "par2": True,
            "bounds": True,
            "monotone": True,
            "continuity": True,
            "gap": True,
            "ub_source": True,
            "core": True,
            "fallback": True,
            "pseudo": True,
            "certification": True,
            "one_cut": True,
            "final_log": True,
            "errors": True,
            "secondary": True,
            "timing": True,
        }
        details: list[str] = []
        seen_run_keys: set[str] = set()
        seen_log_names: set[str] = set()
        for row in rows:
            seed, variant, run_key = _int(row.get("seed")), str(row.get("variant_name")), str(row.get("run_key"))
            run_dir = f"runs/{run_key}"
            required = tuple(f"{run_dir}/{name}" for name in ("run.json", "status.json", "resolved_config.yaml", "error.txt"))
            if not all(source.exists(name) for name in required):
                flags["artifacts"] = False
                details.append(f"missing artifacts: {run_key}")
                continue
            run = source.json_value(f"{run_dir}/run.json")
            status = source.json_value(f"{run_dir}/status.json")
            resolved = source.yaml_value(f"{run_dir}/resolved_config.yaml")
            flags["errors"] &= source.read_text(f"{run_dir}/error.txt").strip() == ""
            resolved_ok, resolved_errors = _resolved_config_ok(resolved, scale, seed, variant)
            flags["resolved"] &= resolved_ok
            details.extend(f"{run_key}: {error}" for error in resolved_errors)
            log_name = str(row.get("iteration_log_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
            log_path = f"iteration_logs/{log_name}"
            if not source.exists(log_path):
                flags["artifacts"] = False
                details.append(f"missing log: {run_key}")
                continue
            log = source.csv_rows(log_path)
            flags["identity"] &= (
                row.get("experiment_name") == expected["experiment"]
                and row.get("instance_size") == scale
                and row.get("git_commit") == FINAL_COMMIT
                and run.get("git_commit") == FINAL_COMMIT
                and run.get("run_key") == run_key
                and status.get("run_key") == run_key
                and run_key not in seen_run_keys
                and log_name not in seen_log_names
            )
            seen_run_keys.add(run_key)
            seen_log_names.add(log_name)
            flags["record_consistency"] &= _record_matches_row(run, row)
            flags["record_consistency"] &= run.get("config_sha256") == config_sha256(resolved)
            solved = _bool(row.get("solved_to_tolerance"))
            gap = _float(row.get("final_gap"))
            flags["status"] &= (
                run.get("state") == "complete"
                and run.get("success") is True
                and run.get("solved_to_tolerance") is True
                and status.get("state") == "complete"
                and status.get("success") is True
                and status.get("solved_to_tolerance") is True
                and solved
                and row.get("status") == "optimal"
                and status.get("status") == "optimal"
                and math.isfinite(gap)
                and gap <= 1.0e-4 + ABS_TOL
            )
            expected_par2 = _float(row.get("runtime")) if solved else 2.0 * _float(row.get("time_limit"))
            flags["par2"] &= _close(row.get("penalized_runtime_par2"), expected_par2, abs_tol=1.0e-6)
            flags["bounds"] &= (
                _bool(row.get("valid_UB"))
                and _bool(row.get("ub_uses_subproblem_bound"))
                and math.isfinite(_float(row.get("lower_bound")))
                and math.isfinite(_float(row.get("upper_bound")))
                and _float(row.get("lower_bound")) <= _float(row.get("upper_bound")) + ABS_TOL
            )
            if not log:
                flags["artifacts"] = False
                continue
            iterations = [_int(item.get("iteration")) for item in log]
            flags["continuity"] &= iterations == list(range(1, len(log) + 1))
            lbs = [_float(item.get("LB")) for item in log]
            ubs = [_float(item.get("UB")) for item in log]
            mp_gaps = [_float(item.get("requested_master_mip_gap")) for item in log]
            sp_gaps = [_float(item.get("subproblem_requested_mip_gap")) for item in log]
            flags["monotone"] &= (
                _all_finite(lbs + ubs + mp_gaps + sp_gaps)
                and _nondecreasing(lbs)
                and _nonincreasing(ubs)
                and _nonincreasing(mp_gaps)
                and _nonincreasing(sp_gaps)
            )
            for item in log:
                lb, ub = _float(item.get("LB")), _float(item.get("UB"))
                expected_gap = max(0.0, (ub - lb) / max(1.0, abs(ub)))
                flags["gap"] &= _close(item.get("global_gap"), expected_gap, abs_tol=1.0e-6)
                flags["ub_source"] &= (
                    _bool(item.get("target_robust_evaluation_used"))
                    and not _bool(item.get("core_point_auxiliary_bound_used_for_UB"))
                    and not _bool(item.get("v3_secondary_bound_used_for_UB"))
                )
                accepted = _bool(item.get("core_point_cut_accepted"))
                attempted = _bool(item.get("core_point_attempted"))
                if accepted:
                    flags["core"] &= (
                        attempted
                        and item.get("core_point_stage1_status") == "optimal"
                        and item.get("core_point_stage2_status") == "optimal"
                        and _bool(item.get("core_point_dual_feasible"))
                        and _float(item.get("core_point_strengthened_value_at_current")) + ABS_TOL
                        >= _float(item.get("core_point_current_value_floor"))
                    )
                if attempted and not accepted:
                    flags["fallback"] &= (
                        bool(str(item.get("core_point_cut_fallback_reason", "")).strip())
                        and _close(item.get("cut_rhs_current"), item.get("core_point_original_value_at_current"), abs_tol=1.0e-5)
                    )
                if not _bool(item.get("subproblem_has_incumbent")):
                    flags["pseudo"] &= not _bool(item.get("cut_added"))
                if _bool(item.get("final_certification_active")):
                    flags["certification"] &= not attempted and not _bool(item.get("v3_secondary_attempted"))
                flags["one_cut"] &= _int(item.get("cuts_added_this_iteration")) <= 1
                flags["secondary"] &= (
                    not _bool(item.get("v3_secondary_attempted"))
                    and not _bool(item.get("v3_secondary_cut_added"))
                    and not _bool(item.get("v3_secondary_bound_used_for_UB"))
                )
            last = log[-1]
            flags["final_log"] &= (
                len(log) == _int(row.get("iterations"))
                and _int(last.get("iteration")) == _int(row.get("iterations"))
                and _close(last.get("LB"), row.get("lower_bound"), abs_tol=1.0e-6)
                and _close(last.get("UB"), row.get("upper_bound"), abs_tol=1.0e-6)
                and _close(last.get("global_gap"), row.get("final_gap"), abs_tol=1.0e-6)
            )
            flags["secondary"] &= all(
                _int(row.get(field)) == 0
                for field in (
                    "v3_secondary_trigger_count",
                    "v3_secondary_solve_count",
                    "v3_secondary_cut_added_count",
                    "v3_secondary_cuts_added",
                )
            )
            if variant == CORE_VARIANT:
                timing_values = [
                    _float(row.get("core_point_total_runtime")),
                    _float(row.get("core_point_stage1_total_runtime")),
                    _float(row.get("core_point_stage2_total_runtime")),
                ]
                flags["timing"] &= _all_finite(timing_values) and all(value >= 0.0 for value in timing_values)
        check_names = {
            "artifacts": "run_artifacts_complete",
            "resolved": "resolved_configs_frozen",
            "identity": "run_identity_commit_and_uniqueness",
            "record_consistency": "run_status_results_consistent",
            "status": "complete_solved_status_semantics",
            "par2": "par2_correct",
            "bounds": "valid_bounds",
            "monotone": "bounds_and_requested_gaps_monotone",
            "continuity": "iteration_logs_complete_and_continuous",
            "gap": "global_gap_formula",
            "ub_source": "only_original_robust_bound_updates_ub",
            "core": "accepted_core_cuts_valid",
            "fallback": "core_failure_falls_back_to_original_cut",
            "pseudo": "no_pseudo_cut_without_incumbent",
            "certification": "certification_disables_core",
            "one_cut": "at_most_one_cut_per_iteration",
            "final_log": "final_log_matches_results",
            "errors": "all_error_files_empty",
            "secondary": "secondary_component_never_used",
            "timing": "core_extra_time_complete",
        }
        checks.extend(_check(f"{scale}_{check_names[key]}", value, details[:20]) for key, value in flags.items())

    by_pair = {(_int(row.get("seed")), str(row.get("variant_name"))): row for row in rows}
    overlap = len(by_pair) == 20 and all(
        max(_float(by_pair[(seed, V1_VARIANT)].get("lower_bound")), _float(by_pair[(seed, CORE_VARIANT)].get("lower_bound")))
        <= min(_float(by_pair[(seed, V1_VARIANT)].get("upper_bound")), _float(by_pair[(seed, CORE_VARIANT)].get("upper_bound"))) + 1.0e-4
        for seed in FINAL_SEEDS[scale]
        if (seed, V1_VARIANT) in by_pair and (seed, CORE_VARIANT) in by_pair
    )
    checks.append(_check(f"{scale}_paired_final_intervals_overlap", overlap))
    resume_valid = (
        manifest.get("skipped_run_count") == expected["skipped"]
        and manifest.get("completed_run_count") == 20
        and manifest.get("remaining_run_count") == 0
        and manifest.get("failed_run_count") == 0
        and len(seen_run_keys) == 20
        and len(seen_log_names) == 20
        and flags["status"]
        and flags["continuity"]
        and flags["record_consistency"]
    )
    checks.append(_check(f"{scale}_resume_record_is_complete_and_nonduplicating", resume_valid))
    return checks, _summary(scale, rows)


def _decision(scales: Mapping[str, dict[str, Any]], correctness: bool) -> tuple[str, dict[str, bool]]:
    if not correctness or any(not scales.get(scale, {}).get("variants") for scale in FINAL_SEEDS):
        return "invalid_run", {"identity_completeness_correctness": False}
    medium, large = scales["medium_large"], scales["large"]
    mv1, mcore = medium["variants"][V1_VARIANT], medium["variants"][CORE_VARIANT]
    lv1, lcore = large["variants"][V1_VARIANT], large["variants"][CORE_VARIANT]
    gates = {
        "identity_completeness_correctness": True,
        "medium_solved_rate_noninferior": mcore["solved_rate"] >= mv1["solved_rate"],
        "medium_mean_par2_within_103_percent": mcore["mean_par2"] <= 1.03 * mv1["mean_par2"],
        "large_solved_rate_noninferior": lcore["solved_rate"] >= lv1["solved_rate"],
        "large_mean_par2_reduction_at_least_7_5_percent": large["par2_reduction_percent"] >= 7.5,
        "large_mean_iterations_reduction_at_least_15_percent": large["iteration_reduction_percent"] >= 15.0,
        "large_par2_pairs_at_least_6_of_10": large["paired_par2_core_not_worse"] >= 6,
        "large_iteration_pairs_at_least_6_of_10": large["paired_iterations_core_lower"] >= 6,
        "large_mean_paired_rank_better": large["core_mean_rank"] < large["v1_mean_rank"],
    }
    return ("final_confirmed" if all(gates.values()) else "final_not_confirmed"), gates


def audit_cut_v3_final_results(
    medium_input: str | Path,
    large_input: str | Path,
    *,
    repo_root: str | Path | None = None,
    expected_archive_sha256: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Audit frozen Final evidence without extracting, modifying, or solving it."""

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    hashes = EXPECTED_ARCHIVE_SHA256 if expected_archive_sha256 is None else expected_archive_sha256
    checks: list[dict[str, Any]] = []
    config_dir = root / "experiments/configs"
    for scale, expected_hash in EXPECTED_INPUT_CONFIG_SHA256.items():
        path = config_dir / str(EXPECTED_SCALE[scale]["config"])
        actual = file_sha256(path).lower() if path.exists() else "missing"
        checks.append(_check(f"{scale}_raw_input_yaml_sha256", actual == expected_hash, actual))
    candidate = config_dir / "selected_cut_strengthened_joint_v3_candidate.yaml"
    candidate_hash = file_sha256(candidate).lower() if candidate.exists() else "missing"
    checks.append(_check("selected_candidate_yaml_sha256", candidate_hash == EXPECTED_CANDIDATE_SHA256, candidate_hash))

    scale_reports: dict[str, dict[str, Any]] = {}
    for scale, input_path in (("medium_large", medium_input), ("large", large_input)):
        try:
            scale_checks, summary = _audit_scale(scale, input_path, hashes.get(scale))
        except Exception as exc:  # noqa: BLE001 - malformed evidence is an invalid run.
            checks.append(_check(f"{scale}_evidence_readable", False, f"{type(exc).__name__}: {exc}"))
            scale_reports[scale] = {"scale": scale, "variants": {}}
        else:
            checks.extend(scale_checks)
            scale_reports[scale] = summary
    medium_observed = set(scale_reports.get("medium_large", {}).get("observed_seeds", []))
    large_observed = set(scale_reports.get("large", {}).get("observed_seeds", []))
    seed_sets_disjoint = (
        medium_observed == set(range(90, 100))
        and large_observed == set(range(100, 110))
        and medium_observed.isdisjoint(large_observed)
    )
    checks.append(_check("final_seed_groups_exact_and_disjoint", seed_sets_disjoint))
    failed_before_decision = [item["check"] for item in checks if not item["passed"]]
    decision, gates = _decision(scale_reports, not failed_before_decision)
    for name, passed in gates.items():
        checks.append(_check(f"decision_{name}", passed))
    bootstrap = _paired_bootstrap(
        list(scale_reports.get("large", {}).get("paired_par2_differences_core_minus_v1", []))
    )
    checks.append(_check("large_auxiliary_bootstrap_reproducible", bootstrap.get("valid") is True))
    failed = [item["check"] for item in checks if not item["passed"]]
    return {
        "audit_name": "cut_strengthened_joint_v3_final_results",
        "created_at": utc_now_iso(),
        "read_only": True,
        "final_run_commit": FINAL_COMMIT,
        "decision": decision,
        "selected_algorithm": CORE_VARIANT,
        "v3_status": "completed" if decision == "final_confirmed" else "not_completed",
        "retuning_allowed": False,
        "seed_replacement_allowed": False,
        "development_validation_pooling_allowed": False,
        "next_authorized_stage": "fairness_diagnostic_only" if decision == "final_confirmed" else "none",
        "all_required_checks_passed": not failed,
        "required_check_count": len(checks),
        "passed_check_count": sum(item["passed"] for item in checks),
        "failed_checks": failed,
        "archive_sha256": dict(hashes),
        "input_config_sha256": dict(EXPECTED_INPUT_CONFIG_SHA256),
        "candidate_config_sha256": EXPECTED_CANDIDATE_SHA256,
        "scales": scale_reports,
        "decision_gates": gates,
        "auxiliary_bootstrap": bootstrap,
        "checks": checks,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Cut-strengthened Joint V3 Final results audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Selected algorithm: `{report['selected_algorithm']}`",
        f"- Checks: {report['passed_check_count']}/{report['required_check_count']} passed",
        "",
        "| Scale | Method | Solved | Mean PAR-2 | Mean iterations |",
        "|---|---|---:|---:|---:|",
    ]
    for scale in ("medium_large", "large"):
        for variant in EXPECTED_VARIANTS:
            value = report["scales"].get(scale, {}).get("variants", {}).get(variant, {})
            lines.append(
                f"| {scale} | {variant} | {value.get('solved_count', 0)}/{value.get('run_count', 0)} | "
                f"{value.get('mean_par2', math.nan):.12f} | {value.get('mean_iterations', math.nan):.1f} |"
            )
    bootstrap = report.get("auxiliary_bootstrap", {})
    lines.extend(
        [
            "",
            "## Auxiliary Large paired bootstrap",
            "",
            f"Mean core-minus-V1 PAR-2: {bootstrap.get('mean_paired_difference')}",
            f"95% percentile CI: {bootstrap.get('confidence_interval')}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit of frozen V3 Final evidence.")
    parser.add_argument("--medium-input", required=True)
    parser.add_argument("--large-input", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    args = parser.parse_args()
    report = audit_cut_v3_final_results(args.medium_input, args.large_input)
    if args.output_json:
        atomic_write_json(args.output_json, report)
    if args.output_markdown:
        atomic_write_text(args.output_markdown, _markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["decision"] == "invalid_run":
        raise SystemExit(2)
    if report["decision"] == "final_not_confirmed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
