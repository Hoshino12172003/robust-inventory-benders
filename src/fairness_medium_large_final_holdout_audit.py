from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Iterable, Mapping
import zipfile

from .experiment_protocol import atomic_write_csv, atomic_write_json
from .fairness_medium_large_final_holdout_runner import (
    CANDIDATES,
    EXPECTED_CANDIDATE_SHA256,
    FINAL_HOLDOUT_EXECUTION_ATTEMPT,
    HOLDOUT_SEEDS,
    PROHIBITED_SEEDS,
    RHOS,
    SCENARIO_COUNT,
    dry_run_report,
    final_holdout_run_plan,
)
from .fairness_scalability_runner import run_directory_id


ATTEMPT5_ARCHIVE_SHA256 = "09B41862A5BFED724EDBEC1E64996B54AA878119F5C0DEDFE5B10126B2525A98"
ATTEMPT5_DECISION = {
    "decision": "stop_final_large_remediation",
    "l0_passed": False,
    "large_frontier_certified": False,
    "l1_authorized": False,
    "m1_authorized": False,
    "additional_large_runs_authorized": False,
    "large_incumbent_usable_as_optimal_result": False,
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _json_bytes(value: bytes, *, name: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain an object.")
    return parsed


def audit_attempt5_archive(path: str | Path) -> dict[str, Any]:
    archive = Path(path)
    before = file_sha256(archive)
    if before != ATTEMPT5_ARCHIVE_SHA256:
        raise ValueError("Large Attempt 5 archive SHA256 mismatch.")
    with zipfile.ZipFile(archive) as source:
        bad = source.testzip()
        if bad is not None:
            raise ValueError(f"Large Attempt 5 CRC failure: {bad}")
        names = source.namelist()
        file_names = sorted(name for name in names if not name.endswith("/"))
        entry_hashes = [
            {
                "entry": name.removeprefix("./"),
                "sha256": _sha256_bytes(source.read(name)),
                "size": len(source.read(name)),
            }
            for name in file_names
        ]
        manifest_name = next(name for name in file_names if name.endswith("manifest.json"))
        manifest = _json_bytes(source.read(manifest_name), name=manifest_name)
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("Attempt 5 manifest identity is missing.")
        run_names = sorted(name for name in file_names if name.endswith("run.json"))
        status_names = sorted(name for name in file_names if name.endswith("status.json"))
        runs = [_json_bytes(source.read(name), name=name) for name in run_names]
        statuses = [_json_bytes(source.read(name), name=name) for name in status_names]
    after = file_sha256(archive)
    if after != before:
        raise ValueError("Large Attempt 5 archive changed during read-only audit.")
    if len(runs) != 2 or len(statuses) != 2:
        raise ValueError("Attempt 5 must contain two run and two status records.")
    if any(run.get("state") != "complete" for run in runs):
        raise ValueError("Attempt 5 run is not complete.")
    baseline = next(run for run in runs if run.get("task_type") == "baseline")
    frontier = next(run for run in runs if run.get("task_type") == "frontier")
    result = frontier.get("result") or {}
    expected = {
        "baseline_scientific_status": baseline.get("scientific_status"),
        "frontier_scientific_status": frontier.get("scientific_status"),
        "frontier_algorithm_status": frontier.get("algorithm_status"),
        "iterations": result.get("iterations"),
        "cuts": result.get("cuts"),
        "lower_bound": result.get("lower_bound"),
        "upper_bound": result.get("upper_bound"),
        "gap": result.get("gap"),
        "penalized_runtime_par2": result.get("penalized_runtime_par2"),
    }
    if expected != {
        "baseline_scientific_status": "certified_robust_optimal",
        "frontier_scientific_status": "time_limit_uncertified",
        "frontier_algorithm_status": "time_limit",
        "iterations": 95,
        "cuts": 121,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "gap": 1.0,
        "penalized_runtime_par2": 3600.0,
    }:
        raise ValueError(f"Attempt 5 scientific result mismatch: {expected}")
    if result.get("post_evaluation") is not None:
        raise ValueError("Uncertified Attempt 5 frontier must not claim post-evaluation success.")
    if any(run.get("scientific_status") == "implementation_error" for run in runs):
        raise ValueError("Attempt 5 contains an implementation error.")
    if identity.get("git_commit") != "0866e4ed20a367af52f28bffc9bab8e621ac514c":
        raise ValueError("Attempt 5 Git identity mismatch.")
    if identity.get("execution_attempt") != 5:
        raise ValueError("Attempt 5 execution identity mismatch.")
    return {
        "audit_schema": "fairness_large_remediation_attempt5_archive_audit_v1",
        "source_archive": archive.name,
        "source_archive_sha256": before,
        "source_archive_sha256_unchanged_after_audit": after == before,
        "zip_crc_valid": True,
        "entry_count": len(names),
        "file_count": len(file_names),
        "directory_count": len(names) - len(file_names),
        "entry_sha256": entry_hashes,
        "identity": identity,
        "completed_run_count": 2,
        "baseline_certified_count": 1,
        "frontier_certified_count": 0,
        "pipeline_error_count": 0,
        "implementation_error_count": 0,
        "frontier": {
            "scientific_status": frontier["scientific_status"],
            "algorithm_status": frontier["algorithm_status"],
            "algorithm_runtime": result["algorithm_runtime"],
            "iterations": result["iterations"],
            "cuts": result["cuts"],
            "lower_bound": result["lower_bound"],
            "upper_bound": result["upper_bound"],
            "gap": result["gap"],
            "penalized_runtime_par2": result["penalized_runtime_par2"],
            "post_evaluation": None,
            "post_evaluation_absence_expected": True,
        },
        "decision": dict(ATTEMPT5_DECISION),
    }


def _seed_values_from_csv(value: bytes) -> set[int]:
    try:
        rows = csv.DictReader(io.StringIO(value.decode("utf-8-sig")))
    except UnicodeDecodeError:
        return set()
    seeds: set[int] = set()
    for row in rows:
        raw = row.get("seed") or row.get("random_seed")
        try:
            seeds.add(int(raw))
        except (TypeError, ValueError):
            pass
    return seeds


def _identity_seeds(value: Any) -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"seed", "random_seed"} and not isinstance(child, bool):
                try:
                    found.add(int(child))
                except (TypeError, ValueError):
                    pass
            elif key in {"seeds", "random_seeds", "holdout_seeds"} and isinstance(child, list):
                for item in child:
                    if not isinstance(item, bool):
                        try:
                            found.add(int(item))
                        except (TypeError, ValueError):
                            pass
            found.update(_identity_seeds(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_identity_seeds(child))
    return found


def _scan_archive(path: Path) -> tuple[set[int], dict[str, Any]]:
    seeds: set[int] = set()
    evidence: dict[str, Any] = {"path": str(path), "sha256": file_sha256(path), "entries_checked": 0}
    with zipfile.ZipFile(path) as source:
        for name in source.namelist():
            if name.endswith("/"):
                continue
            basename = PurePosixPath(name).name
            lowered = name.lower()
            for seed in HOLDOUT_SEEDS:
                if any(token in lowered for token in (f"seed{seed}", f"seed_{seed}", f"seed-{seed}")):
                    seeds.add(seed)
            if basename == "results.csv":
                seeds.update(_seed_values_from_csv(source.read(name)))
                evidence["entries_checked"] += 1
            elif basename in {
                "manifest.json", "scalability_development_manifest.json", "run_manifest.json"
            }:
                try:
                    seeds.update(_identity_seeds(json.loads(source.read(name))))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    raise ValueError(f"Invalid archive identity JSON: {path}!{name}")
                evidence["entries_checked"] += 1
    evidence["seeds"] = sorted(seeds)
    return seeds, evidence


def _scan_result_root(path: Path) -> tuple[set[int], dict[str, Any]]:
    seeds: set[int] = set()
    checked = 0
    if not path.exists():
        return seeds, {"path": str(path), "exists": False, "identity_files_checked": 0, "seeds": []}
    identity_names = {
        "manifest.json", "scalability_development_manifest.json", "run_manifest.json",
        "results.csv", "summary.csv",
    }
    for directory, names, files in os.walk(path):
        names[:] = [name for name in names if name not in {"post_evaluation", "checkpoint"}]
        for filename in files:
            lowered = filename.lower()
            for seed in HOLDOUT_SEEDS:
                if any(token in lowered for token in (f"seed{seed}", f"seed_{seed}", f"seed-{seed}")):
                    seeds.add(seed)
            if filename not in identity_names:
                continue
            target = Path(directory) / filename
            checked += 1
            if filename.endswith(".csv"):
                seeds.update(_seed_values_from_csv(target.read_bytes()))
            else:
                try:
                    seeds.update(_identity_seeds(json.loads(target.read_text(encoding="utf-8"))))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid result identity JSON: {target}") from exc
    return seeds, {"path": str(path), "exists": True, "identity_files_checked": checked, "seeds": sorted(seeds)}


def _tracked_files(repo_root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _is_preregistered_holdout_declaration(relative: str) -> bool:
    """Return true only for the frozen planning surface, never result artifacts."""
    normalized = relative.replace("\\", "/").lower()
    exact = {
        "docs/fairness_medium_large_final_holdout_protocol.md",
        "experiments/configs/fairness_medium_large_final_holdout.yaml",
        "src/fairness_medium_large_final_holdout_audit.py",
        "src/fairness_medium_large_final_holdout_runner.py",
        "tests/test_fairness_medium_large_final_holdout.py",
    }
    return normalized in exact or normalized.startswith(
        "analysis/fairness_medium_large_final_holdout_freeze/"
    )


def _looks_like_committed_execution_artifact(relative: str) -> bool:
    normalized = relative.replace("\\", "/").lower()
    basename = PurePosixPath(normalized).name
    execution_names = {
        "manifest.json",
        "run.json",
        "status.json",
        "results.csv",
        "summary.csv",
        "checkpoint.json",
        "checkpoint_index.json",
        "instance.json",
    }
    return (
        basename in execution_names
        or "/results_" in normalized
        or "/checkpoints/" in normalized
        or "/instances/" in normalized
    )


def audit_holdout_seed_nonaccess(
    *, repo_root: str | Path, archive_roots: Iterable[str | Path],
    result_roots: Iterable[str | Path],
) -> dict[str, Any]:
    repo = Path(repo_root)
    accessed: set[int] = set()
    tracked_actual: list[str] = []
    reservation_mentions: list[str] = []
    for relative in _tracked_files(repo):
        path = repo / relative
        lowered = relative.lower()
        preregistered = _is_preregistered_holdout_declaration(relative)
        execution_artifact = _looks_like_committed_execution_artifact(relative)
        filename_hit = any(
            token in lowered
            for seed in HOLDOUT_SEEDS
            for token in (f"seed{seed}", f"seed_{seed}", f"seed-{seed}")
        )
        if filename_hit and execution_artifact and not preregistered:
            tracked_actual.append(relative)
        if path.suffix.lower() == ".csv" and path.is_file():
            overlap = set(HOLDOUT_SEEDS).intersection(_seed_values_from_csv(path.read_bytes()))
            if overlap:
                if preregistered:
                    reservation_mentions.append(relative)
                elif execution_artifact:
                    tracked_actual.append(relative)
                    accessed.update(overlap)
        elif path.suffix.lower() == ".json" and path.is_file():
            try:
                overlap = set(HOLDOUT_SEEDS).intersection(
                    _identity_seeds(json.loads(path.read_text(encoding="utf-8")))
                )
            except json.JSONDecodeError:
                overlap = set()
            if overlap:
                if preregistered:
                    reservation_mentions.append(relative)
                elif execution_artifact:
                    tracked_actual.append(relative)
                    accessed.update(overlap)
        elif path.suffix.lower() in {".md", ".yaml", ".yml", ".py"} and path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(str(seed) in text for seed in HOLDOUT_SEEDS):
                reservation_mentions.append(relative)
    archive_evidence: list[dict[str, Any]] = []
    archives: list[Path] = []
    for root in archive_roots:
        candidate = Path(root)
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            archives.append(candidate)
        elif candidate.is_dir():
            archives.extend(
                path for path in candidate.rglob("*.zip")
                if ".venv" not in path.parts and not path.name.endswith(".failed.zip")
            )
    for archive in sorted(set(archives), key=lambda value: str(value).lower()):
        seeds, evidence = _scan_archive(archive)
        overlap = set(HOLDOUT_SEEDS).intersection(seeds)
        accessed.update(overlap)
        evidence["holdout_seed_overlap"] = sorted(overlap)
        archive_evidence.append(evidence)
    result_evidence: list[dict[str, Any]] = []
    for root in result_roots:
        seeds, evidence = _scan_result_root(Path(root))
        overlap = set(HOLDOUT_SEEDS).intersection(seeds)
        accessed.update(overlap)
        evidence["holdout_seed_overlap"] = sorted(overlap)
        result_evidence.append(evidence)
    if tracked_actual or accessed:
        raise ValueError(
            "Holdout seed access detected; fail closed: "
            f"seeds={sorted(accessed)}, tracked={sorted(set(tracked_actual))}"
        )
    return {
        "audit_schema": "fairness_medium_large_final_holdout_seed_nonaccess_v1",
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "prohibited_seeds": list(PROHIBITED_SEEDS),
        "holdout_seed_accessed": False,
        "actual_access_evidence": [],
        "tracked_reservation_declarations": sorted(set(reservation_mentions)),
        "tracked_reservations_count_as_access": False,
        "archives": archive_evidence,
        "result_roots": result_evidence,
        "decision": "holdout_seed_set_pristine",
    }


def static_audit(config: Mapping[str, Any], dry_run: Mapping[str, Any]) -> dict[str, Any]:
    specs = final_holdout_run_plan()
    checks = {
        "matrix_110": len(specs) == 110,
        "baseline_10": sum(spec.task_type == "baseline" for spec in specs) == 10,
        "frontier_100": sum(spec.task_type == "frontier" for spec in specs) == 100,
        "unique_run_keys": len({spec.run_key for spec in specs}) == 110,
        "seeds_exact": sorted({spec.seed for spec in specs}) == list(HOLDOUT_SEEDS),
        "rhos_exact": sorted({float(spec.rho) for spec in specs if spec.rho is not None}) == list(RHOS),
        "candidates_exact": sorted({spec.candidate for spec in specs if spec.task_type == "frontier"}) == sorted(CANDIDATES),
        "one_baseline_per_seed": all(
            sum(spec.task_type == "baseline" and spec.seed == seed for spec in specs) == 1
            for seed in HOLDOUT_SEEDS
        ),
        "ten_frontier_per_seed": all(
            sum(spec.task_type == "frontier" and spec.seed == seed for spec in specs) == 10
            for seed in HOLDOUT_SEEDS
        ),
        "scenario_count_1831": dry_run.get("scenario_count") == SCENARIO_COUNT,
        "dry_run_no_instance": dry_run.get("instances_generated") is False,
        "dry_run_no_solver": dry_run.get("solver_called") is False,
        "output_absent": dry_run.get("output_dir_exists") is False,
        "windows_portable": dry_run.get("windows_portability_check") is True,
        "formal_authorization": config.get("formal_run_authorized") is True,
        "candidate_sha": str(config.get("candidate_config_sha256", "")).upper() == EXPECTED_CANDIDATE_SHA256,
        "independent_unit_seed": config.get("statistical_analysis", {}).get("independent_unit") == "seed",
        "cluster_bootstrap_seed": config.get("statistical_analysis", {}).get("cluster_bootstrap_unit") == "seed",
        "seed_rho_not_independent": config.get("statistical_analysis", {}).get("seed_rho_tasks_are_independent") is False,
        "holm_correction": config.get("statistical_analysis", {}).get("multiple_testing_correction") == "Holm",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Final holdout static audit failed: {failed}")
    return {"checks": checks, "passed": len(checks), "total": len(checks), "status": "pass"}


def write_freeze_evidence(
    *, output_dir: str | Path, attempt5: Mapping[str, Any], seed_audit: Mapping[str, Any],
    config_path: str | Path, dry_run: Mapping[str, Any], static: Mapping[str, Any],
) -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs = [
        atomic_write_json(root / "large_attempt5_audit.json", dict(attempt5)),
        atomic_write_json(root / "holdout_seed_access_audit.json", dict(seed_audit)),
        atomic_write_json(root / "decision.json", dict(ATTEMPT5_DECISION)),
        atomic_write_json(
            root / "source_archive_provenance.json",
            {
                "source_archive": "fairness_large_final_remediation_l0_attempt5_results.zip",
                "source_archive_sha256": ATTEMPT5_ARCHIVE_SHA256,
                "audit_mode": "read_only",
                "original_archive_committed": False,
                "formal_run_artifacts_committed": False,
            },
        ),
        atomic_write_json(root / "holdout_dry_run.json", dict(dry_run)),
        atomic_write_json(root / "static_audit.json", dict(static)),
    ]
    specs = final_holdout_run_plan()
    rows = [
        {
            "stage": spec.introduced_stage,
            "task_type": spec.task_type,
            "scale": spec.scale,
            "seed": spec.seed,
            "rho": "NOT_APPLICABLE" if spec.rho is None else spec.rho,
            "candidate": spec.candidate,
            "run_key": spec.run_key,
            "run_directory_id": run_directory_id(spec.run_key),
        }
        for spec in specs
    ]
    outputs.append(
        atomic_write_csv(root / "frozen_run_plan.csv", rows, list(rows[0]))
    )
    summary = [
        {"task_type": "baseline", "planned_count": 10},
        {"task_type": "frontier", "planned_count": 100},
        {"task_type": "all", "planned_count": 110},
    ]
    outputs.append(atomic_write_csv(root / "experiment_matrix_summary.csv", summary, list(summary[0])))
    artifacts = []
    for path in outputs:
        artifacts.append({"artifact": path.name, "sha256": file_sha256(path)})
    artifacts.extend(
        [
            {"artifact": Path(config_path).as_posix(), "sha256": file_sha256(config_path)},
            {"artifact": "source_archive", "sha256": ATTEMPT5_ARCHIVE_SHA256},
        ]
    )
    outputs.append(atomic_write_csv(root / "artifact_sha256.csv", artifacts, ["artifact", "sha256"]))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit and freeze the final fairness experiment protocol")
    parser.add_argument("--large-zip", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--archive-root", action="append", default=[])
    parser.add_argument("--result-root", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    from .fairness_medium_large_final_holdout_runner import _strict_yaml

    config_path = Path(args.config)
    config = _strict_yaml(config_path)
    dry = dry_run_report(config, config_path=config_path)
    attempt5 = audit_attempt5_archive(args.large_zip)
    seed_audit = audit_holdout_seed_nonaccess(
        repo_root=args.repo_root,
        archive_roots=args.archive_root,
        result_roots=args.result_root,
    )
    static = static_audit(config, dry)
    write_freeze_evidence(
        output_dir=args.output_dir,
        attempt5=attempt5,
        seed_audit=seed_audit,
        config_path=config_path,
        dry_run=dry,
        static=static,
    )
    print(json.dumps({"attempt5": attempt5["decision"], "seed_audit": seed_audit["decision"], "dry_run": dry, "static": static}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
