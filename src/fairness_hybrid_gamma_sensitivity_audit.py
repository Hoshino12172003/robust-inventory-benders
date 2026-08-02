from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
from typing import Any
import zipfile

import yaml

from .experiment_protocol import file_sha256
from .fairness_hybrid_gamma_sensitivity_runner import (
    BASE_COMMIT,
    CANDIDATE_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    GAMMAS,
    SEEDS,
    SCALES,
    dry_run,
    expand_plan,
    load_config,
    scenario_count,
    validate_config,
)


FORMAL_ZIP_SHA256 = "BD253A4DE967143B8CD88E7DEADCB821D6062208CB71664B900F9BD8EBDF3839"
DEFAULT_ARCHIVE = Path(r"E:\论文代码\rfhfinal\fairness_hybrid_final_holdout_results.zip")
PROTECTED_SHA256 = {
    "src/benders.py": "37967750EE1AAD5575A9B1FE0B050F012EC21DB58FA277FBEFAA5A48CFEF1D9F",
    "src/scenarios.py": "7294C60DC318F7678F8A4464DAF2CBD85E540842C6C3858BB1D30A9DE7915511",
    "experiments/configs/certified_hybrid_scenario_benders_fairness_d1_candidate.yaml": CANDIDATE_SHA256,
    "analysis/fairness_hybrid_final_holdout_reconciliation/results.corrected.csv": "50EB5823F4C7138E65FA36546B90EE081B48949D2F961F5AFDFAE098A7F0A496",
    "analysis/fairness_hybrid_final_holdout_reconciliation/paper_metrics.json": "044689ABF1ADD1C1FC217FCB5F46B8D280D8659865EE3A3707EBB9FE792F2E37",
}


def _target_seed(value: Any) -> int | None:
    if type(value) is int and value in SEEDS:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) in SEEDS:
        return int(value)
    return None


def _walk_seed_records(value: Any, path: str = "$") -> list[tuple[int, str, set[str]]]:
    found: list[tuple[int, str, set[str]]] = []
    if isinstance(value, dict):
        seed = _target_seed(value.get("seed"))
        if seed is not None:
            found.append((seed, path, set(value)))
        for key, item in value.items():
            found.extend(_walk_seed_records(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_seed_records(item, f"{path}[{index}]"))
    return found


def _classify(relative: str, keys: set[str]) -> str:
    lower = relative.replace("\\", "/").lower()
    if "/instances/" in f"/{lower}" or "instance_sha256" in keys:
        return "generated_instance"
    if any(token in lower for token in ("run.json", "status.json", "checkpoint", "results")) or keys.intersection(
        {"run_key", "algorithm_status", "scientific_status", "best_x_values", "checkpoint_sha256"}
    ):
        return "solved_run"
    if lower.startswith("analysis/") or "formal" in lower or "freeze" in lower or "manifest" in lower:
        return "formal_result_access"
    return "preregistration_declaration"


def _tracked_files(root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [root / name.decode("utf-8") for name in completed.stdout.split(b"\0") if name]


def _audit_structured_file(root: Path, path: Path) -> list[dict[str, Any]]:
    relative = path.relative_to(root).as_posix()
    suffix = path.suffix.lower()
    records: list[dict[str, Any]] = []
    if suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            if any(token in relative.lower() for token in ("results", "instances", "manifest", "checkpoint", "run.json", "status.json")):
                raise RuntimeError(f"fail-closed structured seed audit could not parse {relative}") from exc
            return []
        for seed, location, keys in _walk_seed_records(value):
            records.append({"seed": seed, "path": relative, "location": location, "category": _classify(relative, keys)})
    elif suffix == ".csv":
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                for index, row in enumerate(csv.DictReader(handle), start=2):
                    seed = _target_seed(row.get("seed"))
                    if seed is not None:
                        records.append({"seed": seed, "path": relative, "location": f"row:{index}", "category": _classify(relative, set(row))})
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            if "results" in relative.lower():
                raise RuntimeError(f"fail-closed structured seed audit could not parse {relative}") from exc
    elif suffix in {".yaml", ".yml"}:
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise RuntimeError(f"fail-closed structured seed audit could not parse {relative}") from exc
        if isinstance(value, dict) and value.get("seeds") == SEEDS:
            records.append({"seed": "180-184", "path": relative, "location": "$.seeds", "category": "preregistration_declaration"})
    return records


def audit_repository_seed_access(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in _tracked_files(root):
        if path.exists() and path.suffix.lower() in {".json", ".csv", ".yaml", ".yml"}:
            entries.extend(_audit_structured_file(root, path))
    pattern = r'"seed"[[:space:]]*:[[:space:]]*(180|181|182|183|184)([^0-9]|$)|(^|,)[[:space:]]*(180|181|182|183|184)[[:space:]]*(,|$)'
    grep = subprocess.run(
        ["git", "-C", str(root), "grep", "-n", "-E", pattern, "--", "*.json", "*.csv"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if grep.returncode not in (0, 1):
        raise RuntimeError("fail-closed seed audit could not inspect tracked structured blobs")
    for line in grep.stdout.splitlines():
        relative, location, _text = line.split(":", 2)
        entries.append({
            "seed": "180-184",
            "path": relative,
            "location": f"line:{location}",
            "category": _classify(relative, {"seed"}),
        })
    known_roots = [root / "instances", root / "experiments", root / "analysis"]
    tracked = {path.resolve() for path in _tracked_files(root)}
    for known in known_roots:
        if not known.exists():
            continue
        for path in known.rglob("*"):
            if not path.is_file() or path.resolve() in tracked:
                continue
            if path.suffix.lower() in {".json", ".csv", ".yaml", ".yml"}:
                entries.extend(_audit_structured_file(root, path))
    actual = [entry for entry in entries if entry["category"] != "preregistration_declaration"]
    return {
        "tracked_and_known_root_record_count": len(entries),
        "preregistration_declarations": [entry for entry in entries if entry["category"] == "preregistration_declaration"],
        "generated_instance_evidence": [entry for entry in actual if entry["category"] == "generated_instance"],
        "solved_run_evidence": [entry for entry in actual if entry["category"] == "solved_run"],
        "formal_result_access_evidence": [entry for entry in actual if entry["category"] == "formal_result_access"],
        "actual_access_evidence_count": len(actual),
        "audit_passed": not actual,
    }


def _zip_json_records(name: str, raw: bytes) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"fail-closed ZIP seed audit could not parse {name}") from exc
    return [
        {"seed": seed, "path": name, "location": location, "category": _classify(name, keys)}
        for seed, location, keys in _walk_seed_records(value)
    ]


def _zip_csv_records(name: str, raw: bytes) -> list[dict[str, Any]]:
    try:
        rows = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
        return [
            {"seed": int(row["seed"]), "path": name, "location": f"row:{index}", "category": _classify(name, set(row))}
            for index, row in enumerate(rows, start=2)
            if _target_seed(row.get("seed")) is not None
        ]
    except (UnicodeDecodeError, csv.Error, KeyError) as exc:
        raise RuntimeError(f"fail-closed ZIP seed audit could not parse {name}") from exc


def audit_zip_seed_access(path: Path) -> dict[str, Any]:
    digest = file_sha256(path).upper()
    entries: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        bad_crc = archive.testzip()
        names = archive.namelist()
        for name in names:
            lower = name.lower()
            if lower.endswith(".json"):
                entries.extend(_zip_json_records(name, archive.read(name)))
            elif lower.endswith(".csv"):
                entries.extend(_zip_csv_records(name, archive.read(name)))
    actual = [entry for entry in entries if entry["category"] != "preregistration_declaration"]
    return {
        "path": str(path),
        "sha256": digest,
        "sha_matches_frozen_holdout": digest == FORMAL_ZIP_SHA256,
        "entry_count": len(names),
        "bad_crc_entry": bad_crc,
        "target_seed_records": entries,
        "actual_access_evidence_count": len(actual),
        "audit_passed": bad_crc is None and not actual,
    }


def static_audit(root: Path, archive: Path | None = None) -> dict[str, Any]:
    config_path = root / "experiments/configs/fairness_hybrid_gamma_sensitivity.yaml"
    protocol_path = root / "docs/fairness_hybrid_gamma_sensitivity_protocol.md"
    config = load_config(config_path)
    validate_config(config_path, config)
    plan = expand_plan()
    dry = dry_run(config_path)
    seed = audit_repository_seed_access(root)
    zip_report = audit_zip_seed_access(archive) if archive is not None and archive.exists() else None
    checks = {
        "base_commit_frozen": config["base_commit"] == BASE_COMMIT,
        "config_sha_frozen": EXPECTED_CONFIG_SHA256 != "TO_BE_FROZEN" and file_sha256(config_path).upper() == EXPECTED_CONFIG_SHA256,
        "protocol_sha_frozen": EXPECTED_PROTOCOL_SHA256 != "TO_BE_FROZEN" and file_sha256(protocol_path).upper() == EXPECTED_PROTOCOL_SHA256,
        "protected_files_unchanged": all(file_sha256(root / name).upper() == value for name, value in PROTECTED_SHA256.items()),
        "matrix_60": len(plan) == 60 and dry["baseline"] == dry["frontier"] == 30,
        "unique_identity": dry["unique_run_keys"] == dry["unique_short_directory_ids"] == 60 and dry["duplicate_or_collision_count"] == 0,
        "scenario_counts": all(scenario_count(scale, gamma) == count for scale in SCALES for gamma, count in SCALES[scale]["scenario_counts"].items()),
        "solver_envelopes": dry["algorithm_solver_limit_seconds"] == 108000 and dry["post_evaluation_solver_limit_seconds"] == 997200,
        "dry_run_zero_side_effect": dry["instances_generated"] is dry["solver_called"] is dry["output_dir_exists"] is False,
        "windows_path": dry["windows_path_check"] is True and dry["longest_windows_path_length"] < 220,
        "seed_access_clear": seed["audit_passed"] is True,
        "zip_access_clear": zip_report is None or zip_report["audit_passed"] is True,
        "formal_run_forbidden": config["formal_run_authorized"] is False,
        "no_overwrite": config["overwrite_supported"] is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Gamma sensitivity static audit failed: {failed}")
    return {
        "status": "pass",
        "passed": len(checks),
        "total": len(checks),
        "checks": checks,
        "dry_run": dry,
        "seed_access_audit": seed,
        "zip_seed_access_audit": zip_report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE if DEFAULT_ARCHIVE.exists() else None)
    args = parser.parse_args(argv)
    print(json.dumps(static_audit(args.root, args.archive), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
