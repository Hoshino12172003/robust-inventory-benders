"""Read-only HG1 baseline-Gamma incident audit and deterministic freeze evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
from typing import Any

from .experiment_protocol import atomic_write_csv, atomic_write_json, file_sha256


HG1_SHA256 = "17ABAC73952D7A6C62EFAC313EC5A3771D904750BEE917BC51FBA3F1C76FDD47"
SEEDS = [185, 186, 187, 188, 189]
GAMMAS = [2, 3, 4]
SCENARIOS = {2: 211, 3: 1351, 4: 6196}


class HG1IncidentAuditError(RuntimeError):
    pass


def _check(value: bool, message: str) -> None:
    if not value:
        raise HG1IncidentAuditError(message)


def _strict_json(raw: bytes, member: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HG1IncidentAuditError(f"invalid JSON: {member}") from exc
    _check(isinstance(value, dict), f"JSON root is not object: {member}")
    return value


def _value_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().upper()


def audit_hg1(zip_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(zip_path)
    before = file_sha256(source).upper()
    _check(before == HG1_SHA256, "HG1 ZIP SHA256 mismatch")
    with zipfile.ZipFile(source) as archive:
        names = [info.filename for info in archive.infolist()]
        _check(len(names) == len(set(names)), "duplicate ZIP member")
        _check(archive.testzip() is None, "ZIP CRC failure")
        runs: list[tuple[str, dict[str, Any]]] = []
        statuses: dict[str, dict[str, Any]] = {}
        bytes_read = 0
        for info in archive.infolist():
            if info.is_dir():
                continue
            raw = archive.read(info)
            bytes_read += len(raw)
            if info.filename.endswith("/run.json"):
                runs.append((info.filename, _strict_json(raw, info.filename)))
            elif info.filename.endswith("/status.json"):
                statuses[str(PurePosixPath(info.filename).parent)] = _strict_json(raw, info.filename)
        _check(len(runs) == len(statuses) == 45, "HG1 run/status count mismatch")
        _check(len({run["run_key"] for _, run in runs}) == 45 and
               len({run["run_directory_id"] for _, run in runs}) == 45,
               "HG1 run identity duplicate")
        for name, run in runs:
            status = statuses[str(PurePosixPath(name).parent)]
            _check(run.get("state") == status.get("state") == "complete" and
                   run.get("scientific_status") == status.get("scientific_status"),
                   "HG1 run/status contradiction")

        rows: list[dict[str, Any]] = []
        baselines = [run for _, run in runs if run.get("task_type") == "baseline"]
        _check(len(baselines) == 15, "HG1 baseline count mismatch")
        for run in sorted(baselines, key=lambda value: (value["seed"], value["gamma"])):
            result = run.get("result")
            _check(isinstance(result, dict), "HG1 baseline result missing")
            schedule = result.get("gamma_schedule")
            rows.append({
                "seed": run["seed"], "requested_gamma": run["gamma"],
                "outer_scenario_count": run["scenario_count"],
                "result_projected_gamma": result.get("gamma"),
                "solver_gamma_target": result.get("gamma_target"),
                "solver_active_gamma": result.get("active_gamma"),
                "solver_gamma_schedule": schedule,
                "solver_max_scenarios": result.get("max_scenarios"),
                "lower_bound": result.get("lower_bound"), "upper_bound": result.get("upper_bound"),
                "x_sha256": _value_sha(result.get("best_x_values")),
                "y_sha256": _value_sha(result.get("best_y_values")),
                "internal_gamma_identity_valid": (
                    result.get("gamma_target") == run["gamma"] and
                    result.get("active_gamma") == run["gamma"] and
                    str(schedule) == str(run["gamma"])),
            })
        for seed in SEEDS:
            selected = [row for row in rows if row["seed"] == seed]
            _check(len(selected) == 3 and len({(row["x_sha256"], row["y_sha256"],
                   row["lower_bound"], row["upper_bound"]) for row in selected}) == 1,
                   f"HG1 baseline equality evidence changed for seed {seed}")
        _check(all(row["solver_gamma_target"] == row["solver_active_gamma"] == 2 and
                   str(row["solver_gamma_schedule"]) == "2" for row in rows),
               "HG1 internal Gamma defect evidence changed")

        scientific = {}
        for gamma in GAMMAS:
            scientific[str(gamma)] = {}
            for task in ("hybrid_frontier", "direct_extensive_frontier"):
                chosen = [run for _, run in runs if run.get("gamma") == gamma and run.get("task_type") == task]
                scientific[str(gamma)][task] = sum(
                    run.get("scientific_status") == "certified_robust_optimal" for run in chosen)
        _check(scientific == {
            "2": {"hybrid_frontier": 5, "direct_extensive_frontier": 5},
            "3": {"hybrid_frontier": 0, "direct_extensive_frontier": 0},
            "4": {"hybrid_frontier": 0, "direct_extensive_frontier": 0},
        }, "HG1 scientific status evidence changed")
        audit = {
            "schema": "fairness_high_gamma_hg1_incident_audit_v1",
            "source_archive_sha256_before": before,
            "source_archive_sha256_after": file_sha256(source).upper(),
            "zip_entry_count": len(archive.infolist()),
            "zip_file_count": sum(not info.is_dir() for info in archive.infolist()),
            "zip_directory_count": sum(info.is_dir() for info in archive.infolist()),
            "zip_duplicate_member_count": 0,
            "zip_crc_errors": 0,
            "uncompressed_bytes_read": bytes_read,
            "run_count": len(runs), "status_count": len(statuses),
            "unique_run_key_count": len({run["run_key"] for _, run in runs}),
            "unique_run_directory_count": len({run["run_directory_id"] for _, run in runs}),
            "scientific_status_by_gamma": scientific,
            "baseline_internal_gamma_mismatch_count": sum(not row["internal_gamma_identity_valid"] for row in rows),
            "same_seed_cross_gamma_identical_baseline_count": 5,
            "root_cause": "frozen_candidate_gamma_overrode_requested_gamma_before_base_config; outer result gamma label hid solver gamma_target/active_gamma/schedule",
            "incident": "baseline_gamma_identity_defect",
            "scientifically_usable_gamma2_subset": True,
            "scientifically_usable_gamma3_4": False,
            "previous_attempt_results_reused": False,
            "optimization_rerun_scope": "full_attempt2_15_baseline_15_hybrid_15_direct",
        }
    _check(audit["source_archive_sha256_after"] == before, "HG1 ZIP changed during audit")
    return audit, rows


def write_freeze(zip_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    audit, rows = audit_hg1(zip_path)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "source_archive_provenance.json", {
        "source_archive": Path(zip_path).name,
        "sha256": HG1_SHA256,
        "immutable": True,
        "archive_modified": False,
    })
    atomic_write_json(output / "hg1_incident_audit.json", audit)
    atomic_write_csv(output / "baseline_gamma_identity_by_seed.csv", rows, list(rows[0]))
    decision = {
        "decision": "freeze_hg1_and_require_full_attempt2",
        "incident": "baseline_gamma_identity_defect",
        "scientifically_usable_gamma2_subset": True,
        "scientifically_usable_gamma3_4": False,
        "previous_attempt_results_reused": False,
    }
    atomic_write_json(output / "decision.json", decision)
    (output / "decision.md").write_text(
        "# HG1 incident freeze\n\n"
        "HG1 is frozen as `baseline_gamma_identity_defect`. Gamma 2 remains scientifically usable; "
        "Gamma 3 and 4 are not scientifically usable because their baselines, anchors, and cost budgets "
        "were solved at Gamma 2. Attempt 2 restarts all 45 tasks and reuses no HG1 artifact.\n",
        encoding="utf-8", newline="\n")
    artifacts = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.name == "artifact_sha256.csv" or not path.is_file():
            continue
        artifacts.append({"artifact": path.name, "sha256": file_sha256(path).upper()})
    atomic_write_csv(output / "artifact_sha256.csv", artifacts, ["artifact", "sha256"])
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    audit = write_freeze(args.zip, args.output) if args.output else audit_hg1(args.zip)[0]
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
