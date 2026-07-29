from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any
from zipfile import ZipFile

from .fairness_large_final_remediation import canonical_json_bytes


D2_ARCHIVE_SHA256 = "07746988F93B2CD6E6BDE7B08EB661FCB330B3CA7A3A874A772BE9ED24258271"
D2_GIT_COMMIT = "1a56f23c659f9a5ddaa352017b6ee87d9e9eeaf7"
D2_CONFIG_SHA256 = "ED8F145A9ACAA1AC799DBBDE2BAEBF1A35F2F614FE41B0F73EF8F278690EF63A"
D2_PROTOCOL_SHA256 = "A1D1655F4D66B79ADB9AF28E69F8E04D50F0EAEFB8577F645080D5713D1426BC"
CANDIDATE_SHA256 = "8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0"
CANDIDATE = "certified_hybrid_scenario_benders_fairness"
EXPECTED_SEEDS = [160, 161, 162]
EXPECTED_RHOS = [0.0, 0.01, 0.10]
EXPECTED_SCENARIOS = 4657
EXPECTED_CHUNKS_PER_FRONTIER = 187


class D2ArchiveAuditError(RuntimeError):
    pass


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _normalized_name(name: str) -> str:
    normalized = PurePosixPath(name).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/")


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise D2ArchiveAuditError(f"nonfinite value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def _run_directory_id(run_key: str) -> str:
    return "r_" + hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:24]


def _read_json(files: dict[str, bytes], name: str) -> dict[str, Any]:
    try:
        value = json.loads(files[name].decode("utf-8"))
    except KeyError as exc:
        raise D2ArchiveAuditError(f"missing {name}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D2ArchiveAuditError(f"invalid JSON {name}") from exc
    if not isinstance(value, dict):
        raise D2ArchiveAuditError(f"non-object JSON {name}")
    _assert_finite(value, name)
    return value


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise D2ArchiveAuditError(message)


def _checkpoint_valid(checkpoint: dict[str, Any]) -> bool:
    payload = deepcopy(checkpoint)
    digest = payload.pop("checkpoint_sha256", None)
    return digest == _canonical_hash(payload)


def audit_d2_archive(
    archive_path: str | Path,
    *,
    expected_sha256: str = D2_ARCHIVE_SHA256,
) -> dict[str, Any]:
    archive = Path(archive_path)
    before = _file_sha256(archive)
    _check(before == expected_sha256.upper(), "D2 archive SHA256 mismatch")
    with ZipFile(archive, "r") as source:
        bad_crc = source.testzip()
        _check(bad_crc is None, f"D2 archive CRC failure: {bad_crc}")
        entries = source.infolist()
        names = [_normalized_name(entry.filename) for entry in entries if not entry.is_dir()]
        _check(len(names) == len(set(names)), "duplicate normalized ZIP path")
        files = {
            _normalized_name(entry.filename): source.read(entry)
            for entry in entries
            if not entry.is_dir()
        }
    after = _file_sha256(archive)
    _check(after == before, "D2 archive changed during audit")

    json_count = 0
    csv_count = 0
    for name, payload in files.items():
        if name.endswith(".json"):
            json.loads(payload.decode("utf-8"))
            json_count += 1
        elif name.endswith(".csv"):
            list(csv.reader(io.StringIO(payload.decode("utf-8"))))
            csv_count += 1

    manifest = _read_json(files, "manifest.json")
    identity = manifest.get("identity", {})
    expected_identity = {
        "candidate": CANDIDATE,
        "candidate_sha256": CANDIDATE_SHA256,
        "config_file_sha256": D2_CONFIG_SHA256,
        "execution_attempt": 3,
        "git_commit": D2_GIT_COMMIT,
        "previous_attempt_results_reused": False,
        "protocol_sha256": D2_PROTOCOL_SHA256,
        "rhos": EXPECTED_RHOS,
        "scale": "large",
        "seeds": EXPECTED_SEEDS,
        "solver_parameters": {"FeasibilityTol": 1e-7, "Seed": 0, "Threads": 1},
        "stage": "D2",
    }
    for key, value in expected_identity.items():
        _check(identity.get(key) == value, f"manifest identity mismatch: {key}")
    _check(manifest.get("completed_run_count") == 12, "manifest completed count mismatch")
    _check(manifest.get("certified_solved_count") == 12, "manifest certified count mismatch")
    gate = manifest.get("d2_gate", {})
    _check(gate.get("passed") is True, "D2 gate did not pass")
    _check(gate.get("baseline_certified") == 3, "D2 baseline gate mismatch")
    _check(gate.get("frontier_certified") == 9, "D2 frontier gate mismatch")
    _check(gate.get("selective_rerun_authorized") is False, "selective rerun unexpectedly authorized")

    forward = manifest.get("run_key_to_directory_id", {})
    reverse = manifest.get("directory_id_to_run_key", {})
    _check(len(forward) == len(reverse) == 12, "manifest mapping count mismatch")
    _check({value: key for key, value in forward.items()} == reverse, "manifest reverse mapping drift")

    records: list[dict[str, Any]] = []
    post_chunk_total = 0
    post_scenario_total = 0
    for run_key, directory_id in sorted(forward.items()):
        _check(directory_id == _run_directory_id(run_key), "run directory hash mismatch")
        root = f"runs/{directory_id}"
        record = _read_json(files, f"{root}/run.json")
        status = _read_json(files, f"{root}/status.json")
        parsed_key = json.loads(run_key)
        _check(record.get("run_key") == run_key, "run key mismatch")
        _check(record.get("run_directory_id") == directory_id, "run directory identity mismatch")
        _check(record.get("state") == status.get("state") == "complete", "run/status state mismatch")
        _check(record.get("scientific_status") == status.get("scientific_status"), "scientific status mismatch")
        _check(record.get("algorithm_status") == status.get("algorithm_status"), "algorithm status mismatch")
        _check(record.get("scientific_status") == "certified_robust_optimal", "uncertified D2 record")
        _check(record.get("algorithm_status") == "optimal", "non-optimal D2 algorithm status")
        _check(parsed_key["execution_attempt"] == record.get("execution_attempt") == 3, "attempt mismatch")
        _check(parsed_key["seed"] == record.get("seed"), "seed mismatch")
        _check(record.get("git_commit") == D2_GIT_COMMIT, "run Git mismatch")
        _check(record.get("config_file_sha256") == D2_CONFIG_SHA256, "run config mismatch")
        _check(record.get("candidate_sha256") == CANDIDATE_SHA256, "run candidate mismatch")
        result = record.get("result", {})
        runtime = float(result.get("algorithm_runtime"))
        _check(runtime >= 0.0, "invalid algorithm runtime")
        _check(float(result.get("penalized_runtime_par2")) == runtime, "solved PAR-2 mismatch")

        if record.get("task_type") == "baseline":
            _check(parsed_key["candidate"] == "baseline", "baseline key mismatch")
            _check(result.get("status") == "optimal" and result.get("valid_UB") is True, "invalid baseline")
            _check(float(result.get("gap")) <= 1e-4, "baseline gap exceeds tolerance")
            _check(f"{root}/baseline_checkpoint.json" in files, "missing baseline checkpoint")
        else:
            _check(record.get("candidate") == CANDIDATE, "frontier candidate mismatch")
            _check(float(record.get("rho")) in EXPECTED_RHOS, "frontier rho mismatch")
            checkpoint = _read_json(files, f"{root}/algorithm_checkpoint.json")
            _check(_checkpoint_valid(checkpoint), "algorithm checkpoint hash mismatch")
            checkpoint_identity = checkpoint.get("identity", {})
            _check(checkpoint_identity.get("run_key") == run_key, "checkpoint run key mismatch")
            _check(checkpoint_identity.get("candidate_sha256") == CANDIDATE_SHA256, "checkpoint candidate mismatch")
            _check(checkpoint_identity.get("protocol_sha256") == D2_PROTOCOL_SHA256, "checkpoint protocol mismatch")
            state = checkpoint.get("state", {})
            scenarios = state.get("committed_scenario_sha256_values", [])
            scenario_payloads = state.get("scenario_payloads_by_sha256", {})
            cuts = state.get("committed_farkas_cut_sha256_values", [])
            cut_payloads = state.get("cut_payloads_by_sha256", {})
            _check(len(scenarios) == len(set(scenarios)) and set(scenarios) == set(scenario_payloads), "scenario checkpoint drift")
            _check(len(cuts) == len(set(cuts)) and set(cuts) == set(cut_payloads), "cut checkpoint drift")
            log = result.get("iteration_log", [])
            _check(log and log[-1].get("final_exact_separation_performed") is True, "missing final exact separation")
            _check(log[-1].get("robust_feasibility_certified") is True, "final separation not certified")
            counts = [int(item["scenario_count"]) for item in log]
            _check(all(current >= previous for previous, current in zip(counts, counts[1:])), "scenario count decreased")
            _check(all(current - previous <= 1 for previous, current in zip(counts, counts[1:])), "more than one scenario committed per iteration")
            metadata = result.get("metadata", {})
            _check(metadata.get("robust_feasibility_certified") is True, "metadata robust certificate missing")
            _check(metadata.get("committed_scenario_sha256_values") == scenarios, "run/checkpoint scenario mismatch")
            _check(metadata.get("committed_farkas_cut_sha256_values") == cuts, "run/checkpoint cut mismatch")
            lower = float(result.get("lower_bound"))
            upper = float(result.get("upper_bound"))
            gap_value = float(result.get("gap"))
            _check(lower <= upper + 1e-9, "invalid final bounds")
            _check(abs(gap_value - (upper - lower) / max(1.0, abs(upper))) <= 1e-12, "gap recomputation mismatch")
            evaluation = result.get("post_evaluation", {})
            _check(evaluation.get("valid") is True, "invalid post-evaluation")
            _check(evaluation.get("scenario_count") == EXPECTED_SCENARIOS, "post-evaluation scenario mismatch")
            _check(evaluation.get("objective_t_consistent") is True, "post-evaluation T mismatch")
            _check(evaluation.get("errors") == [], "post-evaluation errors present")
            final_post = _read_json(files, f"{root}/post_evaluation/post_evaluation.json")
            _check(final_post.get("evaluation") == evaluation, "post-evaluation final record mismatch")
            index = _read_json(files, f"{root}/post_evaluation/checkpoint/index.json")
            chunks = index.get("chunks", [])
            _check(len(chunks) == EXPECTED_CHUNKS_PER_FRONTIER, "post chunk count mismatch")
            _check([entry["chunk_index"] for entry in chunks] == list(range(EXPECTED_CHUNKS_PER_FRONTIER)), "post chunk order mismatch")
            scenario_cursor = 0
            for entry in chunks:
                chunk_name = f"{root}/post_evaluation/{entry['relative_path']}"
                payload = files.get(chunk_name)
                _check(payload is not None, f"missing {chunk_name}")
                _check(hashlib.sha256(payload).hexdigest() == entry["sha256"], "post chunk SHA mismatch")
                chunk = _read_json(files, chunk_name)
                _check(chunk.get("chunk_index") == entry["chunk_index"], "chunk index mismatch")
                _check(chunk.get("scenario_start") == scenario_cursor, "chunk scenario order mismatch")
                _check(len(chunk.get("records", [])) == entry["scenario_count"], "chunk record count mismatch")
                scenario_cursor += entry["scenario_count"]
                _check(chunk.get("scenario_end_exclusive") == scenario_cursor, "chunk end mismatch")
            _check(scenario_cursor == EXPECTED_SCENARIOS, "post chunk scenario total mismatch")
            post_chunk_total += len(chunks)
            post_scenario_total += scenario_cursor

        records.append(record)

    baselines = [record for record in records if record["task_type"] == "baseline"]
    frontiers = [record for record in records if record["task_type"] == "frontier"]
    _check(len(baselines) == 3 and len(frontiers) == 9, "D2 task matrix mismatch")
    _check(sorted(record["seed"] for record in baselines) == EXPECTED_SEEDS, "baseline seed matrix mismatch")
    _check(
        sorted((record["seed"], float(record["rho"])) for record in frontiers)
        == sorted((seed, rho) for seed in EXPECTED_SEEDS for rho in EXPECTED_RHOS),
        "frontier matrix mismatch",
    )
    anchors = manifest.get("baseline_anchors", {})
    for seed in EXPECTED_SEEDS:
        seed_frontiers = [record for record in frontiers if record["seed"] == seed]
        _check(len({record["instance_sha256"] for record in seed_frontiers}) == 1, "instance not shared by seed")
        _check(len({record["baseline_run_key"] for record in seed_frontiers}) == 1, "baseline not shared by seed")
        _check(len({record["anchor_sha256"] for record in seed_frontiers}) == 1, "anchor not shared by seed")
        _check(seed_frontiers[0]["anchor_sha256"] == anchors[str(seed)]["anchor_sha256"], "manifest anchor mismatch")

    results_rows = list(csv.DictReader(io.StringIO(files["results.csv"].decode("utf-8"))))
    summary_rows = list(csv.DictReader(io.StringIO(files["summary.csv"].decode("utf-8"))))
    _check(len(results_rows) == 12, "results.csv row count mismatch")
    _check(all(row["scientific_status"] == "certified_robust_optimal" for row in results_rows), "results.csv status mismatch")
    _check(summary_rows, "summary.csv is empty")

    run_summary = []
    for record in sorted(records, key=lambda item: (item["seed"], item["task_type"], str(item["rho"]))):
        result = record["result"]
        run_summary.append(
            {
                "task_type": record["task_type"],
                "seed": record["seed"],
                "rho": record["rho"],
                "scientific_status": record["scientific_status"],
                "algorithm_status": record["algorithm_status"],
                "algorithm_runtime": result.get("algorithm_runtime"),
                "total_wall_runtime": result.get("total_wall_runtime"),
                "penalized_runtime_par2": result.get("penalized_runtime_par2"),
                "iterations": result.get("iterations"),
                "cuts": result.get("cuts"),
                "lower_bound": result.get("lower_bound"),
                "upper_bound": result.get("upper_bound"),
                "gap": result.get("gap"),
                "objective_t": result.get("objective_t"),
                "committed_scenario_count": result.get("metadata", {}).get("committed_scenario_count"),
                "post_evaluation_valid": result.get("post_evaluation", {}).get("valid"),
                "post_evaluation_scenario_count": result.get("post_evaluation", {}).get("scenario_count"),
            }
        )

    return {
        "status": "pass",
        "source_archive_sha256": before,
        "archive_sha256_after_audit": after,
        "crc_valid": True,
        "entry_count": len(entries),
        "file_count": sum(not entry.is_dir() for entry in entries),
        "directory_count": sum(entry.is_dir() for entry in entries),
        "json_file_count": json_count,
        "csv_file_count": csv_count,
        "run_count": len(records),
        "baseline_certified_count": len(baselines),
        "frontier_certified_count": len(frontiers),
        "post_evaluation_valid_count": len(frontiers),
        "post_evaluation_chunk_count": post_chunk_total,
        "post_evaluation_scenario_total": post_scenario_total,
        "git_commit": identity["git_commit"],
        "config_file_sha256": identity["config_file_sha256"],
        "protocol_sha256": identity["protocol_sha256"],
        "candidate_sha256": identity["candidate_sha256"],
        "execution_attempt": identity["execution_attempt"],
        "d2_gate": gate,
        "run_summary": run_summary,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_freeze_evidence(audit: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_sha = audit["source_archive_sha256"]
    decision = {
        "algorithm_development_closed": True,
        "candidate_frozen": CANDIDATE,
        "complete_scenario_blocks_frozen": True,
        "d1_d2_role": "development_evidence_only",
        "d2_baseline_certified": "3/3",
        "d2_frontier_certified": "9/9",
        "decision": "approve_final_cross_scale_holdout_protocol",
        "farkas_certification_frozen": True,
        "final_exact_separation_frozen": True,
        "final_holdout_formal_run_authorized": False,
        "next_authorized_stage": "fairness_hybrid_final_cross_scale_holdout_review_only",
        "source_archive_sha256": source_sha,
        "time_limits_and_tolerances_frozen": True,
        "tuning_closed": True,
    }
    provenance = {
        "archive_sha256": source_sha,
        "archive_sha256_after_audit": audit["archive_sha256_after_audit"],
        "crc_valid": audit["crc_valid"],
        "directory_count": audit["directory_count"],
        "entry_count": audit["entry_count"],
        "file_count": audit["file_count"],
        "original_archive_committed_to_git": False,
        "read_only_audit": True,
    }
    _write_json(output / "decision.json", decision)
    _write_json(output / "source_archive_provenance.json", provenance)
    _write_json(output / "d2_audit.json", audit)
    fields = list(audit["run_summary"][0])
    with (output / "d2_run_summary.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit["run_summary"])
    artifacts = {}
    for name in ("decision.json", "source_archive_provenance.json", "d2_audit.json", "d2_run_summary.csv"):
        artifacts[name] = _file_sha256(output / name)
    with (output / "artifact_sha256.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(["relative_path", "sha256"])
        for name, digest in sorted(artifacts.items()):
            writer.writerow([name, digest])
    return {**artifacts, "artifact_sha256.csv": _file_sha256(output / "artifact_sha256.csv")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = audit_d2_archive(args.archive)
    result: dict[str, Any] = {"audit": audit}
    if args.output is not None:
        result["artifacts"] = write_freeze_evidence(audit, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
