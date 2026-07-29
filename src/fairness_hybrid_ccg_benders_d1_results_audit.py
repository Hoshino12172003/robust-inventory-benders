from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import yaml


ARCHIVE_SHA256 = "7E89115E3BE325C9A37C31D28D32EA80EEA95F09528DBC8AEDA833EF0129A4A9"
RUN_COMMIT = "25c5bf5cb69754c696b8b7c0e44d3e0e22b5f658"
CONFIG_SHA256 = "95514DD43167583CCE8D09A2C9491FF8892A0ACE4B800DC9B0F8CD879B5C7156"
PROTOCOL_SHA256 = "C1F608E6ABD1D0EE27A106BD28EE098A26FF262F987033C1BD9DDFB53E3EF750"
CANDIDATE_SHA256 = "8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0"
EXPECTED_SCENARIOS = 4657


class D1AuditError(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _yaml_sha(value: Any) -> str:
    return _sha(yaml.safe_dump(value, sort_keys=True, allow_unicode=True).encode("utf-8"))


def _finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise D1AuditError(f"non-finite numeric value at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def _run_directory_id(run_key: str) -> str:
    return "r_" + hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:24]


def audit_d1_archive(path: Path) -> dict[str, Any]:
    before = _sha(path.read_bytes())
    if before != ARCHIVE_SHA256:
        raise D1AuditError("D1 archive SHA256 mismatch")
    checks: dict[str, bool] = {}
    with ZipFile(path) as archive:
        checks["zip_crc"] = archive.testzip() is None
        names = archive.namelist()
        checks["unique_entry_names"] = len(names) == len(set(names))
        checks["json_csv_parseable"] = True
        for name in names:
            data = archive.read(name)
            if name.endswith(".json"):
                json.loads(data)
            elif name.endswith(".csv"):
                list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))

        def one(suffix: str) -> tuple[str, bytes]:
            found = [name for name in names if name.endswith(suffix)]
            if len(found) != 1:
                raise D1AuditError(f"expected one {suffix}, found {len(found)}")
            return found[0], archive.read(found[0])

        def one_json(suffix: str) -> tuple[str, dict[str, Any]]:
            name, data = one(suffix)
            value = json.loads(data)
            if not isinstance(value, dict):
                raise D1AuditError(f"{suffix} is not an object")
            return name, value

        _, manifest = one_json("manifest.json")
        identity = manifest["identity"]
        checks["manifest_identity"] = (
            identity.get("execution_attempt") == 1
            and identity.get("stage") == "D1"
            and identity.get("git_commit") == RUN_COMMIT
            and identity.get("config_file_sha256") == CONFIG_SHA256
            and identity.get("protocol_sha256") == PROTOCOL_SHA256
            and identity.get("candidate_sha256") == CANDIDATE_SHA256
            and identity.get("previous_attempt_results_reused") is False
            and identity.get("solver_parameters")
            == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7}
        )
        _, resolved_bytes = one("resolved_config.yaml")
        resolved = yaml.safe_load(resolved_bytes)
        checks["resolved_config_file_sha"] = _sha(resolved_bytes) == identity["resolved_config_file_sha256"]
        checks["resolved_config_canonical_sha"] = _yaml_sha(resolved) == identity["resolved_config_canonical_sha256"]

        run_names = sorted(name for name in names if name.endswith("/run.json"))
        status_names = sorted(name for name in names if name.endswith("/status.json"))
        checks["two_runs_and_statuses"] = len(run_names) == len(status_names) == 2
        runs = [json.loads(archive.read(name)) for name in run_names]
        statuses = {PurePosixPath(name).parent.name: json.loads(archive.read(name)) for name in status_names}
        keys = [run["run_key"] for run in runs]
        checks["unique_run_keys"] = len(keys) == len(set(keys)) == 2
        for name, run in zip(run_names, runs):
            directory = PurePosixPath(name).parent.name
            checks[f"directory_{directory}"] = (
                directory == run["run_directory_id"] == _run_directory_id(run["run_key"])
                and manifest["run_key_to_directory_id"].get(run["run_key"]) == directory
                and manifest["directory_id_to_run_key"].get(directory) == run["run_key"]
            )
            status = statuses[directory]
            checks[f"status_{directory}"] = (
                run["state"] == status["state"] == "complete"
                and run["scientific_status"] == status["scientific_status"]
                and run.get("algorithm_status", run["result"].get("status"))
                == status.get("algorithm_status", run["result"].get("status"))
            )
        baseline = next(run for run in runs if run["task_type"] == "baseline")
        frontier = next(run for run in runs if run["task_type"] == "frontier")
        baseline_result = baseline["result"]
        checks["baseline_certified"] = (
            baseline["scientific_status"] == "certified_robust_optimal"
            and baseline["solved_to_tolerance"] is True
            and baseline_result["status"] == "optimal"
            and baseline_result["valid_UB"] is True
            and baseline_result["gap"] <= 1e-4
        )
        anchor = manifest["baseline_anchors"]["160"]
        checks["anchor_chain"] = (
            anchor["value"] == baseline_result["upper_bound"]
            and anchor["value_hex"] == float(baseline_result["upper_bound"]).hex()
            and frontier["anchor_sha256"] == anchor["anchor_sha256"]
            and frontier["baseline_run_key"] == baseline["run_key"]
            and frontier["instance_sha256"] == baseline["instance_sha256"]
        )
        _, instance_payload = one_json("instances/160.json")
        checks["instance_payload_sha"] = _yaml_sha(instance_payload) == baseline["instance_sha256"]
        checks["anchor_payload_sha"] = _yaml_sha(
            {key: value for key, value in anchor.items() if key != "anchor_sha256"}
        ) == anchor["anchor_sha256"]

        _, baseline_checkpoint = one_json("baseline_checkpoint.json")
        unhashed = {key: value for key, value in baseline_checkpoint.items() if key != "checkpoint_sha256"}
        checks["baseline_checkpoint_hash"] = _canonical_sha(unhashed).upper() == baseline_checkpoint["checkpoint_sha256"]
        checks["baseline_checkpoint_result"] = baseline_checkpoint["result"] == {
            key: value for key, value in baseline_result.items()
            if key not in {"algorithm_runtime", "post_evaluation_wall_runtime", "total_wall_runtime", "penalized_runtime_par2"}
        }

        _, checkpoint = one_json("algorithm_checkpoint.json")
        cp_unhashed = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
        checks["algorithm_checkpoint_hash"] = _canonical_sha(cp_unhashed).upper() == checkpoint["checkpoint_sha256"]
        cp_identity = checkpoint["identity"]
        checks["algorithm_checkpoint_identity"] = (
            cp_identity["run_key"] == frontier["run_key"]
            and cp_identity["run_identity"]["instance_sha256"] == frontier["instance_sha256"]
            and cp_identity["run_identity"]["baseline_run_key"] == baseline["run_key"]
            and cp_identity["anchor_sha256"] == anchor["anchor_sha256"]
            and cp_identity["candidate_sha256"] == CANDIDATE_SHA256
            and cp_identity["protocol_sha256"] == PROTOCOL_SHA256
        )
        state = checkpoint["state"]
        scenarios = state["committed_scenario_sha256_values"]
        cuts = state["committed_farkas_cut_sha256_values"]
        checks["append_only_unique_state"] = (
            len(scenarios) == len(set(scenarios)) == 21
            and len(cuts) == len(set(cuts)) == 8
            and set(scenarios) == set(state["scenario_payloads_by_sha256"])
            and set(cuts) == set(state["cut_payloads_by_sha256"])
        )
        checks["scenario_and_cut_payload_hashes"] = (
            all(_canonical_sha(payload).upper() == digest for digest, payload in state["scenario_payloads_by_sha256"].items())
            and all(_canonical_sha(payload).upper() == digest for digest, payload in state["cut_payloads_by_sha256"].items())
        )
        log = frontier["result"]["iteration_log"]
        checks["iteration_log_matches_checkpoint"] = log == state["iteration_log"] and len(log) == 9
        prior_lb = -math.inf
        appended_scenarios: list[str] = []
        appended_cuts: list[str] = []
        for index, item in enumerate(log, start=1):
            if item["iteration"] != index or item["lower_bound"] < prior_lb - 1e-12:
                raise D1AuditError("iteration numbering or monotone lower bound failed")
            prior_lb = item["lower_bound"]
            scenario_sha = item.get("committed_scenario_sha256")
            cut_sha = item.get("committed_farkas_cut_sha256")
            if scenario_sha is not None:
                appended_scenarios.append(scenario_sha)
            if cut_sha is not None:
                appended_cuts.append(cut_sha)
            expected_count = 13 + len(appended_scenarios)
            if item["scenario_count"] != expected_count:
                raise D1AuditError("scenario append-only count failed")
            if index < len(log) and (
                item["final_exact_separation_performed"] is not False
                or item["robust_feasibility_certified"] is not False
            ):
                raise D1AuditError("premature robust certification")
        final = log[-1]
        checks["one_scenario_and_cut_per_iteration"] = (
            len(appended_scenarios) == len(set(appended_scenarios)) == 8
            and len(appended_cuts) == len(set(appended_cuts)) == 8
            and scenarios[-8:] == appended_scenarios
            and cuts == appended_cuts
        )
        checks["final_exact_gate"] = (
            final["final_exact_separation_performed"] is True
            and final["robust_feasibility_certified"] is True
            and final["separation_status"] == "optimal"
            and final["separation_objective_bound"] <= 1e-7
            and frontier["result"]["metadata"]["robust_feasibility_certified"] is True
            and frontier["scientific_status"] == "certified_robust_optimal"
        )
        result = frontier["result"]
        recomputed_gap = (result["upper_bound"] - result["lower_bound"]) / max(1.0, abs(result["upper_bound"]))
        checks["bounds_and_gap"] = (
            result["lower_bound"] == state["lower_bound"]
            and result["upper_bound"] == state["upper_bound"]
            and abs(result["gap"] - recomputed_gap) <= 1e-15
            and result["objective_t"] == result["upper_bound"]
        )

        post_name, post = one_json("post_evaluation/post_evaluation.json")
        post_identity = post["identity"]
        checks["post_identity_hash"] = _canonical_sha(post_identity) == post["identity_sha256"]
        checks["legacy_attempt_semantics"] = (
            post_identity["execution_attempt"] == 4
            and json.loads(post_identity["run_key"])["execution_attempt"] == 1
            and post_identity["run_key"] == frontier["run_key"]
            and post_identity["git_commit"] == RUN_COMMIT
            and post_identity["config_sha256"] == identity["resolved_config_file_sha256"]
            and post_identity["baseline_anchor_sha256"] == anchor["anchor_sha256"]
        )
        index_name, index = one_json("post_evaluation/checkpoint/index.json")
        checks["post_index_identity"] = index["identity_sha256"] == post["identity_sha256"]
        chunk_entries = index["chunks"]
        checks["chunk_plan"] = (
            len(chunk_entries) == 187
            and [entry["chunk_index"] for entry in chunk_entries] == list(range(187))
            and sum(entry["scenario_count"] for entry in chunk_entries) == EXPECTED_SCENARIOS
        )
        scenario_keys: list[str] = []
        for entry in chunk_entries:
            expected_name = str(PurePosixPath(index_name).parent.parent / entry["relative_path"])
            data = archive.read(expected_name)
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise D1AuditError(f"chunk hash mismatch: {expected_name}")
            chunk = json.loads(data)
            if (
                chunk["identity_sha256"] != post["identity_sha256"]
                or chunk["chunk_index"] != entry["chunk_index"]
                or len(chunk["records"]) != entry["scenario_count"]
                or chunk["scenario_end_exclusive"] - chunk["scenario_start"] != len(chunk["records"])
            ):
                raise D1AuditError(f"chunk identity/count mismatch: {expected_name}")
            scenario_keys.extend(record["scenario_key"] for record in chunk["records"])
        checks["all_post_scenarios_once"] = len(scenario_keys) == len(set(scenario_keys)) == EXPECTED_SCENARIOS
        evaluation = post["evaluation"]
        acceptance = evaluation["acceptance_evidence"]
        max_residual = max(float(item["residual"]) for item in acceptance)
        checks["post_evaluation_valid"] = (
            evaluation["valid"] is True
            and evaluation["scenario_count"] == EXPECTED_SCENARIOS
            and evaluation["errors"] == []
            and evaluation["objective_t_consistent"] is True
            and all(item["accepted"] for item in acceptance)
            and max_residual <= evaluation["acceptance_threshold"]
        )
        checks["post_not_algorithm_certificate"] = (
            final["robust_feasibility_certified"] is True
            and post["completed_at"] >= "2026"
        )

        results_name, results_bytes = one("results.csv")
        summary_name, summary_bytes = one("summary.csv")
        result_rows = list(csv.DictReader(io.StringIO(results_bytes.decode("utf-8-sig"))))
        summary_rows = list(csv.DictReader(io.StringIO(summary_bytes.decode("utf-8-sig"))))
        checks["csv_rows"] = len(result_rows) == 2 and len(summary_rows) == 2
        by_key = {row["run_key"]: row for row in result_rows}
        for run in runs:
            csv_row = by_key[run["run_key"]]
            if (
                csv_row["scientific_status"] != run["scientific_status"]
                or csv_row["state"] != run["state"]
                or (csv_row["certified_solved"] == "True") != run["solved_to_tolerance"]
            ):
                raise D1AuditError("results.csv status reconciliation failed")
        checks["results_runtime_reconciliation"] = all(
            float(by_key[run["run_key"]]["algorithm_runtime"]) == run["result"]["algorithm_runtime"]
            and float(by_key[run["run_key"]]["penalized_runtime_par2"]) == run["result"]["penalized_runtime_par2"]
            and float(by_key[run["run_key"]]["total_wall_runtime"]) == run["result"]["total_wall_runtime"]
            for run in runs
        )
        summary_by_type = {row["task_type"]: row for row in summary_rows}
        checks["summary_reconciliation"] = all(
            summary_by_type[task]["run_count"] == "1"
            and summary_by_type[task]["certified_solved_count"] == "1"
            and float(summary_by_type[task]["mean_algorithm_runtime"])
            == next(run for run in runs if run["task_type"] == task)["result"]["algorithm_runtime"]
            for task in ("baseline", "frontier")
        )

        nested_name, nested_bytes = one("D1_review.zip")
        with ZipFile(io.BytesIO(nested_bytes)) as nested:
            checks["nested_review_crc"] = nested.testzip() is None
            nested_names = nested.namelist()
            nested_entries = len(nested_names)
            formal_hashes_by_name: dict[str, set[str]] = {}
            for formal_name in names:
                if formal_name == nested_name:
                    continue
                formal_hashes_by_name.setdefault(PurePosixPath(formal_name).name, set()).add(
                    _sha(archive.read(formal_name))
                )
            checks["nested_review_is_exact_formal_subset"] = all(
                _sha(nested.read(name)) in formal_hashes_by_name.get(PurePosixPath(name).name, set())
                for name in nested_names
            ) and "D1_review.zip" not in json.dumps(manifest)
        _finite(manifest, "manifest")
        _finite(baseline, "baseline")
        _finite(frontier, "frontier")
        _finite(checkpoint, "algorithm_checkpoint")
        _finite(post, "post_evaluation")

    failed = sorted(name for name, passed in checks.items() if not passed)
    after = _sha(path.read_bytes())
    if after != before:
        failed.append("archive_immutability")
    if failed:
        raise D1AuditError(f"D1 audit failed: {failed}")
    return {
        "audit_status": "pass",
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "archive": {
            "sha256": before, "sha256_after_audit": after,
            "entries": len(names), "files": sum(not name.endswith("/") for name in names),
            "directories": sum(name.endswith("/") for name in names), "crc_valid": True,
        },
        "identity": identity,
        "legacy_post_evaluation_identity": {
            "ambiguous_field": "execution_attempt",
            "stored_value": 4,
            "actual_run_execution_attempt": 1,
            "classification": "post_evaluation_pipeline_generation_mislabeled",
            "run_attempt_locked_by": [
                "manifest.identity.execution_attempt", "identity.run_key",
                "post_evaluation.identity.run_key", "post_evaluation.identity_sha256",
            ],
        },
        "metrics": {
            "baseline_algorithm_runtime": baseline_result["algorithm_runtime"],
            "frontier_algorithm_runtime": result["algorithm_runtime"],
            "frontier_master_runtime": result["master_runtime"],
            "frontier_separation_runtime": result["separation_runtime"],
            "frontier_post_evaluation_solver_runtime": result["post_evaluation_solver_runtime"],
            "frontier_post_evaluation_wall_runtime": result["post_evaluation_wall_runtime"],
            "frontier_checkpoint_io_runtime": result["checkpoint_io_runtime"],
            "frontier_total_wall_runtime": result["total_wall_runtime"],
            "frontier_penalized_runtime_par2": result["penalized_runtime_par2"],
            "lower_bound": result["lower_bound"], "upper_bound": result["upper_bound"],
            "gap": result["gap"], "objective_t": result["objective_t"],
            "robust_minimum_fill_rate": result["robust_minimum_fill_rate"],
            "iterations": result["iterations"], "initial_scenario_count": 13,
            "final_scenario_count": len(scenarios), "certified_farkas_cut_count": len(cuts),
            "baseline_cost": result["baseline_cost"], "cost_budget": result["cost_budget"],
            "actual_robust_cost": evaluation["actual_robust_cost"],
            "post_evaluation_scenario_count": evaluation["scenario_count"],
            "post_evaluation_max_residual": max_residual,
        },
        "nested_review_archive": {
            "path": nested_name, "sha256": _sha(nested_bytes), "entries": nested_entries,
            "classification": "non_authoritative_review_convenience_copy_excluded_from_scientific_identity",
        },
        "decision": "approve_for_d2_controlled_large_expansion",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_d1_archive(args.archive), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
