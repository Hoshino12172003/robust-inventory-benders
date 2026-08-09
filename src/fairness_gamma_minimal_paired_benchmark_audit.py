from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import zipfile

from .experiment_protocol import atomic_write_json, file_sha256
from .fairness_gamma_minimal_paired_benchmark_runner import (
    SEEDS, SOURCE_CANDIDATE_SHA256, SOURCE_COMMIT, SOURCE_CONFIG_SHA256,
    SOURCE_PROTOCOL_SHA256, SOURCE_ZIP_SHA256, SCALES, _source_hybrid_key,
    _source_prefix, canonical_json, dry_run, load_catalog, load_yaml, sha256_value,
    validate_config, validate_solution_payload,
)


class SourceAuditError(RuntimeError):
    pass


def _reject_constant(value: str) -> None:
    raise SourceAuditError(f"non-finite JSON constant: {value}")


def _json(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceAuditError(f"invalid JSON: {name}") from exc
    if not isinstance(value, dict):
        raise SourceAuditError(f"JSON object required: {name}")
    return value


def _member(archive: zipfile.ZipFile, name: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = archive.read(name)
    except KeyError as exc:
        raise SourceAuditError(f"missing source member: {name}") from exc
    return raw, _json(raw, name)


def _baseline_key(scale: str, seed: int) -> str:
    return canonical_json({
        "candidate": "baseline", "execution_attempt": 3, "gamma": 2,
        "rho": "NOT_APPLICABLE", "scale": scale, "seed": seed,
        "stage": "GAMMA_SENSITIVITY", "task_type": "baseline",
    })


def build_source_catalog(source_zip: str | Path) -> dict[str, Any]:
    source_zip = Path(source_zip)
    before = file_sha256(source_zip).upper()
    if before != SOURCE_ZIP_SHA256:
        raise SourceAuditError("Gamma Attempt 3 source ZIP SHA mismatch")
    cells: list[dict[str, Any]] = []
    with zipfile.ZipFile(source_zip) as archive:
        if archive.testzip() is not None:
            raise SourceAuditError("Gamma Attempt 3 source ZIP CRC failure")
        for scale in SCALES:
            prefix = _source_prefix(scale)
            _, manifest = _member(archive, f"{prefix}/manifest.json")
            expected_manifest = {
                "stage": "GAMMA_SENSITIVITY", "scale": scale, "seeds": list(SEEDS),
                "gamma": [0, 1, 2], "rho": [0.025], "execution_attempt": 3,
                "previous_attempt_results_reused": False, "git_commit": SOURCE_COMMIT,
                "config_file_sha256": SOURCE_CONFIG_SHA256,
                "protocol_sha256": SOURCE_PROTOCOL_SHA256,
                "candidate_sha256": SOURCE_CANDIDATE_SHA256,
                "solver_parameters": {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7},
            }
            for field, wanted in expected_manifest.items():
                if manifest.get("identity", {}).get(field) != wanted:
                    raise SourceAuditError(f"source manifest {scale} {field} mismatch")
            for seed in SEEDS:
                hkey = _source_hybrid_key(scale, seed)
                bkey = _baseline_key(scale, seed)
                mapping = manifest.get("run_key_to_directory_id", {})
                hdir, bdir = mapping.get(hkey), mapping.get(bkey)
                if not isinstance(hdir, str) or not isinstance(bdir, str):
                    raise SourceAuditError("source run mapping missing")
                _, hybrid = _member(archive, f"{prefix}/runs/{hdir}/run.json")
                _, hstatus = _member(archive, f"{prefix}/runs/{hdir}/status.json")
                _, baseline = _member(archive, f"{prefix}/runs/{bdir}/run.json")
                _, bstatus = _member(archive, f"{prefix}/runs/{bdir}/status.json")
                instance_name = f"{prefix}/instances/s{seed}_g2.json"
                instance_raw, instance_archive = _member(archive, instance_name)
                instance = instance_archive.get("instance")
                identity = instance_archive.get("identity")
                if not isinstance(instance, dict) or not isinstance(identity, dict):
                    raise SourceAuditError("source instance archive schema mismatch")
                canonical_sha = sha256_value(instance)
                identity_sha = sha256_value(identity)
                file_sha = hashlib.sha256(instance_raw).hexdigest().upper()
                frozen_instance = manifest.get("instance_identities", {}).get(f"s{seed}_g2")
                anchor = manifest.get("baseline_anchors", {}).get(f"s{seed}_g2")
                if not isinstance(frozen_instance, dict) or not isinstance(anchor, dict):
                    raise SourceAuditError("source instance/anchor manifest evidence missing")
                if frozen_instance.get("instance_archive_file_sha256") != file_sha:
                    raise SourceAuditError("source instance file SHA mismatch")
                if frozen_instance.get("instance_canonical_sha256") != canonical_sha:
                    raise SourceAuditError("source instance canonical SHA mismatch")
                if frozen_instance.get("instance_identity_sha256") != identity_sha:
                    raise SourceAuditError("source instance identity SHA mismatch")
                if hybrid.get("run_key") != hkey or baseline.get("run_key") != bkey:
                    raise SourceAuditError("source run key mismatch")
                if hybrid.get("scientific_status") != "certified_robust_optimal" or hstatus.get("state") != "complete":
                    raise SourceAuditError("source Hybrid cell not certified complete")
                if baseline.get("scientific_status") != "certified_robust_optimal" or bstatus.get("state") != "complete":
                    raise SourceAuditError("source baseline not optimal complete")
                if hybrid.get("baseline_run_key") != bkey or anchor.get("baseline_run_key") != bkey:
                    raise SourceAuditError("source baseline reference mismatch")
                upper = float(baseline["result"]["upper_bound"])
                if not baseline["result"].get("valid_UB") or upper.hex() != anchor.get("anchor_value_hex"):
                    raise SourceAuditError("source baseline upper-bound/anchor mismatch")
                if hybrid.get("anchor_sha256") != anchor.get("anchor_sha256"):
                    raise SourceAuditError("source Hybrid/anchor SHA mismatch")
                result = hybrid.get("result", {})
                post = result.get("post_evaluation", {})
                validate_solution_payload(baseline.get("result", {}), instance, baseline=True)
                validate_solution_payload(result, instance)
                if post.get("valid") is not True or post.get("errors") != []:
                    raise SourceAuditError("source Hybrid post-evaluation invalid")
                cells.append({
                    "scale": scale, "seed": seed, "gamma": 2, "rho": "0.025",
                    "source_hybrid_run_key": hkey, "source_hybrid_directory_id": hdir,
                    "source_hybrid_scientific_status": hybrid["scientific_status"],
                    "source_hybrid_algorithm_runtime": result["algorithm_runtime"],
                    "source_hybrid_par2": result.get("penalized_runtime_par2", result["algorithm_runtime"]),
                    "source_hybrid_iterations": result["iterations"],
                    "source_hybrid_scenario_blocks": result["metadata"]["committed_scenario_count"],
                    "source_hybrid_certified_farkas_cuts": result["cuts"],
                    "source_hybrid_objective_t": result["objective_t"],
                    "source_hybrid_actual_robust_cost": post["actual_robust_cost"],
                    "instance_member": instance_name,
                    "instance_canonical_sha256": canonical_sha,
                    "instance_identity_sha256": identity_sha,
                    "instance_file_sha256": file_sha,
                    "baseline_run_key": bkey, "baseline_scientific_status": baseline["scientific_status"],
                    "baseline_upper_bound": upper, "anchor_value": anchor["value"],
                    "anchor_value_hex": anchor["anchor_value_hex"], "anchor_sha256": anchor["anchor_sha256"],
                })
    after = file_sha256(source_zip).upper()
    if after != before:
        raise SourceAuditError("source ZIP changed during read-only audit")
    return {
        "schema": "fairness_gamma_minimal_paired_source_catalog_v1",
        "source_zip_sha256": before, "source_git_commit": SOURCE_COMMIT,
        "source_config_sha256": SOURCE_CONFIG_SHA256, "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "source_candidate_sha256": SOURCE_CANDIDATE_SHA256, "source_execution_attempt": 3,
        "source_previous_attempt_results_reused": False, "source_zip_unchanged": True,
        "cell_count": len(cells), "cells": cells,
    }


def static_audit(root: str | Path, source_zip: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    config_path = root / "experiments/configs/fairness_gamma_minimal_paired_benchmark.yaml"
    config = load_yaml(config_path)
    validate_config(config_path, config)
    frozen = load_catalog(config)
    rebuilt = build_source_catalog(source_zip)["cells"]
    candidate = load_yaml(root / config["candidate_definition"])
    protected = (
        "src/benders.py", "src/scenarios.py", "src/fairness_benders.py",
        "src/fairness_hybrid_ccg_benders.py",
        "analysis/fairness_hybrid_gamma_sensitivity_attempt3_final/results.combined.csv",
        "analysis/fairness_hybrid_gamma_sensitivity_attempt3_final/freeze_manifest.json",
    )
    diff = subprocess.run(
        ["git", "diff", "--name-only", "dde1a4608ec74b3e0e6aadbdfeca55de7767a08a", "--", *protected],
        cwd=root, text=True, capture_output=True, check=True,
    ).stdout.strip()
    before = "gurobipy" in sys.modules
    dry = dry_run(config_path)
    checks = {
        "source_catalog_exact": rebuilt == frozen,
        "source_zip_unchanged": file_sha256(source_zip).upper() == SOURCE_ZIP_SHA256,
        "protected_files_zero_diff": diff == "",
        "matrix_10": dry["reference_frontier"] == 10 and dry["baseline_new"] == dry["hybrid_new"] == 0,
        "dry_run_no_solver_import": dry["gurobipy_imported_by_dry_run"] is False and ("gurobipy" in sys.modules) == before,
        "dry_run_zero_side_effect": not dry["instances_generated"] and not dry["solver_called"] and not dry["output_dir_exists"],
        "windows_path": dry["longest_windows_path_length"] < 220,
        "single_cut": candidate.get("fairness_scalability_strategy") == "single_cut",
        "complete_blocks_disabled": candidate.get("complete_scenario_recourse_blocks_enabled") is False,
        "persistent_and_cache_disabled": candidate.get("persistent_separation_enabled") is False and candidate.get("certified_scenario_cache_enabled") is False,
        "final_exact_required": candidate.get("final_exact_separation_required") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SourceAuditError(f"benchmark static audit failed: {failed}")
    return {"checks": checks, "decision": "approve_benchmark_protocol", "source_pairing_cells": 10, "dry_run": dry}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", required=True)
    parser.add_argument("--output")
    parser.add_argument("--static-root")
    args = parser.parse_args(argv)
    if args.static_root:
        report = static_audit(args.static_root, args.source_zip)
        if args.output:
            atomic_write_json(Path(args.output), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    catalog = build_source_catalog(args.source_zip)
    if args.output:
        atomic_write_json(Path(args.output), catalog)
    print(json.dumps(catalog, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
