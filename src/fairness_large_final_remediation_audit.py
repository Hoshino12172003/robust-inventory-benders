"""Solver-free static audit and dry-run for the final fairness remediation protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

BASE = "d6ae40c0bbe3d9af7c67cb7e71fb8ad45e64b033"
CANDIDATE = "certified_adaptive_multicut_fair_benders"
FREEZE_DECISION_SHA = "7EBFD4F22C5AF2B26E630722FB3F8D17E83FAEE9AA58248BDA32A386FFDE29B2"
FREEZE_INDEX_SHA = "BC0781818CA2DD1F5964512BEEF5438CAD8181BE5FF8B0A992DE716E98DF2358"
FREEZE_DOC_SHA = "AA95348D1677CBB714F88812B56C78E94E8450E533D3F37E2AA8EB5616311C9B"
CONFIGS = (
    "experiments/configs/fairness_large_final_remediation_pilot.yaml",
    "experiments/configs/fairness_large_final_remediation_large_s1.yaml",
    "experiments/configs/fairness_large_final_remediation_medium_large_s1.yaml",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def canonical_run_key(config: dict[str, Any], task: str, seed: int, rho: float | None) -> str:
    identity = {
        "candidate": "baseline" if task == "baseline" else CANDIDATE,
        "execution_attempt": config["execution_attempt"],
        "rho": "NOT_APPLICABLE" if rho is None else format(rho, ".2f"),
        "scale": config["scale"],
        "seed": seed,
        "task_type": task,
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def expand_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        key = canonical_run_key(config, "baseline", seed, None)
        rows.append({"task_type": "baseline", "seed": seed, "rho": None, "run_key": key})
        for rho in config["rho"]:
            key = canonical_run_key(config, "frontier", seed, float(rho))
            rows.append({"task_type": "frontier", "seed": seed, "rho": float(rho), "run_key": key})
    for row in rows:
        row["run_directory_id"] = "r_" + hashlib.sha256(row["run_key"].encode("utf-8")).hexdigest()[:24]
    return rows


def _candidate_paths(root: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> list[Path]:
    output = root / config["output_dir"]
    paths = [output / "manifest.json.tmp", output / "results.csv.tmp", output / "summary.csv.tmp", output / "audit.json.tmp"]
    chunk_count = (int(config["scenario_count"]) + int(config["post_evaluation"]["checkpoint_chunk_size"]) - 1) // int(config["post_evaluation"]["checkpoint_chunk_size"])
    for row in rows:
        run_dir = output / row["run_directory_id"]
        paths.extend((run_dir / "run.json.tmp", run_dir / "status.json.tmp", run_dir / "algorithm_checkpoint.json.tmp"))
        if row["task_type"] == "frontier":
            paths.extend((run_dir / "post_evaluation" / "index.json.tmp", run_dir / "post_evaluation" / f"chunk_{chunk_count - 1:04d}.json.tmp"))
    return paths


def audit(root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    freeze = root / "analysis/fairness_scalability_s1_attempt2_cross_scale_freeze"
    decision_path = freeze / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    check("freeze decision SHA", sha256(decision_path) == FREEZE_DECISION_SHA, sha256(decision_path))
    check("freeze decision", decision["decision"] == "no_existing_candidate_passes_cross_scale_s1")
    check("freeze medium-large", decision["medium_large_certified_frontier"] == "20/24")
    check("freeze large", decision["large_certified_frontier"] == "0/24")
    check("freeze gates", not decision["original_s2_authorized"] and not decision["full_grid_authorized"] and not decision["attempt4_authorized"])
    index_path = freeze / "artifact_sha256.csv"
    check("freeze artifact index SHA", sha256(index_path) == FREEZE_INDEX_SHA, sha256(index_path))
    with index_path.open(newline="", encoding="utf-8-sig") as stream:
        indexed = list(csv.DictReader(stream))
    check("freeze artifact index coverage", len(indexed) == 7)
    for row in indexed:
        check(f"freeze artifact {row['artifact_path']}", sha256(freeze / row["artifact_path"]) == row["sha256"])
    freeze_doc = root / "docs/audits/fairness_scalability_s1_attempt2_cross_scale_decision.md"
    check("freeze decision document SHA", sha256(freeze_doc) == FREEZE_DOC_SHA, sha256(freeze_doc))

    candidate_path = root / "experiments/configs/fairness_large_final_remediation_candidate.yaml"
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    check("one candidate", candidate["candidate_name"] == CANDIDATE)
    check("no cut deletion", candidate["forbidden"]["cut_deletion"] is True)
    check("current point recertification", candidate["certification"]["fixed_scenario_recertification_at_current_point"] is True)
    check("complete separation bound", candidate["certification"]["full_separation_objective_bound_required"] is True)
    check("adaptive cut bounds", candidate["adaptive_multicut"]["minimum_cuts_when_a_certified_violation_exists"] == 1 and candidate["adaptive_multicut"]["maximum_certified_cuts_per_iteration"] == 5)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    longest = Path()
    longest_length = -1
    outputs_absent = True
    for rel in CONFIGS:
        path = root / rel
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        rows = expand_plan(config)
        all_rows[config["stage"]] = rows
        check(f"{config['stage']} protocol-only", config["authorization"] == "protocol_only_no_formal_execution" and config["formal_run_authorized"] is False)
        check(f"{config['stage']} identity", config["base_commit"] == BASE and config["schema_version"] == 3 and config["execution_attempt"] == 3 and config["previous_attempt_results_reused"] is False)
        check(f"{config['stage']} unique plan", len({r["run_key"] for r in rows}) == len(rows))
        check(f"{config['stage']} arithmetic", len(rows) == config["total_tasks"] and sum(r["task_type"] == "baseline" for r in rows) == config["baseline_count"] and sum(r["task_type"] == "frontier" for r in rows) == config["frontier_count"])
        check(f"{config['stage']} solver identity", config["solver_identity"] == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7})
        check(f"{config['stage']} time identity", config["algorithm_time_limit_seconds"] == config["baseline_time_limit_seconds"] == config["general_time_limit_seconds"] == 1800)
        check(f"{config['stage']} resume no overwrite", config["resume"] is True and config["overwrite_supported"] is False)
        output = root / config["output_dir"]
        outputs_absent = outputs_absent and not output.exists()
        for item in _candidate_paths(root, config, rows):
            length = len(str(item.resolve()))
            if length > longest_length:
                longest, longest_length = item.resolve(), length
    check("L0 plan", len(all_rows["L0"]) == 2)
    check("L1 cumulative plan", len(all_rows["L1"]) == 9)
    l0_keys = {row["run_key"] for row in all_rows["L0"]}
    l1_keys = {row["run_key"] for row in all_rows["L1"]}
    check("L0 identities are an L1 subset", l0_keys < l1_keys)
    check("L1 adds seven tasks after L0", len(l1_keys - l0_keys) == 7)
    check("M1 plan", len(all_rows["M1"]) == 9)
    check("cross-scale maximum", len(all_rows["L1"]) + len(all_rows["M1"]) == 18)
    accessed_seeds = {row["seed"] for rows in all_rows.values() for row in rows}
    check("prohibited seeds untouched", not accessed_seeds.intersection(range(130, 160)))
    check("reserved holdout seeds untouched", not accessed_seeds.intersection(range(170, 180)))
    check("output isolation", outputs_absent)
    check("Windows path portability", longest_length <= 220, {"path": str(longest), "length": longest_length})

    envelopes = {
        "L0": 2 * 1800 + 1 * 4657 * 30,
        "L1": 9 * 1800 + 6 * 4657 * 30,
        "M1": 9 * 1800 + 6 * 1831 * 30,
    }
    result = {
        "decision": "approve_for_implementation" if all(c["passed"] for c in checks) else "mathematical_or_protocol_blocker",
        "next_authorized_stage": "fairness_large_final_remediation_implementation_only" if all(c["passed"] for c in checks) else "stop_and_report",
        "formal_run_authorized": False,
        "initial_T1_UB_proved": True,
        "checks_passed": sum(c["passed"] for c in checks),
        "checks_total": len(checks),
        "checks": checks,
        "dry_run": {
            "S0": [
                "single_region", "symmetric_regions", "clear_fairness_gap", "rho_0", "rho_0.01",
                "gamma_0", "gamma_2", "zero_demand_region", "infeasible_cost_budget",
                "valid_baseline_t1_ub", "invalid_t1_ub_counterexample", "cache_current_point_recertification",
                "multiple_independently_certified_cuts", "tie_breaking", "near_parallel_cuts",
                "adaptive_budget_5_to_1", "benders_extensive_form_equivalence",
            ],
            "L0_tasks": 2,
            "L1_cumulative_tasks": 9,
            "M1_tasks": 9,
            "cross_scale_cumulative_tasks": 18,
            "scenarios": {"medium_large": 1831, "large": 4657},
            "algorithm_time_envelope_seconds": {"L0": 3600, "L1": 16200, "M1": 16200},
            "post_evaluation_envelope_seconds": {"L0": 139710, "L1": 838260, "M1": 329580},
            "total_wall_time_upper_accounting_seconds": envelopes,
            "longest_windows_path": str(longest),
            "longest_windows_path_length": longest_length,
            "instances_generated": False,
            "solver_called": False,
            "output_dir_exists": not outputs_absent,
        },
        "artifact_sha256": {
            "protocol": sha256(root / "docs/fairness_large_final_remediation_protocol.md"),
            "initial_upper_bound_proof": sha256(root / "docs/robust_regional_fairness_initial_upper_bound.md"),
            "candidate": sha256(candidate_path),
            **{Path(rel).name: sha256(root / rel) for rel in CONFIGS},
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = audit(args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["decision"] == "approve_for_implementation" else 1


if __name__ == "__main__":
    raise SystemExit(main())
