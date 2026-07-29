from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .experiment_protocol import file_sha256
from .fairness_hybrid_ccg_benders import CANDIDATE, CANDIDATE_SHA256, initial_scenario_plan_identity
from .fairness_large_final_remediation_runner import RemediationGateError
from .fairness_scalability_runner import run_directory_id


STAGE = "FINAL_HOLDOUT"
EXECUTION_ATTEMPT = 1
EXPECTED_CONFIG_SHA256 = "662A106D25B1E8E5A467167046AD8C8B87AD3CD5257FBB0620B477C0CEB73920"
PROTOCOL_SHA256 = "F4B2556575CFB397F0FED75E59773EEC96855EFD3F7604BED6B288AA90AC909D"
D2_DECISION_SHA256 = "E73B01404908BCC49E30A85B5795B30DF9AC927BDFA5F685FA95EF4B5DE607E1"
SEEDS = list(range(170, 180))
RHOS = [0.0, 0.01, 0.025, 0.05, 0.10]
SCALES = {
    "medium_large": {"num_regions": 10, "num_products": 6, "scenario_count": 1831, "output_dir": "experiments/results_fairness_hybrid_final_holdout/ml_a1"},
    "large": {"num_regions": 12, "num_products": 8, "scenario_count": 4657, "output_dir": "experiments/results_fairness_hybrid_final_holdout/lg_a1"},
}


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RemediationGateError("final holdout config must be a mapping")
    return value


def validate_config(path: str | Path, config: dict[str, Any]) -> None:
    source = Path(path)
    if file_sha256(source).upper() != EXPECTED_CONFIG_SHA256:
        raise RemediationGateError("final holdout config SHA mismatch")
    expected = {
        "stage": STAGE,
        "authorization": "protocol_review_only",
        "formal_run_authorized": False,
        "schema_version": 1,
        "execution_attempt": EXECUTION_ATTEMPT,
        "previous_attempt_results_reused": False,
        "d2_decision": "approve_final_cross_scale_holdout_protocol",
        "required_d2_decision_sha256": D2_DECISION_SHA256,
        "required_protocol_sha256": PROTOCOL_SHA256,
        "required_candidate_sha256": CANDIDATE_SHA256,
        "candidate": CANDIDATE,
        "seeds": SEEDS,
        "rho": RHOS,
        "baseline_count": 20,
        "frontier_count": 100,
        "total_tasks": 120,
        "solver_identity": {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7},
        "gamma": 2,
        "tol": 1e-4,
        "baseline_time_limit_seconds": 1800,
        "algorithm_time_limit_seconds": 1800,
        "general_time_limit_seconds": 1800,
        "post_evaluation": {"time_limit_per_scenario_seconds": 30, "checkpoint_chunk_size": 25},
        "runtime_semantics": {"par2_multiplier": 2, "par2_basis": "algorithm_runtime"},
        "resume": True,
        "overwrite_supported": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RemediationGateError(f"final holdout identity drifted: {key}")
    root = Path(__file__).resolve().parents[1]
    if file_sha256(root / config["protocol_document"]).upper() != PROTOCOL_SHA256:
        raise RemediationGateError("final holdout protocol SHA mismatch")
    if file_sha256(root / config["candidate_definition"]).upper() != CANDIDATE_SHA256:
        raise RemediationGateError("final holdout candidate SHA mismatch")
    decision_path = root / config["d2_decision_file"]
    if file_sha256(decision_path).upper() != D2_DECISION_SHA256:
        raise RemediationGateError("D2 decision SHA mismatch")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != "approve_final_cross_scale_holdout_protocol":
        raise RemediationGateError("D2 decision does not authorize holdout protocol")
    for scale, frozen in SCALES.items():
        configured = config.get("scales", {}).get(scale, {})
        if configured != {**frozen, "baseline_count": 10, "frontier_count": 50, "total_tasks": 60}:
            raise RemediationGateError(f"final holdout scale identity drifted: {scale}")
    statistics = config.get("statistics", {})
    if statistics.get("independent_unit") != "seed" or statistics.get("prohibit_seed_rho_independence") is not True:
        raise RemediationGateError("final holdout statistical unit drifted")


def expand_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scale in SCALES:
        for seed in SEEDS:
            tasks = [("baseline", "baseline", "NOT_APPLICABLE")]
            tasks.extend(("frontier", CANDIDATE, f"{rho:.3f}".rstrip("0").rstrip(".")) for rho in RHOS)
            for task_type, candidate, rho in tasks:
                key = json.dumps(
                    {
                        "candidate": candidate,
                        "execution_attempt": EXECUTION_ATTEMPT,
                        "rho": rho,
                        "scale": scale,
                        "seed": seed,
                        "stage": STAGE,
                        "task_type": task_type,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                rows.append(
                    {
                        "scale": scale,
                        "seed": seed,
                        "rho": rho,
                        "candidate": candidate,
                        "task_type": task_type,
                        "run_key": key,
                        "run_directory_id": run_directory_id(key),
                    }
                )
    return rows


def _planned_paths(root: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for scale, frozen in SCALES.items():
        output = root / frozen["output_dir"]
        for name in ("manifest.json", ".manifest.json.tmp", "resolved_config.yaml", "results.csv", "summary.csv", "audit.log"):
            paths.append((f"{scale}_root", output / name))
        for seed in SEEDS:
            paths.append((f"{scale}_instance", output / f"instances/{seed}.json"))
        chunk_index = math_ceil(frozen["scenario_count"], int(config["post_evaluation"]["checkpoint_chunk_size"])) - 1
        for row in (item for item in rows if item["scale"] == scale):
            run = output / "runs" / row["run_directory_id"]
            for kind, name in (
                ("run", "run.json"),
                ("run_tmp", ".run.json.tmp"),
                ("status", "status.json"),
                ("algorithm_checkpoint", "algorithm_checkpoint.json"),
                ("post_index", "post_evaluation/checkpoint/index.json"),
                ("post_chunk_tmp", f"post_evaluation/checkpoint/.chunk_{chunk_index:05d}.json.tmp"),
            ):
                paths.append((f"{scale}_{kind}", run / name))
    return [(kind, path.resolve()) for kind, path in paths]


def math_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def structured_seed_access_evidence(root: Path) -> list[str]:
    evidence: list[str] = []
    excluded = {".git", ".pytest_cache", "__pycache__"}
    for path in root.rglob("*.json"):
        if any(part in excluded for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith("analysis/fairness_hybrid_ccg_benders_d2_decision/"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            value = json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if _contains_access_identity(value):
            evidence.append(relative)
    for scale in SCALES:
        output = root / SCALES[scale]["output_dir"]
        for seed in SEEDS:
            if (output / f"instances/{seed}.json").exists():
                evidence.append((output / f"instances/{seed}.json").relative_to(root).as_posix())
    return sorted(set(evidence))


def _contains_access_identity(value: Any) -> bool:
    if isinstance(value, dict):
        seed = value.get("seed")
        identity_keys = {"run_key", "instance_sha256", "algorithm_status", "checkpoint_sha256", "best_x_values"}
        if isinstance(seed, int) and seed in SEEDS and identity_keys.intersection(value):
            return True
        return any(_contains_access_identity(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_access_identity(item) for item in value)
    return False


def dry_run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config_path, config)
    root = Path(__file__).resolve().parents[1]
    rows = expand_plan()
    paths = _planned_paths(root, config, rows)
    longest_kind, longest = max(paths, key=lambda item: len(str(item[1])))
    by_scale = {}
    for scale, frozen in SCALES.items():
        scale_rows = [row for row in rows if row["scale"] == scale]
        by_scale[scale] = {
            "baseline": sum(row["task_type"] == "baseline" for row in scale_rows),
            "frontier": sum(row["task_type"] == "frontier" for row in scale_rows),
            "total": len(scale_rows),
            "scenario_count": frozen["scenario_count"],
            "output_dir_exists": (root / frozen["output_dir"]).exists(),
            **initial_scenario_plan_identity(
                num_regions=frozen["num_regions"], num_products=frozen["num_products"], gamma=2
            ),
        }
    access = structured_seed_access_evidence(root)
    return {
        "stage": STAGE,
        "candidate": CANDIDATE,
        "seeds": SEEDS,
        "rho": RHOS,
        "baseline": 20,
        "frontier": 100,
        "total": len(rows),
        "unique_run_keys": len({row["run_key"] for row in rows}),
        "duplicate_run_keys": len(rows) - len({row["run_key"] for row in rows}),
        "by_scale": by_scale,
        "independent_unit": "seed",
        "seed_cluster_count": 10,
        "reserved_seed_access_evidence": access,
        "reserved_seed_access_audit_passed": not access,
        "instances_generated": False,
        "solver_called": False,
        "output_dirs_exist": any(value["output_dir_exists"] for value in by_scale.values()),
        "windows_path_check": len(str(longest)) < 220,
        "longest_path_type": longest_kind,
        "longest_path": longest.relative_to(root).as_posix(),
        "longest_path_length": len(str(longest)),
        "formal_run_authorized": False,
    }


def write_protocol_evidence(config_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = dry_run(config_path)
    rows = expand_plan()
    plan_path = output / "frozen_run_plan.csv"
    with plan_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "dry_run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    seed_audit = {
        "access_evidence": report["reserved_seed_access_evidence"],
        "audit_passed": report["reserved_seed_access_audit_passed"],
        "declaration_only_is_not_access": True,
        "formal_run_authorized": False,
        "pre_run_reaudit_required": True,
        "reserved_seeds": SEEDS,
    }
    (output / "seed_access_audit.json").write_text(
        json.dumps(seed_audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    result = {}
    for name in ("dry_run.json", "frozen_run_plan.csv", "seed_access_audit.json"):
        result[name] = file_sha256(output / name).upper()
    return result


def run_holdout(config_path: str | Path) -> None:
    config = load_config(config_path)
    validate_config(config_path, config)
    raise RemediationGateError("formal_run_not_authorized: final holdout requires independent review and pre-run authorization")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.stage != STAGE:
        raise RemediationGateError("only FINAL_HOLDOUT protocol dry-run is available")
    if args.dry_run:
        report: dict[str, Any] = {"dry_run": dry_run(args.config)}
        if args.evidence_output is not None:
            report["artifacts"] = write_protocol_evidence(args.config, args.evidence_output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    run_holdout(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
