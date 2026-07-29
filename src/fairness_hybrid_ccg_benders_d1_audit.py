from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from .experiment_protocol import file_sha256
from .fairness_hybrid_ccg_benders import CANDIDATE_SHA256, PROTOCOL_SHA256
from .fairness_hybrid_ccg_benders_runner import (
    ATTEMPT5_DECISION_SHA256,
    ATTEMPT5_PROVENANCE_SHA256,
    EXPECTED_CONFIG_SHA256,
    dry_run,
    load_config,
)


ARCHIVE_SHA256 = "09B41862A5BFED724EDBEC1E64996B54AA878119F5C0DEDFE5B10126B2525A98"


def audit(root: Path, *, archive: Path | None = None) -> dict[str, object]:
    protocol = root / "docs/fairness_hybrid_ccg_benders_d1_protocol.md"
    candidate = root / "experiments/configs/certified_hybrid_scenario_benders_fairness_d1_candidate.yaml"
    config_path = root / "experiments/configs/fairness_hybrid_ccg_benders_d1.yaml"
    solver = (root / "src/fairness_hybrid_ccg_benders.py").read_text(encoding="utf-8")
    runner = (root / "src/fairness_hybrid_ccg_benders_runner.py").read_text(encoding="utf-8")
    config = load_config(config_path)
    report = dry_run(config_path)
    decision = root / "analysis/fairness_hybrid_ccg_benders_d1_freeze/large_attempt5_stop_decision.json"
    provenance = root / "analysis/fairness_hybrid_ccg_benders_d1_freeze/source_archive_provenance.json"
    inventory = root / "analysis/fairness_hybrid_ccg_benders_d1_freeze/attempt5_archive_inventory.csv"
    checks = {
        "protocol_sha": file_sha256(protocol).upper() == PROTOCOL_SHA256,
        "candidate_sha": file_sha256(candidate).upper() == CANDIDATE_SHA256,
        "config_sha": file_sha256(config_path).upper() == EXPECTED_CONFIG_SHA256,
        "scenario_block_production_recourse": "_recourse_expressions(" in solver,
        "scenario_cost_cap": "first_stage + transport + shortage + service" in solver,
        "regional_fairness_cap": "hybrid_regional_service" in solver,
        "zero_demand_not_applicable": "FAIRNESS_METRIC_TOLERANCE" in solver,
        "valid_master_bound": "model.ObjBound" in solver,
        "monotone_lb": "max(float(lower_bound), master_bound)" in solver,
        "initial_t1_ub": "construct_initial_t1_upper_bound" in solver,
        "monotone_ub": "min(upper_bound, candidate_t)" in solver,
        "farkas_retained": "add_canonical_cut_payload" in solver,
        "current_point_recertification": "CertifiedAdaptiveSeparator" in solver,
        "complete_final_separation": "final_certification=True" in solver,
        "intermediate_certificate_not_terminal": "if chosen is None:" in solver and "final_separation_performed and full.robust_feasibility_certified" in solver,
        "heuristic_not_certificate": "full.robust_feasibility_certified" in solver,
        "one_scenario_per_iteration": "select_one_new_scenario" in solver,
        "no_runtime_selection": "normalized_violation_bucket" in solver and "runtime_driven_scientific_branching" in solver,
        "scenario_append_only": "scenario_order.append(digest)" in solver,
        "checkpoint_identity": "hybrid checkpoint identity mismatch" in solver,
        "checkpoint_hash": "checkpoint_sha256" in solver,
        "resume_order_validation": "checkpoint scenario order or payload drifted" in solver,
        "d1_only_gate": 'args.stage != STAGE' in runner,
        "no_overwrite_cli": 'add_argument("--overwrite"' not in runner,
        "fresh_output": "exist_ok=False" in runner,
        "production_baseline": "_production_solve_baseline" in runner,
        "baseline_checkpoint_hash": "D1 baseline checkpoint hash mismatch" in runner,
        "production_post_evaluation": "_production_post_evaluate" in runner,
        "scientific_classification": "_classify_frontier" in runner,
        "matrix_two": report["total"] == 2 and report["baseline"] == 1 and report["frontier"] == 1,
        "d1_identity": report["scale"] == "large" and report["seed"] == 160 and report["rho"] == 0.0,
        "scenario_count": report["uncertainty_scenarios"] == 4657,
        "dry_no_solver": report["solver_called"] is False,
        "dry_no_instance": report["instances_generated"] is False,
        "dry_no_output": report["output_dir_exists"] is False,
        "windows_path": report["windows_path_check"] is True and report["longest_path_length"] < 220,
        "attempt5_frozen": config["prior_large_attempt5_archive_sha256"] == ARCHIVE_SHA256,
        "attempt5_decision_sha": file_sha256(decision).upper() == ATTEMPT5_DECISION_SHA256,
        "attempt5_provenance_sha": file_sha256(provenance).upper() == ATTEMPT5_PROVENANCE_SHA256,
        "attempt5_inventory_sha": file_sha256(inventory).upper() == "6B5CE363FC2649D7DBC4249BF98AABEEE05F3C196E7E069A6F31799AB951BF6D",
        "seed_is_statistical_unit": "cluster bootstrap resamples seeds" in protocol.read_text(encoding="utf-8"),
        "holdout_excluded": config["seeds"] == [160] and "170" not in json.dumps(config),
    }
    archive_evidence = None
    if archive is not None:
        digest = file_sha256(archive).upper()
        with ZipFile(archive) as handle:
            bad = handle.testzip()
            entries = handle.infolist()
        archive_evidence = {"sha256": digest, "crc_valid": bad is None, "entries": len(entries)}
        checks["archive_sha"] = digest == ARCHIVE_SHA256
        checks["archive_crc"] = bad is None
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"D1 audit failed: {failed}")
    return {"status": "pass", "passed": len(checks), "total": len(checks), "checks": checks, "archive": archive_evidence, "dry_run": report}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--large-attempt5-zip", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root, archive=args.large_attempt5_zip)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
