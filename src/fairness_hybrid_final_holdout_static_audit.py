from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment_protocol import file_sha256
from .fairness_hybrid_final_holdout_runner import (
    CANDIDATE_SHA256,
    D2_DECISION_SHA256,
    EXPECTED_CONFIG_SHA256,
    PROTOCOL_SHA256,
    dry_run,
    expand_plan,
    load_config,
)


PROTECTED_SHA256 = {
    "src/fairness_hybrid_ccg_benders.py": "216CD3A5EF9FB599F625FF5A13F56E93F6F3865981605DFD8F67E27AC3CE5C81",
    "src/fairness_benders.py": "289BEDB61939C9FF2FE2AA116209CA441745FCDE3BDD4E85309688A720DC372E",
    "src/robust_regional_fairness.py": "25A641F0D898E4C3113D06747CE1311FFE4E67BA339B8B4D2FDE11687A260318",
    "src/benders.py": "37967750EE1AAD5575A9B1FE0B050F012EC21DB58FA277FBEFAA5A48CFEF1D9F",
    "src/scenarios.py": "7294C60DC318F7678F8A4464DAF2CBD85E540842C6C3858BB1D30A9DE7915511",
    "experiments/configs/certified_hybrid_scenario_benders_fairness_d1_candidate.yaml": CANDIDATE_SHA256,
}


def audit(root: Path) -> dict[str, object]:
    config_path = root / "experiments/configs/fairness_hybrid_final_cross_scale_holdout.yaml"
    protocol_path = root / "docs/fairness_hybrid_final_cross_scale_holdout_protocol.md"
    decision_path = root / "analysis/fairness_hybrid_ccg_benders_d2_decision/decision.json"
    config = load_config(config_path)
    plan = expand_plan()
    report = dry_run(config_path)
    protocol = protocol_path.read_text(encoding="utf-8")
    delivery = (root / "docs/fairness_hybrid_final_cross_scale_holdout_delivery.md").read_text(encoding="utf-8")
    runner = (root / "src/fairness_hybrid_final_holdout_runner.py").read_text(encoding="utf-8")
    checks = {
        "config_sha": file_sha256(config_path).upper() == EXPECTED_CONFIG_SHA256,
        "protocol_sha": file_sha256(protocol_path).upper() == PROTOCOL_SHA256,
        "decision_sha": file_sha256(decision_path).upper() == D2_DECISION_SHA256,
        "decision_approved": json.loads(decision_path.read_text(encoding="utf-8"))["decision"] == "approve_final_cross_scale_holdout_protocol",
        "protected_files_frozen": all(file_sha256(root / name).upper() == digest for name, digest in PROTECTED_SHA256.items()),
        "matrix_120": len(plan) == report["total"] == 120,
        "matrix_by_scale": all(report["by_scale"][scale][key] == value for scale in ("medium_large", "large") for key, value in (("baseline", 10), ("frontier", 50), ("total", 60))),
        "unique_run_keys": report["unique_run_keys"] == 120 and report["duplicate_run_keys"] == 0,
        "holdout_seeds_only": config["seeds"] == list(range(170, 180)),
        "five_rhos": config["rho"] == [0.0, 0.01, 0.025, 0.05, 0.10],
        "candidate_only": {row["candidate"] for row in plan if row["task_type"] == "frontier"} == {config["candidate"]},
        "solver_identity": config["solver_identity"] == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7},
        "time_limits": config["baseline_time_limit_seconds"] == config["algorithm_time_limit_seconds"] == 1800,
        "post_parameters": config["post_evaluation"] == {"time_limit_per_scenario_seconds": 30, "checkpoint_chunk_size": 25},
        "scenario_counts": report["by_scale"]["medium_large"]["scenario_count"] == 1831 and report["by_scale"]["large"]["scenario_count"] == 4657,
        "formal_authorized_after_merge": config["authorization"] == "formal_execution_authorized_after_merge" and config["formal_run_authorized"] is True,
        "production_pipeline_connected": all(token in runner for token in ("def _run_scale", "solve_certified_hybrid_scenario_benders_fairness", "checkpointed_fairness_post_evaluation", "_aggregate_records")),
        "gate_before_solver_and_writes": runner.index("_formal_git_gate(root)", runner.index("def run_holdout")) < runner.index("deps.configure_solver", runner.index("def run_holdout")) < runner.index("_run_scale(", runner.index("def run_holdout")),
        "detached_clean_main_gate": all(token in runner for token in ("worktree is not clean", "HEAD is not current origin/main", "worktree must be detached")),
        "dry_no_side_effect": report["instances_generated"] is False and report["solver_called"] is False and report["output_dirs_exist"] is False,
        "path_portable": report["windows_path_check"] is True and report["longest_path_length"] < 220,
        "seed_access_clear": report["reserved_seed_access_audit_passed"] is True,
        "seed_is_statistical_unit": config["statistics"]["independent_unit"] == "seed" and "independent experimental unit is the seed" in protocol,
        "cluster_bootstrap": config["statistics"]["cluster_bootstrap_resamples_seed"] is True and "both scales and all five rho values move together" in protocol,
        "ten_pairs_per_rho": config["statistics"]["per_rho_cross_scale_pair_count"] == 10,
        "holm_frozen": config["statistics"]["multiple_testing"] == "Holm_if_five_rho_tests",
        "no_task_independence": config["statistics"]["prohibit_seed_rho_independence"] is True and "must never be treated as 100 independent observations" in protocol,
        "development_closed": "closes algorithm development and tuning" in (root / "docs/fairness_hybrid_ccg_benders_d2_final_decision.md").read_text(encoding="utf-8"),
        "d1_d2_development_only": "D1 and D2 remain development evidence only" in (root / "docs/fairness_hybrid_ccg_benders_d2_final_decision.md").read_text(encoding="utf-8"),
        "no_overwrite": 'add_argument("--overwrite"' not in runner and config["overwrite_supported"] is False,
        "strict_resume": "FINAL_HOLDOUT requires --resume" in runner,
        "cross_scale_outputs": all(scale in config["scales"] for scale in ("medium_large", "large")),
        "delivery_uses_only_frozen_entrypoint": "--stage FINAL_HOLDOUT --resume" in delivery and "--overwrite" not in delivery,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"final holdout static audit failed: {failed}")
    return {"status": "pass", "passed": len(checks), "total": len(checks), "checks": checks, "dry_run": report}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
