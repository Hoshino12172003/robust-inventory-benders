from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment_protocol import file_sha256
from .fairness_hybrid_ccg_benders import CANDIDATE_SHA256
from .fairness_hybrid_ccg_benders_d2_runner import (
    D1_DECISION_SHA256,
    EXPECTED_CONFIG_SHA256,
    PROTOCOL_SHA256,
    dry_run,
    expand_d2_plan,
    load_config,
)


def audit(root: Path) -> dict[str, object]:
    config_path = root / "experiments/configs/fairness_hybrid_ccg_benders_d2.yaml"
    protocol_path = root / "docs/fairness_hybrid_ccg_benders_d2_protocol.md"
    candidate_path = root / "experiments/configs/certified_hybrid_scenario_benders_fairness_d1_candidate.yaml"
    decision_path = root / "analysis/fairness_hybrid_ccg_benders_d1_decision/decision.json"
    runner = (root / "src/fairness_hybrid_ccg_benders_d2_runner.py").read_text(encoding="utf-8")
    solver = (root / "src/fairness_hybrid_ccg_benders.py").read_text(encoding="utf-8")
    config = load_config(config_path)
    plan = expand_d2_plan()
    report = dry_run(config_path)
    checks = {
        "config_sha": file_sha256(config_path).upper() == EXPECTED_CONFIG_SHA256,
        "protocol_sha": file_sha256(protocol_path).upper() == PROTOCOL_SHA256,
        "candidate_sha": file_sha256(candidate_path).upper() == CANDIDATE_SHA256,
        "decision_sha": file_sha256(decision_path).upper() == D1_DECISION_SHA256,
        "d1_approved": json.loads(decision_path.read_text(encoding="utf-8"))["decision"] == "approve_for_d2_controlled_large_expansion",
        "matrix": len(plan) == 12 and report["baseline"] == 3 and report["frontier"] == 9,
        "unique_keys": len({row["run_key"] for row in plan}) == 12,
        "seeds_rhos": config["seeds"] == [160, 161, 162] and config["rho"] == [0.0, 0.01, 0.10],
        "candidate_only": {row["candidate"] for row in plan if row["task_type"] == "frontier"} == {config["candidate"]},
        "three_baselines": sum(row["task_type"] == "baseline" for row in plan) == 3,
        "nine_frontiers": sum(row["task_type"] == "frontier" for row in plan) == 9,
        "no_old_reuse": config["previous_attempt_results_reused"] is False,
        "new_output": "controlled_d2" in config["output_dir"] and "development_d1" not in config["output_dir"],
        "solver_identity": config["solver_identity"] == {"Threads": 1, "Seed": 0, "FeasibilityTol": 1e-7},
        "time_limits": config["baseline_time_limit_seconds"] == config["algorithm_time_limit_seconds"] == 1800,
        "post_identity_unambiguous": "run_execution_attempt=EXECUTION_ATTEMPT" in runner and "post_evaluation_pipeline_generation=4" in runner,
        "initial_ub_identity_projection": "initial_upper_bound_expected_identity(common, anchor)" in runner,
        "formal_gate_before_output": runner.index("_formal_git_gate(root)") < runner.index("output = root / str(config[\"output_dir\"])", runner.index("def run_d2")),
        "clean_detached_main_gate": all(text in runner for text in ('"status", "--porcelain"', "origin/main", "symbolic-ref")),
        "d2_only_gate": 'args.stage != STAGE' in runner and "only D2 is authorized" in runner,
        "no_overwrite": 'add_argument("--overwrite"' not in runner,
        "strict_resume": "D2 requires --resume" in runner and "D2 resume identity mismatch" in runner,
        "atomic_output": "atomic_write_json" in runner and "atomic_write_yaml" in runner,
        "short_directory": "run_directory_id(key)" in runner,
        "path_preflight": report["windows_path_check"] is True and report["longest_path_length"] < 220,
        "dry_no_side_effect": report["solver_called"] is False and report["instances_generated"] is False and report["output_dir_exists"] is False,
        "scenario_append_only": "scenario_order.append(digest)" in solver,
        "cut_append_only": "cut_order.append(candidate.cut_sha256)" in solver,
        "full_exact_gate": "final_separation_performed and full.robust_feasibility_certified" in solver,
        "master_bound_lb": "model.ObjBound" in solver and "max(float(lower_bound), master_bound)" in solver,
        "post_not_certificate": "algorithm_certified" in runner and "_classify_frontier" in runner,
        "uncertified_not_solved": 'scientific == "certified_robust_optimal"' in runner,
        "par2_algorithm_runtime": config["runtime_semantics"] == {"par2_multiplier": 2, "par2_basis": "algorithm_runtime"},
        "pass_gate_nine": config["pass_gate"]["frontier_certified"] == 9,
        "selective_rerun_forbidden": '"selective_rerun_authorized": False' in runner,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"D2 audit failed: {failed}")
    return {"status": "pass", "passed": len(checks), "total": len(checks), "checks": checks, "dry_run": report}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
