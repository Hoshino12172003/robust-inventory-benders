"""Solver-free audit for the final remediation implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .experiment_protocol import file_sha256
from .fairness_large_final_remediation import (
    CANDIDATE,
    CANDIDATE_SHA256,
    INITIAL_UB_THEOREM_SHA256,
    PROTOCOL_SHA256,
)
from .fairness_large_final_remediation_runner import (
    EXPECTED_FILE_SHA256,
    dry_run_remediation,
)


FROZEN = {
    "docs/fairness_large_final_remediation_protocol.md": PROTOCOL_SHA256,
    "docs/robust_regional_fairness_initial_upper_bound.md": INITIAL_UB_THEOREM_SHA256,
    "experiments/configs/fairness_large_final_remediation_candidate.yaml": CANDIDATE_SHA256,
    "experiments/configs/fairness_large_final_remediation_pilot.yaml": EXPECTED_FILE_SHA256["L0"],
    "experiments/configs/fairness_large_final_remediation_large_s1.yaml": EXPECTED_FILE_SHA256["L1"],
    "experiments/configs/fairness_large_final_remediation_medium_large_s1.yaml": EXPECTED_FILE_SHA256["M1"],
}
CONFIGS = {
    "L0": "experiments/configs/fairness_large_final_remediation_pilot.yaml",
    "L1": "experiments/configs/fairness_large_final_remediation_large_s1.yaml",
    "M1": "experiments/configs/fairness_large_final_remediation_medium_large_s1.yaml",
}


def audit(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for relative, expected in FROZEN.items():
        actual = file_sha256(root / relative).upper()
        check(f"frozen SHA {relative}", actual == expected, actual)

    implementation = (root / "src/fairness_large_final_remediation.py").read_text(encoding="utf-8")
    runner = (root / "src/fairness_large_final_remediation_runner.py").read_text(encoding="utf-8")
    tests = (root / "tests/test_fairness_large_final_remediation_implementation.py").read_text(encoding="utf-8")
    required_implementation_tokens = {
        "sole candidate": f'CANDIDATE = "{CANDIDATE}"',
        "initial UB fail closed": "initial_upper_bound_assumption_failure",
        "initial UB evidence": "initial_robust_ub_assumption_checks",
        "persistent separation": "PersistentFairnessSeparation",
        "pattern-only cache": "CertifiedScenarioCache",
        "canonical cut bytes": "canonical_cut_bytes",
        "canonical pattern SHA": "pattern_sha256",
        "exact quantization": "quantized_bucket",
        "relative evidence": "relative_normalized_violation_evidence",
        "integer eligibility threshold": "relative_violation_eligible",
        "adaptive committed count": "committed_master_cut_sha256_values",
        "selection checkpoint": "make_selection_checkpoint",
        "commit checkpoint": "commit_selected_checkpoint",
        "checkpoint validation": "validate_checkpoint_state",
        "full separation certification": "full.robust_feasibility_certified",
        "final certification": "final_certification=certification_active",
        "no cut deletion": "total_committed_unique_master_cuts",
        "post-runtime metadata": "runtime_driven_scientific_branching",
    }
    for name, token in required_implementation_tokens.items():
        check(name, token in implementation)
    check("no Decimal context", "localcontext" not in implementation and ".quantize(" not in implementation)
    check("no runtime batch branch", "reporting_runtime" not in implementation and "runtime share" not in implementation.lower())
    check("production files protected", all(
        path not in implementation for path in ("src/benders.py", "src/scenarios.py")
    ))
    check("runner CLI", all(token in runner for token in (
        'add_argument("--config"', 'add_argument("--stage"',
        'add_argument("--resume"', 'add_argument("--dry-run"',
    )))
    check("runner no overwrite", 'add_argument("--overwrite"' not in runner)
    check("formal gate before execution", "formal_run_not_authorized" in runner)
    check("holdout and old stages rejected", "PROHIBITED_STAGES" in runner)
    check("recovery ledger", "RECOVERABLE_PHASES" in runner and "advance_recovery_ledger" in runner)
    check("fault injection tests", all(token in tests for token in (
        "test_s0_selection_checkpoint_interrupt_resumes_to_clean_result",
        "test_recovery_ledger_faults_resume_without_replaying_committed_phases",
    )))
    check("S0 extensive equivalence tests", "test_s0_adaptive_solver_matches_extensive_form" in tests)
    check("relative boundary tests", "test_relative_union_denominator_integer_formula_and_zero_policies" in tests)
    expected = {"L0": 2, "L1": 9, "M1": 9}
    dry_runs = {}
    for stage, relative in CONFIGS.items():
        report = dry_run_remediation(root / relative, stage=stage, root=root)
        dry_runs[stage] = report
        check(f"{stage} task arithmetic", report["tasks"] == expected[stage], report["tasks"])
        check(f"{stage} no solver", report["solver_called"] is False)
        check(f"{stage} no instances", report["instances_generated"] is False)
        check(f"{stage} output absent", report["output_dir_exists"] is False)
        check(f"{stage} Windows path", report["longest_windows_path_length"] <= 220, report["longest_windows_path_length"])
    check("cross-scale maximum", dry_runs["L1"]["tasks"] + dry_runs["M1"]["tasks"] == 18)
    passed = sum(row["passed"] for row in checks)
    return {
        "audit": "fairness_large_final_remediation_implementation",
        "candidate": CANDIDATE,
        "checks_passed": passed,
        "checks_total": len(checks),
        "passed": passed == len(checks),
        "formal_run_authorized": False,
        "dry_run": dry_runs,
        "checks": checks,
    }


def main() -> int:
    report = audit(Path(__file__).resolve().parents[1])
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
