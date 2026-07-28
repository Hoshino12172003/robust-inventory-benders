"""Identity-locked runner and dry-run for final fairness remediation stages.

The checked-in L0/L1/M1 configurations are intentionally protocol-only.  This
runner therefore exposes the complete gate and identity preflight now, while
refusing every formal invocation before creating an output directory or an
instance.  A later, separately reviewed authorization may change that external
gate; this implementation stage cannot.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .experiment_protocol import atomic_write_json, config_sha256, file_sha256, git_commit, read_json
from .fairness_large_final_remediation import (
    CANDIDATE,
    CANDIDATE_SHA256,
    INITIAL_UB_THEOREM_SHA256,
    PROTOCOL_SHA256,
)
from .fairness_large_final_remediation_audit import (
    CUT_SCHEMA,
    PATTERN_SCHEMA,
    QUANTIZATION_SCHEMA,
    RELATIVE_VIOLATION_SCHEMA,
    _candidate_paths,
    expand_plan,
)


STAGES = {"L0", "L1", "M1"}
PROHIBITED_STAGES = {"holdout", "S2", "s2", "full-grid", "full_grid", "Attempt4", "attempt4"}
EXPECTED_FILE_SHA256 = {
    "L0": "286E60982CE78AF07DCC8AFC7D938491CB30442F184BCB897FC708180DDDB8BB",
    "L1": "08F77E62DCB0252AAA7E5C23B8234A2DCF40D46CC2290E6FB6851F202CE1DE53",
    "M1": "D8AAEAD792369032A47BE230954D0D5CB5EFE3A562FE50BF40D9FA6E10815C19",
}
RECOVERABLE_PHASES = (
    "algorithm_checkpoint",
    "separation_complete",
    "post_evaluation_chunks_complete",
    "aggregation_complete",
    "csv_complete",
)


class RemediationGateError(RuntimeError):
    """A formal stage or identity gate is not satisfied."""


def advance_recovery_ledger(
    path: str | Path,
    *,
    identity: dict[str, Any],
    phase_action: Any,
    failure_injector: Any | None = None,
) -> dict[str, Any]:
    """Advance post-algorithm phases atomically and resume without replaying commits."""
    target = Path(path)
    payload = read_json(target)
    if target.exists() and not isinstance(payload, dict):
        raise RemediationGateError("recovery ledger corrupt")
    if payload is None:
        payload = {
            "schema": "fairness_large_remediation_recovery_ledger_v1",
            "identity": deepcopy(identity),
            "completed_phases": [],
        }
    if payload.get("identity") != identity:
        raise RemediationGateError("recovery ledger identity mismatch")
    completed = payload.get("completed_phases")
    if not isinstance(completed, list) or len(completed) != len(set(completed)):
        raise RemediationGateError("recovery ledger phase list corrupt")
    if completed != list(RECOVERABLE_PHASES[: len(completed)]):
        raise RemediationGateError("recovery ledger phase order corrupt")
    for phase in RECOVERABLE_PHASES[len(completed):]:
        phase_action(phase)
        if failure_injector:
            failure_injector(f"before_{phase}_checkpoint", deepcopy(payload))
        committed = deepcopy(payload)
        committed["completed_phases"] = list(completed) + [phase]
        atomic_write_json(target, committed)
        payload = committed
        completed = payload["completed_phases"]
        if failure_injector:
            failure_injector(f"after_{phase}_checkpoint", deepcopy(payload))
    return payload


def load_remediation_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Remediation config must be a YAML mapping.")
    return payload


def validate_frozen_config(path: str | Path, config: dict[str, Any], *, stage: str) -> None:
    if stage not in STAGES:
        if stage in PROHIBITED_STAGES:
            raise RemediationGateError(f"stage {stage} is permanently unauthorized by this runner")
        raise RemediationGateError("stage must be exactly L0, L1, or M1")
    if config.get("stage") != stage:
        raise RemediationGateError("CLI stage does not match the frozen config stage")
    actual = file_sha256(path).upper()
    if actual != EXPECTED_FILE_SHA256[stage]:
        raise RemediationGateError("frozen stage config file SHA256 mismatch")
    if config.get("required_protocol_sha256") != PROTOCOL_SHA256:
        raise RemediationGateError("frozen protocol SHA256 mismatch")
    if config.get("required_candidate_sha256") != CANDIDATE_SHA256:
        raise RemediationGateError("frozen candidate SHA256 mismatch")
    if config.get("candidate") != CANDIDATE:
        raise RemediationGateError("a second remediation candidate is forbidden")
    if int(config.get("execution_attempt", -1)) != 3:
        raise RemediationGateError("remediation execution attempt identity mismatch")
    if config.get("previous_attempt_results_reused") is not False:
        raise RemediationGateError("previous formal results may not be reused")
    if config.get("overwrite_supported") is not False:
        raise RemediationGateError("overwrite must remain unsupported")
    seeds = [int(value) for value in config.get("seeds", [])]
    if any(130 <= value <= 159 or 170 <= value <= 179 for value in seeds):
        raise RemediationGateError("prohibited or holdout seed requested")
    if seeds != ([160] if stage == "L0" else [160, 161, 162]):
        raise RemediationGateError("development seed matrix drift")
    if [float(value) for value in config.get("rho", [])] != ([0.0] if stage == "L0" else [0.0, 0.01]):
        raise RemediationGateError("rho matrix drift")
    locks = set(config.get("manifest_identity_lock", []))
    required_locks = {
        "git", "config_file_sha256", "resolved_config_file_sha256",
        "resolved_config_canonical_sha256", "protocol_sha256", "candidate_sha256",
        "pattern_schema", "cut_schema", "quantization_schema",
        "relative_violation_identity", "adaptive_batch_schedule",
        "committed_cut_state_machine", "checkpoint_selection_state",
        "solver", "instance", "baseline", "anchor",
    }
    if not required_locks.issubset(locks):
        raise RemediationGateError("manifest scientific identity lock is incomplete")


def validate_stage_gate(stage: str, decision: dict[str, Any] | None) -> None:
    if stage == "L0":
        if decision not in (None, {}):
            raise RemediationGateError("L0 does not consume a prior run decision")
        return
    if not isinstance(decision, dict):
        raise RemediationGateError(f"{stage} requires an identity-locked prior-stage decision")
    if stage == "L1":
        required = {
            "decision": "authorize_L1_after_L0",
            "scientific_status": "certified_robust_optimal",
            "post_evaluation_valid": True,
            "scenario_count": 4657,
            "valid_upper_bound": True,
            "valid_lower_bound": True,
            "implementation_error": False,
            "invalid_post_evaluation": False,
        }
    elif stage == "M1":
        required = {
            "decision": "authorize_M1_after_L1",
            "certified_frontier_count": 4,
            "frontier_count": 6,
            "implementation_error_count": 0,
            "invalid_post_evaluation_count": 0,
            "all_successful_post_evaluations_valid": True,
        }
        if int(decision.get("certified_frontier_count", -1)) < 4:
            raise RemediationGateError("L1 certified frontier is below 4/6")
        required.pop("certified_frontier_count")
    else:
        raise RemediationGateError("unknown remediation stage")
    for field, expected in required.items():
        if decision.get(field) != expected:
            raise RemediationGateError(f"prior-stage decision mismatch: {field}")
    if not isinstance(decision.get("identity_sha256"), str) or len(decision["identity_sha256"]) != 64:
        raise RemediationGateError("prior-stage decision identity SHA256 missing")


def execution_identity(
    root: Path, config_path: Path, config: dict[str, Any], *, stage: str,
    decision: dict[str, Any] | None,
) -> dict[str, Any]:
    decision_hash = None
    if decision:
        decision_hash = hashlib.sha256(
            json.dumps(decision, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest().upper()
    return {
        "schema": "fairness_large_final_remediation_manifest_v1",
        "execution_attempt": 3,
        "stage": stage,
        "git_commit": git_commit(root),
        "config_file_sha256": file_sha256(config_path).upper(),
        "resolved_config_canonical_sha256": config_sha256(config).upper(),
        "protocol_sha256": PROTOCOL_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "candidate": CANDIDATE,
        "initial_ub_theorem_sha256": INITIAL_UB_THEOREM_SHA256,
        "pattern_schema": PATTERN_SCHEMA,
        "cut_schema": CUT_SCHEMA,
        "relative_violation_schema": RELATIVE_VIOLATION_SCHEMA,
        "quantization_schema": QUANTIZATION_SCHEMA,
        "relative_violation_identity": {
            "quantum": "1e-9",
            "threshold_bucket": 100_000_000,
            "deadband_buckets": 2,
            "eligibility": "strictly_greater_than_100000002",
            "denominator": "maximum_normalized_violation_bucket_over_unique_certified_violating_candidates",
        },
        "adaptive_batch_schedule": [[1000, 5], [3000, 3], [5000, 2], [None, 1]],
        "stage_decision_sha256": decision_hash,
        "previous_attempt_results_reused": False,
        "formal_run_authorized": bool(config.get("formal_run_authorized")),
    }


def dry_run_remediation(
    config_path: str | Path,
    *,
    stage: str,
    root: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    repository = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    config = load_remediation_config(config_path)
    validate_frozen_config(config_path, config, stage=stage)
    rows = expand_plan(config)
    if len({row["run_key"] for row in rows}) != len(rows):
        raise RemediationGateError("duplicate run key in frozen plan")
    if len({row["run_directory_id"] for row in rows}) != len(rows):
        raise RemediationGateError("short run-directory hash collision")
    output = repository / str(config["output_dir"])
    paths = _candidate_paths(repository, config, rows)
    longest = max(paths, key=lambda value: len(str(value.resolve())))
    maximum = int(config["path_policy"]["maximum_absolute_path_length"])
    if len(str(longest.resolve())) > maximum:
        raise RemediationGateError("Windows path portability limit exceeded")
    return {
        "stage": stage,
        "tasks": len(rows),
        "baseline_count": sum(row["task_type"] == "baseline" for row in rows),
        "frontier_count": sum(row["task_type"] == "frontier" for row in rows),
        "scenario_count": int(config["scenario_count"]),
        "run_key_count": len(rows),
        "run_directory_id_count": len(rows),
        "longest_windows_path": str(longest.resolve()),
        "longest_windows_path_length": len(str(longest.resolve())),
        "instances_generated": False,
        "solver_called": False,
        "output_dir_exists": output.exists(),
        "formal_run_authorized": False,
        "identity": execution_identity(repository, config_path, config, stage=stage, decision=None),
    }


def run_remediation_stage(
    config_path: str | Path,
    *,
    stage: str,
    resume: bool,
    dry_run: bool,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_remediation_config(config_path)
    validate_frozen_config(config_path, config, stage=stage)
    if dry_run:
        return dry_run_remediation(config_path, stage=stage)
    # This test happens before gate parsing, mkdir, instance generation, module
    # imports that create models, or solver parameter configuration.
    if config.get("formal_run_authorized") is not True or config.get("authorization") != "formal_execution_authorized":
        raise RemediationGateError("formal_run_not_authorized")
    validate_stage_gate(stage, decision)
    raise RemediationGateError(
        "formal execution additionally requires the future implementation review and pre-run audit"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final fairness remediation runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_remediation_stage(
        args.config,
        stage=args.stage,
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
