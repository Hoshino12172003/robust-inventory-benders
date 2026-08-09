from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .experiment_protocol import file_sha256
from .fairness_high_gamma_external_solver_benchmark_runner import (
    GAMMAS, SEEDS, HighGammaGateError, _root, dry_run, expand_plan,
    load_yaml, scenario_count, seed_access_gate, validate_config,
)


def static_audit(config_path: str | Path, *, root_override: Path | None = None) -> dict[str, Any]:
    root = (root_override or _root()).resolve()
    path = Path(config_path).resolve()
    config = load_yaml(path)
    validate_config(path, config)
    before = "gurobipy" in sys.modules
    report = dry_run(path, root_override=Path(r"E:\rfext1"))
    rows = expand_plan()
    if len(rows) != 45 or [scenario_count(value) for value in GAMMAS] != [211, 1351, 6196]:
        raise HighGammaGateError("matrix or scenario count audit failed")
    if report["longest_windows_path_length"] >= 220 or report["gurobipy_imported"]:
        raise HighGammaGateError("dry-run path or solver-import audit failed")
    seed = seed_access_gate(root)
    protected = subprocess.run(
        ["git", "diff", "--quiet", "a1be201fc329780cf128457f56af0df31db8b4da", "--",
         "src/benders.py", "src/scenarios.py"], cwd=root,
    ).returncode == 0
    if not protected:
        raise HighGammaGateError("protected solver core changed")
    return {
        "decision": "approve_pre_run_static_audit",
        "formal_run_started": False,
        "seeds": SEEDS,
        "seed_access_evidence_count": seed["formal_instance_or_solve_access_evidence_count"],
        "tasks": 45,
        "baselines": 15,
        "hybrid_frontiers": 15,
        "direct_extensive_frontiers": 15,
        "scenario_counts": {str(value): scenario_count(value) for value in GAMMAS},
        "unique_run_keys": len({row["run_key"] for row in rows}),
        "unique_directory_ids": len({row["run_directory_id"] for row in rows}),
        "protected_benders_and_scenarios_unchanged": protected,
        "gurobipy_imported_by_audit": ("gurobipy" in sys.modules) and not before,
        "output_dir_exists": (root / config["output_dir"]).exists(),
        "config_sha256": file_sha256(path).upper(),
        "protocol_sha256": file_sha256(root / config["protocol_document"]).upper(),
        "direct_candidate_sha256": file_sha256(root / config["direct_candidate_definition"]).upper(),
        "longest_windows_path": report["longest_windows_path"],
        "longest_windows_path_length": report["longest_windows_path_length"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(static_audit(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
