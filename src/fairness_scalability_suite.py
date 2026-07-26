from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .fairness_scalability_runner import (
    STAGES,
    run_scalability_stage,
    validate_runtime_config,
)


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Scalability config must contain a mapping.")
    return value


def scenario_count_for_size(size: str, gamma: int) -> int:
    dimensions = {"medium_large": 60, "large": 96}
    if size not in dimensions:
        raise ValueError(f"Unsupported frozen size: {size}")
    n = dimensions[size]
    return sum(math.comb(n, k) for k in range(min(int(gamma), n) + 1))


def dry_run_report(config: dict[str, Any], stage: str = "s1") -> dict[str, Any]:
    validate_runtime_config(config)
    seeds = [int(v) for v in config["development_seeds"]]
    s1_seeds = [int(v) for v in config["s1_seeds"]]
    candidates = [str(v) for v in config["scalability_candidates"]]
    s1_rhos = [float(v) for v in config["s1_rho_grid"]]
    s2_rhos = [float(v) for v in config["s2_rho_grid"]]
    full_rhos = [float(v) for v in config["rho_grid"]]
    size = str(config["instance_sizes"][0])
    s1_frontier = len(s1_seeds) * len(s1_rhos) * len(candidates)
    s2_frontier = len(seeds) * len(s2_rhos) * len(candidates)
    remaining_rhos = [rho for rho in full_rhos if rho not in s2_rhos]
    stage_counts = {"s1": 27, "s2": 90, "full-grid": 120}
    new_counts = {"s1": 27, "s2": 63, "full-grid": 30}
    return {
        "experiment_name": config["experiment_name"],
        "authorization": config["authorization"],
        "size": size,
        "development_seeds": seeds,
        "reserved_seeds_accessed": False,
        "candidates": candidates,
        "scenario_count": scenario_count_for_size(size, int(config["gamma_target"])),
        "s0": {"formal_runs": 0, "tiny_deterministic_tests_only": True},
        "s1": {
            "baseline_tasks": len(s1_seeds),
            "frontier_tasks": s1_frontier,
            "total_tasks": len(s1_seeds) + s1_frontier,
        },
        "s2_cumulative": {
            "baseline_tasks": len(seeds),
            "frontier_tasks": s2_frontier,
            "total_tasks": len(seeds) + s2_frontier,
        },
        "full_grid_conditional_additional_tasks": len(seeds) * len(remaining_rhos),
        "complete_staged_unique_tasks": len(seeds) + s2_frontier + len(seeds) * len(remaining_rhos),
        "requested_stage": stage,
        "requested_stage_cumulative_tasks": stage_counts[stage],
        "requested_stage_new_tasks": new_counts[stage],
        "output_dir": config["output_dir"],
        "output_dir_exists": Path(config["output_dir"]).exists(),
        "instances_generated": False,
        "solver_called": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fairness scalability protocol planner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=STAGES, default="s1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--decision")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.dry_run:
        print(json.dumps(dry_run_report(config, args.stage), indent=2, sort_keys=True))
        return 0
    if not args.resume:
        parser.error("Formal scalability execution requires --resume.")
    run_scalability_stage(
        config,
        config_path=config_path,
        stage=args.stage,
        resume=True,
        decision_path=args.decision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
