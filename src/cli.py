from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, resolve_project_path
from .experiment import run_experiment, solve_method
from .experiment_suite import run_experiment_suite
from .instance import generate_instance, load_instance, save_instance


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust inventory Benders research prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Generate a synthetic inventory instance")
    generate.add_argument("--config", default="configs/default.yaml")
    generate.add_argument("--output", default=None)

    solve = sub.add_parser("solve", help="Solve one instance")
    solve.add_argument("--config", default="configs/default.yaml")
    solve.add_argument("--method", default="adaptive_gap_gamma_benders")
    solve.add_argument("--instance", default="data/processed/instance.json")
    solve.add_argument("--summary", default=None)

    experiment = sub.add_parser("experiment", help="Run the configured method comparison")
    experiment.add_argument("--config", default="configs/experiment.yaml")

    experiment_suite = sub.add_parser("experiment-suite", help="Run the reproducible experiment suite")
    experiment_suite.add_argument("--config", default="experiments/configs/baseline_comparison.yaml")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "generate":
        instance = generate_instance(config)
        output = resolve_project_path(
            args.output or config.get("experiment", {}).get("instance_path", "data/processed/instance.json")
        )
        save_instance(instance, output)
        print(json.dumps({"instance": instance.name, "path": str(output)}, ensure_ascii=False, indent=2))
        return

    if args.command == "solve":
        instance = load_instance(resolve_project_path(args.instance))
        result = solve_method(config, instance, args.method)
        summary = result.summary_dict()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.summary:
            target = Path(args.summary)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    if args.command == "experiment":
        results = run_experiment(config)
        print(json.dumps([result.summary_dict() for result in results], ensure_ascii=False, indent=2))

    if args.command == "experiment-suite":
        outputs = run_experiment_suite(config)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
