from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.final_evaluation_analysis import AnalysisIntegrityError, run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and analyze frozen held-out final-evaluation results."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_analysis(
            input_dir=args.input_dir,
            analysis_config_path=args.analysis_config,
            output_dir=args.output_dir,
        )
    except AnalysisIntegrityError as exc:
        print(f"FINAL-EVALUATION ANALYSIS FAILED: {exc}", file=sys.stderr)
        return 2
    print("Final-evaluation analysis completed without invoking the solver.")
    for name, path in sorted(outputs.items()):
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
