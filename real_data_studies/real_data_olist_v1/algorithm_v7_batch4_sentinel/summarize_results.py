from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
HOLDOUT = STUDY_ROOT / "algorithm_v4_holdout_validation"
TIME_LIMIT = 900.0


def main() -> None:
    rows = []
    for path in sorted((HERE / "paired_replay").glob("*.json")):
        paired = json.loads(path.read_text(encoding="utf-8"))
        v7 = paired["results"]["hybrid_v7"]
        pure = paired["results"]["pure_ccg"]
        direct = json.loads(
            (HOLDOUT / "cases" / path.stem / "direct.json").read_text(encoding="utf-8")
        )
        rows.append({
            "case_id": path.stem,
            "run_order": " then ".join(paired["run_order"]),
            "v7_status": v7["status"],
            "pure_status": pure["status"],
            "v7_certified": v7["robust_feasibility_certified"],
            "pure_certified": pure["robust_feasibility_certified"],
            "v7_runtime": v7["runtime"],
            "pure_runtime": pure["runtime"],
            "v7_iterations": v7["iterations"],
            "pure_iterations": pure["iterations"],
            "v7_t": v7["objective_t"],
            "pure_t": pure["objective_t"],
            "direct_t": direct["objective_t"],
            "v7_direct_error": abs(v7["objective_t"] - direct["objective_t"]),
            "v7_faster_when_jointly_certified": bool(
                pure["robust_feasibility_certified"] and v7["runtime"] < pure["runtime"]
            ),
        })

    joint = [row for row in rows if row["v7_certified"] and row["pure_certified"]]
    v7_times = [row["v7_runtime"] for row in joint]
    pure_times = [row["pure_runtime"] for row in joint]
    pure_par2 = [row["pure_runtime"] if row["pure_certified"] else 2 * TIME_LIMIT for row in rows]
    summary = {
        "status": "development_success_on_observed_instances_requires_new_independent_validation",
        "cases": len(rows),
        "v7_certified": sum(row["v7_certified"] for row in rows),
        "pure_ccg_certified": sum(row["pure_certified"] for row in rows),
        "jointly_certified_cases": len(joint),
        "v7_runtime_wins": sum(row["v7_faster_when_jointly_certified"] for row in joint),
        "v7_mean_runtime_joint": statistics.mean(v7_times),
        "pure_mean_runtime_joint": statistics.mean(pure_times),
        "runtime_reduction_percent": 100.0 * (1.0 - sum(v7_times) / sum(pure_times)),
        "geometric_mean_speedup_pure_over_v7": math.exp(
            statistics.mean(math.log(pure / v7) for v7, pure in zip(v7_times, pure_times))
        ),
        "exact_two_sided_sign_test_p": 2.0 / (2.0 ** len(joint)),
        "max_v7_objective_error_against_direct": max(row["v7_direct_error"] for row in rows),
        "v7_par2_mean": statistics.mean(row["v7_runtime"] for row in rows),
        "pure_ccg_par2_mean": statistics.mean(pure_par2),
        "test_data_status": "observed during v4-v7 development; not an independent final holdout",
        "paper_results_modified": False,
    }
    (HERE / "paired_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (HERE / "paired_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
