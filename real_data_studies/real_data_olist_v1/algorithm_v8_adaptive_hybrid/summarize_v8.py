from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
STUDY_ROOT = HERE.parent
HOLDOUT = STUDY_ROOT / "algorithm_v4_holdout_validation"
RESULTS = HERE / "v8_vs_pure_results"


def main() -> None:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        v8 = payload["results"]["v8"]
        pure = payload["results"]["pure_ccg"]
        direct = json.loads(
            (HOLDOUT / "cases" / payload["case_id"] / "direct.json").read_text(encoding="utf-8")
        )
        rows.append({
            "case_id": payload["case_id"],
            "run_order": "->".join(payload["run_order"]),
            "v8_status": v8["status"],
            "pure_status": pure["status"],
            "v8_certified": v8["robust_feasibility_certified"],
            "pure_certified": pure["robust_feasibility_certified"],
            "v8_runtime": v8["runtime"],
            "pure_runtime": pure["runtime"],
            "v8_iterations": v8["iterations"],
            "pure_iterations": pure["iterations"],
            "v8_blocks": v8["blocked_scenarios"],
            "pure_blocks": pure["blocked_scenarios"],
            "v8_cuts": v8["cuts_added_total"],
            "v8_t": v8["objective_t"],
            "pure_t": pure["objective_t"],
            "direct_t": direct["objective_t"],
            "v8_direct_abs_error": abs(v8["objective_t"] - direct["objective_t"]),
            "pure_direct_abs_error": (
                abs(pure["objective_t"] - direct["objective_t"])
                if pure["robust_feasibility_certified"] else None
            ),
        })

    joint = [row for row in rows if row["v8_certified"] and row["pure_certified"]]
    v8_times = [row["v8_runtime"] for row in joint]
    pure_times = [row["pure_runtime"] for row in joint]
    wins = sum(v8 < pure for v8, pure in zip(v8_times, pure_times))
    losses = sum(v8 > pure for v8, pure in zip(v8_times, pure_times))
    mean_v8 = statistics.mean(v8_times)
    mean_pure = statistics.mean(pure_times)
    geometric_speedup = math.exp(
        statistics.mean(math.log(pure / v8) for v8, pure in zip(v8_times, pure_times))
    )
    non_ties = wins + losses
    sign_test_p = min(1.0, 2.0 * (0.5 ** non_ties)) if min(wins, losses) == 0 else None
    summary = {
        "status": "development_comparison_complete_external_confirmation_required",
        "cases": len(rows),
        "v8_certified": sum(row["v8_certified"] for row in rows),
        "pure_ccg_certified": sum(row["pure_certified"] for row in rows),
        "jointly_certified": len(joint),
        "v8_wins": wins,
        "v8_losses": losses,
        "v8_mean_runtime": mean_v8,
        "pure_ccg_mean_runtime": mean_pure,
        "runtime_reduction_percent": 100.0 * (mean_pure - mean_v8) / mean_pure,
        "geometric_mean_speedup": geometric_speedup,
        "two_sided_exact_sign_test_p": sign_test_p,
        "v8_max_direct_objective_abs_error": max(row["v8_direct_abs_error"] for row in rows),
        "v8_cases_with_farkas_cuts": sum(row["v8_cuts"] > 0 for row in rows),
        "v8_total_farkas_cuts": sum(row["v8_cuts"] for row in rows),
        "claim_limit": "The twelve weekly cases were observed during algorithm development and are not an independent holdout.",
    }
    (HERE / "v8_vs_pure_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (HERE / "v8_vs_pure_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
