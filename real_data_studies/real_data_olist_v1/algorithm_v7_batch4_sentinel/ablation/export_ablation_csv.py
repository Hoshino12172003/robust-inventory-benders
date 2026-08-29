from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def export(source: str, output: str, left: str, right: str) -> None:
    rows = []
    for path in sorted((HERE / source).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        lhs = payload["results"][left]
        rhs = payload["results"][right]
        rows.append({
            "case_id": path.stem,
            "run_order": " then ".join(payload["run_order"]),
            f"{left}_status": lhs["status"],
            f"{right}_status": rhs["status"],
            f"{left}_certified": lhs["robust_feasibility_certified"],
            f"{right}_certified": rhs["robust_feasibility_certified"],
            f"{left}_runtime": lhs["runtime"],
            f"{right}_runtime": rhs["runtime"],
            f"{left}_iterations": lhs["iterations"],
            f"{right}_iterations": rhs["iterations"],
            f"{left}_objective_t": lhs["objective_t"],
            f"{right}_objective_t": rhs["objective_t"],
        })
    with (HERE / output).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    export("paired_results", "sentinel_vs_batch4.csv", "hybrid_v7", "batch4_ccg")
    export("batch4_vs_pure_results", "batch4_vs_pure.csv", "batch4_ccg", "pure_ccg")


if __name__ == "__main__":
    main()
