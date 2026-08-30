from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE / "hybrid_v8_formal_holdout"
RESULTS = ROOT / "results"
METHODS = ("hybrid_v8", "pure_ccg", "batch4_ccg", "direct")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    target = RESULTS / "result_freeze.json"
    if target.exists():
        raise RuntimeError("formal result freeze already exists")
    catalog = json.loads((ROOT / "case_catalog.json").read_text(encoding="utf-8"))
    if len(catalog) != 12:
        raise RuntimeError("formal case catalog is incomplete")
    for case in catalog:
        directory = RESULTS / case["case_id"]
        required = ("cost_anchor.json", "run_plan.json") + tuple(f"{method}.json" for method in METHODS)
        for filename in required:
            if not (directory / filename).is_file():
                raise RuntimeError(f"formal result missing: {case['case_id']}/{filename}")
    required_root = ("environment.json", "summary.csv", "summary.json")
    for filename in required_root:
        if not (RESULTS / filename).is_file():
            raise RuntimeError(f"formal result missing: {filename}")
    files = []
    for path in sorted(value for value in RESULTS.rglob("*") if value.is_file()):
        files.append({
            "path": path.relative_to(RESULTS).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        })
    payload = {
        "schema": "m5_formal_holdout_result_freeze_v1",
        "status": "all_preregistered_results_frozen",
        "input_freeze_sha256": sha256(ROOT / "input_freeze.json"),
        "case_count": 12,
        "method_result_count": 48,
        "file_count_excluding_this_manifest": len(files),
        "files": files,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({
        "result_freeze_sha256": sha256(target),
        "files": len(files),
    }, indent=2))


if __name__ == "__main__":
    main()
