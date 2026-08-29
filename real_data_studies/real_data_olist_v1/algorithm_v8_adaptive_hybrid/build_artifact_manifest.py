from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
INCLUDED = (
    "confirmation_protocol.json",
    "environment_snapshot.json",
    "STATUS.json",
    "development_report_zh.md",
    "v8_vs_pure_summary.json",
    "v8_vs_pure_summary.csv",
    "sealed_confirmation/case_catalog.json",
    "sealed_confirmation/summary.json",
    "sealed_confirmation/summary.csv",
)
RESULT_GLOBS = (
    "v8_vs_pure_results/*.json",
    "sealed_confirmation/results/*.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paths = [HERE / value for value in INCLUDED]
    for pattern in RESULT_GLOBS:
        paths.extend(sorted(HERE.glob(pattern)))
    rows = [
        {
            "path": path.relative_to(HERE).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]
    payload = {
        "status": "archived_artifacts_hashed_after_report_generation",
        "file_count": len(rows),
        "files": rows,
    }
    (HERE / "artifact_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote hashes for {len(rows)} artifacts.")


if __name__ == "__main__":
    main()
