from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


REQUIRED = {
    "sales_train_evaluation.csv": {
        "identity_columns": ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        "first_day": "d_1",
        "last_day": "d_1941",
        "expected_rows": 30490,
    },
    "calendar.csv": {
        "identity_columns": ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d"],
        "first_day": None,
        "last_day": None,
        "expected_rows": 1969,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_csv(path: Path, specification: dict) -> dict:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    missing = [column for column in specification["identity_columns"] if column not in header]
    if missing:
        raise RuntimeError(f"{path.name}: missing identity columns {missing}")
    for key in ("first_day", "last_day"):
        value = specification[key]
        if value is not None and value not in header:
            raise RuntimeError(f"{path.name}: missing required day column {value}")
    if rows != specification["expected_rows"]:
        raise RuntimeError(
            f"{path.name}: expected {specification['expected_rows']} rows, found {rows}"
        )
    return {
        "file": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": len(header),
        "header_sha256": hashlib.sha256(
            json.dumps(header, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_directory", type=Path)
    args = parser.parse_args()
    records = []
    for name, specification in REQUIRED.items():
        path = args.raw_directory / name
        if not path.is_file():
            raise RuntimeError(f"missing official M5 file: {path}")
        records.append(inspect_csv(path, specification))
    print(json.dumps({
        "schema": "m5_official_download_structural_audit_v1",
        "status": "pass",
        "files": records,
    }, indent=2))


if __name__ == "__main__":
    main()
