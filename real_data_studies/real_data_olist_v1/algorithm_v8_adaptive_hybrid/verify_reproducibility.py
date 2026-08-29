from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(path: Path) -> int:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for row in manifest["files"]:
        artifact = HERE / row["path"]
        actual = sha256(artifact) if artifact.is_file() else None
        if actual != row["sha256"]:
            failures.append({"path": row["path"], "expected": row["sha256"], "actual": actual})
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))
    return len(manifest["files"])


def main() -> None:
    protocol = json.loads((HERE / "confirmation_protocol.json").read_text(encoding="utf-8"))
    candidate = HERE / protocol["candidate_source"]
    actual_candidate_hash = sha256(candidate).upper()
    if actual_candidate_hash != protocol["candidate_source_sha256"]:
        raise RuntimeError(
            f"candidate source hash mismatch: {actual_candidate_hash}"
        )
    input_count = verify_manifest(HERE / "sealed_confirmation" / "input_freeze.json")
    artifact_count = verify_manifest(HERE / "artifact_manifest.json")
    print(
        f"Verification passed: candidate source, {input_count} sealed inputs, "
        f"and {artifact_count} archived artifacts."
    )


if __name__ == "__main__":
    main()
