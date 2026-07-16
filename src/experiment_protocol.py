from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


TERMINAL_SUCCESS_STATUSES = {"optimal", "time_limit", "iteration_limit", "suboptimal"}


@dataclass(frozen=True)
class ProtocolRunSpec:
    experiment_name: str
    instance_size: str
    seed: int
    variant_name: str
    sensitivity_axis: str | None = None
    sensitivity_value: int | float | str | None = None
    baseline_value: int | float | str | None = None

    @property
    def run_key(self) -> str:
        return stable_run_key(
            experiment_name=self.experiment_name,
            sensitivity_axis=self.sensitivity_axis,
            sensitivity_value=self.sensitivity_value,
            instance_size=self.instance_size,
            seed=self.seed,
            variant_name=self.variant_name,
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_component(value: Any) -> str:
    if value is None or value == "":
        return "none"
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int):
        text = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Run-key values must be finite.")
        text = format(value, ".12g")
    else:
        text = str(value)
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in text)


def stable_run_key(
    *,
    experiment_name: str,
    sensitivity_axis: str | None,
    sensitivity_value: Any,
    instance_size: str,
    seed: int,
    variant_name: str,
) -> str:
    return "__".join(
        (
            _safe_component(experiment_name),
            _safe_component(sensitivity_axis),
            _safe_component(sensitivity_value),
            _safe_component(instance_size),
            f"seed_{int(seed)}",
            _safe_component(variant_name),
        )
    )


def penalized_runtime_par2(
    *, solved_to_tolerance: bool, runtime: float | int | None, time_limit: float | int
) -> float:
    limit = float(time_limit)
    if limit < 0.0 or not math.isfinite(limit):
        raise ValueError("time_limit must be a finite nonnegative value.")
    if solved_to_tolerance:
        if runtime is None:
            raise ValueError("Solved runs require a runtime for PAR-2.")
        value = float(runtime)
        if value < 0.0 or not math.isfinite(value):
            raise ValueError("runtime must be a finite nonnegative value.")
        return value
    return 2.0 * limit


def config_sha256(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: str | Path | None = None) -> str:
    command = ["git"]
    if repo_root is not None:
        command.extend(["-C", str(repo_root)])
    command.extend(["rev-parse", "HEAD"])
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def atomic_write_text(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, target)
    return target


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def atomic_write_yaml(path: str | Path, value: Any) -> Path:
    return atomic_write_text(
        path,
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
    )


def atomic_write_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fields: list[str],
    *,
    value_encoder: Any | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    encode = value_encoder or (lambda value: value)
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: encode(row.get(field)) for field in fields})
    os.replace(temporary, target)
    return target


def read_json(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def is_complete_success_record(record: dict[str, Any] | None) -> bool:
    if not record or record.get("state") != "complete":
        return False
    if record.get("success") is not True:
        return False
    result = record.get("result")
    if not isinstance(result, dict):
        return False
    return str(result.get("status", "")).lower() in TERMINAL_SUCCESS_STATUSES


def decide_run_action(
    record: dict[str, Any] | None,
    *,
    resume: bool,
    overwrite: bool,
) -> str:
    if overwrite:
        return "run_overwrite" if record else "run"
    if is_complete_success_record(record):
        return "skip_success"
    if record is None:
        return "run"
    if resume:
        return "run_resume"
    return "skip_incomplete"


def run_record_path(output_dir: str | Path, run_key: str) -> Path:
    return Path(output_dir) / "runs" / run_key / "run.json"


def load_run_record(output_dir: str | Path, run_key: str) -> dict[str, Any] | None:
    return read_json(run_record_path(output_dir, run_key))


def write_run_state(
    output_dir: str | Path,
    run_key: str,
    *,
    state: str,
    details: dict[str, Any] | None = None,
) -> Path:
    payload = {"run_key": run_key, "state": state, "updated_at": utc_now_iso()}
    payload.update(details or {})
    return atomic_write_json(Path(output_dir) / "runs" / run_key / "status.json", payload)


def build_run_manifest(
    *,
    output_dir: str | Path,
    run_keys: list[str],
    config_hash: str,
    commit: str,
    skipped_run_count: int,
    previous_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed = 0
    solved = 0
    failed = 0
    for run_key in run_keys:
        record = load_run_record(output_dir, run_key)
        if not record or record.get("state") != "complete":
            continue
        completed += 1
        if record.get("solved_to_tolerance") is True:
            solved += 1
        if record.get("success") is not True:
            failed += 1
    expected = len(run_keys)
    now = utc_now_iso()
    return {
        "expected_run_count": expected,
        "completed_run_count": completed,
        "solved_run_count": solved,
        "failed_run_count": failed,
        "skipped_run_count": int(skipped_run_count),
        "remaining_run_count": max(0, expected - completed),
        "config_sha256": config_hash,
        "git_commit": commit,
        "created_at": (previous_manifest or {}).get("created_at", now),
        "updated_at": now,
    }


def update_run_manifest(
    *,
    output_dir: str | Path,
    run_keys: list[str],
    config_hash: str,
    commit: str,
    skipped_run_count: int,
) -> Path:
    path = Path(output_dir) / "run_manifest.json"
    manifest = build_run_manifest(
        output_dir=output_dir,
        run_keys=run_keys,
        config_hash=config_hash,
        commit=commit,
        skipped_run_count=skipped_run_count,
        previous_manifest=read_json(path),
    )
    return atomic_write_json(path, manifest)


def theoretical_maximum_hours(run_count: int, time_limit_seconds: float) -> float:
    return float(run_count) * float(time_limit_seconds) / 3600.0
