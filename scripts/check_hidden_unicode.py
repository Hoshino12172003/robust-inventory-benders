from __future__ import annotations

import argparse
import subprocess
import sys
import unicodedata
from pathlib import Path


ALLOWED_CONTROL_CODEPOINTS = {0x09, 0x0A, 0x0D}
BIDI_CONTROL_CODEPOINTS = {
    0x202A,
    0x202B,
    0x202C,
    0x202D,
    0x202E,
    0x2066,
    0x2067,
    0x2068,
    0x2069,
}
ZERO_WIDTH_CODEPOINTS = {
    0x200B,
    0x200C,
    0x200D,
    0xFEFF,
}
EXPLICIT_CODEPOINTS = BIDI_CONTROL_CODEPOINTS | ZERO_WIDTH_CODEPOINTS | {0x00A0}
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache"}


# This scanner removes only hidden/control characters and preserves normal text.
def git_tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def is_probably_binary(data: bytes) -> bool:
    return b"\x00" in data


def should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def is_hidden_unicode(ch: str) -> bool:
    codepoint = ord(ch)
    category = unicodedata.category(ch)
    if codepoint in EXPLICIT_CODEPOINTS:
        return True
    if category == "Cf":
        return True
    if category == "Cc" and codepoint not in ALLOWED_CONTROL_CODEPOINTS:
        return True
    return False


def unicode_name(ch: str) -> str:
    return unicodedata.name(ch, "<unnamed>")


def scan_text(path: Path, text: str) -> list[str]:
    findings: list[str] = []
    line = 1
    column = 1
    for ch in text:
        if is_hidden_unicode(ch):
            findings.append(
                f"{path.as_posix()}:{line}:{column} U+{ord(ch):04X} {unicode_name(ch)}"
            )
        if ch == "\n":
            line += 1
            column = 1
        else:
            column += 1
    return findings


def clean_text(text: str) -> str:
    return "".join(ch for ch in text if not is_hidden_unicode(ch))


def read_text_file(path: Path) -> str | None:
    data = path.read_bytes()
    if is_probably_binary(data):
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan git tracked files for hidden Unicode characters.")
    parser.add_argument("--fix", action="store_true", help="Delete hidden/control Unicode characters in place.")
    args = parser.parse_args()

    all_findings: list[str] = []
    fixed_files: list[Path] = []

    for path in git_tracked_files():
        if should_skip(path) or not path.exists() or not path.is_file():
            continue
        text = read_text_file(path)
        if text is None:
            continue
        findings = scan_text(path, text)
        if findings:
            all_findings.extend(findings)
            if args.fix:
                cleaned = clean_text(text)
                path.write_bytes(cleaned.encode("utf-8"))
                fixed_files.append(path)

    if all_findings:
        for finding in all_findings:
            print(finding)
        if args.fix:
            print()
            print("Fixed files:")
            for path in fixed_files:
                print(path.as_posix())
        return 1

    print("No hidden Unicode characters found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
