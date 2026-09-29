"""Input validation and deterministic, atomic output helpers."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class InputError(Exception):
    """Invalid input data or an unsafe output path."""


def parse_time(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        raise InputError(f"{label}: expected an ISO 8601 timestamp with UTC offset") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InputError(f"{label}: timestamp needs a UTC offset")
    return parsed.astimezone(timezone.utc)


def format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            header = reader.fieldnames
            if header is None or len(set(header)) != len(header) or any(not name for name in header):
                raise InputError(f"{path}: missing or duplicate CSV headers")
            missing = set(required) - set(header)
            if missing:
                raise InputError(f"{path}: missing columns: {', '.join(sorted(missing))}")
            rows = []
            for number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise InputError(f"{path}:{number}: row has a different number of columns")
                rows.append(row)
            return rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError(f"Cannot read CSV {path}: {exc}") from exc


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"Cannot read JSON {path}: {exc}") from exc


def check_output(path: Path, inputs: tuple[Path, ...], force: bool) -> None:
    target = path.resolve(strict=False)
    if any(target == source.resolve(strict=False) for source in inputs):
        raise InputError("Output path must differ from all input paths")
    if target.exists() and not force:
        raise InputError(f"Output already exists: {path}; pass --force to replace it")


def _atomic_write(path: Path, writer) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            writer(target)
        os.replace(temporary, path)
    except OSError as exc:
        raise InputError(f"Cannot write {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    def emit(target) -> None:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    _atomic_write(path, emit)


def write_json(path: Path, value: Any) -> None:
    def emit(target) -> None:
        json.dump(value, target, ensure_ascii=False, indent=2)
        target.write("\n")

    _atomic_write(path, emit)
