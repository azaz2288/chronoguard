"""Input validation and deterministic, atomic output helpers."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class InputError(Exception):
    """Invalid input data or an unsafe output path."""


@dataclass(frozen=True)
class InputSnapshot:
    """Captured bytes, not a promise that the original path remains unchanged."""

    path: Path
    data: bytes

    def __str__(self) -> str:
        return str(self.path)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @classmethod
    def capture(cls, path: Path) -> InputSnapshot:
        def identity(info, *, path_check=False):
            # Reading can update atime; exclude it from the change check.
            result = (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)
            # Python 3.12 Windows stat/fstat may disagree about ctime after
            # hard-link publication (creation vs metadata-change time).
            return result if path_check and os.name == 'nt' else (*result, info.st_ctime_ns)

        try:
            # Avoid waiting on an ordinary FIFO/device input. The descriptor
            # check below is still required; this is not hostile-path isolation.
            if not stat.S_ISREG(path.stat().st_mode):
                raise InputError(f"{path}: snapshot input must be a regular file")
            with path.open('rb') as source:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise InputError(f"{path}: snapshot input must be a regular file")
                # Bound capture to the observed length plus one growth sentinel.
                data = source.read(before.st_size + 1)
                after = os.fstat(source.fileno())
                if (identity(before) != identity(after) or len(data) != before.st_size
                        or identity(after, path_check=True) != identity(path.stat(), path_check=True)):
                    raise InputError(f"{path}: input changed during capture; retry with stable files")
            return cls(path, data)
        except OSError as exc:
            raise InputError(f"Cannot capture {path}: {exc}") from exc


def _text_source(path: Path | InputSnapshot, encoding: str):
    if isinstance(path, InputSnapshot):
        return io.StringIO(path.data.decode(encoding), newline='')
    return path.open('r', encoding=encoding, newline='')


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


def read_csv(path: Path | InputSnapshot, required: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with _text_source(path, 'utf-8-sig') as source:
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


def read_json(path: Path | InputSnapshot) -> Any:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InputError("JSON contains duplicate object keys")
            result[key] = value
        return result

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise InputError("JSON contains a nonfinite number")
        return result

    def reject_constant(value):
        raise InputError("JSON contains a nonfinite literal")

    try:
        with _text_source(path, 'utf-8') as source:
            return json.load(source, object_pairs_hook=unique_object,
                             parse_float=finite_float, parse_constant=reject_constant)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise InputError(f"Cannot read JSON {path}: {exc}") from exc


def check_output(path: Path, inputs: tuple[Path, ...], force: bool) -> None:
    target = path.resolve(strict=False)
    if any(target == source.resolve(strict=False) for source in inputs):
        raise InputError("Output path must differ from all input paths")
    if target.exists() and not force:
        raise InputError(f"Output already exists: {path}; pass --force to replace it")


def _atomic_write(path: Path, writer, *, force: bool = False) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            writer(target)
            target.flush()
            os.fsync(target.fileno())
        if force:
            os.replace(temporary, path)
        else:
            # A preflight existence check is only UX, not overwrite protection.
            # Publish complete bytes atomically and exclusively; fail closed on
            # filesystems without hard links instead of racing another writer.
            os.link(temporary, path)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise InputError(f"Cannot write {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]], *, force: bool = False) -> None:
    def emit(target) -> None:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    _atomic_write(path, emit, force=force)


def write_json(path: Path, value: Any, *, force: bool = False) -> None:
    def emit(target) -> None:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")

    _atomic_write(path, emit, force=force)
