from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path


def atomic_write_json(path: Path, value: object) -> None:
    """Write JSON through a sibling temporary file, then replace atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_gzip_json(path: Path, value: object) -> None:
    """Write deterministic compressed JSON through a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")
    temporary.write_bytes(gzip.compress(encoded, compresslevel=6, mtime=0))
    temporary.replace(path)


def atomic_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a heterogeneous row list as a stable, union-column CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_gzip_json(path: Path) -> object:
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
