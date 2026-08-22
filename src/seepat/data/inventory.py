from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from hashlib import sha256
from pathlib import Path

import ijson

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    file TEXT PRIMARY KEY,
    original TEXT,
    split TEXT NOT NULL,
    modify_type TEXT NOT NULL,
    audio_model TEXT,
    video_model TEXT,
    fake_segments_json TEXT NOT NULL,
    audio_fake_segments_json TEXT NOT NULL,
    visual_fake_segments_json TEXT NOT NULL,
    video_frames INTEGER,
    audio_frames INTEGER,
    subject_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_videos_split_modify
ON videos(split, modify_type);

CREATE INDEX IF NOT EXISTS idx_videos_original
ON videos(original);
"""


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_metadata(path: Path) -> Iterator[dict[str, object]]:
    with path.open("rb") as stream:
        # AV++ segment timestamps are ordinary floats. Parsing them as floats
        # keeps streamed values JSON-serializable without loading the array.
        yield from ijson.items(stream, "item", use_float=True)


def infer_subject_id(file_path: str) -> str | None:
    parts = Path(file_path).parts
    if len(parts) >= 2:
        return parts[1]
    return None


def _compact_json(value: object) -> str:
    return json.dumps(value if value is not None else [], separators=(",", ":"))


def build_inventory(
    metadata_paths: Sequence[Path],
    database_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    missing = [str(path) for path in metadata_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Metadata files not found: {', '.join(missing)}")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(SCHEMA)
        connection.execute("DELETE FROM videos")

        insert_sql = """
        INSERT INTO videos (
            file, original, split, modify_type, audio_model, video_model,
            fake_segments_json, audio_fake_segments_json, visual_fake_segments_json,
            video_frames, audio_frames, subject_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        source_documents: list[dict[str, object]] = []
        for metadata_path in metadata_paths:
            batch: list[tuple[object, ...]] = []
            row_count = 0
            for record in iter_metadata(metadata_path):
                file_path = str(record["file"])
                batch.append(
                    (
                        file_path,
                        record.get("original"),
                        str(record.get("split") or "unknown"),
                        str(record.get("modify_type") or "unknown"),
                        record.get("audio_model"),
                        record.get("video_model"),
                        _compact_json(record.get("fake_segments")),
                        _compact_json(record.get("audio_fake_segments")),
                        _compact_json(record.get("visual_fake_segments")),
                        record.get("video_frames"),
                        record.get("audio_frames"),
                        infer_subject_id(file_path),
                    )
                )
                row_count += 1
                if len(batch) >= 5_000:
                    connection.executemany(insert_sql, batch)
                    batch.clear()
            if batch:
                connection.executemany(insert_sql, batch)
            connection.commit()
            source_documents.append(
                {
                    "path": str(metadata_path),
                    "bytes": metadata_path.stat().st_size,
                    "sha256": file_sha256(metadata_path),
                    "rows": row_count,
                }
            )

        counts = [
            {"split": split, "modify_type": modify_type, "count": count}
            for split, modify_type, count in connection.execute(
                """
                SELECT split, modify_type, COUNT(*)
                FROM videos
                GROUP BY split, modify_type
                ORDER BY split, modify_type
                """
            )
        ]
        total_rows = int(connection.execute("SELECT COUNT(*) FROM videos").fetchone()[0])
        summary = {
            "total_rows": total_rows,
            "counts": counts,
            "sources": source_documents,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        connection.close()
