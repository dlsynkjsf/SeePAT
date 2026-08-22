from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


def find_7zip(explicit_path: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)

    discovered = shutil.which("7z")
    if discovered:
        candidates.append(Path(discovered))

    candidates.extend(
        (
            Path(".tools/7zip/Files/7-Zip/7z.exe"),
            Path("C:/Program Files/7-Zip/7z.exe"),
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "7z was not found. Install 7-Zip or pass --seven-zip with its executable path."
    )


def read_manifest_video_paths(manifest_path: Path) -> list[str]:
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "file" not in reader.fieldnames:
            raise ValueError("Pilot manifest must contain a 'file' column")
        paths = [str(row["file"]).strip().replace("\\", "/") for row in reader]

    if not paths:
        raise ValueError("Pilot manifest contains no video paths")

    unique_paths = list(dict.fromkeys(paths))
    for item in unique_paths:
        path = PurePosixPath(item)
        if not item or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive path in pilot manifest: {item!r}")
    return unique_paths


def extract_manifest_videos(
    archive_first_volume: Path,
    manifest_path: Path,
    output_dir: Path,
    report_path: Path,
    seven_zip_path: Path | None = None,
    archive_root: str | None = None,
) -> dict[str, object]:
    if not archive_first_volume.is_file():
        raise FileNotFoundError(f"Archive first volume not found: {archive_first_volume}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pilot manifest not found: {manifest_path}")

    seven_zip = find_7zip(seven_zip_path)
    video_paths = read_manifest_video_paths(manifest_path)
    if archive_root is None:
        archive_root = archive_first_volume.name.split(".zip", maxsplit=1)[0]
    archive_root = archive_root.strip("/\\")
    archive_paths = [
        f"{archive_root}/{item}" if archive_root else item for item in video_paths
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="seepat-7z-") as temporary_dir:
        include_path = Path(temporary_dir) / "include.txt"
        include_path.write_text("\n".join(archive_paths) + "\n", encoding="utf-8")
        command = [
            str(seven_zip),
            "x",
            str(archive_first_volume.resolve()),
            f"-o{output_dir.resolve()}",
            f"-i@{include_path.resolve()}",
            "-y",
            "-aoa",
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)

    extracted_root = output_dir / archive_root if archive_root else output_dir
    missing = [item for item in video_paths if not (extracted_root / Path(item)).is_file()]
    extracted_bytes = sum(
        (extracted_root / Path(item)).stat().st_size
        for item in video_paths
        if (extracted_root / Path(item)).is_file()
    )
    report: dict[str, object] = {
        "archive_first_volume": str(archive_first_volume),
        "manifest": str(manifest_path),
        "archive_root": archive_root,
        "output_dir": str(output_dir),
        "extracted_data_root": str(extracted_root),
        "requested_files": len(video_paths),
        "extracted_files": len(video_paths) - len(missing),
        "extracted_bytes": extracted_bytes,
        "missing_files": missing,
        "seven_zip_exit_code": completed.returncode,
        "seven_zip_stdout_tail": completed.stdout[-4000:],
        "seven_zip_stderr_tail": completed.stderr[-4000:],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if completed.returncode != 0:
        raise RuntimeError(
            f"7-Zip extraction failed with exit code {completed.returncode}. "
            f"See {report_path}."
        )
    if missing:
        raise FileNotFoundError(
            f"7-Zip completed but {len(missing)} requested videos were absent. "
            f"See {report_path}."
        )
    return report
