from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def find_media_binary(name: str, explicit_path: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)

    discovered = shutil.which(name)
    if discovered:
        candidates.append(Path(discovered))

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        winget_packages = Path(local_app_data) / "Microsoft/WinGet/Packages"
        candidates.extend(winget_packages.glob(f"Gyan.FFmpeg_*/ffmpeg-*/bin/{name}.exe"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"{name} was not found. Install FFmpeg or configure its explicit path."
    )


def parse_fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", maxsplit=1)
    if float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def _optional_float(value: object, default: float | None = None) -> float | None:
    if value in (None, "N/A", ""):
        return default
    return float(value)


def _optional_int(value: object) -> int | None:
    if value in (None, "N/A", ""):
        return None
    return int(value)


def probe_media(video_path: Path, ffprobe_path: Path | None = None) -> dict[str, object]:
    ffprobe = find_media_binary("ffprobe", ffprobe_path)
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    raw = json.loads(completed.stdout)
    video_stream = next(
        (stream for stream in raw.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    audio_stream = next(
        (stream for stream in raw.get("streams", []) if stream.get("codec_type") == "audio"),
        None,
    )
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    duration_value = video_stream.get("duration") or raw.get("format", {}).get("duration")
    duration_s = _optional_float(duration_value)
    average_frame_rate = parse_fraction(video_stream.get("avg_frame_rate"))
    nominal_frame_rate = parse_fraction(video_stream.get("r_frame_rate"))
    fps = average_frame_rate or nominal_frame_rate
    video_start_time_s = _optional_float(video_stream.get("start_time"), 0.0)
    audio_start_time_s = (
        _optional_float(audio_stream.get("start_time"), 0.0) if audio_stream else None
    )
    return {
        "duration_s": duration_s,
        "fps": fps,
        "average_frame_rate": average_frame_rate,
        "nominal_frame_rate": nominal_frame_rate,
        "video_time_base": video_stream.get("time_base"),
        "video_frame_count": _optional_int(video_stream.get("nb_frames")),
        "video_start_time_s": video_start_time_s,
        "audio_start_time_s": audio_start_time_s,
        "audio_video_start_offset_s": (
            audio_start_time_s - video_start_time_s
            if audio_start_time_s is not None and video_start_time_s is not None
            else None
        ),
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "video_codec": video_stream.get("codec_name"),
        "audio_present": audio_stream is not None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "audio_sample_rate": (
            int(audio_stream["sample_rate"])
            if audio_stream and audio_stream.get("sample_rate")
            else None
        ),
    }


def extract_mono_audio(
    video_path: Path,
    output_path: Path,
    ffmpeg_path: Path | None = None,
    force: bool = False,
) -> Path:
    if output_path.is_file() and not force:
        return output_path

    ffmpeg = find_media_binary("ffmpeg", ffmpeg_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(temporary_path),
    ]
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path
