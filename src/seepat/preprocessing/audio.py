from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from seepat.artifacts import atomic_write_json
from seepat.config import AudioEnhancementSettings
from seepat.preprocessing.media import extract_mono_audio

AUDIO_PIPELINE_VERSION = "audio-v1"


class AudioEnhancementError(RuntimeError):
    """Raised when an enabled audio-enhancement stage cannot complete."""


@dataclass(frozen=True)
class PreparedAudio:
    raw_audio: Path
    alignment_audio: Path
    vocals_audio: Path | None = None
    enhanced_audio: Path | None = None
    demucs_cache_hit: bool | None = None
    deepfilter_cache_hit: bool | None = None
    normalization_cache_hit: bool | None = None

    def report_fields(self) -> dict[str, object]:
        return {
            "audio_enhancement_enabled": self.enhanced_audio is not None,
            "raw_audio_path": str(self.raw_audio),
            "demucs_vocals_path": (
                str(self.vocals_audio) if self.vocals_audio is not None else ""
            ),
            "deepfilter_audio_path": (
                str(self.enhanced_audio) if self.enhanced_audio is not None else ""
            ),
            "alignment_audio_path": str(self.alignment_audio),
            "demucs_cache_hit": self.demucs_cache_hit,
            "deepfilter_cache_hit": self.deepfilter_cache_hit,
            "audio_normalization_cache_hit": self.normalization_cache_hit,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _require_audio_dependencies() -> None:
    missing = [
        package
        for package, module in (
            ("demucs", "demucs"),
            ("deepfilternet", "df"),
            ("torchaudio", "torchaudio"),
        )
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        names = ", ".join(missing)
        raise AudioEnhancementError(
            f"Missing audio dependencies: {names}. Install the project's audio extra first."
        )


def _stage_is_current(output_path: Path, metadata_path: Path, contract: dict[str, object]) -> bool:
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return metadata.get("contract") == contract


def _publish_stage_output(
    source_path: Path,
    output_path: Path,
    metadata_path: Path,
    contract: dict[str, object],
) -> None:
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise AudioEnhancementError(f"Audio stage created no usable output: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        shutil.copy2(source_path, temporary_path)
        temporary_path.replace(output_path)
        atomic_write_json(metadata_path, {"contract": contract})
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _model_environment(cache_dir: Path) -> dict[str, str]:
    cache_dir = cache_dir.resolve()
    torch_cache = cache_dir / "torch"
    xdg_cache = cache_dir / "xdg"
    local_cache = cache_dir / "localappdata"
    for path in (torch_cache, xdg_cache, local_cache):
        path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TORCH_HOME"] = str(torch_cache)
    environment["XDG_CACHE_HOME"] = str(xdg_cache)
    if os.name == "nt":
        environment["LOCALAPPDATA"] = str(local_cache)
    return environment


def _run_audio_command(command: list[str], cache_dir: Path) -> None:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=_model_environment(cache_dir),
    )
    if completed.returncode != 0:
        details = (completed.stdout + "\n" + completed.stderr).strip()[-6000:]
        raise AudioEnhancementError(
            f"Audio command failed with exit code {completed.returncode}:\n{details}"
        )


def separate_vocals(
    input_path: Path,
    output_path: Path,
    settings: AudioEnhancementSettings,
    force: bool = False,
) -> tuple[Path, bool]:
    metadata_path = output_path.with_suffix(".meta.json")
    contract: dict[str, object] = {
        "stage": "demucs_vocals",
        "implementation": AUDIO_PIPELINE_VERSION,
        "input_sha256": _sha256(input_path),
        "demucs_version": _installed_version("demucs"),
        "model": settings.demucs_model,
        "device": settings.demucs_device,
        "segment_seconds": settings.demucs_segment_seconds,
        "shifts": settings.demucs_shifts,
        "overlap": settings.demucs_overlap,
    }
    if not force and _stage_is_current(output_path, metadata_path, contract):
        return output_path, True

    with tempfile.TemporaryDirectory(prefix=".demucs-", dir=output_path.parent) as directory:
        temporary_dir = Path(directory)
        command = [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "--name",
            settings.demucs_model,
            "--device",
            settings.demucs_device,
            "--shifts",
            str(settings.demucs_shifts),
            "--overlap",
            str(settings.demucs_overlap),
            "--out",
            str(temporary_dir),
            "--filename",
            "{track}_{stem}.{ext}",
        ]
        if settings.demucs_segment_seconds is not None:
            command.extend(["--segment", str(settings.demucs_segment_seconds)])
        command.append(str(input_path.resolve()))
        _run_audio_command(command, settings.model_cache_dir)
        generated = temporary_dir / settings.demucs_model / f"{input_path.stem}_vocals.wav"
        _publish_stage_output(generated, output_path, metadata_path, contract)
    return output_path, False


def denoise_vocals(
    input_path: Path,
    output_path: Path,
    settings: AudioEnhancementSettings,
    force: bool = False,
) -> tuple[Path, bool]:
    metadata_path = output_path.with_suffix(".meta.json")
    contract: dict[str, object] = {
        "stage": "deepfilter_denoise",
        "implementation": AUDIO_PIPELINE_VERSION,
        "input_sha256": _sha256(input_path),
        "deepfilternet_version": _installed_version("deepfilternet"),
        "model": settings.deepfilter_model,
        "post_filter": settings.deepfilter_post_filter,
    }
    if not force and _stage_is_current(output_path, metadata_path, contract):
        return output_path, True

    with tempfile.TemporaryDirectory(prefix=".deepfilter-", dir=output_path.parent) as directory:
        temporary_dir = Path(directory)
        command = [
            sys.executable,
            "-m",
            "df.enhance",
            "--model-base-dir",
            settings.deepfilter_model,
            "--output-dir",
            str(temporary_dir),
            "--log-level",
            "error",
            "--no-suffix",
        ]
        if settings.deepfilter_post_filter:
            command.append("--pf")
        command.append(str(input_path.resolve()))
        _run_audio_command(command, settings.model_cache_dir)
        generated = temporary_dir / input_path.name
        _publish_stage_output(generated, output_path, metadata_path, contract)
    return output_path, False


def normalize_alignment_audio(
    input_path: Path,
    output_path: Path,
    ffmpeg_path: Path | None,
    force: bool = False,
) -> tuple[Path, bool]:
    metadata_path = output_path.with_suffix(".meta.json")
    contract: dict[str, object] = {
        "stage": "alignment_audio_normalization",
        "implementation": AUDIO_PIPELINE_VERSION,
        "input_sha256": _sha256(input_path),
        "channels": 1,
        "sample_rate_hz": 16000,
        "codec": "pcm_s16le",
    }
    if not force and _stage_is_current(output_path, metadata_path, contract):
        return output_path, True
    extract_mono_audio(input_path, output_path, ffmpeg_path, force=True)
    atomic_write_json(metadata_path, {"contract": contract})
    return output_path, False


def prepare_alignment_audio(
    raw_audio: Path,
    work_dir: Path,
    settings: AudioEnhancementSettings,
    ffmpeg_path: Path | None,
    force: bool = False,
) -> PreparedAudio:
    if not settings.enabled:
        return PreparedAudio(raw_audio=raw_audio, alignment_audio=raw_audio)

    _require_audio_dependencies()
    work_dir.mkdir(parents=True, exist_ok=True)
    vocals, demucs_hit = separate_vocals(
        raw_audio,
        work_dir / "demucs_vocals.wav",
        settings,
        force=force,
    )
    enhanced, deepfilter_hit = denoise_vocals(
        vocals,
        work_dir / "deepfilter_enhanced.wav",
        settings,
        force=force,
    )
    alignment, normalization_hit = normalize_alignment_audio(
        enhanced,
        work_dir / "alignment_audio.wav",
        ffmpeg_path,
        force=force,
    )
    return PreparedAudio(
        raw_audio=raw_audio,
        vocals_audio=vocals,
        enhanced_audio=enhanced,
        alignment_audio=alignment,
        demucs_cache_hit=demucs_hit,
        deepfilter_cache_hit=deepfilter_hit,
        normalization_cache_hit=normalization_hit,
    )
