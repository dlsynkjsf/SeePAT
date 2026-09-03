from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from seepat.artifacts import atomic_write_json
from seepat.config import AudioEnhancementSettings
from seepat.preprocessing.media import extract_mono_audio, find_media_binary

AUDIO_PIPELINE_VERSION = "audio-v2"
DEEPFILTER_STAGE_VERSION = "deepfilter-v3"
DEEPFILTER_VERSION = "0.5.6"
DEEPFILTER_FLUSH_PADDING_SECONDS = 0.1
FALLBACK_WINDOW_SECONDS = 0.25
FALLBACK_ACTIVE_FLOOR_DB = -40.0
FALLBACK_LIMIT_TOLERANCE_DB = 0.5
FALLBACK_MIN_SUSTAINED_SECONDS = 0.5


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
    deepfilter_fallback_applied: bool | None = None
    deepfilter_max_sustained_attenuation_s: float | None = None

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
            "deepfilter_fallback_applied": self.deepfilter_fallback_applied,
            "deepfilter_max_sustained_attenuation_s": (
                self.deepfilter_max_sustained_attenuation_s
            ),
            "alignment_audio_source": (
                "demucs_fallback"
                if self.deepfilter_fallback_applied
                else "deepfilter"
                if self.enhanced_audio is not None
                else "raw"
            ),
        }


@dataclass(frozen=True)
class DeepFilterResult:
    path: Path
    cache_hit: bool
    fallback_applied: bool
    max_sustained_attenuation_s: float | None


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


def find_deepfilter_binary(explicit_path: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)

    executable_names = (
        ("deep-filter.exe", "deep-filter")
        if os.name == "nt"
        else ("deep-filter", "deep-filter.exe")
    )
    local_tools = Path(".tools/deepfilter")
    candidates.extend(local_tools / name for name in executable_names)
    candidates.extend(sorted(local_tools.glob("deep-filter-*")))

    for name in executable_names:
        discovered = shutil.which(name)
        if discovered:
            candidates.append(Path(discovered))

    non_executable: Path | None = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if os.name != "nt" and not os.access(resolved, os.X_OK):
            non_executable = resolved
            continue
        return resolved

    if non_executable is not None:
        raise AudioEnhancementError(
            f"DeepFilterNet executable is not executable: {non_executable}. "
            "Run chmod +x on it."
        )
    raise AudioEnhancementError(
        "DeepFilterNet's deep-filter executable was not found. Install the pinned "
        "v0.5.6 binary under .tools/deepfilter, add it to PATH, or configure "
        "deepfilter_executable."
    )


@cache
def _cached_deepfilter_identity(
    executable: str, size: int, modified_ns: int
) -> tuple[str, str]:
    del size, modified_ns
    path = Path(executable)
    completed = subprocess.run(
        [str(path), "--version"],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        details = (completed.stdout + "\n" + completed.stderr).strip()[-2000:]
        raise AudioEnhancementError(
            "Could not read the DeepFilterNet executable version "
            f"(exit code {completed.returncode}):\n{details}"
        )
    reported_version = (completed.stdout or completed.stderr).strip()
    if not reported_version:
        reported_version = "unknown"
    if DEEPFILTER_VERSION not in reported_version:
        raise AudioEnhancementError(
            f"Expected deep-filter {DEEPFILTER_VERSION}, but {path} reported "
            f"{reported_version!r}."
        )
    return reported_version, _sha256(path)


def _deepfilter_identity(executable: Path) -> tuple[str, str]:
    stat = executable.stat()
    return _cached_deepfilter_identity(
        str(executable.resolve()), stat.st_size, stat.st_mtime_ns
    )


def _require_audio_dependencies(settings: AudioEnhancementSettings) -> None:
    if importlib.util.find_spec("demucs") is None:
        raise AudioEnhancementError(
            "Missing audio dependency: demucs. Install the project's audio extra first."
        )
    find_deepfilter_binary(settings.deepfilter_executable)


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
    outcome: dict[str, object] | None = None,
) -> None:
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise AudioEnhancementError(f"Audio stage created no usable output: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        shutil.copy2(source_path, temporary_path)
        temporary_path.replace(output_path)
        metadata: dict[str, object] = {"contract": contract}
        if outcome is not None:
            metadata["outcome"] = outcome
        atomic_write_json(metadata_path, metadata)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_stage_outcome(metadata_path: Path) -> dict[str, object]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    outcome = metadata.get("outcome", {})
    return outcome if isinstance(outcome, dict) else {}


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


def _run_media_command(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        details = (completed.stdout + "\n" + completed.stderr).strip()[-6000:]
        raise AudioEnhancementError(
            f"Audio media command failed with exit code {completed.returncode}:\n{details}"
        )


def _wav_properties(path: Path) -> tuple[int, int, int]:
    try:
        with wave.open(str(path), "rb") as stream:
            return stream.getframerate(), stream.getnframes(), stream.getnchannels()
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioEnhancementError(f"Could not read WAV properties from {path}") from exc


def _read_pcm16_mono(path: Path) -> tuple[int, object]:
    try:
        with wave.open(str(path), "rb") as stream:
            if stream.getsampwidth() != 2:
                raise AudioEnhancementError(
                    f"Sustained-attenuation analysis requires 16-bit PCM WAV: {path}"
                )
            sample_rate = stream.getframerate()
            channels = stream.getnchannels()
            frame_count = stream.getnframes()
            payload = stream.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioEnhancementError(f"Could not read PCM WAV samples from {path}") from exc

    import numpy as np

    samples = np.frombuffer(payload, dtype="<i2").astype(np.float64)
    if samples.size != frame_count * channels:
        raise AudioEnhancementError(f"Unexpected PCM sample count in {path}")
    if channels > 1:
        samples = samples.reshape(frame_count, channels).mean(axis=1)
    return sample_rate, samples


def _analyze_sustained_attenuation(
    input_path: Path,
    enhanced_path: Path,
    attenuation_limit_db: float,
) -> dict[str, object]:
    import numpy as np

    input_rate, input_audio = _read_pcm16_mono(input_path)
    enhanced_rate, enhanced_audio = _read_pcm16_mono(enhanced_path)
    if input_rate != enhanced_rate or len(input_audio) != len(enhanced_audio):
        raise AudioEnhancementError(
            "DeepFilter fallback analysis requires matching sample rates and lengths"
        )

    peak = float(np.max(np.abs(input_audio))) if len(input_audio) else 0.0
    window_frames = max(1, round(FALLBACK_WINDOW_SECONDS * input_rate))
    near_limit_threshold_db = -(attenuation_limit_db - FALLBACK_LIMIT_TOLERANCE_DB)
    current_frames = 0
    longest_frames = 0

    if peak > 0:
        for start in range(0, len(input_audio), window_frames):
            input_window = input_audio[start : start + window_frames]
            enhanced_window = enhanced_audio[start : start + window_frames]
            input_rms = float(np.sqrt(np.mean(input_window * input_window)))
            enhanced_rms = float(np.sqrt(np.mean(enhanced_window * enhanced_window)))
            if input_rms <= 0:
                current_frames = 0
                continue
            input_relative_db = 20 * np.log10(input_rms / peak)
            attenuation_db = 20 * np.log10(max(enhanced_rms, 1e-12) / input_rms)
            near_limit = (
                input_relative_db >= FALLBACK_ACTIVE_FLOOR_DB
                and attenuation_db <= near_limit_threshold_db
            )
            if near_limit:
                current_frames += len(input_window)
                longest_frames = max(longest_frames, current_frames)
            else:
                current_frames = 0

    sustained_seconds = longest_frames / input_rate
    return {
        "fallback_applied": sustained_seconds >= FALLBACK_MIN_SUSTAINED_SECONDS,
        "max_sustained_attenuation_seconds": round(sustained_seconds, 6),
        "analysis_window_seconds": FALLBACK_WINDOW_SECONDS,
        "active_floor_db": FALLBACK_ACTIVE_FLOOR_DB,
        "near_limit_threshold_db": near_limit_threshold_db,
        "minimum_sustained_seconds": FALLBACK_MIN_SUSTAINED_SECONDS,
    }


def _pad_audio_tail(
    input_path: Path,
    output_path: Path,
    ffmpeg_path: Path | None,
    padding_seconds: float,
) -> None:
    ffmpeg = find_media_binary("ffmpeg", ffmpeg_path)
    sample_rate, _, channels = _wav_properties(input_path)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-af",
        f"apad=pad_dur={padding_seconds:g}",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    _run_media_command(command)


def _restore_reference_duration(
    input_path: Path,
    reference_path: Path,
    output_path: Path,
    ffmpeg_path: Path | None,
) -> None:
    ffmpeg = find_media_binary("ffmpeg", ffmpeg_path)
    sample_rate, frames, channels = _wav_properties(reference_path)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-af",
        f"aresample={sample_rate},apad,atrim=end_sample={frames}",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    _run_media_command(command)
    output_rate, output_frames, output_channels = _wav_properties(output_path)
    if (output_rate, output_frames, output_channels) != (sample_rate, frames, channels):
        raise AudioEnhancementError(
            "DeepFilterNet duration restoration produced an unexpected WAV shape: "
            f"expected {(sample_rate, frames, channels)}, got "
            f"{(output_rate, output_frames, output_channels)}"
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
    ffmpeg_path: Path | None = None,
    force: bool = False,
) -> DeepFilterResult:
    executable = find_deepfilter_binary(settings.deepfilter_executable)
    deepfilter_version, executable_sha256 = _deepfilter_identity(executable)
    metadata_path = output_path.with_suffix(".meta.json")
    contract: dict[str, object] = {
        "stage": "deepfilter_denoise",
        "implementation": DEEPFILTER_STAGE_VERSION,
        "input_sha256": _sha256(input_path),
        "deepfilternet_version": deepfilter_version,
        "executable_sha256": executable_sha256,
        "model": settings.deepfilter_model,
        "compensate_delay": settings.deepfilter_compensate_delay,
        "post_filter": settings.deepfilter_post_filter,
        "attenuation_limit_db": settings.deepfilter_attenuation_limit_db,
        "fallback_enabled": settings.deepfilter_fallback_enabled,
        "fallback_window_seconds": FALLBACK_WINDOW_SECONDS,
        "fallback_active_floor_db": FALLBACK_ACTIVE_FLOOR_DB,
        "fallback_limit_tolerance_db": FALLBACK_LIMIT_TOLERANCE_DB,
        "fallback_min_sustained_seconds": FALLBACK_MIN_SUSTAINED_SECONDS,
        "preserve_input_duration": settings.deepfilter_compensate_delay,
        "flush_padding_seconds": (
            DEEPFILTER_FLUSH_PADDING_SECONDS
            if settings.deepfilter_compensate_delay
            else 0.0
        ),
    }
    if not force and _stage_is_current(output_path, metadata_path, contract):
        cached_outcome = _read_stage_outcome(metadata_path)
        return DeepFilterResult(
            path=output_path,
            cache_hit=True,
            fallback_applied=bool(cached_outcome.get("fallback_applied", False)),
            max_sustained_attenuation_s=cached_outcome.get(
                "max_sustained_attenuation_seconds"
            ),
        )

    with tempfile.TemporaryDirectory(prefix=".deepfilter-", dir=output_path.parent) as directory:
        temporary_dir = Path(directory)
        generated_dir = temporary_dir / "output"
        generated_dir.mkdir()
        command_input = input_path.resolve()
        if settings.deepfilter_compensate_delay:
            padded_dir = temporary_dir / "input"
            padded_dir.mkdir()
            command_input = padded_dir / "deepfilter_input_padded.wav"
            _pad_audio_tail(
                input_path.resolve(),
                command_input,
                ffmpeg_path,
                DEEPFILTER_FLUSH_PADDING_SECONDS,
            )
        command = [
            str(executable),
            "--output-dir",
            str(generated_dir),
        ]
        if settings.deepfilter_compensate_delay:
            command.append("--compensate-delay")
        if settings.deepfilter_post_filter:
            command.append("--pf")
        command.extend(
            [
                "--atten-lim-db",
                f"{settings.deepfilter_attenuation_limit_db:g}",
                str(command_input),
            ]
        )
        _run_audio_command(command, settings.model_cache_dir)
        generated = generated_dir / command_input.name
        publish_source = generated
        if settings.deepfilter_compensate_delay:
            publish_source = temporary_dir / "deepfilter_duration_restored.wav"
            _restore_reference_duration(
                generated,
                input_path.resolve(),
                publish_source,
                ffmpeg_path,
            )
        outcome: dict[str, object] = {
            "fallback_applied": False,
            "max_sustained_attenuation_seconds": None,
        }
        if settings.deepfilter_fallback_enabled:
            outcome = _analyze_sustained_attenuation(
                input_path.resolve(),
                publish_source,
                settings.deepfilter_attenuation_limit_db,
            )
        _publish_stage_output(
            publish_source,
            output_path,
            metadata_path,
            contract,
            outcome=outcome,
        )
    return DeepFilterResult(
        path=output_path,
        cache_hit=False,
        fallback_applied=bool(outcome["fallback_applied"]),
        max_sustained_attenuation_s=outcome[
            "max_sustained_attenuation_seconds"
        ],
    )


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

    _require_audio_dependencies(settings)
    work_dir.mkdir(parents=True, exist_ok=True)
    vocals, demucs_hit = separate_vocals(
        raw_audio,
        work_dir / "demucs_vocals.wav",
        settings,
        force=force,
    )
    deepfilter = denoise_vocals(
        vocals,
        work_dir / "deepfilter_enhanced.wav",
        settings,
        ffmpeg_path=ffmpeg_path,
        force=force,
    )
    alignment_source = vocals if deepfilter.fallback_applied else deepfilter.path
    alignment, normalization_hit = normalize_alignment_audio(
        alignment_source,
        work_dir / "alignment_audio.wav",
        ffmpeg_path,
        force=force,
    )
    return PreparedAudio(
        raw_audio=raw_audio,
        vocals_audio=vocals,
        enhanced_audio=deepfilter.path,
        alignment_audio=alignment,
        demucs_cache_hit=demucs_hit,
        deepfilter_cache_hit=deepfilter.cache_hit,
        normalization_cache_hit=normalization_hit,
        deepfilter_fallback_applied=deepfilter.fallback_applied,
        deepfilter_max_sustained_attenuation_s=(
            deepfilter.max_sustained_attenuation_s
        ),
    )
