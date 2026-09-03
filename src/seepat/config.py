from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSettings:
    split: str
    pilot_manifest: Path
    extracted_root: Path
    include_video_ids: tuple[str, ...]


@dataclass(frozen=True)
class AudioEnhancementSettings:
    enabled: bool
    model_cache_dir: Path
    demucs_model: str
    demucs_device: str
    demucs_segment_seconds: int | None
    demucs_shifts: int
    demucs_overlap: float
    deepfilter_executable: Path | None
    deepfilter_model: str
    deepfilter_compensate_delay: bool
    deepfilter_post_filter: bool
    deepfilter_attenuation_limit_db: float = 100.0
    deepfilter_fallback_enabled: bool = False


@dataclass(frozen=True)
class PreprocessingSettings:
    output_dir: Path
    ffmpeg_path: Path | None
    ffprobe_path: Path | None
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    mfa_docker_image: str
    mfa_dictionary: str
    mfa_acoustic_model: str
    mfa_cache_dir: Path
    event_window_before_s: float
    event_window_after_s: float
    manipulation_boundary_tolerance_s: float
    min_valid_landmark_ratio: float
    min_bilabial_events: int
    max_debug_overlays_per_video: int
    audio_enhancement: AudioEnhancementSettings


@dataclass(frozen=True)
class PipelineSettings:
    dataset: DatasetSettings
    preprocessing: PreprocessingSettings
    cache_signature: str


def _optional_path(value: object) -> Path | None:
    return Path(str(value)) if value else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError("demucs_segment_seconds must be a whole number")
    return int(numeric)


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section '{key}' must be a YAML mapping")
    return value


def _validate(settings: PreprocessingSettings) -> None:
    if settings.event_window_before_s < 0 or settings.event_window_after_s < 0:
        raise ValueError("Event-window durations cannot be negative")
    if settings.manipulation_boundary_tolerance_s < 0:
        raise ValueError("Manipulation boundary tolerance cannot be negative")
    if not 0 <= settings.min_valid_landmark_ratio <= 1:
        raise ValueError("min_valid_landmark_ratio must be between 0 and 1")
    if settings.min_bilabial_events < 1:
        raise ValueError("min_bilabial_events must be at least 1")
    if settings.max_debug_overlays_per_video < 0:
        raise ValueError("max_debug_overlays_per_video cannot be negative")
    audio = settings.audio_enhancement
    if audio.demucs_device not in {"cpu", "cuda"}:
        raise ValueError("demucs_device must be either 'cpu' or 'cuda'")
    if audio.demucs_segment_seconds is not None and audio.demucs_segment_seconds <= 0:
        raise ValueError("demucs_segment_seconds must be positive when provided")
    if audio.demucs_shifts < 0:
        raise ValueError("demucs_shifts cannot be negative")
    if not 0 <= audio.demucs_overlap < 1:
        raise ValueError("demucs_overlap must be at least 0 and less than 1")
    if audio.deepfilter_model != "DeepFilterNet3":
        raise ValueError(
            "The native DeepFilterNet integration currently supports only DeepFilterNet3"
        )
    if not 0 <= audio.deepfilter_attenuation_limit_db <= 100:
        raise ValueError("deepfilter_attenuation_limit_db must be between 0 and 100")


def load_pipeline_settings(path: Path, pipeline_version: str) -> PipelineSettings:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("Pipeline configuration must be a YAML mapping")

    dataset_raw = _require_mapping(config, "dataset")
    preprocessing_raw = _require_mapping(config, "preprocessing")
    audio_raw = preprocessing_raw.get("audio_enhancement", {})
    if not isinstance(audio_raw, dict):
        raise TypeError(
            "Configuration section 'preprocessing.audio_enhancement' must be a YAML mapping"
        )
    include_video_ids_raw = dataset_raw.get("include_video_ids", [])
    if not isinstance(include_video_ids_raw, list):
        raise TypeError("dataset.include_video_ids must be a YAML list")
    include_video_ids = tuple(str(value).strip() for value in include_video_ids_raw)
    if any(not value for value in include_video_ids):
        raise ValueError("dataset.include_video_ids cannot contain empty values")
    if len(include_video_ids) != len(set(include_video_ids)):
        raise ValueError("dataset.include_video_ids cannot contain duplicates")

    dataset = DatasetSettings(
        split=str(dataset_raw.get("split", "")).strip(),
        pilot_manifest=Path(str(dataset_raw["pilot_manifest"])),
        extracted_root=Path(str(dataset_raw["extracted_root"])),
        include_video_ids=include_video_ids,
    )
    preprocessing = PreprocessingSettings(
        output_dir=Path(str(preprocessing_raw["output_dir"])),
        ffmpeg_path=_optional_path(preprocessing_raw.get("ffmpeg_path")),
        ffprobe_path=_optional_path(preprocessing_raw.get("ffprobe_path")),
        whisper_model=str(preprocessing_raw["whisper_model"]),
        whisper_device=str(preprocessing_raw["whisper_device"]),
        whisper_compute_type=str(preprocessing_raw["whisper_compute_type"]),
        mfa_docker_image=str(preprocessing_raw["mfa_docker_image"]),
        mfa_dictionary=str(preprocessing_raw["mfa_dictionary"]),
        mfa_acoustic_model=str(preprocessing_raw["mfa_acoustic_model"]),
        mfa_cache_dir=Path(str(preprocessing_raw["mfa_cache_dir"])),
        event_window_before_s=float(preprocessing_raw["event_window_before_s"]),
        event_window_after_s=float(preprocessing_raw["event_window_after_s"]),
        manipulation_boundary_tolerance_s=float(
            preprocessing_raw["manipulation_boundary_tolerance_s"]
        ),
        min_valid_landmark_ratio=float(
            preprocessing_raw["min_valid_landmark_ratio"]
        ),
        min_bilabial_events=int(preprocessing_raw["min_bilabial_events"]),
        max_debug_overlays_per_video=int(
            preprocessing_raw["max_debug_overlays_per_video"]
        ),
        audio_enhancement=AudioEnhancementSettings(
            enabled=bool(audio_raw.get("enabled", False)),
            model_cache_dir=Path(
                str(audio_raw.get("model_cache_dir", "data/cache/audio_models"))
            ),
            demucs_model=str(audio_raw.get("demucs_model", "htdemucs")),
            demucs_device=str(audio_raw.get("demucs_device", "cpu")),
            demucs_segment_seconds=_optional_int(
                audio_raw.get("demucs_segment_seconds")
            ),
            demucs_shifts=int(audio_raw.get("demucs_shifts", 0)),
            demucs_overlap=float(audio_raw.get("demucs_overlap", 0.25)),
            deepfilter_executable=_optional_path(
                audio_raw.get("deepfilter_executable")
            ),
            deepfilter_model=str(audio_raw.get("deepfilter_model", "DeepFilterNet3")),
            deepfilter_compensate_delay=bool(
                audio_raw.get("deepfilter_compensate_delay", True)
            ),
            deepfilter_post_filter=bool(
                audio_raw.get("deepfilter_post_filter", False)
            ),
            deepfilter_attenuation_limit_db=float(
                audio_raw.get("deepfilter_attenuation_limit_db", 100.0)
            ),
            deepfilter_fallback_enabled=bool(
                audio_raw.get("deepfilter_fallback_enabled", False)
            ),
        ),
    )
    _validate(preprocessing)

    signature_payload = {
        "pipeline_version": pipeline_version,
        "preprocessing": preprocessing_raw,
    }
    encoded = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    return PipelineSettings(
        dataset=dataset,
        preprocessing=preprocessing,
        cache_signature=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )
