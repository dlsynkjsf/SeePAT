from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSettings:
    pilot_manifest: Path
    extracted_root: Path


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


@dataclass(frozen=True)
class PipelineSettings:
    dataset: DatasetSettings
    preprocessing: PreprocessingSettings
    cache_signature: str


def _optional_path(value: object) -> Path | None:
    return Path(str(value)) if value else None


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


def load_pipeline_settings(path: Path, pipeline_version: str) -> PipelineSettings:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("Pipeline configuration must be a YAML mapping")

    dataset_raw = _require_mapping(config, "dataset")
    preprocessing_raw = _require_mapping(config, "preprocessing")
    dataset = DatasetSettings(
        pilot_manifest=Path(str(dataset_raw["pilot_manifest"])),
        extracted_root=Path(str(dataset_raw["extracted_root"])),
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
