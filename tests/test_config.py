from __future__ import annotations

from pathlib import Path

import pytest

from seepat.config import load_pipeline_settings


def _write_config(path: Path, min_valid_landmark_ratio: float = 0.8) -> None:
    path.write_text(
        f"""
dataset:
  pilot_manifest: data/pilot.csv
  extracted_root: data/videos
preprocessing:
  output_dir: outputs/pilot
  ffmpeg_path: null
  ffprobe_path: null
  whisper_model: base.en
  whisper_device: cpu
  whisper_compute_type: int8
  mfa_docker_image: example/mfa:latest
  mfa_dictionary: english_us_arpa
  mfa_acoustic_model: english_us_arpa
  mfa_cache_dir: data/cache/mfa
  event_window_before_s: 0.2
  event_window_after_s: 0.2
  manipulation_boundary_tolerance_s: 0.04
  min_valid_landmark_ratio: {min_valid_landmark_ratio}
  min_bilabial_events: 1
  max_debug_overlays_per_video: 10
""".lstrip(),
        encoding="utf-8",
    )


def test_load_pipeline_settings_is_typed_and_stable(tmp_path: Path) -> None:
    config_path = tmp_path / "pilot.yaml"
    _write_config(config_path)

    first = load_pipeline_settings(config_path, pipeline_version="pilot-v3")
    second = load_pipeline_settings(config_path, pipeline_version="pilot-v3")

    assert first.dataset.pilot_manifest == Path("data/pilot.csv")
    assert first.preprocessing.min_valid_landmark_ratio == 0.8
    assert first.preprocessing.ffmpeg_path is None
    assert len(first.cache_signature) == 64
    assert first.cache_signature == second.cache_signature


def test_load_pipeline_settings_rejects_invalid_ratio(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    _write_config(config_path, min_valid_landmark_ratio=1.1)

    with pytest.raises(ValueError, match="between 0 and 1"):
        load_pipeline_settings(config_path, pipeline_version="pilot-v3")
