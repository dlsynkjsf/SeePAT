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


def test_train_canary_config_points_to_separate_inputs_and_outputs() -> None:
    settings = load_pipeline_settings(
        Path("configs/train_canary1000.yaml"),
        pipeline_version="pilot-v3",
    )

    assert settings.dataset.pilot_manifest == Path(
        "data/manifests/train_canary1000.csv"
    )
    assert settings.dataset.extracted_root == Path(
        "data/extracted/avpp/train-canary1000/train"
    )
    assert settings.preprocessing.output_dir == Path("outputs/train-canary1000")


@pytest.mark.parametrize(
    ("config_name", "manifest", "extracted_root", "output_dir", "per_video_overlays"),
    [
        (
            "train_subset5000.yaml",
            "data/manifests/train_subset5000.csv",
            "data/extracted/avpp/train-subset5000/train",
            "outputs/train-subset5000",
            1,
        ),
        (
            "val_subset1000.yaml",
            "data/manifests/val_subset1000.csv",
            "data/extracted/avpp/val-subset1000/val",
            "outputs/val-subset1000",
            1,
        ),
    ],
)
def test_scaled_subset_configs_use_isolated_paths(
    config_name: str,
    manifest: str,
    extracted_root: str,
    output_dir: str,
    per_video_overlays: int,
) -> None:
    settings = load_pipeline_settings(
        Path("configs") / config_name,
        pipeline_version="pilot-v3",
    )

    assert settings.dataset.pilot_manifest == Path(manifest)
    assert settings.dataset.extracted_root == Path(extracted_root)
    assert settings.preprocessing.output_dir == Path(output_dir)
    assert settings.preprocessing.max_debug_overlays_per_video == per_video_overlays
