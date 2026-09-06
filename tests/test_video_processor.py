from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from seepat.artifacts import read_gzip_json
from seepat.config import AudioEnhancementSettings
from seepat.preprocessing.alignment import MfaAlignmentError
from seepat.preprocessing.audio import PreparedAudio
from seepat.video_processor import PilotVideoProcessor


def _disabled_audio_settings(tmp_path: Path) -> AudioEnhancementSettings:
    return AudioEnhancementSettings(
        enabled=False,
        model_cache_dir=tmp_path / "models",
        demucs_model="htdemucs",
        demucs_device="cpu",
        demucs_segment_seconds=None,
        demucs_shifts=0,
        demucs_overlap=0.25,
        deepfilter_executable=None,
        deepfilter_model="DeepFilterNet3",
        deepfilter_compensate_delay=True,
        deepfilter_post_filter=False,
    )


def test_mfa_failure_is_reported_as_alignment_failed(
    tmp_path: Path, monkeypatch
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    preprocessing = SimpleNamespace(
        ffprobe_path=None,
        ffmpeg_path=None,
        audio_enhancement=_disabled_audio_settings(tmp_path),
    )
    settings = SimpleNamespace(
        dataset=SimpleNamespace(extracted_root=tmp_path),
        preprocessing=preprocessing,
    )
    transcriber = SimpleNamespace(
        transcribe=lambda audio_path: {"text": "please bring me paper"}
    )

    def fail_alignment(*args, **kwargs):
        raise MfaAlignmentError("Could not align with the current beam size")

    aligner = SimpleNamespace(align=fail_alignment)
    processor = PilotVideoProcessor(
        settings=settings,
        transcriber=transcriber,
        aligner=aligner,
        mouth_analyzer=object(),
    )

    monkeypatch.setattr(
        "seepat.video_processor.probe_media",
        lambda path, ffprobe_path: {"audio_present": True, "fps": 25.0},
    )

    def fake_extract(video_path, output_path, ffmpeg_path, force):
        output_path.write_bytes(b"audio")
        return output_path

    monkeypatch.setattr("seepat.video_processor.extract_mono_audio", fake_extract)

    report, events = processor.process(
        manifest_row={"file": "video.mp4", "split": "train"},
        video_id="video-1",
        work_dir=work_dir,
        force=False,
    )

    assert events == []
    assert report["pipeline_status"] == "failed"
    assert report["eligibility_status"] == "ineligible"
    assert report["exclusion_reason"] == "alignment_failed"
    assert report["error_type"] == "MfaAlignmentError"


def test_enhanced_audio_is_used_for_both_whisper_and_mfa(
    tmp_path: Path, monkeypatch
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    preprocessing = SimpleNamespace(
        ffprobe_path=None,
        ffmpeg_path=None,
        audio_enhancement=_disabled_audio_settings(tmp_path),
    )
    settings = SimpleNamespace(
        dataset=SimpleNamespace(extracted_root=tmp_path),
        preprocessing=preprocessing,
    )
    used_by_whisper: list[Path] = []
    used_by_mfa: list[Path] = []
    alignment_force: list[bool] = []

    def transcribe(audio_path: Path) -> dict[str, str]:
        used_by_whisper.append(audio_path)
        return {"text": "please bring me paper"}

    def align(audio_path: Path, *args, **kwargs):
        used_by_mfa.append(audio_path)
        alignment_force.append(kwargs["force"])
        raise MfaAlignmentError("stop after input check")

    processor = PilotVideoProcessor(
        settings=settings,
        transcriber=SimpleNamespace(transcribe=transcribe),
        aligner=SimpleNamespace(align=align),
        mouth_analyzer=object(),
    )
    enhanced_audio = work_dir / "alignment_audio.wav"
    enhanced_audio.write_bytes(b"enhanced")
    (work_dir / "transcription.json").write_text(
        '{"text": "stale transcript"}', encoding="utf-8"
    )
    monkeypatch.setattr(
        "seepat.video_processor.probe_media",
        lambda path, ffprobe_path: {"audio_present": True, "fps": 25.0},
    )

    def fake_extract(video_path, output_path, ffmpeg_path, force):
        output_path.write_bytes(b"raw")
        return output_path

    monkeypatch.setattr("seepat.video_processor.extract_mono_audio", fake_extract)
    monkeypatch.setattr(
        "seepat.video_processor.prepare_alignment_audio",
        lambda **kwargs: PreparedAudio(
            raw_audio=work_dir / "audio.wav",
            vocals_audio=work_dir / "demucs_vocals.wav",
            enhanced_audio=work_dir / "deepfilter_enhanced.wav",
            alignment_audio=enhanced_audio,
            normalization_cache_hit=False,
        ),
    )

    processor.process(
        manifest_row={"file": "video.mp4", "split": "train"},
        video_id="video-1",
        work_dir=work_dir,
        force=False,
    )

    assert used_by_whisper == [enhanced_audio]
    assert used_by_mfa == [enhanced_audio]
    assert alignment_force == [True]


def test_vild_trace_preserves_frames_references_and_no_labels(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    preprocessing = SimpleNamespace(
        output_dir=output_dir,
        event_window_before_s=0.2,
        event_window_after_s=0.2,
        vild_trace=SimpleNamespace(
            reference_window_seconds=0.08,
            max_reference_windows=4,
            speech_margin_seconds=0.02,
        ),
    )
    processor = PilotVideoProcessor(
        settings=SimpleNamespace(preprocessing=preprocessing),
        transcriber=object(),
        aligner=object(),
        mouth_analyzer=object(),
    )
    event = {
        "event_id": "video-1-000",
        "event_label": "fake",
        "phoneme": "p",
        "phone_start_s": 1.0,
        "phone_end_s": 1.1,
        "video_phone_start_s": 1.0,
        "video_phone_end_s": 1.1,
        "mouth_crop_frame_indices_json": "[8,9,11]",
    }
    frames = [
        {
            "frame_index": index,
            "timestamp_s": index * 0.1,
            "face_count": 1,
            "normalized_vild": 0.1 + index * 0.01,
        }
        for index in range(20)
    ]
    report: dict[str, object] = {}

    processor._write_vild_trace(
        manifest_row={
            "file": "video.mp4",
            "split": "train",
            "source_group": "source-1",
            "subject_id": "person-1",
        },
        video_id="video-1",
        probe={
            "fps": 10.0,
            "duration_s": 2.0,
            "average_frame_rate": 10.0,
            "nominal_frame_rate": 10.0,
            "video_time_base": "1/1000",
            "video_start_time_s": 0.0,
            "audio_start_time_s": 0.0,
        },
        transcription={"segments": [{"start": 0.6, "end": 1.4}]},
        phone_intervals=[
            {"phoneme": "sil", "phone_start_s": 0.0, "phone_end_s": 0.3},
            {"phoneme": "p", "phone_start_s": 0.3, "phone_end_s": 1.7},
            {"phoneme": "sil", "phone_start_s": 1.7, "phone_end_s": 2.0},
        ],
        events=[event],
        trace_result={
            "frames": frames,
            "summary": {
                "attempted_frames": 20,
                "valid_landmark_ratio": 1.0,
            },
        },
        audio_video_offset_s=0.0,
        report=report,
    )

    trace_path = Path(str(report["vild_trace_path"]))
    artifact = read_gzip_json(trace_path)
    assert isinstance(artifact, dict)
    assert artifact["subject_id"] == "person-1"
    assert artifact["bilabial_event_windows"][0]["event_id"] == "video-1-000"
    assert artifact["bilabial_event_windows"][0]["mouth_clip_frame_indices"] == [
        8,
        9,
        11,
    ]
    assert artifact["non_speech_reference"]["windows"]
    assert (
        artifact["non_speech_reference"]["strategy"]
        == "mfa_phone_complement"
    )
    assert "event_label" not in str(artifact)
    assert event["vild_trace_path"] == str(trace_path)
    assert event["vild_trace_event_key"] == "video-1-000"


def test_event_video_timing_applies_recorded_stream_offset(tmp_path: Path) -> None:
    analyzed_intervals: list[tuple[float, float]] = []

    def analyze(**kwargs):
        analyzed_intervals.append((kwargs["phone_start_s"], kwargs["phone_end_s"]))
        return {
            "attempted_frames": 4,
            "valid_landmark_ratio": 1.0,
            "multiple_face_ratio": 0.0,
            "normalized_minimum_closure": 0.1,
            "closure_time_s": 1.15,
            "closure_duration_s": 0.04,
            "closing_velocity": 0.2,
            "opening_velocity": 0.3,
            "mouth_crop_frame_indices": [27, 28, 29, 30],
            "mouth_crop_path": str(tmp_path / "clip.mp4"),
            "overlay_path": None,
            "minimum_overlay_path": None,
        }

    processor = PilotVideoProcessor(
        settings=SimpleNamespace(
            preprocessing=SimpleNamespace(
                output_dir=tmp_path,
                max_debug_overlays_per_video=0,
                event_window_before_s=0.2,
                event_window_after_s=0.2,
                min_valid_landmark_ratio=0.8,
                manipulation_boundary_tolerance_s=0.04,
            )
        ),
        transcriber=object(),
        aligner=object(),
        mouth_analyzer=SimpleNamespace(analyze=analyze),
    )

    events = processor._process_events(
        manifest_row={
            "file": "video.mp4",
            "split": "train",
            "modify_type": "audio_modified",
            "fake_segments_json": "[[1.0,1.3]]",
        },
        video_id="video-1",
        video_path=tmp_path / "video.mp4",
        fps=25.0,
        intervals=[{"phoneme": "p", "phone_start_s": 1.0, "phone_end_s": 1.1}],
        audio_video_offset_s=0.1,
    )

    assert analyzed_intervals[0] == pytest.approx((1.1, 1.2))
    assert events[0]["event_label"] == "fake"
    assert events[0]["mouth_crop_frame_indices_json"] == "[27,28,29,30]"
