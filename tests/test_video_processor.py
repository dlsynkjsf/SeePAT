from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
