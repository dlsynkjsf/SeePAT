from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from seepat.preprocessing.alignment import MfaAlignmentError
from seepat.video_processor import PilotVideoProcessor


def test_mfa_failure_is_reported_as_alignment_failed(
    tmp_path: Path, monkeypatch
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    preprocessing = SimpleNamespace(ffprobe_path=None, ffmpeg_path=None)
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
