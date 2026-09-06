from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from seepat.preprocessing.alignment import (
    MfaAlignmentError,
    MfaDockerAligner,
    ensure_docker_daemon,
    normalize_arpa_phone,
    parse_mfa_json,
    parse_mfa_phone_intervals,
)


def test_normalize_arpa_phone_removes_stress() -> None:
    assert normalize_arpa_phone("M") == "M"
    assert normalize_arpa_phone("b1") == "B"


def test_parse_mfa_json_keeps_only_bilabials(tmp_path: Path) -> None:
    output = tmp_path / "alignment.json"
    output.write_text(
        '{"tiers":{"words":{"entries":[[0.0,0.3,"word"]]},'
        '"phones":{"entries":[[0.0,0.1,"AH0"],[0.1,0.2,"B"]]}}}',
        encoding="utf-8",
    )

    assert parse_mfa_json(output) == [
        {
            "phoneme": "b",
            "phone_start_s": 0.1,
            "phone_end_s": 0.2,
            "speaker": "",
        }
    ]
    assert [row["phoneme"] for row in parse_mfa_phone_intervals(output)] == [
        "ah",
        "b",
    ]


def test_align_wraps_per_video_mfa_failure(tmp_path: Path, monkeypatch) -> None:
    aligner = object.__new__(MfaDockerAligner)
    aligner.dictionary = "english_us_arpa"
    aligner.acoustic_model = "english_us_arpa"
    monkeypatch.setattr(aligner, "ensure_models", lambda: None)

    def fail_alignment(*args, **kwargs) -> None:
        raise RuntimeError("Could not align with the current beam size")

    monkeypatch.setattr(aligner, "_run", fail_alignment)
    audio_path = tmp_path / "source.wav"
    audio_path.write_bytes(b"audio")

    with pytest.raises(MfaAlignmentError, match="current beam size"):
        aligner.align(audio_path, "test transcript", tmp_path / "alignment.json")


def test_align_uses_enhanced_audio_already_inside_work_directory(
    tmp_path: Path, monkeypatch
) -> None:
    aligner = object.__new__(MfaDockerAligner)
    aligner.dictionary = "english_us_arpa"
    aligner.acoustic_model = "english_us_arpa"
    monkeypatch.setattr(aligner, "ensure_models", lambda: None)
    enhanced_audio = tmp_path / "deepfilter_enhanced.wav"
    enhanced_audio.write_bytes(b"enhanced")
    raw_audio = tmp_path / "audio.wav"
    raw_audio.write_bytes(b"raw")
    output_path = tmp_path / "alignment.json"
    captured: list[str] = []

    def create_alignment(arguments: list[str], data_dir: Path | None = None) -> None:
        captured.extend(arguments)
        output_path.write_text(
            '{"tiers":{"phones":{"entries":[[0.0,0.1,"M"]]}}}',
            encoding="utf-8",
        )

    monkeypatch.setattr(aligner, "_run", create_alignment)

    aligner.align(enhanced_audio, "mom", output_path)

    assert "/data/deepfilter_enhanced.wav" in captured
    assert raw_audio.read_bytes() == b"raw"


def test_docker_daemon_check_explains_stopped_desktop(monkeypatch) -> None:
    completed = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="failed to connect to the docker API",
    )
    monkeypatch.setattr("seepat.preprocessing.alignment.shutil.which", lambda name: name)
    monkeypatch.setattr(
        "seepat.preprocessing.alignment.subprocess.run",
        lambda *args, **kwargs: completed,
    )

    with pytest.raises(RuntimeError, match="Docker Desktop is not running"):
        ensure_docker_daemon()
