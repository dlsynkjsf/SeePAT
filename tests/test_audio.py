from __future__ import annotations

from pathlib import Path

from seepat.config import AudioEnhancementSettings
from seepat.preprocessing import audio


def _settings(tmp_path: Path) -> AudioEnhancementSettings:
    return AudioEnhancementSettings(
        enabled=True,
        model_cache_dir=tmp_path / "models",
        demucs_model="htdemucs",
        demucs_device="cuda",
        demucs_segment_seconds=7.0,
        demucs_shifts=0,
        demucs_overlap=0.25,
        deepfilter_model="DeepFilterNet3",
        deepfilter_post_filter=False,
    )


def test_separate_vocals_builds_command_and_reuses_matching_cache(
    tmp_path: Path, monkeypatch
) -> None:
    raw_audio = tmp_path / "audio.wav"
    raw_audio.write_bytes(b"raw audio")
    output_path = tmp_path / "demucs_vocals.wav"
    commands: list[list[str]] = []
    monkeypatch.setattr(audio, "_installed_version", lambda package: "test-version")

    def fake_run(command: list[str], cache_dir: Path) -> None:
        commands.append(command)
        destination = Path(command[command.index("--out") + 1])
        (destination / "htdemucs").mkdir(parents=True)
        (destination / "htdemucs" / "audio_vocals.wav").write_bytes(b"vocals")

    monkeypatch.setattr(audio, "_run_audio_command", fake_run)

    first_path, first_hit = audio.separate_vocals(
        raw_audio, output_path, _settings(tmp_path)
    )
    second_path, second_hit = audio.separate_vocals(
        raw_audio, output_path, _settings(tmp_path)
    )

    assert first_path == second_path == output_path
    assert first_hit is False
    assert second_hit is True
    assert len(commands) == 1
    assert "--two-stems" in commands[0]
    assert "vocals" in commands[0]
    assert commands[0][commands[0].index("--segment") + 1] == "7.0"


def test_denoise_vocals_uses_deepfilternet3_and_publishes_output(
    tmp_path: Path, monkeypatch
) -> None:
    vocals = tmp_path / "demucs_vocals.wav"
    vocals.write_bytes(b"vocals")
    output_path = tmp_path / "deepfilter_enhanced.wav"
    captured: list[str] = []
    monkeypatch.setattr(audio, "_installed_version", lambda package: "test-version")

    def fake_run(command: list[str], cache_dir: Path) -> None:
        captured.extend(command)
        destination = Path(command[command.index("--output-dir") + 1])
        (destination / vocals.name).write_bytes(b"enhanced")

    monkeypatch.setattr(audio, "_run_audio_command", fake_run)

    result, cache_hit = audio.denoise_vocals(
        vocals, output_path, _settings(tmp_path)
    )

    assert result == output_path
    assert cache_hit is False
    assert output_path.read_bytes() == b"enhanced"
    assert captured[captured.index("--model-base-dir") + 1] == "DeepFilterNet3"
    assert "--no-suffix" in captured


def test_prepare_alignment_audio_keeps_raw_audio_when_disabled(tmp_path: Path) -> None:
    raw_audio = tmp_path / "audio.wav"
    raw_audio.write_bytes(b"raw")
    settings = _settings(tmp_path)
    settings = AudioEnhancementSettings(**{**settings.__dict__, "enabled": False})

    prepared = audio.prepare_alignment_audio(
        raw_audio, tmp_path, settings, ffmpeg_path=None
    )

    assert prepared.alignment_audio == raw_audio
    assert prepared.enhanced_audio is None
    assert prepared.report_fields()["audio_enhancement_enabled"] is False


def test_prepare_alignment_audio_runs_stages_in_order(tmp_path: Path, monkeypatch) -> None:
    raw_audio = tmp_path / "audio.wav"
    raw_audio.write_bytes(b"raw")
    calls: list[str] = []
    monkeypatch.setattr(audio, "_require_audio_dependencies", lambda: None)

    def fake_separate(input_path, output_path, settings, force=False):
        calls.append("demucs")
        output_path.write_bytes(b"vocals")
        return output_path, False

    def fake_denoise(input_path, output_path, settings, force=False):
        assert input_path.name == "demucs_vocals.wav"
        calls.append("deepfilter")
        output_path.write_bytes(b"enhanced")
        return output_path, False

    def fake_normalize(input_path, output_path, ffmpeg_path, force=False):
        assert input_path.name == "deepfilter_enhanced.wav"
        calls.append("normalize")
        output_path.write_bytes(b"normalized")
        return output_path, False

    monkeypatch.setattr(audio, "separate_vocals", fake_separate)
    monkeypatch.setattr(audio, "denoise_vocals", fake_denoise)
    monkeypatch.setattr(audio, "normalize_alignment_audio", fake_normalize)

    prepared = audio.prepare_alignment_audio(
        raw_audio, tmp_path, _settings(tmp_path), ffmpeg_path=None
    )

    assert calls == ["demucs", "deepfilter", "normalize"]
    assert prepared.alignment_audio.name == "alignment_audio.wav"
