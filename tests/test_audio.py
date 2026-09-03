from __future__ import annotations

import json
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

from seepat.config import AudioEnhancementSettings
from seepat.preprocessing import audio


def _settings(tmp_path: Path) -> AudioEnhancementSettings:
    return AudioEnhancementSettings(
        enabled=True,
        model_cache_dir=tmp_path / "models",
        demucs_model="htdemucs",
        demucs_device="cuda",
        demucs_segment_seconds=7,
        demucs_shifts=0,
        demucs_overlap=0.25,
        deepfilter_executable=None,
        deepfilter_model="DeepFilterNet3",
        deepfilter_compensate_delay=True,
        deepfilter_post_filter=False,
    )


def test_find_deepfilter_binary_accepts_explicit_path(
    tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / "deep-filter"
    executable.write_bytes(b"binary")
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)

    assert audio.find_deepfilter_binary(executable) == executable.resolve()


def test_deepfilter_identity_rejects_unpinned_version(
    tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / "deep-filter"
    executable.write_bytes(b"binary")
    monkeypatch.setattr(
        audio.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="deep_filter 0.5.4\n",
            stderr="",
        ),
    )

    with pytest.raises(audio.AudioEnhancementError, match="Expected deep-filter 0.5.6"):
        audio._deepfilter_identity(executable)


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
    assert commands[0][commands[0].index("--segment") + 1] == "7"


def test_denoise_vocals_uses_deepfilternet3_and_publishes_output(
    tmp_path: Path, monkeypatch
) -> None:
    vocals = tmp_path / "demucs_vocals.wav"
    vocals.write_bytes(b"vocals")
    output_path = tmp_path / "deepfilter_enhanced.wav"
    executable = tmp_path / "deep-filter.exe"
    executable.write_bytes(b"native binary")
    captured: list[str] = []
    padded: list[tuple[Path, Path, float]] = []
    restored: list[tuple[Path, Path, Path]] = []
    monkeypatch.setattr(audio, "find_deepfilter_binary", lambda explicit: executable)
    monkeypatch.setattr(
        audio,
        "_deepfilter_identity",
        lambda binary: ("deep-filter 0.5.6", "test-sha256"),
    )

    def fake_run(command: list[str], cache_dir: Path) -> None:
        captured.extend(command)
        destination = Path(command[command.index("--output-dir") + 1])
        source = Path(command[-1])
        (destination / source.name).write_bytes(b"enhanced")

    def fake_pad(input_path, padded_path, ffmpeg_path, padding_seconds):
        padded.append((input_path, padded_path, padding_seconds))
        padded_path.write_bytes(b"padded")

    def fake_restore(input_path, reference_path, restored_path, ffmpeg_path):
        restored.append((input_path, reference_path, restored_path))
        restored_path.write_bytes(b"restored")

    monkeypatch.setattr(audio, "_run_audio_command", fake_run)
    monkeypatch.setattr(audio, "_pad_audio_tail", fake_pad)
    monkeypatch.setattr(audio, "_restore_reference_duration", fake_restore)

    result = audio.denoise_vocals(vocals, output_path, _settings(tmp_path))

    assert result.path == output_path
    assert result.cache_hit is False
    assert result.fallback_applied is False
    assert output_path.read_bytes() == b"restored"
    assert captured[0] == str(executable)
    assert "df.enhance" not in captured
    assert "--compensate-delay" in captured
    assert "--pf" not in captured
    assert captured[captured.index("--atten-lim-db") + 1] == "100"
    assert padded[0][0] == vocals.resolve()
    assert padded[0][2] == audio.DEEPFILTER_FLUSH_PADDING_SECONDS
    assert restored[0][1] == vocals.resolve()
    metadata = json.loads(output_path.with_suffix(".meta.json").read_text())
    assert metadata["contract"]["deepfilternet_version"] == "deep-filter 0.5.6"
    assert metadata["contract"]["executable_sha256"] == "test-sha256"
    assert metadata["contract"]["compensate_delay"] is True
    assert metadata["contract"]["attenuation_limit_db"] == 100.0
    assert metadata["contract"]["preserve_input_duration"] is True


def _write_pcm16(path: Path, samples: list[int], sample_rate: int = 100) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(array("h", samples).tobytes())


def test_sustained_attenuation_guard_ignores_brief_loss_and_flags_long_loss(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    brief = tmp_path / "brief.wav"
    sustained = tmp_path / "sustained.wav"
    _write_pcm16(source, [1000] * 100)
    _write_pcm16(brief, [316] * 10 + [1000] * 90)
    _write_pcm16(sustained, [316] * 75 + [1000] * 25)

    brief_result = audio._analyze_sustained_attenuation(source, brief, 10.0)
    sustained_result = audio._analyze_sustained_attenuation(
        source, sustained, 10.0
    )

    assert brief_result["fallback_applied"] is False
    assert sustained_result["fallback_applied"] is True
    assert sustained_result["max_sustained_attenuation_seconds"] == 0.75


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
    monkeypatch.setattr(audio, "_require_audio_dependencies", lambda settings: None)

    def fake_separate(input_path, output_path, settings, force=False):
        calls.append("demucs")
        output_path.write_bytes(b"vocals")
        return output_path, False

    def fake_denoise(input_path, output_path, settings, ffmpeg_path=None, force=False):
        assert input_path.name == "demucs_vocals.wav"
        assert ffmpeg_path is None
        calls.append("deepfilter")
        output_path.write_bytes(b"enhanced")
        return audio.DeepFilterResult(
            path=output_path,
            cache_hit=False,
            fallback_applied=False,
            max_sustained_attenuation_s=None,
        )

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


def test_prepare_alignment_audio_uses_demucs_for_quality_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    raw_audio = tmp_path / "audio.wav"
    raw_audio.write_bytes(b"raw")
    normalized_inputs: list[Path] = []
    monkeypatch.setattr(audio, "_require_audio_dependencies", lambda settings: None)

    def fake_separate(input_path, output_path, settings, force=False):
        output_path.write_bytes(b"vocals")
        return output_path, False

    def fake_denoise(input_path, output_path, settings, ffmpeg_path=None, force=False):
        output_path.write_bytes(b"enhanced")
        return audio.DeepFilterResult(
            path=output_path,
            cache_hit=False,
            fallback_applied=True,
            max_sustained_attenuation_s=1.25,
        )

    def fake_normalize(input_path, output_path, ffmpeg_path, force=False):
        normalized_inputs.append(input_path)
        output_path.write_bytes(b"normalized")
        return output_path, False

    monkeypatch.setattr(audio, "separate_vocals", fake_separate)
    monkeypatch.setattr(audio, "denoise_vocals", fake_denoise)
    monkeypatch.setattr(audio, "normalize_alignment_audio", fake_normalize)

    prepared = audio.prepare_alignment_audio(
        raw_audio, tmp_path, _settings(tmp_path), ffmpeg_path=None
    )

    assert normalized_inputs == [tmp_path / "demucs_vocals.wav"]
    assert prepared.deepfilter_fallback_applied is True
    assert prepared.report_fields()["alignment_audio_source"] == "demucs_fallback"
