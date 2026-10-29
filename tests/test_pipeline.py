from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from seepat import pipeline
from seepat.artifacts import atomic_write_json, file_sha256, read_csv_rows, stable_id
from seepat.pipeline import _load_cached_result, selected_manifest_rows


def _write_result(path: Path, status: str) -> None:
    path.write_text(
        json.dumps(
            {
                "cache_signature": "signature",
                "video_report": {"pipeline_status": status},
                "events": [],
            }
        ),
        encoding="utf-8",
    )


def test_retry_failed_reuses_only_complete_results(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    _write_result(result_path, "failed")

    assert _load_cached_result(result_path, "signature") is not None
    assert _load_cached_result(result_path, "signature", retry_failed=True) is None

    _write_result(result_path, "complete")
    assert _load_cached_result(result_path, "signature", retry_failed=True) is not None


def test_cached_result_requires_its_recorded_vild_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json.gz"
    trace_path.write_bytes(b"trace")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "cache_signature": "signature",
                "video_report": {
                    "pipeline_status": "complete",
                    "bilabial_event_count": 1,
                    "vild_trace_path": str(trace_path),
                    "vild_trace_sha256": file_sha256(trace_path),
                },
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    assert _load_cached_result(
        result_path, "signature", require_vild_trace=True
    ) is not None
    trace_path.write_bytes(b"changed")
    assert _load_cached_result(result_path, "signature", require_vild_trace=True) is None


def test_selected_manifest_rows_filters_by_stable_video_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [{"file": "a.mp4"}, {"file": "b.mp4"}, {"file": "c.mp4"}]
    settings = SimpleNamespace(
        dataset=SimpleNamespace(
            pilot_manifest=Path("pilot.csv"),
            include_video_ids=(stable_id("c.mp4"), stable_id("a.mp4")),
        )
    )
    monkeypatch.setattr("seepat.pipeline.read_csv_rows", lambda path: rows)

    assert selected_manifest_rows(settings) == [rows[0], rows[2]]


def test_selected_manifest_rows_rejects_unknown_video_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        dataset=SimpleNamespace(
            pilot_manifest=Path("pilot.csv"),
            include_video_ids=("missing",),
        )
    )
    monkeypatch.setattr(
        "seepat.pipeline.read_csv_rows", lambda path: [{"file": "a.mp4"}]
    )

    with pytest.raises(ValueError, match="absent from the manifest: missing"):
        selected_manifest_rows(settings)


def _pipeline_fixture(tmp_path, monkeypatch, files):
    settings = SimpleNamespace(
        cache_signature="signature",
        preprocessing=SimpleNamespace(
            output_dir=tmp_path / "outputs",
            whisper_model="unused",
            whisper_device="cpu",
            whisper_compute_type="int8",
            mfa_docker_image="unused",
            mfa_cache_dir=tmp_path / "mfa",
            mfa_dictionary="unused",
            mfa_acoustic_model="unused",
            vild_trace=SimpleNamespace(enabled=False),
        ),
    )
    rows = [{"file": name} for name in files]
    monkeypatch.setattr(pipeline, "load_pipeline_settings", lambda *_: settings)
    monkeypatch.setattr(pipeline, "selected_manifest_rows", lambda *_args, **_kwargs: rows)
    return settings, rows


def _video_report(row, status="complete"):
    return {
        "video_id": stable_id(row["file"]),
        "file": row["file"],
        "pipeline_status": status,
        "eligibility_status": "eligible" if status == "complete" else "ineligible",
        "bilabial_event_count": 0,
        "eligible_event_count": 0,
    }


def _cache_video(settings, row, status="complete"):
    report = _video_report(row, status)
    path = settings.preprocessing.output_dir / "cache" / report["video_id"] / "result.json"
    atomic_write_json(path, {
        "cache_signature": settings.cache_signature,
        "video_report": report,
        "events": [],
    })
    return report


def _record_dependencies(monkeypatch, calls, *, process_error=False):
    def transcriber(**_kwargs):
        calls.append("transcriber")
        return "transcriber-instance"

    def aligner(**_kwargs):
        calls.append("aligner")
        return "aligner-instance"

    class Mouth:
        def __init__(self):
            calls.append("mouth-init")

        def __enter__(self):
            calls.append("mouth-enter")
            return self

        def __exit__(self, error_type, *_args):
            calls.append(("mouth-exit", error_type))

    class Processor:
        def __init__(self, *, settings, transcriber, aligner, mouth_analyzer):
            assert transcriber == "transcriber-instance"
            assert aligner == "aligner-instance"
            assert isinstance(mouth_analyzer, Mouth)
            calls.append("processor")

        def process(self, *, manifest_row, video_id, work_dir, force):
            assert video_id == stable_id(manifest_row["file"])
            assert work_dir.is_dir()
            calls.append(("process", manifest_row["file"], force))
            if process_error:
                raise RuntimeError("processing interrupted")
            return _video_report(manifest_row), []

    monkeypatch.setattr(pipeline, "WhisperTranscriber", transcriber)
    monkeypatch.setattr(pipeline, "MfaDockerAligner", aligner)
    monkeypatch.setattr(pipeline, "MouthEventAnalyzer", Mouth)
    monkeypatch.setattr(pipeline, "PilotVideoProcessor", Processor)


@pytest.mark.parametrize("files", [[], ["a.mp4", "b.mp4"]])
def test_fully_cached_or_empty_run_needs_no_model_or_docker_initialization(
    tmp_path, monkeypatch, files,
):
    settings, rows = _pipeline_fixture(tmp_path, monkeypatch, files)
    expected = [_cache_video(settings, row) for row in rows]

    def unexpected(*_args, **_kwargs):
        raise AssertionError("cached runs must not initialize processing dependencies")

    for name in ("WhisperTranscriber", "MfaDockerAligner", "MouthEventAnalyzer",
                 "PilotVideoProcessor"):
        monkeypatch.setattr(pipeline, name, unexpected)

    result = pipeline.run_pipeline(tmp_path / "unused.yaml")

    assert result["videos_requested"] == len(rows)
    assert result["videos_completed"] == len(rows)
    actual = read_csv_rows(settings.preprocessing.output_dir / "video_manifest.csv")
    assert [row["file"] for row in actual] == [row["file"] for row in expected]


def test_mixed_cache_run_initializes_once_and_preserves_manifest_order(tmp_path, monkeypatch):
    settings, rows = _pipeline_fixture(tmp_path, monkeypatch,
                                       ["a.mp4", "b.mp4", "c.mp4", "d.mp4"])
    _cache_video(settings, rows[0])
    _cache_video(settings, rows[2])
    calls, progress = [], []
    _record_dependencies(monkeypatch, calls)

    result = pipeline.run_pipeline(tmp_path / "unused.yaml",
                                   progress=lambda *args: progress.append(args))

    assert result["videos_completed"] == 4
    assert calls == [
        "transcriber", "aligner", "mouth-init", "mouth-enter", "processor",
        ("process", "b.mp4", False), ("process", "d.mp4", False), ("mouth-exit", None),
    ]
    assert [row["file"] for row in read_csv_rows(
        settings.preprocessing.output_dir / "video_manifest.csv"
    )] == [row["file"] for row in rows]
    assert progress == [
        ("preprocess videos", 0, 4, ""),
        ("preprocess videos", 0, 4, stable_id("a.mp4")),
        ("preprocess videos", 1, 4, stable_id("b.mp4")),
        ("preprocess videos", 2, 4, stable_id("b.mp4")),
        ("preprocess videos", 2, 4, stable_id("c.mp4")),
        ("preprocess videos", 3, 4, stable_id("d.mp4")),
        ("preprocess videos", 4, 4, stable_id("d.mp4")),
    ]


@pytest.mark.parametrize("force,retry_failed,expected", [
    (False, False, []),
    (False, True, [("process", "a.mp4", False)]),
    (True, False, [("process", "a.mp4", True), ("process", "b.mp4", True)]),
])
def test_lazy_processor_respects_force_and_retry_failed(
    tmp_path, monkeypatch, force, retry_failed, expected,
):
    settings, rows = _pipeline_fixture(tmp_path, monkeypatch, ["a.mp4", "b.mp4"])
    _cache_video(settings, rows[0], "failed")
    _cache_video(settings, rows[1])
    calls = []
    _record_dependencies(monkeypatch, calls)

    pipeline.run_pipeline(tmp_path / "unused.yaml", force=force, retry_failed=retry_failed)

    assert [call for call in calls if isinstance(call, tuple) and call[0] == "process"] == expected
    assert calls.count("processor") == bool(expected)


def test_lazy_processor_closes_mouth_analyzer_on_processing_error(tmp_path, monkeypatch):
    _pipeline_fixture(tmp_path, monkeypatch, ["uncached.mp4"])
    calls = []
    _record_dependencies(monkeypatch, calls, process_error=True)

    with pytest.raises(RuntimeError, match="processing interrupted"):
        pipeline.run_pipeline(tmp_path / "unused.yaml")

    assert calls[-1] == ("mouth-exit", RuntimeError)
