from __future__ import annotations

import json
from pathlib import Path

import pytest

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256
from seepat.evidence import FUSION_EVIDENCE_FIELDS
from seepat.xai import (
    FORENSIC_TRACE_VERSION,
    OpenAICompatibleClient,
    build_evidence_bundle,
    generate_forensic_trace,
    render_evidence_summary,
)


def _write_verdict_dir(tmp_path: Path) -> Path:
    verdict_dir = tmp_path / "verdict"
    verdict_dir.mkdir()
    (verdict_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "verdict_version": "verdict-v2",
                "status": "complete",
                "model_name": "swin3d_b_vild_fusion",
                "training_version": "hybrid-fusion-v1",
                "split": "test",
                "threshold": {"value": 0.5, "source": "artifact"},
                "aggregation": "maximum event probability",
            }
        ),
        encoding="utf-8",
    )
    atomic_write_csv(
        verdict_dir / "video_verdicts.csv",
        [
            {
                "video_id": "video-quiet",
                "event_count": 1,
                "maximum_probability": 0.05,
                "threshold": 0.5,
                "verdict": "authentic",
                "label": 0,
            },
            {
                "video_id": "video-flagged",
                "event_count": 2,
                "maximum_probability": 0.92,
                "threshold": 0.5,
                "verdict": "manipulated",
                "label": 1,
            },
        ],
    )
    def event_row(event_id: str, video_id: str, probability: float, *, complete: bool):
        row = {
            "event_id": event_id,
            "video_id": video_id,
            "subject_id": "subject-a",
            "phoneme": "b",
            "label": 0,
            "video_label": 0,
            "manipulated_probability": probability,
            "sync_gap_score": 0.7,
            "evidence_available": 7 if complete else 0,
            "evidence_total": 7,
        }
        for index, field in enumerate(FUSION_EVIDENCE_FIELDS):
            row[field] = str(index) if complete else ""
            row[f"{field}_available"] = complete
        return row

    atomic_write_csv(
        verdict_dir / "event_predictions.csv",
        [
            event_row("flagged-b", "video-flagged", 0.92, complete=True),
            event_row("flagged-a", "video-flagged", 0.4, complete=True),
            event_row("quiet-a", "video-quiet", 0.05, complete=False),
        ],
    )
    evaluation_path = verdict_dir / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["output_sha256"] = {
        "events": file_sha256(verdict_dir / "event_predictions.csv"),
        "videos": file_sha256(verdict_dir / "video_verdicts.csv"),
    }
    atomic_write_json(evaluation_path, evaluation)
    return verdict_dir


def test_build_evidence_bundle_reports_sources_and_top_events(tmp_path: Path) -> None:
    verdict_dir = _write_verdict_dir(tmp_path)

    bundle = build_evidence_bundle(verdict_dir)

    assert bundle["trace_version"] == FORENSIC_TRACE_VERSION
    assert bundle["videos_scored"] == 2
    assert bundle["events_scored"] == 3
    assert bundle["videos"][0]["video_id"] == "video-flagged"
    assert bundle["videos"][0]["events"][0]["event_id"] == "flagged-b"
    assert bundle["videos"][0]["events"][0]["evidence"]["phoneme-viseme residual (z)"]
    assert bundle["events_with_missing_evidence_values"] == 1
    assert bundle["missing_evidence_values"] == 7
    for source in bundle["sources"].values():
        assert source["sha256"]


def test_render_evidence_summary_is_deterministic_and_explanatory(tmp_path: Path) -> None:
    bundle = build_evidence_bundle(_write_verdict_dir(tmp_path))

    summary = render_evidence_summary(bundle)

    assert "video-flagged" in summary
    assert "manipulated" in summary
    assert "explanatory only" in summary
    assert summary == render_evidence_summary(bundle)


def test_generate_forensic_trace_writes_frozen_artifact(tmp_path: Path) -> None:
    verdict_dir = _write_verdict_dir(tmp_path)
    output_path = tmp_path / "trace.json"

    artifact = generate_forensic_trace(verdict_dir, output_path)

    assert output_path.is_file()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["trace_version"] == FORENSIC_TRACE_VERSION
    assert written["mode"] == "deterministic"
    assert written["client"] is None
    assert written["trace"] == written["evidence_summary"]
    assert "cannot modify probabilities" in written["disclaimer"]
    assert artifact["verdict_snapshot"]["videos_scored"] == 2


def test_generate_forensic_trace_filters_video(tmp_path: Path) -> None:
    verdict_dir = _write_verdict_dir(tmp_path)

    artifact = generate_forensic_trace(
        verdict_dir,
        tmp_path / "trace-quiet.json",
        video_id="video-quiet",
    )

    assert artifact["evidence_summary"].count("## Video") == 1
    assert "video-flagged" not in artifact["evidence_summary"]
    with pytest.raises(ValueError, match="No verdict row"):
        generate_forensic_trace(
            verdict_dir,
            tmp_path / "trace-missing.json",
            video_id="video-unknown",
        )


def test_generate_forensic_trace_requires_verdict_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        generate_forensic_trace(tmp_path / "missing", tmp_path / "trace.json")


def test_openai_client_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAICompatibleClient(model="gpt-4o-mini")

    with pytest.raises(RuntimeError, match="Missing API key"):
        client.generate("system", "user")
