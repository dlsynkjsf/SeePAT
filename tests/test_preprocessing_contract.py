from __future__ import annotations

import json
from pathlib import Path

import pytest

from seepat.artifacts import (
    atomic_write_csv,
    atomic_write_gzip_json,
    atomic_write_json,
    file_sha256,
    read_csv_rows,
)
from seepat.preprocessing.contract import audit_preprocessing_contract
from seepat.preprocessing.vild import VILD_TRACE_VERSION


def test_contract_audit_verifies_trace_events_frames_and_references(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    trace_path = output_dir / "vild_traces" / "video-1.json.gz"
    artifact_fields: dict[str, str] = {}
    for name in ("raw_audio", "alignment_audio", "transcription", "alignment"):
        path = output_dir / f"{name}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        artifact_fields[f"{name}_path"] = str(path)
        artifact_fields[f"{name}_sha256"] = file_sha256(path)
    atomic_write_gzip_json(
        trace_path,
        {
            "artifact_version": VILD_TRACE_VERSION,
            "video_id": "video-1",
            "frames": [
                {"timestamp_s": 0.0, "normalized_vild": 0.2},
                {"timestamp_s": 0.04, "normalized_vild": 0.1},
            ],
            "bilabial_event_windows": [
                {
                    "event_id": "video-1-000",
                    "window_start_s": 0.0,
                    "window_end_s": 0.1,
                    "mouth_clip_frame_indices": [0, 1],
                }
            ],
            "non_speech_reference": {
                "strategy": "mfa_phone_complement",
                "windows": [{"start_s": 0.0, "end_s": 0.08}],
            },
            "provenance": artifact_fields,
        },
    )
    trace_hash = file_sha256(trace_path)
    atomic_write_csv(
        output_dir / "video_manifest.csv",
        [
            {
                "video_id": "video-1",
                "pipeline_status": "complete",
                "bilabial_event_count": 1,
                "video_frame_count": 2,
                **artifact_fields,
            }
        ],
    )
    atomic_write_csv(
        output_dir / "bilabial_events.csv",
        [
            {
                "event_id": "video-1-000",
                "video_id": "video-1",
                "eligibility_status": "eligible",
                "vild_trace_path": str(trace_path),
                "vild_trace_sha256": trace_hash,
                "vild_trace_event_key": "video-1-000",
                "mouth_crop_frame_indices_json": json.dumps([0, 1]),
            }
        ],
    )
    trace_index = output_dir / "vild_trace_index.csv"
    atomic_write_csv(
        trace_index,
        [
            {
                "video_id": "video-1",
                "vild_trace_path": str(trace_path),
                "vild_trace_sha256": trace_hash,
                "vild_trace_frame_count": 2,
                "vild_reference_window_count": 1,
            }
        ],
    )
    atomic_write_json(
        output_dir / "run_summary.json",
        {
            "vild_trace_enabled": True,
            "vild_trace_index_sha256": file_sha256(trace_index),
            "vild_trace_videos": 1,
            "vild_reference_windows": 1,
            "bilabial_events": 1,
        },
    )

    report = audit_preprocessing_contract(output_dir, project_root=tmp_path)

    assert report["status"] == "passed"
    assert report["trace_videos"] == 1
    assert report["eligible_events"] == 1
    assert report["reference_windows"] == 1

    events = read_csv_rows(output_dir / "bilabial_events.csv")
    events[0]["mouth_crop_frame_indices_json"] = "[1,0]"
    atomic_write_csv(output_dir / "bilabial_events.csv", events)
    with pytest.raises(ValueError, match="frame map"):
        audit_preprocessing_contract(output_dir, project_root=tmp_path)
