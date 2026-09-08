from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

from seepat.artifacts import file_sha256, read_gzip_json
from seepat.config import VildTraceSettings
from seepat.preprocessing.alignment import SILENCE_PHONES

VILD_TRACE_VERSION = "vild-trace-v2"


def load_vild_trace(path: Path, expected_sha256: str | None = None) -> dict[str, object]:
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError(f"VILD trace hash does not match its manifest: {path}")
    artifact = read_gzip_json(path)
    if not isinstance(artifact, dict):
        raise TypeError(f"VILD trace must contain a JSON object: {path}")
    if artifact.get("artifact_version") != VILD_TRACE_VERSION:
        raise ValueError(f"Unsupported VILD trace version in {path}")
    frames = artifact.get("frames")
    event_windows = artifact.get("bilabial_event_windows")
    if not isinstance(frames, list) or not isinstance(event_windows, list):
        raise TypeError(f"VILD trace is missing frame or event-window data: {path}")
    timestamps = [float(frame["timestamp_s"]) for frame in frames]
    if timestamps != sorted(timestamps):
        raise ValueError(f"VILD trace timestamps are not ordered: {path}")
    event_ids = [str(event["event_id"]) for event in event_windows]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError(f"VILD trace contains duplicate event IDs: {path}")
    return artifact


def event_vild_frames(artifact: dict[str, object], event_id: str) -> list[dict[str, object]]:
    event_windows = artifact.get("bilabial_event_windows")
    frames = artifact.get("frames")
    if not isinstance(event_windows, list) or not isinstance(frames, list):
        raise TypeError("VILD trace has no frame or event-window data")
    matching = [event for event in event_windows if event.get("event_id") == event_id]
    if len(matching) != 1:
        raise KeyError(f"VILD trace has no unique event window for {event_id!r}")
    start = float(matching[0]["window_start_s"])
    end = float(matching[0]["window_end_s"])
    return [frame for frame in frames if start <= float(frame["timestamp_s"]) <= end]


def _clamped_merged_intervals(
    intervals: Sequence[tuple[float, float]],
    duration_s: float,
    margin_s: float,
) -> list[tuple[float, float]]:
    expanded = sorted(
        (
            max(0.0, float(start) - margin_s),
            min(duration_s, float(end) + margin_s),
        )
        for start, end in intervals
        if float(end) > 0 and float(start) < duration_s and float(end) >= float(start)
    )
    merged: list[tuple[float, float]] = []
    for start, end in expanded:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def non_speech_reference_windows(
    duration_s: float,
    speech_intervals: Sequence[tuple[float, float]],
    window_seconds: float,
    max_windows: int,
    speech_margin_seconds: float,
) -> list[dict[str, float | str]]:
    """Select fixed, deterministic windows from the complement of speech segments."""
    if duration_s <= 0:
        return []
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if max_windows < 1:
        raise ValueError("max_windows must be at least 1")
    if speech_margin_seconds < 0:
        raise ValueError("speech_margin_seconds cannot be negative")
    if not speech_intervals:
        return []

    speech = _clamped_merged_intervals(
        speech_intervals,
        duration_s=duration_s,
        margin_s=speech_margin_seconds,
    )
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in speech:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration_s:
        gaps.append((cursor, duration_s))

    candidates: list[tuple[float, float]] = []
    for start, end in gaps:
        count = math.floor(((end - start) + 1e-9) / window_seconds)
        if count < 1:
            continue
        unused = (end - start) - count * window_seconds
        first_start = start + unused / 2
        candidates.extend(
            (
                first_start + index * window_seconds,
                first_start + (index + 1) * window_seconds,
            )
            for index in range(count)
        )

    if len(candidates) > max_windows:
        if max_windows == 1:
            selected = [candidates[len(candidates) // 2]]
        else:
            indices = [
                round(index * (len(candidates) - 1) / (max_windows - 1))
                for index in range(max_windows)
            ]
            selected = [candidates[index] for index in indices]
    else:
        selected = candidates

    return [
        {
            "reference_id": f"reference-{index:03d}",
            "start_s": round(start, 9),
            "end_s": round(end, 9),
        }
        for index, (start, end) in enumerate(selected)
    ]


def _percentile(values: Sequence[float], proportion: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = proportion * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_reference_windows(
    frames: Sequence[dict[str, object]],
    windows: Sequence[dict[str, float | str]],
) -> list[dict[str, object]]:
    """Attach label-independent VILD and face-quality summaries to windows."""
    summarized: list[dict[str, object]] = []
    for window in windows:
        start = float(window["start_s"])
        end = float(window["end_s"])
        selected = [frame for frame in frames if start <= float(frame["timestamp_s"]) < end]
        values = [
            float(frame["normalized_vild"])
            for frame in selected
            if frame.get("normalized_vild") is not None
        ]
        attempted = len(selected)
        multiple = sum(int(frame.get("face_count", 0)) > 1 for frame in selected)
        summary: dict[str, object] = {
            **window,
            "attempted_frames": attempted,
            "valid_frames": len(values),
            "valid_landmark_ratio": len(values) / attempted if attempted else 0.0,
            "multiple_face_ratio": multiple / attempted if attempted else 0.0,
            "normalized_vild_median": None,
            "normalized_vild_iqr": None,
            "normalized_vild_minimum": None,
            "normalized_vild_maximum": None,
            "mean_absolute_vild_delta": None,
        }
        if values:
            deltas = [abs(second - first) for first, second in pairwise(values)]
            summary.update(
                {
                    "normalized_vild_median": statistics.median(values),
                    "normalized_vild_iqr": _percentile(values, 0.75) - _percentile(values, 0.25),
                    "normalized_vild_minimum": min(values),
                    "normalized_vild_maximum": max(values),
                    "mean_absolute_vild_delta": (statistics.fmean(deltas) if deltas else 0.0),
                }
            )
        summarized.append(summary)
    return summarized


def build_vild_trace_artifact(
    *,
    video_id: str,
    source_file: str,
    dataset_split: str,
    source_group: str,
    subject_id: str,
    probe: dict[str, object],
    transcription: dict[str, object],
    phone_intervals: list[dict[str, object]],
    events: list[dict[str, object]],
    trace_result: dict[str, object],
    audio_video_offset_s: float,
    event_window_before_s: float,
    event_window_after_s: float,
    settings: VildTraceSettings,
    provenance: dict[str, object],
) -> dict[str, object]:
    """Build the label-independent, reusable per-video VILD artifact."""
    frames = trace_result.get("frames")
    if not isinstance(frames, list) or not frames:
        raise TypeError("VILD trace frames must be a non-empty list")
    duration_s = probe.get("duration_s")
    if duration_s is None:
        duration_s = float(frames[-1]["timestamp_s"]) + 1 / float(probe["fps"])

    whisper_speech_intervals = [
        (
            float(segment["start"]) + audio_video_offset_s,
            float(segment["end"]) + audio_video_offset_s,
        )
        for segment in (transcription.get("segments") or [])
        if isinstance(segment, dict)
        and segment.get("start") is not None
        and segment.get("end") is not None
    ]
    timed_phone_intervals = [
        {
            "phoneme": interval["phoneme"],
            "start_s": float(interval["phone_start_s"]) + audio_video_offset_s,
            "end_s": float(interval["phone_end_s"]) + audio_video_offset_s,
        }
        for interval in phone_intervals
    ]
    mfa_speech_intervals = [
        (float(interval["start_s"]), float(interval["end_s"]))
        for interval in timed_phone_intervals
        if str(interval["phoneme"]).upper() not in SILENCE_PHONES
    ]
    reference_windows = non_speech_reference_windows(
        duration_s=float(duration_s),
        speech_intervals=mfa_speech_intervals,
        window_seconds=settings.reference_window_seconds,
        max_windows=settings.max_reference_windows,
        speech_margin_seconds=settings.speech_margin_seconds,
    )
    reference_strategy = "mfa_phone_complement"
    if not reference_windows:
        reference_windows = non_speech_reference_windows(
            duration_s=float(duration_s),
            speech_intervals=whisper_speech_intervals,
            window_seconds=settings.reference_window_seconds,
            max_windows=settings.max_reference_windows,
            speech_margin_seconds=settings.speech_margin_seconds,
        )
        reference_strategy = "whisper_segment_complement_fallback"
        if not reference_windows:
            reference_strategy = "unavailable"
    summarized_references = summarize_reference_windows(frames, reference_windows)

    event_windows = []
    for event in events:
        event_windows.append(
            {
                "event_id": event["event_id"],
                "phoneme": event["phoneme"],
                "audio_phone_start_s": event["phone_start_s"],
                "audio_phone_end_s": event["phone_end_s"],
                "video_phone_start_s": event["video_phone_start_s"],
                "video_phone_end_s": event["video_phone_end_s"],
                "window_start_s": max(
                    0.0,
                    float(event["video_phone_start_s"]) - event_window_before_s,
                ),
                "window_end_s": float(event["video_phone_end_s"]) + event_window_after_s,
                "mouth_clip_frame_indices": json.loads(
                    str(event.get("mouth_crop_frame_indices_json", "[]"))
                ),
            }
        )

    return {
        "artifact_version": VILD_TRACE_VERSION,
        "video_id": video_id,
        "source_file": source_file,
        "dataset_split": dataset_split,
        "source_group": source_group,
        "subject_id": subject_id,
        "timing": {
            "fps": probe["fps"],
            "average_frame_rate": probe.get("average_frame_rate"),
            "nominal_frame_rate": probe.get("nominal_frame_rate"),
            "video_time_base": probe.get("video_time_base"),
            "video_start_time_s": probe.get("video_start_time_s"),
            "audio_start_time_s": probe.get("audio_start_time_s"),
            "audio_video_start_offset_s": audio_video_offset_s,
            "duration_s": duration_s,
        },
        "frames": frames,
        "whisper_segments_video_time": [
            {"start_s": start, "end_s": end} for start, end in whisper_speech_intervals
        ],
        "mfa_phone_intervals_video_time": timed_phone_intervals,
        "bilabial_event_windows": event_windows,
        "non_speech_reference": {
            "strategy": reference_strategy,
            "window_seconds": settings.reference_window_seconds,
            "maximum_windows": settings.max_reference_windows,
            "speech_margin_seconds": settings.speech_margin_seconds,
            "windows": summarized_references,
        },
        "provenance": provenance,
        "summary": trace_result["summary"],
    }
