from __future__ import annotations

import json
import traceback
from pathlib import Path

from seepat.artifacts import atomic_write_gzip_json, atomic_write_json, file_sha256
from seepat.config import PipelineSettings
from seepat.preprocessing.alignment import (
    MfaAlignmentError,
    MfaDockerAligner,
    parse_mfa_phone_intervals,
)
from seepat.preprocessing.audio import prepare_alignment_audio
from seepat.preprocessing.eligibility import (
    event_evidence_status,
    video_evidence_status,
)
from seepat.preprocessing.events import classify_event_label, parse_segments
from seepat.preprocessing.face import MouthEventAnalyzer
from seepat.preprocessing.media import extract_mono_audio, probe_media
from seepat.preprocessing.transcription import WhisperTranscriber
from seepat.preprocessing.vild import build_vild_trace_artifact


class PilotVideoProcessor:
    """Run all preprocessing stages for one manifest video."""

    def __init__(
        self,
        settings: PipelineSettings,
        transcriber: WhisperTranscriber,
        aligner: MfaDockerAligner,
        mouth_analyzer: MouthEventAnalyzer,
    ) -> None:
        self.settings = settings
        self.preprocessing = settings.preprocessing
        self.transcriber = transcriber
        self.aligner = aligner
        self.mouth_analyzer = mouth_analyzer

    def process(
        self,
        manifest_row: dict[str, str],
        video_id: str,
        work_dir: Path,
        force: bool,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        relative_path = manifest_row["file"]
        video_path = self.settings.dataset.extracted_root / Path(relative_path)
        report = self._initial_report(manifest_row, video_id, video_path)
        events: list[dict[str, object]] = []

        try:
            if not video_path.is_file():
                raise FileNotFoundError(f"Pilot video is absent: {video_path}")
            probe = probe_media(video_path, self.preprocessing.ffprobe_path)
            report.update(probe)
            if not probe["audio_present"]:
                report["exclusion_reason"] = "audio_missing"
                raise ValueError("Video has no audio stream")
            if not probe["fps"]:
                raise ValueError("Video frame rate is unavailable")

            raw_audio_path = extract_mono_audio(
                video_path,
                work_dir / "audio.wav",
                self.preprocessing.ffmpeg_path,
                force=force,
            )
            prepared_audio = prepare_alignment_audio(
                raw_audio=raw_audio_path,
                work_dir=work_dir,
                settings=self.preprocessing.audio_enhancement,
                ffmpeg_path=self.preprocessing.ffmpeg_path,
                force=force,
            )
            report.update(prepared_audio.report_fields())
            report["raw_audio_sha256"] = file_sha256(prepared_audio.raw_audio)
            report["alignment_audio_sha256"] = file_sha256(
                prepared_audio.alignment_audio
            )
            downstream_force = (
                force or prepared_audio.normalization_cache_hit is False
            )
            transcription = self._transcribe(
                prepared_audio.alignment_audio, work_dir, downstream_force
            )
            report["transcription_path"] = str(work_dir / "transcription.json")
            report["transcription_sha256"] = file_sha256(
                work_dir / "transcription.json"
            )
            report["transcript"] = transcription["text"]
            if not str(transcription["text"]).strip():
                report["exclusion_reason"] = "alignment_failed"
                raise ValueError("Whisper produced an empty transcript")

            alignment_path = work_dir / "alignment.json"
            intervals = self.aligner.align(
                prepared_audio.alignment_audio,
                str(transcription["text"]),
                alignment_path,
                force=downstream_force,
            )
            report["alignment_path"] = str(alignment_path)
            report["alignment_sha256"] = file_sha256(alignment_path)
            phone_intervals = parse_mfa_phone_intervals(alignment_path)
            report["bilabial_event_count"] = len(intervals)
            audio_video_offset_s = float(
                probe.get("audio_video_start_offset_s") or 0.0
            )
            trace_result = None
            if intervals and self.preprocessing.vild_trace.enabled:
                trace_result = self.mouth_analyzer.trace_video(
                    video_path=video_path,
                    fps=float(probe["fps"]),
                )
                if not trace_result["frames"]:
                    raise ValueError("VILD trace contains no decoded video frames")
            events = self._process_events(
                manifest_row=manifest_row,
                video_id=video_id,
                video_path=video_path,
                fps=float(probe["fps"]),
                intervals=intervals,
                audio_video_offset_s=audio_video_offset_s,
            )
            if trace_result is not None:
                self._write_vild_trace(
                    manifest_row=manifest_row,
                    video_id=video_id,
                    probe=probe,
                    transcription=transcription,
                    phone_intervals=phone_intervals,
                    events=events,
                    trace_result=trace_result,
                    audio_video_offset_s=audio_video_offset_s,
                    report=report,
                )
            status, reason, eligible_count = video_evidence_status(
                events, self.preprocessing.min_bilabial_events
            )
            report["eligibility_status"] = status
            report["exclusion_reason"] = reason
            report["eligible_event_count"] = eligible_count
            report["pipeline_status"] = "complete"
        # Dataset-scale preprocessing must record a per-video failure and
        # continue. The exception type and traceback remain in the audit.
        except Exception as error:  # noqa: BLE001
            if isinstance(error, MfaAlignmentError):
                report["exclusion_reason"] = "alignment_failed"
            report["error_type"] = type(error).__name__
            report["error_message"] = str(error)
            report["traceback_tail"] = traceback.format_exc()[-4000:]

        return report, events

    def _transcribe(
        self, audio_path: Path, work_dir: Path, force: bool
    ) -> dict[str, object]:
        transcription_path = work_dir / "transcription.json"
        if transcription_path.is_file() and not force:
            return json.loads(transcription_path.read_text(encoding="utf-8"))
        transcription = self.transcriber.transcribe(audio_path)
        atomic_write_json(transcription_path, transcription)
        return transcription

    def _process_events(
        self,
        manifest_row: dict[str, str],
        video_id: str,
        video_path: Path,
        fps: float,
        intervals: list[dict[str, object]],
        audio_video_offset_s: float,
    ) -> list[dict[str, object]]:
        manipulated_segments = parse_segments(manifest_row.get("fake_segments_json"))
        events: list[dict[str, object]] = []
        for event_index, interval in enumerate(intervals):
            event_id = f"{video_id}-{event_index:03d}"
            video_phone_start_s = float(interval["phone_start_s"]) + audio_video_offset_s
            video_phone_end_s = float(interval["phone_end_s"]) + audio_video_offset_s
            mouth_path = (
                self.preprocessing.output_dir
                / "mouth_event_clips"
                / f"{event_id}.mp4"
            )
            overlay_path = (
                self.preprocessing.output_dir
                / "debug_overlays"
                / f"{event_id}.mp4"
                if event_index < self.preprocessing.max_debug_overlays_per_video
                else None
            )
            face_result = self.mouth_analyzer.analyze(
                video_path=video_path,
                phone_start_s=video_phone_start_s,
                phone_end_s=video_phone_end_s,
                phoneme=str(interval["phoneme"]),
                fps=fps,
                window_before_s=self.preprocessing.event_window_before_s,
                window_after_s=self.preprocessing.event_window_after_s,
                mouth_clip_path=mouth_path,
                overlay_path=overlay_path,
            )
            mouth_crop_frame_indices = face_result.pop(
                "mouth_crop_frame_indices", []
            )
            status, reason = event_evidence_status(
                face_result, self.preprocessing.min_valid_landmark_ratio
            )
            event_label = classify_event_label(
                video_phone_start_s,
                video_phone_end_s,
                manipulated_segments,
                self.preprocessing.manipulation_boundary_tolerance_s,
            )
            events.append(
                {
                    "event_id": event_id,
                    "video_id": video_id,
                    "file": manifest_row["file"],
                    "split": manifest_row.get("split", ""),
                    "phoneme": interval["phoneme"],
                    "phone_start_s": interval["phone_start_s"],
                    "phone_end_s": interval["phone_end_s"],
                    "video_phone_start_s": video_phone_start_s,
                    "video_phone_end_s": video_phone_end_s,
                    "audio_video_start_offset_s": audio_video_offset_s,
                    "event_label": event_label,
                    "training_label_status": (
                        "omit_boundary" if event_label == "ambiguous" else "usable"
                    ),
                    "manipulation_modality": manifest_row.get("modify_type", ""),
                    "mouth_crop_frame_indices_json": json.dumps(
                        mouth_crop_frame_indices,
                        separators=(",", ":"),
                    ),
                    **face_result,
                    "eligibility_status": status,
                    "exclusion_reason": reason,
                }
            )
        return events

    def _write_vild_trace(
        self,
        manifest_row: dict[str, str],
        video_id: str,
        probe: dict[str, object],
        transcription: dict[str, object],
        phone_intervals: list[dict[str, object]],
        events: list[dict[str, object]],
        trace_result: dict[str, object],
        audio_video_offset_s: float,
        report: dict[str, object],
    ) -> None:
        trace_path = (
            self.preprocessing.output_dir / "vild_traces" / f"{video_id}.json.gz"
        )
        artifact = build_vild_trace_artifact(
            video_id=video_id,
            source_file=manifest_row["file"],
            dataset_split=manifest_row.get("split", ""),
            source_group=manifest_row.get("source_group", ""),
            subject_id=manifest_row.get("subject_id", ""),
            probe=probe,
            transcription=transcription,
            phone_intervals=phone_intervals,
            events=events,
            trace_result=trace_result,
            audio_video_offset_s=audio_video_offset_s,
            event_window_before_s=self.preprocessing.event_window_before_s,
            event_window_after_s=self.preprocessing.event_window_after_s,
            settings=self.preprocessing.vild_trace,
            provenance={
                "raw_audio_path": report.get("raw_audio_path"),
                "raw_audio_sha256": report.get("raw_audio_sha256"),
                "transcription_path": report.get("transcription_path"),
                "transcription_sha256": report.get("transcription_sha256"),
                "alignment_path": report.get("alignment_path"),
                "alignment_sha256": report.get("alignment_sha256"),
                "alignment_audio_path": report.get("alignment_audio_path"),
                "alignment_audio_sha256": report.get("alignment_audio_sha256"),
                "alignment_audio_source": report.get("alignment_audio_source"),
                "deepfilter_fallback_applied": report.get(
                    "deepfilter_fallback_applied"
                ),
            },
        )
        atomic_write_gzip_json(trace_path, artifact)
        trace_hash = file_sha256(trace_path)
        for event in events:
            event["vild_trace_path"] = str(trace_path)
            event["vild_trace_sha256"] = trace_hash
            event["vild_trace_event_key"] = event["event_id"]

        trace_summary = trace_result["summary"]
        if not isinstance(trace_summary, dict):
            raise TypeError("VILD trace summary must be a mapping")
        report.update(
            {
                "vild_trace_path": str(trace_path),
                "vild_trace_sha256": trace_hash,
                "vild_trace_frame_count": trace_summary["attempted_frames"],
                "vild_trace_valid_landmark_ratio": trace_summary[
                    "valid_landmark_ratio"
                ],
                "vild_reference_window_count": len(
                    artifact["non_speech_reference"]["windows"]
                ),
            }
        )

    @staticmethod
    def _initial_report(
        manifest_row: dict[str, str], video_id: str, video_path: Path
    ) -> dict[str, object]:
        return {
            "video_id": video_id,
            "file": manifest_row["file"],
            "split": manifest_row.get("split", ""),
            "modify_type": manifest_row.get("modify_type", ""),
            "input_path": str(video_path),
            "eligibility_status": "ineligible",
            "exclusion_reason": "corrupt_media",
            "bilabial_event_count": 0,
            "eligible_event_count": 0,
            "pipeline_status": "failed",
        }
