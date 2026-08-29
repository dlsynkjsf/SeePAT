from __future__ import annotations

import json
import traceback
from pathlib import Path

from seepat.artifacts import atomic_write_json
from seepat.config import PipelineSettings
from seepat.preprocessing.alignment import MfaAlignmentError, MfaDockerAligner
from seepat.preprocessing.eligibility import (
    event_evidence_status,
    video_evidence_status,
)
from seepat.preprocessing.events import classify_event_label, parse_segments
from seepat.preprocessing.face import MouthEventAnalyzer
from seepat.preprocessing.media import extract_mono_audio, probe_media
from seepat.preprocessing.transcription import WhisperTranscriber


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

            audio_path = extract_mono_audio(
                video_path,
                work_dir / "audio.wav",
                self.preprocessing.ffmpeg_path,
                force=force,
            )
            transcription = self._transcribe(audio_path, work_dir, force)
            report["transcript"] = transcription["text"]
            if not str(transcription["text"]).strip():
                report["exclusion_reason"] = "alignment_failed"
                raise ValueError("Whisper produced an empty transcript")

            intervals = self.aligner.align(
                audio_path,
                str(transcription["text"]),
                work_dir / "alignment.json",
                force=force,
            )
            report["bilabial_event_count"] = len(intervals)
            events = self._process_events(
                manifest_row=manifest_row,
                video_id=video_id,
                video_path=video_path,
                fps=float(probe["fps"]),
                intervals=intervals,
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
    ) -> list[dict[str, object]]:
        manipulated_segments = parse_segments(manifest_row.get("fake_segments_json"))
        events: list[dict[str, object]] = []
        for event_index, interval in enumerate(intervals):
            event_id = f"{video_id}-{event_index:03d}"
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
                phone_start_s=float(interval["phone_start_s"]),
                phone_end_s=float(interval["phone_end_s"]),
                phoneme=str(interval["phoneme"]),
                fps=fps,
                window_before_s=self.preprocessing.event_window_before_s,
                window_after_s=self.preprocessing.event_window_after_s,
                mouth_clip_path=mouth_path,
                overlay_path=overlay_path,
            )
            status, reason = event_evidence_status(
                face_result, self.preprocessing.min_valid_landmark_ratio
            )
            event_label = classify_event_label(
                float(interval["phone_start_s"]),
                float(interval["phone_end_s"]),
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
                    "event_label": event_label,
                    "training_label_status": (
                        "omit_boundary" if event_label == "ambiguous" else "usable"
                    ),
                    "manipulation_modality": manifest_row.get("modify_type", ""),
                    **face_result,
                    "eligibility_status": status,
                    "exclusion_reason": reason,
                }
            )
        return events

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
