"""Label-free local video preparation using frozen preprocessing settings."""

import argparse
import math
import pickle
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256, read_csv_rows
from seepat.config import load_pipeline_settings
from seepat.context_inputs import paired_indices
from seepat.live_progress import WorkflowProgress, format_live_progress
from seepat.pipeline import PIPELINE_VERSION
from seepat.preprocessing.eligibility import event_evidence_status, video_evidence_status
from seepat.preprocessing.processor import VideoProcessor
from seepat.preprocessing.vild import load_vild_trace
from seepat.training.calibration import (
    CALIBRATION_VERSION,
    CalibrationOptions,
    score_manifests_with_calibration,
)

from .expanded_context import REPRESENTATION as EXPANDED_REPRESENTATION
from .expanded_context import expanded_indices
from .frozen_inference import (
    _bound,
    _json,
    _runtime_hashes,
    _unchanged,
    validate_freeze,
    validate_inputs,
)

ANNOTATIONS = {"class_id", "video_class_id", "event_class_id", "label", "video_label", "event_label",
               "training_label_status", "modify_type", "manipulation_modality", "fake_segments_json"}


def raw_runtime_hashes(matched=False):
    """Exact implementation map for a future reviewed local execution consent."""
    return {**_runtime_hashes(matched), "local_raw_inputs.py": file_sha256(Path(__file__))}



def raw_environment():
    return {name: version(name) for name in
            ("torch", "faster-whisper", "mediapipe", "opencv-contrib-python", "scikit-learn")}


def validate_raw_request(freeze_path: Path, request_path: Path):
    """Read/hash trusted local bindings only; never construct speech/vision models."""
    request = _json(request_path)
    if (set(request) != {"version", "dataset_source", "split", "videos"}
            or request.get("version") != "raw-local-inputs-v1"
            or request.get("dataset_source") != "local-unlabelled" or request.get("split") != "local"):
        raise ValueError("Only unlabelled local raw inputs are enabled; external/test stays locked")
    validated = validate_freeze(freeze_path)
    bindings = validated["bindings"]
    _bound({"path": str(request_path), "sha256": file_sha256(request_path)}, bindings)
    settings = load_pipeline_settings(validated["paths"]["preprocessing"], PIPELINE_VERSION)
    if not settings.preprocessing.vild_trace.enabled:
        raise ValueError("Frozen preprocessing must retain VILD traces")
    sources = []
    if not isinstance(request["videos"], list) or not request["videos"]:
        raise ValueError("A nonempty local inventory is required")
    for row in request["videos"]:
        if set(row) != {"source", "origin"} or row["origin"] not in {"personal-recording", "local-demo"}:
            raise ValueError("Raw inputs need explicit local origin and no target annotations")
        path = _bound(row["source"], bindings)
        if path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            raise ValueError("Unsupported local video file")
        if "deepfakeeval" in "".join(c for c in str(path).lower() if c.isalnum()):
            raise ValueError("External-test paths remain locked")
        digest = row["source"]["sha256"]
        sources.append({"video_id": f"local-{digest[:24]}", "source_group": f"local-sha256:{digest}",
                        "source": {"path": str(path), "sha256": digest}, "origin": row["origin"]})
    if len({row["video_id"] for row in sources}) != len(sources):
        raise ValueError("Duplicate local source content")
    _unchanged(bindings)
    matched = validated["freeze"]["representation"] != "legacy-bilabial-v1"
    ready = (validated["execution_ready"]
             and validated["freeze"].get("raw_preparation_sha256") == raw_runtime_hashes(matched)
             and validated["freeze"].get("raw_environment") == raw_environment())
    return {**validated, "freeze_path": str(freeze_path.resolve()),
            "status": "passed", "sources": sources, "settings": settings,
            "execution_ready": ready}


def _binding(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def validate_prepared_provenance(spec, rows, requested, bindings):
    """Bind raw source, saved report, trace and caches throughout scoring/restarts."""
    path = _bound(spec["raw_provenance"], bindings)
    provenance = _json(path)
    videos = provenance["videos"]
    if (provenance.get("version") != "raw-local-provenance-v1"
            or provenance.get("scope") not in {"synthetic_engineering", "local_unlabelled_inference"}
            or provenance.get("preprocessing_sha256") != spec["preprocessing_sha256"]
            or len(videos) != len(requested)
            or [row["video_id"] for row in videos] != [row["video_id"] for row in requested]):
        raise ValueError("Raw provenance identity, settings or scope differs")
    _bound(provenance["raw_manifest"], bindings)
    traces = {}
    by_video = {video["video_id"]: video for video in videos}
    for video, expected in zip(videos, requested, strict=True):
        digest = video["source"]["sha256"]
        if (video["video_id"] != f"local-{digest[:24]}" or video["source_group"] != f"local-sha256:{digest}"
                or video["source_group"] != expected["source_group"] or video["status"] != expected["status"]
                or video["origin"] not in {"personal-recording", "local-demo"}):
            raise ValueError("Raw content identity or declared origin differs")
        _bound(video["source"], bindings)
        report = _json(_bound(video["report"], bindings))
        if report["report"]["video_id"] != video["video_id"]:
            raise ValueError("Raw report identity differs")
        if video["trace"]:
            trace = load_vild_trace(_bound(video["trace"], bindings), video["trace"]["sha256"])
            if (trace["video_id"] != video["video_id"] or trace["source_group"] != video["source_group"]
                    or trace["source_file"] != video["source"]["path"] or trace["dataset_split"] != "local"):
                raise ValueError("Raw source/trace identity differs")
            traces[video["video_id"]] = trace
        elif expected["status"] == "ready":
            raise ValueError("Scorable raw inputs require their original trace")
        for item in video["artifacts"]:
            artifact = Path(item["path"]).resolve()
            if not artifact.is_relative_to(path.parent) or artifact.is_symlink():
                raise ValueError("Raw preparation artifacts must remain contained")
            _bound(item, bindings)
    for row in rows:
        trace = traces[row["video_id"]]
        if row.get("kind") != "context":
            video = by_video[row["video_id"]]
            if (row.get("vild_trace_sha256") != video["trace"]["sha256"]
                    or row.get("vild_trace_event_key") != row["event_id"]
                    or Path(row["vild_trace_path"]).resolve() != Path(video["trace"]["path"]).resolve()
                    or not any(event["event_id"] == row["event_id"] for event in trace["bilabial_event_windows"])):
                raise ValueError("Scored bilabial does not belong to its raw trace")
    if spec.get("matched_inputs"):
        packet = _json(Path(spec["matched_inputs"]["path"]))
        for record in packet["records"]:
            if not record["windows"]:
                continue
            frames = {frame["frame_index"]: frame for frame in traces[record["video_id"]]["frames"]}
            if set(frames) != set(range(record["video_frame_count"])):
                raise ValueError("Context source/trace frame membership differs")
            for window in record["windows"]:
                if any(slot["trace_pts_s"] != frames[slot["source_frame"]]["timestamp_s"] for slot in window["slots"]):
                    raise ValueError("Context timestamp differs from original raw trace")
    return provenance["scope"]


def _destination(validated, output):
    output = output.resolve()
    protected = [Path(row["path"]).parent for row in validated["bindings"]]
    for source in validated["bindings"]:
        path = Path(source["path"])
        if output == path or path.is_relative_to(output):
            raise ValueError("Output would contain a protected input")
    for path in protected:
        if output.is_relative_to(path) and path not in {Path(validated["freeze_path"]).parent}:
            raise ValueError("Output is nested inside a protected input directory")
    return output


def export_prepared(validated, output_dir, *, processor_factory=None, engineering=False, progress=None):
    """Export via a speech/detector adapter; unknown labels never become real/fake."""
    validated = {**validated, "bindings": list(validated["bindings"])}
    if not engineering and not validated["execution_ready"]:
        raise ValueError("Raw execution needs a reviewed final local freeze")
    import sklearn

    calibration = validated["paths"]["calibration"]
    artifact = _json(calibration)
    record = calibration.with_suffix(".record.json")
    _bound({"path": str(record), "sha256": file_sha256(record)}, validated["bindings"])
    if (_json(record)["sha256"] != file_sha256(calibration)
            or artifact.get("calibration_version") != CALIBRATION_VERSION
            or artifact.get("sklearn_version") != sklearn.__version__):
        raise ValueError("Frozen population calibration record/version differs")
    CalibrationOptions(**artifact["options"]).validate()
    _unchanged(validated["bindings"])
    if Path(output_dir).is_symlink():
        raise FileExistsError("Output symlinks are not preparation destinations")
    output = _destination(validated, Path(output_dir))
    scope = "synthetic_engineering" if engineering else "local_unlabelled_inference"
    if output.exists():
        saved = _json(output / "summary.json")
        if (saved.get("version") != "raw-local-inputs-v1" or saved.get("status") != "complete"
                or saved.get("scope") != scope or saved.get("bindings") != validated["bindings"]
                or set(saved.get("outputs", {})) != {"inputs", "raw_manifest", "provenance"}):
            raise FileExistsError("Existing output is partial, changed or belongs to another input")
        for name, filename in (("inputs", "inputs.json"), ("raw_manifest", "events_raw.csv"), ("provenance", "provenance.json")):
            item = saved["outputs"][name]
            if Path(item["path"]).resolve() != output / filename or item["sha256"] != file_sha256(output / filename):
                raise FileExistsError("Completed preparation binding changed")
        prepared = validate_inputs(Path(validated["freeze_path"]), output / "inputs.json")
        if (saved.get("bilabials") != len(read_csv_rows(output / "events_raw.csv"))
                or saved.get("videos_requested") != prepared["requested"]
                or saved.get("status_counts") != dict(Counter(row["status"] for row in prepared["requested_videos"]))):
            raise FileExistsError("Saved preparation counts differ")
        return {**saved, "action": "skipped"}
    output.mkdir(parents=True, exist_ok=False)
    if progress:
        progress("initialize speech and vision", 0, len(validated["sources"]), "")
    requested, provenance, all_events, context_records = [], [], [], []
    settings = validated["settings"]
    settings = replace(settings, preprocessing=replace(settings.preprocessing, output_dir=output / "preprocessing"))
    with (processor_factory or local_media_processor)(settings) as processor:
        for index, source in enumerate(validated["sources"], 1):
            if progress:
                progress("prepare local videos", index - 1, len(validated["sources"]), source["video_id"])
            directory = output / "videos" / source["video_id"]
            report, events = processor.process(source, directory)
            if report.get("video_id") != source["video_id"] or any(
                    event.get("video_id") != source["video_id"] for event in events):
                raise ValueError("Preprocessing identity differs from raw source")
            report = {key: value for key, value in report.items() if key not in ANNOTATIONS}
            events = [{key: value for key, value in row.items() if key not in ANNOTATIONS} for row in events]
            retained = []
            for event in events:
                eligibility, reason = event_evidence_status(event, settings.preprocessing.min_valid_landmark_ratio)
                if event.get("eligibility_status") != eligibility:
                    raise ValueError("Preprocessing eligibility differs from frozen label-free rule")
                event.update(exclusion_reason=reason)
                if eligibility == "eligible":
                    if event.get("phoneme") not in {"m", "b", "p"}:
                        raise ValueError("Only aligned bilabials can carry numerical evidence")
                    if "phone_duration_s" not in event:
                        duration = round(float(event["phone_end_s"]) - float(event["phone_start_s"]), 9)
                        if not math.isfinite(duration) or duration < 0:
                            raise ValueError("Invalid aligned bilabial duration")
                        event["phone_duration_s"] = duration
                    clip = Path(event.get("mouth_crop_path", event.get("mouth_clip_path"))).resolve()
                    if not clip.is_relative_to(output):
                        raise ValueError("Preprocessing clip must stay in its fresh package")
                    retained.append({**event, "file": source["source"]["path"], "dataset_split": "local",
                                     "dataset_source": "local-unlabelled", "source_group": source["source_group"],
                                     "kind": "bilabial", "mouth_clip_path": str(clip),
                                     "mouth_clip_sha256": file_sha256(clip)})
            status, _, _ = video_evidence_status(events, settings.preprocessing.min_bilabial_events)
            trace_binding = None
            if report.get("pipeline_status") != "complete":
                status = (report["exclusion_reason"] if report.get("exclusion_reason") in {"alignment_failed", "corrupt_media"}
                          else "preprocessing_failed")
            elif status != "eligible":
                status = "no_eligible_bilabials" if not events else "ineligible"
            else:
                status = "ready"
                trace_path = Path(report["vild_trace_path"]).resolve()
                if not trace_path.is_relative_to(output):
                    raise ValueError("Trace must stay inside the fresh package")
                trace_binding = {"path": str(trace_path), "sha256": report["vild_trace_sha256"]}
                trace = load_vild_trace(trace_path, trace_binding["sha256"])
                if (trace.get("video_id") != source["video_id"] or trace.get("source_group") != source["source_group"]
                        or trace.get("source_file") != source["source"]["path"] or trace.get("dataset_split") != "local"):
                    raise ValueError("Raw source/trace identity differs")
                if any(event.get("vild_trace_path") != str(trace_path)
                       or event.get("vild_trace_sha256") != trace_binding["sha256"]
                       or event.get("vild_trace_event_key") != event["event_id"] for event in retained):
                    raise ValueError("Event and raw trace bindings differ")
                all_events.extend(retained)
            directory.mkdir(parents=True, exist_ok=True)
            context_record = {"video_id": source["video_id"], "source_group": source["source_group"],
                              "dataset_split": "local", "windows": []}
            if validated["freeze"]["representation"] != "legacy-bilabial-v1" and status == "ready" and len(retained) >= 2:
                windows = processor.contexts(source, report, directory)
                for window in windows:
                    if window["technical_ok"]:
                        clip = (directory / window["clip_file"]).resolve()
                        if not clip.is_relative_to(directory.resolve()):
                            raise ValueError("Context path escapes its video directory")
                        window["clip_file"] = clip.relative_to(output).as_posix()
                context_record.update(video_frame_count=report["video_frame_count"], fps=report["fps"], windows=windows)
            context_records.append(context_record)
            atomic_write_json(directory / "report.json", {"report": report, "events": events})
            requested.append({"video_id": source["video_id"], "source_group": source["source_group"],
                              "dataset_source": "local-unlabelled", "status": status})
            artifacts = [_binding(path) for path in sorted(directory.rglob("*")) if path.is_file()]
            provenance.append({**source, "status": status, "report": _binding(directory / "report.json"),
                               "trace": trace_binding, "artifacts": artifacts})
    raw_manifest = output / "events_raw.csv"
    atomic_write_csv(raw_manifest, all_events)
    calibrated = output / "calibrated/events_local_calibrated.csv"
    if progress:
        progress("apply frozen Train calibration; input-video forests only", 0, 0, "")
    if all_events:
        score_manifests_with_calibration(calibration, {"local": raw_manifest}, output / "calibrated")
    else:
        atomic_write_csv(calibrated, [])
        atomic_write_json(calibrated.parent / "summary.json", {
            "mode": "score_only", "calibration_version": CALIBRATION_VERSION,
            "calibration": str(calibration), "calibration_sha256": file_sha256(calibration),
            "scored_manifests": {"local": str(calibrated)},
            "scored_manifest_sha256": {"local": file_sha256(calibrated)},
            "strategy": "No eligible bilabials; no population or input-video fit", "videos_scored": 0})
    spec = {"version": "prepared-inference-v1", "dataset_source": "local-unlabelled", "split": "local",
            "representation": validated["freeze"]["representation"],
            "preprocessing_sha256": validated["freeze"]["preprocessing"]["sha256"],
            "manifest": _binding(calibrated),
            "calibration_summary": _binding(output / "calibrated/summary.json"), "requested_videos": requested}
    if validated["freeze"]["representation"] != "legacy-bilabial-v1":
        indexes = {"control": [], "candidate": []}
        calibrated_rows = read_csv_rows(calibrated)
        for record in context_records:
            control, candidate = paired_indices(
                [row for row in calibrated_rows if row["video_id"] == record["video_id"]], record["windows"])
            indexes["control"].extend(control)
            indexes["candidate"].extend(candidate)
        expanded = validated["freeze"]["representation"] == EXPANDED_REPRESENTATION
        if expanded:
            indexes = {"expanded": []}
            for record in context_records:
                indexes["expanded"].extend(expanded_indices(
                    [row for row in calibrated_rows if row["video_id"] == record["video_id"]], record["windows"]))
        packet = {"version": "expanded-local-inputs-v1" if expanded else "matched-local-inputs-v1", "arm": validated["contract"]["clip_input_contract"]["arm"],
                  "context_protocol_sha256": validated["freeze"]["context_protocol"]["sha256"],
                  "bilabial_manifest_sha256": file_sha256(calibrated), "indexes": indexes, "records": context_records}
        atomic_write_json(output / "matched.json", packet)
        spec["matched_inputs"] = _binding(output / "matched.json")
    atomic_write_json(output / "provenance.json", {"version": "raw-local-provenance-v1",
                                                  "scope": "synthetic_engineering" if engineering else "local_unlabelled_inference",
                                                  "preprocessing_sha256": spec["preprocessing_sha256"],
                                                  "raw_manifest": _binding(raw_manifest), "videos": provenance})
    spec["raw_provenance"] = _binding(output / "provenance.json")
    atomic_write_json(output / "inputs.json", spec)
    validate_inputs(Path(validated["freeze_path"]), output / "inputs.json")
    _unchanged(validated["bindings"])
    summary = {"version": "raw-local-inputs-v1", "status": "complete",
               "scope": "synthetic_engineering" if engineering else "local_unlabelled_inference",
               "bindings": validated["bindings"], "videos_requested": len(requested),
               "status_counts": dict(Counter(row["status"] for row in requested)), "bilabials": len(all_events),
               "outputs": {"inputs": _binding(output / "inputs.json"),
                           "raw_manifest": _binding(raw_manifest), "provenance": spec["raw_provenance"]}}
    atomic_write_json(output / "summary.json", summary)
    if progress:
        progress("complete", len(requested), len(requested), "")
    return summary


class _UnlabelledVideoProcessor(VideoProcessor):
    """Reuse extraction, stripping legacy annotation bookkeeping before persistence."""

    def _process_events(self, *args, **kwargs):
        return [{key: value for key, value in event.items() if key not in ANNOTATIONS}
                for event in super()._process_events(*args, **kwargs)]

    @staticmethod
    def _initial_report(*args, **kwargs):
        return {key: value for key, value in VideoProcessor._initial_report(*args, **kwargs).items()
                if key not in ANNOTATIONS}


class _LocalMedia:
    def __init__(self, processor):
        self.processor = processor

    def process(self, source, directory):
        directory.mkdir(parents=True)
        manifest_row = {"file": source["source"]["path"], "split": "local", "source_group": source["source_group"]}
        original = self.processor
        settings = replace(original.settings, preprocessing=replace(original.preprocessing, output_dir=directory))
        processor = _UnlabelledVideoProcessor(settings, original.transcriber, original.aligner, original.mouth_analyzer)
        report, events = processor.process(manifest_row, source["video_id"], directory, force=False)
        if report.get("vild_trace_path"):
            trace = load_vild_trace(Path(report["vild_trace_path"]), report["vild_trace_sha256"])
            report["video_frame_count"] = len(trace["frames"])
        return report, events

    def contexts(self, source, report, directory):
        from seepat.coverage import context_windows

        from . import context_media as accepted

        trace = load_vild_trace(Path(report["vild_trace_path"]), report["vild_trace_sha256"])
        frames = {int(frame["frame_index"]): frame for frame in trace["frames"]}
        count, fps = len(frames), float(report["fps"])
        if set(frames) != set(range(count)):
            raise ValueError("Context source/trace membership differs")
        windows = context_windows(count, fps)
        needed = {index for window in windows for index in range(window["start_frame"], window["stop_frame_exclusive"])}
        decoded, actual_count = accepted.decode_needed(Path(source["source"]["path"]), needed, count,
                                                              lambda *args: None)
        if actual_count != count:
            raise ValueError("Source decoded count differs from frozen trace")
        video = {"video_id": source["video_id"], "source_group": source["source_group"], "split": "local", "fps": fps}
        with accepted.StatelessMouthEventAnalyzer() as analyzer:
            return [accepted.window_record(video, plan, decoded, frames, analyzer, directory) for plan in windows]


@contextmanager
def local_media_processor(settings):
    """Real user-run speech/alignment/detector seam. Never instantiated by synthetic tests."""
    from seepat.preprocessing.alignment import MfaDockerAligner
    from seepat.preprocessing.face import MouthEventAnalyzer
    from seepat.preprocessing.transcription import WhisperTranscriber

    prep = settings.preprocessing
    aligner = MfaDockerAligner(prep.mfa_docker_image, prep.mfa_cache_dir, prep.mfa_dictionary, prep.mfa_acoustic_model)
    transcriber = WhisperTranscriber(prep.whisper_model, prep.whisper_device, prep.whisper_compute_type)
    with MouthEventAnalyzer() as analyzer:
        yield _LocalMedia(_UnlabelledVideoProcessor(settings, transcriber, aligner, analyzer))


def main(argv=None, *, processor_factory=None, engineering=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--raw-inputs", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--watch-output", type=Path)
    parser.add_argument("--watch-seconds", type=float, default=5)
    args = parser.parse_args(argv)
    if not math.isfinite(args.watch_seconds) or not 0 < args.watch_seconds <= 60:
        parser.error("--watch-seconds must be finite and in (0,60]")
    if args.watch_output is not None:
        if args.freeze or args.raw_inputs or args.run or args.output_dir:
            parser.error("The read-only watcher does not take preparation arguments")
        try:
            while True:
                path = args.watch_output / "summary.progress.json"
                try:
                    record = _json(path)
                except (OSError, ValueError, TypeError):
                    record = {"status": "unavailable", "message": "Waiting for preparation progress; no work is started by this watcher."}
                print(format_live_progress(record), flush=True)
                if record["status"] in {"complete", "failed", "interrupted"}:
                    return record
                time.sleep(args.watch_seconds)
        except KeyboardInterrupt:
            print("Tracking stopped; the worker was not interrupted.", flush=True)
            return {"status": "watching_stopped"}
    if args.freeze is None or args.raw_inputs is None:
        parser.error("--freeze and --raw-inputs are required for verification/preparation")
    if args.run != (args.output_dir is not None):
        parser.error("--run and --output-dir must be supplied together")
    print("RAW PREPARATION | verify local inventory and frozen settings", flush=True)
    tracker, started = None, False

    def progress(phase, done, total, item):
        nonlocal started
        if not started:
            tracker.start_job(1, "local raw-video preparation")
            started = True
        tracker(phase, done, total, item)

    try:
        validated = validate_raw_request(args.freeze, args.raw_inputs)
        if not args.run:
            print("VERIFICATION PASSED | PREPARATION ONLY; no media processing or model construction", flush=True)
            return validated
        tracker = WorkflowProgress(args.freeze, args.output_dir / "summary.json", jobs_total=1)
        result = export_prepared(validated, args.output_dir, processor_factory=processor_factory, engineering=engineering,
                                 progress=progress)
        if result.get("action") == "skipped":
            print("RAW PREPARATION | VERIFIED COMPLETE; SKIPPED", flush=True)
            return result
        tracker.finish()
        print(f"RAW PREPARATION COMPLETE | {args.output_dir / 'summary.json'}", flush=True)
        return result
    except KeyboardInterrupt:
        if started:
            tracker.finish(KeyboardInterrupt())
        print("RAW PREPARATION INTERRUPTED | partial work preserved", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError):
        if started:
            tracker.finish(RuntimeError("Preparation failed; preserve partial work"))
        print("RAW PREPARATION FAILED | contract_or_output_gate; preserve inputs and partial output", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
