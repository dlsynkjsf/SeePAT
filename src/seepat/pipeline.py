from __future__ import annotations

import argparse
import json
from pathlib import Path

from seepat.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_rows,
    stable_id,
)
from seepat.config import PipelineSettings, load_pipeline_settings
from seepat.preprocessing.alignment import MfaDockerAligner
from seepat.preprocessing.face import MouthEventAnalyzer
from seepat.preprocessing.transcription import WhisperTranscriber
from seepat.video_processor import PilotVideoProcessor

PIPELINE_VERSION = "pilot-v3"
ELIGIBILITY_REPORT_FIELDS = (
    "video_id",
    "file",
    "split",
    "modify_type",
    "eligibility_status",
    "exclusion_reason",
    "bilabial_event_count",
    "eligible_event_count",
    "pipeline_status",
    "error_type",
    "error_message",
)


def selected_manifest_rows(
    settings: PipelineSettings,
    limit: int | None = None,
) -> list[dict[str, str]]:
    rows = read_csv_rows(settings.dataset.pilot_manifest)
    included_ids = settings.dataset.include_video_ids
    if included_ids:
        requested = set(included_ids)
        rows = [row for row in rows if stable_id(row["file"]) in requested]
        found = {stable_id(row["file"]) for row in rows}
        missing = requested - found
        if missing:
            raise ValueError(
                "dataset.include_video_ids contains IDs absent from the manifest: "
                + ", ".join(sorted(missing))
            )
    if limit is not None:
        rows = rows[:limit]
    return rows


def _load_cached_result(
    result_path: Path,
    signature: str,
    retry_failed: bool = False,
) -> dict | None:
    if not result_path.is_file():
        return None
    cached = json.loads(result_path.read_text(encoding="utf-8"))
    if cached.get("cache_signature") != signature:
        return None
    if retry_failed and cached.get("video_report", {}).get("pipeline_status") != "complete":
        return None
    return cached


def _write_run_outputs(
    output_dir: Path,
    config_path: Path,
    signature: str,
    video_reports: list[dict[str, object]],
    events: list[dict[str, object]],
) -> dict[str, object]:
    atomic_write_csv(output_dir / "video_manifest.csv", video_reports)
    atomic_write_csv(output_dir / "bilabial_events.csv", events)
    eligibility_rows = [
        {key: report.get(key) for key in ELIGIBILITY_REPORT_FIELDS}
        for report in video_reports
    ]
    atomic_write_csv(output_dir / "eligibility_report.csv", eligibility_rows)

    summary: dict[str, object] = {
        "pipeline_version": PIPELINE_VERSION,
        "config": str(config_path),
        "cache_signature": signature,
        "videos_requested": len(video_reports),
        "videos_completed": sum(
            report["pipeline_status"] == "complete" for report in video_reports
        ),
        "videos_eligible": sum(
            report["eligibility_status"] == "eligible" for report in video_reports
        ),
        "bilabial_events": len(events),
        "eligible_events": sum(
            event["eligibility_status"] == "eligible" for event in events
        ),
        "ambiguous_training_events": sum(
            event["training_label_status"] == "omit_boundary" for event in events
        ),
    }
    atomic_write_json(output_dir / "run_summary.json", summary)
    return summary


def run_pipeline(
    config_path: Path,
    limit: int | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> dict[str, object]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    if force and retry_failed:
        raise ValueError("--force and --retry-failed cannot be used together")

    settings = load_pipeline_settings(config_path, PIPELINE_VERSION)
    preprocessing = settings.preprocessing
    output_dir = preprocessing.output_dir
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = selected_manifest_rows(settings, limit=limit)

    transcriber = WhisperTranscriber(
        model_name=preprocessing.whisper_model,
        device=preprocessing.whisper_device,
        compute_type=preprocessing.whisper_compute_type,
    )
    aligner = MfaDockerAligner(
        image=preprocessing.mfa_docker_image,
        cache_dir=preprocessing.mfa_cache_dir,
        dictionary=preprocessing.mfa_dictionary,
        acoustic_model=preprocessing.mfa_acoustic_model,
    )

    video_reports: list[dict[str, object]] = []
    all_events: list[dict[str, object]] = []
    with MouthEventAnalyzer() as mouth_analyzer:
        processor = PilotVideoProcessor(
            settings=settings,
            transcriber=transcriber,
            aligner=aligner,
            mouth_analyzer=mouth_analyzer,
        )
        for manifest_row in manifest_rows:
            video_id = stable_id(manifest_row["file"])
            work_dir = cache_dir / video_id
            result_path = work_dir / "result.json"
            cached = None if force else _load_cached_result(
                result_path,
                settings.cache_signature,
                retry_failed=retry_failed,
            )
            if cached is not None:
                video_reports.append(cached["video_report"])
                all_events.extend(cached["events"])
                continue

            work_dir.mkdir(parents=True, exist_ok=True)
            report, events = processor.process(
                manifest_row=manifest_row,
                video_id=video_id,
                work_dir=work_dir,
                force=force,
            )
            atomic_write_json(
                result_path,
                {
                    "cache_signature": settings.cache_signature,
                    "video_report": report,
                    "events": events,
                },
            )
            video_reports.append(report)
            all_events.extend(events)

    return _write_run_outputs(
        output_dir=output_dir,
        config_path=config_path,
        signature=settings.cache_signature,
        video_reports=video_reports,
        events=all_events,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    summary = run_pipeline(
        args.config,
        limit=args.limit,
        force=args.force,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
