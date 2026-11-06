from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from seepat.artifacts import atomic_write_json, read_csv_rows, stable_id
from seepat.artifacts import file_sha256 as _sha256
from seepat.config import PipelineSettings, load_pipeline_settings
from seepat.coverage import run_coverage_pilot
from seepat.decision import run_decision_job
from seepat.live_progress import ProgressCallback, WorkflowProgress
from seepat.pipeline import PIPELINE_VERSION, run_pipeline
from seepat.preprocessing.augmentation import run_trace_augmentation
from seepat.preprocessing.contract import audit_preprocessing_contract
from seepat.training.manifest import prepare_training_manifests
from seepat.training.study import run_study_job
from seepat.workflow_config import (
    SUPPORTED_TRAINING_MODELS,  # noqa: F401 -- public configuration API
    ModelTrainingJob,
    NumericalCalibrationJob,
    WorkflowJob,
    WorkflowSettings,  # noqa: F401 -- public configuration API
    _local_inference_api,
    _require_mapping,
    load_workflow_settings,
)

WORKFLOW_VERSION = "local-pipeline-v2"
def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def preprocessing_outputs_are_current(
    settings: PipelineSettings,
    progress: ProgressCallback | None = None,
) -> bool:
    output_dir = settings.preprocessing.output_dir
    summary_path = output_dir / "run_summary.json"
    video_manifest_path = output_dir / "video_manifest.csv"
    required_paths = [
        summary_path,
        video_manifest_path,
        output_dir / "bilabial_events.csv",
        output_dir / "eligibility_report.csv",
    ]
    if settings.preprocessing.vild_trace.enabled:
        required_paths.append(output_dir / "vild_trace_index.csv")
    if not all(path.is_file() for path in required_paths):
        return False

    try:
        source_rows = read_csv_rows(settings.dataset.pilot_manifest)
        video_rows = read_csv_rows(video_manifest_path)
        summary = _read_json(summary_path)
        source_files = Counter(row["file"] for row in source_rows)
        output_files = Counter(row["file"] for row in video_rows)
        expected_ids = Counter(stable_id(row["file"]) for row in source_rows)
        output_ids = Counter(row["video_id"] for row in video_rows)
        base_current = (
            summary.get("pipeline_version") == PIPELINE_VERSION
            and summary.get("cache_signature") == settings.cache_signature
            and summary.get("videos_requested") == len(source_rows)
            and len(video_rows) == len(source_rows)
            and source_files == output_files
            and expected_ids == output_ids
        )
        if not base_current:
            return False
        if not settings.preprocessing.vild_trace.enabled:
            return True

        trace_index_path = output_dir / "vild_trace_index.csv"
        if (
            summary.get("vild_trace_enabled") is not True
            or summary.get("vild_trace_index_sha256") != _sha256(trace_index_path)
        ):
            return False
        trace_rows = read_csv_rows(trace_index_path)
        if summary.get("vild_trace_videos") != len(trace_rows):
            return False
        for index, row in enumerate(trace_rows):
            if progress is not None:
                progress("verify preprocessing traces", index, len(trace_rows), row["video_id"])
            trace_path = Path(row["vild_trace_path"])
            if not trace_path.is_file() or row["vild_trace_sha256"] != _sha256(
                trace_path
            ):
                return False
        if progress is not None:
            progress("verify preprocessing traces", len(trace_rows), len(trace_rows), "")
        expected_trace_ids = {
            row["video_id"]
            for row in video_rows
            if row.get("pipeline_status") == "complete"
            and int(row.get("bilabial_event_count", 0)) > 0
        }
        return expected_trace_ids == {row["video_id"] for row in trace_rows}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def training_outputs_are_current(
    settings: PipelineSettings,
    output_dir: Path,
) -> bool:
    if not settings.dataset.split:
        return False
    preprocessing_dir = settings.preprocessing.output_dir
    inputs = {
        "source_manifest": settings.dataset.pilot_manifest,
        "video_manifest": preprocessing_dir / "video_manifest.csv",
        "event_manifest": preprocessing_dir / "bilabial_events.csv",
    }
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file() or not all(path.is_file() for path in inputs.values()):
        return False

    try:
        summary = _read_json(summary_path)
        recorded_inputs = _require_mapping(
            summary.get("input_artifacts"), "Training input artifacts"
        )
        for name, path in inputs.items():
            recorded = _require_mapping(recorded_inputs.get(name), f"Recorded {name}")
            if recorded.get("sha256") != _sha256(path):
                return False

        combined_path = output_dir / "events.csv"
        if (
            not combined_path.is_file()
            or summary.get("combined_manifest_sha256") != _sha256(combined_path)
        ):
            return False

        split_hashes = _require_mapping(
            summary.get("split_manifest_sha256"), "Split manifest hashes"
        )
        split_paths = _require_mapping(summary.get("split_manifests"), "Split manifests")
        if settings.dataset.split not in split_paths:
            return False
        for split, recorded_path in split_paths.items():
            path = Path(str(recorded_path))
            if not path.is_file() or split_hashes.get(split) != _sha256(path):
                return False

        expected_split_path = output_dir / f"events_{settings.dataset.split}.csv"
        return expected_split_path.is_file()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_workflow_job(
    job: WorkflowJob,
    retry_failed: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    settings = load_pipeline_settings(job.pipeline_config, PIPELINE_VERSION)
    if progress is not None:
        progress("verify preprocessing cache", 0, 0, "")
    preprocessing_current = preprocessing_outputs_are_current(
        settings, **({"progress": progress} if progress is not None else {})
    )

    if preprocessing_current and not retry_failed:
        print(f"[{job.name}] preprocessing: skipped (artifacts are current)")
        pipeline_summary = _read_json(
            settings.preprocessing.output_dir / "run_summary.json"
        )
        preprocessing_action = "skipped"
    else:
        reason = "retrying cached failures" if retry_failed else "artifacts are missing or stale"
        print(f"[{job.name}] preprocessing: running ({reason})")
        if progress is not None:
            progress("preprocess videos", 0, 0, "")
        pipeline_summary = run_pipeline(
            job.pipeline_config,
            retry_failed=retry_failed,
            **({"progress": progress} if progress is not None else {}),
        )
        preprocessing_action = "ran"

    contract_summary = None
    if settings.preprocessing.vild_trace.enabled:
        print(f"[{job.name}] preprocessing contract: auditing")
        contract_summary = audit_preprocessing_contract(
            settings.preprocessing.output_dir,
            **({"progress": progress} if progress is not None else {}),
        )

    if progress is not None:
        progress("verify training manifest", 0, 0, "")
    manifest_current = training_outputs_are_current(settings, job.manifest_output_dir)
    if manifest_current:
        print(f"[{job.name}] training manifest: skipped (artifacts are current)")
        manifest_summary = _read_json(job.manifest_output_dir / "summary.json")
        manifest_action = "skipped"
    else:
        print(f"[{job.name}] training manifest: building")
        if progress is not None:
            progress("build training manifest", 0, 0, "")
        manifest_summary = prepare_training_manifests(
            source_manifest_path=settings.dataset.pilot_manifest,
            video_manifest_path=settings.preprocessing.output_dir / "video_manifest.csv",
            event_manifest_path=settings.preprocessing.output_dir / "bilabial_events.csv",
            output_dir=job.manifest_output_dir,
        )
        manifest_action = "built"

    result = {
        "name": job.name,
        "pipeline_config": job.pipeline_config.as_posix(),
        "preprocessing": {
            "action": preprocessing_action,
            "summary": pipeline_summary,
        },
        "training_manifest": {
            "action": manifest_action,
            "summary": manifest_summary,
        },
    }
    if contract_summary is not None:
        result["preprocessing_contract"] = contract_summary
    return result


def _model_training_options(job: ModelTrainingJob):
    from seepat.training.train import FUSION_MODEL, TrainingOptions
    from seepat.training.video_objective import VIDEO_MAX

    options = TrainingOptions(**job.options)
    options.validate()
    if options.supervision == VIDEO_MAX and job.model != FUSION_MODEL:
        raise ValueError("Production video_max supervision is supported only by full fusion")
    if job.input_profile is not None:
        from seepat.training.matched_inputs import validate_profile

        validate_profile(job.input_profile)
        if (job.model != FUSION_MODEL or options.supervision != VIDEO_MAX
                or options.sequence_length != 16 or options.unfreeze_final_backbone_stages):
            raise ValueError("Matched inputs require frozen fusion/video_max and 16 slots")
    return options


def _model_training_configuration_matches(
    job: ModelTrainingJob,
    run_record: dict[str, Any],
) -> bool:
    from seepat.training.train import (
        FUSION_MODEL,
        SWIN_BASE_MODEL,
        VIDEO_F1_SELECTION,
        model_contract_name,
        training_version_for_model,
    )
    from seepat.training.video_objective import VIDEO_MAX, VIDEO_SUPERVISION_CONTRACT

    try:
        options = _model_training_options(job)
        contract = _require_mapping(run_record.get("resume_contract"), "Resume contract")
        expected_type = (
            "engineering_preflight"
            if options.max_train_batches is not None
            else "training_experiment"
        )
        recorded_options = dict(
            _require_mapping(run_record.get("options"), "Recorded training options")
        )
        expected_options = asdict(options)
        recorded_options.setdefault("positive_class_weight_ratio", None)
        recorded_options.setdefault("selection_metric", VIDEO_F1_SELECTION)
        recorded_options.setdefault("unfreeze_final_backbone_stages", False)
        recorded_options.setdefault("backbone_learning_rate", None)
        recorded_options.setdefault("loss_function", "cross_entropy")
        recorded_options.setdefault("focal_gamma", 2.0)
        recorded_options.setdefault("supervision", "event")
        recorded_options.pop("epochs", None)
        expected_options.pop("epochs")
        if job.device != "auto" and run_record.get("device") != job.device:
            return False
        if job.model == FUSION_MODEL:
            from seepat.evidence import calibrated_manifest_contract

            if job.input_profile is not None:
                from seepat.training.matched_inputs import MatchedClipDataset

                datasets = [MatchedClipDataset(path, job.project_root, split, job.input_profile,
                                               options.sequence_length, options.image_size)
                            for split, path in (("train", job.train_manifest), ("val", job.validation_manifest))]
                train_contract, val_contract = [dataset.calibration_contract for dataset in datasets]
                if (datasets[0].input_contract != datasets[1].input_contract
                        or contract.get("clip_input_contract") != datasets[0].input_contract):
                    return False
            else:
                train_contract = calibrated_manifest_contract(job.train_manifest)
                val_contract = calibrated_manifest_contract(job.validation_manifest)
                if contract.get("clip_input_contract") is not None:
                    return False
            if train_contract != val_contract or contract.get("fusion_inputs") != train_contract:
                return False
        return (
            run_record.get("run_type") == expected_type
            and recorded_options == expected_options
            and contract.get("supervision") == (
                VIDEO_SUPERVISION_CONTRACT if options.supervision == VIDEO_MAX else None
            )
            and (options.supervision != VIDEO_MAX or contract.get("seed") == options.seed)
            and contract.get("training_version") == training_version_for_model(job.model)
            and contract.get("model") == model_contract_name(job.model)
            and contract.get("model_name", SWIN_BASE_MODEL) == job.model
            and contract.get("pretrained") == job.pretrained
            and contract.get("initial_checkpoint")
            == (
                {
                    "path": job.initialize_from.as_posix(),
                    "sha256": _sha256(job.initialize_from),
                }
                if job.initialize_from is not None
                else None
            )
            and contract.get("train_manifest_sha256") == _sha256(job.train_manifest)
            and contract.get("validation_manifest_sha256")
            == _sha256(job.validation_manifest)
        )
    except (OSError, TypeError, ValueError, KeyError):
        return False


def model_training_outputs_are_current(job: ModelTrainingJob) -> bool:
    required_paths = (
        job.train_manifest,
        job.validation_manifest,
        job.output_dir / "run.json",
        job.output_dir / "history.json",
        job.output_dir / "checkpoint_last.pt",
        job.output_dir / "checkpoint_best.pt",
    )
    if not all(path.is_file() for path in required_paths):
        return False
    try:
        run_record = _read_json(job.output_dir / "run.json")
        options = _model_training_options(job)
        completed_epochs = int(run_record.get("completed_epochs", 0))
        completed_as_planned = completed_epochs == options.epochs or (
            run_record.get("stopped_early") is True
            and 0 < completed_epochs <= options.epochs
        )
        return (
            run_record.get("status") == "complete"
            and completed_as_planned
            and _model_training_configuration_matches(job, run_record)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _run_model_training(
    job: ModelTrainingJob,
    resume_from: Path | None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    from seepat.training.train import train_from_manifests

    return train_from_manifests(
        train_manifest=job.train_manifest,
        validation_manifest=job.validation_manifest,
        output_dir=job.output_dir,
        project_root=job.project_root,
        options=_model_training_options(job),
        device_name=job.device,
        pretrained=job.pretrained,
        model_name=job.model,
        resume_from=resume_from,
        initialize_from=job.initialize_from,
        progress=progress,
        **({"input_profile": job.input_profile} if job.input_profile is not None else {}),
    )


def run_model_training_job(
    job: ModelTrainingJob, progress: ProgressCallback | None = None,
) -> dict[str, object]:
    if model_training_outputs_are_current(job):
        print(f"[{job.name}] model training: skipped (artifacts are current)")
        return {
            "name": job.name,
            "action": "skipped",
            "summary": _read_json(job.output_dir / "run.json"),
        }

    if job.readiness_dir is not None:
        from seepat.training.readiness import require_readiness

        if progress is not None:
            progress("verify training readiness", 0, 0, job.name)
        require_readiness(job)

    run_path = job.output_dir / "run.json"
    checkpoint_path = job.output_dir / "checkpoint_last.pt"
    resume_from = None
    if run_path.is_file():
        previous_run = _read_json(run_path)
        if not _model_training_configuration_matches(job, previous_run):
            raise RuntimeError(
                f"[{job.name}] existing training output does not match the workflow config; "
                "use a new output directory"
            )
        if checkpoint_path.is_file():
            resume_from = checkpoint_path

    action = "resumed" if resume_from is not None else "ran"
    print(f"[{job.name}] model training: {action}")
    summary = _run_model_training(job, resume_from, progress=progress)
    return {"name": job.name, "action": action, "summary": summary}


def numerical_calibration_outputs_are_current(job: NumericalCalibrationJob) -> bool:
    from seepat.training.calibration import (
        CalibrationOptions,
        calibration_outputs_are_current,
        score_only_outputs_are_current,
    )

    if job.calibration is not None:
        return score_only_outputs_are_current(
            job.calibration,
            dict(job.score_manifests),
            job.output_dir,
        )
    assert job.train_manifest is not None
    return calibration_outputs_are_current(
        job.train_manifest, dict(job.score_manifests), job.output_dir,
        CalibrationOptions(job.min_video_reference_frames, job.isolation_trees, job.random_seed),
    )


def run_numerical_calibration_job(
    job: NumericalCalibrationJob, progress: ProgressCallback | None = None,
) -> dict[str, object]:
    if progress is not None:
        progress("verify calibration inputs and outputs", 0, 0, "")
    if numerical_calibration_outputs_are_current(job):
        print(f"[{job.name}] numerical calibration: skipped (artifacts are current)")
        return {
            "name": job.name,
            "action": "skipped",
            "summary": _read_json(job.output_dir / "summary.json"),
        }
    from seepat.training.calibration import (
        CalibrationOptions,
        fit_and_score_calibration,
        score_manifests_with_calibration,
    )

    if job.calibration is not None:
        print(f"[{job.name}] numerical calibration: frozen Train parameters, new input forests")
        summary = score_manifests_with_calibration(
            calibration_path=job.calibration,
            score_manifests=dict(job.score_manifests),
            output_dir=job.output_dir,
        )
        return {"name": job.name, "action": "ran", "summary": summary}
    assert job.train_manifest is not None
    print(f"[{job.name}] numerical calibration: Train regression and independent input-video forests")
    summary = fit_and_score_calibration(
        train_manifest=job.train_manifest,
        score_manifests=dict(job.score_manifests),
        output_dir=job.output_dir,
        options=CalibrationOptions(
            min_video_reference_frames=job.min_video_reference_frames,
            isolation_trees=job.isolation_trees,
            random_seed=job.random_seed,
        ),
        progress=progress,
    )
    return {"name": job.name, "action": "ran", "summary": summary}


def run_workflow(
    config_path: Path,
    retry_failed: bool = False,
) -> dict[str, object]:
    settings = load_workflow_settings(config_path)
    report: dict[str, object] = {
        "workflow_version": WORKFLOW_VERSION,
        "config": config_path.as_posix(),
        "jobs": [],
    }
    tasks = (
        [("jobs", job) for job in settings.jobs]
        + [("coverage_pilot", job) for job in settings.coverage_pilot_jobs]
        + [("trace_augmentation", job) for job in settings.trace_augmentation_jobs]
        + [("numerical_calibration", job) for job in settings.numerical_calibration_jobs]
        + [("model_training", job) for job in settings.model_training_jobs]
        + ([("ablation_study", settings.study_job)] if settings.study_job else [])
        + [("decisions", job) for job in settings.decision_jobs]
        + [("local_inference", job) for job in settings.local_inference_jobs]
    )
    progress = WorkflowProgress(
        config_path, settings.report_path, len(tasks),
        model_training_config=settings.model_training_config,
    )
    try:
        for index, (kind, job) in enumerate(tasks, 1):
            progress.start_job(index, job.name)
            if kind == "jobs":
                result = run_workflow_job(job, retry_failed=retry_failed, progress=progress)
            elif kind == "trace_augmentation":
                result = run_trace_augmentation(job, progress=progress)
            elif kind == "coverage_pilot":
                result = run_coverage_pilot(job, progress=progress)
            elif kind == "numerical_calibration":
                result = run_numerical_calibration_job(job, progress=progress)
            elif kind == "local_inference":
                result = _local_inference_api().run_local_job(job, progress=progress)
            elif kind == "decisions":
                result = run_decision_job(job, progress=progress)
            elif kind == "ablation_study":
                result = run_study_job(job, progress=progress)
            else:
                progress("verify or resume model job", 0, 0, "")
                result = run_model_training_job(job, progress=progress)
            report.setdefault(kind, []).append(result)
        atomic_write_json(settings.report_path, report)
    except BaseException as error:
        progress.finish(error)
        raise
    progress.finish(deferred=any(row["action"] == "deferred" for row in report.get("decisions", [])))

    return report


def _local_readable_summary(row):
    """Display sealed small reports only; upstream full verification is separate."""
    from seepat.artifacts import file_sha256
    from seepat.inference import readable_xai

    viewer = Path(readable_xai.__file__)
    viewer_bytes = viewer.read_bytes()
    expected = row["summary"]["contract"]["implementation_sha256"]["readable_xai.py"]
    if hashlib.sha256(viewer_bytes).hexdigest() != expected:
        raise ValueError("Readable XAI source changed; review its release binding")
    binding = row["summary"]["stage_summaries"]["explanation"]
    summary = Path(binding["path"])
    if summary.name != "summary.json" or file_sha256(summary) != binding["sha256"]:
        raise ValueError("Explanation summary differs from completed workflow")
    text = readable_xai.render_saved_report(summary.parent, technical=True, paragraph_layout=True)
    if viewer.read_bytes() != viewer_bytes or file_sha256(summary) != binding["sha256"]:
        raise ValueError("Presentation source changed during display")
    return text


def format_workflow_summary(report: dict[str, object], report_path: Path) -> str:
    """Compact terminal summary; JSON remains the authoritative report."""
    lines = ["", "#" * 72, "WORKFLOW COMPLETE"]
    for row in report.get("coverage_pilot", []):
        summary = row["summary"]
        gain = summary["visual_coverage_gain"]
        text = f"{gain:+.1%}" if gain is not None else "unavailable"
        lines.append(f"  {row['name']} | {row['action']} | visual coverage gain {text}")
        lines.append(f"  Next-step gate: {summary['gate']['decision']} (not a classifier result)")
    for row in report.get("model_training", []):
        summary = row.get("summary", {})
        metric = summary.get("selection_metric", "validation_video_f1")
        value = summary.get(
            "best_selection_value",
            summary.get("best_validation_video_f1"),
        )
        score = f"{float(value):.2%}" if value is not None else "n/a"
        lines.append(
            f"  {row['name']:<34} {row['action']!s:<8} "
            f"best epoch {summary.get('best_epoch', 'n/a')}  |  {metric} {score}"
        )
    for row in report.get("decisions", []):
        current = sum(action in {"ran", "skipped"} for action in row.get("stages", {}).values())
        lines.append(
            f"  {row['name']:<34} {row['action']!s:<8} decisions {current}/3 complete"
        )
    for row in report.get("local_inference", []):
        lines.append(f"  {row['name']} | {row['action']} | local inference 3/3 complete")
        lines.append(_local_readable_summary(row))
    lines.extend((f"Report: {report_path}", "#" * 72))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.workflow")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry cached preprocessing failures while reusing completed work",
    )
    parser.add_argument("--json", action="store_true", help="Also print the full final report")
    args = parser.parse_args()
    report = run_workflow(args.config, retry_failed=args.retry_failed)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            format_workflow_summary(
                report,
                load_workflow_settings(args.config).report_path,
            )
        )


if __name__ == "__main__":
    main()
