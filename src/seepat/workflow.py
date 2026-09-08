from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from seepat.artifacts import atomic_write_json, read_csv_rows, stable_id
from seepat.config import PipelineSettings, load_pipeline_settings
from seepat.pipeline import PIPELINE_VERSION, run_pipeline
from seepat.preprocessing.contract import audit_preprocessing_contract
from seepat.training.manifest import prepare_training_manifests

WORKFLOW_VERSION = "local-pipeline-v2"
SUPPORTED_TRAINING_MODELS = {"swin3d_b", "efficientnet_v2_s_tempcnn"}


@dataclass(frozen=True)
class WorkflowJob:
    name: str
    pipeline_config: Path
    manifest_output_dir: Path


@dataclass(frozen=True)
class ModelTrainingJob:
    name: str
    train_manifest: Path
    validation_manifest: Path
    output_dir: Path
    project_root: Path
    device: str
    pretrained: bool
    options: dict[str, object]
    model: str = "swin3d_b"


@dataclass(frozen=True)
class NumericalCalibrationJob:
    name: str
    train_manifest: Path
    score_manifests: tuple[tuple[str, Path], ...]
    output_dir: Path
    min_subject_reference_frames: int = 16
    isolation_trees: int = 100
    random_seed: int = 20260908


@dataclass(frozen=True)
class WorkflowSettings:
    jobs: tuple[WorkflowJob, ...]
    model_training_jobs: tuple[ModelTrainingJob, ...]
    report_path: Path
    numerical_calibration_jobs: tuple[NumericalCalibrationJob, ...] = ()


def _require_mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a YAML mapping")
    return value


def load_workflow_settings(path: Path) -> WorkflowSettings:
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    config = _require_mapping(raw, "Workflow configuration")

    raw_jobs = config.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("Workflow configuration must contain at least one job")

    jobs: list[WorkflowJob] = []
    names: set[str] = set()
    for index, raw_job in enumerate(raw_jobs, start=1):
        job = _require_mapping(raw_job, f"Workflow job {index}")
        name = str(job.get("name", "")).strip()
        if not name:
            raise ValueError(f"Workflow job {index} has no name")
        if name in names:
            raise ValueError(f"Duplicate workflow job name: {name}")
        names.add(name)
        try:
            pipeline_config = Path(str(job["pipeline_config"]))
            manifest_output_dir = Path(str(job["manifest_output_dir"]))
        except KeyError as error:
            raise ValueError(
                f"Workflow job {name!r} is missing {error.args[0]!r}"
            ) from error
        jobs.append(
            WorkflowJob(
                name=name,
                pipeline_config=pipeline_config,
                manifest_output_dir=manifest_output_dir,
            )
        )

    raw_training = config.get("model_training")
    if raw_training is None:
        raw_training_jobs: list[object] = []
    elif isinstance(raw_training, list):
        raw_training_jobs = raw_training
    else:
        raw_training_jobs = [raw_training]

    model_training_jobs: list[ModelTrainingJob] = []
    model_names: set[str] = set()
    model_output_dirs: set[Path] = set()
    for index, raw_model_job in enumerate(raw_training_jobs, start=1):
        training = _require_mapping(
            raw_model_job,
            f"Model training configuration {index}",
        )
        options = _require_mapping(training.get("options", {}), "Model training options")
        pretrained = training.get("pretrained", True)
        if not isinstance(pretrained, bool):
            raise TypeError("Model training pretrained value must be true or false")
        device = str(training.get("device", "auto"))
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Model training device must be auto, cpu, or cuda")
        model = str(training.get("model", "swin3d_b")).strip()
        if model not in SUPPORTED_TRAINING_MODELS:
            raise ValueError(f"Unsupported model training model: {model!r}")
        try:
            model_training = ModelTrainingJob(
                name=str(training.get("name", "model-training")).strip(),
                train_manifest=Path(str(training["train_manifest"])),
                validation_manifest=Path(str(training["validation_manifest"])),
                output_dir=Path(str(training["output_dir"])),
                project_root=Path(str(training.get("project_root", "."))),
                device=device,
                pretrained=pretrained,
                options={str(key): value for key, value in options.items()},
                model=model,
            )
        except KeyError as error:
            raise ValueError(
                f"Model training configuration is missing {error.args[0]!r}"
            ) from error
        if not model_training.name:
            raise ValueError("Model training configuration has no name")
        if model_training.name in model_names:
            raise ValueError(f"Duplicate model training job name: {model_training.name}")
        if model_training.output_dir in model_output_dirs:
            raise ValueError(
                f"Duplicate model training output directory: {model_training.output_dir}"
            )
        model_names.add(model_training.name)
        model_output_dirs.add(model_training.output_dir)
        model_training_jobs.append(model_training)

    raw_calibration = config.get("numerical_calibration")
    if raw_calibration is None:
        raw_calibration_jobs: list[object] = []
    elif isinstance(raw_calibration, list):
        raw_calibration_jobs = raw_calibration
    else:
        raw_calibration_jobs = [raw_calibration]
    numerical_calibration_jobs: list[NumericalCalibrationJob] = []
    calibration_names: set[str] = set()
    calibration_output_dirs: set[Path] = set()
    for index, raw_calibration_job in enumerate(raw_calibration_jobs, start=1):
        calibration = _require_mapping(
            raw_calibration_job, f"Numerical calibration configuration {index}"
        )
        raw_scoring = _require_mapping(
            calibration.get("score_manifests", {}),
            "Numerical calibration score_manifests",
        )
        score_manifests = tuple(
            (str(name).strip(), Path(str(value)))
            for name, value in sorted(
                raw_scoring.items(), key=lambda item: str(item[0])
            )
        )
        if not score_manifests or any(not name for name, _ in score_manifests):
            raise ValueError("Numerical calibration requires named score_manifests")
        if len({name for name, _ in score_manifests}) != len(score_manifests):
            raise ValueError("Numerical calibration score_manifests names must be unique")
        try:
            calibration_job = NumericalCalibrationJob(
                name=str(calibration.get("name", "numerical-calibration")).strip(),
                train_manifest=Path(str(calibration["train_manifest"])),
                score_manifests=score_manifests,
                output_dir=Path(str(calibration["output_dir"])),
                min_subject_reference_frames=int(
                    calibration.get("min_subject_reference_frames", 16)
                ),
                isolation_trees=int(calibration.get("isolation_trees", 100)),
                random_seed=int(calibration.get("random_seed", 20260908)),
            )
        except KeyError as error:
            raise ValueError(
                f"Numerical calibration configuration is missing {error.args[0]!r}"
            ) from error
        if not calibration_job.name:
            raise ValueError("Numerical calibration configuration has no name")
        if calibration_job.name in calibration_names:
            raise ValueError(
                f"Duplicate numerical calibration job name: {calibration_job.name}"
            )
        if calibration_job.output_dir in calibration_output_dirs:
            raise ValueError(
                f"Duplicate numerical calibration output directory: {calibration_job.output_dir}"
            )
        calibration_names.add(calibration_job.name)
        calibration_output_dirs.add(calibration_job.output_dir)
        numerical_calibration_jobs.append(calibration_job)

    report_path = Path(str(config.get("report", "outputs/workflow_summary.json")))
    return WorkflowSettings(
        jobs=tuple(jobs),
        model_training_jobs=tuple(model_training_jobs),
        report_path=report_path,
        numerical_calibration_jobs=tuple(numerical_calibration_jobs),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_outputs_are_current(
    settings: PipelineSettings,
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
        for row in trace_rows:
            trace_path = Path(row["vild_trace_path"])
            if not trace_path.is_file() or row["vild_trace_sha256"] != _sha256(
                trace_path
            ):
                return False
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
) -> dict[str, object]:
    settings = load_pipeline_settings(job.pipeline_config, PIPELINE_VERSION)
    preprocessing_current = preprocessing_outputs_are_current(settings)

    if preprocessing_current and not retry_failed:
        print(f"[{job.name}] preprocessing: skipped (artifacts are current)")
        pipeline_summary = _read_json(
            settings.preprocessing.output_dir / "run_summary.json"
        )
        preprocessing_action = "skipped"
    else:
        reason = "retrying cached failures" if retry_failed else "artifacts are missing or stale"
        print(f"[{job.name}] preprocessing: running ({reason})")
        pipeline_summary = run_pipeline(
            job.pipeline_config,
            retry_failed=retry_failed,
        )
        preprocessing_action = "ran"

    contract_summary = None
    if settings.preprocessing.vild_trace.enabled:
        print(f"[{job.name}] preprocessing contract: auditing")
        contract_summary = audit_preprocessing_contract(
            settings.preprocessing.output_dir
        )

    manifest_current = training_outputs_are_current(settings, job.manifest_output_dir)
    if manifest_current:
        print(f"[{job.name}] training manifest: skipped (artifacts are current)")
        manifest_summary = _read_json(job.manifest_output_dir / "summary.json")
        manifest_action = "skipped"
    else:
        print(f"[{job.name}] training manifest: building")
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
    from seepat.training.train import TrainingOptions

    options = TrainingOptions(**job.options)
    options.validate()
    return options


def _model_training_configuration_matches(
    job: ModelTrainingJob,
    run_record: dict[str, Any],
) -> bool:
    from seepat.training.train import (
        SWIN_BASE_MODEL,
        model_contract_name,
        training_version_for_model,
    )

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
        recorded_options.pop("epochs", None)
        expected_options.pop("epochs")
        if job.device != "auto" and run_record.get("device") != job.device:
            return False
        return (
            run_record.get("run_type") == expected_type
            and recorded_options == expected_options
            and contract.get("training_version") == training_version_for_model(job.model)
            and contract.get("model") == model_contract_name(job.model)
            and contract.get("model_name", SWIN_BASE_MODEL) == job.model
            and contract.get("pretrained") == job.pretrained
            and contract.get("train_manifest_sha256") == _sha256(job.train_manifest)
            and contract.get("validation_manifest_sha256")
            == _sha256(job.validation_manifest)
        )
    except (OSError, TypeError, ValueError):
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
    )


def run_model_training_job(job: ModelTrainingJob) -> dict[str, object]:
    if model_training_outputs_are_current(job):
        print(f"[{job.name}] model training: skipped (artifacts are current)")
        return {
            "name": job.name,
            "action": "skipped",
            "summary": _read_json(job.output_dir / "run.json"),
        }

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
    summary = _run_model_training(job, resume_from)
    return {"name": job.name, "action": action, "summary": summary}


def numerical_calibration_outputs_are_current(job: NumericalCalibrationJob) -> bool:
    required = (
        job.train_manifest,
        job.output_dir / "summary.json",
        job.output_dir / "calibration.json",
        job.output_dir / "isolation_forests.joblib",
    )
    if not all(path.is_file() for path in required) or not all(
        path.is_file() for _, path in job.score_manifests
    ):
        return False
    try:
        summary = _read_json(job.output_dir / "summary.json")
        if summary.get("train_manifest_sha256") != _sha256(job.train_manifest):
            return False
        if summary.get("calibration_sha256") != _sha256(
            job.output_dir / "calibration.json"
        ):
            return False
        calibration = _read_json(job.output_dir / "calibration.json")
        if calibration.get("isolation_forest_models_sha256") != _sha256(
            job.output_dir / "isolation_forests.joblib"
        ):
            return False
        recorded_options = _require_mapping(
            summary.get("options"), "Calibration options"
        )
        if recorded_options != {
            "min_subject_reference_frames": job.min_subject_reference_frames,
            "isolation_trees": job.isolation_trees,
            "random_seed": job.random_seed,
        }:
            return False
        recorded_inputs = _require_mapping(
            summary.get("score_manifest_sha256"), "Calibration score input hashes"
        )
        recorded_outputs = _require_mapping(
            summary.get("scored_manifest_sha256"), "Calibrated manifest hashes"
        )
        recorded_paths = _require_mapping(
            summary.get("scored_manifests"), "Calibrated manifest paths"
        )
        for name, input_path in job.score_manifests:
            output_path = Path(str(recorded_paths.get(name, "")))
            if (
                recorded_inputs.get(name) != _sha256(input_path)
                or not output_path.is_file()
                or recorded_outputs.get(name) != _sha256(output_path)
            ):
                return False
        return summary.get("reference_frames", 0) >= 2
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_numerical_calibration_job(job: NumericalCalibrationJob) -> dict[str, object]:
    if numerical_calibration_outputs_are_current(job):
        print(f"[{job.name}] numerical calibration: skipped (artifacts are current)")
        return {
            "name": job.name,
            "action": "skipped",
            "summary": _read_json(job.output_dir / "summary.json"),
        }
    from seepat.training.calibration import CalibrationOptions, fit_and_score_calibration

    print(f"[{job.name}] numerical calibration: fitting Train-only evidence")
    summary = fit_and_score_calibration(
        train_manifest=job.train_manifest,
        score_manifests=dict(job.score_manifests),
        output_dir=job.output_dir,
        options=CalibrationOptions(
            min_subject_reference_frames=job.min_subject_reference_frames,
            isolation_trees=job.isolation_trees,
            random_seed=job.random_seed,
        ),
    )
    return {"name": job.name, "action": "ran", "summary": summary}


def run_workflow(
    config_path: Path,
    retry_failed: bool = False,
) -> dict[str, object]:
    settings = load_workflow_settings(config_path)
    jobs = [run_workflow_job(job, retry_failed=retry_failed) for job in settings.jobs]
    report: dict[str, object] = {
        "workflow_version": WORKFLOW_VERSION,
        "config": config_path.as_posix(),
        "jobs": jobs,
    }
    if settings.numerical_calibration_jobs:
        report["numerical_calibration"] = [
            run_numerical_calibration_job(job)
            for job in settings.numerical_calibration_jobs
        ]
    if settings.model_training_jobs:
        report["model_training"] = [
            run_model_training_job(job) for job in settings.model_training_jobs
        ]
    atomic_write_json(settings.report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.workflow")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry cached preprocessing failures while reusing completed work",
    )
    args = parser.parse_args()
    report = run_workflow(args.config, retry_failed=args.retry_failed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
