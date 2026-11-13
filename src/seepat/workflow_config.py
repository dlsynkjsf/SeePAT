"""Workflow job definitions and configuration parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from seepat.coverage import CoveragePilotJob
from seepat.decision import DecisionJob
from seepat.live_progress import live_progress_path
from seepat.preprocessing.augmentation import TraceAugmentationJob
from seepat.training.study import StudyJob

SUPPORTED_TRAINING_MODELS = {
    "swin3d_b",
    "efficientnet_v2_s_tempcnn",
    "swin3d_b_vild_fusion",
    "efficientnet_v2_s_only",
    "tempcnn_gray16",
    "swin3d_b_visual_fusion",
}


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
    readiness_dir: Path | None = None
    initialize_from: Path | None = None
    input_profile: dict[str, str] | None = None


@dataclass(frozen=True)
class NumericalCalibrationJob:
    name: str
    train_manifest: Path | None
    score_manifests: tuple[tuple[str, Path], ...]
    output_dir: Path
    min_video_reference_frames: int = 16
    isolation_trees: int = 100
    random_seed: int = 20260908
    calibration: Path | None = None


@dataclass(frozen=True)
class WorkflowSettings:
    jobs: tuple[WorkflowJob, ...]
    model_training_jobs: tuple[ModelTrainingJob, ...]
    report_path: Path
    numerical_calibration_jobs: tuple[NumericalCalibrationJob, ...] = ()
    trace_augmentation_jobs: tuple[TraceAugmentationJob, ...] = ()
    decision_jobs: tuple[DecisionJob, ...] = ()
    model_training_config: Path | None = None
    study_job: StudyJob | None = None
    coverage_pilot_jobs: tuple[CoveragePilotJob, ...] = ()
    local_inference_jobs: tuple[Any, ...] = ()


def _require_mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a YAML mapping")
    return value


def _local_inference_api():
    from seepat.inference import local_inference_workflow

    return local_inference_workflow


def load_workflow_settings(path: Path) -> WorkflowSettings:
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    config = _require_mapping(raw, "Workflow configuration")

    raw_jobs = config.get("jobs", [])
    if not isinstance(raw_jobs, list):
        raise TypeError("Workflow jobs must be a list")

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

    training_config = None
    training_source = config
    if config.get("model_training_config") is not None:
        training_config = path.parent / str(config["model_training_config"])
        with training_config.open(encoding="utf-8") as stream:
            training_source = _require_mapping(yaml.safe_load(stream), "Model training profile")
        if training_source.get("model_training_config") is not None:
            raise ValueError("Nested model training profiles are not supported")
        if not (training_source.get("model_training") or training_source.get("ablation_study")):
            raise ValueError("Model training profile must contain model_training jobs or an ablation_study")
    raw_training = training_source.get("model_training")
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
                readiness_dir=Path(training["readiness_dir"]) if training.get("readiness_dir") else None,
                initialize_from=(
                    Path(training["initialize_from"])
                    if training.get("initialize_from")
                    else None
                ),
                input_profile=(dict(_require_mapping(training["input_profile"], "Matched input profile"))
                               if training.get("input_profile") is not None else None),
            )
        except KeyError as error:
            raise ValueError(
                f"Model training configuration is missing {error.args[0]!r}"
            ) from error
        if not model_training.name:
            raise ValueError("Model training configuration has no name")
        if model_training.readiness_dir is not None and (
            model_training.readiness_dir.resolve() == model_training.output_dir.resolve()
            or options.get("max_train_batches") is not None
        ):
            raise ValueError("Experiment readiness must use a separate output and an uncapped training job")
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
                train_manifest=(
                    Path(str(calibration["train_manifest"]))
                    if calibration.get("train_manifest")
                    else None
                ),
                score_manifests=score_manifests,
                output_dir=Path(str(calibration["output_dir"])),
                min_video_reference_frames=int(
                    calibration.get("min_video_reference_frames", 16)
                ),
                isolation_trees=int(calibration.get("isolation_trees", 100)),
                random_seed=int(calibration.get("random_seed", 20260908)),
                calibration=(
                    Path(str(calibration["calibration"]))
                    if calibration.get("calibration")
                    else None
                ),
            )
        except KeyError as error:
            raise ValueError(
                f"Numerical calibration configuration is missing {error.args[0]!r}"
            ) from error
        if not calibration_job.name:
            raise ValueError("Numerical calibration configuration has no name")
        if (calibration_job.train_manifest is None) == (calibration_job.calibration is None):
            raise ValueError(
                "Numerical calibration requires exactly one of train_manifest or calibration"
            )
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

    augmentation_jobs = []
    for raw_job in config.get("trace_augmentation", []):
        job = _require_mapping(raw_job, "Trace augmentation job")
        augmentation = TraceAugmentationJob(
            name=str(job["name"]), manifest=Path(job["manifest"]),
            extracted_root=Path(job["extracted_root"]), output_dir=Path(job["output_dir"]),
            max_videos=job.get("max_videos"),
            normalized_tolerance=float(job.get("normalized_tolerance", 1e-5)),
            timestamp_tolerance_s=float(job.get("timestamp_tolerance_s", 1e-6)),
        )
        augmentation.validate()
        if any(old.name == augmentation.name or old.output_dir == augmentation.output_dir
               for old in augmentation_jobs):
            raise ValueError("Duplicate trace augmentation name or output directory")
        augmentation_jobs.append(augmentation)
    decision_jobs = []
    for raw_job in config.get("decisions", []):
        item = _require_mapping(raw_job, "Validation decision job")
        if item.get("split", "val") != "val" or item.get("dataset_source", "AV-Deepfake1M++") != "AV-Deepfake1M++":
            raise ValueError("Decision workflow is restricted to AV++ Validation; external/test remains locked")
        decision = DecisionJob(
            name=str(item["name"]), validation_manifest=Path(item["validation_manifest"]),
            output_dir=Path(item["output_dir"]),
            checkpoint=Path(item["checkpoint"]) if item.get("checkpoint") else None,
            source_manifest=Path(item["source_manifest"]) if item.get("source_manifest") else None,
            project_root=Path(item.get("project_root", ".")), device=str(item.get("device", "auto")),
            batch_size=int(item.get("batch_size", 1)),
            threshold_artifact=(
                Path(item["threshold_artifact"])
                if item.get("threshold_artifact")
                else None
            ),
            input_profile=(dict(_require_mapping(item["input_profile"], "Decision matched profile"))
                           if item.get("input_profile") is not None else None),
        )
        if not decision.name or decision.batch_size < 1 or any(
            old.name == decision.name or old.output_dir == decision.output_dir for old in decision_jobs
        ):
            raise ValueError("Decision jobs require unique names/outputs and positive batch size")
        decision_jobs.append(decision)
    study_job = None
    if training_source.get("ablation_study") is not None:
        item = dict(_require_mapping(training_source["ablation_study"], "Ablation study"))
        for key in ("train_manifest", "validation_manifest", "output_dir", "project_root"):
            if key in item:
                item[key] = Path(item[key])
        if "models" in item:
            item["models"] = tuple(item["models"])
        study_job = StudyJob(**item)
        study_job.validate()
    coverage_jobs = []
    raw_coverage = config.get("coverage_pilot", [])
    if isinstance(raw_coverage, dict):
        raw_coverage = [raw_coverage]
    if not isinstance(raw_coverage, list):
        raise TypeError("coverage_pilot must be a mapping or list")
    for raw_job in raw_coverage:
        item = dict(_require_mapping(raw_job, "Coverage pilot"))
        for key in ("source_manifest", "video_manifest", "event_manifest", "output_dir"):
            item[key] = Path(item[key])
        job = CoveragePilotJob(**item)
        job.validate()
        if any(j.name == job.name or j.output_dir.resolve() == job.output_dir.resolve()
               for j in coverage_jobs):
            raise ValueError("Duplicate coverage pilot name/output")
        coverage_jobs.append(job)
    local_jobs = _local_inference_api().load_local_jobs(config) if 'local_inference' in config else ()
    if not (local_jobs or jobs or augmentation_jobs or numerical_calibration_jobs or model_training_jobs
            or decision_jobs or study_job or coverage_jobs):
        raise ValueError("Workflow configuration must contain at least one job")
    report_path = Path(str(config.get("report", "outputs/workflow_summary.json")))
    for job in local_jobs:
        protected = {job.freeze.resolve(), job.raw_inputs.resolve(), path.resolve()}
        for target in (report_path.resolve(), live_progress_path(report_path).resolve()):
            if target in protected or target.is_relative_to(job.output_dir.resolve()):
                raise ValueError("Workflow report must stay outside local outputs and protected inputs")
    return WorkflowSettings(
        jobs=tuple(jobs),
        model_training_jobs=tuple(model_training_jobs),
        report_path=report_path,
        numerical_calibration_jobs=tuple(numerical_calibration_jobs),
        trace_augmentation_jobs=tuple(augmentation_jobs),
        decision_jobs=tuple(decision_jobs),
        model_training_config=training_config,
        study_job=study_job,
        coverage_pilot_jobs=tuple(coverage_jobs),
        local_inference_jobs=local_jobs,
    )
