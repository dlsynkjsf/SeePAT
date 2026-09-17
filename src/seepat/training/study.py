"""Opt-in source-group folds for the paper's ablations, using the existing trainer.

Official Validation remains development data. Held-out Train folds are scored
only after checkpoint/threshold selection; external data is never accepted here.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256, read_csv_rows
from seepat.live_progress import ProgressCallback

STUDY_VERSION = "source-group-ablation-v1"
STUDY_MODELS = (
    "efficientnet_v2_s_only", "tempcnn_gray16", "efficientnet_v2_s_tempcnn",
    "swin3d_b", "swin3d_b_vild_fusion", "swin3d_b_visual_fusion",
)


@dataclass(frozen=True)
class StudyJob:
    name: str
    train_manifest: Path
    validation_manifest: Path
    output_dir: Path
    phase: str = "prepare"
    folds: int = 5
    seed: int = 20260916
    models: tuple[str, ...] = STUDY_MODELS[:3]
    readiness_epochs: int = 1
    epochs: int = 10
    device: str = "cuda"
    project_root: Path = Path(".")

    def validate(self) -> None:
        if self.phase not in {"prepare", "audit", "readiness", "train", "evaluate", "calibration"}:
            raise ValueError("Study phase must be prepare, audit, readiness, train, evaluate or calibration")
        if not self.name or self.folds < 2 or self.epochs < 1 or self.readiness_epochs not in {1, 2}:
            raise ValueError("Study needs a name, at least two folds, and valid epoch counts")
        if not self.models or len(set(self.models)) != len(self.models) or set(self.models) - set(STUDY_MODELS):
            raise ValueError("Study requires unique supported model names")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("Study device must be cpu, cuda or auto")
        for source in (self.train_manifest, self.validation_manifest):
            if source.resolve().is_relative_to(self.output_dir.resolve()):
                raise ValueError("Study output must be separate from its source manifests")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_rows(rows: list[dict], split: str) -> None:
    if not rows or any(
        row.get("dataset_split") != split
        or row.get("dataset_source", "AV-Deepfake1M++") != "AV-Deepfake1M++"
        for row in rows
    ):
        raise ValueError("Studies accept only declared AV++ Train/Validation rows; external stays locked")
    ids = [row.get("event_id") for row in rows]
    if not all(ids) or len(set(ids)) != len(ids) or any(not row.get("source_group") for row in rows):
        raise ValueError("Study event IDs must be unique and source groups nonempty")
    if {row.get("class_id") for row in rows} != {"0", "1"}:
        raise ValueError("Every study partition must contain both event classes")
    videos = {}
    for row in rows:
        identity = (row["source_group"], row.get("manipulation_modality"))
        if videos.setdefault(row["video_id"], identity) != identity:
            raise ValueError("A video cannot have conflicting source groups or video labels")


def _disjoint(first: list[dict], second: list[dict]) -> None:
    for key in ("event_id", "video_id", "source_group"):
        if {r[key] for r in first} & {r[key] for r in second}:
            raise ValueError(f"Study partitions overlap in {key}")


def source_group_folds(train: list[dict], validation: list[dict], folds: int, seed: int):
    from sklearn.model_selection import StratifiedGroupKFold

    _validate_rows(train, "train")
    _validate_rows(validation, "val")
    _disjoint(train, validation)
    # Sort first so CSV ordering cannot change which source group enters a fold.
    ordered = sorted(train, key=lambda row: row["event_id"])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = []
    for fitting, heldout in splitter.split(
        ordered, [int(r["class_id"]) for r in ordered], [r["source_group"] for r in ordered],
    ):
        pair = ([ordered[i] for i in fitting], [ordered[i] for i in heldout])
        for rows in pair:
            _validate_rows(rows, "train")
        _disjoint(*pair)
        result.append(pair)
    return result


def _recipe(job: StudyJob) -> dict:
    return {"version": STUDY_VERSION, "folds": job.folds, "seed": job.seed,
            "train_manifest": job.train_manifest.as_posix(),
            "validation_manifest": job.validation_manifest.as_posix()}


def prepare_study(job: StudyJob, progress: ProgressCallback | None = None) -> dict:
    import sklearn

    from seepat.training.calibration import (
        DEFAULT_CALIBRATION_OPTIONS,
        calibration_outputs_are_current,
        fit_and_score_calibration,
    )

    job.validate()
    recipe = {**_recipe(job), "sklearn_version": sklearn.__version__,
              "calibration_options": asdict(DEFAULT_CALIBRATION_OPTIONS), "source_hashes": {
        path.as_posix(): file_sha256(path) for path in (job.train_manifest, job.validation_manifest)
    }}
    record_path = job.output_dir / "study.json"
    if record_path.exists() and _read(record_path)["recipe"] != recipe:
        raise ValueError("Existing study uses different inputs/settings; choose a new output directory")
    train, val = read_csv_rows(job.train_manifest), read_csv_rows(job.validation_manifest)
    if any(row.get("calibration_version") for row in train + val):
        raise ValueError("Prepare folds from augmented manifests, not all-Train calibrated exports")
    pairs = source_group_folds(train, val, job.folds, job.seed)
    records = []
    for number, (fitting, heldout) in enumerate(pairs, 1):
        directory = job.output_dir / f"fold_{number}"
        inputs = {name: directory / "inputs" / f"{name}.csv" for name in ("train", "val", "heldout")}
        for name, rows in (("train", fitting), ("val", val), ("heldout", heldout)):
            atomic_write_csv(inputs[name], rows)
        calibrated = directory / "calibrated"
        if progress:
            progress("prepare fold calibration", number - 1, job.folds, f"fold {number}")
        fold_progress = None if progress is None else (
            lambda phase, done, total, item, number=number: progress(f"fold {number}: {phase}", done, total, item))
        if not calibration_outputs_are_current(inputs["train"], inputs, calibrated, DEFAULT_CALIBRATION_OPTIONS):
            fit_and_score_calibration(inputs["train"], inputs, calibrated, progress=fold_progress)
        paths = {name: calibrated / f"events_{name}_calibrated.csv" for name in inputs}
        files = [*inputs.values(), *paths.values(), calibrated / "summary.json", calibrated / "calibration.json"]
        records.append({"fold": number, "inputs": {k: v.as_posix() for k, v in inputs.items()},
                        "manifests": {k: v.as_posix() for k, v in paths.items()},
                        "hashes": {p.as_posix(): file_sha256(p) for p in files}})
    record = {"recipe": recipe, "folds": records,
              "protocol": "Train source-group folds; official Validation selects checkpoints and thresholds",
              "fit_scope": "Each fold fits population regression/phoneme statistics on its fitting Train groups only"}
    atomic_write_json(record_path, record)
    return record


def load_study(job: StudyJob) -> dict:
    """Verify portable exports, including population fit scope, without raw videos."""
    record = _read(job.output_dir / "study.json")
    if any(record["recipe"].get(k) != v for k, v in _recipe(job).items()):
        raise ValueError("Study settings differ from prepared folds")
    if any(Path(path).exists() and file_sha256(Path(path)) != digest
           for path, digest in record["recipe"]["source_hashes"].items()):
        raise ValueError("Original study input changed; use a new study directory")
    if len(record["folds"]) != job.folds or [r["fold"] for r in record["folds"]] != list(range(1, job.folds + 1)):
        raise ValueError("Study fold inventory is incomplete")
    heldout_ids, all_train_ids, validation_hash = set(), None, None
    for fold in record["folds"]:
        if any(file_sha256(Path(path)) != digest for path, digest in fold["hashes"].items()):
            raise ValueError("Study export hash mismatch; rerun preparation or restore the transfer")
        rows = {key: read_csv_rows(Path(path)) for key, path in fold["inputs"].items()}
        for name, split in (("train", "train"), ("heldout", "train"), ("val", "val")):
            _validate_rows(rows[name], split)
        _disjoint(rows["train"], rows["heldout"])
        _disjoint(rows["train"] + rows["heldout"], rows["val"])
        current = {r["event_id"] for r in rows["train"] + rows["heldout"]}
        held = {r["event_id"] for r in rows["heldout"]}
        val_hash = file_sha256(Path(fold["inputs"]["val"]))
        if heldout_ids & held or (all_train_ids is not None and all_train_ids != current):
            raise ValueError("Held-out folds must partition one fixed Train inventory exactly once")
        if validation_hash is not None and validation_hash != val_hash:
            raise ValueError("All folds must share the official Validation inventory")
        heldout_ids.update(held)
        all_train_ids, validation_hash = current, val_hash
        calibrated = Path(fold["manifests"]["train"]).parent
        population = _read(calibrated / "calibration.json")
        if population["train_manifest_sha256"] != file_sha256(Path(fold["inputs"]["train"])):
            raise ValueError("Fold calibration was not fitted on this fold's Train subset")
        for name, path in fold["manifests"].items():
            exported = read_csv_rows(Path(path))
            # Calibration adds columns; no source identity, label or clip may change.
            originals = {r["event_id"]: r for r in rows[name]}
            if len(exported) != len(originals) or {r["event_id"] for r in exported} != set(originals):
                raise ValueError("Calibrated fold changed the event inventory")
            if any(any(row.get(k) != v for k, v in originals[row["event_id"]].items()) for row in exported):
                raise ValueError("Calibrated fold changed an original input column")
    if heldout_ids != all_train_ids:
        raise ValueError("Held-out folds do not cover Train exactly once")
    return record


def model_job(job: StudyJob, fold: dict, model: str, *, readiness: bool):
    from seepat.workflow import ModelTrainingJob

    directory = job.output_dir / f"fold_{fold['fold']}"
    ready_dir = directory / "readiness" / model
    return ModelTrainingJob(
        name=f"fold-{fold['fold']}-{model}", model=model,
        train_manifest=Path(fold["manifests"]["train"]),
        validation_manifest=Path(fold["manifests"]["val"]),
        output_dir=ready_dir if readiness else directory / "training" / model,
        project_root=job.project_root, device=job.device, pretrained=model != "tempcnn_gray16",
        readiness_dir=None if readiness else ready_dir,
        options={"epochs": job.readiness_epochs if readiness else job.epochs,
                 "batch_size": 1, "workers": 0, "learning_rate": 1e-4, "weight_decay": .01,
                 "gradient_accumulation_steps": 4, "sequence_length": 16, "image_size": 224,
                 "seed": job.seed, "amp": False, "freeze_backbone": model != "tempcnn_gray16",
                 "class_weighting": "balanced_global", "early_stopping_patience": 3,
                 "max_train_batches": 16 if readiness else None,
                 "max_validation_batches": 8 if readiness else None},
    )


def run_study_job(job: StudyJob, progress: ProgressCallback | None = None) -> dict:
    job.validate()
    if job.phase == "prepare":
        prepare_study(job, progress)
        return {"name": job.name, "action": "prepared", "folds": job.folds}
    record = load_study(job)
    from seepat.training.study_evaluation import (
        evaluate_calibration,
        evaluate_model,
        write_comparison,
    )
    from seepat.workflow import run_model_training_job

    results = []
    for fold in record["folds"]:
        if job.phase == "calibration":
            if progress:
                progress("compare static/dynamic calibration", fold["fold"] - 1, job.folds, f"fold {fold['fold']}")
            results.extend(evaluate_calibration(job, fold))
            continue
        for model in job.models:
            training = model_job(job, fold, model, readiness=job.phase == "readiness")
            if progress:
                progress(f"study {job.phase}", len(results), job.folds * len(job.models), training.name)
            model_progress = None if progress is None else (
                lambda phase, done, total, item, name=training.name: progress(f"{name}: {phase}", done, total, item))
            if job.phase == "audit":
                from seepat.training.readiness import audit_inputs

                result = audit_inputs(training)
            elif job.phase in {"readiness", "train"}:
                result = run_model_training_job(training, model_progress)
            else:
                result = evaluate_model(job, fold, training, model_progress)
            results.append(result)
    if job.phase in {"evaluate", "calibration"}:
        write_comparison(job.output_dir / f"{job.phase}_comparison", results)
        files = [job.output_dir / f"{job.phase}_comparison.{ext}" for ext in ("csv", "json")]
        for result in results:
            files.extend(Path(path) for path in result.get("artifacts", {}))
            if job.phase == "evaluate":
                training = model_job(job, record["folds"][result["fold"] - 1], result["model"], readiness=False)
                files.extend((training.output_dir / "run.json", Path(_read(training.output_dir / "run.json")["best_checkpoint"]),
                              training.output_dir / "fold_evaluation" / "metrics.json",
                              training.output_dir / "fold_evaluation" / "metrics.record.json"))
            else:
                files.append(job.output_dir / f"fold_{result['fold']}" / "calibration_comparison.json")
        atomic_write_json(job.output_dir / f"{job.phase}_complete.json", {
            "settings": json.loads(json.dumps(asdict(job), default=str)),
            "study_sha256": file_sha256(job.output_dir / "study.json"),
            "artifacts": {p.as_posix(): file_sha256(p) for p in files},
        })
    return {"name": job.name, "action": job.phase, "results": results}


def study_phase_is_current(job: StudyJob) -> bool:
    """Read-only integrity check for --verify; normal watches use live progress."""
    try:
        record = load_study(job)
        if job.phase == "prepare":
            return all(file_sha256(Path(p)) == digest for p, digest in record["recipe"]["source_hashes"].items())
        if job.phase in {"readiness", "train"}:
            from seepat.workflow import model_training_outputs_are_current

            return all(model_training_outputs_are_current(model_job(job, fold, model, readiness=job.phase == "readiness"))
                       for fold in record["folds"] for model in job.models)
        if job.phase == "audit":
            from seepat.training.readiness import audit_inputs

            for fold in record["folds"]:
                for model in job.models:
                    audit_inputs(model_job(job, fold, model, readiness=False))
            return True
        complete = _read(job.output_dir / f"{job.phase}_complete.json")
        return (complete["settings"] == json.loads(json.dumps(asdict(job), default=str))
                and complete["study_sha256"] == file_sha256(job.output_dir / "study.json")
                and all(file_sha256(Path(p)) == digest for p, digest in complete["artifacts"].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False
