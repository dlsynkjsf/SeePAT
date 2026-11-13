"""Inference from frozen models and label-free prepared inputs."""

import argparse
import json
import math
import pickle
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch

from seepat.artifacts import file_sha256, read_csv_rows
from seepat.evidence import (
    EVIDENCE_VERSION,
    FUSION_EVIDENCE_FIELDS,
    calibrated_manifest_contract,
    evidence_feature_values,
)
from seepat.training.dataset import decode_clip_frames, sequence_frame_indices
from seepat.training.train import FUSION_MODEL, build_event_classifier, training_version_for_model
from seepat.verdict import _load_threshold, _score_events, _validate_supervision

from .expanded_context import REPRESENTATION as EXPANDED_REPRESENTATION
from .expanded_context import saved_threshold, validate_supervision

VERSION = "frozen-inference-v2"


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Expected an artifact object")
    return value


def _bound(item, bindings):
    path = Path(item["path"]).resolve()
    if not isinstance(item.get("sha256"), str) or file_sha256(path) != item["sha256"]:
        raise ValueError("Frozen artifact hash mismatch")
    bindings.append({"path": str(path), "sha256": item["sha256"]})
    return path


def _unchanged(bindings):
    for item in bindings:
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Input changed during inference")


def _runtime_hashes(matched=False):
    source = Path(__file__).resolve().parents[2]
    return {"frozen_inference.py": file_sha256(Path(__file__)),
            **({"local_matched_inputs.py": file_sha256(Path(__file__).with_name("local_matched_inputs.py"))}
               if matched else {}),
            **{path.relative_to(source).as_posix(): file_sha256(path) for path in sorted(source.rglob("*.py"))}}


def validate_freeze(freeze_path: Path):
    """Shared trusted-model/config gate, before raw preparation or clip scoring."""
    bindings = []
    _bound({"path": str(freeze_path), "sha256": file_sha256(freeze_path)}, bindings)
    freeze = _json(freeze_path)
    if freeze.get("version") != VERSION or freeze.get("status") not in {"preparation", "final_frozen"}:
        raise ValueError("Unsupported freeze record")
    if freeze.get("representation") not in {"legacy-bilabial-v1", "matched-clips-v1", EXPANDED_REPRESENTATION}:
        raise ValueError("Unsupported input representation")
    matched = freeze["representation"] != "legacy-bilabial-v1"
    expanded = freeze["representation"] == EXPANDED_REPRESENTATION
    paths = {name: _bound(freeze[name], bindings) for name in
             ("checkpoint", "run", "threshold", "calibration", "preprocessing")}
    # The hash check above must pass before loading this trusted project checkpoint.
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    contract, options = checkpoint["resume_contract"], checkpoint["options"]
    if (checkpoint.get("checkpoint_type") != "seepat_evaluation_model"
            or contract.get("model_name") != FUSION_MODEL
            or contract.get("training_version") != training_version_for_model(FUSION_MODEL)
            or (contract.get("clip_input_contract") is not None) != matched):
        raise ValueError("Requires a compatible full-fusion evaluation checkpoint")
    (validate_supervision if expanded else _validate_supervision)(checkpoint, contract)
    if any(type(contract.get(key)) is not int or contract[key] < 1 for key in ("sequence_length", "image_size")):
        raise ValueError("Invalid frozen tensor dimensions")
    if matched:
        from .local_matched_inputs import validate_training_context

        validate_training_context(freeze, contract, bindings, _bound)
    if any(key not in options or options[key] is not None for key in ("max_train_batches", "max_validation_batches")):
        raise ValueError("Preflight checkpoints cannot be released")
    run = _json(paths["run"])
    if (run.get("status") != "complete" or run.get("run_type") != "training_experiment"
            or int(run.get("global_step", 0)) < 1 or run.get("resume_contract") != contract
            or (not expanded and Path(run["best_checkpoint"]).resolve() != paths["checkpoint"])
            or expanded and Path(run["best_checkpoint"]).name != "checkpoint_best.pt"):
        raise ValueError("Checkpoint is not the selected output of a completed fit")
    provenance = {"checkpoint_sha256": freeze["checkpoint"]["sha256"],
                  "validation_manifest_sha256": contract["validation_manifest_sha256"],
                  "model_name": contract["model_name"], "training_version": contract["training_version"],
                  "sequence_length": contract["sequence_length"], "image_size": contract["image_size"],
                  "evidence_version": EVIDENCE_VERSION, "fusion_inputs": contract["fusion_inputs"],
                  **({"supervision": contract["supervision"]} if contract.get("supervision") else {}),
                  **({"clip_input_contract": contract["clip_input_contract"]} if matched else {})}
    threshold = (saved_threshold(freeze, contract, run, bindings, _bound) if expanded
                 else _load_threshold(None, paths["threshold"], provenance))
    _bound(threshold["predictions"], bindings)
    _unchanged(bindings)
    ready = (freeze["status"] == "final_frozen" and freeze.get("approved_scope") == "local_unlabelled_inference"
             and freeze.get("runtime_sha256") == _runtime_hashes(matched))
    return {"freeze": freeze, "paths": paths, "contract": contract, "threshold": threshold,
            "bindings": bindings, "execution_ready": ready}


def load_local_model(checkpoint_path, device):
    """Trusted evaluation state only; no pretrained download or optimizer restoration."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = checkpoint["resume_contract"]
    if (checkpoint.get("checkpoint_type") != "seepat_evaluation_model"
            or contract.get("model_name") != FUSION_MODEL
            or contract.get("training_version") != training_version_for_model(FUSION_MODEL)):
        raise ValueError("Unsupported frozen fusion model")
    expanded = (contract.get("clip_input_contract") or {}).get("version") == EXPANDED_REPRESENTATION
    (validate_supervision if expanded else _validate_supervision)(checkpoint, contract)
    model = build_event_classifier(model_name=FUSION_MODEL, pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), contract


def validate_inputs(freeze_path: Path, input_path: Path):
    """Verify trusted frozen artifacts and local prepared inputs without scoring."""
    validated = validate_freeze(freeze_path)
    freeze, paths, contract = (validated[key] for key in ("freeze", "paths", "contract"))
    threshold, bindings = validated["threshold"], validated["bindings"]
    _bound({"path": str(input_path), "sha256": file_sha256(input_path)}, bindings)
    spec = _json(input_path)
    if (spec.get("version") != "prepared-inference-v1" or spec.get("dataset_source") != "local-unlabelled"
            or spec.get("split") != "local"):
        raise ValueError("Only unlabelled local inputs are enabled; external/test execution stays locked")
    if spec.get("representation") != freeze["representation"]:
        raise ValueError("Unsupported or crossed input representation")
    if spec.get("preprocessing_sha256") != freeze["preprocessing"]["sha256"]:
        raise ValueError("Prepared preprocessing differs from the frozen contract")
    matched = freeze["representation"] != "legacy-bilabial-v1"
    manifest = _bound(spec["manifest"], bindings)
    summary_path = _bound(spec["calibration_summary"], bindings)
    if summary_path != manifest.parent / "summary.json":
        raise ValueError("Prepared manifest must use its own bound calibration summary")
    summary = _json(summary_path)
    if (summary.get("mode") != "score_only"
            or summary.get("calibration_sha256") != freeze["calibration"]["sha256"]
            or Path(summary["calibration"]).resolve() != paths["calibration"]):
        raise ValueError("Inputs must use unchanged frozen Train population calibration")
    rows = read_csv_rows(manifest)
    if calibrated_manifest_contract(manifest, rows, allow_empty=True) != contract["fusion_inputs"]:
        raise ValueError("Prepared evidence contract differs from the checkpoint")
    requested = spec["requested_videos"]
    ids = [row["video_id"] for row in requested]
    statuses = {"ready", "alignment_failed", "preprocessing_failed", "no_eligible_bilabials", "ineligible", "corrupt_media"}
    if (not ids or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) for value in ids)
            or len(ids) != len(set(ids)) or any(row.get("status") not in statuses or not row.get("source_group")
                                               or row.get("dataset_source", "local-unlabelled") != "local-unlabelled"
                                               for row in requested)):
        raise ValueError("Requested videos must be nonempty and unique")
    by_video = {row["video_id"]: row for row in requested}
    event_ids = [row["event_id"] for row in rows]
    if any(not value for value in event_ids) or len(event_ids) != len(set(event_ids)):
        raise ValueError("Prepared event IDs must be unique and nonempty")
    for row in rows:
        source = by_video.get(row["video_id"])
        if (source is None or source.get("status") != "ready" or not row.get("source_group")
                or source.get("source_group") != row["source_group"] or row.get("dataset_split") != "local"
                or row.get("dataset_source", "local-unlabelled") != "local-unlabelled"
                or (row.get("kind") or "bilabial") != "bilabial" or row.get("phoneme") not in {"m", "b", "p"}):
            raise ValueError("Prepared identity, eligibility, split or phoneme mismatch")
        _bound({"path": row["mouth_clip_path"], "sha256": row["mouth_clip_sha256"]}, bindings)
    matched_directory, matched_coverage = None, None
    if matched:
        from .local_matched_inputs import select_local_rows

        rows, matched_directory, matched_coverage = select_local_rows(
            freeze, spec, contract, rows, requested, bindings, _bound)
    raw_scope = None
    raw_ready = True
    if spec.get("raw_provenance"):
        from .local_raw_inputs import raw_runtime_hashes, validate_prepared_provenance

        raw_scope = validate_prepared_provenance(spec, rows, requested, bindings)
        raw_ready = freeze.get("raw_preparation_sha256") == raw_runtime_hashes(matched)
    _unchanged(bindings)
    return {"status": "passed", "execution_ready": validated["execution_ready"] and raw_ready
            and raw_scope != "synthetic_engineering",
            "events": len(rows), "requested": len(requested), "rows": rows, "requested_videos": requested,
            "threshold": threshold, "contract": contract, "checkpoint": str(paths["checkpoint"]),
            "prepared_spec": str(input_path.resolve()),
            "manifest": str(manifest), "matched_directory": matched_directory,
            "matched_coverage": matched_coverage,
            "bindings": bindings, "scope": "local-unlabelled; no external evaluation or accuracy metrics"}


def score_prepared(model, dataset, requested_videos, threshold, *, device="cpu", batch_size=1, progress=None):
    """Numerical scoring seam; caller supplies validated inputs and an evaluation model.

    Tiny concrete Torch models/tensor datasets exercise this seam in synthetic
    tests. This function alone does not certify a checkpoint or input provenance.
    """
    if (type(batch_size) is not int or batch_size < 1 or type(threshold) not in {int, float}
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Invalid scoring batch size or frozen threshold")
    requested = {row["video_id"]: row for row in requested_videos}
    if (not requested or len(requested) != len(requested_videos) or not all(requested)
            or any(not row.get("status") for row in requested_videos)):
        raise ValueError("Requested-video identity/status is invalid")
    event_ids = [row["event_id"] for row in dataset.rows]
    if len(event_ids) != len(set(event_ids)) or any(not value for value in event_ids):
        raise ValueError("Scored event IDs must be unique and nonempty")
    if any(row["video_id"] not in requested or requested[row["video_id"]]["status"] != "ready"
           for row in dataset.rows):
        raise ValueError("Scoring cannot include absent or failed inputs")
    device = torch.device(device)
    model.to(device).eval()
    records = _score_events(model, dataset, device, batch_size, progress) if len(dataset) else []
    maxima = {}
    for record in records:
        # The shared scorer also serves evaluation; local inputs have no ground-truth labels.
        record.pop("label", None)
        record.pop("video_label", None)
        identifier = record["video_id"]
        maxima[identifier] = max(maxima.get(identifier, 0.0), record["manipulated_probability"])
    counts = Counter(record["video_id"] for record in records)
    videos = []
    for source in requested_videos:
        identifier = source["video_id"]
        probability = maxima.get(identifier)
        videos.append({"video_id": identifier, "event_count": counts[identifier],
                       "maximum_probability": probability if probability is not None else "",
                       "threshold": threshold,
                       "verdict": ("not_evaluated" if probability is None else
                                   "manipulated" if probability >= threshold else "authentic"),
                       "reason": ("" if probability is not None else source["status"]
                                  if source["status"] != "ready" else "no_scorable_clips")})
    return {"events": records, "videos": videos, "metrics": None, "labels_available": False,
            "aggregation": "maximum event probability",
            "interpretation": "Video evidence contributions, not validated manipulation localization"}


class PreparedDataset:
    """Frozen bilabial/context pixels without the training loader's required labels."""

    def __init__(self, validated):
        self.rows = validated["rows"]
        self.length = validated["contract"]["sequence_length"]
        self.size = validated["contract"]["image_size"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import numpy as np

        row = self.rows[index]
        path = Path(row["mouth_clip_path"])
        if file_sha256(path) != row["mouth_clip_sha256"]:
            raise ValueError("Prepared clip changed before decoding")
        context = row.get("kind") == "context"
        frames = decode_clip_frames(path, self.size, source_size=(112, 112) if context else None)
        if context:
            if len(frames) != 16 or self.length != 16:
                raise ValueError("Context decoder must retain exactly 16 stored slots")
            indices, mask = list(range(16)), row["_frame_mask"]
        else:
            indices, mask = sequence_frame_indices(len(frames), self.length)
        array = np.ascontiguousarray(np.stack([frames[position] for position in indices]))
        if context:
            array[~np.asarray(mask, dtype=bool)] = 0
            values, available = [0.0] * len(FUSION_EVIDENCE_FIELDS), [False] * len(FUSION_EVIDENCE_FIELDS)
        else:
            values, available = evidence_feature_values(row)
        return {"video": torch.from_numpy(array).permute(3, 0, 1, 2).float().div_(255),
                "frame_mask": torch.tensor(mask, dtype=torch.bool),
                "evidence_features": torch.tensor(values, dtype=torch.float32),
                "evidence_feature_mask": torch.tensor(available, dtype=torch.bool),
                "label": torch.tensor(-1), "video_label": torch.tensor(-1),
                "event_id": row["event_id"], "video_id": row["video_id"]}


def _destination(validated, output_dir):
    directory = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Preserve existing output; inspect completion before recovery")
    protected = [Path(validated["checkpoint"]).parent, Path(validated["manifest"]).parent]
    if validated.get("matched_directory"):
        protected.append(Path(validated["matched_directory"]))
    protected.extend(Path(row["mouth_clip_path"]).resolve().parent for row in validated["rows"])
    if any(directory.is_relative_to(path) for path in protected):
        raise ValueError("Output must be outside frozen model/prepared clip directories")
    return directory


def validate_scored(validated, scored):
    """Shared decision/coverage checks for fresh exports and sealed result readers."""
    if (scored.get("labels_available") is not False or scored.get("metrics") is not None
            or scored.get("aggregation") != "maximum event probability"
            or [row["event_id"] for row in scored["events"]] != [row["event_id"] for row in validated["rows"]]
            or [row["video_id"] for row in scored["videos"]] != [row["video_id"] for row in validated["requested_videos"]]
            or any(row["threshold"] != validated["threshold"]["value"] for row in scored["videos"])
            or any("label" in row or "video_label" in row for row in scored["events"])):
        raise ValueError("Scored output differs from its validated input/decision contract")
    maxima, counts = {}, Counter()
    for record, original in zip(scored["events"], validated["rows"], strict=True):
        probability = record["manipulated_probability"]
        if (record["video_id"] != original["video_id"] or record.get("phoneme") != original["phoneme"]
                or type(probability) not in {int, float} or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("Invalid event identity or score")
        if validated["contract"].get("clip_input_contract") is not None:
            if record.get("kind") != original["kind"]:
                raise ValueError("Matched output cannot change clip kind")
            if original["kind"] == "context" and (
                record.get("evidence_available") != 0 or record.get("evidence_total") != len(FUSION_EVIDENCE_FIELDS)
                or any(record.get(f"{field}_available") is not False or record.get(field) != ""
                       for field in FUSION_EVIDENCE_FIELDS)
            ):
                raise ValueError("Context output must retain unavailable biological evidence")
        identifier = record["video_id"]
        counts[identifier] += 1
        maxima[identifier] = max(maxima.get(identifier, 0.0), probability)
    for row, source in zip(scored["videos"], validated["requested_videos"], strict=True):
        probability = maxima.get(row["video_id"])
        expected = ("not_evaluated" if probability is None else "manipulated"
                    if probability >= row["threshold"] else "authentic")
        reason = "" if probability is not None else source["status"] if source["status"] != "ready" else "no_scorable_clips"
        if (row["maximum_probability"] != (probability if probability is not None else "")
                or row["verdict"] != expected or row["event_count"] != counts[row["video_id"]]
                or row["reason"] != reason or "label" in row or "video_label" in row):
            raise ValueError("Video result disagrees with maximum aggregation or coverage")


def read_result(validated, directory: Path, *, engineering=False):
    """Verify a completed numerical result without decoding, scoring or writing."""
    bindings = list(validated["bindings"])
    _bound({"path": str(directory / "summary.json"), "sha256": file_sha256(directory / "summary.json")}, bindings)
    saved = _json(directory / "summary.json")
    if (saved.get("status") != "complete" or saved.get("version") != VERSION
            or saved.get("scope") != ("synthetic_engineering" if engineering else "local_unlabelled_inference")
            or saved.get("bindings") != validated["bindings"] or saved.get("contract") != validated["contract"]
            or saved.get("threshold") != validated["threshold"] or saved.get("metrics") is not None
            or saved.get("labels_available") is not False
            or saved.get("matched_coverage") != validated.get("matched_coverage")
            or set(saved.get("output_sha256", {})) != {"event_predictions.csv", "video_verdicts.csv", "report.md"}):
        raise ValueError("Completed numerical result differs from frozen inputs")
    for name, digest in saved["output_sha256"].items():
        _bound({"path": str(directory / name), "sha256": digest}, bindings)
    events, videos = read_csv_rows(directory / "event_predictions.csv"), read_csv_rows(directory / "video_verdicts.csv")
    for row in events:
        row["manipulated_probability"] = float(row["manipulated_probability"])
        for name in ("evidence_available", "evidence_total"):
            row[name] = int(row[name])
        for field in FUSION_EVIDENCE_FIELDS:
            flag = row[f"{field}_available"]
            if flag not in {"True", "False"}:
                raise ValueError("Invalid saved evidence availability")
            row[f"{field}_available"] = flag == "True"
            row[field] = float(row[field]) if flag == "True" else row[field]
    for row in videos:
        row["event_count"], row["threshold"] = int(row["event_count"]), float(row["threshold"])
        row["maximum_probability"] = float(row["maximum_probability"]) if row["maximum_probability"] else ""
    scored = {"events": events, "videos": videos, "labels_available": False, "metrics": None,
              "aggregation": "maximum event probability"}
    validate_scored(validated, scored)
    _unchanged(bindings)
    return {"summary": saved, "scored": scored, "bindings": bindings}


def export_result(validated, scored, output_dir: Path, *, engineering=False):
    """Seal caller-supplied scored results; synthetic tests are explicitly labelled."""
    directory = _destination(validated, output_dir)
    if not engineering and not validated["execution_ready"]:
        raise ValueError("Preparation record cannot authorize real inference")
    validate_scored(validated, scored)
    from seepat.artifacts import atomic_write_csv

    _unchanged(validated["bindings"])
    directory.mkdir(parents=True, exist_ok=False)
    atomic_write_csv(directory / "event_predictions.csv", scored["events"])
    atomic_write_csv(directory / "video_verdicts.csv", scored["videos"])
    scope = "synthetic_engineering" if engineering else "local_unlabelled_inference"
    lines = ["# SeePAT inference result", "",
             "ENGINEERING CHECK: synthetic result, not thesis performance." if engineering else
             "LOCAL UNLABELLED INFERENCE: not an external-test evaluation.", ""]
    for row in scored["videos"]:
        lines.append(f"- {row['video_id']}: {row['verdict']}; {row['event_count']} analyzed clips; "
                     f"maximum score {row['maximum_probability'] if row['maximum_probability'] != '' else 'unavailable'}; "
                     f"frozen cutoff {row['threshold']}." + (f" Reason: {row['reason']}." if row["reason"] else ""))
    lines += ["", "Unknown ground truth: accuracy and other classification metrics are not computed.",
              ("The verdict is a model result, not proof. Scores are not calibrated confidence and "
               "clip contributions are not verified manipulation localization."), "",
              ("Full measurements and availability flags: event_predictions.csv. "
               "Explicit unevaluated coverage: video_verdicts.csv.")]
    with (directory / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    _unchanged(validated["bindings"])
    summary = {"version": VERSION, "status": "complete", "scope": scope,
               "generated_at_utc": datetime.now(UTC).isoformat(), "metrics": None,
               "labels_available": False, "bindings": validated["bindings"],
               "contract": validated["contract"], "threshold": validated["threshold"],
               **({"matched_coverage": validated["matched_coverage"]} if validated.get("matched_coverage") is not None else {}),
               "output_sha256": {name: file_sha256(directory / name) for name in
                                 ("event_predictions.csv", "video_verdicts.csv", "report.md")}}
    with (directory / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    if args.run != (args.output_dir is not None):
        parser.error("--run and --output-dir must be supplied together")
    print("FROZEN INFERENCE | verify frozen contracts; external/test execution stays locked", flush=True)
    try:
        result = validate_inputs(args.freeze, args.inputs)
        if args.run and not result["execution_ready"]:
            print("FROZEN INFERENCE FAILED | not_final_frozen; no scoring authorized", file=sys.stderr)
            raise SystemExit(1)
        if args.run:
            if args.batch_size < 1:
                raise ValueError("Invalid batch size")
            if args.output_dir.exists() or args.output_dir.is_symlink():
                saved = _json(args.output_dir / "summary.json")
                if (saved.get("status") != "complete" or saved.get("version") != VERSION
                        or saved.get("scope") != "local_unlabelled_inference"
                        or saved.get("bindings") != result["bindings"] or saved.get("contract") != result["contract"]
                        or saved.get("threshold") != result["threshold"]
                        or saved.get("matched_coverage") != result.get("matched_coverage")
                        or set(saved.get("output_sha256", {})) != {"event_predictions.csv", "video_verdicts.csv", "report.md"}
                        or any(file_sha256(args.output_dir / name) != sha for name, sha in saved["output_sha256"].items())):
                    raise FileExistsError("Existing output is partial, changed or belongs to another input")
                _unchanged(result["bindings"])
                print("FROZEN INFERENCE | VERIFIED COMPLETE; SKIPPED", flush=True)
                return {**saved, "action": "skipped"}
            _destination(result, args.output_dir)
            from .frozen_inference import load_local_model as load_evaluation_model

            dataset = PreparedDataset(result)
            if len(dataset):
                model, loaded_contract = load_evaluation_model(Path(result["checkpoint"]), torch.device(args.device))
                if loaded_contract != result["contract"]:
                    raise ValueError("Model contract changed before scoring")
            else:
                model = torch.nn.Identity()

            def progress(phase, done, total, item):
                print(f"FROZEN INFERENCE | {phase} | {done}/{total} | {item}", flush=True)

            scored = score_prepared(model, dataset, result["requested_videos"], result["threshold"]["value"],
                                    device=args.device, batch_size=args.batch_size, progress=progress)
            summary = export_result(result, scored, args.output_dir)
            print(f"FROZEN INFERENCE COMPLETE | {args.output_dir / 'summary.json'}", flush=True)
            return summary
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError):
        print("FROZEN INFERENCE FAILED | input_contract_failed; preserve inputs", file=sys.stderr)
        raise SystemExit(1) from None
    print("VERIFICATION PASSED | PREPARATION ONLY; this does not run inference", flush=True)
    print(json.dumps({key: result[key] for key in ("status", "events", "requested", "execution_ready")}))
    return result


if __name__ == "__main__":
    main()
