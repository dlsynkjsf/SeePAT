"""Opt-in, hash-bound matched bilabial/context inputs. Old event data stay unchanged."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from seepat.artifacts import file_sha256, read_csv_rows
from seepat.context_inputs import INPUT_VERSION, paired_indices
from seepat.data.sampling import canonical_source_group
from seepat.evidence import FUSION_EVIDENCE_FIELDS, evidence_coverage
from seepat.training.dataset import FEATURE_FIELDS, MouthEventDataset, video_class_id

PROFILE_VERSION = "matched-clips-v1"


def validate_profile(profile: dict) -> dict:
    if not isinstance(profile, dict) or set(profile) != {
        "version", "summary_sha256", "initial_checkpoint_sha256"
    } or profile["version"] != PROFILE_VERSION:
        raise ValueError("Unsupported matched input profile")
    for key in ("summary_sha256", "initial_checkpoint_sha256"):
        value = profile[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Matched inputs require a pinned {key}")
    return dict(profile)


def inside(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Matched input path escapes its declared root")
    return path


class MatchedClipDataset:
    """Load original bilabials verbatim and explicit unlabelled visual contexts.

    ``label=-1`` is an unavailable event target, never a training class. This
    dataset is valid only for complete-video supervision with both clip kinds.
    """

    def __init__(self, manifest_path: Path, project_root: Path, dataset_split: str,
                 profile: dict, sequence_length: int = 16, image_size: int = 224):
        if dataset_split not in {"train", "val"} or sequence_length != 16 or image_size < 1:
            raise ValueError("Matched inputs require Train/Validation, 16 slots and a positive size")
        self.profile = validate_profile(profile)
        self.root = project_root.resolve()
        self.directory = manifest_path.resolve().parent
        summary_path = self.directory / "summary.json"
        if file_sha256(summary_path) != self.profile["summary_sha256"]:
            raise ValueError("Matched export summary hash differs from the pinned profile")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        contract = summary["contract"]
        if (summary.get("status") != "complete" or summary.get("input_version") != INPUT_VERSION
                or contract != json.loads((self.directory / "contract.json").read_text(encoding="utf-8"))
                or summary.get("engineering_gate") != "review_for_training_wiring"):
            raise ValueError("Matched inputs require a completed, gate-passing export")
        if manifest_path.name not in {f"control_{dataset_split}.csv", f"candidate_{dataset_split}.csv"}:
            raise ValueError("Matched index name/split is incompatible")
        self.arm = manifest_path.name.split("_")[0]
        for name, expected in summary["output_sha256"].items():
            if file_sha256(inside(self.directory, name)) != expected:
                raise ValueError("Matched export index hash mismatch")
        originals = {}
        for split in ("train", "val"):
            matches = [name for name in contract["input_sha256"]
                       if Path(name).name.startswith("events_") and name.endswith("_calibrated.csv")
                       and (Path(name).name == "events_train_calibrated.csv") == (split == "train")]
            if len(matches) != 1:
                raise ValueError("Matched export has ambiguous original event-manifest roles")
            path = inside(self.root, matches[0])
            if file_sha256(path) != contract["input_sha256"][matches[0]]:
                raise ValueError("Original calibrated input changed")
            originals[split] = path
        self.bilabials = MouthEventDataset(originals[dataset_split], self.root,
                                           dataset_split=dataset_split, sequence_length=16,
                                           image_size=image_size, require_calibration=True)
        self.calibration_contract = self.bilabials.calibration_contract
        self.input_contract = {
            "version": PROFILE_VERSION, "profile": self.profile, "arm": self.arm,
            "index_sha256": summary["output_sha256"],
            "source_manifest_sha256": {split: file_sha256(path) for split, path in originals.items()},
            "policy": "cap16; >=half bilabial; same per-video counts; context event target unavailable",
            "context_masks": "16 saved crop-valid/padding flags; invalid decoded pixels zeroed",
            "context_evidence": "all values zero placeholders, all availability flags false",
        }
        self.image_size = image_size
        self.rows, self.contexts, self.base_indices = [], {}, {}
        original_by_video = defaultdict(list)
        for index, row in enumerate(self.bilabials.rows):
            self.base_indices[row["event_id"]] = index
            original_by_video[row["video_id"]].append(row)
        indexes = {arm: read_csv_rows(self.directory / f"{arm}_{dataset_split}.csv")
                   for arm in ("control", "candidate")}
        index_fields = {arm: list(dict.fromkeys(key for row in rows for key in row))
                        for arm, rows in indexes.items()}
        for arm, rows in indexes.items():
            if any(row.get("dataset_split") != dataset_split for row in rows):
                raise ValueError("Matched index contains another split")
            ids = [row.get("clip_id") for row in rows]
            if not all(ids) or len(ids) != len(set(ids)):
                raise ValueError("Matched indexes need unique nonempty clip IDs")
        index_groups = {arm: defaultdict(list) for arm in indexes}
        for arm, rows in indexes.items():
            for row in rows:
                index_groups[arm][row["video_id"]].append(row)
            if set(index_groups[arm]) != set(original_by_video):
                raise ValueError("Matched indexes changed scorable video membership")
        selected_contexts = {}
        for identifier, original_rows in original_by_video.items():
            record_path = inside(self.directory, f"records/{identifier}/record.json")
            relative = f"records/{identifier}/record.json"
            if file_sha256(record_path) != summary["record_sha256"].get(relative):
                raise ValueError("Matched per-video record hash differs")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            canonical = canonical_source_group(original_rows[0]["source_group"], None)
            if record.get("video_id") != identifier or record.get("split") != dataset_split or record.get("source_group") != canonical:
                raise ValueError("Matched record identity differs")
            expected = paired_indices(original_rows, record["windows"])
            for arm, planned in zip(("control", "candidate"), expected, strict=True):
                fields = index_fields[arm]
                actual = index_groups[arm][identifier]
                if actual != [{key: str(row.get(key, "")) for key in fields} for row in planned]:
                    raise ValueError("Matched per-video selection or count differs from the frozen planner")
            for window in record["windows"]:
                if not window["technical_ok"]:
                    continue
                mask = window["frame_mask"]
                if (len(mask) != 16 or any(type(v) is not bool for v in mask) or not any(mask)
                        or window.get("evidence_available") is not False
                        or window.get("phoneme") or window.get("event_class_id")):
                    raise ValueError("Context frame masks or unavailable evidence are invalid")
                selected_contexts[window["context_id"]] = (window, record_path.parent)
        for item in indexes[self.arm]:
            if item["kind"] == "bilabial":
                original = self.bilabials.rows[self.base_indices[item["source_event_id"]]]
                row = {**original, "kind": "bilabial", "source_group": item["source_group"]}
            elif item["kind"] == "context":
                window, directory = selected_contexts[item["clip_id"]]
                original = original_by_video[item["video_id"]][0]
                clip = inside(directory, window["clip_file"])
                row = {**item, "event_id": item["clip_id"], "class_id": "-1", "phoneme": "",
                       "manipulation_modality": original["manipulation_modality"],
                       "mouth_clip_path": clip.relative_to(self.root).as_posix()}
                self.contexts[item["clip_id"]] = window
            else:
                raise ValueError("Unknown matched clip kind")
            self.rows.append(row)
        if not self.rows:
            raise ValueError("Matched dataset has no scorable clips")

    def __len__(self):
        return len(self.rows)

    def evidence_coverage(self):
        return evidence_coverage(self.rows)

    def audit_clips(self):
        for row in self.rows:
            path = inside(self.root, row["mouth_clip_path"])
            if not path.is_file():
                raise ValueError(f"Missing matched clip: {path}")
            if row["kind"] == "context" and file_sha256(path) != self.contexts[row["event_id"]]["clip_sha256"]:
                raise ValueError("Context clip hash mismatch")
        return dict(Counter(row["kind"] for row in self.rows))

    def __getitem__(self, index):
        import cv2
        import numpy as np
        import torch

        row = self.rows[index]
        if row["kind"] == "bilabial":
            sample = self.bilabials[self.base_indices[row["event_id"]]]
            return {**sample, "source_group": row["source_group"], "kind": "bilabial"}
        window = self.contexts[row["event_id"]]
        path = inside(self.root, row["mouth_clip_path"])
        if file_sha256(path) != window["clip_sha256"]:
            raise ValueError("Context clip hash mismatch")
        capture = cv2.VideoCapture(str(path))
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                         (self.image_size, self.image_size)))
        finally:
            capture.release()
        if len(frames) != 16:
            raise ValueError("Context decoder must retain exactly 16 stored slots")
        array = np.stack(frames)
        array[~np.array(window["frame_mask"], dtype=bool)] = 0
        return {
            "video": torch.from_numpy(np.ascontiguousarray(array)).permute(3, 0, 1, 2).float().div_(255),
            "frame_mask": torch.tensor(window["frame_mask"], dtype=torch.bool),
            "decoded_frame_count": len(frames),
            "features": torch.zeros(len(FEATURE_FIELDS)), "feature_mask": torch.zeros(len(FEATURE_FIELDS), dtype=torch.bool),
            "evidence_features": torch.zeros(len(FUSION_EVIDENCE_FIELDS)),
            "evidence_feature_mask": torch.zeros(len(FUSION_EVIDENCE_FIELDS), dtype=torch.bool),
            "label": torch.tensor(-1, dtype=torch.long), "video_label": torch.tensor(video_class_id(row["manipulation_modality"])),
            "event_id": row["event_id"], "video_id": row["video_id"], "source_group": row["source_group"],
            "manipulation_modality": row["manipulation_modality"], "phoneme": "", "kind": "context",
        }


def input_dataset(manifest_path, project_root, dataset_split, sequence_length=16, image_size=224,
                  require_calibration=False, allow_empty=False, input_profile=None):
    if input_profile is not None:
        if not require_calibration or allow_empty:
            raise ValueError("Matched inputs require fusion calibration and a nonempty paired cohort")
        return MatchedClipDataset(manifest_path, project_root, dataset_split, input_profile,
                                  sequence_length, image_size)
    return MouthEventDataset(manifest_path, project_root, dataset_split, sequence_length,
                             image_size, require_calibration, allow_empty)
