from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from seepat.artifacts import read_csv_rows

FEATURE_FIELDS = (
    "normalized_minimum_closure",
    "closure_time_s",
    "closure_duration_s",
    "closing_velocity",
    "opening_velocity",
    "valid_landmark_ratio",
    "multiple_face_ratio",
    "phone_duration_s",
)

VIDEO_MODALITY_LABELS = {
    "real": 0,
    "audio_modified": 1,
    "visual_modified": 1,
    "both_modified": 1,
}


def video_class_id(manipulation_modality: str) -> int:
    try:
        return VIDEO_MODALITY_LABELS[manipulation_modality.strip()]
    except KeyError as error:
        raise ValueError(
            f"Unsupported manipulation modality: {manipulation_modality!r}"
        ) from error


def sequence_frame_indices(frame_count: int, sequence_length: int) -> tuple[list[int], list[bool]]:
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")

    if frame_count >= sequence_length:
        if sequence_length == 1:
            return [frame_count // 2], [True]
        indices = [
            round(index * (frame_count - 1) / (sequence_length - 1))
            for index in range(sequence_length)
        ]
        return indices, [True] * sequence_length

    padding = sequence_length - frame_count
    return (
        list(range(frame_count)) + [frame_count - 1] * padding,
        [True] * frame_count + [False] * padding,
    )


def numeric_feature_values(row: dict[str, str]) -> tuple[list[float], list[bool]]:
    values: list[float] = []
    mask: list[bool] = []
    for field in FEATURE_FIELDS:
        try:
            value = float(row.get(field, ""))
        except (TypeError, ValueError):
            value = 0.0
            available = False
        else:
            available = math.isfinite(value)
            if not available:
                value = 0.0
        values.append(value)
        mask.append(available)
    return values, mask


class MouthEventDataset:
    """Loads fixed-length mouth clips and numerical bilabial features."""

    def __init__(
        self,
        manifest_path: Path,
        project_root: Path = Path("."),
        dataset_split: str | None = None,
        sequence_length: int = 16,
        image_size: int = 224,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        if image_size < 1:
            raise ValueError("image_size must be at least 1")

        rows = read_csv_rows(manifest_path)
        if dataset_split is not None:
            rows = [row for row in rows if row.get("dataset_split") == dataset_split]
        if not rows:
            qualifier = f" for split {dataset_split!r}" if dataset_split else ""
            raise ValueError(f"Training manifest contains no events{qualifier}")

        required_fields = {
            "event_id",
            "video_id",
            "dataset_split",
            "source_group",
            "manipulation_modality",
            "phoneme",
            "class_id",
            "mouth_clip_path",
        }
        for row_number, row in enumerate(rows, start=2):
            missing = sorted(field for field in required_fields if not row.get(field, "").strip())
            if missing:
                raise ValueError(
                    f"Training manifest row {row_number} is missing: {', '.join(missing)}"
                )
            if row["class_id"] not in {"0", "1"}:
                raise ValueError(f"Training manifest row {row_number} has an invalid class_id")

        self.rows = rows
        self.project_root = project_root
        self.sequence_length = sequence_length
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            import cv2
            import numpy as np
            import torch
        except ImportError as error:
            raise RuntimeError(
                'Clip loading requires the training dependencies: pip install -e ".[train]"'
            ) from error

        row = self.rows[index]
        relative_path = Path(row["mouth_clip_path"].replace("\\", "/"))
        clip_path = (
            relative_path if relative_path.is_absolute() else self.project_root / relative_path
        )
        capture = cv2.VideoCapture(str(clip_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open mouth clip: {clip_path}")

        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(frame, (self.image_size, self.image_size)))
        finally:
            capture.release()
        if not frames:
            raise RuntimeError(f"Mouth clip contains no readable frames: {clip_path}")

        indices, frame_mask = sequence_frame_indices(len(frames), self.sequence_length)
        array = np.ascontiguousarray(np.stack([frames[position] for position in indices]))
        video = torch.from_numpy(array).permute(3, 0, 1, 2).float().div_(255.0)
        feature_values, feature_mask = numeric_feature_values(row)

        return {
            "video": video,
            "frame_mask": torch.tensor(frame_mask, dtype=torch.bool),
            "features": torch.tensor(feature_values, dtype=torch.float32),
            "feature_mask": torch.tensor(feature_mask, dtype=torch.bool),
            "label": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "event_id": row["event_id"],
            "video_id": row["video_id"],
            "source_group": row["source_group"],
            "manipulation_modality": row["manipulation_modality"],
            "video_label": torch.tensor(
                video_class_id(row["manipulation_modality"]),
                dtype=torch.long,
            ),
            "phoneme": row["phoneme"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.training.dataset")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--split")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    dataset = MouthEventDataset(
        manifest_path=args.manifest,
        project_root=args.project_root,
        dataset_split=args.split,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
    )
    sample = dataset[0]
    report = {
        "dataset_events": len(dataset),
        "event_id": sample["event_id"],
        "video_id": sample["video_id"],
        "video_shape": list(sample["video"].shape),
        "feature_shape": list(sample["features"].shape),
        "valid_frames": int(sample["frame_mask"].sum().item()),
        "label": int(sample["label"].item()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
