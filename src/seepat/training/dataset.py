from __future__ import annotations

from pathlib import Path
from typing import Any

from seepat.artifacts import read_csv_rows
from seepat.evidence import calibrated_manifest_contract, evidence_feature_values, finite_float
from seepat.evidence import evidence_coverage as _evidence_coverage

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
        value = finite_float(row.get(field, ""))
        values.append(value if value is not None else 0.0)
        mask.append(value is not None)
    return values, mask


def decode_clip_frames(path: Path, image_size: int, *, source_size=None):
    """Decode RGB frames once; callers retain their own sampling and label rules."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open mouth clip: {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_size is not None and frame.shape[:2] != source_size:
                raise ValueError("Context storage must retain 112px mouth-region frames")
            frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                     (image_size, image_size)))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Mouth clip contains no readable frames: {path}")
    return frames


class MouthEventDataset:
    """Fixed-length mouth clips with motion features and calibrated evidence masks."""

    def __init__(
        self,
        manifest_path: Path,
        project_root: Path = Path("."),
        dataset_split: str | None = None,
        sequence_length: int = 16,
        image_size: int = 224,
        require_calibration: bool = False,
        allow_empty: bool = False,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        if image_size < 1:
            raise ValueError("image_size must be at least 1")

        rows = read_csv_rows(manifest_path)
        self.calibration_contract = (
            calibrated_manifest_contract(manifest_path, rows, allow_empty=allow_empty)
            if require_calibration else None
        )
        if require_calibration and dataset_split is not None and any(
            row.get("dataset_split") != dataset_split for row in rows
        ):
            raise ValueError("Fusion training manifests must contain only the requested split")
        if dataset_split is not None:
            rows = [row for row in rows if row.get("dataset_split") == dataset_split]
        if not rows and not allow_empty:
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

    def evidence_coverage(self) -> dict[str, float]:
        """Fraction of events with each biological-alignment evidence field."""
        return _evidence_coverage(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        relative_path = Path(row["mouth_clip_path"].replace("\\", "/"))
        clip_path = (
            relative_path if relative_path.is_absolute() else self.project_root / relative_path
        )
        try:
            import numpy as np
            import torch

            frames = decode_clip_frames(clip_path, self.image_size)
        except ImportError as error:
            raise RuntimeError(
                'Clip loading requires the training dependencies: pip install -e ".[train]"'
            ) from error
        except ValueError as error:
            raise RuntimeError(str(error)) from error

        indices, frame_mask = sequence_frame_indices(len(frames), self.sequence_length)
        array = np.ascontiguousarray(np.stack([frames[position] for position in indices]))
        video = torch.from_numpy(array).permute(3, 0, 1, 2).float().div_(255.0)
        feature_values, feature_mask = numeric_feature_values(row)
        evidence_values, evidence_mask = evidence_feature_values(row)

        return {
            "video": video,
            "frame_mask": torch.tensor(frame_mask, dtype=torch.bool),
            "decoded_frame_count": len(frames),
            "features": torch.tensor(feature_values, dtype=torch.float32),
            "feature_mask": torch.tensor(feature_mask, dtype=torch.bool),
            "evidence_features": torch.tensor(evidence_values, dtype=torch.float32),
            "evidence_feature_mask": torch.tensor(evidence_mask, dtype=torch.bool),
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
