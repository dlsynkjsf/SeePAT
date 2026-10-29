from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from seepat.artifacts import read_csv_rows
from seepat.evidence import calibrated_manifest_contract, evidence_feature_values
from seepat.evidence import evidence_coverage as _evidence_coverage
from seepat.training.clip_cache import read_clip_array
from seepat.training.clip_cache import sequence_frame_indices as _sequence_frame_indices

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


def sequence_frame_indices(frame_count: int, sequence_length: int) -> tuple[list[int], list[bool]]:
    """Retain the public dataset selection API used by existing experiments."""
    return _sequence_frame_indices(frame_count, sequence_length)


def video_class_id(manipulation_modality: str) -> int:
    try:
        return VIDEO_MODALITY_LABELS[manipulation_modality.strip()]
    except KeyError as error:
        raise ValueError(
            f"Unsupported manipulation modality: {manipulation_modality!r}"
        ) from error


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
    """Loads fixed-length mouth clips and numerical bilabial features.

    Each sample exposes both feature groups: the base motion features used by
    the CNN-temporal baseline and the calibrated biological-alignment evidence
    consumed by the hybrid fusion model (see :mod:`seepat.evidence`).
    """

    def __init__(
        self,
        manifest_path: Path,
        project_root: Path = Path("."),
        dataset_split: str | None = None,
        sequence_length: int = 16,
        image_size: int = 224,
        require_calibration: bool = False,
        allow_empty: bool = False,
        clip_cache_dir: Path | None = None,
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
        self.clip_cache_dir = clip_cache_dir

    def __len__(self) -> int:
        return len(self.rows)

    def evidence_coverage(self) -> dict[str, float]:
        """Fraction of events with each biological-alignment evidence field."""
        return _evidence_coverage(self.rows)

    def clip_source(self, index: int) -> tuple[Path, None, None]:
        """Locate source pixels without decoding or caching targets/evidence."""
        relative_path = Path(self.rows[index]["mouth_clip_path"].replace("\\", "/"))
        clip_path = (
            relative_path if relative_path.is_absolute() else self.project_root / relative_path
        )
        return clip_path, None, None

    def sample_metadata(self, index: int) -> dict[str, Any]:
        """Return live manifest-derived sample fields independently of pixels."""
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                'Clip loading requires the training dependencies: pip install -e ".[train]"'
            ) from error

        row = self.rows[index]
        feature_values, feature_mask = numeric_feature_values(row)
        evidence_values, evidence_mask = evidence_feature_values(row)

        return {
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        metadata = self.sample_metadata(index)
        import torch

        clip_path, _, _ = self.clip_source(index)
        array, frame_mask, decoded_count = read_clip_array(
            clip_path, self.sequence_length, self.image_size, self.clip_cache_dir,
        )
        return {
            "video": torch.from_numpy(array).permute(3, 0, 1, 2).float().div_(255.0),
            "frame_mask": torch.tensor(frame_mask, dtype=torch.bool),
            "decoded_frame_count": decoded_count,
            **metadata,
        }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.training.dataset")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--split")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--clip-cache-dir", type=Path)
    args = parser.parse_args()

    dataset = MouthEventDataset(
        manifest_path=args.manifest,
        project_root=args.project_root,
        dataset_split=args.split,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        clip_cache_dir=args.clip_cache_dir,
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
