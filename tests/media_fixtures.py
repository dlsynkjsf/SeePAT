"""Synthetic media and calibrated manifests for integration tests."""

from itertools import chain

import numpy as np
import pytest

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256
from seepat.evidence import CALIBRATED_INPUT_VERSION, FUSION_EVIDENCE_FIELDS


def write_clip(path, frames, *, fps=25, codec="mp4v", skip_unavailable=False):
    import cv2

    frames = iter(frames)
    first = next(frames)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, first.shape[1::-1])
    if not writer.isOpened() and skip_unavailable:
        pytest.skip(f"Synthetic {codec} codec unavailable")
    assert writer.isOpened(), f"Cannot write synthetic {codec} clip"
    try:
        for frame in chain((first,), frames):
            writer.write(frame)
    finally:
        writer.release()


def calibrated_inputs(tmp_path):
    clip = tmp_path / "clip.avi"
    write_clip(clip, (np.full((8, 8, 3), 30 + i * 20, dtype=np.uint8) for i in range(4)),
               codec="MJPG", fps=5, skip_unavailable=True)
    population = tmp_path / "calibration.json"
    atomic_write_json(population, {"calibration_version": CALIBRATED_INPUT_VERSION})
    paths = {}
    for split in ("train", "val"):
        rows = []
        for index in range(2):
            rows.append(
                {
                    **{field: "0" for field in FUSION_EVIDENCE_FIELDS},
                    "event_id": f"{split}-{index}",
                    "video_id": f"{split}-{index}",
                    "source_group": f"{split}-{index}",
                    "dataset_split": split,
                    "class_id": str(index),
                    "phoneme": "p",
                    "mouth_clip_path": clip.as_posix(),
                    "manipulation_modality": "real" if index == 0 else "both_modified",
                    "closure_time_s": "0.2",
                    "video_phone_start_s": "0.21",
                    "calibration_version": CALIBRATED_INPUT_VERSION,
                    "isolation_forest_available": "True" if index == 0 else "False",
                    "isolation_forest_anomaly_score": "0" if index == 0 else "",
                    "isolation_forest_scope": "input_video" if index == 0 else "unavailable",
                    "isolation_forest_unavailable_reason": (
                        "" if index == 0 else "insufficient_video_reference_frames"
                    ),
                }
            )
        paths[split] = tmp_path / f"events_{split}_calibrated.csv"
        atomic_write_csv(paths[split], rows)
    atomic_write_json(
        tmp_path / "summary.json",
        {
            "scored_manifests": {k: p.as_posix() for k, p in paths.items()},
            "scored_manifest_sha256": {k: file_sha256(p) for k, p in paths.items()},
            "calibration": population.as_posix(),
            "calibration_sha256": file_sha256(population),
        },
    )
    return paths

