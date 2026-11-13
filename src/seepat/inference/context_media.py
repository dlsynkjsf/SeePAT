"""Stateless mouth-region cropping for context clips."""
from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

from seepat.artifacts import file_sha256
from seepat.preprocessing.face import MouthEventAnalyzer
from seepat.preprocessing.face import mouth_crop_bounds as crop_bounds


def decode_needed(path: Path, needed: set[int], expected_count: int, progress) -> tuple[dict, int]:
    """Sequential decoding avoids unverified random-seek frame indices; bounded to 8 videos."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Cannot open selected source media")
    frames, index = {}, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in needed:
                frames[index] = (frame, float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
            index += 1
            if index % 100 == 0:
                progress("decode selected source video", index, expected_count, path.name)
    finally:
        capture.release()
    if set(frames) != needed:
        raise ValueError("Decoded source does not contain every planned frame")
    return frames, index


class StatelessMouthEventAnalyzer(MouthEventAnalyzer):
    """Use the existing detector/landmark guard, without carrying previous-frame ROIs."""

    def _create_face_mesh(self):
        return self.mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.5,
        )


class CaptureMesh:
    """Observe the real mesh result without rerunning inference or selecting a face."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.result = None

    def process(self, image):
        self.result = self.mesh.process(image)
        return self.result


def face_boxes(result, width: int, height: int) -> list[list[float]]:
    faces = getattr(result, "multi_face_landmarks", None) or []
    boxes = []
    for face in faces:
        xs = [p.x * width for p in face.landmark]
        ys = [p.y * height for p in face.landmark]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return boxes


def fixed_window_bounds(point_sets: list[dict], width: int, height: int):
    """One 1.8x square around the union of valid lip estimates in a context window."""
    if not point_sets:
        return None
    points = {index: point for index, point in enumerate(
        point for points in point_sets for point in points.values())}
    if any(not math.isfinite(float(v)) for point in points.values() for v in point):
        raise ValueError("Nonfinite mouth landmarks")
    bounds = crop_bounds(points, width, height)
    return bounds if bounds[2] > bounds[0] and bounds[3] > bounds[1] else None


def window_record(video: dict, window: dict, decoded: dict, trace_frames: dict,
                  analyzer, directory: Path) -> dict:
    """Same stateless inference and fixed union crop as accepted v3, without reels."""
    import cv2
    import numpy as np

    identifier = video["video_id"]
    fps = float(video["fps"])
    start, stop = window["start_frame"], window["stop_frame_exclusive"]
    measured = {}
    mesh = analyzer._create_face_mesh()
    capture = CaptureMesh(mesh)
    try:
        for index in range(start, stop):
            frame, _ = decoded[index]
            faces, vild, points, _, _ = analyzer._measure_frame(frame, capture)
            measured[index] = {"faces": faces,
                               "points": points if faces == 1 and vild is not None and points else None,
                               "face_boxes": face_boxes(capture.result, frame.shape[1], frame.shape[0])}
    finally:
        mesh.close()
    height, width = decoded[start][0].shape[:2]
    bounds = fixed_window_bounds([row["points"] for row in measured.values() if row["points"]],
                                    width, height)
    slots, crops = [], []
    for index, planned in zip(window["source_frames"], window["frame_mask"], strict=True):
        frame, pts = decoded[index]
        trace_pts = float(trace_frames[index]["timestamp_s"])
        item = measured[index]
        timing_ok = (math.isfinite(pts) and pts >= 0 and math.isfinite(trace_pts)
                     and abs(pts - index / fps) <= 1 / fps + 1e-6
                     and abs(pts - trace_pts) <= 1 / fps + 1e-6)
        valid = bool(planned and item["points"] is not None and bounds is not None)
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        if valid:
            x1, y1, x2, y2 = bounds
            crop = cv2.resize(frame[y1:y2, x1:x2], (112, 112))
        crops.append(crop)
        slots.append({"source_frame": index, "planned_valid": planned, "crop_valid": valid,
                      "face_count": item["faces"], "detected_face_boxes": item["face_boxes"],
                      "decoder_pts_s": pts if math.isfinite(pts) else None,
                      "trace_pts_s": trace_pts if math.isfinite(trace_pts) else None,
                      "timing_ok": timing_ok})
    fraction, multiple, usable = window_status(slots)
    context_id = f"{identifier}-context-{window['slot']:03d}"
    record = {"context_id": context_id, "video_id": identifier, "dataset_split": video["split"],
              "source_group": video["source_group"], "kind": "context",
              "start_frame": start, "stop_frame_exclusive": stop,
              "source_frames": window["source_frames"], "planned_mask": window["frame_mask"],
              "frame_mask": [row["crop_valid"] for row in slots], "slots": slots,
              "window_bbox": bounds, "usable_frame_fraction": fraction,
              "multiple_face_slots": multiple, "technical_ok": usable,
              "clip_file": "", "clip_sha256": "",
              "evidence_available": False, "phoneme": "", "event_class_id": ""}
    if usable:
        clip = directory / f"{context_id}.mp4"
        write_clip(clip, crops, fps)
        record.update(clip_file=clip.name, clip_sha256=file_sha256(clip))
    return record


def window_status(slots: list[dict]) -> tuple[float, int, bool]:
    active = [row for row in slots if row["planned_valid"]]
    if not active:
        raise ValueError("Context window has no nonpadding slots")
    fraction = sum(row["crop_valid"] for row in active) / len(active)
    multiple = sum(row["face_count"] > 1 for row in active)
    monotonic = all(a["decoder_pts_s"] is not None and b["decoder_pts_s"] is not None
                    and b["decoder_pts_s"] > a["decoder_pts_s"] for a, b in pairwise(active))
    usable = (fraction >= .8 and multiple == 0 and monotonic
              and all(row["timing_ok"] for row in active))
    return fraction, multiple, usable


def write_clip(path: Path, crops: list, fps: float) -> None:
    """Same mp4v/112px storage family as original bilabial clips; exactly 16 slots."""
    import cv2

    if len(crops) != 16 or path.exists():
        raise ValueError("Context encoding requires 16 slots and a fresh destination")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (112, 112))
    if not writer.isOpened():
        raise RuntimeError("Cannot create context clip")
    try:
        for crop in crops:
            writer.write(crop)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (112, 112, 3):
                raise ValueError("Encoded context dimensions changed")
            count += 1
    finally:
        capture.release()
    if count != 16:
        raise ValueError("Encoded context must retain all 16 slots, including masked slots")
