from __future__ import annotations

import math
from pathlib import Path
from typing import Self

INNER_UPPER_LIP = 13
INNER_LOWER_LIP = 14
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
LIP_LANDMARKS = (
    0,
    13,
    14,
    17,
    37,
    39,
    40,
    61,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    95,
    146,
    178,
    181,
    185,
    191,
    267,
    269,
    270,
    291,
    308,
    310,
    311,
    312,
    314,
    317,
    318,
    321,
    324,
    375,
    402,
    405,
    409,
    415,
)


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _velocity(delta: float, elapsed_s: float) -> float | None:
    if elapsed_s <= 0:
        return None
    return delta / elapsed_s


def timestamp_to_frame(timestamp_s: float, fps: float, round_up: bool = False) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    scaled = max(0.0, timestamp_s) * fps
    return math.ceil(scaled) if round_up else math.floor(scaled)


def event_frame_bounds(
    phone_start_s: float,
    phone_end_s: float,
    fps: float,
    window_before_s: float,
    window_after_s: float,
) -> tuple[int, int]:
    if phone_end_s < phone_start_s:
        raise ValueError("phone_end_s cannot precede phone_start_s")
    if window_before_s < 0 or window_after_s < 0:
        raise ValueError("event-window durations cannot be negative")

    start_s = max(0.0, phone_start_s - window_before_s)
    end_s = max(start_s, phone_end_s + window_after_s)
    return (
        timestamp_to_frame(start_s, fps),
        timestamp_to_frame(end_s, fps, round_up=True),
    )


def normalized_vild(
    upper_lip: tuple[float, float],
    lower_lip: tuple[float, float],
    left_corner: tuple[float, float],
    right_corner: tuple[float, float],
    minimum_mouth_width: float = 1.0,
) -> float | None:
    mouth_width = _distance(left_corner, right_corner)
    if mouth_width <= minimum_mouth_width:
        return None
    return _distance(upper_lip, lower_lip) / mouth_width


class MouthEventAnalyzer:
    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                'Install preprocessing dependencies with: pip install -e ".[preprocess]"'
            ) from error

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This MediaPipe build does not expose solutions.face_mesh; "
                "install a compatible 0.10.x build."
            )
        self.cv2 = cv2
        self.mp = mp
        self.min_detection_confidence = min_detection_confidence
        self.face_mesh = self._create_face_mesh()

    def _create_face_mesh(self):
        return self.mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def _measure_frame(self, frame, face_mesh=None):
        mesh = face_mesh or self.face_mesh
        height, width = frame.shape[:2]
        result = mesh.process(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
        faces = result.multi_face_landmarks or []
        if len(faces) != 1:
            return len(faces), None, None, None, None
        landmarks = faces[0].landmark
        points = {
            index: (landmarks[index].x * width, landmarks[index].y * height)
            for index in LIP_LANDMARKS
        }
        all_points = [(landmark.x * width, landmark.y * height) for landmark in landmarks]
        x_values, y_values = zip(*all_points, strict=True)
        face_bbox_size_px = math.hypot(max(x_values) - min(x_values), max(y_values) - min(y_values))
        raw_vild_px = math.dist(points[INNER_UPPER_LIP], points[INNER_LOWER_LIP])
        vild = normalized_vild(
            points[INNER_UPPER_LIP],
            points[INNER_LOWER_LIP],
            points[LEFT_MOUTH_CORNER],
            points[RIGHT_MOUTH_CORNER],
        )
        return 1, vild, points, raw_vild_px, face_bbox_size_px

    def trace_video(self, video_path: Path, fps: float) -> dict[str, object]:
        """Measure normalized VILD once per decoded frame for later feature work."""
        if fps <= 0:
            raise ValueError("fps must be positive")
        cv2 = self.cv2
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open {video_path}")

        frames: list[dict[str, object]] = []
        valid_frames = 0
        multiple_face_frames = 0
        opencv_timestamp_frames = 0
        fallback_timestamp_frames = 0
        last_timestamp = -math.inf
        trace_mesh = self._create_face_mesh()
        try:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                position_s = position_ms / 1000
                if (
                    math.isfinite(position_s)
                    and position_s >= 0
                    and (frame_index == 0 or position_s > last_timestamp)
                ):
                    timestamp_s = position_s
                    opencv_timestamp_frames += 1
                else:
                    timestamp_s = frame_index / fps
                    if frame_index > 0 and timestamp_s <= last_timestamp:
                        timestamp_s = last_timestamp + 1 / fps
                    fallback_timestamp_frames += 1
                last_timestamp = timestamp_s

                face_count, vild, _, raw_vild_px, face_bbox_size_px = self._measure_frame(
                    frame, trace_mesh
                )
                if face_count > 1:
                    multiple_face_frames += 1
                if vild is not None:
                    valid_frames += 1
                frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_s": round(timestamp_s, 9),
                        "face_count": face_count,
                        "normalized_vild": round(vild, 9) if vild is not None else None,
                        "raw_vild_px": round(raw_vild_px, 9) if raw_vild_px is not None else None,
                        "face_bbox_size_px": (
                            round(face_bbox_size_px, 9)
                            if face_bbox_size_px is not None
                            else None
                        ),
                    }
                )
                frame_index += 1
        finally:
            capture.release()
            trace_mesh.close()

        attempted_frames = len(frames)
        return {
            "frames": frames,
            "summary": {
                "attempted_frames": attempted_frames,
                "valid_frames": valid_frames,
                "valid_landmark_ratio": (
                    valid_frames / attempted_frames if attempted_frames else 0.0
                ),
                "multiple_face_ratio": (
                    multiple_face_frames / attempted_frames if attempted_frames else 0.0
                ),
                "opencv_timestamp_frames": opencv_timestamp_frames,
                "fps_fallback_timestamp_frames": fallback_timestamp_frames,
            },
        }

    def close(self) -> None:
        self.face_mesh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def analyze(
        self,
        video_path: Path,
        phone_start_s: float,
        phone_end_s: float,
        phoneme: str,
        fps: float,
        window_before_s: float,
        window_after_s: float,
        mouth_clip_path: Path,
        overlay_path: Path | None,
    ) -> dict[str, object]:
        cv2 = self.cv2
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open {video_path}")

        start_frame, end_frame = event_frame_bounds(
            phone_start_s,
            phone_end_s,
            fps,
            window_before_s,
            window_after_s,
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        mouth_clip_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_mouth = mouth_clip_path.with_name(
            f".{mouth_clip_path.stem}.tmp{mouth_clip_path.suffix}"
        )
        mouth_writer = cv2.VideoWriter(
            str(temporary_mouth),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (112, 112),
        )
        overlay_writer = None
        temporary_overlay = None

        measurements: list[tuple[float, float]] = []
        attempted_frames = 0
        valid_frames = 0
        multiple_face_frames = 0
        mouth_crop_frame_indices: list[int] = []
        minimum_frame_image = None
        minimum_seen = math.inf

        try:
            frame_index = start_frame
            while frame_index <= end_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                attempted_frames += 1
                timestamp_s = frame_index / fps
                height, width = frame.shape[:2]
                face_count, vild, points, _, _ = self._measure_frame(frame)

                if overlay_path is not None and overlay_writer is None:
                    overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary_overlay = overlay_path.with_name(
                        f".{overlay_path.stem}.tmp{overlay_path.suffix}"
                    )
                    overlay_writer = cv2.VideoWriter(
                        str(temporary_overlay),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )

                if face_count != 1:
                    if face_count > 1:
                        multiple_face_frames += 1
                    if overlay_writer is not None:
                        cv2.putText(
                            frame,
                            f"/{phoneme}/ faces={face_count} VILD=NA",
                            (7, 17),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (0, 0, 255),
                            1,
                        )
                        overlay_writer.write(frame)
                    frame_index += 1
                    continue

                if vild is None or points is None:
                    frame_index += 1
                    continue
                measurements.append((timestamp_s, vild))
                valid_frames += 1

                x_values = [point[0] for point in points.values()]
                y_values = [point[1] for point in points.values()]
                center_x = (min(x_values) + max(x_values)) / 2
                center_y = (min(y_values) + max(y_values)) / 2
                side = max(max(x_values) - min(x_values), max(y_values) - min(y_values)) * 1.8
                side = max(side, 24.0)
                x1 = max(0, int(center_x - side / 2))
                y1 = max(0, int(center_y - side / 2))
                x2 = min(width, int(center_x + side / 2))
                y2 = min(height, int(center_y + side / 2))
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    mouth_writer.write(cv2.resize(crop, (112, 112)))
                    mouth_crop_frame_indices.append(frame_index)

                if overlay_writer is not None:
                    for point in points.values():
                        cv2.circle(frame, (round(point[0]), round(point[1])), 1, (0, 255, 0), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
                    in_phone = phone_start_s <= timestamp_s <= phone_end_s
                    cv2.putText(
                        frame,
                        f"/{phoneme}/ t={timestamp_s:.3f}s VILD={vild:.4f}",
                        (7, 17),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 255, 0),
                        1,
                    )
                    cv2.putText(
                        frame,
                        f"aligned interval active={in_phone}",
                        (7, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 255, 0),
                        1,
                    )
                    overlay_writer.write(frame)
                    if vild < minimum_seen:
                        minimum_seen = vild
                        minimum_frame_image = frame.copy()
                frame_index += 1
        finally:
            capture.release()
            mouth_writer.release()
            if overlay_writer is not None:
                overlay_writer.release()

        if measurements:
            temporary_mouth.replace(mouth_clip_path)
        elif temporary_mouth.exists():
            temporary_mouth.unlink()
        if overlay_path is not None and temporary_overlay is not None:
            if measurements and temporary_overlay.exists():
                temporary_overlay.replace(overlay_path)
            elif temporary_overlay.exists():
                temporary_overlay.unlink()

        minimum_overlay_path = None
        if overlay_path is not None and minimum_frame_image is not None:
            minimum_overlay_path = overlay_path.with_name(
                f"{overlay_path.stem}_minimum.png"
            )
            cv2.putText(
                minimum_frame_image,
                "MINIMUM CLOSURE",
                (7, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 0, 255),
                1,
            )
            cv2.imwrite(str(minimum_overlay_path), minimum_frame_image)

        valid_ratio = valid_frames / attempted_frames if attempted_frames else 0.0
        if not measurements:
            return {
                "attempted_frames": attempted_frames,
                "valid_landmark_ratio": valid_ratio,
                "multiple_face_ratio": (
                    multiple_face_frames / attempted_frames if attempted_frames else 0.0
                ),
                "normalized_minimum_closure": None,
                "closure_time_s": None,
                "closure_duration_s": None,
                "closing_velocity": None,
                "opening_velocity": None,
                "mouth_crop_frame_indices": [],
                "mouth_crop_path": None,
                "overlay_path": None,
                "minimum_overlay_path": None,
            }

        minimum_index = min(range(len(measurements)), key=lambda index: measurements[index][1])
        minimum_time, minimum_vild = measurements[minimum_index]
        maximum_vild = max(value for _, value in measurements)
        closure_cutoff = minimum_vild + 0.10 * (maximum_vild - minimum_vild)
        closure_frames = sum(value <= closure_cutoff for _, value in measurements)
        frame_duration = 1 / fps
        first_time, first_vild = measurements[0]
        last_time, last_vild = measurements[-1]

        return {
            "attempted_frames": attempted_frames,
            "valid_landmark_ratio": valid_ratio,
            "multiple_face_ratio": (
                multiple_face_frames / attempted_frames if attempted_frames else 0.0
            ),
            "normalized_minimum_closure": minimum_vild,
            "closure_time_s": minimum_time,
            "closure_duration_s": closure_frames * frame_duration,
            "closing_velocity": _velocity(first_vild - minimum_vild, minimum_time - first_time),
            "opening_velocity": _velocity(last_vild - minimum_vild, last_time - minimum_time),
            "mouth_crop_frame_indices": mouth_crop_frame_indices,
            "mouth_crop_path": str(mouth_clip_path),
            "overlay_path": str(overlay_path) if overlay_path is not None else None,
            "minimum_overlay_path": (
                str(minimum_overlay_path) if minimum_overlay_path is not None else None
            ),
        }
