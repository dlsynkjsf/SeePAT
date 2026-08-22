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
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

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

        start_s = max(0.0, phone_start_s - window_before_s)
        end_s = max(start_s, phone_end_s + window_after_s)
        start_frame = max(0, math.floor(start_s * fps))
        end_frame = max(start_frame, math.ceil(end_s * fps))
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
                result = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                faces = result.multi_face_landmarks or []

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

                if len(faces) != 1:
                    if len(faces) > 1:
                        multiple_face_frames += 1
                    if overlay_writer is not None:
                        cv2.putText(
                            frame,
                            f"/{phoneme}/ faces={len(faces)} VILD=NA",
                            (7, 17),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (0, 0, 255),
                            1,
                        )
                        overlay_writer.write(frame)
                    frame_index += 1
                    continue

                landmarks = faces[0].landmark
                points = {
                    index: (landmarks[index].x * width, landmarks[index].y * height)
                    for index in LIP_LANDMARKS
                }
                mouth_width = _distance(
                    points[LEFT_MOUTH_CORNER], points[RIGHT_MOUTH_CORNER]
                )
                if mouth_width <= 1.0:
                    frame_index += 1
                    continue
                vild = _distance(points[INNER_UPPER_LIP], points[INNER_LOWER_LIP]) / mouth_width
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
            "mouth_crop_path": str(mouth_clip_path),
            "overlay_path": str(overlay_path) if overlay_path is not None else None,
            "minimum_overlay_path": (
                str(minimum_overlay_path) if minimum_overlay_path is not None else None
            ),
        }
