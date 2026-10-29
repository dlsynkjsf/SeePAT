"""Exact selected-frame decoding with an optional content-addressed disk cache.

Only RGB pixels, frame masks and decoded counts are cached. Targets, evidence
and split membership continue to come from the current verified manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

from seepat.artifacts import file_sha256

CLIP_CACHE_VERSION = "selected-rgb-clip-v1"
SELECTION_POLICY = "round-uniform-endpoints; short-repeat-final; singleton-middle-v1"


def sequence_frame_indices(frame_count: int, sequence_length: int) -> tuple[list[int], list[bool]]:
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if frame_count >= sequence_length:
        if sequence_length == 1:
            return [frame_count // 2], [True]
        return [round(i * (frame_count - 1) / (sequence_length - 1))
                for i in range(sequence_length)], [True] * sequence_length
    padding = sequence_length - frame_count
    return (list(range(frame_count)) + [frame_count - 1] * padding,
            [True] * frame_count + [False] * padding)


def _plan(frame_count: int, sequence_length: int, context_frame_mask):
    if context_frame_mask is not None:
        if frame_count != 16:
            raise ValueError("Context decoder must retain exactly 16 stored slots")
        return list(range(16)), list(context_frame_mask)
    return sequence_frame_indices(frame_count, sequence_length)


def _read_selected(path, sequence_length, image_size, context_frame_mask):
    import cv2
    import numpy as np

    def decode(planned_count=None):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open mouth clip: {path}")
        # Container counts are advisory only. A mismatch triggers a sequential
        # second pass using the actual readable count, never arbitrary seeking.
        if planned_count is None:
            hint = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            planned_count = int(hint) if math.isfinite(hint) and hint >= 1 else 0
        if context_frame_mask is not None:
            planned_count = 16
        indices = _plan(planned_count, sequence_length, context_frame_mask)[0] if planned_count else []
        wanted = set(indices)
        selected, actual_count = {}, 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if actual_count in wanted:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    selected[actual_count] = cv2.resize(rgb, (image_size, image_size))
                actual_count += 1
        finally:
            capture.release()
        return selected, actual_count, planned_count

    frames, count, planned = decode()
    if count < 1:
        raise RuntimeError(f"Mouth clip contains no readable frames: {path}")
    indices, mask = _plan(count, sequence_length, context_frame_mask)
    if count != planned:
        frames, repeated_count, _ = decode(count)
        if repeated_count != count:
            raise RuntimeError(f"Mouth clip decoded count changed between passes: {path}")
    array = np.ascontiguousarray(np.stack([frames[index] for index in indices]))
    mask = np.asarray(mask, dtype=np.bool_)
    if context_frame_mask is not None:
        array[~mask] = 0
    return array, mask, count


def _contract(clip_sha256, sequence_length, image_size, context_frame_mask):
    import cv2

    return {
        "version": CLIP_CACHE_VERSION,
        "clip_sha256": clip_sha256,
        "opencv_version": cv2.__version__,
        "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        "image_size": image_size,
        "sequence_length": sequence_length,
        "selection_policy": SELECTION_POLICY,
        "pixels": "uint8 RGB; cvtColor BGR2RGB then resize INTER_LINEAR",
        "context_frame_mask": context_frame_mask,
    }


def _payload_hash(array, mask, count):
    digest = hashlib.sha256()
    digest.update(array.tobytes(order="C"))
    digest.update(mask.tobytes(order="C"))
    digest.update(str(count).encode("ascii"))
    return digest.hexdigest()


def _load_entry(path, contract):
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as entry:
            if set(entry.files) != {"rgb", "frame_mask", "decoded_count", "metadata"}:
                return None
            array, mask, count = entry["rgb"], entry["frame_mask"], entry["decoded_count"]
            metadata = json.loads(entry["metadata"].item())
            if count.shape != () or count.dtype != np.dtype("int64"):
                return None
            count = int(count.item())
            if count < 1:
                return None
            _, expected_mask = _plan(count, contract["sequence_length"], contract["context_frame_mask"])
            size, length = contract["image_size"], contract["sequence_length"]
            if (array.dtype != np.uint8 or array.shape != (length, size, size, 3)
                    or mask.dtype != np.bool_ or mask.shape != (length,)
                    or mask.tolist() != expected_mask or not array.flags.c_contiguous
                    or metadata != {"contract": contract,
                                    "payload_sha256": _payload_hash(array, mask, count)}):
                return None
            if contract["context_frame_mask"] is not None and array[~mask].any():
                return None
            return array, mask, count
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        # Incomplete/old/corrupt caches are replaceable derived data. A source
        # hash mismatch is checked separately and must never become a cache hit.
        return None


def _write_entry(path, contract, array, mask, count):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, rgb=array, frame_mask=mask,
                     decoded_count=np.asarray(count, dtype=np.int64),
                     metadata=np.asarray(json.dumps({"contract": contract,
                         "payload_sha256": _payload_hash(array, mask, count)}, sort_keys=True)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_clip_array(path: Path, sequence_length: int, image_size: int,
                    cache_dir: Path | None = None, *, expected_sha256: str | None = None,
                    context_frame_mask: list[bool] | None = None):
    """Return selected uint8 RGB frames, bool mask and actual decoded count.

    Source bytes are rehashed on every cached access. Context inputs additionally
    enforce their already-frozen source hash and retain all 16 stored slots.
    """
    if sequence_length < 1 or image_size < 1:
        raise ValueError("sequence_length and image_size must be positive")
    if context_frame_mask is not None and (
        sequence_length != 16 or len(context_frame_mask) != 16
        or any(type(v) is not bool for v in context_frame_mask) or not any(context_frame_mask)
    ):
        raise ValueError("Context frame mask requires 16 boolean slots with an observed frame")
    source_hash = file_sha256(path) if cache_dir is not None or expected_sha256 is not None else None
    if expected_sha256 is not None and source_hash != expected_sha256:
        raise ValueError("Context clip hash mismatch")
    contract, target = None, None
    if cache_dir is not None:
        contract = _contract(source_hash, sequence_length, image_size, context_frame_mask)
        key = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        target = Path(cache_dir) / key[:2] / f"{key}.npz"
        cached = _load_entry(target, contract)
        if cached is not None:
            return cached
    result = _read_selected(path, sequence_length, image_size, context_frame_mask)
    if target is not None:
        if file_sha256(path) != source_hash:
            raise ValueError("Clip source changed while creating its decoded cache")
        _write_entry(target, contract, *result)
    return result
