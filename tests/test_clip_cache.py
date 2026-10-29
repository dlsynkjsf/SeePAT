"""Pixel parity, bounded selection, provenance and invalidation for decoded caches."""

import cv2
import numpy as np
import pytest

from seepat.artifacts import atomic_write_csv, file_sha256
from seepat.training import clip_cache
from seepat.training.dataset import FEATURE_FIELDS, MouthEventDataset, sequence_frame_indices


def write_clip(path, count, offset=0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (24, 20))
    assert writer.isOpened()
    yy, xx = np.indices((20, 24))
    for index in range(count):
        image = np.stack(((xx * 7 + offset + index * 3) % 256,
                          (yy * 9 + index * 11) % 256,
                          ((xx + yy) * 5 + index * 17) % 256), axis=-1).astype(np.uint8)
        writer.write(image)
    writer.release()


def legacy_pixels(path, length, size):
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (size, size)))
    finally:
        capture.release()
    indices, mask = sequence_frame_indices(len(frames), length)
    return np.ascontiguousarray(np.stack([frames[i] for i in indices])), np.array(mask), len(frames)


@pytest.mark.parametrize("count,length,size", [(3, 16, 8), (16, 16, 12), (37, 16, 9), (8, 1, 8)])
def test_selected_decoder_matches_legacy_pixels_exactly(tmp_path, count, length, size, monkeypatch):
    path = tmp_path / "clip.mp4"
    write_clip(path, count)
    expected = legacy_pixels(path, length, size)
    resize = cv2.resize
    calls = []

    def counted_resize(*args, **kwargs):
        calls.append(1)
        return resize(*args, **kwargs)

    monkeypatch.setattr(cv2, "resize", counted_resize)
    actual = clip_cache.read_clip_array(path, length, size)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]
    assert len(calls) == min(count, length)


@pytest.mark.parametrize("reported_count", [0, 999])
def test_container_count_is_advisory_and_fallback_remains_exact(tmp_path, monkeypatch, reported_count):
    path = tmp_path / "clip.mp4"
    write_clip(path, 9)
    expected = legacy_pixels(path, 4, 8)
    capture_class = cv2.VideoCapture

    class IncorrectCount:
        def __init__(self, name):
            self.capture = capture_class(name)

        def isOpened(self):
            return self.capture.isOpened()

        def get(self, key):
            return reported_count if key == cv2.CAP_PROP_FRAME_COUNT else self.capture.get(key)

        def read(self):
            return self.capture.read()

        def release(self):
            self.capture.release()

    monkeypatch.setattr(cv2, "VideoCapture", IncorrectCount)
    actual = clip_cache.read_clip_array(path, 4, 8)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_warm_cache_avoids_decode_and_keeps_uncompressed_uint8(tmp_path, monkeypatch):
    path = tmp_path / "clip.mp4"
    write_clip(path, 7)
    cache = tmp_path / "cache"
    first = clip_cache.read_clip_array(path, 16, 8, cache)
    entry, = cache.rglob("*.npz")
    with np.load(entry, allow_pickle=False) as saved:
        assert saved["rgb"].dtype == np.uint8
        assert saved["rgb"].shape == (16, 8, 8, 3)

    def no_decode(*_args):
        raise AssertionError("Warm cache reopened its decoder")

    monkeypatch.setattr(clip_cache, "_read_selected", no_decode)
    second = clip_cache.read_clip_array(path, 16, 8, cache)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_source_size_and_policy_changes_have_distinct_entries(tmp_path, monkeypatch):
    path = tmp_path / "clip.mp4"
    write_clip(path, 5)
    cache = tmp_path / "cache"
    old = clip_cache.read_clip_array(path, 16, 8, cache)[0]
    clip_cache.read_clip_array(path, 16, 9, cache)
    clip_cache.read_clip_array(path, 8, 8, cache)
    monkeypatch.setattr(clip_cache, "SELECTION_POLICY", "new-test-selection-version")
    clip_cache.read_clip_array(path, 16, 8, cache)
    write_clip(path, 5, offset=80)
    changed = clip_cache.read_clip_array(path, 16, 8, cache)[0]
    assert not np.array_equal(old, changed)
    assert len(list(cache.rglob("*.npz"))) == 5


@pytest.mark.parametrize("corruption", ["truncated", "pixels", "count", "mask", "metadata"])
def test_corrupt_entries_are_regenerated_from_verified_source(tmp_path, monkeypatch, corruption):
    path = tmp_path / "clip.mp4"
    write_clip(path, 3)
    cache = tmp_path / "cache"
    expected = clip_cache.read_clip_array(path, 16, 8, cache)
    entry, = cache.rglob("*.npz")
    if corruption == "truncated":
        entry.write_bytes(b"broken zip")
    else:
        with np.load(entry, allow_pickle=False) as stored:
            values = {key: stored[key] for key in stored.files}
        if corruption == "pixels":
            values["rgb"][0, 0, 0, 0] ^= 1
        elif corruption == "count":
            values["decoded_count"] = np.asarray(-1, dtype=np.int64)
        elif corruption == "mask":
            values["frame_mask"][:] = True
        else:
            values["metadata"] = np.asarray("{}")
        np.savez(entry, **values)
    original_decode = clip_cache._read_selected
    calls = []

    def counted(*args):
        calls.append(1)
        return original_decode(*args)

    monkeypatch.setattr(clip_cache, "_read_selected", counted)
    regenerated = clip_cache.read_clip_array(path, 16, 8, cache)
    assert len(calls) == 1
    assert np.array_equal(regenerated[0], expected[0])
    assert np.array_equal(regenerated[1], expected[1])
    assert regenerated[2] == expected[2]


def test_context_cache_keeps_slots_zeroing_and_frozen_hash(tmp_path, monkeypatch):
    path = tmp_path / "context.mp4"
    write_clip(path, 16)
    cache = tmp_path / "cache"
    mask = [True] * 15 + [False]
    expected_hash = file_sha256(path)
    actual = clip_cache.read_clip_array(path, 16, 8, cache,
        expected_sha256=expected_hash, context_frame_mask=mask)
    assert actual[2] == 16
    assert actual[1].tolist() == mask
    assert not actual[0][-1].any() and actual[0][0].any()
    # The same source with another frozen mask must not reuse the first pixels.
    other_mask = [False] + [True] * 15
    other = clip_cache.read_clip_array(path, 16, 8, cache,
        expected_sha256=expected_hash, context_frame_mask=other_mask)
    assert not other[0][0].any() and other[0][-1].any()
    assert len(list(cache.rglob("*.npz"))) == 2
    write_clip(path, 16, offset=100)
    with pytest.raises(ValueError, match="Context clip hash mismatch"):
        clip_cache.read_clip_array(path, 16, 8, cache,
            expected_sha256=expected_hash, context_frame_mask=mask)


def test_context_requires_exactly_sixteen_decoded_slots(tmp_path):
    path = tmp_path / "short.mp4"
    write_clip(path, 15)
    with pytest.raises(ValueError, match="exactly 16"):
        clip_cache.read_clip_array(path, 16, 8, tmp_path / "cache",
            expected_sha256=file_sha256(path), context_frame_mask=[True] * 16)


def test_dataset_cache_preserves_labels_evidence_and_float_pixels(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "clip.mp4"
    write_clip(path, 3)
    manifest = tmp_path / "events.csv"
    row = {"event_id": "event", "video_id": "video", "dataset_split": "train",
           "source_group": "source", "manipulation_modality": "visual_modified",
           "phoneme": "p", "class_id": "0", "mouth_clip_path": path.name,
           **{field: "0.25" for field in FEATURE_FIELDS}}
    atomic_write_csv(manifest, [row])
    original = MouthEventDataset(manifest, tmp_path, image_size=8)[0]
    cached = MouthEventDataset(manifest, tmp_path, image_size=8, clip_cache_dir=tmp_path / "cache")[0]
    assert original.keys() == cached.keys()
    for key, value in original.items():
        assert torch.equal(value, cached[key]) if torch.is_tensor(value) else value == cached[key]
    assert cached["label"].item() == 0 and cached["video_label"].item() == 1
    # Pixel cache never supplies stale targets/numerical evidence.
    atomic_write_csv(manifest, [{**row, "class_id": "1", "closure_time_s": "0.75"}])
    changed = MouthEventDataset(manifest, tmp_path, image_size=8, clip_cache_dir=tmp_path / "cache")[0]
    assert changed["label"].item() == 1
    assert changed["features"][FEATURE_FIELDS.index("closure_time_s")].item() == 0.75
    assert torch.equal(changed["video"], cached["video"])
