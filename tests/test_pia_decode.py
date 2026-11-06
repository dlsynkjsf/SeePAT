import numpy as np
import pytest

from scripts import pia_cpu_consumed_runtime as original
from scripts.pia_cpu_decode_runtime import selected_embeddings


class Capture:
    def __init__(self, count=10):
        self.index, self.count, self.retrieved, self.closed = -1, count, 0, False

    def isOpened(self):
        return not self.closed

    def grab(self):
        self.index += 1
        return self.index < self.count

    def retrieve(self):
        self.retrieved += 1
        return True, self.index

    def read(self):
        return self.retrieve() if self.grab() else (False, None)

    def release(self):
        self.closed = True


def test_sparse_decode_preserves_selected_frames_without_materializing_discarded_frames():
    capture = Capture()
    values, decoded = selected_embeddings(capture, lambda frame: [frame] * 512, [0, 4, 9])
    assert values == {index: [index] * 512 for index in [0, 4, 9]} and decoded == 10
    assert capture.retrieved == 3 and capture.closed


def test_generated_video_selected_pixels_equal_read_reference(tmp_path):
    import cv2

    path = str(tmp_path / 'synthetic.mp4')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (96, 64))
    assert writer.isOpened()
    for index in range(40):
        writer.write(np.full((64, 96, 3), index * 6, dtype=np.uint8))
    writer.release()
    indices = [0, 4, 19, 39]
    reference, count = original.selected_embeddings(cv2.VideoCapture(path), lambda frame: frame.copy(), indices)
    actual, decoded = selected_embeddings(cv2.VideoCapture(path), lambda frame: frame.copy(), indices)
    assert count == decoded == 40 and actual.keys() == reference.keys()
    assert all(np.array_equal(actual[index], reference[index]) for index in indices)


def test_sparse_decode_short_empty_invalid_and_failed_retrieval_release_capture():
    for indices in ([], [0, 9]):
        capture = Capture(2)
        assert selected_embeddings(capture, lambda frame: frame, indices)[0] == ({} if not indices else {0:0})
        assert capture.closed
    capture = Capture()
    with pytest.raises(ValueError):
        selected_embeddings(capture, lambda frame: frame, [3, 1])
    assert capture.closed
    capture = Capture()
    capture.retrieve = lambda: (False, None)
    with pytest.raises(RuntimeError, match='could not be retrieved'):
        selected_embeddings(capture, lambda frame: frame, [0])
    assert capture.closed
