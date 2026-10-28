import pytest
import torch

from seepat.artifacts import atomic_write_csv, atomic_write_gzip_json, file_sha256
from seepat.training import sequence_inputs as inputs
from seepat.training.context_dataset import ExpandedContextDataset
from seepat.training.dataset import MouthEventDataset


class PreparedClips:
    def __init__(self, root, prefix="train", label=0):
        self.calibration_contract = {"version": "synthetic-frozen-calibration"}
        self.root = root
        self.rows, self.samples = [], []
        for index, kind in enumerate(("bilabial", "context")):
            clip = root / f"{prefix}-{index}.mp4"
            clip.write_bytes(b"synthetic fixture; injected decoder does not open this")
            row = {
                "event_id": f"{prefix}-{index}", "video_id": prefix, "file": f"{prefix}.mp4",
                "source_group": prefix, "dataset_split": "train" if prefix == "train" else "val",
                "kind": kind, "class_id": "0" if index == 0 else "-1",
                "phoneme": "p" if index == 0 else "", "mouth_clip_path": clip.name,
                "manipulation_modality": "real" if not label else "both_modified",
                "phone_start_s": "1", "phone_end_s": "1.2",
                "video_phone_start_s": "1", "video_phone_end_s": "1.2",
                "audio_video_start_offset_s": "0", "mouth_crop_frame_indices_json": "[10,12,14]",
            }
            self.rows.append(row)
            self.samples.append({
                "video": torch.zeros(3, 16, 8, 8), "frame_mask": torch.tensor([True] * 3 + [False] * 13),
                "decoded_frame_count": 3, "features": torch.zeros(8),
                "feature_mask": torch.zeros(8, dtype=torch.bool), "evidence_features": torch.zeros(7),
                "evidence_feature_mask": torch.zeros(7, dtype=torch.bool),
                "label": torch.tensor(int(row["class_id"])), "video_label": torch.tensor(label),
                "event_id": row["event_id"], "video_id": prefix, "kind": kind, "phoneme": row["phoneme"],
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.samples[index]

    def evidence_coverage(self):
        return {}


def trace_for(row):
    return {
        "artifact_version": "vild-trace-v1", "video_id": row["video_id"],
        "source_file": row["file"], "source_group": row["source_group"],
        "dataset_split": row["dataset_split"], "timing": {"audio_video_start_offset_s": 0, "fps": 10},
        "frames": [{"frame_index": i, "timestamp_s": t, "normalized_vild": v, "face_count": 1}
                   for i, t, v in ((10, 1.0, 0.03), (12, 1.1, 0), (14, 1.2, 0.04))],
        "bilabial_event_windows": [{
            "event_id": row["event_id"], "phoneme": "p", "audio_phone_start_s": 1,
            "audio_phone_end_s": 1.2, "video_phone_start_s": 1, "video_phone_end_s": 1.2,
            "window_start_s": 0.8, "window_end_s": 1.4, "mouth_clip_frame_indices": [10, 12, 14],
        }],
    }


def test_export_and_both_arms_reuse_identical_clips_and_explicit_missingness(tmp_path):
    clips = PreparedClips(tmp_path)
    for row in clips.rows:
        row.update(vild_trace_path="fixture", vild_trace_sha256="fixture")
    inventories = {role: [{"video_id": ds.rows[0]["video_id"], "status": "paired"}]
                   for role, ds in (("train", clips), ("inner", PreparedClips(tmp_path, "inner", 1)))}
    datasets = {"train": clips, "inner": PreparedClips(tmp_path, "inner", 1)}
    for row in datasets["inner"].rows:
        row.update(vild_trace_path="fixture", vild_trace_sha256="fixture")
    output = tmp_path / "timing"
    result = inputs.export_timing_inputs(datasets, inventories, output, {"test": "fixed"}, trace_loader=trace_for)
    assert result["gate"] == "ready_for_readiness" and result["new_pixels"] == 0
    assert result["roles"]["train"]["videos_with_timing"] == 1
    enabled = inputs.SequenceInputDataset(clips, output, "train", True)
    control = inputs.SequenceInputDataset(clips, output, "train", False)
    assert enabled.input_contract == control.input_contract and enabled.rows == control.rows
    assert enabled[0]["timing_mask"].sum() == 3 and not control[0]["timing_mask"].any()
    assert not enabled[1]["timing_mask"].any() and enabled[1]["label"].item() == -1
    assert torch.equal(enabled[0]["video"], control[0]["video"])


def test_real_decoder_and_verified_synthetic_trace_reconcile_before_export(tmp_path):
    import cv2
    import numpy as np

    datasets, inventories = {}, {}
    for role in ("train", "inner"):
        fixture = PreparedClips(tmp_path, role, int(role == "inner"))
        row = fixture.rows[0]
        path = tmp_path / row["mouth_clip_path"]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
        assert writer.isOpened()
        for value in (20, 100, 200):
            writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
        writer.release()
        trace = tmp_path / f"{role}.json.gz"
        atomic_write_gzip_json(trace, trace_for(row))
        row.update(vild_trace_path=str(trace), vild_trace_sha256=file_sha256(trace))
        manifest = tmp_path / f"{role}.csv"
        atomic_write_csv(manifest, [row])
        base = MouthEventDataset(manifest, tmp_path, image_size=8)
        index = [{"video_id": row["video_id"], "source_group": row["source_group"],
                  "dataset_split": row["dataset_split"], "kind": "bilabial",
                  "source_event_id": row["event_id"], "clip_id": row["event_id"]}]
        datasets[role] = ExpandedContextDataset(base, index, {}, tmp_path, tmp_path)
        inventories[role] = [{"video_id": row["video_id"], "status": "paired"}]
    output = tmp_path / "export"
    inputs.export_timing_inputs(datasets, inventories, output, {"frozen": "synthetic"})
    result = inputs.SequenceInputDataset(datasets["train"], output, "train", True)[0]
    assert result["decoded_frame_count"] == 3 and result["timing_mask"].sum() == 3
    (tmp_path / datasets["train"].rows[0]["mouth_clip_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="clip"):
        inputs.SequenceInputDataset(datasets["train"], output, "train", True)[0]


def test_missing_timing_is_retained_but_blocks_training_and_partial_output_is_preserved(tmp_path):
    datasets = {r: PreparedClips(tmp_path, r, int(r == "inner")) for r in ("train", "inner")}
    inventories = {r: [{"video_id": r, "status": "paired"}, {"video_id": f"{r}-missing", "status": "unevaluated"}]
                   for r in datasets}
    output = tmp_path / "missing_timing"
    result = inputs.export_timing_inputs(datasets, inventories, output, {"frozen": "test"})
    assert result["gate"] == "stop_and_review_timing_coverage"
    assert result["roles"]["train"]["unavailable"] == 1
    with pytest.raises(ValueError, match="coverage"):
        inputs.SequenceInputDataset(datasets["train"], output, "train", True)
    assert inputs.export_timing_inputs({}, {}, output, {"frozen": "test"}) == result
    job_hash = file_sha256(output / "job.json")
    with pytest.raises(ValueError):
        inputs.export_timing_inputs({}, {}, output, {"different": "binding"})
    assert file_sha256(output / "job.json") == job_hash
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "keep.txt").write_text("interrupted", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        inputs.export_timing_inputs(datasets, inventories, partial, {"frozen": "test"})
    assert (partial / "keep.txt").read_text(encoding="utf-8") == "interrupted"
