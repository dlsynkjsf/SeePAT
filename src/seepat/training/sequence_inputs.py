"""Sealed timing export and two views of the same original mouth clips."""

from collections import defaultdict

import torch

from seepat.artifacts import atomic_write_json, file_sha256
from seepat.models.sequence_evidence import TIMING_VERSION, event_timing_sequence
from seepat.preprocessing.augmentation import augmented_trace_for_row
from seepat.training import experiment as runtime
from seepat.training.matched_inputs import inside


def export_timing_inputs(datasets, inventories, output, binding, *,
                         trace_loader=augmented_trace_for_row, progress=None):
    job = {"stage": "inputs", "binding": binding, "timing_version": TIMING_VERSION}
    if output.exists():
        return runtime.verify_output(output, job)
    if set(datasets) != {"train", "inner"} or set(inventories) != set(datasets):
        raise ValueError("Timing export requires exactly Train and inner Validation")
    with runtime.output_stage(output, job, progress) as complete:
        roles = {}
        for number, (role, dataset) in enumerate(datasets.items(), 1):
            if progress:
                progress.start_job(number, role)
            records, covered, eligible = {}, set(), set()
            available_slots, timing_clips = 0, 0
            current_video, trace = None, None
            grouped = defaultdict(list)
            for index, row in enumerate(dataset.rows):
                grouped[row["video_id"]].append(index)
            for finished, index in enumerate(i for group in grouped.values() for i in group):
                row = dataset.rows[index]
                if progress:
                    progress("verify existing clips and export timing", finished, len(dataset), row["event_id"])
                sample = dataset[index]
                clip = inside(dataset.root, row["mouth_clip_path"])
                if row["kind"] == "bilabial":
                    if row["video_id"] != current_video:
                        current_video = row["video_id"]
                        trace = trace_loader(row) if row.get("vild_trace_path") and row.get("vild_trace_sha256") else None
                    eligible.add(row["video_id"])
                timing = event_timing_sequence(
                    row, trace if row["kind"] == "bilabial" else None,
                    decoded_frame_count=sample["decoded_frame_count"],
                )
                if (timing["timing_mask"] & ~sample["frame_mask"]).any():
                    raise ValueError("Timing export disagrees with actual decoded RGB slots")
                if timing["timing_mask"].any():
                    covered.add(row["video_id"])
                    timing_clips += 1
                available_slots += int(timing["timing_mask"].sum())
                if row["event_id"] in records:
                    raise ValueError("Duplicate timing clip identity")
                records[row["event_id"]] = {
                    "video_id": row["video_id"], "kind": row["kind"],
                    "source_group": row["source_group"], "dataset_split": row["dataset_split"],
                    "clip_sha256": file_sha256(clip), "decoded_frame_count": sample["decoded_frame_count"],
                    "trace_sha256": row.get("vild_trace_sha256", "") if row["kind"] == "bilabial" else "",
                    "augmentation_sha256": row.get("vild_augmentation_sha256", "") if row["kind"] == "bilabial" else "",
                    **{key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in timing.items()},
                }
            expected = {r["video_id"] for r in inventories[role] if r["status"] == "paired"}
            if expected != eligible:
                raise ValueError("Timing export changed original usable membership")
            atomic_write_json(output / f"{role}.json", records)
            roles[role] = {"requested": len(inventories[role]), "usable": len(expected),
                           "unavailable": len(inventories[role]) - len(expected),
                           "clips": len(records), "videos_with_timing": len(covered),
                           "clips_with_timing": timing_clips, "available_timing_slots": available_slots}
        gate = all(r["usable"] > 0 and r["videos_with_timing"] * 5 >= r["usable"] * 4 for r in roles.values())
        return complete({"status": "complete", "roles": roles, "new_pixels": 0, "model_calls": 0,
                         "gate": "ready_for_readiness" if gate else "stop_and_review_timing_coverage"})


class SequenceInputDataset:
    """Decode original pixels and verify their sealed timing; control disables only timing."""

    def __init__(self, base, directory, role, enabled):
        summary = runtime.verify_output(directory)
        if summary["gate"] != "ready_for_readiness":
            raise ValueError("Timing coverage gate failed; preserve inputs and review")
        self.base, self.rows, self.root = base, base.rows, base.root
        self.enabled = enabled
        self.records = runtime.read_json(directory / f"{role}.json")
        if set(self.records) != {r["event_id"] for r in self.rows}:
            raise ValueError("Timing records change clip membership")
        self.calibration_contract = base.calibration_contract
        self.input_contract = {"version": TIMING_VERSION, "inputs_seal_sha256": file_sha256(directory / "seal.json"),
                               "context_target": "unavailable -1", "bags": "unchanged saved expanded sample"}

    def __len__(self):
        return len(self.base)

    def evidence_coverage(self):
        return self.base.evidence_coverage()

    def __getitem__(self, index):
        row = self.rows[index]
        record = self.records[row["event_id"]]
        if (any(record[k] != row[k] for k in ("video_id", "kind", "source_group", "dataset_split"))
                or row["kind"] == "bilabial" and (
                    record["trace_sha256"] != row.get("vild_trace_sha256", "")
                    or record["augmentation_sha256"] != row.get("vild_augmentation_sha256", ""))
                or file_sha256(inside(self.root, row["mouth_clip_path"])) != record["clip_sha256"]):
            raise ValueError("Original clip or timing provenance changed")
        sample = self.base[index]
        if sample["decoded_frame_count"] != record["decoded_frame_count"]:
            raise ValueError("Original decoded clip count changed")
        values = torch.tensor(record["timing_features"], dtype=torch.float32)
        mask = torch.tensor(record["timing_mask"], dtype=torch.bool)
        if (values.shape != (16, 3) or mask.shape != (16,)
                or (mask & ~sample["frame_mask"]).any() or not torch.isfinite(values).all()
                or values[~mask].any() or record["timing_version"] != TIMING_VERSION
                or row["kind"] == "context" and mask.any()):
            raise ValueError("Timing shape, mask or version changed")
        return {**sample, "timing_version": TIMING_VERSION,
                "timing_features": values if self.enabled else torch.zeros_like(values),
                "timing_mask": mask if self.enabled else torch.zeros_like(mask)}
