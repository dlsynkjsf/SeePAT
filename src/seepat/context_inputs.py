"""Opt-in Phase-5 input planning; never masquerades context as a phoneme event.

The existing event trainer deliberately cannot consume these index records.
Training/evaluation adapters must verify the input contract before using them.
"""

from __future__ import annotations

import math
from collections import defaultdict

from seepat.artifacts import stable_id
from seepat.data.sampling import canonical_source_group
from seepat.training.dataset import sequence_frame_indices

INPUT_VERSION = "matched-context-inputs-v1"
CLIP_CAP = 16


def evenly_spaced(items: list, count: int) -> list:
    if type(count) is not int or not 0 <= count <= len(items):
        raise ValueError("Selection count must be an integer within available items")
    if not count:
        return []
    positions, _ = sequence_frame_indices(len(items), count)
    return [items[index] for index in positions]


def validate_tables(sources: dict, videos: dict, events: dict) -> dict[str, list[dict]]:
    """Validate original Train/inner membership, without consulting fake intervals."""
    grouped = defaultdict(list)
    group_splits: dict[str, str] = {}
    seen_events: set[str] = set()
    seen_videos: set[str] = set()
    for split in ("train", "val"):
        source_by_file = {row["file"]: row for row in sources[split]}
        video_by_file = {row["file"]: row for row in videos[split]}
        if (len(source_by_file) != len(sources[split])
                or len(video_by_file) != len(videos[split])
                or source_by_file.keys() != video_by_file.keys()):
            raise ValueError("Original source/terminal membership differs or has duplicates")
        for file, source in source_by_file.items():
            video = video_by_file[file]
            identifier = video["video_id"]
            group = canonical_source_group(source.get("source_group") or source.get("original"), file)
            if (source.get("split") != split or video.get("split") != split
                    or identifier != stable_id(file)
                    or identifier in seen_videos or group in group_splits):
                raise ValueError("Duplicate/leaking source groups, video IDs or wrong split")
            if source.get("modify_type") != video.get("modify_type"):
                raise ValueError("Source and terminal categories disagree")
            if source["modify_type"] not in {"real", "audio_modified", "visual_modified", "both_modified"}:
                raise ValueError("Unsupported original source category")
            seen_videos.add(identifier)
            group_splits[group] = split
        for event in events[split]:
            source = source_by_file.get(event["file"])
            video = video_by_file.get(event["file"])
            if (source is None or video is None or event["dataset_split"] != split
                    or event["video_id"] != video["video_id"]
                    or canonical_source_group(event["source_group"], None) != canonical_source_group(
                        source.get("source_group") or source.get("original"), source["file"])
                    or event["manipulation_modality"] != source["modify_type"]
                    or event["event_id"] in seen_events or event["class_id"] not in {"0", "1"}
                    or (source["modify_type"] == "real" and event["class_id"] != "0")
                    or event["phoneme"] not in {"m", "b", "p"}
                    or video["pipeline_status"] != "complete"
                    or video["eligibility_status"] != "eligible"):
                raise ValueError("Bilabial identity, label, eligibility or provenance mismatch")
            seen_events.add(event["event_id"])
            grouped[video["video_id"]].append(event)
    return dict(grouped)


def paired_indices(events: list[dict], contexts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Equal per-video scoring counts, >= half bilabial, no duplicate padding votes.

    Context eligibility is crop/timing-based. Neither event class nor fake interval
    participates in ordering/selection. Fewer contexts means fewer replacements,
    not new windows or replacement videos. A one-event video remains unchanged.
    """
    if len({row["event_id"] for row in events}) != len(events):
        raise ValueError("Duplicate bilabial event IDs")
    if len({row["context_id"] for row in contexts}) != len(contexts):
        raise ValueError("Duplicate context IDs")
    if any(type(row.get("technical_ok")) is not bool for row in contexts):
        raise ValueError("Every planned context needs an explicit technical status")
    if not events:
        return [], []
    first = events[0]
    identity = (first["video_id"], first["dataset_split"],
                canonical_source_group(first["source_group"], None))
    if any((row["video_id"], row["dataset_split"],
            canonical_source_group(row["source_group"], None)) != identity
           for row in events + contexts):
        raise ValueError("Mixed video, split or source group in clip selection")
    if any(not math.isfinite(float(row["video_phone_start_s"])) for row in events):
        raise ValueError("Bilabial positions must be finite")
    ordered = sorted(events, key=lambda row: (float(row["video_phone_start_s"]), row["event_id"]))
    base = evenly_spaced(ordered, min(CLIP_CAP, len(ordered)))
    usable = sorted((row for row in contexts if row["technical_ok"]),
                    key=lambda row: (int(row["start_frame"]), row["context_id"]))
    replacement_count = min(len(base) // 2, len(usable), 8)

    def bilabial(row):
        return {"clip_id": row["event_id"], "kind": "bilabial",
                "source_event_id": row["event_id"], "video_id": row["video_id"],
                "dataset_split": row["dataset_split"], "source_group": identity[2]}

    control = [bilabial(row) for row in base]
    candidate = [bilabial(row) for row in evenly_spaced(base, len(base) - replacement_count)]
    for row in evenly_spaced(usable, replacement_count):
        candidate.append({"clip_id": row["context_id"], "kind": "context",
                          "source_event_id": "", "video_id": row["video_id"],
                          "dataset_split": row["dataset_split"], "source_group": identity[2],
                          "context_record_id": row["context_id"]})
    return control, candidate
