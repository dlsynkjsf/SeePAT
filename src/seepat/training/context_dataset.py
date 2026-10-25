"""Expanded-context decoder used by both experiment drivers."""

from collections import defaultdict
from pathlib import Path

from seepat.data.sampling import canonical_source_group
from seepat.training.matched_inputs import MatchedClipDataset, inside

CONTEXT_INPUTS = "outputs/context_budget_inputs_v1"


class ExpandedContextDataset(MatchedClipDataset):
    """Decode expanded bags while preserving provenance and missing-feature masks."""

    def __init__(self, bilabials, index, records, project_root, export_root):
        self.root, export_root = Path(project_root).resolve(), Path(export_root).resolve()
        self.image_size, self.bilabials = 224, bilabials
        self.calibration_contract = bilabials.calibration_contract
        self.base_indices = {row["event_id"]: i for i, row in enumerate(bilabials.rows)}
        if len(self.base_indices) != len(bilabials.rows):
            raise ValueError("Duplicate original bilabial event identity")
        grouped = defaultdict(list)
        for row in bilabials.rows:
            grouped[row["video_id"]].append(row)
        self.rows, self.contexts = [], {}
        for item in index:
            identifier, clip_id = item["video_id"], item["clip_id"]
            original = grouped[identifier][0]
            if (item["dataset_split"] != original["dataset_split"]
                    or canonical_source_group(item["source_group"], None) != canonical_source_group(original["source_group"], None)):
                raise ValueError("Study index changes split/source provenance")
            if item["kind"] == "bilabial":
                row = bilabials.rows[self.base_indices[item["source_event_id"]]]
                if row["event_id"] != clip_id or row["video_id"] != identifier:
                    raise ValueError("Original bilabial identity changed")
                inside(self.root, row["mouth_clip_path"])
                self.rows.append({**row, "kind": "bilabial", "source_group": item["source_group"]})
            elif item["kind"] == "context":
                windows = [w for w in records[identifier]["windows"] if w["context_id"] == clip_id]
                if len(windows) != 1:
                    raise ValueError("Missing/duplicate original context")
                window = windows[0]
                mask = window["frame_mask"]
                if (window["technical_ok"] is not True or len(mask) != 16 or not any(mask)
                        or any(type(v) is not bool for v in mask)
                        or window["video_id"] != identifier or window["dataset_split"] != item["dataset_split"]
                        or canonical_source_group(window["source_group"], None) != canonical_source_group(item["source_group"], None)
                        or window["evidence_available"] is not False or window["phoneme"] != ""
                        or window["event_class_id"] != "" or window["clip_file"] != f"{clip_id}.mp4"):
                    raise ValueError("Context mask, identity or unavailable evidence differs")
                clip = inside(export_root, f"records/{identifier}/{window['clip_file']}")
                if not clip.is_relative_to(self.root):
                    raise ValueError("Context path escapes the project")
                self.contexts[clip_id] = window
                self.rows.append({**item, "event_id": clip_id, "class_id": "-1", "phoneme": "",
                                  "manipulation_modality": original["manipulation_modality"],
                                  "mouth_clip_path": clip.relative_to(self.root).as_posix()})
            else:
                raise ValueError("Unknown study clip kind")
        if not self.rows or len({r["event_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Study dataset must be nonempty with unique clip identities")
