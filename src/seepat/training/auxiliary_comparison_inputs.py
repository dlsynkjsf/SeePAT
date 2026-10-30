"""Fixed larger-study membership and Train-only exposure labels on existing crops."""

import hashlib
import importlib.util
from collections import Counter, defaultdict

import torch

from seepat.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    read_csv_rows,
    stable_id,
)
from seepat.data.sampling import canonical_source_group
from seepat.models.sequence_evidence import event_timing_sequence
from seepat.training import experiment as runtime
from seepat.training.auxiliary_inputs import verified_sample
from seepat.training.auxiliary_visual import balance_targets
from seepat.training.context_dataset import CONTEXT_INPUTS, ExpandedContextDataset
from seepat.training.dataset import MouthEventDataset
from seepat.training.matched_inputs import inside

VERSION = "larger-auxiliary-comparison-v1"
OUTPUT = "outputs/engineering/larger_auxiliary_comparison_v1"
CATEGORIES = ("audio_modified", "both_modified", "real", "visual_modified")
SEED = "auxiliary-larger-20261030-v1"
MONITOR = "outputs/engineering/smooth_loss_pilot_v1/selection"
ROLES = {
    "train": ("train", "outputs/numerical_calibration/guarded_per_video_v3/events_train_calibrated.csv",
              "outputs/phase5_context_inputs_v1"),
    "inner": ("val", "outputs/numerical_calibration/guarded_per_video_v3/events_val_calibrated.csv",
              "outputs/phase5_context_inputs_v1"),
    "outer": ("val", "outputs/numerical_calibration/phase5_outer_v1/events_phase5_outer_calibrated.csv",
              "outputs/phase5_outer_context_inputs_v1"),
}
ANNOTATIONS = "data/manifests/train_subset5000.csv"


def select_train(rows, exclusions, *, per_category=250):
    """Stratify by category; availability and annotation contents never choose IDs."""
    if per_category < 1 or any(r["dataset_split"] != "val" for r in exclusions):
        raise ValueError("Invalid selection budget or Validation exclusion role")
    excluded_ids = {r["video_id"] for r in exclusions}
    excluded_groups = {canonical_source_group(r["source_group"], None) for r in exclusions}
    identities, groups = set(), set()
    for row in rows:
        group = canonical_source_group(row["source_group"], None)
        if row["video_id"] in identities or group in groups:
            raise ValueError("Train inventory has duplicate identities/source groups")
        if (row["dataset_split"] != "train" or row["category"] not in CATEGORIES
                or row["video_id"] in excluded_ids or group in excluded_groups):
            raise ValueError("Train/Validation source separation or category differs")
        identities.add(row["video_id"])
        groups.add(group)

    def rank(row):
        text = "\0".join((SEED, canonical_source_group(row["source_group"], None), row["video_id"]))
        return hashlib.sha256(text.encode()).hexdigest(), row["video_id"]

    selected = []
    for category in CATEGORIES:
        category_rows = sorted((r for r in rows if r["category"] == category), key=rank)
        if len(category_rows) < per_category:
            raise ValueError("Insufficient distinct source groups; selection rule stays fixed")
        selected.extend(category_rows[:per_category])
    return selected


def validate_membership(inventories, indexes):
    ids, groups, clip_ids = set(), set(), set()
    for role, rows in inventories.items():
        lookup = {}
        for row in rows:
            group = canonical_source_group(row["source_group"], None)
            if (row["video_id"] in ids or group in groups or row["dataset_split"] != ROLES[role][0]
                    or row["category"] not in CATEGORIES or row["status"] not in {"paired", "unevaluated"}):
                raise ValueError("Requested source separation/availability differs")
            ids.add(row["video_id"])
            groups.add(group)
            lookup[row["video_id"]] = row
        grouped = defaultdict(list)
        for clip in indexes[role]:
            original = lookup[clip["video_id"]]
            if (clip["clip_id"] in clip_ids or clip["kind"] not in {"bilabial", "context"}
                    or clip["dataset_split"] != ROLES[role][0]
                    or canonical_source_group(clip["source_group"], None)
                    != canonical_source_group(original["source_group"], None)):
                raise ValueError("Selected clip identity/source differs")
            clip_ids.add(clip["clip_id"])
            grouped[clip["video_id"]].append(clip)
        for row in rows:
            clips = grouped[row["video_id"]]
            if (len(clips) != int(row["expanded_clips"]) or bool(clips) != (row["status"] == "paired")
                    or not clips and not row["reason"] or len(clips) > 16
                    or clips and (not any(c["kind"] == "bilabial" for c in clips)
                                  or sum(c["kind"] == "context" for c in clips) > 8)):
                raise ValueError("Selected complete bag or original failure differs")


def prepare_selection(project, binding):
    output = inside(project, f"{OUTPUT}/selection")
    job = {"stage": "selection", "study": VERSION, "binding": binding}
    if output.exists():
        runtime.verify_output(output, job)
        return read_selection(project)
    cache = inside(project, CONTEXT_INPUTS)
    inventories = {"inner": read_csv_rows(inside(project, f"{MONITOR}/inventory_inner.csv")),
                   "outer": read_csv_rows(cache / "inventory_outer.csv")}
    original_inner = read_csv_rows(cache / "inventory_inner.csv")
    if (len(original_inner) != 1000 or Counter(r["category"] for r in inventories["inner"]) != dict.fromkeys(CATEGORIES, 16)
            or Counter(r["category"] for r in inventories["outer"]) != dict.fromkeys(CATEGORIES, 250)
            or any(r not in original_inner for r in inventories["inner"])):
        raise ValueError("Original monitor/comparison membership differs")
    full_train = read_csv_rows(cache / "inventory_train.csv")
    if len(full_train) != 5000:
        raise ValueError("Original Train pool must retain all 5,000 requests")
    inventories["train"] = select_train(full_train, [*original_inner, *inventories["outer"]])
    indexes = {}
    for role in ROLES:
        chosen = {r["video_id"] for r in inventories[role]}
        indexes[role] = [r for r in read_csv_rows(cache / f"expanded_{role}.csv") if r["video_id"] in chosen]
    validate_membership(inventories, indexes)
    if indexes["inner"] != read_csv_rows(inside(project, f"{MONITOR}/index_inner.csv")):
        raise ValueError("Original monitor clip order/membership differs")
    if ((sum(r["status"] == "paired" for r in inventories["inner"]), len(indexes["inner"])) != (62, 639)
            or (sum(r["status"] == "paired" for r in inventories["outer"]), len(indexes["outer"])) != (965, 10404)):
        raise ValueError("Frozen Validation coverage/clip counts differ")
    with runtime.output_stage(output, job) as complete:
        for role in ROLES:
            atomic_write_csv(output / f"inventory_{role}.csv", inventories[role])
            atomic_write_csv(output / f"index_{role}.csv", indexes[role])
        complete({"status": "complete", "selection_seed": SEED, "annotations_read": False,
                  "requests": {r: len(v) for r, v in inventories.items()}})
    return inventories, indexes


def read_selection(project):
    directory = inside(project, f"{OUTPUT}/selection")
    runtime.verify_output(directory)
    inventories = {r: read_csv_rows(directory / f"inventory_{r}.csv") for r in ROLES}
    indexes = {r: read_csv_rows(directory / f"index_{r}.csv") for r in ROLES}
    validate_membership(inventories, indexes)
    return inventories, indexes


def target_api(project):
    # Reuse the frozen target rule without moving or editing its historical tool.
    spec = importlib.util.spec_from_file_location("accepted_clip_targets", inside(project, "scripts/audit_clip_targets.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_dataset(project, role, index):
    split, events, export = ROLES[role]
    original = MouthEventDataset(inside(project, events), project, dataset_split=split, require_calibration=True)
    records = runtime.read_json(inside(project, f"{CONTEXT_INPUTS}/records_{role}.json"))
    return ExpandedContextDataset(original, index, records, project, inside(project, export))


def export_case(base, indices, inventory, output, binding, *, source=None, audit=None,
                support_loader=None, progress=None):
    """One sealed video is the export restart unit; no pixel files are written."""
    job = {"study": VERSION, "stage": "prepared-video", "binding": binding, "inventory": inventory}
    if output.exists():
        runtime.verify_output(output, job)
        return runtime.read_json(output / "case.json")
    if (inventory["dataset_split"] != "train" and source is not None
            or source is not None and (source["split"] != "train"
                                      or source.get("dataset_source", "AV-Deepfake1M++") != "AV-Deepfake1M++"
                                      or stable_id(source["file"]) != inventory["video_id"]
                                      or source["modify_type"] != inventory["category"]
                                      or canonical_source_group(source.get("source_group") or source.get("original"), source["file"])
                                      != canonical_source_group(inventory["source_group"], None))):
        raise ValueError("Auxiliary annotations are Train-only and bound to the original source")
    if len(indices) != int(inventory["expanded_clips"]) or bool(indices) != (inventory["status"] == "paired"):
        raise ValueError("Export changed original available/unavailable clip membership")
    records, media, clips = {}, {}, []
    for index in indices:
        row = base.rows[index]
        if any(row[k] != inventory[k] for k in ("video_id", "source_group", "dataset_split")):
            raise ValueError("Decoded input belongs to another requested source")
        sample = base[index]
        mask = sample["frame_mask"]
        if mask.shape != (16,) or mask.dtype != torch.bool or not mask.any():
            raise ValueError("Existing RGB support changed")
        name = inside(base.root, row["mouth_clip_path"]).relative_to(base.root).as_posix()
        clip_hash = file_sha256(inside(base.root, name))
        record = {k: row[k] for k in ("video_id", "source_group", "dataset_split", "kind")}
        record.update(clip_sha256=clip_hash, decoded_frame_count=sample["decoded_frame_count"],
                      frame_mask=mask.tolist(), original_event_class_id=row["class_id"],
                      original_phoneme=row["phoneme"],
                      trace_sha256=row.get("vild_trace_sha256", "") if row["kind"] == "bilabial" else "",
                      augmentation_sha256=row.get("vild_augmentation_sha256", "") if row["kind"] == "bilabial" else "")
        records[row["event_id"]] = record
        media[name] = {"bytes": inside(base.root, name).stat().st_size, "sha256": clip_hash}
        clips.append({"clip_id": row["event_id"], "kind": row["kind"], "video_id": row["video_id"],
                      "source_group": row["source_group"], "dataset_split": row["dataset_split"],
                      "event": row if row["kind"] == "bilabial" else None, "timing": record})
        if progress:
            progress("verify existing crop; no pixel export", len(records), len(indices), row["event_id"])
    target_record = None
    if source is not None and indices:
        events = [c["event"] for c in clips if c["event"] is not None]
        if (not events or any(source["file"] != e["file"] or e["manipulation_modality"] != source["modify_type"]
                              or e["class_id"] not in {"0", "1"} or e["phoneme"] not in {"m", "b", "p"} for e in events)
                or canonical_source_group(source.get("source_group") or source.get("original"), source["file"])
                != canonical_source_group(inventory["source_group"], None)):
            raise ValueError("Train annotation file/source group differs")
        case = {"inventory": inventory, "source": source, "clips": clips}
        trace, context = support_loader(case)
        _, _, clock_reason = audit.video_clock(trace)
        for clip in clips:
            row, record = clip["event"], clip["timing"]
            if row is not None and not clock_reason:
                data = event_timing_sequence(row, trace, decoded_frame_count=record["decoded_frame_count"])
                record["timing_frame_indices"] = list(data["timing_frame_indices"])
        target_record = audit.audit_case(case, trace, context)
        for item in target_record["clips"]:
            record = records[item["clip_id"]]
            record.update(target=item["aux_visual_target"], target_mask=item["aux_target_mask"])
            if item["support"] and [s["rgb_valid"] for s in item["support"]] != record["frame_mask"]:
                raise ValueError("Target support differs from actual decoded RGB mask")
    result = {"inventory": inventory, "records": records, "media": media, "target_provenance": target_record}
    with runtime.output_stage(output, job) as complete:
        atomic_write_json(output / "case.json", result)
        complete({"status": "complete", "clips": len(records), "new_pixels": 0, "model_calls": 0})
    return result


def prepare_inputs(project, binding, *, progress=None):
    inventories, indexes = prepare_selection(project, binding)
    selected_binding = {**binding, "selection_seal_sha256": file_sha256(inside(project, f"{OUTPUT}/selection/seal.json"))}
    output = inside(project, f"{OUTPUT}/inputs")
    job = {"study": VERSION, "stage": "inputs", "binding": selected_binding}
    if output.exists():
        return runtime.verify_output(output, job)
    # Membership is already sealed before annotation contents are read.
    source_rows = read_csv_rows(inside(project, ANNOTATIONS))
    sources = {stable_id(r["file"]): r for r in source_rows}
    if len(sources) != len(source_rows) or len(source_rows) != 5000:
        raise ValueError("Original Train annotation membership differs")
    audit, context_summary = target_api(project), runtime.read_json(inside(project, f"{ROLES['train'][2]}/summary.json"))
    prepared, media, case_seals = {}, {}, {}
    for number, role in enumerate(ROLES, 1):
        if progress:
            progress.start_job(number, role)
        base = base_dataset(project, role, indexes[role])
        grouped, records = defaultdict(list), {}
        for index, row in enumerate(base.rows):
            grouped[row["video_id"]].append(index)
        for finished, original in enumerate(inventories[role], 1):
            directory = inside(project, f"{OUTPUT}/preparation/{role}/{original['video_id']}")
            case = export_case(base, grouped[original["video_id"]], original, directory, selected_binding,
                               source=sources[original["video_id"]] if role == "train" else None,
                               audit=audit, support_loader=lambda case: audit.load_support(project, case, context_summary))
            records.update(case["records"])
            media.update(case["media"])
            case_seals[directory.relative_to(project).as_posix()] = file_sha256(directory / "seal.json")
            if progress:
                progress("verify prepared video records; resume sealed videos", finished, len(inventories[role]), original["video_id"])
        prepared[role] = records
    balance = balance_targets(prepared["train"].values())
    with runtime.output_stage(output, job) as complete:
        for role, records in prepared.items():
            atomic_write_json(output / f"{role}.json", records)
        atomic_write_json(output / "media_catalog.json", media)
        atomic_write_json(output / "case_seals.json", case_seals)
        roles = {r: {"requested": len(inventories[r]), "usable": sum(v["status"] == "paired" for v in inventories[r]),
                     "unavailable": sum(v["status"] != "paired" for v in inventories[r]), "clips": len(prepared[r])} for r in ROLES}
        readiness = {r: [v["video_id"] for category in CATEGORIES
                         for v in sorted((v for v in inventories[r] if v["category"] == category and v["status"] == "paired"),
                                         key=lambda v: v["video_id"])[:4 if r == "train" else 1]] for r in ("train", "inner")}
        if len(readiness["train"]) != 16 or len(readiness["inner"]) != 4:
            raise ValueError("Readiness requires exactly four/one usable videos per category")
        return complete({"status": "complete", "roles": roles, "auxiliary_balance": balance, "readiness": readiness,
                         "targets": dict(Counter(str(r["target"]) for r in prepared["train"].values())),
                         "new_pixels": 0, "model_calls": 0, "optimizer_updates": 0, "gate": "ready_for_readiness"})


class ComparisonDataset:
    """The same bound inference inputs in all models; auxiliary labels on Train only."""

    def __init__(self, base, directory, role, *, videos=None):
        summary = runtime.verify_output(directory)
        if summary["gate"] != "ready_for_readiness" or role not in ROLES:
            raise ValueError("Comparison inputs are not ready")
        self.base, self.role = base, role
        all_records = runtime.read_json(directory / f"{role}.json")
        if set(all_records) != {r["event_id"] for r in base.rows}:
            raise ValueError("Prepared inference clip membership changed")
        if role != "train" and any("target" in r or "target_mask" in r for r in all_records.values()):
            raise ValueError("Validation inference must be label-free")
        if role == "train" and balance_targets(all_records.values()) != summary["auxiliary_balance"]:
            raise ValueError("Train-only auxiliary balance changed")
        self.indices = [i for i, row in enumerate(base.rows) if videos is None or row["video_id"] in videos]
        self.rows = [base.rows[i] for i in self.indices]
        self.records = all_records
        self.auxiliary_balance = summary["auxiliary_balance"]
        self.calibration_contract = base.calibration_contract
        self.input_contract = {"version": VERSION, "inputs_seal_sha256": file_sha256(directory / "seal.json"),
                               "policy": "same expanded bags; auxiliary targets affect Train loss only"}

    def __len__(self):
        return len(self.rows)

    def evidence_coverage(self):
        from seepat.evidence import evidence_coverage
        return evidence_coverage(self.rows)

    def __getitem__(self, index):
        record = self.records[self.rows[index]["event_id"]]
        sample = verified_sample(self.base, self.indices[index], record)
        if self.role == "train":
            return {**sample, "auxiliary_target": torch.tensor(record["target"], dtype=torch.long),
                    "auxiliary_mask": torch.tensor(record["target_mask"], dtype=torch.bool)}
        return sample
