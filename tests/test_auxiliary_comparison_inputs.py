from copy import deepcopy

import pytest

from seepat.training.auxiliary_comparison_inputs import select_train


def inventory():
    return [{"video_id": f"{category}-{n}", "source_group": f"source/{category}-{n}",
             "category": category, "dataset_split": "train", "status": "paired",
             "reason": "", "expanded_clips": "2"}
            for category in ("real", "audio_modified", "visual_modified", "both_modified")
            for n in range(3)]


def test_selection_is_order_and_annotation_blind_and_retains_selected_failures():
    rows = inventory()
    selected = select_train(rows, [], per_category=1)
    assert [r["video_id"] for r in selected] == [
        "audio_modified-1", "both_modified-0", "real-2", "visual_modified-2"]
    changed = deepcopy(rows)
    chosen = next(r for r in changed if r["video_id"] == "real-2")
    chosen.update(status="unevaluated", reason="existing_failure", expanded_clips="0")
    for row in changed:
        row.update(visual_fake_segments_json="unread", probability="unread", sharpness="unread")
    again = select_train(list(reversed(changed)), [], per_category=1)
    assert [r["video_id"] for r in again] == [r["video_id"] for r in selected]
    assert next(r for r in again if r["video_id"] == "real-2")["status"] == "unevaluated"


def test_overlap_including_unavailable_and_duplicate_groups_stop_before_selection():
    rows = inventory()
    excluded = [{"video_id": "absent-val", "source_group": rows[0]["source_group"],
                 "dataset_split": "val", "status": "unevaluated"}]
    with pytest.raises(ValueError, match="separation"):
        select_train(rows, excluded, per_category=1)
    rows[1]["source_group"] = rows[0]["source_group"]
    with pytest.raises(ValueError, match="duplicate"):
        select_train(rows, [], per_category=1)


def test_new_export_reuses_accepted_frame_targets_and_sealed_video_restart(tmp_path):
    import torch
    from test_clip_target_audit import audit, case_fixture

    from seepat.artifacts import file_sha256
    from seepat.training.auxiliary_comparison_inputs import export_case

    case, trace, context = case_fixture()
    class ExistingCrops:
        def __init__(self):
            self.root, self.rows = tmp_path, []
            for clip in case["clips"]:
                path = tmp_path / (clip["clip_id"] + ".bin")
                path.write_bytes(b"synthetic existing crop")
                self.rows.append({**(clip["event"] or {}), "event_id": clip["clip_id"],
                                  "kind": clip["kind"], "mouth_clip_path": path.name,
                                  "class_id": "0" if clip["event"] else "-1", "phoneme": "p" if clip["event"] else "",
                                  **{k: case["inventory"][k] for k in ("video_id", "source_group", "dataset_split")}})
                if clip["kind"] == "context":
                    context["windows"][0]["clip_sha256"] = file_sha256(path)
        def __getitem__(self, index):
            return {"decoded_frame_count": 3 if index == 0 else 16,
                    "frame_mask": torch.tensor([True] * 3 + [False] * 13)}
    base = ExistingCrops()
    output = tmp_path / "prepared"
    result = export_case(base, [0, 1], case["inventory"], output, {"fixture": True},
                         source=case["source"], audit=audit, support_loader=lambda _: (trace, context))
    assert [r["target"] for r in result["records"].values()] == [1, 1]
    assert [r["original_event_class_id"] for r in result["records"].values()] == ["0", "-1"]
    assert export_case(None, None, case["inventory"], output, {"fixture": True}) == result
    unknown = export_case(base, [0, 1], case["inventory"], tmp_path / "unknown", {"fixture": True},
                          source=case["source"], audit=audit, support_loader=lambda _: (None, None))
    assert all(r["target"] == -1 and r["target_mask"] is False for r in unknown["records"].values())
    bad_clock = deepcopy(trace)
    bad_clock["timing"]["fps"] = 0
    unknown_clock = export_case(base, [0, 1], case["inventory"], tmp_path / "unknown-clock", {"fixture": True},
                                source=case["source"], audit=audit, support_loader=lambda _: (bad_clock, context))
    assert all(r["target"] == -1 and r["target_mask"] is False for r in unknown_clock["records"].values())
    changed = deepcopy(case["source"])
    changed["split"] = "val"
    with pytest.raises(ValueError, match="Train-only"):
        export_case(base, [0, 1], case["inventory"], tmp_path / "wrong", {}, source=changed)
    assert not (tmp_path / "wrong").exists()


def test_label_free_validation_adapter_preserves_masks_and_unknown_train_targets(tmp_path):
    from test_auxiliary_inputs import SavedClips, targets

    from seepat.training.auxiliary_comparison_inputs import ComparisonDataset
    from seepat.training.auxiliary_inputs import export_inputs

    bases = {r: SavedClips(tmp_path / "media", r) for r in ("train", "inner")}
    saved = {r: b.records for r, b in bases.items()}
    target_rows = targets(bases["train"])
    target_rows[next(iter(target_rows))].update(target=-1, target_mask=False)
    inventories = {r: [{"video_id": f"{r}-{n}", "status": "paired"} for n in (0, 1)] for r in bases}
    output = tmp_path / "inputs"
    export_inputs(bases, inventories, saved, target_rows, output, {})
    train, inner = (ComparisonDataset(bases[r], output, r) for r in ("train", "inner"))
    assert train[0]["auxiliary_target"].item() == -1 and not train[0]["auxiliary_mask"].item()
    assert "auxiliary_target" not in inner[0]
    assert train.input_contract == inner.input_contract
    before = inner[0]["frame_mask"]
    assert before.tolist() == [True] * 16
    (bases["inner"].root / bases["inner"].rows[0]["mouth_clip_path"]).write_bytes(b"changed crop")
    with pytest.raises(ValueError, match="clip"):
        inner[0]
