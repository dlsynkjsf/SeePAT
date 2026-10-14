"""Synthetic matched-count planning; no real media/models/data jobs."""

import copy

import pytest

from seepat.artifacts import stable_id
from seepat.context_inputs import evenly_spaced, paired_indices, validate_tables


def event(index, **changes):
    return {"event_id": f"video-event-{index:03d}", "video_id": "video",
            "source_group": "source", "dataset_split": "train",
            "video_phone_start_s": str(index / 2), "class_id": str(index % 2),
            "phoneme": "m", **changes}


def context(index, usable=True, **changes):
    return {"context_id": f"video-context-{index:03d}", "video_id": "video",
            "source_group": "source", "dataset_split": "train",
            "start_frame": index * 20, "technical_ok": usable, **changes}


@pytest.mark.parametrize("bilabials", [0, 1, 2, 3, 7, 16, 17, 50])
@pytest.mark.parametrize("usable", [0, 1, 4, 8])
def test_counts_bilabial_floor_and_original_evidence_are_preserved(bilabials, usable):
    events = [event(index) for index in range(bilabials)]
    windows = [context(index, index < usable) for index in range(8)]
    before = copy.deepcopy((events, windows))
    control, candidate = paired_indices(events, windows)
    assert len(control) == len(candidate) == min(16, bilabials)
    assert sum(row["kind"] == "bilabial" for row in candidate) >= (len(candidate) + 1) // 2
    assert sum(row["kind"] == "context" for row in candidate) == min(usable, bilabials // 2, 8)
    assert len({row["clip_id"] for row in candidate}) == len(candidate)
    assert (events, windows) == before
    assert all("class_id" not in row and "phoneme" not in row for row in candidate)


def test_annotations_labels_and_input_order_do_not_choose_clips():
    events = [event(index) for index in range(40)]
    contexts = [context(index) for index in range(8)]
    expected = paired_indices(events, contexts)
    for row in events:
        row.update(class_id="1", event_label="fake", fake_segments_json="[[0,1000]]")
    assert paired_indices(list(reversed(events)), list(reversed(contexts))) == expected


def test_one_event_and_no_usable_context_fallback_are_identical():
    control, candidate = paired_indices([event(0)], [context(index) for index in range(8)])
    assert control == candidate
    control, candidate = paired_indices([event(index) for index in range(8)], [context(0, False)])
    assert control == candidate


@pytest.mark.parametrize("change", [{"video_id": "other"}, {"dataset_split": "val"},
                                     {"source_group": "other"}])
def test_cross_video_or_split_context_is_rejected(change):
    with pytest.raises(ValueError, match="Mixed"):
        paired_indices([event(0), event(1)], [context(0, **change)])


def test_duplicate_ids_and_unknown_technical_status_stop():
    with pytest.raises(ValueError, match="Duplicate bilabial"):
        paired_indices([event(0), event(0)], [])
    with pytest.raises(ValueError, match="Duplicate context"):
        paired_indices([event(0)], [context(0), context(0)])
    with pytest.raises(ValueError, match="explicit technical"):
        paired_indices([event(0)], [context(0, technical_ok="False")])


def test_even_selection_never_adds_padding_votes():
    assert evenly_spaced(list(range(5)), 3) == [0, 2, 4]
    assert evenly_spaced([], 0) == []
    with pytest.raises(ValueError):
        evenly_spaced([1], 2)


def group_tables(group):
    """Minimal replay of the real Validation path-serialization pattern."""
    file = "vox_celeb_2/id07982/xBDzoHwuPvA/00433/real.mp4"
    raw = "VoxCeleb2//dev/mp4/id07982/xBDzoHwuPvA/00433.mp4"
    source = {"file": file, "split": "val", "original": raw,
              "source_group": raw, "modify_type": "real"}
    video = {**source, "video_id": stable_id(file), "pipeline_status": "complete",
             "eligibility_status": "eligible"}
    sample = {**event(0), "event_id": f"{video['video_id']}-000", "file": file,
              "video_id": video["video_id"], "dataset_split": "val", "source_group": group,
              "manipulation_modality": "real", "class_id": "0"}
    return ({"train": [], "val": [source]}, {"train": [], "val": [video]},
            {"train": [], "val": [sample]})


@pytest.mark.parametrize("group", [
    "VoxCeleb2//dev/mp4/id07982/xBDzoHwuPvA/00433.mp4",
    "VoxCeleb2/dev/mp4/id07982/xBDzoHwuPvA/00433.mp4",
    "VoxCeleb2\\dev\\mp4\\id07982\\xBDzoHwuPvA\\00433.mp4",
])
def test_validator_compares_equivalent_group_paths_canonically_without_mutation(group):
    sources, videos, events = group_tables(group)
    before = copy.deepcopy((sources, videos, events))
    grouped = validate_tables(sources, videos, events)
    assert list(grouped.values()) == [events["val"]]
    assert (sources, videos, events) == before


def test_paired_selection_uses_same_canonical_identity_as_validator():
    raw = "VoxCeleb2//dev/mp4/person/source.mp4"
    canonical = "VoxCeleb2/dev/mp4/person/source.mp4"
    events = [event(index, source_group=raw) for index in range(4)]
    contexts = [context(0, source_group=canonical)]
    before = copy.deepcopy((events, contexts))
    control, candidate = paired_indices(events, contexts)
    assert len(control) == len(candidate) == 4
    assert any(row["kind"] == "context" for row in candidate)
    assert {row["source_group"] for row in control + candidate} == {canonical}
    assert (events, contexts) == before


@pytest.mark.parametrize("group", ["", "VoxCeleb2/dev/mp4/other/source.mp4"])
def test_canonical_comparison_still_rejects_missing_or_genuinely_different_groups(group):
    sources, videos, events = group_tables(group)
    with pytest.raises(ValueError):
        validate_tables(sources, videos, events)
