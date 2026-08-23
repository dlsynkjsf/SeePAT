from __future__ import annotations

import pytest

pytest.importorskip("torch")
overfit_module = pytest.importorskip("seepat.training.overfit")
select_balanced_event_indices = overfit_module.select_balanced_event_indices


def test_balanced_smoke_selection_uses_distinct_source_groups() -> None:
    rows = [
        {"class_id": "1", "source_group": "fake-a"},
        {"class_id": "1", "source_group": "fake-a"},
        {"class_id": "1", "source_group": "fake-b"},
        {"class_id": "0", "source_group": "fake-a"},
        {"class_id": "0", "source_group": "real-a"},
        {"class_id": "0", "source_group": "real-b"},
    ]

    indices = select_balanced_event_indices(rows, events_per_class=2)

    assert indices == [0, 2, 4, 5]
    assert len({rows[index]["source_group"] for index in indices}) == 4


def test_balanced_smoke_selection_rejects_insufficient_groups() -> None:
    rows = [
        {"class_id": "1", "source_group": "fake-a"},
        {"class_id": "0", "source_group": "real-a"},
    ]

    with pytest.raises(ValueError, match="distinct-source"):
        select_balanced_event_indices(rows, events_per_class=2)
