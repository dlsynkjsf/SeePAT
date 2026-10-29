from __future__ import annotations

import gc
import os
import random

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from seepat.training.loading import ReusableTrainingLoader


class IndexedDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {"index": index, "pid": os.getpid()}


def indexes(loader):
    return [value for batch in loader for value in batch["index"].tolist()]


def test_same_loader_changes_only_ordered_plan_and_keeps_complete_bags():
    reusable = ReusableTrainingLoader(IndexedDataset(), 1, 0, torch.device("cpu"))
    # Two complete video bags, with deliberately non-sorted event indices.
    first = reusable.for_epoch([3, 1, 5, 6, 2], 9)
    assert indexes(first) == [3, 1, 5, 6, 2]
    second = reusable.for_epoch([6, 2, 3, 1, 5], 10)
    assert second is first
    assert indexes(second) == [6, 2, 3, 1, 5]
    assert first.num_workers == 0 and not first.persistent_workers
    reusable.close()
    with pytest.raises(RuntimeError, match="closed"):
        reusable.for_epoch([0], 1)


def test_replanning_active_epoch_is_rejected_until_last_planned_batch():
    reusable = ReusableTrainingLoader(IndexedDataset(), 2, 0, torch.device("cpu"))
    iterator = iter(reusable.for_epoch([0, 1, 2], 4))
    assert next(iterator)["index"].tolist() == [0, 1]
    with pytest.raises(RuntimeError, match="Finish"):
        reusable.for_epoch([4, 5], 5)
    assert next(iterator)["index"].tolist() == [2]
    # No extra next() is required: readiness loops can break at a planned cap.
    second = reusable.for_epoch([4, 5], 5)
    assert indexes(second) == [4, 5]
    with pytest.raises(RuntimeError, match="replaced epoch"):
        next(iterator)
    reusable.close()


def test_empty_plan_and_repeated_indices_are_explicit_and_global_rng_is_unchanged():
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    with ReusableTrainingLoader(IndexedDataset(), 2, 0, torch.device("cpu")) as reusable:
        assert indexes(reusable.for_epoch([], 17)) == []
        assert indexes(reusable.for_epoch([5, 5, 0], 18)) == [5, 5, 0]
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]


@pytest.mark.parametrize("indices", [[-1], [8], [True], [1.5]])
def test_invalid_global_indices_are_rejected(indices):
    with (
        ReusableTrainingLoader(IndexedDataset(), 1, 0, torch.device("cpu")) as reusable,
        pytest.raises(ValueError, match="index"),
    ):
        reusable.for_epoch(indices, 1)


def test_spawned_worker_is_reused_across_epoch_plans():
    reusable = ReusableTrainingLoader(IndexedDataset(), 1, 1, torch.device("cpu"))
    loader = reusable.for_epoch([0, 1, 2], 40)
    first = list(loader)
    second_loader = reusable.for_epoch([5, 4, 3], 41)
    second = list(second_loader)
    assert second_loader is loader
    assert [batch["index"].item() for batch in first] == [0, 1, 2]
    assert [batch["index"].item() for batch in second] == [5, 4, 3]
    first_pids = {batch["pid"].item() for batch in first}
    assert first_pids == {batch["pid"].item() for batch in second}
    assert len(first_pids) == 1 and os.getpid() not in first_pids
    reusable.close()
    del loader, second_loader
    gc.collect()
