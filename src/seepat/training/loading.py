"""Reusable loaders for deterministic, explicitly ordered training epochs.

Only the parent process owns the index plan. Worker datasets remain unchanged,
so persistent workers can load a new plan without being respawned each epoch.
The caller supplies the exact old sampling order and must finish each planned
epoch before replacing it. This helper does not reseed arbitrary stochastic
transforms inside already-running workers; current SeePAT datasets are fixed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Self

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class _OrderedPlan(Sampler[int]):
    def __init__(self) -> None:
        self.indices: tuple[int, ...] = ()

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class _EpochIterator:
    def __init__(self, iterator, owner: ReusableTrainingLoader, batches: int) -> None:
        self.iterator = iterator
        self.owner = owner
        self.epoch = owner._epoch
        self.remaining = batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.epoch != self.owner._epoch:
            raise RuntimeError("This training iterator belongs to a replaced epoch plan")
        try:
            batch = next(self.iterator)
        except BaseException:
            self.owner._active = False
            raise
        self.remaining -= 1
        # Bounded readiness loops may break after the final planned batch instead
        # of asking for StopIteration. Every planned index has then been consumed.
        if self.remaining == 0:
            self.owner._active = False
        return batch


class _PlannedDataLoader(DataLoader):
    def __init__(self, owner: ReusableTrainingLoader, *args, **kwargs) -> None:
        self.owner = owner
        super().__init__(*args, **kwargs)

    def __iter__(self):
        if self.owner._closed:
            raise RuntimeError("The reusable training loader is closed")
        if self.owner._active:
            raise RuntimeError("Finish the current training epoch before iterating again")
        self.owner._active = len(self) > 0
        try:
            return _EpochIterator(super().__iter__(), self.owner, len(self))
        except BaseException:
            self.owner._active = False
            raise


class ReusableTrainingLoader:
    """Keep one DataLoader and its workers for explicit successive index plans.

    ``for_epoch`` takes global indices into ``dataset`` in their required order,
    including an already flattened complete-video plan when appropriate. It
    never shuffles or modifies global RNG state. Its private torch generator
    seeds worker initialization and is reset to the supplied seed each epoch.

    ``close`` releases this object's DataLoader reference without private torch
    shutdown hooks. Callers should also release any retained loader/iterator
    references so PyTorch can promptly clean up persistent workers.
    """

    def __init__(
        self, dataset: Dataset[Any], batch_size: int, workers: int, device: torch.device,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise ValueError("workers must be a nonnegative integer")
        self.dataset = dataset
        self._plan = _OrderedPlan()
        self._generator = torch.Generator()
        self._active = False
        self._closed = False
        self._epoch = 0
        self._loader: DataLoader[Any] | None = _PlannedDataLoader(
            self, dataset, batch_size=batch_size, sampler=self._plan,
            num_workers=workers, pin_memory=torch.device(device).type == "cuda",
            persistent_workers=workers > 0, generator=self._generator,
        )

    def for_epoch(self, indices: list[int], seed: int) -> DataLoader[Any]:
        if self._closed:
            raise RuntimeError("The reusable training loader is closed")
        if self._active:
            raise RuntimeError("Finish the current training epoch before replacing its plan")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if any(isinstance(index, bool) or not isinstance(index, int)
               or not 0 <= index < len(self.dataset) for index in indices):
            raise ValueError("Every planned index must be an integer within the dataset")
        self._generator.manual_seed(seed)
        self._plan.indices = tuple(indices)
        self._epoch += 1
        assert self._loader is not None
        return self._loader

    def close(self) -> None:
        self._closed = True
        self._active = False
        self._loader = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
