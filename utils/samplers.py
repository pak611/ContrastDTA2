from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Dict, Iterable, Iterator, List, Sequence

import torch
from torch.utils.data import Sampler


class GroupedBatchSampler(Sampler[List[int]]):
    """
    Yield batches that contain at least `per_target` samples per target when possible,
    using round-robin across target groups. Falls back gracefully when groups are small.

    Args:
        target_ids: a sequence of per-item target ids aligned with dataset indices
        batch_size: desired batch size
        per_target: number of samples per target to include per batch (best-effort)
        drop_last: drop tail batch smaller than batch_size
        shuffle: shuffle group order and indices each epoch
        seed: optional seed for determinism
    """

    def __init__(
        self,
        target_ids: Sequence[int],
        batch_size: int,
        per_target: int = 2,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        self.target_ids = list(target_ids)
        self.batch_size = int(batch_size)
        self.per_target = max(1, int(per_target))
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = seed

        # build groups
        groups: Dict[int, List[int]] = defaultdict(list)
        for idx, tid in enumerate(self.target_ids):
            groups[int(tid)].append(idx)
        self.groups = groups
        self.group_keys = list(groups.keys())

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        keys = list(self.group_keys)
        if self.shuffle:
            rng.shuffle(keys)
        # materialize deques and shuffle indices within each group
        deques = []
        for k in keys:
            idxs = list(self.groups[k])
            if self.shuffle:
                rng.shuffle(idxs)
            deques.append(deque(idxs))
        # round-robin draw per_target indices per group to form batches
        batch: List[int] = []
        while deques:
            i = 0
            while i < len(deques) and len(batch) < self.batch_size:
                dq = deques[i]
                taken = 0
                while dq and (taken < self.per_target) and (len(batch) < self.batch_size):
                    batch.append(dq.popleft())
                    taken += 1
                # remove empty deques
                if not dq:
                    deques.pop(i)
                    continue
                i += 1
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            else:
                # no more full batches
                break
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        # approximate number of batches
        n = len(self.target_ids)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
