# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Committed-cursor wrapper for deterministic, prefetched batch samplers."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler

__all__ = ["ReplayableBatchSampler"]

_STATE_VERSION = 1


class ReplayableBatchSampler(Sampler[list[int]]):
    """Separate actually consumed batches from batches yielded to worker prefetch.

    The wrapped sampler remains the authority for deterministic rank/epoch batch plans.
    This wrapper materializes that plan compactly and advances its committed cursor only
    after the training loop confirms the logical sample IDs it consumed.
    """

    def __init__(self, sampler: Sampler[list[int]]) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError(f"sampler must be a torch Sampler, got {type(sampler).__name__}.")
        dataset = getattr(sampler, "dataset", None)
        metadata = getattr(dataset, "metadata", None)
        if not isinstance(metadata, Sequence):
            raise TypeError("sampler.dataset.metadata must be a sequence.")

        self.sampler = sampler
        self.dataset = dataset
        self.epoch = int(getattr(sampler, "epoch", 0))
        self.committed_batches = 0
        self.sample_slots_consumed = 0
        self._yielded_batches = 0
        self._flat_indices = torch.empty(0, dtype=torch.int64)
        self._offsets = torch.zeros(1, dtype=torch.int64)
        self._plan_sha256 = ""
        self._build_plan()

    def _build_plan(self) -> None:
        self.sampler.set_epoch(self.epoch)
        self.sampler.load_state_dict({"epoch": self.epoch, "batches_yielded": 0})
        batches = [tuple(int(index) for index in batch) for batch in self.sampler]
        if not batches:
            raise ValueError("replayable batch plan must contain at least one batch.")
        if any(not batch for batch in batches):
            raise ValueError("replayable batch plan cannot contain an empty batch.")

        flat = [index for batch in batches for index in batch]
        offsets = [0]
        for batch in batches:
            offsets.append(offsets[-1] + len(batch))
        self._flat_indices = torch.tensor(flat, dtype=torch.int64)
        self._offsets = torch.tensor(offsets, dtype=torch.int64)

        digest = hashlib.sha256()
        digest.update(b"modelopt-pdd-batch-plan-v1\0")
        digest.update(struct.pack(">q", self.epoch))
        digest.update(struct.pack(">q", int(getattr(self.sampler, "rank", 0))))
        digest.update(struct.pack(">q", int(getattr(self.sampler, "num_replicas", 1))))
        for batch in batches:
            digest.update(struct.pack(">q", len(batch)))
            for index in batch:
                digest.update(struct.pack(">q", index))
        self._plan_sha256 = digest.hexdigest()
        self._yielded_batches = self.committed_batches

    @property
    def plan_sha256(self) -> str:
        return self._plan_sha256

    @property
    def remaining_batches(self) -> int:
        return len(self) - self.committed_batches

    def _batch_indices(self, batch_index: int) -> list[int]:
        if not 0 <= batch_index < len(self):
            raise IndexError(f"batch_index={batch_index} is outside [0, {len(self)}).")
        start = int(self._offsets[batch_index])
        end = int(self._offsets[batch_index + 1])
        return self._flat_indices[start:end].tolist()

    def _sample_ids(self, batch_index: int) -> tuple[str, ...]:
        sample_ids: list[str] = []
        for index in self._batch_indices(batch_index):
            item = self.dataset.metadata[index]
            if not isinstance(item, Mapping) or not isinstance(item.get("sample_id"), str):
                raise ValueError(f"dataset.metadata[{index}] has no string sample_id.")
            sample_ids.append(item["sample_id"])
        return tuple(sample_ids)

    def expected_next_sample_ids(self) -> tuple[str, ...]:
        """Return the next committed batch's logical IDs without consuming it."""
        if self.committed_batches == len(self):
            return ()
        return self._sample_ids(self.committed_batches)

    def commit(self, sample_ids: Sequence[str]) -> None:
        """Advance the durable cursor after verifying the collated logical IDs."""
        if isinstance(sample_ids, str) or not isinstance(sample_ids, Sequence):
            raise TypeError("sample_ids must be a sequence of strings.")
        actual = tuple(sample_ids)
        if any(not isinstance(sample_id, str) for sample_id in actual):
            raise TypeError("sample_ids must contain only strings.")
        expected = self.expected_next_sample_ids()
        if not expected:
            raise RuntimeError("cannot commit beyond the end of the batch plan.")
        if actual != expected:
            raise RuntimeError(
                f"consumed sample IDs do not match the committed cursor: "
                f"expected={expected}, actual={actual}."
            )
        self.committed_batches += 1
        self.sample_slots_consumed += len(actual)

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be an integer >= 0.")
        if epoch == self.epoch:
            return
        if self.committed_batches != len(self):
            raise RuntimeError("cannot change epoch before every planned batch is committed.")
        self.epoch = epoch
        self.committed_batches = 0
        self._build_plan()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _STATE_VERSION,
            "epoch": self.epoch,
            "committed_batches": self.committed_batches,
            "sample_slots_consumed": self.sample_slots_consumed,
            "plan_sha256": self.plan_sha256,
            "next_sample_ids": list(self.expected_next_sample_ids()),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("replayable sampler state must be a mapping.")
        expected_keys = {
            "schema_version",
            "epoch",
            "committed_batches",
            "sample_slots_consumed",
            "plan_sha256",
            "next_sample_ids",
        }
        if set(state) != expected_keys:
            raise ValueError(
                f"replayable sampler state keys mismatch: expected={sorted(expected_keys)}, "
                f"actual={sorted(state)}."
            )
        if state["schema_version"] != _STATE_VERSION:
            raise ValueError(f"unsupported replayable sampler schema {state['schema_version']!r}.")
        epoch = state["epoch"]
        committed = state["committed_batches"]
        consumed = state["sample_slots_consumed"]
        if type(epoch) is not int or epoch < 0:
            raise ValueError("saved epoch must be an integer >= 0.")
        if type(committed) is not int or committed < 0:
            raise ValueError("saved committed_batches must be an integer >= 0.")
        if type(consumed) is not int or consumed < 0:
            raise ValueError("saved sample_slots_consumed must be an integer >= 0.")
        if not isinstance(state["plan_sha256"], str):
            raise TypeError("saved plan_sha256 must be a string.")
        if not isinstance(state["next_sample_ids"], list) or any(
            not isinstance(sample_id, str) for sample_id in state["next_sample_ids"]
        ):
            raise TypeError("saved next_sample_ids must be a list of strings.")

        self.epoch = epoch
        self.committed_batches = 0
        self._build_plan()
        if committed > len(self):
            raise ValueError(
                f"saved committed_batches={committed} exceeds plan length {len(self)}."
            )
        if self.plan_sha256 != state["plan_sha256"]:
            raise RuntimeError("reconstructed batch plan does not match the saved plan hash.")
        self.committed_batches = committed
        self.sample_slots_consumed = consumed
        self._yielded_batches = committed
        if list(self.expected_next_sample_ids()) != state["next_sample_ids"]:
            raise RuntimeError("reconstructed next sample IDs do not match the checkpoint.")

    def __iter__(self) -> Iterator[list[int]]:
        start = self.committed_batches
        flat_indices = self._flat_indices
        offsets = self._offsets
        total_batches = int(offsets.numel() - 1)
        self._yielded_batches = start
        for batch_index in range(start, total_batches):
            batch_start = int(offsets[batch_index])
            batch_end = int(offsets[batch_index + 1])
            self._yielded_batches = batch_index + 1
            yield flat_indices[batch_start:batch_end].tolist()

    def __len__(self) -> int:
        return int(self._offsets.numel() - 1)
