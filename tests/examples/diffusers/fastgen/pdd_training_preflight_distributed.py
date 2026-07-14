# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-rank proof that training-input failures propagate before the model call."""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.distributed as dist

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

from pdd_finetune import _collective_training_batch, _collective_training_iterator


class _Sampler:
    def __init__(self, sample_ids: tuple[str, ...]) -> None:
        self.sample_ids = sample_ids
        self.epoch = 0
        self.remaining_batches = 1

    def expected_next_sample_ids(self) -> tuple[str, ...]:
        return self.sample_ids

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.remaining_batches = 1


class _Loader:
    def __init__(self, batch: dict, *, fail: bool) -> None:
        self.batch = batch
        self.fail = fail

    def __iter__(self):
        if self.fail:
            raise OSError("injected iterator construction failure")
        return iter([self.batch])


def _batch(sample_id: str) -> dict:
    return {
        "image_latents": torch.ones(1, 3, 4, 4),
        "text_embeddings": torch.ones(1, 5, 6),
        "text_embeddings_mask": torch.ones(1, 5, dtype=torch.bool),
        "metadata": {"sample_ids": [sample_id]},
    }


def _expect_collective_failure(iterator, sampler: _Sampler, expected: str) -> None:
    message = None
    try:
        _collective_training_batch(
            iterator,
            sampler=sampler,
            resume=None,
            resume_pending=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
            require_negative_condition=False,
            expected_batch_size=1,
        )
    except RuntimeError as error:
        message = str(error)
    messages: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(messages, message)
    assert all(item is not None and expected in item for item in messages)


def main() -> None:
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        sample_id = f"sample-rank-{rank}"
        expected_ids = ("wrong-rank-0",) if rank == 0 else (sample_id,)
        _expect_collective_failure(
            iter([_batch(sample_id)]),
            _Sampler(expected_ids),
            "committed cursor",
        )
        dist.barrier()

        malformed = _batch(sample_id)
        if rank == 1:
            malformed.pop("text_embeddings")
        _expect_collective_failure(
            iter([malformed]),
            _Sampler((sample_id,)),
            "missing required keys",
        )
        dist.barrier()

        iterator_message = None
        try:
            _collective_training_iterator(
                _Loader(_batch(sample_id), fail=rank == 0),
                _Sampler((sample_id,)),
            )
        except RuntimeError as error:
            iterator_message = str(error)
        iterator_messages: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(iterator_messages, iterator_message)
        assert all(
            item is not None and "iterator construction" in item for item in iterator_messages
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
