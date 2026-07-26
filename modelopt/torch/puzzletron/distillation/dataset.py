# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Puzzle-KD text dataset adapter for native AutoModel global KD."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from datasets import DatasetDict, load_from_disk
from torch.utils.data import IterableDataset

from ..utils.data.dataset import ConstantLengthDataset
from ..utils.data.packed_memmap import PackedTokenMemmapDataset

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "collate_puzzletron_llm_batch",
    "make_puzzletron_chat_dataset",
    "make_puzzletron_llm_dataset",
    "make_puzzletron_llm_overfit_dataset",
]


def collate_puzzletron_llm_batch(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack fixed-length tensor samples without list-oriented padding.

    Puzzletron's packed-token cache already emits equally sized tensors.  The
    generic AutoModel collator is intended for variable-length Python lists and
    attempts ``tensor + list`` while padding these samples.  Keep this collator
    dataset-specific and preserve AutoModel's usual ``padding_mask`` contract.
    """
    if not batch:
        raise ValueError("cannot collate an empty Puzzletron LLM batch")
    keys = tuple(batch[0])
    if any(tuple(sample) != keys for sample in batch):
        raise ValueError("Puzzletron LLM samples have inconsistent fields")
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [sample[key] for sample in batch]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"Puzzletron LLM field {key!r} must contain tensors")
        shapes = {tuple(value.shape) for value in values}
        if len(shapes) != 1:
            raise ValueError(
                f"Puzzletron LLM field {key!r} is not fixed-length: {sorted(shapes)}"
            )
        result[key] = torch.stack(values, dim=0)
    if "attention_mask" in result:
        result["padding_mask"] = ~result["attention_mask"].bool()
    return result


def make_puzzletron_chat_dataset(
    tokenizer,
    dataset_path: str,
    split: str = "train",
    num_samples: int | None = None,
    seq_length: int = 4096,
    seed: int = 444,
    **_: object,
):
    """Load saved, Hub, JSON, or Parquet messages through AutoModel chat formatting."""
    from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
    from nemo_automodel.components.datasets.llm.formatting_utils import _add_pad_token

    path = Path(dataset_path)
    saved_to_disk = path.is_dir() and any(
        (path / marker).is_file()
        for marker in ("dataset_dict.json", "dataset_info.json", "state.json")
    )
    if not saved_to_disk:
        dataset = ChatDataset(
            dataset_path,
            tokenizer,
            split=None if split == "__auto__" else split,
            seq_length=int(seq_length),
            padding="do_not_pad",
            truncation=True,
            shuffle_seed=int(seed),
        )
        if isinstance(dataset.dataset, list):
            random.Random(int(seed)).shuffle(dataset.dataset)
        if num_samples is not None:
            count = min(int(num_samples), len(dataset.dataset))
            if hasattr(dataset.dataset, "select"):
                dataset.dataset = dataset.dataset.select(range(count))
            else:
                dataset.dataset = dataset.dataset[:count]
        return dataset

    loaded = load_from_disk(path)
    if isinstance(loaded, DatasetDict):
        if split == "__auto__":
            split = "validation" if "validation" in loaded else next(iter(loaded))
        if split not in loaded:
            raise KeyError(f"dataset {dataset_path} has no split {split!r}: {list(loaded)}")
        loaded = loaded[split]
    loaded = loaded.shuffle(seed=int(seed))
    if num_samples is not None:
        loaded = loaded.select(range(min(int(num_samples), len(loaded))))

    dataset = ChatDataset.__new__(ChatDataset)
    dataset.tokenizer = tokenizer
    dataset.seq_length = int(seq_length)
    dataset.padding = "do_not_pad"
    dataset.truncation = True
    dataset.start_of_turn_token = None
    dataset.mask_reasoning_content = False
    dataset.mask_history = False
    dataset.unshifted = False
    dataset.skip_invalid_samples = False
    dataset.dataset = loaded
    eos_token_id = getattr(tokenizer, "eos_token_id", 0)
    dataset.pad_token_id = _add_pad_token(tokenizer) or eos_token_id
    return dataset


class _LimitedAutoModelDataset(IterableDataset):
    def __init__(
        self,
        source: IterableDataset,
        count: int,
        *,
        num_shards: int = 1,
        shard_index: int = 0,
    ):
        self.source = source
        self.count = int(count)
        self.num_shards = int(num_shards)
        self.shard_index = int(shard_index)

    def shard(self, num_shards: int, index: int):
        """Return the DP-local strided view expected by AutoModel train_ft."""
        if num_shards < 1 or not 0 <= index < num_shards:
            raise ValueError(f"invalid dataset shard {index}/{num_shards}")
        return type(self)(
            self.source,
            self.count,
            num_shards=int(num_shards),
            shard_index=int(index),
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for index, sample in enumerate(self.source):
            if index >= self.count:
                return
            if index % self.num_shards != self.shard_index:
                continue
            input_ids = sample["input_ids"]
            labels = sample.get("targets", sample.get("labels"))
            if labels is None:
                raise KeyError("Puzzle-KD sample has neither targets nor labels")
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
            }


def make_puzzletron_llm_dataset(
    tokenizer,
    dataset_path: str,
    split: str = "train",
    num_samples: int = 2048,
    seq_length: int = 131072,
    seed: int = 444,
    packed_token_cache_path: str | None = None,
    **_: object,
) -> IterableDataset:
    """Pack the real local Puzzle-KD messages into deterministic fixed-length samples."""
    if packed_token_cache_path:
        return _PackedAutoModelDataset(
            PackedTokenMemmapDataset(
                packed_token_cache_path,
                limit=int(num_samples),
                sequence_length=int(seq_length),
            )
        )

    loaded = load_from_disk(dataset_path)
    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise KeyError(f"dataset {dataset_path} has no split {split!r}: {list(loaded)}")
        loaded = loaded[split]
    loaded = loaded.shuffle(seed=int(seed))
    packed = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=loaded,
        infinite=True,
        seq_length=int(seq_length),
        num_of_sequences=4,
        content_field="messages",
        fim_rate=0.0,
        fim_spm_rate=0.0,
        seed=int(seed),
        label_shift=False,
        bos_rate=1.0,
    )
    return _LimitedAutoModelDataset(packed, int(num_samples))


class _PackedAutoModelDataset(IterableDataset):
    def __init__(
        self,
        source: PackedTokenMemmapDataset,
        *,
        num_shards: int = 1,
        shard_index: int = 0,
    ):
        self.source = source
        self.num_shards = int(num_shards)
        self.shard_index = int(shard_index)

    def shard(self, num_shards: int, index: int):
        """Keep CP replicas aligned while giving each DP rank unique samples."""
        if num_shards < 1 or not 0 <= index < num_shards:
            raise ValueError(f"invalid dataset shard {index}/{num_shards}")
        return type(self)(
            self.source,
            num_shards=int(num_shards),
            shard_index=int(index),
        )

    def __iter__(self):
        for index in range(self.shard_index, len(self.source), self.num_shards):
            sample = self.source[index]
            input_ids = sample["input_ids"]
            yield {
                "input_ids": input_ids,
                "labels": sample["targets"],
                "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
            }


class _FrozenAutoModelDataset(IterableDataset):
    """A replayable DP-shardable snapshot used by distillation overfit."""

    def __init__(self, samples, *, num_shards: int = 1, shard_index: int = 0):
        self.samples = tuple(samples)
        self.num_shards = int(num_shards)
        self.shard_index = int(shard_index)

    def shard(self, num_shards: int, index: int):
        if num_shards < 1 or not 0 <= index < num_shards:
            raise ValueError(f"invalid dataset shard {index}/{num_shards}")
        return type(self)(
            self.samples,
            num_shards=int(num_shards),
            shard_index=int(index),
        )

    def __iter__(self):
        for index in range(self.shard_index, len(self.samples), self.num_shards):
            yield {key: value.clone() for key, value in self.samples[index].items()}

    def __len__(self):
        remaining = max(0, len(self.samples) - self.shard_index)
        return (remaining + self.num_shards - 1) // self.num_shards


def make_puzzletron_llm_overfit_dataset(
    tokenizer,
    dataset_path: str,
    split: str = "train",
    num_samples: int = 128,
    seq_length: int = 128,
    seed: int = 444,
    packed_token_cache_path: str | None = None,
    **kwargs: object,
) -> IterableDataset:
    """Materialize one deterministic minibatch and replay it every epoch."""
    source = make_puzzletron_llm_dataset(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        split=split,
        num_samples=int(num_samples),
        seq_length=int(seq_length),
        seed=int(seed),
        packed_token_cache_path=packed_token_cache_path,
        **kwargs,
    )
    samples = list(source)
    if len(samples) != int(num_samples):
        raise RuntimeError(
            f"distillation overfit expected {num_samples} frozen samples, got {len(samples)}"
        )
    return _FrozenAutoModelDataset(samples)
