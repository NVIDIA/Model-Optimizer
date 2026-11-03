# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-mapped packed SFT data with explicit recurrent phase boundaries."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from megatron.bridge.data.base import DatasetProvider
from megatron.bridge.data.packing.in_batch import build_mcore_thd_sequence_batch_from_rows
from torch.utils.data import Dataset

__all__ = ["LinearAttentionPackedDatasetConfig"]


class _PackedStateDataset(Dataset):
    def __init__(self, root, split, seq_length, samples, seed, cp_size, pad_token_id):
        root = Path(root) / split
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["sequence_length"] != seq_length:
            raise ValueError("Prepared pack width differs from --seq_length")
        if manifest["alignment"] % (2 * cp_size):
            raise ValueError("Prepared pack alignment must be divisible by twice CP size")
        self.tokens = np.memmap(root / "tokens.bin", mode="r", dtype="<u4")
        self.sequences = np.load(root / "sequences.npy", mmap_mode="r", allow_pickle=False)
        self.packs = np.load(root / "packs.npy", mmap_mode="r", allow_pickle=False)
        self.order = np.random.default_rng(seed).permutation(len(self.packs))
        if not len(self.packs) or self.sequences.shape[1] != 3 or self.packs.shape[1] != 2:
            raise ValueError(
                "Packed state data needs sequences [offset,length,prefix] and packs [start,count]"
            )
        self.samples = samples
        self.seq_length = seq_length
        self.alignment = manifest["alignment"]
        self.pad_token_id = pad_token_id

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        start, count = self.packs[self.order[index % len(self.order)]]
        rows, prefixes = [], []
        for offset, length, prefix in self.sequences[start : start + count]:
            if not 0 <= prefix < length - 1 or offset < 0 or offset + length > len(self.tokens):
                raise ValueError("Invalid packed token offset or prompt/completion boundary")
            ids = torch.from_numpy(self.tokens[offset : offset + length].astype(np.int64))
            positions = torch.arange(length - 1)
            rows.append(
                {
                    "tokens": ids[:-1],
                    "labels": ids[1:],
                    "position_ids": positions,
                    "loss_mask": (positions >= max(int(prefix) - 1, 0)).float(),
                }
            )
            prefixes.append(int(prefix))
        return rows, prefixes

    def collate_fn(self, batch):
        if len(batch) != 1:
            raise ValueError("Offline packed linear-attention data requires --mbs 1")
        rows, prefixes = batch[0]
        result = build_mcore_thd_sequence_batch_from_rows(
            rows,
            token_key="tokens",
            sequence_length=self.seq_length,
            pad_token_id=self.pad_token_id,
            pad_to_multiple_of=self.alignment,
            pad_to_max_length=True,
            emit_padding_mask=True,
        )
        result["prefill_lengths"] = prefixes
        result["sequence_lengths"] = [row["tokens"].numel() for row in rows]
        return result


@dataclass(kw_only=True)
class LinearAttentionPackedDatasetConfig(DatasetProvider):
    """Read offline packs through Bridge's standard sampler and training loop.

    Each split contains tokens.bin (uint32), sequences.npy ([offset,length,prefix]),
    packs.npy ([first_sequence,count]), and manifest.json (sequence_length, alignment).
    The data builder owns conversation-level splitting and records source provenance.
    """

    dataset_root: str
    seq_length: int
    context_parallel_size: int = 1
    seed: int = 2026
    do_validation: bool = True
    skip_getting_attention_mask_from_dataset: bool = True
    dataloader_type: str = "single"

    def build_datasets(self, context):
        """Build deterministic pack streams on every TP and CP rank."""

        def build(split, samples):
            return _PackedStateDataset(
                self.dataset_root,
                split,
                self.seq_length,
                samples,
                self.seed,
                self.context_parallel_size,
                context.tokenizer.eos_id,
            )

        return (
            build("training", context.train_samples),
            build("validation", context.valid_samples) if self.do_validation else None,
            None,
        )
