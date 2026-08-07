# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only fixed-sequence token storage shared by every distributed stage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["PackedTokenMemmapDataset"]


class PackedTokenMemmapDataset(Dataset):
    """Map ``N x (S+1)`` uint32 tokens to shifted input/target tensors."""

    def __init__(
        self,
        path: str | Path,
        *,
        limit: int | None = None,
        sequence_length: int | None = None,
    ):
        self.path = Path(path)
        metadata_path = self.path.with_suffix(self.path.suffix + ".json")
        if not self.path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"packed token cache is incomplete: data={self.path} metadata={metadata_path}. "
                "Run tokenize_data with caches that write these paths "
                "(see train_token_cache_path / validation_token_cache_path), "
                "or remove a stale manifests/tokenize_data.json with status=skipped."
            )
        self.metadata = json.loads(metadata_path.read_text())
        if self.metadata.get("status") != "complete":
            raise RuntimeError(f"packed token cache is not complete: {metadata_path}")
        self.num_samples = int(self.metadata["num_samples"])
        self.seq_length = int(self.metadata["seq_length"])
        self.sequence_length = (
            self.seq_length if sequence_length is None else int(sequence_length)
        )
        if not 0 < self.sequence_length <= self.seq_length:
            raise ValueError(
                "packed token cache sequence_length must be in "
                f"[1, {self.seq_length}], got {self.sequence_length}"
            )
        expected_bytes = self.num_samples * (self.seq_length + 1) * np.dtype(np.uint32).itemsize
        if self.path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"packed token cache size {self.path.stat().st_size} != {expected_bytes}"
            )
        self.limit = self.num_samples if limit is None else min(int(limit), self.num_samples)
        self._tokens = None

    def _array(self):
        if self._tokens is None:
            self._tokens = np.memmap(
                self.path,
                dtype=np.uint32,
                mode="r",
                shape=(self.num_samples, self.seq_length + 1),
            )
        return self._tokens

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self._array()[int(index), : self.sequence_length + 1]
        # Copy from the read-only map so PyTorch never exposes a writable view
        # and pinned-memory transfer can proceed independently of mmap lifetime.
        values = torch.from_numpy(np.array(row, dtype=np.int64, copy=True))
        return {"input_ids": values[:-1], "targets": values[1:]}
