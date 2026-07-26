# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical sample metadata handling for padded and packed activation hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["flatten_sample_tokens"]


def flatten_sample_tokens(
    tensor: torch.Tensor,
    *,
    scored_dim: int,
    sequence_ids: torch.Tensor,
    sequence_cursor: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return ``[tokens, other_features, scored]`` and aligned sample IDs.

    Padded activations begin with ``[batch, sequence]`` while packed THD
    activations begin with one flattened token dimension. ``sequence_ids``
    remains the canonical two-dimensional metadata in both cases.
    """
    if tensor.ndim < 2:
        raise ValueError(f"sample-aware activation must have at least two dimensions, got {tuple(tensor.shape)}")
    if sequence_ids.ndim != 2:
        raise ValueError("sequence_ids must be [batch, sequence]")

    scored_dim %= tensor.ndim
    available_rows = int(sequence_ids.shape[0]) - int(sequence_cursor)
    sequence = int(sequence_ids.shape[1])
    if (
        tensor.ndim >= 3
        and int(tensor.shape[0]) <= available_rows
        and int(tensor.shape[1]) == sequence
    ):
        rows = int(tensor.shape[0])
        token_dims = 2
    else:
        if sequence < 1 or int(tensor.shape[0]) % sequence:
            raise ValueError(
                f"packed activation shape {tuple(tensor.shape)} does not align with "
                f"sequence_ids shape {tuple(sequence_ids.shape)}"
            )
        rows = int(tensor.shape[0]) // sequence
        token_dims = 1
    if rows < 1 or rows > available_rows:
        raise ValueError("activation consumed more rows than the canonical batch metadata")
    if scored_dim < token_dims:
        raise ValueError("scored_dim cannot select a batch, sequence, or packed-token dimension")

    stop = int(sequence_cursor) + rows
    ids = sequence_ids[int(sequence_cursor) : stop].reshape(-1)
    values = tensor.movedim(scored_dim, -1)
    scored_size = int(values.shape[-1])
    token_count = int(ids.numel())
    if values.numel() % (token_count * scored_size):
        raise ValueError(
            f"activation shape {tuple(tensor.shape)} cannot be aligned with {token_count} tokens"
        )
    flat = values.reshape(token_count, -1, scored_size)
    return flat, ids, stop
