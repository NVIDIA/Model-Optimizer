# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pipeline-parallel helpers for native AutoModel VLM stages."""

from __future__ import annotations

from typing import Any

import torch
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate

__all__ = ["set_pp_vlm_chunk_specs"]


def set_pp_vlm_chunk_specs(
    schedule, kwargs: dict[str, Any], *, batch_size: int | None = None
) -> None:
    """Chunk batch-major PP inputs and replicate non-batch metadata.

    PyTorch's default chunks every tensor on dimension zero. That is wrong for
    mRoPE ``[axes, batch, sequence]`` and for CP metadata such as singleton
    cumulative-length tensors. A singleton kwarg also reduces the entire
    schedule to one chunk, while the PP schedule still expects all configured
    microbatches. Infer only genuine batch axes and replicate the rest.
    """
    if batch_size is None:
        for value in kwargs.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                batch_size = int(value.shape[0])
                break
    if batch_size is not None:
        schedule._args_chunk_spec = (TensorChunkSpec(0),)

    def chunk_spec(key: str, value: Any):
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            return _Replicate()
        if key == "position_ids" and value.ndim == 3 and value.shape[1] == batch_size:
            return TensorChunkSpec(1)
        if value.shape[0] == batch_size:
            return TensorChunkSpec(0)
        return _Replicate()

    schedule._kwargs_chunk_spec = {
        key: chunk_spec(key, value)
        for key, value in kwargs.items()
    }
