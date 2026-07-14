# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral losses used by Puzzletron block-local distillation."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import torch

from ..tools.kd_model import normalized_mse_loss

__all__ = [
    "batched_normalized_mse_loss",
    "normalized_mse_loss",
    "resolve_local_kd_loss",
    "vectorwise_normalized_mse_loss",
]


def vectorwise_normalized_mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute relative L2 per last-dimension vector, then average."""
    return batched_normalized_mse_loss(
        input,
        target,
        epsilon,
        batch_dims=range(input.ndim - 1),
    )


def batched_normalized_mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
    batch_dims: Sequence[int] = (0,),
) -> torch.Tensor:
    """Compute relative L2 independently over the non-batch dimensions."""
    input_shape = tuple(input.shape)
    target_shape = tuple(target.shape)
    if epsilon <= 0:
        raise ValueError(f"epsilon must be strictly positive, got {epsilon!r}")
    try:
        raw_batch_dims = tuple(operator.index(dim) for dim in batch_dims)
    except TypeError as exc:
        raise ValueError(
            "batch_dims must be an iterable of integer dimensions; "
            f"got {batch_dims!r} for input shape {input_shape} and target shape {target_shape}"
        ) from exc

    resolved_batch_dims: list[int] = []
    for dim in raw_batch_dims:
        if dim < -input.ndim or dim >= input.ndim:
            raise ValueError(
                f"batch_dims contains invalid dimension {dim} for input.ndim={input.ndim}; "
                f"input shape={input_shape}, target shape={target_shape}, "
                f"batch_dims={raw_batch_dims}, norm_dims=None"
            )
        resolved_batch_dims.append(dim % input.ndim)
    if len(set(resolved_batch_dims)) != len(resolved_batch_dims):
        raise ValueError(
            "batch_dims contains duplicate dimensions after normalization; "
            f"input shape={input_shape}, target shape={target_shape}, "
            f"batch_dims={tuple(resolved_batch_dims)}, norm_dims=None"
        )

    norm_dims = tuple(d for d in range(input.ndim) if d not in set(resolved_batch_dims))
    if input.ndim != target.ndim:
        raise ValueError(
            "input and target must have the same number of dimensions; "
            f"input shape={input_shape}, target shape={target_shape}, "
            f"batch_dims={tuple(resolved_batch_dims)}, norm_dims={norm_dims}"
        )
    if input_shape != target_shape:
        mismatched_dims = tuple(
            dim
            for dim, (input_size, target_size) in enumerate(zip(input_shape, target_shape))
            if input_size != target_size
        )
        raise ValueError(
            "input and target shapes must match exactly; "
            f"mismatched_dims={mismatched_dims}, input shape={input_shape}, "
            f"target shape={target_shape}, batch_dims={tuple(resolved_batch_dims)}, "
            f"norm_dims={norm_dims}"
        )

    numerator = ((input - target) ** 2).sum(dim=norm_dims)
    denominator = (target**2).sum(dim=norm_dims) + epsilon
    return (numerator / denominator).mean()


_LOCAL_KD_LOSSES = {
    "normalized_mse_loss": normalized_mse_loss,
    "vectorwise_normalized_mse_loss": vectorwise_normalized_mse_loss,
    "batched_normalized_mse_loss": batched_normalized_mse_loss,
}


def resolve_local_kd_loss(name: str):
    """Resolve a configured local-KD loss without importing an execution backend."""
    try:
        return _LOCAL_KD_LOSSES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported AutoModel local KD block loss {name!r}") from exc

