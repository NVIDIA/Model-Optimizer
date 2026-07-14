# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight-only surgery for latent-projected mixture-of-experts layers.

The layer descriptor supplies tensor names; this module owns the mathematical
contract.  A full-rank transform is function preserving and orders the latent
coordinates so a smaller child can be realized by taking a prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, MutableMapping, Sequence

import torch

__all__ = [
    "LatentMoETensorLayout",
    "LatentMoETransform",
    "apply_latent_moe_transform",
    "apply_latent_moe_sort",
    "compute_latent_moe_transform",
    "reverse_latent_moe_transform",
]


@dataclass(frozen=True)
class LatentMoETensorLayout:
    """Logical latent-MoE tensors in a checkpoint.

    Split HF checkpoints use one key per expert.  A native grouped checkpoint
    adapter can expose views using the same logical sequence, keeping the
    decomposition independent of the storage format.
    """

    fc1_key: str
    fc2_key: str
    expert_up_keys: tuple[str, ...]
    expert_down_keys: tuple[str, ...]

    def validate(self, state_dict: Mapping[str, torch.Tensor]) -> int:
        if not self.expert_up_keys or len(self.expert_up_keys) != len(self.expert_down_keys):
            raise ValueError("latent MoE requires matching, non-empty expert up/down tensors")
        missing = [
            key
            for key in (self.fc1_key, self.fc2_key, *self.expert_up_keys, *self.expert_down_keys)
            if key not in state_dict
        ]
        if missing:
            raise KeyError(f"latent MoE checkpoint is missing tensors: {missing[:8]}")
        fc1 = state_dict[self.fc1_key]
        fc2 = state_dict[self.fc2_key]
        if fc1.ndim != 2 or fc2.ndim != 2:
            raise ValueError("latent projections must be rank-2 tensors")
        latent = int(fc1.shape[0])
        if fc2.shape[1] != latent:
            raise ValueError(
                f"latent projection mismatch: {self.fc1_key} rows={latent}, "
                f"{self.fc2_key} columns={fc2.shape[1]}"
            )
        for up_key, down_key in zip(self.expert_up_keys, self.expert_down_keys):
            up, down = state_dict[up_key], state_dict[down_key]
            if up.ndim != 2 or up.shape[1] != latent:
                raise ValueError(f"{up_key} must have latent input width {latent}, got {tuple(up.shape)}")
            if down.ndim != 2 or down.shape[0] != latent:
                raise ValueError(
                    f"{down_key} must have latent output width {latent}, got {tuple(down.shape)}"
                )
        return latent


@dataclass(frozen=True)
class LatentMoETransform:
    """Full-rank input basis and output factorization for a latent MoE."""

    input_basis: torch.Tensor
    output_basis: torch.Tensor
    output_compressor: torch.Tensor
    transformed_fc2: torch.Tensor


def _finite_square(
    name: str,
    value: torch.Tensor,
    size: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != (size, size):
        shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
        raise ValueError(f"{name} must have shape {(size, size)}, got {shape}")
    value = value.detach().to(dtype=torch.float64, device=device)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return (value + value.T) * 0.5


def compute_latent_moe_transform(
    fc1: torch.Tensor,
    fc2: torch.Tensor,
    expert_ups: Sequence[torch.Tensor],
    expert_weights: torch.Tensor,
    latent_cov_in: torch.Tensor,
    latent_cov_out: torch.Tensor,
    *,
    compute_device: torch.device | str = "cpu",
) -> LatentMoETransform:
    """Compute the proper Deci sensitivity-weighted full-rank transform."""

    compute_device = torch.device(compute_device)
    w_in = fc1.detach().to(dtype=torch.float64, device=compute_device)
    w_out = fc2.detach().to(dtype=torch.float64, device=compute_device)
    latent, _ = w_in.shape
    if w_out.ndim != 2 or w_out.shape[1] != latent:
        raise ValueError(f"fc2 must have shape [hidden, {latent}], got {tuple(w_out.shape)}")
    # The scorer collects z = W_in x directly, so this is exactly the old
    # W_in Sigma_x W_in^T matrix without materializing hidden_size^2 covariance.
    z_in = _finite_square(
        "latent_cov_in", latent_cov_in, latent, device=compute_device
    )

    alpha = expert_weights.detach().to(dtype=torch.float64, device=compute_device).flatten()
    if alpha.numel() != len(expert_ups):
        raise ValueError(
            f"expert_weights has {alpha.numel()} entries for {len(expert_ups)} expert tensors"
        )
    if not torch.isfinite(alpha).all() or torch.any(alpha < 0) or not torch.any(alpha > 0):
        raise ValueError("expert_weights must be finite, non-negative, and contain a positive value")
    alpha = alpha / alpha.sum()
    sensitivity = torch.zeros((latent, latent), dtype=torch.float64, device=compute_device)
    for weight, expert_up in zip(alpha, expert_ups):
        if weight <= 0:
            continue
        up = expert_up.detach().to(dtype=torch.float64, device=compute_device)
        if up.ndim != 2 or up.shape[1] != latent:
            raise ValueError(f"expert up tensor must have {latent} columns, got {tuple(up.shape)}")
        sensitivity.addmm_(up.T, up, beta=1.0, alpha=float(weight))

    # S @ Z is generally non-symmetric.  The ordered left singular vectors are
    # the input basis used by the original Nemotron/Deci implementation.
    input_basis, _, _ = torch.linalg.svd(sensitivity @ z_in, full_matrices=True)

    z_out = _finite_square(
        "latent_cov_out",
        latent_cov_out,
        latent,
        device=compute_device,
    )
    q, r = torch.linalg.qr(w_out, mode="reduced")
    if tuple(r.shape) != (latent, latent):
        raise ValueError(
            f"fc2 must have at least latent_dim={latent} rows for reduced QR, got {tuple(w_out.shape)}"
        )
    metric = (r @ z_out @ r.T)
    metric = (metric + metric.T) * 0.5
    _, vectors = torch.linalg.eigh(metric)
    output_basis = vectors.flip(dims=(1,)).contiguous()
    transformed_fc2 = q @ output_basis
    output_compressor = output_basis.T @ r
    return LatentMoETransform(
        input_basis=input_basis,
        output_basis=output_basis,
        output_compressor=output_compressor,
        transformed_fc2=transformed_fc2,
    )


def reverse_latent_moe_transform(transform: LatentMoETransform) -> LatentMoETransform:
    """Reverse both ordered latent bases without changing the represented function."""

    latent = int(transform.input_basis.shape[1])
    order = torch.arange(latent - 1, -1, -1, dtype=torch.long)
    return LatentMoETransform(
        input_basis=transform.input_basis[:, order].contiguous(),
        output_basis=transform.output_basis[:, order].contiguous(),
        output_compressor=transform.output_compressor[order, :].contiguous(),
        transformed_fc2=transform.transformed_fc2[:, order].contiguous(),
    )


def _bias_key(weight_key: str) -> str:
    return weight_key[: -len(".weight")] + ".bias" if weight_key.endswith(".weight") else weight_key + ".bias"


def _cast_like(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return value.to(device=target.device, dtype=target.dtype)


def apply_latent_moe_sort(
    state_dict: MutableMapping[str, torch.Tensor],
    layout: LatentMoETensorLayout,
    *,
    latent_cov_in: torch.Tensor,
    expert_weights: torch.Tensor,
    latent_cov_out: torch.Tensor,
    reverse: bool = False,
    compute_device: torch.device | str = "cpu",
    tensor_loader: Callable[[str], torch.Tensor | None] | None = None,
) -> LatentMoETransform:
    """Apply a function-preserving full-rank latent transform in place.

    ``state_dict`` may be one checkpoint shard.  ``tensor_loader`` supplies the
    original logical tensors needed to compute a single global transform; only
    tensors resident in ``state_dict`` are rewritten.
    """

    def original(key: str) -> torch.Tensor:
        value = tensor_loader(key) if tensor_loader is not None else None
        if value is None:
            value = state_dict.get(key)
        if value is None:
            raise KeyError(f"latent MoE checkpoint is missing tensor {key}")
        return value

    if not layout.expert_up_keys or len(layout.expert_up_keys) != len(layout.expert_down_keys):
        raise ValueError("latent MoE requires matching, non-empty expert up/down tensors")
    source = {key: original(key) for key in (layout.fc1_key, layout.fc2_key, *layout.expert_up_keys)}
    fc1 = source[layout.fc1_key]
    fc2 = source[layout.fc2_key]
    transform = compute_latent_moe_transform(
        fc1,
        fc2,
        [source[key] for key in layout.expert_up_keys],
        expert_weights,
        latent_cov_in,
        latent_cov_out,
        compute_device=compute_device,
    )
    if reverse:
        transform = reverse_latent_moe_transform(transform)
    apply_latent_moe_transform(state_dict, layout, transform)
    return transform


def apply_latent_moe_transform(
    state_dict: MutableMapping[str, torch.Tensor],
    layout: LatentMoETensorLayout,
    transform: LatentMoETransform,
) -> None:
    """Apply a precomputed full-rank transform to tensors resident in one shard."""

    p = transform.input_basis
    compressor = transform.output_compressor

    if layout.fc1_key in state_dict:
        target = state_dict[layout.fc1_key]
        target_p = p.to(device=target.device)
        state_dict[layout.fc1_key] = _cast_like(
            target_p.T @ target.detach().to(device=target.device, dtype=torch.float64), target
        )
    fc1_bias_key = _bias_key(layout.fc1_key)
    if fc1_bias_key in state_dict:
        bias = state_dict[fc1_bias_key]
        bias_p = p.to(device=bias.device)
        state_dict[fc1_bias_key] = _cast_like(
            bias_p.T @ bias.detach().to(device=bias.device, dtype=torch.float64), bias
        )

    for up_key in layout.expert_up_keys:
        if up_key not in state_dict:
            continue
        up = state_dict[up_key]
        up_p = p.to(device=up.device)
        state_dict[up_key] = _cast_like(
            up.detach().to(device=up.device, dtype=torch.float64) @ up_p, up
        )

    for down_key in layout.expert_down_keys:
        if down_key not in state_dict:
            continue
        down = state_dict[down_key]
        down_compressor = compressor.to(device=down.device)
        state_dict[down_key] = _cast_like(
            down_compressor @ down.detach().to(device=down.device, dtype=torch.float64), down
        )
        down_bias_key = _bias_key(down_key)
        if down_bias_key in state_dict:
            bias = state_dict[down_bias_key]
            bias_compressor = compressor.to(device=bias.device)
            state_dict[down_bias_key] = _cast_like(
                bias_compressor @ bias.detach().to(device=bias.device, dtype=torch.float64), bias
            )

    if layout.fc2_key in state_dict:
        transformed_fc2 = transform.transformed_fc2.to(device=state_dict[layout.fc2_key].device)
        state_dict[layout.fc2_key] = _cast_like(
            transformed_fc2, state_dict[layout.fc2_key]
        )
