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

"""Framework-neutral projection, training, and sampling primitives for PDD.

This module owns projection layout and fusion plus the data-dependent and
data-free objectives and block sampler. Model calls and architecture-specific
packing remain behind the adapter protocol and belong in
``modelopt.torch.fastgen.plugins``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from ..config import PDDConfig
from ..flow_matching import (
    add_noise,
    fusion_coefficients,
    integrate_interval_velocities,
    make_shifted_flow_grid,
)

__all__ = [
    "PDDLayerSpec",
    "PDDModelAdapter",
    "PDDOutputProjection",
    "PDDPipeline",
    "convert_to_pdd_output_projection",
]

PDDHeadLayout = Literal["channel_major", "patch_major"]
_HEAD_LAYOUTS = ("channel_major", "patch_major")


def _require_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return value


@dataclass(frozen=True)
class PDDLayerSpec:
    """Immutable description of an architecture's final PDD projection.

    ``channel_major`` stores widened outputs as ``[head, base_output]``.
    ``patch_major`` stores them as ``[patch, head, output_channel]`` and
    therefore requires the unpatched ``output_channels`` count.
    """

    projection_path: str
    head_layout: PDDHeadLayout
    output_channels: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.projection_path, str) or not self.projection_path:
            raise ValueError("projection_path must be a non-empty dotted module path.")
        if any(not part for part in self.projection_path.split(".")):
            raise ValueError(
                f"projection_path contains an empty component: {self.projection_path!r}."
            )
        if self.head_layout not in _HEAD_LAYOUTS:
            raise ValueError(
                f"head_layout must be one of {_HEAD_LAYOUTS}, got {self.head_layout!r}."
            )
        if self.head_layout == "channel_major":
            if self.output_channels is not None:
                raise ValueError("channel_major layout does not use output_channels.")
        else:
            _require_int(self.output_channels, name="output_channels")


class PDDOutputProjection(nn.Linear):
    """A widened linear projection with one output head per PDD interval.

    ``forward`` returns the full widened output unless an explicit fusion tuple
    ``(start, end, grid)`` is supplied. Fused parameters are computed in float32
    and applied without mutating the module or replacing its registered weights.
    """

    def __init__(
        self,
        in_features: int,
        base_out_features: int,
        grid_size: int,
        layer_spec: PDDLayerSpec,
        *,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize an unpopulated widened projection with validated layout metadata."""
        _require_int(in_features, name="in_features")
        _require_int(base_out_features, name="base_out_features")
        _require_int(grid_size, name="grid_size")
        if not isinstance(layer_spec, PDDLayerSpec):
            raise TypeError(f"layer_spec must be PDDLayerSpec, got {type(layer_spec).__name__}.")
        if type(bias) is not bool:
            raise ValueError(f"bias must be bool, got {bias!r}.")
        if layer_spec.head_layout == "patch_major":
            output_channels = layer_spec.output_channels
            if output_channels is None or base_out_features % output_channels != 0:
                raise ValueError(
                    f"base_out_features={base_out_features} must be divisible by "
                    f"output_channels={output_channels} for patch_major layout."
                )

        super().__init__(
            in_features,
            base_out_features * grid_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.base_out_features = base_out_features
        self.grid_size = grid_size
        self.layer_spec = layer_spec

    @property
    def patch_factor(self) -> int:
        """Number of output patches represented by the base projection."""
        if self.layer_spec.head_layout == "channel_major":
            return 1
        output_channels = self.layer_spec.output_channels
        if output_channels is None:  # guarded by PDDLayerSpec validation
            raise RuntimeError("patch_major projection is missing output_channels.")
        return self.base_out_features // output_channels

    @classmethod
    def from_linear(
        cls,
        base_linear: nn.Linear,
        grid_size: int,
        layer_spec: PDDLayerSpec,
    ) -> PDDOutputProjection:
        """Convert a loaded base linear without modifying it.

        Repeating a compatible conversion is idempotent. A PDD projection with
        conflicting grid, layout, path, or channel metadata is rejected.
        """
        if isinstance(base_linear, cls):
            if base_linear.grid_size != grid_size or base_linear.layer_spec != layer_spec:
                raise ValueError(
                    "existing PDDOutputProjection is incompatible with the requested "
                    f"grid/spec: existing=({base_linear.grid_size}, {base_linear.layer_spec}), "
                    f"requested=({grid_size}, {layer_spec})."
                )
            return base_linear
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"base_linear must be nn.Linear, got {type(base_linear).__name__}.")

        projection = cls(
            base_linear.in_features,
            base_linear.out_features,
            grid_size,
            layer_spec,
            bias=base_linear.bias is not None,
            device=base_linear.weight.device,
            dtype=base_linear.weight.dtype,
        )
        with torch.no_grad():
            projection.weight.copy_(projection._repeat_base_tensor(base_linear.weight))
            if projection.bias is not None and base_linear.bias is not None:
                projection.bias.copy_(projection._repeat_base_tensor(base_linear.bias))
        projection.weight.requires_grad_(base_linear.weight.requires_grad)
        if projection.bias is not None and base_linear.bias is not None:
            projection.bias.requires_grad_(base_linear.bias.requires_grad)
        projection.train(base_linear.training)
        return projection

    def _repeat_base_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Repeat a base weight or bias according to the configured head layout."""
        if tensor.shape[0] != self.base_out_features:
            raise ValueError(
                f"base tensor first dimension must be {self.base_out_features}, "
                f"got {tensor.shape[0]}."
            )
        trailing_shape = tensor.shape[1:]
        if self.layer_spec.head_layout == "channel_major":
            return (
                tensor.reshape(1, self.base_out_features, *trailing_shape)
                .expand(self.grid_size, self.base_out_features, *trailing_shape)
                .reshape(self.out_features, *trailing_shape)
                .clone()
            )

        output_channels = self.layer_spec.output_channels
        if output_channels is None:  # guarded by PDDLayerSpec validation
            raise RuntimeError("patch_major projection is missing output_channels.")
        return (
            tensor.reshape(self.patch_factor, output_channels, *trailing_shape)
            .unsqueeze(1)
            .expand(self.patch_factor, self.grid_size, output_channels, *trailing_shape)
            .reshape(self.out_features, *trailing_shape)
            .clone()
        )

    def _tensor_by_head(self, tensor: torch.Tensor) -> torch.Tensor:
        """View widened weight or bias as ``[head, base_output, ...]``."""
        trailing_shape = tensor.shape[1:]
        if self.layer_spec.head_layout == "channel_major":
            return tensor.reshape(self.grid_size, self.base_out_features, *trailing_shape)

        output_channels = self.layer_spec.output_channels
        if output_channels is None:  # guarded by PDDLayerSpec validation
            raise RuntimeError("patch_major projection is missing output_channels.")
        return (
            tensor.reshape(
                self.patch_factor,
                self.grid_size,
                output_channels,
                *trailing_shape,
            )
            .movedim(1, 0)
            .reshape(self.grid_size, self.base_out_features, *trailing_shape)
        )

    def _fused_parameters(
        self,
        start: int,
        end: int,
        grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute block-fused parameters for the half-open interval ``[start, end)``."""
        if isinstance(start, bool) or isinstance(end, bool):
            raise TypeError("start and end must be integers, not bool.")
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("start and end must be Python integers.")
        if grid.ndim != 1 or grid.shape[0] != self.grid_size + 1:
            raise ValueError(
                f"grid must contain {self.grid_size + 1} nodes, got shape {tuple(grid.shape)}."
            )
        coefficients = fusion_coefficients(grid, start, end).to(
            device=self.weight.device,
            dtype=torch.float32,
        )
        head_weights = self._tensor_by_head(self.weight)[start:end]
        fused_weight = torch.einsum("n,n...->...", coefficients, head_weights.float()).to(
            self.weight.dtype
        )
        if self.bias is None:
            return fused_weight, None
        head_bias = self._tensor_by_head(self.bias)[start:end]
        fused_bias = torch.einsum("n,n...->...", coefficients, head_bias.float()).to(
            self.bias.dtype
        )
        return fused_weight, fused_bias

    def forward(
        self,
        input: torch.Tensor,
        *,
        fusion: tuple[int, int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply the widened projection or an explicitly selected fused block."""
        if fusion is None:
            return F.linear(input, self.weight, self.bias)
        start, end, grid = fusion
        fused_weight, fused_bias = self._fused_parameters(start, end, grid)
        return F.linear(input, fused_weight, fused_bias)


def _get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    """Return an already registered nested module at ``path``."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be nn.Module, got {type(model).__name__}.")
    if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
        raise ValueError(f"path must be a non-empty dotted module path, got {path!r}.")
    try:
        return model.get_submodule(path)
    except AttributeError as error:
        raise ValueError(
            f"module path {path!r} does not resolve to a registered module."
        ) from error


def _replace_module_by_path(model: nn.Module, path: str, replacement: nn.Module) -> nn.Module:
    """Replace an existing nested module and return the previous module."""
    if not isinstance(replacement, nn.Module):
        raise TypeError(f"replacement must be nn.Module, got {type(replacement).__name__}.")
    previous = _get_module_by_path(model, path)
    parent_path, _, name = path.rpartition(".")
    parent = _get_module_by_path(model, parent_path) if parent_path else model
    setattr(parent, name, replacement)
    return previous


def convert_to_pdd_output_projection(
    model: nn.Module,
    layer_spec: PDDLayerSpec,
    grid_size: int,
) -> PDDOutputProjection:
    """Explicitly replace ``layer_spec.projection_path`` with a PDD projection."""
    current = _get_module_by_path(model, layer_spec.projection_path)
    if not isinstance(current, nn.Linear):
        raise TypeError(
            f"PDD projection at {layer_spec.projection_path!r} must be nn.Linear, "
            f"got {type(current).__name__}."
        )
    projection = PDDOutputProjection.from_linear(current, grid_size, layer_spec)
    if projection is not current:
        replaced = _replace_module_by_path(model, layer_spec.projection_path, projection)
        if replaced is not current:
            raise RuntimeError("projection changed during synchronous PDD conversion.")
    return projection


class PDDModelAdapter(Protocol):
    """Architecture adapter used by the framework-neutral PDD pipeline."""

    def student_all_heads(
        self,
        model: nn.Module,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Return canonical ``[batch, head, *latent_shape]`` student velocities."""
        ...

    def student_fused_block(
        self,
        model: nn.Module,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        start: int,
        end: int,
        grid: torch.Tensor,
        condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Return one base-shaped velocity from the fused projection block."""
        ...

    def teacher_velocity(
        self,
        model: nn.Module,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        negative_condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Return the adapter-specific guided teacher velocity."""
        ...


class PDDPipeline:
    """PDD losses and fused sampler over a single core-owned grid."""

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module | None,
        config: PDDConfig,
        adapter: PDDModelAdapter,
    ) -> None:
        """Store the models/config/adapter and freeze the optional training teacher."""
        if not isinstance(config, PDDConfig):
            raise TypeError(f"config must be PDDConfig, got {type(config).__name__}.")
        self.student = student
        self.teacher = None if teacher is None else teacher.eval().requires_grad_(False)
        self.config = config
        self.adapter = adapter

    def time_grid(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Construct this pipeline's sole shifted rectified-flow grid."""
        return make_shifted_flow_grid(
            self.config.grid_size,
            self.config.flow_shift,
            max_t=self.config.grid_max_t,
            device=device,
            dtype=torch.float32,
        )

    @staticmethod
    def _validate_state(state: torch.Tensor, *, name: str) -> None:
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(state).__name__}.")
        if state.ndim < 2 or state.shape[0] <= 0:
            raise ValueError(f"{name} must have shape [batch, *latent_shape], got {state.shape}.")
        if not state.dtype.is_floating_point:
            raise TypeError(f"{name} must use a real floating-point dtype, got {state.dtype}.")

    @staticmethod
    def _model_kwargs(model_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
        if model_kwargs is None:
            return {}
        if not isinstance(model_kwargs, Mapping):
            raise TypeError(
                f"model_kwargs must be a mapping or None, got {type(model_kwargs).__name__}."
            )
        return dict(model_kwargs)

    @staticmethod
    def _normalize_velocity(
        velocity: torch.Tensor,
        *,
        expected_shape: torch.Size,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if not isinstance(velocity, torch.Tensor):
            raise TypeError(f"{name} must return a tensor, got {type(velocity).__name__}.")
        if velocity.shape != expected_shape:
            raise ValueError(
                f"{name} must return shape {tuple(expected_shape)}, got {tuple(velocity.shape)}."
            )
        if velocity.device != device:
            raise ValueError(f"{name} returned device {velocity.device}, expected {device}.")
        if not velocity.dtype.is_floating_point:
            raise TypeError(
                f"{name} must return a real floating-point tensor, got {velocity.dtype}."
            )
        return velocity.to(torch.float32)

    @staticmethod
    def _validate_explicit_index(
        index: torch.Tensor,
        *,
        name: str,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(index, torch.Tensor):
            raise TypeError(f"{name} must be an integer tensor, got {type(index).__name__}.")
        if index.shape != (batch_size,):
            raise ValueError(f"{name} must have shape ({batch_size},), got {tuple(index.shape)}.")
        if index.device != device:
            raise ValueError(f"{name} must be on {device}, got {index.device}.")
        if index.dtype == torch.bool or index.dtype.is_floating_point or index.dtype.is_complex:
            raise TypeError(f"{name} must use an integer dtype, got {index.dtype}.")
        return index.to(torch.long)

    def _resolve_indices(
        self,
        *,
        batch_size: int,
        device: torch.device,
        n: torch.Tensor | None,
        k: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_size = self.config.grid_size
        block_min = self.config.block_size_min
        block_max = self.config.block_size_max
        if n is None and k is not None:
            raise ValueError(
                "explicit k requires explicit n so their joint support is deterministic."
            )
        if n is None:
            n = block_min * torch.randint(
                0,
                grid_size // block_min,
                (batch_size,),
                device=device,
                generator=generator,
            )
        else:
            n = self._validate_explicit_index(
                n,
                name="n",
                batch_size=batch_size,
                device=device,
            )
            torch._assert_async(
                torch.all((n >= 0) & (n < grid_size) & (n.remainder(block_min) == 0)),
                f"n must be aligned to {block_min} and satisfy 0 <= n < {grid_size}.",
            )

        upper = torch.minimum(n + block_max, torch.full_like(n, grid_size))
        if k is None:
            interval_ids = torch.arange(grid_size, device=device)
            support = (interval_ids[None] >= n[:, None]) & (interval_ids[None] < upper[:, None])
            k = torch.multinomial(support.to(torch.float32), 1, generator=generator).squeeze(1)
        else:
            k = self._validate_explicit_index(
                k,
                name="k",
                batch_size=batch_size,
                device=device,
            )
            torch._assert_async(
                torch.all((k >= n) & (k < upper)),
                "k must satisfy n <= k < min(n + block_size_max, grid_size).",
            )
        return n, k

    @staticmethod
    def _rms_per_sample(value: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, value.ndim))
        return value.square().mean(dim=dims).sqrt()

    def _compute_loss_from_state(
        self,
        state: torch.Tensor,
        *,
        grid: torch.Tensor,
        condition: Any = None,
        negative_condition: Any = None,
        model_kwargs: Mapping[str, Any] | None = None,
        n: torch.Tensor,
        k: torch.Tensor,
        collect_metrics: bool = True,
        advance_intervals: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
        """Compute the shared PDD target from an already constructed state ``x_n``."""
        if self.teacher is None:
            raise RuntimeError("PDD training requires a teacher model.")
        kwargs = self._model_kwargs(model_kwargs)
        state_fp32 = state.to(torch.float32)
        batch_size = state.shape[0]
        time_n = grid[n]
        broadcast_shape = (batch_size,) + (1,) * (state.ndim - 1)

        student_heads = self.adapter.student_all_heads(
            self.student,
            state_fp32,
            time_n,
            condition=condition,
            **kwargs,
        )
        expected_head_shape = torch.Size((batch_size, self.config.grid_size, *state.shape[1:]))
        student_heads = self._normalize_velocity(
            student_heads,
            expected_shape=expected_head_shape,
            device=state.device,
            name="student_all_heads",
        )
        with torch.no_grad():
            x_bar_k = integrate_interval_velocities(state_fp32, student_heads, grid, n, k)

        batch_ids = torch.arange(batch_size, device=state.device)
        student_target = student_heads[batch_ids, k]
        time_k = grid[k]
        with torch.no_grad():
            teacher_query = x_bar_k.detach()
            teacher_first = self.adapter.teacher_velocity(
                self.teacher,
                teacher_query,
                time_k,
                condition=condition,
                negative_condition=negative_condition,
                **kwargs,
            )
            teacher_first = self._normalize_velocity(
                teacher_first,
                expected_shape=state.shape,
                device=state.device,
                name="teacher_velocity",
            )
            if self.config.teacher_integrator == "euler":
                teacher_target = teacher_first
            else:
                delta_k = grid[k + 1] - time_k
                midpoint_state = (
                    teacher_query + 0.5 * delta_k.reshape(broadcast_shape) * teacher_first
                )
                midpoint_time = time_k + 0.5 * delta_k
                teacher_target = self.adapter.teacher_velocity(
                    self.teacher,
                    midpoint_state,
                    midpoint_time,
                    condition=condition,
                    negative_condition=negative_condition,
                    **kwargs,
                )
                teacher_target = self._normalize_velocity(
                    teacher_target,
                    expected_shape=state.shape,
                    device=state.device,
                    name="teacher_velocity",
                )

            next_state = None
            if advance_intervals is not None:
                next_n = n + advance_intervals
                torch._assert_async(
                    torch.all(next_n <= self.config.grid_size),
                    "data-free state advance must not pass the final PDD interval.",
                )
                next_state = integrate_interval_velocities(
                    state_fp32,
                    student_heads,
                    grid,
                    n,
                    next_n,
                ).detach()

        squared_error = (student_target - teacher_target).square()
        loss = squared_error.mean()
        metric_dims = tuple(range(1, squared_error.ndim))
        with torch.no_grad():
            metrics = {
                "student_target_mse": squared_error.mean(dim=metric_dims).detach(),
            }
            if collect_metrics:
                all_head_dims = tuple(range(1, student_heads.ndim))
                metrics.update(
                    n=n.detach(),
                    k=k.detach(),
                    target_span=(k - n + 1).detach(),
                    student_velocity_rms=self._rms_per_sample(student_target).detach(),
                    teacher_velocity_rms=self._rms_per_sample(teacher_target).detach(),
                    reconstructed_state_rms=self._rms_per_sample(x_bar_k).detach(),
                    all_student_heads_finite=torch.isfinite(student_heads)
                    .all(dim=all_head_dims)
                    .detach(),
                    student_target_finite=torch.isfinite(student_target)
                    .all(dim=metric_dims)
                    .detach(),
                    teacher_target_finite=torch.isfinite(teacher_target)
                    .all(dim=metric_dims)
                    .detach(),
                    reconstructed_state_finite=torch.isfinite(x_bar_k)
                    .all(dim=metric_dims)
                    .detach(),
                    loss_finite=torch.isfinite(loss).detach(),
                )
        return loss, metrics, next_state

    def compute_loss(
        self,
        data: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        condition: Any = None,
        negative_condition: Any = None,
        model_kwargs: Mapping[str, Any] | None = None,
        n: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        collect_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the exact data-dependent PDD objective for one batch."""
        if type(collect_metrics) is not bool:
            raise TypeError("collect_metrics must be a bool.")
        self._validate_state(data, name="data")
        if noise is None:
            noise_fp32 = torch.randn(
                data.shape,
                device=data.device,
                dtype=torch.float32,
                generator=generator,
            )
        else:
            self._validate_state(noise, name="noise")
            if noise.shape != data.shape:
                raise ValueError(
                    f"noise must match data shape {tuple(data.shape)}, got {tuple(noise.shape)}."
                )
            if noise.device != data.device:
                raise ValueError(f"noise must be on {data.device}, got {noise.device}.")
            noise_fp32 = noise.to(torch.float32)

        batch_size = data.shape[0]
        grid = self.time_grid(data.device)
        n, k = self._resolve_indices(
            batch_size=batch_size,
            device=data.device,
            n=n,
            k=k,
            generator=generator,
        )
        state = add_noise(data.to(torch.float32), noise_fp32, grid[n])
        loss, metrics, _ = self._compute_loss_from_state(
            state,
            grid=grid,
            condition=condition,
            negative_condition=negative_condition,
            model_kwargs=model_kwargs,
            n=n,
            k=k,
            collect_metrics=collect_metrics,
        )
        return loss, metrics

    def compute_data_free_loss(
        self,
        state: torch.Tensor,
        *,
        n: torch.Tensor,
        condition: Any = None,
        negative_condition: Any = None,
        model_kwargs: Mapping[str, Any] | None = None,
        k: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        collect_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Compute Algorithm 3 and return the detached state carried to ``n + L_min``."""
        if type(collect_metrics) is not bool:
            raise TypeError("collect_metrics must be a bool.")
        self._validate_state(state, name="state")
        grid = self.time_grid(state.device)
        n, k = self._resolve_indices(
            batch_size=state.shape[0],
            device=state.device,
            n=n,
            k=k,
            generator=generator,
        )
        loss, metrics, next_state = self._compute_loss_from_state(
            state,
            grid=grid,
            condition=condition,
            negative_condition=negative_condition,
            model_kwargs=model_kwargs,
            n=n,
            k=k,
            collect_metrics=collect_metrics,
            advance_intervals=self.config.block_size_min,
        )
        if next_state is None:  # guarded by advance_intervals above
            raise RuntimeError("data-free PDD did not produce a carried state.")
        return loss, metrics, next_state, (n + self.config.block_size_min).detach()

    def _validate_blocks(self, blocks: Sequence[int] | None) -> tuple[int, ...]:
        if blocks is None:
            resolved = tuple(self.config.inference_blocks)
        else:
            if isinstance(blocks, str | bytes) or not isinstance(blocks, Sequence):
                raise TypeError("blocks must be a sequence of integer interval counts.")
            resolved = tuple(blocks)
        if not resolved:
            raise ValueError("blocks must contain at least one interval count.")
        for index, block in enumerate(resolved):
            if type(block) is not int or block <= 0:
                raise ValueError(f"blocks[{index}] must be a positive integer, got {block!r}.")
        total = sum(resolved)
        if total != self.config.grid_size:
            raise ValueError(f"blocks must sum to grid_size={self.config.grid_size}, got {total}.")
        return resolved

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        *,
        condition: Any = None,
        blocks: Sequence[int] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """Sample from raw RF noise with one fused call per contiguous block."""
        self._validate_state(noise, name="noise")
        kwargs = self._model_kwargs(model_kwargs)
        resolved_blocks = self._validate_blocks(blocks)
        grid = self.time_grid(noise.device)
        # Derive fusion coefficients from the high-precision schedule, then cast
        # them to the FP32 decoding dtype used for state/time integration.
        fusion_grid = make_shifted_flow_grid(
            self.config.grid_size,
            self.config.flow_shift,
            max_t=self.config.grid_max_t,
            device=noise.device,
            dtype=torch.float64,
        )
        current = (noise.to(torch.float64) * self.config.grid_max_t).to(torch.float32)
        start = 0
        for block in resolved_blocks:
            end = start + block
            time = grid[start].expand(noise.shape[0])
            velocity = self.adapter.student_fused_block(
                self.student,
                current,
                time,
                start=start,
                end=end,
                grid=fusion_grid,
                condition=condition,
                **kwargs,
            )
            velocity = self._normalize_velocity(
                velocity,
                expected_shape=current.shape,
                device=noise.device,
                name="student_fused_block",
            )
            current = current + (grid[end] - grid[start]) * velocity
            start = end
        return current
