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

"""Framework-neutral output projection primitives for PDD.

This module owns only projection layout, conversion, reconstruction metadata,
and in-forward fusion. Model calls and architecture-specific packing belong to
adapters in ``modelopt.torch.fastgen.plugins``.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from ..config import PDDConfig
from ..flow_matching import fusion_coefficients

__all__ = [
    "PDDLayerSpec",
    "PDDMetadata",
    "PDDOutputProjection",
    "convert_to_pdd_output_projection",
    "get_module_by_path",
    "replace_module_by_path",
]

PDDHeadLayout = Literal["channel_major", "patch_major"]
_HEAD_LAYOUTS = ("channel_major", "patch_major")
_METADATA_SCHEMA_VERSION = 1


def _require_exact_keys(mapping: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if any(not isinstance(key, str) for key in mapping):
        raise ValueError(f"{name} keys must all be strings.")
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} keys mismatch: missing={missing}, extra={extra}.")


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a strict primitive mapping."""
        return {
            "projection_path": self.projection_path,
            "head_layout": self.head_layout,
            "output_channels": self.output_channels,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PDDLayerSpec:
        """Deserialize a strict primitive mapping."""
        if not isinstance(data, Mapping):
            raise TypeError(f"layer_spec must be a mapping, got {type(data).__name__}.")
        _require_exact_keys(
            data,
            {"projection_path", "head_layout", "output_channels"},
            name="layer_spec",
        )
        if not isinstance(data["projection_path"], str):
            raise ValueError("layer_spec.projection_path must be a string.")
        if not isinstance(data["head_layout"], str):
            raise ValueError("layer_spec.head_layout must be a string.")
        output_channels = data["output_channels"]
        if output_channels is not None and type(output_channels) is not int:
            raise ValueError("layer_spec.output_channels must be an integer or null.")
        return cls(
            projection_path=data["projection_path"],
            head_layout=data["head_layout"],
            output_channels=output_channels,
        )


@dataclass(frozen=True)
class PDDMetadata:
    """Versioned minimum metadata required to reconstruct a PDD projection."""

    grid_size: int
    flow_shift: float
    block_size_min: int
    block_size_max: int
    inference_blocks: tuple[int, ...]
    teacher_integrator: Literal["euler", "midpoint"]
    layer_spec: PDDLayerSpec
    projection_in_features: int
    projection_out_features: int
    projection_bias: bool
    schema_version: int = _METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_int(self.schema_version, name="schema_version")
        if self.schema_version != _METADATA_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported PDD metadata schema_version={self.schema_version}; "
                f"expected {_METADATA_SCHEMA_VERSION}."
            )
        _require_int(self.grid_size, name="grid_size")
        if type(self.flow_shift) is not float:
            raise ValueError(f"flow_shift must be a float, got {self.flow_shift!r}.")
        _require_int(self.block_size_min, name="block_size_min")
        _require_int(self.block_size_max, name="block_size_max")
        if not isinstance(self.inference_blocks, tuple) or any(
            type(block) is not int for block in self.inference_blocks
        ):
            raise ValueError("inference_blocks must be a tuple of integers.")
        if self.teacher_integrator not in ("euler", "midpoint"):
            raise ValueError(
                "teacher_integrator must be either 'euler' or 'midpoint', got "
                f"{self.teacher_integrator!r}."
            )
        _require_int(self.projection_in_features, name="projection_in_features")
        _require_int(self.projection_out_features, name="projection_out_features")
        if type(self.projection_bias) is not bool:
            raise ValueError(f"projection_bias must be bool, got {self.projection_bias!r}.")
        if not isinstance(self.layer_spec, PDDLayerSpec):
            raise TypeError(
                f"layer_spec must be PDDLayerSpec, got {type(self.layer_spec).__name__}."
            )
        if self.layer_spec.head_layout == "patch_major":
            output_channels = self.layer_spec.output_channels
            if output_channels is None or self.projection_out_features % output_channels != 0:
                raise ValueError(
                    f"projection_out_features={self.projection_out_features} must be divisible by "
                    f"output_channels={output_channels} for patch_major layout."
                )

        PDDConfig(
            grid_size=self.grid_size,
            flow_shift=self.flow_shift,
            block_size_min=self.block_size_min,
            block_size_max=self.block_size_max,
            inference_blocks=list(self.inference_blocks),
            student_sample_steps=len(self.inference_blocks),
            teacher_integrator=self.teacher_integrator,
        )

    @classmethod
    def from_config(cls, config: PDDConfig, projection: PDDOutputProjection) -> PDDMetadata:
        """Build reconstruction metadata from a validated config and projection."""
        if not isinstance(config, PDDConfig):
            raise TypeError(f"config must be PDDConfig, got {type(config).__name__}.")
        if not isinstance(projection, PDDOutputProjection):
            raise TypeError(
                f"projection must be PDDOutputProjection, got {type(projection).__name__}."
            )
        if config.grid_size != projection.grid_size:
            raise ValueError(
                f"config grid_size={config.grid_size} does not match projection "
                f"grid_size={projection.grid_size}."
            )
        return cls(
            grid_size=config.grid_size,
            flow_shift=config.flow_shift,
            block_size_min=config.block_size_min,
            block_size_max=config.block_size_max,
            inference_blocks=tuple(config.inference_blocks),
            teacher_integrator=config.teacher_integrator,
            layer_spec=projection.layer_spec,
            projection_in_features=projection.in_features,
            projection_out_features=projection.base_out_features,
            projection_bias=projection.bias is not None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a strict, JSON/YAML-safe mapping."""
        return {
            "schema_version": self.schema_version,
            "grid_size": self.grid_size,
            "flow_shift": self.flow_shift,
            "block_size_min": self.block_size_min,
            "block_size_max": self.block_size_max,
            "inference_blocks": list(self.inference_blocks),
            "teacher_integrator": self.teacher_integrator,
            "layer_spec": self.layer_spec.to_dict(),
            "base_projection": {
                "in_features": self.projection_in_features,
                "out_features": self.projection_out_features,
                "bias": self.projection_bias,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PDDMetadata:
        """Deserialize a strict, versioned metadata mapping."""
        if not isinstance(data, Mapping):
            raise TypeError(f"PDD metadata must be a mapping, got {type(data).__name__}.")
        _require_exact_keys(
            data,
            {
                "schema_version",
                "grid_size",
                "flow_shift",
                "block_size_min",
                "block_size_max",
                "inference_blocks",
                "teacher_integrator",
                "layer_spec",
                "base_projection",
            },
            name="PDD metadata",
        )
        schema_version = _require_int(data["schema_version"], name="schema_version")
        if type(data["flow_shift"]) is not float:
            raise ValueError(f"flow_shift must be a float, got {data['flow_shift']!r}.")
        if not isinstance(data["inference_blocks"], list) or any(
            type(block) is not int for block in data["inference_blocks"]
        ):
            raise ValueError("inference_blocks must be a list of integers.")
        if data["teacher_integrator"] not in ("euler", "midpoint"):
            raise ValueError(
                "teacher_integrator must be either 'euler' or 'midpoint', got "
                f"{data['teacher_integrator']!r}."
            )
        base_projection = data["base_projection"]
        if not isinstance(base_projection, Mapping):
            raise TypeError("base_projection must be a mapping.")
        _require_exact_keys(
            base_projection,
            {"in_features", "out_features", "bias"},
            name="base_projection",
        )
        if type(base_projection["bias"]) is not bool:
            raise ValueError("base_projection.bias must be bool.")

        return cls(
            schema_version=schema_version,
            grid_size=_require_int(data["grid_size"], name="grid_size"),
            flow_shift=data["flow_shift"],
            block_size_min=_require_int(data["block_size_min"], name="block_size_min"),
            block_size_max=_require_int(data["block_size_max"], name="block_size_max"),
            inference_blocks=tuple(data["inference_blocks"]),
            teacher_integrator=data["teacher_integrator"],
            layer_spec=PDDLayerSpec.from_dict(data["layer_spec"]),
            projection_in_features=_require_int(
                base_projection["in_features"], name="base_projection.in_features"
            ),
            projection_out_features=_require_int(
                base_projection["out_features"], name="base_projection.out_features"
            ),
            projection_bias=base_projection["bias"],
        )


@dataclass(frozen=True)
class _FusionRequest:
    start: int
    end: int
    grid: torch.Tensor


class PDDOutputProjection(nn.Linear):
    """A widened linear projection with one output head per PDD interval.

    Outside :meth:`fuse_block`, ``forward`` returns the full widened output.
    Inside the context, ``forward`` computes the selected block's weighted
    projection in float32 and returns the base-sized output. Fusion state is
    synchronous and thread-owned; nested contexts in one thread are supported,
    while concurrent access from another thread is rejected.
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
        self._fusion_stack: list[_FusionRequest] = []
        self._fusion_owner_thread: int | None = None
        self._fusion_lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        """Exclude the process-local lock while preserving ordinary module deepcopy."""
        with self._fusion_lock:
            if self._fusion_stack:
                raise RuntimeError("cannot copy or serialize an active PDD fusion context.")
            state = super().__getstate__()
            state.pop("_fusion_lock", None)
            return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore module state with a fresh process-local fusion lock."""
        super().__setstate__(state)
        self._fusion_lock = threading.Lock()

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

    @contextlib.contextmanager
    def fuse_block(self, start: int, end: int, grid: torch.Tensor) -> Iterator[PDDOutputProjection]:
        """Temporarily return the fused base-sized projection for ``[start, end)``.

        Contexts may nest synchronously in one thread. Because fusion selection is
        stored on the module, a second thread may neither enter a context nor call
        ``forward`` until the owning context exits.
        """
        if isinstance(start, bool) or isinstance(end, bool):
            raise TypeError("start and end must be integers, not bool.")
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("start and end must be Python integers.")
        if grid.ndim != 1 or grid.shape[0] != self.grid_size + 1:
            raise ValueError(
                f"grid must contain {self.grid_size + 1} nodes, got shape {tuple(grid.shape)}."
            )
        fusion_coefficients(grid, start, end)

        thread_id = threading.get_ident()
        request = _FusionRequest(start=start, end=end, grid=grid)
        with self._fusion_lock:
            if self._fusion_stack and self._fusion_owner_thread != thread_id:
                raise RuntimeError("PDD fusion context is already active in another thread.")
            if not self._fusion_stack:
                self._fusion_owner_thread = thread_id
            self._fusion_stack.append(request)
        try:
            yield self
        finally:
            with self._fusion_lock:
                if not self._fusion_stack or self._fusion_stack[-1] is not request:
                    raise RuntimeError("PDD fusion contexts exited out of order.")
                self._fusion_stack.pop()
                if not self._fusion_stack:
                    self._fusion_owner_thread = None

    def _fused_parameters(
        self, request: _FusionRequest
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute block-fused parameters from the registered widened parameter."""
        coefficients = fusion_coefficients(request.grid, request.start, request.end).to(
            device=self.weight.device,
            dtype=torch.float32,
        )
        head_weights = self._tensor_by_head(self.weight)[request.start : request.end]
        fused_weight = torch.einsum("n,n...->...", coefficients, head_weights.float()).to(
            self.weight.dtype
        )
        if self.bias is None:
            return fused_weight, None
        head_bias = self._tensor_by_head(self.bias)[request.start : request.end]
        fused_bias = torch.einsum("n,n...->...", coefficients, head_bias.float()).to(
            self.bias.dtype
        )
        return fused_weight, fused_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the widened or currently scoped fused projection."""
        with self._fusion_lock:
            if not self._fusion_stack:
                request = None
            else:
                if self._fusion_owner_thread != threading.get_ident():
                    raise RuntimeError("PDD fused forward was called from a non-owning thread.")
                request = self._fusion_stack[-1]
        if request is None:
            return F.linear(input, self.weight, self.bias)
        fused_weight, fused_bias = self._fused_parameters(request)
        return F.linear(input, fused_weight, fused_bias)


def get_module_by_path(model: nn.Module, path: str) -> nn.Module:
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


def replace_module_by_path(model: nn.Module, path: str, replacement: nn.Module) -> nn.Module:
    """Replace an existing nested module and return the previous module."""
    if not isinstance(replacement, nn.Module):
        raise TypeError(f"replacement must be nn.Module, got {type(replacement).__name__}.")
    previous = get_module_by_path(model, path)
    parent_path, _, name = path.rpartition(".")
    parent = get_module_by_path(model, parent_path) if parent_path else model
    setattr(parent, name, replacement)
    return previous


def convert_to_pdd_output_projection(
    model: nn.Module,
    layer_spec: PDDLayerSpec,
    grid_size: int,
) -> PDDOutputProjection:
    """Explicitly replace ``layer_spec.projection_path`` with a PDD projection."""
    current = get_module_by_path(model, layer_spec.projection_path)
    if not isinstance(current, nn.Linear):
        raise TypeError(
            f"PDD projection at {layer_spec.projection_path!r} must be nn.Linear, "
            f"got {type(current).__name__}."
        )
    projection = PDDOutputProjection.from_linear(current, grid_size, layer_spec)
    if projection is not current:
        replaced = replace_module_by_path(model, layer_spec.projection_path, projection)
        if replaced is not current:
            raise RuntimeError("projection changed during synchronous PDD conversion.")
    return projection
