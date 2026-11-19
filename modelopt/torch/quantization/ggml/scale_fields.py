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

"""Codec-owned native IQ scale fields. Code and additive bits are immutable."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .common import GGMLFormat

__all__ = ["NativeScaleSpec", "decode_scale_fields", "replace_scale_fields"]


@dataclass(frozen=True)
class NativeScaleSpec:
    """Legal native field geometry, independent of optimizer and model family."""

    local_min: int = 0
    local_max: int = 0
    local_multiplier: int = 1
    local_bias: int = 0
    divisor: int = 1
    read_local: Callable[[torch.Tensor], torch.Tensor] | None = None
    write_local: Callable[[torch.Tensor, torch.Tensor], None] | None = None

    def __post_init__(self) -> None:
        if (self.read_local is None) != (self.write_local is None):
            raise ValueError("Native local scale readers and writers must be paired")

    def read(
        self, blocks: torch.Tensor, read_global: Callable[[torch.Tensor], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Read FP32 global scales and optional native integer local fields."""
        return (
            read_global(blocks).float(),
            None if self.read_local is None else self.read_local(blocks),
            self.local_max,
        )

    def replace(
        self,
        blocks: torch.Tensor,
        scale: torch.Tensor,
        local: torch.Tensor | None,
        write_global: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Replace scale bits in a copy, refusing nonrepresentable local fields."""
        if (
            scale.is_complex()
            or scale.numel() != len(blocks)
            or not torch.isfinite(scale.half()).all()
        ):
            raise ValueError("Global IQ scales must be finite FP16 values, one per block")
        if self.read_local is None:
            if local is not None:
                raise ValueError("This IQ format has no local scale field")
        elif (
            local is None
            or local.shape != self.read_local(blocks).shape
            or not (
                torch.isfinite(local)
                & (local >= self.local_min)
                & (local <= self.local_max)
                & (local == local.round())
            ).all()
        ):
            raise ValueError("Local IQ scales must be legal native integer fields")
        result = write_global(blocks, scale.reshape(-1))
        if self.write_local is not None:
            self.write_local(result, local)
        return result


def _native_scale_spec(fmt: "GGMLFormat") -> NativeScaleSpec:
    if fmt.scale_spec is None or fmt.payload_scales is None or fmt.replace_payload_scales is None:
        raise ValueError(f"{fmt.name} does not expose native IQ scale fields")
    return fmt.scale_spec


def _scale_blocks(packed: torch.Tensor, fmt: "GGMLFormat") -> torch.Tensor:
    if packed.dim() == 0 or packed.dtype != torch.uint8 or packed.shape[-1] != fmt.block_bytes:
        raise ValueError(
            f"packed must be uint8 with last dimension {fmt.block_bytes}, "
            f"got {packed.dtype} {tuple(packed.shape)}"
        )
    return packed.reshape(-1, fmt.block_bytes)


def decode_scale_fields(
    packed: torch.Tensor, fmt: "GGMLFormat"
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Read fields using the codec descriptor, without decoding the full weight."""
    return _native_scale_spec(fmt).read(
        _scale_blocks(packed, fmt),
        fmt.payload_scales,  # type: ignore[arg-type]  # Validated by _native_scale_spec.
    )


@torch.no_grad()
def replace_scale_fields(
    packed: torch.Tensor,
    fmt: "GGMLFormat",
    global_scale: torch.Tensor,
    local: torch.Tensor | None,
) -> torch.Tensor:
    """Return a native payload with only its legal scale fields replaced."""
    return (
        _native_scale_spec(fmt)
        .replace(
            _scale_blocks(packed, fmt),
            global_scale,
            local,
            fmt.replace_payload_scales,  # type: ignore[arg-type]  # Validated by _native_scale_spec.
        )
        .reshape(packed.shape)
    )
