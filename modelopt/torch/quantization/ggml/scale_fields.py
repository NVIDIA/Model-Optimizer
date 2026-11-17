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

import torch


@dataclass(frozen=True)
class NativeScaleSpec:
    """Legal native field geometry, independent of optimizer and model family."""

    local_min: int = 0
    local_max: int = 0
    local_multiplier: int = 1
    local_bias: int = 0
    divisor: int = 1
    read_local: Callable | None = None
    write_local: Callable | None = None
    read_global: Callable | None = None
    write_global: Callable | None = None

    def read(self, blocks):
        """Read FP32 global scales and optional native integer local fields."""
        scale = (
            blocks[:, :2].contiguous().view(torch.float16).reshape(-1)
            if self.read_global is None
            else self.read_global(blocks)
        )
        return (
            scale.float(),
            None if self.read_local is None else self.read_local(blocks),
            self.local_max,
        )

    def replace(self, blocks, scale, local):
        """Replace scale bits in a copy, refusing nonrepresentable local fields."""
        if scale.numel() != len(blocks) or not torch.isfinite(scale.half()).all():
            raise ValueError("Global IQ scales must be finite FP16 values, one per block")
        if self.read_local is None:
            if local is not None:
                raise ValueError("This IQ format has no local scale field")
        elif (
            local is None
            or local.shape != self.read_local(blocks).shape
            or not torch.isfinite(local).all()
            or (local < self.local_min).any()
            or (local > self.local_max).any()
            or not torch.equal(local, local.round())
        ):
            raise ValueError("Local IQ scales must be legal native integer fields")
        result = blocks.clone()
        if self.write_global is None:
            result[:, :2] = scale.half().contiguous().view(torch.uint8).reshape(-1, 2)
        else:
            result = self.write_global(result, scale)
        if self.write_local is not None:
            self.write_local(result, local)
        return result


def decode_scale_fields(packed, fmt):
    """Read fields using the codec descriptor, without decoding the full weight."""
    if fmt.scale_spec is None:
        raise ValueError(f"{fmt.name} does not expose native IQ scale fields")
    return fmt.scale_spec.read(packed.reshape(-1, fmt.block_bytes))


@torch.no_grad()
def replace_scale_fields(packed, fmt, global_scale, local):
    """Return a native payload with only its legal scale fields replaced."""
    if fmt.scale_spec is None:
        raise ValueError(f"{fmt.name} does not expose native IQ scale fields")
    return fmt.scale_spec.replace(packed.reshape(-1, fmt.block_bytes), global_scale, local).reshape(
        packed.shape
    )
