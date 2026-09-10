# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared utilities for the distillation trainer."""

from __future__ import annotations

import torch
import torch.distributed as dist

DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def to_dtype(name: str) -> torch.dtype:
    """Convert a dtype string to a ``torch.dtype``."""
    if name not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{name}'. Choose from: {list(DTYPE_MAP)}")
    return DTYPE_MAP[name]


def is_global_rank0() -> bool:
    """Return True on the global rank-0 process (or when not distributed)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def resolve_layer_pairs(
    modules: list[str | list[str]],
) -> list[tuple[str, str]]:
    """Normalize layer specs into ``(teacher_path, student_path)`` pairs.

    Plain strings are treated as shared paths (self-distillation shorthand).
    Two-element lists map ``[teacher_path, student_path]`` explicitly.
    """
    pairs: list[tuple[str, str]] = []
    for entry in modules:
        if isinstance(entry, str):
            pairs.append((entry, entry))
        else:
            pairs.append((entry[0], entry[1]))
    return pairs


def get_seq_length(
    latents: torch.Tensor, patch_size: int | tuple[int, ...] = 1
) -> int | None:
    """Patchified token count from a 5-D latent ``[B, C, F, H, W]``.

    Args:
        patch_size: int (uniform) or tuple ``(pt, ph, pw)`` per dimension.

    Returns None for non-5D tensors.
    """
    if latents.ndim != 5:
        return None
    _, _, f, h, w = latents.shape
    if isinstance(patch_size, int):
        pt = ph = pw = patch_size
    else:
        pt, ph, pw = patch_size
    return (f // pt) * (h // ph) * (w // pw)
