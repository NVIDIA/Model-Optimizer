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

"""Single source of truth for the NVFP4 FP8 scale-candidate set.

Pure PyTorch, no Triton dependency, so it can be imported from both the kernel
wrapper (which is triton-gated) and the reference Python sweep in the
:class:`NVFP4MSECalibrator` (which must work without triton too).
"""

import torch

_E4M3_MIN_POSITIVE_CODE = 1
_E4M3_MAX_FINITE_CODE = 126


def fp8_scale_candidates(
    device: torch.device | str = "cpu", fp8_max_for_normalization: float = 448.0
) -> torch.Tensor:
    """Return finite positive E4M3 values divided by the configured normalization max."""
    uint8_values = torch.arange(0, 128, dtype=torch.uint8, device=device)
    fp8_values = uint8_values.view(torch.float8_e4m3fn).float()
    valid_mask = torch.isfinite(fp8_values) & (fp8_values > 0)
    return fp8_values[valid_mask] / fp8_max_for_normalization


def fp8_scale_codes(
    amax: torch.Tensor,
    global_amax: torch.Tensor,
    fp8_max_for_normalization: float = 448.0,
) -> torch.Tensor:
    """Return each block's max-derived finite positive E4M3 scale code."""
    global_amax = global_amax.detach().to(device=amax.device, dtype=torch.float32)
    safe_global_amax = torch.where(global_amax > 0, global_amax, torch.ones_like(global_amax))
    normalized_scale = (
        amax.detach().to(dtype=torch.float32) * fp8_max_for_normalization / safe_global_amax
    )
    codes = normalized_scale.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int16)
    return codes.clamp(_E4M3_MIN_POSITIVE_CODE, _E4M3_MAX_FINITE_CODE).to(torch.uint8)


def fp8_scale_offset_candidates(
    amax: torch.Tensor,
    global_amax: torch.Tensor,
    offset_range: tuple[int, int],
    fp8_max_for_normalization: float = 448.0,
) -> torch.Tensor:
    """Return normalized per-block candidates for an inclusive E4M3 code-offset range."""
    min_offset, max_offset = offset_range
    base_codes = fp8_scale_codes(amax, global_amax, fp8_max_for_normalization).to(torch.int16)
    offsets = torch.arange(
        min_offset, max_offset + 1, device=amax.device, dtype=torch.int16
    ).reshape((-1,) + (1,) * amax.ndim)
    codes = (base_codes.unsqueeze(0) + offsets).clamp(
        _E4M3_MIN_POSITIVE_CODE, _E4M3_MAX_FINITE_CODE
    )
    return fp8_scale_candidates(amax.device, fp8_max_for_normalization)[codes.to(torch.long) - 1]
