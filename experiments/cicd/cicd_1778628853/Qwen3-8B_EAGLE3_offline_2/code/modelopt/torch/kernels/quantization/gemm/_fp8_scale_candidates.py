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


def fp8_scale_candidates(device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the 126 valid finite positive FP8 E4M3 scale candidates / 448."""
    uint8_values = torch.arange(0, 128, dtype=torch.uint8, device=device)
    fp8_values = uint8_values.view(torch.float8_e4m3fn).float()
    valid_mask = torch.isfinite(fp8_values) & (fp8_values > 0)
    return fp8_values[valid_mask] / 448.0
