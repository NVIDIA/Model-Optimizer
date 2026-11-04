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

"""Native hierarchical scale fields for GGML-compatible block formats."""

import torch

__all__ = []


def decode_scale_fields(packed, fmt):
    """Read only native scale fields; codebooks, index and delta bits stay fixed."""
    name = fmt.name
    blocks = packed.reshape(-1, fmt.block_bytes)
    if name == "iq1_m":
        words = blocks[:, 48:56:2].int() | (blocks[:, 49:56:2].int() << 8)
        bits = sum(((words[:, i] >> 12) & 15) << (4 * i) for i in range(4))
        global_scale = bits.to(torch.int16).contiguous().view(torch.float16).float()
        shifts = torch.tensor([0, 3, 6, 9], device=blocks.device)
        local = ((words.unsqueeze(-1) >> shifts) & 7).reshape(-1, 16)
    else:
        global_scale = blocks[:, :2].contiguous().view(torch.float16).reshape(-1).float()
        if name == "q8_0":
            local = None
        elif name == "iq1_s":
            local = (blocks[:, 35:50:2].int() >> 4) & 7
        elif name == "iq2_xxs":
            local = blocks[:, 9::8].int() >> 4
        else:
            start = 74 if name == "iq2_s" else 66
            values = blocks[:, start:].int()
            local = torch.stack((values & 15, values >> 4), dim=-1).reshape(-1, 16)
    return global_scale, local, 7 if name.startswith("iq1") else 15


@torch.no_grad()
def replace_scale_fields(packed, fmt, global_scale, local):
    """Replace legal IQ scale fields while preserving element and catalog bits."""
    packed = packed.clone()
    blocks = packed.reshape(-1, fmt.block_bytes)
    bits = global_scale.half().contiguous().view(torch.int16).int()
    if fmt.name == "iq1_m":
        words = torch.zeros((len(blocks), 4), dtype=torch.int32, device=blocks.device)
        for index in range(4):
            words[:, index] = ((bits >> (4 * index)) & 15) << 12
            for slot in range(4):
                words[:, index] |= local[:, index * 4 + slot].int() << (3 * slot)
        blocks[:, 48:56:2] = (words & 255).to(torch.uint8)
        blocks[:, 49:56:2] = (words >> 8).to(torch.uint8)
    else:
        blocks[:, :2] = global_scale.half().contiguous().view(torch.uint8).reshape(-1, 2)
        if fmt.name == "iq1_s":
            blocks[:, 35:50:2] = (blocks[:, 35:50:2] & 143) | (local.to(torch.uint8) << 4)
        elif fmt.name == "iq2_xxs":
            blocks[:, 9::8] = (blocks[:, 9::8] & 15) | (local.to(torch.uint8) << 4)
        elif local is not None:
            start = 74 if fmt.name == "iq2_s" else 66
            blocks[:, start:] = (local[:, 0::2].int() | (local[:, 1::2].int() << 4)).to(torch.uint8)
    return packed
