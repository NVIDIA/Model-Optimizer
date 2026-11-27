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

"""Enable FlashAttention 2 for Qwen3.6 on GB300 in NeMo 26.08.00.

Transformer Engine 2.17.1 excludes GB300 (SM103) from FlashAttention 2 for
head dimensions above 192. Qwen3.6 uses 256, and the unfused fallback ran
out of GPU memory. This patch lets Transformer Engine select FlashAttention 2
to reduce memory use.

The GB300 QAD retest on NeMo 26.10 RC8 used automatic attention selection.
Transformer Engine selected native FusedAttention and ran without this patch.
"""

import torch
import transformer_engine

__all__ = ["enable_sm103_flash"]


def enable_sm103_flash() -> None:
    """Allow GB300 to use FlashAttention 2 at head dimension 256 on the original stack."""
    sm103 = (10, 3)
    if transformer_engine.__version__ != "2.17.1+4329ff84":
        return
    if torch.cuda.get_device_capability() != sm103:
        return

    # Only the pinned Transformer Engine build needs this private selector.
    from transformer_engine.pytorch.attention.dot_product_attention import utils

    # Transformer Engine uses this capability list to select FlashAttention 2 for
    # head dimensions above 192. Add SM103 to the list in memory. The checks skip
    # an already patched selector and catch changes to the expected list.
    selector = utils.get_attention_backend
    code = selector.__code__
    eligible = ((8, 0), (9, 0), (10, 0), (12, 0))
    patched = (*eligible[:-1], sm103, eligible[-1])
    if patched in code.co_consts:
        return
    if code.co_consts.count(eligible) != 1:
        raise RuntimeError(
            "Cannot apply the GB300 FlashAttention 2 patch. "
            "Expected exactly one copy of the GPU capability list in the Transformer Engine selector."
        )
    selector.__code__ = code.replace(
        co_consts=tuple(patched if value == eligible else value for value in code.co_consts)
    )
