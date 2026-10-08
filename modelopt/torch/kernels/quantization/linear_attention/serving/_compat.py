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

"""Keep native vLLM imports and state layout changes at the kernel boundary."""

from importlib import import_module
from importlib.util import find_spec

import vllm

__all__ = []

# vLLM 0.16 transposed the recurrent cache; ModelOpt keeps [H,K,V] for QDQ/autograd.
STATE_V_FIRST = vllm.__version_tuple__[:2] >= (0, 16)
_FLA_OPS = (
    "vllm.third_party.flash_linear_attention.ops"
    if find_spec("vllm.third_party.flash_linear_attention") is not None
    else "vllm.model_executor.layers.fla.ops"
)


def fla_module(name):
    return import_module(f"{_FLA_OPS}.{name}")


def state_layout(state):
    """Convert between ModelOpt and native layout; transposing is its own inverse."""
    return state.transpose(-1, -2).contiguous() if STATE_V_FIRST else state.contiguous()
