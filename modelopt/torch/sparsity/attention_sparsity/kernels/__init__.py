# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Triton attention kernels for sparse attention optimization."""

import torch

from modelopt.torch.utils import import_plugin

IS_AVAILABLE = False
context_attention_fwd = None
context_attention = None
register_triton_attention = None
register_diffusers_triton_attention = None
register_ltx_triton_attention = None
register_ltx_head_cache_attention = None
register_diffusers_head_cache_attention = None
set_sparse24 = None

if torch.cuda.is_available():
    with import_plugin(
        "triton",
        msg_if_missing=(
            "Your device is potentially capable of using the triton attention "
            "kernel. Try to install triton with `pip install triton`."
        ),
    ):
        from .triton_unified_attention import context_attention as _context_attention
        from .triton_unified_attention import context_attention_fwd as _context_attention_fwd

        context_attention_fwd = _context_attention_fwd
        context_attention = _context_attention
        IS_AVAILABLE = True
        with import_plugin("transformers"):
            from .hf_triton_attention import register_triton_attention as _register_triton_attention
            from .hf_triton_attention import set_sparse24 as _set_sparse24

            register_triton_attention = _register_triton_attention
            set_sparse24 = _set_sparse24
            _register_triton_attention()

        with import_plugin("diffusers"):
            from .diffusers_triton_attention import (
                register_diffusers_triton_attention as _register_diffusers_triton_attention,
            )

            register_diffusers_triton_attention = _register_diffusers_triton_attention

        with import_plugin("ltx_core"):
            from .ltx_triton_attention import (
                register_ltx_triton_attention as _register_ltx_triton_attention,
            )

            register_ltx_triton_attention = _register_ltx_triton_attention

# Head cache kernels (no Triton dependency, just need ltx_core or diffusers)
with import_plugin("ltx_core"):
    from .ltx_head_cache_attention import (
        register_ltx_head_cache_attention as _register_ltx_head_cache,
    )

    register_ltx_head_cache_attention = _register_ltx_head_cache

with import_plugin("diffusers"):
    from .diffusers_head_cache_attention import (
        register_diffusers_head_cache_attention as _register_diffusers_head_cache,
    )

    register_diffusers_head_cache_attention = _register_diffusers_head_cache

__all__ = [
    "IS_AVAILABLE",
    "context_attention",
    "context_attention_fwd",
    "register_diffusers_head_cache_attention",
    "register_diffusers_triton_attention",
    "register_ltx_head_cache_attention",
    "register_ltx_triton_attention",
    "register_triton_attention",
    "set_sparse24",
]
