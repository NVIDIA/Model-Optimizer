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

"""Unit tests for ``_get_all_non_persistent_buffers_set``.

This helper is what ``bypass_factory_fn`` uses to decide which buffers belong
to ``owned_buffers`` (and therefore get checkpointed) versus which are
recomputed on every forward (RoPE caches, attention masks, etc.). A regression
that drops the module-name prefix would cause the post-resume model to silently
load buffers under wrong names.
"""

import torch
import torch.nn as nn

from modelopt.torch.puzzletron.bypass_distillation.stitched_model_factory import (
    _get_all_non_persistent_buffers_set,
)


def test_persistent_buffer_excluded_non_persistent_included():
    m = nn.Module()
    m.register_buffer("p", torch.zeros(1), persistent=True)
    m.register_buffer("np", torch.zeros(1), persistent=False)
    out = _get_all_non_persistent_buffers_set(m)
    assert out == {"np"}


def test_nested_submodule_paths_are_fully_qualified():
    """Sub-module non-persistent buffers must surface as ``submodule_name.buffer_name``
    so the matching key in ``state_dict()`` and the bypass save/restore code agree."""
    outer = nn.Module()
    inner = nn.Module()
    inner.register_buffer("nb", torch.zeros(1), persistent=False)
    outer.add_module("inner", inner)
    out = _get_all_non_persistent_buffers_set(outer)
    assert out == {"inner.nb"}


def test_mix_of_persistent_and_non_persistent_in_nested_module():
    """The full discrimination: only the nested non-persistent buffer should
    appear, with its fully-qualified path."""
    outer = nn.Module()
    inner = nn.Module()
    inner.register_buffer("keep", torch.zeros(1), persistent=True)  # persistent → excluded
    inner.register_buffer("rope_cache", torch.zeros(1), persistent=False)
    outer.add_module("attn", inner)
    outer.register_buffer("global_keep", torch.zeros(1), persistent=True)  # → excluded
    out = _get_all_non_persistent_buffers_set(outer)
    assert out == {"attn.rope_cache"}
