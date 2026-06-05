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


def test_non_persistent_buffers_are_reported_with_qualified_paths():
    """Only non-persistent buffers should appear, including nested names."""
    outer = nn.Module()
    outer.register_buffer("global_keep", torch.zeros(1), persistent=True)
    outer.register_buffer("scratch", torch.zeros(1), persistent=False)

    inner = nn.Module()
    inner.register_buffer("keep", torch.zeros(1), persistent=True)
    inner.register_buffer("rope_cache", torch.zeros(1), persistent=False)
    outer.add_module("attn", inner)

    out = _get_all_non_persistent_buffers_set(outer)
    assert out == {"scratch", "attn.rope_cache"}
