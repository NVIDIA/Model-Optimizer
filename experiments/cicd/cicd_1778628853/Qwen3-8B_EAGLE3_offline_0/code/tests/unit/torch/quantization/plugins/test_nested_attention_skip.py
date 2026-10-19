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

"""Tests for the per-instance nested-attention skip in the HF quantization plugin."""

import pytest
import torch.nn as nn

pytest.importorskip("transformers")

from modelopt.torch.quantization.plugins.huggingface import _wraps_nested_attention


def _attn(name, child=None):
    """Build a module whose class name ends with ``Attention`` and optionally wraps ``child``."""
    cls = type(name, (nn.Module,), {"__init__": lambda self: nn.Module.__init__(self)})
    m = cls()
    if child is not None:
        m.inner = child
    return m


def test_wraps_nested_attention_flags_only_wrappers_per_instance():
    """Leaf attention is not a wrapper; wrappers (any level) are; same class reused is
    checked per-instance."""
    leaf = _attn("SelfAttention")
    wrapper = _attn("ViTAttention", child=_attn("ViTSelfAttention"))
    outer = _attn("OuterAttention", child=wrapper)
    reused_wrapper = _attn("ReusedAttention", child=_attn("ReusedAttention"))

    assert not _wraps_nested_attention(leaf)
    assert _wraps_nested_attention(wrapper)
    assert _wraps_nested_attention(outer) and _wraps_nested_attention(outer.inner)
    assert _wraps_nested_attention(reused_wrapper)
    assert not _wraps_nested_attention(reused_wrapper.inner)
