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

"""Unit tests for sparse attention vLLM worker compatibility helpers."""

import pytest

pytest.importorskip("vllm")

from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    ModelOptSparseAttentionImpl,
    _clone_sparse_impl,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl


def _make_old_impl():
    """Create a vLLM FlashAttention impl with initialized runtime state."""
    return FlashAttentionImpl(
        num_heads=2,
        head_size=64,
        scale=0.125,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=128,
        kv_cache_dtype="auto",
    )


def test_clone_sparse_impl_preserves_runtime_state():
    """Clone helper should preserve vLLM's initialized impl state."""
    old_impl = _make_old_impl()
    old_impl.future_attr = object()

    new_impl = _clone_sparse_impl(old_impl)

    assert isinstance(new_impl, ModelOptSparseAttentionImpl)
    assert new_impl is not old_impl
    assert new_impl.sliding_window == old_impl.sliding_window
    assert new_impl.future_attr is old_impl.future_attr
    assert new_impl.__dict__.items() >= old_impl.__dict__.items()


def test_clone_sparse_impl_rejects_non_none_sinks():
    """vLLM attention sinks must fail fast until the sparse kernel supports them."""
    old_impl = _make_old_impl()
    old_impl.sinks = object()

    with pytest.raises(NotImplementedError, match="sinks"):
        _clone_sparse_impl(old_impl)
