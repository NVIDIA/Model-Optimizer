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

import math
from contextlib import nullcontext

import pytest
import torch

pytest.importorskip("vllm")

from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from modelopt.torch.sparsity.attention_sparsity.plugins import vllm as vllm_plugin
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    ModelOptSparseAttentionImpl,
    _build_sparse_kw,
    _clone_sparse_impl,
)


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


def test_forward_delegates_cascade_metadata_to_vllm(monkeypatch):
    """Cascade/prefix-cache metadata should use vLLM's native implementation."""
    impl = _clone_sparse_impl(_make_old_impl())
    q = torch.zeros(1, impl.num_heads, impl.head_size, dtype=torch.float16)
    kv_cache = torch.zeros(2, 1, 16, impl.num_kv_heads, impl.head_size, dtype=torch.float16)
    output = torch.empty_like(q)
    attn_metadata = type("AttnMetadata", (), {"use_cascade": True})()
    called = {}

    def fake_forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache_arg,
        attn_metadata_arg,
        output_arg=None,
        output_scale=None,
        output_block_scale=None,
    ):
        called["self"] = self
        called["kv_cache"] = kv_cache_arg
        called["attn_metadata"] = attn_metadata_arg
        output_arg.fill_(3)
        return output_arg

    monkeypatch.setattr(FlashAttentionImpl, "forward", fake_forward)

    result = impl.forward(
        layer=None,
        query=q,
        key=q,
        value=q,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
    )

    assert result is output
    assert called == {
        "self": impl,
        "kv_cache": kv_cache,
        "attn_metadata": attn_metadata,
    }
    assert torch.all(result == 3)


@pytest.mark.parametrize(
    ("sparse_kw", "max_query_len", "max_seq_len"),
    [
        ({"skip_softmax_threshold": 0.001}, 1, 128),
        (
            {
                "threshold_scale_factor": {
                    "formula": "a * exp(b * target_sparsity)",
                    "prefill": {"a": 10.0, "b": 0.0},
                },
                "target_sparse_ratio": {"prefill": 0.5},
            },
            4,
            4,
        ),
    ],
)
def test_forward_delegates_launches_without_effective_sparse_work(
    monkeypatch, sparse_kw, max_query_len, max_seq_len
):
    """When no validated sparse path is active, use vLLM FlashAttention."""
    impl = _clone_sparse_impl(_make_old_impl())
    impl.sparse_kw = sparse_kw
    q = torch.zeros(max_query_len, impl.num_heads, impl.head_size, dtype=torch.float16)
    kv_cache = torch.zeros(
        2, 1, max_seq_len, impl.num_kv_heads, impl.head_size, dtype=torch.float16
    )
    output = torch.empty_like(q)
    attn_metadata = type(
        "AttnMetadata",
        (),
        {
            "num_actual_tokens": max_query_len,
            "max_query_len": max_query_len,
            "max_seq_len": max_seq_len,
            "query_start_loc": torch.tensor([0, max_query_len], dtype=torch.int32),
            "seq_lens": torch.tensor([max_seq_len], dtype=torch.int32),
            "block_table": torch.zeros(1, 1, dtype=torch.int32),
        },
    )()
    called = {}

    def fake_attention(*args, **kwargs):
        raise AssertionError("ModelOpt Triton kernel should not be called")

    def fake_forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache_arg,
        attn_metadata_arg,
        output_arg=None,
        output_scale=None,
        output_block_scale=None,
    ):
        called["attn_metadata"] = attn_metadata_arg
        output_arg.fill_(5)
        return output_arg

    monkeypatch.setattr(vllm_plugin, "triton_attention", fake_attention)
    monkeypatch.setattr(FlashAttentionImpl, "forward", fake_forward)

    maybe_warns = (
        pytest.warns(UserWarning, match="outside the valid lambda range")
        if "threshold_scale_factor" in sparse_kw
        else nullcontext()
    )
    with maybe_warns:
        result = impl.forward(
            layer=None,
            query=q,
            key=q,
            value=q,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )

    assert called["attn_metadata"] is attn_metadata
    assert result is output
    assert torch.all(result == 5)


def test_forward_resolves_calibrated_skip_softmax_threshold(monkeypatch):
    """Forward should convert checkpoint calibration params to kernel threshold."""
    max_query_len = 128
    seq_len = 128
    expected_scale = 2.0 * math.exp(3.0 * 0.4)
    impl = _clone_sparse_impl(_make_old_impl())
    impl.sparse_kw = {
        "threshold_scale_factor": {
            "formula": "a * exp(b * target_sparsity)",
            "prefill": {"a": 2.0, "b": 3.0},
            "decode": {"a": 0.1, "b": 1.0},
        },
        "target_sparse_ratio": {"prefill": 0.4, "decode": 0.6},
    }
    q = torch.zeros(max_query_len, impl.num_heads, impl.head_size, dtype=torch.float16)
    kv_cache = torch.zeros(2, 1, seq_len, impl.num_kv_heads, impl.head_size, dtype=torch.float16)
    attn_metadata = type(
        "AttnMetadata",
        (),
        {
            "num_actual_tokens": max_query_len,
            "max_query_len": max_query_len,
            "max_seq_len": seq_len,
            "query_start_loc": torch.tensor([0, max_query_len], dtype=torch.int32),
            "seq_lens": torch.tensor([seq_len], dtype=torch.int32),
            "block_table": torch.zeros(1, 1, dtype=torch.int32),
        },
    )()
    captured = {}

    def fake_attention(q, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(q)

    monkeypatch.setattr(vllm_plugin, "triton_attention", fake_attention)

    impl.forward(
        layer=None,
        query=q,
        key=q,
        value=q,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=torch.empty_like(q),
    )

    assert captured["skip_softmax_threshold"] == pytest.approx(expected_scale / seq_len)
    assert "threshold_scale_factor" not in captured
    assert "target_sparse_ratio" not in captured


def test_resolve_calibrated_skip_softmax_threshold_for_decode():
    """Calibration conversion is phase-aware even when decode later delegates."""
    sparse_kw = {
        "threshold_scale_factor": {
            "formula": "a * exp(b * target_sparsity)",
            "decode": {"a": 0.1, "b": 1.0},
        },
        "target_sparse_ratio": {"decode": 0.6},
    }

    vllm_plugin._resolve_skip_softmax_calibration(
        sparse_kw,
        is_prefill=False,
        max_seq_len=256,
    )

    assert sparse_kw == {"skip_softmax_threshold": pytest.approx(0.1 * math.exp(1.0 * 0.6) / 256)}


def test_resolve_calibrated_skip_softmax_warns_and_disables_for_large_threshold():
    """A derived lambda >= 1 is invalid and disables calibrated skip-softmax."""
    sparse_kw = {
        "threshold_scale_factor": {
            "formula": "a * exp(b * target_sparsity)",
            "decode": {"a": 925.492, "b": 0.0},
        },
        "target_sparse_ratio": {"decode": 0.5},
    }

    with pytest.warns(UserWarning, match="outside the valid lambda range"):
        vllm_plugin._resolve_skip_softmax_calibration(
            sparse_kw,
            is_prefill=False,
            max_seq_len=256,
        )

    assert "skip_softmax_threshold" not in sparse_kw
    assert "threshold_scale_factor" not in sparse_kw
    assert "target_sparse_ratio" not in sparse_kw


def test_build_sparse_kw_restores_checkpoint_sparse_metadata():
    """Checkpoint metadata is converted into ModelOpt Triton kwargs."""
    layer_cfg = {
        "sparsity_n": 2,
        "sparsity_m": 4,
        "dense_sink_tokens": 3,
        "dense_recent_tokens": 64,
        "threshold_scale_factor": {"prefill": {"a": 1.0, "b": 2.0}},
        "target_sparse_ratio": {"prefill": 0.5},
    }

    assert _build_sparse_kw(layer_cfg) == {
        "sparsity_n": 2,
        "sparsity_m": 4,
        "dense_sink_tokens": 3,
        "dense_recent_tokens": 64,
        "threshold_scale_factor": {"prefill": {"a": 1.0, "b": 2.0}},
        "target_sparse_ratio": {"prefill": 0.5},
    }


def test_forward_delegates_sparse_nm_only_decode_to_vllm(monkeypatch):
    """N:M sparse softmax is prefill-only, so N:M-only decode uses vLLM."""
    impl = _clone_sparse_impl(_make_old_impl())
    impl.sparse_kw = {
        "sparsity_n": 2,
        "sparsity_m": 4,
        "dense_sink_tokens": 4,
        "dense_recent_tokens": 128,
    }
    q = torch.zeros(1, impl.num_heads, impl.head_size, dtype=torch.float16)
    kv_cache = torch.zeros(2, 1, 16, impl.num_kv_heads, impl.head_size, dtype=torch.float16)
    attn_metadata = type(
        "AttnMetadata",
        (),
        {
            "num_actual_tokens": 1,
            "max_query_len": 1,
            "max_seq_len": 16,
            "query_start_loc": torch.tensor([0, 1], dtype=torch.int32),
            "seq_lens": torch.tensor([16], dtype=torch.int32),
            "block_table": torch.zeros(1, 1, dtype=torch.int32),
        },
    )()

    def fake_attention(q, **kwargs):
        raise AssertionError("N:M-only decode should not call ModelOpt Triton")

    def fake_forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache_arg,
        attn_metadata_arg,
        output_arg=None,
        output_scale=None,
        output_block_scale=None,
    ):
        output_arg.fill_(7)
        return output_arg

    monkeypatch.setattr(vllm_plugin, "triton_attention", fake_attention)
    monkeypatch.setattr(FlashAttentionImpl, "forward", fake_forward)

    output = torch.empty_like(q)
    result = impl.forward(
        layer=None,
        query=q,
        key=q,
        value=q,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
    )

    assert result is output
    assert torch.all(result == 7)


def test_forward_allows_chunked_prefill_metadata(monkeypatch):
    """vLLM V1 can pass suffix-Q/chunked-prefill metadata; the kernel handles it."""
    impl = _clone_sparse_impl(_make_old_impl())
    impl.sparse_kw = {"sparsity_n": 2, "sparsity_m": 4}
    q_len = 4
    kv_len = 10
    q = torch.zeros(q_len, impl.num_heads, impl.head_size, dtype=torch.float16)
    kv_cache = torch.zeros(2, 1, 16, impl.num_kv_heads, impl.head_size, dtype=torch.float16)
    attn_metadata = type(
        "AttnMetadata",
        (),
        {
            "num_actual_tokens": q_len,
            "max_query_len": q_len,
            "max_seq_len": kv_len,
            "query_start_loc": torch.tensor([0, q_len], dtype=torch.int32),
            "seq_lens": torch.tensor([kv_len], dtype=torch.int32),
            "block_table": torch.zeros(1, 1, dtype=torch.int32),
        },
    )()
    captured = {}

    def fake_attention(q, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(q)

    monkeypatch.setattr(vllm_plugin, "triton_attention", fake_attention)

    impl.forward(
        layer=None,
        query=q,
        key=q,
        value=q,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=torch.empty_like(q),
    )

    assert captured["is_causal"] is True
    torch.testing.assert_close(captured["b_seq_len"], torch.tensor([q_len], dtype=torch.int32))
    torch.testing.assert_close(captured["b_seq_len_k"], torch.tensor([kv_len], dtype=torch.int32))
