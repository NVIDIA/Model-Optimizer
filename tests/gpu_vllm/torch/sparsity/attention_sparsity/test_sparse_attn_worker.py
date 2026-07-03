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

"""Tests for sparse attention vLLM worker compatibility helpers."""

import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
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


def test_nvfp4_bmm_mapping_and_unsupported_format_failure():
    """Enabled P/V quantizers map only dynamic block-16 NVFP4."""
    assert vllm_plugin._p_qdq_from_layer(SimpleNamespace()) == (None, 1.0)
    assert vllm_plugin._v_qdq_from_layer(SimpleNamespace()) == (None, None)

    good = SimpleNamespace(
        is_enabled=True,
        is_nvfp4_dynamic=True,
        block_sizes={-1: 16},
        _amax=torch.tensor(6.0 * 448.0),
    )
    layer = SimpleNamespace(p_bmm_quantizer=good, v_bmm_quantizer=good)
    assert vllm_plugin._p_qdq_from_layer(layer) == ("nvfp4", 6.0 * 448.0)
    assert vllm_plugin._v_qdq_from_layer(layer) == ("nvfp4", 6.0 * 448.0)

    good.block_sizes = {-1: 32}
    with pytest.raises(NotImplementedError, match="p_bmm_quantizer"):
        vllm_plugin._p_qdq_from_layer(layer)
    with pytest.raises(NotImplementedError, match="v_bmm_quantizer"):
        vllm_plugin._v_qdq_from_layer(layer)


def test_quantized_decode_finalizes_v_then_calls_split_k_kernel(monkeypatch):
    """Pure decode finalizes V before dispatching the valid query rows to split-K."""
    impl = _clone_sparse_impl(_make_old_impl())

    class UnreadableAmax:
        def numel(self):
            return 1

        def __float__(self):
            raise AssertionError("forward read live quantizer amax")

    quantizer = SimpleNamespace(
        is_enabled=True,
        is_nvfp4_dynamic=True,
        block_sizes={-1: 16},
        _amax=UnreadableAmax(),
    )
    q_inputs = []

    def quantize_q(query):
        assert query.dtype == torch.float32
        q_inputs.append(query.clone())
        return query + 1

    layer = SimpleNamespace(
        p_bmm_quantizer=quantizer,
        q_bmm_quantizer=quantize_q,
        v_bmm_quantizer=quantizer,
        _query_quant_in_kernel=True,
    )
    impl.quant_kw = {
        "p_qdq": "nvfp4",
        "p_qdq_amax": 1.0,
        "v_qdq": "nvfp4",
        "v_qdq_amax": 6.0 * 448.0,
    }
    q = torch.full((4, impl.num_heads, impl.head_size), 2.0, dtype=torch.float16)
    q[2:] = 10_000
    kv_cache = torch.zeros(2, 4, 16, impl.num_kv_heads, impl.head_size, dtype=torch.float16)
    metadata = SimpleNamespace(
        num_actual_tokens=q.shape[0],
        max_query_len=1,
        max_seq_len=34,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([16, 34], dtype=torch.int32),
        block_table=torch.zeros(2, 3, dtype=torch.int32),
    )
    calls = {}

    def fake_finalize(value_cache, block_table, v_lo, v_hi, **kwargs):
        calls["finalize"] = (v_lo.clone(), v_hi.clone(), kwargs)

    def fake_decode(query, key_cache, value_cache, block_table, seq_lens, **kwargs):
        calls["query"] = query.clone()
        calls["decode"] = (key_cache, value_cache, block_table, seq_lens, kwargs)
        return torch.ones_like(query)

    monkeypatch.setattr(vllm_plugin, "fake_quant_v_onwrite", fake_finalize)
    monkeypatch.setattr(vllm_plugin, "triton_decode_attention", fake_decode)
    monkeypatch.setattr(
        vllm_plugin,
        "triton_attention",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("shared kernel")),
    )
    monkeypatch.setattr(
        FlashAttentionImpl,
        "forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("native fallback")),
    )

    output = torch.empty_like(q)
    assert impl.forward(layer, q, q, q, kv_cache, metadata, output=output) is output
    v_lo, v_hi, finalizer_kw = calls["finalize"]
    torch.testing.assert_close(v_lo, torch.tensor([0, 32], dtype=torch.int32))
    torch.testing.assert_close(v_hi, torch.tensor([16, 32], dtype=torch.int32))
    assert finalizer_kw == {"page_size": 16, "v_qdq_scale": 1.0, "decode": True}
    key_cache, value_cache, block_table, seq_lens, decode_kw = calls["decode"]
    assert key_cache.data_ptr() == kv_cache[0].data_ptr()
    assert value_cache.data_ptr() == kv_cache[1].data_ptr()
    assert block_table is metadata.block_table
    assert seq_lens is metadata.seq_lens
    assert calls["query"].shape[0] == metadata.seq_lens.shape[0]
    assert decode_kw["p_qdq"] == "nvfp4"
    assert decode_kw["v_qdq"] == "nvfp4"
    assert decode_kw["v_cache_quantized"] is True
    assert len(q_inputs) == 1 and q_inputs[0].shape[0] == q.shape[0]
    torch.testing.assert_close(q_inputs[0][:2], q[:2].float())
    assert torch.all(q_inputs[0][2:] == 0)
    assert calls["query"].dtype == torch.float32


def test_quantized_skip_softmax_decode_stays_on_shared_kernel(monkeypatch):
    """Split-local maxima must not change calibrated skip-softmax semantics."""
    impl = _clone_sparse_impl(_make_old_impl())
    impl.quant_kw = {
        "p_qdq": "nvfp4",
        "p_qdq_amax": 1.0,
        "v_qdq": "nvfp4",
        "v_qdq_amax": 6.0 * 448.0,
    }
    impl.sparse_kw = {"skip_softmax_threshold": 0.001}
    q = torch.zeros(1, impl.num_heads, impl.head_size, dtype=torch.float16)
    kv_cache = torch.zeros(2, 1, 16, impl.num_kv_heads, impl.head_size, dtype=torch.float16)
    metadata = SimpleNamespace(
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=16,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([16], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
    )
    captured = {}

    monkeypatch.setattr(vllm_plugin, "fake_quant_v_onwrite", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        vllm_plugin,
        "triton_decode_attention",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("split-K kernel")),
    )

    def fake_attention(query, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(query)

    monkeypatch.setattr(vllm_plugin, "triton_attention", fake_attention)
    output = torch.empty_like(q)
    assert impl.forward(None, q, q, q, kv_cache, metadata, output=output) is output
    assert captured["skip_softmax_threshold"] == pytest.approx(0.001)
    assert captured["p_qdq"] == "nvfp4"
    assert captured["v_qdq"] == "nvfp4"


def test_active_cascade_fails_loud():
    """Cascade must not silently drop an active ModelOpt transform."""
    impl = _clone_sparse_impl(_make_old_impl())
    impl.sparse_kw = {"skip_softmax_threshold": 0.001}
    q = torch.zeros(1, impl.num_heads, impl.head_size, dtype=torch.float16)
    cache = torch.zeros(2, 1, 16, impl.num_kv_heads, impl.head_size, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="cascade"):
        impl.forward(None, q, q, q, cache, SimpleNamespace(use_cascade=True), output=q)


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
