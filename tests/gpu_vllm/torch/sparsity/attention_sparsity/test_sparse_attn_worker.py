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

import builtins
import importlib.util
import math
from contextlib import nullcontext
from itertools import accumulate
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import vllm
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends import flashinfer as flashinfer_backend
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadata,
    FlashInferMetadataBuilder,
)
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from modelopt.torch.sparsity.attention_sparsity.plugins import vllm as vllm_plugin
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    ModelOptSparseAttentionImpl,
    _build_sparse_kw,
    _clone_sparse_impl,
    get_flashinfer_sparse_impl_cls,
    patch_flashinfer_metadata_builder,
    select_sparse_impl_cls,
)

_WORKER_PATH = Path(__file__).parents[5] / "examples/vllm_serve/sparse_attn_worker.py"


def _load_worker_module(name="sparse_attn_worker_test"):
    spec = importlib.util.spec_from_file_location(name, _WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_worker_import_does_not_resolve_quant_only_apis(monkeypatch):
    forbidden = {
        "vllm.config.compilation",
        "vllm.v1.attention.backend",
        "modelopt.torch.quantization.conversion",
        "modelopt.torch.quantization.nn",
        "modelopt.torch.quantization.plugins.vllm",
    }
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        fromlist = kwargs.get("fromlist", args[2] if len(args) > 2 else ())
        requested = {name, *(f"{name}.{item}" for item in fromlist or ())}
        if blocked := requested & forbidden:
            raise AssertionError(
                f"quant-only import during sparse module load: {sorted(blocked)[0]}"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(vllm, "__version__", "0.9.0")
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    worker_module = _load_worker_module("sparse_attn_worker_import_test")

    assert worker_module.__all__ == ["SparseAttnWorker", "QuantSparseAttnWorker"]


@pytest.mark.parametrize(
    ("class_name", "installer_name", "expected_kwargs"),
    [
        ("SparseAttnWorker", "install_vllm_sparse_attention_from_checkpoint", {}),
        (
            "QuantSparseAttnWorker",
            "install_vllm_nvfp4_attention",
            {"sparse_cfg": "checkpoint"},
        ),
    ],
)
def test_public_workers_install_after_base_load(
    monkeypatch, capsys, class_name, installer_name, expected_kwargs
):
    worker_module = _load_worker_module(f"worker_order_{class_name}")
    events = []
    model_runner = object()

    def fake_base_load(worker, *_args, **_kwargs):
        events.append("base")
        worker.model_runner = model_runner
        worker.vllm_config = SimpleNamespace(additional_config=None)

    def fake_install(actual_runner, **kwargs):
        events.append(("install", actual_runner, kwargs))
        return SimpleNamespace(
            installed_count=0 if class_name == "SparseAttnWorker" else 1,
            backend_counts={"TestImpl": 1},
            sparse_algorithm=None,
        )

    monkeypatch.setattr(BaseWorker, "load_model", fake_base_load)
    monkeypatch.setattr(worker_module, installer_name, fake_install)

    instance = object.__new__(getattr(worker_module, class_name))
    assert instance.load_model() is None
    assert events == ["base", ("install", model_runner, expected_kwargs)]
    output = capsys.readouterr().out
    if class_name == "SparseAttnWorker":
        assert "No sparse_attention_config found" in output
        assert "hf_sa.py" in output
    else:
        assert "Installed NVFP4 attention (quant+sparse) on 1 layers: {'TestImpl': 1}" in output


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


def _make_old_flashinfer_impl():
    """Create a bare FlashInfer impl without requiring a live vLLM config."""
    impl = object.__new__(FlashInferImpl)
    impl.__dict__.update(
        num_heads=2,
        num_kv_heads=2,
        head_size=64,
        scale=0.125,
        sinks=None,
        alibi_slopes=None,
        logits_soft_cap=None,
        kv_cache_dtype="auto",
    )
    return impl


def _make_flashinfer_impl(*, sparse=False, quantized=False):
    impl = _clone_sparse_impl(_make_old_flashinfer_impl(), get_flashinfer_sparse_impl_cls())
    impl.sparse_kw = {"sparsity_n": 2, "sparsity_m": 4} if sparse else {}
    impl.quant_kw = {
        "p_qdq": "nvfp4" if quantized else None,
        "p_qdq_amax": 1.0,
        "v_qdq": "nvfp4" if quantized else None,
        "v_qdq_amax": 6.0 * 448.0 if quantized else None,
    }
    return impl


@pytest.fixture
def isolated_flashinfer_builder_patch():
    saved_build = FlashInferMetadataBuilder.build
    vllm_plugin._reset_flashinfer_state_for_tests()
    try:
        yield
    finally:
        FlashInferMetadataBuilder.build = saved_build
        vllm_plugin._reset_flashinfer_state_for_tests()


def test_flashinfer_metadata_builder_patch_stashes_common_metadata(
    monkeypatch, isolated_flashinfer_builder_patch
):
    """The real builder result must retain the common metadata contract."""
    monkeypatch.setattr(flashinfer_backend, "use_trtllm_attention", lambda *_args, **_kwargs: True)
    assert patch_flashinfer_metadata_builder() is True
    builder = object.__new__(FlashInferMetadataBuilder)
    builder.__dict__.update(
        reorder_batch_threshold=1,
        page_size=16,
        num_qo_heads=2,
        num_kv_heads=2,
        dcp_world_size=1,
        cache_dtype=torch.float16,
        q_data_type=torch.float16,
        attention_config=SimpleNamespace(use_trtllm_attention=True),
        has_sinks=False,
        use_trtllm_decode_attention=True,
        use_dcp=False,
    )
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([16], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=16,
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        causal=False,
    )

    metadata = builder.build(0, common, fast_build=False)

    assert isinstance(metadata, FlashInferMetadata)
    for target, source in vllm_plugin._FLASHINFER_METADATA_FIELDS.items():
        actual = getattr(metadata, target)
        expected = getattr(common, source)
        if isinstance(expected, torch.Tensor):
            assert actual is expected
        else:
            assert actual == expected


def test_select_and_clone_flashinfer_impl_preserves_runtime_state(
    isolated_flashinfer_builder_patch,
):
    old_impl = _make_old_flashinfer_impl()

    new_cls = select_sparse_impl_cls(old_impl)
    new_impl = _clone_sparse_impl(old_impl, new_cls)

    assert new_cls is get_flashinfer_sparse_impl_cls()
    assert isinstance(new_impl, FlashInferImpl)
    assert type(new_impl).__name__ == "ModelOptSparseFlashInferImpl"
    if hasattr(FlashInferImpl, "do_kv_cache_update"):
        assert type(new_impl).do_kv_cache_update is FlashInferImpl.do_kv_cache_update
    assert select_sparse_impl_cls(new_impl) is None


def _flashinfer_metadata(*, query_lens=(1, 1), seq_lens=(16, 34), max_query_len=None, mixed=False):
    query_start_loc = list(accumulate(query_lens, initial=0))
    max_query_len = max_query_len or max(query_lens)
    metadata = SimpleNamespace(
        use_cascade=False,
        _modelopt_block_table=torch.zeros(len(seq_lens), 3, dtype=torch.int32),
        _modelopt_seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        _modelopt_query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        _modelopt_num_actual_tokens=sum(query_lens),
        _modelopt_max_query_len=max_query_len,
        _modelopt_max_seq_len=max(seq_lens),
        _modelopt_causal=max_query_len > 1,
        slot_mapping=torch.arange(sum(query_lens), dtype=torch.int64),
    )
    if mixed:
        metadata.num_decodes = 1
        metadata.num_prefills = len(query_lens) - 1
        metadata.num_decode_tokens = query_lens[0]
        metadata.num_prefill_tokens = sum(query_lens[1:])
    return metadata


def _flash_attention_metadata(q_len, kv_len):
    return SimpleNamespace(
        num_actual_tokens=q_len,
        max_query_len=q_len,
        max_seq_len=kv_len,
        query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
        seq_lens=torch.tensor([kv_len], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
    )


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_flashinfer_quantized_decode_preserves_cache_layout(monkeypatch, layout):
    """Both FI layouts expose one logical cache shape with different strides."""
    impl = _make_flashinfer_impl(quantized=True)
    page_size, num_heads, head_dim = 16, 2, 64
    if layout == "NHD":
        kv_cache = torch.zeros(4, 2, page_size, num_heads, head_dim, dtype=torch.float16)
    else:
        physical = torch.zeros(4, 2, num_heads, page_size, head_dim, dtype=torch.float16)
        kv_cache = physical.permute(0, 1, 3, 2, 4)
    metadata = _flashinfer_metadata()
    query = torch.zeros(4, num_heads, head_dim, dtype=torch.float16)
    layer = SimpleNamespace(
        _query_quant_in_kernel=True,
        q_bmm_quantizer=lambda value: value,
    )
    calls = {}

    def fake_finalize(value_cache, block_table, v_lo, v_hi, **kwargs):
        calls["finalize"] = (value_cache, block_table, kwargs)

    def fake_decode(query, key_cache, value_cache, block_table, seq_lens, **kwargs):
        calls["decode"] = (key_cache, value_cache, block_table, seq_lens, kwargs)
        return torch.ones_like(query)

    monkeypatch.setattr(vllm_plugin, "fake_quant_v_onwrite", fake_finalize)
    monkeypatch.setattr(vllm_plugin, "triton_decode_attention", fake_decode)

    output = torch.empty_like(query)
    assert impl.forward(layer, query, query, query, kv_cache, metadata, output=output) is output

    key_cache, value_cache, block_table, seq_lens, decode_kw = calls["decode"]
    assert key_cache.shape == (4, page_size, num_heads, head_dim)
    assert value_cache.shape == key_cache.shape
    assert key_cache.stride() == kv_cache[:, 0].stride()
    assert value_cache.stride() == kv_cache[:, 1].stride()
    assert key_cache.data_ptr() == kv_cache[:, 0].data_ptr()
    assert value_cache.data_ptr() == kv_cache[:, 1].data_ptr()
    assert block_table is metadata._modelopt_block_table
    assert seq_lens is metadata._modelopt_seq_lens
    assert decode_kw["page_size"] == page_size
    assert decode_kw["p_qdq"] == "nvfp4"
    assert decode_kw["v_qdq"] == "nvfp4"
    finalized_cache, finalized_table, finalize_kw = calls["finalize"]
    assert finalized_cache.data_ptr() == value_cache.data_ptr()
    assert finalized_table is block_table
    assert finalize_kw["page_size"] == page_size


def test_flashinfer_sparse_prefill_uses_shared_triton_kernel(monkeypatch):
    """FlashInfer must route 2:4 prefill through the current ModelOpt kernel."""
    impl = _make_flashinfer_impl(sparse=True)
    page_size, num_heads, head_dim = 16, 2, 64
    physical = torch.zeros(4, 2, num_heads, page_size, head_dim, dtype=torch.float16)
    kv_cache = physical.permute(0, 1, 3, 2, 4)
    metadata = _flashinfer_metadata(query_lens=(2, 2), max_query_len=4)
    query = torch.zeros(4, num_heads, head_dim, dtype=torch.float16)
    captured = {}

    def fake_attention(query, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(query)

    monkeypatch.setattr(vllm_plugin, "triton_attention", fake_attention)
    output = torch.empty_like(query)

    assert (
        impl.forward(SimpleNamespace(), query, query, query, kv_cache, metadata, output=output)
        is output
    )
    assert captured["sparsity_n"] == 2
    assert captured["sparsity_m"] == 4
    assert captured["page_size"] == page_size
    assert captured["k_cache"].stride() == kv_cache[:, 0].stride()
    assert captured["v_cache"].stride() == kv_cache[:, 1].stride()
    assert captured["block_table"] is metadata._modelopt_block_table
    assert captured["b_seq_len_k"] is metadata._modelopt_seq_lens


@pytest.mark.parametrize(("updates_in_forward", "expected_writes"), [(True, 1), (False, 0)])
def test_flashinfer_legacy_forward_writes_kv_cache(
    monkeypatch, updates_in_forward, expected_writes
):
    """vLLM releases where FlashInfer updates in forward must retain that write."""
    impl = _make_flashinfer_impl(sparse=True)
    impl.kv_sharing_target_layer_name = None
    query = torch.zeros(4, 2, 64, dtype=torch.float16)
    kv_cache = torch.zeros(4, 2, 16, 2, 64, dtype=torch.float16)
    metadata = _flashinfer_metadata(query_lens=(2, 2), max_query_len=4)
    writes = []

    monkeypatch.setattr(FlashInferBackend, "forward_includes_kv_cache_update", updates_in_forward)
    monkeypatch.setattr(vllm_plugin, "_flashinfer_cache_write", lambda *args: writes.append(args))
    monkeypatch.setattr(
        vllm_plugin, "triton_attention", lambda query, **kwargs: torch.zeros_like(query)
    )

    layer = SimpleNamespace()
    impl.forward(layer, query, query, query, kv_cache, metadata, output=torch.empty_like(query))

    assert len(writes) == expected_writes
    if expected_writes:
        assert writes == [(layer, query, query, kv_cache, metadata, impl)]


def test_flashinfer_q_only_transform_does_not_fallback(monkeypatch):
    """Withheld Q QDQ must run even when sparse/P/V transforms are inactive."""
    impl = _make_flashinfer_impl()
    query = torch.zeros(4, 2, 64, dtype=torch.float16)
    kv_cache = torch.zeros(4, 2, 16, 2, 64, dtype=torch.float16)
    metadata = _flashinfer_metadata(query_lens=(2, 2), max_query_len=4)
    captured = {}
    layer = SimpleNamespace(
        _query_quant_in_kernel=True,
        q_bmm_quantizer=lambda value: value + 1,
    )

    monkeypatch.setattr(
        FlashInferImpl,
        "forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("native fallback")),
    )

    def fake_attention(query, **kwargs):
        captured["query"] = query
        return torch.zeros_like(query)

    monkeypatch.setattr(vllm_plugin, "triton_attention", fake_attention)

    impl.forward(layer, query, query, query, kv_cache, metadata, output=torch.empty_like(query))

    assert torch.all(captured["query"] == 1)


def test_flashinfer_legacy_inactive_launch_writes_only_in_native_fallback(monkeypatch):
    impl = _make_flashinfer_impl(sparse=True)
    impl.kv_sharing_target_layer_name = None
    query = torch.zeros(1, 2, 64, dtype=torch.float16)
    kv_cache = torch.zeros(1, 2, 16, 2, 64, dtype=torch.float16)
    metadata = _flashinfer_metadata(query_lens=(1,), seq_lens=(16,))
    manual_writes = []
    native_calls = []

    monkeypatch.setattr(FlashInferBackend, "forward_includes_kv_cache_update", True)
    monkeypatch.setattr(
        vllm_plugin, "_flashinfer_cache_write", lambda *args: manual_writes.append(args)
    )
    monkeypatch.setattr(
        FlashInferImpl,
        "forward",
        lambda *args, **kwargs: native_calls.append(True) or kwargs.get("output", args[7]),
    )

    impl.forward(
        SimpleNamespace(), query, query, query, kv_cache, metadata, output=torch.empty_like(query)
    )

    assert manual_writes == []
    assert native_calls == [True]


@pytest.mark.parametrize("quantized", [False, True], ids=["sparse-only", "quantized"])
def test_flashinfer_mixed_batch_splits_decode_and_prefill(monkeypatch, quantized):
    prefill_tokens = 17
    impl = _make_flashinfer_impl(sparse=True, quantized=quantized)
    query = torch.zeros(1 + prefill_tokens, 2, 64, dtype=torch.float16)
    kv_cache = torch.zeros(4, 2, 16, 2, 64, dtype=torch.float16)
    metadata = _flashinfer_metadata(query_lens=(1, prefill_tokens), mixed=True)
    layer = SimpleNamespace(
        _query_quant_in_kernel=quantized,
        q_bmm_quantizer=lambda value: value,
    )
    calls = {"native": [], "decode": [], "prefill": [], "finalize": []}

    def fake_decode(query, *args, **kwargs):
        calls["decode"].append((query, kwargs))
        return torch.zeros_like(query)

    def fake_attention(query, **kwargs):
        calls["prefill"].append((query, kwargs))
        return torch.zeros_like(query)

    monkeypatch.setattr(
        FlashInferImpl,
        "forward",
        lambda *args, **kwargs: calls["native"].append(True) or args[7],
    )
    monkeypatch.setattr(
        vllm_plugin,
        "fake_quant_v_onwrite",
        lambda *args, **kwargs: calls["finalize"].append(kwargs),
    )
    monkeypatch.setattr(vllm_plugin, "triton_decode_attention", fake_decode)
    monkeypatch.setattr(vllm_plugin, "triton_attention", fake_attention)

    impl.forward(layer, query, query, query, kv_cache, metadata, output=torch.empty_like(query))

    assert len(calls["prefill"]) == 1
    prefill_query, prefill_kw = calls["prefill"][0]
    assert prefill_query.shape[0] == prefill_tokens
    assert prefill_kw["sparsity_n"] == 2
    assert prefill_kw["sparsity_m"] == 4
    torch.testing.assert_close(
        prefill_kw["b_seq_len"], torch.tensor([prefill_tokens], dtype=torch.int32)
    )

    if quantized:
        assert calls["native"] == []
        assert len(calls["decode"]) == 1
        decode_query, decode_kw = calls["decode"][0]
        assert decode_query.shape[0] == 1
        assert decode_kw["p_qdq"] == "nvfp4"
        assert "sparsity_n" not in decode_kw
        assert prefill_kw["p_qdq"] == "nvfp4"
        assert [kw["max_new_tokens"] for kw in calls["finalize"]] == [1, prefill_tokens]
    else:
        assert calls["native"] == [True]
        assert calls["decode"] == []
        assert calls["finalize"] == []


def _make_flash_attention_impl(*, sparse=False, quantized=False):
    impl = _clone_sparse_impl(_make_old_impl())
    impl.sparse_kw = {"sparsity_n": 2, "sparsity_m": 4} if sparse else {}
    impl.quant_kw = {
        "p_qdq": "nvfp4" if quantized else None,
        "p_qdq_amax": 1.0,
        "v_qdq": "nvfp4" if quantized else None,
        "v_qdq_amax": 6.0 * 448.0 if quantized else None,
    }
    return impl


def _flash_attention_mixed_metadata(decode_len=1, prefill_len=17):
    query_lens = (decode_len, prefill_len)
    seq_lens = (16, 34)
    return SimpleNamespace(
        use_cascade=False,
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        query_start_loc=torch.tensor(list(accumulate(query_lens, initial=0)), dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table=torch.zeros(len(seq_lens), 3, dtype=torch.int32),
        causal=True,
        num_decodes=1,
        num_prefills=1,
        num_decode_tokens=decode_len,
        num_prefill_tokens=prefill_len,
    )


@pytest.mark.parametrize("quantized", [False, True], ids=["sparse-only", "quantized"])
def test_flash_attention_mixed_batch_splits_decode_and_prefill(monkeypatch, quantized):
    """FlashAttention must split mixed decode+prefill batches so decode rows take
    the decode schedule -- NVFP4 P-QDQ is schedule-sensitive, so a decode result
    must not depend on a co-scheduled prefill (matches the FlashInfer adapter)."""
    prefill_tokens = 17
    impl = _make_flash_attention_impl(sparse=True, quantized=quantized)
    query = torch.zeros(1 + prefill_tokens, 2, 64, dtype=torch.float16)
    kv_cache = torch.zeros(2, 4, 16, 2, 64, dtype=torch.float16)
    metadata = _flash_attention_mixed_metadata(decode_len=1, prefill_len=prefill_tokens)
    layer = SimpleNamespace(
        _query_quant_in_kernel=quantized,
        q_bmm_quantizer=lambda value: value,
    )
    calls = {"native": [], "decode": [], "prefill": [], "finalize": []}

    def fake_decode(query, *args, **kwargs):
        calls["decode"].append((query, kwargs))
        return torch.full_like(query, 5)

    def fake_attention(query, **kwargs):
        calls["prefill"].append((query, kwargs))
        return torch.full_like(query, 7)

    def fake_native(*args, **kwargs):
        calls["native"].append(True)
        output = kwargs["output"] if "output" in kwargs else args[7]
        return output.fill_(3)

    monkeypatch.setattr(
        FlashAttentionImpl,
        "forward",
        fake_native,
    )
    monkeypatch.setattr(
        vllm_plugin,
        "fake_quant_v_onwrite",
        lambda *args, **kwargs: calls["finalize"].append(kwargs),
    )
    monkeypatch.setattr(vllm_plugin, "triton_decode_attention", fake_decode)
    monkeypatch.setattr(vllm_plugin, "triton_attention", fake_attention)

    result = impl.forward(
        layer, query, query, query, kv_cache, metadata, output=torch.empty_like(query)
    )

    assert torch.all(result[1:] == 7)
    assert torch.all(result[:1] == (5 if quantized else 3))

    assert len(calls["prefill"]) == 1
    prefill_query, prefill_kw = calls["prefill"][0]
    assert prefill_query.shape[0] == prefill_tokens
    assert prefill_kw["sparsity_n"] == 2
    assert prefill_kw["sparsity_m"] == 4

    if quantized:
        assert calls["native"] == []
        assert len(calls["decode"]) == 1
        decode_query, decode_kw = calls["decode"][0]
        assert decode_query.shape[0] == 1
        assert decode_kw["p_qdq"] == "nvfp4"
        assert "sparsity_n" not in decode_kw
    else:
        assert calls["native"] == [True]
        assert calls["decode"] == []


def test_flashinfer_invalid_mixed_metadata_has_no_side_effects(monkeypatch):
    impl = _make_flashinfer_impl(sparse=True)
    metadata = _flashinfer_metadata(query_lens=(1, 17), mixed=True)
    metadata._modelopt_num_actual_tokens += 1
    query = torch.zeros(18, 2, 64, dtype=torch.float16)
    kv_cache = torch.zeros(4, 2, 16, 2, 64, dtype=torch.float16)

    def unexpected_side_effect(*args, **kwargs):
        pytest.fail("invalid metadata triggered attention side effects")

    monkeypatch.setattr(FlashInferImpl, "forward", unexpected_side_effect)
    for name in ("_flashinfer_cache_write", "triton_attention"):
        monkeypatch.setattr(vllm_plugin, name, unexpected_side_effect)

    with pytest.raises(
        ValueError,
        match="Mixed-batch token counts do not match common metadata",
    ):
        impl.forward(
            SimpleNamespace(),
            query,
            query,
            query,
            kv_cache,
            metadata,
            output=torch.empty_like(query),
        )


@pytest.mark.parametrize(
    ("failure", "active", "expected"),
    [
        (
            "metadata",
            True,
            "FlashInfer metadata is missing the ModelOpt attention transform fields: "
            + ", ".join(vllm_plugin._FLASHINFER_METADATA_FIELDS),
        ),
        ("metadata", False, None),
        ("output", True, "Fused attention output quantization is unsupported"),
    ],
)
def test_flashinfer_transform_safety_for_unsupported_metadata(
    monkeypatch, failure, active, expected
):
    impl = _make_flashinfer_impl(sparse=active)
    metadata = (
        _flashinfer_metadata()
        if failure == "output"
        else SimpleNamespace(use_cascade=failure == "cascade")
    )
    native_calls = []
    query = torch.zeros(1, 2, 64, dtype=torch.float16)
    output = torch.empty_like(query)

    def fake_forward(self, *args, **kwargs):
        native_calls.append((self, args[5]))
        output = args[6]
        output.fill_(9)
        return output

    monkeypatch.setattr(FlashInferImpl, "forward", fake_forward)

    if active:
        with pytest.raises(NotImplementedError) as exc:
            impl.forward(
                None,
                query,
                query,
                query,
                torch.empty(0),
                metadata,
                output=output,
                output_scale=torch.ones(1),
            )
        assert str(exc.value) == expected
        assert native_calls == []
    else:
        assert (
            impl.forward(
                None,
                query,
                query,
                query,
                torch.empty(0),
                metadata,
                output=output,
                output_scale=torch.ones(1),
            )
            is output
        )
        assert native_calls == [(impl, metadata)]
        assert torch.all(output == 9)


@pytest.mark.parametrize("quantized", [False, True], ids=["sparse-only", "quantized"])
def test_flashinfer_cascade_falls_back_for_sparse_only_but_rejects_quant(monkeypatch, quantized):
    """Cascade is unimplemented by the kernel. A sparse-only transform can safely
    delegate to the native dense path; quantization must not be silently dropped
    (it would change numerics), so it is rejected instead."""
    impl = _make_flashinfer_impl(sparse=True, quantized=quantized)
    metadata = SimpleNamespace(use_cascade=True)
    query = torch.zeros(1, 2, 64, dtype=torch.float16)
    output = torch.empty_like(query)
    native_calls = []

    def fake_forward(self, *args, **kwargs):
        native_calls.append(True)
        out = args[6]
        out.fill_(9)
        return out

    monkeypatch.setattr(FlashInferImpl, "forward", fake_forward)

    if quantized:
        with pytest.raises(NotImplementedError) as exc:
            impl.forward(None, query, query, query, torch.empty(0), metadata, output=output)
        assert str(exc.value) == (
            "vLLM cascade attention is incompatible with active ModelOpt attention quantization"
        )
        assert native_calls == []
    else:
        assert (
            impl.forward(None, query, query, query, torch.empty(0), metadata, output=output)
            is output
        )
        assert native_calls == [True]
        assert torch.all(output == 9)


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
        (
            {
                "sparsity_n": 2,
                "sparsity_m": 4,
                "dense_sink_tokens": 4,
                "dense_recent_tokens": 128,
            },
            1,
            16,
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
    attn_metadata = _flash_attention_metadata(max_query_len, max_seq_len)
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
    attn_metadata = _flash_attention_metadata(max_query_len, seq_len)
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


def test_unsupported_nvfp4_bmm_block_size_raises():
    """P/V mappings reject NVFP4 quantizers that are not block-16."""
    quantizer = SimpleNamespace(
        is_enabled=True,
        is_nvfp4_dynamic=True,
        block_sizes={-1: 32},
    )
    layer = SimpleNamespace(p_bmm_quantizer=quantizer, v_bmm_quantizer=quantizer)
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
    assert finalizer_kw == {
        "max_new_tokens": 1,
        "page_size": 16,
        "v_qdq_scale": 1.0,
    }
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
    metadata = _flash_attention_metadata(1, 16)
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


def test_forward_allows_chunked_prefill_metadata(monkeypatch):
    """vLLM V1 can pass suffix-Q/chunked-prefill metadata; the kernel handles it."""
    impl = _clone_sparse_impl(_make_old_impl())
    impl.sparse_kw = {"sparsity_n": 2, "sparsity_m": 4}
    q_len = 4
    kv_len = 10
    q = torch.zeros(q_len, impl.num_heads, impl.head_size, dtype=torch.float16)
    kv_cache = torch.zeros(2, 1, 16, impl.num_kv_heads, impl.head_size, dtype=torch.float16)
    attn_metadata = _flash_attention_metadata(q_len, kv_len)
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
