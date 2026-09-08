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

"""Tests for the vLLM MLA attention runtime installer and impl adapter."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

# The MLA adapter targets the vLLM 0.26-style prefill-backend architecture.
pytest.importorskip("vllm.v1.attention.backends.mla.prefill.base")

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl

from modelopt.torch.quantization.plugins import vllm as quant_plugin
from modelopt.torch.sparsity.attention_sparsity.plugins import vllm_mla, vllm_runtime

_MLA_ATTENTION = vllm_runtime._import_mla_attention_type()

pytestmark = pytest.mark.skipif(
    _MLA_ATTENTION is None, reason="vLLM MLAAttention type is unavailable"
)


def _bare_mla_attention(impl_cls=TritonMLAImpl, num_heads=8):
    module = object.__new__(_MLA_ATTENTION)
    nn.Module.__init__(module)
    module.kv_cache_dtype = "auto"
    module.use_sparse = False
    module.indexer = None
    module.q_pad_num_heads = None
    module.num_heads = num_heads
    module.kv_lora_rank = 128
    module.qk_rope_head_dim = 64
    module.qk_nope_head_dim = 128
    module.qk_head_dim = 192
    module.v_head_dim = 128
    module.device = torch.device("cpu")
    module.dtype = torch.float16
    impl = object.__new__(impl_cls)
    impl.scale = 192**-0.5
    impl.num_heads = num_heads
    impl.kv_lora_rank = module.kv_lora_rank
    impl.qk_rope_head_dim = module.qk_rope_head_dim
    module.impl = impl
    return module


def _model_runner(model):
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(sparse_attention_config=None), dtype=torch.float16
    )
    return SimpleNamespace(
        model=model,
        model_config=model_config,
        cascade_attn_enabled=True,
        vllm_config=SimpleNamespace(
            model_config=model_config,
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                enable_dbo=False,
                use_ubatching=False,
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False, cache_dtype="auto"),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
            kv_transfer_config=None,
            speculative_config=None,
        ),
    )


@pytest.fixture
def patched_parallel_state(monkeypatch):
    monkeypatch.setattr(
        quant_plugin,
        "create_parallel_state",
        lambda: quant_plugin.ParallelState(data_parallel_group=None),
    )


class TestMLAInstall:
    def test_nvfp4_install_configures_quantizers_and_impl(self, patched_parallel_state):
        attention = _bare_mla_attention()
        runner = _model_runner(nn.ModuleDict({"mla_attn": attention}))

        report = vllm_runtime.install_vllm_nvfp4_attention(runner, sparse_cfg=None)

        assert isinstance(attention, quant_plugin._QuantVLLMMLAAttention)
        assert type(attention.impl) is vllm_mla.ModelOptMLAImpl
        for name in ("q", "kv_c", "k_pe", "k_mha", "p", "v_mha"):
            quantizer = getattr(attention, f"{name}_bmm_quantizer")
            assert quantizer.is_enabled
            assert quantizer.is_nvfp4_dynamic
        for name in ("kv_c", "k_pe", "k_mha", "v_mha"):
            amax = getattr(attention, f"{name}_bmm_quantizer")._amax
            assert float(amax) == 6.0 * 448.0
        assert attention._query_quant_in_kernel is True
        assert not hasattr(attention, "_value_quant_in_kernel")
        # Quantized-BMM model: the module flag keeps the latent cache raw by
        # skipping the module-side kv_c/k_pe quant; operands are quantized
        # in-kernel (prefill projected K/V; decode on-read latent K/V).
        assert attention._skip_module_kv_quant is True
        assert attention.impl.quant_kw.prefill == {
            "q_quant": "nvfp4",
            "q_amax": None,
            "k_quant": "nvfp4",
            "k_amax": 6.0 * 448.0,
            "p_quant": "nvfp4",
            "p_amax": 1.0,
            "v_quant": "nvfp4",
            "v_amax": 6.0 * 448.0,
        }
        assert attention.impl.quant_kw.decode == {
            "p_qdq": "nvfp4",
            "p_qdq_amax": 1.0,
            "k_qdq": "nvfp4",
            "k_qdq_amax": 6.0 * 448.0,
            "v_qdq": "nvfp4",
            "v_qdq_amax": 6.0 * 448.0,
        }
        assert attention.impl.sparse_kw == {}
        assert report.installed_layers == ("mla_attn",)
        assert report.quantized_layers == ("mla_attn",)
        assert report.backend_counts == {"ModelOptMLAImpl": 1}
        assert runner.cascade_attn_enabled is False

    def test_fp8_q_and_v_formats(self, patched_parallel_state):
        attention = _bare_mla_attention()
        runner = _model_runner(nn.ModuleDict({"mla_attn": attention}))

        vllm_runtime.install_vllm_nvfp4_attention(
            runner, sparse_cfg=None, q_format="fp8", v_format="fp8"
        )

        assert attention._query_quant_in_kernel is False
        # Module-level FP8 Q means neither kernel applies a Q transform.
        assert attention.impl.quant_kw.prefill["q_quant"] is None
        assert attention.impl.quant_kw.prefill["v_quant"] == "fp8"
        assert attention.impl.quant_kw.prefill["v_amax"] == 448.0
        assert float(attention.q_bmm_quantizer._amax) == 448.0

    def test_unsupported_mla_backend_rejected_before_mutation(self, patched_parallel_state):
        attention = _bare_mla_attention(impl_cls=FlashAttentionImpl)
        original_impl = attention.impl
        runner = _model_runner(nn.ModuleDict({"mla_attn": attention}))

        with pytest.raises(NotImplementedError, match="TRITON_MLA"):
            vllm_runtime.install_vllm_nvfp4_attention(runner, sparse_cfg=None)

        assert attention.impl is original_impl
        assert not hasattr(attention, "q_bmm_quantizer")
        assert runner.cascade_attn_enabled is True

    @pytest.mark.parametrize(
        ("mutation", "match"),
        [
            ({"kv_cache_dtype": "fp8"}, "FP8 KV cache"),
            ({"indexer": object()}, "sparse-indexer"),
            ({"use_sparse": True}, "sparse-indexer"),
            ({"q_pad_num_heads": 256}, "q_pad_num_heads"),
            ({"v_head_dim": 100}, "v_head_dim"),
        ],
    )
    def test_layer_gates(self, patched_parallel_state, mutation, match):
        attention = _bare_mla_attention()
        for name, value in mutation.items():
            setattr(attention, name, value)
        runner = _model_runner(nn.ModuleDict({"mla_attn": attention}))

        with pytest.raises(NotImplementedError, match=match):
            vllm_runtime.install_vllm_nvfp4_attention(runner, sparse_cfg=None)

    def test_skip_softmax_sparse_config_rejected(self, patched_parallel_state, monkeypatch):
        attention = _bare_mla_attention()
        runner = _model_runner(nn.ModuleDict({"mla_attn": attention}))
        monkeypatch.setattr(
            vllm_runtime,
            "_sparse_kwargs",
            lambda name, cfg: {"skip_softmax_threshold": 0.5},
        )

        with pytest.raises(NotImplementedError, match="skip-softmax is unsupported on MLA"):
            vllm_runtime.install_vllm_nvfp4_attention(runner, sparse_cfg={"*": {"enable": True}})

    def test_sparse_only_install_ignores_mla(self):
        attention = _bare_mla_attention()
        original_impl = attention.impl
        runner = _model_runner(nn.ModuleDict({"mla_attn": attention}))
        runner.model_config.hf_config.sparse_attention_config = {
            "config_groups": {
                "group_0": {"algorithm": "sparse_softmax", "sparsity_n": 2, "sparsity_m": 4}
            }
        }

        report = vllm_runtime.install_vllm_sparse_attention_from_checkpoint(runner)

        assert attention.impl is original_impl
        assert not hasattr(attention, "q_bmm_quantizer")
        assert report.installed_layers == ()


def _bare_mla_impl(*, p_qdq="nvfp4", prefill_active=True, sparse_kw=None):
    impl = object.__new__(vllm_mla.ModelOptMLAImpl)
    impl.scale = 0.25
    impl.kv_lora_rank = 128
    impl.qk_rope_head_dim = 64
    mode = "nvfp4" if prefill_active else None
    impl.quant_kw = vllm_mla._MLAQuantKw(
        prefill={
            "q_quant": mode,
            "q_amax": None,
            "k_quant": mode,
            "k_amax": None,
            "p_quant": mode,
            "p_amax": 1.0,
            "v_quant": mode,
            "v_amax": None,
        },
        decode={
            "p_qdq": p_qdq,
            "p_qdq_amax": 1.0,
            "k_qdq": mode,
            "k_qdq_amax": None,
            "v_qdq": mode,
            "v_qdq_amax": None,
        },
    )
    impl.sparse_kw = dict(sparse_kw or {})
    return impl


class TestMLAImplDispatch:
    def test_forward_mqa_dispatches_to_decode_kernel(self, monkeypatch):
        impl = _bare_mla_impl()
        recorded = {}

        def _record_decode(q, latent_cache, block_table, b_seq_len, **kwargs):
            recorded["q"] = q
            recorded["latent_cache"] = latent_cache
            recorded["block_table"] = block_table
            recorded["b_seq_len"] = b_seq_len
            recorded.update(kwargs)
            return torch.zeros(q.shape[0], q.shape[1], 128), torch.zeros(q.shape[0], q.shape[1])

        monkeypatch.setattr(vllm_mla, "mla_attention_decode", _record_decode)

        def _poison(self, *args, **kwargs):
            raise AssertionError("native TritonMLAImpl.forward_mqa must not run")

        monkeypatch.setattr(TritonMLAImpl, "forward_mqa", _poison)

        layer = SimpleNamespace(_query_quant_in_kernel=False)
        cache = torch.zeros(4, 16, 192, dtype=torch.float16)
        ql_nope = torch.zeros(2, 8, 128, dtype=torch.float16)
        q_pe = torch.zeros(2, 8, 64, dtype=torch.float16)
        metadata = SimpleNamespace(
            decode=SimpleNamespace(
                block_table=torch.zeros(2, 4, dtype=torch.int32),
                seq_lens=torch.tensor([5, 9], dtype=torch.int32),
            )
        )

        o, lse = impl.forward_mqa((ql_nope, q_pe), cache, metadata, layer)

        assert recorded["q"].shape == (2, 8, 192)  # tuple q concatenated
        assert recorded["latent_cache"] is cache
        assert recorded["block_table"] is metadata.decode.block_table
        assert recorded["b_seq_len"] is metadata.decode.seq_lens
        assert recorded["softmax_scale"] == impl.scale
        assert recorded["num_kv_splits"] == 32
        assert recorded["p_qdq"] == "nvfp4"
        assert recorded["p_qdq_amax"] == 1.0
        # Decode quantizes K (feature) and V (token) on read from the raw cache.
        assert recorded["k_qdq"] == "nvfp4"
        assert recorded["v_qdq"] == "nvfp4"
        assert recorded["kv_lora_rank"] == 128
        assert recorded["qk_rope_head_dim"] == 64
        assert recorded["out_dtype"] == cache.dtype
        assert o.shape == (2, 8, 128)
        assert lse.shape == (2, 8)

    def test_forward_mqa_no_transform_uses_native_path(self, monkeypatch):
        impl = _bare_mla_impl(p_qdq=None, prefill_active=False)
        called = {}

        def _native(self, q, cache, metadata, layer):
            called["native"] = True
            return "native-result"

        monkeypatch.setattr(TritonMLAImpl, "forward_mqa", _native)
        layer = SimpleNamespace(_query_quant_in_kernel=False)

        result = impl.forward_mqa(torch.zeros(1, 8, 192), None, None, layer)

        assert called.get("native") is True
        assert result == "native-result"

    @pytest.mark.parametrize("super_raises", [False, True])
    def test_forward_mha_swaps_and_restores_prefill_backend(self, monkeypatch, super_raises):
        impl = _bare_mla_impl()
        base_backend = SimpleNamespace(
            num_heads=8,
            scale=impl.scale,
            kv_lora_rank=128,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            vllm_config=None,
        )
        prefill_metadata = SimpleNamespace(prefill_backend=base_backend)
        metadata = SimpleNamespace(prefill=prefill_metadata)
        seen = {}

        def _fake_super(self, q, kv_c, k_pe, cache, attn_metadata, k_scale, output, output_scale):
            seen["backend_during_call"] = attn_metadata.prefill.prefill_backend
            if super_raises:
                raise RuntimeError("boom")

        monkeypatch.setattr(TritonMLAImpl, "forward_mha", _fake_super)

        args = (None, None, None, None, metadata, None, None, None)
        if super_raises:
            with pytest.raises(RuntimeError, match="boom"):
                impl.forward_mha(*args)
        else:
            impl.forward_mha(*args)

        assert isinstance(seen["backend_during_call"], vllm_mla._ModelOptMLAPrefillBackend)
        assert seen["backend_during_call"]._prefill_metadata is prefill_metadata
        assert prefill_metadata.prefill_backend is base_backend  # restored

    def test_forward_mha_no_transform_uses_native_backend(self, monkeypatch):
        impl = _bare_mla_impl(p_qdq=None, prefill_active=False)
        base_backend = SimpleNamespace(prefill_backend=None)
        prefill_metadata = SimpleNamespace(prefill_backend=base_backend)
        metadata = SimpleNamespace(prefill=prefill_metadata)
        seen = {}

        def _fake_super(self, q, kv_c, k_pe, cache, attn_metadata, k_scale, output, output_scale):
            seen["backend_during_call"] = attn_metadata.prefill.prefill_backend

        monkeypatch.setattr(TritonMLAImpl, "forward_mha", _fake_super)

        impl.forward_mha(None, None, None, None, metadata, None, None, None)

        assert seen["backend_during_call"] is base_backend  # untouched


def test_impl_keeps_cache_raw_no_update_override():
    """Quantized-BMM model: the latent cache stays RAW, so the impl must NOT
    override do_kv_cache_update (which would re-introduce a cache-write quant).
    Decode instead quantizes K/V on read; prefill quantizes projected operands.
    """
    assert vllm_mla.ModelOptMLAImpl.do_kv_cache_update is TritonMLAImpl.do_kv_cache_update
