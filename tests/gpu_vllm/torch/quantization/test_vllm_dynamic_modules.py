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

"""End-to-end tests for the vLLM fakequant dynamic modules.

Boots ``vllm.LLM`` on tiny HF models (saved via
``_test_utils.torch.transformers_models``) and runs ``mtq.quantize`` inside the
worker via ``LLM.collective_rpc``. Asserts every ``_QuantVLLM…`` class is
installed and every enabled quantizer ends up with a registered tensor-level
``_amax`` after calibration. Mirrors the
``examples/vllm_serve/fakequant_worker.py`` production path.

Architectures: TinyLlama (Linear + Attention), TinyQwen3MoE (+ FusedMoE),
TinyDeepseekV3 (+ MLAAttention).
"""

from __future__ import annotations

import gc
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _test_utils.torch.transformers_models import (
    create_tiny_deepseek_v3_dir,
    create_tiny_llama_dir,
    create_tiny_qwen3_moe_dir,
)
from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.plugins import vllm as vllm_plugin
from modelopt.torch.quantization.plugins.vllm import (
    _ATTENTION_TYPES,
    VllmMLAAttention,
    _QuantFusedMoEBase,
    _QuantVLLMAttention,
    _VLLMParallelLinear,
    build_vllm_attention_quant_cfg,
    configure_vllm_nvfp4_attention_quantizers,
    disable_compilation,
)


class _NativeAttention(torch.nn.Module):
    def forward(self, query, key, value, *args, **kwargs):
        return query, key, value


class _TestQuantVLLMAttention(_QuantVLLMAttention, _NativeAttention):
    pass


def _new_attention(cls):
    attention = object.__new__(cls)
    torch.nn.Module.__init__(attention)
    return attention


def _nvfp4_quantizer(*, block_size=16, enabled=True):
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1),
            block_sizes={-1: block_size, "type": "dynamic", "scale_bits": (4, 3)},
            enable=enabled,
        )
    )
    return quantizer


def test_attention_setup_keeps_qkv_only_checkpoint_surface(monkeypatch):
    monkeypatch.setattr(
        vllm_plugin,
        "create_parallel_state",
        lambda: vllm_plugin.ParallelState(data_parallel_group=None),
    )
    attention = _new_attention(_TestQuantVLLMAttention)

    attention._setup()

    quantizer_names = ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer")
    assert set(dict(attention.named_children())) == set(quantizer_names)
    for name in quantizer_names:
        getattr(attention, name).amax = torch.tensor(1.0)
    assert set(attention.state_dict()) == {f"{name}._amax" for name in quantizer_names}
    assert not hasattr(attention, "_query_quant_in_kernel")
    assert not hasattr(attention, "_value_quant_in_kernel")

    attention.k_bmm_quantizer = _nvfp4_quantizer()
    attention.v_bmm_quantizer = _nvfp4_quantizer()
    attention.device, attention.dtype = torch.device("cpu"), torch.float32
    attention.modelopt_post_restore()
    assert not hasattr(attention.k_bmm_quantizer, "_amax")
    assert not hasattr(attention.v_bmm_quantizer, "_amax")


def test_configure_vllm_nvfp4_attention_quantizers_is_attention_scoped(monkeypatch):
    monkeypatch.setattr(
        vllm_plugin,
        "create_parallel_state",
        lambda: vllm_plugin.ParallelState(data_parallel_group=None),
    )
    attention = object.__new__(vllm_plugin.vllm_attention.Attention)
    torch.nn.Module.__init__(attention)
    linear = torch.nn.Linear(4, 4)
    attention.unrelated_linear = linear
    original_linear_type = type(linear)

    converted = configure_vllm_nvfp4_attention_quantizers(
        attention, device="cpu", dtype=torch.bfloat16
    )

    assert converted is attention
    assert isinstance(converted, _QuantVLLMAttention)
    assert converted.device == torch.device("cpu")
    assert converted.dtype == torch.bfloat16
    assert type(linear) is original_linear_type
    for name in ("q", "k", "p", "v"):
        quantizer = getattr(converted, f"{name}_bmm_quantizer")
        assert quantizer.is_enabled
        assert quantizer.is_nvfp4_dynamic
        assert quantizer.block_sizes[-1] == 16
    assert not hasattr(converted.q_bmm_quantizer, "_amax")
    assert not hasattr(converted.p_bmm_quantizer, "_amax")
    assert converted.k_bmm_quantizer._amax == 6.0 * 448.0
    assert converted.v_bmm_quantizer._amax == 6.0 * 448.0
    assert not hasattr(converted, "_query_quant_in_kernel")
    assert not hasattr(converted, "_value_quant_in_kernel")


def test_configure_vllm_nvfp4_attention_quantizers_preserves_and_moves_amax(monkeypatch):
    monkeypatch.setattr(
        vllm_plugin,
        "create_parallel_state",
        lambda: vllm_plugin.ParallelState(data_parallel_group=None),
    )
    attention = object.__new__(vllm_plugin.vllm_attention.Attention)
    torch.nn.Module.__init__(attention)
    converted = configure_vllm_nvfp4_attention_quantizers(
        attention, device="cpu", dtype=torch.float16
    )
    for name, value in zip(("q", "k", "p", "v"), (13.0, 17.0, 23.0, 19.0), strict=True):
        getattr(converted, f"{name}_bmm_quantizer").amax = torch.tensor(value)
    reconfigured = configure_vllm_nvfp4_attention_quantizers(
        converted, device="cpu", dtype=torch.float16
    )

    assert reconfigured is converted
    assert converted.q_bmm_quantizer._amax == 13.0
    assert converted.k_bmm_quantizer._amax == 17.0
    assert converted.p_bmm_quantizer._amax == 23.0
    assert converted.v_bmm_quantizer._amax == 19.0

    configure_vllm_nvfp4_attention_quantizers(converted, device="meta", dtype=torch.float16)
    for name in ("q", "k", "p", "v"):
        assert getattr(converted, f"{name}_bmm_quantizer")._amax.device.type == "meta"


def test_quant_vllm_attention_forward_skips_only_in_kernel_qv_quantization():
    attention = _new_attention(_TestQuantVLLMAttention)
    attention.q_bmm_quantizer = Mock(side_effect=lambda inputs: inputs + 1)
    attention.k_bmm_quantizer = Mock(side_effect=lambda inputs: inputs + 2)
    attention.v_bmm_quantizer = Mock(side_effect=lambda inputs: inputs + 3)
    query = torch.tensor(10)
    key = torch.tensor(20)
    value = torch.tensor(30)

    assert not hasattr(attention, "_query_quant_in_kernel")
    assert not hasattr(attention, "_value_quant_in_kernel")
    quantized = attention(query, key, value)
    attention._query_quant_in_kernel = True
    query_in_kernel = attention(query, key, value)
    attention._value_quant_in_kernel = True
    qv_in_kernel = attention(query, key, value)

    assert quantized[:3] == (torch.tensor(11), torch.tensor(22), torch.tensor(33))
    assert query_in_kernel[:3] == (query, torch.tensor(22), torch.tensor(33))
    assert qv_in_kernel[:3] == (query, torch.tensor(22), value)
    assert attention.q_bmm_quantizer.call_count == 1
    assert attention.k_bmm_quantizer.call_count == 3
    assert attention.v_bmm_quantizer.call_count == 2


def test_attention_kv_defaults_set_only_uncalibrated_dynamic_block16_quantizers():
    calibrated_amax = 7.25
    layer = SimpleNamespace(
        q_bmm_quantizer=_nvfp4_quantizer(),
        k_bmm_quantizer=_nvfp4_quantizer(),
        v_bmm_quantizer=_nvfp4_quantizer(),
        p_bmm_quantizer=_nvfp4_quantizer(),
    )
    layer.v_bmm_quantizer.amax = calibrated_amax

    vllm_plugin._set_vllm_attention_kv_default_amax(layer, torch.device("cpu"))

    assert layer.k_bmm_quantizer._amax.item() == 6.0 * 448.0
    assert layer.v_bmm_quantizer._amax.item() == calibrated_amax
    assert not hasattr(layer.q_bmm_quantizer, "_amax")
    assert not hasattr(layer.p_bmm_quantizer, "_amax")


def test_attention_kv_defaults_ignore_unsupported_quantizers():
    for quantizer in (
        TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3))),
        _nvfp4_quantizer(block_size=32),
        _nvfp4_quantizer(enabled=False),
    ):
        layer = SimpleNamespace(k_bmm_quantizer=quantizer, v_bmm_quantizer=quantizer)
        vllm_plugin._set_vllm_attention_kv_default_amax(layer, torch.device("cpu"))
        assert not hasattr(quantizer, "_amax")


def _quantize_and_summarize(self):
    """Run on the worker via ``LLM.collective_rpc``.

    Module-level so it survives pickle over engine-core IPC. ``self`` is the
    vLLM worker — needed to drive ``model_runner._dummy_run`` from the
    calibration forward_loop. Returns a JSON-able summary.
    """
    model = self.get_model()

    def _forward_loop(_model):
        # ``num_tokens=1`` is enough for the ``"max"`` calibrator.
        self.model_runner._dummy_run(1)

    with disable_compilation(model):
        mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=_forward_loop)

    parallel_linear_counts: dict[str, int] = {}
    moe_count = 0
    attention_count = 0
    mla_count = 0
    missing_quantizers: list[str] = []
    quantizers_without_amax: list[str] = []
    enabled_quantizer_count = 0

    def _missing(module, name, slots):
        return (
            f"{name}.{slot}"
            for slot in slots
            if not isinstance(getattr(module, slot, None), TensorQuantizer)
        )

    for name, module in model.named_modules():
        if isinstance(module, _VLLMParallelLinear):
            kind = type(module).__name__
            parallel_linear_counts[kind] = parallel_linear_counts.get(kind, 0) + 1
            missing_quantizers.extend(
                _missing(module, name, ("input_quantizer", "weight_quantizer", "output_quantizer"))
            )
        elif isinstance(module, _QuantFusedMoEBase):
            moe_count += 1
            missing_quantizers.extend(
                _missing(
                    module,
                    name,
                    (
                        "w13_input_quantizer",
                        "w2_input_quantizer",
                        "w13_weight_quantizer",
                        "w2_weight_quantizer",
                    ),
                )
            )
        elif VllmMLAAttention is not None and isinstance(module, VllmMLAAttention):
            mla_count += 1
            missing_quantizers.extend(
                _missing(
                    module, name, ("q_bmm_quantizer", "kv_c_bmm_quantizer", "k_pe_bmm_quantizer")
                )
            )
        elif isinstance(module, _ATTENTION_TYPES):
            attention_count += 1
            missing_quantizers.extend(
                _missing(module, name, ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer"))
            )

        # Static-amax invariant: every enabled quantizer must own an ``_amax``
        # after calibration. ``kv_b_proj`` is exempt — vLLM's MLA decode path
        # reads its weight directly and never calls its forward.
        if isinstance(module, TensorQuantizer) and module.is_enabled:
            enabled_quantizer_count += 1
            if not hasattr(module, "_amax") and "kv_b_proj" not in name:
                quantizers_without_amax.append(name)

    return {
        "parallel_linear_counts": parallel_linear_counts,
        "moe_count": moe_count,
        "attention_count": attention_count,
        "mla_count": mla_count,
        "missing_quantizers": missing_quantizers,
        "quantizers_without_amax": quantizers_without_amax,
        "enabled_quantizer_count": enabled_quantizer_count,
    }


def _boot_llm(model_dir, **extra):
    """Construct a vLLM engine on a tiny model.

    MoE fixtures override with ``moe_backend="triton"`` (pins the Triton
    experts kernel whose module-level entries the modelopt plugin patches —
    FlashInfer/TRTLLM kernels bypass them) and ``enable_expert_parallel=True``
    (keeps modelopt's MoE-specific calibration paths live).
    """
    return LLM(
        model=str(model_dir),
        enforce_eager=True,
        gpu_memory_utilization=0.2,
        max_model_len=64,
        max_num_seqs=1,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        **extra,
    )


def _shutdown_llm(llm):
    del llm
    gc.collect()
    cleanup_dist_env_and_memory(shutdown_ray=False)


@pytest.fixture(scope="module")
def tiny_llama_llm(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiny_llama")
    # Helper default ``max_position_embeddings=32`` would clash with vLLM's
    # ``max_model_len=64`` set in ``_boot_llm``.
    model_dir = create_tiny_llama_dir(tmp, max_position_embeddings=64)
    llm = _boot_llm(model_dir)
    try:
        yield llm
    finally:
        _shutdown_llm(llm)


@pytest.fixture(scope="module")
def tiny_qwen3_moe_llm(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiny_qwen3_moe")
    # head_dim=64 with num_heads=2 is broadly supported by vLLM's attention backends.
    model_dir = create_tiny_qwen3_moe_dir(
        tmp,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        vocab_size=128,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
    )
    llm = _boot_llm(model_dir, moe_backend="triton", enable_expert_parallel=True)
    try:
        yield llm
    finally:
        _shutdown_llm(llm)


@pytest.fixture(scope="module")
def tiny_deepseek_llm(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiny_deepseek")
    model_dir = create_tiny_deepseek_v3_dir(tmp)
    llm = _boot_llm(model_dir, moe_backend="triton", enable_expert_parallel=True)
    try:
        yield llm
    finally:
        _shutdown_llm(llm)


def _assert_quantizer_amax_is_static(summary):
    """Every enabled quantizer must own a registered ``_amax`` after
    calibration. Missing ``_amax`` → repr ``amax=dynamic`` → regression.
    """
    assert summary["enabled_quantizer_count"] > 0, summary
    assert summary["quantizers_without_amax"] == [], summary["quantizers_without_amax"]


def test_tiny_llama_quantize(tiny_llama_llm):
    """Covers QKV/Row/MergedColumn ParallelLinear + Attention on a dense Llama."""
    summaries = tiny_llama_llm.collective_rpc(_quantize_and_summarize)
    summary = summaries[0]

    assert summary["missing_quantizers"] == [], summary["missing_quantizers"]

    parallel_linear_counts = summary["parallel_linear_counts"]
    # Each decoder layer contributes one of each. With num_hidden_layers=2:
    assert parallel_linear_counts.get("QuantQKVParallelLinear", 0) >= 2, parallel_linear_counts
    # o_proj + down_proj per layer
    assert parallel_linear_counts.get("QuantRowParallelLinear", 0) >= 4, parallel_linear_counts
    assert parallel_linear_counts.get("QuantMergedColumnParallelLinear", 0) >= 2, (
        parallel_linear_counts
    )

    # Llama uses the base Attention type — one per decoder layer.
    assert summary["attention_count"] >= 2, summary

    # No MoE in a dense Llama.
    assert summary["moe_count"] == 0

    _assert_quantizer_amax_is_static(summary)


def test_tiny_qwen3_moe_quantize(tiny_qwen3_moe_llm):
    """Tiny Qwen3-MoE adds FusedMoE coverage on top of the dense linears."""
    summaries = tiny_qwen3_moe_llm.collective_rpc(_quantize_and_summarize)
    summary = summaries[0]

    assert summary["missing_quantizers"] == [], summary["missing_quantizers"]

    parallel_linear_counts = summary["parallel_linear_counts"]
    assert parallel_linear_counts.get("QuantQKVParallelLinear", 0) >= 2, parallel_linear_counts
    assert parallel_linear_counts.get("QuantRowParallelLinear", 0) >= 2, parallel_linear_counts

    # decoder_sparse_step=1 → every layer is MoE. With 2 layers we expect ≥2 FusedMoE.
    assert summary["moe_count"] >= 2, summary
    assert summary["attention_count"] >= 2, summary

    _assert_quantizer_amax_is_static(summary)


def test_tiny_deepseek_mla_quantize(tiny_deepseek_llm):
    """Tiny DeepSeek-V3 covers MLAAttention (and again FusedMoE)."""
    summaries = tiny_deepseek_llm.collective_rpc(_quantize_and_summarize)
    summary = summaries[0]

    assert summary["missing_quantizers"] == [], summary["missing_quantizers"]
    assert summary["mla_count"] >= 2, summary
    # ``first_k_dense_replace=0`` → every layer is MoE.
    assert summary["moe_count"] >= 2, summary

    _assert_quantizer_amax_is_static(summary)


def test_configure_vllm_attention_quantizers_fp8_bmm2(monkeypatch):
    monkeypatch.setattr(
        vllm_plugin,
        "create_parallel_state",
        lambda: vllm_plugin.ParallelState(data_parallel_group=None),
    )
    attention = object.__new__(vllm_plugin.vllm_attention.Attention)
    torch.nn.Module.__init__(attention)

    converted = configure_vllm_nvfp4_attention_quantizers(
        attention,
        device="cpu",
        dtype=torch.bfloat16,
        cfg=build_vllm_attention_quant_cfg(p_format="fp8", v_format="fp8"),
    )

    # BMM1 unchanged: Q/K dynamic block-16 NVFP4 (F1)
    for name in ("q", "k"):
        quantizer = getattr(converted, f"{name}_bmm_quantizer")
        assert quantizer.is_enabled and quantizer.is_nvfp4_dynamic
        assert quantizer.block_sizes[-1] == 16
    assert converted.k_bmm_quantizer._amax == 6.0 * 448.0
    # BMM2: P/V per-tensor FP8 E4M3 with fixed amax (P=1.0, V=448) (F3)
    for name, amax in (("p", 1.0), ("v", 448.0)):
        quantizer = getattr(converted, f"{name}_bmm_quantizer")
        assert quantizer.is_enabled
        assert quantizer.num_bits == (4, 3)
        assert not quantizer.block_sizes
        assert float(quantizer._amax) == amax
    # idempotent: calibrated amax survives reconfiguration
    converted.v_bmm_quantizer.amax = torch.tensor(96.0)
    reconfigured = configure_vllm_nvfp4_attention_quantizers(
        converted,
        device="cpu",
        dtype=torch.bfloat16,
        cfg=build_vllm_attention_quant_cfg(p_format="fp8", v_format="fp8"),
    )
    assert float(reconfigured.v_bmm_quantizer._amax) == 96.0
