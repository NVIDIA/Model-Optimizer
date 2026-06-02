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

import pytest
from _test_utils.torch.transformers_models import (
    create_tiny_deepseek_v3_dir,
    create_tiny_llama_dir,
    create_tiny_qwen3_moe_dir,
)
from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.plugins.vllm import (
    _ATTENTION_TYPES,
    VllmMLAAttention,
    _QuantFusedMoEBase,
    _VLLMParallelLinear,
    disable_compilation,
)


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
