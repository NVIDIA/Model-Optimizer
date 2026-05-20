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

Each test:

1. Saves a tiny HF model to disk via the helpers in ``_test_utils.torch.transformers_models``.
2. Boots ``vllm.LLM(model=tiny_dir, enforce_eager=True, …)`` — vLLM constructs all of
   its parallel linears / FusedMoE / Attention modules during model load.
3. Uses ``LLM.collective_rpc(...)`` to run ``mtq.quantize`` (with a real
   engine-mediated forward loop) on the loaded model inside the worker process,
   plus a structural assertion about which ``_QuantVLLM…`` classes are now
   present and a regression guard that every enabled quantizer ends up with a
   registered tensor-level ``_amax`` (i.e. amax is *static*, not recomputed
   per forward).

This exercises the same code path that ``examples/vllm_serve/fakequant_worker.py``
uses in production, so it covers the registry walk, the
``vllm_replace_quant_module_hook`` (which stamps device/dtype on Attention
modules before conversion), and ``create_parallel_state`` against a fully
initialized vLLM distributed environment.

Architectures covered:

- **TinyLlama** → ``QKVParallelLinear``, ``RowParallelLinear``,
  ``MergedColumnParallelLinear``, and ``Attention``.
- **TinyQwen3MoE** → adds ``FusedMoE``.
- **TinyDeepseekV3** → adds ``MLAAttention``.
"""

from __future__ import annotations

import gc

import pytest
import vllm.model_executor.layers.fused_moe.layer as vllm_fused_moe_layer
import vllm.model_executor.layers.linear as vllm_linear
from _test_utils.torch.transformers_models import (
    create_tiny_deepseek_v3_dir,
    create_tiny_llama_dir,
    create_tiny_qwen3_moe_dir,
)
from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.plugins.vllm import (
    _ATTENTION_TYPES,
    VllmMLAAttention,
    _QuantFusedMoEBase,
    _VLLMParallelLinear,
    disable_compilation,
)

# Sizes picked so vLLM accepts the head_size (must be supported by the chosen
# attention backend). head_size=64 with num_heads=2 is broadly supported.
_LLAMA_OVERRIDES = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "max_position_embeddings": 128,
    "vocab_size": 128,
    "head_dim": 64,
}

_QWEN3_MOE_OVERRIDES = {
    **_LLAMA_OVERRIDES,
    "moe_intermediate_size": 64,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
}


def _quantize_and_summarize(self):
    """Run on the worker via ``LLM.collective_rpc``.

    Must be a module-level function so it survives pickle when shipped over the
    engine-core IPC. ``self`` is the vLLM worker — we need it to drive the
    engine-mediated forward (``self.model_runner._dummy_run``) during
    calibration; ``apply_model`` only hands us the model and that's not enough
    to push activations through the full forward path.

    Steps performed against the loaded vLLM model:

    1. ``mtq.quantize`` with a static-amax recipe (NVFP4 is the canary — that
       is where dynamic-amax regressions have actually surfaced — but the
       invariant we check holds for every PTQ recipe that uses calibration:
       after calibration, every enabled quantizer must have its tensor-level
       ``_amax`` registered as a buffer, not recomputed per forward).
    2. Walk the model and record per-architecture coverage plus the
       static-amax property.

    Returns a small JSON-able summary so the parent process can assert without
    shipping tensors back.
    """
    model = self.get_model()

    def _forward_loop(_model):
        # Engine-mediated forward; same path ``examples/vllm_serve/fakequant_worker.py``
        # uses for warmup. ``num_tokens=1`` is enough to flow activations through every
        # quantizer once, which is all the ``"max"`` calibrator needs.
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

        # Static-amax invariant: after calibration, every enabled quantizer
        # must own an ``_amax`` buffer. Missing ``_amax`` means the quantizer
        # falls back to per-forward dynamic computation (the repr shows
        # ``amax=dynamic``) — the regression this test guards against.
        # ``kv_b_proj`` is exempt: vLLM's MLA decode path absorbs the linear
        # (reads ``kv_b_proj.weight`` directly, never calls its forward), so
        # neither quantizer sees data via ``_dummy_run``. Tracked separately.
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

    ``**extra`` overrides defaults — used by MoE-flavored fixtures to enable
    expert parallelism so vLLM selects an MoE backend whose kernel dispatch
    still flows through ``vllm.model_executor.layers.fused_moe.fused_moe``'s
    module-level entries (the seam the modelopt vLLM plugin patches to run
    calibration through the ``w13/w2`` quantizers). Without EP, vLLM ≥ 0.21
    picks ``MoEPrepareAndFinalizeNoDPEPModular`` which bypasses those entries.
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
    model_dir = create_tiny_llama_dir(tmp, **_LLAMA_OVERRIDES)
    llm = _boot_llm(model_dir)
    try:
        yield llm
    finally:
        _shutdown_llm(llm)


@pytest.fixture(scope="module")
def tiny_qwen3_moe_llm(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiny_qwen3_moe")
    model_dir = create_tiny_qwen3_moe_dir(tmp, **_QWEN3_MOE_OVERRIDES)
    llm = _boot_llm(model_dir, enable_expert_parallel=True)
    try:
        yield llm
    finally:
        _shutdown_llm(llm)


@pytest.fixture(scope="module")
def tiny_deepseek_llm(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiny_deepseek")
    model_dir = create_tiny_deepseek_v3_dir(tmp)
    llm = _boot_llm(model_dir, enable_expert_parallel=True)
    try:
        yield llm
    finally:
        _shutdown_llm(llm)


def _assert_quantizer_amax_is_static(summary):
    """Regression guard for the general PTQ invariant: after calibration every
    enabled ``TensorQuantizer`` must own a registered ``_amax`` buffer (the
    tensor-level scale is static). Quantizers without ``_amax`` fall back to
    per-forward dynamic computation — that is the regression this guards
    against; some vLLM versions have left enabled quantizers in that state.
    Listing the offending module paths in the failure message makes the
    triage obvious.
    """
    assert summary["enabled_quantizer_count"] > 0, summary
    assert summary["quantizers_without_amax"] == [], summary["quantizers_without_amax"]


def test_tiny_llama_quantize(tiny_llama_llm):
    """A tiny Llama loaded into vLLM has its parallel linears + Attention layers
    upgraded by ``mtq.replace_quant_module``.

    Expected coverage for a Llama-shaped model:
      - QKVParallelLinear (q/k/v as one fused projection)
      - RowParallelLinear (o_proj, down_proj)
      - MergedColumnParallelLinear (gate_up_proj)
      - Attention (one per layer)
    """
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


def test_tiny_qwen3_moe_quantize_via_vllm(tiny_qwen3_moe_llm):
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


@pytest.mark.parametrize(
    "vllm_cls",
    [
        vllm_linear.RowParallelLinear,
        vllm_linear.ColumnParallelLinear,
        vllm_linear.MergedColumnParallelLinear,
        vllm_linear.QKVParallelLinear,
        vllm_fused_moe_layer.FusedMoE,
    ],
)
def test_registry_registration(vllm_cls):
    """Every quantizable vLLM class has a registered QuantModule mapping.

    Pure registry check — no GPU / engine boot — so it runs even when the
    heavier fixtures are skipped.
    """
    assert vllm_cls in QuantModuleRegistry


def test_registry_has_mla_attention():
    assert VllmMLAAttention is not None
    assert VllmMLAAttention in QuantModuleRegistry
