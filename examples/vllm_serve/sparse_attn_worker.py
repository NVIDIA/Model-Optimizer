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

"""Custom vLLM workers for sparse attention.

``SparseAttnWorker``: Replaces ``FlashAttentionImpl`` with
``ModelOptSparseAttentionImpl`` on each Attention module after model loading.
The sparse impl uses the ModelOpt Triton kernel for both prefill and decode.

``SparseQuantWorker``: Applies quantization first, then sparse attention via
direct module walk (registry stacking does not work due to ``_DMRegistryCls``
forward identity check).

Usage:
    SPARSE_ATTN_CFG=SPARSE_SOFTMAX_DEFAULT python vllm_serve_sparse_attn.py \\
        meta-llama/Llama-3.1-8B --enforce-eager
"""

import fnmatch
import functools
import json
import os
from typing import Any

import torch
from fakequant_worker import disable_compilation
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.kernels.triton_fa import attention as triton_attention

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

sparse_config: dict[str, Any] = {
    "sparse_cfg": os.environ.get("SPARSE_ATTN_CFG", None),
    "calib_config_path": os.environ.get("SPARSE_CALIB_CONFIG_PATH", None),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


_DEFAULT_SPARSE_CFG = {
    "sparse_cfg": {
        "*attn*": {
            "sparsity_n": 2,
            "sparsity_m": 4,
            "num_sink_tokens": 0,
            "dense_window_size": 1,
            "enable": True,
        },
        "default": {"enable": False},
    },
}


def _build_sparse_config(env_config: dict[str, Any]) -> dict | None:
    """Build sparse_cfg dict from env vars."""
    cfg_name = env_config["sparse_cfg"]
    if cfg_name is None:
        return None
    # Try looking up preset from mtsa, fall back to default
    cfg = getattr(mtsa, cfg_name, None)
    if cfg is not None:
        return cfg
    # Use built-in default if name matches
    if cfg_name in ("SPARSE_SOFTMAX_DEFAULT", "default"):
        return _DEFAULT_SPARSE_CFG
    raise ValueError(
        f"Unknown sparse config: {cfg_name}. Set SPARSE_ATTN_CFG to 'default' or a valid preset name."
    )


def _load_sparse_config(path: str) -> dict:
    """Load offline calibration config JSON."""
    with open(path) as f:
        calib_cfg = json.load(f)

    sparse_cfg = {}
    for pattern, layer_cfg in calib_cfg.items():
        if pattern == "calibration":
            sparse_cfg[pattern] = layer_cfg
            continue
        layer_cfg.setdefault("method", "triton_sparse_softmax")
        layer_cfg.setdefault("backend", "triton")
        layer_cfg.setdefault("enable", True)
        sparse_cfg[pattern] = layer_cfg
    sparse_cfg["default"] = {"enable": False}

    return {"sparse_cfg": sparse_cfg}


def _match_sparse_config(module_name: str, sparse_cfg: dict) -> dict | None:
    """Match a module name against sparse_cfg patterns."""
    cfg = sparse_cfg.get("sparse_cfg", sparse_cfg)
    for pattern, layer_cfg in cfg.items():
        if pattern in ("default", "calibration"):
            continue
        if fnmatch.fnmatch(module_name, pattern):
            return layer_cfg
    return None


def _sparse_attention_forward(module, query, key, value, kv_cache, attn_metadata, **kwargs):
    """Sparse attention forward — used by SparseQuantWorker for direct module patching."""
    if not getattr(module, "_sparse_enabled", False):
        return module._original_forward(query, key, value, kv_cache, attn_metadata, **kwargs)

    from vllm._custom_ops import reshape_and_cache_flash

    reshape_and_cache_flash(
        key,
        value,
        kv_cache,
        attn_metadata.slot_mapping,
        module.impl.kv_cache_dtype,
        getattr(module.impl, "k_scale", 1.0),
        getattr(module.impl, "v_scale", 1.0),
    )

    # Unpack paged KV cache
    k_cache = kv_cache[:, 0]  # [num_blocks, page_size, num_kv_heads, head_dim]
    v_cache = kv_cache[:, 1]
    page_size = k_cache.shape[1]

    output = torch.empty_like(query)
    sm_scale = module.impl.scale
    sparse_kw = module._sparse_kw

    # Paged KV kwargs
    paged_kw = {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "page_size": page_size,
    }

    if attn_metadata.num_prefill_tokens > 0:
        pm = attn_metadata.prefill
        n = attn_metadata.num_prefill_tokens
        output[:n] = triton_attention(
            q=query[:n],
            k=query[:0],  # dummy, not used in paged mode
            v=query[:0],
            b_start_loc=pm.query_start_loc,
            b_seq_len=pm.seq_lens_q,
            max_input_len=int(pm.seq_lens_q.max().item()),
            is_causal=True,
            softmax_scale=sm_scale,
            b_seq_len_k=pm.seq_lens,
            max_input_len_k=int(pm.seq_lens.max().item()),
            block_table=pm.block_tables,
            **paged_kw,
            **sparse_kw,
        )

    if attn_metadata.num_decode_tokens > 0:
        dm = attn_metadata.decode
        offset = attn_metadata.num_prefill_tokens
        nd = attn_metadata.num_decode_tokens
        output[offset : offset + nd] = triton_attention(
            q=query[offset : offset + nd],
            k=query[:0],  # dummy, not used in paged mode
            v=query[:0],
            b_start_loc=dm.query_start_loc,
            b_seq_len=torch.ones(nd, dtype=torch.int32, device=query.device),
            max_input_len=1,
            is_causal=True,
            softmax_scale=sm_scale,
            b_seq_len_k=dm.seq_lens,
            max_input_len_k=int(dm.seq_lens.max().item()),
            block_table=dm.block_tables,
            **paged_kw,
            **sparse_kw,
        )

    return output


def _apply_sparse_to_attention_modules(model, sparse_cfg: dict):
    """Walk model modules, patch attention layers with sparse forward.

    Used by SparseQuantWorker where registry-based mtsa.sparsify() cannot
    find already-quantized attention modules (forward identity check fails).
    """
    from vllm.attention.layer import Attention as VLLMAttention

    for name, module in model.named_modules():
        if not isinstance(module, VLLMAttention):
            continue

        layer_cfg = _match_sparse_config(name, sparse_cfg)
        if layer_cfg is None or not layer_cfg.get("enable", True):
            continue

        # Build kernel kwargs from layer config
        sparse_kw = {}
        sparsity_n = layer_cfg.get("sparsity_n", 0)
        if sparsity_n > 0:
            sparse_kw["sparsity_n"] = sparsity_n
            sparse_kw["sparsity_m"] = layer_cfg.get("sparsity_m", 4)
            sparse_kw["num_sink_tokens"] = layer_cfg.get("num_sink_tokens", 0)
            sparse_kw["dense_window_size"] = layer_cfg.get("dense_window_size", 1)
        threshold = layer_cfg.get("skip_softmax_threshold", None)
        if threshold:
            sparse_kw["skip_softmax_threshold"] = threshold

        module._sparse_enabled = True
        module._sparse_kw = sparse_kw

        original_forward = module.forward
        module._original_forward = original_forward
        module.forward = functools.partial(_sparse_attention_forward, module)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


class SparseAttnWorker(BaseWorker):
    """vLLM worker that uses the ModelOpt sparse attention backend.

    Replaces FlashAttentionImpl with ModelOptSparseAttentionImpl on each
    Attention module right after model loading — before any forward pass
    (including determine_available_memory profiling).
    """

    def load_model(self, *args, **kwargs) -> None:
        """Load model, then replace attention impl with sparse variant."""
        super().load_model(*args, **kwargs)

        if sparse_config["calib_config_path"]:
            cfg = _load_sparse_config(sparse_config["calib_config_path"])
        else:
            cfg = _build_sparse_config(sparse_config)

        if cfg is None:
            return

        from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
            ModelOptSparseAttentionImpl,
            set_sparse_config,
        )

        set_sparse_config(cfg)

        from vllm.attention.layer import Attention as VLLMAttention

        model = self.model_runner.model
        if hasattr(model, "unwrap"):
            model = model.unwrap()

        patched = 0
        for name, module in model.named_modules():
            if isinstance(module, VLLMAttention):
                old_impl = module.impl
                module.impl = ModelOptSparseAttentionImpl(
                    num_heads=old_impl.num_heads,
                    head_size=old_impl.head_size,
                    scale=old_impl.scale,
                    num_kv_heads=old_impl.num_kv_heads,
                    alibi_slopes=old_impl.alibi_slopes,
                    sliding_window=None,
                    kv_cache_dtype=old_impl.kv_cache_dtype,
                    logits_soft_cap=old_impl.logits_soft_cap,
                    attn_type=old_impl.attn_type,
                    kv_sharing_target_layer_name=old_impl.kv_sharing_target_layer_name,
                )
                patched += 1
        print(f"[ModelOpt] Sparse attention: replaced impl on {patched} attention layers")


class SparseQuantWorker(BaseWorker):
    """vLLM worker that applies quantization + sparse attention.

    Quantization uses the standard registry-based ``mtq.quantize()``.
    Sparse attention uses direct module walk because the registry cannot
    match already-quantized attention modules (forward identity check).
    """

    def compile_or_warm_up_model(self) -> None:
        """Apply quantization then sparse attention before warm-up."""
        from .fakequant_worker import _fakequant_run_prolog_worker, quant_config

        model = self.model_runner.model
        if hasattr(model, "unwrap"):
            model = model.unwrap()

        with disable_compilation(model):
            # Step 1: Quantize
            if quant_config["quant_cfg"] or quant_config["kv_quant_cfg"]:
                _fakequant_run_prolog_worker(self)

            # Step 2: Apply sparse attention via direct module walk
            if sparse_config["calib_config_path"]:
                cfg = _load_sparse_config(sparse_config["calib_config_path"])
            elif sparse_config["sparse_cfg"]:
                cfg = _build_sparse_config(sparse_config)
            else:
                cfg = None

            if cfg is not None:
                _apply_sparse_to_attention_modules(model, cfg)

        super().compile_or_warm_up_model()
