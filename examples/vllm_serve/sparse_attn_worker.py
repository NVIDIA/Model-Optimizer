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
import json
import os
from typing import Any

from fakequant_worker import disable_compilation
from vllm.attention.layer import Attention as VLLMAttention
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import ModelOptSparseAttentionImpl

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


def _replace_attention_impl(worker, config: dict):
    """Replace FlashAttentionImpl with ModelOptSparseAttentionImpl on all Attention layers.

    Shared by SparseAttnWorker and SparseQuantWorker.
    """
    if config["calib_config_path"]:
        cfg = _load_sparse_config(config["calib_config_path"])
    else:
        cfg = _build_sparse_config(config)

    if cfg is None:
        return

    model = worker.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()

    patched = 0
    for name, module in model.named_modules():
        if not isinstance(module, VLLMAttention):
            continue

        # Match per-layer sparse config using name-based patterns
        layer_cfg = _match_sparse_config(name, cfg)
        if layer_cfg is None or not layer_cfg.get("enable", True):
            continue

        method = layer_cfg.get("method", "triton_sparse_softmax")
        backend = layer_cfg.get("backend", "triton")
        if backend != "triton" or method not in {"triton_sparse_softmax", "triton_skip_softmax"}:
            raise ValueError(
                f"{name}: unsupported sparse config for vLLM worker "
                f"(backend={backend!r}, method={method!r}). "
                "Only backend='triton' with method='triton_sparse_softmax' or "
                "'triton_skip_softmax' is supported."
            )

        # Build per-layer sparse kwargs
        sparse_kw = {}
        sparsity_n = layer_cfg.get("sparsity_n", 0)
        if sparsity_n > 0:
            sparse_kw["sparsity_n"] = sparsity_n
            sparse_kw["sparsity_m"] = layer_cfg.get("sparsity_m", 4)
            sparse_kw["num_sink_tokens"] = layer_cfg.get("num_sink_tokens", 0)
            sparse_kw["dense_window_size"] = layer_cfg.get("dense_window_size", 1)
        threshold = layer_cfg.get("skip_softmax_threshold")
        if threshold:
            sparse_kw["skip_softmax_threshold"] = threshold

        # Replace impl and store per-layer config
        old_impl = module.impl
        new_impl = ModelOptSparseAttentionImpl(
            num_heads=old_impl.num_heads,
            head_size=old_impl.head_size,
            scale=old_impl.scale,
            num_kv_heads=old_impl.num_kv_heads,
            alibi_slopes=old_impl.alibi_slopes,
            sliding_window=None,  # overwritten below
            kv_cache_dtype=old_impl.kv_cache_dtype,
            logits_soft_cap=old_impl.logits_soft_cap,
            attn_type=old_impl.attn_type,
            kv_sharing_target_layer_name=old_impl.kv_sharing_target_layer_name,
        )
        # Copy the already-transformed sliding_window tuple directly,
        # since __init__ transforms int -> (sw-1, 0) and we can't reverse it.
        new_impl.sliding_window = old_impl.sliding_window
        # Store per-layer sparse kwargs on the impl for forward() to read
        new_impl.sparse_kw = sparse_kw
        module.impl = new_impl
        patched += 1
    print(f"[ModelOpt] Sparse attention: replaced impl on {patched} attention layers")


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
        _replace_attention_impl(self, sparse_config)


class SparseQuantWorker(BaseWorker):
    """vLLM worker that applies quantization + sparse attention.

    Quantization uses the standard registry-based ``mtq.quantize()``.
    Sparse attention replaces FlashAttentionImpl with ModelOptSparseAttentionImpl
    (same approach as SparseAttnWorker).
    """

    def load_model(self, *args, **kwargs) -> None:
        """Load model, then replace attention impl with sparse variant."""
        super().load_model(*args, **kwargs)
        _replace_attention_impl(self, sparse_config)

    def compile_or_warm_up_model(self) -> None:
        """Apply quantization before warm-up."""
        from fakequant_worker import _fakequant_run_prolog_worker, quant_config

        model = self.model_runner.model
        if hasattr(model, "unwrap"):
            model = model.unwrap()

        with disable_compilation(model):
            if quant_config["quant_cfg"] or quant_config["kv_quant_cfg"]:
                _fakequant_run_prolog_worker(self)

        super().compile_or_warm_up_model()
