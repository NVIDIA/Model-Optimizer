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

"""Custom vLLM worker for sparse attention.

``SparseAttnWorker``: Replaces ``FlashAttentionImpl`` with
``ModelOptSparseAttentionImpl`` on each Attention module after model loading.
The sparse impl uses the ModelOpt Triton kernel for sparse prefill launches.
Decode-only launches and launches without active sparse work delegate back to
vLLM FlashAttention.

Configuration flows exclusively through the loaded checkpoint's
``sparse_attention_config`` block (written by ModelOpt's HF export). If the
checkpoint has no such block, the worker logs a message and passes through
unchanged.

Quantization combined with sparse attention is not handled by this worker
and will land in a follow-up PR once the combined path is tested.

Usage:
    python vllm_serve_sparse_attn.py <path/to/modelopt-exported-ckpt>
"""

import importlib

try:
    _has_legacy_attention_layer = importlib.util.find_spec("vllm.attention.layer") is not None
except (ModuleNotFoundError, ValueError):
    _has_legacy_attention_layer = False

if _has_legacy_attention_layer:
    from vllm.attention.layer import Attention as VLLMAttention
else:
    from vllm.model_executor.layers.attention import Attention as VLLMAttention

from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from modelopt.torch.sparsity.attention_sparsity.plugins.sparse_attn_config import (
    load_from_checkpoint_metadata,
    match_sparse_config,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    _build_sparse_kw,
    _clone_sparse_impl,
    parse_attn_quant_env,
    select_sparse_impl_cls,
)


def _replace_attention_impl(worker):
    """Replace the attention impl with the ModelOpt sparse impl on all Attention layers.

    Supports the FlashAttention and FlashInfer backends (the matching sparse impl
    is selected per layer). Configuration comes from the checkpoint's
    ``sparse_attention_config`` metadata and/or the ``MODELOPT_ATTN_*`` env knobs
    (NVFP4 BMMs / mixed-precision softmax / N:M sparse softmax) — the latter mirror
    the env-driven ``vllm_serve_fakequant`` flow and let one served checkpoint toggle
    configs with no re-export. No-op only if neither is present.
    """
    hf_config = getattr(worker.model_runner.model_config, "hf_config", None)
    detected = load_from_checkpoint_metadata(hf_config)
    env_q = parse_attn_quant_env()
    if detected is None and not env_q:
        print(
            "[ModelOpt] No sparse_attention_config and no MODELOPT_ATTN_* env knobs; "
            "skipping sparse attention. Run examples/llm_sparsity/attention_sparsity/"
            "hf_sa.py to export a config, or set e.g. MODELOPT_ATTN_NVFP4=q,k,p,v."
        )
        return
    cfg, preset_name = detected if detected is not None else (None, "env-attn-quant")
    print(f"[ModelOpt] Sparse attention config: algo -> {preset_name}; env knobs -> {env_q}")

    # Env N:M sparse softmax (prefill) augments the per-layer kwargs; the NVFP4 /
    # mixed-precision-softmax knobs ride on the impl as attn_quant_kw.
    env_nm = (
        {
            "sparsity_n": env_q["sparsity_n"],
            "sparsity_m": env_q["sparsity_m"],
            "dense_sink_tokens": 0,
            "dense_recent_tokens": 128,
        }
        if "sparsity_n" in env_q
        else {}
    )
    env_attn_quant = {k: env_q[k] for k in ("nvfp4", "fp16_softmax", "softmax_quant") if k in env_q}

    model = worker.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()

    patched = 0
    skipped_backends: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, VLLMAttention):
            continue

        if cfg is not None:
            layer_cfg = match_sparse_config(name, cfg)
            if layer_cfg is None or not layer_cfg.get("enable", True):
                continue
        else:
            layer_cfg = {}  # env-only: apply the env knobs to every attention layer

        sparse_kw = _build_sparse_kw(layer_cfg)
        sparse_kw.update(env_nm)  # env N:M sparse softmax augments/overrides
        if not sparse_kw and not env_attn_quant:
            # Neither metadata nor env enables any sparse/quant feature here.
            continue
        new_cls = select_sparse_impl_cls(module.impl)
        if new_cls is None:
            # Unsupported backend (not FlashAttention / FlashInfer) — leave it on
            # vLLM's native impl rather than mis-cloning into the wrong base.
            skipped_backends.add(type(module.impl).__name__)
            continue
        try:
            new_impl = _clone_sparse_impl(module.impl, new_cls)
        except NotImplementedError:
            skipped_backends.add(type(module.impl).__name__)
            continue
        new_impl.sparse_kw = sparse_kw
        new_impl.attn_quant_kw = env_attn_quant  # NVFP4 BMMs + mixed-precision softmax
        module.impl = new_impl
        patched += 1
    print(f"[ModelOpt] Sparse attention: replaced impl on {patched} attention layers")
    if skipped_backends:
        print(
            f"[ModelOpt] Sparse attention: left {sorted(skipped_backends)} layers unchanged "
            "(unsupported backend — serve under FLASH_ATTN or FLASHINFER)."
        )


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
        _replace_attention_impl(self)
