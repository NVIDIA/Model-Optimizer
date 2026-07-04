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

``SparseAttnWorker`` replaces the native FlashAttention or FlashInfer impl with
the matching ModelOpt adapter on each Attention module after model loading.
The sparse impl uses the ModelOpt Triton kernel for sparse prefill launches and
delegates inactive launches to the selected native backend.

Configuration flows exclusively through the loaded checkpoint's
``sparse_attention_config`` block (written by ModelOpt's HF export). If the
checkpoint has no such block, the worker logs a message and passes through
unchanged.

This worker remains sparse-only; the compact quant+sparse path uses
``quant_sparse_attn_worker.QuantSparseAttnWorker``.

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
    select_sparse_impl_cls,
)


def _replace_attention_impl(worker):
    """Install the backend-matched ModelOpt sparse impl on attention layers.

    The sole configuration source is the checkpoint's ``sparse_attention_config``
    metadata. No-op if the checkpoint has no such block.
    """
    hf_config = getattr(worker.model_runner.model_config, "hf_config", None)
    detected = load_from_checkpoint_metadata(hf_config)
    if detected is None:
        print(
            "[ModelOpt] No sparse_attention_config found in the checkpoint; "
            "skipping sparse attention. Run examples/llm_sparsity/"
            "attention_sparsity/hf_sa.py to calibrate and export a checkpoint "
            "with the config embedded."
        )
        return
    cfg, preset_name = detected
    print(f"[ModelOpt] Sparse attention config: algo -> {preset_name}")

    model = worker.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()

    plans = []
    errors = []
    for name, module in model.named_modules():
        if not isinstance(module, VLLMAttention):
            continue

        layer_cfg = match_sparse_config(name, cfg)
        if layer_cfg is None or not layer_cfg.get("enable", True):
            continue

        sparse_kw = _build_sparse_kw(layer_cfg)
        if not sparse_kw:
            # Keep vLLM's original impl when the exported layer config does not
            # enable any sparse feature.
            continue
        new_impl_cls = select_sparse_impl_cls(module.impl)
        if new_impl_cls is None:
            errors.append(f"{name or '<root>'}: unsupported backend {type(module.impl).__name__}")
            continue
        try:
            new_impl = _clone_sparse_impl(module.impl, new_impl_cls)
        except (NotImplementedError, TypeError) as err:
            errors.append(f"{name or '<root>'}: {err}")
            continue
        plans.append((module, new_impl, sparse_kw))

    if errors:
        raise NotImplementedError(
            "Unsupported ModelOpt sparse attention plan:\n  - " + "\n  - ".join(errors)
        )

    installed = {}
    for module, new_impl, sparse_kw in plans:
        new_impl.sparse_kw = sparse_kw
        module.impl = new_impl
        impl_name = type(new_impl).__name__
        installed[impl_name] = installed.get(impl_name, 0) + 1
    print(
        f"[ModelOpt] Sparse attention: replaced impl on {len(plans)} attention layers: {installed}"
    )


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


class SparseAttnWorker(BaseWorker):
    """vLLM worker that uses the ModelOpt sparse attention backend.

    Replaces FlashAttention or FlashInfer with its matching ModelOpt adapter on
    each Attention module right after model loading, before any forward pass.
    """

    def load_model(self, *args, **kwargs) -> None:
        """Load model, then replace attention impl with sparse variant."""
        super().load_model(*args, **kwargs)
        _replace_attention_impl(self)
