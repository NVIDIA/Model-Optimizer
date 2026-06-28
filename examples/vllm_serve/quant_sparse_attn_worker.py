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

"""Custom vLLM worker for combined attention quantization + sparse attention.

``QuantSparseAttnWorker`` runs ModelOpt fakequant restore **and** installs
``ModelOptSparseAttentionImpl`` on each attention layer, so a single served
checkpoint runs attention quant (Q/K quantized by the ``_QuantVLLMAttention``
pre-step, P/V quantized in-kernel) together with skip-softmax sparsity.

Ordering matters: the quant-restore prolog runs first so the attention layers
become ``_QuantVLLMAttention`` carrying the ``q/k/v/p_bmm_quantizer`` config; the
sparse impl is then installed and the ``_value_quant_in_kernel`` gate is flipped
so V is fake-quantized along the *keys* axis in-kernel by the sparse impl rather
than along head_dim by the (now-skipped) pre-step.

Configuration:
- Quantization: the same env knobs as ``fakequant_worker`` (``MODELOPT_STATE_PATH``,
  ``QUANT_CFG``, ``KV_QUANT_CFG``, ...).
- Sparsity: the checkpoint's ``sparse_attention_config`` block (as in ``sparse_attn_worker``).

Usage:
    python vllm_serve_quant_sparse_attn.py <path/to/modelopt-exported-ckpt>
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

# Reuse the env-driven quant config + restore prolog from the fakequant worker (sibling module;
# the launcher puts examples/vllm_serve on sys.path so this import resolves in each worker).
from fakequant_worker import FakeQuantWorker, _fakequant_run_prolog_worker, quant_config
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from modelopt.torch.sparsity.attention_sparsity.plugins.sparse_attn_config import (
    load_from_checkpoint_metadata,
    match_sparse_config,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    _build_sparse_kw,
    _clone_sparse_impl,
    _p_qdq_from_layer,
    _v_qdq_from_layer,
)


def _install_quant_sparse_attn(worker) -> None:
    """Install ``ModelOptSparseAttentionImpl`` on attention layers (quant + sparse together).

    Runs AFTER the quant-restore prolog. A layer gets the sparse impl when it has *either* a
    sparse feature (skip-softmax / N:M from ``sparse_attention_config``) *or* active attention
    quant (an enabled ``p/v_bmm_quantizer``) — the sparse impl is also what applies in-kernel
    P/V quant, so a quant-only layer still needs it. For quant-active layers the
    ``_value_quant_in_kernel`` gate is set so the head_dim V pre-step is skipped and V is
    fake-quantized along the keys axis in-kernel instead (avoiding a double-quant of V).
    """
    hf_config = getattr(worker.model_runner.model_config, "hf_config", None)
    detected = load_from_checkpoint_metadata(hf_config)
    cfg, preset = detected if detected is not None else (None, None)
    if preset is not None:
        print(f"[ModelOpt] Sparse attention config: algo -> {preset}")

    model = worker.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()

    patched = sparse_only = quant_layers = 0
    for name, module in model.named_modules():
        if not isinstance(module, VLLMAttention):
            continue

        # Sparse features for this layer (empty dict if no/disabled sparse config).
        sparse_kw: dict = {}
        if cfg is not None:
            layer_cfg = match_sparse_config(name, cfg)
            if layer_cfg is not None and layer_cfg.get("enable", True):
                sparse_kw = _build_sparse_kw(layer_cfg)

        # Active attention quant on this (restored) layer.
        p_qdq, _ = _p_qdq_from_layer(module)
        v_qdq, _ = _v_qdq_from_layer(module)
        quant_active = p_qdq is not None or v_qdq is not None

        if not sparse_kw and not quant_active:
            continue  # neither sparse nor quant active -> keep vLLM's native impl

        new_impl = _clone_sparse_impl(module.impl)
        new_impl.sparse_kw = sparse_kw
        module.impl = new_impl
        if quant_active and hasattr(module, "_value_quant_in_kernel"):
            module._value_quant_in_kernel = True
            quant_layers += 1
        elif not quant_active:
            sparse_only += 1
        patched += 1

    print(
        f"[ModelOpt] Quant+sparse attention: installed sparse impl on {patched} layers "
        f"({quant_layers} quant-active, {sparse_only} sparse-only)"
    )


class QuantSparseAttnWorker(FakeQuantWorker):
    """vLLM worker that restores quantization and installs the sparse attention impl.

    Inherits ``determine_available_memory`` (compilation disabled during profiling) from
    ``FakeQuantWorker`` and runs both the quant restore and the sparse-impl install in
    ``compile_or_warm_up_model`` (after memory profiling, before the warm-up forward).
    """

    def compile_or_warm_up_model(self) -> float:
        # 1) Quant-restore -> attention layers become _QuantVLLMAttention with their quantizers.
        if (
            quant_config["quant_cfg"]
            or quant_config["kv_quant_cfg"]
            or quant_config["modelopt_state_path"]
            or quant_config["recipe_path"]
        ):
            _fakequant_run_prolog_worker(self)
        # 2) Install the sparse impl + flip the in-kernel-V gate (needs the restored quantizers).
        _install_quant_sparse_attn(self)
        # 3) Base worker warm-up (skip FakeQuantWorker's prolog — already run above). Must return
        # the compilation time (seconds): vLLM V1 takes max() across TP workers.
        return BaseWorker.compile_or_warm_up_model(self)
