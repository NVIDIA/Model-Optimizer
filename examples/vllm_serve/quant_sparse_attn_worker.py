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

Ordering matters: the attention-quant attach runs first so the attention layers
become ``_QuantVLLMAttention`` carrying the ``q/k/v/p_bmm_quantizer`` config; the
sparse impl is then installed and the ``_value_quant_in_kernel`` gate is flipped
so V is fake-quantized along the *keys* axis in-kernel by the sparse impl rather
than along head_dim by the (now-skipped) pre-step.

Attention-quant attach is selected by ``ATTN_QUANT_MODE``:
- ``"impl_swap"`` (default): attention-ONLY conversion (``_attach_attention_quant_impl_swap``).
  It converts just the vLLM ``Attention`` modules and never touches Linear/MoE, so it composes
  with an already-realquant checkpoint (whose Linears would otherwise trip the
  ``_VLLMParallelLinear`` ``quant_method`` assert under ``mtq.quantize``).
- ``"mtq"``: the legacy ``fakequant_worker`` restore prolog (whole-model ``replace_quant_module``).

Configuration:
- Quantization: ``ATTN_QUANT_MODE`` (see above); in ``"mtq"`` mode the same env knobs as
  ``fakequant_worker`` (``MODELOPT_STATE_PATH``, ``QUANT_CFG``, ``KV_QUANT_CFG``, ...).
- Sparsity: the checkpoint's ``sparse_attention_config`` block (as in ``sparse_attn_worker``).

Usage:
    python vllm_serve_quant_sparse_attn.py <path/to/modelopt-exported-ckpt>
"""

import importlib
import os

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

from modelopt.torch.quantization.conversion import set_quantizer_by_cfg
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.vllm import _ATTENTION_TYPES, post_restore_vllm_attentions
from modelopt.torch.sparsity.attention_sparsity.plugins.sparse_attn_config import (
    load_from_checkpoint_metadata,
    match_sparse_config,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    _assert_bmm_quant_supported,
    _build_sparse_kw,
    _clone_sparse_impl,
    _p_qdq_from_layer,
    _v_qdq_from_layer,
    select_sparse_impl_cls,
)


def _attach_attention_quant_impl_swap(model) -> None:
    """Attach NVFP4 attention quant by converting ONLY the attention modules (no ``mtq.quantize``).

    Why not ``mtq.quantize`` / ``_fakequant_run_prolog_worker``: that path calls
    ``replace_quant_module``, which walks the whole model and wraps every registered module type,
    including ``RowParallelLinear`` / ``ColumnParallelLinear`` / ``QKVParallelLinear`` / ``FusedMoE``.
    On an already-**realquant** checkpoint those Linears carry a real quant method (e.g.
    ``ModelOptFp8LinearMethod``), and ``_VLLMParallelLinear._setup`` asserts
    ``type(self.quant_method) is UnquantizedLinearMethod`` -> ``AssertionError`` at serve time.

    This attach is attention-ONLY: it converts each vLLM ``Attention`` in place to
    ``_QuantVLLMAttention`` (adding the ``q/k/v/p_bmm_quantizer`` sub-quantizers) and never touches
    any Linear/MoE module, so it composes cleanly with a realquant checkpoint. It then configures the
    four BMM quantizers to dynamic block-16 NVFP4 via ``set_quantizer_by_cfg`` (deny-all first, then
    enable+configure the targets) and applies the K/V global-scale-1.0 default via
    ``post_restore_vllm_attentions``. ``_install_quant_sparse_attn`` must run AFTER this to read the
    quantizers and install ``ModelOptSparseAttentionImpl``.
    """
    if hasattr(model, "unwrap"):
        model = model.unwrap()

    def _convert_attention_only(parent) -> None:
        for name, child in parent.named_children():
            if isinstance(child, _ATTENTION_TYPES) and type(child) in QuantModuleRegistry:
                # Convert on the parent (mirrors conversion.py:_replace_quant_module). Do NOT
                # convert any non-attention module -- that is what avoids the Linear/MoE assert.
                setattr(parent, name, QuantModuleRegistry.convert(child))
            # Recurse into whichever module now lives at parent.name so nested modules are covered.
            _convert_attention_only(getattr(parent, name))

    _convert_attention_only(model)

    # Dynamic block-16 NVFP4 (num_bits (2,1), dynamic scale (4,3)); matches the KV-default test cfg.
    nvfp4 = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    set_quantizer_by_cfg(
        model,
        [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*q_bmm_quantizer", "cfg": nvfp4, "enable": True},
            {"quantizer_name": "*k_bmm_quantizer", "cfg": nvfp4, "enable": True},
            {"quantizer_name": "*v_bmm_quantizer", "cfg": nvfp4, "enable": True},
            {"quantizer_name": "*p_bmm_quantizer", "cfg": nvfp4, "enable": True},
        ],
    )

    # Apply the K/V global-scale-1.0 default to layers with no calibrated _amax (dynamic Q/P skip it).
    post_restore_vllm_attentions(model)


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
    skipped_backends: set[str] = set()
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
        # Fail loud on an enabled BMM2 quantizer the kernel cannot map (else it is
        # dropped silently -- the kernel skips it and, for V, _value_quant_in_kernel
        # below would skip the pre-step too). Before the `continue` so an unmapped-only
        # layer cannot slip past.
        _assert_bmm_quant_supported(module, name)
        quant_active = p_qdq is not None or v_qdq is not None

        if not sparse_kw and not quant_active:
            continue  # neither sparse nor quant active -> keep vLLM's native impl

        # Select the sparse impl for this layer's attention backend (FlashAttention
        # vs FlashInfer). For FlashInfer this also installs the metadata-builder patch
        # that exposes the dense paged metadata the Triton kernel needs; None means an
        # unsupported backend, which we leave on vLLM's native impl.
        new_cls = select_sparse_impl_cls(module.impl)
        if new_cls is None:
            skipped_backends.add(type(module.impl).__name__)
            continue
        try:
            new_impl = _clone_sparse_impl(module.impl, new_cls)
        except NotImplementedError:
            skipped_backends.add(type(module.impl).__name__)
            continue
        new_impl.sparse_kw = sparse_kw
        module.impl = new_impl
        # Only let the kernel own V quant when V maps to a supported in-kernel format.
        # _value_quant_in_kernel tells _QuantVLLMAttention.forward to skip its own
        # v_bmm_quantizer pre-step (V's keys-axis NVFP4 blocks can't be formed per token);
        # gating on quant_active instead would skip that pre-step even when V is unquantized.
        if v_qdq is not None and hasattr(module, "_value_quant_in_kernel"):
            module._value_quant_in_kernel = True
        if quant_active:
            quant_layers += 1
        else:
            sparse_only += 1
        patched += 1

    print(
        f"[ModelOpt] Quant+sparse attention: installed sparse impl on {patched} layers "
        f"({quant_layers} quant-active, {sparse_only} sparse-only)"
    )
    if skipped_backends:
        print(
            f"[ModelOpt] Quant+sparse attention: left {sorted(skipped_backends)} layers unchanged "
            "(unsupported backend — serve under FLASH_ATTN or FLASHINFER)."
        )

    if quant_layers:
        # vLLM cascade attention routes shared-prefix batches through native attention,
        # silently dropping the NVFP4 P/V quant. Disable it so every request goes through
        # the quantized ModelOpt kernel. No-op where cascade is already off (e.g. vLLM
        # 0.22 FlashInfer hardcodes it off); pairs with the fail-loud guard in the impl.
        runner = worker.model_runner
        if getattr(runner, "cascade_attn_enabled", False):
            runner.cascade_attn_enabled = False
            print("[ModelOpt] Disabled vLLM cascade attention (incompatible with attention quant)")


class QuantSparseAttnWorker(FakeQuantWorker):
    """vLLM worker that restores quantization and installs the sparse attention impl.

    Inherits ``determine_available_memory`` (compilation disabled during profiling) from
    ``FakeQuantWorker`` and runs both the quant restore and the sparse-impl install in
    ``compile_or_warm_up_model`` (after memory profiling, before the warm-up forward).
    """

    def compile_or_warm_up_model(self) -> float:
        # 1) Attach attention quant -> attention layers become _QuantVLLMAttention with quantizers.
        #    ATTN_QUANT_MODE selects how:
        #    - "impl_swap" (default): attention-ONLY convert (composes with a realquant checkpoint;
        #      never touches Linear/MoE, so it avoids the _VLLMParallelLinear quant_method assert).
        #    - "mtq": legacy mtq.quantize restore prolog (whole-model replace_quant_module).
        attn_quant_mode = os.environ.get("ATTN_QUANT_MODE", "impl_swap")
        if attn_quant_mode == "impl_swap":
            _attach_attention_quant_impl_swap(self.model_runner.model)
        elif (
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
