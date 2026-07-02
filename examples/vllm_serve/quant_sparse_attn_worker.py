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

Install is a fail-loud, two-pass transaction:

1. **Preflight** (:func:`_preflight_quant_sparse_attn`) is a pure resolver: it validates
   the engine-wide and per-layer support matrix and returns one install record per
   decoder self-attention layer to transform. It never converts a module or assigns an
   impl. Any unsupported configuration aggregates into a single ``NotImplementedError``
   raised at startup with layer-qualified reasons — no requested quant or sparsity
   transform is ever silently dropped.
2. **Attach quant** makes the attention layers carry ``q/k/v/p_bmm_quantizer`` config.
3. **Prepare then commit** (:func:`_prepare_quant_sparse_attn` / :func:`_commit_quant_sparse_attn`)
   constructs every replacement impl first and only assigns them once the whole model
   prepared successfully, so a failure leaves every original ``module.impl`` and
   ``_value_quant_in_kernel`` gate unchanged.

Attention-quant attach is selected by ``ATTN_QUANT_MODE``:
- ``"impl_swap"`` (default): attention-ONLY conversion (:func:`_attach_attention_quant_impl_swap`).
  It converts just the preflight-approved decoder ``Attention`` modules and never touches
  Linear/MoE, so it composes with an already-realquant checkpoint (whose Linears would
  otherwise trip the ``_VLLMParallelLinear`` ``quant_method`` assert under ``mtq.quantize``).
- ``"mtq"``: the legacy ``fakequant_worker`` restore prolog (whole-model ``replace_quant_module``);
  preflight then runs against the restored quantizers.

Supported configuration (everything else fails at startup): decoder self-attention on the
FlashAttention or FlashInfer backend, fp16/bf16 KV cache, ``dcp_world_size == 1``, page size a
multiple of 16, and no sliding window / ALiBi / logits soft-cap / sinks / prefix caching /
cross-layer KV sharing / KV connector / speculative decoding / cascade.

Usage:
    python vllm_serve_quant_sparse_attn.py <path/to/modelopt-exported-ckpt>
"""

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass

from fakequant_worker import FakeQuantWorker, _fakequant_run_prolog_worker, quant_config
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from modelopt.torch.quantization.conversion import set_quantizer_by_cfg
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.vllm import (
    _ATTENTION_TYPES,
    post_restore_vllm_attentions,
    vllm_attention,
)
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

# Reuse the concrete vLLM ``Attention`` class the quantization plugin resolved (via its
# ``_import_attention_module`` symbol-based selection) so isinstance/registry checks here match
# the class the plugin registered and the served model instantiates.
VLLMAttention = vllm_attention.Attention


def _import_attention_type():
    """Return vLLM's ``AttentionType`` enum from whichever path this version exposes."""
    for path in (
        "vllm.v1.attention.backend",
        "vllm.attention",
        "vllm.attention.backends.abstract",
    ):
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        if hasattr(module, "AttentionType"):
            return module.AttentionType
    raise ImportError("Could not import vLLM AttentionType from any supported path")


AttentionType = _import_attention_type()

# The NVFP4 group size; the paged-cache page size must be a whole multiple of it.
_NVFP4_GROUP_SIZE = 16
# vLLM attention-backend impl class names whose layout the ModelOpt sparse kernel supports.
_SUPPORTED_BACKEND_IMPLS = ("FlashAttentionImpl", "FlashInferImpl")

# Dynamic block-16 NVFP4 (num_bits (2,1), dynamic scale (4,3)); matches the KV-default test cfg.
_NVFP4_ATTN_CFG = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
}


# ---------------------------------------------------------------------------
# Capability preflight (pure: no module conversion, no impl assignment)
# ---------------------------------------------------------------------------


def _validate_global_runtime(worker) -> list[str]:
    """Return engine-wide reasons the ModelOpt attention plan cannot be served."""
    config = worker.model_runner.vllm_config
    parallel = config.parallel_config
    cache = config.cache_config
    reasons: list[str] = []

    dcp = getattr(parallel, "decode_context_parallel_size", 1)
    if dcp != 1:
        reasons.append(f"decode_context_parallel_size={dcp} (only 1 is supported)")
    if getattr(cache, "enable_prefix_caching", False):
        reasons.append(
            "prefix caching is enabled (V is baked in place, so cross-request prefix reuse "
            "would read another request's quantized V)"
        )
    if getattr(config, "kv_transfer_config", None) is not None:
        reasons.append(
            "a KV transfer/connector config is set (external KV movement is unsupported)"
        )
    if getattr(config, "speculative_config", None) is not None:
        reasons.append("speculative decoding is enabled (V is baked in place)")
    block_size = getattr(cache, "block_size", None)
    if not isinstance(block_size, int) or block_size <= 0 or block_size % _NVFP4_GROUP_SIZE:
        reasons.append(
            f"cache page size {block_size} is not a positive multiple of the NVFP4 group size "
            f"{_NVFP4_GROUP_SIZE}"
        )
    model_config = getattr(config, "model_config", None)
    if model_config is not None and not getattr(model_config, "enforce_eager", False):
        reasons.append(
            "enforce_eager is not set (CUDA-graph capture is not yet validated for the ModelOpt "
            "attention path; launch with --enforce-eager)"
        )
    return reasons


def _unsupported_attention_reasons(module) -> list[str]:
    """Return per-layer reasons this attention module cannot run the ModelOpt kernel.

    Pure: inspects the module and its backend impl by name/attribute only. Backend support is
    checked by impl class name so this does not trigger the FlashInfer metadata-builder patch
    (that side effect is deferred to :func:`_prepare_quant_sparse_attn`).
    """
    reasons: list[str] = []

    # Regular Q/K/V paged attention only. MLA is a distinct class (not an ``Attention`` subclass);
    # cross/encoder-only subclass ``Attention`` but are caught by the attn_type check below. MLA has
    # ``attn_type == DECODER``, so it would slip past that check -- reject it by layout here.
    if not isinstance(module, VLLMAttention):
        reasons.append(
            f"attention layout {type(module).__name__} is unsupported "
            "(needs regular decoder self-attention with a Q/K/V paged cache; MLA is not supported)"
        )

    attn_type = getattr(module, "attn_type", AttentionType.DECODER)
    if attn_type != AttentionType.DECODER:
        reasons.append(f"attn_type={attn_type} (only DECODER self-attention is supported)")

    head_size = getattr(module, "head_size", None)
    if isinstance(head_size, int) and head_size % _NVFP4_GROUP_SIZE != 0:
        reasons.append(
            f"head_size={head_size} is not a multiple of the NVFP4 block size {_NVFP4_GROUP_SIZE} "
            "(Q/K are block-quantized on the flattened head axis, so a block would span two heads)"
        )

    impl = getattr(module, "impl", None)
    impl_name = type(impl).__name__ if impl is not None else "None"
    if impl_name not in _SUPPORTED_BACKEND_IMPLS:
        reasons.append(
            f"attention backend {impl_name} is not supported "
            f"(need one of {', '.join(_SUPPORTED_BACKEND_IMPLS)})"
        )
    if getattr(module, "sliding_window", None) is not None:
        reasons.append("sliding_window is set")
    if getattr(module, "kv_sharing_target_layer_name", None) is not None:
        reasons.append("kv_sharing_target_layer_name is set (cross-layer KV sharing)")
    kv_cache_dtype = str(getattr(module, "kv_cache_dtype", "") or "")
    if kv_cache_dtype.startswith("fp8"):
        reasons.append(f"kv_cache_dtype={kv_cache_dtype!r} (only fp16/bf16 KV cache is supported)")
    if getattr(impl, "alibi_slopes", None) is not None:
        reasons.append("alibi_slopes are set")
    logits_soft_cap = getattr(impl, "logits_soft_cap", None)
    if logits_soft_cap:  # None or 0 == disabled
        reasons.append(f"logits_soft_cap={logits_soft_cap} is set")
    if getattr(impl, "sinks", None) is not None or getattr(impl, "has_sinks", False):
        reasons.append("attention sinks are set")

    # An enabled P/V BMM2 quantizer whose format the kernel cannot map must fail loud: the
    # kernel would skip it and (for V) ``_value_quant_in_kernel`` would skip the pre-step too,
    # leaving the operand silently un-quantized. (No-op pre-conversion under impl_swap, where the
    # quantizers do not exist yet; re-checked with layer context in prepare.)
    if getattr(getattr(module, "p_bmm_quantizer", None), "is_enabled", False) and (
        _p_qdq_from_layer(module)[0] is None
    ):
        reasons.append(
            "p_bmm_quantizer is enabled but its quant format is unsupported "
            "(supported: per-tensor FP8, block-16 dynamic NVFP4)"
        )
    if getattr(getattr(module, "v_bmm_quantizer", None), "is_enabled", False) and (
        _v_qdq_from_layer(module)[0] is None
    ):
        reasons.append(
            "v_bmm_quantizer is enabled but its quant format is unsupported "
            "(supported: per-tensor FP8, block-16 dynamic NVFP4)"
        )
    return reasons


def _attention_quant_requested(module, quant_will_be_configured: bool) -> bool:
    """Whether attention quant is (or will be) active on this layer.

    ``quant_will_be_configured`` is True for the impl-swap attach (which configures the fixed
    NVFP4 recipe on every decoder self-attention layer) and False for the legacy mtq path (quant
    is active where a P/V BMM2 quantizer is enabled after restore — mappable or not; an unmappable
    one is reported as a support error rather than skipped).
    """
    if quant_will_be_configured:
        return getattr(module, "attn_type", AttentionType.DECODER) == AttentionType.DECODER
    return any(
        getattr(getattr(module, attr, None), "is_enabled", False)
        for attr in ("p_bmm_quantizer", "v_bmm_quantizer")
    )


@dataclass(frozen=True)
class _AttentionInstallRecord:
    """One decoder attention layer the ModelOpt plan will transform (resolved, immutable)."""

    name: str
    sparse_kw: Mapping[str, object]


def _load_sparse_config(worker):
    hf_config = getattr(worker.model_runner.model_config, "hf_config", None)
    detected = load_from_checkpoint_metadata(hf_config)
    return detected if detected is not None else (None, None)


def _sparse_kw_for_layer(name: str, cfg) -> dict:
    if cfg is None:
        return {}
    layer_cfg = match_sparse_config(name, cfg)
    if layer_cfg is not None and layer_cfg.get("enable", True):
        return _build_sparse_kw(layer_cfg)
    return {}


def _unwrapped_model(worker):
    model = worker.model_runner.model
    return model.unwrap() if hasattr(model, "unwrap") else model


def _named_vllm_attentions(model):
    # Enumerate ALL attention layouts (regular + MLA/cross/encoder), not just regular ``Attention``,
    # so an unsupported layout (e.g. an all-MLA model) is surfaced and rejected by preflight rather
    # than yielding an empty plan that serves with no requested transform.
    for name, module in model.named_modules():
        if isinstance(module, _ATTENTION_TYPES):
            yield name, module


def _preflight_quant_sparse_attn(worker, quant_will_be_configured: bool):
    """Resolve the per-layer install plan and validate the full support matrix.

    Pure resolver: inspects model/engine state but does not convert modules, assign
    ``module.impl``, or flip ``_value_quant_in_kernel``. Aggregates every global and per-layer
    problem and raises a single ``NotImplementedError`` before returning any records, so an
    unsupported configuration fails at startup with layer-qualified diagnostics rather than
    silently dropping a requested transform.
    """
    model = _unwrapped_model(worker)

    cfg, preset = _load_sparse_config(worker)
    if preset is not None:
        print(f"[ModelOpt] Sparse attention config: algo -> {preset}")

    errors: list[str] = list(_validate_global_runtime(worker))
    records: list[_AttentionInstallRecord] = []
    for name, module in _named_vllm_attentions(model):
        sparse_kw = _sparse_kw_for_layer(name, cfg)
        quant_requested = _attention_quant_requested(module, quant_will_be_configured)
        if not sparse_kw and not quant_requested:
            continue  # neither sparse nor quant active -> keep vLLM's native impl
        reasons = _unsupported_attention_reasons(module)
        if reasons:
            errors.extend(f"{name}: {reason}" for reason in reasons)
            continue
        records.append(_AttentionInstallRecord(name=name, sparse_kw=sparse_kw))

    if errors:
        raise NotImplementedError(
            "Unsupported ModelOpt attention plan (no transform will be silently dropped):\n  - "
            + "\n  - ".join(errors)
        )
    return tuple(records)


# ---------------------------------------------------------------------------
# Attach attention quant (impl-swap): convert only preflight-approved layers
# ---------------------------------------------------------------------------


def _attach_attention_quant_impl_swap(model, allowed_names) -> None:
    """Attach NVFP4 attention quant by converting ONLY the preflight-approved decoder layers.

    Why not ``mtq.quantize`` / ``_fakequant_run_prolog_worker``: that path calls
    ``replace_quant_module``, which walks the whole model and wraps every registered module type,
    including ``RowParallelLinear`` / ``ColumnParallelLinear`` / ``QKVParallelLinear`` / ``FusedMoE``.
    On an already-**realquant** checkpoint those Linears carry a real quant method (e.g.
    ``ModelOptFp8LinearMethod``), and ``_VLLMParallelLinear._setup`` asserts
    ``type(self.quant_method) is UnquantizedLinearMethod`` -> ``AssertionError`` at serve time.

    This attach is attention-ONLY: it converts each approved vLLM ``Attention`` in place to
    ``_QuantVLLMAttention`` (adding the ``q/k/v/p_bmm_quantizer`` sub-quantizers) and never touches
    any Linear/MoE module, so it composes cleanly with a realquant checkpoint. It then configures
    the four BMM quantizers to dynamic block-16 NVFP4 via ``set_quantizer_by_cfg`` (deny-all first,
    then enable+configure the targets) and applies the K/V global-scale-1.0 default via
    ``post_restore_vllm_attentions``. ``_install_quant_sparse_attn`` must run AFTER this to read the
    quantizers and install ``ModelOptSparseAttentionImpl``.

    ``allowed_names`` is the set of ``model.named_modules`` names preflight approved for
    conversion; every other module (including any non-decoder attention) is left untouched.
    """
    if hasattr(model, "unwrap"):
        model = model.unwrap()
    allowed = set(allowed_names)

    def _convert_approved(parent, prefix) -> None:
        for name, child in parent.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if (
                isinstance(child, _ATTENTION_TYPES)
                and type(child) in QuantModuleRegistry
                and full in allowed
            ):
                # Convert on the parent (mirrors conversion.py:_replace_quant_module). Do NOT
                # convert any non-attention module -- that is what avoids the Linear/MoE assert.
                setattr(parent, name, QuantModuleRegistry.convert(child))
            # Recurse into whichever module now lives at parent.name so nested modules are covered.
            _convert_approved(getattr(parent, name), full)

    _convert_approved(model, "")

    set_quantizer_by_cfg(
        model,
        [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*q_bmm_quantizer", "cfg": _NVFP4_ATTN_CFG, "enable": True},
            {"quantizer_name": "*k_bmm_quantizer", "cfg": _NVFP4_ATTN_CFG, "enable": True},
            {"quantizer_name": "*v_bmm_quantizer", "cfg": _NVFP4_ATTN_CFG, "enable": True},
            {"quantizer_name": "*p_bmm_quantizer", "cfg": _NVFP4_ATTN_CFG, "enable": True},
        ],
    )

    # Apply the K/V global-scale-1.0 default to layers with no calibrated _amax (dynamic Q/P skip it).
    post_restore_vllm_attentions(model)


# ---------------------------------------------------------------------------
# Two-pass install: prepare all replacements, then commit atomically
# ---------------------------------------------------------------------------


@dataclass
class _PreparedInstall:
    """A constructed-but-not-yet-assigned sparse impl for one attention layer."""

    module: object
    new_impl: object
    quant_active: bool
    value_in_kernel: bool


def _prepare_quant_sparse_attn(worker, records) -> list[_PreparedInstall]:
    """Construct every replacement impl WITHOUT assigning it (all-or-nothing).

    Reads the live (converted/restored) module for each record, validates its BMM2 quant format
    with layer context, resolves the backend sparse impl (patching the FlashInfer metadata builder
    here, not in the pure preflight), and clones it. A failure — unsupported format, unclonable
    impl (e.g. FlashAttention sinks) — raises before any assignment, leaving every original
    ``module.impl`` and ``_value_quant_in_kernel`` gate unchanged.
    """
    modules = dict(_unwrapped_model(worker).named_modules())

    prepared: list[_PreparedInstall] = []
    for record in records:
        module = modules[record.name]
        # Fail loud on an enabled BMM2 quantizer the kernel cannot map (layer-qualified). Not
        # caught here -- an unsupported quant format is a startup error, never a silent drop.
        _assert_bmm_quant_supported(module, record.name)
        p_qdq, _ = _p_qdq_from_layer(module)
        v_qdq, _ = _v_qdq_from_layer(module)

        new_cls = select_sparse_impl_cls(module.impl)
        if new_cls is None:
            raise NotImplementedError(
                f"{record.name}: attention backend {type(module.impl).__name__} has no supported "
                "ModelOpt sparse impl (preflight should have rejected this)."
            )
        new_impl = _clone_sparse_impl(module.impl, new_cls)
        new_impl.sparse_kw = dict(record.sparse_kw)
        prepared.append(
            _PreparedInstall(
                module=module,
                new_impl=new_impl,
                quant_active=p_qdq is not None or v_qdq is not None,
                # Let the kernel own V quant only when V maps to a supported in-kernel format;
                # gating on quant_active would skip the head_dim pre-step even for unquantized V.
                value_in_kernel=v_qdq is not None and hasattr(module, "_value_quant_in_kernel"),
            )
        )
    return prepared


def _commit_quant_sparse_attn(prepared) -> None:
    """Assign every prepared impl, then transfer in-kernel V ownership.

    Pure assignment (cannot fail): after prepare succeeds for the whole model, install all impls,
    then flip ``_value_quant_in_kernel`` only on V-quantized layers so the head_dim V pre-step is
    skipped in favor of keys-axis in-kernel V quant (avoiding a double-quant of V).
    """
    for item in prepared:
        item.module.impl = item.new_impl
    for item in prepared:
        if item.value_in_kernel:
            item.module._value_quant_in_kernel = True


def _install_quant_sparse_attn(worker, records) -> None:
    """Two-pass, fail-loud install of ``ModelOptSparseAttentionImpl`` on the plan's layers."""
    prepared = _prepare_quant_sparse_attn(worker, records)
    _commit_quant_sparse_attn(prepared)

    quant_layers = sum(1 for item in prepared if item.quant_active)
    sparse_only = len(prepared) - quant_layers
    print(
        f"[ModelOpt] Quant+sparse attention: installed sparse impl on {len(prepared)} layers "
        f"({quant_layers} quant-active, {sparse_only} sparse-only)"
    )

    if prepared:
        # vLLM cascade attention routes shared-prefix batches through native attention, silently
        # dropping the ModelOpt transform — the NVFP4 P/V quant AND the skip-softmax sparsity. Disable
        # it for ANY installed plan (quant- or sparse-only) so every request goes through the ModelOpt
        # kernel. No-op where cascade is already off (e.g. vLLM 0.22 FlashInfer hardcodes it off);
        # pairs with the fail-loud guard in the impl.
        runner = worker.model_runner
        if getattr(runner, "cascade_attn_enabled", False):
            runner.cascade_attn_enabled = False
            print(
                "[ModelOpt] Disabled vLLM cascade attention (incompatible with the ModelOpt plan)"
            )


def _resolve_attn_quant_mode() -> str:
    """Return the validated ``ATTN_QUANT_MODE`` or raise (no silent fall-through to un-quantized).

    A typo must not silently enter the legacy ``mtq`` branch, and ``mtq`` with no quant source would
    serve un-quantized -- both raise here rather than serving the wrong thing.
    """
    mode = os.environ.get("ATTN_QUANT_MODE", "impl_swap")
    if mode not in ("impl_swap", "mtq"):
        raise ValueError(
            f"ATTN_QUANT_MODE={mode!r} is invalid (expected 'impl_swap' or 'mtq'). "
            "A typo must not silently fall through to an un-quantized serve."
        )
    if mode == "mtq" and not (
        quant_config["quant_cfg"]
        or quant_config["kv_quant_cfg"]
        or quant_config["modelopt_state_path"]
        or quant_config["recipe_path"]
    ):
        raise ValueError(
            "ATTN_QUANT_MODE=mtq requires a quant source (QUANT_CFG / KV_QUANT_CFG / "
            "MODELOPT_STATE_PATH / RECIPE_PATH); none set, which would serve un-quantized. "
            "Use ATTN_QUANT_MODE=impl_swap to configure NVFP4 attention without a checkpoint."
        )
    return mode


class QuantSparseAttnWorker(FakeQuantWorker):
    """vLLM worker that restores quantization and installs the sparse attention impl.

    Inherits ``determine_available_memory`` (compilation disabled during profiling) from
    ``FakeQuantWorker`` and runs the preflight, the quant attach, and the atomic sparse-impl
    install in ``compile_or_warm_up_model`` (after memory profiling, before the warm-up forward).
    """

    def compile_or_warm_up_model(self) -> float:
        # ATTN_QUANT_MODE selects how attention quant is attached:
        # - "impl_swap" (default): attention-ONLY convert (composes with a realquant checkpoint;
        #   never touches Linear/MoE, so it avoids the _VLLMParallelLinear quant_method assert).
        #   Preflight runs on the raw decoder attention layers, then only approved layers convert.
        # - "mtq": legacy mtq.quantize restore prolog (whole-model replace_quant_module); preflight
        #   then runs against the restored quantizers.
        attn_quant_mode = _resolve_attn_quant_mode()
        if attn_quant_mode == "impl_swap":
            records = _preflight_quant_sparse_attn(self, quant_will_be_configured=True)
            _attach_attention_quant_impl_swap(
                self.model_runner.model, {record.name for record in records}
            )
        else:  # "mtq": legacy whole-model restore prolog (quant source verified above).
            _fakequant_run_prolog_worker(self)
            records = _preflight_quant_sparse_attn(self, quant_will_be_configured=False)

        # Install the sparse impl + flip the in-kernel-V gate (needs the restored quantizers).
        _install_quant_sparse_attn(self, records)

        # Base worker warm-up (skip FakeQuantWorker's prolog — already run above). Must return the
        # compilation time (seconds): vLLM V1 takes max() across TP workers.
        return BaseWorker.compile_or_warm_up_model(self)
