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

"""Code that export quantized Hugging Face models for deployment."""

import json
import tempfile
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    from .diffusers_utils import is_diffusers_object

    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.fsdp import FSDPModule

from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.qtensor import MXFP8QTensor, NVFP4QTensor
from modelopt.torch.quantization.qtensor.base_qtensor import QTensorWrapper
from modelopt.torch.quantization.qtensor.nvfp4_tensor import _cast_per_block_scale_to_fp8
from modelopt.torch.quantization.utils import quantizer_attr_names
from modelopt.torch.quantization.utils.core_utils import has_accelerate_offload
from modelopt.torch.utils.distributed import is_fsdp2_model

try:
    from modelopt.torch.sparsity.attention_sparsity.conversion import export_sparse_attention_config
except ImportError:
    export_sparse_attention_config = None

# Importing the built-in handlers installs their entries in the two registries.
from . import hf_export_handlers as _hf_export_handlers  # noqa: F401
from .convert_hf_config import convert_hf_quant_config_format
from .hf_export_prep import (
    _add_mtp_exclusions,
    _patch_revert_weight_conversion,
    _prepare_moe_inputs,
    _resolve_export_dtype,
    _sanitize_generation_config_for_save,
    _unpatch_revert_weight_conversion,
    _warn_on_unsynced_moe_gate_up,
    requantize_resmooth_fused_llm_layers,
)
from .model_config import (
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_REAL,
    QUANTIZATION_FP8_PC_PT,
    QUANTIZATION_MXFP8,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_NVFP4_AWQ,
    QUANTIZATION_NVFP4_SVDQUANT,
    QUANTIZATION_W4A8_AWQ,
    QUANTIZATION_W4A8_NVFP4_FP8,
    QUANTIZATION_W4A16_NVFP4,
)
from .model_utils import _reorder_canonical_first
from .plugins import SpeculativeDecodingExporter, has_spec_opt, sanitize_hf_config_for_deployment
from .quant_aware_conversion import (
    build_reverse_name_mapper,
    revert_quant_config_names,
    revert_weight_conversion_quant_aware,
)
from .quant_utils import (
    get_activation_scaling_factor,
    get_quant_config,
    get_quantization_format,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    maybe_transpose_expert_weight_dimensions,
    postprocess_state_dict,
    sync_tied_input_amax,
    to_quantized_weight,
)
from .registry import ExportContext, ExportModuleRegistry

__all__ = ["export_hf_checkpoint", "export_speculative_decoding"]


def _compressed_per_block_scale(
    weight_quantizer: TensorQuantizer, weight: QTensorWrapper
) -> torch.Tensor | None:
    """Per-block scale captured at compression time, in the modelopt E4M3 layout.

    ``NVFP4QTensor.quantize(..., try_tensorrt=True)`` returns a cutlass-swizzled 1-D uint8 scale
    when TensorRT-LLM is available on an FP4-capable device, so normalize it the way
    ``NVFP4QTensor.dequantize`` does before it is used as an exported ``weight_scale``.
    """
    scale = getattr(weight_quantizer, "_scale", None)
    if scale is None or not (scale.dtype == torch.uint8 and scale.ndim == 1):
        return scale
    try:
        from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import (
            cutlass_fp4_scale_to_modelopt_fp4_scale,
        )
    except ImportError as e:
        raise ImportError(
            "This weight was compressed by TensorRT-LLM, so its NVFP4 block scale is "
            "cutlass-swizzled, but tensorrt_llm cannot be imported to convert it for export."
        ) from e
    return cutlass_fp4_scale_to_modelopt_fp4_scale(scale, weight.metadata["shape"][-2:])


def _export_quantized_weight(
    sub_module: nn.Module,
    dtype: torch.dtype,
    weight_name: str = "weight",
    _tied_cache: dict[int, nn.Module] | None = None,
):
    """For the given weight attr of the sub_module, export the quantization info of it.

    The export includes converting weight tensor to correct quantized values and quantized dtype,
    and registering scaling factors.

    Tied-weight dedup is opt-in via ``_tied_cache``: the setattr below replaces
    ``.weight`` with a fresh ``nn.Parameter`` wrapping packed bytes, breaking
    any HF-level tie. When the caller passes a ``_tied_cache`` dict (keyed by
    the pre-pack ``weight.data_ptr()``), the alias step at the end re-points
    ``weight`` / ``weight_scale`` / ``weight_scale_2`` at a previously-processed
    module sharing the same source memory so the downstream data_ptr dedup can
    collapse them. The cache is owned by the caller (typically
    ``_export_transformers_checkpoint``) and scoped to one export invocation;
    when ``_tied_cache`` is ``None`` (the default) the alias step is skipped
    entirely. Uses memory identity only — no ``_tied_weights_keys`` lookup,
    no-op for non-tied modules.
    """
    quantization_format = get_quantization_format(sub_module)
    if quantization_format == QUANTIZATION_NONE:
        return

    block_size = get_weight_block_size(sub_module, weight_name)
    quantizer_attrs = quantizer_attr_names(weight_name)
    weight: nn.Parameter = getattr(sub_module, weight_name)

    if weight.is_meta:
        raise RuntimeError(
            f"Weight '{weight_name}' of {type(sub_module).__name__} is a meta tensor during "
            "export. If the model was loaded with disk/CPU offload, use export_hf_checkpoint() "
            "which dispatches to the streaming writer that materialises weights layer-by-layer."
        )

    # Capture source identity BEFORE any tensor-creating operation below.
    # For HF-tied weights this matches across all modules sharing the
    # underlying Parameter; the cache lookup at the end of this function
    # uses it to detect ties whose Python identity is about to be broken
    # by the setattr on `weight_name` further down.
    _tied_source_data_ptr = weight.data_ptr()
    weight_quantizer: TensorQuantizer | SequentialQuantizer = getattr(
        sub_module, quantizer_attrs.weight_quantizer
    )
    input_quantizer: TensorQuantizer | SequentialQuantizer | None = getattr(
        sub_module, quantizer_attrs.input_quantizer, None
    )
    output_quantizer: TensorQuantizer | SequentialQuantizer | None = getattr(
        sub_module, quantizer_attrs.output_quantizer, None
    )

    # Already real-quantized weights (``mtq.compress`` / ``hf_ptq --low_memory_mode``) hold packed
    # nibbles -- half the logical last dim -- so per-block scales cannot be recomputed from them.
    # Use the scale the quantizer captured at compression time instead.
    uses_compressed_nvfp4_scale = isinstance(weight, QTensorWrapper) and quantization_format in [
        QUANTIZATION_NVFP4,
        QUANTIZATION_NVFP4_AWQ,
        QUANTIZATION_NVFP4_SVDQUANT,
        QUANTIZATION_W4A16_NVFP4,
    ]
    compressed_weight_scale = (
        _compressed_per_block_scale(weight_quantizer, weight)
        if uses_compressed_nvfp4_scale
        else None
    )
    compressed_weight_scale_2 = (
        getattr(weight_quantizer, "_double_scale", None) if uses_compressed_nvfp4_scale else None
    )
    use_compressed_scale = (
        compressed_weight_scale is not None and compressed_weight_scale_2 is not None
    )

    if quantization_format == QUANTIZATION_FP8:
        # Convert amax to float32
        weight_quantizer._amax = weight_quantizer._amax.to(torch.float32)

        if weight_quantizer._amax.dim() == 1:
            # Per-tensor amax
            weight_scaling_factor = torch.tensor(
                weight_quantizer.amax.item() / weight_quantizer.maxbound
            )
        else:
            # Per-channel amax
            weight_scaling_factor = torch.tensor(weight_quantizer.amax / weight_quantizer.maxbound)

        sub_module.register_buffer(
            quantizer_attrs.weight_scale,
            weight_scaling_factor,
        )

        if hasattr(input_quantizer, "_amax"):
            assert input_quantizer is not None
            input_quantizer._amax = input_quantizer._amax.to(torch.float32)

            sub_module.register_buffer(
                quantizer_attrs.input_scale,
                get_activation_scaling_factor(
                    sub_module, input_quantizer_name=quantizer_attrs.input_quantizer
                ).squeeze(),
            )

        if hasattr(output_quantizer, "_amax"):
            assert output_quantizer is not None
            output_quantizer._amax = output_quantizer._amax.to(torch.float32)
    else:
        # Register weight_scale and input_scale
        if quantization_format == QUANTIZATION_FP8_PB_REAL:
            sub_module.register_buffer(
                quantizer_attrs.weight_scale,
                weight_quantizer._scale.to(torch.float32),
            )
            del weight_quantizer._scale
        elif quantization_format == QUANTIZATION_MXFP8:
            # MXFP8 uses dynamic block quantization with E8M0 scales (uint8)
            weight = getattr(sub_module, weight_name)
            e8m0_scale = MXFP8QTensor.get_weights_scaling_factor_from_quantizer(
                weight, weight_quantizer
            )
            sub_module.register_buffer(quantizer_attrs.weight_scale, e8m0_scale)
            if hasattr(weight_quantizer, "_scale") and weight_quantizer._scale is not None:
                del weight_quantizer._scale
        elif not use_compressed_scale:
            sub_module.register_buffer(
                quantizer_attrs.weight_scale, get_weight_scaling_factor(sub_module, weight_name)
            )

        if (
            input_quantizer is not None
            and "disabled" not in repr(input_quantizer)
            and input_quantizer.amax is not None
        ):
            sub_module.register_buffer(
                quantizer_attrs.input_scale,
                get_activation_scaling_factor(
                    sub_module, input_quantizer_name=quantizer_attrs.input_quantizer
                ).squeeze(),
            )

    if quantization_format in [
        QUANTIZATION_NVFP4_AWQ,
        QUANTIZATION_NVFP4_SVDQUANT,
        QUANTIZATION_NVFP4,
        QUANTIZATION_W4A16_NVFP4,
        QUANTIZATION_W4A8_AWQ,
        QUANTIZATION_W4A8_NVFP4_FP8,
    ]:
        # Register weight_scale_2
        sub_module.register_buffer(
            quantizer_attrs.weight_scale_2,
            get_weight_scaling_factor_2(sub_module, weight_name).squeeze(),
        )

    weight_scale: torch.Tensor | None = getattr(sub_module, quantizer_attrs.weight_scale, None)
    weight_scale_2: torch.Tensor | None = getattr(sub_module, quantizer_attrs.weight_scale_2, None)

    # Transpose weight for bmm-style expert quantization (llama4, gpt-oss)
    # Check if this is a BMM-style expert weight that needs transposition
    is_bmm_expert_weight = weight.dim() == 3 and any(
        expert_type in type(sub_module).__name__
        for expert_type in ["Llama4TextExperts", "GptOssExperts"]
    )
    # NVFP4StaticQuantizer + BMM-style experts: route through the static-aware
    # ``_from_quantizer`` helper so the pinned per-block ``_amax`` (e.g. set by
    # the MXFP4->NVFP4 cast to ``6 * 2^k_j``) is used to derive the FP8
    # per-block scale. The plain ``get_weights_scaling_factor`` would ignore
    # ``_amax`` and recompute per-block max from the BF16 weight, which
    # rebuckets nibbles and loses bit-exactness when ``max_nibble < 6``.

    if quantization_format in [
        QUANTIZATION_NVFP4,
        QUANTIZATION_NVFP4_AWQ,
        QUANTIZATION_NVFP4_SVDQUANT,
        QUANTIZATION_W4A16_NVFP4,
    ]:
        # Transpose weight from (num_experts, input_dim, output_dim) to (num_experts, output_dim, input_dim)
        # for NVFP4 quantization functions that expect input_dim as the last dimension for block quantization
        weight, _ = maybe_transpose_expert_weight_dimensions(
            weight, is_bmm_expert_weight=is_bmm_expert_weight
        )

        if use_compressed_scale and weight_scale_2 is not None:
            # Dequant is ``nibble * weight_scale * weight_scale_2``; the stored per-block scale is
            # normalized against the compression-time global scale, so rescale to keep that product.
            # The nibbles cannot be re-quantized here (the high-precision weight is gone), so once
            # ``preprocess_linear_fusion`` unifies ``weight_scale_2`` over a fused group the ratio
            # below is 1 only for the member owning the group max; the others take one extra E4M3
            # rounding (<= half-ULP, 6.25%). Avoiding that needs a shared scale at compress time.
            assert compressed_weight_scale is not None and compressed_weight_scale_2 is not None
            device = compressed_weight_scale.device
            weight_scale = _cast_per_block_scale_to_fp8(
                compressed_weight_scale.float()
                * compressed_weight_scale_2.float().to(device)
                / weight_scale_2.float().to(device)
            )
        elif NVFP4QTensor._is_static_quantizer(weight_quantizer):
            weight_scale = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
                weight_quantizer,
                weight,
                weight_scale_2,
            )[0]
        else:
            weight_scale = NVFP4QTensor.get_weights_scaling_factor(
                weight,
                block_size=block_size,
                weights_scaling_factor_2=weight_scale_2,
            )[0]

        quantized_weight = to_quantized_weight(
            weight.to(dtype),
            weight_scale,
            quantization_format,
            weight_scale_2,
            block_size,
        )

        quantized_weight, weight_scale = maybe_transpose_expert_weight_dimensions(
            quantized_weight, weight_scale, is_bmm_expert_weight=is_bmm_expert_weight
        )
    elif quantization_format == QUANTIZATION_FP8_PC_PT and is_bmm_expert_weight:
        # For FP8_PC_PT with BMM-style experts, transpose only the weight (not weight_scale)
        weight, _ = maybe_transpose_expert_weight_dimensions(
            weight, is_bmm_expert_weight=is_bmm_expert_weight
        )

        quantized_weight = to_quantized_weight(
            weight.to(dtype),
            weight_scale,
            quantization_format,
            weight_scale_2,
            block_size,
        )

        # Transpose back to original BMM format
        quantized_weight, _ = maybe_transpose_expert_weight_dimensions(
            quantized_weight, is_bmm_expert_weight=is_bmm_expert_weight
        )
    else:
        quantized_weight = to_quantized_weight(
            weight.to(dtype),
            weight_scale,
            quantization_format,
            weight_scale_2,
            block_size,
        )

    setattr(sub_module, weight_name, nn.Parameter(quantized_weight, requires_grad=False))

    # Register the corrected weight_scale as a buffer
    if weight_scale is not None:
        sub_module.register_buffer(quantizer_attrs.weight_scale, weight_scale)

    # Tied-weight dedup: if a previously-processed module shared the same
    # source weight memory, alias the packed weight + scale buffers so the
    # downstream data_ptr dedup in postprocess_state_dict can collapse them.
    # input_scale is safe to alias because sync_tied_input_amax (earlier in
    # this export) already max-merged the per-side amaxes. Gated on the
    # caller-owned _tied_cache so the dedup state is scoped to one export.
    if _tied_cache is not None:
        _prior = _tied_cache.get(_tied_source_data_ptr)
        if _prior is not None and _prior is not sub_module:
            if hasattr(_prior, weight_name):
                setattr(sub_module, weight_name, getattr(_prior, weight_name))
            for _attr in (
                quantizer_attrs.weight_scale,
                quantizer_attrs.weight_scale_2,
                quantizer_attrs.input_scale,
            ):
                if not hasattr(_prior, _attr):
                    continue
                if _attr in sub_module._buffers:
                    del sub_module._buffers[_attr]
                elif hasattr(sub_module, _attr):
                    delattr(sub_module, _attr)
                sub_module.register_buffer(_attr, getattr(_prior, _attr))
        else:
            _tied_cache[_tied_source_data_ptr] = sub_module

    torch.cuda.empty_cache()


def _dispatch_export_handler(name: str, sub_module: nn.Module, ctx: ExportContext) -> None:
    """QLoRA skip, unpack-weight preprocessing, and handler dispatch for one module."""
    if ctx.is_modelopt_qlora and hasattr(sub_module, "base_layer"):
        return
    # Restore unpacked weight so the export path can read the live quantizer state.
    if hasattr(sub_module, "weight_packed") or (
        "QuantFP8Linear" in type(sub_module).__name__ and sub_module.weight.element_size() <= 1
    ):
        sub_module.unpack_weight()
    handler = ExportModuleRegistry.match(sub_module)
    if handler is not None:
        handler(name, sub_module, ctx)


def _process_quantized_modules(
    model: nn.Module,
    dtype: torch.dtype,
    is_modelopt_qlora: bool = False,
) -> None:
    """Process all quantized modules in model, export weights in-place.

    This function iterates through all modules in the model and invokes the first matching
    handler in :data:`ExportModuleRegistry`. Modules matching no handler are left untouched.

    Args:
        model: The model containing quantized modules.
        dtype: The data type for weight conversion.
        is_modelopt_qlora: Whether the model is a modelopt-trained QLoRA model.
            If True, modules with base_layer attribute are skipped.
    """
    # Per-call tied-weight dedup caches inside the context. Created fresh on
    # every invocation so cache state is scoped to one export and cannot leak
    # into a later call (see ExportContext).
    ctx = ExportContext(model=model, dtype=dtype, is_modelopt_qlora=is_modelopt_qlora)
    fsdp_module_to_reshard = None

    for name, sub_module in model.named_modules():
        # Optimization to perform resharding only once per decoder layer to avoid extra communication overhead
        if isinstance(sub_module, FSDPModule):
            # Every time we encounter a new FSDPModule, the previous decoder layer is fully processed.
            # We need to reshard the previous FSDPModule to prevent potential OOM.
            # This hack reduces the number of unshard reshard operations, to avoid unnecessary communication.
            if fsdp_module_to_reshard is not None:
                fsdp_module_to_reshard.reshard()

            fsdp_module_to_reshard = sub_module

        _dispatch_export_handler(name, sub_module, ctx)


def _export_transformers_checkpoint(
    model: nn.Module,
    dtype: torch.dtype | None = None,
    is_modelopt_qlora: bool = False,
    **kwargs,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exports the torch model to the packed checkpoint with original HF naming.

    The packed checkpoint will be consumed by the TensorRT-LLM unified converter.

    Builds the whole quantized state dict in memory, so it requires every weight to be
    resident. Models with accelerate CPU/disk offload are rejected here and handled by
    :func:`_export_transformers_checkpoint_streaming`, which materializes one layer at a
    time; :func:`export_hf_checkpoint` picks between the two.

    Args:
        model: the full torch model to export. The actual quantized model may be a submodule.
        dtype: the weights data type to export the unquantized layers or the default model data type if None.

    Returns:
        post_state_dict: Dict containing quantized weights
        quant_config: config information to export hf_quant_cfg.json

    Raises:
        NotImplementedError: if the model has accelerate offload hooks.
    """
    dtype = _resolve_export_dtype(model, dtype)
    _prepare_moe_inputs(model, dtype, is_modelopt_qlora)

    # Resmooth and requantize fused layers
    # TODO: Handle mixed precision
    requantize_resmooth_fused_llm_layers(model)

    # Offloaded models need their weights materialized layer-by-layer, which this
    # whole-state-dict path cannot do; export_hf_checkpoint() streams them instead.
    if has_accelerate_offload(model):
        raise NotImplementedError(
            "_export_transformers_checkpoint does not support disk/CPU-offloaded models. "
            "Use export_hf_checkpoint() which dispatches to _export_transformers_checkpoint_streaming."
        )

    # Remove all hooks from the model
    try:
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(model, recurse=True)
    except ImportError:
        pass  # no accelerate installed → no offload hooks exist to remove

    quant_config = get_quant_config(model, is_modelopt_qlora=is_modelopt_qlora)

    _add_mtp_exclusions(model, quant_config)

    _warn_on_unsynced_moe_gate_up(model)

    # Merge per-side input_quantizer amaxes BEFORE _process_quantized_modules,
    # so the merged value flows into input_scale derivation downstream.
    synced_input = sync_tied_input_amax(model)
    if synced_input:
        print(
            f"sync_tied_input_amax: max-merged input_quantizer amaxes across "
            f"{synced_input} tied module group(s)"
        )

    # Process all quantized modules and export weights
    from modelopt.torch.quantization.plugins.huggingface import _reconstruct_fused_moe_linear

    _process_quantized_modules(model, dtype, is_modelopt_qlora)
    _reconstruct_fused_moe_linear(model)

    if is_fsdp2_model(model):
        # FSDP2: gather the full (unsharded) state_dict to CPU on rank 0.
        quantized_state_dict = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    else:
        # Non-FSDP2: assumes a replicated model (rank 0 has the full state dict).
        quantized_state_dict = model.state_dict()

    # We define kv cache scale as amax / 448 for both FP8 and NVFP4 KV cache quantization.
    kv_cache_max_bound = 448
    kv_cache_format = quant_config["quantization"]["kv_cache_quant_algo"]

    # Reorder so canonical-side tied keys (per HF's _tied_weights_keys)
    # iterate first into postprocess_state_dict's first-wins data_ptr dedup.
    # Self-gated to DiffusionGemma inside _reorder_canonical_first; no-op
    # for every other model.
    quantized_state_dict = _reorder_canonical_first(quantized_state_dict, model)

    quantized_state_dict = postprocess_state_dict(
        quantized_state_dict, kv_cache_max_bound, kv_cache_format, is_modelopt_qlora
    )

    return quantized_state_dict, quant_config


# TODO: Remove this workaround once HuggingFace fixes revert_weight_conversion to handle
# scalar (0-d) tensors. transformers' Chunk.convert() calls torch.chunk() on quantization
# scale buffers that are 0-d scalars, raising RuntimeError ("chunk expects at least a
# 1-dimensional tensor"). Confirmed in transformers 5.12.0.
# See: transformers/core_model_loading.py, Chunk.convert()


def export_speculative_decoding(
    model: torch.nn.Module,
    dtype: torch.dtype | None = None,
    export_dir: Path | str = tempfile.gettempdir(),
) -> None:
    """Export speculative decoding HuggingFace model checkpoint."""
    assert has_spec_opt(model), "Model is not optimized for speculative decoding."

    exporter: SpeculativeDecodingExporter = model.get_exporter()
    exporter.export(export_dir, dtype)


def _write_hf_export_config(
    model: nn.Module,
    hf_quant_config: dict | None,
    export_dir: Path,
) -> None:
    """Write hf_quant_config.json (if quantized) and embed quantization_config into config.json."""
    quantization_details = (hf_quant_config or {}).get("quantization", {})
    is_quantized_export = (
        quantization_details.get("quant_algo") is not None
        or quantization_details.get("kv_cache_quant_algo") is not None
    )
    quantization_config = None
    if hf_quant_config is not None and is_quantized_export:
        with open(f"{export_dir}/hf_quant_config.json", "w") as file:
            json.dump(hf_quant_config, file, indent=4)
        quantization_config = convert_hf_quant_config_format(hf_quant_config)

    original_config = f"{export_dir}/config.json"
    with open(original_config) as file:
        config_data = json.load(file)
    sanitize_hf_config_for_deployment(config_data, model)
    if quantization_config is not None:
        config_data["quantization_config"] = quantization_config
    if export_sparse_attention_config is not None:
        sparse_attn_config = export_sparse_attention_config(model)
        if sparse_attn_config is not None:
            config_data["sparse_attention_config"] = sparse_attn_config
    with open(original_config, "w") as file:
        json.dump(config_data, file, indent=4)


def export_hf_checkpoint(
    model: Any,
    dtype: torch.dtype | None = None,
    export_dir: Path | str = tempfile.gettempdir(),
    save_modelopt_state: bool = False,
    components: list[str] | None = None,
    extra_state_dict: dict[str, torch.Tensor] | None = None,
    max_shard_size: int | str = "10GB",
    **kwargs,
):
    """Export quantized HuggingFace model checkpoint (transformers or diffusers).

    This function automatically detects whether the model is from transformers
    or diffusers and applies the appropriate export logic.

    Under ``torch.distributed`` (e.g. FSDP2), all ranks participate in the
    collective state-dict gather inside ``_export_transformers_checkpoint``;
    only rank 0 writes files. A final barrier syncs the other ranks.

    Args:
        model: The full torch model to export. The actual quantized model may be a submodule.
            Supports both transformers models (e.g., LlamaForCausalLM) and diffusers
            models/pipelines (e.g., StableDiffusionPipeline, UNet2DConditionModel).
        dtype: The weights data type to export the unquantized layers or the default
            model data type if None.
        export_dir: The target export path.
        save_modelopt_state: Whether to save the modelopt state_dict.
        components: Only used for diffusers pipelines. Optional list of component names
            to export. If None, all quantized components are exported.
        extra_state_dict: Extra state dictionary to add to the exported model.
        max_shard_size: Maximum size of each safetensors shard file. Defaults to "10GB".
        **kwargs: Runtime-specific post-processing options forwarded to
            :func:`_postprocess_safetensors` for diffusion model exports.
            See its docstring for supported keys.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    is_diffusers_obj = False
    if HAS_DIFFUSERS:
        is_diffusers_obj = is_diffusers_object(model)
    if is_diffusers_obj:
        # Imported here rather than at module scope: the diffusers exporter imports the
        # shared module-walking helpers from this module, so a top-level import would be
        # circular. The cycle goes away once those helpers move to their own modules.
        from .unified_export_diffusers import _export_diffusers_checkpoint

        _export_diffusers_checkpoint(
            model,
            dtype,
            export_dir,
            components,
            max_shard_size,
            **kwargs,
        )
        return

    is_distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and is_fsdp2_model(model)
    )
    # Offloaded models take the streaming path: it materializes one layer at a time and
    # writes each straight to a shard file, so peak memory is one layer plus one shard
    # buffer instead of the whole quantized state dict.
    _offloaded = has_accelerate_offload(model)

    try:
        if _offloaded:
            # Imported here rather than at module scope: the streaming exporter imports the
            # shared prep helpers from this module, so a top-level import would be circular.
            from .unified_export_hf_streaming import _export_transformers_checkpoint_streaming

            if save_modelopt_state:
                warnings.warn(
                    "save_modelopt_state=True is not supported in the streaming offload export "
                    "path and will be ignored."
                )
            _, hf_quant_config = _export_transformers_checkpoint_streaming(
                model,
                dtype,
                export_dir=export_dir,
                max_shard_size=max_shard_size,
                extra_state_dict=extra_state_dict,
                **kwargs,
            )
            if getattr(model, "hf_quantizer", None) is not None:
                model.hf_quantizer = None
            try:
                name_mapper = build_reverse_name_mapper(model)
                if name_mapper is not None and hf_quant_config:
                    revert_quant_config_names(hf_quant_config.get("quantization", {}), name_mapper)
            except Exception as exc:
                warnings.warn(
                    f"Quant-aware reverse weight conversion skipped ({exc}); exported tensor "
                    "names may not match the original HF hub checkpoint."
                )
            _write_hf_export_config(model, hf_quant_config, export_dir)
            return

        post_state_dict, hf_quant_config = _export_transformers_checkpoint(model, dtype, **kwargs)

        # Remove hf_quantizer from model so post_state_dict can be exported.
        if getattr(model, "hf_quantizer", None) is not None:
            model.hf_quantizer = None

        export_state_dict = {**post_state_dict, **(extra_state_dict or {})}

        # transformers may have applied a load-time conversion_mapping (fused gate_up_proj,
        # renamed MoE leaves, reordered model/language_model prefix), so the in-memory names
        # differ from the original hub checkpoint. Reverse it quantization-aware so exported
        # tensor names stay aligned with the hub checkpoint (the unified-checkpoint contract).
        # transformers' own revert_weight_conversion errors on 0-d scalar scale tensors, so we
        # do it here. The same rename is applied to the quant-config module references
        # (exclude_modules / quantized_layers keys) so a deployment loader matches them against
        # the reverted hub-named modules (otherwise an excluded BF16 layer is loaded as quantized
        # and fails). Best-effort and atomic: any failure (an op we cannot reverse yet,
        # transformers API drift, unexpected shapes) falls back to the in-memory names for BOTH
        # weights and config so they stay mutually consistent.
        try:
            name_mapper = build_reverse_name_mapper(model)
            export_state_dict = revert_weight_conversion_quant_aware(model, export_state_dict)
            if name_mapper is not None and hf_quant_config:
                revert_quant_config_names(hf_quant_config.get("quantization", {}), name_mapper)
        except Exception as exc:
            warnings.warn(
                f"Quant-aware reverse weight conversion skipped ({exc}); exported tensor "
                "names may not match the original HF hub checkpoint."
            )

        # Under torch.distributed only rank 0 writes; others sync at the finally barrier.
        if is_distributed and torch.distributed.get_rank() != 0:
            return

        # Keep transformers' own revert_weight_conversion disabled (the quant-aware reverse
        # above replaces it): it can't handle quantized state dicts (RuntimeError on 0-d scalar
        # scale tensors). Patch both the source and importing module since modeling_utils does
        # `from core_model_loading import revert_weight_conversion`.
        _patches = _patch_revert_weight_conversion()

        _sanitize_generation_config_for_save(model)

        # TODO: parallelize the disk write across ranks (avoid single-process speed + rank-0 OOM).
        try:
            model.save_pretrained(
                export_dir,
                state_dict=export_state_dict,
                save_modelopt_state=save_modelopt_state,
                max_shard_size=max_shard_size,
            )
        finally:
            _unpatch_revert_weight_conversion(_patches)

        _write_hf_export_config(model, hf_quant_config, export_dir)

    except Exception as e:
        warnings.warn(
            "Cannot export model to the model_config. The modelopt-optimized model state_dict"
            " can be saved with torch.save for further inspection."
        )
        raise e
    finally:
        if is_distributed:
            torch.distributed.barrier()
