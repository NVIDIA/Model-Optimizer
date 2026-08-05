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

"""Per-module quantized weight export.

The leaf of the export pipeline: packing one module's weight into its quantized
representation and registering the scale buffers beside it, plus the registry dispatch
and the whole-model walk that drive it.

Like :mod:`hf_export_prep`, this depends on nothing else in the export package, so the
exporters and the MoE/handler plugins can import it directly instead of lazily.
"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule

from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.qtensor import MXFP8QTensor, NVFP4QTensor
from modelopt.torch.quantization.qtensor.base_qtensor import QTensorWrapper
from modelopt.torch.quantization.qtensor.nvfp4_tensor import _cast_per_block_scale_to_fp8
from modelopt.torch.quantization.utils import quantizer_attr_names

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
from .quant_utils import (
    get_activation_scaling_factor,
    get_quantization_format,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    maybe_transpose_expert_weight_dimensions,
    to_quantized_weight,
)
from .registry import ExportContext, ExportModuleRegistry


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
