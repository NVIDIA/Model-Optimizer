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

"""Utils for quantization including scaling factors adjustments."""

import logging
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any
from warnings import warn

import torch
import torch.nn as nn

from modelopt import __version__
from modelopt.torch.quantization.model_calib import (
    enable_stats_collection,
    finish_stats_collection,
    svd,
)
from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear
from modelopt.torch.quantization.qtensor import (
    FP8QTensor,
    MXFP4QTensor,
    MXFP8QTensor,
    NVFP4QTensor,
    QTensorWrapper,
)
from modelopt.torch.quantization.utils import (
    QuantizerAttrNames,
    quantizer_attr_names,
    weight_attr_names,
)
from modelopt.torch.utils import clear_cuda_cache

from ..quantization.nn import SequentialQuantizer, TensorQuantizer
from .model_config import (
    KV_CACHE_FP8,
    KV_CACHE_INT8,
    KV_CACHE_NVFP4,
    KV_CACHE_NVFP4_AFFINE,
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_REAL,
    QUANTIZATION_FP8_PB_WO,
    QUANTIZATION_FP8_PC_PT,
    QUANTIZATION_INT4_AWQ,
    QUANTIZATION_INT8_SQ,
    QUANTIZATION_INT8_WO,
    QUANTIZATION_MXFP4,
    QUANTIZATION_MXFP8,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_NVFP4_AWQ,
    QUANTIZATION_NVFP4_SVDQUANT,
    QUANTIZATION_W4A8_AWQ,
    QUANTIZATION_W4A8_MXFP4_FP8,
    QUANTIZATION_W4A8_NVFP4_FP8,
)

logger = logging.getLogger(__name__)


def get_scaling_factor_from_weight(weight, group_size) -> torch.tensor:
    """Calculate the weight scaling factor for a given group size."""
    [n, k] = weight.shape

    if group_size != 0:
        # int4_awq
        if k % group_size != 0:
            raise NotImplementedError(
                "Weight shape is not divisible for block size for block quantization."
            )
        weight = weight.reshape(n, k // group_size, group_size)
        maxbound = 7.0
    else:
        # int8_sq
        maxbound = 127.0
    amax = weight.abs().max(dim=-1)[0].float()

    weights_scaling_factor = amax / maxbound

    # Let's filter the zeros in the scaling factor if the weights are zero
    # to avoid the divided-by-zero error..
    weights_scaling_factor[weights_scaling_factor == 0] = 1.0

    return weights_scaling_factor


def maybe_transpose_expert_weight_dimensions(
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None = None,
    is_bmm_expert_weight: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Transpose the last two dimensions of expert weights.

    This function transposes expert weights between the two layouts:
    - (num_experts, input_dim, output_dim) ↔ (num_experts, output_dim, input_dim)

    Since transpose(-2, -1) is self-inverse, this function can be used for both
    forward and backward transformations. This is needed for quantization functions
    that expect the last dimension to be the input dimension for block quantization.
    Specifically used for bmm-style expert weights in models like llama4 and gpt-oss.

    Args:
        weight: The weight tensor to transpose. Expected shape for experts: (num_experts, dim1, dim2)
        weight_scale: Optional weight scaling factor tensor to transpose alongside weight
        is_bmm_expert_weight: Whether this is an expert weight (3D tensor) that needs transposition

    Returns:
        Tuple of (transposed_weight, transposed_weight_scale)
    """
    if not is_bmm_expert_weight or weight.dim() != 3:
        return weight, weight_scale

    transposed_weight = weight.transpose(-2, -1)
    transposed_weight_scale = weight_scale.transpose(-2, -1) if weight_scale is not None else None

    return transposed_weight, transposed_weight_scale


def resmooth_and_get_scale(
    merged_weights: torch.Tensor,
    pre_quant_scales: list[torch.Tensor],
    ranks: int,
    group_size: int,
    new_pre_quant_scale: torch.Tensor | None = None,
    quantization: str | None = QUANTIZATION_NONE,
):
    """Resmooths weights from a single or multiple ranks and get scaling factors and amax.

    Args:
        merged_weights: Merged weights from ranks.
        pre_quant_scales: List of pre-quantization scales for each rank.
        ranks: Number of ranks.
        group_size: Group size of the quantization block.
        new_pre_quant_scale (optional): If not provided, weights will be resmoothed using
            the average of pre_quant_scales.

    Returns:
        weights: Resmoothed weights.
        weight_scaling_factors: Resmoothed scaling factors.
        avg_pre_quant_scale: Calculated average of the quantization scale.
    """
    if new_pre_quant_scale is None:
        new_pre_quant_scale = torch.stack(pre_quant_scales).mean(dim=0)

    assert len(pre_quant_scales) > 0 and new_pre_quant_scale.numel() == merged_weights.shape[1], (
        "Shape of pre_quant_scales and weights do not match."
    )
    weights = torch.chunk(merged_weights, ranks, dim=0)

    scales = []
    new_weights = []
    for i, p_scaling_factor in enumerate(pre_quant_scales):
        # De smooth & Re smooth
        weight = (
            weights[i]
            * p_scaling_factor.type(weights[i].dtype)
            / new_pre_quant_scale.type(weights[i].dtype)
        )
        new_weights.append(weight)
        # If NVFP4_AWQ then we view the scales as uint8 to allow for cat later
        if quantization in [QUANTIZATION_NVFP4_AWQ, QUANTIZATION_NVFP4_SVDQUANT]:
            scale, _ = NVFP4QTensor.get_weights_scaling_factor(weight, group_size).view(torch.uint8)
        else:
            scale = get_scaling_factor_from_weight(weight, group_size)
        scales.append(scale)

    resmoothed_scales = torch.cat(scales, dim=0)

    return (
        torch.cat(new_weights, dim=0),
        resmoothed_scales.view(torch.float8_e4m3fn)
        if quantization in [QUANTIZATION_NVFP4_AWQ, QUANTIZATION_NVFP4_SVDQUANT]
        else resmoothed_scales,  # if NVFP4_AWQ we view the scales back as float8_e4m3fn after cat
        new_pre_quant_scale,
    )


def adjust_attn_amax_values(module):
    """Adjusts the amax values for the attention layers."""
    projection_prefixes = ["q", "k", "v"]
    max_amax = float("-inf")
    proj_layers = []

    # Find all projection layers whose names contain 'q', 'k', or 'v'
    for name, sub_module in module.named_children():
        for prefix in projection_prefixes:
            if (
                prefix in name
                and hasattr(sub_module, "weight_quantizer")
                and hasattr(sub_module.weight_quantizer, "amax")
            ):
                proj_layers.append(sub_module)
                max_amax = max(max_amax, sub_module.weight_quantizer.amax.item())

    if not proj_layers:
        raise ValueError(
            "No projection layers with the specified prefixes ('q', 'k', 'v') have amax attributes"
        )

    assert max_amax > 0, "max_amax must be positive."

    # Set all amax values to the maximum found
    for proj_layer in proj_layers:
        proj_layer.weight_quantizer.amax.fill_(max_amax)


def get_scaling_factor(quantizer: TensorQuantizer) -> torch.Tensor:
    """Returns scaling factor from the quantizer as torch.Tensor."""
    if not quantizer.is_enabled:
        return None

    amax = quantizer.export_amax()
    if amax is None:
        return None

    # tensorrt_llm uses float as the scaling_factors.
    if quantizer.num_bits == (2, 1):
        scaling_factor = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(quantizer)
    else:
        scaling_factor = amax.float() / quantizer.maxbound

    assert torch.all(scaling_factor > 0), f"scaling factor {scaling_factor} not positive."

    return scaling_factor


def _ensure_weight_quantizer_calibrated(
    weight_quantizer: TensorQuantizer, weight: torch.Tensor, module_name: str = ""
) -> None:
    """Calibrate weight quantizer if amax is not set.

    This is a lazy calibration pattern used during export when weight quantizers
    may not have been calibrated during the main calibration phase.

    Args:
        weight_quantizer: The weight quantizer to calibrate
        weight: The weight tensor to use for calibration
        module_name: Optional module name for better warning messages
    """
    if not hasattr(weight_quantizer, "_amax") or weight_quantizer._amax is None:
        warn(
            f"Weight quantizer{f' for {module_name}' if module_name else ''} was not calibrated. "
            f"Computing amax from weights. This may occur if: "
            f"some experts were not activated during calibration (expected for MoE models), try increasing --calib_size"
        )
        weight_quantizer.reset_amax()
        enable_stats_collection(weight_quantizer)
        weight_quantizer(weight)
        finish_stats_collection(weight_quantizer)


def get_activation_scaling_factor(
    module: nn.Module, input_quantizer_name: str = "input_quantizer"
) -> torch.Tensor:
    """Returns the activation scaling factor."""
    # If NVFP4, return activation scaling factor from NVFP4QTensor
    input_quantizer = getattr(module, input_quantizer_name, None)
    if input_quantizer is None:
        return None

    if get_quantization_format(module) in [
        QUANTIZATION_NVFP4,
        QUANTIZATION_NVFP4_AWQ,
        QUANTIZATION_NVFP4_SVDQUANT,
    ]:
        return NVFP4QTensor.get_activation_scaling_factor(input_quantizer)
    return get_scaling_factor(input_quantizer)


def get_weight_scaling_factor(module: nn.Module, weight_name: str = "weight") -> torch.Tensor:
    """Returns the weight scaling factor."""
    # module.weight_quantizer could be a TensorQuantizer (for algorithms except W4A8) or
    # a SequentialQuantizer (for W4A8). In the latter case, we need to get the scaling factor from the
    # first quantizer of the SequentialQuantizer instance.

    weight: nn.Parameter = getattr(module, weight_name)
    weight_quantizer: TensorQuantizer | SequentialQuantizer | None = getattr(
        module, quantizer_attr_names(weight_name).weight_quantizer, None
    )

    if weight_quantizer is None:
        return None

    if isinstance(weight_quantizer, SequentialQuantizer):
        return get_scaling_factor(weight_quantizer[0])

    quantization_format = get_quantization_format(module)
    # If NVFP4, we need to return quantized per_block scaling factors
    if quantization_format in [
        QUANTIZATION_NVFP4,
        QUANTIZATION_NVFP4_AWQ,
        QUANTIZATION_NVFP4_SVDQUANT,
        QUANTIZATION_W4A8_NVFP4_FP8,
    ]:
        # Calibrate weight quantizer if amax is not set
        module_name = f"{type(module).__name__}.{weight_name}"
        _ensure_weight_quantizer_calibrated(weight_quantizer, weight, module_name)

        if quantization_format == QUANTIZATION_W4A8_NVFP4_FP8:
            # weight_scaling_factor_2 for w4a8 needs to be amax/448, so that the wsf is in range 448/6.
            # This is because the kernel dequantizes weight to fp8, which is in range 448.
            weight_scaling_factor_2 = weight_quantizer._amax.float() / 448.0
        else:
            weight_scaling_factor_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(
                weight_quantizer
            )
        return NVFP4QTensor.get_weights_scaling_factor(
            weight,
            weight_quantizer.block_sizes[-1],
            weight_scaling_factor_2.to(weight.device),
        )[0]

    if quantization_format in [QUANTIZATION_W4A8_MXFP4_FP8, QUANTIZATION_MXFP4]:
        return MXFP4QTensor.quantize(weight, block_size=weight_quantizer.block_sizes[-1])[
            1
        ].reshape(*weight.shape[:-1], -1)

    if quantization_format == QUANTIZATION_MXFP8:
        return MXFP8QTensor.get_weights_scaling_factor_from_quantizer(weight, weight_quantizer)
    return get_scaling_factor(weight_quantizer)


def get_weight_scaling_factor_2(module: nn.Module, weight_name: str = "weight") -> torch.Tensor:
    """Returns the secondary weight scaling factor."""
    weight_quantizer = getattr(module, quantizer_attr_names(weight_name).weight_quantizer, None)

    if weight_quantizer is None:
        return None

    quantization_format = get_quantization_format(module)

    # Calibrate weight quantizer if amax is not set for all NVFP4 variants
    if quantization_format in [
        QUANTIZATION_NVFP4,
        QUANTIZATION_NVFP4_AWQ,
        QUANTIZATION_NVFP4_SVDQUANT,
        QUANTIZATION_W4A8_NVFP4_FP8,
    ]:
        weight = getattr(module, weight_name)
        module_name = f"{type(module).__name__}.{weight_name}"
        _ensure_weight_quantizer_calibrated(weight_quantizer, weight, module_name)

    if quantization_format in [
        QUANTIZATION_NVFP4,
        QUANTIZATION_NVFP4_AWQ,
        QUANTIZATION_NVFP4_SVDQUANT,
    ]:
        return NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(weight_quantizer)
    elif quantization_format == QUANTIZATION_W4A8_NVFP4_FP8:
        # weight_scaling_factor_2 for w4a8 needs to be amax/448, so that the wsf is in range 448/6.
        # This is because the kernel dequantizes weight to fp8, which is in range 448.
        return weight_quantizer._amax.float() / 448.0

    # SequentialQuantizer is required
    if not isinstance(weight_quantizer, SequentialQuantizer) or not weight_quantizer[-1].is_enabled:
        return None

    assert len(weight_quantizer) == 2, (
        "modelopt only supports 2 sequential quantization layers for now"
    )
    return get_scaling_factor(weight_quantizer[-1])


def get_prequant_scaling_factor(module: nn.Module) -> torch.Tensor:
    """Returns the prequant scaling factor."""
    prequant_scaling_factor = (
        module.input_quantizer._pre_quant_scale.squeeze()
        if hasattr(module, "input_quantizer")
        and hasattr(module.input_quantizer, "_pre_quant_scale")
        else None
    )

    if prequant_scaling_factor is not None:
        assert torch.all(prequant_scaling_factor > 0), (
            f"prequant scaling factor {prequant_scaling_factor} not positive."
        )
    return prequant_scaling_factor


def get_kv_cache_bias(kv_module: nn.Module) -> list[torch.Tensor]:
    """Returns the kv_cache bias if _bias_value is set. Else returns None."""
    kv_bias = []
    for quantizer in ["k_bmm_quantizer", "v_bmm_quantizer"]:
        quantizer_module = getattr(kv_module, quantizer, None)
        kv_bias.append(getattr(quantizer_module, "_bias_value", None))
    return kv_bias


def get_kv_cache_scaling_factor(self_attention_module: nn.Module) -> list[torch.Tensor | None]:
    """Get the K and V BMM scaling factors for the self attention module.

    Args:
        self_attention_module: The self attention module to get the K and V BMM scaling factors from.

    Returns:
        The K and V BMM scaling factors.
    """
    if not hasattr(self_attention_module, "k_bmm_quantizer") or not hasattr(
        self_attention_module, "v_bmm_quantizer"
    ):
        return [None, None]

    scaling_factors = [
        get_scaling_factor(getattr(self_attention_module, quantizer))
        for quantizer in ("k_bmm_quantizer", "v_bmm_quantizer")
    ]

    # For FP8, we recommend default kv cache scaling factor to be 1.
    if get_kv_cache_dtype(self_attention_module) == KV_CACHE_FP8:
        for i, factor in enumerate(scaling_factors):
            if factor.item() > 0.5:
                warn(
                    f"Warning: Large KV activation detected: {factor.item()}, "
                    "Quantized KV cache may lead to higher accuracy drop."
                )
            scaling_factors[i] = torch.max(
                factor, torch.tensor([1.0], dtype=torch.float, device=factor.device)
            )
    return scaling_factors


def get_kv_cache_dtype(modules: list[nn.Module] | nn.Module) -> str | None:
    """Returns the kv_cache dtype.

    If num_bits of output_quantizer is (4, 3) then returns FP8; if it is 8, returns int8,
    otherwise returns None.

    Args:
        modules: The module or list of modules to inspect.

    Returns:
        The kv_cache dtype.
    """
    num_bits_list = []
    is_affine = True

    if isinstance(modules, nn.Module):
        modules = [modules]

    for module in modules:
        # Case where the module has both k_bmm_quantizer and v_bmm_quantizer
        # Still check for output quantizer for the unified_megatron_export path
        for quantizer in ("k_bmm_quantizer", "v_bmm_quantizer", "output_quantizer"):
            quantizer_attr = getattr(module, quantizer, None)
            if quantizer_attr and quantizer_attr.is_enabled:
                num_bits_list.append(quantizer_attr.num_bits)
                is_affine &= hasattr(quantizer_attr, "_bias_value")

    return _compute_kv_cache_dtype(num_bits_list)


def _compute_kv_cache_dtype(num_bits_list: list[int | tuple[int, int]]) -> str | None:
    """Returns the kv_cache dtype.

    If num_bits of output_quantizer is (4, 3) then returns FP8; if it is 8, returns int8,
    otherwise returns None.

    Args:
        modules: The module or list of modules to inspect.

    Returns:
        The kv_cache dtype.
    """
    is_affine = True

    if (4, 3) in num_bits_list:
        return KV_CACHE_FP8
    elif 8 in num_bits_list:
        return KV_CACHE_INT8
    elif (2, 1) in num_bits_list and is_affine:
        return KV_CACHE_NVFP4_AFFINE
    elif (2, 1) in num_bits_list:
        return KV_CACHE_NVFP4
    else:
        return QUANTIZATION_NONE


def get_weight_block_size(module: nn.Module, weight_name: str = "weight") -> int:
    """Returns the weight block size."""
    weight_quantizer = getattr(module, quantizer_attr_names(weight_name).weight_quantizer, None)

    if weight_quantizer is None:
        return 0

    if isinstance(weight_quantizer, SequentialQuantizer):
        weight_quantizer = weight_quantizer[0]

    if not weight_quantizer.is_enabled:
        return 0

    block_sizes = weight_quantizer.block_sizes

    if block_sizes:
        return block_sizes[-1]
    return 0


def get_quantization_format(module) -> str | None:
    """Gets the quantization string.

    Gets the quantization string by iterating through the module and its children.
    The first non-None quantization string is returned.
    """

    def _get_quantization_from_layer(layer, quantizer_attr_names: QuantizerAttrNames):
        weight_quantizer = getattr(layer, quantizer_attr_names.weight_quantizer, None)
        input_quantizer = getattr(layer, quantizer_attr_names.input_quantizer, None)

        if weight_quantizer is None or not weight_quantizer.is_enabled:
            return QUANTIZATION_NONE

        # Handle SequentialQuantizer
        if isinstance(weight_quantizer, SequentialQuantizer):
            assert (
                len(weight_quantizer) == 2
                and weight_quantizer[0].num_bits == 4
                and weight_quantizer[1].num_bits == (4, 3)
            ), "Unsupported SequentialQuantizer configuration"
            assert (
                weight_quantizer[0].block_sizes
                and len(weight_quantizer[0].block_sizes) > 0
                and weight_quantizer[0].block_sizes[-1] > 0
            ), "Invalid block_sizes for SequentialQuantizer"

            return QUANTIZATION_W4A8_AWQ

        # Handle individual num_bits cases
        if weight_quantizer.num_bits == 4:
            assert len(weight_quantizer.block_sizes) > 0 and weight_quantizer.block_sizes[-1] > 0, (
                "Invalid block_sizes for INT4 quantizer"
            )
            return QUANTIZATION_INT4_AWQ

        if weight_quantizer.num_bits == 8:
            if input_quantizer is not None and input_quantizer.is_enabled:
                return QUANTIZATION_INT8_SQ
            else:
                return QUANTIZATION_INT8_WO

        if weight_quantizer.num_bits == (4, 3):
            if weight_quantizer.block_sizes:
                assert weight_quantizer.block_sizes[-1] > 0, "Invalid block_sizes for FP8 quantizer"
                # Check if this is MXFP8 (dynamic block quantization with scale_bits (8, 0))
                block_sizes = getattr(weight_quantizer, "block_sizes")
                if (
                    isinstance(block_sizes, dict)
                    and block_sizes.get("type", "static") == "dynamic"
                    and block_sizes.get("scale_bits") == (8, 0)
                ):
                    return QUANTIZATION_MXFP8
                if weight_quantizer.fake_quant:
                    return QUANTIZATION_FP8_PB_WO
                else:
                    return QUANTIZATION_FP8_PB_REAL
            if weight_quantizer.axis == 0:
                return QUANTIZATION_FP8_PC_PT
            return QUANTIZATION_FP8

        if weight_quantizer.num_bits == (2, 1):
            # FP4 formats are all block quantization
            block_sizes = getattr(weight_quantizer, "block_sizes")
            scale_bits = block_sizes.get("scale_bits")

            if input_quantizer is not None and hasattr(weight_quantizer, "svdquant_lora_a"):
                return QUANTIZATION_NVFP4_SVDQUANT
            if input_quantizer is not None and hasattr(input_quantizer, "_pre_quant_scale"):
                return QUANTIZATION_NVFP4_AWQ
            if getattr(layer, "fused_with_prequant", False):
                return QUANTIZATION_NVFP4_AWQ
            assert input_quantizer is not None, (
                f"input_quantizer is None for {quantizer_attr_names}"
            )
            if (
                block_sizes.get("type", "static") == "dynamic"
                and scale_bits == (8, 0)
                and input_quantizer.is_enabled
                and input_quantizer.num_bits == (4, 3)
                and input_quantizer.block_sizes is None
            ):
                return QUANTIZATION_W4A8_MXFP4_FP8
            if (
                block_sizes.get("type", "static") == "dynamic"
                and scale_bits == (4, 3)
                and input_quantizer.is_enabled
                and input_quantizer.num_bits == (4, 3)
                and input_quantizer.block_sizes is None
            ):
                return QUANTIZATION_W4A8_NVFP4_FP8
            if scale_bits == (4, 3):
                return QUANTIZATION_NVFP4
            elif scale_bits == (8, 0):
                return QUANTIZATION_MXFP4

        # Raise error for unsupported num_bits
        raise NotImplementedError(
            f"Unsupported quantizer with num_bits: {weight_quantizer.num_bits}"
        )

    for weight_name in weight_attr_names(module):
        quantization = _get_quantization_from_layer(module, quantizer_attr_names(weight_name))
        if quantization != QUANTIZATION_NONE:
            return quantization

    for _, layer in module.named_children():
        format = get_quantization_format(layer)
        if format != QUANTIZATION_NONE:
            return format

    return QUANTIZATION_NONE


def _prefix_wildcard_summarize_exclude_modules(unquantized_layers, quantized_layers):
    """Generate a summarization of the quantization layer configs using prefix wildcards.

    Prefix wildcards means we only consider wildcards that is a prefix with a star in the end.
    We do not consider other wildcards such as: a*b.
    """

    def all_matching_prefix_wildcards(name):
        # include all possible prefix wildcards, and the exact name itself
        wildcards = {name}
        for i in range(len(name) + 1):
            wildcards.add(name[:i] + "*")
        return wildcards

    def next_formatted_matching_prefix_wildcards(name: str) -> Generator[list[str], None, None]:
        """Enumerate formatted prefix wildcards. A result may be a combination of prefix wildcards.

        Formatted here means we only consider wildcards at dot split. We need two patterns.

        1. a single wildcard: module_name*
        2. a set of 2 wildcards: {module_name, module_name.*}. We need this pattern set because
           module_name* may match other modules with module_name as a prefix.
        """
        for i in range(len(name)):
            if name[i] == ".":
                yield [name[:i] + "*"]
                yield [name[:i], name[:i] + ".*"]
        # in the end, itself only is a wildcard
        yield [name]

    # any of the wildcard in this set cannot be present in the result
    negative_wild_candidates = set()
    for layer in quantized_layers:
        negative = all_matching_prefix_wildcards(layer)
        negative_wild_candidates.update(negative)
        logger.debug(
            f"Quantized layer {layer}, prefix wildcards {negative} identified as negative wildcards"
        )

    res_summary = set()
    for layer in unquantized_layers:
        candidate_wildcards = []
        for wildcards in next_formatted_matching_prefix_wildcards(layer):
            if any(wildcard in negative_wild_candidates for wildcard in wildcards):
                # need a more specific wildcard
                logger.debug(
                    f"Unquantized layer {layer}, prefix wildcards {wildcards} invalidated by negative wildcards"
                )
                continue
            if all(wildcard in res_summary for wildcard in wildcards):
                # we get covered already, do not need to move forward, and clear candidate
                logger.debug(
                    f"Unquantized layer {layer}, prefix wildcards {wildcards} already covered"
                )
                candidate_wildcards = []
                break
            # find one, now terminate the search
            candidate_wildcards = wildcards
            logger.debug(
                f"Unquantized layer {layer}, prefix wildcards {wildcards} identified as a new match"
            )
            break
        res_summary.update(candidate_wildcards)
    return res_summary


def process_layer_quant_config(layer_config_dict):
    """Processes per layer quantization information for TRTLLM export to quant_cfg.json."""
    per_layer_config: dict[str, Any] = {
        "quant_algo": None,
        "kv_cache_quant_algo": None,
        "quantized_layers": {},
    }
    layer_config: dict[str, Any] = {}
    # Set of quantization formats used.
    quantization_formats = set()
    quantization_config = None
    exclude_modules = []

    for k, v in layer_config_dict.items():
        if "awq_block_size" in k:
            continue

        # Get layer name for constructing quantized_layers dictionary under per_layer_config
        prefix = ".".join(k.rsplit(".", 1)[:-1])
        awq_key = prefix + ".awq_block_size"

        # Get the corresponding AWQ block size
        block_size_value = layer_config_dict.get(awq_key, 0)

        if v == "fp8":
            layer_config = {"quant_algo": "FP8"}
        elif v == "fp8_pc_pt":
            layer_config = {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"}
        elif v == "int4_awq":
            layer_config = {
                "quant_algo": "W4A16_AWQ",
                "group_size": block_size_value,
                "has_zero_point": False,
                "pre_quant_scale": True,
            }
        elif v == "w4a8_awq":
            layer_config = {
                "quant_algo": "W4A8_AWQ",
                "group_size": block_size_value,
                "has_zero_point": False,
                "pre_quant_scale": True,
            }
        elif v == "int8_wo":
            layer_config = {"quant_algo": "W8A16"}
        elif v == "int8_sq":
            layer_config = {"quant_algo": "W8A8_SQ_PER_CHANNEL"}
        elif v == "nvfp4":
            layer_config = {
                "quant_algo": "NVFP4",
                "group_size": block_size_value,
            }
        elif v == "nvfp4_awq":
            layer_config = {
                "quant_algo": "NVFP4_AWQ",
                "group_size": block_size_value,
                "has_zero_point": False,
                "pre_quant_scale": True,
            }
        elif v == "w4a8_nvfp4_fp8":
            layer_config = {
                "quant_algo": "W4A8_NVFP4_FP8",
                "group_size": block_size_value,
            }
        elif v == "w4a8_mxfp4_fp8":
            layer_config = {
                "quant_algo": "W4A8_MXFP4_FP8",
                "group_size": block_size_value,
            }
        elif v == "nvfp4_svdquant":
            layer_config = {
                "quant_algo": "NVFP4_SVD",
                "group_size": block_size_value,
            }
        elif v == "mxfp8":
            layer_config = {
                "quant_algo": "MXFP8",
                "group_size": block_size_value,
            }
        else:
            layer_config = {"quant_algo": v}

        if layer_config["quant_algo"] != QUANTIZATION_NONE:
            quantization_formats.add(str(layer_config))
            quantization_config = layer_config
            per_layer_config["quantized_layers"].update({prefix: layer_config})
        else:
            exclude_modules.append(prefix)

    # If we have more than one quantization format, infer MIXED_PRECISION
    if len(quantization_formats) > 1:
        per_layer_config["quant_algo"] = "MIXED_PRECISION"
    elif len(quantization_formats) == 1 and quantization_config is not None:
        per_layer_config.update(quantization_config)
        per_layer_config["exclude_modules"] = sorted(
            _prefix_wildcard_summarize_exclude_modules(
                exclude_modules, per_layer_config["quantized_layers"].keys()
            )
        )
        per_layer_config.pop("quantized_layers")

    return per_layer_config


def pack_int4_in_uint8(weight, weights_scaling_factor):
    """Packs the INT4 weights into uint8 tensor."""
    out_dim = weight.shape[-2]
    assert out_dim % 2 == 0, f"Cannot pack weight. Out dimension {out_dim} is not an even number."
    in_dim = weight.shape[-1]
    block_size = weight.shape[-1] // weights_scaling_factor.shape[-1]

    # Scale, round, and clamp to the signed 4-bit range [-8..7].
    int8_tensor = (
        (weight / weights_scaling_factor[..., :, torch.arange(in_dim) // block_size])
        .round()
        .clamp(-8, 7)
        .to(torch.int8)
    )

    # -- Handle the MoE (3D) case vs. the 2D case --
    if int8_tensor.dim() == 3:
        # Dimensions might be (experts, out_dim, in_dim)
        transpose = int8_tensor.permute(0, 2, 1)  # -> (experts, in_dim, out_dim)
        # Reshape to group two output channels (out_dim // 2) and keep an extra dimension of size 2
        transpose = transpose.reshape(-1, in_dim, out_dim // 2, 2)  # (E, in_dim, out_dim//2, 2)

        # Pack two 4-bit values (val0,val1) into a single byte:
        val0 = transpose[..., 0] & 0x0F
        val1 = transpose[..., 1] & 0x0F
        packed_byte = val0 | (val1 << 4)

        # Transpose back to the shape (experts, out_dim // 2, in_dim)
        return packed_byte.permute(0, 2, 1).contiguous().view(torch.uint8)

    else:
        # 2D weights: shape typically (out_dim, in_dim)
        # Transpose to (in_dim, out_dim)
        reshaped = int8_tensor.T.reshape(in_dim, out_dim // 2, 2)

        # Pack two 4-bit values into one byte
        val0 = reshaped[..., 0] & 0x0F
        val1 = reshaped[..., 1] & 0x0F
        packed_byte = val0 | (val1 << 4)

        # Return shape (out_dim // 2, in_dim)
        return packed_byte.T.contiguous().view(torch.uint8)


def to_quantized_weight(
    weight: torch.Tensor,
    weights_scaling_factor: torch.Tensor,
    quantization: str,
    weights_scaling_factor2: torch.Tensor | None = None,
    block_size: int | None = None,
):
    """Converts the weight to the quantized (packed) format."""
    if weights_scaling_factor is not None:
        weights_scaling_factor = weights_scaling_factor.to(weight.device)

    if weights_scaling_factor2 is not None:
        weights_scaling_factor2 = weights_scaling_factor2.to(weight.device)

    # For compressed weights, we directly return the data from wrapper
    if isinstance(weight, QTensorWrapper):
        return weight.data

    if quantization == QUANTIZATION_FP8:
        # Fix RuntimeError: Promotion for Float8 Types is not supported, attempted to promote Float8_e4m3fn and Float
        # in speculative decoding fp8 model export
        if weight.dtype == torch.float8_e4m3fn:
            warn("Skipping quantization: weight already in fp8 format")
            return weight

        if weight.dim() == 3:
            # for MOE stacked weights
            # Clear GPU cache to avoid pontential GPU OOM issues for large models.
            clear_cuda_cache()
            return (weight / weights_scaling_factor.unsqueeze(-1)).to(torch.float8_e4m3fn)
        return (weight / weights_scaling_factor).to(torch.float8_e4m3fn)

    if quantization in [QUANTIZATION_INT8_SQ, QUANTIZATION_INT8_WO]:
        return (weight / weights_scaling_factor[:, None]).round().clamp(-128, 127).to(torch.int8)

    if quantization == QUANTIZATION_MXFP8:
        return MXFP8QTensor.quantize_with_scale(weight, weights_scaling_factor)

    if quantization == QUANTIZATION_FP8_PB_WO:
        return FP8QTensor.quantize(
            weight, weights_scaling_factor.squeeze(), block_sizes={-1: block_size, -2: block_size}
        )[0]._quantized_data

    if quantization == QUANTIZATION_FP8_PC_PT:
        if weight.dim() == 3:
            # Handle different scale tensor shapes
            if weights_scaling_factor.dim() == 1:
                # Per-expert scaling only: (num_experts,) -> (num_experts, 1, 1)
                return (weight / weights_scaling_factor[:, None, None]).to(torch.float8_e4m3fn)
            elif weights_scaling_factor.dim() == 2:
                # Per-channel scaling: check which dimension matches
                if weights_scaling_factor.shape[0] != weight.shape[0]:
                    raise ValueError(
                        f"First dimension (num_experts) mismatch for FP8_PC_PT quantization. "
                        f"weight shape: {weight.shape}, scale shape: {weights_scaling_factor.shape}"
                    )
                if weight.shape[-1] == weight.shape[-2]:
                    raise ValueError(
                        f"Ambiguous scaling dimension for FP8_PC_PT quantization with square weight matrix. "
                        f"weight shape: {weight.shape}, scale shape: {weights_scaling_factor.shape}. "
                        f"Cannot determine if scaling should be applied to input_dim or output_dim."
                    )
                if weights_scaling_factor.shape[-1] == weight.shape[-1]:
                    # (num_experts, input_dim) -> (num_experts, 1, input_dim), BMM-style
                    return (weight / weights_scaling_factor.unsqueeze(-2)).to(torch.float8_e4m3fn)
                elif weights_scaling_factor.shape[-1] == weight.shape[-2]:
                    # (num_experts, output_dim) -> (num_experts, output_dim, 1), Standard MoE case
                    return (weight / weights_scaling_factor.unsqueeze(-1)).to(torch.float8_e4m3fn)
                else:
                    raise ValueError(
                        f"Cannot determine correct unsqueeze dimension for FP8_PC_PT quantization. "
                        f"weight shape: {weight.shape}, scale shape: {weights_scaling_factor.shape}"
                    )
        return (weight / weights_scaling_factor[:, None]).to(torch.float8_e4m3fn)

    if quantization in [QUANTIZATION_INT4_AWQ, QUANTIZATION_W4A8_AWQ]:
        return pack_int4_in_uint8(weight, weights_scaling_factor)

    if quantization in [
        QUANTIZATION_NVFP4,
        QUANTIZATION_NVFP4_AWQ,
        QUANTIZATION_W4A8_NVFP4_FP8,
        QUANTIZATION_NVFP4_SVDQUANT,
    ]:
        assert block_size is not None, "Block size not passed. Unable to quantize to NVFP4 format."
        assert weights_scaling_factor2 is not None, (
            "Weights scaling factor 2 not passed. Unable to quantize to NVFP4 format"
        )
        # If MoE reshape weights_scaling_factor2 to enable quantize operations
        return NVFP4QTensor.quantize(
            weight,
            block_size,
            weights_scaling_factor,
            weights_scaling_factor2.view(-1, 1, 1)
            if weights_scaling_factor2.dim() != 0
            else weights_scaling_factor2,
        )[0]._quantized_data

    if quantization in [QUANTIZATION_W4A8_MXFP4_FP8, QUANTIZATION_MXFP4]:
        return MXFP4QTensor.quantize(weight, block_size=block_size)[0]._quantized_data

    raise NotImplementedError(f"quantization format {quantization} not supported")


def from_quantized_weight(
    weight: torch.Tensor,
    weights_scaling_factor: torch.Tensor,
    quantization: str,
    torch_dtype,
):
    """Converts the quantized weight to the target torch_dtype format."""
    if weight.element_size() >= 2 or weights_scaling_factor is None or not quantization:
        # No need to unquantize the weight.
        return weight.to(torch_dtype)

    if quantization == QUANTIZATION_FP8:
        # safe tensors does not support fp8 yet. So we pack the tensors as int8
        return weight.view(torch.float8_e4m3fn).to(torch_dtype) * weights_scaling_factor.to(
            torch_dtype
        )

    if quantization in [QUANTIZATION_INT8_SQ, QUANTIZATION_INT8_WO]:
        return weight.to(torch_dtype) * weights_scaling_factor[:, None].to(torch_dtype)

    raise NotImplementedError(f"quantization format {quantization} not supported")


def postprocess_state_dict(
    state_dict: dict,
    maxbound: float,
    quantization: str | None,
    is_modelopt_qlora: bool = False,
) -> dict:
    """Filters out keys related to weight quantizers and updates KV cache related keys.

    Args:
        state_dict: The full model state_dict.
        maxbound: The maximum bound value for the output quantizer.
        quantization: The KV cache quantization format.
        is_modelopt_qlora: Whether the model is a modelopt-trained QLoRA model.

    Returns:
        The filtered state_dict without unnecessary keys like '_amax' and non KV cache output quantizers.
    """
    replacements = {
        "k_bmm_quantizer._amax": "k_proj.k_scale",
        "v_bmm_quantizer._amax": "v_proj.v_scale",
        "k_bmm_quantizer._bias_value": "k_proj.k_bias",
        "v_bmm_quantizer._bias_value": "v_proj.v_bias",
        "input_quantizer._pre_quant_scale": "pre_quant_scale",
    }
    skip_keys = [
        "output_quantizer",
        "_amax",
        "_bias_value",
        "input_quantizer._pre_quant_scale",
        "weight_shape",
    ]

    # For modelopt-trained LoRA models, we need to remove the base_layer prefix from the keys for deployment
    if is_modelopt_qlora:
        replacements.update(
            {
                "base_layer.weight": "weight",
                "base_layer.input_scale": "input_scale",
                "base_layer.weight_scale": "weight_scale",
            }
        )
        skip_keys.append("base_layer")

    post_state_dict = {}

    for key, value in state_dict.items():
        # Skip problematic parameters for specific model architectures, e.g., Nemotron Nano VL models
        if key == "vision_model.radio_model.summary_idxs":
            logger.info(f"Removing problematic parameter: {key}")
            continue

        # Skip keys not related to quantizers
        if all(skip_key not in key for skip_key in skip_keys):
            post_state_dict[key] = value
            continue

        # Apply replacements if the key matches any suffix in the replacements dict
        for old_suffix, new_suffix in replacements.items():
            if key.endswith(old_suffix):
                prefix = key[: -len(old_suffix)]

                if "_amax" in key:
                    assert quantization in [KV_CACHE_FP8, KV_CACHE_NVFP4, KV_CACHE_NVFP4_AFFINE], (
                        "Invalid KV cache quantization format."
                    )
                    assert maxbound > 0, "Maxbound must be greater than zero."

                    value = value.float() / maxbound

                    # Warn if scale exceeds threshold
                    if quantization == KV_CACHE_FP8 and value.item() > 0.5:
                        logger.warning(
                            "Large KV activations detected. Quantized KV cache may lead to higher accuracy drop. "
                            "Setting KV cache scaling factor to at least 1."
                        )

                    # Ensure scale is at least 1 for KV_CACHE_FP8
                    # We export real value for KV_CACHE_NVFP4
                    if quantization == KV_CACHE_FP8:
                        value.clamp_(min=1.0)
                post_state_dict[prefix + new_suffix] = value
                break

    # Squeeze scales with a leading dimension of 1
    for key, value in post_state_dict.items():
        if (
            "scale" in key
            and isinstance(value, torch.Tensor)
            and value.dim() == 3
            and value.shape[0] == 1
        ):
            post_state_dict[key] = value.squeeze(0)

    # remove real quant parameters from the state dict
    keys_to_delete = []
    for key, value in post_state_dict.items():
        if any(
            key.endswith("weight_quantizer." + q_key)
            for q_key in RealQuantLinear.list_of_scale_tensors
        ):
            keys_to_delete.append(key)

    # remove LoRA adapters from state dict
    if is_modelopt_qlora:
        for key in post_state_dict:
            if "lora" in key and key not in keys_to_delete:
                keys_to_delete.append(key)
    # Check for tied weights and remove duplicates
    seen_tensors = {}

    # Remove any tied weights if found.
    for key, value in post_state_dict.items():
        if isinstance(value, torch.Tensor):
            # Use tensor data pointer to identify tied weights
            tensor_id = value.data_ptr()
            if tensor_id in seen_tensors:
                # This is a tied weight, mark for deletion and warn
                keys_to_delete.append(key)
                logger.warning(
                    f"Found tied weight: '{key}' is tied to '{seen_tensors[tensor_id]}'. "
                    f"Removing duplicate '{key}' from the exported state dict."
                )
            else:
                seen_tensors[tensor_id] = key

    for key in keys_to_delete:
        del post_state_dict[key]

    return post_state_dict


def all_items_same(item_list):
    """Checks if all elements in the provided list are the same."""
    return all(x == item_list[0] for x in item_list)


def _update_pre_quant_scale(module, new_pre_quant_scale):
    old_pre_quant_scale = module.input_quantizer._pre_quant_scale
    # do the processing in fp32 for numerical stability
    dtype = module.weight.dtype
    module.weight = nn.Parameter(
        (
            module.weight.to(torch.float32)
            * old_pre_quant_scale.to(dtype=torch.float32, device=module.weight.device)
            / new_pre_quant_scale.to(dtype=torch.float32, device=module.weight.device)
        ).to(dtype)
    )
    module.input_quantizer.pre_quant_scale = new_pre_quant_scale

    # Redo weights collection
    module.weight_quantizer.reset_amax()
    enable_stats_collection(module.weight_quantizer)
    module.weight_quantizer(module.weight)
    finish_stats_collection(module.weight_quantizer)


def _update_svdquant(modules, new_pre_quant_scale):
    """Updates the pre_quant_scale, svdquant_lora_a and svdquant_lora_b matrices when pre_quant_scale is changed."""
    new_pre_quant_scale = new_pre_quant_scale.to(torch.float32)
    lora_a = [m.weight_quantizer.svdquant_lora_a.to(torch.float32) for m in modules]
    lora_b = [m.weight_quantizer.svdquant_lora_b.to(torch.float32) for m in modules]
    weight = [m.weight.to(torch.float32) for m in modules]
    old_pre_quant_scale = [m.input_quantizer._pre_quant_scale.to(torch.float32) for m in modules]
    weight = [
        (w + (lb @ la)) * (s / new_pre_quant_scale)
        for w, la, lb, s in zip(weight, lora_a, lora_b, old_pre_quant_scale)
    ]
    weight_concatenated = torch.cat(weight, dim=0)
    lb, la = svd(weight_concatenated, rank=lora_a[0].shape[0])
    weight_concatenated -= lb @ la
    weight_concatenated = weight_concatenated.to(modules[0].weight.dtype)
    la = la.to(modules[0].weight_quantizer.svdquant_lora_a.dtype)
    lb = lb.to(modules[0].weight_quantizer.svdquant_lora_b.dtype)
    new_pre_quant_scale = new_pre_quant_scale.to(modules[0].input_quantizer.pre_quant_scale.dtype)

    index = 0
    for i, module in enumerate(modules):
        module.input_quantizer.pre_quant_scale = new_pre_quant_scale
        module.weight_quantizer.svdquant_lora_a = la
        assert lora_b[i].shape[0] == module.weight.shape[0]
        module.weight_quantizer.svdquant_lora_b = lb[index : index + lora_b[i].shape[0], :]
        module.weight = nn.Parameter(weight_concatenated[index : index + lora_b[i].shape[0], :])
        index += lora_b[i].shape[0]
        # Redo weights collection
        module.weight_quantizer.reset_amax()
        enable_stats_collection(module.weight_quantizer)
        module.weight_quantizer(module.weight)
        finish_stats_collection(module.weight_quantizer)


# Format: (list of target modules, tuple of (linear_to_fuse_into, linear_from_with_scale))
PQS_FUSE_MODULE_MAPPING = [
    # Attention: Fuse o_proj's pre_quant_scale into v_proj's output dimension
    # Mathematical equivalence:
    #   Before: o_proj_out = [attn @ (v_proj_in @ v_proj.W^T)^T * scale] @ o_proj.W^T
    #   After:  o_proj_out = [attn @ (v_proj_in @ (v_proj.W * scale)^T)^T] @ o_proj.W^T
    (["LlamaAttention", "Qwen3Attention", "Qwen3MoeAttention"], ("v_proj", "o_proj")),
    # MLP: Fuse down_proj's pre_quant_scale into up_proj's output dimension
    # Mathematical equivalence:
    #   Before: down_proj_out = {[act_fn(self.gate_proj(x)) * up_proj(x)] * scale} @ down_proj.W^T
    #   After:  down_proj_out = {[act_fn(self.gate_proj(x)) * (up_proj(x) * scale)]} @ down_proj.W^T
    (["LlamaMLP", "Qwen3MLP", "Qwen3MoeMLP"], ("up_proj", "down_proj")),
]


def fuse_prequant_to_linear(model: torch.nn.Module, fuse_grouped_heads=False):
    """Fuse pre_quant_scale to the linear weights if possible.

    Args:
        model: The model to fuse pre_quant_scale to.
        fuse_grouped_heads: If True, fuse the pre_quant_scale even if dimension between pre_quant_scale
            and linear weights is not the same.

    Returns:
        fused_modules: A list of modules of which pre_quant_scale is fused to the previous linear layer.
    """
    # Fuse pre_quant_scale to the linear weights
    for _, module in model.named_modules():
        for module_map in PQS_FUSE_MODULE_MAPPING:
            target_module_list = module_map[0]
            linear_pair = module_map[1]
            if any(module_name in type(module).__name__ for module_name in target_module_list):
                linear_fuse_into = module.get_submodule(linear_pair[0])
                linear_pqs_from = module.get_submodule(linear_pair[1])
                if hasattr(linear_pqs_from, "input_quantizer") and hasattr(
                    linear_pqs_from.input_quantizer, "_pre_quant_scale"
                ):
                    pre_quant_scale = linear_pqs_from.input_quantizer._pre_quant_scale

                    # for GQA/MQA models, we can apply averaging to the pre_quant_scale for shared head groups
                    if pre_quant_scale.numel() != linear_fuse_into.weight.shape[-2]:
                        if (
                            not fuse_grouped_heads
                            or "attention" not in type(module).__name__.lower()
                        ):
                            warn(
                                f"Skipping pattern fuse prequant for {type(module).__name__}"
                                f"pre_quant_scale dim {pre_quant_scale.numel()} != "
                                f"out_channel dim {linear_fuse_into.weight.shape[-2]}"
                            )
                            continue
                        config = module.config
                        num_kv_heads = config.num_key_value_heads
                        kv_head_dim = linear_fuse_into.weight.shape[0] // num_kv_heads
                        n_rep = pre_quant_scale.numel() // num_kv_heads // kv_head_dim

                        # Reshape:(num_kv_heads, n_rep, kv_head_dim)
                        # n_rep is the number of query group
                        averaged_scale = pre_quant_scale.view(
                            num_kv_heads, n_rep, kv_head_dim
                        ).mean(dim=1)

                        # To update o_proj, we need to repeat back to original shape
                        repeated_scale = (
                            averaged_scale.unsqueeze(1)
                            .expand(num_kv_heads, n_rep, kv_head_dim)
                            .reshape(-1)
                        )
                        # Update o_proj's pre_quant_scale
                        _update_pre_quant_scale(linear_pqs_from, repeated_scale)

                        # Use averaged scale (flattened) for v_proj fusion
                        pre_quant_scale = averaged_scale.reshape(-1)

                    # Fuse the pre_quant_scale to weight
                    linear_fuse_into.weight = torch.nn.Parameter(
                        linear_fuse_into.weight * pre_quant_scale.view(-1, 1)
                    )
                    if hasattr(linear_fuse_into, "bias") and linear_fuse_into.bias is not None:
                        linear_fuse_into.bias = torch.nn.Parameter(
                            linear_fuse_into.bias * pre_quant_scale
                        )

                    # Recalibrate the weight quantizer for linear_fuse_into
                    linear_fuse_into.weight_quantizer.reset_amax()
                    enable_stats_collection(linear_fuse_into.weight_quantizer)
                    linear_fuse_into.weight_quantizer(linear_fuse_into.weight)
                    finish_stats_collection(linear_fuse_into.weight_quantizer)

                    delattr(linear_pqs_from.input_quantizer, "_pre_quant_scale")
                    setattr(linear_pqs_from, "fused_with_prequant", True)


def _layernorm_uses_weight_plus_one(module: torch.nn.Module) -> bool:
    if any(
        name in type(module).__name__
        for name in ["LayerNorm1P", "GemmaRMSNorm", "Gemma2RMSNorm", "Gemma3RMSNorm"]
    ):
        return True

    return bool(hasattr(module, "zero_centered_gamma") and module.zero_centered_gamma)


def fuse_prequant_layernorm(
    layernorm_module: torch.nn.Module,
    modules: list[torch.Tensor],
):
    """Scales layernorm weights with avg_pre_quant_scale of the modules list and sets pre_quant_scales to be deleted.

    original:
        layernorm_output = (normalization(input) * weight) + bias
        layernorm_output_scaled = layernorm_output * pre_quant_scale

    fused:
        fused_weight = weight * avg_pre_quant_scale
        fused_bias = bias * avg_pre_quant_scale
        layernorm_output_scaled = (normalization(input) * fused_weight) + fused_bias
    """
    pre_quant_scale = getattr(modules[0].input_quantizer, "_pre_quant_scale").to(
        layernorm_module.weight.device
    )
    if _layernorm_uses_weight_plus_one(layernorm_module):
        # For norms that use (1 + weight) in forward, fold pre_quant_scale into the effective weight.
        fused_weight = (layernorm_module.weight + 1.0) * pre_quant_scale - 1.0
    else:
        fused_weight = layernorm_module.weight * pre_quant_scale
    layernorm_module.weight = torch.nn.Parameter(fused_weight.to(layernorm_module.weight.dtype))
    if hasattr(layernorm_module, "bias") and layernorm_module.bias is not None:
        layernorm_module.bias = torch.nn.Parameter(layernorm_module.bias * pre_quant_scale)
    # Pre_quant_scales of modules must not be exported, since they have been fused with layernorm
    for module in modules:
        delattr(module.input_quantizer, "_pre_quant_scale")
        setattr(module, "fused_with_prequant", True)


def preprocess_linear_fusion(modules: list[torch.nn.Module], resmooth_only=False):
    """Preprocess the quantized linears that we plan to fuse.

    Use resmooth_only for MOE experts as each individual expert is not fused.
    """
    quantization_format_list = [get_quantization_format(module) for module in modules]
    assert all_items_same(quantization_format_list), "Modules have different quantization formats"

    # Activation
    if hasattr(modules[0], "input_quantizer"):
        # Resmooth
        if modules[0].input_quantizer.pre_quant_scale is not None:
            avg_prequant_scale = torch.mean(
                torch.stack([module.input_quantizer.pre_quant_scale for module in modules]),
                dim=0,
            )

            if all(
                getattr(m.weight_quantizer, "svdquant_lora_a", None) is not None for m in modules
            ):
                _update_svdquant(modules, avg_prequant_scale)
            else:
                for module in modules:
                    if not torch.equal(module.input_quantizer.pre_quant_scale, avg_prequant_scale):
                        _update_pre_quant_scale(module, avg_prequant_scale)

        if resmooth_only:
            return

        if modules[0].input_quantizer.is_enabled and modules[0].input_quantizer.amax is not None:
            assert modules[0].input_quantizer.amax.numel() == 1, (
                "Only support scalar input quant amax"
            )

            input_amax = torch.max(torch.stack([module.input_quantizer.amax for module in modules]))
            for module in modules:
                module.input_quantizer.amax = input_amax

    # Weight
    if hasattr(modules[0], "weight_quantizer"):
        is_seq_quant = isinstance(modules[0].weight_quantizer, SequentialQuantizer)

        if is_seq_quant:
            if modules[0].weight_quantizer[-1].is_enabled:
                assert len(modules[0].weight_quantizer) == 2
                weight_amax = torch.max(
                    torch.stack([module.weight_quantizer[-1].amax for module in modules])
                )
                for module in modules:
                    module.weight_quantizer[-1].amax = weight_amax

        elif (
            modules[0].weight_quantizer.is_enabled
            and modules[0].weight_quantizer.amax is not None
            and modules[0].weight_quantizer.amax.numel() == 1
        ):
            weight_amax = torch.max(
                torch.stack([module.weight_quantizer.amax for module in modules])
            )
            for module in modules:
                module.weight_quantizer.amax = weight_amax


def get_quant_config(
    model: nn.Module,
    is_modelopt_qlora: bool = False,
) -> dict[str, Any]:
    """Generate quantization config for a model.

    The model should be the root model. It can be fully quantized, partially quantized or
    mixed-precision quantized.

    Args:
        model: The PyTorch model to make config for.
        is_modelopt_qlora: Whether the model is a modelopt-trained QLoRA model.

    Returns:
        Dictionary containing the quantization configuration
    """
    # Find first quantized linear layer to determine quantization format
    quantization_format = None
    block_size = None

    # Create base config
    quant_config = {
        "producer": {
            "name": "modelopt",
            "version": __version__,
        },
    }

    default_quantization = {
        "quant_algo": None,
        "kv_cache_quant_algo": None,
    }

    quant_config["quantization"] = default_quantization

    # Layer config dict holds quantization format of each layer.
    # It also holds awq_block_size information for applicable layers.
    layer_config_dict = {}

    kv_cache_format = QUANTIZATION_NONE
    for name, module in dict(model.named_modules()).items():
        # Check for standard quantizers or any quantizers from weight attributes
        weight_names = list(weight_attr_names(module))
        has_quantizers = any(
            hasattr(module, quantizer_attr_names(weight_name).weight_quantizer)
            or hasattr(module, quantizer_attr_names(weight_name).input_quantizer)
            for weight_name in weight_names
        )

        # Skip LORA module and adapters.
        # ModelOpt does not currently quantize these layers in QLoRA path.
        skip_layer = is_modelopt_qlora and (
            hasattr(module, "base_layer") or "lora_A" in name or "lora_B" in name
        )

        if has_quantizers and not skip_layer:
            quantization_format = get_quantization_format(module)

            # For MoE expert modules, we need to extract block size from the correct weight quantizer
            # Try to get block size from each weight attribute (e.g., gate_up_proj, down_proj)
            block_size = 0

            for weight_name in weight_names:
                weight_block_size = get_weight_block_size(module, weight_name)
                if weight_block_size > 0:
                    block_size = weight_block_size
                    break

            # Fallback to default weight quantizer if no specific weight quantizer found
            if block_size == 0:
                block_size = get_weight_block_size(module)

            # Construct per layer config dictionary
            layer_config_dict[name + ".quantization"] = quantization_format
            layer_config_dict[name + ".awq_block_size"] = block_size

        not_enabled = SimpleNamespace(is_enabled=False)

        # Find kv cache quant format
        if (
            getattr(module, "k_bmm_quantizer", not_enabled).is_enabled
            or getattr(module, "v_bmm_quantizer", not_enabled).is_enabled
            or getattr(module, "output_quantizer", not_enabled).is_enabled
        ):
            module_kv_quant = get_kv_cache_dtype(module)
            if kv_cache_format == QUANTIZATION_NONE:
                kv_cache_format = module_kv_quant
            else:
                assert kv_cache_format == module_kv_quant, (
                    "Do not support mixed precision kv cache quantization"
                )

    # Process per layer quantization config dict
    quant_config["quantization"].update(process_layer_quant_config(layer_config_dict))

    if kv_cache_format is not None:
        quant_config["quantization"]["kv_cache_quant_algo"] = kv_cache_format

    return quant_config


def has_quantized_modules(model: nn.Module) -> bool:
    """Check if a model has any quantized modules.

    Args:
        model: The model to check.

    Returns:
        True if the model contains quantized modules, False otherwise.
    """
    return any(
        get_quantization_format(sub_module) != QUANTIZATION_NONE
        for _, sub_module in model.named_modules()
    )
