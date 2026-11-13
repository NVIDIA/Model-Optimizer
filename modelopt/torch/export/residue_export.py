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

"""Native FP4 weight export with the ResServe residue metadata contract."""

from collections import Counter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelopt.torch.quantization.nn import ResidueQuantLinear, TensorQuantizer
from modelopt.torch.quantization.utils.core_utils import has_accelerate_offload
from modelopt.torch.utils.distributed import is_fsdp2_model

from .quantized_weight_export import (
    capture_quantized_weight_export_state,
    export_quantized_weight_tensors,
)

__all__: list[str] = []


def _residue_layer_metadata(module):
    """Validate one residue layer and describe its deployment channel layout."""
    weight = module.weight
    if weight.is_meta or hasattr(weight, "to_local") or not weight.is_floating_point():
        raise NotImplementedError("Residue export requires resident, uncompressed weights.")
    width = module.in_features
    indices = module.input_quantizer._salient_indices
    if indices is None:
        raise ValueError("Calibrate residue channels before export.")
    if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Residue channels must be a one-dimensional integer tensor.")
    selected = indices.tolist()
    if selected != sorted(set(selected)) or any(i < 0 or i >= width for i in selected):
        raise ValueError("Residue channels must be sorted, unique and in range.")
    full = selected == list(range(width))
    mode = "mext_r1" if full else "extended_k" if selected else "standard"
    formats = []
    for quantizer in (module.input_quantizer, module.weight_quantizer):
        blocks = quantizer.block_sizes or {}
        bits = blocks.get("scale_bits")
        if (
            not isinstance(quantizer, TensorQuantizer)
            or not quantizer.is_enabled
            or not quantizer.fake_quant
            or quantizer.num_bits != (2, 1)
            or blocks.get("type") != "dynamic"
            or bits not in ((4, 3), (8, 0))
            or blocks.get(-1) != (16 if bits == (4, 3) else 32)
            or quantizer.backend is not None
        ):
            raise NotImplementedError("Residue export requires paired dynamic NVFP4 or MXFP4.")
        if quantizer.pre_quant_scale is not None:
            raise NotImplementedError("Fuse pre_quant_scale before residue deployment export.")
        if bits == (4, 3) and (quantizer.amax is None or quantizer.amax.numel() != 1):
            raise ValueError("NVFP4 residue export requires calibrated scalar amax values.")
        formats.append(bits)
    if formats[0] != formats[1]:
        raise ValueError("Residue activation and weight formats must match.")
    if width % 128 or module.out_features % 32:
        raise ValueError(
            "Residue deployment requires input width divisible by 128 and output by 32."
        )
    if selected and not full:
        per_block = len(selected) * 8 // width
        if (
            per_block not in (1, 2, 4)
            or len(selected) != width // 8 * per_block
            or any(value // 8 != position // per_block for position, value in enumerate(selected))
        ):
            raise ValueError(
                "Residue deployment requires 1, 2 or 4 selected channels per block of 8."
            )
    if hasattr(module, "_original_in_features") or weight.shape != (module.out_features, width):
        raise ValueError("Residue weight dimensions do not match the calibrated channels.")
    if getattr(module, "output_quantizer", None) is not None and module.output_quantizer.is_enabled:
        raise NotImplementedError("Residue export does not support output quantization.")
    return {"salient_indices": selected, "num_salient": len(selected), "runtime_mode": mode}


def collect_residue_export_metadata(model):
    """Validate export eligibility and shared channels without mutating the model."""
    residue = {name: m for name, m in model.named_modules() if isinstance(m, ResidueQuantLinear)}
    if not residue:
        return {}
    if (
        is_fsdp2_model(model)
        or has_accelerate_offload(model)
        or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
    ):
        raise NotImplementedError("Residue export requires a resident, single-process model.")
    uses = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
    if any(uses[id(m.weight)] > 1 for m in residue.values()):
        raise NotImplementedError("Residue export does not support tied weights.")
    layers = {name: _residue_layer_metadata(m) for name, m in residue.items()}
    for name, module in residue.items():
        group = getattr(module, "_residue_group", ())
        if group and (
            name not in group
            or any(
                other not in layers
                or tuple(getattr(residue[other], "_residue_group", ())) != tuple(group)
                or residue[other].in_features != module.in_features
                or residue[other].input_quantizer.block_sizes != module.input_quantizer.block_sizes
                or layers[other] != layers[name]
                for other in group
            )
        ):
            raise ValueError(f"Incompatible declared residue input group: {group}")
    extended = {}
    for name, m in residue.items():
        layer = layers[name]
        if layer["runtime_mode"] != "extended_k":
            continue
        width = m.in_features + layer["num_salient"]
        if m.weight_quantizer.block_sizes["scale_bits"] == (8, 0):
            width += -width % 128
        extended[name] = width
    return {"layers": layers, "extended_linear_dims": extended}


@torch.no_grad()
def export_residue_weight(module, dtype):
    """Pack residue weights with native FP4 scales, leaving the live module unchanged."""
    metadata = _residue_layer_metadata(module)
    weight = module.weight.detach()
    if metadata["runtime_mode"] == "extended_k":
        weight = torch.cat(
            [weight, weight.index_select(-1, module.input_quantizer._salient_indices)], dim=-1
        )
    is_mx = module.weight_quantizer.block_sizes["scale_bits"] == (8, 0)
    padding = -weight.shape[-1] % 128 if is_mx else 0
    if padding:
        weight = F.pad(weight, (0, padding))
    # A tensor-only view reuses native packing without exporting residue as an ordinary Linear.
    view = nn.Module()
    view.register_buffer("weight", weight)
    view.input_quantizer = module.input_quantizer
    view.weight_quantizer = module.weight_quantizer
    if not is_mx:
        # Normalize zero amax using the native export convention without changing QDQ state.
        view.weight_quantizer = deepcopy(module.weight_quantizer)
        view.weight_quantizer.amax = module.weight_quantizer.export_amax()
    # Bound native packing temporaries while retaining the calibrated global scales.
    rows = max(1, 2**22 // weight.shape[-1])
    parts = []
    for start in range(0, weight.shape[0], rows):
        view.weight = weight[start : start + rows]
        state = capture_quantized_weight_export_state(view, cpu=False)
        assert state is not None
        parts.append(export_quantized_weight_tensors(view.weight, state, dtype))
    result = parts[0]
    if len(parts) > 1:
        for key in ("weight", "weight_scale"):
            result[key] = torch.cat([part[key] for part in parts], dim=0)
    return result
