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

"""Export GGML-quantized HF models as INT8 weight-only compressed-tensors checkpoints for vLLM."""

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY, GGMLFormat
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils.core_utils import has_accelerate_offload
from modelopt.torch.utils.distributed import is_fsdp2_model
from modelopt.torch.utils.plugins.hf_checkpoint_utils import (
    copy_off_index_safetensors,
    indexed_weight_map,
)

from ..unified_export_hf import _copy_non_model_files_from_source, _source_checkpoint
from .vllm_fakequant_hf import _carry_over_unplaced_weights, _copy_source_config

__all__ = ["export_hf_ggml_int8_checkpoint"]


def export_hf_ggml_int8_checkpoint(model: nn.Module, export_dir: Path | str) -> None:
    """Export a GGML-quantized model as an INT8 weight-only compressed-tensors checkpoint.

    Every GGML weight is an integer level of at most 8 bits times the step of its 16- or
    32-weight sub-block, so storing those levels with the steps as group scales reproduces the
    GGML weights up to rounding the steps to the model dtype, and vLLM serves the checkpoint
    with its integer weight-only kernels. Unquantized weights are written as they are. Every
    enabled quantizer must be a GGML weight quantizer, which is checked before anything is
    written.

    Args:
        model: In-memory model quantized with GGML weight formats.
        export_dir: Output directory.
    """
    if has_accelerate_offload(model) or is_fsdp2_model(model):
        raise NotImplementedError(
            "GGML INT8 export needs every weight resident; offloaded and FSDP2 models are not "
            "supported."
        )
    ggml_layers = _ggml_layers(model)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {k: v for k, v in model.state_dict().items() if "quantizer" not in k}
    with torch.inference_mode():
        for name, (weight, quantizer, ggml_format) in ggml_layers.items():
            shape = torch.tensor(weight.shape)
            # pack() returns the payload fake quant decoded, including codes GPTQ pinned.
            levels, steps = ggml_format.decode_int8(ggml_format.pack(weight, quantizer), shape)
            del state_dict[f"{name}.weight"]
            # compressed-tensors packs 8-bit values offset by 128, four little-endian per int32.
            state_dict[f"{name}.weight_packed"] = (
                (levels.view(torch.uint8) ^ 0x80).view(torch.int32).cpu()
            )
            state_dict[f"{name}.weight_scale"] = steps.to(weight.dtype).cpu()
            state_dict[f"{name}.weight_shape"] = shape
    model.save_pretrained(export_dir, state_dict=state_dict, save_modelopt_state=False)
    _carry_over_unplaced_weights(export_dir, model)
    copy_off_index_safetensors(_source_checkpoint(model), export_dir)
    _copy_non_model_files_from_source(model, export_dir)
    _copy_source_config(model, export_dir)

    config_path = export_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = _compressed_tensors_config(export_dir)
    config_path.write_text(json.dumps(config, indent=4))


def _ggml_layers(model: nn.Module) -> dict[str, tuple[torch.Tensor, TensorQuantizer, GGMLFormat]]:
    """Each GGML-quantized layer's weight, quantizer and format, refusing any other quantizer."""
    layers, unsupported = {}, []
    for name, quantizer in model.named_modules():
        if not (isinstance(quantizer, TensorQuantizer) and quantizer.is_enabled):
            continue
        layer, _, attr = name.rpartition(".")
        weight = getattr(model.get_submodule(layer), "weight", None)
        num_bits = quantizer.num_bits
        ggml_format = GGML_FORMAT_REGISTRY.get(num_bits) if isinstance(num_bits, str) else None
        if attr != "weight_quantizer" or ggml_format is None or weight is None or weight.dim() != 2:
            unsupported.append(name)
            continue
        layers[layer] = (weight, quantizer, ggml_format)
    if unsupported:
        raise ValueError(
            "GGML INT8 export writes only GGML weight quantizers of 2-D weights, but these "
            f"quantizers are enabled too: {unsupported[:8]}"
        )
    if not layers:
        raise ValueError("The model has no GGML-quantized weights to export.")
    return layers


def _compressed_tensors_config(export_dir: Path) -> dict:
    """One config group per INT8 group size, targeting the layers by their saved names.

    The names are read back from the checkpoint because saving can rename modules; vLLM maps
    explicit layer names onto its own modules, but not ``re:`` patterns.
    """
    shapes = {}
    for shard in sorted(set(indexed_weight_map(export_dir).values())):
        with safe_open(export_dir / shard, framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 -- a safe_open handle is not a dict
                if key.endswith((".weight_packed", ".weight_scale")):
                    shapes[key] = f.get_slice(key).get_shape()
    layers_by_group_size: dict[int, list[str]] = {}
    for key, shape in sorted(shapes.items()):
        if key.endswith(".weight_packed"):
            layer = key.removesuffix(".weight_packed")
            group_size = shape[-1] * 4 // shapes[f"{layer}.weight_scale"][-1]
            layers_by_group_size.setdefault(group_size, []).append(layer)
    config_groups = {
        f"group_{i}": {
            "targets": layers,
            "weights": {
                "num_bits": 8,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": group_size,
                "dynamic": False,
            },
            "input_activations": None,
            "output_activations": None,
            "format": "pack-quantized",
        }
        for i, (group_size, layers) in enumerate(sorted(layers_by_group_size.items()))
    }
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": config_groups,
        "ignore": [],
    }
