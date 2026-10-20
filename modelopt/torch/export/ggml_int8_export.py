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

"""Rewrite a GGML Hugging Face export as an INT8 weight-only compressed-tensors checkpoint."""

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY, GGMLFormat
from modelopt.torch.utils.plugins.hf_checkpoint_utils import indexed_weight_map

__all__ = ["convert_ggml_export_to_int8"]


def convert_ggml_export_to_int8(export_dir: str | os.PathLike) -> None:
    """Rewrite a GGML export in place as INT8 weight-only compressed-tensors, which vLLM serves.

    Every GGML weight is an integer level of at most 8 bits times the step of its 16- or
    32-weight sub-block, so storing those levels with the steps as group scales reproduces the
    GGML decoding up to rounding the steps to the model dtype. Unquantized tensors are kept as
    they are. The checkpoint is validated before anything is rewritten.

    Args:
        export_dir: A directory written by :func:`export_hf_checkpoint` whose quantized layers
            are all GGML formats.
    """
    export_dir = Path(export_dir)
    config = json.loads((export_dir / "config.json").read_text())
    quant_config_path = export_dir / "hf_quant_config.json"
    quantization = json.loads(quant_config_path.read_text())["quantization"]
    scale_dtype = getattr(torch, config.get("dtype") or config["torch_dtype"])
    weight_map = indexed_weight_map(export_dir)
    layer_formats = _ggml_layer_formats(export_dir, weight_map, quantization)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for shard in sorted({weight_map[f"{layer}.weight"] for layer in layer_formats}):
        _convert_shard(export_dir / shard, layer_formats, scale_dtype, device)

    for layer in layer_formats:
        shard = weight_map.pop(f"{layer}.weight")
        weight_map.update(
            {
                f"{layer}.{suffix}": shard
                for suffix in ("weight_packed", "weight_scale", "weight_shape")
            }
        )
    index_path = export_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index["weight_map"] = dict(sorted(weight_map.items()))
        index_path.write_text(json.dumps(index, indent=2))

    config["quantization_config"] = _compressed_tensors_config(layer_formats)
    (export_dir / "config.json").write_text(json.dumps(config, indent=4))
    # It describes the GGML payloads this replaced, and runtimes that read it would misload.
    quant_config_path.unlink()


def _ggml_layer_formats(
    export_dir: Path, weight_map: dict[str, str], quantization: dict
) -> dict[str, GGMLFormat]:
    """The GGML format of every layer holding a packed payload, refusing anything else."""
    if quantization.get("kv_cache_quant_algo") is not None:
        raise ValueError("INT8 conversion of a GGML export does not carry KV cache quantization.")
    mixed = quantization.get("quant_algo") == "MIXED_PRECISION"
    layer_configs = quantization.get("quantized_layers", {}) if mixed else {}
    algos = (
        {cfg.get("quant_algo") for cfg in layer_configs.values()}
        if mixed
        else {quantization.get("quant_algo")}
    )
    unsupported = sorted(str(a) for a in algos if str(a).lower() not in GGML_FORMAT_REGISTRY)
    if unsupported:
        raise ValueError(f"Only GGML layers convert to INT8, but the export has {unsupported}.")

    layer_formats = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(export_dir / shard, framework="pt") as f:
            # GGML payloads are the only uint8 weights an export writes.
            payloads = [
                name
                for name in f.keys()  # noqa: SIM118 -- a safe_open handle is not a dict
                if name.endswith(".weight") and f.get_slice(name).get_dtype() == "U8"
            ]
        for name in payloads:
            layer = name.removesuffix(".weight")
            algo = layer_configs[layer]["quant_algo"] if mixed else quantization["quant_algo"]
            layer_formats[layer] = GGML_FORMAT_REGISTRY[algo.lower()]
    if not layer_formats:
        raise ValueError(f"{export_dir} holds no GGML weights to convert.")
    return layer_formats


def _convert_shard(
    path: Path, layer_formats: dict[str, GGMLFormat], scale_dtype: torch.dtype, device: str
) -> None:
    tensors = {}
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()
        for name in f.keys():  # noqa: SIM118 -- a safe_open handle is not a dict
            layer = name.removesuffix(".weight")
            ggml_format = layer_formats.get(layer) if name.endswith(".weight") else None
            if ggml_format is None:
                tensors[name] = f.get_tensor(name)
                continue
            payload = f.get_tensor(name).to(device)
            shape = torch.tensor((*payload.shape[:-2], payload.shape[-2] * ggml_format.block_size))
            levels, steps = ggml_format.decode_int8(payload, shape)
            # compressed-tensors packs 8-bit values offset by 128, four little-endian per int32.
            tensors[f"{layer}.weight_packed"] = (
                (levels.view(torch.uint8) ^ 0x80).view(torch.int32).cpu()
            )
            tensors[f"{layer}.weight_scale"] = steps.to(scale_dtype).cpu()
            tensors[f"{layer}.weight_shape"] = shape
    tmp_path = path.with_suffix(".tmp")
    save_file(tensors, tmp_path, metadata=metadata)
    os.replace(tmp_path, path)


def _compressed_tensors_config(layer_formats: dict[str, GGMLFormat]) -> dict:
    """One config group per INT8 group size, targeting its layers by name."""
    layers_by_group_size: dict[int, list[str]] = {}
    for layer, ggml_format in sorted(layer_formats.items()):
        layers_by_group_size.setdefault(ggml_format.int8_group_size, []).append(layer)
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
