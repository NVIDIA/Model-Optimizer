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

"""Export MiniMax-M3 with an MXFP8 base and NVFP4 routed experts.

Non-routed-expert tensors are copied unchanged from the vendor MXFP8
checkpoint. Routed experts are quantized directly from the BF16 checkpoint,
one MoE layer at a time, to avoid loading the complete model or double
quantizing the vendor weights.

Usage:
    python hf_ptq_mixed_mxfp8_nvfp4.py \\
        --mxfp8_ckpt /models/minimax-m3-mxfp8 \\
        --bf16_ckpt /models/minimax-m3-bf16 \\
        --recipe huggingface/minimax_m3_vl/ptq/nvfp4_experts_only \\
        --output_ckpt /workspace/quant/minimax-m3-mxfp8-nvfp4-mixed \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

import modelopt.torch.quantization as mtq
from modelopt import __version__
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

BLOCK_SIZE = 16

_EXPERT_WEIGHT_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>w[123])\.weight$"
)
_EXPERT_TENSOR_RE = re.compile(
    r"^language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w[123]\."
)


def _log(message: str) -> None:
    print(message, flush=True)


def _load_index(checkpoint: Path) -> dict[str, str]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    return index["weight_map"]


def _expert_projection(weight: torch.Tensor) -> nn.Linear:
    output_features, input_features = weight.shape
    linear = nn.Linear(
        input_features,
        output_features,
        bias=False,
        device=weight.device,
        dtype=weight.dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    return linear


class _ExpertLayerModel(nn.Module):
    """One MoE layer with module paths matching the expert-only recipe."""

    def __init__(self, weights: dict[tuple[int, str], torch.Tensor]):
        super().__init__()
        self.block_sparse_moe = nn.Module()
        self.block_sparse_moe.experts = nn.ModuleDict()
        for (expert, projection), weight in weights.items():
            expert_key = str(expert)
            if expert_key not in self.block_sparse_moe.experts:
                self.block_sparse_moe.experts[expert_key] = nn.Module()
            setattr(
                self.block_sparse_moe.experts[expert_key],
                projection,
                _expert_projection(weight),
            )

    def projection(self, expert: int, name: str) -> nn.Linear:
        return getattr(self.block_sparse_moe.experts[str(expert)], name)


def _pack_nvfp4(
    weight: torch.Tensor,
    quantizer: nn.Module,
    weight_scale_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_scale, weight_scale_2 = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
        quantizer, weight, weight_scale_2
    )
    result = NVFP4QTensor.quantize(weight, BLOCK_SIZE, weight_scale, weight_scale_2)
    quantized = result[0] if isinstance(result, tuple) else result
    return quantized._quantized_data, weight_scale, weight_scale_2


def _quantize_layer(
    weights: dict[tuple[int, str], torch.Tensor],
    quantize_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    model = _ExpertLayerModel(weights)
    mtq.quantize(model, quantize_config, forward_loop=lambda _: None)

    output: dict[str, torch.Tensor] = {}
    for expert in sorted({expert for expert, _ in weights}):
        w1_quantizer = model.projection(expert, "w1").weight_quantizer
        w3_quantizer = model.projection(expert, "w3").weight_quantizer
        w1_scale_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(w1_quantizer)
        w3_scale_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(w3_quantizer)
        shared_w13_scale_2 = torch.maximum(w1_scale_2.reshape(()), w3_scale_2.reshape(()))

        for projection in ("w1", "w2", "w3"):
            linear = model.projection(expert, projection)
            quantizer = linear.weight_quantizer
            if projection in ("w1", "w3"):
                weight_scale_2 = shared_w13_scale_2
            else:
                weight_scale_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(
                    quantizer
                ).reshape(())

            packed, weight_scale, weight_scale_2 = _pack_nvfp4(
                linear.weight, quantizer, weight_scale_2
            )
            key = f"experts.{expert}.{projection}"
            output[f"{key}.weight"] = packed.cpu()
            output[f"{key}.weight_scale"] = weight_scale.cpu()
            output[f"{key}.weight_scale_2"] = weight_scale_2.cpu().reshape(())

            output[f"{key}.input_scale"] = torch.tensor(1.0, dtype=torch.float32).reshape(())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _is_routed_expert_tensor(key: str) -> bool:
    return bool(_EXPERT_TENSOR_RE.match(key))


def _build_quant_config(
    mxfp8_map: dict[str, str],
    nvfp4_expert_modules: list[str],
    exclude_modules: list[str],
) -> dict[str, Any]:
    quantized_layers: dict[str, dict[str, Any]] = {}
    for key in mxfp8_map:
        if not key.endswith(".weight_scale_inv"):
            continue
        module = key.removesuffix(".weight_scale_inv")
        if "block_sparse_moe.experts." not in module:
            quantized_layers[module] = {"quant_algo": "MXFP8"}

    for module in nvfp4_expert_modules:
        quantized_layers[module] = {"quant_algo": "NVFP4", "group_size": BLOCK_SIZE}
    return {
        "producer": {"name": "modelopt", "version": __version__},
        "quant_method": "modelopt",
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "quant_method": "modelopt",
            "kv_cache_quant_algo": None,
            "exclude_modules": exclude_modules,
            "quantized_layers": quantized_layers,
        },
    }


def _load_layer_weights(
    checkpoint: Path,
    weight_map: dict[str, str],
    keys: list[str],
    device: str,
) -> dict[tuple[int, str], torch.Tensor]:
    weights: dict[tuple[int, str], torch.Tensor] = {}
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        keys_by_shard[weight_map[key]].append(key)

    for shard, shard_keys in keys_by_shard.items():
        with safe_open(str(checkpoint / shard), framework="pt", device="cpu") as handle:
            for key in shard_keys:
                match = _EXPERT_WEIGHT_RE.match(key)
                if match is None:
                    continue
                expert = int(match.group("expert"))
                projection = match.group("projection")
                weights[(expert, projection)] = handle.get_tensor(key).to(device)
    return weights


def _quantize_experts(
    bf16: Path,
    destination: Path,
    weight_map: dict[str, str],
    quantize_config: dict[str, Any],
    device: str,
) -> tuple[dict[str, str], list[str]]:
    keys_by_layer: dict[int, list[str]] = defaultdict(list)
    for key in weight_map:
        match = _EXPERT_WEIGHT_RE.match(key)
        if match:
            keys_by_layer[int(match.group("layer"))].append(key)
    if not keys_by_layer:
        raise ValueError(f"No routed-expert weights found in {bf16}")

    new_index: dict[str, str] = {}
    expert_modules: list[str] = []
    layers = sorted(keys_by_layer)
    _log(f"[mixed] quantizing {len(layers)} BF16 MoE layers to NVFP4")
    for layer_index, layer in enumerate(layers, start=1):
        weights = _load_layer_weights(bf16, weight_map, keys_by_layer[layer], device)
        quantized = _quantize_layer(weights, quantize_config)
        tensors = {
            f"language_model.model.layers.{layer}.block_sparse_moe.{key}": tensor
            for key, tensor in quantized.items()
        }
        shard_name = f"experts-layer-{layer:03d}.safetensors"
        save_file(tensors, str(destination / shard_name))
        for key in tensors:
            new_index[key] = shard_name
            if key.endswith(".weight"):
                expert_modules.append(key.removesuffix(".weight"))
        _log(
            f"[mixed] layer {layer} ({layer_index}/{len(layers)}): "
            f"wrote {len(tensors)} NVFP4 tensors"
        )
    return new_index, expert_modules


def _copy_mxfp8_base(
    checkpoint: Path,
    destination: Path,
    weight_map: dict[str, str],
    new_index: dict[str, str],
) -> None:
    for shard_index, shard in enumerate(sorted(set(weight_map.values()))):
        tensors = {}
        with safe_open(str(checkpoint / shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118
                if not _is_routed_expert_tensor(key):
                    tensors[key] = handle.get_tensor(key)
        if not tensors:
            continue

        output_name = f"base-mxfp8-{shard_index:05d}.safetensors"
        save_file(tensors, str(destination / output_name))
        for key in tensors:
            new_index[key] = output_name
        _log(f"[mixed] base shard {shard} -> {output_name}: {len(tensors)} tensors")


def _copy_ancillary_files(source: Path, destination: Path) -> None:
    generated_files = {
        "config.json",
        "hf_quant_config.json",
        "model.safetensors.index.json",
    }
    for item in source.iterdir():
        if item.name in generated_files or item.name.endswith(".safetensors"):
            continue
        if item.is_file():
            shutil.copy2(item, destination / item.name)


def _rename_checkpoint_shards(destination: Path, weight_map: dict[str, str]) -> dict[str, str]:
    shard_names = list(dict.fromkeys(weight_map.values()))
    renamed_shards = {
        name: f"model-{index:05d}-of-{len(shard_names):05d}.safetensors"
        for index, name in enumerate(shard_names, start=1)
    }
    for old_name, new_name in renamed_shards.items():
        (destination / old_name).replace(destination / new_name)
    return {key: renamed_shards[shard] for key, shard in weight_map.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mxfp8_ckpt", required=True, help="vendor MiniMax-M3-MXFP8 checkpoint")
    parser.add_argument("--bf16_ckpt", required=True, help="BF16 source for routed experts")
    parser.add_argument("--recipe", required=True, help="expert-only NVFP4 recipe")
    parser.add_argument("--output_ckpt", required=True, help="mixed checkpoint output")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mxfp8 = Path(args.mxfp8_ckpt)
    bf16 = Path(args.bf16_ckpt)
    destination = Path(args.output_ckpt)
    destination.mkdir(parents=True, exist_ok=True)

    recipe = load_recipe(args.recipe)
    quantize_config = recipe.quantize.model_dump()
    mxfp8_map = _load_index(mxfp8)
    bf16_map = _load_index(bf16)

    new_index, expert_modules = _quantize_experts(
        bf16,
        destination,
        bf16_map,
        quantize_config,
        args.device,
    )
    _copy_mxfp8_base(mxfp8, destination, mxfp8_map, new_index)
    new_index = _rename_checkpoint_shards(destination, new_index)

    mxfp8_config = json.loads((mxfp8 / "config.json").read_text())
    vendor_quantization = mxfp8_config.get("quantization_config", {})
    mixed_quant_config = _build_quant_config(
        mxfp8_map,
        expert_modules,
        list(vendor_quantization.get("ignored_layers", []) or []),
    )
    mxfp8_config["quantization_config"] = mixed_quant_config["quantization"]

    (destination / "config.json").write_text(json.dumps(mxfp8_config, indent=2))
    (destination / "hf_quant_config.json").write_text(json.dumps(mixed_quant_config, indent=2))
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": "pt"}, "weight_map": new_index}, indent=2)
    )
    _copy_ancillary_files(mxfp8, destination)
    _log(f"[mixed] done -> {destination}")


if __name__ == "__main__":
    main()
