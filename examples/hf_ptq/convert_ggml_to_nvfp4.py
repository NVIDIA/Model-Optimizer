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

"""Transcode a GGML checkpoint exported by hf_ptq.py into an NVFP4 checkpoint that vLLM serves.

python hf_ptq.py --pyt_ckpt_path <model> --recipe general/ptq/iq1_s_w1a4 --export_path <ggml_dir>
python convert_ggml_to_nvfp4.py <ggml_dir> <nvfp4_dir>
vllm serve <nvfp4_dir>

An export with activation scales (the *_w1a4 and *_w2a4 recipes) becomes NVFP4 W4A4, one without
them NVFP4 weight-only (W4A16_NVFP4). Either evaluates the GGML weights as an FP4 GEMM computes
with them: each decoded weight is transcoded to NVFP4 the way those recipes fake-quantize it.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from convert_ggml_to_int8 import ggml_layer_formats, write_converted_checkpoint
from safetensors import safe_open

from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.quantization.ggml import GGMLFormat
from modelopt.torch.quantization.ggml.nvfp4 import nvfp4_global_scale, nvfp4_transcode_scales
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.utils.plugins.hf_checkpoint_utils import indexed_weight_map

# Projections vLLM fuses into one GEMM, which takes a single per-tensor weight scale.
_FUSED_PROJECTIONS = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("w1", "w3"),
    ("in_proj_qkv", "in_proj_z"),
)


def convert_ggml_export_to_nvfp4(
    ggml_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    max_shard_size: int = 10 * 2**30,
) -> None:
    """Write a GGML export as a ModelOpt NVFP4 checkpoint, W4A4 if it holds activation scales.

    Every GGML weight is decoded and transcoded to NVFP4 as the ``transcode: nvfp4`` fake quant of
    the ``*_w1a4`` and ``*_w2a4`` recipes does, so the checkpoint computes what calibration
    simulated. The one difference: projections vLLM fuses share the largest of their power-of-two
    per-tensor scales, which moves only blocks near the bottom of E4M3's range. Activation scales,
    unquantized tensors and the other files are copied as they are; the GGML export is left
    unchanged.

    Args:
        ggml_dir: A Hugging Face export whose quantized layers are all GGML formats, as written
            by hf_ptq.py with a GGML recipe.
        output_dir: Where to write the NVFP4 checkpoint. Must be empty or not exist yet.
        max_shard_size: Target bytes per output safetensors shard; a larger tensor gets a shard
            of its own. Conversion holds one output shard in memory at a time.
    """
    ggml_dir = Path(ggml_dir)
    hf_quant_config = json.loads((ggml_dir / "hf_quant_config.json").read_text())
    layer_formats = ggml_layer_formats(ggml_dir)
    weight_map = indexed_weight_map(ggml_dir)
    with_input_scale = {layer for layer in layer_formats if f"{layer}.input_scale" in weight_map}
    if with_input_scale and with_input_scale != set(layer_formats):
        missing = sorted(set(layer_formats) - with_input_scale)
        raise ValueError(f"Only some GGML layers have activation scales; {missing} do not.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_scales = _global_scales(ggml_dir, weight_map, layer_formats, device)

    def convert(name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        layer = name.removesuffix(".weight")
        ggml_format = layer_formats.get(layer) if name.endswith(".weight") else None
        if ggml_format is None:
            return {name: tensor}
        decoded = _decode(tensor.to(device), ggml_format)
        block_scales, global_scale = nvfp4_transcode_scales(decoded, global_scales[layer])
        packed, _, _ = NVFP4QTensor.quantize(decoded, 16, block_scales, global_scale)
        return {
            name: packed._quantized_data.cpu(),
            f"{layer}.weight_scale": block_scales.cpu(),
            # A copy: fused projections share one scale tensor, and safetensors refuses aliases.
            f"{layer}.weight_scale_2": global_scale.to("cpu", copy=True),
        }

    quantization = hf_quant_config["quantization"]
    hf_quant_config["quantization"] = {
        "quant_algo": "NVFP4" if with_input_scale else "W4A16_NVFP4",
        "kv_cache_quant_algo": None,
        "group_size": 16,
        "exclude_modules": quantization.get("exclude_modules", []),
    }
    config = json.loads((ggml_dir / "config.json").read_text())
    config["quantization_config"] = convert_hf_quant_config_format(hf_quant_config)
    write_converted_checkpoint(
        ggml_dir,
        output_dir,
        convert,
        {"config.json": config, "hf_quant_config.json": hf_quant_config},
        max_shard_size,
    )


def _decode(payload: torch.Tensor, ggml_format: GGMLFormat) -> torch.Tensor:
    """A ``[rows, blocks, block bytes]`` payload decoded in float32, as fake quant decodes it."""
    shape = torch.tensor((payload.shape[0], payload.shape[1] * ggml_format.block_size))
    return ggml_format.dequantize(payload, shape, dtype=torch.float32)


def _global_scales(
    ggml_dir: Path,
    weight_map: dict[str, str],
    layer_formats: dict[str, GGMLFormat],
    device: str,
) -> dict[str, torch.Tensor]:
    """Each layer's per-tensor NVFP4 scale, the largest of its fused projections' own scales."""
    own: dict[str, torch.Tensor] = {}
    layers_by_shard = defaultdict(list)
    for layer in layer_formats:
        layers_by_shard[weight_map[f"{layer}.weight"]].append(layer)
    for shard, layers in sorted(layers_by_shard.items()):
        with safe_open(ggml_dir / shard, framework="pt", device=device) as f:
            for layer in layers:
                decoded = _decode(f.get_tensor(f"{layer}.weight"), layer_formats[layer])
                own[layer] = nvfp4_global_scale(decoded)

    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for layer in own:
        parent, _, leaf = layer.rpartition(".")
        group = next((i for i, g in enumerate(_FUSED_PROJECTIONS) if leaf in g), None)
        # A layer outside every fused group keeps a group of its own.
        groups[(parent, group) if group is not None else (layer, -1)].append(layer)
    shared = {}
    for members in groups.values():
        scale = torch.stack([own[m] for m in members]).amax()
        shared.update(dict.fromkeys(members, scale))
    return shared


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ggml_dir", help="Checkpoint exported with a GGML recipe (IQ, q8_0).")
    parser.add_argument("output_dir", help="Where to write the NVFP4 checkpoint; must be empty.")
    parser.add_argument(
        "--max_shard_size_gb",
        type=float,
        default=10,
        help="Target size of each output shard in GiB; a larger tensor gets a shard of its own.",
    )
    args = parser.parse_args()
    convert_ggml_export_to_nvfp4(
        args.ggml_dir, args.output_dir, max_shard_size=int(args.max_shard_size_gb * 2**30)
    )
    print(f"NVFP4 checkpoint written to {args.output_dir}")


if __name__ == "__main__":
    main()
