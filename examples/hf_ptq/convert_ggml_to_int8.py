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

"""Expand a GGML checkpoint exported by hf_ptq.py into an INT8 checkpoint that vLLM serves.

python hf_ptq.py --pyt_ckpt_path <model> --recipe general/ptq/iq1_s --export_path <ggml_dir>
python convert_ggml_to_int8.py <ggml_dir> <int8_dir>
vllm serve <int8_dir>
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY, GGMLFormat
from modelopt.torch.utils.plugins.hf_checkpoint_utils import (
    copy_non_model_files,
    copy_off_index_safetensors,
    indexed_weight_map,
)

_INDEX = "model.safetensors.index.json"


def convert_ggml_export_to_int8(
    ggml_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    max_shard_size: int = 10 * 2**30,
) -> None:
    """Write a GGML export as an INT8 weight-only compressed-tensors checkpoint, which vLLM serves.

    Every GGML weight is an integer level of at most 8 bits times the step of its 16- or
    32-weight sub-block, so storing those levels with the steps as group scales reproduces the
    GGML weights up to rounding the steps to the model dtype. Unquantized tensors and the other
    files are copied as they are; the GGML export is left unchanged.

    Args:
        ggml_dir: A Hugging Face export whose quantized layers are all GGML formats, as written
            by hf_ptq.py with a GGML recipe.
        output_dir: Where to write the INT8 checkpoint. Must be empty or not exist yet.
        max_shard_size: Target bytes per output safetensors shard; a larger tensor gets a shard
            of its own. Conversion holds one output shard in memory at a time.
    """
    ggml_dir, output_dir = Path(ggml_dir), Path(output_dir)
    config = json.loads((ggml_dir / "config.json").read_text())
    quantization = json.loads((ggml_dir / "hf_quant_config.json").read_text())["quantization"]
    # Multimodal configs such as GLM-5.3-Flash's keep the dtype in text_config only.
    text_config = config.get("text_config", {})
    dtype = (
        config.get("dtype")
        or config.get("torch_dtype")
        or text_config.get("dtype")
        or text_config["torch_dtype"]
    )
    scale_dtype = getattr(torch, dtype)
    weight_map = indexed_weight_map(ggml_dir)
    layer_formats = _ggml_layer_formats(ggml_dir, weight_map, quantization)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{output_dir} is not empty.")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        shards = sorted(set(weight_map.values()))
        new_weight_map = _write_int8_shards(
            ggml_dir, shards, output_dir, layer_formats, scale_dtype, max_shard_size
        )
        total_size = sum((output_dir / s).stat().st_size for s in set(new_weight_map.values()))
        index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
        (output_dir / _INDEX).write_text(json.dumps(index, indent=2))
        config["quantization_config"] = _compressed_tensors_config(layer_formats)
        (output_dir / "config.json").write_text(json.dumps(config, indent=4))
        # Both skip files already written, and copy_non_model_files skips hf_quant_config.json,
        # which describes the GGML payloads this replaced.
        copy_off_index_safetensors(ggml_dir, output_dir)
        copy_non_model_files(ggml_dir, output_dir)
    except BaseException:
        # Empty the directory again, so a rerun is not refused over a partial checkpoint.
        for path in output_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        raise


def _ggml_layer_formats(
    ggml_dir: Path, weight_map: dict[str, str], quantization: dict
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
        with safe_open(ggml_dir / shard, framework="pt") as f:
            # GGML payloads are the only uint8 weights an export writes.
            payloads = {
                name: f.get_slice(name).get_shape()
                for name in f.keys()  # noqa: SIM118 -- a safe_open handle is not a dict
                if name.endswith(".weight") and f.get_slice(name).get_dtype() == "U8"
            }
        for name, shape in payloads.items():
            layer = name.removesuffix(".weight")
            # A 2-D weight's payload is [rows, blocks, block bytes]; weight_shape and the int32
            # packing assume that layout.
            if len(shape) != 3:
                raise ValueError(f"{layer} is not a 2-D weight; only 2-D weights convert to INT8.")
            if mixed and layer not in layer_configs:
                raise ValueError(f"{layer} holds a uint8 weight but is not in quantized_layers.")
            algo = layer_configs[layer]["quant_algo"] if mixed else quantization["quant_algo"]
            layer_formats[layer] = GGML_FORMAT_REGISTRY[algo.lower()]
    if not layer_formats:
        raise ValueError(f"{ggml_dir} holds no GGML weights to convert.")
    return layer_formats


def _write_int8_shards(
    ggml_dir: Path,
    shards: list[str],
    output_dir: Path,
    layer_formats: dict[str, GGMLFormat],
    scale_dtype: torch.dtype,
    max_shard_size: int,
) -> dict[str, str]:
    """Rewrite every shard's tensors into size-bounded output shards; return the weight map."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pending: dict[str, torch.Tensor] = {}
    written: list[list[str]] = []

    def flush():
        save_file(pending, output_dir / f"part-{len(written):05d}.safetensors", {"format": "pt"})
        written.append(list(pending))
        pending.clear()

    for shard in shards:
        with safe_open(ggml_dir / shard, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118 -- a safe_open handle is not a dict
                layer = name.removesuffix(".weight")
                ggml_format = layer_formats.get(layer) if name.endswith(".weight") else None
                if ggml_format is None:
                    tensors = {name: f.get_tensor(name)}
                else:
                    tensors = _int8_tensors(
                        layer, f.get_tensor(name), ggml_format, scale_dtype, device
                    )
                if pending and _nbytes(pending) + _nbytes(tensors) > max_shard_size:
                    flush()
                pending.update(tensors)
    flush()

    weight_map = {}
    for i, names in enumerate(written):
        shard_name = f"model-{i + 1:05d}-of-{len(written):05d}.safetensors"
        os.replace(output_dir / f"part-{i:05d}.safetensors", output_dir / shard_name)
        weight_map.update(dict.fromkeys(names, shard_name))
    return dict(sorted(weight_map.items()))


def _int8_tensors(
    layer: str,
    payload: torch.Tensor,
    ggml_format: GGMLFormat,
    scale_dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor]:
    """A layer's GGML payload as compressed-tensors ``weight_packed``, ``weight_scale`` and shape."""
    shape = torch.tensor((*payload.shape[:-2], payload.shape[-2] * ggml_format.block_size))
    levels, steps = ggml_format.decode_int8(payload.to(device), shape)
    return {
        # compressed-tensors packs 8-bit values offset by 128, four little-endian per int32.
        f"{layer}.weight_packed": (levels.view(torch.uint8) ^ 0x80).view(torch.int32).cpu(),
        f"{layer}.weight_scale": steps.to(scale_dtype).cpu(),
        f"{layer}.weight_shape": shape,
    }


def _nbytes(tensors: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors.values())


def _compressed_tensors_config(layer_formats: dict[str, GGMLFormat]) -> dict:
    """One config group per INT8 group size, targeting its layers by name.

    vLLM maps explicit layer names onto its own modules, but not ``re:`` patterns.
    """
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ggml_dir", help="Checkpoint exported with a GGML recipe (IQ, q8_0).")
    parser.add_argument("output_dir", help="Where to write the INT8 checkpoint; must be empty.")
    parser.add_argument(
        "--max_shard_size_gb",
        type=float,
        default=10,
        help="Target size of each output shard in GiB; a larger tensor gets a shard of its own.",
    )
    args = parser.parse_args()
    convert_ggml_export_to_int8(
        args.ggml_dir, args.output_dir, max_shard_size=int(args.max_shard_size_gb * 2**30)
    )
    print(f"INT8 checkpoint written to {args.output_dir}")


if __name__ == "__main__":
    main()
