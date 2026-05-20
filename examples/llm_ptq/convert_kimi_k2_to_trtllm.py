#!/usr/bin/env python3
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

"""
Convert Kimi-K2-Thinking compressed-tensors checkpoint to TRT-LLM GPTQ format.

The source checkpoint uses:
- int32 packing: 8 int4 values per int32
- group_size: 32
- symmetric quantization

TRT-LLM GPTQ format uses:
- int32 packing: 8 int4 values per int32 (same)
- Requires .qweight, .scales, .qzeros tensors
"""

import argparse
import json
from pathlib import Path

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm


def unpack_int32_to_int4(weight_packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack int32 tensor containing 8 int4 values into int8 tensor.

    Args:
        weight_packed: Shape (N, K/8) dtype int32

    Returns:
        unpacked: Shape (N, K) dtype int8 with values in range [-8, 7]
    """
    # Convert int32 to uint8 view to extract nibbles
    w_packed_uint8 = weight_packed.contiguous().view(torch.uint8)

    # Each int32 = 4 bytes, each byte has 2 int4 values
    # So shape (N, K/8) int32 -> (N, K/8, 4) uint8 -> (N, K/2) uint8
    n, k_div_8 = weight_packed.shape
    w_packed_uint8 = w_packed_uint8.view(n, k_div_8 * 4)

    # Allocate output: (N, K) where K = K_div_8 * 8
    k = k_div_8 * 8
    w_unpacked = torch.zeros(n, k, dtype=torch.int8)

    # Extract low and high nibbles
    w_unpacked[:, 0::2] = (w_packed_uint8 & 0x0F).to(torch.int8)
    w_unpacked[:, 1::2] = (w_packed_uint8 >> 4).to(torch.int8)

    # Convert from uint4 [0, 15] to int4 [-8, 7]
    # Values > 7 should be interpreted as negative
    w_unpacked[w_unpacked > 7] -= 16

    return w_unpacked.contiguous()


def pack_int4_to_int32_gptq(weight_unpacked: torch.Tensor) -> torch.Tensor:
    """
    Pack int8 tensor (with int4 values) into int32 GPTQ format.

    Args:
        weight_unpacked: Shape (N, K) dtype int8 with values in range [-8, 7]

    Returns:
        packed: Shape (N, K/8) dtype int32
    """
    n, k = weight_unpacked.shape
    assert k % 8 == 0, "K must be divisible by 8"

    # Convert int4 [-8, 7] to uint4 [0, 15]
    w_uint4 = weight_unpacked.clone()
    w_uint4[w_uint4 < 0] += 16
    w_uint4 = w_uint4.to(torch.uint8)

    # Pack 2 uint4 into 1 uint8
    w_packed_uint8 = torch.zeros(n, k // 2, dtype=torch.uint8)
    w_packed_uint8 = (w_uint4[:, 1::2] << 4) | (w_uint4[:, 0::2])

    # Reshape to int32
    w_packed_int32 = (
        w_packed_uint8.view(n, k // 8, 4).view(torch.uint8).view(n, k // 8).view(torch.int32)
    )

    return w_packed_int32.contiguous()


def convert_compressed_tensor_to_gptq(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: list[int],
    group_size: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Convert compressed-tensors format to GPTQ format for TRT-LLM.

    Args:
        weight_packed: Shape (N, K/8) dtype int32
        weight_scale: Shape (N, K/group_size) dtype bfloat16
        weight_shape: [N, K] - original weight shape
        group_size: Quantization group size

    Returns:
        Dictionary with:
            - qweight: Shape (K/8, N) dtype int32 (transposed!)
            - scales: Shape (K/group_size, N) dtype fp16 (transposed!)
            - qzeros: Shape (K/group_size, N/8) dtype int32 (for symmetric, all zeros)
    """
    n, k = weight_shape

    # TRT-LLM expects weights transposed: (K, N) instead of (N, K)
    # But packed format keeps the packing dimension, so:
    # Source: (N, K/8) -> Target: (K/8, N)
    qweight = weight_packed.t().contiguous()

    # Scales also need transpose: (N, K/group_size) -> (K/group_size, N)
    scales = weight_scale.t().contiguous().to(torch.float16)

    # For symmetric quantization, we use zero as the zero-point
    # GPTQ format expects qzeros in packed format
    # Shape: (K/group_size, N/8) since zeros are also packed 8 per int32
    num_groups = k // group_size
    # Create zeros tensor - for symmetric quantization, zero-point is 8 (middle of [0,15])
    # Pack as uint4: each int32 contains 8 nibbles of value 8
    # Create as bytes first then view as int32
    qzeros_uint8 = torch.full((num_groups, n // 2), 0x88, dtype=torch.uint8)
    qzeros = qzeros_uint8.view(torch.int32).contiguous()

    return {
        "qweight": qweight,
        "scales": scales,
        "qzeros": qzeros,
    }


def convert_checkpoint(
    input_dir: str,
    output_dir: str,
    num_shards: int | None = None,
    skip_existing: bool = True,
):
    """
    Convert all shards from compressed-tensors to GPTQ format.

    Args:
        input_dir: Source checkpoint directory
        output_dir: Output checkpoint directory
        num_shards: Number of shards to process (None = all)
        skip_existing: Skip conversion if output shard already exists
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all safetensors files
    shard_files = sorted(input_path.glob("model-*.safetensors"))
    if not shard_files:
        raise ValueError(f"No model shards found in {input_dir}")

    if num_shards is not None:
        shard_files = shard_files[:num_shards]

    print(f"Found {len(shard_files)} shards to process")

    # Track weight mapping for index file
    new_weight_map = {}

    # Load and convert each shard
    for shard_idx, shard_file in enumerate(tqdm(shard_files, desc="Processing shards")):
        shard_name = shard_file.name
        output_file = output_path / shard_name

        # Check if output already exists
        if skip_existing and output_file.exists():
            print(f"\n⏭️  Skipping {shard_name} (already exists)")
            # Still need to build the weight_map from existing file
            with safetensors.safe_open(str(output_file), framework="pt", device="cpu") as f:
                for key in f:
                    new_weight_map[key] = shard_name
            continue

        print(f"\n🔄 Converting {shard_file.name}...")

        # Load source shard
        source_tensors = {}
        with safetensors.safe_open(str(shard_file), framework="pt", device="cpu") as f:
            for key in f:
                source_tensors[key] = f.get_tensor(key)

        # Convert tensors
        output_tensors = {}

        for key, tensor in tqdm(source_tensors.items(), desc="Converting tensors", leave=False):
            if key.endswith(".weight_packed"):
                # This is a quantized weight - convert to GPTQ format
                base_key = key[: -len(".weight_packed")]
                scale_key = base_key + ".weight_scale"
                shape_key = base_key + ".weight_shape"

                if scale_key in source_tensors and shape_key in source_tensors:
                    weight_shape = source_tensors[shape_key].tolist()

                    # Convert to GPTQ format
                    gptq_tensors = convert_compressed_tensor_to_gptq(
                        weight_packed=tensor,
                        weight_scale=source_tensors[scale_key],
                        weight_shape=weight_shape,
                        group_size=32,
                    )

                    # Save with GPTQ naming convention and track in weight_map
                    qweight_key = base_key + ".qweight"
                    scales_key = base_key + ".scales"
                    qzeros_key = base_key + ".qzeros"

                    output_tensors[qweight_key] = gptq_tensors["qweight"]
                    output_tensors[scales_key] = gptq_tensors["scales"]
                    output_tensors[qzeros_key] = gptq_tensors["qzeros"]

                    new_weight_map[qweight_key] = shard_name
                    new_weight_map[scales_key] = shard_name
                    new_weight_map[qzeros_key] = shard_name
                else:
                    print(f"Warning: Missing scale or shape for {key}")

            elif key.endswith((".weight_scale", ".weight_shape")):
                # Skip these as they're handled above
                continue
            else:
                # Keep non-quantized tensors as-is
                output_tensors[key] = tensor
                new_weight_map[key] = shard_name

        # Save converted shard
        safetensors.torch.save_file(output_tensors, str(output_file))
        print(f"✅ Saved to {output_file}")

    # Copy config.json and update quantization settings
    config_file = input_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

        # Remove HuggingFace quantization_config if present
        config.pop("quantization_config", None)

        # Add TRT-LLM native quantization config for GPTQ
        config["quantization"] = {
            "quant_algo": "W4A16_GPTQ",  # TRT-LLM's enum value
            "group_size": 32,
            "has_zero_point": True,  # GPTQ uses asymmetric quantization
            "pre_quant_scale": False,  # No pre-quantization scaling
        }

        output_config_file = output_path / "config.json"
        with open(output_config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nSaved config to {output_config_file}")

    # Generate new safetensors index file
    index_data = {
        "metadata": {
            "total_size": sum(
                (output_path / shard_file.name).stat().st_size for shard_file in shard_files
            )
        },
        "weight_map": new_weight_map,
    }

    index_file = output_path / "model.safetensors.index.json"
    with open(index_file, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"\nGenerated index file: {index_file}")
    print(f"  Total tensors: {len(new_weight_map)}")

    # Copy other necessary files
    import shutil

    # JSON files (tokenizer and generation config)
    for file in [
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]:
        src = input_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            print(f"Copied: {file}")

    # Python files (model architecture, custom tokenizers)
    for file in ["configuration_deepseek.py", "modeling_deepseek.py", "tokenization_kimi.py"]:
        src = input_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            print(f"Copied: {file}")

    # Tokenizer model files
    for file in ["tiktoken.model", "tokenizer.model", "sentencepiece.model"]:
        src = input_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            print(f"Copied: {file}")

    # Template files
    for file in ["chat_template.jinja", "chat_template.json"]:
        src = input_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            print(f"Copied: {file}")

    print(f"\n✓ Conversion complete! Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kimi-K2 checkpoint to TRT-LLM GPTQ format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/scratch.omniml_data_1/models/Kimi-K2-Thinking",
        help="Input checkpoint directory with compressed-tensors format",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/scratch.omniml_data_2/zhiyuc/checkpoints/Kimi-K2-Thinking-GPTQ",
        help="Output directory for GPTQ format checkpoint",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of shards to convert (default: all)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-convert shards even if they already exist (default: skip existing)",
    )

    args = parser.parse_args()

    convert_checkpoint(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
