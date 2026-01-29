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


"""Custom Megatron mapping and safetensors utility."""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open, save_file

from modelopt.torch.utils import import_plugin

with import_plugin("megatron"):
    from megatron.core.parallel_state import (
        get_expert_model_parallel_rank,
        get_expert_model_parallel_world_size,
        get_expert_tensor_parallel_rank,
        get_expert_tensor_parallel_world_size,
        get_pipeline_model_parallel_rank,
        get_pipeline_model_parallel_world_size,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

MAX_SAFETENSOR_SIZE = 4000000000
PER_RANK_PARTITIONS = 10


@dataclass
class ParallelConfig:
    """The parallel configuration including sharding_dim and parallel_group.

    Args:
        sharding_dim: The dimension (0 or 1) to shard the tensor.
        parallel_group: The parallel group (TP or ETP) to shard the tensor.
    """

    sharding_dim: int | None = None
    parallel_group: str | None = None


COL_TP = {"parallel_config": ParallelConfig(sharding_dim=0, parallel_group="TP")}
ROW_TP = {"parallel_config": ParallelConfig(sharding_dim=1, parallel_group="TP")}
COL_ETP = {"parallel_config": ParallelConfig(sharding_dim=0, parallel_group="ETP")}
ROW_ETP = {"parallel_config": ParallelConfig(sharding_dim=1, parallel_group="ETP")}
PACK_COL_ETP = {"parallel_config": ParallelConfig(sharding_dim=2, parallel_group="ETP")}
PACK_ROW_ETP = {"parallel_config": ParallelConfig(sharding_dim=1, parallel_group="ETP")}
PACK_EP = {"parallel_config": ParallelConfig(sharding_dim=0, parallel_group="EP")}
REPLICATE = {}


class CustomModuleMapping:
    """A custom module mapping from Megatron Core to its HF counter part."""

    def __init__(
        self, func_name: str = "", target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}
    ):
        """Create a custom module mapping."""
        self.func_name = func_name
        self.target_name_or_prefix = target_name_or_prefix
        self.func_kwargs = func_kwargs


class NameRemapping(CustomModuleMapping):
    """A custom module mapping that renames of the modules."""

    def __init__(self, target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}):
        """Create a custom module mapping that renames of the modules."""
        super().__init__(
            func_name="name_remapping",
            target_name_or_prefix=target_name_or_prefix,
            func_kwargs=func_kwargs,
        )


class QKVMerging(CustomModuleMapping):
    """A custom module mapping that merges Q, K, V."""

    def __init__(self, target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}):
        """Create a custom module mapping that merges Q, K, V."""
        super().__init__(
            func_name="qkv_merging",
            target_name_or_prefix=target_name_or_prefix,
            func_kwargs=func_kwargs,
        )


class GroupedMLPMerging(CustomModuleMapping):
    """A custom module mapping that merges up_proj and down_proj for Grouped MLP."""

    def __init__(self, target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}):
        """Create a custom module mapping that merges up_proj and down_proj for Grouped MLP."""
        super().__init__(
            func_name="grouped_mlp_merging",
            target_name_or_prefix=target_name_or_prefix,
            func_kwargs=func_kwargs,
        )


class GatedMLPMerging(CustomModuleMapping):
    """A custom module mapping that merges gate_proj and up_proj."""

    def __init__(self, target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}):
        """Create a custom module mapping that merges gate_proj and up_proj."""
        super().__init__(
            func_name="gated_mlp_merging",
            target_name_or_prefix=target_name_or_prefix,
            func_kwargs=func_kwargs,
        )


class QKVSlicing(CustomModuleMapping):
    """A custom module mapping that slices Q, K, V."""

    def __init__(self, target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}):
        """Create a custom module mapping that slices Q, K, V."""
        super().__init__(
            func_name="qkv_slicing",
            target_name_or_prefix=target_name_or_prefix,
            func_kwargs=func_kwargs,
        )


class SelfAttentionScaling(CustomModuleMapping):
    """A custom module mapping that scales self attention."""

    def __init__(self, target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}):
        """Create a custom module mapping that scales self attention."""
        super().__init__(
            func_name="self_attention_scaling",
            target_name_or_prefix=target_name_or_prefix,
            func_kwargs=func_kwargs,
        )


class GatedMLPSlicing(CustomModuleMapping):
    """A custom module mapping that slices gate_proj and up_proj."""

    def __init__(self, target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}):
        """Create a custom module mapping that slices gate_proj and up_proj."""
        super().__init__(
            func_name="gated_mlp_slicing",
            target_name_or_prefix=target_name_or_prefix,
            func_kwargs=func_kwargs,
        )


class PackNameRemapping(CustomModuleMapping):
    """A custom module mapping that packs module after name remapping."""

    def __init__(self, target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}):
        """Create a custom module mapping that packs and renames module."""
        super().__init__(
            func_name="pack_name_remapping",
            target_name_or_prefix=target_name_or_prefix,
            func_kwargs=func_kwargs,
        )


class UnpackNameRemapping(CustomModuleMapping):
    """A custom module mapping that unpacks module after name remapping."""

    def __init__(self, target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}):
        """Create a custom module mapping that unpacks module after name remapping."""
        super().__init__(
            func_name="unpack_name_remapping",
            target_name_or_prefix=target_name_or_prefix,
            func_kwargs=func_kwargs,
        )


class PackNameRemappingGPT(CustomModuleMapping):
    """A custom module mapping that packs module after name remapping."""

    def __init__(self, target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}):
        """Create a custom module mapping that packs and renames module."""
        super().__init__(
            func_name="pack_name_remapping_gpt_oss",
            target_name_or_prefix=target_name_or_prefix,
            func_kwargs=func_kwargs,
        )


class UnpackNameRemappingGPT(CustomModuleMapping):
    """A custom module mapping that unpacks module after name remapping."""

    def __init__(self, target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}):
        """Create a custom module mapping that unpacks module after name remapping."""
        super().__init__(
            func_name="unpack_name_remapping_gpt_oss",
            target_name_or_prefix=target_name_or_prefix,
            func_kwargs=func_kwargs,
        )


def save_safetensors(state_dict, save_directory: str | os.PathLike):
    """Save safetensors with pipeline model parallel support."""
    pp_rank = get_pipeline_model_parallel_rank()
    pp_size = get_pipeline_model_parallel_world_size()

    all_safetensors = [{} for i in range(PER_RANK_PARTITIONS)]

    local_idx = 0
    local_total_size = 0

    for key, val in state_dict.items():
        tensor_size = val.numel() * val.element_size()
        if (
            local_total_size + tensor_size > MAX_SAFETENSOR_SIZE
            and local_idx < PER_RANK_PARTITIONS - 1
        ):
            local_idx += 1
            local_total_size = 0
        all_safetensors[local_idx][key] = val
        local_total_size += tensor_size

    for local_idx, tensors in enumerate(all_safetensors):
        global_idx = PER_RANK_PARTITIONS * pp_rank + local_idx
        global_count = PER_RANK_PARTITIONS * pp_size
        filename = f"model-{global_idx + 1:05d}-of-{global_count:05d}"
        meta_filename = filename + ".json"
        ckpt_filename = filename + ".safetensors"

        weight_map = {}
        local_total_size = 0

        for key, val in tensors.items():
            local_total_size += val.numel() * val.element_size()
            weight_map[key] = ckpt_filename

        with open(save_directory + "/" + meta_filename, "w") as f:
            json.dump(
                {"metadata": {"total_size": local_total_size}, "weight_map": weight_map},
                f,
                indent=4,
            )
        save_file(tensors, save_directory + "/" + ckpt_filename, metadata={"format": "pt"})

    # Barrier to ensure all ranks have written the metadata
    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        global_count = PER_RANK_PARTITIONS * pp_size
        safetensor_index = {
            "metadata": {"total_size": 0},
            "weight_map": {},
        }
        for global_idx in range(global_count):
            meta_filename = f"model-{global_idx + 1:05d}-of-{global_count:05d}.json"
            with open(save_directory + "/" + meta_filename) as f:
                shard = json.load(f)
            safetensor_index["metadata"]["total_size"] += shard["metadata"]["total_size"]
            safetensor_index["weight_map"].update(shard["weight_map"])

        with open(save_directory + "/model.safetensors.index.json", "w") as f:
            json.dump(safetensor_index, f, indent=4)


def _get_safetensors_file(pretrained_model_path: str | Path, key: str) -> Path | None:
    """Given a tensor key return the safetensors file that contains this tensor if exists.

    Args:
        pretrained_model_path: The path to the pretrained model.
        key: The key of the tensor to get.
    """
    safetensors_file = Path(pretrained_model_path) / "model.safetensors"
    safetensors_index_file = Path(pretrained_model_path) / "model.safetensors.index.json"
    if safetensors_file.is_file():
        pass
    elif safetensors_index_file.is_file():
        with open(safetensors_index_file) as f:
            safetensors_index = json.load(f)
        safetensors_file = (
            (Path(pretrained_model_path) / safetensors_index["weight_map"][key])
            if key in safetensors_index["weight_map"]
            else None
        )
    else:
        raise ValueError("Only safetensors (single of multi- files) are supported.")
    return safetensors_file


def _get_safetensor_slices(
    pretrained_model_path: str | Path, key: str, parallel_config: ParallelConfig | None = None
):
    """Given a tensor key return the safetensors slice if exists.

    Depending on the parallel_config, the tensor will be sharded along the sharding_dim over
    the size of the parallel group (equally divided over each rank in the group).
    The sharding_dim can be 0 or 1, which corresponds to the column parallel or row parallel,
    respectively. The parallel group can be tensor model parallel (TP) group or expert tensor
    parallel (ETP) group.

    Args:
        pretrained_model_path: The path to the pretrained model.
        key: The key of the tensor to get.
        parallel_config: The parallel configuration including sharding_dim and parallel_group.
    """
    safetensors_file = _get_safetensors_file(pretrained_model_path, key)

    if safetensors_file is None:
        return None

    with safe_open(safetensors_file, framework="pt") as f:
        if key not in f.keys():  # noqa: SIM118
            tensor = None
        elif parallel_config is None:
            tensor = f.get_tensor(key)
        else:
            tensor_slice = f.get_slice(key)
            assert tensor_slice is not None
            shape = tensor_slice.get_shape()
            assert len(shape) in (1, 2, 3, 4), f"Shape {shape} is not supported!"
            # 1 for bias case
            # 3 for packed MoE case (Llama4)
            # 4 for packed MoE case with mxfp4 dtype (gpt-oss)

            # MCore tensor parallel model sharding
            sharding_dim = parallel_config.sharding_dim
            parallel_group = parallel_config.parallel_group

            if parallel_group == "EP":
                # For packed EP case, Llama4 uses 3D tensor for local experts
                tp_size = get_expert_tensor_parallel_world_size()
                if tp_size > 1:
                    raise ValueError("Packed MoE import only supports ETP=1.")
                if len(shape) not in (2, 3, 4):
                    raise ValueError(
                        "For LLama4, Packed MoE import only supports 3D tensor in shape [num_experts, in_dim, out_dim]."
                        "For gpt-oss, Packed MoE import only supports 2D tensor of MOE bias, "
                        "3D tensor of MOE scales, 4D tensor of MOE blocks"
                    )
                if sharding_dim != 0:
                    raise ValueError("Packed MoE import only supports sharding_dim=0.")
                ep_rank = get_expert_model_parallel_rank()
                ep_size = get_expert_model_parallel_world_size()
                per_rank_size = shape[sharding_dim] // ep_size
                rank_offset = ep_rank * per_rank_size
                if shape[sharding_dim] % ep_size > 0:
                    raise ValueError(
                        "{} {} @ dim {} is not a multiple of {}".format(
                            key, shape, sharding_dim, ep_size
                        )
                    )
                if len(shape) == 2:
                    tensor = tensor_slice[rank_offset : rank_offset + per_rank_size, :]
                elif len(shape) == 3:
                    tensor = tensor_slice[rank_offset : rank_offset + per_rank_size, :, :]
                elif len(shape) == 4:
                    tensor = tensor_slice[rank_offset : rank_offset + per_rank_size, :, :, :]
            else:
                if parallel_group == "TP":
                    tp_rank = get_tensor_model_parallel_rank()
                    tp_size = get_tensor_model_parallel_world_size()
                elif parallel_group == "ETP":
                    tp_rank = get_expert_tensor_parallel_rank()
                    tp_size = get_expert_tensor_parallel_world_size()
                else:
                    raise ValueError(f"Unsupported parallel group [tp|etp]: {parallel_group}")
                per_rank_size = shape[sharding_dim] // tp_size
                rank_offset = tp_rank * per_rank_size

                if shape[sharding_dim] % tp_size > 0:
                    raise ValueError(
                        "{} {} @ dim {} is not a multiple of {}".format(
                            key, shape, sharding_dim, tp_size
                        )
                    )
                if len(shape) == 2:
                    if sharding_dim in (1, -1):
                        tensor = tensor_slice[:, rank_offset : rank_offset + per_rank_size]
                    else:
                        tensor = tensor_slice[rank_offset : rank_offset + per_rank_size, :]
                elif len(shape) == 3:
                    # For packed ETP case, Llama4 uses 3D tensor for local experts
                    # For gpt-oss case, gpt-oss uses 3D tensor for scales of local experts
                    if sharding_dim == 1:
                        tensor = tensor_slice[:, rank_offset : rank_offset + per_rank_size, :]
                    elif sharding_dim == 2:
                        tensor = tensor_slice[:, :, rank_offset : rank_offset + per_rank_size]
                    else:
                        raise ValueError(
                            f"Unsupported sharding_dim: {sharding_dim} for shape: {shape}"
                        )
                elif len(shape) == 1:
                    # For bias case
                    tensor = tensor_slice[rank_offset : rank_offset + per_rank_size]
                elif len(shape) == 4:
                    # For packed ETP case, gpt-oss uses 4D tensor for local experts
                    if sharding_dim == 1:
                        tensor = tensor_slice[:, rank_offset : rank_offset + per_rank_size, :, :]
                    elif sharding_dim == 2:
                        tensor = tensor_slice[:, :, rank_offset : rank_offset + per_rank_size, :]
                else:
                    raise ValueError(f"Unsupported shape: {shape}")
    return tensor


def _get_dsfp8_weight_scale_inv(
    pretrained_model_path: str | Path, key: str, parallel_config: ParallelConfig | None = None
):
    """Given a weight tensor key return weight_scale_inv if exists.

    Args:
        pretrained_model_path: The path to the pretrained model.
        key: The key of the tensor to get.
        parallel_config: The parallel configuration including sharding_dim and parallel_group.
    """
    if not key.endswith("weight"):
        return None
    return _get_safetensor_slices(pretrained_model_path, key + "_scale_inv", parallel_config)


def get_safetensor(
    pretrained_model_path: str | Path,
    key: str,
    parallel_config: ParallelConfig | None = None,
    dequantize: bool = False,
) -> torch.Tensor:
    """Get a safetensor from the sharded checkpoint.

    Args:
        pretrained_model_path: The path to the pretrained model.
        key: The key of the tensor to get.
        parallel_config: The parallel configuration including sharding_dim and parallel_group.
    """
    tensor = _get_safetensor_slices(pretrained_model_path, key, parallel_config)

    if tensor is None:
        raise ValueError(f"Key [{key}] does not exist!")

    if dequantize:
        # DSFP8 (weight-only 128x128-blocked FP8)
        if tensor.dtype is torch.float8_e4m3fn:
            weight_scale_inv = _get_dsfp8_weight_scale_inv(
                pretrained_model_path, key, parallel_config
            )
            if weight_scale_inv is None:
                raise ValueError(f"Fail to dequantize [{key}]! weight_scale_inv not found!")
            else:
                # [TODO]: may need padding here
                out_dim, in_dim = tensor.shape
                blk_out_dim, blk_in_dim = weight_scale_inv.shape

                tensor = tensor.to(torch.bfloat16)
                if out_dim % 128 > 0 or in_dim % 128 > 0:
                    padded_tensor = torch.zeros(
                        blk_out_dim * 128, blk_in_dim * 128, dtype=torch.bfloat16
                    )
                    padded_tensor[:out_dim, :in_dim] = tensor
                else:
                    padded_tensor = tensor

                padded_tensor = padded_tensor.reshape(blk_out_dim, 128, blk_in_dim, 128)
                weight_scale_inv = weight_scale_inv.to(torch.bfloat16)
                weight_scale_inv = weight_scale_inv.reshape(blk_out_dim, 1, blk_in_dim, 1)
                padded_tensor = padded_tensor * weight_scale_inv
                padded_tensor = padded_tensor.view(blk_out_dim * 128, blk_in_dim * 128)

                if out_dim % 128 > 0 or in_dim % 128 > 0:
                    tensor = padded_tensor[:out_dim, :in_dim]
                else:
                    tensor = padded_tensor

    return tensor.contiguous()


def dequantize_mxfp4_to_bf16(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    """Dequantize MXFP4 blocks and scales to BF16 format.

    Args:
        blocks: The quantized blocks tensor (U8 dtype)
        scales: The scales tensor (U8 dtype)
        dtype: Target dtype for dequantization (default: torch.bfloat16)
        rows_per_chunk: Number of rows to process per chunk

    Returns:
        Dequantized tensor in the specified dtype
    """
    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"

    # Convert scales from U8 to int32 and subtract 127 (bias)
    scales = scales.to(torch.int32) - 127

    fp4_values = [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    lut = torch.tensor(fp4_values, dtype=dtype, device=blocks.device)

    *prefix_shape, g, b = blocks.shape
    rows_total = math.prod(prefix_shape) * g

    blocks = blocks.reshape(rows_total, b)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, b * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, g, b * 2).view(*prefix_shape, g * b * 2)
