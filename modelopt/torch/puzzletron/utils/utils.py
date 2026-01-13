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

import dataclasses
import json
import os
from copy import deepcopy
from typing import Any

import torch

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
)


def calculate_kv_dim(n_heads_in_group: int, n_head: int, n_embd: int) -> int:
    """Calculate the key-value dimension for grouped-query attention.

    TODO: Consider a better place for this function.

    Args:
        n_heads_in_group: Number of attention heads per key-value group.
        n_head: Total number of attention heads.
        n_embd: Embedding dimension.

    Returns:
        Combined dimension for key and value tensors (2 * n_kv_heads * head_size).
    """
    if n_heads_in_group is None:
        return 0
    n_kv_heads = n_head // n_heads_in_group
    head_size = n_embd // n_head
    kv_dim = 2 * n_kv_heads * head_size
    return kv_dim


def raise_unknown_subblock_config_error(subblock_config: Any) -> None:
    """Raise an error for invalid subblock configuration types.

    TODO: Consider a better place for this function.

    Args:
        subblock_config: The invalid subblock configuration object.

    Raises:
        ValueError: Always raised with a message indicating the expected types.
    """
    raise ValueError(
        f"subblock_config should be an instance of FFNConfig or AttentionConfig, instead got {type(subblock_config)}"
    )


def sizeof_dtype(dtype: torch.dtype) -> int | float:
    """Return the size in bytes of the given data type.

    TODO: Consider a better place for this function.

    Args:
        dtype: PyTorch data type or custom type string (e.g., 'nvfp4').

    Returns:
        Size in bytes of the data type. Special case: 'nvfp4' returns ~0.588 bytes.
    """
    if dtype == "nvfp4":
        return 1 / 1.7
    return torch.tensor([], dtype=dtype).element_size()


def load_json(file_path: str):
    """Load and parse a JSON file.

    TODO: Consider a better place for this function.

    Args:
        file_path: Path to the JSON file to load.

    Returns:
        Parsed JSON data as a Python object, or None if the file doesn't exist.
    """
    if not os.path.exists(file_path):
        print("file does not exist {file_path}")
        return None

    with open(file=file_path) as f:
        return json.load(f)


def solution_to_str(block_configs: list[dict[str, Any] | BlockConfig]) -> str:
    """Convert a list of block configurations to a human-readable string representation.

    TODO: Consider a better place for this function.
    Better place for this and subsequent related function would be in __repr__ function in class
    BlockConfig so when we print it or do str(block_config), it automatically
    prints in this custom formatted string

    Args:
        block_configs: List of BlockConfig dataclasses or dicts containing layer configurations.

    Returns:
        Multi-line string with each block's configuration on a separate line.
    """
    block_configs = deepcopy(block_configs)
    reps = []
    for block_idx, block_config in enumerate(block_configs):
        rep = f"block_{block_idx}:".ljust(9)
        rep += block_config_to_str(block_config)
        reps.append(rep)
    rep = "\n".join(reps) + "\n"
    return rep


def block_config_to_str(block_config: BlockConfig | dict[str, Any] | None) -> str | None:
    """Convert a BlockConfig to a human-readable string representation.

    TODO: Consider a better place for this function.

    Args:
        block_config: BlockConfig dataclass or dict containing attention and ffn configs.

    Returns:
        Formatted string with attention and FFN information, or None if input is None.
    """
    if block_config is None:
        return None
    rep = ""
    if dataclasses.is_dataclass(block_config):
        block_config = dataclasses.asdict(block_config)
    for subblock_name in ["attention", "ffn"]:
        subblock_config = block_config[subblock_name]
        rep += subblock_config_to_str(subblock_config, subblock_name)
    return rep


def subblock_config_to_str(
    subblock_config: FFNConfig | AttentionConfig | dict[str, Any] | None,
    subblock_name: None | str = None,
) -> str | None:
    """Convert a subblock config (FFN, Attention, Mamba, or MoE) to string.

    TODO: Consider a better place for this function.

    Args:
        subblock_config: FFNConfig, AttentionConfig dataclass or dict.
        subblock_name: Name of subblock ('ffn', 'attention', 'mamba', 'moe').
                      Auto-detected if subblock_config is a dataclass.

    Returns:
        Formatted string showing subblock type and key parameters (e.g., intermediate_size,
        n_heads_in_group), or None if input is None.
    """
    if subblock_config is None:
        return None
    subblock_name = (
        "ffn"
        if isinstance(subblock_config, FFNConfig)
        else "mamba"
        if isinstance(subblock_config, AttentionConfig) and subblock_config.is_mamba
        else "attention"
        if isinstance(subblock_config, AttentionConfig)
        else subblock_name
    )
    assert subblock_name is not None, "Must provide subblock_name if subblock_config is a dict."

    if dataclasses.is_dataclass(subblock_config):
        subblock_config = dataclasses.asdict(subblock_config)

    if subblock_name == "attention" and subblock_config.get("mamba") is not None:
        subblock_name = "mamba"

    if subblock_name == "ffn" and subblock_config.get("moe") is not None:
        subblock_name = "moe"

    rep = f"  {subblock_name}"
    if subblock_config.get("no_op"):
        rep += "  no_op".ljust(8)
    elif subblock_config.get("replace_with_linear"):
        rep += "  linear".ljust(8)
    elif subblock_name == "ffn":
        intermediate_size = subblock_config["intermediate_size"]
        rep += f"  intermediate_{intermediate_size}".ljust(8)
    elif subblock_name == "attention":
        n_heads_in_group = subblock_config["n_heads_in_group"]
        rep += f"  gqa_{n_heads_in_group}".ljust(8)
    elif subblock_name == "mamba":
        mamba_num_heads = subblock_config["mamba"]["num_heads"]
        mamba_head_dim = subblock_config["mamba"]["head_dim"]
        rep += f"  num_heads_{mamba_num_heads}  head_dim_{mamba_head_dim}".ljust(8)
    elif subblock_name == "moe":
        moe_num_local_experts = subblock_config["moe"]["num_local_experts"]
        moe_expert_intermediate_dim = subblock_config["moe"]["expert_intermediate_dim"]
        shared_expert_intermediate_dim = subblock_config["moe"]["shared_expert_intermediate_dim"]
        num_experts_per_tok = subblock_config["moe"]["num_experts_per_tok"]
        rep += f"  num_experts_{moe_num_local_experts}  expert_intermediate_dim_{moe_expert_intermediate_dim}  shared_expert_intermediate_dim_{shared_expert_intermediate_dim}  num_experts_per_tok_{num_experts_per_tok}".ljust(
            8
        )
    else:
        raise ValueError(f"subblock_config_to_str: unrecognized subblock_name: {subblock_name}.")

    return rep


class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):
    def __init__(self, device=None, dtype=None):
        """Create tensors with given device and dtype and don't run initialization
           (but instead use "empty tensors", i.e. uninitialized memory).

            device: `torch.device` to work with
            dtype: `torch.dtype` to work with

        Example::
            with EmptyInitOnDevice("cuda", dtype=torch.bfloat16):
                model = LLaMA(model_config)
            model.load_state_dict(torch.load("llama-lit/7B/lit-llama.pth"))
        """
        self.device = device
        self.dtype = dtype

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            else:
                return args[0]
        if (
            self.device is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("device") is None
        ):
            kwargs["device"] = self.device
        if (
            self.dtype is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("dtype") is None
        ):
            kwargs["dtype"] = self.dtype
        return func(*args, **kwargs)
