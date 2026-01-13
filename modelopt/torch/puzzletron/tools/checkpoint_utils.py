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
# mypy: ignore-errors

"""It provides general utilities for loading and initializing PyTorch model checkpoints,
particularly for DeciLM models.
"""

import concurrent.futures
import warnings
from functools import partial
from pathlib import Path
from typing import Literal, TypeVar

import torch
from safetensors.torch import load_file as safe_load_file
from torch import nn
from transformers import AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import load_model_config
from modelopt.torch.puzzletron.tools.common import infer_weights_dtype

SAFETENSORS_SUBBLOCKS_DIR_NAME = "subblocks_safetensors"
PTH_SUBBLOCKS_DIR_NAME = "subblocks"
STATE_DICT_FILE_NAME = "model.pth"

warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")


def load_state_dict(checkpoint_dir: Path | str) -> dict[str, torch.Tensor]:
    checkpoint_dir = _normalize_checkpoint_dir(checkpoint_dir)

    if (state_dict_path := checkpoint_dir / STATE_DICT_FILE_NAME).exists():
        return torch.load(state_dict_path, map_location="cpu", weights_only=False)

    if (safetensors_subblocks_dir := checkpoint_dir / SAFETENSORS_SUBBLOCKS_DIR_NAME).exists():
        return _load_state_dict_from_subblocks(safetensors_subblocks_dir)

    if (pth_subblocks_dir := checkpoint_dir / PTH_SUBBLOCKS_DIR_NAME).exists():
        return _load_state_dict_from_subblocks(pth_subblocks_dir)

    if (checkpoint_dir / SAFE_WEIGHTS_INDEX_NAME).exists() or (
        checkpoint_dir / SAFE_WEIGHTS_NAME
    ).exists():
        from modelopt.torch.puzzletron.tools.sharded_checkpoint_utils import (
            load_sharded_state_dict,  # local import to avoid circular import
        )

        return load_sharded_state_dict(checkpoint_dir)

    raise FileNotFoundError(
        f"Couldn't find state dict path or subblocks dir inside {checkpoint_dir}"
    )


def _normalize_checkpoint_dir(checkpoint_dir: Path | str) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent
    return checkpoint_dir


def _load_state_dict_from_subblocks(subblocks_dir: Path) -> dict[str, torch.Tensor]:
    torch_paths = list(subblocks_dir.glob("*.pth"))
    safetensors_paths = list(subblocks_dir.glob("*.safetensors"))

    if len(torch_paths) != 0:
        load_fn = partial(torch.load, map_location="cpu", weights_only=False)
        file_paths = torch_paths
    elif len(safetensors_paths) != 0:
        load_fn = safe_load_file
        file_paths = safetensors_paths
    else:
        raise ValueError(f"No tensor files found in {subblocks_dir=}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        state_dict_shards = list(executor.map(load_fn, file_paths))

    state_dict = {k: v for shard in state_dict_shards for k, v in shard.items()}
    return state_dict


NNModule = TypeVar("NNModule", bound=nn.Module)


def init_module_with_state_dict(
    state_dict: dict[str, torch.Tensor],
    module_cls: type[NNModule],
    *init_args,
    **init_kwargs,
) -> NNModule:
    weights_dtype = infer_weights_dtype(state_dict)
    module = init_empty_module(module_cls, weights_dtype, *init_args, **init_kwargs)
    module.load_state_dict(state_dict)
    return module


def init_empty_module(
    module_cls: type[NNModule],
    dtype: torch.dtype,
    *init_args,
    **init_kwargs,
) -> NNModule:
    default_dtype = torch.get_default_dtype()
    current_device = torch.ones(1).device
    torch.set_default_dtype(dtype)
    module = skip_init(module_cls, *init_args, device=current_device, **init_kwargs)
    torch.set_default_dtype(default_dtype)
    return module


def skip_init(module_cls, *args, **kwargs) -> nn.Module:
    """Heavily inspired by torch.nn.utils.skip_init but does not require the module to accept a "device" kwarg."""
    if not issubclass(module_cls, torch.nn.Module):
        raise RuntimeError(f"Expected a Module; got {module_cls}")

    final_device = kwargs.pop("device", "cpu")
    with torch.device("meta"):
        module = module_cls(*args, **kwargs)

    module = module.to_empty(device=final_device)
    return module


def is_valid_decilm_checkpoint(checkpoint_dir: Path | str) -> bool:
    """Validate that a checkpoint is in DeciLM format (has block_configs).

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        True if checkpoint is valid DeciLM format, False otherwise
    """
    try:
        model_config = load_model_config(checkpoint_dir)
        if model_config.block_configs is None:
            warnings.warn(
                f"Skipping checkpoint '{checkpoint_dir}' - not in DeciLM format (missing block_configs)"
            )
            return False
        return True
    except Exception as e:
        warnings.warn(f"Skipping checkpoint '{checkpoint_dir}' - failed to load config: {e}")
        return False


def copy_tokenizer(
    source_dir_or_tokenizer_name: Path | str,
    target_dir: Path | str,
    on_failure: Literal["raise", "warn"] = "raise",
) -> None:
    """Prefer loading the tokenizer from huggingface hub (when tokenizer_name.txt file is available)
    to avoid collision between transformers versions.
    """
    source_tokenizer_name_path = Path(source_dir_or_tokenizer_name) / "tokenizer_name.txt"
    if source_tokenizer_name_path.exists():
        source_dir_or_tokenizer_name = source_tokenizer_name_path.read_text().strip()

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source_dir_or_tokenizer_name, trust_remote_code=True
        )
    except Exception:
        message = f"Couldn't load tokenizer from '{source_dir_or_tokenizer_name}'"
        if on_failure == "raise":
            raise FileNotFoundError(message)
        else:
            warnings.warn(message)

    if tokenizer is not None:
        target_dir = Path(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)
        tokenizer.save_pretrained(target_dir)

        target_tokenizer_name_path = target_dir / "tokenizer_name.txt"
        is_given_tokenizer_name_as_argument = not Path(source_dir_or_tokenizer_name).exists()
        if is_given_tokenizer_name_as_argument:
            target_tokenizer_name_path.write_text(source_dir_or_tokenizer_name)
