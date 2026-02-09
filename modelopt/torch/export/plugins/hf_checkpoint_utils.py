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

"""Hugging Face checkpoint utility."""

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import safe_open
from tqdm import tqdm


def copy_remote_code(
    pretrained_model_path: str | os.PathLike,
    save_directory: str | os.PathLike,
):
    """Copy remote code from pretrained model to save directory.

    For models that keep configuration and modeling files as part of the checkpoint,
    we need to copy them to the export directory for seamless integration with inference
    frameworks.

    Args:
        pretrained_model_path: Path to the pretrained model.
        save_directory: Path to the save directory.

    Raises:
        ValueError: If the pretrained model path is not a directory.
    """
    hf_checkpoint_path = Path(pretrained_model_path)
    save_dir = Path(save_directory)

    if not hf_checkpoint_path.is_dir():
        raise ValueError(
            f"Invalid pretrained model path: {pretrained_model_path}. It should be a directory."
        )

    for py_file in hf_checkpoint_path.glob("*.py"):
        if py_file.is_file():
            shutil.copy(py_file, save_dir / py_file.name)


def load_multimodal_components(
    pretrained_model_path: str | os.PathLike,
) -> dict[str, torch.Tensor]:
    """Load multimodal components from safetensors file.

    Args:
        pretrained_model_path: Path to the pretrained model.

    Returns:
        A dictionary of multimodal components.
    """
    hf_checkpoint_path = Path(pretrained_model_path)
    if not hf_checkpoint_path.is_dir():
        raise ValueError(
            f"Invalid pretrained model path: {pretrained_model_path}. It should be a directory."
        )

    safetensors_file = Path(hf_checkpoint_path) / "model.safetensors"
    safetensors_index_file = Path(hf_checkpoint_path) / "model.safetensors.index.json"

    multimodal_state_dict = {}

    if safetensors_file.is_file():
        print(f"Loading multimodal components from single file: {safetensors_file}")
        with safe_open(safetensors_file, framework="pt") as f:
            multimodal_keys = [
                key
                for key in f.keys()  # noqa: SIM118
                if key.startswith(("multi_modal_projector", "vision_model"))
            ]
            for key in tqdm(multimodal_keys, desc="Loading multimodal tensors"):
                multimodal_state_dict[key] = f.get_tensor(key)

    elif safetensors_index_file.is_file():
        print(f"Loading multimodal components from sharded model: {hf_checkpoint_path}")
        with open(safetensors_index_file) as f:
            safetensors_index = json.load(f)

        # For multimodal models, vision_model and multi_modal_projector are in the first shard
        all_shard_files = sorted(set(safetensors_index["weight_map"].values()))
        first_shard_file = all_shard_files[0]  # e.g., "model-00001-of-00050.safetensors"

        # Load multimodal components from the first shard file
        safetensors_filepath = Path(hf_checkpoint_path) / first_shard_file
        print(f"Loading multimodal components from {first_shard_file}")

        with safe_open(safetensors_filepath, framework="pt") as f:
            shard_keys = list(f.keys())
            multimodal_keys_in_shard = [
                k for k in shard_keys if k.startswith(("multi_modal_projector", "vision_model"))
            ]

            if multimodal_keys_in_shard:
                print(
                    f"Found {len(multimodal_keys_in_shard)} multimodal tensors in {first_shard_file}"
                )
                for key in tqdm(multimodal_keys_in_shard, desc="Loading multimodal tensors"):
                    multimodal_state_dict[key] = f.get_tensor(key)
            else:
                print(f"No multimodal components found in {first_shard_file}")

    else:
        print(f"Warning: No safetensors files found in {hf_checkpoint_path}")

    print(f"Successfully loaded {len(multimodal_state_dict)} multimodal tensors")
    return multimodal_state_dict
