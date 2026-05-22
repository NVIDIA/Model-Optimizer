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
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors.torch import safe_open
from tqdm import tqdm

_HF_HUB_OFFLINE_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


def _is_hf_hub_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in _HF_HUB_OFFLINE_TRUE_VALUES


def _copy_python_files(source_dir: Path, save_dir: Path) -> None:
    for py_file in source_dir.glob("*.py"):
        shutil.copy2(py_file, save_dir / py_file.name)


def copy_hf_ckpt_remote_code(
    pretrained_model_path: str | os.PathLike, save_directory: str | os.PathLike
):
    """Copy remote code from pretrained model to save directory.

    For models that keep configuration and modeling files as part of the checkpoint,
    we need to copy them to the export directory for seamless integration with inference
    frameworks.

    If ``pretrained_model_path`` is a local directory, Python files are copied directly.
    If it's a HF Hub model ID (e.g. ``nvidia/NVIDIA-Nemotron-Nano-12B-v2``), the Hub
    snapshot is resolved first and Python files are copied from that snapshot. When
    ``HF_HUB_OFFLINE`` is set, the snapshot must already be available in the local
    Hugging Face cache.

    Args:
        pretrained_model_path: Local path to the pretrained model or HuggingFace Hub model ID.
        save_directory: Path to the save directory.
    """
    hf_checkpoint_path = Path(pretrained_model_path)
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    if hf_checkpoint_path.is_dir():
        _copy_python_files(hf_checkpoint_path, save_dir)
    else:
        local_files_only = _is_hf_hub_offline()
        try:
            source_dir = Path(
                snapshot_download(
                    repo_id=str(pretrained_model_path),
                    allow_patterns=["*.py"],
                    local_files_only=local_files_only,
                )
            )
        except LocalEntryNotFoundError as exc:
            if local_files_only:
                raise RuntimeError(
                    f"Could not copy Python sidecar files for {pretrained_model_path!r} because "
                    "HF_HUB_OFFLINE is enabled and the files are not available in the local "
                    "Hugging Face cache. Populate the cache with the model's *.py files or pass "
                    "a local pretrained model directory."
                ) from exc
            raise

        _copy_python_files(source_dir, save_dir)


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


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
)


def copy_tokenizer_from_local_ckpt(src: str | os.PathLike, dst: str | os.PathLike):
    """Copy tokenizer files verbatim from a local HF checkpoint dir.

    Preserves the pre-transformers-v5 PreTrainedTokenizer(Fast) layout (full
    ``added_tokens_decoder`` in ``tokenizer_config.json`` + ``special_tokens_map.json``).

    Args:
        src: Source HF checkpoint directory. Must be a local path.
        dst: Destination directory; created if missing.
    """
    if not os.path.isdir(src):
        raise ValueError(f"Invalid source path: {src}. It should be a directory.")
    os.makedirs(dst, exist_ok=True)
    for fn in _TOKENIZER_FILES:
        p = os.path.join(src, fn)
        if os.path.isfile(p):
            shutil.copy2(p, dst)
