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
import warnings
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors.torch import safe_open
from tqdm import tqdm

_HF_HUB_OFFLINE_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


def _as_nonnegative_int(value: Any) -> int | None:
    """Return ``value`` as an int when it is a non-negative integer."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _count_mtp_layer_prefixes(prefixes: list[Any] | tuple[Any, ...]) -> int | None:
    """Count actual MTP layer prefixes, excluding broad prefixes like ``mtp``."""
    layer_prefixes = {
        prefix
        for prefix in prefixes
        if isinstance(prefix, str)
        and (parts := prefix.split("."))
        and len(parts) >= 2
        and parts[-2] == "layers"
        and parts[-1].isdigit()
    }
    return len(layer_prefixes) or None


def _get_num_nextn_predict_layers(config_data: dict[str, Any], model: Any) -> int | None:
    """Get the number of next-token-prediction layers from config metadata."""
    num_nextn_predict_layers = _as_nonnegative_int(config_data.get("num_nextn_predict_layers"))
    if num_nextn_predict_layers is not None:
        return num_nextn_predict_layers

    model_config = getattr(model, "config", None)
    if model_config is not None:
        num_nextn_predict_layers = _as_nonnegative_int(
            getattr(model_config, "num_nextn_predict_layers", None)
        )
        if num_nextn_predict_layers is not None:
            return num_nextn_predict_layers

    mtp_layer_prefixes = getattr(model, "_mtp_layer_prefixes", None)
    if isinstance(mtp_layer_prefixes, (list, tuple)):
        return _count_mtp_layer_prefixes(mtp_layer_prefixes)

    return None


def _get_rope_theta(config_data: dict[str, Any], model: Any) -> Any:
    rope_theta = config_data.get("rope_theta")
    if rope_theta is not None:
        return rope_theta

    model_config = getattr(model, "config", None)
    if model_config is None:
        return None

    return getattr(model_config, "rope_theta", None)


def _sanitize_llama3_rope_config(config_data: dict[str, Any], model: Any) -> None:
    rope_theta = _get_rope_theta(config_data, model)
    if rope_theta is None:
        return

    for key in ("rope_parameters", "rope_scaling"):
        rope_config = config_data.get(key)
        if not isinstance(rope_config, dict):
            continue

        rope_type = rope_config.get("rope_type", rope_config.get("type"))
        if rope_type == "llama3" and "rope_theta" not in rope_config:
            rope_config["rope_theta"] = rope_theta


def sanitize_hf_config_for_deployment(config_data: dict[str, Any], model: Any) -> None:
    """Sanitize exported Hugging Face config metadata for deployment runtimes.

    Transformers 5.x validates that ``len(layer_types) == num_hidden_layers``.
    Some checkpoints include MTP/next-token-prediction layer entries after the
    main decoder layer entries. Those extra entries are not part of
    ``num_hidden_layers`` and make deployment stacks fail while loading config.
    """
    _sanitize_llama3_rope_config(config_data, model)

    num_hidden_layers = _as_nonnegative_int(config_data.get("num_hidden_layers"))
    layer_types = config_data.get("layer_types")
    if num_hidden_layers is None or not isinstance(layer_types, list):
        return

    num_layer_types = len(layer_types)
    if num_layer_types == num_hidden_layers:
        return

    num_nextn_predict_layers = _get_num_nextn_predict_layers(config_data, model)
    if (
        num_layer_types > num_hidden_layers
        and num_nextn_predict_layers == num_layer_types - num_hidden_layers
    ):
        warnings.warn(
            "Trimming config.layer_types from "
            f"{num_layer_types} to {num_hidden_layers} entries so it matches "
            "num_hidden_layers; the removed entries correspond to "
            "num_nextn_predict_layers.",
            stacklevel=2,
        )
        config_data["layer_types"] = layer_types[:num_hidden_layers]


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
    prefixes: tuple[str, ...] = ("multi_modal_projector", "vision_model"),
) -> dict[str, torch.Tensor]:
    """Load multimodal components from safetensors file.

    Args:
        pretrained_model_path: Path to the pretrained model.
        prefixes: Tensor key prefixes to select.  Defaults to the LLaVA-style
            ``multi_modal_projector`` / ``vision_model`` prefixes.  Pass
            ``("model.visual.",)`` for Qwen3-VL checkpoints.

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
                if key.startswith(prefixes)
            ]
            for key in tqdm(multimodal_keys, desc="Loading multimodal tensors"):
                multimodal_state_dict[key] = f.get_tensor(key)

    elif safetensors_index_file.is_file():
        print(f"Loading multimodal components from sharded model: {hf_checkpoint_path}")
        with open(safetensors_index_file) as f:
            safetensors_index = json.load(f)

        all_shard_files = sorted(set(safetensors_index["weight_map"].values()))
        for shard_file in all_shard_files:
            safetensors_filepath = Path(hf_checkpoint_path) / shard_file
            with safe_open(safetensors_filepath, framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    if key.startswith(prefixes):
                        multimodal_state_dict[key] = f.get_tensor(key)

    else:
        print(f"Warning: No safetensors files found in {hf_checkpoint_path}")

    print(f"Successfully loaded {len(multimodal_state_dict)} multimodal tensors")
    return multimodal_state_dict


def copy_non_safetensor_files_from_ckpt(src: str | os.PathLike, dst: str | os.PathLike):
    """Copy every non-safetensors file from a local HF checkpoint dir verbatim.

    Use as a baseline so tokenizer files, remote_code ``*.py``, README, LICENSE, etc.
    are preserved from the source. The caller is expected to overwrite the files
    modelopt owns (``config.json``, ``generation_config.json``, ``hf_quant_config.json``,
    ``preprocessor_config.json``) after this step.

    Args:
        src: Source HF checkpoint directory. Must be a local path.
        dst: Destination directory; created if missing.
    """
    if not os.path.isdir(src):
        raise ValueError(f"Invalid source path: {src}. It should be a directory.")
    os.makedirs(dst, exist_ok=True)
    for entry in os.listdir(src):
        sp = os.path.join(src, entry)
        if not os.path.isfile(sp):
            continue
        if entry.endswith(".safetensors") or entry == "model.safetensors.index.json":
            continue
        shutil.copy2(sp, dst)
