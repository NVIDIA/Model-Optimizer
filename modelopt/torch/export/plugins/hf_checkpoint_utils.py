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

import fnmatch
import json
import os
import shutil
import warnings
from collections.abc import Iterable
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
    """Return rope_theta from exported config data or the in-memory model config."""
    rope_theta = config_data.get("rope_theta")
    if rope_theta is not None:
        return rope_theta

    model_config = getattr(model, "config", None)
    if model_config is None:
        return None

    return getattr(model_config, "rope_theta", None)


def _sanitize_llama3_rope_config(config_data: dict[str, Any], model: Any) -> None:
    """Fill missing llama3 rope_theta in rope config metadata when available."""
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


# transformers' native NemotronHConfig (registered as model_type "nemotron_h") stores
# layer types as a list and derives ``hybrid_override_pattern``/``num_hidden_layers`` as
# read-only properties (transformers/models/nemotron_h/configuration_nemotron_h.py).
# Plain ``to_dict()`` serialization -- used by ``save_pretrained`` -- only captures
# ``__dict__``, so neither property makes it into the exported config.json. Many
# NemotronH checkpoints on the Hub (e.g. nvidia/Nemotron-Cascade-2-30B-A3B) still ship
# their own older, bundled trust_remote_code configuration_nemotron_h.py written against
# the *opposite* schema: ``hybrid_override_pattern``/``num_hidden_layers`` as real
# fields, with ``layers_block_type`` as a computed property that has no setter. Loading
# such a checkpoint's exported config.json back through its own bundled class then fails
# (or, if it doesn't fail outright, silently loses the pattern/layer-count metadata).
# Reproduced three times in a row on real nvidia/Nemotron-Cascade-2-30B-A3B NVFP4
# exports. This mapping is the exact inverse of transformers' own
# ``NemotronHConfig._list_to_pattern``, so it reconstructs precisely what a native
# transformers load would have derived.
_NEMOTRON_H_PATTERN_CHAR_BY_LAYER_TYPE = {
    "linear_attention": "M",
    "moe": "E",
    "full_attention": "*",
    "mlp": "-",
}


def _restore_nemotron_h_legacy_schema_if_dropped(config_data: dict[str, Any]) -> None:
    """Reconstruct NemotronH's legacy ``hybrid_override_pattern``/``num_hidden_layers``.

    No-op for anything but a NemotronH export (``model_type == "nemotron_h"``), and a
    no-op if ``hybrid_override_pattern`` is already present (nothing was dropped). If
    ``layers_block_type`` contains a value outside the four known block types, this
    warns and leaves ``config.json`` in the new schema rather than guessing.
    """
    if config_data.get("model_type") != "nemotron_h":
        return
    layer_types = config_data.get("layers_block_type")
    if not isinstance(layer_types, list) or "hybrid_override_pattern" in config_data:
        return

    try:
        pattern = "".join(
            _NEMOTRON_H_PATTERN_CHAR_BY_LAYER_TYPE[layer_type] for layer_type in layer_types
        )
    except (KeyError, TypeError) as e:
        # KeyError: a recognized-but-unlisted string block type. TypeError: a malformed
        # entry (list, dict, ...) that isn't even hashable for the dict lookup.
        warnings.warn(
            f"Cannot reconstruct NemotronH's legacy hybrid_override_pattern: unrecognized "
            f"layers_block_type entry {e}. Leaving config.json in the new schema; a "
            "bundled trust_remote_code configuration_nemotron_h.py written against the "
            "legacy schema may fail to load it.",
            stacklevel=2,
        )
        return

    config_data["hybrid_override_pattern"] = pattern
    config_data["num_hidden_layers"] = len(layer_types)
    del config_data["layers_block_type"]
    config_data.pop("mtp_layers_block_type", None)


def sanitize_hf_config_for_deployment(config_data: dict[str, Any], model: Any) -> None:
    """Sanitize exported Hugging Face config metadata for deployment runtimes.

    Fix conservative deployment-only config incompatibilities:

    * add missing llama3 ``rope_theta`` metadata when available;
    * trim trailing MTP/next-token-prediction ``layer_types`` entries only when
      the mismatch is exactly explained by next-token-prediction metadata;
    * reconstruct NemotronH's legacy ``hybrid_override_pattern``/``num_hidden_layers``
      when a native-transformers export dropped them (see
      :func:`_restore_nemotron_h_legacy_schema_if_dropped`).
    """
    _sanitize_llama3_rope_config(config_data, model)
    _restore_nemotron_h_legacy_schema_if_dropped(config_data)

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
        pretrained_model_path: Directory or HuggingFace repo id of the pretrained model.
        prefixes: Tensor key prefixes to select.  Defaults to the LLaVA-style
            ``multi_modal_projector`` / ``vision_model`` prefixes.  Pass
            ``("model.visual.",)`` for Qwen3-VL checkpoints.

    Returns:
        A dictionary of multimodal components.
    """
    hf_checkpoint_path = Path(pretrained_model_path)
    if not hf_checkpoint_path.is_dir():
        # Also accept a repo id, which is what the example scripts pass to quantize.py.
        # Fetched in two stages: the vision tower is a small fraction of a VLM checkpoint, so
        # pulling every shard to keep a few would waste tens of GB.
        local_files_only = _is_hf_hub_offline()
        repo_id = str(pretrained_model_path)
        try:
            index_dir = Path(
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=["model.safetensors.index.json"],
                    local_files_only=local_files_only,
                )
            )
        except (LocalEntryNotFoundError, OSError, ValueError) as exc:
            raise ValueError(
                f"Invalid pretrained model path: {pretrained_model_path}. It should be a "
                "directory or an available HuggingFace repo id."
            ) from exc

        index_file = index_dir / "model.safetensors.index.json"
        if index_file.is_file():
            try:
                weight_map = json.loads(index_file.read_text())["weight_map"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Malformed safetensors index in {repo_id}.") from exc
            wanted = sorted(
                {shard for key, shard in weight_map.items() if key.startswith(prefixes)}
            )
        else:
            wanted = ["model.safetensors"]  # unsharded checkpoint
        # Kept separate from the resolution failure above: a hub outage or a full disk here is
        # retryable, not a bad path.
        hf_checkpoint_path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=["model.safetensors.index.json", *wanted],
                local_files_only=local_files_only,
            )
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

        all_shard_files = sorted(
            {
                shard
                for key, shard in safetensors_index["weight_map"].items()
                if key.startswith(prefixes)
            }
        )
        for shard_file in all_shard_files:
            safetensors_filepath = Path(hf_checkpoint_path) / shard_file
            with safe_open(safetensors_filepath, framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    if key.startswith(prefixes):
                        multimodal_state_dict[key] = f.get_tensor(key)

    else:
        print(f"Warning: No safetensors files found in {hf_checkpoint_path}")

    if not multimodal_state_dict:
        raise ValueError(
            f"No tensors under {prefixes} in {pretrained_model_path}; the vision tower would be "
            "missing from the export. The checkpoint's prefixes have likely changed."
        )

    print(f"Successfully loaded {len(multimodal_state_dict)} multimodal tensors")
    return multimodal_state_dict


def _matches_any_pattern(file_name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(file_name, pattern) for pattern in patterns)


def copy_non_safetensor_files_from_ckpt(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    *,
    exclude_files: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
) -> list[str]:
    """Copy every non-safetensors file from a local HF checkpoint dir verbatim.

    Use as a baseline so tokenizer files, remote_code ``*.py``, README, LICENSE, etc.
    are preserved from the source. Callers can exclude additional files or patterns when
    copying after export-owned metadata has already been written.

    Args:
        src: Source HF checkpoint directory. Must be a local path.
        dst: Destination directory; created if missing.
        exclude_files: Exact file names to skip.
        exclude_patterns: Glob patterns for additional files to skip.

    Returns:
        File names copied into ``dst``.
    """
    if not os.path.isdir(src):
        raise ValueError(f"Invalid source path: {src}. It should be a directory.")
    exclude_files = set(exclude_files or ())
    exclude_patterns = tuple(exclude_patterns or ())
    copied_files = []
    os.makedirs(dst, exist_ok=True)
    for entry in sorted(os.listdir(src)):
        if entry in exclude_files or _matches_any_pattern(entry, exclude_patterns):
            continue
        sp = os.path.join(src, entry)
        if not os.path.isfile(sp):
            continue
        if entry.endswith(".safetensors") or entry == "model.safetensors.index.json":
            continue
        try:
            shutil.copy2(sp, dst)
        except OSError as error:
            warnings.warn(f"Failed to copy checkpoint sidecar {entry}: {error}")
            continue
        copied_files.append(entry)
    return copied_files
