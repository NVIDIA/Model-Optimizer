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

"""
Utilities for loading and saving Hugging Face-format checkpoints (``AutoConfig`` + optional ``block_configs``).
"""

import concurrent.futures
import dataclasses
import fcntl
import os
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import torch
import transformers
from safetensors.torch import save_file as safe_save_file
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from modelopt.torch.utils import json_dumps

from ..block_config import maybe_cast_block_configs

if TYPE_CHECKING:
    from ..anymodel.model_descriptor import ModelDescriptor
from .logger import mprint

__all__ = [
    "SAFETENSORS_SUBBLOCKS_DIR_NAME",
    "PTH_SUBBLOCKS_DIR_NAME",
    "RELATIVE_SUBBLOCKS_DIR",
    "force_cache_dynamic_modules",
    "load_model_config",
    "init_model_from_config",
    "save_checkpoint",
    "save_subblocks",
    "save_model_config",
]

SAFETENSORS_SUBBLOCKS_DIR_NAME = "subblocks_safetensors"
PTH_SUBBLOCKS_DIR_NAME = "subblocks"
RELATIVE_SUBBLOCKS_DIR = Path(SAFETENSORS_SUBBLOCKS_DIR_NAME)


# TODO: (esegal) Should ask the model for something like this
NON_LAYER_MODULE_TO_FILE_TYPE = {
    "model.embed_tokens": "embeddings",
    "model.norm": "lm_head",
    "lm_head": "lm_head",
}
MODULE_WITHIN_LAYER_TO_FILE_TYPE = {
    "input_layernorm": "attention",
    "self_attn": "attention",
    "post_attention_layernorm": "ffn",
    "mlp": "ffn",
    "parallel_blocks": "multi_block",
}
LAYERS_MODULE_NAME = "model.layers"


def force_cache_dynamic_modules(
    config: PretrainedConfig, checkpoint_dir: Path | str, trust_remote_code: bool = False
):
    has_remote_code = (
        hasattr(config, "auto_map")
        and isinstance(config.auto_map, dict)
        and "AutoConfig" in config.auto_map.keys()
    )
    if has_remote_code and trust_remote_code:
        for class_reference in config.auto_map.values():
            _ = get_class_from_dynamic_module(class_reference, checkpoint_dir)


def load_model_config(
    checkpoint_dir: Path | str,
    model_config_overrides: Mapping | None = None,
    ignore_unexpected_config_keys: bool = False,
    trust_remote_code: bool = False,
):
    """Load model configuration from a checkpoint directory.

    Args:
        checkpoint_dir: Path to the checkpoint directory (e.g. containing config.json).
        model_config_overrides: Optional mapping of config overrides.
        ignore_unexpected_config_keys: If True, ignore unexpected config keys.
        trust_remote_code: If True, allows execution of custom code from the model repository.
            This is a security risk if the model source is untrusted. Only set to True if you
            trust the source of the model. Defaults to False for security.

    Returns:
        Loaded model configuration (PretrainedConfig).
    """
    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    if model_config_overrides is None:
        model_config_overrides = {}

    config, unused_kwargs = AutoConfig.from_pretrained(
        checkpoint_dir,
        trust_remote_code=trust_remote_code,
        return_unused_kwargs=True,
        **model_config_overrides,
    )
    if hasattr(config, "block_configs"):
        config.block_configs = maybe_cast_block_configs(config.block_configs)

    force_cache_dynamic_modules(config, checkpoint_dir, trust_remote_code=trust_remote_code)

    if not ignore_unexpected_config_keys:
        if unused_kwargs:
            raise ValueError(f"Unexpected config keys: {unused_kwargs.keys()}")

    return config


def _get_model_class_from_config(config: PretrainedConfig) -> type:
    """Resolve HuggingFace model class from ``config.architectures`` (see puzzletron checkpoint_utils_hf)."""
    if hasattr(config, "architectures") and config.architectures:
        model_class_name = config.architectures[0]
        if hasattr(transformers, model_class_name):
            return getattr(transformers, model_class_name)
        mprint(
            f"Warning: {model_class_name} not found in transformers, "
            "falling back to AutoModelForCausalLM"
        )
    return AutoModelForCausalLM


def _get_auto_class_for_trust_remote_code(config: PretrainedConfig) -> type:
    """Pick the right Auto class for a trust_remote_code model by inspecting auto_map.

    When a model requires trust_remote_code, the native transformers class resolved from
    config.architectures must NOT be used directly — it may have a different module structure
    than the trust_remote_code class (e.g. NemotronH: native uses ``model.`` prefix, but the
    trust_remote_code class uses ``backbone.`` prefix, causing key mismatches throughout the
    pipeline). Instead, we route through the appropriate Auto class so that from_config()
    resolves the class via auto_map, picking up the correct trust_remote_code implementation.

    Models declare which Auto class they support via config.auto_map. We walk a priority list
    so that CausalLM models and VL models (AutoModelForConditionalGeneration or similar) are
    both handled correctly.
    """
    auto_map = getattr(config, "auto_map", {})
    priority = [
        "AutoModelForCausalLM",
        "AutoModelForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModel",
    ]
    for name in priority:
        if name in auto_map and hasattr(transformers, name):
            return getattr(transformers, name)
    return AutoModelForCausalLM


def init_model_from_config(
    config: PretrainedConfig,
    *,
    trust_remote_code: bool = False,
    **kwargs,
) -> PreTrainedModel:
    """Build a model from config on meta/uninitialized weights (used e.g. for subblock param counts).

    ``trust_remote_code`` defaults to False (only ``AutoModelForCausalLM.from_config`` uses it).
    Pass True when loading configs that rely on custom modeling code from the checkpoint.
    """
    model_class = _get_model_class_from_config(config)
    if trust_remote_code:
        auto_cls = _get_auto_class_for_trust_remote_code(config)
        return auto_cls.from_config(config, trust_remote_code=trust_remote_code, **kwargs)
    if model_class is AutoModelForCausalLM:
        return AutoModelForCausalLM.from_config(config, **kwargs)
    # Concrete model classes (e.g. GptOssForCausalLM, Qwen3VLMoeForConditionalGeneration):
    # _from_config forwards kwargs to __init__, which does not accept trust_remote_code.
    return model_class._from_config(config, **kwargs)


def save_checkpoint(
    model: PreTrainedModel, checkpoint_dir: Path | str, descriptor: "ModelDescriptor"
) -> None:
    _save_checkpoint(model.config, model.state_dict(), checkpoint_dir, descriptor)


def _save_checkpoint(
    model_config: PretrainedConfig,
    state_dict: dict[str, torch.Tensor],
    checkpoint_dir: Path | str,
    descriptor: "ModelDescriptor",
    max_workers: int | None = None,  # Now optional - will auto-calculate if None
) -> None:
    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Save config
    save_model_config(model_config, checkpoint_dir)

    # Phase 2: Build weight map using descriptor and write index
    subblock_keys = descriptor.get_weight_groups(
        layer_names=state_dict.keys(),
        num_hidden_layers=model_config.num_hidden_layers,
    )

    weight_map = {}
    for subblock, layer_keys in subblock_keys.items():
        weight_map_entries = {
            key: f"subblocks_safetensors/{subblock}.safetensors" for key in layer_keys
        }
        weight_map.update(weight_map_entries)

    # Handle tie_word_embeddings - remove from state_dict and weight_map BEFORE writing index
    output_emb_weight_name = f"{descriptor.output_embedding_name()}.weight"
    if getattr(model_config, "tie_word_embeddings", False) and output_emb_weight_name in state_dict:
        state_dict = {k: v for k, v in state_dict.items() if k != output_emb_weight_name}
        weight_map = {k: v for k, v in weight_map.items() if k != output_emb_weight_name}

    # Write index (now without tied embedding)
    index = {"metadata": {"format": "pt"}, "weight_map": weight_map}
    index_path = checkpoint_dir / SAFE_WEIGHTS_INDEX_NAME
    index_json = json_dumps(index)
    _write_file_process_safe(index_json, index_path)

    # Phase 3: Save subblocks
    save_subblocks(
        state_dict,
        checkpoint_dir,
        weight_map=weight_map,
        multi_threaded=True,
        max_workers=max_workers,
    )


def save_subblocks(
    state_dict: dict[str, torch.Tensor],
    checkpoint_dir: Path | str,
    weight_map: dict[str, str] | None = None,
    multi_threaded: bool = True,
    max_workers: int | None = None,  # Now optional - will auto-calculate if None
) -> None:
    mprint("=== Starting save_subblocks detailed profiling ===")
    subblocks_start_time = time.time()

    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    # Step 1: Build weight map (use provided or build from state_dict)
    weight_map_start_time = time.time()
    if weight_map is None:
        weight_map = _build_safetensors_weight_map(
            state_dict=state_dict,
            non_layer_module_to_file_type=NON_LAYER_MODULE_TO_FILE_TYPE,
            module_within_layer_to_file_type=MODULE_WITHIN_LAYER_TO_FILE_TYPE,
            layers_module_name=LAYERS_MODULE_NAME,
        )
    weight_name_to_filename = {k: checkpoint_dir / v for k, v in weight_map.items()}
    weight_map_time = time.time() - weight_map_start_time
    mprint(f"  Step 1 - Build weight map: {weight_map_time:.2f}s ({len(weight_map)} mappings)")

    # Step 2: Create subblocks directory
    dir_create_start_time = time.time()
    subblocks_path = checkpoint_dir / SAFETENSORS_SUBBLOCKS_DIR_NAME
    subblocks_path.mkdir(parents=True, exist_ok=True)
    dir_create_time = time.time() - dir_create_start_time
    mprint(f"  Step 2 - Create directory: {dir_create_time:.2f}s")

    # Step 3: Organize tensors by file
    organize_start_time = time.time()
    filename_to_partial_state_dict = defaultdict(dict)
    total_tensor_size = 0
    for weight_name, weight in state_dict.items():
        if weight_name in weight_map:
            # Ensure tensor is contiguous and on CPU for faster I/O
            tensor = (
                weight.contiguous().cpu() if weight.device.type != "cpu" else weight.contiguous()
            )
            filename_to_partial_state_dict[weight_name_to_filename[weight_name]][weight_name] = (
                tensor
            )
            total_tensor_size += weight.numel() * weight.element_size()
    organize_time = time.time() - organize_start_time
    mprint(
        f"  Step 3 - Organize tensors: {organize_time:.2f}s ({total_tensor_size / (1024**3):.2f}GB total)"
    )

    # Step 4: Prepare save arguments and auto-calculate optimal I/O workers
    prepare_start_time = time.time()
    safe_save_kwargs = [
        {"tensors": partial_state_dict, "filename": filename, "metadata": {"format": "pt"}}
        for filename, partial_state_dict in filename_to_partial_state_dict.items()
    ]

    # Auto-calculate optimal I/O workers: min(cpu_count, num_files)
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        num_files = len(safe_save_kwargs)
        max_workers = min(cpu_count, num_files)
        mprint(
            f"  Auto-calculated I/O workers: min({cpu_count} CPUs, {num_files} files) = {max_workers}"
        )
    else:
        mprint(f"  Using specified I/O workers: {max_workers}")

    prepare_time = time.time() - prepare_start_time
    mprint(f"  Step 4 - Prepare save args: {prepare_time:.2f}s ({len(safe_save_kwargs)} files)")

    # Step 5: Save files with optimal worker count
    save_start_time = time.time()
    if multi_threaded:
        mprint(f"  Using multi-threaded saving with {max_workers} workers...")

        def optimized_safe_save(kwargs):
            try:
                safe_save_file(**kwargs)
                return True
            except Exception as e:
                mprint(f"  Error saving {kwargs['filename']}: {e}")
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(optimized_safe_save, safe_save_kwargs))

        # Check for any failures
        failed_saves = sum(1 for r in results if not r)
        if failed_saves > 0:
            raise RuntimeError(f"  {failed_saves} shard file(s) failed to save")
    else:
        mprint("  Using single-threaded saving...")
        for kwargs in safe_save_kwargs:
            safe_save_file(**kwargs)

    save_time = time.time() - save_start_time
    mprint(f"  Step 5 - Save files: {save_time:.2f}s ({max_workers} workers)")

    subblocks_total_time = time.time() - subblocks_start_time
    mprint(f"=== save_subblocks completed in {subblocks_total_time:.2f}s ===")
    mprint(
        f"  Breakdown: WeightMap {weight_map_time:.1f}s + DirCreate {dir_create_time:.1f}s + "
        f"Organize {organize_time:.1f}s + Prepare {prepare_time:.1f}s + Save {save_time:.1f}s"
    )

    # Calculate effective I/O speed
    io_speed_gbps = (total_tensor_size / (1024**3)) / save_time if save_time > 0 else 0
    mprint(f"  Effective I/O speed: {io_speed_gbps:.2f} GB/s ({max_workers} workers)")
    mprint(f"  Save operation was {save_time / subblocks_total_time * 100:.1f}% of total time")


def _write_text(content: str, f: BinaryIO) -> None:
    f.write(content.encode("utf-8"))


def _write_file_process_safe(
    content: Any,
    path: Path | str,
    write_fn: Callable[[Any, BinaryIO], None] = _write_text,
) -> None:
    """
    Write a file in a multi-process safe way.
    If another process tries to write the same file using this method, the current process
    "gives up" and assumes that the matter is being taken care of by another process.

    write_fn is a function that receives file contents and a binary file object,
    and writes the content to the file. It can be _write_text (defined above), or torch.save,
    or a similar function (not safetensors.torch.save_file since it expects a path).
    """
    # Open with "ab+" so the file is not truncated before the lock is acquired.
    # Once we hold the exclusive lock we seek to the start and truncate explicitly.
    with open(path, "ab+") as f:
        # Try to acquire an exclusive, non-blocking lock
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return  # Exit immediately if the lock is not acquired

        f.seek(0)
        f.truncate()
        write_fn(content, f)  # Write the content if lock is acquired
        f.flush()  # Ensure data is written to disk

        # Release the lock
        fcntl.flock(f, fcntl.LOCK_UN)


def _build_safetensors_weight_map(
    *,
    state_dict: dict[str, torch.Tensor],
    non_layer_module_to_file_type: dict[str, str],
    module_within_layer_to_file_type: dict[str, str],
    layers_module_name: str,
) -> dict[str, Path]:
    weight_map = {}
    unmapped_weight_names = []
    for weight_name in state_dict:
        found_match = False
        for module_name, file_type in non_layer_module_to_file_type.items():
            if weight_name.startswith(f"{module_name}."):
                weight_map[weight_name] = str(RELATIVE_SUBBLOCKS_DIR / f"{file_type}.safetensors")
                found_match = True
        if not found_match:
            if weight_name.startswith(f"{layers_module_name}."):
                name_parts = weight_name[len(layers_module_name) + 1 :].split(".")
                layer_index = name_parts[0]
                name_within_layer = ".".join(name_parts[1:])

                for module_name, file_type in module_within_layer_to_file_type.items():
                    if name_within_layer.startswith(f"{module_name}."):
                        weight_map[weight_name] = str(
                            RELATIVE_SUBBLOCKS_DIR / f"block_{layer_index}_{file_type}.safetensors"
                        )
                        found_match = True

        if not found_match:
            unmapped_weight_names.append(weight_name)

    if len(unmapped_weight_names) > 0:
        raise ValueError(
            f"Unmapped weight names: {unmapped_weight_names}\n"
            f"Add them to the `non_layer_module_to_file_type` or "
            f"`module_within_layer_to_file_type` dictionaries."
        )

    return weight_map


def save_model_config(model_config: PretrainedConfig, checkpoint_dir: Path | str) -> None:
    if hasattr(model_config, "block_configs"):
        model_config.block_configs = [
            dataclasses.asdict(conf) if dataclasses.is_dataclass(conf) else conf
            for conf in model_config.block_configs
        ]
    model_config.save_pretrained(checkpoint_dir)
