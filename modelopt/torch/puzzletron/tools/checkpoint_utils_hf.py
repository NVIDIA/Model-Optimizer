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

"""Provides utilities for loading and saving PyTorch model checkpoints in the Hugging Face format,
particularly for DeciLM models.
"""

import concurrent.futures
import fcntl
import os
import shutil
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

import torch
from safetensors.torch import save_file as safe_save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from modelopt.torch.puzzletron.decilm import deci_lm_hf_code
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.configuration_decilm import DeciLMConfig
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.modeling_decilm import DeciLMForCausalLM
from modelopt.torch.puzzletron.tools.common import infer_weights_dtype
from modelopt.torch.puzzletron.tools.logger import mprint
from modelopt.torch.puzzletron.tools.post_init_sparse import SparsityMethod
from modelopt.torch.puzzletron.tools.robust_json import json_dumps

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

warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")


def load_checkpoint(
    checkpoint_dir: Path | str,
    model_config_overrides: dict | None = None,
    ignore_unexpected_config_keys: bool = False,
) -> DeciLMForCausalLM:
    """Unlike AutoModelForCausalLM.from_pretrained, the models loaded by this function use your
    local repo code, not the code inside the checkpoint.
    """
    from modelopt.torch.puzzletron.tools.checkpoint_utils import (
        load_state_dict,  # prevent circular import
    )

    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    model_config = load_model_config(
        checkpoint_dir, model_config_overrides, ignore_unexpected_config_keys
    )

    # Without sparsity we could have done:
    # model = DeciLMForCausalLM.from_pretrained(pretrained_model_name_or_path=checkpoint_dir, config=model_config)
    state_dict = load_state_dict(checkpoint_dir)
    state_dict, sparsity_masks = SparsityMethod.fix_state_dict_inplace(state_dict, verbose=True)
    dtype = infer_weights_dtype(state_dict)
    model = DeciLMForCausalLM.from_pretrained(
        pretrained_model_name_or_path=None,
        config=model_config,
        state_dict=state_dict,
        torch_dtype=dtype,
    )
    SparsityMethod().apply_masks(model, sparsity_masks)

    return model


def load_model_config(
    checkpoint_dir: Path | str,
    model_config_overrides: Mapping | None = None,
    ignore_unexpected_config_keys: bool = False,
) -> DeciLMConfig:
    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    if model_config_overrides is None:
        model_config_overrides = {}

    config, unused_kwargs = DeciLMConfig.from_pretrained(
        checkpoint_dir, return_unused_kwargs=True, **model_config_overrides
    )

    if not ignore_unexpected_config_keys:
        if unused_kwargs:
            raise ValueError(f"Unexpected config keys: {unused_kwargs.keys()}")

    return config


def save_checkpoint(model: DeciLMForCausalLM, checkpoint_dir: Path | str) -> None:
    _save_checkpoint(model.config, model.state_dict(), checkpoint_dir)


def _save_checkpoint(
    model_config: DeciLMConfig,
    state_dict: dict[str, torch.Tensor],
    checkpoint_dir: Path | str,
    max_workers: int | None = None,  # Now optional - will auto-calculate if None
) -> None:
    mprint("=== Starting _save_checkpoint detailed profiling ===")
    total_start_time = time.time()

    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    # Phase 1: Create directory and save config
    phase1_start_time = time.time()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_config.save_pretrained(checkpoint_dir)
    phase1_time = time.time() - phase1_start_time
    mprint(f"Phase 1 - Directory creation and config save: {phase1_time:.2f}s")

    # Phase 2: Save subblocks (main model weights) with auto-calculated worker count
    phase2_start_time = time.time()
    save_subblocks(
        state_dict,
        checkpoint_dir,
        multi_threaded=True,
        max_workers=max_workers,  # Will auto-calculate if None
    )
    phase2_time = time.time() - phase2_start_time
    mprint(f"Phase 2 - Save subblocks (model weights): {phase2_time:.2f}s")

    # Phase 3: Save safetensors index
    phase3_start_time = time.time()
    save_safetensors_index(model_config, checkpoint_dir)
    phase3_time = time.time() - phase3_start_time
    mprint(f"Phase 3 - Save safetensors index: {phase3_time:.2f}s")

    # Phase 4: Copy HF code
    phase4_start_time = time.time()
    copy_deci_lm_hf_code(checkpoint_dir)
    phase4_time = time.time() - phase4_start_time
    mprint(f"Phase 4 - Copy HF code: {phase4_time:.2f}s")

    total_time = time.time() - total_start_time
    mprint(f"=== _save_checkpoint completed in {total_time:.2f}s ===")
    mprint(
        f"Breakdown: Config {phase1_time:.1f}s + Subblocks {phase2_time:.1f}s + "
        f"Index {phase3_time:.1f}s + HF code {phase4_time:.1f}s"
    )
    mprint(
        f"Save percentage breakdown: Config {phase1_time / total_time * 100:.1f}% + "
        f"Subblocks {phase2_time / total_time * 100:.1f}% + "
        f"Index {phase3_time / total_time * 100:.1f}% + "
        f"HF code {phase4_time / total_time * 100:.1f}%"
    )

    # Performance metrics
    if phase2_time > 0:
        subblocks_percentage = phase2_time / total_time * 100
        actual_workers = max_workers if max_workers else "auto"
        mprint(
            f"I/O optimization: Subblocks were {subblocks_percentage:.1f}% of total save time "
            f"(max_workers={actual_workers})"
        )


def split_checkpoint_to_subblocks(checkpoint_dir: Path | str) -> None:
    from modelopt.torch.puzzletron.tools.checkpoint_utils import (
        load_state_dict,  # prevent circular import
    )

    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    model_config = load_model_config(checkpoint_dir)
    state_dict = load_state_dict(checkpoint_dir)
    save_subblocks(state_dict, checkpoint_dir)

    if (index_path := checkpoint_dir / SAFE_WEIGHTS_INDEX_NAME).exists():
        index_path.rename(checkpoint_dir / f"before_splitting.{SAFE_WEIGHTS_INDEX_NAME}")
    save_safetensors_index(model_config, checkpoint_dir)


def save_subblocks(
    state_dict: dict[str, torch.Tensor],
    checkpoint_dir: Path | str,
    multi_threaded: bool = True,
    max_workers: int | None = None,  # Now optional - will auto-calculate if None
) -> None:
    mprint("=== Starting save_subblocks detailed profiling ===")
    subblocks_start_time = time.time()

    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    # Step 1: Build weight map
    weight_map_start_time = time.time()
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
            mprint(f"  Warning: {failed_saves} files failed to save")
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


def save_safetensors_index(
    model_config: DeciLMConfig,
    checkpoint_dir: Path | str,
) -> None:
    mprint("=== Starting save_safetensors_index profiling ===")
    index_start_time = time.time()

    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    # Step 1: Create fake model on meta device
    fake_model_start_time = time.time()
    with torch.device("meta"):
        fake_model = DeciLMForCausalLM(model_config)
    fake_model_time = time.time() - fake_model_start_time
    mprint(f"  Step 1 - Create fake model: {fake_model_time:.2f}s")

    # Step 2: Build weight map
    weight_map_start_time = time.time()
    weight_map = _build_safetensors_weight_map(
        state_dict=fake_model.state_dict(),
        non_layer_module_to_file_type=NON_LAYER_MODULE_TO_FILE_TYPE,
        module_within_layer_to_file_type=MODULE_WITHIN_LAYER_TO_FILE_TYPE,
        layers_module_name=LAYERS_MODULE_NAME,
    )
    weight_map_time = time.time() - weight_map_start_time
    mprint(f"  Step 2 - Build weight map: {weight_map_time:.2f}s ({len(weight_map)} mappings)")

    # Step 3: Create and write index
    write_start_time = time.time()
    index = {"metadata": {"format": "pt"}, "weight_map": weight_map}
    index_path = checkpoint_dir / SAFE_WEIGHTS_INDEX_NAME
    index_json = json_dumps(index)
    _write_file_process_safe(index_json, index_path)
    write_time = time.time() - write_start_time
    mprint(f"  Step 3 - Write index file: {write_time:.2f}s ({len(index_json)} chars)")

    index_total_time = time.time() - index_start_time
    mprint(f"=== save_safetensors_index completed in {index_total_time:.2f}s ===")
    mprint(
        f"  Breakdown: FakeModel {fake_model_time:.1f}s + WeightMap {weight_map_time:.1f}s + Write {write_time:.1f}s"
    )


def _write_text(content: str, f: BinaryIO) -> None:
    f.write(content.encode("utf-8"))


def _write_file_process_safe(
    content: Any,
    path: Path | str,
    write_fn: Callable[[Any, BinaryIO], None] = _write_text,
) -> None:
    """Write a file in a multi-process safe way.
    If another process tries to write the same file using this method, the current process
    "gives up" and assumes that the matter is being taken care of by another process.

    write_fn is a function that receives file contents and a binary file object,
    and writes the content to the file. It can be _write_text (defined above), or torch.save,
    or a similar function (not safetensors.torch.save_file since it expects a path).
    """
    with open(path, "wb") as f:
        # Try to acquire an exclusive, non-blocking lock
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return  # Exit immediately if the lock is not acquired

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


# Not really needed
def save_model_config(model_config: DeciLMConfig, checkpoint_dir: Path | str) -> None:
    model_config.save_pretrained(checkpoint_dir)


def copy_deci_lm_hf_code(output_dir: Path | str) -> None:
    """Copy the deci_lm_hf_code directory to the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_dir = Path(deci_lm_hf_code.__file__).parent
    for path in code_dir.glob("*.py"):
        shutil.copy(path, output_dir / path.name)
