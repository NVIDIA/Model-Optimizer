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

"""Calc subblock stats to compute memory and runtime statistics for subblocks."""

import dataclasses
import json
import os
import warnings
from functools import partial
from itertools import product
from pathlib import Path
from typing import Iterable, Type, TypeVar

import pandas as pd
import torch
from immutabledict import immutabledict
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm
from transformers import PretrainedConfig

from modelopt.torch.nas.subblock_stats.calc_subblock_params_and_memory import (
    calc_subblock_active_params,
    calculate_non_block_memory,
    calculate_non_block_params,
    calculate_subblock_memory,
    calculate_subblock_params,
)
from modelopt.torch.utils import json_dump

from ..anymodel.model_descriptor import ModelDescriptor, ModelDescriptorFactory
from ..block_config import AttentionConfig, BlockConfig, FFNConfig, SubblockConfig
from ..replacement_library.replacement_utils import parse_layer_replacement
from ..tools.checkpoint_utils import load_model_config
from ..tools.logger import mprint
from ..utils.parsing import format_global_config

__all__ = [
    "calculate_subblock_stats",
    "launch_calc_subblock_stats",
]

# Type variable for dataclasses
T_DataClass = TypeVar("T_DataClass")

"""
Usage:
python -m modelopt.torch.puzzletron.subblock_stats.calc_subblock_stats PUZZLE_DIR [ --runtime_stats ]

--runtime_stats_enabled=False (the default) means that the code won't benchmark runtime,
  only memory stats will be calculated. If you want to benchmark runtime, run inside an trtllm docker.

"""


def calculate_subblock_stats(
    calc_subblock_stats_config: DictConfig,
    teacher_dir: Path,
    model_config: PretrainedConfig,
    descriptor: Type[ModelDescriptor],
    master_puzzle_dir: Path,
    subblock_configs: list[immutabledict[str, AttentionConfig | FFNConfig]],
    batch_size: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    prefill_queue_size: int,
    n_embd: int,
    n_head: int,
    vocab_size: int,
    runtime_stats_enabled: bool,
    use_cuda_graph: bool,
    weights_dtype: torch.dtype,
    activations_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    allocate_prefill_query: bool,
    moe_stats_file: str | Path | None = None,
) -> dict:
    if runtime_stats_enabled:
        from modelopt.torch.nas.subblock_stats.calc_runtime_stats import calc_runtime_for_subblocks

    gpu = None if not torch.cuda.is_available() else torch.cuda.get_device_name()
    subblock_stats = {
        "args": dict(
            gpu=gpu,
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            generation_seq_len=generation_seq_len,
            prefill_queue_size=prefill_queue_size,
            n_embd=n_embd,
            n_head=n_head,
            vocab_size=vocab_size,
            runtime_stats=runtime_stats_enabled,
            use_cuda_graph=use_cuda_graph,
            weights_dtype=str(weights_dtype),
            activations_dtype=str(activations_dtype),
            kv_cache_dtype=str(kv_cache_dtype),
        ),
        "non_block": dict(),
        "subblocks": list(),
    }
    # Compute runtime stats for unique subblocks only
    if runtime_stats_enabled:
        subblock_configs_nolayerindex = set(
            [subblock_config["subblock_config"] for subblock_config in subblock_configs]
        )

        # dict[SubblockConfig, float], float
        # TODO: Manage default values for calc_subblock_stats_config in one place, e.g. within a dataclass for hydra config.
        synth_dataset_num_requests = calc_subblock_stats_config.get("runtime_stats", {}).get(
            "synth_dataset_num_requests", 200
        )
        runtime_stats_config = calc_subblock_stats_config.get("runtime_stats", {})

        rt_prefill_seq_len = runtime_stats_config.get("prefill_seq_len", prefill_seq_len)
        rt_generation_seq_len = runtime_stats_config.get("generation_seq_len", generation_seq_len)

        runtime_by_subblock_dict, non_block_runtime_ms = calc_runtime_for_subblocks(
            subblock_configs_nolayerindex,
            runtime_stats_config,
            vocab_size,
            n_embd,
            n_head,
            master_puzzle_dir,
            teacher_dir,
            synth_dataset_num_requests,
            rt_prefill_seq_len,
            rt_generation_seq_len,
        )

    sorted_subblock_config = sorted(
        subblock_configs, key=lambda subblock_config: subblock_config["subblock_config"]
    )
    it = (
        tqdm(sorted_subblock_config, desc="Measuring subblock runtimes")
        if runtime_stats_enabled
        else sorted_subblock_config
    )
    for subblock_config_indexed in it:
        subblock_config = subblock_config_indexed["subblock_config"]
        parent_layer_indices = subblock_config_indexed["parent_layer_indices"]

        if runtime_stats_enabled:
            total_runtime_ms = runtime_by_subblock_dict[subblock_config]
            prefill_runtime_ms = None
            decode_runtime_ms = None
        else:
            total_runtime_ms, prefill_runtime_ms, decode_runtime_ms = None, None, None

        subblock_memory = calculate_subblock_memory(
            subblock_config,
            batch_size,
            prefill_seq_len,
            generation_seq_len,
            prefill_queue_size,
            n_embd,
            n_head,
            weights_dtype,
            kv_cache_dtype,
            allocate_prefill_query,
            model_config=model_config,
            descriptor=descriptor,
        )
        if not isinstance(subblock_memory, dict):
            subblock_memory = {"memory_mib": subblock_memory, "kv_cache_memory_mib": 0.0}

        subblock_params = calculate_subblock_params(model_config, subblock_config, descriptor)
        if moe_stats_file is not None:
            subblock_active_params = calc_subblock_active_params(
                subblock_config,
                descriptor,
                n_embd,
                moe_stats_file,
                batch_size,
                parent_layer_indices[0],
            )
        else:
            subblock_active_params = subblock_params
        subblock_stats["subblocks"].append(
            {
                "subblock_config": subblock_config,
                "subblock_config_class": type(subblock_config).__name__,
                "runtime_ms": total_runtime_ms,
                "prefill_runtime_ms": prefill_runtime_ms,
                "decode_runtime_ms": decode_runtime_ms,
                "num_params": subblock_params,
                "active_params": subblock_active_params,
                "parent_layer_index": parent_layer_indices[0],
                **subblock_memory,
            }
        )

    if not runtime_stats_enabled:
        non_block_runtime_ms = None
    non_block_memory = calculate_non_block_memory(n_embd, vocab_size, weights_dtype)
    non_block_params = calculate_non_block_params(n_embd, vocab_size)

    subblock_stats["non_block"] = {
        "runtime_ms": non_block_runtime_ms,
        "memory_mib": non_block_memory,
        "num_params": non_block_params,
    }
    return subblock_stats


def launch_calc_subblock_stats(cfg: DictConfig) -> None:
    """
    Launch the calc subblock stats function with Hydra configuration.
    """
    mprint(f"Calculating subblock stats for puzzle directory: {cfg.puzzle_dir}")
    mprint(f"Teacher directory: {cfg.teacher_dir}")
    mprint(
        f"Calc subblock stats config: {format_global_config(cfg.calc_subblock_stats, title='Calc subblock stats')}"
    )

    descriptor = ModelDescriptorFactory.get(cfg.descriptor)
    calculate_subblock_stats_for_puzzle_dir(
        cfg.calc_subblock_stats,
        master_puzzle_dir=cfg.puzzle_dir,
        teacher_dir=cfg.teacher_dir,
        descriptor=descriptor,
        model_hidden_sizes=cfg.calc_subblock_stats.get("model_hidden_sizes", OmegaConf.create([])),
        ffn_hidden_sizes=cfg.calc_subblock_stats.get("ffn_hidden_sizes", OmegaConf.create([])),
        batch_sizes=cfg.calc_subblock_stats.batch_sizes,
        prefill_seq_len=cfg.calc_subblock_stats.prefill_seq_len,
        generation_seq_len=cfg.calc_subblock_stats.generation_seq_len,
        num_active_tokens_override=cfg.calc_subblock_stats.get("num_active_tokens_override", None),
        prefill_queue_size=cfg.calc_subblock_stats.prefill_queue_size,
        allocate_prefill_query=cfg.calc_subblock_stats.get("allocate_prefill_query", False),
        runtime_stats_enabled=cfg.calc_subblock_stats.get("runtime_stats", {}).get(
            "enabled", False
        ),
        merge_with_existing_stats=cfg.calc_subblock_stats.merge_with_existing_stats,
        subblock_stats_filename=cfg.calc_subblock_stats.subblock_stats_filename,
        moe_stats_filename=cfg.calc_subblock_stats.moe_stats_filename,
    )


def calculate_subblock_stats_for_puzzle_dir(
    calc_subblock_stats_config: DictConfig,
    master_puzzle_dir: Path | str,
    teacher_dir: Path | str,
    descriptor: Type[ModelDescriptor],
    model_hidden_sizes: ListConfig,
    ffn_hidden_sizes: ListConfig,
    batch_sizes: Iterable[int] = (1, 8, 16, 32, 64, 128, 256),
    prefill_seq_len: int = 2048,
    generation_seq_len: int = 2048,
    num_active_tokens_override: int | None = None,
    prefill_queue_size: int = 0,  # it's an infery-llm thing
    allocate_prefill_query: bool = False,
    runtime_stats_enabled: bool = False,  # Compute runtime statistics.
    merge_with_existing_stats: bool = False,
    subblock_stats_filename: str = "subblock_stats.json",
    moe_stats_filename: str = "moe_stats.json",
) -> None:
    # ==== START === Setup for attach-helper ====
    # import sys
    # import os
    # sys.path.insert(0, os.environ["ATTACH_HELPER_INSTALLATION_PATH"])
    # from attach_helper import debugging_setup
    # debugging_setup()  # You can optionally pass a name to identify the job (e.g. `debugging_setup(name="my_script")`)
    # ==== END === Setup for attach-helper ====
    if isinstance(batch_sizes, str):
        batch_sizes = [
            int(batch_size) for batch_size in batch_sizes.strip("[]").replace(" ", "").split(",")
        ]

    master_puzzle_dir = Path(master_puzzle_dir)
    teacher_dir = (
        Path(teacher_dir) if teacher_dir is not None else master_puzzle_dir / "ckpts" / "teacher"
    )
    trust_remote_code = descriptor.requires_trust_remote_code()
    model_config = load_model_config(teacher_dir, trust_remote_code=trust_remote_code)
    # Get language model config for LM-specific attributes (VL models have nested config)
    lm_config = descriptor.get_language_model_config(model_config)
    subblock_configs = _load_subblock_configs(master_puzzle_dir, ffn_hidden_sizes)

    subblock_stats_file = master_puzzle_dir / subblock_stats_filename
    if subblock_stats_file.exists() and not merge_with_existing_stats:
        raise ValueError(
            f"Subblock stats file {subblock_stats_file} already exists and `merge_with_existing_stats` was set to False."
        )

    if subblock_stats_file.exists():
        with open(subblock_stats_file) as f:
            subblock_stats = json.load(f)
    else:
        subblock_stats = []

    moe_stats_file = master_puzzle_dir / moe_stats_filename
    if not moe_stats_file.exists():
        warnings.warn(
            f"MOE stats file {moe_stats_file} does not exist, can't calculate num active params"
        )
        moe_stats_file = None

    subblock_stats_args = {immutabledict(x["args"]) for x in subblock_stats}

    data_types = [
        ("nvfp4", "nvfp4", "nvfp4"),
        (torch.int8, torch.int8, torch.int8),
        (torch.int8, torch.int8, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
    ]

    model_hidden_sizes = model_hidden_sizes + [
        lm_config.hidden_size
    ]  # add a teacher model hidden size
    for batch_size, (
        weights_dtype,
        activations_dtype,
        kv_cache_dtype,
    ), model_hidden_size in product(batch_sizes, data_types, model_hidden_sizes):
        if num_active_tokens_override is not None:
            prefill_seq_len = generation_seq_len = int(num_active_tokens_override / batch_size / 2)

        curr_runtime_stats_enabled = (
            runtime_stats_enabled if weights_dtype == torch.bfloat16 else False
        )

        curr_subblock_stats = calculate_subblock_stats(
            calc_subblock_stats_config,
            teacher_dir=teacher_dir,
            model_config=model_config,
            descriptor=descriptor,
            master_puzzle_dir=master_puzzle_dir,
            subblock_configs=subblock_configs,
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            generation_seq_len=generation_seq_len,
            prefill_queue_size=prefill_queue_size,
            n_embd=model_hidden_size,
            n_head=lm_config.num_attention_heads,
            vocab_size=lm_config.vocab_size,
            runtime_stats_enabled=curr_runtime_stats_enabled,
            use_cuda_graph=True,
            weights_dtype=weights_dtype,
            activations_dtype=activations_dtype,
            kv_cache_dtype=kv_cache_dtype,
            allocate_prefill_query=allocate_prefill_query,
            moe_stats_file=moe_stats_file,
        )

        if immutabledict(curr_subblock_stats["args"]) in subblock_stats_args:
            raise ValueError(
                f"Failed merging subblock_stats. The following arguments already existed in the file: {curr_subblock_stats['args']}"
            )

        subblock_stats.append(curr_subblock_stats)

    json_dump(subblock_stats, subblock_stats_file)

    mprint(subblock_stats_file)


def _load_subblock_configs(
    master_puzzle_dir: Path, ffn_hidden_sizes: ListConfig
) -> list[SubblockConfig]:
    try:
        subblock_configs = _load_subblock_configs_from_replacement_library(master_puzzle_dir)
    except FileNotFoundError:
        subblock_configs = _load_subblock_configs_from_subblock_library(master_puzzle_dir)

    # Extend subblock stats calculation space with ffn_hidden_sizes defined in the calc_subblock_stats section of the model config yaml file.
    extra_ffn_subblock_configs = []
    for ffn_hidden_size in ffn_hidden_sizes:
        # Use FFNConfig defaults (hidden_act will use its default value)
        ffn_config = FFNConfig(intermediate_size=ffn_hidden_size)
        extra_ffn_subblock_configs.append(
            immutabledict({"subblock_config": ffn_config, "parent_layer_indices": tuple([-1])})
        )  # -1 to indicate that this sublock has no parent layer
    subblock_configs.extend(extra_ffn_subblock_configs)

    return subblock_configs


def _load_subblock_configs_from_subblock_library(master_puzzle_dir: Path) -> list[SubblockConfig]:
    subblocks_df = pd.read_json(master_puzzle_dir / "subblock_library.json")
    subblocks_df["attention_config"] = subblocks_df["attention_config"].apply(
        partial(_dataclass_from_dict, cls=AttentionConfig)
    )
    subblocks_df["ffn_config"] = subblocks_df["ffn_config"].apply(
        partial(_dataclass_from_dict, cls=FFNConfig)
    )
    attention_configs = subblocks_df["attention_config"].dropna().drop_duplicates().tolist()
    ffn_configs = subblocks_df["ffn_config"].dropna().drop_duplicates().tolist()
    # Wrap in the same dict format expected by calculate_subblock_stats() callers.
    # Use parent_layer_indices=(-1,) to indicate no specific parent layer.
    subblock_configs = [
        immutabledict({"subblock_config": cfg, "parent_layer_indices": (-1,)})
        for cfg in attention_configs + ffn_configs
    ]
    return subblock_configs


def _load_subblock_configs_from_replacement_library(
    master_puzzle_dir: Path,
) -> list[SubblockConfig]:
    """Load unique subblocks from replacement_library.json, e.g.,
    256 = 32*8 unique sublocks will be returned for a model with 32 layers and the search space of
    4 intermediate_size + teacher_intermediate_size + ffn_noop + att_op (teacher) + att_noop.

    Args:
        master_puzzle_dir (Path): Directory with "replacement_library.json" file

    Returns:
        list[SubblockConfig]:
    """
    replacement_library = json.loads((master_puzzle_dir / "replacement_library.json").read_text())
    subblock_configs = set()
    for layer_replacement in replacement_library:
        layer_replacement = parse_layer_replacement(layer_replacement)

        for block_config in layer_replacement["child_block_configs"]:
            block_config: BlockConfig
            attention_frozen_dict = immutabledict(
                {
                    "subblock_config": block_config.attention,
                    "parent_layer_indices": tuple(layer_replacement["parent_layer_indices"]),
                }
            )
            ffn_frozen_dict = immutabledict(
                {
                    "subblock_config": block_config.ffn,
                    "parent_layer_indices": tuple(layer_replacement["parent_layer_indices"]),
                }
            )
            subblock_configs.add(attention_frozen_dict)
            subblock_configs.add(ffn_frozen_dict)

            if block_config.parallel_blocks is not None:
                for block_idx, internal_block_config in enumerate(block_config.parallel_blocks):
                    attention_frozen_dict = immutabledict(
                        {
                            "subblock_config": internal_block_config.attention,
                            "parent_layer_indices": tuple(
                                layer_replacement["parent_layer_indices"]
                            ),
                            "inner_block_idx": block_idx,
                        }
                    )
                    ffn_frozen_dict = immutabledict(
                        {
                            "subblock_config": internal_block_config.ffn,
                            "parent_layer_indices": tuple(
                                layer_replacement["parent_layer_indices"]
                            ),
                            "inner_block_idx": block_idx,
                        }
                    )
                    subblock_configs.add(attention_frozen_dict)
                    subblock_configs.add(ffn_frozen_dict)

    subblock_configs = list(subblock_configs)
    return subblock_configs


T_DataClass: TypeVar = Type[dataclasses.dataclass]


def _dataclass_from_dict(
    d: dict | T_DataClass | None,
    cls: T_DataClass,
) -> T_DataClass | None:
    if isinstance(d, cls):
        return d
    if isinstance(d, dict):
        return cls(**d)
    if pd.isna(d):
        return None
    raise ValueError(f"_dataclass_from_dict: unrecognized {type(d)=} {d=}")
