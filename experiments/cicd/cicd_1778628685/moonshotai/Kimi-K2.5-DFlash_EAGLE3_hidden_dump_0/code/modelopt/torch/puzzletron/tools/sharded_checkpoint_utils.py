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
Provides utilities for distributed loading, saving, and manipulation of
large language model checkpoints across multiple GPUs/processes.

Uses native HuggingFace models with deci_x_patcher for heterogeneous layer configurations.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import transformers
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

import modelopt.torch.utils.distributed as dist

from ..utils.dummy_modules import DummyLMHead, DummyWTE
from ..utils.misc import EmptyInitOnDevice
from .checkpoint_utils import load_model_config, load_state_dict
from .checkpoint_utils_hf import _get_auto_class_for_trust_remote_code
from .logger import mprint

__all__ = [
    "set_submodule",
    "load_and_shard_model",
    "create_sharded_model",
    "load_sharded_state_dict",
    "is_in_safetensors_format",
]


def set_submodule(model: nn.Module, module_name: str, new_submodule: nn.Module) -> None:
    """Set a submodule on a model by dotted path."""
    parts = module_name.split(".")
    parent_path = ".".join(parts[:-1])
    attr = parts[-1]
    parent_module = model.get_submodule(parent_path) if parent_path else model
    setattr(parent_module, attr, new_submodule)


def create_local_shard_(model, owned_block_indexes: set[int], descriptor, runtime):
    # Get language model config (handles nested configs like Qwen3-VL's text_config)
    lm_config = descriptor.get_language_model_config(model.config)
    all_block_indexes = set(range(lm_config.num_hidden_layers))
    has_first_block = 0 in owned_block_indexes
    has_last_block = max(all_block_indexes) in owned_block_indexes

    unowned_block_indexes = all_block_indexes - owned_block_indexes
    for block_index in unowned_block_indexes:
        decoder_layer_name = descriptor.layer_block_name(block_index)
        decoder_layer = model.get_submodule(decoder_layer_name)
        set_submodule(
            model,
            decoder_layer_name,
            descriptor.create_dummy_block(decoder_layer, block_index=block_index),
        )

    # If we have the last block with tied embeddings, keep embed_tokens so lm_head works.
    # load_sharded_state_dict will load embed_tokens.weight from the first shard's checkpoint file,
    # and since they're tied, lm_head.weight gets populated too.
    if not has_first_block and not (has_last_block and model.config.tie_word_embeddings):
        set_submodule(
            model,
            descriptor.input_embedding_name(),
            DummyWTE(lm_config.hidden_size, dtype=runtime.dtype),
        )

    if not has_last_block:
        set_submodule(model, descriptor.final_norm_name(), nn.Identity())
        if not (model.config.tie_word_embeddings and has_first_block):
            set_submodule(model, descriptor.output_embedding_name(), DummyLMHead(lm_config))

    return model


def _get_model_class_from_config(config: PretrainedConfig):
    """
    Get the model class from config.architectures field.
    Works for any model registered in transformers (CausalLM, VL models, etc.).
    Falls back to AutoModelForCausalLM if architectures is not available.
    """
    if hasattr(config, "architectures") and config.architectures:
        model_class_name = config.architectures[0]
        if hasattr(transformers, model_class_name):
            return getattr(transformers, model_class_name)
        mprint(
            f"Warning: {model_class_name} not found in transformers, falling back to AutoModelForCausalLM"
        )
    return AutoModelForCausalLM


def load_and_shard_model(
    descriptor,
    checkpoint_path: str | Path,
    owned_block_indexes: set[int] | Literal["auto"] = "auto",
    model_config: PretrainedConfig | None = None,
):
    checkpoint_path = Path(checkpoint_path)
    runtime = SimpleNamespace(
        device=torch.device(dist.local_rank()),
        dtype=torch.bfloat16,
        global_rank=dist.rank(),
        world_size=dist.size(),
        is_main_process=dist.is_master(),
        is_last_process=dist.is_last_process(),
        use_autocast=True,  # Default: use autocast; descriptor can override
    )

    with runtime.device:
        if model_config is None:
            trust_remote_code = descriptor.requires_trust_remote_code()
            model_config = load_model_config(checkpoint_path, trust_remote_code=trust_remote_code)

        num_hidden_layers = descriptor.get_language_model_config(model_config).num_hidden_layers
        if owned_block_indexes == "auto":
            owned_block_indexes = set(
                np.array_split(np.arange(num_hidden_layers), runtime.world_size)[
                    runtime.global_rank
                ]
            )

        mprint("Initializing model shards")
        # Pass block_configs explicitly so patcher works for VL models where
        # decoder layers receive nested config (e.g., text_config) without block_configs
        from ..anymodel.puzzformer import deci_x_patcher

        with deci_x_patcher(
            model_descriptor=descriptor, block_configs=getattr(model_config, "block_configs", None)
        ):
            model_shard = create_sharded_model(
                runtime=runtime,
                descriptor=descriptor,
                model_config=model_config,
                owned_block_indexes=owned_block_indexes,
            )

        if (checkpoint_path / SAFE_WEIGHTS_NAME).exists() or (
            checkpoint_path / SAFE_WEIGHTS_INDEX_NAME
        ).exists():
            mprint("Loading shard state_dict from safetensors")
            shard_keys = [
                *[name for name, _ in model_shard.named_parameters()],
                *[name for name, _ in model_shard.named_buffers()],
            ]
            shard_state_dict = load_sharded_state_dict(
                model_name_or_path=str(checkpoint_path),
                keys_to_load=shard_keys,
                device=runtime.device,
            )

            # strict=False: allows missing lm_head.weight when tie_word_embeddings=True (e.g., Llama 3.2 3B)
            model_shard.load_state_dict(shard_state_dict, strict=False, assign=True)

            del shard_state_dict

            # Re-tie weights after load_state_dict with assign=True, which severs the tie.
            # Needed on first rank (owns embed_tokens) and last rank (owns lm_head).
            has_first_block = 0 in owned_block_indexes
            has_last_block = (num_hidden_layers - 1) in owned_block_indexes
            if model_config.tie_word_embeddings and (has_first_block or has_last_block):
                model_shard.tie_weights()

            # On the last rank with tied embeddings, we kept embed_tokens in create_local_shard_()
            # just to load the weight and tie it to lm_head. Now replace it with a dummy so it
            # doesn't interfere with the pipeline forward pass (only rank 0 should run embed_tokens).
            if model_config.tie_word_embeddings and has_last_block and not has_first_block:
                set_submodule(
                    model_shard,
                    descriptor.input_embedding_name(),
                    DummyWTE(model_config.hidden_size, dtype=runtime.dtype),
                )
        else:
            mprint("Loading state_dict in main process")
            state_dict = load_state_dict(checkpoint_path) if runtime.is_main_process else None

            mprint("Distributing model to shards")
            load_state_dict_to_shards(model_shard=model_shard, loaded_state_dict=state_dict)
            del state_dict

        descriptor.init_rotary_embedding(model_shard, runtime)

        model_shard.type(runtime.dtype)

        # Configure autocast based on model descriptor (some models like Qwen3-VL MoE
        # have dtype bugs under autocast)
        runtime.use_autocast = descriptor.uses_autocast()

    params_on_meta_device = [
        param_name
        for param_name, param in model_shard.named_parameters()
        if param.device == torch.device("meta")
    ]
    assert len(params_on_meta_device) == 0, (
        f"[global_rank={runtime.global_rank}]  Couldn't load params {params_on_meta_device}"
    )

    return model_shard


def create_sharded_model(
    runtime,
    descriptor,
    model_config: PretrainedConfig,
    owned_block_indexes: set[int],
    device: str | torch.device | None = "meta",
    dtype: torch.dtype | None = torch.float32,
):
    if isinstance(device, str):
        device = torch.device(device)

    dist.barrier()

    with EmptyInitOnDevice(device="meta", dtype=dtype):
        # Get model class from config.architectures (works for CausalLM, VL models, etc.)
        model_class = _get_model_class_from_config(model_config)
        trust_remote_code = descriptor.requires_trust_remote_code()
        if trust_remote_code:
            auto_cls = _get_auto_class_for_trust_remote_code(model_config)
            model = auto_cls.from_config(model_config, trust_remote_code=trust_remote_code)
        elif model_class is AutoModelForCausalLM:
            model = AutoModelForCausalLM.from_config(model_config)
        else:
            model = model_class._from_config(model_config)
        create_local_shard_(
            model=model,
            owned_block_indexes=owned_block_indexes,
            descriptor=descriptor,
            runtime=runtime,
        )

    if device != torch.device("meta"):
        local_shard_state_dict = {
            k: torch.empty_like(v, device=device) for k, v in model.state_dict().items()
        }
        model.load_state_dict(local_shard_state_dict, assign=True)

    return model


def load_state_dict_to_shards(
    model_shard: torch.nn.Module, loaded_state_dict: dict | None = None
) -> None:
    from ..sewing_kit.utils import distributed_isend_obj, distributed_recv_obj

    model_shard.to("meta")
    local_state_dict_keys = list(model_shard.state_dict().keys())

    if dist.is_master():
        gathered_state_dict_keys = [None] * dist.size()
        torch.distributed.gather_object(local_state_dict_keys, gathered_state_dict_keys)

        assert loaded_state_dict is not None
        loaded_state_dict = {k.replace("_orig_mod.", ""): v for k, v in loaded_state_dict.items()}

        works: list[torch.distributed.Work] = []
        for i, shard_keys in enumerate(gathered_state_dict_keys[1:]):
            process_id = i + 1
            shard_state_dict = {k: v for k, v in loaded_state_dict.items() if k in shard_keys}
            process_works = distributed_isend_obj(shard_state_dict, process_id)
            works.extend(process_works)

        for work in works:
            work.wait()

        shard_state_dict = {
            k: v for k, v in loaded_state_dict.items() if k in local_state_dict_keys
        }
    else:
        torch.distributed.gather_object(local_state_dict_keys)
        shard_state_dict = distributed_recv_obj()

    print(f"{dist.rank()} loaded state_dict shard")

    missing_keys, unexpected_keys = model_shard.load_state_dict(
        shard_state_dict, strict=False, assign=True
    )
    assert len(unexpected_keys) == 0
    assert all("dummy_param" in key for key in missing_keys)

    model_shard.cuda(dist.local_rank())

    dist.barrier()


def save_sharded_model(
    model_shard: torch.nn.Module | dict[str, torch.Tensor], out_path: str | Path
):
    """
    out_path is usually output_checkpoint_path / "model.safetensors"
    """
    dist.barrier()

    if isinstance(model_shard, torch.nn.Module):
        shard_state_dict = model_shard.state_dict()
    elif isinstance(model_shard, dict):
        shard_state_dict = model_shard
    else:
        raise ValueError(f"Unrecognized model shard type: {type(model_shard)}")

    shard_state_dict = {k: v.cpu() for k, v in shard_state_dict.items()}
    total_shard_size = sum(
        weight.numel() * weight.element_size() for weight in shard_state_dict.values()
    )

    num_shards = dist.size()
    idx = dist.rank()

    out_path = Path(out_path)
    shard_file = out_path.with_stem(f"{out_path.stem}-{idx + 1:05d}-of-{num_shards:05d}")

    shard_metadata = {
        "total_shard_size": total_shard_size,
        "shard_keys": list(shard_state_dict.keys()),
        "shard_file": str(shard_file),
    }

    if dist.is_master():
        shard_metadatas = [{} for _ in range(dist.size())]
        torch.distributed.gather_object(shard_metadata, shard_metadatas, dst=0)
        total_size = sum(x["total_shard_size"] for x in shard_metadatas)
        metadata = {"total_size": total_size}
        weight_map: dict[str, str] = {}
        for shard_metadata in shard_metadatas:
            weight_map.update(
                {k: Path(shard_metadata["shard_file"]).name for k in shard_metadata["shard_keys"]}
            )

        index = {"metadata": metadata, "weight_map": weight_map}
        index_path = Path(str(out_path) + ".index.json")
        index_path.write_text(json.dumps(index, indent=2))

    else:
        torch.distributed.gather_object(shard_metadata, dst=0)

    if out_path.suffix == ".safetensors":
        safe_save_file(shard_state_dict, shard_file, metadata={"format": "pt"})
    else:
        torch.save(shard_state_dict, shard_file)

    dist.barrier()


def load_sharded_state_dict(
    model_name_or_path: str | Path,
    keys_to_load: Iterable[str] | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    keys_to_load: entire state_dict if None, else partial state_dict containing only these keys
    """
    shard_paths = _resolve_shard_paths(model_name_or_path)
    # print(f"shard_paths: {shard_paths}")
    partial_state_dict = {}
    for safetensors_path in shard_paths:
        if keys_to_load is None:
            shard = safe_load_file(safetensors_path)
            partial_state_dict.update(shard)
        else:
            with safe_open(safetensors_path, framework="pt", device=str(device)) as f:
                for key in f.keys():  # noqa: SIM118 - safe_open objects require .keys(), not directly iterable
                    if key in keys_to_load:
                        partial_state_dict[key] = f.get_tensor(key)
    return partial_state_dict


def _resolve_shard_paths(model_name_or_path: str) -> list[str]:
    try:
        unsharded_path = cached_file(model_name_or_path, SAFE_WEIGHTS_NAME)
        return [unsharded_path]
    except OSError:
        index_path = cached_file(model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
        shard_paths, _ = get_checkpoint_shard_files(model_name_or_path, index_path)
        return shard_paths


def is_in_safetensors_format(checkpoint_dir: Path) -> bool:
    return len(list(checkpoint_dir.glob("*.safetensors"))) > 0
