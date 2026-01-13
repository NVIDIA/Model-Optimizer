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

"""Provides utilities for distributed loading, saving, and manipulation of
large language model checkpoints across multiple GPUs/processes.
"""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files
from typing_extensions import override

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.configuration_decilm import DeciLMConfig
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.modeling_decilm import (
    DeciLMDecoderLayer,
    DeciLMForCausalLM,
    rope_type_to_class,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils import load_model_config, load_state_dict
from modelopt.torch.puzzletron.tools.logger import mprint
from modelopt.torch.puzzletron.utils.utils import EmptyInitOnDevice


class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_load_state_dict_post_hook(self.load_state_dict_post_hook)

    @staticmethod
    def load_state_dict_post_hook(
        module: torch.nn.Module, incompatible_keys: torch.nn.modules.module._IncompatibleKeys
    ) -> None:
        incompatible_keys.missing_keys.clear()
        incompatible_keys.unexpected_keys.clear()


class DummyBlock(DummyModule):
    def __init__(self, config: DeciLMConfig, block_index: int):
        super().__init__()
        self.config = config
        self.block_index = block_index

    @override
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor | tuple[torch.Tensor, None]:
        if self.config.block_return_only_hidden_states:
            return x
        else:
            return x, None


class DummyWTE(DummyModule):
    def __init__(self, config: DeciLMConfig, dtype: torch.dtype | None = None):
        super().__init__()
        self.n_embd = config.get_hidden_size()
        self.dtype = dtype

    @override
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape  # noqa: N806
        result = torch.ones((B, T, self.n_embd), dtype=self.dtype, device=input_ids.device)
        return result


class DummyLMHead(DummyModule):
    def __init__(self, config: DeciLMConfig):
        super().__init__()
        self.vocab_size = config.vocab_size

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # noqa: N806
        result = torch.ones((B, T, self.vocab_size), dtype=x.dtype, device=x.device)
        return result


def create_local_shard_(model: DeciLMForCausalLM, owned_block_indexes: set[int]):
    all_block_indexes = set(range(len(model.model.layers)))
    has_first_block = 0 in owned_block_indexes
    has_last_block = max(all_block_indexes) in owned_block_indexes

    unowned_block_indexes = all_block_indexes - owned_block_indexes
    for block_index in unowned_block_indexes:
        model.model.layers[block_index] = cast(
            "DeciLMDecoderLayer", DummyBlock(model.config, block_index)
        )

    if not has_first_block:
        model.set_input_embeddings(DummyWTE(model.config))

    if not has_last_block:
        model.model.set_final_layer_norm(nn.Identity())
        if not (model.config.tie_word_embeddings and has_first_block):
            model.set_output_embeddings(DummyLMHead(model.config))

    return model


def create_dummy_model(
    model_config: DeciLMConfig,
    dtype: torch.dtype,
) -> DeciLMForCausalLM:
    with torch.device("meta"):
        model = DeciLMForCausalLM(model_config)

    rope_cls = rope_type_to_class[model_config.position_embedding_type]
    model.model.rotary_emb = rope_cls(config=model.config)

    model.model.set_input_embeddings(DummyWTE(model.config, dtype))
    model.model.set_final_layer_norm(nn.Identity())
    model.set_output_embeddings(DummyLMHead(model.config))

    for block_index in range(model_config.get_num_hidden_layers()):
        model.model.layers[block_index] = DummyBlock(model.config, block_index)

    return model


def load_and_shard_model(
    checkpoint_path: str | Path,
    owned_block_indexes: set[int] | Literal["auto"] = "auto",
    model_config: DeciLMConfig | None = None,
    model_config_overrides: Mapping | None = None,
    model_dtype: torch.dtype = torch.bfloat16,
) -> DeciLMForCausalLM:
    checkpoint_path = Path(checkpoint_path)
    with torch.device(dist.local_rank()):
        if model_config is None:
            model_config = load_model_config(
                checkpoint_path, model_config_overrides, ignore_unexpected_config_keys=True
            )

        if owned_block_indexes == "auto":
            owned_block_indexes = set(
                np.array_split(np.arange(model_config.get_num_hidden_layers()), dist.size())[
                    dist.rank()
                ]
            )

        mprint("Initializing model shards")
        model_shard = create_sharded_model(
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
                device=torch.device(dist.local_rank()),
            )

            new_names = set(shard_state_dict.keys())
            mprint(f"{new_names=}")
            model_shard.load_state_dict(shard_state_dict, assign=True)

            del shard_state_dict

            if model_config.tie_word_embeddings and (0 in owned_block_indexes):
                # re-tie the weights in case the connection was severed
                model_shard.tie_weights()
        else:
            mprint("Loading state_dict in main process")
            state_dict = load_state_dict(checkpoint_path) if dist.is_master() else None

            mprint("Distributing model to shards")
            load_state_dict_to_shards(model_shard=model_shard, loaded_state_dict=state_dict)
            del state_dict

        model_shard.type(model_dtype)

    params_on_meta_device = [
        param_name
        for param_name, param in model_shard.named_parameters()
        if param.device == torch.device("meta")
    ]
    assert len(params_on_meta_device) == 0, (
        f"[global_rank={dist.rank()}]  Couldn't load params {params_on_meta_device}"
    )

    return model_shard


def create_sharded_model(
    model_config: DeciLMConfig,
    owned_block_indexes: set[int],
    device: str | torch.device | None = "meta",
    dtype: torch.dtype | None = torch.float32,
):
    if isinstance(device, str):
        device = torch.device(device)

    dist.barrier()

    with EmptyInitOnDevice(device="meta", dtype=dtype):
        model = DeciLMForCausalLM(model_config)
        create_local_shard_(model=model, owned_block_indexes=owned_block_indexes)

    if device != torch.device("meta"):
        local_shard_state_dict = {
            k: torch.empty_like(v, device=device) for k, v in model.state_dict().items()
        }

        model.load_state_dict(local_shard_state_dict, assign=True)

    return model


def load_state_dict_to_shards(
    model_shard: torch.nn.Module, loaded_state_dict: dict | None = None
) -> None:
    from modelopt.torch.puzzletron.sewing_kit.utils import (
        distributed_isend_obj,
        distributed_recv_obj,
    )

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
    """out_path is usually output_checkpoint_path / "model.safetensors" """
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
    """keys_to_load: entire state_dict if None, else partial state_dict containing only these keys"""
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
