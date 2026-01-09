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

"""Megatron core distributed checkpointing plugin for sharded ``modelopt_state``."""

# TODO: Add unit tests for this plugin
import copy
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from megatron.core import dist_checkpointing, mpu
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.core.dist_checkpointing.strategies.common import COMMON_STATE_FNAME
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.transformer.module import Float16Module

import modelopt
import modelopt.torch.opt as mto
import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils.network import SUPPORTED_WRAPPERS

SUPPORTED_WRAPPERS[Float16Module] = "module"

DROP_SUBSTRINGS = [
    "fp4",
    "fp8",
    "tp_",
    "parallel",
    "cuda_graph",
    "init_",
    "cpu",
    "recompute",
    "inference",
    "pipeline",
    "comm",
    "batch",
]


def remove_per_module_state(
    modelopt_state: dict[str, Any],
) -> None:
    """Remove metadata from the modelopt_state.

    The metadata of the modelopt_state contains keys which may change with different pipeline
    and expert parallelism. As a result, the metadata must be stored as several ShardedObject with
    global and local layer offset mapping.

    Args:
        modelopt_state: the state_dict that contains all algorithms that have been applied
            to the given model.
    """
    if "modelopt_state_dict" not in modelopt_state:
        return

    for mode, config in modelopt_state["modelopt_state_dict"]:
        metadata = config.get("metadata", None)
        if metadata is not None:
            _ = metadata.pop("quantizer_state", None)
            _ = metadata.pop("subnet_config", None)
            _ = metadata.pop("real_quantizer_state", None)
            _ = metadata.pop("q_tensor_state", None)
        else:
            config["metadata"] = {}


def save_modelopt_state(model: list[torch.nn.Module], state_dict: dict[str, Any]) -> None:
    """Save modelopt_state as a part of the per rank state_dict.

    NOTE: Only used for Megatron-LM.

    Args:
        model: the modelopt optimized model
        state_dict: the current modelopt optimized model state_dict to store
    """
    if not mto.ModeloptStateManager.is_converted(model[0]):
        return
    if len(model) == 1:
        state_dict["modelopt_state"] = mto.modelopt_state(model[0])
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            state_dict[f"modelopt_state_{i}"] = mto.modelopt_state(model[i])


def restore_modelopt_state(model: list[torch.nn.Module], state_dict: dict[str, Any]) -> None:
    """Restore modelopt_state from the per rank state_dict.

    NOTE: Only used for Megatron-LM.

    Args:
        model: the model to restore the modelopt optimization
        state_dict: the loaded state_dict to extract
    """
    if (
        len(model) == 1
        and "modelopt_state" in state_dict
        and not mto.ModeloptStateManager.is_converted(model[0])
    ):
        model[0] = mto.restore_from_modelopt_state(model[0], state_dict["modelopt_state"])
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            if f"modelopt_state_{i}" in state_dict and not mto.ModeloptStateManager.is_converted(
                model[i]
            ):
                model[i] = mto.restore_from_modelopt_state(
                    model[i], state_dict[f"modelopt_state_{i}"]
                )


def save_sharded_modelopt_state(
    model: list[torch.nn.Module],
    checkpoint_name: str | Path,
    sharded_strategy: tuple[str, int] | None = None,
    prefix: str = "",
) -> None:
    """Save modelopt_state in the sharded state_dict format.

    Args:
        model: the model to restore the modelopt optimization
        checkpoint_name: the checkpoint folder path
        sharded_strategy: configures sharded tensors saving behavior and backend
        prefix: the prefix to add to the modelopt_state keys ("model." for NeMo)
    """

    def _parse_transformer_config(transformer_config: dict) -> dict:
        config = {}

        for k, v in transformer_config.items():
            if any(substring in k for substring in DROP_SUBSTRINGS):
                continue
            if isinstance(v, (bool, int, str)):
                config[k] = v
            else:
                config[k] = str(v)

        return config

    # Save own version of run config, if not already saved by the framework.
    if dist.is_master() and not os.path.exists(f"{checkpoint_name}/run_config.yaml"):
        run_config_name = f"{checkpoint_name}/modelopt_run_config.yaml"
        # We avoid deepcopy since some attributes in Megatron-Bridge config cannot be deepcopied.
        config_dict = _parse_transformer_config(model[0].config.__dict__)
        config_dict["nvidia_modelopt_version"] = modelopt.__version__
        with open(run_config_name, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    if not mto.ModeloptStateManager.is_converted(model[0]):
        return
    if len(model) > 1:
        raise ValueError("sharded_modelopt_state does not support virtual pipeline parallel!")
    modelopt_checkpoint_name = f"{checkpoint_name}/modelopt_state"
    if dist.is_master():
        os.makedirs(modelopt_checkpoint_name, exist_ok=True)
    modelopt_state = copy.deepcopy(mto.modelopt_state(model[0]))
    remove_per_module_state(modelopt_state)
    dist_checkpointing.save(modelopt_state, modelopt_checkpoint_name, sharded_strategy)


def _load_extra_state_from_sharded_checkpoint(
    model: torch.nn.Module,
    checkpoint_name: str | Path,
    prefix: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Load extra state from sharded checkpoint.

    Note: since extra_state is a subset of full the sharded_state_dict, we use
        strict=StrictHandling.LOG_UNEXPECTED instead of LOG_ALL.

    Args:
        model: the model to load extra state into
        checkpoint_name: the checkpoint folder path
        prefix: the prefix to add to the modelopt_state keys
        metadata: the metadata for distributed checkpointing

    Note:
        The metadata includes several breaking changes. For example, `singleton_local_shards`
        is set to `True` (was not set before) in megatron-core-0.15.0. This flag affects the
        sharded state_dict format and must be consistent between saving and loading.
    """
    sharded_state_dict = model.sharded_state_dict(prefix=prefix)
    extra_sharded_state_dict = {k: v for k, v in sharded_state_dict.items() if "_extra_state" in k}
    extra_state_dict = dist_checkpointing.load(
        extra_sharded_state_dict,
        checkpoint_name,
        get_default_load_sharded_strategy(checkpoint_name),
        strict=StrictHandling.LOG_UNEXPECTED,
    )
    extra_state_dict_no_prefix = {}

    for k, v in extra_state_dict.items():
        if k.startswith(prefix):
            extra_state_dict_no_prefix[k[len(prefix) :]] = v
    model.load_state_dict(extra_state_dict_no_prefix, strict=False)


def restore_sharded_modelopt_state(
    model: list[torch.nn.Module],
    checkpoint_name: str | Path,
    prefix: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Restore modelopt_state from the sharded state_dict format.

    Args:
        model: the model to restore the modelopt optimization
        checkpoint_name: the checkpoint folder path
        prefix: the prefix to add to the modelopt_state keys ("model." for NeMo)
        metadata: the metadata for distributed checkpointing

    Note:
        The metadata includes several breaking changes. For example, `singleton_local_shards`
        is set to `True` (was not set before) in megatron-core-0.15.0. This flag affects the
        sharded state_dict format and must be consistent between saving and loading.
    """
    if len(model) > 1:
        raise ValueError("sharded_modelopt_state does not support virtual pipeline parallel!")

    modelopt_checkpoint_name = f"{checkpoint_name}/modelopt_state"

    # Early return if the model already has a modelopt_state or the checkpoint does not exist.
    if not os.path.exists(modelopt_checkpoint_name) or mto.ModeloptStateManager.is_converted(
        model[0]
    ):
        return

    # Loading the common modelopt_state (replicated on all ranks)
    common_modelopt_state = torch.load(
        modelopt_checkpoint_name + "/" + COMMON_STATE_FNAME, weights_only=False
    )

    modelopt_load_version = common_modelopt_state["modelopt_version"]

    print(f"nvidia-modelopt ckpt/inst version: {modelopt_load_version}/{modelopt.__version__}")

    # After 0.29, we no longer store (or shard) any quantizer_state in the modelopt_state.
    # quantizer_state (or other per-module state) is stored with the main distributed
    # checkpoint as extra_state at the QuantModule level.
    #
    # The process of resuming modelopt_state becomes 2-phase:
    # 1. Load the global modelopt_state and call mto.restore_from_modelopt_state.
    #    Modes are restored in order. Modes with per-module state stored as
    #    extra_state are partially restored (stop at DynamicModule replacement)
    #
    model[0] = mto.restore_from_modelopt_state(model[0], common_modelopt_state)

    _load_extra_state_from_sharded_checkpoint(model[0], checkpoint_name, prefix, metadata=metadata)
