# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Support quantization and save/resore for Megatron."""

import contextlib
import pickle  # nosec
import types
from typing import Any

import megatron.core.transformer.mlp as megatron_mlp
import regex as re
import torch
from megatron.core.parallel_state import get_data_parallel_group

from ..dynamic import DynamicModule


def ensure_metadata_has_dp_cp_group(metadata):
    """Ensure `metadata` is a dict containing `dp_cp_group` entry.

    This function is adapted from megatron-lm's megatron.core.transformer.utils to avoid
    dependency on megatron-lm's specific version.

    Note:
        This is a temporary method and will be removed once this function is merged to
        megatron.core.transformer.utils in the main branch of megatron-lm.
    """
    # Create a copy to avoid modifying the original metadata dict
    # This prevents ProcessGroup from leaking into state dict
    if metadata is None:
        new_metadata = {}
    else:
        new_metadata = dict(metadata)
    if "dp_cp_group" not in new_metadata:
        try:
            new_metadata["dp_cp_group"] = get_data_parallel_group(with_context_parallel=True)
        except (AssertionError, RuntimeError):
            # Fallback if context parallel is not initialized
            new_metadata["dp_cp_group"] = get_data_parallel_group()
    return new_metadata


def _modelopt_get_extra_state(self):
    """Populating the extra_state when state_dict() is called.

    This is a top-level get_extra_state function which will first call
    subclass's get_extra_state (if any) then go through all registered
    get_extra_state callbacks

    If there is no extra_state, None is returned. Otherwise, the dictionary
    is serialized (via pickle) into a byte tensor following
    TransformerEngine's approach. In this case, the extra_state,
    """
    try:
        extra_state = super().get_extra_state()  # type: ignore[misc]
    except RuntimeError:
        # If the base class didn't implement get_extra_state(), then
        # exception with RuntimeError
        extra_state = None

    # TransformerEngine's extra_state may be torch.Tensor
    extra_state = {"extra_state": extra_state} if extra_state is not None else {}

    for callback in self.modelopt_get_extra_state_callbacks:
        extra_state.update(callback(self))

    # If no extra_state involved, return None instead of an empty dict.
    if len(extra_state) == 0:
        return None

    # Serialize state into byte tensor
    torch.cuda.synchronize()
    state_serialized = bytearray(pickle.dumps(extra_state))  # nosec
    state_serialized = torch.frombuffer(state_serialized, dtype=torch.uint8)

    return state_serialized


def _modelopt_set_extra_state(self, state: Any):
    """Restore quantizer_state when load_state_dict() is called.

    This is a top-level set_extra_state function which will first call
    subclass's set_extra_state (if any) then go through all registered
    set_extra_state callbacks
    """
    if state is None:
        return

    if isinstance(state, torch.Tensor):
        # Default format: byte tensor with pickled data
        #
        # TODO: possible deserialization improvement
        # https://github.com/NVIDIA/TensorRT-LLM/commits/main/tensorrt_llm/serialization.py
        extra_state = pickle.loads(state.detach().cpu().numpy().tobytes())  # nosec
    else:
        raise RuntimeError("Unsupported extra_state format.")

    # If the base class didn't implement get_extra_state(), then exception with RuntimeError
    with contextlib.suppress(RuntimeError):
        # TransformerEngine's extra_state is stored separately.
        super().set_extra_state(extra_state.get("extra_state", extra_state))  # type: ignore[misc]

    for callback in self.modelopt_set_extra_state_callbacks:
        callback(self, extra_state)


def register_modelopt_extra_state_callbacks(
    module: torch.nn.Module,
    get_extra_state_func,
    set_extra_state_func,
):
    """Register modelopt extra_state callbacks.

    Since we use module's extra_state to store per-module state of each mode, we need
    a registry to register all extra_state callbacks to apply since a module can be
    converted multiple times (e.g. sparsity + quantization).

    This function first check if the registry exists, then insert the callbacks to
    the registry.
    """
    if not hasattr(module, "modelopt_get_extra_state_callbacks"):
        module.modelopt_get_extra_state_callbacks = []
        module.get_extra_state = types.MethodType(_modelopt_get_extra_state, module)
    if not hasattr(module, "modelopt_set_extra_state_callbacks"):
        module.modelopt_set_extra_state_callbacks = []
        module.set_extra_state = types.MethodType(_modelopt_set_extra_state, module)
        module.allow_post_restore = True
    if get_extra_state_func not in module.modelopt_get_extra_state_callbacks:
        module.modelopt_get_extra_state_callbacks += [get_extra_state_func]
    if set_extra_state_func not in module.modelopt_set_extra_state_callbacks:
        module.modelopt_set_extra_state_callbacks += [set_extra_state_func]


class _MegatronMLP(DynamicModule):
    """Module to support special handling of `linear_fc1` in `sharded_state_dict()` of MCore `MLP`.

    See https://github.com/NVIDIA/Megatron-LM/blob/0657a52d5a5bfea3f74bcc19eb620211b71f8671/megatron/core/transformer/mlp.py#L151.
    """

    _modelopt_state_keys = []

    def _setup(self):
        pass

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Overriding the default to support scalar sharding.

        Note:
            singleton_local_shards needs to be added to the metadata as well as
            apply_swiglu_sharded_factory to handle the swiglu case.
        """
        if metadata is None:
            metadata = {}
        metadata["singleton_local_shards"] = True
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if not self.config.gated_linear_unit:
            return sharded_state_dict
        for k, v in sharded_state_dict.items():
            if "linear_fc1" in k and any(
                re.compile(pattern).match(k) for pattern in self._modelopt_state_keys
            ):
                sharded_state_dict[k] = megatron_mlp.apply_swiglu_sharded_factory(
                    v,
                    sharded_offsets,
                    metadata["singleton_local_shards"],
                )
        return sharded_state_dict
