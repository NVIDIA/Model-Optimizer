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

"""Provides a function to register activation hooks for a model.
Activation hooks are used to compute activation scores for pruning."""

from typing import Type

import torch

from modelopt.torch.prune.importance_hooks.base_hooks import ForwardHook as ActivationsHook

from ...tools.logger import aprint
from ...utils.dummy_modules import DummyBlock, DummyModule

__all__ = ["register_activation_hooks"]


def register_activation_hooks(
    model,
    activation_hooks_kwargs: dict,
    pruning_mixin,
    hook_class: Type[ActivationsHook],
) -> dict[str, ActivationsHook]:
    """Register activation hooks using the pruning mixin approach.

    Args:
        model: The model to register hooks on.
        activation_hooks_kwargs: Keyword arguments passed to hook constructors.
        pruning_mixin: The pruning mixin that defines which modules to hook.
        hook_class: The hook class to instantiate for each module.

    Returns:
        Dictionary mapping module names to hook instances.
    """
    activation_hooks_kwargs["model"] = model

    if hook_class not in pruning_mixin.supported_hooks():
        raise ValueError(
            f"Hook class not supported for {pruning_mixin.__class__.__name__}, "
            f"must be in {pruning_mixin.supported_hooks()}"
        )

    module_names_to_hook = pruning_mixin.get_module_names_to_hook(model)
    activation_hooks = dict()
    for block_idx, module_name in module_names_to_hook:
        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            # Module doesn't exist on this rank's shard (e.g., in distributed setup)
            continue

        # Skip dummy modules - they don't have real activations to hook
        if isinstance(module, (DummyModule, DummyBlock)):
            continue

        block_config = None
        if block_idx is not None:
            block_config = model.config.block_configs[block_idx]
        curr_activation_hooks_kwargs = {
            **activation_hooks_kwargs,
            "block_config": block_config,
        }

        hook = hook_class(module, curr_activation_hooks_kwargs)
        module.register_forward_hook(hook)
        activation_hooks[module_name] = hook

    if len(activation_hooks) == 0:
        # In distributed mode, it's okay for a rank to have 0 hooks if it doesn't own
        # the target modules (e.g., with hybrid patterns like "*-" where different
        # ranks own different layer types). However, we still want to catch real bugs
        # where no hooks are found at all.
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if is_distributed:
            aprint(
                "No hooks registered on this rank. This is expected if this rank "
                "doesn't own any layers matching the hook pattern (e.g., in hybrid "
                "patterns with distributed model sharding)."
            )
        else:
            raise ValueError("couldn't find any hooks")

    if len(activation_hooks) > 0:
        aprint(f"Found the following hooks: {activation_hooks.keys()}")
    return activation_hooks
