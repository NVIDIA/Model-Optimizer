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

"""Utilities for optimization."""

import types
from collections.abc import Generator
from contextlib import contextmanager

import torch.nn as nn
from torch.distributed._composable_state import _get_module_state
from torch.distributed.fsdp import FSDPModule

from modelopt.torch.utils import unwrap_model

from .dynamic import DynamicModule, DynamicSpace
from .hparam import Hparam

__all__ = [
    "get_hparam",
    "is_configurable",
    "is_dynamic",
    "named_dynamic_modules",
    "named_hparams",
    "search_space_size",
]


class _DynamicSpaceUnwrapped(DynamicSpace):
    """A wrapper for the DynamicSpace class to handle unwrapping of model wrappers like DDP.

    This is useful to ensure that configurations are valid for both vanilla models and wrapped
    models, see :meth:`unwrap_models<modelopt.torch.utils.network.unwrap_model>` for supported wrappers.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__(unwrap_model(model))


def is_configurable(model: nn.Module) -> bool:
    """Check if the model is configurable."""
    return _DynamicSpaceUnwrapped(model).is_configurable()


def is_dynamic(model: nn.Module) -> bool:
    """Check if the model is dynamic."""
    return _DynamicSpaceUnwrapped(model).is_dynamic()


def named_hparams(
    model: nn.Module, configurable: bool | None = None, unique: bool | None = None
) -> Generator[tuple[str, Hparam], None, None]:
    """Recursively yield the name and instance of *all* hparams."""
    yield from _DynamicSpaceUnwrapped(model).named_hparams(configurable, unique)


def named_dynamic_modules(model: nn.Module) -> Generator[tuple[str, DynamicModule], None, None]:
    """Recursively yield the name and instance of *all* dynamic modules."""
    yield from _DynamicSpaceUnwrapped(model).named_dynamic_modules()


def get_hparam(model: nn.Module, name: str) -> Hparam:
    """Get the hparam with the given name."""
    return _DynamicSpaceUnwrapped(model).get_hparam(name)


def search_space_size(model: nn.Module) -> int:
    """Return the size of the search space."""
    return _DynamicSpaceUnwrapped(model).size()


@contextmanager
def forward_with_reshard(model: nn.Module):
    """Context manager to keep the mesh info of FSDPModule during forward pass.

    FSDP2 models don't automatically reshard FSDPParam in the root module after forward passes with
    auto_reshard. This prevents accessing original DTensor parameters via .parameters() after any forward
    pass. Since calibration requires forward passes, initializing optimizers after calibration
    would use unsharded parameters (not original). Without manual resharding, there's no clean way
    to access original DTensor parameters. We have two options, 1) initialize optimizers before
    calibration, but this is hard to control when integrating to third-party frameworks.
    2) use this context manager to reshard FSDPParam in the root module after forward
    passes.
    """

    def _lazy_init_retain_mesh_info(self):
        if self._fsdp_param_group and not hasattr(self, "_post_forward_mesh_info_before"):
            self._post_forward_mesh_info_before = self._fsdp_param_group.post_forward_mesh_info
        self._lazy_init_original()
        if self._fsdp_param_group and not hasattr(self, "_post_forward_mesh_info_after"):
            self._post_forward_mesh_info_after = self._fsdp_param_group.post_forward_mesh_info
            self._fsdp_param_group.post_forward_mesh_info = self._post_forward_mesh_info_before

    fsdp_states = []

    for module in model.modules():
        if isinstance(module, FSDPModule):
            fsdp_state = _get_module_state(module)
            if not hasattr(fsdp_state, "_lazy_init_original"):
                fsdp_state._lazy_init_original = fsdp_state._lazy_init
                fsdp_state._lazy_init = types.MethodType(_lazy_init_retain_mesh_info, fsdp_state)

                fsdp_states.append(fsdp_state)
    yield
    for fsdp_state in fsdp_states:
        if fsdp_state._fsdp_param_group and hasattr(fsdp_state, "_post_forward_mesh_info_after"):
            fsdp_state._fsdp_param_group.post_forward_mesh_info = (
                fsdp_state._post_forward_mesh_info_after
            )
            delattr(fsdp_state, "_post_forward_mesh_info_before")
            delattr(fsdp_state, "_post_forward_mesh_info_after")
        fsdp_state._lazy_init = fsdp_state._lazy_init_original
        delattr(fsdp_state, "_lazy_init_original")
