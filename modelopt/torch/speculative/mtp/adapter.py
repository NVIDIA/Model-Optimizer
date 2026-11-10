# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Adapter dispatch for checkpoint-native MTP training."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn

from ..config import MTPConfig

__all__ = [
    "NativeMTPAdapter",
    "NativeMTPCheckpointError",
    "create_native_mtp_boost_model",
    "get_native_mtp_adapter",
    "register_native_mtp_adapter",
]


class NativeMTPCheckpointError(ValueError):
    """An unsupported or inconsistent native MTP checkpoint."""


class NativeMTPAdapter(ABC):
    """Identify a native checkpoint and construct its module and frozen endpoints."""

    name: str

    @classmethod
    @abstractmethod
    def supports(cls, model_path: str | Path) -> bool:
        """Identify the architecture from checkpoint metadata without loading weights."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create(
        cls,
        model_path: str | Path,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        config: MTPConfig,
    ) -> nn.Module:
        """Return ``create_mtp_model(native_module, embedding, lm_head, config)``.

        Adapters handle native state layouts and placement. They must not load the
        target backbone for offline training.
        """
        raise NotImplementedError


_ADAPTERS: dict[str, type[NativeMTPAdapter]] = {}


def register_native_mtp_adapter(adapter: type[NativeMTPAdapter]) -> type[NativeMTPAdapter]:
    """Register one named checkpoint implementation; duplicate names are rejected."""
    if not getattr(adapter, "name", None):
        raise TypeError("Native MTP adapters must define a nonempty name")
    if adapter.name in _ADAPTERS:
        raise ValueError(f"Native MTP adapter {adapter.name!r} is already registered")
    _ADAPTERS[adapter.name] = adapter
    return adapter


def get_native_mtp_adapter(model_path: str | Path, adapter: str = "auto") -> type[NativeMTPAdapter]:
    """Resolve exactly one matching adapter, or validate the explicitly selected one."""
    if adapter != "auto":
        selected = _ADAPTERS.get(adapter)
        if selected is None or not selected.supports(model_path):
            raise NativeMTPCheckpointError(f"Adapter {adapter!r} does not support {model_path}")
        return selected
    matches = [candidate for candidate in _ADAPTERS.values() if candidate.supports(model_path)]
    if len(matches) != 1:
        raise NativeMTPCheckpointError(
            f"Expected one native MTP adapter for {model_path}; matched {[a.name for a in matches]}"
        )
    return matches[0]


def create_native_mtp_boost_model(
    model_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    mtp_adapter: str = "auto",
    mtp_ttt_steps: int = 1,
    mtp_mix_hidden_states: bool = False,
    mtp_mix_hidden_states_mode: str = "uniform_layer_sample",
    mtp_loss_type: str = "lk_lambda",
    mtp_lk_eta: float = 3.0,
    mtp_loss_decay_factor: float = 1.0,
) -> nn.Module:
    """Load native MTP weights for offline training through a registered adapter."""
    config = MTPConfig(
        mtp_adapter=mtp_adapter,
        mtp_ttt_steps=mtp_ttt_steps,
        mtp_mix_hidden_states=mtp_mix_hidden_states,
        mtp_mix_hidden_states_mode=mtp_mix_hidden_states_mode,
        mtp_loss_type=mtp_loss_type,
        mtp_lk_eta=mtp_lk_eta,
        mtp_loss_decay_factor=mtp_loss_decay_factor,
    )
    selected = get_native_mtp_adapter(model_path, config.mtp_adapter)
    config.mtp_adapter = selected.name
    return selected.create(model_path, dtype=dtype, device=device, config=config)
