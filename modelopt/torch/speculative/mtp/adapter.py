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
    """Identify and load native MTP; FakeBase supplies the frozen target components."""

    name: str

    @classmethod
    @abstractmethod
    def supports(cls, model_path: str | Path) -> bool:
        """Identify the architecture from checkpoint metadata without loading weights."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_mtp(
        cls,
        model_path: str | Path,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> nn.Module:
        """Load an MTPModule with native config, state layout and placement, without the backbone."""
        raise NotImplementedError

    @classmethod
    def create(
        cls,
        model_path: str | Path,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        config: MTPConfig,
    ) -> nn.Module:
        """Load the partial target and attach native MTP to shared EAGLE training.

        Override for checkpoint layouts or distributed loading that need a specialized
        FakeBase construction. Never load the target backbone for offline training.
        """
        # Transformers is optional; dispatch and checkpoint detection do not require it.
        from ..plugins.modeling_fakebase import FakeBaseModel
        from ..plugins.modeling_mtp import create_mtp_model

        base_model = FakeBaseModel.from_source(str(model_path)).to(device=device, dtype=dtype)
        mtp = cls.load_mtp(model_path, dtype=dtype, device=device)
        return create_mtp_model(mtp, base_model, config)


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
    if not _ADAPTERS:
        raise NativeMTPCheckpointError(
            "No native MTP adapters are registered. Import the adapter module before selecting one."
        )
    if adapter != "auto":
        selected = _ADAPTERS.get(adapter)
        if selected is None:
            raise NativeMTPCheckpointError(
                f"Unknown native MTP adapter {adapter!r}. Registered adapters: {sorted(_ADAPTERS)}"
            )
        if not selected.supports(model_path):
            raise NativeMTPCheckpointError(f"Adapter {adapter!r} does not support {model_path}")
        return selected
    matches = [candidate for candidate in _ADAPTERS.values() if candidate.supports(model_path)]
    if len(matches) != 1:
        raise NativeMTPCheckpointError(
            f"Expected one native MTP adapter for {model_path}; matched {[a.name for a in matches]}. "
            f"Registered adapters: {sorted(_ADAPTERS)}"
        )
    return matches[0]


def create_native_mtp_boost_model(
    model_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    **config_kwargs,
) -> nn.Module:
    """Load native MTP weights for offline training through a registered adapter.

    Additional keyword arguments are validated by :class:`MTPConfig`, which defines their defaults.
    """
    config = MTPConfig(**config_kwargs)
    selected = get_native_mtp_adapter(model_path, config.mtp_adapter)
    config.mtp_adapter = selected.name
    return selected.create(model_path, dtype=dtype, device=device, config=config)
