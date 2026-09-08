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

"""Registry and public entry points for checkpoint-native MTP training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn

__all__ = [
    "NativeMTPAdapter",
    "NativeMTPCheckpointError",
    "attach_native_mtp_online_target",
    "create_native_mtp_boost_model",
    "export_native_mtp_checkpoint",
    "get_native_mtp_adapter",
    "load_native_mtp_boost_checkpoint",
    "register_native_mtp_adapter",
]


class NativeMTPCheckpointError(ValueError):
    """Raised when a checkpoint is not a supported native-MTP checkpoint."""


class NativeMTPAdapter(ABC):
    """Model-specific loader and exporter for a checkpoint-native MTP."""

    name: str

    @classmethod
    @abstractmethod
    def supports(cls, model_path: str | Path) -> bool:
        """Return whether ``model_path`` is a checkpoint handled by this adapter."""

    @classmethod
    @abstractmethod
    def create(
        cls,
        model_path: str | Path,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        rollout_steps: int,
        hsm_mode: str,
    ) -> nn.Module:
        """Construct a trainable native-MTP model from ``model_path``."""

    @classmethod
    @abstractmethod
    def export(
        cls,
        model: nn.Module,
        base_checkpoint: str | Path,
        output_path: str | Path,
        *,
        state_dict: dict[str, torch.Tensor] | None,
    ) -> Path:
        """Write a deployment checkpoint containing ``model``'s native MTP."""

    @classmethod
    def attach_online_target(
        cls,
        model: nn.Module,
        model_path: str | Path,
        target_checkpoint: str | Path,
        *,
        device: torch.device | str,
        max_batch_size: int,
        max_seq_len: int,
    ) -> None:
        """Attach the model-specific frozen target used for online features."""
        raise NativeMTPCheckpointError(
            f"Native MTP adapter {cls.name!r} does not support online training"
        )


_ADAPTERS: dict[str, type[NativeMTPAdapter]] = {}


def register_native_mtp_adapter(adapter: type[NativeMTPAdapter]) -> type[NativeMTPAdapter]:
    """Register a checkpoint-native MTP adapter.

    Adapters are deliberately small and model-specific.  A checkpoint must be
    accepted by exactly one adapter; this avoids accidentally treating an
    arbitrary model's similarly-named tensors as a native MTP.
    """
    if not getattr(adapter, "name", None):
        raise TypeError("Native MTP adapters must define a non-empty name")
    if adapter.name in _ADAPTERS:
        raise ValueError(f"Native MTP adapter {adapter.name!r} is already registered")
    _ADAPTERS[adapter.name] = adapter
    return adapter


def get_native_mtp_adapter(model_path: str | Path, adapter: str = "auto") -> type[NativeMTPAdapter]:
    """Resolve the one adapter that supports ``model_path``.

    ``auto`` is intentionally conservative: it only succeeds when one
    registered adapter positively identifies the checkpoint.
    """
    # Importing the implementation here keeps codec-only imports free of the
    # optional Transformers dependency.
    from . import deepseek_v4  # noqa: F401

    if adapter != "auto":
        try:
            selected = _ADAPTERS[adapter]
        except KeyError as exc:
            supported = ", ".join(sorted(_ADAPTERS)) or "none"
            raise NativeMTPCheckpointError(
                f"Unknown native MTP adapter {adapter!r}; registered adapters: {supported}"
            ) from exc
        if not selected.supports(model_path):
            raise NativeMTPCheckpointError(
                f"Adapter {adapter!r} does not support checkpoint {str(model_path)!r}"
            )
        return selected

    matches = [candidate for candidate in _ADAPTERS.values() if candidate.supports(model_path)]
    if len(matches) != 1:
        if not matches:
            raise NativeMTPCheckpointError(
                "No native MTP adapter supports this checkpoint. Expected a model-specific "
                "checkpoint with an explicit native MTP."
            )
        names = ", ".join(candidate.name for candidate in matches)
        raise NativeMTPCheckpointError(
            f"Checkpoint is ambiguous across native MTP adapters: {names}"
        )
    return matches[0]


def create_native_mtp_boost_model(
    model_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    adapter: str = "auto",
    rollout_steps: int = 1,
    hsm_mode: str = "off",
) -> nn.Module:
    """Load a checkpoint-native MTP as a standalone trainable draft model.

    The returned module owns only the MTP as trainable state.  Its copied token
    embedding and LM head are frozen, allowing offline training without loading
    the target backbone.
    """
    selected = get_native_mtp_adapter(model_path, adapter)
    return selected.create(
        model_path,
        dtype=dtype,
        device=device,
        rollout_steps=rollout_steps,
        hsm_mode=hsm_mode,
    )


def attach_native_mtp_online_target(
    model: nn.Module,
    model_path: str | Path,
    target_checkpoint: str | Path,
    *,
    device: torch.device | str,
    max_batch_size: int,
    max_seq_len: int,
    adapter: str = "auto",
) -> None:
    """Attach a frozen native target that produces MTP features online."""
    selected = get_native_mtp_adapter(model_path, adapter)
    selected.attach_online_target(
        model,
        model_path,
        target_checkpoint,
        device=device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )


def export_native_mtp_checkpoint(
    model: nn.Module,
    base_checkpoint: str | Path,
    output_path: str | Path,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    adapter: str = "auto",
) -> Path:
    """Export a trained native MTP into a DSV4-compatible checkpoint directory.

    For FSDP, pass the full state dictionary returned by
    ``Accelerator.get_state_dict``.  The caller must invoke this writer only on
    the main process after that collective has completed.
    """
    selected = get_native_mtp_adapter(base_checkpoint, adapter)
    return selected.export(
        model,
        base_checkpoint,
        output_path,
        state_dict=state_dict,
    )


def load_native_mtp_boost_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> nn.Module:
    """Load a resumable BF16 MTP-boost checkpoint into a freshly-created model.

    The checkpoint intentionally contains only ``mtp.*`` tensors; frozen shared
    embedding and LM-head tensors always come from the native base checkpoint
    passed to :func:`create_native_mtp_boost_model`.  ``checkpoint_path`` can
    be either the file written by :meth:`NativeMTPBoostModel.save_pretrained`
    or its containing Trainer checkpoint directory.
    """
    path = Path(checkpoint_path)
    if path.is_dir():
        candidates = (
            path / "mtp_boost.pt",
            path / "model.safetensors",
            path / "pytorch_model.bin",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected mtp_boost.pt, model.safetensors, or pytorch_model.bin at {checkpoint_path!s}"
        )

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Loading a safetensors MTP checkpoint requires safetensors.") from exc
        state = load_file(path, device="cpu")
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch before the ``weights_only`` argument.
            state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise NativeMTPCheckpointError(f"Native MTP training state at {path} is not a mapping")
    if isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]

    normalized = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if isinstance(key, str) and isinstance(value, torch.Tensor)
    }
    mtp_state = {key: value for key, value in normalized.items() if key.startswith("mtp.")}
    if not mtp_state:
        raise NativeMTPCheckpointError(f"No mtp.* tensors found in native MTP state {path}")
    missing, unexpected = model.load_state_dict(mtp_state, strict=False)
    missing_mtp = [key for key in missing if key.startswith("mtp.")]
    if missing_mtp or unexpected:
        raise NativeMTPCheckpointError(
            "Native MTP training state does not match the created adapter model: "
            f"missing={missing_mtp[:8]}, unexpected={unexpected[:8]}"
        )
    return model
