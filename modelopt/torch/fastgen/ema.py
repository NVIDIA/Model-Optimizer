# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exponential moving average of a student network, FSDP2 DTensor aware.

Ported from ``FastGen/fastgen/callbacks/ema.py`` (lines 20-169) but exposed as a plain
class rather than a framework-specific callback. The caller decides when to call
:meth:`update` (typically after ``optimizer.step()``), how to persist the shadow state
(via :meth:`state_dict`), and when to publish the EMA weights back to a target module
(via :meth:`copy_to`).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .config import EMAConfig

__all__ = ["ExponentialMovingAverage"]


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _resolve_dtype(config_dtype: str | None, fallback: torch.dtype) -> torch.dtype:
    """Map an ``EMAConfig.dtype`` string to a ``torch.dtype``.

    ``config_dtype is None`` falls through to ``fallback`` (the live parameter's dtype).
    """
    if config_dtype is None:
        return fallback
    try:
        return _DTYPE_MAP[config_dtype]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported EMA dtype {config_dtype!r}; expected one of {sorted(_DTYPE_MAP)} or None."
        ) from exc


def _is_distributed_tensor(t: torch.Tensor) -> bool:
    """Return True if ``t`` is a ``torch.distributed.DTensor`` supporting ``full_tensor()``."""
    return hasattr(t, "full_tensor") and callable(t.full_tensor)


def _gather_full(param: torch.Tensor, *, fsdp2: bool) -> torch.Tensor:
    """Return a materialised full tensor for ``param``.

    Mirrors the FSDP2 branch in ``FastGen/fastgen/callbacks/ema.py:128-139``: if CPU
    offloading is enabled the local shard must be moved to CUDA before ``full_tensor()``
    can perform the all-gather (which requires a CUDA backend).
    """
    if fsdp2 and _is_distributed_tensor(param):
        if param.device.type == "cpu":
            return param.to("cuda").full_tensor()
        return param.full_tensor()
    return param


def _strip_checkpoint_prefix(name: str) -> str:
    """Remove the ``_checkpoint_wrapped_module.`` prefix injected by FSDP2 activation checkpointing."""
    return name.replace("_checkpoint_wrapped_module.", "")


class ExponentialMovingAverage:
    """FSDP2-aware EMA tracker for a PyTorch module.

    The tracker stores a shadow state dict: parameters are promoted per
    :attr:`EMAConfig.dtype` (default fp32) while buffers are kept in the live module's
    dtype. Buffers are replicated across ranks and stepped via ``copy_`` rather than
    ``lerp_``, so the bf16-roundoff argument that motivates parameter promotion
    doesn't apply — preserving the live dtype makes the buffer restore exact.

    By default the tracker materialises the full tensor per parameter
    (``mode='full_tensor'``) so the EMA represents the globally averaged weights even
    when the model is sharded across ranks. A ``mode='local_shard'`` fallback is
    available for memory-constrained settings — it does not all-gather and therefore
    each rank holds an EMA of its local shard only.

    Example::

        ema = ExponentialMovingAverage(student, EMAConfig(decay=0.999))
        for step in range(max_steps):
            ...  # compute loss, backward, optimizer.step()
            ema.update(student, iteration=step)

        ema.copy_to(student_for_eval)  # publish for inference
    """

    def __init__(self, model: nn.Module, config: EMAConfig) -> None:
        """Pre-allocate the shadow state from ``model``'s parameters and buffers."""
        self.config = config
        self._shadow: dict[str, torch.Tensor] = {}
        self._buffer_shadow: dict[str, torch.Tensor] = {}
        self._initialized = False

        # Pre-allocate shadow storage as a deepcopy of the live parameters on their
        # current devices. Shadow dtype is promoted to ``EMAConfig.dtype`` (default
        # fp32) so EMA updates remain meaningful even when the live model is
        # bf16/fp16: the per-step increment ``(live - shadow) * (1 - beta)`` rounds
        # to zero in bf16 (unit roundoff ~2^-8 of |shadow|) long before the live
        # weights have converged. Pass ``dtype=None`` to fall back to the live
        # parameter's dtype.
        with torch.no_grad():
            for name, p in model.named_parameters():
                clean = _strip_checkpoint_prefix(name)
                full = _gather_full(p.detach(), fsdp2=config.fsdp2)
                target_dtype = _resolve_dtype(config.dtype, full.dtype)
                self._shadow[clean] = copy.deepcopy(full).to(dtype=target_dtype)
            # Buffers are replicated (not averaged) across ranks and stepped via
            # ``copy_``, so the bf16-roundoff argument that drives ``EMAConfig.dtype``
            # on parameters doesn't apply — keep buffers in the live dtype for exact
            # restore.
            for name, b in model.named_buffers():
                clean = _strip_checkpoint_prefix(name)
                self._buffer_shadow[clean] = copy.deepcopy(b.detach())

    # ------------------------------------------------------------------ #
    #  Decay schedules                                                   #
    # ------------------------------------------------------------------ #

    def _beta(self, iteration: int) -> float:
        cfg = self.config
        if cfg.type == "constant":
            return cfg.decay
        if cfg.type == "power":
            # (1 - 1/iter) ** (gamma + 1); iteration must be > 0 for this to be finite.
            safe_iter = max(iteration, 1)
            return (1.0 - 1.0 / safe_iter) ** (cfg.gamma + 1.0)
        if cfg.type == "halflife":
            ema_halflife_nimg = cfg.halflife_kimg * 1000.0
            cur_nimg = iteration * cfg.batch_size
            if cfg.rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * cfg.rampup_ratio)
            return 0.5 ** (cfg.batch_size / max(ema_halflife_nimg, 1e-8))
        raise ValueError(f"Unsupported EMA type: {cfg.type!r}")

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def update(self, model: nn.Module, *, iteration: int) -> None:
        """Update the shadow state from ``model`` at the given iteration.

        Skips updates before :attr:`EMAConfig.start_iter`. On the iteration that equals
        ``start_iter`` the shadow is (re-)initialised from the live weights; after that
        it is updated with ``shadow = beta * shadow + (1 - beta) * live``.
        """
        if iteration < self.config.start_iter:
            return

        if iteration == self.config.start_iter or not self._initialized:
            self._copy_from_model(model)
            self._initialized = True
            return

        beta = self._beta(iteration)

        for name, p in model.named_parameters():
            clean = _strip_checkpoint_prefix(name)
            if clean not in self._shadow:
                continue
            shadow = self._shadow[clean]
            if self.config.mode == "full_tensor":
                live = _gather_full(p.detach(), fsdp2=self.config.fsdp2)
            else:
                live = p.detach().to_local() if _is_distributed_tensor(p) else p.detach()
            shadow.lerp_(live.to(device=shadow.device, dtype=shadow.dtype), 1.0 - beta)

        # Buffers are replicated across ranks under FSDP2, so we just copy.
        for name, b in model.named_buffers():
            clean = _strip_checkpoint_prefix(name)
            if clean in self._buffer_shadow:
                shadow = self._buffer_shadow[clean]
                shadow.copy_(b.detach().to(device=shadow.device, dtype=shadow.dtype))

    @torch.no_grad()
    def copy_to(self, target: nn.Module) -> None:
        """Load the shadow state into ``target`` (which should share the tracked module's structure).

        The target is expected to be an unsharded module (i.e. the caller has unwrapped
        any FSDP2 wrappers before calling). For sharded targets, prefer saving the
        shadow via :meth:`state_dict` and reloading it through the framework's usual
        checkpoint path.
        """
        for name, p in target.named_parameters():
            clean = _strip_checkpoint_prefix(name)
            if clean in self._shadow:
                shadow = self._shadow[clean]
                p.data.copy_(shadow.to(device=p.device, dtype=p.dtype))
        for name, b in target.named_buffers():
            clean = _strip_checkpoint_prefix(name)
            if clean in self._buffer_shadow:
                shadow = self._buffer_shadow[clean]
                b.data.copy_(shadow.to(device=b.device, dtype=b.dtype))

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the shadow state (parameters + buffers) for checkpointing."""
        merged: dict[str, torch.Tensor] = {}
        merged.update(self._shadow)
        merged.update(self._buffer_shadow)
        return merged

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore the shadow state from a previously saved dict."""
        for k, v in state.items():
            if k in self._shadow:
                shadow = self._shadow[k]
                shadow.copy_(v.to(device=shadow.device, dtype=shadow.dtype))
            elif k in self._buffer_shadow:
                shadow = self._buffer_shadow[k]
                shadow.copy_(v.to(device=shadow.device, dtype=shadow.dtype))
        self._initialized = True

    # ------------------------------------------------------------------ #
    #  Internals                                                         #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _copy_from_model(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            clean = _strip_checkpoint_prefix(name)
            if clean not in self._shadow:
                continue
            shadow = self._shadow[clean]
            live = _gather_full(p.detach(), fsdp2=self.config.fsdp2)
            shadow.copy_(live.to(device=shadow.device, dtype=shadow.dtype))
        for name, b in model.named_buffers():
            clean = _strip_checkpoint_prefix(name)
            if clean in self._buffer_shadow:
                shadow = self._buffer_shadow[clean]
                shadow.copy_(b.detach().to(device=shadow.device, dtype=shadow.dtype))
