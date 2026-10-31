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

"""Reuse fake-quantized weights across the micro-batches of one QAT / QAD optimizer step.

Weights only change at the optimizer step, but every micro-batch forward and every activation
recomputation fake-quantizes them again. Inside :func:`weight_cache_scope`, an enrolled weight
quantizer computes its output once and reuses it, with a straight-through gradient to the weight.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary, ref

import torch
from torch import nn

__all__ = [
    "WeightCacheStats",
    "disable_weight_cache",
    "enable_weight_cache",
    "invalidate_weight_cache",
    "weight_cache_scope",
]

# Kept outside the quantizers so that ModelOpt state, deepcopy and checkpoints never carry them.
_ENROLLED: WeakKeyDictionary = WeakKeyDictionary()  # quantizer -> weakref to its model (chunk)
_ACTIVE: WeakKeyDictionary = WeakKeyDictionary()  # quantizer -> _Session of its open scope
_num_open_scopes = 0  # a plain int so the uncached path stays a cheap, compile-friendly check


@dataclass
class WeightCacheStats:
    """Counters of one :func:`weight_cache_scope`."""

    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    retained_bytes: int = 0
    peak_bytes: int = 0


@dataclass
class _Entry:
    key: tuple
    # Holds the source's storage so that its address cannot be reused by another tensor. For a source
    # made afresh on each call (e.g. an onloaded weight), this keeps one more copy alive per entry.
    anchor: torch.Tensor
    value: torch.Tensor
    value_version: int


@dataclass
class _Session:
    stats: WeightCacheStats = field(default_factory=WeightCacheStats)
    entries: dict = field(default_factory=dict)

    def add(self, quantizer, key, source, value) -> _Entry:
        entry = _Entry(key, source.detach(), value, value._version)
        self.entries[quantizer] = entry
        self.stats.misses += 1
        self.stats.retained_bytes += value.untyped_storage().nbytes()
        self.stats.peak_bytes = max(self.stats.peak_bytes, self.stats.retained_bytes)
        return entry

    def drop(self, quantizer) -> None:
        entry = self.entries.pop(quantizer, None)
        if entry is not None:
            self.stats.retained_bytes -= entry.value.untyped_storage().nbytes()
            self.stats.invalidations += 1


class _CachedIdentity(torch.autograd.Function):
    """Return the cached payload, passing the gradient straight through to the weight."""

    @staticmethod
    def forward(ctx, weight, payload):
        ctx.set_materialize_grads(False)
        return payload

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError("The fake-quant weight cache does not support double backward.")
        return grad, None


def enable_weight_cache(
    model: nn.Module | list[nn.Module],
    select: Callable[[str, nn.Module], bool] | None = None,
) -> int:
    """Enroll ``model``'s weight quantizers in the fake-quant weight cache.

    Enrollment alone changes nothing; enrolled quantizers reuse their output only inside
    :func:`weight_cache_scope`. A cache hit passes the gradient straight through to the weight, so
    each enrolled quantizer must have a straight-through backward: ``pass_through_bwd=True``, which
    a registered custom ``backend`` must also honor, and no LSQ.

    Args:
        model: The quantized model, or a list of its model chunks (e.g. Megatron virtual pipeline
            stages). Each chunk owns the quantizers found in it, so a chunk can later be passed alone.
        select: Optional ``select(name, quantizer)`` filter over the weight quantizers, e.g. to
            cache only the routed experts of an MoE model.

    Returns:
        The number of enrolled weight quantizers.
    """
    # local: ..nn imports this utils package at module scope
    from ..nn import QuantModule, TensorQuantizer

    models = _as_list(model)
    previous = _enrolled(models)
    if any(q in _ACTIVE for q in previous):
        raise RuntimeError("Cannot change the weight cache enrollment inside weight_cache_scope.")

    selected = {}
    for m in models:
        names = {id(q): name for name, q in m.named_modules()}
        for module in m.modules():
            if not isinstance(module, QuantModule):
                continue
            for quantizer in module.iter_weight_quantizers_for_calibration():
                name = names.get(id(quantizer), "")
                if select is not None and not select(name, quantizer):
                    continue
                if not isinstance(quantizer, TensorQuantizer):
                    raise ValueError(
                        f"{name}: only TensorQuantizer weight quantizers can be cached."
                    )
                if not quantizer.is_enabled:
                    continue
                if not quantizer._pass_through_bwd:
                    raise ValueError(f"{name}: the weight cache requires pass_through_bwd=True.")
                if getattr(quantizer, "_lsq", False):  # LSQ's backward is not straight-through
                    raise ValueError(f"{name}: the weight cache does not support LSQ quantizers.")
                if any(p.requires_grad for p in quantizer.parameters()):
                    raise ValueError(f"{name}: the weight cache does not support learned scales.")
                owner = _ENROLLED.get(quantizer)
                if owner is not None and owner() is not None and owner() is not m:
                    raise ValueError(
                        f"{name} is already enrolled under another model; disable it there first."
                    )
                if selected.setdefault(quantizer, m) is not m:
                    raise ValueError(f"{name} is in more than one of the given model chunks.")
    if not selected:
        raise ValueError("No enabled weight quantizers were selected for the weight cache.")

    for quantizer in previous:
        del _ENROLLED[quantizer]
    for quantizer, m in selected.items():
        _ENROLLED[quantizer] = ref(m)
    return len(selected)


def disable_weight_cache(model: nn.Module | list[nn.Module]) -> None:
    """Remove ``model``'s weight quantizers from the weight cache."""
    enrolled = _enrolled(_as_list(model))
    if any(q in _ACTIVE for q in enrolled):
        raise RuntimeError("Cannot change the weight cache enrollment inside weight_cache_scope.")
    for quantizer in enrolled:
        del _ENROLLED[quantizer]


@contextmanager
def weight_cache_scope(model: nn.Module | list[nn.Module]) -> Iterator[WeightCacheStats]:
    """Reuse the fake-quantized weights of ``model``'s enrolled quantizers within this scope.

    Wrap all forward and backward passes of one optimizer step (every micro-batch and activation
    recomputation) and close the scope before ``optimizer.step()``:

    .. code-block:: python

        mtq.utils.enable_weight_cache(model)
        for batch in loader:
            with mtq.utils.weight_cache_scope(model) as stats:
                for micro_batch in batch:
                    loss_fn(model(micro_batch)).backward()
            optimizer.step()

    An in-place update of a weight inside the scope recomputes its entry, unless it bypasses the
    version counter through ``.data``. Call :func:`invalidate_weight_cache` before changing
    quantizer state such as ``amax``.

    Yields:
        The :class:`WeightCacheStats` of this scope; its ``retained_bytes`` is zero after it closes.
    """
    global _num_open_scopes
    quantizers = _enrolled(_as_list(model))
    if not quantizers:
        raise ValueError("Call enable_weight_cache before opening weight_cache_scope.")
    if any(q in _ACTIVE for q in quantizers):
        raise RuntimeError("weight_cache_scope cannot be nested over the same quantizers.")
    session = _Session()
    for quantizer in quantizers:
        _ACTIVE[quantizer] = session
    _num_open_scopes += 1
    try:
        yield session.stats
    finally:
        _num_open_scopes -= 1
        for quantizer in quantizers:
            _ACTIVE.pop(quantizer, None)
        session.entries.clear()
        session.stats.retained_bytes = 0


def invalidate_weight_cache(model: nn.Module | list[nn.Module]) -> None:
    """Drop the cached weights of ``model`` in the open scope, e.g. before changing ``amax``."""
    for quantizer in _enrolled(_as_list(model)):
        session = _ACTIVE.get(quantizer)
        if session is not None:
            session.drop(quantizer)


def cached_fake_quantize(quantizer, inputs: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """``quantizer._fake_quantize(inputs)``, reused while ``quantizer`` is in an open scope.

    ``source`` is the tensor the quantizer received, from which ``inputs`` was derived (e.g. by
    padding). It keys the entry, so that a derived tensor made afresh on each call still hits.
    """
    session = _ACTIVE.get(quantizer) if _num_open_scopes else None
    if session is None:
        return quantizer._fake_quantize(inputs)

    key = _input_key(source)
    entry = session.entries.get(quantizer)
    if entry is not None and entry.key != key:
        session.drop(quantizer)
        entry = None
    if entry is None:
        with torch.no_grad():
            value = quantizer._fake_quantize(inputs)
        if (value.shape, value.dtype, value.device) != (inputs.shape, inputs.dtype, inputs.device):
            raise ValueError("A cached fake quant must keep the weight's shape, dtype and device.")
        if value.untyped_storage().data_ptr() == inputs.untyped_storage().data_ptr():
            raise ValueError("A cached fake quant must not return a view of the weight.")
        entry = session.add(quantizer, key, source, value)
    elif entry.value._version != entry.value_version:
        session.drop(quantizer)
        raise RuntimeError("A cached fake-quantized weight was modified in place.")
    else:
        session.stats.hits += 1
    return _CachedIdentity.apply(inputs, entry.value)


def _as_list(model) -> list[nn.Module]:
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    if not models or not all(isinstance(m, nn.Module) for m in models):
        raise TypeError("Expected a module or a non-empty list of modules.")
    return models


def _enrolled(models: list[nn.Module]) -> list[nn.Module]:
    roots = {id(m) for m in models}
    return [q for q, owner in _ENROLLED.items() if id(owner()) in roots]


def _input_key(inputs: torch.Tensor) -> tuple:
    # The version counter changes on any in-place update, e.g. an optimizer step.
    return (
        inputs.untyped_storage().data_ptr(),
        inputs.storage_offset(),
        tuple(inputs.shape),
        inputs.stride(),
        inputs.dtype,
        inputs.device,
        inputs._version,
    )
