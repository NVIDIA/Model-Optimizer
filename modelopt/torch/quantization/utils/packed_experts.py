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

"""Fused MoE experts kept in their checkpoint's compressed format, and decompressed on use.

A checkpoint that stores its routed experts in a 4-bit format holds them in about 4.25 bits a
weight; decompressed at load they take 16. :func:`pack_experts` keeps them as stored instead: each
projection ``<proj>`` of the experts module holds its packed values, beside the other tensors the
format stores, ``<proj><suffix>``. MXFP4 (:class:`PackedFormat`) stores its E8M0 scales as
``<proj>_scales``.

Every call of the module's forward runs on the decompressed projections, put in place of the packed
ones for that call (:func:`unpacked`), so the module computes what its decompressing load would. A
reader outside the forward opens the same window through
:func:`~modelopt.torch.quantization.utils.enable_weight_access_and_writeback`, and the export
decompresses the module for good (:func:`unpack_experts`) before it writes it. The packed format
cannot hold a change, so the window refuses one, as a fold does: calibration that updates weights in
place needs the experts decompressed at load. A projection sharded over the experts (a ``DTensor``)
decompresses this rank's experts, sharded alike.
"""

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from .numeric_utils import dequantize_mxfp4

__all__ = [
    "PackedFormat",
    "holds_packed_experts",
    "is_packed",
    "pack_experts",
    "unpack_experts",
    "unpacked",
]

_STATE = "_modelopt_packed_experts"


class PackedFormat:
    """How each expert of a packed projection is stored, and how it decompresses: MXFP4 here."""

    def stored(
        self, name: str, out: int, inp: int
    ) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """The tensors one expert of projection ``name``, ``(out, inp)`` decompressed, is stored as.

        ``suffix -> (shape, dtype)``, the suffix ``""`` naming the projection's own parameter.
        """
        return {"": ((out, inp // 2), torch.uint8), "_scales": ((out, inp // 32), torch.uint8)}

    def decompress(self, name: str, stored: dict[str, torch.Tensor], dtype: torch.dtype):
        """Experts ``(e, ...)`` of projection ``name``, stored as :meth:`stored` says, decompressed."""
        return dequantize_mxfp4(stored[""], stored["_scales"], dtype)


@dataclass
class _PackedExperts:
    """How a module's projections are packed, and the window open on it, if any."""

    # Each packed projection's shape decompressed, as a decompressing load holds it.
    shapes: dict[str, torch.Size]
    # Whether the decompressed projections are stored (num_experts, in, out), as GPT-OSS's are.
    transposed: bool
    dtype: torch.dtype
    packed_format: PackedFormat
    # The suffixes of the tensors stored beside each packed projection.
    siblings: tuple[str, ...]
    # The packed projections the open window put aside, by name.
    packed: dict[str, nn.Parameter] = field(default_factory=dict)


def is_packed(module: nn.Module) -> bool:
    """Whether ``module``'s expert projections are stored packed (:func:`pack_experts`)."""
    return getattr(module, _STATE, None) is not None


def holds_packed_experts(module: nn.Module) -> bool:
    """Whether ``module``'s expert projections hold packed bytes now: packed, with no window open."""
    state = getattr(module, _STATE, None)
    return state is not None and not state.packed


def pack_experts(
    module: nn.Module,
    projections: "tuple[str, ...]" = ("gate_up_proj", "down_proj"),
    *,
    transposed: bool = False,
    packed_format: PackedFormat | None = None,
) -> None:
    """Store ``module``'s expert ``projections`` packed, for a load to fill with the checkpoint's.

    Each is a 3-D parameter, ``(num_experts, out, in)``, or ``(num_experts, in, out)`` when
    ``transposed``. It is replaced by the tensors ``packed_format`` (MXFP4 by default) stores it as,
    on the same device: a packed one of the same name, and its siblings. It decompresses to the
    parameter's dtype. The module's forward then runs on the decompressed projections. Checked before
    anything changes, so a refusal leaves the module as it was.
    """
    packed_format = packed_format or PackedFormat()
    if is_packed(module):
        raise ValueError(f"{type(module).__name__}'s experts are already packed.")
    if "forward" in vars(module):
        # The window must open inside whatever wraps the forward, as transformers' expert parallelism does.
        raise NotImplementedError(
            f"{type(module).__name__} already has its forward wrapped; pack its experts before."
        )
    weights: dict[str, nn.Parameter] = {}
    layouts = {}
    for name in projections:
        weight = getattr(module, name, None)
        if not isinstance(weight, nn.Parameter) or weight.dim() != 3:
            raise ValueError(f"{type(module).__name__}.{name} is not a 3-D expert parameter.")
        out, inp = (weight.shape[2], weight.shape[1]) if transposed else weight.shape[1:]
        if inp % 32:
            raise ValueError(
                f"{type(module).__name__}.{name} {tuple(weight.shape)}: the packed formats pack "
                "the input dimension in blocks of 32."
            )
        weights[name], layouts[name] = weight, packed_format.stored(name, out, inp)
    dtypes = {weight.dtype for weight in weights.values()}
    if len(dtypes) != 1:
        raise ValueError(f"{type(module).__name__}'s expert projections differ in dtype: {dtypes}.")
    for name, weight in weights.items():
        for suffix, (shape, dtype) in layouts[name].items():
            packed = torch.empty(len(weight), *shape, dtype=dtype, device=weight.device)
            setattr(module, name + suffix, nn.Parameter(packed, requires_grad=False))
    siblings = tuple(suffix for suffix in layouts[next(iter(layouts))] if suffix)
    shapes = {name: weight.shape for name, weight in weights.items()}
    state = _PackedExperts(shapes, transposed, dtypes.pop(), packed_format, siblings)
    setattr(module, _STATE, state)
    # On the instance, not by a subclass, so the module keeps its class's forward, by which modules
    # are matched to their quantized versions; looked up when called, so a quantized one is reached,
    # which is why converting the module keeps it (DynamicModule.convert).
    forward = functools.partial(_forward, module)
    forward._modelopt_calls_class_forward = True  # type: ignore[attr-defined]
    module.forward = forward


def _forward(module: nn.Module, *args, **kwargs):
    with unpacked(module):
        return type(module).forward(module, *args, **kwargs)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _decompress(module: nn.Module, state: _PackedExperts, name: str, packed: torch.Tensor):
    """Packed projection ``name``, decompressed and laid out as a decompressing load holds it.

    This rank's experts of a projection sharded over the experts, sharded alike.
    """
    stored = {"": _local(packed)}
    stored.update((suffix, _local(getattr(module, name + suffix))) for suffix in state.siblings)
    decompress = functools.partial(state.packed_format.decompress, name, dtype=state.dtype)
    if not state.transposed:
        weight = decompress(stored)
    else:
        experts, device = len(stored[""]), stored[""].device
        weight = torch.empty(experts, *state.shapes[name][1:], dtype=state.dtype, device=device)
        for expert in range(experts):  # one expert at a time bounds the transient
            weight[expert] = decompress({k: v[expert : expert + 1] for k, v in stored.items()})[0].T
    if not isinstance(packed, DTensor):
        return weight
    shape = (packed.shape[0], *weight.shape[1:])
    stride = torch.empty(shape, device="meta").stride()
    return DTensor.from_local(
        weight, packed.device_mesh, packed.placements, run_check=False, shape=shape, stride=stride
    )


@contextmanager
def unpacked(module: nn.Module, writeback: bool = False):
    """Put ``module``'s decompressed expert projections in place of the packed ones while open.

    A no-op for a module whose experts are not packed, or already open. On exit the packed ones are
    back, unless :func:`unpack_experts` decompressed the module for good inside. With ``writeback``,
    a decompressed projection replaced or changed is refused: the packed format cannot hold the
    change. Compared by value, as an update through ``.data`` leaves no other trace.
    """
    state: _PackedExperts | None = getattr(module, _STATE, None)
    if state is None or state.packed:
        yield
        return
    packed = {name: module._parameters[name] for name in state.shapes}
    unpacked_weights = {
        name: nn.Parameter(_decompress(module, state, name, blocks), requires_grad=False)
        for name, blocks in packed.items()
    }
    state.packed = packed
    # Set on the module's parameters, not through setattr: a quantized module reads its dynamic
    # attributes from there.
    module._parameters.update(unpacked_weights)
    try:
        yield
    finally:
        if is_packed(module):  # unless unpack_experts decompressed the module for good inside
            current = {name: module._parameters[name] for name in unpacked_weights}
            module._parameters.update(packed)
            state.packed = {}
            changed = writeback and [
                name
                for name, weight in unpacked_weights.items()
                if current[name] is not weight
                or not torch.equal(
                    _local(weight), _local(_decompress(module, state, name, packed[name]))
                )
            ]
            if changed:
                raise ValueError(
                    f"{type(module).__name__}.{', '.join(changed)} changed, but its experts are "
                    "stored packed, which cannot hold the change. Load the checkpoint with its "
                    "experts decompressed to update them."
                )


def unpack_experts(module: nn.Module) -> None:
    """Decompress ``module``'s packed expert projections for good, and drop what they stored."""
    state = getattr(module, _STATE, None)
    if state is None:
        return
    for name in state.shapes:
        # An open window holds them decompressed already.
        weight = (
            module._parameters[name]
            if state.packed
            else _decompress(module, state, name, module._parameters[name])
        )
        module._parameters[name] = nn.Parameter(weight.data, requires_grad=False)
        for suffix in state.siblings:
            delattr(module, name + suffix)
    delattr(module, _STATE)
    forward = vars(module).get("forward")
    if isinstance(forward, functools.partial) and forward.func is _forward:
        # Wrapped since, as transformers' expert parallelism wraps it, it stays: it calls through.
        del module.forward
