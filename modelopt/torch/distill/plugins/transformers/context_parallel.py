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


"""Chronological CP layouts and scoped native SDPA, with explicit dense fallbacks."""

import contextvars
import importlib
import threading
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from torch.overrides import TorchFunctionMode

__all__ = ["CPSequenceLayout"]

_DISPATCH_LOCK = threading.RLock()
_ACTIVE_LAYOUT: contextvars.ContextVar["CPSequenceLayout | None"] = contextvars.ContextVar(
    "modelopt_cp_layout", default=None
)


class _SDPA(nn.Module):
    def forward(self, *args, **kwargs):
        return F.scaled_dot_product_attention(*args, **kwargs)


class _AttentionMode(TorchFunctionMode):
    def __init__(self, layout, attention):
        super().__init__()
        self.layout = layout
        self.attention = attention

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is not F.scaled_dot_product_attention:
            return func(*args, **kwargs)
        mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
        dropout = args[4] if len(args) > 4 else kwargs.get("dropout_p", 0.0)
        causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)
        if mask is not None or not causal or dropout:
            raise ValueError(
                "Native CP SDPA requires unmasked, deterministic full causal attention; "
                "use the explicit full-context fallback for other attention semantics."
            )
        qkv = [
            args[index] if index < len(args) else kwargs[name]
            for index, name in enumerate(("query", "key", "value"))
        ]
        if any(
            isinstance(value, DTensor) or value.shape[2] != self.layout.local_length
            for value in qkv
        ):
            raise ValueError(
                "Native CP SDPA expects local dense [batch, heads, sequence, dim] shards."
            )
        return self.attention(*args, **kwargs)


class CPSequenceLayout:
    """Own sequence placement on an existing one-dimensional context mesh.

    Head-tail placement matches Torch's native load balancer. Padding is appended
    chronologically and must carry zero validity. This object never creates a
    process group, infers tensor dimensions, or changes model parameter placement.
    """

    def __init__(self, mesh, sequence_length, *, load_balance=True, device=None):
        """Create explicit sequence indices; native attention requires Torch 2.14 APIs."""
        if mesh.ndim != 1 or sequence_length <= 0:
            raise ValueError("CP requires a one-dimensional mesh and a positive sequence length.")
        self.mesh = mesh
        self.sequence_length = sequence_length
        self.world_size = mesh.size()
        self.rank = mesh.get_local_rank()
        self.load_balance = load_balance
        multiple = self.world_size * (2 if load_balance else 1)
        self.padded_length = ((sequence_length + multiple - 1) // multiple) * multiple
        self.local_length = self.padded_length // self.world_size
        self.device = torch.device(device or mesh.device_type)
        chronological = torch.arange(self.padded_length, device=self.device)
        if load_balance:
            chunks = chronological.reshape(2 * self.world_size, -1)
            rank_ids = torch.arange(self.world_size, device=self.device)
            self.reorder_indices = torch.stack(
                (chunks[rank_ids], chunks[-rank_ids - 1]), 1
            ).flatten()
        else:
            self.reorder_indices = chronological
        self.local_indices = self.reorder_indices.reshape(self.world_size, -1)[self.rank]

    def shard(self, tensor, dim, *, pad_value=0):
        """Select this rank's head-tail tokens, padding only the global sequence tail."""
        dim %= tensor.ndim
        if tensor.shape[dim] != self.sequence_length:
            raise ValueError("Full tensor length differs from the CP layout.")
        if self.padded_length != self.sequence_length:
            shape = list(tensor.shape)
            shape[dim] = self.padded_length - self.sequence_length
            tensor = torch.cat((tensor, tensor.new_full(shape, pad_value)), dim=dim)
        return tensor.index_select(dim, self.local_indices.to(tensor.device))

    @contextmanager
    def attention(self):
        """Scope Torch's module-wrapper CP dispatcher without monkeypatching global SDPA.

        Keep this scope open through student backward: native CP backward also
        dispatches distributed attention operators. Unsupported blocks must run
        outside this scope through replicated full-context execution instead.
        """
        if self.world_size == 1:
            yield
            return
        active = _ACTIVE_LAYOUT.get()
        if active is self:
            yield
            return
        if active is not None:
            raise ValueError("Different CP layouts cannot share one nested attention scope.")
        try:
            native = importlib.import_module(
                "torch.distributed.tensor.experimental._context_parallel._attention"
            )
            style_class = native._ContextParallel
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "Scoped native CP requires Torch's module-wrapper attention API."
            ) from error
        style = style_class(seq_dim=2, attention_type=style_class.AttentionType.SDPA)
        attention = style._apply(_SDPA(), self.mesh)
        with _DISPATCH_LOCK:
            previous_handlers = DTensor._op_dispatcher._custom_op_handlers
            previous_balance = native._cp_options.enable_load_balance
            token = _ACTIVE_LAYOUT.set(self)
            try:
                native._cp_options.enable_load_balance = self.load_balance
                native._enable_cp_dtensor_dispatcher()
                # Preserve unrelated handlers registered after the native module's import.
                DTensor._op_dispatcher._custom_op_handlers = {
                    **previous_handlers,
                    **native.custom_ops,
                }
                with _AttentionMode(self, attention):
                    yield
            finally:
                native._disable_cp_dtensor_dispatcher()
                DTensor._op_dispatcher._custom_op_handlers = previous_handlers
                native._cp_options.enable_load_balance = previous_balance
                _ACTIVE_LAYOUT.reset(token)
