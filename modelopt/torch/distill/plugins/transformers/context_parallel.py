# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chronological CP layouts and scoped native SDPA, with explicit dense fallbacks."""

import contextvars
import importlib
import threading
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.nn.functional import all_gather
from torch.distributed.tensor import DTensor
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten_with_path, tree_unflatten

__all__ = ["CPSequenceLayout"]

_DISPATCH_LOCK = threading.RLock()
_ACTIVE_LAYOUT = contextvars.ContextVar("modelopt_cp_layout", default=None)
_SUSPEND_ATTENTION = contextvars.ContextVar("modelopt_cp_full_context", default=False)


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
        if func is not F.scaled_dot_product_attention or _SUSPEND_ATTENTION.get():
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


def _map_sequence_leaves(tree, dimensions, function):
    leaves, specification = tree_flatten_with_path(tree)
    found = set()
    output = []
    for path, value in leaves:
        keys = tuple(
            getattr(key, "key", getattr(key, "idx", getattr(key, "name", None))) for key in path
        )
        if keys in dimensions:
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Declared sequence input {keys} is not a tensor.")
            value = function(value, dimensions[keys])
            found.add(keys)
        output.append(value)
    missing = set(dimensions) - found
    if missing:
        raise ValueError(f"Declared sequence paths are absent: {sorted(missing, key=str)}")
    return tree_unflatten(output, specification)


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
        self.restore_indices = self.reorder_indices.argsort()

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

    def gather(self, tensor, dim):
        """Differentiably gather local shards and undo head-tail order before execution."""
        dim %= tensor.ndim
        if isinstance(tensor, DTensor) or tensor.shape[dim] != self.local_length:
            raise ValueError("CP gather expects a local dense tensor matching the layout.")
        if self.world_size > 1:
            tensor = torch.cat(
                all_gather(tensor.contiguous(), group=self.mesh.get_group()), dim=dim
            )
        tensor = tensor.index_select(dim, self.restore_indices.to(tensor.device))
        return tensor.narrow(dim, 0, self.sequence_length)

    def partition_valid_mask(self, valid_mask, dim=1):
        """Keep a full-context mask but count each real token on exactly one CP rank."""
        dim %= valid_mask.ndim
        if valid_mask.shape[dim] != self.sequence_length:
            raise ValueError("Validity length differs from the CP layout.")
        owned = torch.zeros(self.padded_length, device=valid_mask.device, dtype=torch.bool)
        owned[self.local_indices.to(valid_mask.device)] = True
        shape = [1] * valid_mask.ndim
        shape[dim] = self.sequence_length
        return valid_mask.bool() & owned[: self.sequence_length].reshape(shape)

    def replicated_batch(self, batch, *, valid_mask=None, sequence_dim=1):
        """Preserve all full-context inputs; partition only explicit loss-token validity.

        Every rank runs the complete forward/replay. CP gradient reduction recovers
        the global objective, but this fallback does not reduce activation memory
        or sequence computation. No structured input dimensions are guessed.
        """
        valid_mask = batch.get("valid_mask") if valid_mask is None else valid_mask
        if valid_mask is None:
            raise ValueError("Replicated CP requires an explicit token-validity mask.")
        return {**batch, "valid_mask": self.partition_valid_mask(valid_mask, sequence_dim)}

    def full_context(self, function, args, kwargs, *, input_dims, output_dims):
        """Gather declared sequence leaves, run dense math, then shard declared outputs.

        Paths address ``{'args': args, 'kwargs': kwargs}``; output paths address
        the returned pytree. Example: ``input_dims={('args', 0): 1}`` and
        ``output_dims={(): 1}`` for a tensor-returning recurrent block. Tensor
        gathering and output selection remain differentiable during replay.
        """
        inputs = _map_sequence_leaves({"args": args, "kwargs": kwargs}, input_dims, self.gather)
        token = _SUSPEND_ATTENTION.set(True)
        try:
            output = function(*inputs["args"], **inputs["kwargs"])
        finally:
            _SUSPEND_ATTENTION.reset(token)
        return _map_sequence_leaves(output, output_dims, self.shard)

    @contextmanager
    def attention(self):
        """Scope Torch's module-wrapper CP dispatcher without monkeypatching global SDPA.

        Keep this scope open through student backward: native CP backward also
        dispatches distributed attention operators. Unsupported blocks must run
        through ``full_context`` or replicated full-context execution instead.
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
