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

"""Base class for parallelism-aware activation-scoring hooks.

A scoring hook is a forward hook that accumulates a per-target statistic during a
calibration pass and, at :meth:`finalize`, assembles the final per-element score
by composing the reduction primitives (see :mod:`..reduction`) so the result is
identical across any parallel layout. Each subclass ports one legacy importance
hook from ``modelopt.torch.prune.importance_hooks`` while adding the scored-axis
GATHER (TP/EP) and token-axis SUM-reduce (``dp_cp``).

Lifecycle::

    hook = SomeScorer(module, groups, block_idx=i)
    handle = hook.register()          # module.register_forward_hook(hook)
    ... run the forward over the calibration data ...
    score = hook.finalize()           # gathered + reduced; identical on every rank
    hook.remove()
"""

from contextlib import contextmanager

import torch

from ..reduction import MeshGroups

__all__ = ["ScoringHook"]


class ScoringHook:
    """Base forward hook for activation scoring.

    Attributes:
        target_type: ``"ffn"`` | ``"attn"`` | ``"moe"`` — which pruning target this
            scores; set by the subclass and used by the output writer.
        method: the legacy method name this reproduces (e.g. ``"independent"``).
        block_idx / name: optional identity attached by the target resolver so the
            output can be keyed canonically.
    """

    target_type: str = ""
    method: str = ""
    write_all_ep_ranks: bool = False
    _nested_disable_depth: int = 0
    checkpoint_tensor_names: tuple[str, ...] = ()
    checkpoint_scalar_names: tuple[str, ...] = ()

    def __init__(
        self,
        module,
        groups: MeshGroups,
        *,
        block_idx: int | None = None,
        name: str | None = None,
    ):
        self.module = module
        self.groups = groups
        self.block_idx = block_idx
        self.name = name
        self._handle = None
        self._handles = []
        # When False, the registered dispatch is a no-op. Used to skip the pipeline's
        # one-time shape-inference forward (run on all-zero tensors), which would otherwise
        # pollute the accumulated statistics.
        self.enabled = True
        self._canonical_sequence_ids: torch.Tensor | None = None
        self._canonical_num_samples = 0
        self._canonical_sequence_cursors: dict[str, int] = {}

    def set_batch_metadata(self, *, sequence_ids: torch.Tensor, num_samples: int) -> None:
        """Set canonical token validity for one forward batch.

        ``sequence_ids`` is always ``[batch, sequence]`` and uses ``-1`` for
        padding. Scorers that flatten token dimensions call
        :meth:`_flatten_valid_tokens` or :meth:`_valid_token_mask`; both also
        account for pipeline microbatching with an independent cursor per
        observation stream.
        """
        if sequence_ids.ndim != 2:
            raise ValueError("scorer sequence_ids must be [batch, sequence]")
        if int(num_samples) < 0:
            raise ValueError("scorer num_samples must be non-negative")
        self._canonical_sequence_ids = sequence_ids
        self._canonical_num_samples = int(num_samples)
        self._canonical_sequence_cursors.clear()

    def _aligned_token_ids(
        self,
        tensor: torch.Tensor,
        *,
        trailing_dims: int,
        stream: str,
    ) -> torch.Tensor | None:
        """Align canonical IDs with padded ``[B,T,...]`` or packed ``[BT,...]``."""
        sequence_ids = self._canonical_sequence_ids
        if sequence_ids is None:
            return None
        if trailing_dims < 0 or tensor.ndim <= trailing_dims:
            raise ValueError(
                f"cannot align scorer tensor shape {tuple(tensor.shape)} "
                f"with trailing_dims={trailing_dims}"
            )
        cursor = self._canonical_sequence_cursors.get(stream, 0)
        available_rows = int(sequence_ids.shape[0]) - cursor
        sequence = int(sequence_ids.shape[1])
        leading_dims = tensor.ndim - trailing_dims
        if (
            leading_dims >= 2
            and int(tensor.shape[0]) <= available_rows
            and int(tensor.shape[1]) == sequence
        ):
            rows = int(tensor.shape[0])
        else:
            tokens = int(tensor.shape[0])
            if sequence < 1 or tokens % sequence:
                raise ValueError(
                    f"scorer tensor shape {tuple(tensor.shape)} does not align with "
                    f"sequence_ids shape {tuple(sequence_ids.shape)}"
                )
            rows = tokens // sequence
        if rows < 1 or rows > available_rows:
            raise ValueError(
                "scorer activation consumed more rows than the canonical batch: "
                f"stream={stream!r} cursor={cursor} rows={rows} available={available_rows}"
            )
        stop = cursor + rows
        self._canonical_sequence_cursors[stream] = stop
        return sequence_ids[cursor:stop].reshape(-1).to(device=tensor.device)

    def _flatten_valid_tokens(
        self,
        tensor: torch.Tensor,
        *,
        trailing_dims: int = 1,
        stream: str = "default",
    ) -> torch.Tensor:
        """Flatten token dimensions and remove canonical padding positions."""
        trailing_shape = tuple(tensor.shape[-trailing_dims:]) if trailing_dims else ()
        flat = tensor.reshape(-1, *trailing_shape)
        ids = self._aligned_token_ids(
            tensor,
            trailing_dims=trailing_dims,
            stream=stream,
        )
        if ids is None:
            return flat
        if int(ids.numel()) != int(flat.shape[0]):
            raise ValueError(
                f"canonical token count {ids.numel()} does not match scorer tensor "
                f"token count {flat.shape[0]}"
            )
        return flat[ids >= 0]

    def _valid_token_mask(
        self,
        tensor: torch.Tensor,
        *,
        trailing_dims: int = 1,
        stream: str = "default",
    ) -> torch.Tensor | None:
        """Return the flattened canonical valid-token mask for ``tensor``."""
        ids = self._aligned_token_ids(
            tensor,
            trailing_dims=trailing_dims,
            stream=stream,
        )
        return None if ids is None else ids >= 0

    def __call__(self, module, args, output):
        """Forward-hook entry point — capture and accumulate. Implemented by subclasses."""
        raise NotImplementedError

    def _dispatch(self, module, args, output):
        """Registered forward hook: gate on ``enabled`` then delegate to ``__call__``."""
        if not self.enabled or ScoringHook._nested_disable_depth:
            return None
        return self(module, args, output)

    def step_iteration(self) -> None:
        """Cross-rank synchronization point, called by the recipe after each batch.

        Additive scorers reduce only once at :meth:`finalize`, so this is a no-op for
        them. Stateful scorers (e.g. the iterative FFN scorer) override it to reduce the
        per-iteration accumulation across the data-partition group and advance their
        greedy state in lock-step on every rank.
        """

    def finalize(self) -> dict:
        """Return the per-target score dict (gathered + reduced). Implemented by subclasses.

        The returned dict mirrors the legacy hook's ``to_dict()`` keys so the
        downstream pruning step consumes it unchanged (e.g. ``{"score": ...}``).
        """
        raise NotImplementedError

    def checkpoint_state(self) -> dict:
        """Return declared rank-local additive state needed for exact resume."""
        state = {}
        for name in self.checkpoint_tensor_names:
            value = getattr(self, name)
            state[name] = value.detach().cpu() if value is not None else None
        for name in self.checkpoint_scalar_names:
            state[name] = getattr(self, name)
        return state

    def load_checkpoint_state(self, state: dict) -> None:
        """Restore declared state produced by :meth:`checkpoint_state`."""
        expected = {*self.checkpoint_tensor_names, *self.checkpoint_scalar_names}
        if set(state) != expected:
            if state or expected:
                raise RuntimeError(
                    f"{type(self).__name__} resume keys={sorted(state)}, "
                    f"expected={sorted(expected)}"
                )
            return
        try:
            device = next(self.module.parameters()).device
        except StopIteration:
            device = None
        for name in self.checkpoint_tensor_names:
            saved = state[name]
            current = getattr(self, name)
            if saved is None:
                setattr(self, name, None)
            elif current is not None:
                if tuple(saved.shape) != tuple(current.shape):
                    raise RuntimeError(
                        f"{type(self).__name__}.{name} resume shape "
                        f"{tuple(saved.shape)} != {tuple(current.shape)}"
                    )
                current.copy_(saved.to(device=current.device, dtype=current.dtype))
            else:
                setattr(self, name, saved.to(device=device))
        for name in self.checkpoint_scalar_names:
            setattr(self, name, state[name])

    def register(self):
        """Register this hook as a forward hook on its module and return the handle."""
        if self._handles:
            return self._handles[0]
        self._handle = self.module.register_forward_hook(self._dispatch)
        self._handles.append(self._handle)
        return self._handle

    def _register_handle(self, handle):
        """Track an auxiliary hook owned by a multi-observation scorer."""
        self._handles.append(handle)
        if self._handle is None:
            self._handle = handle
        return handle

    def remove(self) -> None:
        """Remove the registered forward hook, if any."""
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._handle = None

    @classmethod
    @contextmanager
    def nested_forward_disabled(cls):
        """Temporarily disable all Puzzletron scorers during scorer-owned re-forwards.

        Some MoE signals intentionally run nested forwards with modified routing
        decisions. Those forwards should not pollute unrelated hooks registered in
        the same unified activation pass, such as shared-expert or Mamba hooks.
        """
        ScoringHook._nested_disable_depth += 1
        try:
            yield
        finally:
            ScoringHook._nested_disable_depth -= 1
