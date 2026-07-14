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

"""Parallelism-aware reduction primitives for AutoModel activation scoring.

A scored quantity (e.g. per-channel FFN importance, per-KV-head importance,
per-expert importance) lives on two kinds of axes, reduced differently:

* **scored axis** — the dimension we rank (FFN intermediate channels, KV heads,
  MoE experts). When sharded across **tensor parallel** (FFN/attention) or
  **expert parallel** (MoE experts) it must be **GATHERED / concatenated in
  rank order** — never summed.
* **token axis** — batch / sequence positions. Per-token contributions are
  summed locally and then **SUM all-reduced across the data-partition group**
  (``dp_cp``: data-parallel × FSDP × context-parallel).

Composing a token SUM-reduce with a scored-axis GATHER yields a final per-element
vector that is identical on every rank and **independent of the parallel layout
or node count**. This module provides the primitives; each ported hook composes
them (see ``hooks/`` in later milestones).

Sequence parallel (SP) needs no special handling here: it shards the sequence
only *outside* the MLP/attention output projection, so at the hook point (the
input to ``down_proj`` / ``o_proj``) the sequence is full and only the feature
dim is TP-sharded — covered by the scored-axis gather.

The module depends only on ``torch`` (``torch.distributed`` + ``DTensor``) so it
can be unit-tested with a plain gloo/CPU process group, without NeMo or a real
model.
"""

from dataclasses import dataclass

import torch
import torch.distributed as c10d
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor, Shard

__all__ = [
    "MeshGroups",
    "finalize_additive",
    "full_weight",
    "gather_scored_axis",
    "is_writer",
    "reduce_token_sum",
    "to_local_with_feature_group",
    "writer_shard_id",
]


def _group_size(group: ProcessGroup | None) -> int:
    """Number of ranks in ``group`` (1 when ``group`` is ``None``)."""
    return c10d.get_world_size(group) if group is not None else 1


def _group_rank(group: ProcessGroup | None) -> int:
    """This process's rank within ``group`` (0 when ``group`` is ``None``)."""
    return c10d.get_rank(group) if group is not None else 0


def _first_axis_group(mesh, names: tuple[str, ...]) -> ProcessGroup | None:
    """Return the 1-D process group for the first of ``names`` present in ``mesh``.

    Handles both native mesh dims (``mesh[name]``) and flattened axes such as
    ``"dp_cp"`` that NeMo creates via ``DeviceMesh._flatten`` (looked up in the
    root mesh's ``_flatten_mapping``). Returns ``None`` when none are found, which
    makes the corresponding reduction a no-op (correct for a size-1 axis).
    """
    if mesh is None:
        return None
    dim_names = tuple(getattr(mesh, "mesh_dim_names", None) or ())
    root = mesh._get_root_mesh() if hasattr(mesh, "_get_root_mesh") else None
    flatten_mapping = getattr(root, "_flatten_mapping", None) if root is not None else None
    for name in names:
        sub = None
        if name in dim_names:
            sub = mesh[name]
        elif isinstance(flatten_mapping, dict) and name in flatten_mapping:
            sub = flatten_mapping[name]
        if sub is not None:
            try:
                return sub.get_group()
            except Exception:
                return None
    return None


@dataclass
class MeshGroups:
    """Process groups for the four axis classes that matter to scoring.

    A ``None`` group means "that axis is size 1 / absent"; reductions over it are
    no-ops. Construct directly from process groups (handy for tests) or resolve
    from a live NeMo ``DeviceMesh`` via :meth:`from_device_mesh`.

    Attributes:
        token_group: ranks that shard the **token** axis (``dp_cp``) — SUM-reduced.
        cp_group: context-parallel ranks that shard only the sequence axis.
        tp_group: tensor-parallel ranks that shard the scored axis for FFN/attn —
            GATHERED. (Usually derived per-tensor from the DTensor placement via
            :func:`to_local_with_feature_group`; kept here for reference/diagnostics.)
        ep_group: expert-parallel ranks that shard the MoE expert axis — GATHERED.
        pp_group: pipeline-parallel ranks; each PP stage owns disjoint modules and
            writes its own output shard.
    """

    token_group: ProcessGroup | None = None
    cp_group: ProcessGroup | None = None
    tp_group: ProcessGroup | None = None
    ep_group: ProcessGroup | None = None
    pp_group: ProcessGroup | None = None

    @classmethod
    def from_device_mesh(
        cls,
        device_mesh,
        *,
        moe_mesh=None,
        token_axes: tuple[str, ...] = ("dp_cp", "dp_shard_cp", "dp"),
        cp_axes: tuple[str, ...] = ("cp",),
        tp_axes: tuple[str, ...] = ("tp",),
        pp_axes: tuple[str, ...] = ("pp",),
        ep_axes: tuple[str, ...] = ("ep",),
    ) -> "MeshGroups":
        """Resolve groups from a NeMo ``DeviceMesh`` (+ optional ``moe_mesh``).

        ``*_axes`` are tried in order; the first present axis name wins. Defaults
        match NeMo's mesh naming (flattened ``dp_cp`` for the token axis, ``tp``,
        ``pp``, and ``ep`` on the MoE mesh).
        """
        return cls(
            token_group=_first_axis_group(device_mesh, token_axes),
            cp_group=_first_axis_group(device_mesh, cp_axes),
            tp_group=_first_axis_group(device_mesh, tp_axes),
            ep_group=_first_axis_group(moe_mesh if moe_mesh is not None else device_mesh, ep_axes),
            pp_group=_first_axis_group(device_mesh, pp_axes),
        )

    @property
    def token_rank(self) -> int:
        return _group_rank(self.token_group)

    @property
    def tp_rank(self) -> int:
        return _group_rank(self.tp_group)

    @property
    def cp_rank(self) -> int:
        return _group_rank(self.cp_group)

    @property
    def ep_rank(self) -> int:
        return _group_rank(self.ep_group)

    @property
    def pp_rank(self) -> int:
        return _group_rank(self.pp_group)

    @property
    def token_size(self) -> int:
        return _group_size(self.token_group)

    @property
    def tp_size(self) -> int:
        return _group_size(self.tp_group)

    @property
    def cp_size(self) -> int:
        return _group_size(self.cp_group)

    @property
    def ep_size(self) -> int:
        return _group_size(self.ep_group)

    @property
    def pp_size(self) -> int:
        return _group_size(self.pp_group)


def to_local_with_feature_group(
    tensor: torch.Tensor, feature_dim: int = -1
) -> tuple[torch.Tensor, ProcessGroup | None]:
    """Return ``(local_tensor, feature_group)`` for a captured activation.

    For a :class:`DTensor` sharded on ``feature_dim`` (e.g. the input to a
    RowwiseParallel ``down_proj``/``o_proj`` under tensor parallel), returns the
    rank-local shard (token dims left exactly as they are, so any context-parallel
    / data-parallel token sharding stays local for the later token SUM-reduce) and
    the process group across which ``feature_dim`` is sharded. Pass that group to
    :func:`gather_scored_axis` (or :func:`finalize_additive`) to reassemble the
    full scored axis.

    For a replicated DTensor or a plain tensor, returns the local/full tensor and
    ``None`` (no gather needed — the feature axis is already complete locally).

    Note: only the first mesh dim found sharding ``feature_dim`` is returned; the
    feature axis is expected to be sharded across at most one mesh dim (TP).
    """
    if not isinstance(tensor, DTensor):
        return tensor, None
    fdim = feature_dim % tensor.ndim
    feature_group: ProcessGroup | None = None
    for mesh_dim, placement in enumerate(tensor.placements):
        if isinstance(placement, Shard) and placement.dim == fdim:
            feature_group = tensor.device_mesh.get_group(mesh_dim)
            break
    return tensor.to_local(), feature_group


def gather_scored_axis(
    local: torch.Tensor, group: ProcessGroup | None, dim: int = 0
) -> torch.Tensor:
    """All-gather ``local`` across ``group`` and concat along ``dim`` in rank order.

    Reconstructs the full scored axis from contiguous ``Shard(dim)`` shards (the
    layout produced by tensor/expert parallel). Returns ``local`` unchanged when
    ``group`` is ``None`` or size 1.

    Assumes equal shard sizes across the group (true for TP/EP, which require the
    scored dimension to be divisible by the parallel degree).
    """
    size = _group_size(group)
    if size == 1:
        return local
    local = local.contiguous()
    gathered = [torch.empty_like(local) for _ in range(size)]
    c10d.all_gather(gathered, local, group=group)
    return torch.cat(gathered, dim=dim)


def reduce_token_sum(tensor: torch.Tensor, group: ProcessGroup | None) -> torch.Tensor:
    """In-place SUM all-reduce of ``tensor`` across the token-partition ``group``.

    Use for per-channel partial sums and token counts accumulated over a rank's
    local token shard. No-op when ``group`` is ``None`` or size 1. Returns
    ``tensor`` (mutated in place).
    """
    if _group_size(group) == 1:
        return tensor
    c10d.all_reduce(tensor, op=c10d.ReduceOp.SUM, group=group)
    return tensor


def full_weight(weight: torch.Tensor) -> torch.Tensor:
    """Materialize a (possibly :class:`DTensor`) weight to a plain replicated tensor.

    Weights have no token axis, so a full all-gather across all sharded mesh dims
    (``DTensor.full_tensor``) is the simplest correct way to compute, e.g., a
    per-output-channel norm.
    """
    return weight.full_tensor() if isinstance(weight, DTensor) else weight


def finalize_additive(
    local_partial: torch.Tensor,
    *,
    feature_group: ProcessGroup | None,
    groups: "MeshGroups",
    scored_dim: int = 0,
) -> torch.Tensor:
    """Standard additive finalize: SUM across the token group, then GATHER the scored axis.

    ``local_partial`` is a per-(local-scored-element) vector already summed over
    this rank's local tokens. ``feature_group`` is the TP/EP group that shards it
    along ``scored_dim`` — obtained from :func:`to_local_with_feature_group` (TP)
    or :attr:`MeshGroups.ep_group` (experts); ``None`` means no gather.

    Operates on a clone so the caller's accumulator is left intact (safe to call
    for resume/diagnostics). The token reduce runs first on the small per-element
    vector, then the gather assembles the global scored axis; the two groups are
    orthogonal so the order does not affect the result.
    """
    reduced = reduce_token_sum(local_partial.clone(), groups.token_group)
    return gather_scored_axis(reduced, feature_group, dim=scored_dim)


def is_writer(groups: "MeshGroups") -> bool:
    """Single-writer rule for output dedup.

    After gather+reduce, every rank in a PP stage holds identical scores, so only
    one rank per stage should write. We pick ``token==0 and tp==0 and ep==0``;
    each PP stage writes the modules it owns, and the union over PP shards is the
    full model (no duplicate module keys).
    """
    return groups.token_rank == 0 and groups.tp_rank == 0 and groups.ep_rank == 0


def writer_shard_id(groups: "MeshGroups") -> int:
    """Output shard id for the writing rank = its pipeline-parallel stage index.

    0 when pipeline parallel is not used (all modules written by a single shard).
    """
    return groups.pp_rank
