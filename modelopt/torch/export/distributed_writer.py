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

"""Pieces of a no-gather distributed HF checkpoint writer.

Every tensor is sent to exactly one owner rank over batched point-to-point sends and
written from there, so no rank ever materializes the whole model and no unit is
replicated across ranks.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def _nbytes(v) -> int:
    """Byte size. DTensor.numel() is GLOBAL, so this is consistent across ranks."""
    return int(v.numel() * v.element_size())


def _is_full_world_shard0(v, world: int, _mesh_memo: dict | None = None) -> bool:
    """True when ``v`` is sharded on dim 0 over a 1-D mesh that is exactly global ranks ``0..world-1``.

    Restricting the point-to-point fast path to that case keeps the peer in ``P2POp`` unambiguous --
    a plain global rank on the default process group, with no mesh-local/global translation to get
    wrong. Any other layout is rejected by :func:`_materialize_owned` rather than routed through a
    slower path, so no untested transport can silently produce a mis-assembled tensor.
    ``_mesh_memo`` keeps the answer per mesh, since FSDP2 gives every parameter the same mesh.
    """
    if len(v.placements) != 1 or not v.placements[0].is_shard(0):
        return False
    mesh = v.device_mesh
    if mesh.ndim != 1:
        return False
    if _mesh_memo is None:
        return mesh.mesh.flatten().tolist() == list(range(world))
    key = id(mesh)
    if key not in _mesh_memo:
        _mesh_memo[key] = mesh.mesh.flatten().tolist() == list(range(world))
    return _mesh_memo[key]


def _run_to_completion(ops: list) -> None:
    """Run the point-to-point ``ops`` and wait for all of them, then empty ``ops``.

    Emptying the list frees the send buffers, which the ops hold, before the caller assembles.
    """
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()
    ops.clear()


# Keys per round of transfers: a round's op count does not grow with its keys.
_KEYS_PER_ROUND = 4096


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _gather_shard0_to_owners(
    local_sd: dict, keys: list, owner: dict, all_shapes: list, rank: int, world: int
) -> None:
    """Gather each dim-0-sharded DTensor to its owner and drop it elsewhere, in place.

    What a rank owes a peer goes in one flat buffer per (peer, dtype), which both sides build from
    the same sorted keys and owner map, so every send meets a receive of its length.
    ``all_shapes[r][k]`` is the shape of rank ``r``'s shard of ``k``; the caller checked they tile.
    """
    for base in range(0, len(keys), _KEYS_PER_ROUND):
        _transfer_round(
            local_sd, keys[base : base + _KEYS_PER_ROUND], owner, all_shapes, rank, world
        )


def _transfer_round(
    local_sd: dict, part: list, owner: dict, all_shapes: list, rank: int, world: int
) -> None:
    """One round of :func:`_gather_shard0_to_owners`, whose buffers are freed when it returns."""
    # (peer, dtype) -> keys. A rank sends its shard of every key the peer owns; the owner
    # receives, from every other rank, its shard of every key this rank owns. Both sides build
    # the identical grouping from shared state. Empty shards are skipped on both sides.
    send_groups: dict = {}
    recv_groups: dict = {}
    for k in part:
        dt = local_sd[k].dtype
        o = owner[k]
        if o != rank:
            if _numel(all_shapes[rank][k]):
                send_groups.setdefault((o, dt), []).append(k)
        else:
            for r in range(world):
                if r != rank and _numel(all_shapes[r][k]):
                    recv_groups.setdefault((r, dt), []).append(k)

    ops: list = []
    pending: list = []
    for peer, dt in sorted(send_groups, key=lambda kd: (kd[0], str(kd[1]))):
        ks = sorted(send_groups[(peer, dt)])
        # Packed straight into its op: no name here then holds the copy past the transfer.
        ops.append(
            dist.P2POp(
                dist.isend, torch.cat([local_sd[k].to_local().reshape(-1) for k in ks]), peer
            )
        )
    for srcr, dt in sorted(recv_groups, key=lambda kd: (kd[0], str(kd[1]))):
        ks = sorted(recv_groups[(srcr, dt)])
        total = sum(_numel(all_shapes[srcr][k]) for k in ks)
        buf = torch.empty(total, dtype=dt, device=local_sd[ks[0]].to_local().device)
        pending.append((srcr, ks, buf))
        ops.append(dist.P2POp(dist.irecv, buf, srcr))

    _run_to_completion(ops)

    # Unpack each peer's flat buffer back into per-key views, in the same order it was packed.
    parts_by_key: dict = {}
    for srcr, ks, buf in pending:
        off = 0
        for k in ks:
            shape = all_shapes[srcr][k]
            n = _numel(shape)
            parts_by_key.setdefault(k, {})[srcr] = buf[off : off + n].view(shape)
            off += n

    # Rank order is mesh order (checked by _is_full_world_shard0), so concatenating the parts in
    # rank order reproduces exactly what full_tensor() would have built.
    for k in part:
        if owner[k] != rank:
            del local_sd[k]
            continue
        got = parts_by_key.get(k, {})
        pieces = []
        for r in range(world):
            if r == rank:
                pieces.append(local_sd[k].to_local())
            elif r in got:
                pieces.append(got[r])
        local_sd[k] = torch.cat(pieces, dim=0)


def _materialize_owned(
    local_sd: dict, expert_key_set: set, rank: int, world: int, together: list = ()
) -> None:
    """Make every tensor in ``local_sd`` whole and owned by exactly one rank, in place.

    Experts must come whole, plain and disjoint per rank, as the caller splits them; one still a
    DTensor is refused. Each dense key goes to one owner, largest first onto the lightest rank
    (seeded with each rank's expert bytes), and keys this rank does not own are dropped. Each group
    in ``together`` (the same on every rank, which is checked) goes to one owner as a unit, with any
    group it shares a key with, since the reverse conversion merges a group only where it is whole.
    Afterwards nothing is a DTensor or partial. On top of the shards it is handed, which the model
    keeps, a rank holds at most about twice its owned set: the shards it receives and the tensors
    assembled from them.
    """
    expert_keys = sorted(k for k in local_sd if k in expert_key_set)
    dense_keys = sorted(k for k in local_sd if k not in expert_key_set)
    _mesh_memo: dict = {}

    # Placements are local metadata: classify first, so the shard shapes ride in the one metadata
    # collective below, which also catches a rank that classified differently.
    replicated, shard0, unsupported, plain, empty0 = [], [], [], [], []
    for k in dense_keys:
        v = local_sd[k]
        if not isinstance(v, DTensor):
            plain.append(k)
        elif all(p.is_replicate() for p in v.placements):
            replicated.append(k)
        elif _is_full_world_shard0(v, world, _mesh_memo):
            # An empty global tensor cannot be reassembled: FSDP2 still hands one rank a padded
            # 1-row chunk, which concatenating would turn into a row. Its owner creates it instead.
            (empty0 if v.numel() == 0 else shard0).append(k)
        else:
            unsupported.append((k, str(v.placements), v.device_mesh.ndim))

    # A key outside ``together`` is a unit of its own, which keeps the order of a model without merge
    # groups. Groups sharing a key are one unit, whatever their order: split, one of them would be a
    # merge its owner cannot complete. Expert keys stay out: a rank holds its experts whole already.
    unit_of = {k: (k,) for k in dense_keys}
    for group in together:
        unit = tuple(sorted({j for k in group if k in unit_of for j in unit_of[k]}))
        for k in unit:
            unit_of[k] = unit
    merged_units = sorted({u for u in unit_of.values() if len(u) > 1})

    # One collective for all the metadata the steps below need. Every refusal below is decided from
    # it, so it is the same on every rank: one that raised alone would leave its peers waiting.
    dense_sizes = {k: _nbytes(local_sd[k]) for k in dense_keys}
    dense_kinds = {k: (str(local_sd[k].dtype), tuple(local_sd[k].shape)) for k in dense_keys}
    my_meta = (
        dense_keys,
        sum(_nbytes(local_sd[k]) for k in expert_keys),
        {k: tuple(local_sd[k].to_local().shape) for k in shard0},
        {k: tuple(local_sd[k].shape) for k in shard0},
        [(k, pl, f"mesh.ndim={nd}") for k, pl, nd in unsupported[:3]],
        {k: (str(local_sd[k].dtype), local_sd[k].to_local().device.type) for k in shard0},
        [k for k in expert_keys if isinstance(local_sd[k], DTensor)][:3],
        dense_kinds,
        merged_units,
    )
    meta: list = [None] * world
    dist.all_gather_object(meta, my_meta)
    # Everything below is keyed on the dense key list, so a rank that disagreed about it would HANG
    # in a collective rather than fail. Check once, up front, and make that a diagnosable error.
    if any(m[0] != dense_keys for m in meta):
        mine = set(dense_keys)
        diff = {r: sorted(set(m[0]) ^ mine)[:5] for r, m in enumerate(meta) if m[0] != dense_keys}
        raise RuntimeError(
            f"rank {rank}: dense key sets differ across ranks, so the export would deadlock; "
            f"sample divergences per rank: {diff}"
        )
    # Reject rather than silently taking a slower, untested route. The point-to-point transport
    # covers replicated, plain, and dim-0 shards on a 1-D full-world mesh; anything else (2-D mesh,
    # non-dim-0 shard, partial placement) has no correct handling here. Raising names the offending
    # keys, which is recoverable; emitting a mis-assembled tensor would not be.
    unsupported_by_rank = {r: m[4] for r, m in enumerate(meta) if m[4]}
    if unsupported_by_rank:
        raise RuntimeError(
            f"rank {rank}: tensor(s) have a placement the no-gather export cannot materialize; "
            f"first few per rank: {unsupported_by_rank}"
        )
    # Experts skip the classification above and go to the write as they are, which cannot serialize
    # a DTensor: refuse one here, before any rank moves or writes anything.
    sharded_experts = {r: m[6] for r, m in enumerate(meta) if m[6]}
    if sharded_experts:
        raise RuntimeError(
            f"rank {rank}: expert tensor(s) are still DTensors, but each rank writes its experts as "
            f"it holds them; first few per rank: {sharded_experts}"
        )
    expert_bytes = [m[1] for m in meta]
    all_shapes = [m[2] for m in meta]
    global_shapes = [m[3] for m in meta]
    shard_kinds = [m[5] for m in meta]

    # all_shapes[r]'s keys are rank r's dim-0 classification: name a rank that disagrees, which would
    # otherwise surface as a bare KeyError below.
    _mine = set(all_shapes[rank])
    _bad = {
        r: sorted(set(all_shapes[r]) ^ _mine)[:3]
        for r in range(world)
        if set(all_shapes[r]) != _mine
    }
    if _bad:
        raise RuntimeError(
            f"rank {rank}: ranks disagree on which tensors are dim-0 sharded, so the export cannot "
            f"assemble them; per-rank symmetric differences (first few): {dict(list(_bad.items())[:3])}"
        )

    # Concatenating the shards rebuilds the tensor only if they tile it: dim-0 extents summing to
    # the global dim 0, the other dims equal. A padded chunk from uneven sharding would concatenate
    # into a wrong tensor rather than fail, so check, against the gathered global shapes.
    untiled, mixed = [], []
    for k in shard0:
        # A shard is sent in the sender's dtype and received in the owner's: ranks that disagree
        # would mismatch the transfer's byte counts and hang in it. So would its device.
        if len({shard_kinds[r][k] for r in range(world)}) > 1:
            mixed.append((k, [shard_kinds[r][k] for r in range(world)][:3]))
        gshapes = {global_shapes[r][k] for r in range(world)}
        gshape = next(iter(gshapes))
        shard_shapes = [all_shapes[r][k] for r in range(world)]
        if len(gshapes) != 1 or not (
            sum(s[0] for s in shard_shapes) == gshape[0]
            and all(tuple(s[1:]) == gshape[1:] for s in shard_shapes)
        ):
            untiled.append((k, sorted(gshapes), shard_shapes[:3]))
    if mixed:
        raise RuntimeError(
            f"rank {rank}: {len(mixed)} dim-0 sharded tensor(s) have a different dtype or device on "
            f"different ranks, so their shards cannot be moved to one rank; first few: {mixed[:2]}"
        )
    # A peer's shards of one dtype travel in one buffer, and a round's transfers in one batch: neither
    # spans devices, as a CPU-offloaded shard beside one moved to the accelerator would need.
    devices = sorted({shard_kinds[rank][k][1] for k in shard0})
    if len(devices) > 1:
        raise RuntimeError(
            f"rank {rank}: dim-0 sharded tensors are on more than one device type ({devices}), "
            "which the transport cannot move in one batch; put them on one device first."
        )
    if untiled:
        raise RuntimeError(
            f"rank {rank}: {len(untiled)} dim-0 sharded tensor(s) do not tile their global shape, "
            f"so concatenating the local shards would not reconstruct them (to_local() may be "
            f"returning padded chunks, or the ranks disagree on the global shape); first few: "
            f"{untiled[:2]}"
        )
    # Every rank assigns the owners below from these sizes, so ranks that disagree on one (a plain
    # buffer left in another dtype on one rank) would disagree on who keeps which tensor: one could
    # be dropped everywhere or written twice, or a shard sent to a rank not waiting for it. A dtype or
    # shape that differs at the same size would leave the owner's version in the checkpoint.
    reshaped = {
        r: sorted(k for k in dense_keys if m[7][k] != dense_kinds[k])[:3]
        for r, m in enumerate(meta)
        if m[7] != dense_kinds
    }
    if reshaped:
        raise RuntimeError(
            f"rank {rank}: dense tensor(s) have a different dtype or shape on different ranks, so "
            f"the ranks would not agree on which of them writes each; first few per rank: {reshaped}"
        )
    # So would ranks that group the merged keys differently.
    regrouped = {
        r: sorted(set(map(tuple, m[8])) ^ set(merged_units))[:3]
        for r, m in enumerate(meta)
        if m[8] != merged_units
    }
    if regrouped:
        raise RuntimeError(
            f"rank {rank}: ranks group the keys a merge needs on one rank differently, so the ranks "
            "would not agree on which of them writes each; the units that differ from this rank's, "
            f"first few per rank: {regrouped}"
        )

    # Deterministic on every rank: the sort and the argmin both break ties explicitly.
    unit_bytes = {u: sum(dense_sizes[k] for k in u) for u in set(unit_of.values())}
    loads = list(expert_bytes)
    owner: dict = {}
    for unit in sorted(unit_bytes, key=lambda u: (-unit_bytes[u], u)):
        r = min(range(world), key=lambda i: (loads[i], i))
        for k in unit:
            owner[k] = r
        loads[r] += unit_bytes[unit]

    # Replicated and plain values are already whole on every rank -- the owner just unwraps, the
    # rest just forget. No communication at all.
    for k in empty0:
        if owner[k] == rank:
            _v = local_sd[k]
            local_sd[k] = torch.empty(tuple(_v.shape), dtype=_v.dtype, device=_v.to_local().device)
        else:
            del local_sd[k]
    for k in replicated:
        if owner[k] == rank:
            local_sd[k] = local_sd[k].to_local()
        else:
            del local_sd[k]
    for k in plain:
        if owner[k] != rank:
            del local_sd[k]

    if shard0:
        _gather_shard0_to_owners(local_sd, shard0, owner, all_shapes, rank, world)
