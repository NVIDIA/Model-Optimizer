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

"""Pieces of a no-gather distributed HF checkpoint writer.

Every tensor is sent to exactly one owner rank over batched point-to-point sends and
written from there, so no rank ever materializes the whole model and no unit is
replicated across ranks.
"""

import json
import warnings
from pathlib import Path

import torch
import torch.nn as nn


def _size_to_bytes(size: "str | int") -> int:
    """Parse an HF-style shard-size string (``"5GB"``, ``"500MB"``, ``"1GiB"``) to bytes.

    Matches transformers' decimal convention (GB == 10**9). Bare ints pass through.
    """
    if isinstance(size, int):
        return size
    s = str(size).strip().upper()
    units = {
        "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40,
        "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
    }  # fmt: skip
    for unit in ("KIB", "MIB", "GIB", "TIB", "KB", "MB", "GB", "TB"):
        if s.endswith(unit):
            return int(float(s[: -len(unit)]) * units[unit])
    return int(float(s))


def _even_bins(keys: list, sizes: dict, n: int, max_bytes: int) -> list:
    """Partition ``keys`` into ~``n`` EVEN-sized bins (by bytes), each <= ``max_bytes`` where possible.

    Longest-processing-time greedy (largest item -> lightest bin) balances the bins, so N files come out
    ~``total/N`` each rather than (N-1) full files + a small remainder -- ``max_shard_size`` is an UPPER
    BOUND, not the target. ``n`` counts the bins for the tensors within the bound, and is bumped (up to
    one file per tensor) if a lumpy item pushes a bin over it. A tensor larger than ``max_bytes`` gets
    a file of its own on top, as in HF, and the rest still pack around it.
    Returns the non-empty bins (``list[list[key]]``).
    """
    oversized = [[k] for k in keys if sizes[k] > max_bytes]
    fits = [k for k in keys if sizes[k] <= max_bytes]
    n = max(1, min(n, len(fits))) if fits else 1
    ordered = sorted(fits, key=lambda k: -sizes[k])
    while True:
        bins: list = [[] for _ in range(n)]
        loads = [0] * n
        for k in ordered:
            j = min(range(n), key=lambda i: loads[i])
            bins[j].append(k)
            loads[j] += sizes[k]
        if n >= len(fits) or max(loads) <= max_bytes:
            return oversized + [b for b in bins if b]
        n += 1


def _nbytes(v) -> int:
    """Byte size. DTensor.numel() is GLOBAL, so this is consistent across ranks."""
    return int(v.numel() * v.element_size())


def _is_full_world_shard0(v, world: int, _mesh_memo: dict | None = None) -> bool:
    """True when ``v`` is sharded on dim 0 over a 1-D mesh that is exactly global ranks ``0..world-1``.

    Restricting the point-to-point fast path to that case keeps the peer in ``P2POp`` unambiguous --
    a plain global rank on the default process group, with no mesh-local/global translation to get
    wrong. Any other layout is rejected by :func:`_materialize_owned` rather than routed through a
    slower path, so no untested transport can silently produce a mis-assembled tensor.

    ``_mesh_memo`` caches the mesh-layout answer by mesh identity. Reading ``mesh.mesh`` back to the
    host is a device sync, and FSDP2 gives every parameter the SAME mesh object, so without the memo
    a few hundred keys cost a few hundred redundant syncs.
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


def _gather_shard0_to_owners(
    local_sd: dict, keys: list, owner: dict, all_shapes: list, rank: int, world: int
) -> None:
    """Gather each dim-0-sharded DTensor to its owner and drop it elsewhere, in place.

    ``full_tensor()`` is an ALL-gather: it lands a full copy on every rank, so moving N bytes costs
    ``world x N`` and every rank allocates the whole tensor even though one rank keeps it. Here each
    shard travels once, to its owner only.

    Transfers are bucketed BY PEER rather than per key: everything this rank owes a given peer is
    concatenated into one flat buffer per (peer, dtype) and sent as a single operation. One op per
    (key, peer) pair does not survive leaving NVLink -- at 235B that was ~12.7k operations and took
    660s, with the payload only a few GB. Bucketing makes it O(world x dtypes) operations.

    ``all_shapes[r][k]`` is rank ``r``'s local dim-0 extent for key ``k``; the caller has already
    checked that the shards tile the global tensor exactly.

    Deadlock safety: both sides derive their buckets from the same globally-sorted key list and the
    same owner map, so every send has a matching receive of exactly the same length.
    """
    import torch.distributed as dist

    def _numel(shape):
        n = 1
        for s in shape:
            n *= s
        return n

    device = local_sd[keys[0]].to_local().device

    # Chunk the key list so the in-flight receive buffers stay bounded. Within a chunk the op count
    # is O(world x dtypes) regardless of how many keys it holds, so chunks can be large.
    chunk = 4096
    for base in range(0, len(keys), chunk):
        part = keys[base : base + chunk]

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
        keepalive: list = []
        pending: list = []
        for peer, dt in sorted(send_groups, key=lambda kd: (kd[0], str(kd[1]))):
            ks = sorted(send_groups[(peer, dt)])
            buf = torch.cat([local_sd[k].to_local().reshape(-1) for k in ks])
            keepalive.append(buf)  # must outlive the wait
            ops.append(dist.P2POp(dist.isend, buf, peer))
        for srcr, dt in sorted(recv_groups, key=lambda kd: (kd[0], str(kd[1]))):
            ks = sorted(recv_groups[(srcr, dt)])
            total = sum(_numel(all_shapes[srcr][k]) for k in ks)
            buf = torch.empty(total, dtype=dt, device=device)
            pending.append((srcr, ks, buf))
            ops.append(dist.P2POp(dist.irecv, buf, srcr))

        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()

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
        del keepalive, pending, parts_by_key


def _materialize_owned(local_sd: dict, expert_key_set: set, rank: int, world: int) -> None:
    """Make every tensor in ``local_sd`` WHOLE and owned by exactly one rank, in place.

    Experts are already disjoint per rank (each rank split its own shard). Every dense fqn is
    assigned to ONE rank -- largest-first onto the currently-lightest rank, seeded with each rank's
    expert bytes so the per-rank totals level out -- and then moved there by the cheapest transport
    its layout allows (see :func:`_gather_shard0_to_owners`). Keys this rank does not own are dropped.

    This is the step that lets the reverse conversion run: afterwards nothing is a DTensor and no
    tensor is partial, so transformers' conversion ops see the whole tensors they were written for.
    It is NOT a gather to a single rank -- peak memory is this rank's owned set, not the model.
    """
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    expert_keys = sorted(k for k in local_sd if k in expert_key_set)
    dense_keys = sorted(k for k in local_sd if k not in expert_key_set)
    _mesh_memo: dict = {}

    # Classify FIRST: placements are local DTensor metadata, so this needs no communication, and
    # doing it up front means the shard shapes can ride along in the single metadata collective
    # below. Every rank sees the same placements and so agrees on the buckets.
    replicated, shard0, unsupported, plain, empty0 = [], [], [], [], []
    for k in dense_keys:
        v = local_sd[k]
        if not isinstance(v, DTensor):
            plain.append(k)
        elif v.numel() == 0:
            # A zero-element global tensor (e.g. a projection that is absent in some layers, so its
            # dim 0 is 0) cannot be reassembled from shards: FSDP2 still hands one rank a PADDED
            # 1-row chunk while the rest get genuinely empty ones, so concatenating would fabricate
            # a row the global tensor does not have. There is no data to move -- the owner emits the
            # empty tensor directly.
            empty0.append(k)
        elif all(p.is_replicate() for p in v.placements):
            replicated.append(k)
        elif _is_full_world_shard0(v, world, _mesh_memo):
            shard0.append(k)
        else:
            unsupported.append((k, str(v.placements), v.device_mesh.ndim))

    # Reject rather than silently taking a slower, untested route. The point-to-point transport
    # covers replicated, plain, and dim-0 shards on a 1-D full-world mesh; anything else (2-D mesh,
    # non-dim-0 shard, partial placement) has no correct handling here. Raising names the offending
    # keys, which is recoverable; emitting a mis-assembled tensor would not be.
    if unsupported:
        raise RuntimeError(
            f"rank {rank}: {len(unsupported)} tensor(s) have a placement the no-gather export "
            f"cannot materialize; first few: "
            f"{[(k, pl, f'mesh.ndim={nd}') for k, pl, nd in unsupported[:3]]}"
        )

    # ONE collective for all the metadata: the key list (to catch divergence), this rank's expert
    # byte count (to seed load balancing), and its local dim-0 extents (to size receive buffers).
    # These were three separate all_gather_object round trips, each pickling a ~900-entry payload.
    my_meta = (
        dense_keys,
        sum(_nbytes(local_sd[k]) for k in expert_keys),
        {k: tuple(local_sd[k].to_local().shape) for k in shard0},
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
    expert_bytes = [m[1] for m in meta]
    all_shapes = [m[2] for m in meta]

    dense_sizes = {k: _nbytes(local_sd[k]) for k in dense_keys}
    # Deterministic on every rank: the sort and the argmin both break ties explicitly.
    loads = list(expert_bytes)
    owner: dict = {}
    for k in sorted(dense_keys, key=lambda k: (-dense_sizes[k], k)):
        r = min(range(world), key=lambda i: (loads[i], i))
        owner[k] = r
        loads[r] += dense_sizes[k]

    # Replicated and plain values are already whole on every rank -- the owner just unwraps, the
    # rest just forget. No communication at all.
    for k in empty0:
        if owner[k] == rank:
            _v = local_sd[k]
            local_sd[k] = torch.empty(tuple(_v.shape), dtype=_v.dtype, device="cpu")
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

    # Concatenating local shards reproduces the global tensor only if they tile it exactly: dim-0
    # extents summing to the global dim 0, every other dim equal. Uneven sharding can leave
    # to_local() returning a PADDED chunk, which would concatenate into a wrong (too large,
    # misaligned) tensor rather than failing -- so verify instead of assuming. Derived from the
    # same gathered shapes on every rank, so the split is agreed.
    # all_shapes[r] is keyed by rank r's shard0 bucket, so its key SET is that rank's classification.
    # Those buckets come from per-rank DTensor metadata, and a rank that disagrees makes the loop
    # below raise a bare KeyError from deep inside the writer -- which says nothing about the real
    # mistake. (Seen for real: a packed-weight rewrap re-inferred the global shape as local x world,
    # so under uneven sharding each rank computed a different global shape and classified
    # differently.) Name the divergence instead.
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

    untiled = []
    for k in shard0:
        gshape = tuple(local_sd[k].shape)
        shard_shapes = [all_shapes[r][k] for r in range(world)]
        if not (
            sum(s[0] for s in shard_shapes) == gshape[0]
            and all(tuple(s[1:]) == gshape[1:] for s in shard_shapes)
        ):
            untiled.append((k, gshape, shard_shapes[:3]))
    if untiled:
        raise RuntimeError(
            f"rank {rank}: {len(untiled)} dim-0 sharded tensor(s) do not tile their global shape, "
            f"so concatenating the local shards would not reconstruct them (to_local() may be "
            f"returning padded chunks); first few: {untiled[:2]}"
        )
    if shard0:
        _gather_shard0_to_owners(local_sd, shard0, owner, all_shapes, rank, world)


def _revert_whole_sd(model: nn.Module, local_sd: dict) -> bool:
    """Reverse the transformers conversion mapping on this rank's whole tensors, in place.

    Safe to run per-rank on a disjoint subset: renames are per-key, and a split's outputs stay on
    the rank that owned the input. Every value is a whole plain tensor by now, so no placement
    handling is needed. Best-effort and atomic, mirroring the single-process path -- on any
    failure the in-memory names are kept for the whole dict rather than half-applied.
    """
    from .quant_aware_conversion import revert_weight_conversion_quant_aware

    try:
        reverted = revert_weight_conversion_quant_aware(model, local_sd)
    except Exception as exc:
        warnings.warn(
            f"Quant-aware reverse weight conversion skipped ({exc}); exported tensor "
            "names may not match the original HF hub checkpoint."
        )
        return False
    # revert_weight_conversion_quant_aware returns the INPUT dict itself when the model has no
    # reverse rules (a dense model with no conversion mapping). Clearing then updating would then
    # read from the dict just emptied and wipe the state dict, so only rebuild on a fresh object.
    if reverted is not local_sd:
        local_sd.clear()
        local_sd.update(reverted)
    return True


def _in_rank_order(values: list[list]) -> list:
    """``all_reduce``'s combine that keeps every rank's value: each rank's own is wrapped in a list."""
    return [value for (value,) in values]


def _write_even_shards(
    local_sd: dict, export_dir: "str | Path", max_shard_size: "str | int", rank: int, world: int
) -> None:
    """Write this rank's whole tensors directly as final HF safetensors; rank 0 writes the index.

    Every tensor is whole on exactly one rank, so there is no consolidation step. This rank's set is
    split into even shards, numbered into a global range (prefix-sum of per-rank file counts), and
    written with ``safetensors.save_file``. Must run AFTER the reverse conversion -- the reverse
    renames keys and splits tensors, so the sizes and names bin-packed here have to be the final ones.

    Each step a rank can fail alone runs inside :func:`~modelopt.torch.utils.distributed.all_reduce`,
    which raises on every rank when one fails: a rank leaving early would otherwise meet its peers
    in a different collective, and they would hang or raise something unrelated.
    """
    import math

    from safetensors.torch import save_file

    from modelopt.torch.utils.distributed import all_reduce

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = _size_to_bytes(max_shard_size)

    write_keys = sorted(local_sd)
    sizes = {k: _nbytes(local_sd[k]) for k in write_keys}
    # Tensors over the bound get files of their own, so only the rest decide how many to aim for.
    fits = sum(size for size in sizes.values() if size <= max_bytes)
    my_files = _even_bins(write_keys, sizes, max(1, math.ceil(fits / max_bytes)), max_bytes)
    counts = all_reduce("shard counts", lambda: [len(my_files)], _in_rank_order)
    n_total = sum(counts)
    if n_total == 0:
        raise RuntimeError(
            "distributed export produced no shards on any rank: every tensor was dropped "
            "before the write (state dict emptied upstream)"
        )
    base = sum(counts[:rank])

    def write_mine() -> tuple[dict, int]:
        weight_map: dict = {}
        my_bytes = 0
        for i, keys in enumerate(my_files):
            fname = f"model-{base + i + 1:05d}-of-{n_total:05d}.safetensors"
            tensors = {k: local_sd[k].detach().to("cpu").contiguous() for k in keys}
            save_file(tensors, str(export_dir / fname), metadata={"format": "pt"})
            for k in keys:
                weight_map[k] = fname
                # Once a tensor is on disk this rank has no further use for it. Dropping it here
                # keeps peak memory at roughly one shard above the owned set instead of holding all.
                del local_sd[k]
            my_bytes += sum(_nbytes(t) for t in tensors.values())
            del tensors
        return weight_map, my_bytes

    # One collective, not two: at this point each rank has already finished writing, so the latency
    # of these round trips is pure overhead on the critical path.
    gathered = all_reduce("shards written", lambda: [write_mine()], _in_rank_order)
    full_map: dict = {}
    total_bytes = 0
    for m, b in gathered:
        full_map.update(m)
        total_bytes += b
    # The reverse conversion checks for rename collisions, but only within one rank's slice.
    # A collision ACROSS ranks would otherwise be silently swallowed by this dict update and
    # ship a checkpoint that is quietly missing a tensor. Every rank checks, so all of them raise.
    n_written = sum(len(m) for m, _ in gathered)
    if n_written != len(full_map):
        raise RuntimeError(
            f"tensor name collision across ranks: {n_written} tensors written but only "
            f"{len(full_map)} distinct names in the index"
        )

    def write_index() -> bool:
        if rank == 0:
            index = {"metadata": {"total_size": total_bytes}, "weight_map": full_map}
            with open(export_dir / "model.safetensors.index.json", "w") as f:
                json.dump(index, f, indent=2)
        return True

    # Also the closing barrier: no rank returns before the index is on disk.
    all_reduce("index written", write_index, all)
