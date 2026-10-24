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

"""No-gather distributed HF checkpoint writer for FSDP2.

Every tensor is sent to exactly one owner rank over batched point-to-point sends and
written from there, so no rank ever materializes the whole model and no unit is
replicated across ranks. Selected by ``MODELOPT_FSDP2_EXPORT=nogather``.
"""

import json
import math
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor

from modelopt.torch.utils.distributed import all_reduce

from .quant_aware_conversion import (
    _QUANT_STATE_SUFFIXES,
    QuantConversionUnsupportedError,
    _build_reverse_rules,
    revert_weight_conversion_quant_aware,
)
from .unified_export_hf_streaming import _parse_shard_size

# A checkpoint's files: its weights, sharded as save_pretrained names them or in one file, and index.
_CHECKPOINT_FILES = re.compile(r"model(-\d{5}-of-\d{5})?\.safetensors(\.index\.json)?")

from .moe_utils import _FUSED_PROJ, _fused_experts_prefixes, _split_local_fused_module


def distributed_save_hf_checkpoint(
    model: nn.Module,
    export_dir: "str | Path",
    maxbound: float,
    kv_cache_format: "str | dict[str, dict[str, str]] | None",
    max_shard_size: "str | int" = "10GB",
    is_modelopt_qlora: bool = False,
    extra_state_dict: "dict[str, torch.Tensor] | None" = None,
) -> None:
    """No-gather distributed HF export.

    Thin wrapper: hf_ptq exports under ``torch.inference_mode()``, so run the entire write with inference mode DISABLED
    via a nested ``inference_mode(False)`` context -- the ``.detach()`` in ``get_model_state_dict`` and the version-
    counted ops in the write path both reject inference tensors. The context form reliably restores normal mode for this
    scope (a decorator did not, under the caller's already-active inference_mode). ``no_grad`` too:
    ``inference_mode(False)`` re-ENABLES autograd, so a bare ``clone()``/``detach()`` on an inference tensor would try
    to set up grad and read its (missing) version counter -- ``no_grad`` avoids that while the clones still land as
    normal tensors because inference mode is off.
    """
    with torch.inference_mode(False), torch.no_grad():
        _distributed_save_hf_checkpoint_impl(
            model,
            export_dir,
            maxbound,
            kv_cache_format,
            max_shard_size,
            is_modelopt_qlora,
            extra_state_dict,
        )


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
    (seeded with each rank's expert bytes), and keys this rank does not own are dropped. Ranks are
    weighed by bytes, not free device memory: a large embedding or ``lm_head`` must fit on its owner.
    Each group in ``together`` (the same on every rank, which is checked) goes to one owner as a
    unit, with any group it shares a key with, since the reverse conversion merges a group only where
    it is whole.
    Afterwards nothing is a DTensor or partial. On top of the shards it is handed, which the model
    keeps, a rank holds in each round of transfers copies of its shards of keys other ranks own plus
    the shards it receives, and then those received shards plus the tensors assembled from them.
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
        # Batched P2P that is NCCL's first call on the group needs every rank in it, and the object
        # collective above runs on Gloo where the group has it: every rank initializes NCCL here.
        dist.all_reduce(torch.zeros(1, device=local_sd[shard0[0]].to_local().device))
        _gather_shard0_to_owners(local_sd, shard0, owner, all_shapes, rank, world)


def _merge_groups(model: nn.Module, keys) -> list:
    """The keys the model's reverse conversion merges into one tensor, one list per merged tensor.

    A merge needs its whole group on one rank, so :func:`_materialize_owned` keeps each together.
    A weight's quantization state goes with it: the merge refuses a quantized weight only where it
    sees its scales, which a weight-specific pattern does not match. A mapping that cannot be
    reversed has none: its reverse falls back to the in-memory names anyway. Groups are matched on
    the names before the reverse splits run: a merge whose sources a split produces is whole on one
    rank only because a split's outputs stay with the fused key they come from.
    """
    try:
        merge_rules = _build_reverse_rules(model)[1]
    except QuantConversionUnsupportedError:
        return []
    key_set = set(keys)
    groups: dict = {}
    for i, rule in enumerate(merge_rules):
        for k in keys:
            target, source_pattern = rule.converter.rename_source_key(k)
            if source_pattern is None:
                continue
            group = groups.setdefault((i, target), [])
            group.append(k)
            if k.endswith(".weight"):
                module = k.removesuffix(".weight")
                group += [
                    module + leaf for leaf in _QUANT_STATE_SUFFIXES if module + leaf in key_set
                ]
    return list(groups.values())


def _revert_whole_sd(model: nn.Module, local_sd: dict) -> bool:
    """Reverse the transformers conversion mapping on this rank's whole tensors, in place.

    Each rank reverts the disjoint subset it owns. Every value is a whole plain tensor by now, so no
    placement handling is needed. Renames and splits act on one tensor at a time, so a subset is
    all they need. A merge needs its whole group, which ownership keeps on one rank
    (:func:`_merge_groups`); a rank holding none of a rule's groups skips the rule, and a rule no
    rank matches is refused, as the single-process path refuses one. Best-effort and atomic across
    the ranks, mirroring the single-process path, which keeps the in-memory names for the whole dict
    on any failure. If the reverse fails on any rank, every rank keeps them, rather than writing a
    checkpoint that mixes the two. Collective.
    """
    reverted = local_sd

    def revert() -> tuple[str | None, list]:
        nonlocal reverted
        unmatched: list = []
        try:
            reverted = revert_weight_conversion_quant_aware(
                model, local_sd, unmatched_merges=unmatched
            )
        except Exception as exc:
            # An exception may carry no message: name its type, so the warning says something.
            return str(exc) or type(exc).__name__, []
        return None, unmatched

    def combine(results: list) -> list:
        failures = [
            f"rank {r}: {error}" for r, (error, _) in enumerate(results) if error is not None
        ]
        return failures or sorted(set.intersection(*(set(unmatched) for _, unmatched in results)))

    failures = all_reduce("reverse weight conversion", revert, combine)
    if failures:
        warnings.warn(
            f"Quant-aware reverse weight conversion skipped ({failures[0]}); exported tensor "
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
    """Write this rank's whole tensors as HF safetensors shards; rank 0 writes the index.

    Each rank splits its tensors into even shards, numbered into one global range. Runs after the
    reverse conversion, whose renames and splits fix the names and sizes packed here.

    Every file is written under a temporary name and renamed into place only once every rank has
    written all of its own, the index last. It never overwrites a checkpoint: a directory that already
    holds one is refused on every rank before anything is written. A failure removes the temporary
    files, but one during the final renames leaves the shards already renamed, which a retry into the
    directory is refused over until they are removed. Each step a rank can fail alone runs in
    :func:`~modelopt.torch.utils.distributed.all_reduce`, which raises on every rank when one fails.

    Every name in ``local_sd`` is written, two names for one tensor included: the caller drops tied
    weights by name beforehand, as the streaming export drops ``_tied_weights_keys``.
    """
    export_dir = Path(export_dir)
    max_bytes = _parse_shard_size(max_shard_size)

    write_keys = sorted(local_sd)
    sizes = {k: _nbytes(local_sd[k]) for k in write_keys}
    # Tensors over the bound get files of their own, so only the rest decide how many to aim for.
    fits = sum(size for size in sizes.values() if size <= max_bytes)
    my_files = _even_bins(write_keys, sizes, max(1, math.ceil(fits / max_bytes)), max_bytes)

    def plan_mine() -> list:
        # In the first collective, which carries one rank's failure to every rank: its filesystem can
        # fail to create the directory, or show a checkpoint there, which the export never overwrites.
        export_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            p.name for p in export_dir.iterdir() if _CHECKPOINT_FILES.fullmatch(p.name)
        )
        if existing:
            raise FileExistsError(
                f"{export_dir} already holds a checkpoint ({', '.join(existing[:3])}), which the "
                "distributed export does not overwrite; remove those files (an export that failed "
                "during its final renames can leave some behind) or export to another directory."
            )
        return [(len(my_files), write_keys)]

    planned = all_reduce("shard plan", plan_mine, _in_rank_order)
    # The reverse conversion checks for rename collisions, but only within one rank's slice. A
    # collision ACROSS ranks would be silently swallowed by the index's dict update and ship a
    # checkpoint quietly missing a tensor, so every rank refuses it here, before any file is written.
    names = Counter(k for _, keys in planned for k in keys)
    collisions = sorted(k for k, n in names.items() if n > 1)
    if collisions:
        raise RuntimeError(
            f"tensor name collision across ranks: {len(collisions)} name(s) held by more than one "
            f"rank; first few: {collisions[:5]}"
        )
    counts = [n for n, _ in planned]
    n_total = sum(counts)
    if n_total == 0:
        raise RuntimeError(
            "distributed export produced no shards on any rank: every tensor was dropped "
            "before the write (state dict emptied upstream)"
        )
    base = sum(counts[:rank])
    staged: list = []  # this rank's files, each written as <name>.tmp until the commit
    index_name = "model.safetensors.index.json"

    def stage(name: str) -> Path:
        # Recorded before the write: one that fails midway leaves a partial file behind.
        staged.append(name)
        return export_dir / f"{name}.tmp"

    def write_mine() -> tuple[dict, int]:
        weight_map: dict = {}
        my_bytes = 0
        for i, keys in enumerate(my_files):
            fname = f"model-{base + i + 1:05d}-of-{n_total:05d}.safetensors"
            path = stage(fname)
            tensors: dict = {}
            ptrs: set = set()
            for k in keys:
                t = local_sd[k].detach().to("cpu").contiguous()
                # Once copied this rank has no further use for its own tensor. The writer never
                # allocates on the device, so dropping it here does not lower the peak (the owned
                # set at entry); it frees the tensor now instead of after the whole file is written.
                del local_sd[k]
                # save_file refuses tensors sharing memory, and neither call copies a tensor that is
                # already on the CPU and contiguous, so two names for one tensor arrive aliased. Copy
                # rather than drop one: every name has to reach the index.
                if t.data_ptr() in ptrs:
                    t = t.clone()
                ptrs.add(t.data_ptr())
                tensors[k] = t
            save_file(tensors, str(path), metadata={"format": "pt"})
            for k in keys:
                weight_map[k] = fname
            my_bytes += sum(_nbytes(t) for t in tensors.values())
            del tensors
        return weight_map, my_bytes

    def write_index() -> bool:
        if rank == 0:
            index = {"metadata": {"total_size": total_bytes}, "weight_map": full_map}
            with open(stage(index_name), "w") as f:
                json.dump(index, f, indent=2)
        return True

    def commit(names: list) -> bool:
        for name in names:
            os.replace(export_dir / f"{name}.tmp", export_dir / name)
        return True

    # A failed export removes the temporary files it wrote, and only those. What was written is gone
    # from ``local_sd``, so a retry needs the state dict built again.
    try:
        gathered = all_reduce("shards written", lambda: [write_mine()], _in_rank_order)
        full_map: dict = {}
        total_bytes = 0
        for m, b in gathered:
            full_map.update(m)
            total_bytes += b
        all_reduce("index written", write_index, all)
        # Every file is written: only now does the directory change, the index last, so no index
        # names a shard that is not in place. Also the closing barrier.
        all_reduce("shards committed", lambda: commit([n for n in staged if n != index_name]), all)
        all_reduce("index committed", lambda: commit([n for n in staged if n == index_name]), all)
    except BaseException:
        left = []
        for name in staged:
            # Best effort: the filesystem that failed the write can fail this too, and its error
            # would replace the one naming the rank that failed, so the files it keeps are named in
            # a warning instead. A file already renamed into place has no temporary file left.
            try:
                (export_dir / f"{name}.tmp").unlink(missing_ok=True)
            except OSError:
                left.append(f"{name}.tmp")
        if left:
            warnings.warn(
                f"rank {rank}: could not remove {len(left)} temporary file(s) the failed export "
                f"wrote to {export_dir}: {left[:5]}"
            )
        raise


def _distributed_save_hf_checkpoint_impl(
    model: nn.Module,
    export_dir: "str | Path",
    maxbound: float,
    kv_cache_format: "str | dict[str, dict[str, str]] | None",
    max_shard_size: "str | int" = "10GB",
    is_modelopt_qlora: bool = False,
    extra_state_dict: "dict[str, torch.Tensor] | None" = None,
) -> None:
    """Distributed HF safetensors export -- no rank-0 full-model host-RAM gather.

    Writes an already-processed, FSDP2-sharded model to ``export_dir`` as HF safetensors, entirely
    distributed. No rank ever holds the whole model:

      1. Fused expert weights (when present): each rank splits its LOCAL shard into per-expert keys
         with global indices and keeps them in place -- no communication. Experts are therefore
         already disjoint across ranks. dp-replica ranks under DP x EP skip this.
      2. Dense / non-expert weights: each fqn is assigned to exactly ONE owner rank and moved there
         (:func:`_materialize_owned`), so every tensor becomes whole on its owner while the set stays
         spread across ranks. Replicated and plain values need no transport; dim-0 shards travel
         point-to-point, bucketed per peer.
      3. The reverse weight conversion runs per-rank on those whole tensors -- it cannot run while
         they are sharded, because transformers' conversion ops are written against whole tensors
         and have no way to express a placement.
      4. Every rank bin-packs its own owned set into even shards, numbered into a global range, and
         writes them directly with ``safetensors.save_file``. Rank 0 unions the weight index.

    Peak memory per rank is its owned set, not the model. Placements the point-to-point transport
    does not cover are rejected rather than routed through an untested slower path.
    """
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.tensor import DTensor

    from .model_utils import TiedWeightMap
    from .quant_utils import postprocess_state_dict

    # Name-based tied-weight dedup, matching the gather path. ``fully_shard`` splits a shared
    # nn.Parameter into distinct per-module shards, so a declared tie surfaces here as two
    # independent DTensors and both sides would be written. The map is derived from names and is
    # identical on every rank, so the alias is dropped consistently; postprocess_state_dict skips a
    # group whose canonical is missing from this rank's slice rather than orphaning the alias.
    tied_map = TiedWeightMap(model)

    export_dir = Path(export_dir)
    rank, world = dist.get_rank(), dist.get_world_size()
    # Where the expert input-scale all-reduce below runs: on the GPU under NCCL (distributed_run
    # sets up "cpu:gloo,cuda:nccl"), on the CPU under a gloo-only group.
    device = (
        torch.device("cpu")
        if dist.get_backend() == "gloo"
        else torch.device("cuda", torch.cuda.current_device())
    )

    # This writer targets a FULLY FSDP2-sharded model -- no expert parallelism. FSDP2
    # shards the fused 3-D expert weight on dim 0 (the expert axis), so fused experts are just
    # DTensors sharded across the fsdp mesh: each per-expert key's TRUE global index comes from the
    # DTensor's own dim-0 offset (``_split_local_fused_module`` reads it), so there is no EP block base
    # (ep_rank=0) and no dp-replica dedup (dp_idx=0 -> every rank writes its own experts). The EP-hybrid
    # (a2a experts) path -- which resolves ep_rank/dp_idx from the parallel-group provider -- is out of
    # scope here.
    ep_rank, dp_idx = 0, 0
    # Total config expert count, to tell a per-EP-group expert tensor (size < total) from one that
    # spans all experts (classic DP x EP) when computing the per-expert global index offset.
    _cfg = getattr(model, "config", None)

    def _experts_of(c):
        if c is None:
            return 0
        return int(
            getattr(c, "num_experts", 0)
            or getattr(c, "num_local_experts", 0)
            or getattr(c, "n_routed_experts", 0)  # nemotron_h
            or 0
        )

    # VLMs nest the text-model fields under ``config.text_config`` (the quantized backbone is
    # ``model.language_model``), so fall back to it when the top-level config has no expert count.
    total_experts = _experts_of(_cfg) or _experts_of(getattr(_cfg, "text_config", None))
    # DP x EP: experts are dp-REPLICATED across the EDP group (each holds a full EP-sharded copy). Only
    # the first replica (dp_idx == 0) writes them; replica ranks (dp_idx > 0) skip the per-expert split.
    # dp_idx is the EDP-group rank resolved above -- 0 when experts are not EDP-replicated (dense /
    # FSDP2 / EP-without-DP), so every such rank writes.
    write_experts = dp_idx == 0
    sharded_sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False))

    # Re-materialize every value as a NORMAL tensor UP FRONT. hf_ptq exports under torch.inference_mode(),
    # so these are inference tensors; the inference flag is STICKY -- clone/reshape/detach/slice all
    # propagate it -- and the expert split, scale reduce and write below all do version-counted ops
    # that reject inference tensors. The only way to drop the flag is to copy the data into a freshly
    # ALLOCATED tensor (here, under the wrapper's inference_mode(False)). For a DTensor, copy the LOCAL
    # shard and rewrap so mesh/placements survive. Bounded to this rank's shard.
    def _to_normal(v):
        if isinstance(v, DTensor):
            _loc = v.to_local()
            _new = torch.empty_like(_loc)
            _new.copy_(_loc)
            # Pass the ORIGINAL global shape/stride: from_local() otherwise re-infers them as
            # local x world, which is only right for an EVEN shard. This helper runs over every
            # DTensor in the state dict just before classification, so with an uneven shard
            # (b_proj: 96 rows over 64 ranks) or a zero-size one (mlp_res_proj) each rank derives a
            # DIFFERENT global shape for the same tensor -- and then classifies it differently,
            # which breaks the export with no hint of where the shape went wrong.
            return DTensor.from_local(
                _new,
                v.device_mesh,
                v.placements,
                run_check=False,
                shape=v.shape,
                stride=v.stride(),
            )
        if v.is_inference():
            _new = torch.empty_like(v)
            _new.copy_(v)
            return _new
        return v

    # Re-materialize IN PLACE, not as `{k: _to_normal(v) for ...}`. The comprehension keeps the ENTIRE
    # original dict alive while building a full second copy -> a transient 2x of the whole state dict.
    # That is fine at 30B (24GB -> 48GB) but OOMs at large MoE: at 235B the write-point state dict is
    # ~157GB/rank (fused expert weights are sharded, but the per-expert NVFP4 scale tensors are
    # replicated on every rank), and doubling it hits ~314GB > GPU. Overwriting each key frees the old
    # value immediately, so peak stays ~1x. (Import of ep-dp's in-place/streaming write handling.)
    for _k in list(sharded_sd.keys()):
        sharded_sd[_k] = _to_normal(sharded_sd[_k])
    prefixes = _fused_experts_prefixes(sharded_sd)
    prefix_set = set(prefixes)

    def _is_expert_key(key: str) -> bool:
        for proj in _FUSED_PROJ:
            for suffix in ("", "_weight_scale", "_weight_scale_2", "_input_scale"):
                if key.endswith(f".experts.{proj}{suffix}"):
                    if key[: key.rfind(".experts.") + len(".experts")] in prefix_set:
                        return True
        return False

    # Probe the DTensor placement of a sample expert + dense weight: directly shows what state the
    # generation forwards left the sharded params in -- FSDP2 (Shard(0)/Replicate/gathered) or EP
    # (expert dim-0 Shard, or a plain local shard for the a2a transport).

    # Sync the shared per-module activation (input) scales across ranks (global max), so every
    # expert of a module gets the same input_scale regardless of which rank owns it.
    in_keys = sorted(k for k in sharded_sd if k.endswith("_input_scale") and _is_expert_key(k))
    if in_keys:
        stacked = torch.stack(
            [
                sharded_sd[k].detach().to(device=device, dtype=torch.float32).reshape(())
                for k in in_keys
            ]
        )
        dist.all_reduce(stacked, op=dist.ReduceOp.MAX)
        for j, k in enumerate(in_keys):
            sharded_sd[k] = stacked[j].clone()
    # (1) Experts: split this rank's local shard into per-expert keys (global idx); stays local.
    # Skipped on dp-replica ranks under DP x EP (their experts are written by dp group 0).
    local_sd: dict = {}
    if write_experts:
        for pi, prefix in enumerate(prefixes):
            local_sd.update(
                _split_local_fused_module(
                    prefix, sharded_sd, ep_rank=ep_rank, total_experts=total_experts
                )
            )
        local_sd = postprocess_state_dict(
            local_sd, maxbound, kv_cache_format, is_modelopt_qlora, tied_map=tied_map
        )
        # In place (see above): avoid a transient 2x copy of the per-expert split.
        for _k in list(local_sd.keys()):
            local_sd[_k] = local_sd[_k].detach().contiguous()
    # At this point local_sd holds ONLY this rank's (already split, per-expert) experts -- capture the
    # set so _materialize_owned can tell disjoint experts from replicated dense.
    expert_key_set = set(local_sd)

    # (2) Dense / non-expert: leave sharded for now -- FSDP2 leaves DTensors (dim-0 shard); EP leaves
    # plain replicated tensors. Do NOT gather to rank 0; _materialize_owned below moves each tensor to
    # a single OWNER rank instead, so the set stays spread across ranks. postprocess runs on every rank
    # (dict-level key renaming + small scalar scale math -- safe on DTensors).
    nonexpert_keys = [k for k in sharded_sd if not _is_expert_key(k)]
    dense_sd = {k: sharded_sd[k] for k in nonexpert_keys}
    dense_sd = postprocess_state_dict(
        dense_sd, maxbound, kv_cache_format, is_modelopt_qlora, tied_map=tied_map
    )
    local_sd.update(dense_sd)
    # No explicit gc.collect() here: the del drops the last references and CPython frees acyclic
    # objects immediately. Collecting a ~900-entry DTensor state dict measured 0.89s of the 0.89s
    # this phase used to cost -- and because it ran at different speeds per rank, the next
    # collective absorbed the skew and looked like gather cost.
    del sharded_sd, dense_sd
    # (3) Whole-tensor redistribution -> reverse conversion -> write. Each tensor ends up WHOLE on
    # exactly one rank (never gathered to a single rank), which is what lets the reverse run: the
    # transformers conversion ops are written against whole tensors and cannot express a placement.
    # The reverse must precede the shard layout, since it renames keys and splits tensors.
    _materialize_owned(local_sd, expert_key_set, rank, world)
    _revert_whole_sd(model, local_sd)
    # Tensors the model does not own -- whatever the caller hands in via extra_state_dict, e.g.
    # weights read straight off the source checkpoint that the FSDP2 loader drops. They are plain
    # CPU tensors, already whole and identical on every rank, so ONE rank adds them to its
    # write set; the index union and
    # the cross-rank duplicate check below then treat them like any other owned key. Rank 0 takes
    # them wholesale: they are small next to the model, and spreading them would cost a collective
    # to agree on the split. They go in AFTER the reverse conversion because they carry their
    # original hub names and were never converted -- reverting them would rename them wrongly. This
    # mirrors the gather path, which merges extra_state_dict into the already-reverted dict.
    from modelopt.torch.utils.distributed import all_reduce

    def check_no_overlap() -> None:
        if rank != 0 or not extra_state_dict:
            return
        collisions = sorted(set(extra_state_dict) & set(local_sd))
        if collisions:
            raise RuntimeError(
                f"extra_state_dict overlaps the exported model weights on {len(collisions)} "
                f"key(s), so the checkpoint would be ambiguous; first few: {collisions[:5]}"
            )

    # The export entry hands the tensors it carries over to rank 0 alone, so whether a rank holds
    # any cannot decide whether it joins: every rank does, and a collision rank 0 finds then raises
    # on all of them, instead of leaving the others in the write's first collective.
    all_reduce("extra_state_dict overlap", check_no_overlap, lambda _: None)
    if extra_state_dict and rank == 0:
        local_sd.update(extra_state_dict)
    _write_even_shards(local_sd, export_dir, max_shard_size, rank, world)
