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

"""torch.distribute utils."""

import json
import warnings
from contextlib import contextmanager
from io import BytesIO
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import torch

from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils import safe_load

from .model_config_utils import (
    model_config_from_dict,
    model_config_to_dict,
    restore_model_config,
    split_config_and_weights,
)


class NFSWorkspace:
    """A shared workspace implementation using Network File Storage (NFS).

    NOTE: all read/write/modifition to the NFS dir do not involve any collective
          communication nor barrier. It is users' responsibility to synchronize
          all ranks (local and remove processes).

    This implementation uses `torch.save` and `safe_load` (`torch.load(weights_only=True)`) for serialization.

    Args:
        workspace_path: the path to the NFS directory for postprocess cross rank communication.
            If not provided, SharedMemory will be used instead.
    """

    def __init__(self, workspace_path: Path | str | None = None):
        """Create the NFS work dir and clean up existing existing state files."""
        self.path = Path("") if workspace_path is None else Path(workspace_path)
        self._is_initialized = workspace_path is not None
        self.rank = dist.rank()
        if self.is_initialized:
            if self.rank == 0:
                self.path.mkdir(parents=True, exist_ok=True)
            self.state_path = self._get_state_path(self.rank)
            self._clean_up()

    @property
    def is_initialized(self):
        """Whether the workspace is initialized."""
        return self._is_initialized

    def write_configs_and_weights(self, config_json: dict[str, Any], weights: dict[str, Any]):
        """All ranks write the state file to the shared NFS dir.

        Args:
            config_json: model or module config in json
            weights: module weights in torch's state_dict format
        """
        if not self.is_initialized:
            raise ValueError("NFSWorkspace is not initialized!")
        self._clean_up()
        torch.save({"config": config_json, "weight": weights}, self.state_path)

    def read_configs_and_weights_from_rank(
        self, target_rank: int
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """All ranks read the target_rank state file.

        Args:
            target_rank: the target rank

        Returns:
            the model/module config and the weights
        """
        if not self.is_initialized:
            raise ValueError("NFSWorkspace is not initialized!")
        state_path = self._get_state_path(target_rank)
        if state_path.exists():
            state = safe_load(state_path, map_location="cpu")
            return state["config"], state["weight"]
        else:
            return None, None

    def _get_state_path(self, target_rank: int) -> Path:
        """Return the state file name of a particular rank.

        Args:
            target_rank: the target rank

        Returns:
            the state file path of the target rank
        """
        if not self.is_initialized:
            raise ValueError("NFSWorkspace is not initialized!")
        return self.path.joinpath(f"rank_{target_rank}_state.pth")

    def _clean_up(self):
        """Remove existing state files."""
        if not self.is_initialized:
            raise ValueError("NFSWorkspace is not initialized!")
        self.state_path.unlink(missing_ok=True)


@contextmanager
def get_tensors_parallel(tensor: torch.Tensor, ranks: list[int], group=None):
    """Gathers the tensors across distributed processes using shm.

    Args:
        tensor: the tensor that each rank want to pass to the first rank.
            The tensors across the ranks need to have the same size.
        ranks: the list of the ranks
        group: the barrier sync group.

    Yields:
        the first rank in the ranks has the full access of the tensors across all the ranks.
        the other ranks returns an empty list

    The shm will be destroyed after consumption.
    """
    assert tensor is not None
    assert len(ranks) > 1
    local_rank = dist.rank()
    shm_writer = None
    shm_readers = []
    tensor = tensor.cpu()

    is_merged_rank = local_rank == ranks[0]
    # Create shm and copy the tensor to the shm if not the merged rank.
    # Assume each tensor need up to 2KB additional space for metadata.
    if not is_merged_rank:
        shm_writer = SharedMemory(name=f"rank_{local_rank}", create=True, size=tensor.nbytes + 2048)
        torch.save(tensor, shm_writer._mmap)  # type: ignore[attr-defined]
    # All ranks wait for this to complete.
    dist.barrier(group)

    tensors = []
    # The merged rank gather the tensor from the other ranks (including itself).
    if is_merged_rank:
        for rank in ranks:
            if rank == ranks[0]:
                tensors.append(tensor)
            else:
                shm = SharedMemory(name=f"rank_{rank}", create=False)
                assert shm.buf is not None
                shared_tensor = torch.load(BytesIO(shm.buf))
                tensors.append(shared_tensor)
                shm_readers.append(shm)
    try:
        # Send the tensor list to the consumer.
        # The merged rank will get a valid tensor while the other ranks an empty tensor.
        yield tensors
    finally:
        # Reader closes the shms.
        if shm_readers:
            for shm in shm_readers:
                shm.close()

        # All ranks wait for the reader to close the shms.
        dist.barrier(group)

        # Writer frees the shm resource.
        if shm_writer is not None:
            shm_writer.close()
            shm_writer.unlink()


@contextmanager
def get_configs_parallel(config, ranks: list[int], group, workspace_path: Path | str | None = None):
    """Gathers the layer config across distributed processes using shm or NFS.

    Args:
        config: the config (nullable) that each rank want to pass to the first rank.
        ranks: the list of the ranks
        group: the barrier sync group.
        workspace_path: the path to the NFS directory for postprocess cross rank communication.

    Yields:
        the first rank in the ranks has the full access of the configs across all the ranks.
        the other ranks returns an empty list

    When workspace_path is provided, an NFSWorkspace object is created to perform communication
    across ranks. Otherwise, `SharedMemory` is used for local multi-process communication.
    The shm will be destroyed after consumption.
    """
    assert len(ranks) > 1
    local_rank = dist.rank()
    shm_writer = None
    shm_readers = []
    nfs_workspace = NFSWorkspace(workspace_path)

    is_merged_rank = local_rank == ranks[0]

    def _get_weights_nbytes(weights_dict: dict[str, torch.Tensor]):
        total_nbytes = 0
        for k, v in weights_dict.items():
            # Assume each tensor need up to 2KB additional space for metadata.
            # In reality this should be much smaller.
            total_nbytes = total_nbytes + len(k) + v.nbytes + 2048

        return total_nbytes

    # Create shm and copy the serialized config to the shm if not the merged rank.
    if not is_merged_rank:
        if config is not None:
            config_dict = model_config_to_dict(config)
            # Add additional config type name to the dict so we can later pick the right config type.
            config_dict["__name__"] = str(type(config).__name__)
            weights = {}
            split_config_and_weights(config_dict, weights)

            config_json = json.dumps(config_dict)

            if nfs_workspace.is_initialized:
                # All ranks except for the master merge rank write to the NFS dir.
                nfs_workspace.write_configs_and_weights(config_dict, weights)
            else:
                # SHM data structure: 8B json size, serialized json bytes and the weights dict.
                shm_writer = SharedMemory(
                    name=f"rank_{local_rank}_config",
                    create=True,
                    size=(8 + len(config_json) + _get_weights_nbytes(weights)),
                )
                assert shm_writer.buf is not None

                # Write json length to the shm
                shm_writer.buf[:8] = len(config_json).to_bytes(8, "little")

                # Write json to the shm
                shm_writer.buf[8 : len(config_json) + 8] = config_json.encode()

                # Write np tensors to the shm.
                shm_writer._mmap.seek(len(config_json) + 8)  # type: ignore[attr-defined]
                torch.save(weights, shm_writer._mmap)  # type: ignore[attr-defined]
        else:
            # If the config is None, we just store the empty 0.
            shm_writer = SharedMemory(
                name=f"rank_{local_rank}_config",
                create=True,
                size=8,
            )
            assert shm_writer.buf is not None

            shm_writer.buf[:8] = (0).to_bytes(8, "little")

    # All ranks wait for this to complete.
    dist.barrier(group)

    configs = []
    if is_merged_rank:
        for rank in ranks:
            if rank == ranks[0]:
                configs.append(config)
            elif nfs_workspace.is_initialized:
                # The master merge rank read other configs from the NFS dir.
                config_dict, weights = nfs_workspace.read_configs_and_weights_from_rank(rank)
                if config_dict is not None:
                    restore_model_config(config_dict, weights)
                    config = model_config_from_dict(config_dict)
                    configs.append(config)
            else:
                shm = SharedMemory(name=f"rank_{rank}_config", create=False)
                assert shm.buf is not None
                len_json = int.from_bytes(shm.buf[:8], "little")

                if len_json != 0:
                    config_dict = json.loads(shm.buf[8 : 8 + len_json].tobytes().decode())
                    weights = torch.load(BytesIO(shm.buf[8 + len_json :]))
                    restore_model_config(config_dict, weights)
                    config = model_config_from_dict(config_dict)

                    configs.append(config)
                    shm_readers.append(shm)
    try:
        # Send the config list to the consumer.
        # The merged rank will get a valid config list while the other ranks an empty list.
        yield configs
    finally:
        # Reader closes the shms.
        if shm_readers:
            for shm in shm_readers:
                shm.close()

        # All ranks wait for the reader to close the shms.
        dist.barrier(group)

        # Writer frees the shm resource.
        if shm_writer is not None:
            shm_writer.close()
            shm_writer.unlink()

# ----- No-gather distributed HF export (FSDP2): per-rank owner gather + parallel write -----
import torch.nn as nn  # noqa: E402  (distributed_save_hf_checkpoint's signature uses nn.Module)

from .moe_utils import _FUSED_PROJ, _fused_experts_prefixes, _split_local_fused_module

@contextmanager


def distributed_save_hf_checkpoint(
    model: nn.Module,
    export_dir: "str | Path",
    maxbound: float,
    kv_cache_format: "str | None",
    max_shard_size: "str | int" = "10GB",
    is_modelopt_qlora: bool = False,
    extra_state_dict: "dict[str, torch.Tensor] | None" = None,
) -> None:
    """No-gather distributed HF export. Thin wrapper: hf_ptq exports under ``torch.inference_mode()``,
    so run the entire write with inference mode DISABLED via a nested ``inference_mode(False)`` context
    -- the ``.detach()`` in ``get_model_state_dict`` and the version-counted ops in the write path both
    reject inference
    tensors. The context form reliably restores normal mode for this scope (a decorator did not, under the
    caller's already-active inference_mode). ``no_grad`` too: ``inference_mode(False)`` re-ENABLES autograd,
    so a bare ``clone()``/``detach()`` on an inference tensor would try to set up grad and read its
    (missing) version counter -- ``no_grad`` avoids that while the clones still land as normal tensors
    because inference mode is off."""
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
    BOUND, not the target. Bumps ``n`` (up to one file per tensor) if a lumpy item pushes a bin over the
    bound; a single tensor larger than ``max_bytes`` gets its own (over-bound) file, matching HF behavior.
    Returns the non-empty bins (``list[list[key]]``)."""
    n = max(1, min(n, len(keys))) if keys else 1
    ordered = sorted(keys, key=lambda k: -sizes[k])
    while True:
        bins: list = [[] for _ in range(n)]
        loads = [0] * n
        for k in ordered:
            j = min(range(n), key=lambda i: loads[i])
            bins[j].append(k)
            loads[j] += sizes[k]
        if n >= len(keys) or not loads or max(loads) <= max_bytes:
            return [b for b in bins if b]
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
        for (peer, dt) in sorted(send_groups, key=lambda kd: (kd[0], str(kd[1]))):
            ks = sorted(send_groups[(peer, dt)])
            buf = torch.cat([local_sd[k].to_local().reshape(-1) for k in ks])
            keepalive.append(buf)  # must outlive the wait
            ops.append(dist.P2POp(dist.isend, buf, peer))
        for (srcr, dt) in sorted(recv_groups, key=lambda kd: (kd[0], str(kd[1]))):
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
    replicated, shard0, unsupported, plain = [], [], [], []
    for k in dense_keys:
        v = local_sd[k]
        if not isinstance(v, DTensor):
            plain.append(k)
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


def _write_even_shards(
    local_sd: dict, export_dir: "str | Path", max_shard_size: "str | int", rank: int, world: int
) -> None:
    """Write this rank's whole tensors directly as final HF safetensors; rank 0 unions the index.

    Every tensor is whole on exactly one rank, so there is no consolidation step. This rank's set is
    split into even shards, numbered into a global range (prefix-sum of per-rank file counts), and
    written with ``safetensors.save_file``. Must run AFTER the reverse conversion -- the reverse
    renames keys and splits tensors, so the sizes and names bin-packed here have to be the final ones.
    """
    import json
    import math

    import torch.distributed as dist
    from safetensors.torch import save_file

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = _size_to_bytes(max_shard_size)

    write_keys = sorted(local_sd)
    sizes = {k: _nbytes(local_sd[k]) for k in write_keys}
    total = sum(sizes.values())
    my_files = _even_bins(write_keys, sizes, max(1, math.ceil(total / max_bytes)), max_bytes)
    n_local = len(my_files)
    counts: list = [None] * world
    dist.all_gather_object(counts, n_local)
    n_total = sum(counts)
    if n_total == 0:
        raise RuntimeError(
            "distributed export produced no shards on any rank: every tensor was dropped "
            "before the write (state dict emptied upstream)"
        )
    base = sum(counts[:rank])

    weight_map: dict = {}
    my_bytes = 0
    for i, keys in enumerate(my_files):
        fname = f"model-{base + i + 1:05d}-of-{n_total:05d}.safetensors"
        tensors = {k: local_sd[k].detach().to("cpu").contiguous() for k in keys}
        save_file(tensors, str(export_dir / fname), metadata={"format": "pt"})
        for k in keys:
            weight_map[k] = fname
            # Once a tensor is on disk this rank has no further use for it. Dropping it here keeps
            # peak memory at roughly one shard above the owned set instead of holding all of them.
            del local_sd[k]
        my_bytes += sum(_nbytes(t) for t in tensors.values())
        del tensors

    # One collective, not two: at this point each rank has already finished writing, so the latency
    # of these round trips is pure overhead on the critical path.
    gathered: list = [None] * world
    dist.all_gather_object(gathered, (weight_map, my_bytes))
    if rank == 0:
        full_map: dict = {}
        total_bytes = 0
        for m, b in gathered:
            full_map.update(m)
            total_bytes += b
        # The reverse conversion checks for rename collisions, but only within one rank's slice.
        # A collision ACROSS ranks would otherwise be silently swallowed by this dict update and
        # ship a checkpoint that is quietly missing a tensor.
        n_written = sum(len(m) for m, _ in gathered)
        if n_written != len(full_map):
            raise RuntimeError(
                f"tensor name collision across ranks: {n_written} tensors written but only "
                f"{len(full_map)} distinct names in the index"
            )
        index = {"metadata": {"total_size": total_bytes}, "weight_map": full_map}
        with open(export_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
    dist.barrier()


def _distributed_save_hf_checkpoint_impl(
    model: nn.Module,
    export_dir: "str | Path",
    maxbound: float,
    kv_cache_format: "str | None",
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
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # This writer targets a FULLY FSDP2-sharded model -- no expert parallelism. FSDP2
    # shards the fused 3-D expert weight on dim 0 (the expert axis), so fused experts are just
    # DTensors sharded across the fsdp mesh: each per-expert key's TRUE global index comes from the
    # DTensor's own dim-0 offset (``_split_local_fused_module`` reads it), so there is no EP block base
    # (ep_rank=0) and no dp-replica dedup (dp_idx=0 -> every rank writes its own experts). The EP-hybrid
    # (a2a experts) path -- which resolves ep_rank/dp_idx from the parallel-group provider -- is out of
    # scope here.
    ep_rank, ep_size, dp_idx = 0, 1, 0
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
            return DTensor.from_local(_new, v.device_mesh, v.placements, run_check=False)
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

    # Tensors the model does not own -- MTP weights read straight off the source checkpoint by
    # hf_ptq's load_mtp_weights, which the FSDP2 loader drops. They are plain CPU tensors, already
    # whole and identical on every rank, so ONE rank adds them to its write set; the index union and
    # the cross-rank duplicate check below then treat them like any other owned key. Rank 0 takes
    # them wholesale: they are small next to the model, and spreading them would cost a collective
    # to agree on the split. They go in AFTER the reverse conversion because they carry their
    # original hub names and were never converted -- reverting them would rename them wrongly. This
    # mirrors the gather path, which merges extra_state_dict into the already-reverted dict.
    if extra_state_dict and rank == 0:
        collisions = sorted(set(extra_state_dict) & set(local_sd))
        if collisions:
            raise RuntimeError(
                f"extra_state_dict overlaps the exported model weights on {len(collisions)} key(s), "
                f"so the checkpoint would be ambiguous; first few: {collisions[:5]}"
            )
        local_sd.update(extra_state_dict)

    _write_even_shards(local_sd, export_dir, max_shard_size, rank, world)
