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

# ----- No-gather distributed HF export (FSDP2): DCP per-rank shard write -----
import torch.nn as nn  # noqa: E402  (distributed_save_hf_checkpoint's signature uses nn.Module)

from ._export_common import _size_to_bytes
from .moe_utils import _FUSED_PROJ, _fused_experts_prefixes, _split_local_fused_module

@contextmanager
def _finfo_accepts_int_dtypes():
    """Work around a torch DCP consolidation bug on integer dtypes.

    ``torch.distributed.checkpoint._consolidate_hf_safetensors._parse_input_metadata`` computes each
    tensor's byte size as ``torch.finfo(dtype).bits // 8`` unconditionally, which raises ``TypeError``
    for integer dtypes -- e.g. the ``uint8``-packed NVFP4 weights. ``torch.iinfo`` also exposes
    ``.bits`` and consolidation only reads ``.bits``, so we temporarily shim ``torch.finfo`` to fall
    back to ``torch.iinfo`` for non-floating dtypes. Present in torch 2.9-2.13 (the supported range);
    remove once fixed upstream. Scoped to the ``dcp.save`` call.
    """
    _orig_finfo = torch.finfo

    def _finfo_or_iinfo(dtype):
        try:
            return _orig_finfo(dtype)
        except TypeError:
            return torch.iinfo(dtype)

    torch.finfo = _finfo_or_iinfo
    try:
        yield
    finally:
        torch.finfo = _orig_finfo


def _bin_pack_fqn_to_index(sizes: dict, max_shard_size: "str | int") -> dict:
    """Deterministically bin-pack tensor FQNs into ~``max_shard_size`` files -> ``{fqn: file_index}``.

    file_index is 1..N; HuggingFaceStorageWriter turns it into ``model-<idx>-of-<N>.safetensors``. Keys
    are sorted by name so every rank computes the SAME mapping from the same global size map. Each file
    gets at least one tensor even if a single tensor exceeds ``max_shard_size``.
    """
    max_bytes = _size_to_bytes(max_shard_size)
    mapping: dict = {}
    file_idx, cur = 1, 0
    for k in sorted(sizes):
        sz = sizes[k]
        if cur > 0 and cur + sz > max_bytes:
            file_idx += 1
            cur = 0
        mapping[k] = file_idx
        cur += sz
    return mapping


def distributed_save_hf_checkpoint(
    model: nn.Module,
    export_dir: "str | Path",
    maxbound: float,
    kv_cache_format: "str | None",
    max_shard_size: "str | int" = "10GB",
    is_modelopt_qlora: bool = False,
) -> None:
    """No-gather distributed HF export. Thin wrapper: hf_ptq exports under ``torch.inference_mode()``,
    so run the entire write with inference mode DISABLED via a nested ``inference_mode(False)`` context
    -- the ``.detach()`` in ``get_model_state_dict`` and DCP's version-counter reads both reject inference
    tensors. The context form reliably restores normal mode for this scope (a decorator did not, under the
    caller's already-active inference_mode). ``no_grad`` too: ``inference_mode(False)`` re-ENABLES autograd,
    so a bare ``clone()``/``detach()`` on an inference tensor would try to set up grad and read its
    (missing) version counter -- ``no_grad`` avoids that while the clones still land as normal tensors
    because inference mode is off."""
    with torch.inference_mode(False), torch.no_grad():
        _distributed_save_hf_checkpoint_impl(
            model, export_dir, maxbound, kv_cache_format, max_shard_size, is_modelopt_qlora
        )


def _distributed_save_hf_checkpoint_impl(
    model: nn.Module,
    export_dir: "str | Path",
    maxbound: float,
    kv_cache_format: "str | None",
    max_shard_size: "str | int" = "10GB",
    is_modelopt_qlora: bool = False,
) -> None:
    """Distributed HF safetensors export via torch DCP -- no rank-0 full-model host-RAM gather.

    Writes an already-processed, FSDP2-sharded model to ``export_dir`` as
    consolidated HF safetensors, entirely distributed:
      - Fused expert weights (when present): each rank splits its LOCAL shard into per-expert keys with global indices
        and keeps them in place -- no gather. dp-replica ranks under DP x EP skip this.
      - Dense / non-expert weights: kept sharded (FSDP2 DTensor dim-0 shards, or EP replicated plain
        tensors), NOT gathered to rank 0.

    All keys are bin-packed into a global ``fqn_to_index_mapping`` (identical on every rank); then
    ``dcp.save`` with ``HuggingFaceStorageWriter(save_distributed=True)`` has each rank write only its
    own keys into ``export_dir/sharded/``. ``consolidate_safetensors_files_on_every_rank`` merges
    those into the final ``model-XXXXX-of-NNNNN.safetensors`` with the output files partitioned across
    ranks (parallel); rank 0 writes only the index. No rank ever holds the full model in host RAM.
    """
    import gc
    import shutil

    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import HuggingFaceStorageWriter
    from torch.distributed.checkpoint._consolidate_hf_safetensors import (
        consolidate_safetensors_files_on_every_rank,
    )
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.tensor import DTensor

    from .quant_utils import postprocess_state_dict

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
    # propagate it -- and the expert split + scale reduce + DCP write below all do version-counted ops
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

    sharded_sd = {k: _to_normal(v) for k, v in sharded_sd.items()}

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
        local_sd = postprocess_state_dict(local_sd, maxbound, kv_cache_format, is_modelopt_qlora)
        local_sd = {k: v.detach().contiguous() for k, v in local_sd.items()}

    # (2) Dense / non-expert: keep sharded -- FSDP2 leaves DTensors (dim-0 shard); EP leaves plain
    # replicated tensors. Do NOT gather to rank 0: DCP writes each DTensor's shards per-rank (parallel,
    # no rank-0 full-model host-RAM gather) and dedups replicated plain tensors. postprocess runs on
    # every rank (dict-level key renaming + small scalar scale math -- safe on DTensors).
    nonexpert_keys = [k for k in sharded_sd if not _is_expert_key(k)]
    dense_sd = {k: sharded_sd[k] for k in nonexpert_keys}
    dense_sd = postprocess_state_dict(dense_sd, maxbound, kv_cache_format, is_modelopt_qlora)
    local_sd.update(dense_sd)
    del sharded_sd, dense_sd
    gc.collect()

    # torch DCP HuggingFaceStorageWriter consolidation writes 0-dim (scalar) tensors as ZERO (shape ()
    # -> value dropped); shape (1,) and larger survive. modelopt stores per-tensor scales (input_scale/
    # weight_scale/weight_scale_2) as 0-dim scalars, so promote any 0-dim tensor to (1,) to preserve its
    # value. Per-tensor () vs (1,) scales are equivalent for deployment; applies to plain + DTensor.
    # (Values are already normal tensors -- re-materialized right after get_model_state_dict.)
    local_sd = {k: (v.reshape(1) if v.dim() == 0 else v) for k, v in local_sd.items()}

    # (3) Global shard layout: bin-pack ALL keys into ~max_shard_size files so every rank passes the
    # SAME fqn_to_index_mapping to the writer. A DTensor's byte size is its GLOBAL size (numel() is
    # global) -- the consolidated file holds the whole tensor; plain keys use their own size. The
    # gather dedups replicated/DTensor keys (present on every rank) and unions the disjoint expert keys.
    local_sizes = {k: int(v.numel() * v.element_size()) for k, v in local_sd.items()}
    gathered: list = [None] * world
    dist.all_gather_object(gathered, local_sizes)
    all_sizes: dict = {}
    for g in gathered:
        all_sizes.update(g)
    fqn_to_index_mapping = _bin_pack_fqn_to_index(all_sizes, max_shard_size)
    n_files = max(fqn_to_index_mapping.values()) if fqn_to_index_mapping else 1

    # (4) Distributed write, then DISTRIBUTED consolidation. save_distributed has each rank write only
    # its own keys (dense DTensor shards + this rank's plain experts) into export_dir/sharded/. We set
    # enable_consolidation=False because the writer's built-in consolidation runs on RANK 0 ONLY -- a
    # serial re-read+rewrite of the whole model that dominated export time for large models (~700s for
    # Kimi-K2 vs ~50s writing). Instead consolidate_safetensors_files_on_every_rank partitions the
    # output files across ranks (idx % world_size) so every rank merges its own subset in parallel;
    # rank 0 then writes only the small index. The finfo shim covers the integer-dtype (NVFP4 uint8)
    # consolidation bug; 0-dim scales were already promoted to (1,) above.
    sharded_dir = export_dir / "sharded"
    writer = HuggingFaceStorageWriter(
        str(sharded_dir),
        fqn_to_index_mapping=fqn_to_index_mapping,
        save_distributed=True,
        enable_consolidation=False,
        thread_count=8,
    )
    with _finfo_accepts_int_dtypes():
        dcp.save(local_sd, storage_writer=writer)
        dist.barrier()
        consolidate_safetensors_files_on_every_rank(
            input_dir=str(sharded_dir),
            output_dir=str(export_dir),
            fqn_to_index_mapping=fqn_to_index_mapping,
            num_threads=8,
        )

    # (5) Drop the intermediate per-rank sharded/ dir.
    if rank == 0:
        shutil.rmtree(sharded_dir, ignore_errors=True)
    dist.barrier()
