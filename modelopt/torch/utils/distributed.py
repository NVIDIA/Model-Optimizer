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

"""Utility functions for using torch.distributed."""

import functools
import io
import os
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import timedelta
from typing import Any
from warnings import warn

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed.tensor import DTensor

__all__ = [
    "DistributedProcessGroup",
    "ParallelState",
    "backend",
    "barrier",
    "fsdp2_shard",
    "fsdp2_wrap",
    "fsdp_aware_forward_loop",
    "is_available",
    "is_initialized",
    "is_master",
    "load_fsdp2_causal_lm",
    "rank",
    "shard_dataloader",
    "size",
]


def is_available() -> bool:
    """Returns whether the distributed package is available."""
    return torch.distributed.is_available()


def is_initialized() -> bool:
    """Returns whether the distributed package is initialized."""
    return is_available() and torch.distributed.is_initialized()


def backend() -> str | None:
    """Returns the distributed backend."""
    if is_initialized():
        return "torch"
    return None


def size(group=None) -> int:
    """Returns the number of processes."""
    if backend() == "torch":
        return torch.distributed.get_world_size(group=group)
    return 1


def rank(group=None) -> int:
    """Returns the rank of the current process."""
    if backend() == "torch":
        return torch.distributed.get_rank(group=group)
    return 0


def local_rank() -> int:
    """Returns the local rank of the current process."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    warn("LOCAL_RANK environment variable not found. Using global rank instead.")
    return rank()


def is_master(group=None) -> bool:
    """Returns whether the current process is the master process."""
    return rank(group=group) == 0


def is_last_process(group=None) -> bool:
    """Returns whether the current process is the last process."""
    return rank(group=group) == size(group=group) - 1


def _serialize(obj: Any) -> torch.Tensor:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    storage = torch.UntypedStorage.from_buffer(buffer.getvalue(), dtype=torch.uint8)
    tensor = torch.ByteTensor(storage)
    return tensor


def _deserialize(tensor: torch.Tensor, size: int | None = None) -> Any:
    buffer = tensor.numpy().tobytes()
    if size is not None:
        buffer = buffer[:size]
    # Security NOTE: weights_only=False is used here on internally-generated buffer, not on untrusted user input
    obj = torch.load(io.BytesIO(buffer), weights_only=False)
    return obj


def _broadcast(tensor: torch.Tensor, src: int = 0, group=None) -> None:
    if backend() == "torch":
        torch.distributed.broadcast(tensor, src, group)


def broadcast(obj: Any, src: int = 0, group=None) -> Any:
    """Broadcasts an object from the source to all other processes."""
    if size() == 1:
        return obj

    # serialize
    if rank() == src:
        tensor = _serialize(obj).cuda()

    # broadcast the tensor size
    tensor_size = (
        torch.LongTensor([tensor.numel()]).cuda() if rank() == src else torch.LongTensor([0]).cuda()
    )
    _broadcast(tensor_size, src=src, group=group)

    # broadcast the tensor
    if rank() != src:
        tensor = torch.ByteTensor(size=(tensor_size.item(),)).cuda()
    _broadcast(tensor, src=src, group=group)

    # deserialize
    if rank() != src:
        obj = _deserialize(tensor.cpu())
    return obj


def _allgather(tensors: list[torch.Tensor], tensor: torch.Tensor, group=None) -> None:
    if backend() == "torch":
        torch.distributed.all_gather(tensors, tensor, group)


def allgather(obj: Any, group=None) -> list[Any]:
    """Gathers an object from all processes into a list."""
    if size(group) == 1:
        return [obj]

    # serialize
    tensor = _serialize(obj).cuda()

    # gather the tensor size
    tensor_size = torch.LongTensor([tensor.numel()]).cuda()
    tensor_sizes = [torch.LongTensor([0]).cuda() for _ in range(size(group))]
    _allgather(tensor_sizes, tensor_size, group)
    tensor_sizes = [int(tensor_size.item()) for tensor_size in tensor_sizes]
    max_size = max(tensor_sizes)

    # gather the tensor
    tensors = [torch.ByteTensor(size=(max_size,)).cuda() for _ in tensor_sizes]
    if tensor_size != max_size:
        padding = torch.ByteTensor(size=(max_size - tensor_size,)).cuda()
        tensor = torch.cat((tensor, padding), dim=0)
    _allgather(tensors, tensor, group)

    # deserialize
    objs = []
    for tensor_size, tensor in zip(tensor_sizes, tensors):
        obj = _deserialize(tensor.cpu(), size=tensor_size)
        objs.append(obj)
    return objs


def allreduce(obj: Any, reduction: str = "sum", group=None) -> Any:
    """Reduces an object from all processes."""
    objs = allgather(obj, group)
    if reduction == "sum":
        return sum(objs)
    else:
        raise NotImplementedError(reduction)


def barrier(group=None) -> None:
    """Synchronizes all processes."""
    if size() == 1:
        return
    if backend() == "torch":
        torch.distributed.barrier(group=group)


def master_only(func):
    """Decorator to run a function only on the master process and broadcast the result."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return broadcast(func(*args, **kwargs) if is_master() else None)

    return wrapper


def setup(timeout: timedelta | None = None):
    """Sets up the distributed environment."""
    torch.cuda.set_device(local_rank())
    if not is_initialized():
        torch.distributed.init_process_group("cpu:gloo,cuda:nccl", timeout=timeout)


def cleanup():
    """Cleans up the distributed environment."""
    if is_initialized():
        with suppress(Exception):
            barrier()
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# FSDP2 helpers — used by examples/llm_ptq to run PTQ calibration under FSDP2.
# ---------------------------------------------------------------------------


def fsdp2_wrap(
    model,
    override_cls_name: str | None = None,
    mp_policy=None,
    device=None,
    cpu_offload: bool = False,
):
    """Apply FSDP2 ``fully_shard`` to each decoder layer of ``model``.

    Decoder layers are auto-detected via ``LayerActivationCollector.get_decoder_layers``;
    pass ``override_cls_name`` to force a specific block class instead.

    Args:
        mp_policy: ``MixedPrecisionPolicy`` for compute/reduce dtype (``None`` = no cast).
        device: stream each layer here just before sharding (avoids holding the full
            model on GPU at once); ``None`` shards in place.
        cpu_offload: attach ``CPUOffloadPolicy`` so each shard lives on CPU between
            forwards and streams to GPU per-layer. Trades PCIe traffic for GPU memory;
            use only when the per-rank shard is the binding constraint.

    The root is intentionally NOT sharded — ``embed_tokens``/``lm_head`` stay plain
    replicated tensors, since a DTensor ``embed_tokens.weight`` breaks the embedding
    lookup on plain ``input_ids`` during layerwise calibration.
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

    from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

    if override_cls_name:
        layers = [m for m in model.modules() if type(m).__name__ == override_cls_name]
        if not layers:
            raise RuntimeError(f"No modules of class {override_cls_name!r} found in model")
    else:
        layers = LayerActivationCollector.get_decoder_layers(model)
        if not layers:
            raise RuntimeError(
                "Could not auto-detect decoder layers; pass override_cls_name explicitly."
            )
    fsdp_kwargs: dict[str, Any] = {"reshard_after_forward": True}
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
    for layer in layers:
        if device is not None:
            layer.to(device)
        fully_shard(layer, **fsdp_kwargs)
    return model


def fsdp2_shard(model, device, src_state_dict=None, mp_policy=None, cpu_offload=False):
    """Shard a model across the current process group (accelerate-style rank-0 load).

    Caller contract: ``model`` is built on every rank with params on ``meta`` and
    buffers on CPU (use ``init_empty_weights(include_buffers=False)`` around
    ``from_config``). Rank 0 additionally passes ``src_state_dict`` captured from a
    real CPU model loaded via ``from_pretrained``; other ranks pass ``{}``.

    ``set_model_state_dict(broadcast_from_rank0=True)`` (step 3) is a collective, so
    every rank must reach it: non-rank-0 ranks pass ``{}`` (empty, not ``None``) to
    participate in the broadcast. ``src_state_dict=None`` skips the broadcast entirely
    and must therefore be ``None`` on *all* ranks (e.g. sharding a model that will be
    loaded later) — mixing ``None`` with a populated dict across ranks will hang.

    Also sets ``model._original_architectures`` (FSDP2 wrapping can clobber
    ``config.architectures``, which export reads back).

    Set ``cpu_offload=True`` to attach FSDP2's ``CPUOffloadPolicy`` to wrapped
    layers (each rank's shard lives on CPU between forwards). See
    ``fsdp2_wrap`` docstring for the trade-off.

    Root is never sharded (see ``fsdp2_wrap`` docstring). embed_tokens and
    lm_head stay as plain replicated tensors on every rank.

    Steps:
    1. ``fsdp2_wrap`` — apply ``fully_shard`` to decoder layers.
    2. Materialize: meta params → empty GPU storage; real CPU buffers → GPU
       (preserves their values; ``to_empty`` is NOT used because it would wipe
       buffers).
    3. ``set_model_state_dict(broadcast_from_rank0=True)`` — fills params and
       persistent buffers from rank 0.
    4. ``model.tie_weights()`` — restore tied embeddings (no-op for untied).
    5. Freeze params (so ``patch_fsdp_mp_dtypes`` trainable-only check passes).
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    model._original_architectures = list(model.config.architectures or [])

    fsdp2_wrap(model, mp_policy=mp_policy, cpu_offload=cpu_offload)

    # With CPU offload, FSDP2 requires DTensor params on CPU at lazy_init time
    # (it streams them to GPU per-layer during forward). Also, set_model_state_dict
    # rejects mixed-device models — so materialize everything on CPU first,
    # broadcast, then promote non-DTensor params + buffers to GPU after.
    _materialize_meta_model(model, torch.device("cpu") if cpu_offload else device)

    if src_state_dict is not None:
        set_model_state_dict(
            model,
            src_state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )

    if cpu_offload:
        _promote_non_dtensor_to_gpu(model, device)

    if hasattr(model, "tie_weights"):
        model.tie_weights()

    model.requires_grad_(False)

    return model


def shard_dataloader(loader, rank: int, world_size: int):
    """Wrap a DataLoader with a DistributedSampler so each rank sees a unique shard.

    ``drop_last=False`` keeps per-rank batch counts equal (else a rank exits
    calibration early and hangs the others on FSDP2 collectives), at the cost of the
    sampler repeating up to ``world_size - 1`` samples to pad the even split.
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    sampler = DistributedSampler(
        loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        collate_fn=loader.collate_fn,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )


def fsdp_aware_forward_loop(wrapped_model, dataloader, device=None):
    """Build an ``mtq.quantize`` ``forward_loop`` that respects FSDP wrapping.

    ``mtq.quantize`` hands ``forward_loop`` the *unwrapped* inner module, and calling
    that bypasses FSDP's pre/post-forward hooks (no all-gather/reshard) — breaking
    calibration. This closure ignores that argument and calls the captured *wrapped*
    model instead.

    TODO: ``transformers_trainer.py`` (QLoRA path) has the same logic inlined in
    ``_quantize_model``; consolidate it onto this helper.
    """
    from tqdm import tqdm

    def calibrate(_unwrapped_model):
        for batch in tqdm(dataloader, desc="Calibrating", disable=not is_master()):
            if device is not None:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
            wrapped_model(**batch)

    return calibrate


def broadcast_state_dict(
    state_dict_or_none: dict | None,
    src: int,
    device: torch.device,
    pg=None,
) -> dict:
    """Broadcast a dict of CPU tensors from rank ``src`` to all ranks.

    Two phases: (1) broadcast metadata (key list + shape/dtype) via
    ``broadcast_object_list``, (2) broadcast each tensor via ``dist.broadcast``.
    Source rank passes the populated dict; non-source ranks pass ``None``.
    Returns a dict of tensors on ``device`` on every rank.
    """
    is_src = torch.distributed.get_rank() == src
    meta: list[Any] = (
        [{name: (tuple(t.shape), t.dtype) for name, t in state_dict_or_none.items()}]
        if is_src and state_dict_or_none is not None
        else [None]
    )
    torch.distributed.broadcast_object_list(meta, src=src, group=pg)
    meta_dict = meta[0]
    assert meta_dict is not None, f"src rank {src} passed no state dict to broadcast"

    src_state_dict = state_dict_or_none or {}
    out: dict[str, torch.Tensor] = {}
    for name, (shape, dtype) in meta_dict.items():
        if is_src:
            t = src_state_dict[name].to(device, non_blocking=True)
        else:
            t = torch.empty(shape, dtype=dtype, device=device)
        torch.distributed.broadcast(t, src=src, group=pg)
        out[name] = t
    return out


def _read_safetensors_state_dict(
    ckpt_path: str,
    weight_map: dict,
    select: Callable[[str], bool],
) -> dict:
    """Read tensors whose name satisfies ``select`` from safetensors files.

    Groups param names by file to avoid re-opening. Returns CPU tensors.
    Uses ``safe_open`` so only the requested tensors' bytes are read.
    """
    from safetensors import safe_open

    by_file: dict[str, list[str]] = {}
    for name, file in weight_map.items():
        if select(name):
            by_file.setdefault(file, []).append(name)

    state: dict[str, torch.Tensor] = {}
    for file, names in by_file.items():
        with safe_open(os.path.join(ckpt_path, file), framework="pt", device="cpu") as f:
            for name in names:
                state[name] = f.get_tensor(name)
    return state


def _materialize_meta_model(model: nn.Module, materialize_device: torch.device) -> None:
    """Replace meta-device params/buffers with empty tensors on ``materialize_device``.

    Triggers FSDP2's ``_apply`` override on wrapped modules, which calls
    ``reset_sharded_param`` to refresh FSDP's internal state.
    """

    def _fn(t):
        is_meta_dtensor = isinstance(t, DTensor) and t._local_tensor.is_meta
        if is_meta_dtensor or (not isinstance(t, DTensor) and t.is_meta):
            return torch.empty_like(t, device=materialize_device)
        return t.to(materialize_device)

    model._apply(_fn)


def _promote_non_dtensor_to_gpu(model: nn.Module, device: torch.device) -> None:
    """Move all non-DTensor params + buffers in ``model`` to ``device`` in-place.

    Used after CPU-offload loading: decoder DTensor shards stay on CPU (FSDP2
    streams them to GPU per layer), while root-level plain params and buffers
    need to live on GPU so forwards work.
    """
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None or isinstance(param, DTensor):
                continue
            module._parameters[name] = nn.Parameter(
                param.data.to(device), requires_grad=param.requires_grad
            )
        for name, buf in list(module._buffers.items()):
            if buf is None or isinstance(buf, DTensor):
                continue
            module._buffers[name] = buf.to(device)


def _load_via_parallel_read(
    ckpt_path: str,
    device: torch.device,
    rank: int,
    world_size: int,
    trust_remote_code: bool,
    mp_policy,
    cpu_offload: bool,
    weight_map: dict,
):
    """Parallel-read path: each rank reads its share of decoder layers from disk.

    Phase D: each rank reads its owned layers from disk in parallel.
    Phase E: per-layer broadcast from owner to all ranks; shard locally into the
    FSDP2 DTensor via ``set_model_state_dict(broadcast_from_rank0=False)``.
    Phase F: rank 0 reads + broadcasts non-decoder params; loaded into the
    unwrapped root.
    """
    from accelerate import init_empty_weights
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    from transformers import AutoConfig, AutoModelForCausalLM

    from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

    hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=trust_remote_code)
    dtype = getattr(hf_config, "torch_dtype", None) or torch.bfloat16

    # Phase A: meta skeleton on every rank. include_buffers=False so computed
    # buffers (e.g. rotary inv_freq, often non-persistent) are built for real on
    # CPU here rather than stranded on meta with nothing to materialize them.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            hf_config, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
    model.eval()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Phase B: wrap decoder layers (root NOT wrapped). Discover prefixes.
    decoder_layers = LayerActivationCollector.get_decoder_layers(model)
    if decoder_layers is None:
        raise RuntimeError("Could not auto-detect decoder layers for parallel-read loader.")
    module_to_name = {m: n for n, m in model.named_modules()}
    layer_prefixes = [module_to_name[layer] + "." for layer in decoder_layers]
    model._original_architectures = list(model.config.architectures or [])

    fsdp2_wrap(model, mp_policy=mp_policy, cpu_offload=cpu_offload)

    # Phase C: materialize meta → empty tensors. With cpu_offload, DTensor shards
    # land on CPU (FSDP2 streams them to GPU per-layer during forward).
    _materialize_meta_model(model, torch.device("cpu") if cpu_offload else device)

    # Phase D: each rank reads its owned layers from disk in parallel.
    owned: dict[int, dict] = {}
    for layer_idx in range(len(decoder_layers)):
        if layer_idx % world_size == rank:
            prefix = layer_prefixes[layer_idx]

            def _has_prefix(n: str, p: str = prefix) -> bool:
                return n.startswith(p)

            owned[layer_idx] = _read_safetensors_state_dict(ckpt_path, weight_map, _has_prefix)

    # Phase E: per-layer broadcast + shard. Broadcasts run on GPU (NCCL requires
    # CUDA tensors); with cpu_offload we copy back to CPU before writing into the
    # CPU-resident DTensor shard.
    for layer_idx, layer in enumerate(decoder_layers):
        src = layer_idx % world_size
        layer_state_full = broadcast_state_dict(owned.get(layer_idx), src=src, device=device)
        prefix = layer_prefixes[layer_idx]
        stripped = {k[len(prefix) :]: v for k, v in layer_state_full.items()}
        if cpu_offload:
            stripped = {k: v.cpu() for k, v in stripped.items()}
        # Slice each rank's local DTensor shard from the full tensor it already holds
        # (broadcast_from_rank0=False → no collective needed).
        set_model_state_dict(
            layer,
            stripped,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False),
        )
        if src == rank:
            del owned[layer_idx]
        del layer_state_full, stripped

    # Phase F: non-decoder params (embed, lm_head, norm) — rank 0 reads + broadcasts.
    layer_prefix_tuple = tuple(layer_prefixes)
    non_layer = (
        _read_safetensors_state_dict(
            ckpt_path, weight_map, lambda n: not n.startswith(layer_prefix_tuple)
        )
        if rank == 0
        else None
    )
    non_layer = broadcast_state_dict(non_layer, src=0, device=device)
    if cpu_offload:
        non_layer = {k: v.cpu() for k, v in non_layer.items()}
    missing, unexpected = model.load_state_dict(non_layer, strict=False, assign=False)
    real_missing = [k for k in missing if not k.startswith(layer_prefix_tuple)]
    if real_missing:
        warn(f"Missing non-layer keys on rank {rank}: {real_missing[:5]}...")
    if unexpected:
        warn(f"Unexpected keys in non-layer state dict on rank {rank}: {unexpected[:3]}...")

    if cpu_offload:
        _promote_non_dtensor_to_gpu(model, device)

    # Phase G: tie weights, freeze.
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model.requires_grad_(False)

    return model


def load_fsdp2_causal_lm(
    ckpt_path: str,
    device: torch.device,
    rank: int,
    world_size: int = 1,
    *,
    trust_remote_code: bool = False,
    mp_policy=None,
    cpu_offload: bool = False,
):
    """Load and FSDP2-shard a HuggingFace causal LM.

    Reusable loader with no dependency on argparse / CLI semantics.

    Default path: **parallel read** — each rank reads its share of decoder
    layers from disk in parallel, broadcasts to other ranks. Eliminates the
    rank-0 disk bottleneck. Handles ``cpu_offload`` internally.

    Fallback path (when no ``model.safetensors.index.json`` exists): rank-0
    ``from_pretrained`` + ``set_model_state_dict`` broadcast via
    :func:`fsdp2_shard`.

    Both paths produce identical sharded models (same FSDP2 wrap layout, root
    unsharded, decoder layers DTensor-sharded across the FSDP mesh).
    """
    import json

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    # Resolve ckpt_path: local dir as-is, otherwise HF Hub ID — rank 0 downloads,
    # others wait at the barrier so we don't contend on the cache lock.
    resolved_path: str | None = ckpt_path
    if not os.path.isdir(ckpt_path):
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            snapshot_download = None
        if snapshot_download is not None:
            if rank == 0:
                resolved_path = snapshot_download(ckpt_path)
            if is_initialized():
                barrier()
            if rank != 0:
                resolved_path = snapshot_download(ckpt_path)
        else:
            resolved_path = None

    index_path = (
        os.path.join(resolved_path, "model.safetensors.index.json") if resolved_path else None
    )
    if resolved_path is not None and index_path is not None and os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return _load_via_parallel_read(
            ckpt_path=resolved_path,
            device=device,
            rank=rank,
            world_size=world_size,
            trust_remote_code=trust_remote_code,
            mp_policy=mp_policy,
            cpu_offload=cpu_offload,
            weight_map=weight_map,
        )

    # Fallback: rank-0 from_pretrained + broadcast via fsdp2_shard.
    hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=trust_remote_code)
    dtype = getattr(hf_config, "torch_dtype", None) or torch.bfloat16

    if rank == 0:
        src_model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        src_model.eval()
        src_state_dict = src_model.state_dict()
    else:
        src_model = None
        src_state_dict = {}

    # Meta skeleton on every rank; include_buffers=False keeps computed buffers
    # (e.g. rotary inv_freq) real on CPU instead of stranded on meta.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            hf_config, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
    model.eval()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    sharded = fsdp2_shard(
        model,
        device,
        src_state_dict=src_state_dict,
        mp_policy=mp_policy,
        cpu_offload=cpu_offload,
    )
    del src_model, src_state_dict
    return sharded


class DistributedProcessGroup:
    """A convenient wrapper around torch.distributed.ProcessGroup objects."""

    def __init__(self, group: torch.distributed.ProcessGroup | int | None = None):
        """Initialize the distributed process group."""
        self.group = group

    def is_initialized(self) -> bool:
        """Check if the distributed process group is initialized."""
        return backend() == "torch" and self.group != -1

    def rank(self) -> int:
        """Get the rank of the current process group."""
        return rank(group=self.group) if self.is_initialized() else -1

    def world_size(self) -> int:
        """Get the world size of the current process group."""
        return size(group=self.group) if self.is_initialized() else -1

    def __repr__(self) -> str:
        return f"group: {self.group}, initialized: {self.is_initialized()}, world size: {self.world_size()}"

    @staticmethod
    def get_dist_syncd_obj(
        obj: Any,
        groups: "DistributedProcessGroup | list[DistributedProcessGroup]",
        op: Callable,
    ):
        """Get the distributed synchronized object across the specified distributed groups."""

        def _get_dist_syncd_obj_across_group(obj, group: DistributedProcessGroup):
            if not group.is_initialized():
                return obj
            obj_list = [None] * group.world_size()
            torch.distributed.all_gather_object(obj_list, obj, group=group.group)
            return op(obj_list)

        for group in groups if isinstance(groups, list) else [groups]:
            obj = _get_dist_syncd_obj_across_group(obj, group)

        return obj


class ParallelState:
    """A class to manage various parallel groups such as data parallel, tensor parallel etc.

    Specify the parallel groups of type :class:`torch.distributed.ProcessGroup` for the current module.
    If the parallel group is not used, it should be set to `-1`.
    if a parallel group is `None`, it will use the default PyTorch distributed process group which is the whole world.
    """

    def __init__(
        self,
        data_parallel_group: torch.distributed.ProcessGroup | int | None = None,
        tensor_parallel_group: torch.distributed.ProcessGroup | int | None = -1,
        expert_model_parallel_group: torch.distributed.ProcessGroup | int | None = -1,
    ):
        """Initialize the parallel state."""
        self.data_parallel_group = DistributedProcessGroup(data_parallel_group)
        self.tensor_parallel_group = DistributedProcessGroup(tensor_parallel_group)
        self.expert_model_parallel_group = DistributedProcessGroup(expert_model_parallel_group)

    def __repr__(self) -> str:
        parallel_groups = (
            f"data_parallel_group: {self.data_parallel_group}, "
            f"tensor_parallel_group: {self.tensor_parallel_group}, "
            f"expert_model_parallel_group: {self.expert_model_parallel_group}"
        )
        return parallel_groups


def get_group(ranks: list[int]):
    """Returns the process group if torch.distributed.is_initialized()."""
    # NCCL has an issue with calling barrier. So we just use the gloo backebnd for group barriers.
    return torch.distributed.new_group(ranks, backend="gloo") if is_initialized() else None


def is_dtensor_sharded(model):
    """Returns True if the model is using DTensor."""
    return any(isinstance(param, DTensor) for param in model.parameters()) or any(
        isinstance(param, DTensor) for param in model.buffers()
    )


class FileLock:
    """Mutex object for writing to a file atomically using the O_EXCL directive on Unix filesystems."""

    def __init__(
        self,
        lockfile_path: str,
        all_acquire: bool = False,
        poll_time: float = 0.25,
    ):
        """Constructor.

        Args:
            lockfile_path: Path to a nonexistent file to be used as the locking mechanism.
            all_acquire: Will keep retrying to acquire a lock if True.
            poll_time: Sleep interval between retries.
        """
        self.lockfile_path = lockfile_path
        self.all_acquire = all_acquire
        self.poll_time = poll_time
        self.handle = None

    def try_acquire(self):
        try:
            self.handle = os.open(self.lockfile_path, os.O_CREAT | os.O_EXCL)
            return True
        except FileExistsError:
            return False

    def wait(self):
        while os.path.exists(self.lockfile_path):
            time.sleep(self.poll_time)

    def release(self):
        if self.handle is not None:
            os.close(self.handle)
        os.remove(self.lockfile_path)

    def __enter__(self):
        while True:
            if self.try_acquire() or not self.all_acquire:
                break
            self.wait()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
