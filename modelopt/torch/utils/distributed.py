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
from contextlib import contextmanager, suppress
from datetime import timedelta
from typing import Any
from warnings import warn

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed.tensor import DTensor

__all__ = [
    "DistributedProcessGroup",
    "Fsdp2StateDictAdapter",
    "ParallelState",
    "backend",
    "barrier",
    "fsdp2_shard",
    "fsdp2_wrap",
    "fsdp_aware_forward_loop",
    "init_params_on_meta",
    "is_available",
    "is_initialized",
    "is_master",
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

    Decoder layers are auto-detected via
    ``modelopt.torch.quantization.utils.layerwise_calib.LayerActivationCollector.get_decoder_layers``.
    Pass ``override_cls_name`` to force a specific transformer block class. Pass
    ``mp_policy`` (a ``torch.distributed.fsdp.MixedPrecisionPolicy``) to control
    compute / reduce dtype; default ``None`` means no upcast / downcast.

    Pass ``device`` to stream each layer to that device just before sharding it
    (avoids holding the full model on GPU simultaneously). When ``device`` is
    ``None``, layers are sharded on whatever device they're already on.

    Set ``cpu_offload=True`` to attach FSDP2's ``CPUOffloadPolicy`` to each
    wrapped layer. Each rank's shard then lives on CPU between forward passes
    and is streamed to GPU per-layer (H2D + all-gather + compute + reshard +
    D2H). Useful when the per-rank decoder shard is the binding GPU constraint
    (e.g., 200B+ models on tight GPU budgets) or when you want headroom for a
    larger calibration batch. Adds PCIe traffic per layer per batch; on
    setups where the model already fits comfortably it usually slows the run.

    The root module is intentionally NOT sharded — ``embed_tokens`` and
    ``lm_head`` stay as plain replicated tensors. This costs ~few-GiB per rank
    (full copies of embed + lm_head) but unifies the layerwise and non-layerwise
    code paths: a DTensor-wrapped ``embed_tokens.weight`` raises a "mixed Tensor
    / DTensor" error when modelopt's layerwise calibration passes plain
    ``input_ids`` into the embedding lookup, and FSDP2's root pre-forward hook
    doesn't auto-wrap LongTensor inputs.
    """
    from torch.distributed.fsdp import fully_shard

    from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

    if override_cls_name:
        layers = [m for m in model.modules() if type(m).__name__ == override_cls_name]
        if not layers:
            raise RuntimeError(f"No modules of class {override_cls_name!r} found in model")
    else:
        layers = LayerActivationCollector.get_decoder_layers(model)
        if layers is None:
            raise RuntimeError(
                "Could not auto-detect decoder layers; pass override_cls_name explicitly."
            )
    fsdp_kwargs: dict[str, Any] = {"reshard_after_forward": True}
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy
    if cpu_offload:
        from torch.distributed.fsdp import CPUOffloadPolicy

        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
    for layer in layers:
        if device is not None:
            layer.to(device)
        fully_shard(layer, **fsdp_kwargs)
    return model


@contextmanager
def init_params_on_meta():
    """Replicate ``accelerate.init_empty_weights(include_buffers=False)``.

    Inside this context, ``nn.Module.register_parameter`` is patched so newly
    registered parameters land on the ``meta`` device (zero CPU bytes). Buffer
    registration is NOT patched — buffers are computed normally on CPU during
    ``__init__`` (e.g. ``Qwen2RotaryEmbedding.__init__`` produces a real CPU
    ``inv_freq`` from config).

    Use around ``from_config(...)`` on non-rank-0 ranks to build a meta-skeleton
    that ``fsdp2_shard`` will materialize via ``set_model_state_dict``
    broadcast from rank 0.
    """
    original = nn.Module.register_parameter

    def patched(self, name, param):
        original(self, name, param)
        if param is not None:
            p = self._parameters[name]
            self._parameters[name] = nn.Parameter(p.to("meta"), requires_grad=p.requires_grad)

    nn.Module.register_parameter = patched
    try:
        yield
    finally:
        nn.Module.register_parameter = original


def fsdp2_shard(model, device, rank, src_state_dict=None, mp_policy=None, cpu_offload=False):
    """Shard a model across the current process group (accelerate-style rank-0 load).

    Caller contract: ``model`` is built on every rank with params on ``meta`` and
    buffers on CPU (use ``init_params_on_meta`` around ``from_config``). Rank 0
    additionally passes ``src_state_dict`` captured from a real CPU model loaded
    via ``from_pretrained``; other ranks pass ``None`` or ``{}``.

    Set ``cpu_offload=True`` to attach FSDP2's ``CPUOffloadPolicy`` to wrapped
    layers (each rank's shard lives on CPU between forwards). See
    ``fsdp2_wrap`` docstring for the trade-off.

    Root is never sharded (see ``fsdp2_wrap`` docstring). embed_tokens and
    lm_head stay as plain replicated tensors on every rank.

    Steps (each timed and logged per-rank for diagnostics):
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
    materialize_device = torch.device("cpu") if cpu_offload else device

    def _materialize(t):
        is_meta_dtensor = isinstance(t, DTensor) and t._local_tensor.is_meta
        if is_meta_dtensor or (not isinstance(t, DTensor) and t.is_meta):
            return torch.empty_like(t, device=materialize_device)
        return t.to(materialize_device)

    model._apply(_materialize)

    if src_state_dict is not None:
        set_model_state_dict(
            model,
            src_state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )

    if cpu_offload:
        # FSDP-managed (DTensor) params stay on CPU — FSDP2 streams them per layer.
        # Move everything else (root-level plain params + all buffers) to GPU now.
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

    if hasattr(model, "tie_weights"):
        model.tie_weights()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


def shard_dataloader(loader, rank: int, world_size: int):
    """Wrap a DataLoader with a DistributedSampler so each rank sees a unique shard.

    Preserves the input loader's ``batch_size``, ``collate_fn``, ``num_workers``,
    and ``pin_memory``.
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

    ``mtq.quantize`` strips the FSDP wrapper before calling ``forward_loop``,
    handing the user the unwrapped inner module. Calling the unwrapped module
    bypasses FSDP's pre/post-forward hooks (no all-gather, no reshard), which
    breaks calibration on FSDP2. The closure returned here captures the outer
    *wrapped* model and ignores the ``unwrapped_model`` argument that
    ``mtq.quantize`` passes in.

    Used by ``examples/llm_ptq/hf_ptq.py`` under ``--use_fsdp2``.

    TODO: ``modelopt/torch/quantization/plugins/transformers_trainer.py`` (the
    QLoRA path) currently has the same logic inlined inside ``_quantize_model``.
    Consolidate that call site to use this helper too.
    """
    from tqdm import tqdm

    def calibrate(_unwrapped_model):
        for batch in tqdm(dataloader, desc="Calibrating"):
            if device is not None and isinstance(batch, dict):
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
    assert meta_dict is not None

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


def load_state_dict_into_fsdp2_layer(layer: nn.Module, full_state_dict: dict) -> None:
    """Load full (replicated) tensors into an FSDP2-wrapped layer's DTensor local shards.

    Each rank already has the full tensor; we just need to shard locally.
    Uses ``set_model_state_dict(broadcast_from_rank0=False)`` — each rank holds the
    full tensor in ``full_state_dict``, so no collective is needed; the helper just
    slices each rank's local shard from the full tensor.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    set_model_state_dict(
        layer,
        full_state_dict,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False),
    )


class Fsdp2StateDictAdapter:
    """Adapter exposing ``.get_state_dict(model)`` for FSDP2-sharded models.

    Satisfies the ``accelerator=`` kwarg of
    ``modelopt.torch.export.unified_export_hf._export_transformers_checkpoint``.
    Backed by ``get_model_state_dict`` which materializes a full unsharded state
    dict on every rank (with CPU offload to bound peak GPU memory during gather).
    """

    def get_state_dict(self, model):
        """Return the full unsharded state dict gathered from FSDP2 shards (CPU-offloaded)."""
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        return get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )


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
