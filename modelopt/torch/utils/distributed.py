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
from torch.distributed.tensor import DTensor

__all__ = [
    "DistributedProcessGroup",
    "Fsdp2StateDictAdapter",
    "ParallelState",
    "backend",
    "barrier",
    "fsdp2_shard",
    "fsdp2_wrap",
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


def fsdp2_wrap(model, override_cls_name: str | None = None, mp_policy=None):
    """Apply FSDP2 ``fully_shard`` to each decoder layer of ``model``.

    Decoder layers are auto-detected via
    ``modelopt.torch.quantization.utils.layerwise_calib.LayerActivationCollector.get_decoder_layers``.
    Pass ``override_cls_name`` to force a specific transformer block class. Pass
    ``mp_policy`` (a ``torch.distributed.fsdp.MixedPrecisionPolicy``) to control
    compute / reduce dtype; default ``None`` means no upcast / downcast.

    The root module is intentionally not sharded so embeddings / lm_head stay as
    plain tensors (avoids DTensor / plain-tensor mismatches with modelopt's
    layerwise forward patching).
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
    for layer in layers:
        fully_shard(layer, **fsdp_kwargs)
    return model


def fsdp2_shard(model, device, rank, mp_policy=None):
    """Shard a loaded model across the current process group.

    Expects rank 0 to pass a real CPU model and other ranks to pass a meta
    skeleton with matching structure. After this call every rank holds its
    per-rank GPU shard, populated from rank 0's source.

    Steps: stash ``_original_architectures`` (FSDP2 may mutate
    ``model.config.architectures``); capture rank-0's state_dict; ``fsdp2_wrap``
    per decoder layer; ``to_empty`` allocates per-rank GPU shard storage;
    ``set_model_state_dict(broadcast_from_rank0=True)`` streams the data; freeze
    params (needed by ``patch_fsdp_mp_dtypes``' trainable-only check).
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    model._original_architectures = list(model.config.architectures or [])
    cpu_state_dict = model.state_dict() if rank == 0 else {}

    fsdp2_wrap(model, mp_policy=mp_policy)
    model.to_empty(device=device)
    set_model_state_dict(
        model,
        cpu_state_dict,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )
    # TODO(temp workaround): FSDP2's _init_mp_dtypes asserts uniform dtype across
    # trainable params. patch_fsdp_mp_dtypes narrows the check to trainable-only;
    # freezing here makes trainable empty so mixed-dtype models (Nemotron-H, etc.)
    # pass. PTQ doesn't need gradients anyway.
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
