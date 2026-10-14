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
import gc
import io
import itertools
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, TypeVar
from warnings import warn

import torch
import torch.distributed
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

__all__ = [
    "DistributedProcessGroup",
    "DistributedState",
    "DistributionPlan",
    "ParallelState",
    "all_reduce",
    "backend",
    "barrier",
    "broadcast_from",
    "distributed_forward_loop",
    "distributed_run",
    "distribution_plan",
    "expert_parallel_state",
    "fsdp2_wrap",
    "gather_bytes",
    "gather_state_dicts",
    "is_available",
    "is_distributed_launch",
    "is_distributed_run",
    "is_fsdp2_model",
    "is_initialized",
    "is_master",
    "rank",
    "set_distribution_plan",
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


def is_distributed_launch() -> bool:
    """True if launched distributed (WORLD_SIZE>1), readable BEFORE the process group is set up.

    Env-based, so parse-time code can detect the launch mode before :func:`distributed_run` sets the
    process group up. Once it has, :func:`is_distributed_run` reads the live group.
    """
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_distributed_run() -> bool:
    """True if a real multi-rank job is running: the process group is initialized and world size > 1.

    Runtime state (checks the live process group), unlike :func:`is_distributed_launch` which reads
    only the launch env. A single process or a 1-rank group returns False. Backend-agnostic -- true
    for any backend that initialized ``torch.distributed`` (HF EP/DP loader, Megatron, plain DDP).
    """
    return is_initialized() and size() > 1


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
    # Explicit so TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD cannot re-enable pickle for peer payloads. Custom
    # classes sent through broadcast/allgather must be registered with add_safe_globals.
    return torch.load(io.BytesIO(buffer), weights_only=True)


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


def gather_bytes(
    data: bytes, dst: int, group=None, chunk_bytes: int = 1 << 28
) -> list[bytearray] | None:
    """Gathers byte strings to global rank ``dst`` only, without unpickling anything on the way.

    Unlike ``torch.distributed.gather_object`` it never runs pickle, and it moves ``chunk_bytes`` per
    rank at a time to bound device memory. Returns the per-rank bytes on ``dst`` and ``None`` elsewhere.
    """
    n = size(group)
    device = torch.device("cuda" if torch.distributed.get_backend(group) == "nccl" else "cpu")
    local = (
        torch.frombuffer(bytearray(data), dtype=torch.uint8)
        if data
        else torch.empty(0, dtype=torch.uint8)
    )
    sizes_t = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(n)]
    torch.distributed.all_gather(sizes_t, torch.tensor([len(data)], device=device), group=group)
    sizes = [int(t.item()) for t in sizes_t]

    is_dst = torch.distributed.get_rank() == dst
    out = [bytearray(s) for s in sizes] if is_dst else None
    for off in range(0, max(sizes), chunk_bytes):
        length = min(chunk_bytes, max(sizes) - off)
        buf = torch.zeros(length, dtype=torch.uint8, device=device)
        piece = local[off : off + length]
        buf[: piece.numel()] = piece.to(device)
        bufs = [torch.empty_like(buf) for _ in range(n)] if is_dst else None
        torch.distributed.gather(buf, bufs, dst=dst, group=group)
        if bufs is not None and out is not None:
            for i, b in enumerate(bufs):
                k = min(length, max(0, sizes[i] - off))
                out[i][off : off + k] = b[:k].cpu().numpy().tobytes()
    return out


def gather_state_dicts(state: dict, dst: int, group=None) -> Iterator[dict]:
    """Gathers ``state`` from every rank of ``group`` to global rank ``dst``, loaded with ``weights_only``.

    Yields one rank's dict at a time on ``dst`` (to bound host memory) and nothing elsewhere. The collective
    runs eagerly, so every rank must call this before iterating.
    """
    buf = io.BytesIO()
    torch.save(state, buf)
    gathered = gather_bytes(buf.getvalue(), dst, group)
    del buf

    def _load() -> Iterator[dict]:
        while gathered:
            # Explicit weights_only so TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD cannot re-enable pickle for peers.
            yield torch.load(io.BytesIO(gathered.pop(0)), map_location="cpu", weights_only=True)

    return _load()


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


@dataclass(frozen=True)
class DistributionPlan:
    """How a model is distributed over the ranks of a run.

    * ``world_size`` -- the ranks that hold the model; ``1`` for a single process.
    * ``ep_size`` -- the expert-parallel degree: an MoE's routed experts are sharded over groups of
      ``ep_size`` ranks. ``1`` shards none.
    * ``dense`` -- how the rest of the model, all of it without expert parallelism, is placed:
      ``"replicate"`` gives every rank a whole copy, ``"fsdp"`` FSDP2-shards it.

    Every rank calibrates on its own data, so data parallelism spans the whole world, and each expert
    shard is replicated ``edp = world_size / ep_size`` times.
    """

    world_size: int = 1
    ep_size: int = 1
    dense: Literal["replicate", "fsdp"] = "replicate"

    def __post_init__(self) -> None:
        """Reject a plan no load could carry out, before any rank starts on it."""
        for name in ("world_size", "ep_size"):
            # Below 1 a degree is meaningless: ep_size=-2 would divide the world and read as no
            # expert parallelism, ep_size=0 would divide by zero.
            if getattr(self, name) < 1:
                raise ValueError(f"{name}={getattr(self, name)} must be at least 1.")
        if self.world_size % self.ep_size != 0:
            raise ValueError(f"ep_size={self.ep_size} must divide world_size={self.world_size}")
        if self.dense not in ("replicate", "fsdp"):
            raise ValueError(f"dense={self.dense!r}: expected 'replicate' or 'fsdp'.")

    @property
    def edp(self) -> int:
        """Expert-data-parallel degree: how many ranks hold a replica of each expert shard."""
        return self.world_size // self.ep_size

    @property
    def is_expert_parallel(self) -> bool:
        """Whether routed experts are sharded across ranks."""
        return self.ep_size > 1

    @property
    def is_fsdp_dense(self) -> bool:
        """Whether the dense part is FSDP2-sharded rather than replicated."""
        return self.dense == "fsdp"


_PLAN_ATTR = "_modelopt_distribution_plan"


def set_distribution_plan(model: torch.nn.Module, plan: DistributionPlan) -> None:
    """Record that ``model`` is distributed as ``plan``, on it and on every submodule.

    Code handed a submodule, such as a VLM's language model, then reads the same plan as code handed
    the whole model. ``load_pretrained`` records the plan it loads by; code that distributes a model
    itself records it here before calibrating or exporting the model.

    A model is distributed once, and its plan recorded once: recording a plan on a model that, or
    any of whose submodules, already records one raises, the same plan included, and leaves the
    model as it was. No code reloads or re-distributes a model it loaded.
    """
    for module in model.modules():
        recorded = getattr(module, _PLAN_ATTR, None)
        if recorded is not None:
            raise ValueError(
                f"{type(module).__name__} is already recorded as distributed by {recorded!r}; a "
                f"model is distributed and its plan recorded once, so {plan!r} cannot be recorded "
                "over it."
            )
    for module in model.modules():
        setattr(module, _PLAN_ATTR, plan)


def distribution_plan(model: torch.nn.Module) -> DistributionPlan:
    """How ``model`` is distributed: the plan recorded on it (:func:`set_distribution_plan`).

    A model with no plan recorded is a single process's, unless it shows otherwise: one that is
    FSDP2-sharded, or held in a run of several ranks, raises rather than be taken for one.
    """
    plan = getattr(model, _PLAN_ATTR, None)
    if plan is not None:
        return plan
    sharded = is_fsdp2_model(model)
    if sharded or size() > 1:
        how = "is FSDP2-sharded" if sharded else f"is held in a run of {size()} ranks"
        raise RuntimeError(
            f"{type(model).__name__} {how}, but has no distribution plan recorded. Load it with "
            "load_pretrained, or record how it is distributed with set_distribution_plan()."
        )
    return DistributionPlan()


_V = TypeVar("_V")


def broadcast_from(label: str, fn: Callable[[], _V], rank: int = 0) -> _V:
    """Return ``fn()`` run on ``rank`` only, on every rank.

    For a value that depends on what each rank reads for itself -- the environment, its GPU, files
    on its node -- so that ranks computing it for themselves could disagree. If ``fn`` raises, every
    rank raises, instead of waiting in a broadcast ``rank`` never sends. Ranks broadcasting
    different labels at the same point fail. Collective: every rank calls it at the same point,
    with the same ``rank``. Outside a distributed run ``fn`` runs alone.
    """
    if not is_distributed_run():
        return fn()
    if not 0 <= rank < size():
        raise ValueError(f"rank={rank} is not a rank of this {size()}-rank run.")
    sending = torch.distributed.get_rank() == rank
    value: Any = None
    error = None
    if sending:
        try:
            value = fn()
        except Exception as e:
            # Still send: the other ranks would otherwise wait in the broadcast for this one.
            error = e
    # One message from the sending rank: its label, so a rank broadcasting another learns the two
    # are out of step, and its value or its error. The error goes as text: the other ranks only
    # need to stop, and an exception is not guaranteed to pickle.
    message = [(label, value, None if error is None else f"{type(error).__name__}: {error}")]
    torch.distributed.broadcast_object_list(message, group_src=rank)
    sent_label, value, sent_error = message[0]
    if error is not None:
        raise error
    if sent_error is not None:
        raise RuntimeError(f"rank {rank} failed to compute {label}: {sent_error}")
    if sent_label != label:
        raise RuntimeError(
            f"Ranks broadcast out of step: this rank broadcast {label!r} while rank {rank} sent "
            f"{sent_label!r}. Every rank must broadcast them in the same order."
        )
    return value


def all_reduce(label: str, fn: Callable[[], _V], combine: Callable[[list[_V]], _V]) -> _V:
    """Return ``combine`` of every rank's ``fn()``, on every rank.

    ``combine`` takes the ranks' values in rank order and returns the one every rank gets, e.g.
    ``min`` for a batch size each rank's memory bounds, where one rank's choice could run another
    out of memory. If ``fn`` raises on any rank, every rank raises, instead of waiting in the gather
    that rank never joins. Ranks reducing different labels at the same point fail. Collective;
    outside a distributed run ``fn`` runs alone.
    """
    if not is_distributed_run():
        return combine([fn()])
    value, error = None, None
    try:
        value = fn()
    except Exception as e:
        # Still join the gather: the other ranks would otherwise wait in it for this one.
        error = e
    mine = (label, value, None if error is None else f"{type(error).__name__}: {error}")
    gathered: list[Any] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, mine)
    if error is not None:
        raise error
    for rank_, (_, _, rank_error) in enumerate(gathered):
        if rank_error is not None:
            raise RuntimeError(f"rank {rank_} failed to compute {label}: {rank_error}")
    labels = sorted({rank_label for rank_label, _, _ in gathered})
    if labels != [label]:
        raise RuntimeError(
            f"Ranks reduced out of step: {labels} at the same point. Every rank must reduce "
            "them in the same order."
        )
    # The gathered values are unpickled copies, so the result shares nothing with the callers'.
    return combine([rank_value for _, rank_value, _ in gathered])


@dataclass(frozen=True)
class DistributedState:
    """This process's place in a run: its rank, the run's size, and the device it computes on.

    :func:`distributed_run` yields it, for a single process too (rank 0 of a run of one).
    """

    rank: int
    world_size: int
    device: torch.device | str

    @property
    def is_main(self) -> bool:
        """Whether this is rank 0, the rank that writes the run's outputs."""
        return self.rank == 0


@contextmanager
def distributed_run(timeout: timedelta | None = None, device: torch.device | str = "cuda"):
    """Own the run's process group for the block, and yield this process's state.

    On a distributed launch (``WORLD_SIZE>1``) this sets the process group up, binding this rank's
    CUDA device, with ``timeout``. It yields this rank's :class:`DistributedState`, on that CUDA
    device. A single process sets nothing up and is rank 0 of a run of one, on ``device``.

    On exit, even if the body raises, it destroys the group the run set up: the launch's, or one the
    body set up, such as the one-rank group a single process's FSDP2 load shards through. A group
    that was already there when the run began is the caller's, and is left as it is. A rank whose
    body raised in a run of several ranks exits through :func:`abort` instead, and the launcher
    tears the job down: destroying the group would wait on peers that may sit in a collective this
    rank never reaches.
    """
    launched = is_distributed_launch()
    group_was_there = is_initialized()
    if launched:
        setup(timeout=timeout)
    try:
        yield (
            DistributedState(rank(), size(), torch.device(f"cuda:{local_rank()}"))
            if launched
            else DistributedState(0, 1, device)
        )
    finally:
        # If the body is unwinding due to an exception, PRINT it here BEFORE anything that waits on
        # the peers. A per-rank exception (only some ranks raise) leaves the rest inside a
        # calibration a2a collective, so this rank's cleanup barrier or destroy_process_group would
        # DEADLOCK with the original exception never surfaced. A run of several ranks that owns its
        # group therefore aborts the rank: abort() prints the traceback and exits.
        exc = sys.exc_info()[1]
        owns_group = not group_was_there and is_initialized()
        if exc is not None:
            print(
                f"[distributed_run][rank {os.environ.get('RANK', '?')}] body raised "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if owns_group and not isinstance(exc, SystemExit) and size() > 1:
                abort()
            traceback.print_exc()
            sys.stderr.flush()
        if owns_group:
            cleanup()


def expert_parallel_state(model: torch.nn.Module) -> "ParallelState | None":
    """Return the ``ParallelState`` of the model's routed experts, or ``None`` if none has one.

    The expert-parallel loader seeds every routed-expert module with a ``ParallelState`` that carries the
    expert-model-parallel group (``load_and_prepare_ep``); no other module of a HuggingFace model has one.
    """
    for module in model.modules():
        state = getattr(module, "_parallel_state", None)
        if state is not None and state.expert_model_parallel_group.is_initialized():
            return state
    return None


class _StridedIterableShard(IterableDataset):
    """Rank-strided shard of a streaming ``IterableDataset`` (no ``len()`` required).

    ``DistributedSampler`` only works with map-style datasets. Streaming calibration datasets (e.g. the
    Nemotron image-text ``NemotronTarPlusJsonlIterable``) are iterable, so each rank instead builds its
    own identical stream (deterministic seed) and takes slot ``rank`` from every complete group of
    ``world_size`` consecutive samples. Any trailing incomplete group is dropped so every rank yields
    exactly the same count -- preserving the equal-per-rank-batch invariant that keeps per-step EP
    collectives from hanging when one rank runs short. A stream shorter than ``world_size`` has no
    complete group, so it is refused rather than leaving every rank with nothing to calibrate on. Each
    DataLoader worker process iterates its own copy of the stream, so the workers split the groups
    between them rather than each yielding them all.
    ``vlm_dataset_utils._ShardedIterable`` cannot stand in: it pads every rank to a count its own loader
    knows, while this wraps any loader's stream, which may know none, and splits across its workers.
    """

    def __init__(self, base: IterableDataset, rank: int, world_size: int):
        self.base = base
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        info = get_worker_info()
        num_workers, worker = (info.num_workers, info.id) if info is not None else (1, 0)
        it = iter(self.base)
        for index in itertools.count():
            group = list(itertools.islice(it, self.world_size))
            if len(group) < self.world_size:
                if index == 0:
                    # Every rank and worker sees the same stream, so all of them refuse it together.
                    raise ValueError(
                        f"The calibration stream has {len(group)} sample(s), fewer than the "
                        f"{self.world_size} ranks: no rank would get a sample to calibrate on."
                    )
                return  # drop incomplete trailing group -> equal counts across ranks
            if index % num_workers == worker:
                yield group[self.rank]


def shard_dataloader(loader, rank: int, world_size: int):
    """Wrap a DataLoader so each rank sees a unique shard.

    For map-style datasets a ``DistributedSampler`` is used. ``drop_last=False`` keeps per-rank batch
    counts equal (else a rank exits calibration early and hangs the others on the per-step collectives),
    at the cost of the sampler repeating up to ``world_size - 1`` samples to pad the even split. For
    streaming ``IterableDataset`` datasets (which have no ``len()`` and so can't use a sampler) each rank
    strides its own deterministic copy of the stream via :class:`_StridedIterableShard`. Every DataLoader
    worker must see that whole stream: a dataset that splits itself across workers by
    ``get_worker_info()`` would be split twice and lose samples, so load one with ``num_workers=0``.

    Forwards all non-sampler DataLoader settings from ``loader`` (workers, pinning,
    prefetch, init fn, generator, ...).
    """
    if isinstance(loader.dataset, IterableDataset):
        return DataLoader(
            _StridedIterableShard(loader.dataset, rank, world_size),
            batch_size=loader.batch_size,
            drop_last=loader.drop_last,
            collate_fn=loader.collate_fn,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            timeout=loader.timeout,
            worker_init_fn=loader.worker_init_fn,
            multiprocessing_context=loader.multiprocessing_context,
            generator=loader.generator,
            prefetch_factor=loader.prefetch_factor,
            persistent_workers=loader.persistent_workers,
            pin_memory_device=getattr(loader, "pin_memory_device", ""),
        )
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
        drop_last=loader.drop_last,
        sampler=sampler,
        collate_fn=loader.collate_fn,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers,
        pin_memory_device=getattr(loader, "pin_memory_device", ""),
    )


def _reinstall_a2a_forwards(model, ep_mesh) -> int:
    """Re-plant the torch_a2a TP forward wrappers that quantizer insertion clobbered.

    transformers installs its TP input/output transforms by REPLACING ``module.forward`` with an
    instance attribute (TensorParallelLayer.install_forward). For torch_a2a_experts that wrapper is
    what passes the EP ``process_group`` into the experts forward. ModelOpt's quantizer insertion
    rebinds ``module.forward`` on the same instance, dropping the wrapper -- after which
    ``torch_a2a_experts_forward`` receives ``process_group=None`` and raises.

    Called at the top of the calibration loop: after conversion (so it wraps the QUANTIZED forward)
    and before the first forward. Idempotent via a per-module flag, so repeated calibration passes
    do not stack wrappers (which would inject the process_group argument twice). ``ep_mesh`` is the
    mesh the routed experts are sharded over; without one there is nothing to re-plant.
    """
    if ep_mesh is None:
        return 0
    try:
        from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES
    except ImportError:  # older transformers
        from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES
    try:
        from transformers.distributed.tensor_parallel import _get_parameter_tp_plan
    except ImportError:
        from transformers.integrations.tensor_parallel import _get_parameter_tp_plan

    plan = getattr(model, "tp_plan", None) or {}
    if not plan:
        return 0

    # install_forward() -- and the clobber it causes -- only exist on transformers >=5.16. Older
    # releases install their TP input/output transforms as forward PRE-HOOKS, which survive
    # ModelOpt rebinding ``module.forward``, so there is nothing to re-plant and calling the
    # missing method would AttributeError. No-op rather than fail on 5.9/5.13.
    if not any(
        hasattr(ALL_PARALLEL_STYLES.get(s), "install_forward")
        for s in ("torch_a2a_experts", "torch_a2a_router")
    ):
        return 0
    n = 0
    for name, mod in model.named_modules():
        if getattr(mod, "_modelopt_a2a_forward_reinstalled", False):
            continue
        style_name = _get_parameter_tp_plan(parameter_name=name, tp_plan=plan, is_weight=False)
        if (
            style_name in ("torch_a2a_experts", "torch_a2a_router")
            and style_name in ALL_PARALLEL_STYLES
        ):
            ALL_PARALLEL_STYLES[style_name].install_forward(mod, ep_mesh)
            mod._modelopt_a2a_forward_reinstalled = True
            n += 1
    if n and is_master():
        print(
            f"[ep] re-installed {n} torch_a2a forward wrapper(s) clobbered by quantizer insertion",
            flush=True,
        )
    return n


def _release_cached_cuda_memory():
    """Hand the caching allocator's reserved-but-unused pool back to the driver.

    Only a process that started CUDA has a pool to hand back; elsewhere the collection alone costs
    about a second.
    """
    if torch.cuda.is_initialized():
        gc.collect()
        torch.cuda.empty_cache()


def distributed_forward_loop(model, dataloader, device=None, ep_mesh=None):
    """Build an ``mtq.quantize`` ``forward_loop`` for a distributed (e.g. expert-parallel) model.

    Moves each batch onto ``device``, when given, and runs the captured ``model`` forward (the EP
    all-reduce / all-to-all collectives fire inside that forward). A model whose plan
    (:func:`distribution_plan`) is expert-parallel passes the ``ep_mesh`` its experts are sharded
    over, for the all-to-all forwards that quantizer insertion drops to be re-installed before the
    first forward; any other model passes none. ``mtq.quantize`` hands the
    ``forward_loop`` callback an inner module that this closure ignores, calling the captured
    top-level ``model`` instead so the full distributed forward (and its collectives) runs.

    KV/state caching is disabled for the duration (via ``_disable_use_cache``), mirroring the
    single-node ``_forward_loop``. This distributed path previously omitted the guard, so it ran
    calibration at ``config.use_cache`` (default ``True``); for hybrid Mamba/attention models
    (e.g. NemotronH) the cache is mutated in-place, which both wastes memory and breaks
    calibration correctness.
    """
    # Local import to avoid any import cycle within modelopt.torch.utils.
    from modelopt.torch.utils.dataset_utils import _disable_use_cache

    plan = distribution_plan(model)
    if plan.is_expert_parallel != (ep_mesh is not None):
        raise ValueError(
            f"{plan!r}: an expert-parallel model calibrates with the ep_mesh its routed experts "
            "are sharded over, and any other model without one."
        )

    def calibrate(_inner_model):
        _reinstall_a2a_forwards(model, ep_mesh)
        with _disable_use_cache(model):
            # Reclaim the caching allocator's reserved-but-unused pool left by the load/quantizer-insertion
            # memory peak. Without this, torch hoards ~the whole GPU (e.g. 91 GB live weights vs 275 GB
            # reserved), leaving ~0 GB free, so the distributed a2a's NCCL comm buffers cannot be allocated
            # ("Failed to CUDA calloc") at the very first all_to_all. Freeing the cache hands that memory
            # back to the driver for NCCL + the calibration activations.
            _release_cached_cuda_memory()
            for batch in tqdm(dataloader, desc="Calibrating", disable=not is_master()):
                if device is not None:
                    batch = {
                        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()
                    }
                model(**batch)
            # Symmetric to the pre-forward reclaim above, and for the same reason one stage later:
            # the calibration forwards leave their activations in the caching allocator's reserved
            # pool, so whatever runs next (amax sync, export collectives) can find the device full
            # even though little is live. Measured on Kimi-K3 (EP64, 184 GiB cards) right after the
            # single calib forward: alloc 86.0 GiB but reserved 156.6 GiB -> only 0.7 GiB free, and
            # the job then hung until the 2 h wall clock instead of exporting. NCCL comm buffers are
            # allocated OUTSIDE the torch allocator, so a full reserved pool starves them.
            _release_cached_cuda_memory()

    return calibrate


def cleanup():
    """Cleans up the distributed environment.

    The barrier is skipped when unwinding from an error, since peers may be blocked in a collective
    this rank will never reach. ``SystemExit`` is treated as a clean exit (every rank reaches it).
    That is not sufficient on its own -- ``destroy_process_group`` below blocks for the same reason
    -- so error paths must call :func:`abort` before reaching this ``finally``.
    """
    if is_initialized():
        exc = sys.exc_info()[1]
        if exc is None or isinstance(exc, SystemExit):
            with suppress(Exception):
                barrier()
        torch.distributed.destroy_process_group()


def abort(exit_code: int = 1) -> None:
    """Print the active exception and exit this rank immediately.

    Call from an ``except`` block in a distributed entrypoint. Both a barrier and
    ``destroy_process_group`` stall when peers are blocked in a collective this rank will never
    reach, and the traceback only prints once the enclosing ``finally`` returns -- so the run looks
    hung rather than failed. Exiting lets the launcher (e.g. torchrun) terminate the peers.
    ``SystemExit`` is re-raised instead, since every rank reaches an intentional exit.
    """
    exc = sys.exc_info()[1]
    if isinstance(exc, SystemExit):
        raise exc
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def is_fsdp2_model(model) -> bool:
    """Return True if any submodule of ``model`` has been wrapped with FSDP2 ``fully_shard``."""
    return any(isinstance(m, FSDPModule) for m in model.modules())


def _off_dtype_params(model) -> set[torch.nn.Parameter]:
    """Params whose dtype differs from the model's dominant (by element count) param dtype.

    FSDP2 needs one dtype per shard group, but HF models routinely keep a few params in fp32 for
    stability (e.g. MoE router gates). Pass these to ``fully_shard(ignored_params=...)``.

    TODO: Drop this and shard the off-dtype params once a stable PyTorch release includes FSDP2
    mixed-precision parameter dtype support (already on nightly).
    """
    numel_by_dtype: dict[torch.dtype, int] = {}
    for param in model.parameters():
        numel_by_dtype[param.dtype] = numel_by_dtype.get(param.dtype, 0) + param.numel()
    if len(numel_by_dtype) <= 1:
        return set()

    # Lazy import: logging imports this module at top level (circular).
    from modelopt.torch.utils.logging import warn_rank_0

    dominant = max(numel_by_dtype, key=lambda d: numel_by_dtype[d])
    off_dtype = {n: p for n, p in model.named_parameters() if p.dtype != dominant}
    off_numel = sum(numel_by_dtype[d] for d in numel_by_dtype if d != dominant)
    names = sorted(off_dtype)
    warn_rank_0(
        f"Model has mixed parameter dtypes {set(numel_by_dtype)}; FSDP2 needs one dtype per shard "
        f"group, so {len(names)} non-{dominant} parameter(s) "
        f"({100 * off_numel / sum(numel_by_dtype.values()):.2f}% of elements) will stay replicated "
        f"rather than sharded: {names[:3]}{' ...' if len(names) > 3 else ''}"
    )
    return set(off_dtype.values())


def _move_to_fsdp_device(model, params: set[torch.nn.Parameter]) -> None:
    """Move ``params`` onto the device FSDP2 computes on for ``model``.

    ``fully_shard`` only moves the params it manages, so ignored ones would be stranded on
    whatever device the caller built the model on. Meta params are left alone for deferred init.

    The device comes from a sharded param's mesh rather than its local shard: under
    ``cpu_offload`` the shard rests on CPU while compute still happens on the accelerator.
    """
    # Lazy import: logging imports this module at top level (circular).
    from modelopt.torch.utils.logging import warn_rank_0

    mesh = next((p.device_mesh for p in model.parameters() if isinstance(p, DTensor)), None)
    if mesh is None:
        warn_rank_0(
            f"FSDP2 sharded no parameter of {type(model).__name__}, so the compute device for "
            f"{len(params)} unsharded off-dtype parameter(s) cannot be determined; leaving them "
            "where they are. Move them to the compute device or the forward will fail."
        )
        return

    device = (
        torch.device("cpu")
        if mesh.device_type == "cpu"
        else torch.device(mesh.device_type, getattr(torch, mesh.device_type).current_device())
    )
    for param in params:
        if not param.is_meta and param.device != device:
            param.data = param.data.to(device)


def fsdp2_wrap(model, shard_root=True, mp_policy=None, cpu_offload: bool = False):
    """FSDP2-shard a HF causal LM's decoder layers, and the root unless ``shard_root=False``.

    It loads a model, so it lives with the loaders now:
    :func:`modelopt.torch.utils.plugins.model_load_utils.fsdp2_wrap`. This keeps the old path.
    """
    from modelopt.torch.utils.plugins.model_load_utils import fsdp2_wrap as wrap

    return wrap(model, shard_root=shard_root, mp_policy=mp_policy, cpu_offload=cpu_offload)


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
            t = src_state_dict[name].to(device)
        else:
            t = torch.empty(shape, dtype=dtype, device=device)
        torch.distributed.broadcast(t, src=src, group=pg)
        out[name] = t
    return out


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
