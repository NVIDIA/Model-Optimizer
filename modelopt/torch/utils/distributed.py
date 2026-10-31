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
import types
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, TypeVar
from warnings import warn

import torch
import torch.distributed
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
)
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
    "recorded_distribution_plan",
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


def _deserialize(tensor: torch.Tensor, size: int | None = None, map_location=None) -> Any:
    buffer = tensor.numpy().tobytes()
    if size is not None:
        buffer = buffer[:size]
    # Explicit so TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD cannot re-enable pickle for peer payloads. Custom
    # classes sent through broadcast/allgather must be registered with add_safe_globals.
    return torch.load(io.BytesIO(buffer), map_location=map_location, weights_only=True)


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


# A plan is a value the collectives carry, and they load what the ranks send with weights_only. A
# loaded copy skips __post_init__: it is the plan the sending rank built, and checked there.
torch.serialization.add_safe_globals([DistributionPlan])

_PLAN_ATTR = "_modelopt_distribution_plan"


def _recorded_plan(model: torch.nn.Module) -> tuple[str, DistributionPlan] | None:
    """Which of ``model`` and its submodules records a plan, named, and that plan; else ``None``."""
    for name, module in model.named_modules():
        recorded = getattr(module, _PLAN_ATTR, None)
        if recorded is not None:
            where = f"{type(module).__name__} {name!r}" if name else type(module).__name__
            return where, recorded
    return None


def set_distribution_plan(model: torch.nn.Module, plan: DistributionPlan) -> None:
    """Record that ``model`` is distributed as ``plan``, on it and on every submodule it has now.

    Code handed a submodule, such as a VLM's language model, then reads the same plan as code handed
    the whole model; a module added afterwards records none. Code that distributes a model records
    its plan here before calibrating or exporting the model. The plan says how the model is placed
    over the run; the :class:`ParallelState` a quantized module holds names the process groups its
    collectives run on.

    A model is distributed once, and its plan recorded once: recording a plan on a model that, or
    any of whose submodules, already records one raises, the same plan included, and leaves the
    model as it was. No code reloads or re-distributes a model it loaded.

    In a run of several ranks, a plan for another number of ranks raises too, and records nothing:
    a single process's plan there would be taken at its word, and the model replicated. A plan
    recorded before the run's group is set up is not checked, nor one in a run of one, where a rank
    can stand in for several.
    """
    if size() > 1 and plan.world_size != size():
        raise ValueError(
            f"{plan!r} is for {plan.world_size} ranks, but the run it is recorded in has "
            f"{size()}; a plan's world_size is the ranks that hold the model."
        )
    if (found := _recorded_plan(model)) is not None:
        where, recorded = found
        raise ValueError(
            f"{where} is already recorded as distributed by {recorded!r}; a model is distributed "
            f"and its plan recorded once, so {plan!r} cannot be recorded over it."
        )
    for module in model.modules():
        setattr(module, _PLAN_ATTR, plan)


def distribution_plan(model: torch.nn.Module) -> DistributionPlan:
    """How ``model`` is distributed: the plan recorded on it (:func:`set_distribution_plan`).

    A model with no plan recorded is a single process's, unless it shows otherwise: one that is
    FSDP2-sharded, or held in a run of several ranks, raises rather than be taken for one. That
    includes a module added to a model after its plan was recorded: a module does not know the
    model it was added to, so read the plan off that model. A run is its live process group: before
    a launch sets its group up, each of its processes runs alone with a model of its own, so a model
    with no plan reads as a single process's there.
    """
    plan = recorded_distribution_plan(model)
    if plan is not None:
        return plan
    sharded = is_fsdp2_model(model)
    if sharded or size() > 1:
        how = "is FSDP2-sharded" if sharded else f"is held in a run of {size()} ranks"
        raise RuntimeError(
            f"{type(model).__name__} {how}, but has no distribution plan recorded. Load it with "
            "load_pretrained, or record how it is distributed with set_distribution_plan(); a "
            "module added to a model after its plan was recorded has none, so read the plan off "
            "that model."
        )
    return DistributionPlan()


def recorded_distribution_plan(model: torch.nn.Module) -> DistributionPlan | None:
    """The plan :func:`set_distribution_plan` recorded on ``model``, or ``None`` if none was.

    For code that also serves models distributed without a plan, such as Megatron's, which
    :func:`distribution_plan` refuses in a run of several ranks.
    """
    return getattr(model, _PLAN_ATTR, None)


_V = TypeVar("_V")


def _sendable(error: BaseException | None) -> tuple[bool, Any] | None:
    """``error`` as this rank sends it to the others (see :func:`_received`), ``None`` for none.

    An error goes as its text, since an exception need not pickle; a SystemExit as its code, for
    every rank to exit with.
    """
    if error is None:
        return None
    if isinstance(error, SystemExit):
        code = error.code
        return True, code if code is None or isinstance(code, int) else _text(code)
    return False, f"{type(error).__name__}: {_text(error)}"


def _text(value: Any) -> str:
    """``str(value)``, or a stand-in if that raises.

    Formatting a failure must not keep its rank from the collective that tells the others.
    """
    try:
        return str(value)
    except Exception:
        return f"<unprintable {type(value).__name__}>"


def _refuse_a_launch_without_its_group(label: str) -> None:
    """Raise if this process is a rank of a multi-rank launch whose process group is not set up.

    Each rank would otherwise compute ``label`` alone, and the ranks could disagree with nothing
    telling them so.
    """
    if is_distributed_launch() and not is_initialized():
        raise RuntimeError(
            f"{label!r} needs this {os.environ['WORLD_SIZE']}-rank launch's process group, which is "
            "not set up yet; compute it inside distributed_run(), which sets the group up."
        )


def _received(sent: tuple[bool, Any], src: int, label: str) -> BaseException:
    """The exception to raise for what rank ``src`` sent (:func:`_sendable`) on failing ``label``."""
    exited, detail = sent
    if exited:
        return SystemExit(detail)
    return RuntimeError(f"rank {src} failed to compute {label}: {detail}")


def _encoded(value: Any) -> torch.Tensor:
    """``value`` serialized for :func:`broadcast_from` and :func:`all_reduce`, and loaded back once.

    The load-back makes a value that cannot round-trip fail inside the caller's guard, which tells
    every rank, rather than inside the collective on one rank alone. It loads on the producing rank
    only: a load that depends on the loading rank (a class only the producer can import) can still
    fail on a peer alone.
    """
    payload = _serialize(value)
    _decoded(payload)
    return payload


def _decoded(payload: torch.Tensor) -> Any:
    # On the CPU on every rank, the producing one too: a tensor's saved device is the producer's.
    return _deserialize(payload, map_location="cpu")


def broadcast_from(label: str, fn: Callable[[], _V], src: int = 0) -> _V:
    """Return ``fn()``, run on ``src`` only, on every rank.

    For a value ranks computing it for themselves could disagree on (their environment, GPU or
    files). If ``fn`` raises or its value does not round-trip (:func:`_encoded`), every rank raises
    rather than wait for the broadcast: ``src`` its own error, the others a SystemExit with its
    code if it exited, else a RuntimeError. Every rank, ``src`` too, gets a copy loaded from the
    bytes sent, with any tensor in it on the CPU; the value must load with ``weights_only``, so a
    class in it must be registered with ``torch.serialization.add_safe_globals``. Collective: every
    rank calls it at the same point with the same ``label`` and ``src``; a receiving rank whose
    ``label`` differs raises, but ``src`` itself cannot tell and returns its value. In a single
    process ``fn``'s value comes back as it is; a rank of a launch whose process group is not set up
    raises.
    """
    _refuse_a_launch_without_its_group(label)
    if not 0 <= src < size():
        raise ValueError(f"src={src} is not a rank of this {size()}-rank run.")
    if not is_distributed_run():
        return fn()
    sending = rank() == src
    payload: Any = None
    error = None
    if sending:
        try:
            payload = _encoded(fn())
        except BaseException as e:
            # Still send: the other ranks would otherwise wait in the broadcast for this one. A
            # SystemExit or KeyboardInterrupt too, which argparse-style checks raise.
            payload, error = None, e
    # One message from the sending rank: its label, so a rank broadcasting another learns the two
    # are out of step, and its serialized value or its error. The object collective pickles the
    # message itself between the job's ranks; weights_only bounds the value inside it.
    message = [(label, payload, _sendable(error))]
    torch.distributed.broadcast_object_list(message, group_src=src)
    sent_label, payload, sent_error = message[0]
    if error is not None:
        raise error
    if sent_error is not None:
        raise _received(sent_error, src, label)
    if sent_label != label:
        raise RuntimeError(
            f"Ranks broadcast out of step: this rank broadcast {label!r} while rank {src} sent "
            f"{sent_label!r}. Every rank must broadcast them in the same order."
        )
    return _decoded(payload)


def all_reduce(label: str, fn: Callable[[], _V], combine: Callable[[list[_V]], _V]) -> _V:
    """Return ``combine`` of every rank's ``fn()``, in rank order, on every rank.

    E.g. ``combine=min`` for a batch size each rank's memory bounds. If ``fn`` fails on any rank, or
    its value does not round-trip (:func:`_encoded`), every rank raises instead of waiting in the
    gather: if any rank's failure is not a SystemExit, a rank with such an error raises it and the
    others a RuntimeError naming the first; else each rank exits, with its own code or the first
    one's. Collective, with the same ``label`` on every rank; ranks out of step raise. ``combine``
    gets copies loaded from the bytes the ranks sent, with any tensor in them on the CPU; in a
    single process it gets ``fn()``'s value as it is. The values must load with ``weights_only``, so
    a class in them must be registered with ``torch.serialization.add_safe_globals``. A rank of a
    launch whose process group is not set up raises.

    Unlike :func:`allreduce`, it runs ``fn`` itself, so a rank whose ``fn`` fails still joins.
    """
    _refuse_a_launch_without_its_group(label)
    if not is_distributed_run():
        return combine([fn()])
    payload, error = None, None
    try:
        payload = _encoded(fn())
    except BaseException as e:
        # Still join the gather: the other ranks would otherwise wait in it for this one, a
        # SystemExit or KeyboardInterrupt included.
        payload, error = None, e
    mine = (label, payload, _sendable(error))
    gathered: list[Any] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, mine)
    failures = [(rank_, sent) for rank_, (_, _, sent) in enumerate(gathered) if sent is not None]
    if failures:
        # One rule on every rank: any error that is not an exit makes every rank raise one, for an
        # exiting rank would wait in distributed_run's exit barrier for an aborted peer.
        errors = [(rank_, sent) for rank_, sent in failures if not sent[0]]
        failed_rank, sent = (errors or failures)[0]
        if error is not None and isinstance(error, SystemExit) == sent[0]:
            raise error
        raise _received(sent, failed_rank, label)
    labels = sorted({rank_label for rank_label, _, _ in gathered})
    if labels != [label]:
        raise RuntimeError(
            f"Ranks reduced out of step: {labels} at the same point. Every rank must reduce "
            "them in the same order."
        )
    # Every rank's value is loaded from what it sent, this rank's too, so the result shares nothing
    # with the callers'.
    return combine([_decoded(rank_payload) for _, rank_payload, _ in gathered])


@dataclass(frozen=True)
class DistributedState:
    """This process's place in a run: its rank, the run's size, and the device it computes on.

    :func:`distributed_run` yields it, for a single process too (rank 0 of a run of one).
    """

    rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        """Whether this is rank 0, the rank that writes the run's outputs."""
        return self.rank == 0


@contextmanager
def distributed_run(
    timeout: timedelta | None = None, device: torch.device | str = "cuda"
) -> Iterator[DistributedState]:
    """Own the run's process group for the block, and yield this process's state.

    On a distributed launch (``WORLD_SIZE>1``) this binds this rank's CUDA device, and sets the
    process group up with ``timeout`` unless the caller set one up already. It yields this rank's
    :class:`DistributedState`, on that CUDA device, and refuses any other ``device``. Without a
    launch it sets nothing up, and the state is on ``device``. The state's rank and size are always
    the live group's, which the block's collectives run on: rank 0 of a run of one for a single
    process, and this rank's place in a group the caller set up without the launch's environment.
    A ``timeout`` cannot apply to a group of several ranks the caller set up, so passing one then
    warns. A single process leaves it unused, without a warning, also in a one-rank group of the
    caller's: no rank there waits on a peer, so an entry point may pass one for its launches only.

    On exit, even if the body raises, it destroys the group the run set up: the launch's, or one the
    body set up, such as the one-rank group a single process's FSDP2 load shards through. A group
    that was already there when the run began is the caller's, and is left as it is. A rank whose
    body raised, in a run of several ranks on a group that was not there when it began, exits
    through :func:`abort` instead, and the launcher tears the job down: destroying the group would
    wait on peers that may sit in a collective this rank never reaches. In a run that found the
    caller's group, the error goes on to the caller, which owns the peers.

    A ``SystemExit`` is not an error: the group is destroyed as on a normal exit, which waits on
    every rank, so the exit has to be every rank's, as one raised inside :func:`broadcast_from` or
    :func:`all_reduce` is. A rank that exits alone waits in that teardown until the group times out.
    """
    launched = is_distributed_launch()
    if launched and torch.device(device) != torch.device("cuda"):
        raise ValueError(
            f"distributed_run(device={device!r}): a distributed launch runs each rank on its own "
            "CUDA device, chosen from LOCAL_RANK; pass device='cuda' or run a single process."
        )
    group_was_there = is_initialized()
    if timeout is not None and is_distributed_run():
        warn(
            f"distributed_run(timeout={timeout}) does not apply: the process group was set up before "
            "the run, and a group's timeout is fixed when it is set up. Pass it where the group is."
        )
    if launched:
        # setup() binds the device even to a group the caller set up, which may not have bound one:
        # the state's device is then the one this rank's NCCL collectives run on.
        setup(timeout=timeout)
    # The body's own exception: sys.exc_info() would also report one a caller up the stack is
    # handling, which would make a clean exit from a run entered in an except block abort the rank,
    # or destroy its group without the barrier.
    exc: BaseException | None = None
    try:
        yield DistributedState(
            rank(),
            size(),
            torch.device(f"cuda:{local_rank()}") if launched else torch.device(device),
        )
    except BaseException as e:
        exc = e
        raise
    finally:
        # If the body is unwinding due to an exception, PRINT it here BEFORE anything that waits on
        # the peers. A per-rank exception (only some ranks raise) leaves the rest inside a
        # calibration a2a collective, so this rank's cleanup barrier or destroy_process_group would
        # DEADLOCK with the original exception never surfaced. A run of several ranks that owns its
        # group therefore aborts the rank: abort() prints the traceback and exits. Anywhere else the
        # error goes on to the caller, traceback and all.
        owns_group = not group_was_there and is_initialized()
        # Every rank reaches a SystemExit, an intentional exit: it is not an error to report.
        if exc is not None and not isinstance(exc, SystemExit):
            print(
                f"[distributed_run][rank {rank()}] body raised {type(exc).__name__}: {_text(exc)}",
                flush=True,
            )
            if owns_group and size() > 1:
                abort()
        if owns_group:
            _destroy_group(exc)


class _StridedIterableShard(IterableDataset):
    """Rank-strided shard of a streaming ``IterableDataset`` (no ``len()`` required).

    ``DistributedSampler`` only works with map-style datasets. Streaming calibration datasets (e.g. the
    Nemotron image-text ``NemotronTarPlusJsonlIterable``) are iterable, so each rank instead builds its
    own copy of the stream, the same on every rank as :func:`shard_dataloader` requires,
    and takes slot ``rank`` from every complete group of ``world_size`` consecutive samples. Any
    trailing incomplete group is dropped so every rank yields exactly the same count -- preserving
    the equal-per-rank-batch invariant that keeps per-step EP collectives from hanging when one rank
    runs short. A stream shorter than ``world_size`` has no complete group, so it is refused rather
    than leaving every rank with nothing to calibrate on. It is iterated in the loader's own process
    (:func:`shard_dataloader` refuses workers for a stream). Every rank still reads and decodes every
    sample, keeping one in ``world_size``: for a stream costly to decode, a map-style dataset, or a
    source that shards itself, saves that work.
    ``vlm_dataset_utils._ShardedIterable`` cannot stand in: it pads every rank to a count its own loader
    knows, while this wraps any loader's stream, which may know none.
    """

    def __init__(self, base: IterableDataset, rank: int, world_size: int):
        self.base = base
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        it = iter(self.base)
        for index in itertools.count():
            group = list(itertools.islice(it, self.world_size))
            if len(group) < self.world_size:
                if index == 0:
                    # shard_dataloader requires the same stream on every rank and worker, so all of
                    # them refuse it together.
                    raise ValueError(
                        f"The calibration stream has {len(group)} sample(s), fewer than the "
                        f"{self.world_size} ranks: no rank would get a sample to calibrate on."
                    )
                return  # drop incomplete trailing group -> equal counts across ranks
            yield group[self.rank]


def shard_dataloader(loader: DataLoader, rank: int, world_size: int) -> DataLoader:
    """Wrap a DataLoader so each rank sees a unique shard.

    For map-style datasets a ``DistributedSampler`` is used. ``drop_last=False`` keeps per-rank batch
    counts equal (else a rank exits calibration early and hangs the others on the per-step collectives),
    at the cost of the sampler repeating up to ``world_size - 1`` samples to pad the even split. It
    replaces the loader's sampler, so a shuffling one becomes the rank's unshuffled share, and one that
    picks samples itself (a subset, a sampler over another source than the loader's dataset, a
    random sampler drawing its own number of samples or with replacement, or a batch sampler other
    than a plain ``BatchSampler``) is refused rather than silently widened. For streaming
    ``IterableDataset`` datasets (which have no ``len()``
    and so can't use a sampler) each rank strides its own copy of the stream via
    :class:`_StridedIterableShard`. That stream must yield the same samples in the same order on
    every rank, or the shards overlap and the ranks' step counts can differ, which hangs the
    collectives: one that shuffles needs a fixed seed of its own, as the ranks' RNGs need not agree.
    A stream loaded with workers is refused: a dataset that splits itself across them by
    ``get_worker_info()``, as a Hugging Face streaming dataset does, would be split twice and
    silently lose samples.

    Forwards all non-sampler DataLoader settings from ``loader`` (batching, workers, pinning,
    prefetch, init fn, generator, ...). With one rank there is nothing to shard: ``loader`` itself
    comes back, its sampler and shuffling included.
    """
    if world_size == 1:
        return loader
    batching = loader.batch_sampler
    if not isinstance(loader.dataset, IterableDataset):
        # A plain BatchSampler, built from batch_size or passed in, batches what its sampler picks;
        # with auto-batching off there is none, and the loader's sampler picks each item.
        if type(batching) is BatchSampler:
            picker = batching.sampler
        else:
            picker = loader.sampler if batching is None else batching
        known = type(picker) in (SequentialSampler, RandomSampler, DistributedSampler)
        # A sampler over another source than the loader's dataset, or a RandomSampler drawing its
        # own number of samples or with replacement, picks which and how many it reads, as a subset
        # does; one that orders or shuffles the whole dataset does not.
        source = getattr(picker, "data_source", getattr(picker, "dataset", ()))
        narrowing = known and (
            len(source) != len(loader.dataset)
            or (
                isinstance(picker, RandomSampler)
                and (picker.replacement or picker.num_samples != len(loader.dataset))
            )
        )
        if not known or narrowing:
            raise ValueError(
                f"shard_dataloader: the loader's {type(picker).__name__} picks which samples it "
                "reads, and the rank's DistributedSampler would replace it; pass a loader over "
                "just the samples to calibrate on (a torch.utils.data.Subset) instead."
            )
    if isinstance(loader.dataset, IterableDataset):
        if loader.num_workers:
            raise ValueError(
                f"shard_dataloader: a streaming loader with num_workers={loader.num_workers} is "
                "refused: a dataset that splits itself across the workers by get_worker_info() "
                "would be split twice, by worker and by rank, and lose samples. Load it with "
                "num_workers=0."
            )
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
        batch_size=loader.batch_size if batching is None else batching.batch_size,
        drop_last=loader.drop_last if batching is None else batching.drop_last,
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
    and before the first forward. Only a forward quantizer insertion rebound is planted again: a
    wrapper still in place -- transformers' from the load, on a module quantization left alone, or
    one planted here on an earlier pass -- is kept, since wrapping it again would run its transforms
    twice. ``ep_mesh`` is the mesh the routed experts are sharded over; without one there is nothing
    to re-plant.
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

    n = 0
    for name, mod in model.named_modules():
        style_name = _get_parameter_tp_plan(parameter_name=name, tp_plan=plan, is_weight=False)
        if style_name not in ("torch_a2a_experts", "torch_a2a_router"):
            continue
        # install_forward() -- and the clobber it causes -- only exist on transformers >=5.16. Older
        # releases install their TP input/output transforms as forward PRE-HOOKS, which survive
        # ModelOpt rebinding ``module.forward``: nothing to re-plant. Checked per style, the one
        # about to be called.
        style = ALL_PARALLEL_STYLES.get(style_name)
        if not hasattr(style, "install_forward"):
            continue
        # install_forward stores a plain function as the instance's forward, and quantizer
        # insertion (bind_forward_method) a bound method over it: only the latter lost its wrapper.
        if not isinstance(vars(mod).get("forward"), types.MethodType):
            continue
        style.install_forward(mod, ep_mesh)
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


def distributed_forward_loop(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | str | None = None,
    ep_mesh: "torch.distributed.device_mesh.DeviceMesh | None" = None,
) -> Callable[[torch.nn.Module], None]:
    """Build an ``mtq.quantize`` ``forward_loop`` for a distributed (e.g. expert-parallel) model.

    Moves each batch onto ``device``, when given, and runs the captured ``model`` forward (the EP
    all-reduce / all-to-all collectives fire inside that forward). A model whose plan
    (:func:`distribution_plan`) is expert-parallel passes the ``ep_mesh`` its experts are sharded
    over, for the all-to-all forwards that quantizer insertion drops to be re-installed before the
    first forward; any other model passes none. ``mtq.quantize`` hands the
    ``forward_loop`` callback an inner module that this closure ignores, calling the captured
    top-level ``model`` instead so the full distributed forward (and its collectives) runs.

    The forwards run without gradients and with KV/state caching disabled (``_disable_use_cache``), as
    the single-node ``_forward_loop`` runs them: a hybrid Mamba/attention model mutates its cache in
    place. Each batch is a mapping of the forward's keyword arguments, as ``model(**batch)`` takes;
    only its tensor values move to ``device``, so tensors nested in a value (a list of images, say)
    must already be there.
    """
    # Local import to avoid any import cycle within modelopt.torch.utils.
    from modelopt.torch.utils.dataset_utils import _disable_use_cache

    plan = distribution_plan(model)
    if plan.is_expert_parallel != (ep_mesh is not None):
        raise ValueError(
            f"{plan!r}: an expert-parallel model calibrates with the ep_mesh its routed experts "
            "are sharded over, and any other model without one."
        )
    if ep_mesh is not None and ep_mesh.size() != plan.ep_size:
        raise ValueError(f"{plan!r}: the ep_mesh spans {ep_mesh.size()} ranks, not {plan.ep_size}.")

    def calibrate(_inner_model):
        _reinstall_a2a_forwards(model, ep_mesh)
        with _disable_use_cache(model), torch.no_grad():
            # Hand the caching allocator's unused pool back: NCCL allocates its buffers outside it.
            _release_cached_cuda_memory()
            for batch in tqdm(dataloader, desc="Calibrating", disable=not is_master()):
                if not isinstance(batch, Mapping):
                    raise TypeError(
                        f"distributed_forward_loop: the loader yields {type(batch).__name__}, not a "
                        "mapping of the forward's keyword arguments."
                    )
                if device is not None:
                    batch = {
                        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()
                    }
                model(**batch)
            # Again for what runs next (amax sync, export collectives), past the forwards' pool.
            _release_cached_cuda_memory()

    return calibrate


def cleanup():
    """Cleans up the distributed environment.

    The barrier is skipped when unwinding from an error, since peers may be blocked in a collective
    this rank will never reach. ``SystemExit`` is treated as a clean exit (every rank reaches it).
    That is not sufficient on its own -- ``destroy_process_group`` below blocks for the same reason
    -- so error paths must call :func:`abort` before reaching this ``finally``.
    """
    _destroy_group(sys.exc_info()[1])


def _destroy_group(exc: BaseException | None) -> None:
    """:func:`cleanup` for a rank unwinding from ``exc``, or from nothing if it is ``None``.

    For a caller that knows what it unwinds from: :func:`cleanup` reads ``sys.exc_info()``, which
    also reports an exception a caller up the stack is handling.
    """
    if is_initialized():
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
    exc = sys.exception()
    if isinstance(exc, SystemExit):
        raise exc
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def is_fsdp2_model(model) -> bool:
    """Return True if any submodule of ``model`` has been wrapped with FSDP2 ``fully_shard``."""
    return any(isinstance(m, FSDPModule) for m in model.modules())


def fsdp2_wrap(model, shard_root=True, mp_policy=None, cpu_offload: bool = False):
    """Forward to :func:`~modelopt.torch.utils.fsdp2_utils.fsdp2_wrap`, where it moved."""
    # Lazy import: fsdp2_utils imports modelopt.torch.utils.logging, which imports this module.
    from modelopt.torch.utils.fsdp2_utils import fsdp2_wrap as wrap

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
