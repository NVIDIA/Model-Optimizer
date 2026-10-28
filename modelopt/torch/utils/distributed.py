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
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, fully_shard
from torch.distributed.tensor import DTensor

__all__ = [
    "DistributedProcessGroup",
    "DistributedState",
    "DistributionPlan",
    "ParallelState",
    "all_reduce",
    "backend",
    "barrier",
    "broadcast_from",
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
    "set_distribution_plan",
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
    for name, module in model.named_modules():
        recorded = getattr(module, _PLAN_ATTR, None)
        if recorded is not None:
            where = f"{type(module).__name__} {name!r}" if name else type(module).__name__
            raise ValueError(
                f"{where} is already recorded as distributed by {recorded!r}; a model is "
                f"distributed and its plan recorded once, so {plan!r} cannot be recorded over it."
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
    plan = getattr(model, _PLAN_ATTR, None)
    if plan is not None:
        return plan
    sharded = is_fsdp2_model(model)
    if sharded or size() > 1:
        how = "is FSDP2-sharded" if sharded else f"is held in a run of {size()} ranks"
        raise RuntimeError(
            f"{type(model).__name__} {how}, but has no distribution plan recorded. Record how it "
            "is distributed with set_distribution_plan(); a module added to a model after its plan "
            "was recorded has none, so read the plan off that model."
        )
    return DistributionPlan()


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


def _received(sent: tuple[bool, Any], rank: int, label: str) -> BaseException:
    """The exception to raise for what ``rank`` sent (:func:`_sendable`) on failing ``label``."""
    exited, detail = sent
    if exited:
        return SystemExit(detail)
    return RuntimeError(f"rank {rank} failed to compute {label}: {detail}")


def _encoded(value: Any) -> bytes:
    """``value`` serialized for :func:`broadcast_from` and :func:`all_reduce`, and loaded back once.

    The load-back makes a value that cannot round-trip fail inside the caller's guard, which tells
    every rank, rather than inside the collective on one rank alone. It loads on the producing rank
    only: a load that depends on the loading rank (a class only the producer can import) can still
    fail on a peer alone. torch.save stands in for pickle, which bandit refuses under modelopt/.
    """
    buffer = io.BytesIO()
    torch.save(value, buffer)
    payload = buffer.getvalue()
    _decoded(payload)
    return payload


def _decoded(payload: bytes) -> Any:
    """The value :func:`_encoded` serialized into ``payload``.

    A full load, not :func:`_deserialize`'s weights_only one: the values are whatever the callers
    compute, of any class, and the peers are this job's own ranks, whose payloads torch's object
    collectives unpickle just as fully. A tensor in the value loads on the CPU on every rank, the
    producing one too: the device it was saved from is the producer's, not this rank's.
    """
    return torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=False,  # this job's own ranks sent the bytes over its process group: no file
    )


def broadcast_from(label: str, fn: Callable[[], _V], rank: int = 0) -> _V:
    """Return ``fn()``, run on ``rank`` only, on every rank.

    For a value ranks computing it for themselves could disagree on (their environment, GPU or
    files). If ``fn`` raises or its value does not round-trip (:func:`_encoded`), every rank raises
    rather than wait for the broadcast: ``rank`` its own error, the others a SystemExit with its
    code if it exited, else a RuntimeError. Every rank, ``rank`` too, gets a copy loaded from the
    bytes sent, with any tensor in it on the CPU. Collective: every rank calls it at the same point
    with the same ``label`` and ``rank``; a receiving rank whose ``label`` differs raises, but
    ``rank`` itself cannot tell and returns its value. In a single process ``fn``'s value comes back
    as it is; a rank of a launch whose process group is not set up raises.
    """
    _refuse_a_launch_without_its_group(label)
    if not 0 <= rank < size():
        raise ValueError(f"rank={rank} is not a rank of this {size()}-rank run.")
    if not is_distributed_run():
        return fn()
    sending = torch.distributed.get_rank() == rank
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
    # are out of step, and its serialized value or its error.
    message = [(label, payload, _sendable(error))]
    torch.distributed.broadcast_object_list(message, group_src=rank)
    sent_label, payload, sent_error = message[0]
    if error is not None:
        raise error
    if sent_error is not None:
        raise _received(sent_error, rank, label)
    if sent_label != label:
        raise RuntimeError(
            f"Ranks broadcast out of step: this rank broadcast {label!r} while rank {rank} sent "
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
    single process it gets ``fn()``'s value as it is. A rank of a launch whose process group is not
    set up raises.

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
    device: torch.device | str

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
            rank(), size(), torch.device(f"cuda:{local_rank()}") if launched else device
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
    """Auto-detect a HF causal-LM's decoder layers and FSDP2 ``fully_shard`` each one.

    By default (``shard_root=True``) the root module is wrapped too, so embed/lm_head/norm are
    sharded instead of replicated per rank; pass ``shard_root=False`` to leave the root replicated
    (only decoder layers sharded). Returns the detected decoder layers so callers can reuse the
    detection result.

    Parameters whose dtype differs from the model's dominant one are excluded from the wrap (see
    :func:`_off_dtype_params`), since FSDP2 rejects a shard group that mixes dtypes. They stay
    replicated and are moved onto the shards' device, which ``fully_shard`` does not do for the
    params it ignores.
    """
    # Lazy import: layerwise_calib imports this module at top level (circular).
    from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

    decoder_layers = LayerActivationCollector.get_decoder_layers(model)
    if decoder_layers is None:
        raise RuntimeError(
            "Could not auto-detect decoder layers; FSDP2 wrap requires a standard HF causal-LM layout."
        )

    fsdp_kwargs: dict[str, Any] = {"reshard_after_forward": True}
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
    ignored_params = _off_dtype_params(model)
    if ignored_params:
        fsdp_kwargs["ignored_params"] = ignored_params

    # Snapshot/restore config.architectures: some HF builders mutate it during fully_shard.
    config = getattr(model, "config", None)
    architectures = list(getattr(config, "architectures", []) or [])
    for layer in decoder_layers:
        fully_shard(layer, **fsdp_kwargs)
    if shard_root:
        fully_shard(model, **fsdp_kwargs)
    if ignored_params:
        _move_to_fsdp_device(model, ignored_params)
    if config is not None and architectures:
        config.architectures = architectures

    return decoder_layers


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
