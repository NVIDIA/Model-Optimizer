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

import copy
import functools
import inspect
import io
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import timedelta
from typing import Any, Generic, TypeVar
from warnings import warn

import torch
import torch.distributed
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, fully_shard
from torch.distributed.tensor import DTensor

__all__ = [
    "DistributedProcessGroup",
    "DistributionPlan",
    "ModelOptSession",
    "ParallelState",
    "Synced",
    "backend",
    "barrier",
    "fsdp2_wrap",
    "gather_bytes",
    "gather_state_dicts",
    "get_active_mo_session",
    "is_available",
    "is_distributed_launch",
    "is_distributed_run",
    "is_fsdp2_model",
    "is_initialized",
    "is_master",
    "modelopt_session",
    "rank",
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

    Env-based (does NOT touch the guarded session -- see :func:`get_active_mo_session`), so parse-time code
    can detect the launch mode before :func:`modelopt_session` exists. After the session is entered, prefer
    ``get_active_mo_session().distributed``.
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


class DistributionPlan:
    """How a HuggingFace MoE model is distributed for PTQ, as decided by ``decide_distribution_plan``.

    * ``ep_size``   -- expert-parallel degree: the routed experts are sharded over ``ep_size`` ranks.
    * ``fsdp_size`` -- dense-FSDP degree: ``1`` replicates the dense (non-expert) part on every rank; ``>1``
                       FSDP2-shards it alongside the expert-parallel experts.

    Every rank calibrates on its own data, so data-parallelism spans the whole world, and each expert
    shard is replicated ``edp = world_size / ep_size`` times.
    """

    def __init__(self, world_size: int, ep_size: int = 1, fsdp_size: int = 1) -> None:
        """Build the plan, rejecting an expert-parallel degree that cannot divide the world."""
        if world_size % ep_size != 0:
            raise ValueError(f"ep_size={ep_size} must divide world_size={world_size}")
        self.world_size = world_size
        self.ep_size = ep_size
        self.fsdp_size = fsdp_size

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
        """Whether the dense part is FSDP-sharded rather than replicated."""
        return self.fsdp_size > 1

    def __repr__(self) -> str:
        dense = f"fsdp_size={self.fsdp_size}" if self.is_fsdp_dense else "dense=replicate"
        return (
            f"DistributionPlan(world={self.world_size}, ep_size={self.ep_size}, edp={self.edp}, "
            f"{dense})"
        )


_T = TypeVar("_T")
_V = TypeVar("_V")


class Synced(Generic[_T]):
    """A fact of the session every rank holds the same value of, declared by its type.

    ``plan: Synced[DistributionPlan]`` on the session makes ``session.plan`` this handle. Read the
    value with ``session.plan.value``: local, ``None`` until it is set. Set it with
    :meth:`broadcast_from`, one rank computing it, or :meth:`all_reduce`, every rank computing its own
    and the results combined; both are collective, so every rank calls them at the same point.
    Assigning ``session.plan`` raises, so nothing communicates behind an ``=``. Declare one only for a
    fact a later stage reads back: a one-off value nothing reads back is
    :meth:`ModelOptSession.broadcast_from` or :meth:`ModelOptSession.all_reduce`, which keep nothing.
    """

    def __init__(self, holder: "_SyncedHolder", name: str) -> None:
        """The fact ``name`` of ``holder``."""
        self._holder = holder
        self._name = name

    @property
    def value(self) -> _T | None:
        """The value every rank holds, read locally; ``None`` until one is set."""
        return self._holder._synced.get(self._name)

    def broadcast_from(self, fn: Callable[[], _T], rank: int = 0) -> _T:
        """Set the fact to ``fn()`` run on ``rank`` only; every rank holds and returns that value.

        See :meth:`ModelOptSession.broadcast_from`.
        """
        label = self._holder._label(self._name)
        return self._hold(self._holder._sync_session().broadcast_from(label, fn, rank))

    def all_reduce(self, fn: Callable[[], _T], combine: Callable[[list[_T]], _T]) -> _T:
        """Set the fact to ``combine`` of every rank's ``fn()``; every rank holds and returns it.

        See :meth:`ModelOptSession.all_reduce`.
        """
        label = self._holder._label(self._name)
        return self._hold(self._holder._sync_session().all_reduce(label, fn, combine))

    def _hold(self, value: _T) -> _T:
        # The holder keeps its own copy: a later mutation of the caller's object stays the caller's.
        self._holder._synced[self._name] = copy.deepcopy(value)
        return self._holder._synced[self._name]


class _SyncedAttribute:
    """What a ``Synced[...]`` annotation installs: reading it gives the fact's :class:`Synced` handle."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, holder, owner=None):
        return self if holder is None else Synced(holder, self.name)

    def __set__(self, holder, value) -> None:
        raise AttributeError(
            f"{type(holder).__name__}.{self.name} is synced: set it with .broadcast_from(fn) or "
            ".all_reduce(fn, combine), which every rank calls at the same point."
        )


class _SyncedHolder:
    """Base of a class with synced facts: each ``Synced[...]`` annotation of a subclass becomes one."""

    _synced: dict[str, Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name, annotation in inspect.get_annotations(cls, eval_str=True).items():
            if getattr(annotation, "__origin__", None) is Synced:
                setattr(cls, name, _SyncedAttribute(name))

    def _sync_session(self) -> "ModelOptSession":
        """The session whose ranks hold this object's synced facts."""
        raise NotImplementedError

    def _label(self, name: str) -> str:
        """How the fact ``name`` is named to the other ranks, which must name the same one."""
        return name


class ModelOptSession(_SyncedHolder):
    """The live distributed session of a PTQ run: this process's identity, plus the model's EP mesh.

    One class covers both cases: ``distributed`` is False for a single process (``WORLD_SIZE==1``) and
    True for a real ``torch.distributed`` launch (``WORLD_SIZE>1``). Two parts, with different lifetimes:

    * process IDENTITY (``distributed``/``rank``/``world_size``/``local_rank``/``device``/``is_main``) --
      stable for the whole run, populated once by :func:`modelopt_session`. Reading ANY before
      initialization raises, so code cannot silently consume a stale/zero default before the process
      group is set up.
    * ``ep_mesh`` -- the expert-parallel device mesh of the loaded model, set by ``load_and_prepare_ep``
      (``None`` until then, and for a model loaded without expert parallelism).

    Facts annotated :class:`Synced`, such as ``plan``, hold the same value on every rank.
    """

    #: The :class:`DistributionPlan` of the model this session runs, the same on every rank.
    plan: Synced[DistributionPlan]

    def __init__(self) -> None:
        """Create an empty (uninitialized) session; populate identity with :meth:`initialize`."""
        self._values: dict[str, Any] | None = None
        self._synced: dict[str, Any] = {}
        self.ep_mesh = None

    def initialize(
        self,
        *,
        distributed: bool,
        rank: int,
        world_size: int,
        local_rank: int,
        device,
        is_main: bool,
    ) -> None:
        """Populate the session fields (called by ``modelopt_session``, or directly for tests)."""
        self._values = {
            "distributed": distributed,
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "device": device,
            "is_main": is_main,
        }

    def _reset(self) -> None:
        self._values = None
        self._synced = {}
        self.ep_mesh = None

    def _sync_session(self) -> "ModelOptSession":
        return self

    def broadcast_from(self, label: str, fn: Callable[[], _V], rank: int = 0) -> _V:
        """Return ``fn()`` run on ``rank`` only, on every rank; keep nothing.

        For a value that depends on what each rank reads for itself -- the environment, its GPU, files
        on its node -- so that ranks computing it for themselves could disagree. If ``fn`` raises, every
        rank raises, instead of waiting in a broadcast ``rank`` never sends. Ranks broadcasting
        different labels at the same point fail. Collective: every rank calls it at the same point,
        with the same ``rank``. Outside a distributed session ``fn`` runs alone. A fact a later stage
        reads back is a :class:`Synced` attribute, set with its own ``broadcast_from``.
        """
        if not (self.initialized and self.distributed):
            return fn()
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank={rank} is not a rank of this {self.world_size}-rank run.")
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

    def all_reduce(self, label: str, fn: Callable[[], _V], combine: Callable[[list[_V]], _V]) -> _V:
        """Return ``combine`` of every rank's ``fn()``, on every rank; keep nothing.

        For a value no later stage reads back from the session; a fact that one does is a
        :class:`Synced` attribute, set with its own ``all_reduce``. ``combine`` takes the ranks' values
        in rank order and returns the one every rank gets, e.g. ``min`` for a batch size each rank's
        memory bounds, where one rank's choice could run another out of memory. If ``fn`` raises on any
        rank, every rank raises, instead of waiting in the gather that rank never joins. Ranks reducing
        different labels at the same point fail. Collective; outside a distributed session ``fn`` runs
        alone.
        """
        if not (self.initialized and self.distributed):
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
        for rank, (_, _, rank_error) in enumerate(gathered):
            if rank_error is not None:
                raise RuntimeError(f"rank {rank} failed to compute {label}: {rank_error}")
        labels = sorted({rank_label for rank_label, _, _ in gathered})
        if labels != [label]:
            raise RuntimeError(
                f"Ranks reduced out of step: {labels} at the same point. Every rank must reduce "
                "them in the same order."
            )
        # The gathered values are unpickled copies, so the result shares nothing with the callers'.
        return combine([rank_value for _, rank_value, _ in gathered])

    @property
    def initialized(self) -> bool:
        """Whether ``modelopt_session`` has populated the state."""
        return self._values is not None

    def _get(self, key: str):
        if self._values is None:
            raise RuntimeError(
                "session accessed before initialization; enter "
                "modelopt.torch.utils.distributed.modelopt_session() first"
            )
        return self._values[key]

    @property
    def distributed(self) -> bool:
        """True for a distributed launch (WORLD_SIZE>1), False for single-process."""
        return self._get("distributed")

    @property
    def rank(self) -> int:
        """Global rank of this process (0 single-process)."""
        return self._get("rank")

    @property
    def world_size(self) -> int:
        """Total number of processes (1 single-process)."""
        return self._get("world_size")

    @property
    def local_rank(self) -> int:
        """Node-local rank of this process (0 single-process)."""
        return self._get("local_rank")

    @property
    def device(self):
        """This process's torch device (``cuda:local_rank`` distributed; the caller default otherwise)."""
        return self._get("device")

    @property
    def is_main(self) -> bool:
        """Whether this is the main process (rank 0)."""
        return self._get("is_main")


# The process-wide ACTIVE session: a single STABLE object created once at import and NEVER rebound.
# :func:`modelopt_session` mutates it in place via :meth:`ModelOptSession.initialize` / ``_reset`` rather
# than assigning a new object -- so any held reference stays valid and never goes stale. Module-PRIVATE
# (not exported): reach it only through :func:`get_active_mo_session`. It exists before initialization;
# reading its IDENTITY fields (``rank`` etc.) then raises via the per-field guard.
_active_mo_session: ModelOptSession = ModelOptSession()


def get_active_mo_session(required: bool = True) -> ModelOptSession:
    """Return the process-wide session.

    * ``required=True`` (default) -- fail-fast: raise if no session has been entered yet by
      :func:`modelopt_session`. Use when the caller genuinely needs a live session.
    * ``required=False`` -- return the singleton regardless of initialization. Its ``ep_mesh`` is
      readable (``None``) even before a session is entered, so this is for defensive reads deep in public
      APIs (export / calibration) that may run with no active session.
    """
    if required and not _active_mo_session.initialized:
        raise RuntimeError(
            "no active session; enter modelopt.torch.utils.distributed.modelopt_session() first"
        )
    return _active_mo_session


@contextmanager
def modelopt_session(timeout: timedelta | None = None, device=None):
    """Populate the active session for the block, then tear down.

    On a distributed launch (WORLD_SIZE>1) this initializes the process group (binding the local CUDA
    device) with ``timeout`` and records rank/world_size/local_rank/``cuda:local_rank``/is_main; it
    DESTROYS the group on exit even if the body raises. Single-process records the trivial single-rank
    topology -- with ``device`` as the device (the caller's default, e.g. the ``--device`` CLI value, since
    there is no local rank to derive one) -- and enter/exit are no-ops around the body.

    Yields the session (also reachable via :func:`get_active_mo_session`); it is reset on exit.
    """
    owns_group = is_distributed_launch()
    if owns_group:
        setup(timeout=timeout)
        _active_mo_session.initialize(
            distributed=True,
            rank=rank(),
            world_size=size(),
            local_rank=local_rank(),
            device=torch.device(f"cuda:{local_rank()}"),
            is_main=rank() == 0,
        )
    else:
        _active_mo_session.initialize(
            distributed=False,
            rank=0,
            world_size=1,
            local_rank=0,
            device=torch.device(device) if device is not None else None,
            is_main=True,
        )
    try:
        yield _active_mo_session
    finally:
        # If the body is unwinding due to an exception, PRINT it here BEFORE cleanup()'s collective
        # barrier. A per-rank exception (only some ranks raise) otherwise makes those ranks enter the
        # cleanup barrier while the rest are still inside a calibration a2a collective -> cross-rank
        # DEADLOCK, with the original exception never surfaced (pending behind the hung barrier).
        exc = sys.exc_info()[1]
        if exc is not None:
            print(
                f"[modelopt_session][rank {os.environ.get('RANK', '?')}] body raised "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            sys.stderr.flush()
        if owns_group:
            cleanup()
        _active_mo_session._reset()


def cleanup():
    """Cleans up the distributed environment.

    The barrier is skipped when unwinding from an error, since peers may be blocked in a collective
    this rank will never reach. ``SystemExit`` is treated as a clean exit (every rank reaches it).
    That is not sufficient on its own -- ``destroy_process_group`` below blocks for the same reason
    -- so error paths must call :func:`abort` before reaching this ``finally``.
    """
    if is_initialized():
        exc = sys.exception()
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
