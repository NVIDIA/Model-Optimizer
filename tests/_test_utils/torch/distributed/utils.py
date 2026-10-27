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

import contextlib
import faulthandler
import gc
import os
import queue
import signal
import socket
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    return port


def _init_backend(backend):
    """Mirror ``modelopt.torch.utils.distributed.setup``, which pairs NCCL with a CPU backend.

    Bare ``"nccl"`` registers no backend for CPU tensors, so a collective on a CPU-resident
    shard -- gathering an FSDP2 ``cpu_offload`` parameter, say -- fails with "No backend type
    associated with device type cpu" in tests but not in production.
    """
    return "cpu:gloo,cuda:nccl" if backend == "nccl" else backend


def init_process(rank, size, job=None, backend="gloo", port=None):
    """Initialize the distributed environment."""

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["LOCAL_WORLD_SIZE"] = str(size)
    os.environ["WANDB_DISABLED"] = "true"

    port = str(get_free_port()) if port is None else str(port)

    # We need to use a different port for each tests to avoid conflicts
    os.environ["MASTER_PORT"] = port

    dist.init_process_group(_init_backend(backend), rank=rank, world_size=size)
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(rank)
    torch.manual_seed(1234)
    if job is not None:
        job(rank, size)


def spawn_multiprocess_job(size, job, backend="gloo"):
    port = get_free_port()
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(size):
        p = ctx.Process(target=init_process, args=(rank, size, job, backend, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

        # Ensure that all processes have exited successfully
        assert not p.exitcode


def get_live_pool(pools, key, make_pool):
    """Return ``pools[key]``, building it with ``make_pool`` if it is missing or was killed.

    Fixtures hand pools out through this, so a hang kills the pool for one test instead of
    failing every later test that uses it.
    """
    pool = pools.get(key)
    if pool is None or pool.broken:
        pools[key] = pool = make_pool()
    return pool


_BASELINE_TORCH_FLAGS = {}


def _capture_baseline_torch_flags():
    """Remember the torch settings a freshly spawned worker starts with."""
    _BASELINE_TORCH_FLAGS.update(
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )


def reset_worker_state(rank, world_size):
    """Return a persistent worker to the state it had right after spawning.

    ``set_seed`` and similar helpers change process-global torch settings; run this job at the
    boundary between test modules that share workers.
    """
    torch.backends.cudnn.deterministic = _BASELINE_TORCH_FLAGS["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = _BASELINE_TORCH_FLAGS["cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32 = _BASELINE_TORCH_FLAGS["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = _BASELINE_TORCH_FLAGS["matmul_tf32"]
    torch.use_deterministic_algorithms(_BASELINE_TORCH_FLAGS["deterministic_algorithms"])
    torch.manual_seed(1234)
    gc.collect()
    torch.cuda.empty_cache()


def default_worker_teardown(rank, world_size):
    """Minimal cleanup between tests in persistent workers."""
    try:
        from accelerate.state import AcceleratorState

        AcceleratorState._reset_state()
    except ImportError:
        pass
    except Exception as e:
        print(f"Error resetting AcceleratorState: {e}")
    torch.cuda.empty_cache()


def reseeding_worker_teardown(rank, world_size):
    """``default_worker_teardown``, then the seed a freshly spawned rank starts from.

    A job that draws a different amount of randomness on each rank would otherwise leave the next
    job's ranks out of step, where spawned ranks would start it in step.
    """
    default_worker_teardown(rank, world_size)
    torch.manual_seed(1234)


class DistributedWorkerPool:
    """Persistent worker pool that keeps distributed processes alive across multiple test dispatches.

    Instead of spawning/destroying processes per test (which adds ~10s overhead each time),
    workers are spawned once and reuse the same ``torch.distributed`` process group.
    Use with a module-scoped pytest fixture to share workers across all tests in a file.

    Usage::

        pool = DistributedWorkerPool(
            world_size=2, backend="nccl", teardown_fn=default_worker_teardown
        )


        def _test_fn(rank, size): ...


        pool.run(_test_fn)
        pool.run(partial(other_fn, arg1))
        pool.shutdown()
    """

    def __init__(
        self, world_size, backend="nccl", teardown_fn=default_worker_teardown, ready_timeout=300
    ):
        assert world_size > 0, "World size must be greater than 0"
        self.world_size = world_size
        # Set once a job was interrupted: the workers are then killed, see ``run``.
        self.broken = False
        self.job_timeout = 600
        self.failure_grace = 30  # seconds the other ranks get to finish once one has failed
        ctx = mp.get_context("spawn")
        self._cmd_queues = [ctx.Queue() for _ in range(world_size)]
        self._result_queue = ctx.Queue()
        self._processes = []

        port = get_free_port()
        for rank in range(world_size):
            p = ctx.Process(
                target=self._worker_loop,
                args=(
                    rank,
                    world_size,
                    backend,
                    port,
                    self._cmd_queues[rank],
                    self._result_queue,
                    teardown_fn,
                ),
            )
            p.start()
            self._processes.append(p)

        # ready_timeout is 300 s: cold imports of the Megatron/TE/modelopt stack take 75-100 s
        # under coverage in CI
        try:
            for _ in range(world_size):
                msg = self._get(ready_timeout)
                assert msg == "ready", f"Worker failed to initialize: {msg}"
        except BaseException:
            self._abort()  # do not leave workers behind when the start fails
            raise

    def _get(self, timeout):
        """Return the next message from the workers; fail within a second if a worker died.

        A worker that exits (an illegal memory access aborts the process, an OOM kill) would
        otherwise only be noticed when ``timeout`` runs out.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self._result_queue.get(
                    timeout=max(0.01, min(1.0, deadline - time.monotonic()))
                )
            except queue.Empty:  # noqa: PERF203
                dead = [
                    (rank, p.exitcode) for rank, p in enumerate(self._processes) if not p.is_alive()
                ]
                if dead:
                    raise RuntimeError(
                        f"Worker process(es) died (rank, exit code): {dead}"
                    ) from None
                if time.monotonic() >= deadline:
                    raise

    @staticmethod
    def _worker_loop(rank, world_size, backend, port, cmd_queue, result_queue, teardown_fn):
        if hasattr(signal, "SIGUSR1"):
            # The parent signals this to get the stack of a hung job into the CI log.
            faulthandler.register(signal.SIGUSR1, all_threads=True)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group(_init_backend(backend), rank=rank, world_size=world_size)
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(rank)
        torch.manual_seed(1234)
        _capture_baseline_torch_flags()
        result_queue.put("ready")

        while True:
            cmd = cmd_queue.get()
            if cmd is None:
                break
            fn, args, kwargs = cmd
            status = "ok"
            tb = None
            reported = False
            try:
                fn(rank, world_size, *args, **kwargs)
            except Exception:
                status = "error"
                tb = traceback.format_exc()
                # Report before teardown: it may block on a collective whose peer is still stuck in
                # ``fn``, and the parent's failure grace period starts when it hears of a failure.
                result_queue.put((status, rank, tb))
                reported = True
            finally:
                if teardown_fn is not None:
                    try:
                        teardown_fn(rank, world_size)
                    except Exception as e:
                        print(f"Error tearing down worker: {e}\n{traceback.format_exc()}")
                        if not reported:  # else the job failure is already sent; the teardown error is printed above
                            status = "error"
                            tb = f"[teardown] {traceback.format_exc()}"
            if not reported:
                result_queue.put((status, rank, tb))

        dist.destroy_process_group()

    def run(self, fn, *args, **kwargs):
        """Dispatch ``fn`` to all workers and block until completion.

        ``fn`` is called as ``fn(rank, world_size, *args, **kwargs)`` and must be picklable
        (top-level function or ``functools.partial`` of one).

        If the job fails or the wait is interrupted (e.g. by ``pytest-timeout``), the workers may
        still be inside the job or its teardown, typically blocked in a collective, and every later
        ``run`` would hang until its own timeout. The workers are killed instead and the pool is
        marked ``broken``; callers that want to continue need a new pool (``get_live_pool``). After
        a rank fails, its peers get ``failure_grace`` seconds to report before the kill.
        """
        if self.broken:
            raise RuntimeError("Worker pool was killed after an earlier job hung; create a new one")
        errors = []
        deadline = None  # set when the first rank fails: its peers get ``failure_grace`` in total
        try:
            for q in self._cmd_queues:
                q.put((fn, args, kwargs))
            for _ in range(self.world_size):
                timeout = self.job_timeout if deadline is None else deadline - time.monotonic()
                try:
                    status, rank, tb = self._get(max(timeout, 0))
                except RuntimeError as e:  # a worker died: keep what the other ranks reported
                    if errors:
                        raise RuntimeError(f"{e}\n" + "\n".join(errors)) from None
                    raise
                if status == "error":
                    errors.append(f"--- Rank {rank} ---\n{tb}")
                    deadline = deadline or time.monotonic() + self.failure_grace
        except queue.Empty:
            self._abort()
            if not errors:
                raise
            errors.append("--- The other ranks did not finish after a rank failed; killed ---")
        except BaseException:
            self._abort()
            raise

        if errors:
            if not self.broken:
                # Every rank replied, but a failed rank reports before its teardown, which can still
                # be blocked; never hand these workers to another job.
                self._abort(dump_stacks=False)
            raise RuntimeError("Worker(s) failed:\n" + "\n".join(errors))

    def _abort(self, dump_stacks=True):
        """Kill the workers, first printing their stacks (to stderr, for the CI log) if asked."""
        self.broken = True
        if dump_stacks and hasattr(signal, "SIGUSR1"):
            for p in self._processes:
                with contextlib.suppress(OSError):
                    os.kill(p.pid, signal.SIGUSR1)
            time.sleep(3)  # let faulthandler write before the workers go away
        for p in self._processes:
            p.kill()
            p.join(timeout=10)

    def shutdown(self):
        """Signal all workers to exit and wait for them to finish."""
        for q in self._cmd_queues:
            q.put(None)
        for p in self._processes:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                # Ensure the terminated process is fully reaped to avoid zombies.
                p.join(timeout=10)


def synchronize_state_dict(model: nn.Module):
    state_dict = model.state_dict()
    for v in state_dict.values():
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
    model.load_state_dict(state_dict)
