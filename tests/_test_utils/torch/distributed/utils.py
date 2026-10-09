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

import gc
import os
import queue
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


def destroy_extra_process_groups():
    """Destroy every process group except the default one.

    Libraries that create groups per model (e.g. Megatron's ``initialize_model_parallel``, ~20 NCCL
    groups) drop their references on teardown but not the groups themselves. In a worker that
    lives through hundreds of jobs they pile up (communicators, one watchdog thread each).
    """
    from torch.distributed.distributed_c10d import _get_default_group, _world

    default = _get_default_group()
    for group in [g for g in _world.pg_map if g is not default]:
        try:
            dist.destroy_process_group(group)
        except Exception as e:  # noqa: PERF203
            print(f"Error destroying process group: {e}")


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

    A job that times out, is interrupted (e.g. by pytest-timeout) or loses a worker process can leave
    ranks stuck inside a collective and stale results in the queue, so the pool is marked broken and
    respawned on the next ``run``. A job that merely raises leaves the workers healthy and is reused.
    """

    # After this many failed (re)starts in a row ``run`` fails at once instead of waiting out the
    # ready timeout again for every remaining test.
    _MAX_START_FAILURES = 2

    def __init__(
        self,
        world_size,
        backend="nccl",
        teardown_fn=default_worker_teardown,
        ready_timeout=300,
        run_timeout=600,
    ):
        assert world_size > 0, "World size must be greater than 0"
        self.world_size = world_size
        self._backend = backend
        self._teardown_fn = teardown_fn
        self._ready_timeout = ready_timeout
        self._run_timeout = run_timeout
        self._broken = False
        self._start_failures = 0
        self._cmd_queues = []
        self._result_queue = None
        self._processes = []
        self._start()

    def _start(self):
        ctx = mp.get_context("spawn")
        self._cmd_queues = [ctx.Queue() for _ in range(self.world_size)]
        self._result_queue = ctx.Queue()
        self._processes = []

        port = get_free_port()
        for rank in range(self.world_size):
            p = ctx.Process(
                target=self._worker_loop,
                args=(
                    rank,
                    self.world_size,
                    self._backend,
                    port,
                    self._cmd_queues[rank],
                    self._result_queue,
                    self._teardown_fn,
                ),
            )
            p.start()
            self._processes.append(p)

        try:
            for _ in range(self.world_size):
                # Cold imports of the Megatron/TE/modelopt stack take 75-100 s under coverage in CI
                msg = self._get(self._ready_timeout)
                assert msg == "ready", f"Worker failed to initialize: {msg}"
        except BaseException:
            self._start_failures += 1
            self._kill()
            raise
        self._start_failures = 0
        self._broken = False

    def _get(self, timeout):
        """Return the next result message; fail early if a worker process died."""
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
                    raise TimeoutError(
                        f"No message from the worker pool within {timeout} s"
                    ) from None

    def _kill(self):
        """Stop every worker without waiting for them to finish their current job."""
        for p in self._processes:
            if p.is_alive():
                p.terminate()
        for p in self._processes:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join(timeout=10)
        for q in [*self._cmd_queues, self._result_queue]:
            if q is not None:
                q.cancel_join_thread()
                q.close()

    @staticmethod
    def _worker_loop(rank, world_size, backend, port, cmd_queue, result_queue, teardown_fn):
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
            try:
                fn(rank, world_size, *args, **kwargs)
            except Exception:
                status = "error"
                tb = traceback.format_exc()
            finally:
                if teardown_fn is not None:
                    try:
                        teardown_fn(rank, world_size)
                    except Exception as e:
                        print(f"Error tearing down worker: {e}")
                        status = "error"
                        teardown_tb = traceback.format_exc()
                        tb = (tb + "\n" if tb else "") + f"[teardown] {teardown_tb}"
            result_queue.put((status, rank, tb))

        dist.destroy_process_group()

    def run(self, fn, *args, **kwargs):
        """Dispatch ``fn`` to all workers and block until completion.

        ``fn`` is called as ``fn(rank, world_size, *args, **kwargs)`` and must be picklable
        (top-level function or ``functools.partial`` of one).
        """
        if self._broken or not all(p.is_alive() for p in self._processes):
            if self._start_failures >= self._MAX_START_FAILURES:
                raise RuntimeError(
                    f"The worker pool could not be restarted ({self._start_failures} failed "
                    "attempts in a row); failing instead of waiting for it again."
                )
            self._kill()
            self._start()

        for q in self._cmd_queues:
            q.put((fn, args, kwargs))

        errors = []
        try:
            for _ in range(self.world_size):
                status, rank, tb = self._get(self._run_timeout)
                if status == "error":
                    errors.append(f"--- Rank {rank} ---\n{tb}")
        except BaseException:
            # Timeout (pytest-timeout's included), dead worker or interrupt: a rank may still be
            # inside ``fn``, and results of the abandoned job may still arrive. Respawn next time.
            self._broken = True
            raise

        if errors:
            raise RuntimeError("Worker(s) failed:\n" + "\n".join(errors))

    def shutdown(self):
        """Signal all workers to exit and wait for them to finish."""
        if not self._broken:
            # A clean exit lets the workers flush their coverage data
            for q in self._cmd_queues:
                q.put(None)
            for p in self._processes:
                p.join(timeout=60)
        self._kill()


def synchronize_state_dict(model: nn.Module):
    state_dict = model.state_dict()
    for v in state_dict.values():
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
    model.load_state_dict(state_dict)
