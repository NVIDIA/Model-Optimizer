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

import os
import socket
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


def init_process(rank, size, job=None, backend="gloo", port=None):
    """Initialize the distributed environment."""

    os.environ["MASTER_ADDR"] = "localhost"

    port = str(get_free_port()) if port is None else str(port)

    # We need to use a different port for each tests to avoid conflicts
    os.environ["MASTER_PORT"] = port

    dist.init_process_group(backend, rank=rank, world_size=size)
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(rank)
    torch.manual_seed(1234)
    if job is not None:
        job(rank, size)


def spawn_multiprocess_job(size, job, backend="gloo"):
    port = get_free_port()

    processes = []
    mp.set_start_method("spawn", force=True)
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, job, backend, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

        # Ensure that all processes have exited successfully
        assert not p.exitcode


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
    """

    def __init__(self, world_size, backend="nccl", teardown_fn=default_worker_teardown):
        assert world_size > 0, "World size must be greater than 0"
        self.world_size = world_size
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

        for _ in range(world_size):
            msg = self._result_queue.get(timeout=120)
            assert msg == "ready", f"Worker failed to initialize: {msg}"

    @staticmethod
    def _worker_loop(rank, world_size, backend, port, cmd_queue, result_queue, teardown_fn):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(rank)
        torch.manual_seed(1234)
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
        for q in self._cmd_queues:
            q.put((fn, args, kwargs))

        errors = []
        for _ in range(self.world_size):
            status, rank, tb = self._result_queue.get(timeout=600)
            if status == "error":
                errors.append(f"--- Rank {rank} ---\n{tb}")

        if errors:
            raise RuntimeError("Worker(s) failed:\n" + "\n".join(errors))

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
