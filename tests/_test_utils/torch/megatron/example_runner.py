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
"""Run ``examples/megatron_bridge`` scripts in-process instead of via ``torchrun``.

A ``torchrun`` launch spends ~25s importing torch/megatron/modelopt before doing any work, and one
example test runs several. Driving the script's ``main()`` in the pytest process instead takes the
full suite from ~26min to ~4min on one GPU.

The script's own ``get_args()`` still runs, so CLI flags and recipe strings stay covered. Not
covered: the ``torchrun`` invocation and the ``__main__`` block (``dist.setup()``/``dist.abort()``).
Multi-rank commands are left alone -- the pytest process is a single rank.
"""

import contextlib
import gc
import importlib
import io
import logging
import os
import sys
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from _test_utils.examples.run_command import MODELOPT_ROOT
from _test_utils.torch.distributed.utils import get_free_port
from megatron.bridge.training.config import CheckpointConfig
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
from megatron.core.rerun_state_machine import destroy_rerun_state_machine

import modelopt.torch.utils.distributed as dist


def _use_spawn_for_async_checkpointing() -> None:
    """Stop async checkpoint saving from forking a process that already owns CUDA/NCCL.

    ``CheckpointConfig.async_write_results_mp_mode`` defaults to ``"fork"``. Under ``torchrun`` that
    is fine -- the fork happens in a fresh process. Reusing a worker means the fork happens after
    CUDA and NCCL are initialised, and it deadlocks: the manager server reaches ``serve_forever``
    while the caller never returns from ``get_write_results_queue``. Confirmed with py-spy.
    """
    with contextlib.suppress(Exception):
        CheckpointConfig.async_write_results_mp_mode = "spawn"


@contextlib.contextmanager
def _capture_output():
    """Capture what a step writes here, including lines emitted through logging handlers."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    targets = [logging.getLogger()]
    targets += [
        lg
        for lg in logging.root.manager.loggerDict.values()
        if isinstance(lg, logging.Logger) and (lg.handlers or not lg.propagate)
    ]
    levels = {}
    for lg in targets:
        lg.addHandler(handler)
        levels[lg] = lg.level
        if lg.level > logging.INFO:
            lg.setLevel(logging.INFO)
    try:
        with contextlib.redirect_stdout(buf):
            yield buf
    finally:
        for lg, level in levels.items():
            lg.removeHandler(handler)
            lg.setLevel(level)


@contextlib.contextmanager
def _capture_output_fd():
    """Capture at the file-descriptor level, for use inside a worker process.

    Loggers configured *during* the step (Megatron-Bridge sets its own up in ``setup()``) are
    invisible to handlers installed beforehand, so the in-process approach misses them here.
    """
    holder = [""]
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as tmp:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_out, saved_err = os.dup(1), os.dup(2)
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        try:
            yield holder
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
            tmp.seek(0)
            holder[0] = tmp.read()


def reset_megatron_global_state() -> None:
    """Drop the global state a finished example step leaves behind.

    Steps share one interpreter, so anything global has to be put back or it leaks into the next
    test. Failure-tolerant on purpose: this also runs after a failed step, where the model may be
    half-built, and it must not mask the error that got us here.
    """
    with contextlib.suppress(Exception):
        parallel_state.destroy_model_parallel()
    with contextlib.suppress(Exception):
        # A separate singleton from the parallel state, initialised by the training path.
        destroy_rerun_state_machine()
    with contextlib.suppress(Exception):
        # The async checkpoint worker is a *class-level* singleton, so it outlives the step that
        # started it and the next one inherits a stale process. Under torchrun the process exits
        # and takes it with it; a reused worker has to close it explicitly.
        if AsyncCallsQueue._persistent_caller is not None:
            AsyncCallsQueue._persistent_caller.close(abort=True)
            AsyncCallsQueue._persistent_caller = None
    with contextlib.suppress(Exception):
        # Collect first: empty_cache() only returns blocks nothing references, and a finished step
        # can leave its model reachable from the imported module. Without this a later test ran
        # against a fragmented allocator ~9x slower.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_example_in_process(cmd_parts: list[str], example_path: str) -> str:
    """Drive an example script's real ``get_args()`` + ``main()`` here. Returns its stdout."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(get_free_port()))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.setup()
        torch.cuda.set_device(dist.local_rank())

    _use_spawn_for_async_checkpointing()
    script = next(p for p in cmd_parts if str(p).endswith(".py"))
    argv = [str(p) for p in cmd_parts[cmd_parts.index(script) :]]
    example_dir = str(MODELOPT_ROOT / "examples" / example_path)
    sys.path.insert(0, example_dir)
    output = ""
    try:
        module = importlib.import_module(Path(script).stem)
        reset_megatron_global_state()  # a previous step left its own parallel state behind
        with patch.object(sys, "argv", argv):
            args = module.get_args()
        try:
            with _capture_output() as buf:
                module.main(args)
        except SystemExit as e:
            if e.code not in (0, None):
                raise RuntimeError(f"{script} exited with code {e.code}") from e
        finally:
            output = buf.getvalue()
        return output
    finally:
        sys.path.remove(example_dir)
        print(output)  # keep the script's output in the test log


# --- Multi-rank dispatch ---------------------------------------------------------------------
# The pytest process is a single rank, so a command asking for N>1 ranks cannot run in-process.
# A pool of persistent workers gives real ranks while still paying the import cost once.
_pool_provider = None


def set_worker_pool_provider(provider) -> None:
    """Register ``provider(world_size) -> DistributedWorkerPool | None``."""
    global _pool_provider
    _pool_provider = provider


def requested_world_size(cmd_parts: list[str]) -> int | None:
    """World size a ``torchrun`` command asks for, if it says."""
    for part in cmd_parts:
        if str(part).startswith("--nproc_per_node="):
            return int(str(part).split("=", 1)[1])
    return None


def _reinit_process_group(rank: int, world_size: int, master_port: int) -> None:
    """Give the step a fresh process group, as a ``torchrun`` launch would.

    Megatron-Bridge tears down what it calls framework-owned distributed resources when a run ends
    ("Bridge is aborting framework-owned distributed resources..."), which includes the pool's own
    group. The next step then finds a dead group and its rendezvous is refused. Persistent
    *processes* are what save the import cost; the group itself is cheap to rebuild per step.
    """
    if world_size <= 1:
        return
    with contextlib.suppress(Exception):
        if dist.is_initialized():
            torch.distributed.destroy_process_group()
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.distributed.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def run_example_step_in_worker(rank, world_size, cmd_parts, example_path, output_path, master_port):
    """Run one example step inside a persistent worker. Top-level so it stays picklable."""
    _use_spawn_for_async_checkpointing()
    reset_megatron_global_state()
    _reinit_process_group(rank, world_size, master_port)
    env_before = os.environ.copy()
    script = next(p for p in cmd_parts if str(p).endswith(".py"))
    argv = [str(p) for p in cmd_parts[cmd_parts.index(script) :]]
    example_dir = str(MODELOPT_ROOT / "examples" / example_path)
    sys.path.insert(0, example_dir)
    output = ""
    try:
        module = importlib.import_module(Path(script).stem)
        with patch.object(sys, "argv", argv):
            args = module.get_args()
        try:
            with _capture_output_fd() as cap:
                module.main(args)
        except SystemExit as e:
            if e.code not in (0, None):
                raise RuntimeError(f"{script} exited with code {e.code}") from e
        finally:
            output = cap[0]
    finally:
        sys.path.remove(example_dir)
        print(output)
        # Every rank writes: Megatron prints some lines tests assert on with ``print_rank_last``,
        # so rank 0's output alone is incomplete whenever world_size > 1.
        Path(f"{output_path}.{rank}").write_text(output, encoding="utf-8")
        os.environ.clear()
        os.environ.update(env_before)
        reset_megatron_global_state()


def _drivable(script: str, example_path: str) -> bool:
    """Whether the script exposes the ``get_args()`` + ``main()`` shape this runner drives.

    Duck-typed rather than an allowlist so a new script needs no bookkeeping, and one that does not
    follow the convention (``generate_vllm.py`` has no ``get_args``) simply keeps using torchrun.
    """
    example_dir = str(MODELOPT_ROOT / "examples" / example_path)
    sys.path.insert(0, example_dir)
    try:
        module = importlib.import_module(Path(script).stem)
        return callable(getattr(module, "get_args", None)) and callable(
            getattr(module, "main", None)
        )
    except Exception:
        return False
    finally:
        sys.path.remove(example_dir)


def run_example_step(cmd_parts: list[str], example_path: str) -> str | None:
    """Dispatch an example step in-process (1 rank) or to a worker pool (N ranks).

    Returns the script's stdout, or ``None`` to fall back to a ``torchrun`` subprocess.
    """
    if os.environ.get("MODELOPT_NO_INPROCESS_EXAMPLES"):
        return None
    script = next((str(p) for p in cmd_parts if str(p).endswith(".py")), None)
    if script is None or not _drivable(script, example_path):
        return None
    world_size = requested_world_size(cmd_parts)
    if world_size == 1:
        return run_example_in_process(cmd_parts, example_path)
    pool = _pool_provider(world_size) if (world_size and _pool_provider) else None
    if pool is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        out_file = Path(td) / "stdout.txt"
        pool.run(
            partial(
                run_example_step_in_worker,
                cmd_parts=cmd_parts,
                example_path=example_path,
                output_path=str(out_file),
                master_port=get_free_port(),  # one port for all ranks, fresh each step
            )
        )
        parts = [
            Path(f"{out_file}.{r}").read_text(encoding="utf-8")
            for r in range(world_size)
            if Path(f"{out_file}.{r}").exists()
        ]
        return "\n".join(parts)
