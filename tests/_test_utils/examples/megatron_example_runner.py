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
"""Run ``examples/megatron_bridge`` scripts without a ``torchrun`` launch per step.

A launch spends ~25s importing torch/megatron/modelopt before doing any work, and one example test
runs several. Single-rank commands drive the script's ``main()`` directly in the pytest process,
which is where that cost disappears (the suite goes from ~21min to ~4min on one GPU). Multi-rank
commands drive ``torchrun`` itself in-process, which only saves the launcher's own imports -- the
pytest process cannot be more than one rank -- but keeps torchrun's fresh worker processes.

The script's own ``get_args()`` still runs, so CLI flags and recipe strings stay covered. Not
covered: the ``torchrun`` invocation and the ``__main__`` block (``dist.setup()``, ``dist.abort()``
and ``dist.cleanup()``).

Megatron and torch.distributed.run are imported lazily on purpose: this module is imported at
collection time, and importing ``megatron.bridge`` initialises CUDA in the pytest process, which
would hold a context on device 0 for the whole session even when every step runs under torchrun.
"""

import contextlib
import gc
import importlib
import importlib.util
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import torch
from _test_utils.examples.run_command import _ORPHAN_PIPE_TIMEOUT_S, MODELOPT_ROOT
from _test_utils.torch.distributed.utils import get_free_port

import modelopt.torch.utils.distributed as dist

# torchrun's PContext installs handlers for these and never restores them.
_LAUNCHER_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)


@contextlib.contextmanager
def _capture_output():
    """Capture everything a step emits, as one ordered stream, while still streaming it live.

    Three sources have to end up in the same place: ``print`` from the script, records from loggers
    Megatron-Bridge uses for lines tests assert on, and fd-level writes from ``torchrun`` workers
    and C extensions. Under pytest's fd capture ``sys.stdout``/``sys.stderr`` are objects over
    pytest's own temp file rather than fds 1/2, so both a redirect and an fd swap are needed.

    Tee rather than buffer: a job-level timeout SIGKILLs the runner without running any ``finally``,
    and that is exactly when the log is wanted.
    """
    holder = [""]
    chunks: list[str] = []
    read_fd, write_fd = os.pipe()
    live = os.fdopen(os.dup(1), "w", buffering=1, errors="replace")  # the fd we do not touch

    def _drain():
        with os.fdopen(read_fd, "r", errors="replace") as pipe:
            for line in pipe:
                live.write(line)
                chunks.append(line)

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    sink = os.fdopen(os.dup(1), "w", buffering=1, errors="replace")  # now the pipe
    handler = logging.StreamHandler(sink)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    others = [lg for lg in root.manager.loggerDict.values() if isinstance(lg, logging.Logger)]
    # Lower levels only for loggers that own their output; doing it for every logger switches on
    # torch dynamo/inductor INFO and floods the pipe, which cost the 1-GPU suite 3m43 -> 9m01.
    # A propagating logger still needs its own level lowered, or it drops the record before root.
    levels = {lg: lg.level for lg in [root, *others] if lg.handlers or not lg.propagate}
    # Attach the handler only where the record cannot reach root, or it is written twice.
    handled = [root, *(lg for lg in others if not lg.propagate)]
    try:
        for lg in levels:
            if lg.level > logging.INFO:
                lg.setLevel(logging.INFO)
        for lg in handled:
            lg.addHandler(handler)
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield holder
    finally:
        for lg in handled:
            lg.removeHandler(handler)
        for lg, level in levels.items():
            lg.setLevel(level)
        sink.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(write_fd)  # EOF for the reader
        reader.join(_ORPHAN_PIPE_TIMEOUT_S)  # bounded: an orphan may still hold the pipe
        live.close()
        holder[0] = "".join(chunks)


def reset_megatron_global_state() -> None:
    """Drop the global state a finished single-rank step leaves behind.

    Steps that run here share one interpreter, so anything global has to be put back or it leaks
    into the next test. Failure-tolerant on purpose: this also runs after a failed step, where the
    model may be half-built, and it must not mask the error that got us here.
    """
    with contextlib.suppress(Exception):
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
    with contextlib.suppress(Exception):
        # A separate singleton from the parallel state, initialised by the training path.
        from megatron.core.rerun_state_machine import destroy_rerun_state_machine

        destroy_rerun_state_machine()
    with contextlib.suppress(Exception):
        # Collect first: empty_cache() only returns blocks nothing references, and a finished step
        # can leave its model reachable from the imported module. Without this a later test ran
        # against a fragmented allocator ~9x slower.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_example_module(script: str, example_path: str):
    """Import an example script under a namespaced module name.

    ``import_module("quantize")`` would consult ``sys.modules`` first and could pick up an
    unrelated top-level module of the same name, and would leave the example cached under a
    generic name.
    """
    example_dir = str(MODELOPT_ROOT / "examples" / example_path)
    name = f"_modelopt_example_{example_path.replace('/', '_')}_{Path(script).stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(example_dir) / Path(script).name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, example_dir)  # the script's own sibling imports
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    finally:
        sys.path.remove(example_dir)
    return module


def _require_drivable(script: str, example_path: str) -> None:
    """Check the script exposes the ``get_args()`` + ``main()`` shape this runner drives."""
    module = _load_example_module(script, example_path)
    if not callable(getattr(module, "get_args", None)) or not callable(
        getattr(module, "main", None)
    ):
        raise AssertionError(f"{script} must define get_args() and main(args)")


def requested_world_size(cmd_parts: list[str]) -> int | None:
    """World size a ``torchrun`` command asks for, or ``None`` if it is not a plain integer."""
    for part in cmd_parts:
        if str(part).startswith("--nproc_per_node="):
            value = str(part).split("=", 1)[1]
            return int(value) if value.isdigit() else None  # "gpu"/"auto": keep torchrun
    return None


def run_example_in_process(cmd_parts: list[str], example_path: str) -> str:
    """Drive a single-rank step's real ``get_args()`` + ``main()`` here. Returns its stdout."""
    # Set unconditionally: the per-test environment restore drops these while the process group
    # stays initialised, so a later step would otherwise run with a live group and no launch vars.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_PORT", str(get_free_port()))
        dist.setup()
        torch.cuda.set_device(dist.local_rank())

    script = next(p for p in cmd_parts if str(p).endswith(".py"))
    argv = [str(p) for p in cmd_parts[cmd_parts.index(script) :]]
    example_dir = str(MODELOPT_ROOT / "examples" / example_path)
    cwd = os.getcwd()
    os.chdir(example_dir)  # match the subprocess path, which runs with the example dir as cwd
    native = [""]  # bound up front: readable even if the capture fails to start
    try:
        module = _load_example_module(str(script), example_path)
        reset_megatron_global_state()  # a previous step left its own parallel state behind
        with patch.object(sys, "argv", argv):
            args = module.get_args()
        try:
            with _capture_output() as native:
                module.main(args)
        except SystemExit as e:
            if e.code not in (0, None):
                raise RuntimeError(f"{script} exited with code {e.code}") from e
        return native[0]
    except BaseException as e:
        # The caller matches transient-HuggingFace markers on this alongside the traceback.
        e.captured_output = native[0]
        raise
    finally:
        os.chdir(cwd)


def run_torchrun_in_process(cmd_parts: list[str], example_path: str) -> str:
    """Drive ``torchrun`` itself in-process, letting it spawn fresh workers as usual.

    Used for multi-rank steps. Only the launcher's imports are saved, but the workers are fresh
    processes, so none of the global state a reused process would inherit applies. Borrowed from
    Megatron-Bridge's own functional tests.
    """
    from torch.distributed.run import main as torchrun_main

    example_dir = MODELOPT_ROOT / "examples" / example_path
    script = next(p for p in cmd_parts if str(p).endswith(".py"))
    argv = [str(p) for p in cmd_parts]
    argv[argv.index(str(script))] = str(example_dir / Path(script).name)
    if not any(a.startswith(("--master_port", "--master-port")) for a in argv):
        # The rendezvous store now lives in a long-lived process, so do not rely on torchrun's
        # fixed default port being free by the time the next step starts.
        argv.insert(1, f"--master_port={get_free_port()}")
    # PContext.start() installs its own handlers for these and never puts them back; with a
    # subprocess launcher the process exit did that for us. Losing them would break pytest's
    # Ctrl-C and CI cancellation handling for the rest of the session.
    saved_handlers = {s: signal.getsignal(s) for s in _LAUNCHER_SIGNALS}
    cwd = os.getcwd()
    os.chdir(example_dir)
    cap = [""]
    try:
        with _capture_output() as cap, patch.object(sys, "argv", argv):
            torchrun_main()
        return cap[0]
    except BaseException as e:
        # ChildFailedError only carries the worker's traceback when the entrypoint is decorated
        # with @record, which the examples are not -- so the captured output is all the caller has.
        e.captured_output = cap[0]
        raise
    finally:
        for sig, handler in saved_handlers.items():
            signal.signal(sig, handler)
        os.chdir(cwd)


def run_example_step(cmd_parts: list[str], example_path: str) -> str:
    """Run an example step without shelling out.

    Every step must be ``torchrun --nproc_per_node=<int> <script>.py``, with the script exposing
    ``get_args()`` + ``main()``. Anything else raises: falling back to a subprocess would still
    pass, only ~6x slower, so a step that breaks the convention has to fail instead of quietly
    costing the suite its speed-up.
    """
    script = next((str(p) for p in cmd_parts if str(p).endswith(".py")), None)
    if script is None:
        raise AssertionError(f"step must invoke a .py script: {cmd_parts}")
    world_size = requested_world_size(cmd_parts)
    if world_size is None:
        raise AssertionError(f"--nproc_per_node must be a plain integer: {cmd_parts}")
    if world_size > 1:
        # torchrun imports the script in fresh children, so get_args()/main() need not exist here.
        return run_torchrun_in_process(cmd_parts, example_path)
    _require_drivable(script, example_path)
    return run_example_in_process(cmd_parts, example_path)
