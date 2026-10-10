# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Run ``python <script>.py`` example steps in one long-lived worker instead of one launch per step.

A fresh interpreter spends 10-20 s importing torch, transformers and modelopt before an example does any
work, and an example test runs several steps. The worker pays that once. Each step runs like
``python script.py`` (``runpy`` as ``__main__``, ``sys.argv``, the example directory as cwd and first on
``sys.path``); afterwards everything it changed in the process is put back, so the next step starts clean
while the imports stay cached. Environment variables are applied to ``os.environ``, so those read at
interpreter start-up or at import time keep the worker's values.

The worker is a separate process, not the pytest process, so a hung step can be killed (that step fails and
the next one gets a fresh worker) and pytest holds no CUDA context. The worker is also recycled after any
failed step. ``run_command.run_example_command`` hands an installed ``ExampleRunner`` the steps it
``accepts``; every other step, and every step when ``MODELOPT_EXAMPLE_RUNNER=subprocess``, still runs as a
subprocess. ``MODELOPT_EXAMPLE_RUNNER=inprocess`` runs the steps in the calling process instead: no
isolation, but ``pdb`` works.
"""

import contextlib
import gc
import importlib
import logging
import multiprocessing
import os
import random
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parents[3] / "examples"

_SIGNALS = [getattr(signal, n) for n in ("SIGTERM", "SIGINT", "SIGHUP") if hasattr(signal, n)]
# A clean worker exit lets its atexit handlers run, which is how coverage writes its data.
_WORKER_EXIT_TIMEOUT_S = 30
_KILL_TIMEOUT_S = 5


class StepFailed(subprocess.CalledProcessError):
    """A step exited non-zero, raised, or took its worker down; ``output`` is what it printed."""

    def __init__(self, returncode: int, cmd: list[str], output: str):
        super().__init__(returncode, cmd, output=output)
        # What run_example_command scans for transient Hub errors.
        self.captured_output = output


class StepTimeout(subprocess.TimeoutExpired):
    """A step outlived its timeout and its worker was killed; ``output`` is what it had printed."""

    def __init__(self, cmd: list[str], timeout: float, output: str):
        super().__init__(cmd, timeout, output=output)
        self.captured_output = output


def _loggers() -> list[logging.Logger]:
    with logging._lock:  # a getLogger() on another thread mutates loggerDict
        named = [
            lg for lg in logging.root.manager.loggerDict.values() if isinstance(lg, logging.Logger)
        ]
    return [logging.root, *named]


def _torch_flags(torch) -> list[tuple]:
    """The (object, attribute) pairs a script may flip, e.g. onnx_ptq/evaluation.py sets cudnn's."""
    backends = torch.backends
    return [
        (backends.cudnn, "deterministic"),
        (backends.cudnn, "benchmark"),
        (backends.cudnn, "allow_tf32"),
        (backends.cuda.matmul, "allow_tf32"),
    ]


def _purge_example_modules(before: set[str], examples_dir: Path) -> set[str]:
    """Drop modules the step imported from ``examples/``: ``utils``, ``config``, sibling example dirs.

    The next step may bring a different module of the same name. Third-party and modelopt modules
    stay cached, which is the saving.
    """
    root = examples_dir.resolve()
    purged = set()
    for name in set(sys.modules) - before:
        file = getattr(sys.modules[name], "__file__", None)
        if file and Path(file).resolve().is_relative_to(root):
            purged.add(name)
            del sys.modules[name]
    return purged


def _restore_loggers(snapshot: dict, purged: set[str]) -> None:
    """Restore levels everywhere, handler lists only where the script owns them.

    That is root, ``__main__`` and loggers named after purged modules: a library may configure its
    own logger lazily during a step and must keep its handler.
    """
    owned = {"__main__", *(name.split(".")[0] for name in purged)}
    for lg in _loggers():
        is_owned = lg is logging.root or lg.name.split(".")[0] in owned
        if lg in snapshot:
            handlers, level, propagate, disabled = snapshot[lg]
            lg.setLevel(level)
            lg.propagate, lg.disabled = propagate, disabled
            if is_owned:
                lg.handlers[:] = handlers
        elif is_owned:
            lg.handlers[:] = []
            lg.setLevel(logging.NOTSET)


@contextlib.contextmanager
def _restored_state(examples_dir: Path):
    """Put back what a step changed in the process, which ``python script.py`` discards on exit."""
    argv, path, cwd, environ = list(sys.argv), list(sys.path), os.getcwd(), dict(os.environ)
    modules = set(sys.modules)
    signal_handlers = {sig: signal.getsignal(sig) for sig in _SIGNALS}
    loggers = {lg: (list(lg.handlers), lg.level, lg.propagate, lg.disabled) for lg in _loggers()}
    log_disable = logging.root.manager.disable
    start_method = multiprocessing.get_start_method(allow_none=True)
    py_random = random.getstate()
    numpy = sys.modules.get("numpy")
    np_random = numpy.random.get_state() if numpy else None
    torch = sys.modules.get("torch")
    torch_rng = torch.get_rng_state() if torch else None
    torch_flags = (
        [(obj, name, getattr(obj, name)) for obj, name in _torch_flags(torch)] if torch else []
    )
    with warnings.catch_warnings():
        try:
            yield
        finally:
            sys.argv[:] = argv
            sys.path[:] = path
            os.chdir(cwd)
            os.environ.clear()
            os.environ.update(environ)
            for sig, handler in signal_handlers.items():
                # A handler installed from C reads back as None and cannot be restored.
                if handler is not None:
                    signal.signal(sig, handler)
            logging.disable(log_disable)
            if multiprocessing.get_start_method(allow_none=True) != start_method:
                multiprocessing.set_start_method(start_method, force=True)
            random.setstate(py_random)
            if np_random is not None:
                numpy.random.set_state(np_random)
            if torch is not None:
                torch.set_rng_state(torch_rng)
                for obj, name, value in torch_flags:
                    setattr(obj, name, value)
            _restore_loggers(loggers, _purge_example_modules(modules, examples_dir))
            gc.collect()
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()


def _drain(sink) -> str:
    sink.flush()
    sink.seek(0)
    return sink.read()


def run_step(
    cmd: list[str],
    cwd: Path | str,
    env: dict[str, str] | None,
    sink,
    hooks: Iterable[Callable] = (),
    examples_dir: Path = EXAMPLES_DIR,
) -> tuple[int, str]:
    """Run ``cmd`` (``[python, script, *args]``) in this process as ``python script ...`` in ``cwd``.

    Everything written to fds 1 and 2, by this process or its children, goes to ``sink``. ``env``
    replaces ``os.environ`` for the step. Each of ``hooks`` is a callable returning a context
    manager that wraps the step, for global state the generic restore does not know about.
    Returns ``(returncode, output)``. Only a ``BaseException`` that is not a script failure, such
    as ``KeyboardInterrupt`` or a pytest timeout, propagates, with ``captured_output`` set.
    """
    script = Path(cwd, cmd[1])
    sink.seek(0)
    sink.truncate()
    sys.stdout.flush()
    sys.stderr.flush()
    saved_fds = (os.dup(1), os.dup(2))
    saved_streams = (sys.stdout, sys.stderr)
    returncode = 0
    try:
        with _restored_state(examples_dir), contextlib.ExitStack() as stack:
            try:
                # Child processes and C extensions write to the fds; pytest's capture replaces
                # sys.stdout and sys.stderr with objects over its own files, so swap those too.
                os.dup2(sink.fileno(), 1)
                os.dup2(sink.fileno(), 2)
                sys.stdout = sys.stderr = sink
                os.chdir(cwd)
                if env is not None:
                    os.environ.clear()
                    os.environ.update(env)
                sys.argv[:] = [str(script), *cmd[2:]]
                sys.path.insert(0, str(script.parent))
                importlib.invalidate_caches()
                for hook in hooks:
                    stack.enter_context(hook())
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as e:  # argparse errors, ``sys.exit(1)`` after a logged failure
                if e.code not in (0, None):
                    returncode = e.code if isinstance(e.code, int) else 1
                    if not isinstance(e.code, int):
                        print(e.code, file=sys.stderr)
            except Exception:
                traceback.print_exc()
                returncode = 1
    except BaseException as e:
        e.captured_output = _drain(sink)
        raise
    finally:
        sys.stdout, sys.stderr = saved_streams
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        os.close(saved_fds[0])
        os.close(saved_fds[1])
    return returncode, _drain(sink)


def _preload(modules: Iterable[str]) -> None:
    """Import what ``_restored_state`` snapshots, so a first step cannot change it unnoticed."""
    for name in modules:
        with contextlib.suppress(ImportError):
            importlib.import_module(name)


def _open_sink(path: str):
    return open(path, "w+", encoding="utf-8", errors="replace", buffering=1)


def _worker_main(conn, sink_path, hooks, examples_dir, preload) -> None:
    # Own session and process group: killing the worker takes a step's children along.
    os.setsid()
    # Steps may start DataLoader workers or Pools, which a daemonic process is not allowed to.
    multiprocessing.current_process().daemon = False
    _preload(preload)
    sink = _open_sink(sink_path)
    while True:
        try:
            request = conn.recv()
        except EOFError:
            return
        if request is None:
            return
        cmd, cwd, env = request
        try:
            reply = run_step(cmd, cwd, env, sink, hooks, Path(examples_dir))
        except BaseException:
            reply = (1, traceback.format_exc())
        conn.send(reply)


class ExampleRunner:
    """Run the example steps listed in ``scripts`` in a long-lived worker process.

    Install it with ``run_command.set_in_process_runner``. ``scripts`` are the names of the scripts
    (as in ``["python", name, ...]``) known to be safe to run repeatedly in one process. ``hooks``
    wrap every step, see ``run_step``; the worker imports them, so they must be module-level.
    ``timeout`` bounds one step; by default only pytest-timeout does, and it kills the worker too.
    ``preload`` names modules imported before the first step.
    """

    def __init__(
        self,
        scripts: Iterable[str],
        hooks: Iterable[Callable] = (),
        examples_dir: Path | str = EXAMPLES_DIR,
        timeout: float | None = None,
        preload: Iterable[str] = ("torch",),
        in_process: bool | None = None,
    ):
        self.scripts = frozenset(scripts)
        self._hooks = tuple(hooks)
        self._examples_dir = Path(examples_dir)
        self._timeout = timeout
        self._preload = tuple(preload)
        if in_process is None:
            in_process = os.environ.get("MODELOPT_EXAMPLE_RUNNER") == "inprocess"
        self._in_process = in_process
        self._sink_dir = tempfile.mkdtemp(prefix="example_runner_")
        self._sink_path = os.path.join(self._sink_dir, "output.txt")
        # In-process only, and never closed: a handler built during a step may still hold it.
        self._sink = None
        self._proc = None
        self._conn = None

    def accepts(self, cmd_parts: list[str]) -> bool:
        """Whether the step is ``python <one of scripts> ...``."""
        interpreters = {"python", "python3", Path(sys.executable).name}
        return (
            len(cmd_parts) > 1
            and Path(str(cmd_parts[0])).name in interpreters
            and str(cmd_parts[1]) in self.scripts
        )

    def __call__(
        self,
        cmd_parts: list[str],
        example_path: str,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Run a step in ``examples/<example_path>`` and return its output.

        Raises ``StepFailed`` or ``StepTimeout``. ``env`` is the step's whole environment; ``None``
        keeps the worker's own.
        """
        cmd = [str(part) for part in cmd_parts]
        cwd = self._examples_dir / example_path
        timeout = self._timeout if timeout is None else timeout
        output = ""
        try:
            if self._in_process:
                output = self._run_in_process(cmd, cwd, env)
            else:
                output = self._run_in_worker(cmd, cwd, env, timeout)
            return output
        except BaseException as e:
            output = getattr(e, "captured_output", "")
            raise
        finally:
            print(output)  # keep the step's output in the test log

    def close(self) -> None:
        self._stop_worker()
        shutil.rmtree(self._sink_dir, ignore_errors=True)

    def _run_in_process(self, cmd: list[str], cwd: Path, env: dict[str, str] | None) -> str:
        if self._sink is None:
            _preload(self._preload)
            self._sink = _open_sink(self._sink_path)
        returncode, output = run_step(cmd, cwd, env, self._sink, self._hooks, self._examples_dir)
        if returncode != 0:
            raise StepFailed(returncode, cmd, output)
        return output

    def _run_in_worker(
        self, cmd: list[str], cwd: Path, env: dict[str, str] | None, timeout: float | None
    ) -> str:
        self._start_worker()
        try:
            self._conn.send((cmd, str(cwd), env))
            reply = self._conn.recv() if self._conn.poll(timeout) else None
        except (EOFError, OSError):
            # The worker died during the step: segfault, OOM kill.
            output = self._partial_output()
            raise StepFailed(self._kill_worker(), cmd, output) from None
        except BaseException as e:
            # Interrupted while waiting, e.g. by pytest-timeout: do not leave the step running.
            e.captured_output = self._partial_output()
            self._kill_worker()
            raise
        if reply is None:
            # Hung: only this step fails, the next one gets a fresh worker.
            output = self._partial_output()
            self._kill_worker()
            raise StepTimeout(cmd, timeout, output)
        returncode, output = reply
        if returncode != 0:
            self._stop_worker()  # do not trust the state a failed step left behind
            raise StepFailed(returncode, cmd, output)
        return output

    def _start_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        context = multiprocessing.get_context("spawn")
        self._conn, child_conn = context.Pipe()
        args = (child_conn, self._sink_path, self._hooks, str(self._examples_dir), self._preload)
        self._proc = context.Process(target=_worker_main, args=args, daemon=True)
        self._proc.start()
        child_conn.close()  # only the worker holds this end, so its death shows up as EOF

    def _partial_output(self) -> str:
        try:
            return Path(self._sink_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _kill_worker(self) -> int:
        """SIGKILL the worker's process group and return the worker's exit code."""
        proc, self._proc = self._proc, None
        with contextlib.suppress(ProcessLookupError, PermissionError):
            # Before the pid is reaped, so it cannot have been recycled.
            os.killpg(proc.pid, signal.SIGKILL)
        proc.join(_KILL_TIMEOUT_S)
        return -1 if proc.exitcode is None else proc.exitcode

    def _stop_worker(self) -> None:
        """Let the worker exit on its own, so its coverage data is written, or kill it."""
        proc = self._proc
        if proc is None:
            return
        if proc.is_alive():
            with contextlib.suppress(OSError):
                self._conn.send(None)
            proc.join(_WORKER_EXIT_TIMEOUT_S)
        if proc.is_alive():
            self._kill_worker()
        else:
            self._proc = None
