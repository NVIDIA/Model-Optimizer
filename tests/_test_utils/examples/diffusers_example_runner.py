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
"""Run ``examples/diffusers`` scripts in the pytest process instead of one subprocess per step.

A ``python <script>.py`` launch spends ~20s importing torch/diffusers/modelopt and initializing
CUDA before doing any work, which is most of a step on the tiny test models. Here the script runs
through ``runpy.run_path(..., run_name="__main__")``, so its argument parsing and ``__main__``
block run exactly as on the command line, and the process-wide state a script changes is put back
afterwards: sibling modules it imported by bare name (``config``, ``utils``, ``quantize``, ...),
the ``torch.nn.RMSNorm`` swap in ``quantize.py``/``diffusion_trt.py``, logging, warning filters,
the environment, the working directory and ``sys.path``.
"""

import contextlib
import gc
import io
import logging
import os
import runpy
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
from _test_utils.examples.run_command import MODELOPT_ROOT


def _local_module_names(example_dir: Path) -> set[str]:
    """Top-level names a script in ``example_dir`` can import by bare name."""
    return {p.stem for p in example_dir.glob("*.py")} | {
        p.name for p in example_dir.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
    }


def _pop_modules(names: set[str]) -> dict:
    """Remove ``names`` and their submodules from ``sys.modules``; return what was removed."""
    popped = {key: mod for key, mod in sys.modules.items() if key.split(".", 1)[0] in names}
    for key in popped:
        del sys.modules[key]
    return popped


def _all_loggers() -> list[logging.Logger]:
    with logging._lock:  # any getLogger() from a background thread mutates loggerDict
        others = list(logging.Logger.manager.loggerDict.values())
    return [logging.getLogger(), *(lg for lg in others if isinstance(lg, logging.Logger))]


@contextlib.contextmanager
def _restored_logging(stream: io.StringIO):
    """Put back logger levels, and drop handlers left writing to ``stream`` once it is discarded.

    ``quantize.py`` adds a ``StreamHandler(sys.stdout)`` on every run, and ``sys.stdout`` is
    ``stream`` while it runs.
    """
    levels = {lg: lg.level for lg in _all_loggers()}
    try:
        yield
    finally:
        for lg in _all_loggers():
            for handler in list(lg.handlers):
                if getattr(handler, "stream", None) is stream:
                    lg.removeHandler(handler)
            if lg in levels:
                lg.setLevel(levels[lg])


def run_example_in_process(cmd_parts: list[str], example_path: str) -> str:
    """Run ``python <script>.py <args>`` from ``examples/<example_path>`` here. Returns its output.

    Only stdout/stderr are captured (what tests assert on); fd-level writes from native code go
    to pytest's own capture as before.
    """
    if len(cmd_parts) < 2 or cmd_parts[0] != "python" or not str(cmd_parts[1]).endswith(".py"):
        raise ValueError(f"Expected 'python <script>.py ...', got {cmd_parts}")
    example_dir = MODELOPT_ROOT / "examples" / example_path
    script = example_dir / str(cmd_parts[1])
    argv = [str(script), *(str(p) for p in cmd_parts[2:])]

    local_names = _local_module_names(example_dir)
    # Same-named modules imported elsewhere would shadow the script's own siblings.
    shadowed = _pop_modules(local_names)
    rmsnorm = (torch.nn.RMSNorm, torch.nn.modules.normalization.RMSNorm)
    env_before = os.environ.copy()
    cwd = os.getcwd()
    output = io.StringIO()
    try:
        sys.path.insert(0, str(example_dir))  # as `python <script>.py` does
        os.chdir(example_dir)  # match the subprocess path, which runs from the example dir
        with (
            _restored_logging(output),
            warnings.catch_warnings(),
            patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(output),
        ):
            runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            err = RuntimeError(f"{script.name} exited with code {e.code}")
            err.captured_output = output.getvalue()
            raise err from e
    except BaseException as e:
        # The caller matches transient-HuggingFace markers on this alongside the traceback.
        e.captured_output = output.getvalue()
        raise
    finally:
        os.chdir(cwd)
        sys.path.remove(str(example_dir))
        _pop_modules(local_names)
        sys.modules.update(shadowed)
        torch.nn.RMSNorm, torch.nn.modules.normalization.RMSNorm = rmsnorm
        for key in os.environ.keys() - env_before.keys():
            del os.environ[key]
        os.environ.update(env_before)
        # The pipeline is unreachable once the script returns; give its memory back before the
        # next step rather than whenever the allocator happens to need it.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(output.getvalue())  # keep the step's output in the test log
    return output.getvalue()
