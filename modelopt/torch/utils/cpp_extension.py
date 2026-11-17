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

"""Utility functions for loading CPP / CUDA extensions."""

import contextlib
import functools
import os
import shutil
import sysconfig
import warnings
from collections.abc import Iterator
from pathlib import Path
from time import time
from types import ModuleType
from typing import Any

import torch
import torch.utils.cpp_extension as torch_cpp_extension
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from torch.utils.cpp_extension import load

__all__ = ["load_cpp_extension"]

_IS_WINDOWS = os.name == "nt"


@functools.cache
def _get_msvc_env() -> dict[str, str]:
    """Return the MSVC build environment (``vcvarsall.bat``) for the host, or ``{}`` if unavailable."""
    plat_spec = {"win-amd64": "amd64", "win-arm64": "arm64"}.get(sysconfig.get_platform())
    if plat_spec is None:
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # setuptools warns that _get_vc_env is private
            return torch_cpp_extension._get_vc_env(plat_spec)
    except Exception:
        return {}


@functools.cache
def _warn_msvc_env_activated() -> None:
    warnings.warn(
        "MSVC C++ compiler (cl.exe) was not found on PATH; ModelOpt set up the Visual Studio C++"
        " build environment to build and load its CUDA extensions."
    )


@contextlib.contextmanager
def _temporary_environ(env: dict[str, str]) -> Iterator[None]:
    """Apply ``env`` to ``os.environ`` and restore the previous values on exit."""
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_cpp_extension(
    name: str,
    sources: list[str | Path],
    cuda_version_specifiers: str | None,
    fail_msg: str = "",
    raise_if_failed: bool = False,
    **load_kwargs: Any,
) -> ModuleType | None:
    """Load a C++ / CUDA extension using torch.utils.cpp_extension.load() if the current CUDA version satisfies it.

    Loading first time may take a few mins because of the compilation, but subsequent loads are instantaneous.

    Args:
        name: Name of the extension.
        sources: Source files to compile.
        cuda_version_specifiers: Specifier (e.g. ">=11.8,<12") for CUDA versions required to enable the extension.
        fail_msg: Additional message to display if the extension fails to load.
        raise_if_failed: Raise an exception if the extension fails to load.
        **load_kwargs: Keyword arguments to torch.utils.cpp_extension.load().
    """
    ext = None
    print(f"Loading extension {name}...")
    start = time()
    # Outside a Visual Studio developer prompt, torch's JIT build runs ``where cl`` (to find
    # ``link.exe``) before it activates MSVC itself, and it cannot activate MSVC on Windows ARM64.
    # That breaks every first load in a process, even when the cached build is up to date, so we
    # activate MSVC ourselves when ``cl`` is missing.
    msvc_env: dict[str, str] = {}

    if torch.version.cuda is None or not torch.cuda.is_available():
        fail_msg = f"Skipping extension {name} because CUDA is not available."
    elif cuda_version_specifiers and Version(torch.version.cuda) not in SpecifierSet(
        cuda_version_specifiers
    ):
        fail_msg = (
            f"Skipping extension {name} because the current CUDA version {torch.version.cuda}"
            f" does not satisfy the specifiers {cuda_version_specifiers}."
        )
    elif _IS_WINDOWS and not shutil.which("cl") and not (msvc_env := _get_msvc_env()):
        fail_msg = (
            f"Cannot build extension {name}: MSVC C++ compiler (cl.exe) not found on PATH and no"
            " Visual Studio C++ build tools were found. Install Visual Studio Build Tools with the"
            " C++ workload (ARM64 tools on Windows ARM64), or run from a Visual Studio Developer"
            f" prompt.\n{fail_msg}"
        ).rstrip()
    else:
        if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
            device_capability = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{device_capability[0]}.{device_capability[1]}"
        if _IS_WINDOWS:
            # Define USE_CUDA so PyTorch's compiled_autograd.h takes its Windows-safe branch;
            # otherwise, nvcc + MSVC fail with "error C2872: 'std': ambiguous symbol".
            # See https://github.com/pytorch/pytorch/issues/148317
            for key in ("extra_cflags", "extra_cuda_cflags"):
                flags = list(load_kwargs.get(key, []))
                if not any("USE_CUDA" in flag for flag in flags):
                    flags.append("-DUSE_CUDA=1")
                load_kwargs[key] = flags
        if msvc_env:
            _warn_msvc_env_activated()
        try:
            with _temporary_environ(msvc_env):
                ext = load(name, sources, **load_kwargs)
        except Exception as e:
            # torch records the build inputs before building, so a retry in this process would
            # skip the build and import whatever library is on disk, unchecked. Forget them so a
            # retry builds again (or fails the same way).
            torch_cpp_extension.JIT_EXTENSION_VERSIONER.entries.pop(name, None)
            if not fail_msg:
                fail_msg = f"Unable to load extension {name} and falling back to CPU version."
            fail_msg = f"{e}\n{fail_msg}"
            # RuntimeError can be raised if there are any errors while compiling the extension.
            # OSError can be raised if CUDA_HOME path is not set correctly.
            # subprocess.CalledProcessError can be raised on `-runtime` images where c++ is not installed.

    if ext is None:
        if raise_if_failed:
            raise RuntimeError(fail_msg)
        else:
            warnings.warn(fail_msg)
    else:
        print(f"Loaded extension {name} in {time() - start:.1f} seconds")
    return ext
