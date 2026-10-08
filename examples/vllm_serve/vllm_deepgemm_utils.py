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

"""Serve with a custom DeepGEMM: a prebuilt wheel, e.g. of a fork with the indexer scorer numerics.

vLLM imports a ``deep_gemm`` on the Python path before its vendored copy. ``--deepgemm-url`` (or
``$DEEPGEMM_URL``) names a package index with prebuilt DeepGEMM wheels, or one wheel. From an index,
pip selects the newest version that has a wheel for this Python and machine, or the version that
``--deepgemm-version`` (``$DEEPGEMM_VERSION``) pins. Before vLLM starts its workers, the launcher
installs the wheel with ``pip install --target`` into a folder of its cache, once per version and
source and under a lock, checks in a subprocess that ``import deep_gemm`` loads it with the installed
torch, and puts the folder first on the Python path of the server's processes. Nothing is built,
and the Python environment itself does not change.

Nothing here imports vLLM, so this can be tested without it.
"""

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote, urlsplit

import torch
from packaging import tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

# TODO: make this generic when another prebuilt kernel package is needed. Only the names below, the
# flags and the import check are DeepGEMM's. It only helps packages that vLLM imports by name from
# the Python path; vLLM builds some, such as FlashMLA, into its own wheel.
DEEPGEMM_URL_ENV = "DEEPGEMM_URL"
DEEPGEMM_VERSION_ENV = "DEEPGEMM_VERSION"
# The root of the cache of installed DeepGEMM wheels (see deepgemm_cache_dir).
DEEPGEMM_CACHE_ENV = "MODELOPT_DEEPGEMM_CACHE_DIR"

_PROJECT = "deep-gemm"  # the normalized name of deep_gemm on an index
# Wheels only: nothing is built, e.g. from the unrelated deep_gemm source distribution on PyPI. An
# installed deep_gemm does not count, or pip would select nothing that it satisfies.
_PIP_OPTIONS = (
    "--no-deps",
    "--no-input",
    "--disable-pip-version-check",
    "--ignore-installed",
    "--only-binary",
    ":all:",
)
# A pip run reads an index and downloads a wheel of tens of MB, while it can hold the cache lock.
_PIP_TIMEOUT = 1800  # seconds
_IMPORT_TIMEOUT = 600  # seconds
# Installs stage in the cache under its lock; staging folders older than this are left by launches
# that were killed while installing.
_STALE_STAGING = 24 * 3600  # seconds
# Run by a fresh interpreter with the installed folder first on the Python path, like the workers.
_IMPORT_CHECK = "import deep_gemm; print(deep_gemm.__file__)"


def add_deepgemm_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--deepgemm-url`` and ``--deepgemm-version`` to the launcher's parser."""
    parser.add_argument(
        "--deepgemm_url",
        "--deepgemm-url",
        default=None,
        help=(
            "Serve with the DeepGEMM wheel that pip selects from this package index "
            "(https://.../simple), or with this wheel (a path or URL), installed into a cache "
            "folder that is put first on the Python path before vLLM starts. Default: "
            f"${DEEPGEMM_URL_ENV}."
        ),
    )
    parser.add_argument(
        "--deepgemm_version",
        "--deepgemm-version",
        default=None,
        help=(
            "The version to install from the package index of --deepgemm-url instead of the "
            "newest one, e.g. 2.8.0.post210 or, exactly, 2.8.0.post210+g6fec872c.torch2.13.cu130. "
            f"Default: ${DEEPGEMM_VERSION_ENV}."
        ),
    )


def resolve_deepgemm_args(args: argparse.Namespace) -> None:
    """Install the DeepGEMM that ``--deepgemm-url`` or ``$DEEPGEMM_URL`` names, if any, and use it."""
    url = args.deepgemm_url or os.environ.get(DEEPGEMM_URL_ENV)
    version = args.deepgemm_version or os.environ.get(DEEPGEMM_VERSION_ENV)
    if not url:
        if version:
            raise ValueError(
                f"{DEEPGEMM_VERSION_ENV} {version} pins a version of the package index that "
                f"{DEEPGEMM_URL_ENV} names, which is not set."
            )
        return
    folder, installed = install_deepgemm(url, version)
    prepend_python_path(folder)
    print(f"[deepgemm] serving with deep_gemm {installed} from {url}: {folder}")


def install_deepgemm(url: str, version: str | None = None) -> tuple[Path, str]:
    """Install the DeepGEMM that ``url`` names into a folder of the cache, unless it is there.

    Args:
        url: A package index, or a wheel: a path, or an http(s) or file URL.
        version: The version to install from the package index ``url``.

    Returns:
        The folder, from which ``import deep_gemm`` loads the installed DeepGEMM with the installed
        torch, and the installed version.
    """
    name = unquote(urlsplit(url).path) if "://" in url else url
    if name.endswith(".whl"):
        if version is not None:
            raise ValueError(
                f"{DEEPGEMM_VERSION_ENV} pins a version of a package index, but {DEEPGEMM_URL_ENV} "
                f"{url} is a wheel."
            )
        installed, requirement = _wheel_version(Path(name).name), [url]
    else:
        pin = _parse_version(version) if version is not None else None
        # A version with its local label is exact: once installed, the index is not needed.
        installed = str(pin) if pin is not None and pin.local else _resolve(url, pin)
        requirement = ["--index-url", url, f"deep_gemm=={installed}"]
    wheels = deepgemm_cache_dir() / "wheels"
    folder = wheels / _folder_name(installed, url)
    if not folder.is_dir():  # complete, as published by a rename; a read-only cache works too
        with _locked(wheels):
            if not folder.is_dir():
                print(f"[deepgemm] installing deep_gemm {installed} from {url}", flush=True)
                _pip_install(requirement, folder, url, installed)  # checks the import first
                return folder, installed
    _check_import(folder, installed, url)  # again: the environment can change
    return folder, installed


def prepend_python_path(folder: Path) -> None:
    """Put ``folder`` first on the Python path of this process and of the processes it starts."""
    path = str(folder)
    if path not in sys.path:
        sys.path.insert(0, path)
        importlib.invalidate_caches()
    python_path = os.environ.get("PYTHONPATH")
    if not python_path:
        os.environ["PYTHONPATH"] = path
    elif python_path.split(os.pathsep)[0] != path:
        os.environ["PYTHONPATH"] = f"{path}{os.pathsep}{python_path}"


def deepgemm_cache_dir() -> Path:
    """The root of the cache of installed DeepGEMM wheels.

    ``$MODELOPT_DEEPGEMM_CACHE_DIR``, else ``modelopt/deepgemm`` under ``$VLLM_CACHE_ROOT``, else
    ``~/.cache/modelopt/deepgemm``.
    """
    if root := os.environ.get(DEEPGEMM_CACHE_ENV):
        return Path(root).expanduser().absolute()
    if vllm_root := os.environ.get("VLLM_CACHE_ROOT"):
        return Path(vllm_root).expanduser().absolute() / "modelopt" / "deepgemm"
    return Path.home() / ".cache" / "modelopt" / "deepgemm"


def _pip() -> list[str]:
    # subprocess: pip has no supported Python API. Safe: an argument list without a shell, and pip
    # installs only wheels, from the index or wheel that the user passes and must trust like any
    # package source.
    return [sys.executable, "-m", "pip", "install", *_PIP_OPTIONS]


def _run_pip(command: list[str], what: str, **kwargs) -> subprocess.CompletedProcess:
    """Run the pip ``command`` for ``what`` within ``_PIP_TIMEOUT``."""
    try:
        return subprocess.run(command, timeout=_PIP_TIMEOUT, check=False, **kwargs)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"pip did not finish {what} in {_PIP_TIMEOUT} s.") from None


def _parse_version(version: str) -> Version:
    try:
        return Version(version)
    except InvalidVersion:
        raise ValueError(f"{DEEPGEMM_VERSION_ENV} {version!r} is no version.") from None


def _wheel_version(name: str) -> str:
    """The version of the deep_gemm wheel file ``name``."""
    try:
        project, version, _, _ = parse_wheel_filename(name)
    except InvalidWheelFilename as error:
        raise ValueError(f"{name}: {error}") from None
    if canonicalize_name(project) != _PROJECT:
        raise ValueError(f"{name} is no deep_gemm wheel.")
    return str(version)


def _resolve(index: str, pin: Version | None) -> str:
    """The version of deep_gemm that pip selects on ``index``: the pinned or the newest one."""
    requirement = f"deep_gemm=={pin}" if pin is not None else "deep_gemm"
    # pip prints its own errors, such as an http index on another host without --trusted-host.
    result = _run_pip(
        [*_pip(), "--dry-run", "--quiet", "--report", "-", "--index-url", index, requirement],
        f"selecting deep_gemm on {index}",
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pip finds no deep_gemm wheel for {_python_tag()} on {platform.machine()} on {index}, "
            f"see its output above. {DEEPGEMM_URL_ENV} takes a package index (https://.../simple) "
            "or a wheel; to serve a DeepGEMM source tree, install it into the environment with "
            "`pip install --no-deps --no-build-isolation <source>` instead."
        )
    selected = json.loads(result.stdout)["install"]
    if len(selected) != 1:  # --no-deps and --ignore-installed: deep_gemm only
        raise RuntimeError(f"pip selected {len(selected)} packages for deep_gemm on {index}.")
    return selected[0]["metadata"]["version"]


def _python_tag() -> str:
    return f"{tags.interpreter_name()}{tags.interpreter_version()}"


def _folder_name(version: str, url: str) -> str:
    """The cache folder of ``version`` from ``url``, which no other source of it shares."""
    return f"{version}-{_python_tag()}-{platform.machine()}-{_source_id(url)}"


def _source_id(url: str) -> str:
    """A local wheel by its content, as a rebuild can keep its version; else the URL."""
    parts = urlsplit(url)
    path = Path(unquote(parts.path) if parts.scheme == "file" else url)
    if parts.scheme in ("", "file") and path.is_file():
        source = path.read_bytes()
    else:
        source = url.rstrip("/").encode()
    return hashlib.sha256(source).hexdigest()[:12]


def _pip_install(requirement: list[str], folder: Path, url: str, version: str) -> None:
    """``pip install --target`` into ``folder``: published complete, and only if it imports.

    Everyone can read it.
    """
    staging = Path(tempfile.mkdtemp(prefix=".install-", dir=folder.parent))
    try:
        command = [*_pip(), "--target", str(staging), *requirement]
        if _run_pip(command, f"installing deep_gemm from {url}").returncode != 0:
            raise RuntimeError(f"Installing deep_gemm from {url} failed, see the pip output above.")
        for directory, _, files in os.walk(staging):
            os.chmod(directory, 0o755)  # mkdtemp's 0700 would hide it from other users
            for path in (os.path.join(directory, name) for name in files):
                if not os.path.islink(path):
                    os.chmod(path, os.stat(path).st_mode | 0o444)
        _check_import(staging, version, url)
        try:
            staging.rename(folder)
        except OSError:
            if not folder.is_dir():  # else another process published it first
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _last_line(text: str) -> str:
    return (text.strip().splitlines() or [""])[-1]


def _check_import(folder: Path, version: str, url: str) -> None:
    """Check that ``import deep_gemm`` loads the one installed in ``folder``, with this torch.

    vLLM would fall back to its own copy without an error.
    """
    python_path = os.environ.get("PYTHONPATH")
    env = {
        **os.environ,
        "PYTHONPATH": f"{folder}{os.pathsep}{python_path}" if python_path else str(folder),
    }
    try:
        # subprocess: a fresh interpreter, as a worker starts, runs the fixed _IMPORT_CHECK without
        # a shell.
        result = subprocess.run(
            [sys.executable, "-c", _IMPORT_CHECK],
            capture_output=True,
            text=True,
            env=env,
            cwd=folder,
            timeout=_IMPORT_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"import deep_gemm {version} from {url} timed out.") from None
    if result.returncode != 0:
        error = _last_line(result.stderr) or "no error message"
        raise RuntimeError(
            f"deep_gemm {version} from {url} does not import ({error}): it needs a build for this "
            f"torch {torch.__version__} with CUDA {torch.version.cuda} and glibc "
            f"{platform.libc_ver()[1] or '?'} or older. Pin one with --deepgemm-version or "
            f"{DEEPGEMM_VERSION_ENV}."
        )
    loaded = Path(_last_line(result.stdout)).resolve()
    if not loaded.is_relative_to(folder.resolve()):
        raise RuntimeError(
            f"import deep_gemm loads {loaded} instead of deep_gemm {version} from {url}: "
            "another deep_gemm comes first on the Python path, such as an editable install. "
            "Uninstall it."
        )


@contextlib.contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """Hold an exclusive lock on the cache ``directory``, which it creates.

    Also removes the old staging folders that launches killed while installing left in it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[deepgemm] waiting for another launch installing into {directory}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        for staging in directory.glob(".install-*"):
            with contextlib.suppress(OSError):
                if time.time() - staging.stat().st_mtime > _STALE_STAGING:
                    shutil.rmtree(staging, ignore_errors=True)
        yield
