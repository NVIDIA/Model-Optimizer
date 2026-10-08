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

"""Installing the DeepGEMM that the vLLM fake-quant server imports, into a folder of a cache.

``vllm_deepgemm_utils`` imports no vLLM. pip and the import check are stubbed, except in the last
tests, which install tiny pure-Python ``deep_gemm`` wheels with the real pip, from a local package
index or a file, and check the import in a real subprocess.
"""

import argparse
import base64
import fcntl
import hashlib
import importlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "vllm_serve"

LABEL = "torch2.13.cu130"
NEWEST = f"2.8.0.post5+gbbbbbbb.{LABEL}"
ANY = "py3-none-any"  # the tags of a wheel that every Python installs
INDEX = "https://pypi.example.com/simple"
GLIBC = platform.libc_ver()[1] or "?"
PIP = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-input"]
PIP += ["--disable-pip-version-check", "--ignore-installed", "--only-binary", ":all:"]


def _isolated_module(monkeypatch, tmp_path):
    """``vllm_deepgemm_utils`` with HOME under ``tmp_path``.

    The variables it reads are unset; ``PYTHONPATH`` and ``sys.path``, which it sets, are restored.
    """
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))  # also restores sys.path
    module = importlib.import_module("vllm_deepgemm_utils")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in (
        module.DEEPGEMM_URL_ENV,
        module.DEEPGEMM_VERSION_ENV,
        module.DEEPGEMM_CACHE_ENV,
        "VLLM_CACHE_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    python_path = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", python_path or "")
    if python_path is None:
        monkeypatch.delenv("PYTHONPATH")
    return module


@pytest.fixture
def deepgemm(monkeypatch, tmp_path):
    return _isolated_module(monkeypatch, tmp_path)


def _wheels():
    return Path.home() / ".cache/modelopt/deepgemm/wheels"


class _Subprocesses:
    """Stubbed pip and import check.

    A pip dry run selects ``selected``, pip installs a fake deep_gemm, and the check loads it.
    Records each command with its keyword arguments.
    """

    def __init__(self, deepgemm, monkeypatch, returncode=0):
        self.calls, self.returncode, self.import_check = [], returncode, None
        self.selected = NEWEST
        monkeypatch.setattr(deepgemm.subprocess, "run", self)

    def __call__(self, command, **kwargs):
        self.calls.append(SimpleNamespace(command=command, kwargs=kwargs))
        if command[1] == "-c":
            folder = Path(kwargs["cwd"])
            assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(folder)
            if self.import_check is not None:
                return self.import_check(command)
            stdout = f"{folder / 'deep_gemm/__init__.py'}\n"
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        if self.returncode != 0:
            return subprocess.CompletedProcess(command, self.returncode, stdout="")
        if "--dry-run" in command:
            report = {"install": [{"metadata": {"name": "deep_gemm", "version": self.selected}}]}
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(report))
        target = _target(command)
        (target / "deep_gemm").mkdir()
        (target / "deep_gemm/__init__.py").write_text("__version__ = '2.8.0'\n")
        (target / "deep_gemm/_C.so").write_bytes(b"\x7fELF")
        (target / "deep_gemm/_C.so").chmod(0o700)
        return subprocess.CompletedProcess(command, 0)

    def pip(self, dry_run=False):
        """The pip installs, or the pip dry runs."""
        return [
            call
            for call in self.calls
            if call.command[1:4] == ["-m", "pip", "install"]
            and ("--dry-run" in call.command) == dry_run
        ]


def _target(command):
    return Path(command[command.index("--target") + 1])


def test_index_installs_the_version_pip_selects_once(deepgemm, monkeypatch):
    subprocesses = _Subprocesses(deepgemm, monkeypatch)
    folder, version = deepgemm.install_deepgemm(INDEX)

    assert (folder, version) == (_wheels() / deepgemm._folder_name(NEWEST, INDEX), NEWEST)
    source = hashlib.sha256(INDEX.encode()).hexdigest()[:12]
    assert folder.name == f"{NEWEST}-{deepgemm._python_tag()}-{os.uname().machine}-{source}"
    (dry_run,) = subprocesses.pip(dry_run=True)
    assert dry_run.command == [
        *PIP,
        *["--dry-run", "--quiet", "--report", "-", "--index-url", INDEX, "deep_gemm"],
    ]
    # Only the report is captured: pip's own errors reach the terminal.
    assert dry_run.kwargs["stdout"] == subprocess.PIPE and "stderr" not in dry_run.kwargs
    assert dry_run.kwargs["timeout"] == deepgemm._PIP_TIMEOUT
    (install,) = subprocesses.pip()
    assert install.kwargs["timeout"] == deepgemm._PIP_TIMEOUT
    staging = _target(install.command)
    assert staging.parent == _wheels() and not staging.exists()  # published by a rename
    assert install.command == [
        *PIP,
        *["--target", str(staging), "--index-url", INDEX, f"deep_gemm=={NEWEST}"],
    ]
    assert (folder / "deep_gemm/__init__.py").is_file()
    assert subprocesses.calls[-1].command == [sys.executable, "-c", deepgemm._IMPORT_CHECK]

    # Installed: pip selects the version again, and only the import is checked.
    subprocesses.calls.clear()
    assert deepgemm.install_deepgemm(INDEX) == (folder, version)
    assert len(subprocesses.pip(dry_run=True)) == 1 and subprocesses.pip() == []


def test_index_version_can_be_pinned(deepgemm, monkeypatch):
    subprocesses = _Subprocesses(deepgemm, monkeypatch)
    older = f"2.8.0.post3+gaaaaaaa.{LABEL}"
    subprocesses.selected = older
    # Without its local label: pip selects the build of that version.
    assert deepgemm.install_deepgemm(INDEX, "2.8.0.post3")[1] == older
    (dry_run,) = subprocesses.pip(dry_run=True)
    assert dry_run.command[-1] == "deep_gemm==2.8.0.post3"
    assert subprocesses.pip()[-1].command[-1] == f"deep_gemm=={older}"

    # With it: exact, so pip does not select, and once it is installed pip does not run.
    subprocesses.calls.clear()
    exact = "2.8.0.post6+gccccccc.torch2.14.cu130"
    assert deepgemm.install_deepgemm(INDEX, exact.upper()) == (  # normalized
        _wheels() / deepgemm._folder_name(exact, INDEX),
        exact,
    )
    assert subprocesses.pip(dry_run=True) == []
    assert subprocesses.pip()[-1].command[-1] == f"deep_gemm=={exact}"
    subprocesses.calls.clear()
    deepgemm.install_deepgemm(INDEX, exact)
    assert [call.command[1] for call in subprocesses.calls] == ["-c"]


@pytest.mark.parametrize("form", ["path", "file", "https"])
def test_wheel_is_installed_once_per_version(deepgemm, monkeypatch, tmp_path, form):
    subprocesses = _Subprocesses(deepgemm, monkeypatch)
    wheel = tmp_path / f"deep_gemm-{NEWEST}-{ANY}.whl"
    url = {
        "path": str(wheel),
        "file": wheel.as_uri(),
        "https": f"https://files.example.com/{wheel.name.replace('+', '%2B')}",
    }[form]
    folder, version = deepgemm.install_deepgemm(url)
    assert (folder, version) == (_wheels() / deepgemm._folder_name(NEWEST, url), NEWEST)
    assert subprocesses.pip(dry_run=True) == []  # the file name has the version
    (install,) = subprocesses.pip()
    assert install.command == [*PIP, "--target", str(_target(install.command)), url]
    assert deepgemm.install_deepgemm(url) == (folder, version)
    assert len(subprocesses.pip()) == 1


def test_sources_of_a_version_install_separately(deepgemm, monkeypatch, tmp_path):
    """A rebuilt wheel or another index with the same version is not served from the cache."""
    subprocesses = _Subprocesses(deepgemm, monkeypatch)
    wheel = tmp_path / f"deep_gemm-{NEWEST}-{ANY}.whl"
    wheel.write_bytes(b"first build")
    first, _ = deepgemm.install_deepgemm(str(wheel))
    assert deepgemm.install_deepgemm(wheel.as_uri())[0] == first  # the same file
    wheel.write_bytes(b"rebuilt with the same version")
    rebuilt, _ = deepgemm.install_deepgemm(str(wheel))
    index, _ = deepgemm.install_deepgemm(INDEX)
    other_index, _ = deepgemm.install_deepgemm("https://other.example.com/simple")
    assert len({first, rebuilt, index, other_index}) == 4
    assert all(folder.is_dir() for folder in (first, rebuilt, index, other_index))
    assert len(subprocesses.pip()) == 4
    # Each source reuses its own folder.
    assert deepgemm.install_deepgemm(str(wheel))[0] == rebuilt
    assert deepgemm.install_deepgemm(f"{INDEX}/")[0] == index
    assert len(subprocesses.pip()) == 4


def test_a_build_that_does_not_import_is_not_cached(deepgemm, monkeypatch):
    """The import is checked before the install is published, so a fixed build can be retried."""
    subprocesses = _Subprocesses(deepgemm, monkeypatch)
    subprocesses.import_check = lambda command: subprocess.CompletedProcess(
        command, 1, stdout="", stderr="ImportError: wrong torch\n"
    )
    with pytest.raises(RuntimeError, match=r"does not import \(ImportError: wrong torch\)"):
        deepgemm.install_deepgemm(INDEX, NEWEST)
    assert [path.name for path in _wheels().iterdir()] == [".lock"]  # no folder, no staging
    subprocesses.import_check = None  # fixed, e.g. the environment
    folder, _ = deepgemm.install_deepgemm(INDEX, NEWEST)
    assert folder.is_dir() and len(subprocesses.pip()) == 2


def test_pip_timeouts_release_the_cache(deepgemm, monkeypatch):
    """A pip run that hangs fails the launch, leaves no staging folder and releases the lock."""

    def hang(command, **kwargs):
        assert kwargs["timeout"] == deepgemm._PIP_TIMEOUT
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(deepgemm.subprocess, "run", hang)
    with pytest.raises(RuntimeError, match=r"pip did not finish selecting deep_gemm on .* in"):
        deepgemm.install_deepgemm(INDEX)
    with pytest.raises(RuntimeError, match=r"pip did not finish installing deep_gemm from .* in"):
        deepgemm.install_deepgemm(INDEX, NEWEST)  # under the lock
    assert [path.name for path in _wheels().iterdir()] == [".lock"]
    with (_wheels() / ".lock").open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize(
    ("url", "version", "match"),
    [
        (INDEX, "latest", "DEEPGEMM_VERSION 'latest' is no version"),
        (f"https://example.com/deep_gemm-2.8.0-{ANY}.whl", "2.8.0", r"pins .* but .* is a wheel"),
        (
            f"https://example.com/other-1.0-{ANY}.whl",
            None,
            r"other-1.0-py3-none-any.whl is no deep_",
        ),
        ("dist/deep_gemm.whl", None, r"deep_gemm.whl: Invalid wheel filename"),
    ],
    ids=["version", "wheel-pin", "other-wheel", "wheel-name"],
)
def test_inputs_are_checked_before_pip_runs(deepgemm, monkeypatch, url, version, match):
    subprocesses = _Subprocesses(deepgemm, monkeypatch)
    with pytest.raises(ValueError, match=match):
        deepgemm.install_deepgemm(url, version)
    assert subprocesses.calls == []


def test_pip_failures_are_reported(deepgemm, monkeypatch):
    """Nothing is built: a git URL or source tree finds no wheel, with a hint to build one."""
    _Subprocesses(deepgemm, monkeypatch, returncode=1)
    git = "git+https://git.example.com/group/deepgemm.git@main"
    with pytest.raises(RuntimeError) as error:
        deepgemm.install_deepgemm(git)
    assert str(error.value).startswith(
        f"pip finds no deep_gemm wheel for {deepgemm._python_tag()} on {os.uname().machine} on "
        f"{git}, see its output above."
    )
    assert "build its wheel with `pip wheel --no-deps" in str(error.value)
    with pytest.raises(RuntimeError, match=re.escape(f"from {INDEX} failed, see the pip output")):
        deepgemm.install_deepgemm(INDEX, NEWEST)
    assert [path.name for path in _wheels().iterdir()] == [".lock"]  # no folder, no staging


def test_version_needs_a_url(deepgemm):
    with pytest.raises(ValueError, match=r"DEEPGEMM_VERSION 2.8.0 pins .* which is not set"):
        deepgemm.resolve_deepgemm_args(
            argparse.Namespace(deepgemm_url=None, deepgemm_version="2.8.0")
        )


def test_import_check_names_what_went_wrong(deepgemm, monkeypatch, tmp_path):
    subprocesses = _Subprocesses(deepgemm, monkeypatch)
    subprocesses.import_check = lambda command: subprocess.CompletedProcess(
        command,
        1,
        stdout="",
        stderr="Traceback ...\nImportError: _C.so: undefined symbol: _ZN3c10\n",
    )
    with pytest.raises(RuntimeError) as error:
        deepgemm.install_deepgemm(INDEX)
    assert str(error.value).startswith(
        f"deep_gemm {NEWEST} from {INDEX} does not import (ImportError: _C.so: undefined symbol: "
        "_ZN3c10): it needs a build for this torch "
    )
    assert str(error.value).endswith(
        f"and glibc {GLIBC} or older. Pin one with --deepgemm-version or DEEPGEMM_VERSION."
    )

    elsewhere = tmp_path / "src/deep_gemm/__init__.py"
    subprocesses.import_check = lambda command: subprocess.CompletedProcess(
        command, 0, stdout=f"Failed to load legacy kernels\n{elsewhere}\n", stderr=""
    )
    with pytest.raises(RuntimeError, match=re.escape(f"loads {elsewhere} instead of deep_gemm")):
        deepgemm.install_deepgemm(INDEX)


def test_install_holds_the_cache_lock(deepgemm, monkeypatch):
    """Concurrent launches sharing the cache wait while one of them installs."""
    subprocesses, locked = _Subprocesses(deepgemm, monkeypatch), []

    def run(command, **kwargs):
        if command[1:3] == ["-m", "pip"]:
            with (_wheels() / ".lock").open("a") as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(other, fcntl.LOCK_UN)
                    locked.append(False)
                except BlockingIOError:
                    locked.append(True)
        return subprocesses(command, **kwargs)

    _wheels().mkdir(parents=True)
    monkeypatch.setattr(deepgemm.subprocess, "run", run)
    deepgemm.install_deepgemm(INDEX)
    assert locked == [False, True]  # the dry run, then the install


def test_install_rechecks_after_waiting_for_the_lock(deepgemm, monkeypatch, capsys):
    """Another launch installed the same version while this one waited: no second install."""
    subprocesses, flock = _Subprocesses(deepgemm, monkeypatch), fcntl.flock
    folder = _wheels() / deepgemm._folder_name(NEWEST, INDEX)

    def wait_for_other_launch(file, operation):
        if operation & fcntl.LOCK_NB:
            (folder / "deep_gemm").mkdir(parents=True)
            raise BlockingIOError
        return flock(file, operation)

    monkeypatch.setattr(deepgemm.fcntl, "flock", wait_for_other_launch)
    assert deepgemm.install_deepgemm(INDEX) == (folder, NEWEST)
    assert subprocesses.pip() == []
    assert "waiting for another launch" in capsys.readouterr().out


def test_installed_versions_need_no_lock(deepgemm, monkeypatch, tmp_path):
    """A cache that this user cannot write, e.g. shared or read-only, serves what it holds."""
    subprocesses = _Subprocesses(deepgemm, monkeypatch)
    wheel = str(tmp_path / f"deep_gemm-{NEWEST}-{ANY}.whl")
    installed = [deepgemm.install_deepgemm(INDEX), deepgemm.install_deepgemm(wheel)]

    def unwritable(directory):
        raise PermissionError(13, "Permission denied", str(directory / ".lock"))

    monkeypatch.setattr(deepgemm, "_locked", unwritable)
    subprocesses.calls.clear()
    assert [deepgemm.install_deepgemm(INDEX), deepgemm.install_deepgemm(wheel)] == installed
    assert subprocesses.pip() == []


def test_install_removes_what_killed_installs_left(deepgemm, monkeypatch):
    """Installs stage under the lock; staging folders that old are left by killed launches."""
    _Subprocesses(deepgemm, monkeypatch)
    stale, recent = _wheels() / ".install-stale", _wheels() / ".install-recent"
    for staging in (stale, recent):
        (staging / "deep_gemm").mkdir(parents=True)
    day_ago = time.time() - deepgemm._STALE_STAGING - 60
    os.utime(stale, (day_ago, day_ago))
    deepgemm.install_deepgemm(INDEX)
    assert not stale.exists() and recent.is_dir()


def test_installed_folder_is_readable_by_everyone(deepgemm, monkeypatch):
    _Subprocesses(deepgemm, monkeypatch)
    umask = os.umask(0o077)
    try:
        folder, _ = deepgemm.install_deepgemm(INDEX)
    finally:
        os.umask(umask)
    modes = {
        path.relative_to(folder).as_posix(): stat.S_IMODE(path.stat().st_mode)
        for path in [folder, *folder.rglob("*")]
    }
    assert modes == {
        ".": 0o755,
        "deep_gemm": 0o755,
        "deep_gemm/__init__.py": 0o644,
        "deep_gemm/_C.so": 0o744,
    }


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "home/.cache/modelopt/deepgemm"),
        ({"VLLM_CACHE_ROOT": "vllm"}, "vllm/modelopt/deepgemm"),
        ({"VLLM_CACHE_ROOT": "vllm", "MODELOPT_DEEPGEMM_CACHE_DIR": "mine"}, "mine"),
    ],
    ids=["home", "vllm-cache-root", "override"],
)
def test_cache_dir_follows_the_environment(deepgemm, monkeypatch, tmp_path, env, expected):
    _Subprocesses(deepgemm, monkeypatch)
    for name, value in env.items():
        monkeypatch.setenv(name, str(tmp_path / value))
    assert deepgemm.deepgemm_cache_dir() == tmp_path / expected
    folder, _ = deepgemm.install_deepgemm(INDEX)
    assert folder.parent == tmp_path / expected / "wheels"


@pytest.mark.parametrize(
    ("argv", "env", "expected"),
    [
        (["--deepgemm-url", INDEX, "--deepgemm-version", "1"], {"DEEPGEMM_VERSION": "2"}, "1"),
        (["--deepgemm_url", INDEX], {"DEEPGEMM_URL": "/other", "DEEPGEMM_VERSION": "2"}, "2"),
        ([], {"DEEPGEMM_URL": INDEX}, None),
        ([], {}, None),
    ],
    ids=["flags", "underscores", "env", "none"],
)
def test_flags_override_env_and_the_install_is_used(deepgemm, monkeypatch, argv, env, expected):
    """The installed folder comes first on the Python path of the launcher and its processes."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PYTHONPATH", "/user/path")
    installs, folder = [], Path("/cache/wheels/2.8.0.post5-cp312-x86_64")

    def install(url, version):
        installs.append((url, version))
        return folder, "2.8.0.post5"

    monkeypatch.setattr(deepgemm, "install_deepgemm", install)
    parser = argparse.ArgumentParser()
    deepgemm.add_deepgemm_args(parser)
    deepgemm.resolve_deepgemm_args(parser.parse_args(argv))
    if not argv and not env:
        assert installs == [] and os.environ["PYTHONPATH"] == "/user/path"
        assert str(folder) not in sys.path
        return
    assert installs == [(INDEX, expected)]
    assert sys.path[0] == str(folder)
    assert os.environ["PYTHONPATH"] == f"{folder}{os.pathsep}/user/path"
    deepgemm.prepend_python_path(folder)  # again: no change
    assert sys.path.count(str(folder)) == 1
    assert os.environ["PYTHONPATH"] == f"{folder}{os.pathsep}/user/path"


def test_python_path_changes_do_not_outlive_the_test(tmp_path):
    """The fixture's monkeypatch undoes the launcher's Python path changes."""
    environ, path = dict(os.environ), list(sys.path)
    with pytest.MonkeyPatch.context() as monkeypatch:
        deepgemm = _isolated_module(monkeypatch, tmp_path)
        monkeypatch.setattr(deepgemm, "install_deepgemm", lambda url, version: (tmp_path, "2.8.0"))
        deepgemm.resolve_deepgemm_args(argparse.Namespace(deepgemm_url="/x", deepgemm_version=None))
        assert sys.path[0] == str(tmp_path)
    assert dict(os.environ) == environ and sys.path == path


# The real pip installs tiny pure-Python deep_gemm wheels from a file:// index or a file.


def _make_wheel(directory, version, tag=ANY):
    """A pure-Python ``deep_gemm`` wheel of ``version`` whose package records its version."""
    dist_info = f"deep_gemm-{version}.dist-info"
    files = {
        "deep_gemm/__init__.py": f"__version__ = {version!r}\n",
        f"{dist_info}/METADATA": f"Metadata-Version: 2.1\nName: deep_gemm\nVersion: {version}\n",
        f"{dist_info}/WHEEL": (
            f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: {tag}\n"
        ),
    }
    record = []
    for name, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=")
        record.append(f"{name},sha256={digest.decode()},{len(text.encode())}")
    files[f"{dist_info}/RECORD"] = "\n".join([*record, f"{dist_info}/RECORD,,", ""])
    path = directory / f"deep_gemm-{version}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        for name, text in files.items():
            wheel.writestr(name, text)
    return path


@pytest.fixture
def real_pip(deepgemm, monkeypatch, tmp_path):
    """pip without the user's configuration; a local PEP 503 index of tiny wheels, its URL."""
    for name in [name for name in os.environ if name.startswith("PIP_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)
    files = tmp_path / "files"
    files.mkdir()
    wheels = [
        _make_wheel(files, f"2.8.0.post3+gaaaaaaa.{LABEL}"),
        _make_wheel(files, NEWEST),
        _make_wheel(files, f"2.8.0.post7+gddddddd.{LABEL}", tag="py3-none-win_amd64"),
    ]
    page = tmp_path / "simple" / "deep-gemm"
    page.mkdir(parents=True)
    links = "".join(f'<a href="{wheel.as_uri()}">{wheel.name}</a><br/>\n' for wheel in wheels)
    (page / "index.html").write_text(f"<!DOCTYPE html>\n<html><body>\n{links}</body></html>\n")
    return (tmp_path / "simple").as_uri()


def test_real_pip_installs_the_newest_wheel_for_this_machine(deepgemm, real_pip):
    folder, version = deepgemm.install_deepgemm(real_pip)
    assert version == NEWEST  # not post7, which is for Windows
    assert (folder / "deep_gemm/__init__.py").read_text() == f"__version__ = {NEWEST!r}\n"
    assert (folder / f"deep_gemm-{NEWEST}.dist-info/METADATA").is_file()
    # The import check ran in a fresh interpreter: an install that does not import fails it.
    (folder / "deep_gemm/__init__.py").write_text("raise ImportError('wrong torch')\n")
    with pytest.raises(RuntimeError, match=r"does not import \(ImportError: wrong torch\)"):
        deepgemm.install_deepgemm(real_pip)


def test_real_pip_installs_a_pinned_version_and_the_launcher_uses_it(deepgemm, real_pip, capsys):
    older = f"2.8.0.post3+gaaaaaaa.{LABEL}"
    deepgemm.resolve_deepgemm_args(
        argparse.Namespace(deepgemm_url=real_pip, deepgemm_version="2.8.0.post3")
    )
    folder = _wheels() / deepgemm._folder_name(older, real_pip)
    assert sys.path[0] == str(folder) and os.environ["PYTHONPATH"].startswith(str(folder))
    result = subprocess.run(
        [sys.executable, "-c", "import deep_gemm; print(deep_gemm.__version__)"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path.home().parent,
    )
    assert result.stdout == f"{older}\n"
    assert f"serving with deep_gemm {older} from {real_pip}: {folder}" in capsys.readouterr().out


def test_real_pip_ignores_an_installed_deep_gemm(deepgemm, real_pip, monkeypatch, tmp_path):
    """An installed deep_gemm, e.g. in site-packages, neither satisfies nor pins the selection."""
    installed = tmp_path / "site"
    with zipfile.ZipFile(_make_wheel(tmp_path, f"2.8.0.post3+gaaaaaaa.{LABEL}")) as wheel:
        wheel.extractall(installed)
    monkeypatch.setenv("PYTHONPATH", str(installed))
    assert deepgemm.install_deepgemm(real_pip)[1] == NEWEST
    folder, version = deepgemm.install_deepgemm(real_pip, "2.8.0.post3")  # the installed version
    assert version == f"2.8.0.post3+gaaaaaaa.{LABEL}"
    assert (folder / "deep_gemm/__init__.py").is_file()


def test_real_pip_installs_a_wheel_file(deepgemm, real_pip, tmp_path):
    wheel = _make_wheel(tmp_path, NEWEST)
    for url in (str(wheel), wheel.as_uri()):
        folder, version = deepgemm.install_deepgemm(url)
        assert version == NEWEST
        assert (folder / "deep_gemm/__init__.py").read_text() == f"__version__ = {NEWEST!r}\n"
