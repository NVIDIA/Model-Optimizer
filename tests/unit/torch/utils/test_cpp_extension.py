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

import os
import warnings

import pytest
import torch
import torch.utils.cpp_extension as torch_cpp_extension

from modelopt.torch.utils import cpp_extension


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")


def test_failed_build_is_retried_not_skipped(monkeypatch, tmp_path, fake_cuda):
    """A retry in the same process must rebuild, not import an unchecked library from disk."""
    monkeypatch.setattr(cpp_extension, "_IS_WINDOWS", False)
    builds = []

    def failing_build(**kwargs):
        builds.append(kwargs["name"])
        raise RuntimeError("Command '['where', 'cl']' returned non-zero exit status 1.")

    monkeypatch.setattr(torch_cpp_extension, "_write_ninja_file_and_build_library", failing_build)
    source = tmp_path / "ext.cpp"
    source.write_text("int f() { return 0; }\n")
    name = "modelopt_test_failed_build_ext"

    for _ in range(2):
        with pytest.raises(RuntimeError, match="where"):
            cpp_extension.load_cpp_extension(
                name, [source], None, raise_if_failed=True, build_directory=str(tmp_path)
            )
    assert builds == [name, name]
    assert name not in torch_cpp_extension.JIT_EXTENSION_VERSIONER.entries


@pytest.fixture
def windows_without_cl(monkeypatch, fake_cuda):
    monkeypatch.setattr(cpp_extension, "_IS_WINDOWS", True)
    monkeypatch.setattr(cpp_extension.shutil, "which", lambda cmd: None)
    loads = []

    def fake_load(name, sources, **kwargs):
        loads.append((os.environ.get("MODELOPT_TEST_VC"), os.environ["PATH"]))
        return name

    monkeypatch.setattr(cpp_extension, "load", fake_load)
    return loads


def test_missing_cl_activates_msvc_and_warns_once(monkeypatch, windows_without_cl):
    monkeypatch.setattr(
        cpp_extension, "_get_msvc_env", lambda: {"MODELOPT_TEST_VC": "1", "PATH": "vc_bin"}
    )
    monkeypatch.delenv("MODELOPT_TEST_VC", raising=False)
    cpp_extension._warn_msvc_env_activated.cache_clear()
    path = os.environ["PATH"]

    with pytest.warns(UserWarning, match=r"cl\.exe"):
        assert cpp_extension.load_cpp_extension("ext", [], None) == "ext"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cpp_extension.load_cpp_extension("ext", [], None) == "ext"

    assert windows_without_cl == [("1", "vc_bin")] * 2
    assert "MODELOPT_TEST_VC" not in os.environ
    assert os.environ["PATH"] == path


def test_missing_cl_and_msvc_fails_without_building(monkeypatch, windows_without_cl):
    monkeypatch.setattr(cpp_extension, "_get_msvc_env", dict)

    with pytest.raises(RuntimeError, match=r"cl\.exe"):
        cpp_extension.load_cpp_extension("ext", [], None, raise_if_failed=True)
    with pytest.warns(UserWarning, match=r"cl\.exe"):
        assert cpp_extension.load_cpp_extension("ext", [], None) is None
    assert windows_without_cl == []
