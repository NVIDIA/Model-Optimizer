# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for Modelopt's serialization utilities."""

from io import BytesIO
from pickle import UnpicklingError

import pytest
import torch

from modelopt.torch.opt.config import ModeloptBaseConfig
from modelopt.torch.utils import safe_load


class MockConfig(ModeloptBaseConfig):
    """A mock configuration class for testing serialization."""

    name: str = "mock"


def test_safe_load_with_modelopt_config():
    """Verify that safe_load can handle ModeloptBaseConfig subclasses with weights_only=True."""
    config = MockConfig(name="test_serialization")
    state = {"config": config}

    buffer = BytesIO()
    torch.save(state, buffer)
    data = buffer.getvalue()

    # safe_load defaults to weights_only=True
    loaded_state = safe_load(data)

    assert isinstance(loaded_state["config"], MockConfig)
    assert loaded_state["config"].name == "test_serialization"


def test_safe_load_basic_types():
    """Verify that safe_load can handle basic types (standard torch.load functionality)."""
    state = {"t": torch.ones(2), "v": [1, 2, 3], "d": {"a": 1}}

    buffer = BytesIO()
    torch.save(state, buffer)
    data = buffer.getvalue()

    loaded_state = safe_load(data)

    assert torch.allclose(loaded_state["t"], torch.ones(2))
    assert loaded_state["v"] == [1, 2, 3]
    assert loaded_state["d"]["a"] == 1


def test_safe_load_with_path(tmp_path):
    """Verify that safe_load can handle file paths."""
    state = {"data": 42}
    file_path = tmp_path / "test.pt"

    torch.save(state, file_path)

    loaded_state = safe_load(file_path)

    assert loaded_state["data"] == 42


class _UnsafeObj:
    """Not registered in torch safe globals — unpickling fails with weights_only=True."""

    def __init__(self, v):
        self.v = v


def test_safe_load_env_var_bypasses_weights_only(tmp_path, monkeypatch):
    """Verify TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 allows safe_load to load objects unsafe for weights_only."""
    file_path = tmp_path / "unsafe.pt"
    torch.save({"obj": _UnsafeObj(42)}, file_path)

    # Always fails when weights_only is not set (default=True)
    with pytest.raises(UnpicklingError):
        safe_load(file_path)

    # With the env var, safe_load (no explicit weights_only) defers to torch's default=False
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    loaded = safe_load(file_path)
    assert loaded["obj"].v == 42
