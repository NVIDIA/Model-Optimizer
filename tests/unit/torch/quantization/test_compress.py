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

"""Regression tests for optional compression dependencies."""

import builtins
import importlib.util
import warnings
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("error_type", [None, ModuleNotFoundError, AttributeError])
def test_compress_optional_megatron_import(monkeypatch, error_type):
    # Inject a dependency failure without requiring a Megatron/TE installation.
    original_import = builtins.__import__
    plugin = SimpleNamespace(
        **{
            name: type(name, (), {})
            for name in (
                "_MegatronColumnParallelLinear",
                "_MegatronRowParallelLinear",
                "_RealQuantMegatronColumnParallelLinear",
                "_RealQuantMegatronRowParallelLinear",
            )
        }
    )

    def import_with_optional_dependency(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "plugins.megatron" and level == 1:
            if error_type is not None:
                raise error_type("optional Megatron dependency failed")
            return plugin
        return original_import(name, globals, locals, fromlist, level)

    # Load a separate module instance so the process-wide compression registry is unchanged.
    original_spec = importlib.util.find_spec("modelopt.torch.quantization.compress")
    spec = importlib.util.spec_from_file_location(
        "modelopt.torch.quantization._compress_import_test", original_spec.origin
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(builtins, "__import__", import_with_optional_dependency)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec.loader.exec_module(module)

    assert module.mcore_available is (error_type is None)
    assert callable(module.compress)
    if error_type is None:
        assert module._MegatronColumnParallelLinear is plugin._MegatronColumnParallelLinear
    elif error_type is AttributeError:
        assert any("optional Megatron dependency failed" in str(w.message) for w in caught)
