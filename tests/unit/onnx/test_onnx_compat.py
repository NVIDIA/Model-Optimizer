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

"""Tests for the ONNX compatibility shim (_onnx_compat)."""

import importlib
from unittest import mock

import ml_dtypes
import numpy as np
import onnx.helper
import pytest

PATCHED_ATTRS = ["float32_to_bfloat16", "float32_to_float8e4m3"]


def _reload_compat():
    from modelopt.onnx import _onnx_compat

    importlib.reload(_onnx_compat)


@pytest.fixture
def _clean_helper():
    """Save, remove, and restore patched attrs around each test."""
    saved = {attr: getattr(onnx.helper, attr, None) for attr in PATCHED_ATTRS}
    for attr in PATCHED_ATTRS:
        if hasattr(onnx.helper, attr):
            delattr(onnx.helper, attr)
    yield
    for attr in PATCHED_ATTRS:
        if saved[attr] is not None:
            setattr(onnx.helper, attr, saved[attr])
        elif hasattr(onnx.helper, attr):
            delattr(onnx.helper, attr)


@pytest.mark.usefixtures("_clean_helper")
class TestPatchOnnxHelperRemovedApis:
    @pytest.mark.parametrize("attr", PATCHED_ATTRS)
    def test_patched_when_missing(self, attr):
        _reload_compat()
        assert callable(getattr(onnx.helper, attr))

    @pytest.mark.parametrize("attr", PATCHED_ATTRS)
    def test_existing_not_overwritten(self, attr):
        sentinel = object()
        setattr(onnx.helper, attr, sentinel)
        _reload_compat()
        assert getattr(onnx.helper, attr) is sentinel

    def test_no_error_when_ml_dtypes_unavailable(self):
        real_import = __import__

        def _block_ml_dtypes(name, *args, **kwargs):
            if name == "ml_dtypes":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=_block_ml_dtypes):
            _reload_compat()


@pytest.mark.parametrize(
    ("value", "kwargs", "expected"),
    [
        (0.0, {}, 0),
        (
            1.0,
            {},
            int.from_bytes(np.float32(1.0).astype(ml_dtypes.bfloat16).tobytes(), "little"),
        ),
        (
            1.0,
            {"truncate": True},
            int.from_bytes(np.float32(1.0).astype(ml_dtypes.bfloat16).tobytes(), "little"),
        ),
    ],
)
def test_bfloat16_conversion(value, kwargs, expected):
    from modelopt.onnx import _onnx_compat  # noqa: F401

    if not hasattr(onnx.helper, "float32_to_bfloat16"):
        pytest.skip("ml_dtypes missing")
    assert onnx.helper.float32_to_bfloat16(value, **kwargs) == expected


@pytest.mark.parametrize(
    ("value", "kwargs", "expected"),
    [
        (0.0, {}, 0),
        (
            1.0,
            {},
            int(np.float32(1.0).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)),
        ),
        (
            0.5,
            {"fn": True, "uz": True},
            int(np.float32(0.5).astype(ml_dtypes.float8_e4m3fnuz).view(np.uint8)),
        ),
        (
            1.0,
            {"scale": 2.0, "saturate": False},
            int(np.float32(1.0).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)),
        ),
        (
            1.0,
            {"fn": False},
            int(np.float32(1.0).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)),
        ),
    ],
)
def test_float8e4m3_conversion(value, kwargs, expected):
    from modelopt.onnx import _onnx_compat  # noqa: F401

    if not hasattr(onnx.helper, "float32_to_float8e4m3"):
        pytest.skip("ml_dtypes missing")
    assert onnx.helper.float32_to_float8e4m3(value, **kwargs) == expected
