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

import json
import os
import sys
from pathlib import Path
from tempfile import gettempdir
from types import ModuleType
from unittest import mock

import pytest

# Patch mock tensorrt module
tensorrt = ModuleType("tensorrt")
tensorrt.Logger = mock.Mock()
tensorrt.tensorrt = mock.Mock()
tensorrt.__version__ = "10.10"
sys.modules["tensorrt"] = tensorrt

from modelopt.torch._deploy._runtime.tensorrt.engine_builder import build_engine, profile_engine


@pytest.fixture
def setup_mocks():
    tmp_path = Path(gettempdir()) / "modelopt_build/trt_artifacts"
    engine_bytes = b"engine_bytes"
    dummy_hash = "997930f2"
    expected_engine_bytes = b"\x99y0\xf2" + b"A" * 28 + b",engine_bytes"

    os.makedirs(tmp_path / "onnx", exist_ok=True)
    (tmp_path / "model.engine").write_bytes(engine_bytes)

    (tmp_path / f"{dummy_hash}-profile.json").write_text(
        json.dumps([{"count": 1}, {"name": "dummy_layer", "averageMs": 0.001}])
    )
    (tmp_path / f"{dummy_hash}-layerInfo.json").write_text(
        json.dumps({"Layers": [{"Name": "dummy_layer"}]})
    )

    mock_onnx = mock.Mock()
    mock_onnx.model_name = "model"

    with (
        mock.patch(
            "modelopt.torch._deploy._runtime.tensorrt.engine_builder._run_command"
        ) as mock_run,
        mock.patch(
            "modelopt.torch._deploy._runtime.tensorrt.engine_builder.TemporaryDirectory"
        ) as mock_tmp,
        mock.patch(
            "modelopt.torch._deploy._runtime.tensorrt.engine_builder.prepend_hash_to_bytes"
        ) as mock_hash,
    ):
        mock_run.return_value = (0, "success")
        mock_tmp.return_value.__enter__.return_value = str(tmp_path)
        mock_hash.return_value = expected_engine_bytes
        yield tmp_path, mock_run, mock_onnx, dummy_hash


def _assert_engine_saved(tmp_path, dummy_hash, engine_bytes, out):
    assert not (tmp_path / "model.engine").exists()
    assert (tmp_path / f"{dummy_hash}-model.engine").exists()
    assert engine_bytes is not None and engine_bytes.startswith(b"\x99y0\xf2")
    assert out == "success"


def test_build_engine(setup_mocks):
    tmp_path, mock_run, mock_onnx, dummy_hash = setup_mocks
    engine_bytes, out = build_engine(onnx_bytes=mock_onnx, verbose=True, output_dir=tmp_path)

    mock_run.assert_called_once()
    _assert_engine_saved(tmp_path, dummy_hash, engine_bytes, out)


def test_build_engine_dynamic_shapes(setup_mocks):
    tmp_path, mock_run, mock_onnx, dummy_hash = setup_mocks
    shapes = {
        "minShapes": {"input": [1, 3, 244, 244]},
        "optShapes": {"input": [16, 3, 244, 244]},
        "maxShapes": {"input": [32, 3, 244, 244]},
    }
    engine_bytes, out = build_engine(
        onnx_bytes=mock_onnx, dynamic_shapes=shapes, output_dir=tmp_path
    )

    mock_run.assert_called_once()
    _assert_engine_saved(tmp_path, dummy_hash, engine_bytes, out)


@pytest.mark.parametrize("enable_layerwise", [False, True])
@pytest.mark.parametrize("profiling_runs", [1, 2])
def test_profile_engine_variants(setup_mocks, enable_layerwise, profiling_runs):
    tmp_path, mock_run, _, dummy_hash = setup_mocks
    engine_bytes = b"\x99y0\xf2" + b"A" * 28 + b",engine_bytes"

    kwargs = {
        "engine_bytes": engine_bytes,
        "onnx_node_names": [],
        "output_dir": tmp_path,
        "profiling_runs": profiling_runs,
        "enable_layerwise_profiling": enable_layerwise,
    }

    profiled_data, out = profile_engine(**kwargs)
    mock_run.assert_called_once()

    if enable_layerwise:
        assert profiled_data == {"dummy_layer": 0.001}
        assert (tmp_path / f"{dummy_hash}-profile.json").exists()
        assert (tmp_path / f"{dummy_hash}-layerInfo.json").exists()
    else:
        assert profiled_data == {}

    assert out == "success"
