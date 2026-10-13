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

import importlib
import sys
import warnings
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples/diffusers/quantization"
_LOCAL_MODULES = (
    "calib",
    "calib.plugin_calib",
    "calibration",
    "config",
    "models_utils",
    "pipeline_manager",
    "quantize_config",
    "utils",
    "quantize",
    "diffusion_trt",
    "onnx_utils",
    "onnx_utils.export",
)
_WARNING = "The Diffusers ONNX/TensorRT workflow is deprecated"


@pytest.fixture(scope="module")
def cli_modules():
    # The example uses bare sibling imports; isolate them from other example tests.
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(_EXAMPLE_DIR))
        for name in _LOCAL_MODULES:
            patch.delitem(sys.modules, name, raising=False)
        try:
            yield {name: importlib.import_module(name) for name in ("quantize", "diffusion_trt")}
        finally:
            for name in _LOCAL_MODULES:
                sys.modules.pop(name, None)


class _StopBeforeModelSetupError(Exception):
    pass


@pytest.mark.parametrize(
    ("script", "args", "warns"),
    [
        pytest.param("quantize", ["--onnx-dir", "onnx"], True, id="quantize-onnx"),
        pytest.param(
            "quantize",
            ["--restore-from", "checkpoint.pt", "--onnx-dir", "onnx"],
            True,
            id="restore-onnx",
        ),
        pytest.param("quantize", ["--hf-ckpt-dir", "hf"], False, id="hf-only"),
        pytest.param("quantize", [], False, id="quantize-only"),
        pytest.param("diffusion_trt", [], True, id="default-trt"),
        pytest.param("diffusion_trt", ["--onnx-load-path", "model.onnx"], True, id="load-onnx"),
        pytest.param(
            "diffusion_trt", ["--trt-engine-load-path", "model.plan"], True, id="load-engine"
        ),
        pytest.param("diffusion_trt", ["--torch"], False, id="torch"),
        pytest.param("diffusion_trt", ["--torch", "--torch-compile"], False, id="torch-compile"),
    ],
)
def test_onnx_trt_deprecation(cli_modules, monkeypatch, script, args, warns):
    module = cli_modules[script]
    monkeypatch.setattr(sys, "argv", [script, "--model", "sdxl-1.0", *args])
    # quantize.main replaces both RMSNorm aliases; restore them after each case.
    monkeypatch.setattr(torch.nn, "RMSNorm", torch.nn.RMSNorm)
    monkeypatch.setattr(
        torch.nn.modules.normalization, "RMSNorm", torch.nn.modules.normalization.RMSNorm
    )
    stop = Mock(side_effect=_StopBeforeModelSetupError)
    if script == "quantize":
        monkeypatch.setattr(module, "setup_logging", stop)
    else:
        monkeypatch.setattr(module.PipelineManager, "create_pipeline_from", stop)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(_StopBeforeModelSetupError):
            module.main()

    stop.assert_called_once()
    deprecations = [warning for warning in captured if _WARNING in str(warning.message)]
    assert len(deprecations) == int(warns)
    if warns:
        warning = deprecations[0]
        assert warning.category is FutureWarning
        assert "0.48.0" in str(warning.message)
        assert "0.49.0" in str(warning.message)
        assert "--hf-ckpt-dir" in str(warning.message)
