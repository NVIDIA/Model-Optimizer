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

"""Exercise the actual biased rank-three NVFP4 exporter and TensorRT parser."""

import importlib.util
from pathlib import Path

import onnx
import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata
from modelopt.torch._deploy.utils.torch_onnx import _prepare_nvfp4_fp16_export


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("route", ["api", "example"])
def test_nvfp4_fp16_biased_linear_export(tmp_path, route):
    """Exercise the exported dtype contract without bypassing validation."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 requires Blackwell")
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=True)).cuda().eval()
    x = torch.randn(1, 2, 16, device="cuda")
    with torch.no_grad():
        mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=lambda m: m(x))
        if route == "api":
            _prepare_nvfp4_fp16_export(model)
            data, _ = get_onnx_bytes_and_metadata(
                model, (x.half(),), weights_dtype="fp16", model_name="nvfp4_bias_B"
            )
            OnnxBytes.from_bytes(data).write_to_disk(str(tmp_path), clean_dir=False)
        else:
            example_path = Path(__file__).parents[4] / "examples/onnx_ptq/download_example_onnx.py"
            spec = importlib.util.spec_from_file_location("nvfp4_example", example_path)
            example = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(example)
            example.export_to_onnx(
                model, x.shape, str(tmp_path / "nvfp4_bias_B.onnx"), "cuda", weights_dtype="fp16"
            )
        assert torch.isfinite(model(x.half())).all()
    graph = onnx.load(tmp_path / "nvfp4_bias_B.onnx")
    onnx.checker.check_model(graph, full_check=True)
    onnx.shape_inference.infer_shapes(graph, strict_mode=True, check_type=True)
    assert any(node.op_type == "Add" for node in graph.graph.node)
