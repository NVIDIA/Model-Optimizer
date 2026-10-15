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

from pathlib import Path

import pytest
from _test_utils.examples.run_command import run_example_command
from test_diffusers import DIFFUSER_MODELS, DiffuserModel


@pytest.mark.parametrize("model", DIFFUSER_MODELS)
def test_diffusers_onnx_trt(model: DiffuserModel, tmp_path: Path) -> None:
    onnx_dir = tmp_path / f"{model.name}_{model.format_type}_onnx"
    model.quantize(tmp_path, "--onnx-dir", str(onnx_dir))
    model.restore(tmp_path, "--onnx-dir", str(onnx_dir))
    run_example_command(
        [
            "python",
            "diffusion_trt.py",
            "--model",
            model.name,
            "--override-model-path",
            model.path,
            "--onnx-load-path",
            str(onnx_dir / "model.onnx"),
            "--dq-only",
            "--torch-autocast",
            "--num-inference-steps",
            "2",
        ],
        "diffusers/quantization",
    )
