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
from typing import NamedTuple

import pytest
from _test_utils.examples.models import FLUX_SCHNELL_PATH, SD3_PATH, SDXL_PATH
from _test_utils.examples.run_command import run_example_command
from _test_utils.torch.misc import minimum_sm


class DiffuserModel(NamedTuple):
    dtype: str
    name: str
    path: str
    format_type: str
    quant_algo: str
    collect_method: str

    def _run_cmd(self, script: str, *args: str) -> None:
        cmd_args = [
            "python",
            script,
            "--model",
            self.name,
            "--override-model-path",
            self.path,
        ]
        cmd_args.extend(args)
        run_example_command(cmd_args, "diffusers/quantization")

    def _format_args(self) -> list[str]:
        return [
            "--calib-size",
            "4",
            "--percentile",
            "1.0",
            "--alpha",
            "0.8",
            "--n-steps",
            "2",
            "--batch-size",
            "2",
            "--format",
            self.format_type,
            "--collect-method",
            self.collect_method,
            "--quant-algo",
            self.quant_algo,
        ]

    def quantize(self, tmp_path: Path) -> None:
        self._run_cmd(
            "quantize.py",
            *self._format_args(),
            "--trt-high-precision-dtype",
            self.dtype,
            "--quantized-torch-ckpt-save-path",
            str(tmp_path / f"{self.name}_{self.format_type}.pt"),
            "--onnx-dir",
            str(tmp_path / f"{self.name}_{self.format_type}_onnx"),
        )

    def restore(self, tmp_path: Path) -> None:
        self._run_cmd(
            "quantize.py",
            *self._format_args(),
            "--trt-high-precision-dtype",
            self.dtype,
            "--restore-from",
            str(tmp_path / f"{self.name}_{self.format_type}.pt"),
            "--onnx-dir",
            str(tmp_path / f"{self.name}_{self.format_type}_onnx"),
        )

    def inference(self, tmp_path: Path) -> None:
        self._run_cmd(
            "diffusion_trt.py",
            "--onnx-load-path",
            str(tmp_path / f"{self.name}_{self.format_type}_onnx/model.onnx"),
            "--dq-only",
            "--torch-autocast",
            "--num-inference-steps",
            "2",
        )


@pytest.mark.parametrize(
    "model",
    [
        DiffuserModel(
            name="flux-schnell",
            path=FLUX_SCHNELL_PATH,
            dtype="BFloat16",
            format_type="int8",
            quant_algo="smoothquant",
            collect_method="min-mean",
        ),
        DiffuserModel(
            name="sd3-medium",
            path=SD3_PATH,
            dtype="Half",
            format_type="int8",
            quant_algo="smoothquant",
            collect_method="min-mean",
        ),
        pytest.param(
            DiffuserModel(
                name="sd3-medium",
                path=SD3_PATH,
                dtype="Half",
                format_type="fp8",
                quant_algo="max",
                collect_method="default",
            ),
            marks=minimum_sm(89),
        ),
        pytest.param(
            DiffuserModel(
                name="sdxl-1.0",
                path=SDXL_PATH,
                dtype="Half",
                format_type="fp8",
                quant_algo="max",
                collect_method="default",
            ),
            marks=minimum_sm(89),
        ),
        pytest.param(
            DiffuserModel(
                name="sdxl-1.0",
                path=SDXL_PATH,
                dtype="Half",
                format_type="fp4",
                quant_algo="max",
                collect_method="default",
            ),
            marks=minimum_sm(100),
        ),
        DiffuserModel(
            name="sdxl-1.0",
            path=SDXL_PATH,
            dtype="Half",
            format_type="int8",
            quant_algo="smoothquant",
            collect_method="min-mean",
        ),
    ],
    ids=[
        "flux_schnell_bf16_int8_smoothquant_3.0_min_mean",
        "sd3_medium_fp16_int8_smoothquant_3.0_min_mean",
        "sd3_medium_fp16_fp8_max_3.0_default",
        "sdxl_1.0_fp16_fp8_max_3.0_default",
        "sdxl_1.0_fp16_fp4_max_3.0_default",
        "sdxl_1.0_fp16_int8_smoothquant_3.0_min_mean",
    ],
)
def test_diffusers_quantization(
    model: DiffuserModel,
    tmp_path: Path,
) -> None:
    model.quantize(tmp_path)
    model.restore(tmp_path)
    model.inference(tmp_path)
