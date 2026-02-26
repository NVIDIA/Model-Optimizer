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

from pathlib import Path
from typing import NamedTuple

import pytest
from _test_utils.examples.models import FLUX_SCHNELL_PATH, SDXL_1_0_PATH
from _test_utils.examples.run_command import run_example_command
from _test_utils.torch.misc import minimum_sm


class DiffuserHfExportModel(NamedTuple):
    name: str
    path: str
    dtype: str
    format_type: str
    quant_algo: str
    collect_method: str

    def quantize_and_export_hf(self, tmp_path: Path) -> Path:
        hf_ckpt_dir = tmp_path / f"{self.name}_{self.format_type}_hf_ckpt"
        cmd_args = [
            "python",
            "quantize.py",
            "--model",
            self.name,
            "--override-model-path",
            self.path,
            "--calib-size",
            "8",
            "--batch-size",
            "2",
            "--n-steps",
            "20",
            "--percentile",
            "1.0",
            "--alpha",
            "0.8",
            "--format",
            self.format_type,
            "--quant-algo",
            self.quant_algo,
            "--collect-method",
            self.collect_method,
            "--trt-high-precision-dtype",
            self.dtype,
            "--hf-ckpt-dir",
            str(hf_ckpt_dir),
        ]
        run_example_command(cmd_args, "diffusers/quantization")
        return hf_ckpt_dir


@pytest.mark.parametrize(
    "model",
    [
        DiffuserHfExportModel(
            name="sdxl-1.0",
            path=SDXL_1_0_PATH,
            dtype="Half",
            format_type="int8",
            quant_algo="smoothquant",
            collect_method="min-mean",
        ),
        DiffuserHfExportModel(
            name="flux-schnell",
            path=FLUX_SCHNELL_PATH,
            dtype="BFloat16",
            format_type="int8",
            quant_algo="smoothquant",
            collect_method="min-mean",
        ),
        pytest.param(
            DiffuserHfExportModel(
                name="sdxl-1.0",
                path=SDXL_1_0_PATH,
                dtype="Half",
                format_type="fp8",
                quant_algo="max",
                collect_method="default",
            ),
            marks=minimum_sm(89),
        ),
        pytest.param(
            DiffuserHfExportModel(
                name="flux-schnell",
                path=FLUX_SCHNELL_PATH,
                dtype="BFloat16",
                format_type="fp4",
                quant_algo="max",
                collect_method="default",
            ),
            marks=minimum_sm(89),
        ),
    ],
    ids=[
        "sdxl_1.0_int8_smoothquant_min_mean",
        "flux_schnell_int8_smoothquant_min_mean",
        "sdxl_1.0_fp8_max_default",
        "flux_schnell_fp4_max_default",
    ],
)
def test_diffusers_hf_ckpt_export(model: DiffuserHfExportModel, tmp_path: Path) -> None:
    hf_ckpt_dir = model.quantize_and_export_hf(tmp_path)

    assert hf_ckpt_dir.exists(), f"HF checkpoint directory was not created: {hf_ckpt_dir}"

    config_files = list(hf_ckpt_dir.rglob("config.json"))
    assert len(config_files) > 0, f"No config.json found in {hf_ckpt_dir}"

    weight_files = list(hf_ckpt_dir.rglob("*.safetensors")) + list(hf_ckpt_dir.rglob("*.bin"))
    assert len(weight_files) > 0, f"No weight files (.safetensors or .bin) found in {hf_ckpt_dir}"
