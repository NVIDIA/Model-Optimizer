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


import pytest
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command


# TODO: Add accuracy evaluation after we upgrade TRT version to 10.12
@pytest.mark.parametrize(
    ("quantize_mode", "onnx_save_path", "calib_size", "num_score_steps"),
    [
        ("fp8", "vit_base_patch16_224.fp8.onnx", "1", "1"),
        ("int8", "vit_base_patch16_224.int8.onnx", "1", "1"),
        ("nvfp4", "vit_base_patch16_224.nvfp4.onnx", "1", "1"),
        ("mxfp8", "vit_base_patch16_224.mxfp8.onnx", "1", "1"),
        ("int4_awq", "vit_base_patch16_224.int4_awq.onnx", "1", "1"),
        ("auto", "vit_base_patch16_224.auto.onnx", "1", "1"),
    ],
)
def test_torch_onnx(quantize_mode, onnx_save_path, calib_size, num_score_steps):
    cmd_parts = extend_cmd_parts(
        ["python", "torch_quant_to_onnx.py"],
        quantize_mode=quantize_mode,
        onnx_save_path=onnx_save_path,
        calibration_data_size=calib_size,
        num_score_steps=num_score_steps,
    )
    run_example_command(cmd_parts, "torch_onnx")
