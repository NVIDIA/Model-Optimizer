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


import pytest
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command


@pytest.mark.parametrize(
    ("hf_model_path", "dtype", "lm_head"),
    [
        ("Qwen/Qwen2-0.5B-Instruct", "fp16", "fp16"),
        ("Qwen/Qwen2-0.5B-Instruct", "fp8", "fp16"),
        ("Qwen/Qwen3-0.6B", "int4_awq", "fp16"),
        ("Qwen/Qwen3-0.6B", "nvfp4", "fp16"),
    ],
)
def test_llm_export_onnx(tmp_path, hf_model_path, dtype, lm_head):
    cmd_parts = extend_cmd_parts(
        ["python", "llm_export.py"],
        hf_model_path=hf_model_path,
        dtype=dtype,
        lm_head=lm_head,
        output_dir=str(tmp_path),
        calib_size=1,
    )
    run_example_command(cmd_parts, "torch_onnx")
