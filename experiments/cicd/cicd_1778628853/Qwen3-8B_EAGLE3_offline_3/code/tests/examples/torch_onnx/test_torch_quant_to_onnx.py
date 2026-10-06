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

# TODO: Add int4_awq once the INT4 exporter supports non-MatMul/Gemm consumer patterns
# (e.g., DQ -> Reshape -> Slice in small ViT / SwinTransformer ONNX graphs).
_QUANT_MODES = ["fp8", "int8", "mxfp8", "nvfp4", "auto"]

_MODELS = {
    "vit_tiny": ("vit_tiny_patch16_224", '{"depth": 1}'),
    "swin_tiny": ("swin_tiny_patch4_window7_224", '{"depths": [1, 1, 1, 1]}'),
    "swinv2_tiny": ("swinv2_tiny_window8_256", '{"depths": [1, 1, 1, 1]}'),
    "resnet50": ("resnet50", None),
}


@pytest.mark.parametrize("quantize_mode", _QUANT_MODES)
@pytest.mark.parametrize("model_key", list(_MODELS))
def test_torch_onnx(model_key, quantize_mode):
    timm_model_name, model_kwargs = _MODELS[model_key]
    onnx_save_path = f"{model_key}.{quantize_mode}.onnx"

    cmd_parts = extend_cmd_parts(
        ["python", "torch_quant_to_onnx.py"],
        timm_model_name=timm_model_name,
        model_kwargs=model_kwargs,
        quantize_mode=quantize_mode,
        onnx_save_path=onnx_save_path,
        calibration_data_size="1",
        num_score_steps="1",
    )
    cmd_parts.extend(["--no_pretrained", "--trt_build"])
    run_example_command(cmd_parts, "torch_onnx")
