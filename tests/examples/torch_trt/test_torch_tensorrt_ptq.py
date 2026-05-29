# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command

# Recipe variants the example ships. Mirrors the parametrization style of
# ``tests/examples/torch_onnx/test_torch_quant_to_onnx.py``.
_PRECISIONS = ["fp8", "nvfp4"]

# Tiny ViT config (~1 encoder block) so the test stays under a few seconds
# of GPU time while exercising every code path the recipe touches: encoder
# Linear weight/input quantizers, attention BMM + softmax quantizers,
# per-block LayerNorm output quantizer, and the patch-embed Conv / final
# vit.layernorm / classifier skip rules.
_TINY_VIT_KWARGS = {
    "num_hidden_layers": 1,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 2,
}


@pytest.mark.parametrize("precision", _PRECISIONS)
def test_torch_tensorrt_ptq(precision):
    """End-to-end: load tiny ViT -> mtq.quantize via recipe -> torch_tensorrt.compile.

    Runs against the smallest viable ``ViTForImageClassification`` config so
    the test stays fast; ``--no_pretrained`` skips the multi-GB pretrained
    download. The example's CLI exits non-zero if any step (calibration,
    quantization, TRT compile) fails or if the compiled-model argmax doesn't
    match the fake-quant argmax on the sample input.
    """
    pytest.importorskip("torch_tensorrt")

    cmd_parts = extend_cmd_parts(
        ["python", "torch_tensorrt_ptq.py"],
        model_id="google/vit-base-patch16-224",
        precision=precision,
        calib_samples="4",
        batch_size="1",
        model_kwargs=json.dumps(_TINY_VIT_KWARGS),
    )
    cmd_parts.append("--no_pretrained")
    run_example_command(cmd_parts, "torch_trt")
