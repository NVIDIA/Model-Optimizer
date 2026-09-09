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
import torch
import transformers
from _test_utils.torch import transformers_models
from safetensors import safe_open

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.export import export_hf_checkpoint

_VISION_RECIPE = "fp8_vision-kv_none"
_JOINT_RECIPE = "fp8_vision_lm-kv_fp8_cast"


def _get_tiny_qwen_vlm(model_type):
    if model_type == "qwen3_vl":
        return transformers_models.get_tiny_qwen3vl()
    if not all(hasattr(transformers, name) for name in ("Qwen3_5Config", "Qwen3_5MoeConfig")):
        pytest.skip("Qwen3.5-VL requires a newer Transformers release")
    return transformers_models.get_tiny_qwen3_5_vl_offline()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="FP8 vision encoder export requires compute capability 8.9 or newer",
)
@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [
        ("qwen3_vl", "Qwen3VLForConditionalGeneration"),
        ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
    ],
)
@pytest.mark.parametrize(
    ("recipe", "quantizes_language", "quantizes_kv"),
    [
        (_VISION_RECIPE, False, False),
        (_JOINT_RECIPE, True, True),
    ],
)
def test_qwen_vision_recipe_calibrates_and_exports(
    tmp_path, model_type, architecture, recipe, quantizes_language, quantizes_kv
):
    model = _get_tiny_qwen_vlm(model_type).to("cuda").eval()
    model.config.architectures = [architecture]
    quant_cfg = load_recipe(f"huggingface/{model_type}/ptq/{recipe}").quantize.model_dump()
    vision_config = model.config.vision_config
    pixel_width = (
        vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2
    )
    pixel_values = torch.randn(4, pixel_width, dtype=torch.bfloat16, device="cuda")
    image_grid_thw = torch.tensor([[1, 2, 2]], device="cuda")
    input_ids = torch.randint(5, 16, (1, 4), device="cuda")

    def calibration_forward(_model):
        model.model.visual(pixel_values, image_grid_thw)
        if quantizes_language:
            model(input_ids=input_ids)

    mtq.quantize(model, quant_cfg, forward_loop=calibration_forward)

    export_path = tmp_path / f"{model_type}-{recipe}"
    export_hf_checkpoint(model, export_dir=export_path)

    with (export_path / "hf_quant_config.json").open() as config_file:
        quantization = json.load(config_file)["quantization"]
    assert quantization["quant_algo"] == "FP8"
    assert (quantization["kv_cache_quant_algo"] == "FP8") is quantizes_kv
    assert ("model.language_model*" in quantization["exclude_modules"]) is not quantizes_language

    tensor_names = set()
    for checkpoint_path in export_path.glob("*.safetensors"):
        with safe_open(str(checkpoint_path), framework="pt") as checkpoint:
            tensor_names.update(checkpoint.keys())
    scale_names = {name for name in tensor_names if name.endswith(("input_scale", "weight_scale"))}
    assert scale_names
    assert any(name.startswith("model.visual.") for name in scale_names)
    assert any(".attn.qkv.weight_scale" in name for name in scale_names)
    assert any(".merger.linear_fc1.weight_scale" in name for name in scale_names)
    if model_type == "qwen3_vl":
        assert any(
            ".deepstack_merger_list.0.linear_fc1.weight_scale" in name for name in scale_names
        )
    assert not any("patch_embed" in name for name in scale_names)
    assert (
        any(name.startswith("model.language_model.") for name in scale_names) is quantizes_language
    )
