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

import pytest
import transformers
from _test_utils.torch import transformers_models

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe

_VISION_RECIPE = "fp8_vision-kv_none"
_JOINT_RECIPE = "fp8_vision_lm-kv_fp8_cast"


def _get_tiny_qwen_vlm(model_type):
    if model_type == "qwen3_vl":
        return transformers_models.get_tiny_qwen3vl()
    if not all(hasattr(transformers, name) for name in ("Qwen3_5Config", "Qwen3_5MoeConfig")):
        pytest.skip("Qwen3.5-VL requires a newer Transformers release")
    return transformers_models.get_tiny_qwen3_5_vl_offline()


@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_5"])
@pytest.mark.parametrize(
    ("recipe", "quantizes_language", "quantizes_kv"),
    [
        (_VISION_RECIPE, False, False),
        (_JOINT_RECIPE, True, True),
    ],
)
def test_qwen_vision_recipes_select_expected_quantizers(
    model_type, recipe, quantizes_language, quantizes_kv
):
    model = _get_tiny_qwen_vlm(model_type)
    assert model.config.model_type == model_type
    quant_cfg = load_recipe(f"model_type/{model_type}/ptq/{recipe}").quantize.model_dump()

    mtq.quantize(model, quant_cfg, forward_loop=None)
    modules = dict(model.named_modules())
    enabled = {name for name, module in modules.items() if getattr(module, "is_enabled", False)}

    expected_visual_suffixes = {
        "blocks.0.attn.qkv.weight_quantizer",
        "blocks.0.attn.qkv.input_quantizer",
        "blocks.0.attn.proj.weight_quantizer",
        "blocks.0.attn.proj.input_quantizer",
        "blocks.0.mlp.linear_fc1.weight_quantizer",
        "blocks.0.mlp.linear_fc1.input_quantizer",
        "blocks.0.mlp.linear_fc2.weight_quantizer",
        "blocks.0.mlp.linear_fc2.input_quantizer",
        "merger.linear_fc1.weight_quantizer",
        "merger.linear_fc1.input_quantizer",
        "merger.linear_fc2.weight_quantizer",
        "merger.linear_fc2.input_quantizer",
    }
    if model_type == "qwen3_vl":
        expected_visual_suffixes.update(
            {
                "deepstack_merger_list.0.linear_fc1.weight_quantizer",
                "deepstack_merger_list.0.linear_fc1.input_quantizer",
                "deepstack_merger_list.0.linear_fc2.weight_quantizer",
                "deepstack_merger_list.0.linear_fc2.input_quantizer",
            }
        )
    assert {f"model.visual.{suffix}" for suffix in expected_visual_suffixes} <= enabled
    assert not any("patch_embed" in name for name in enabled)

    vision_weight = modules["model.visual.blocks.0.attn.qkv.weight_quantizer"]
    vision_input = modules["model.visual.blocks.0.attn.qkv.input_quantizer"]
    assert vision_weight.num_bits == (4, 3)
    assert vision_input.num_bits == (4, 3)

    language_quantizers = {name for name in enabled if name.startswith("model.language_model.")}
    assert bool(language_quantizers) is quantizes_language
    if not quantizes_language:
        assert all(name.startswith("model.visual.") for name in enabled)

    enabled_bmm = {name for name in enabled if name.endswith("_bmm_quantizer")}
    assert bool(enabled_bmm) is quantizes_kv
    assert all(name.startswith("model.language_model.") for name in enabled_bmm)
