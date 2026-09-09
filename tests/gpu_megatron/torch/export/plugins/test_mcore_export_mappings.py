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

"""Guards on the Megatron-Core to Hugging Face export mappings."""

import pytest

from modelopt.torch.export.plugins.mcore_common import all_mcore_hf_export_mapping

# ``_GPTModelExporter`` only emits ``k_scale`` / ``v_scale`` and sets ``kv_cache_quant_algo``
# for layers whose architecture mapping defines ``core_attention``. A missing entry is silent:
# calibration still runs and the Megatron checkpoint keeps its KV quantizers, but the exported
# HuggingFace checkpoint serves an unquantized KV cache.
KV_SCALE_EXPORT_PREFIXES = {
    "LlamaForCausalLM": "model.layers.{}.self_attn.",
    "NemotronForCausalLM": "backbone.layers.{}.mixer.",
    "NemotronHForCausalLM": "backbone.layers.{}.mixer.",
    "Qwen2ForCausalLM": "model.layers.{}.self_attn.",
    "Qwen3ForCausalLM": "model.layers.{}.self_attn.",
    "Qwen3MoeForCausalLM": "model.layers.{}.self_attn.",
    # VLMs derive from the text mapping via ``with_language_model_prefix``.
    "Qwen3VLForConditionalGeneration": "model.language_model.layers.{}.self_attn.",
    "Qwen3_5ForConditionalGeneration": "model.language_model.layers.{}.self_attn.",
    "Qwen3_5MoeForConditionalGeneration": "model.language_model.layers.{}.self_attn.",
}


@pytest.mark.parametrize(("arch", "prefix"), sorted(KV_SCALE_EXPORT_PREFIXES.items()))
def test_export_mapping_emits_kv_cache_scales(arch, prefix):
    mapping = all_mcore_hf_export_mapping[arch]
    assert "core_attention" in mapping, (
        f"{arch} exports a quantized KV cache without k_scale / v_scale"
    )
    rule = mapping["core_attention"]
    assert rule.func_name == "self_attention_scaling"
    assert rule.target_name_or_prefix == prefix
