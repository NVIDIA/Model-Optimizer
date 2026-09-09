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


# Routed experts must export one entry per expert: vLLM's quantized MoE loader has no parameter
# for a packed `experts.down_proj_weight_scale_2`, so a packed layout cannot be served.
PER_EXPERT_MOE_ARCHS = {
    "Qwen3_5MoeForConditionalGeneration": "model.language_model.layers.{}.mlp.experts.{}",
    "Qwen3MoeForCausalLM": "model.layers.{}.mlp.experts.{}",
    "NemotronHForCausalLM": "backbone.layers.{}.mixer.experts.{}",
}


@pytest.mark.parametrize("arch", sorted(PER_EXPERT_MOE_ARCHS))
def test_moe_export_is_per_expert(arch):
    mapping = all_mcore_hf_export_mapping[arch]
    assert not mapping.get("use_packed_local_experts"), (
        f"{arch} packs routed experts; vLLM cannot load packed quantized expert scales"
    )
    for rule in ("experts.linear_fc1", "experts.linear_fc2", "local_experts.linear_fc1"):
        if rule in mapping:
            assert "pack_name_remapping" not in mapping[rule].func_name


def test_qwen3_5_moe_expert_names_match_released_checkpoint():
    """Layer 7 / expert 3 must match the released checkpoint's tensor names."""
    mapping = all_mcore_hf_export_mapping["Qwen3_5MoeForConditionalGeneration"]

    fc1 = mapping["experts.linear_fc1"]
    assert fc1.func_kwargs["gate_proj_name"] == "gate_proj"
    assert fc1.func_kwargs["up_proj_name"] == "up_proj"
    # _grouped_mlp_slicing appends the trailing "." itself.
    expert = fc1.target_name_or_prefix.format(7).format(3) + "."
    assert expert == "model.language_model.layers.7.mlp.experts.3."
    assert expert + "gate_proj." == "model.language_model.layers.7.mlp.experts.3.gate_proj."

    fc2 = mapping["experts.linear_fc2"].target_name_or_prefix.format(7).format(3) + "."
    assert fc2 == "model.language_model.layers.7.mlp.experts.3.down_proj."
