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

"""HF text roots, hybrid roles and position layouts; forwards remain upstream-owned."""

from .base import TransformersFamilyAdapter
from .registry import register_transformers_family

__all__ = []


class _HybridAttentionFamily(TransformersFamilyAdapter):
    def unit_attributes(self, block, unit_types):
        linear = getattr(block, "block_type", None) == "linear_attention" or getattr(
            block, "is_linear_attn", False
        )
        result = []
        for role in unit_types:
            if linear and "self_attn" in block._modules and role in ("attention", "gdn"):
                if role == "gdn":
                    result.append("self_attn")
            else:
                result.extend(super().unit_attributes(block, (role,)))
        return list(dict.fromkeys(result))


class _OmniFamily(_HybridAttentionFamily):
    def teacher_forward_target(self, model):
        return getattr(model, "thinker", model)


class _SoftcapFamily(TransformersFamilyAdapter):
    def transform_logits(self, model, logits):
        config = self.teacher_forward_target(model).config.get_text_config()
        multiplier = getattr(config, "output_multiplier", None)
        if multiplier is not None:
            logits = logits * multiplier
        cap = getattr(
            self.teacher_forward_target(model),
            "final_logit_softcapping",
            config.final_logit_softcapping,
        )
        return (logits / cap).tanh() * cap


_CAUSAL_ROOTS = ("model.layers", "layers")
_VISION_ROOTS = ("model.language_model.layers", "language_model.layers", "model.layers", "layers")

for name in ("llama", "qwen3", "qwen3_moe", "deepseek_v3"):
    register_transformers_family(TransformersFamilyAdapter(name, _CAUSAL_ROOTS, False))
for name in ("deepseek_v32", "deepseek_v4", "glm_moe_dsa", "mimo_v2_flash", "mimo_v2"):
    register_transformers_family(TransformersFamilyAdapter(name, _CAUSAL_ROOTS))
for name in ("qwen3_next", "kimi_linear", "bailing_hybrid"):
    child = "linear_attn.conv1d" if name == "qwen3_next" else "self_attn.conv1d"
    projections = ("self_attn.forget_gate.f_proj",) if name == "bailing_hybrid" else ()
    register_transformers_family(
        _HybridAttentionFamily(
            name,
            _CAUSAL_ROOTS,
            functional_children=(child,),
            additional_fsdp_children=projections,
            batch_packed_documents=name == "qwen3_next",
        )
    )
for name in ("qwen3_5", "qwen3_5_moe", "qwen4_exp"):
    register_transformers_family(
        _HybridAttentionFamily(
            name,
            _VISION_ROOTS,
            functional_children=("linear_attn.conv1d",),
            batch_packed_documents=True,
        ),
        name,
        name + "_text",
    )
register_transformers_family(
    _HybridAttentionFamily("qwen3_vl_moe", _VISION_ROOTS), "qwen3_vl_moe", "qwen3_vl_moe_text"
)
register_transformers_family(
    _OmniFamily(
        "qwen3_omni_moe",
        (
            "thinker.model.language_model.layers",
            "thinker.model.layers",
            "model.language_model.layers",
            "language_model.layers",
            "model.layers",
            "layers",
        ),
    ),
    "qwen3_omni_moe",
    "qwen3_omni_moe_thinker",
    "qwen3_omni_moe_text",
)
register_transformers_family(
    _HybridAttentionFamily(
        "glm5_next",
        _VISION_ROOTS,
        functional_children=("self_attn.conv1d",),
        additional_fsdp_children=(
            "self_attn.forget_gate.f_a_proj",
            "self_attn.forget_gate.f_b_proj",
        ),
    ),
    "glm5_next",
    "glm5_next_text",
)
for name in ("minimax_m3_vl",):
    register_transformers_family(
        TransformersFamilyAdapter(name, _VISION_ROOTS), name, name + "_text"
    )
for name in ("muse_spark", "mistral_large4"):
    register_transformers_family(_SoftcapFamily(name, _VISION_ROOTS), name, name + "_text")
register_transformers_family(TransformersFamilyAdapter("kimi_k25", _VISION_ROOTS))
register_transformers_family(
    _HybridAttentionFamily("kimi_k3", ("language_model.model.layers", "model.layers", "layers"))
)

for name in ("deepseek_v41",):
    register_transformers_family(TransformersFamilyAdapter(name, _CAUSAL_ROOTS))
