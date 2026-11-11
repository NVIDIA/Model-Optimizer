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

"""Custom mapping from Qwen3.5-VL Hugging Face models to Megatron Core models.

Qwen3.5 interleaves GatedDeltaNet linear-attention layers (fused ``in_proj``, split here into
HF's ``in_proj_qkv`` / ``_z`` / ``_b`` / ``_a``) with gated full-attention layers, and adds MoE
shared experts. Only the language model is exported; the vision tower is copied from HF.
Routed experts are written one entry per expert, exported from either expert layout.
"""

from .mcore_custom import (
    GatedDeltaNetSlicing,
    GatedMLPSlicing,
    GroupedGatedMLPSlicing,
    GroupedMLPSlicing,
    NameRemapping,
    with_language_model_prefix,
)
from .mcore_qwen import qwen3_causal_lm_export

# Vision-tower weights copied straight from the HF checkpoint (never quantized).
QWEN3_5_VL_VISION_PREFIXES = ("model.visual.",)

# Qwen3.5 adds linear-attention layers and shared experts on top of the Qwen3 rules.
_qwen3_5_extra_export: dict = {
    # Linear attention (GatedDeltaNet). ``linear_attn`` splits the fused in_proj; the rest
    # are plain renames of the surrounding parameters.
    "linear_attn": GatedDeltaNetSlicing("model.layers.{}.linear_attn."),
    "linear_attn.conv1d": NameRemapping("model.layers.{}.linear_attn.conv1d."),
    "linear_attn.A_log": NameRemapping("model.layers.{}.linear_attn.A_log"),
    "linear_attn.dt_bias": NameRemapping("model.layers.{}.linear_attn.dt_bias"),
    # Megatron's GDN output norm is zero-centered; HF's RMSNorm gamma is centered on 1.
    "linear_attn.out_norm": NameRemapping(
        "model.layers.{}.linear_attn.norm.", {"zero_centered_gamma": True}
    ),
    "linear_attn.out_proj": NameRemapping("model.layers.{}.linear_attn.out_proj."),
    # Grouped experts export one entry per expert with gate/up split, matching the released NVFP4
    # checkpoints: vLLM has no parameter for a packed `experts.down_proj_weight_scale_2`.
    # ``local_experts.*`` (SequentialMLP) already does this in the Qwen3 rules.
    "experts.linear_fc1": GroupedGatedMLPSlicing("model.layers.{}.mlp.experts.{{}}"),
    "experts.linear_fc2": GroupedMLPSlicing("model.layers.{}.mlp.experts.{{}}.down_proj"),
    # MoE shared experts (routed experts + router come from the Qwen3 rules).
    "shared_experts.linear_fc1": GatedMLPSlicing("model.layers.{}.mlp.shared_expert."),
    "shared_experts.linear_fc2": NameRemapping("model.layers.{}.mlp.shared_expert.down_proj."),
    "shared_experts.gate_weight": NameRemapping("model.layers.{}.mlp.shared_expert_gate.weight"),
}

qwen3_5_vl_causal_lm_export = with_language_model_prefix(
    {**qwen3_causal_lm_export, **_qwen3_5_extra_export}
)
