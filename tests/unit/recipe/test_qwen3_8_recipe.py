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

"""Wildcard-precedence test for the Qwen3.8-2.4T-A95B checkpoint-mirror PTQ recipe.

This recipe leans entirely on wildcard precedence over ``base_disable_all``: the
broad ``*linear_attn*`` rules deliberately reach ``linear_attn.conv1d`` (an
``nn.Conv1d``, which ModelOpt wraps with quantizers), so the gated-delta conv1d
is FP8 -- matching the published ``nvidia/Qwen3.8-2.4T-A95B-NVFP4`` checkpoint,
whose ``hf_quant_config.json`` lists ``linear_attn.conv1d`` as FP8 on every
gated-delta layer. The test pins that placement (and the trailing ``*mtp*``
disable winning over the broad enables) so it can't silently drift again.
"""

import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.plugins.huggingface import register_fused_experts_on_the_fly

RECIPE = (
    "models/Qwen/Qwen3.8-2.4T-A95B/ptq/nvfp4_experts_mse-fp8_self_attn-fp8_linear_attn-kv_fp8_cast"
)

_H = 32


class _FusedExperts(nn.Module):
    """Fused MoE experts, mirroring the grouped ``gate_up_proj`` / ``down_proj`` layout."""

    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.intermediate_dim = _H
        self.gate_up_proj = nn.Parameter(torch.randn(2, _H, 2 * _H))
        self.down_proj = nn.Parameter(torch.randn(2, _H, _H))

    def forward(self, hidden_states):
        return hidden_states


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _FusedExperts()
        self.gate = nn.Linear(_H, 2, bias=False)
        self.shared_expert = nn.Linear(_H, _H, bias=False)


class _SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(_H, _H, bias=False)
        self.k_proj = nn.Linear(_H, _H, bias=False)
        self.v_proj = nn.Linear(_H, _H, bias=False)
        self.o_proj = nn.Linear(_H, _H, bias=False)


class _LinearAttention(nn.Module):
    """Gated-delta linear attention: projections plus a depthwise causal ``conv1d``."""

    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(_H, 3 * _H, bias=False)
        self.in_proj_z = nn.Linear(_H, _H, bias=False)
        self.in_proj_a = nn.Linear(_H, _H, bias=False)
        self.in_proj_b = nn.Linear(_H, _H, bias=False)
        self.out_proj = nn.Linear(_H, _H, bias=False)
        self.conv1d = nn.Conv1d(_H, _H, kernel_size=3, groups=_H, bias=False)


class _GatedDeltaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = _LinearAttention()
        self.mlp = _MoE()


class _FullAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _SelfAttention()
        self.mlp = _MoE()


class _MTP(nn.Module):
    """Multi-token-prediction block; carries the same submodules but stays BF16."""

    def __init__(self):
        super().__init__()
        self.linear_attn = _LinearAttention()
        self.mlp = _MoE()


class _Qwen38Model(nn.Module):
    """Tiny stand-in for the ``qwen3_5_moe_text`` hybrid-attention MoE."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        # Hybrid attention: a gated-delta (linear-attention) layer + a full-attention layer.
        self.model.layers = nn.ModuleList([_GatedDeltaLayer(), _FullAttentionLayer()])
        self.model.mtp = _MTP()
        self.lm_head = nn.Linear(_H, _H, bias=False)


def _fp8(quantizer):
    return quantizer.is_enabled and quantizer.num_bits == (4, 3)


def test_qwen3_8_recipe_quantizer_precedence():
    model = _Qwen38Model()
    register_fused_experts_on_the_fly(model)

    config = load_recipe(RECIPE).quantize.model_dump()
    # The recipe calibrates with MSE + an FP8-scale sweep; that behaviour is covered
    # elsewhere. Here we only assert quantizer *placement*, so drop the algorithm to
    # avoid needing a calibration forward pass.
    assert config["algorithm"]["method"] == "mse"
    config["algorithm"] = None
    mtq.quantize(model, config)

    gated = model.model.layers[0]  # gated-delta / linear-attention layer
    full = model.model.layers[1]  # full-attention layer

    # Routed experts -> NVFP4: static weight scales, dynamic block-16 input scales.
    for mlp in (gated.mlp, full.mlp):
        weight = mlp.experts.gate_up_proj_weight_quantizers[0]
        assert weight.is_enabled
        assert weight.num_bits == (2, 1)
        assert weight.block_sizes[-1] == 16
        assert weight.block_sizes["type"] == "static"

        inputs = mlp.experts.gate_up_proj_input_quantizer
        assert inputs.is_enabled
        assert inputs.num_bits == (2, 1)
        assert inputs.block_sizes["type"] == "dynamic"

        # Router gate and shared expert stay BF16.
        assert mlp.gate.weight_quantizer.is_enabled is False
        assert mlp.shared_expert.weight_quantizer.is_enabled is False

    # Self-attention -> FP8 W8A8 on every projection.
    for proj in (
        full.self_attn.q_proj,
        full.self_attn.k_proj,
        full.self_attn.v_proj,
        full.self_attn.o_proj,
    ):
        assert _fp8(proj.weight_quantizer)
        assert _fp8(proj.input_quantizer)

    # Linear-attention -> FP8 W8A8 on the projections AND the conv1d. The conv1d
    # assertion is the regression guard: nn.Conv1d is a registered quant module, so
    # the broad `*linear_attn*` rules must reach `linear_attn.conv1d`.
    la = gated.linear_attn
    for proj in (la.in_proj_qkv, la.in_proj_z, la.in_proj_a, la.in_proj_b, la.out_proj, la.conv1d):
        assert _fp8(proj.weight_quantizer)
        assert _fp8(proj.input_quantizer)
    assert gated.linear_attn.conv1d.weight_quantizer.is_enabled is True

    # MTP block stays BF16: the trailing `*mtp*` disable wins over the broad enables,
    # including over `*linear_attn*` for the MTP's own conv1d/projections.
    mtp = model.model.mtp
    assert mtp.linear_attn.conv1d.weight_quantizer.is_enabled is False
    assert mtp.linear_attn.out_proj.weight_quantizer.is_enabled is False
    assert mtp.mlp.experts.gate_up_proj_input_quantizer.is_enabled is False

    # lm_head stays BF16 (never re-enabled after base_disable_all).
    assert model.lm_head.weight_quantizer.is_enabled is False
