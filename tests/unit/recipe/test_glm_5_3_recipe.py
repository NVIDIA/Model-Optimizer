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

"""Wildcard-precedence test for the GLM-5.3-Flash-BF16 checkpoint-mirror PTQ recipe.

The recipe relies on wildcard scoping over ``base_disable_all`` rather than an
explicit per-module map, so a few non-obvious matches decide correctness:

* ``*.experts.*`` needs a literal ``.experts.``, so ``mlp.shared_experts.*`` is
  *not* matched and the shared experts stay BF16.
* The vision tower reuses the language MLP's leaf names (``mlp.gate_proj`` /
  ``up_proj`` / ``down_proj``), so the dense-MLP patterns match ``model.visual.*``
  too -- only the ``*visual*`` disable (which must follow them) keeps the
  vision tower in BF16.

This pins that behaviour so it can't silently drift.
"""

import pytest
import torch.nn as nn
from _test_utils.torch import transformers_models

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe

_RECIPE = "models/zai-org/GLM-5.3-Flash-BF16/ptq/nvfp4_experts_dense_mlp-kv_fp8_cast"
_H = 32


class _MLP(nn.Module):
    """Plain MLP leaf names, shared by the dense MLP, the shared experts, and vision."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(_H, _H, bias=False)
        self.up_proj = nn.Linear(_H, _H, bias=False)
        self.down_proj = nn.Linear(_H, _H, bias=False)


class _VisionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(_H, 3 * _H)
        self.proj = nn.Linear(_H, _H)


class _VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()  # same gate_proj / up_proj / down_proj leaf names as the language MLP
        self.attn = _VisionAttention()


def _nvfp4(quantizer):
    return quantizer.is_enabled and quantizer.num_bits == (2, 1)


def _fp8(quantizer):
    return quantizer.is_enabled and quantizer.num_bits == (4, 3)


def test_glm_5_3_recipe_quantizer_precedence():
    pytest.importorskip("transformers.models.glm5_next", reason="needs transformers>=5.16.1")
    model = transformers_models.get_tiny_glm5_next()

    config = load_recipe(_RECIPE).quantize.model_dump()
    # The recipe uses plain max calibration; here we only assert quantizer placement,
    # so drop the algorithm to avoid needing a calibration forward pass.
    assert config["algorithm"]["method"] == "max"
    config["algorithm"] = None
    mtq.quantize(model, config)

    dense, sparse = model.model.language_model.layers

    # Routed experts (fused into 3D params, one weight quantizer per expert) -> NVFP4 W4A4.
    experts = sparse.mlp.experts
    for name in ("gate_up_proj", "down_proj"):
        assert _nvfp4(getattr(experts, f"{name}_input_quantizer"))
        assert all(_nvfp4(q) for q in getattr(experts, f"{name}_weight_quantizers"))

    # Dense MLP (layers 0-2) -> NVFP4.
    for proj in (dense.mlp.gate_proj, dense.mlp.up_proj, dense.mlp.down_proj):
        assert _nvfp4(proj.weight_quantizer)
        assert _nvfp4(proj.input_quantizer)

    # Sparse-MLA KV cache -> FP8.
    assert _fp8(sparse.self_attn.k_bmm_quantizer)
    assert _fp8(sparse.self_attn.v_bmm_quantizer)

    # The whole vision tower stays BF16 -- the load-bearing case: its blocks and merger reuse
    # gate_proj/up_proj/down_proj, so the dense-MLP patterns match them and only the later
    # `*visual*` disable keeps them off.
    for name, module in model.model.visual.named_modules():
        if name.endswith("quantizer"):
            assert module.is_enabled is False, name

    # Shared experts stay BF16: `*.experts.*` needs a literal `.experts.`, so `shared_experts`
    # is skipped. (The router `mlp.gate` is not an nn.Linear, so it gets no quantizer at all.)
    for proj in (
        sparse.mlp.shared_experts.gate_proj,
        sparse.mlp.shared_experts.up_proj,
        sparse.mlp.shared_experts.down_proj,
    ):
        assert proj.weight_quantizer.is_enabled is False
        assert proj.input_quantizer.is_enabled is False

    # Both attention families' projections stay BF16, including the KDA conv1d and the indexer.
    for attn in (dense.self_attn, sparse.self_attn):
        for name, module in attn.named_modules():
            if name.endswith(("weight_quantizer", "input_quantizer")):
                assert module.is_enabled is False, name

    # Embeddings and lm_head stay BF16.
    assert model.model.language_model.embed_tokens.weight_quantizer.is_enabled is False
    assert model.lm_head.weight_quantizer.is_enabled is False


class _McoreMLP(nn.Module):
    """Megatron-Bridge MLP leaf names (dense MLP, each local expert, and the shared experts)."""

    def __init__(self):
        super().__init__()
        self.linear_fc1 = nn.Linear(_H, 2 * _H, bias=False)
        self.linear_fc2 = nn.Linear(_H, _H, bias=False)


class _McoreMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.Module()
        self.experts.local_experts = nn.ModuleList([_McoreMLP(), _McoreMLP()])
        self.shared_experts = _McoreMLP()


class _McoreLayer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.inner_layer = nn.Module()  # mHC wraps each block as `<layer>.inner_layer`
        self.inner_layer.mlp = mlp


class _McoreGLM53Flash(nn.Module):
    """Megatron-Bridge naming: a dense and an MoE decoder layer, the MTP layer, and vision."""

    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.decoder = nn.Module()
        self.language_model.decoder.layers = nn.ModuleList(
            [_McoreLayer(_McoreMLP()), _McoreLayer(_McoreMoE())]
        )
        self.language_model.mtp = nn.Module()
        self.language_model.mtp.layers = nn.ModuleList([_McoreLayer(_McoreMoE())])
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([_VisionBlock()])


def test_glm_5_3_recipe_megatron_names():
    model = _McoreGLM53Flash()
    config = load_recipe(_RECIPE).quantize.model_dump()
    config["algorithm"] = None
    mtq.quantize(model, config)

    dense, sparse = (layer.inner_layer.mlp for layer in model.language_model.decoder.layers)
    mtp = model.language_model.mtp.layers[0].inner_layer.mlp

    # Dense MLP and routed experts -> NVFP4 W4A4.
    for mlp in (dense, *sparse.experts.local_experts):
        for proj in (mlp.linear_fc1, mlp.linear_fc2):
            assert _nvfp4(proj.weight_quantizer)
            assert _nvfp4(proj.input_quantizer)

    # Shared experts, the whole MTP layer, and the vision tower stay BF16.
    for mlp in (sparse.shared_experts, mtp.shared_experts, *mtp.experts.local_experts):
        for proj in (mlp.linear_fc1, mlp.linear_fc2):
            assert proj.weight_quantizer.is_enabled is False
            assert proj.input_quantizer.is_enabled is False
    vmlp = model.visual.blocks[0].mlp
    for proj in (vmlp.gate_proj, vmlp.up_proj, vmlp.down_proj):
        assert proj.weight_quantizer.is_enabled is False
