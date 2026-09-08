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

"""Wildcard-precedence tests for the GLM-5.3-Flash checkpoint-mirror PTQ recipes.

Both recipes rely on wildcard scoping over ``base_disable_all`` rather than an
explicit per-module map, so a few non-obvious matches decide correctness:

* ``*.experts.*`` needs a literal ``.experts.``, so ``mlp.shared_experts.*`` is
  *not* matched and the shared experts stay BF16.
* In the experts+dense-MLP recipe the vision tower reuses the language MLP's leaf
  names (``mlp.gate_proj`` / ``up_proj`` / ``down_proj``), so the dense-MLP patterns
  match ``model.visual.*`` too -- only the trailing ``*visual*`` disable (which must
  stay last) keeps the vision tower in BF16.
* ``*mlp.gate_proj*`` must not catch the router ``mlp.gate``.

These pin that behaviour so it can't silently drift.
"""

import pytest
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe

_RECIPE_DIR = "models/zai-org/GLM-5.3-Flash/ptq"
_H = 32


class _MLP(nn.Module):
    """Plain MLP leaf names, shared by the dense MLP, each routed/shared expert, and vision."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(_H, _H, bias=False)
        self.up_proj = nn.Linear(_H, _H, bias=False)
        self.down_proj = nn.Linear(_H, _H, bias=False)


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([_MLP(), _MLP()])
        self.shared_experts = _MLP()
        self.gate = nn.Linear(_H, 2, bias=False)  # router


class _KDA(nn.Module):
    """KDA linear attention (projections plus a depthwise causal conv1d)."""

    def __init__(self):
        super().__init__()
        self.in_proj_qkvz = nn.Linear(_H, _H, bias=False)
        self.conv1d = nn.Conv1d(_H, _H, kernel_size=3, groups=_H, bias=False)
        self.out_proj = nn.Linear(_H, _H, bias=False)


class _MLA(nn.Module):
    """NoPE sparse-MLA attention (and the vision attention, which reuses the leaf names)."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(_H, _H, bias=False)
        self.kv_proj = nn.Linear(_H, _H, bias=False)
        self.o_proj = nn.Linear(_H, _H, bias=False)


class _DenseLayer(nn.Module):
    """Layers 0-2: KDA attention + a plain (dense) MLP."""

    def __init__(self):
        super().__init__()
        self.linear_attn = _KDA()
        self.mlp = _MLP()


class _SparseLayer(nn.Module):
    """Layers 3-44: MLA attention + an MoE block."""

    def __init__(self):
        super().__init__()
        self.self_attn = _MLA()
        self.mlp = _MoE()


class _VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()  # same gate_proj / up_proj / down_proj leaf names as the language MLP
        self.attn = _MLA()


class _GLM53Flash(nn.Module):
    """Tiny stand-in for the ``glm5_next`` VLM MoE (one dense + one sparse layer + vision)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_DenseLayer(), _SparseLayer()])
        self.model.visual = nn.Module()
        self.model.visual.blocks = nn.ModuleList([_VisionBlock()])
        self.lm_head = nn.Linear(_H, _H, bias=False)


def _nvfp4(quantizer):
    return quantizer.is_enabled and quantizer.num_bits == (2, 1)


@pytest.mark.parametrize(
    ("recipe", "quantizes_dense_mlp"),
    [
        ("nvfp4_experts_only-kv_fp8_cast", False),
        ("nvfp4_experts_dense_mlp-kv_fp8_cast", True),
    ],
)
def test_glm_5_3_recipe_quantizer_precedence(recipe, quantizes_dense_mlp):
    model = _GLM53Flash()

    config = load_recipe(f"{_RECIPE_DIR}/{recipe}").quantize.model_dump()
    # Both recipes use plain max calibration; here we only assert quantizer placement,
    # so drop the algorithm to avoid needing a calibration forward pass.
    assert config["algorithm"]["method"] == "max"
    config["algorithm"] = None
    mtq.quantize(model, config)

    dense = model.model.language_model.layers[0]
    sparse = model.model.language_model.layers[1]

    # Routed experts -> NVFP4 W4A4 in both recipes.
    for expert in sparse.mlp.experts:
        for proj in (expert.gate_proj, expert.up_proj, expert.down_proj):
            assert _nvfp4(proj.weight_quantizer)
            assert _nvfp4(proj.input_quantizer)

    # Dense MLP (layers 0-2) -> NVFP4 only in the experts+dense-MLP recipe.
    for proj in (dense.mlp.gate_proj, dense.mlp.up_proj, dense.mlp.down_proj):
        assert proj.weight_quantizer.is_enabled is quantizes_dense_mlp
        assert proj.input_quantizer.is_enabled is quantizes_dense_mlp
        if quantizes_dense_mlp:
            assert proj.weight_quantizer.num_bits == (2, 1)

    # Vision tower stays BF16 in BOTH recipes. For experts+dense-MLP this is the
    # load-bearing case: the vision MLP reuses gate_proj/up_proj/down_proj, so the
    # dense-MLP patterns match it and only the trailing `*visual*` disable keeps it off.
    vblock = model.model.visual.blocks[0]
    for proj in (vblock.mlp.gate_proj, vblock.mlp.up_proj, vblock.mlp.down_proj):
        assert proj.weight_quantizer.is_enabled is False
        assert proj.input_quantizer.is_enabled is False

    # Shared experts and the router gate stay BF16: `*.experts.*` needs a literal
    # `.experts.` (so `shared_experts` is skipped), and `*mlp.gate_proj*` doesn't match
    # the router `mlp.gate`.
    for proj in (
        sparse.mlp.shared_experts.gate_proj,
        sparse.mlp.shared_experts.up_proj,
        sparse.mlp.shared_experts.down_proj,
    ):
        assert proj.weight_quantizer.is_enabled is False
    assert sparse.mlp.gate.weight_quantizer.is_enabled is False

    # Both attention families stay BF16, including the KDA conv1d.
    assert dense.linear_attn.conv1d.weight_quantizer.is_enabled is False
    assert dense.linear_attn.in_proj_qkvz.weight_quantizer.is_enabled is False
    assert dense.linear_attn.out_proj.weight_quantizer.is_enabled is False
    for proj in (sparse.self_attn.q_proj, sparse.self_attn.kv_proj, sparse.self_attn.o_proj):
        assert proj.weight_quantizer.is_enabled is False

    # lm_head stays BF16.
    assert model.lm_head.weight_quantizer.is_enabled is False
