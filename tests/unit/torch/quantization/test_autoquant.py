# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import io
import math
from types import SimpleNamespace

import pytest
import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.quantization.models import SimpleConv, SimpleConvLinear, SimpleLinear

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.algorithms as quant_algorithms
from modelopt.torch.opt.searcher import LPS
from modelopt.torch.quantization._auto_quantize_cost import (
    EXCLUDED_MODULE_NAME_PATTERNS_KEY,
    get_auto_quantize_cost_model,
    infer_active_moe_expert_ratio,
    normalize_auto_quantize_constraints,
)
from modelopt.torch.quantization.algorithms import (
    AutoQuantizeGradientSearcher,
    AutoQuantizeKLDivSearcher,
    QuantRecipe,
    QuantRecipeHparam,
    estimate_quant_compression,
)
from modelopt.torch.quantization.config import _base_disable_all, _default_disabled_quantizer_cfg
from modelopt.torch.quantization.conversion import set_quantizer_by_cfg
from modelopt.torch.quantization.plugins.huggingface import _QuantFusedExperts
from modelopt.torch.utils import safe_load, safe_save
from modelopt.torch.utils.distributed import DistributedProcessGroup


def test_auto_quantize_score_chunked_matches_full_product(monkeypatch):
    monkeypatch.setattr(quant_algorithms, "_AUTO_QUANTIZE_SCORE_CHUNK_SIZE", 11)
    grad_output = torch.linspace(-2.0, 2.0, steps=3 * 5 * 7, dtype=torch.float16).reshape(3, 5, 7)
    output_diff = torch.linspace(1.5, -1.5, steps=3 * 5 * 7, dtype=torch.float16).reshape(3, 5, 7)

    score = quant_algorithms._get_auto_quantize_score(grad_output, output_diff)
    expected = (grad_output.float() * output_diff.float()).clamp(-1e10, 1e10).square().sum()

    assert torch.allclose(score, expected)
    with pytest.raises(ValueError, match="same number of elements"):
        quant_algorithms._get_auto_quantize_score(torch.ones(2), torch.ones(3))


def test_auto_quantize_hidden_recon_score_normalized_mse():
    reference = torch.tensor([[1.0, 2.0]])
    quantized = torch.tensor([[2.0, 4.0]])

    score = quant_algorithms._get_hidden_recon_score(reference, quantized)

    expected = (quantized - reference).square().mean() / reference.square().mean()
    assert score.item() == pytest.approx(expected.item())
    with pytest.raises(ValueError, match="same shape"):
        quant_algorithms._get_hidden_recon_score(torch.ones(2), torch.ones(3))


class _AttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(32, 32)
        self.k_proj = torch.nn.Linear(32, 32)
        self.v_proj = torch.nn.Linear(32, 32)
        self.o_proj = torch.nn.Linear(32, 32)

    def forward(self, x):
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            x = layer(x)
        return x


class _QwenStyleSelfAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(32, 32)
        self.k_proj = torch.nn.Linear(32, 32)
        self.v_proj = torch.nn.Linear(32, 32)
        self.o_proj = torch.nn.Linear(32, 32)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(torch.tanh(q + k) * torch.sigmoid(v))


class _QwenStyleLinearAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = torch.nn.Linear(32, 32)
        self.in_proj_z = torch.nn.Linear(32, 32)
        self.in_proj_a = torch.nn.Linear(32, 32)
        self.in_proj_b = torch.nn.Linear(32, 32)
        self.out_proj = torch.nn.Linear(32, 32)

    def forward(self, x):
        qkv = self.in_proj_qkv(x)
        z = torch.sigmoid(self.in_proj_z(x))
        a = torch.tanh(self.in_proj_a(x))
        b = torch.sigmoid(self.in_proj_b(x))
        return self.out_proj(torch.tanh(qkv + a) * z + b)


class _QwenStyleKeywordSelfAttention(_QwenStyleSelfAttention):
    def forward(self, hidden_states):
        return super().forward(hidden_states)


class _QwenStyleKeywordLinearAttention(_QwenStyleLinearAttention):
    def forward(self, hidden_states):
        return super().forward(hidden_states)


class _QwenStyleHybridAttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "self_attn": _QwenStyleSelfAttention(),
                        "linear_attn": _QwenStyleLinearAttention(),
                    }
                )
            ]
        )
        self.mlp = torch.nn.Linear(32, 32)

    def forward(self, x):
        layer = self.layers[0]
        x = layer["self_attn"](x) + layer["linear_attn"](x)
        return self.mlp(x)

    def get_input(self):
        return torch.randn(1, 4, 32)


class _QwenStyleKeywordHybridAttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "self_attn": _QwenStyleKeywordSelfAttention(),
                        "linear_attn": _QwenStyleKeywordLinearAttention(),
                    }
                )
            ]
        )
        self.mlp = torch.nn.Linear(32, 32)

    def forward(self, x):
        layer = self.layers[0]
        x = layer["self_attn"](hidden_states=x) + layer["linear_attn"](hidden_states=x)
        return self.mlp(x)

    def get_input(self):
        return torch.randn(1, 4, 32)


class _CacheDefaultModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.linear = torch.nn.Linear(32, 32)
        self.use_cache_seen = []

    def forward(self, x):
        self.use_cache_seen.append(self.config.use_cache)
        return self.linear(x)

    def get_input(self):
        return torch.randn(1, 4, 32)


class TransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _AttentionLayer()
        self.mlp = torch.nn.Linear(32, 32)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x

    def get_input(self):
        return torch.randn(1, 4, 32)


class _AutoQuantMoeModel(torch.nn.Module):
    def __init__(self, num_experts_attr="num_experts"):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_experts_per_tok=2))
        setattr(self.config.text_config, num_experts_attr, 8)
        self.mlp = torch.nn.Module()
        self.mlp.experts = torch.nn.ModuleList()
        for _ in range(2):
            expert = torch.nn.Module()
            expert.gate_proj = torch.nn.Linear(32, 32)
            expert.up_proj = torch.nn.Linear(32, 32)
            expert.down_proj = torch.nn.Linear(32, 32)
            self.mlp.experts.append(expert)
        self.mlp.shared_expert = torch.nn.Linear(32, 32)

    def forward(self, x):
        y = self.mlp.shared_expert(x)
        for expert in self.mlp.experts:
            y = y + expert.down_proj(expert.gate_proj(x) + expert.up_proj(x))
        return y

    def get_input(self):
        return torch.randn(1, 4, 32)


class _QwenStyleSharedExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(32, 32)
        self.up_proj = torch.nn.Linear(32, 32)
        self.down_proj = torch.nn.Linear(32, 32)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) + self.up_proj(x))


class _QwenStyleSharedExpertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.shared_expert = _QwenStyleSharedExpert()

    def forward(self, x):
        return self.mlp.shared_expert(x)

    def get_input(self):
        return torch.randn(1, 4, 32)


class _QwenStyleFusedExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 64, 32))
        self.down_proj = torch.nn.Parameter(torch.randn(2, 32, 32))
        self.act_fn = torch.nn.SiLU()

    def forward(self, hidden_states):
        output = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            gate_up = torch.nn.functional.linear(hidden_states, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = up * self.act_fn(gate)
            output += torch.nn.functional.linear(intermediate, self.down_proj[expert_idx])
        return output


class _QwenStyleFusedExpertsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.experts = _QuantFusedExperts.convert(_QwenStyleFusedExperts())

    def forward(self, hidden_states):
        return self.mlp.experts(hidden_states)

    def get_input(self):
        return torch.randn(1, 4, 32)


@pytest.mark.parametrize(
    ("quant_cfg", "other_quant_cfg", "is_less_than"),
    [
        (mtq.FP8_DEFAULT_CFG, None, True),
        (mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG, True),
        (None, mtq.INT8_DEFAULT_CFG, False),
    ],
)
def test_quant_recipe(quant_cfg, other_quant_cfg, is_less_than):
    qr_this = QuantRecipe(quant_cfg)
    qr_other = QuantRecipe(other_quant_cfg)
    assert (qr_this < qr_other) == is_less_than

    qr_this_duplicate = QuantRecipe(quant_cfg)
    assert qr_this_duplicate in {qr_this}


def test_quant_recipe_hparam():
    model_test = torch.nn.Linear(4, 16)
    model_ref = torch.nn.Linear(4, 16)
    model_ref.load_state_dict(model_test.state_dict())

    model_test = mtq.quantize(model_test, mtq.INT8_DEFAULT_CFG)
    model_ref = mtq.quantize(model_ref, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)

    search_recipes = [
        QuantRecipe(mtq.INT8_DEFAULT_CFG),
        QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG),
    ]
    hparam = QuantRecipeHparam(
        search_recipes,
        quant_modules=[model_test],
    )
    model_test._register_hparam("quant_recipe", hparam)
    assert model_test.quant_recipe == QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    assert model_test.get_hparam("quant_recipe").choices == sorted(
        [*search_recipes, QuantRecipe(quant_cfg=None)]
    )

    model_test.quant_recipe = QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    inputs = torch.randn(1, 4, 4)
    output_test = model_test(inputs)
    output_ref = model_ref(inputs)

    assert torch.allclose(output_test, output_ref)


def test_quant_recipe_hparam_cost_weight():
    model_test = mtq.quantize(torch.nn.Linear(4, 16), mtq.INT8_DEFAULT_CFG)
    search_recipes = [QuantRecipe(mtq.INT8_DEFAULT_CFG)]
    hparam = QuantRecipeHparam(
        search_recipes,
        quant_modules=[model_test],
        quant_module_names=["layers.0.mlp.experts.0.down_proj"],
        cost_weight=0.25,
    )

    dense_cost = hparam.get_cost(QuantRecipe(quant_cfg=None))
    int8_cost = hparam.get_cost(QuantRecipe(mtq.INT8_DEFAULT_CFG))

    assert dense_cost == pytest.approx(model_test.weight.numel() * 0.25)
    assert int8_cost == pytest.approx(model_test.weight.numel() * 0.25 * 0.5)


def test_quant_recipe_hparam_zero_cost_weight():
    model_test = mtq.quantize(torch.nn.Linear(4, 16), mtq.INT8_DEFAULT_CFG)
    hparam = QuantRecipeHparam(
        [QuantRecipe(mtq.INT8_DEFAULT_CFG)],
        quant_modules=[model_test],
        quant_module_names=["visual.blocks.0.attn.qkv"],
        cost_weight=0.0,
    )

    assert hparam.get_cost(QuantRecipe(quant_cfg=None)) == pytest.approx(0.0)
    assert hparam.get_cost(QuantRecipe(mtq.INT8_DEFAULT_CFG)) == pytest.approx(0.0)


def test_auto_quantize_cost_model_excludes_explicit_module_name_patterns():
    visual = torch.nn.Linear(4, 16)
    mtp = torch.nn.Linear(4, 16)
    lm_head = torch.nn.Linear(4, 16)
    cost_model = get_auto_quantize_cost_model("weight")
    cost_constraints = {EXCLUDED_MODULE_NAME_PATTERNS_KEY: ["*visual*", "*vision_tower*", "*mtp*"]}

    total_weight_size = cost_model.total_weight_size(
        [
            ("model.visual.blocks.0.attn.qkv", visual),
            ("model.mtp.layers.0.mlp", mtp),
            ("lm_head", lm_head),
        ],
        is_auto_quantize_module=lambda module: True,
        cost_constraints=cost_constraints,
    )

    assert total_weight_size == pytest.approx(lm_head.weight.numel())
    assert cost_model.module_cost_weight(
        ["model.visual.blocks.0.attn.qkv"], cost_constraints
    ) == pytest.approx(0.0)
    assert cost_model.module_cost_weight(
        ["model.mtp.layers.0.mlp"], cost_constraints
    ) == pytest.approx(0.0)
    assert cost_model.module_cost_weight(
        ["model.visual.blocks.0.attn.qkv", "lm_head"], cost_constraints
    ) == pytest.approx(1.0)


def test_auto_quantize_cost_model_counts_visual_modules_by_default():
    visual = torch.nn.Linear(4, 16)
    cost_model = get_auto_quantize_cost_model("weight")

    total_weight_size = cost_model.total_weight_size(
        [("visual.blocks.0.attn.qkv", visual)],
        is_auto_quantize_module=lambda module: True,
        cost_constraints={},
    )

    assert total_weight_size == pytest.approx(visual.weight.numel())


def test_active_moe_cost_model_counts_fused_experts_without_weight():
    fused_experts = torch.nn.Module()
    fused_experts.gate_up_proj = torch.nn.Parameter(torch.empty(2, 3, 5))
    fused_experts.down_proj = torch.nn.Parameter(torch.empty(2, 5, 3))
    visual = torch.nn.Linear(4, 16)
    cost_model = get_auto_quantize_cost_model("active_moe")

    total_weight_size = cost_model.total_weight_size(
        [
            ("layers.0.mlp.experts", fused_experts),
            ("model.visual.blocks.0.attn.qkv", visual),
        ],
        is_auto_quantize_module=lambda module: True,
        cost_constraints={
            "active_moe_expert_ratio": 0.25,
            EXCLUDED_MODULE_NAME_PATTERNS_KEY: ["*visual*"],
        },
    )

    assert total_weight_size == pytest.approx(
        (fused_experts.gate_up_proj.numel() + fused_experts.down_proj.numel()) * 0.25
    )


@pytest.mark.parametrize("num_experts_attr", ["num_experts", "num_local_experts"])
def test_auto_quantize_active_moe_cost_model(num_experts_attr):
    model = _AutoQuantMoeModel(num_experts_attr)

    _, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0, "cost_model": "active_moe"},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
    )

    assert search_history["cost_model"] == "active_moe"
    assert search_history["active_moe_expert_ratio"] == pytest.approx(0.25)
    weighted_no_quant_cost = sum(
        stats["costs"][-1] for stats in search_history["candidate_stats"].values()
    )
    assert search_history["cost_denominator"] == pytest.approx(weighted_no_quant_cost)
    routed_stats = [
        stats
        for stats in search_history["candidate_stats"].values()
        if any("mlp.experts" in name for name in stats["module_names"])
    ]
    shared_stats = [
        stats
        for stats in search_history["candidate_stats"].values()
        if any("mlp.shared_expert" in name for name in stats["module_names"])
    ]
    assert routed_stats
    assert shared_stats
    assert all(stats["cost_weight"] == pytest.approx(0.25) for stats in routed_stats)
    assert all(stats["cost_weight"] == pytest.approx(1.0) for stats in shared_stats)
    assert all("active_costs" not in stats for stats in search_history["candidate_stats"].values())


def test_auto_quantize_hidden_recon_scores_grouped_parent_attention():
    model = _QwenStyleHybridAttentionModel()

    _, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        num_calib_steps=2,
        num_score_steps=2,
        method="hidden_recon",
        quant_grouping_scheme="runtime_fused+linear_attn_layer+self_attn_layer",
        score_component_tracking="batch",
    )

    assert search_history["method"] == "hidden_recon"
    assert search_history["scoring_signature"]["method"] == "hidden_recon"
    assert "loss_func" not in search_history["scoring_signature"] or (
        search_history["scoring_signature"]["loss_func"] is None
    )

    self_attn_stats = [
        stats
        for stats in search_history["candidate_stats"].values()
        if {"layers.0.self_attn.q_proj", "layers.0.self_attn.o_proj"}.issubset(
            set(stats["module_names"])
        )
    ]
    linear_attn_stats = [
        stats
        for stats in search_history["candidate_stats"].values()
        if {"layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.out_proj"}.issubset(
            set(stats["module_names"])
        )
    ]
    assert len(self_attn_stats) == 1
    assert len(linear_attn_stats) == 1

    for stats in [*self_attn_stats, *linear_attn_stats]:
        assert any(score > 0 for score in stats["scores"])
        assert "score_components" in stats

    score_module_names = {
        record["score_module_name"]
        for stats in [*self_attn_stats, *linear_attn_stats]
        for record in stats["score_components"]
    }
    assert "layers.0.self_attn" in score_module_names
    assert "layers.0.linear_attn" in score_module_names


def test_auto_quantize_hidden_recon_score_windows_record_components():
    model = _QwenStyleHybridAttentionModel()

    _, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        num_calib_steps=2,
        num_score_steps=2,
        method="hidden_recon",
        quant_grouping_scheme="runtime_fused+linear_attn_layer+self_attn_layer",
        score_component_tracking="batch",
        hidden_recon_score_windows=["full", "last:1"],
        hidden_recon_score_reduce="max",
    )

    assert search_history["scoring_signature"]["hidden_recon_score_windows"] == [
        "full",
        "last:1",
    ]
    assert search_history["scoring_signature"]["hidden_recon_score_reduce"] == "max"
    assert search_history["score_component_metadata"]["hidden_recon_score_windows"] == [
        "full",
        "last:1",
    ]
    assert search_history["score_component_metadata"]["hidden_recon_score_reduce"] == "max"

    score_windows = {
        record["score_window"]
        for stats in search_history["candidate_stats"].values()
        for record in stats.get("score_components", [])
    }
    assert {"full", "last:1", "aggregate"}.issubset(score_windows)


def test_auto_quantize_hidden_recon_disables_and_restores_use_cache():
    model = _CacheDefaultModel()

    mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        num_calib_steps=2,
        num_score_steps=2,
        method="hidden_recon",
    )

    assert model.config.use_cache is True
    assert False in model.use_cache_seen


def test_auto_quantize_gradient_disables_and_restores_use_cache():
    model = _CacheDefaultModel()

    mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        method="gradient",
    )

    assert model.config.use_cache is True
    assert False in model.use_cache_seen


def test_active_moe_ratio_requires_single_config_object():
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        num_experts_per_tok=2,
        text_config=SimpleNamespace(num_experts=8),
    )

    assert infer_active_moe_expert_ratio(model) is None


def test_active_moe_search_prefers_budget_lower_bound():
    searcher = AutoQuantizeGradientSearcher()
    searcher.config = {"cost_model": "active_moe"}
    searcher.cost_model = "active_moe"
    searcher.candidate_stats = {
        "layers.0.mlp.quant_recipe": {
            "formats": ["under_budget", "near_budget"],
            "costs": [1.0, 4.95],
            "scores": [0.0, 10.0],
        }
    }

    best_recipes, is_satisfied = searcher.run_search_with_stats(5.0)

    assert is_satisfied
    assert best_recipes["layers.0.mlp.quant_recipe"]["format"] == "near_budget"


def test_lps_normalizes_tiny_objective_coefficients():
    constraints = {"cost": (3.0, 3.0)}
    candidate_costs = {"cost": [[1.0, 2.0], [1.0, 2.0]]}
    tiny_scores = [[2.0e-16, 1.0e-16], [1.0e-16, 3.0e-16]]

    solver = LPS(
        name="tiny",
        constraints=constraints,
        constraints_to_candidate_costs=candidate_costs,
        candidate_scores=tiny_scores,
    )
    scaled_solver = LPS(
        name="scaled",
        constraints=constraints,
        constraints_to_candidate_costs=candidate_costs,
        candidate_scores=[[2.0, 1.0], [1.0, 3.0]],
    )

    max_normalized_score = max(
        abs(score) for layer_scores in solver.candidate_scores for score in layer_scores
    )
    assert max_normalized_score == pytest.approx(1.0)
    assert solver() == scaled_solver() == ([1, 0], "Optimal")


def test_lps_sanitizes_non_finite_objective_coefficients():
    solver = LPS(
        name="nonfinite",
        constraints={"cost": (2.0, 2.0)},
        constraints_to_candidate_costs={"cost": [[1.0, 1.0], [1.0, 1.0]]},
        candidate_scores=[[0.0, float("inf")], [float("nan"), 1.0]],
    )

    assert all(
        math.isfinite(score) for layer_scores in solver.candidate_scores for score in layer_scores
    )
    assert solver() == ([0, 1], "Optimal")


def test_lps_solves_top_k_with_no_good_cuts():
    solver = LPS(
        name="topk",
        constraints={"cost": (2.0, 2.0)},
        constraints_to_candidate_costs={"cost": [[1.0, 1.0], [1.0, 1.0]]},
        candidate_scores=[[0.0, 10.0], [1.0, 2.0]],
    )

    assert solver.solve_top_k(3) == [
        ([0, 0], "Optimal"),
        ([0, 1], "Optimal"),
        ([1, 0], "Optimal"),
    ]
    assert solver() == ([0, 0], "Optimal")
    with pytest.raises(ValueError, match="positive"):
        solver.solve_top_k(0)


def test_lps_prunes_single_effective_choice_layers_for_top_k():
    solver = LPS(
        name="topk_pruned",
        constraints={"cost": (8.0, 9.0)},
        constraints_to_candidate_costs={"cost": [[1.0, 2.0], [4.0, 4.0, 4.0], [3.0]]},
        candidate_scores=[[0.0, 10.0], [5.0, 5.0, 5.0], [1.0]],
    )

    assert solver._fixed_selection_and_variable_layers() == ([-1, 0, 0], [0])
    assert solver.solve_top_k(3) == [
        ([0, 0, 0], "Optimal"),
        ([1, 0, 0], "Optimal"),
    ]


def test_lps_all_fixed_layers_budget_status():
    feasible_solver = LPS(
        name="all_fixed_feasible",
        constraints={"cost": (2.0, 2.0)},
        constraints_to_candidate_costs={"cost": [[1.0, 1.0], [1.0]]},
        candidate_scores=[[0.0, 0.0], [1.0]],
    )
    infeasible_solver = LPS(
        name="all_fixed_infeasible",
        constraints={"cost": (3.0, 3.0)},
        constraints_to_candidate_costs={"cost": [[1.0, 1.0], [1.0]]},
        candidate_scores=[[0.0, 0.0], [1.0]],
    )

    assert feasible_solver.solve_top_k(4) == [([0, 0], "Optimal")]
    assert infeasible_solver.solve_top_k(4) == [([0, 0], "Infeasible")]


def test_auto_quantize_active_weighted_score_model():
    model = torch.nn.Linear(4, 16)
    constraints = normalize_auto_quantize_constraints(
        model, {"effective_bits": 4.8, "score_model": "active_weighted"}
    )
    assert constraints["score_model"] == "active_weighted"

    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = constraints

    assert searcher._candidate_scores_for_search(
        {"scores": [2.0, 4.0], "costs": [8.0, 4.0], "cost_weight": 0.25}
    ) == pytest.approx([0.5, 1.0])


def test_auto_quantize_active_weighted_score_model_restored_state_fallback():
    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = {"effective_bits": 4.8, "score_model": "active_weighted"}

    assert searcher._candidate_scores_for_search(
        {
            "scores": [2.0, 4.0],
            "costs": [2.0, 1.0],
            "element_costs": [8.0, 4.0],
        }
    ) == pytest.approx([0.5, 1.0])


def test_auto_quantize_per_element_score_model():
    model = torch.nn.Linear(4, 16)
    constraints = normalize_auto_quantize_constraints(
        model, {"effective_bits": 4.8, "score_model": "per_element"}
    )
    assert constraints["score_model"] == "per_element"

    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = constraints

    assert searcher._candidate_scores_for_search(
        {
            "scores": [2.0, 4.0],
            "costs": [2.0, 1.0],
            "element_costs": [8.0, 4.0],
            "cost_weight": 0.25,
        }
    ) == pytest.approx([0.25, 1.0])


def test_auto_quantize_per_element_score_model_restored_state_fallback():
    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = {"effective_bits": 4.8, "score_model": "per_element"}

    assert searcher._candidate_scores_for_search(
        {"scores": [2.0, 4.0], "costs": [2.0, 1.0], "cost_weight": 0.25}
    ) == pytest.approx([0.25, 1.0])


def test_auto_quantize_persists_objective_constraints_for_replay():
    model = SimpleLinear()

    _, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0, "score_model": "per_element"},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
    )

    assert search_state["best"]["constraints"]["score_model"] == "per_element"
    replay = mtq.get_auto_quantize_candidate_packets(search_state)
    assert replay["constraints"]["score_model"] == "per_element"


def test_auto_quantize_per_active_score_model():
    model = torch.nn.Linear(4, 16)
    constraints = normalize_auto_quantize_constraints(
        model, {"effective_bits": 4.8, "score_model": "per_active"}
    )
    assert constraints["score_model"] == "per_active"

    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = constraints

    assert searcher._candidate_scores_for_search(
        {"scores": [2.0, 4.0], "costs": [8.0, 4.0], "cost_weight": 0.25}
    ) == pytest.approx([0.25, 1.0])


def test_auto_quantize_response_risk_hparam_penalty():
    model = torch.nn.Linear(4, 16)
    constraints = normalize_auto_quantize_constraints(
        model,
        {
            "effective_bits": 4.8,
            "response_risk": {
                "entries": [
                    {
                        "hparam": "model.layers.0.linear_attn/layer.quant_recipe",
                        "format": "FP8",
                        "risk": "3.0",
                    }
                ],
            },
        },
    )

    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = constraints

    assert searcher._candidate_scores_for_search(
        {
            "formats": [QuantRecipe(None), QuantRecipe("FP8_DEFAULT_CFG")],
            "scores": [1.0, 1.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.0.linear_attn.in_proj_qkv"],
        },
        "model.layers.0.linear_attn/layer.quant_recipe",
    ) == pytest.approx([1.0, 4.0])


def test_auto_quantize_response_risk_category_layer_penalty_not_global():
    model = torch.nn.Linear(4, 16)
    constraints = normalize_auto_quantize_constraints(
        model,
        {
            "effective_bits": 4.8,
            "response_risk": {
                "entries": [
                    {"category": "linear_attn", "layer": "0", "format": "FP8", "risk": "2.0"},
                    {"category": "linear_attn", "layer": "1", "format": "FP8", "risk": "5.0"},
                ],
            },
        },
    )

    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = constraints

    assert searcher._candidate_scores_for_search(
        {
            "formats": [QuantRecipe(None), QuantRecipe("FP8_DEFAULT_CFG")],
            "scores": [0.0, 0.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.0.linear_attn.out_proj"],
        },
        "model.layers.0.linear_attn/layer.quant_recipe",
    ) == pytest.approx([0.0, 2.0])


def test_auto_quantize_response_risk_matches_root_level_layers():
    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = {
        "effective_bits": 4.8,
        "response_risk": {
            "entries": [{"category": "linear_attn", "layer": "0", "format": "FP8", "risk": "2.0"}]
        },
    }

    scores = searcher._candidate_scores_for_search(
        {
            "formats": [QuantRecipe(None), QuantRecipe("FP8_DEFAULT_CFG")],
            "scores": [0.0, 0.0],
            "costs": [2.0, 1.0],
            "module_names": ["layers.0.linear_attn.out_proj"],
        },
        "layers.0.linear_attn/layer.quant_recipe",
    )

    assert scores == pytest.approx([0.0, 2.0])


@pytest.mark.parametrize(
    ("legacy_category", "module_name", "hparam_name"),
    [
        (
            "linear_attn_layer",
            "model.layers.0.linear_attn.out_proj",
            "model.layers.0.linear_attn/layer.quant_recipe",
        ),
        (
            "self_attn_layer",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.self_attn/layer.quant_recipe",
        ),
    ],
)
def test_auto_quantize_response_risk_legacy_layer_category_alias(
    legacy_category, module_name, hparam_name
):
    model = torch.nn.Linear(4, 16)
    constraints = normalize_auto_quantize_constraints(
        model,
        {
            "effective_bits": 4.8,
            "response_risk": {
                "entries": [
                    {
                        "category": legacy_category,
                        "layer": "0",
                        "format": "W4A16_NVFP4",
                        "risk": "7.0",
                    }
                ],
            },
        },
    )

    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = constraints

    assert searcher._candidate_scores_for_search(
        {
            "formats": [
                QuantRecipe(None),
                QuantRecipe(
                    {"quant_cfg": [{"quantizer_name": "*", "enable": False}]},
                    name="W4A16_NVFP4",
                ),
            ],
            "scores": [0.0, 0.0],
            "costs": [2.0, 1.0],
            "module_names": [module_name],
        },
        hparam_name,
    ) == pytest.approx([0.0, 7.0])


def test_auto_quantize_response_risk_source_path_metric_filter(tmp_path):
    source = tmp_path / "risk.tsv"
    source.write_text(
        "category\tlayer\tformat\trisk\trisk_metric\n"
        "linear_attn\t0\tFP8\t2.0\taggregate_parser_on_response_risk\n"
        "linear_attn\t0\tFP8\t99.0\tother_metric\n"
        "linear_attn\t1\tFP8\t5.0\taggregate_parser_on_response_risk\n",
        encoding="utf-8",
    )
    model = torch.nn.Linear(4, 16)
    constraints = normalize_auto_quantize_constraints(
        model,
        {
            "effective_bits": 4.8,
            "response_risk": {
                "source_path": str(source),
                "risk_metric": "aggregate_parser_on_response_risk",
                "scale": 0.5,
            },
        },
    )

    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = constraints

    assert searcher._candidate_scores_for_search(
        {
            "formats": [QuantRecipe(None), QuantRecipe("FP8_DEFAULT_CFG")],
            "scores": [0.0, 0.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.0.linear_attn.in_proj_qkv"],
        },
        "model.layers.0.linear_attn/layer.quant_recipe",
    ) == pytest.approx([0.0, 1.0])


def test_auto_quantize_response_risk_source_path_loads_all_metrics_by_default(tmp_path):
    source = tmp_path / "risk.tsv"
    source.write_text(
        "category\tlayer\tformat\trisk\trisk_metric\n"
        "linear_attn\t0\tFP8\t2.0\tlcb_acceptance_margin_deficit\n"
        "linear_attn\t0\tFP8\t3.0\tresponse_health_margin_deficit\n",
        encoding="utf-8",
    )
    model = torch.nn.Linear(4, 16)
    constraints = normalize_auto_quantize_constraints(
        model,
        {
            "effective_bits": 4.8,
            "response_risk": {
                "source_path": str(source),
            },
        },
    )

    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = constraints

    assert searcher._candidate_scores_for_search(
        {
            "formats": [QuantRecipe(None), QuantRecipe("FP8_DEFAULT_CFG")],
            "scores": [0.0, 0.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.0.linear_attn.in_proj_qkv"],
        },
        "model.layers.0.linear_attn/layer.quant_recipe",
    ) == pytest.approx([0.0, 5.0])


def test_auto_quantize_response_risk_category_star_layer_is_global():
    model = torch.nn.Linear(4, 16)
    constraints = normalize_auto_quantize_constraints(
        model,
        {
            "effective_bits": 4.8,
            "response_risk": {
                "entries": [
                    {"category": "shared_expert", "layer": "*", "format": "FP8", "risk": "3.0"},
                    {"category": "shared_expert", "layer": "1", "format": "FP8", "risk": "5.0"},
                ],
            },
        },
    )

    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = constraints

    assert searcher._candidate_scores_for_search(
        {
            "formats": [QuantRecipe(None), QuantRecipe("FP8_DEFAULT_CFG")],
            "scores": [0.0, 0.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.0.mlp.shared_expert.gate_proj"],
        },
        "model.layers.0.mlp.shared_expert.quant_recipe",
    ) == pytest.approx([0.0, 3.0])


def test_auto_quantize_response_risk_changes_lps_selection():
    bf16 = QuantRecipe(None)
    fp8 = QuantRecipe("FP8_DEFAULT_CFG")
    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = {
        "effective_bits": 4.8,
        "cost_lower_bound": 0.99,
        "response_risk": {
            "entries": [
                {
                    "hparam": "model.layers.0.linear_attn/layer.quant_recipe",
                    "format": "FP8",
                    "risk": "10.0",
                }
            ],
        },
    }
    searcher.candidate_stats = {
        "model.layers.0.linear_attn/layer.quant_recipe": {
            "formats": [bf16, fp8],
            "scores": [0.0, 0.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.0.linear_attn.out_proj"],
        },
        "model.layers.1.linear_attn/layer.quant_recipe": {
            "formats": [bf16, fp8],
            "scores": [0.0, 0.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.1.linear_attn.out_proj"],
        },
    }

    best_recipes, is_satisfied = searcher.run_search_with_stats(3.0)

    assert is_satisfied
    assert best_recipes["model.layers.0.linear_attn/layer.quant_recipe"]["format"] == bf16
    assert best_recipes["model.layers.1.linear_attn/layer.quant_recipe"]["format"] == fp8


def test_auto_quantize_candidate_rerank_validation():
    model = torch.nn.Linear(4, 16)
    constraints = normalize_auto_quantize_constraints(
        model,
        {
            "effective_bits": 4.8,
            "candidate_rerank": {
                "enabled": True,
                "top_k": 4,
                "launch_authority_default": "no",
                "entries": [{"signature": "abc", "rerank_score": "-1.0"}],
                "provenance": "unit_test",
            },
        },
    )

    assert constraints["candidate_rerank"]["enabled"] is True
    assert constraints["candidate_rerank"]["top_k"] == 4
    assert constraints["candidate_rerank"]["launch_authority_default"] == "no"
    assert constraints["candidate_rerank"]["id_field"] == "signature"
    assert constraints["candidate_rerank"]["score_field"] == "rerank_score"
    assert constraints["candidate_rerank"]["scale"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="top_k"):
        normalize_auto_quantize_constraints(
            model, {"effective_bits": 4.8, "candidate_rerank": {"top_k": 0}}
        )


def test_auto_quantize_candidate_signature_keeps_exact_recipe_identity():
    signatures = {
        quant_algorithms._auto_quantize_candidate_signature(
            {"layer.quant_recipe": {"format": QuantRecipe(recipe)}}
        )
        for recipe in (
            "FP8_DEFAULT_CFG",
            "FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG",
            "W4A8_MXFP4_FP8_CFG",
        )
    }

    assert len(signatures) == 3
    assert (
        quant_algorithms._auto_quantize_recipe_name(QuantRecipe("W4A8_MXFP4_FP8_CFG"))
        == "W4A8_MXFP4_FP8"
    )


def test_auto_quantize_candidate_packets_do_not_change_best_recipe():
    bf16 = QuantRecipe(None)
    fp8 = QuantRecipe("FP8_DEFAULT_CFG")
    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = {
        "effective_bits": 4.8,
        "cost_lower_bound": 1.0,
        "candidate_rerank": {"enabled": True, "top_k": 2},
    }
    searcher.cost_denominator = 4.0
    searcher.candidate_stats = {
        "model.layers.0.linear_attn/layer.quant_recipe": {
            "formats": [bf16, fp8],
            "scores": [0.0, 1.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.0.linear_attn.out_proj"],
        },
        "model.layers.1.linear_attn/layer.quant_recipe": {
            "formats": [bf16, fp8],
            "scores": [0.0, 2.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.1.linear_attn.out_proj"],
        },
    }

    best_recipes, is_satisfied = searcher.run_search_with_stats(3.0)

    assert is_satisfied
    assert best_recipes["model.layers.0.linear_attn/layer.quant_recipe"]["format"] == fp8
    assert best_recipes["model.layers.1.linear_attn/layer.quant_recipe"]["format"] == bf16
    assert len(searcher.last_candidate_packets) == 2
    assert searcher.last_candidate_packets[0]["recipe_info"] == best_recipes
    assert searcher.last_candidate_packets[0]["effective_bits"] == pytest.approx(12.0)
    assert searcher.last_candidate_packets[0]["lp_packet_id"] == 0
    assert searcher.last_candidate_packets[0]["rerank_rank"] == 0
    assert searcher.last_candidate_packets[0]["rerank_score"] == pytest.approx(0.0)
    assert searcher.last_candidate_packets[0]["rerank_source_match"] is False
    assert searcher.last_candidate_packets[0]["rerank_objective_scores"] == pytest.approx(
        searcher.last_candidate_packets[0]["objective_scores"]
    )
    assert (
        searcher.last_candidate_packets[1]["recipe"][
            "model.layers.0.linear_attn/layer.quant_recipe"
        ]
        == bf16
    )
    assert searcher.last_candidate_packets[1]["lp_packet_id"] == 1
    assert searcher.last_candidate_packets[1]["rerank_rank"] == 1


def test_auto_quantize_candidate_packets_can_be_reranked_by_signature():
    bf16 = QuantRecipe(None)
    fp8 = QuantRecipe("FP8_DEFAULT_CFG")
    candidate_stats = {
        "model.layers.0.linear_attn/layer.quant_recipe": {
            "formats": [bf16, fp8],
            "scores": [0.0, 1.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.0.linear_attn.out_proj"],
        },
        "model.layers.1.linear_attn/layer.quant_recipe": {
            "formats": [bf16, fp8],
            "scores": [0.0, 2.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.1.linear_attn.out_proj"],
        },
    }

    baseline_searcher = AutoQuantizeGradientSearcher()
    baseline_searcher.constraints = {
        "effective_bits": 4.8,
        "cost_lower_bound": 1.0,
        "candidate_rerank": {"enabled": True, "top_k": 2},
    }
    baseline_searcher.cost_denominator = 4.0
    baseline_searcher.candidate_stats = candidate_stats
    baseline_recipes, is_satisfied = baseline_searcher.run_search_with_stats(3.0)
    assert is_satisfied

    second_packet_signature = baseline_searcher.last_candidate_packets[1]["signature"]
    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = {
        "effective_bits": 4.8,
        "cost_lower_bound": 1.0,
        "candidate_rerank": {
            "enabled": True,
            "top_k": 2,
            "entries": [{"signature": second_packet_signature, "rerank_score": "-2.0"}],
        },
    }
    searcher.cost_denominator = 4.0
    searcher.candidate_stats = candidate_stats

    best_recipes, is_satisfied = searcher.run_search_with_stats(3.0)

    assert is_satisfied
    assert best_recipes != baseline_recipes
    assert best_recipes["model.layers.0.linear_attn/layer.quant_recipe"]["format"] == bf16
    assert best_recipes["model.layers.1.linear_attn/layer.quant_recipe"]["format"] == fp8
    assert searcher.last_candidate_packets[0]["signature"] == second_packet_signature
    assert searcher.last_candidate_packets[0]["lp_packet_id"] == 1
    assert searcher.last_candidate_packets[0]["rerank_rank"] == 0
    assert searcher.last_candidate_packets[0]["rerank_score"] == pytest.approx(-2.0)
    assert searcher.last_candidate_packets[0]["rerank_source_match"] is True


def test_auto_quantize_candidate_packets_can_be_reranked_by_family_rule():
    bf16 = QuantRecipe(None)
    fp8 = QuantRecipe("FP8_DEFAULT_CFG")
    w4a16 = QuantRecipe(
        {"quant_cfg": [{"quantizer_name": "*", "enable": False}]},
        name="W4A16_NVFP4",
    )
    candidate_stats = {
        "model.layers.0.mlp.shared_expert.quant_recipe": {
            "formats": [w4a16, bf16],
            "scores": [0.0, 0.1],
            "costs": [1.0, 2.0],
            "module_names": ["model.layers.0.mlp.shared_expert.gate_proj"],
        },
        "model.layers.0.linear_attn/layer.quant_recipe": {
            "formats": [fp8],
            "scores": [0.0],
            "costs": [1.0],
            "module_names": ["model.layers.0.linear_attn.out_proj"],
        },
    }

    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = {
        "effective_bits": 12.0,
        "cost_lower_bound": 0.0,
        "candidate_rerank": {
            "enabled": True,
            "top_k": 2,
            "family_entries": [
                {
                    "family": "shared_expert",
                    "format": "W4A16_NVFP4",
                    "min_frac": "1.0",
                    "rerank_score": "1.0",
                }
            ],
        },
    }
    searcher.cost_denominator = 4.0
    searcher.candidate_stats = candidate_stats

    best_recipes, is_satisfied = searcher.run_search_with_stats(3.0)

    assert is_satisfied
    assert best_recipes["model.layers.0.mlp.shared_expert.quant_recipe"]["format"] == bf16
    assert (
        searcher.last_candidate_packets[0]["recipe"][
            "model.layers.0.mlp.shared_expert.quant_recipe"
        ]
        == bf16
    )
    assert searcher.last_candidate_packets[0]["rerank_signature_score"] == pytest.approx(0.0)
    assert searcher.last_candidate_packets[0]["rerank_family_score"] == pytest.approx(0.0)
    assert searcher.last_candidate_packets[0]["rerank_family_match"] is False
    assert (
        searcher.last_candidate_packets[1]["recipe"][
            "model.layers.0.mlp.shared_expert.quant_recipe"
        ]
        == w4a16
    )
    assert searcher.last_candidate_packets[1]["rerank_family_score"] == pytest.approx(1.0)
    assert searcher.last_candidate_packets[1]["rerank_family_match"] is True


def test_auto_quantize_candidate_family_rerank_can_use_hparam_counts():
    packet = {
        "family_format_counts": {
            "shared_expert": {
                "BF16": 1,
                "W4A16_NVFP4": 3,
            }
        },
        "hparam_family_format_counts": {
            "shared_expert": {
                "BF16": 1,
                "W4A16_NVFP4": 1,
            }
        },
    }
    module_rule = {
        "family": "shared_expert",
        "format": "W4A16_NVFP4",
        "score": 1.0,
        "mode": "packet",
        "count_level": "module",
        "min_frac": "0.70",
        "min_count": None,
        "max_count": None,
        "max_frac": None,
    }
    hparam_rule = {
        **module_rule,
        "count_level": "hparam",
    }

    assert AutoQuantizeGradientSearcher._candidate_family_rerank_score(
        packet, [module_rule]
    ) == pytest.approx(1.0)
    assert AutoQuantizeGradientSearcher._candidate_family_rerank_score(
        packet, [hparam_rule]
    ) == pytest.approx(0.0)


def test_auto_quantize_returns_candidate_packets_in_search_history():
    model = _QwenStyleHybridAttentionModel()

    _, search_history = mtq.auto_quantize(
        model,
        constraints={
            "effective_bits": 6.0,
            "candidate_rerank": {
                "enabled": True,
                "top_k": 2,
                "launch_authority_default": "no",
            },
        },
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        quant_grouping_scheme="runtime_fused+linear_attn_layer+self_attn_layer",
    )

    candidate_packets = search_history["best"]["candidate_packets"]

    assert len(candidate_packets) == 2
    assert candidate_packets[0]["recipe"] == search_history["best"]["recipe"]
    assert candidate_packets[0]["launch_authority"] == "no"
    assert candidate_packets[0]["lp_packet_id"] == 0
    assert candidate_packets[0]["rerank_rank"] == 0
    assert candidate_packets[0]["rerank_source_match"] is False
    assert candidate_packets[0]["rerank_score"] == pytest.approx(0.0)
    assert candidate_packets[0]["rerank_objective_scores"] == pytest.approx(
        candidate_packets[0]["objective_scores"]
    )
    assert candidate_packets[0]["signature"]
    assert isinstance(candidate_packets[0]["recipe_info"], dict)
    assert candidate_packets[0]["effective_bits"] is not None


def test_resolve_best_recipe_applies_candidate_rerank_from_restored_state():
    bf16 = QuantRecipe(None)
    fp8 = QuantRecipe("FP8_DEFAULT_CFG")
    candidate_stats = {
        "model.layers.0.linear_attn/layer.quant_recipe": {
            "formats": [bf16, fp8],
            "scores": [0.0, 1.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.0.linear_attn.out_proj"],
        },
        "model.layers.1.linear_attn/layer.quant_recipe": {
            "formats": [bf16, fp8],
            "scores": [0.0, 2.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.1.linear_attn.out_proj"],
        },
    }

    baseline_searcher = AutoQuantizeGradientSearcher()
    baseline_searcher.constraints = {
        "effective_bits": 12.0,
        "cost_lower_bound": 1.0,
        "candidate_rerank": {"enabled": True, "top_k": 2},
    }
    baseline_searcher.cost_denominator = 4.0
    baseline_searcher.candidate_stats = candidate_stats
    baseline_recipes, is_satisfied = baseline_searcher.run_search_with_stats(3.0)
    assert is_satisfied

    second_packet_signature = baseline_searcher.last_candidate_packets[1]["signature"]
    restored_recipe = quant_algorithms._resolve_best_recipe(
        {
            "method": "gradient",
            "candidate_stats": candidate_stats,
            "cost_denominator": 4.0,
            "cost_model": "weight",
            "cost": {},
        },
        {
            "effective_bits": 12.0,
            "cost_lower_bound": 1.0,
            "candidate_rerank": {
                "enabled": True,
                "top_k": 2,
                "entries": [{"signature": second_packet_signature, "rerank_score": "-2.0"}],
            },
        },
    )

    assert restored_recipe != {name: info["format"] for name, info in baseline_recipes.items()}
    assert restored_recipe["model.layers.0.linear_attn/layer.quant_recipe"] == bf16
    assert restored_recipe["model.layers.1.linear_attn/layer.quant_recipe"] == fp8


def test_auto_quantize_explicit_cost_lower_bound():
    model = torch.nn.Linear(4, 16)
    constraints = normalize_auto_quantize_constraints(
        model, {"effective_bits": 4.8, "cost_lower_bound": 0.99}
    )

    assert constraints["cost_lower_bound"] == pytest.approx(0.99)

    searcher = AutoQuantizeGradientSearcher()
    searcher.constraints = constraints
    assert searcher._get_search_lower_bounds() == [pytest.approx(0.99)]

    with pytest.raises(ValueError, match="cost_lower_bound"):
        normalize_auto_quantize_constraints(model, {"effective_bits": 4.8, "cost_lower_bound": 1.1})


def test_auto_quantize_kl_div_rejects_cost_lower_bound():
    searcher = AutoQuantizeKLDivSearcher()
    searcher.constraints = {"effective_bits": 16.0, "cost_lower_bound": 0.9}
    searcher.candidate_stats = {}

    with pytest.raises(ValueError, match="cost_lower_bound is not supported"):
        searcher.run_search_with_stats(1.0)


def test_auto_quantize_kl_div_rejects_candidate_rerank():
    searcher = AutoQuantizeKLDivSearcher()
    searcher.constraints = {
        "effective_bits": 8.0,
        "candidate_rerank": {"enabled": True, "top_k": 2},
    }
    searcher.candidate_stats = {}

    with pytest.raises(ValueError, match="candidate_rerank is not supported"):
        searcher.run_search_with_stats(1.0)


@pytest.mark.parametrize(
    ("unsupported_constraint", "message"),
    [
        ({"cost_lower_bound": 0.9}, "cost_lower_bound is not supported"),
        (
            {"candidate_rerank": {"enabled": True, "top_k": 2}},
            "candidate_rerank is not supported",
        ),
    ],
)
def test_auto_quantize_kl_div_rejects_unsupported_controls_early(unsupported_constraint, message):
    with pytest.raises(ValueError, match=message):
        mtq.auto_quantize(
            torch.nn.Linear(4, 4),
            constraints={"effective_bits": 8.0, **unsupported_constraint},
            quantization_formats=[mtq.INT8_DEFAULT_CFG],
            method="kl_div",
        )


# use this config to test custom quantization config
INT8_CUSTOM_QUANT_TEST_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": "smoothquant",
}


@pytest.mark.parametrize(
    "model_cls",
    [SimpleConv, SimpleConvLinear, SimpleLinear, TransformerBlock],
)
@pytest.mark.parametrize(
    ("search_formats", "min_bits", "search_bits"),
    [
        ([mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG], 4.0, 6.0),
        ([mtq.INT4_AWQ_CFG, mtq.INT8_SMOOTHQUANT_CFG], 4.0, 6.0),
        ([mtq.INT4_AWQ_CFG, INT8_CUSTOM_QUANT_TEST_CFG], 4.0, 6.0),
        ([mtq.INT8_SMOOTHQUANT_CFG], 8.0, 11.0),
        ([None, mtq.INT8_SMOOTHQUANT_CFG], 8.0, 11.0),
    ],
)
@pytest.mark.parametrize(
    "method",
    ["gradient", "kl_div"],
)
def test_auto_quantize(model_cls, search_formats, min_bits, search_bits, method):
    model = model_cls()

    def loss_func(output):
        return output.sum()

    best_model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": search_bits},
        quantization_formats=search_formats,
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
        method=method,
    )
    assert isinstance(search_history, dict)
    assert search_history["best"]["is_satisfied"]
    effective_bits_from_search = search_history["best"]["constraints"]["effective_bits"]
    assert effective_bits_from_search <= search_bits and effective_bits_from_search >= min_bits, (
        "Search failed!"
    )

    if model_cls == TransformerBlock:
        hparam = model.attn.q_proj.get_hparam("quant_recipe")
        for layer in [model.attn.k_proj, model.attn.v_proj]:
            assert layer.get_hparam("quant_recipe") == hparam
        assert ("attn.q_proj.quant_recipe" in search_history["candidate_stats"]) != (
            "attn.k_proj.quant_recipe" in search_history["candidate_stats"]
        )

    # test restore
    buffer = io.BytesIO()
    mto.save(best_model, buffer)
    buffer.seek(0)
    new_model = model_cls()
    new_model = mto.restore(new_model, buffer)

    input = model.get_input()
    output_ref = best_model(input)
    output_test = new_model(input)
    assert torch.allclose(output_ref, output_test)


def test_auto_quantize_disable_layers():
    model = TransformerBlock()

    def loss_func(output):
        return output.sum()

    best_model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 5.0},
        quantization_formats=[
            mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            mtq.INT8_DEFAULT_CFG,
        ],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        disabled_layers=["*mlp*"],
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
    )

    assert not best_model.mlp.input_quantizer.is_enabled


def test_auto_quantize_disabled_layers_no_poison():
    """disabled_layers must only affect the matched layers, not all subsequent layer groups."""
    model = TransformerBlock()

    best_model, _ = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 5.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        disabled_layers=["*mlp*"],
        num_calib_steps=2,
        num_score_steps=2,
    )

    assert not best_model.mlp.input_quantizer.is_enabled
    hparam = best_model.attn.q_proj.get_hparam("quant_recipe")
    assert QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG) in hparam.choices


def test_auto_quantize_groups_qwen_shared_expert_projection_family():
    model = _QwenStyleSharedExpertModel()

    model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
    )

    shared_expert = model.mlp.shared_expert
    hparam = shared_expert.gate_proj.get_hparam("quant_recipe")

    assert shared_expert.up_proj.get_hparam("quant_recipe") is hparam
    assert shared_expert.down_proj.get_hparam("quant_recipe") is hparam
    assert set(hparam.quant_module_names) == {
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "mlp.shared_expert.down_proj",
    }
    assert set(hparam.score_modules) == {model.mlp}
    assert (
        len(
            [
                stats
                for stats in search_history["candidate_stats"].values()
                if set(stats["module_names"])
                == {
                    "mlp.shared_expert.gate_proj",
                    "mlp.shared_expert.up_proj",
                    "mlp.shared_expert.down_proj",
                }
            ]
        )
        == 1
    )


def test_auto_quantize_scores_fused_routed_experts_at_parent_mlp():
    model = _QwenStyleFusedExpertsModel()
    searcher = AutoQuantizeGradientSearcher()
    searcher.model = model
    searcher.config = {"cost": {}}
    searcher._cost_model = get_auto_quantize_cost_model("weight")

    recipes = [
        QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG),
        QuantRecipe(mtq.INT8_DEFAULT_CFG),
        QuantRecipe(None),
    ]
    searcher.insert_hparams_after_merge_rules(model, recipes)

    hparam = model.mlp.experts.get_hparam("quant_recipe")

    assert set(hparam.quant_modules) == {model.mlp.experts}
    assert set(hparam.quant_module_names) == {"mlp.experts"}
    assert set(hparam.score_modules) == {model.mlp}


def test_get_auto_quantize_config_replays_fused_expert_quantizers():
    model = _QwenStyleFusedExpertsModel()
    searcher = AutoQuantizeGradientSearcher()
    searcher.model = model
    searcher.config = {"cost": {}}
    searcher._cost_model = get_auto_quantize_cost_model("weight")
    searcher.candidate_stats = {}
    searcher._score_component_records_by_hparam = {}

    fp8_recipe = QuantRecipe(mtq.FP8_DEFAULT_CFG)
    searcher.insert_hparams_after_merge_rules(model, [fp8_recipe])
    searcher.initialize_candidate_stats()

    hparam_name, candidate_stat = next(iter(searcher.candidate_stats.items()))
    assert candidate_stat["quantizer_attrs"]["mlp.experts"] == [
        "gate_up_proj_input_quantizer",
        "gate_up_proj_weight_quantizer",
        "down_proj_input_quantizer",
        "down_proj_weight_quantizer",
    ]
    search_state = {
        "best": {"recipe": {hparam_name: fp8_recipe}},
        "candidate_stats": searcher.candidate_stats,
    }

    config = mtq.get_auto_quantize_config(search_state)
    fresh_model = _QwenStyleFusedExpertsModel()
    set_quantizer_by_cfg(fresh_model, config["quant_cfg"])

    experts = fresh_model.mlp.experts
    assert experts.gate_up_proj_input_quantizer.is_enabled
    assert experts.down_proj_input_quantizer.is_enabled
    assert all(quantizer.is_enabled for quantizer in experts.gate_up_proj_weight_quantizers)
    assert all(quantizer.is_enabled for quantizer in experts.down_proj_weight_quantizers)


def test_auto_quantize_scores_attention_projection_groups_at_attention_output():
    model = _QwenStyleHybridAttentionModel()

    model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
    )

    self_attn = model.layers[0]["self_attn"]
    qkv_hparam = self_attn.q_proj.get_hparam("quant_recipe")
    o_hparam = self_attn.o_proj.get_hparam("quant_recipe")

    assert self_attn.k_proj.get_hparam("quant_recipe") is qkv_hparam
    assert self_attn.v_proj.get_hparam("quant_recipe") is qkv_hparam
    assert o_hparam is not qkv_hparam
    assert set(qkv_hparam.score_modules) == {self_attn}
    assert set(o_hparam.score_modules) == {self_attn}
    assert set(qkv_hparam.quant_module_names) == {
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
    }
    assert set(o_hparam.quant_module_names) == {"layers.0.self_attn.o_proj"}

    linear_attn = model.layers[0]["linear_attn"]
    qkvz_hparam = linear_attn.in_proj_qkv.get_hparam("quant_recipe")
    ba_hparam = linear_attn.in_proj_a.get_hparam("quant_recipe")
    out_hparam = linear_attn.out_proj.get_hparam("quant_recipe")

    assert linear_attn.in_proj_z.get_hparam("quant_recipe") is qkvz_hparam
    assert linear_attn.in_proj_b.get_hparam("quant_recipe") is ba_hparam
    assert out_hparam is not qkvz_hparam
    assert out_hparam is not ba_hparam
    assert set(qkvz_hparam.score_modules) == {linear_attn}
    assert set(ba_hparam.score_modules) == {linear_attn}
    assert set(out_hparam.score_modules) == {linear_attn}
    assert set(qkvz_hparam.quant_module_names) == {
        "layers.0.linear_attn.in_proj_qkv",
        "layers.0.linear_attn.in_proj_z",
    }
    assert set(ba_hparam.quant_module_names) == {
        "layers.0.linear_attn.in_proj_a",
        "layers.0.linear_attn.in_proj_b",
    }
    assert set(out_hparam.quant_module_names) == {"layers.0.linear_attn.out_proj"}

    grouped_names = [
        set(stats["module_names"]) for stats in search_history["candidate_stats"].values()
    ]
    assert {
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
    } in grouped_names
    assert {
        "layers.0.linear_attn.in_proj_qkv",
        "layers.0.linear_attn.in_proj_z",
    } in grouped_names


def test_auto_quantize_runtime_fused_linear_self_attention_layer_grouping():
    model = _QwenStyleHybridAttentionModel()

    model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        quant_grouping_scheme="runtime_fused+linear_attn_layer+self_attn_layer",
    )

    self_attn = model.layers[0]["self_attn"]
    self_hparam = self_attn.q_proj.get_hparam("quant_recipe")
    assert self_attn.k_proj.get_hparam("quant_recipe") is self_hparam
    assert self_attn.v_proj.get_hparam("quant_recipe") is self_hparam
    assert self_attn.o_proj.get_hparam("quant_recipe") is self_hparam
    assert set(self_hparam.quant_module_names) == {
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.self_attn.o_proj",
    }
    assert set(self_hparam.score_modules) == {self_attn}

    linear_attn = model.layers[0]["linear_attn"]
    linear_hparam = linear_attn.in_proj_qkv.get_hparam("quant_recipe")
    ba_hparam = linear_attn.in_proj_a.get_hparam("quant_recipe")
    assert linear_attn.in_proj_z.get_hparam("quant_recipe") is linear_hparam
    assert linear_attn.out_proj.get_hparam("quant_recipe") is linear_hparam
    assert linear_attn.in_proj_b.get_hparam("quant_recipe") is ba_hparam
    assert ba_hparam is not linear_hparam
    assert set(linear_hparam.quant_module_names) == {
        "layers.0.linear_attn.in_proj_qkv",
        "layers.0.linear_attn.in_proj_z",
        "layers.0.linear_attn.out_proj",
    }
    assert set(ba_hparam.quant_module_names) == {
        "layers.0.linear_attn.in_proj_a",
        "layers.0.linear_attn.in_proj_b",
    }
    assert set(linear_hparam.score_modules) == {linear_attn}

    grouped_names = [
        set(stats["module_names"]) for stats in search_history["candidate_stats"].values()
    ]
    assert set(self_hparam.quant_module_names) in grouped_names
    assert set(linear_hparam.quant_module_names) in grouped_names


def test_auto_quantize_linear_attention_layer_grouping_keeps_disabled_ab_separate():
    model = _QwenStyleHybridAttentionModel()

    mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        disabled_layers=["*linear_attn.in_proj_a*", "*linear_attn.in_proj_b*"],
        quant_grouping_scheme="runtime_fused+linear_attn_layer",
    )

    linear_attn = model.layers[0]["linear_attn"]
    linear_hparam = linear_attn.in_proj_qkv.get_hparam("quant_recipe")
    ba_hparam = linear_attn.in_proj_a.get_hparam("quant_recipe")

    assert linear_attn.in_proj_z.get_hparam("quant_recipe") is linear_hparam
    assert linear_attn.out_proj.get_hparam("quant_recipe") is linear_hparam
    assert linear_attn.in_proj_b.get_hparam("quant_recipe") is ba_hparam
    assert ba_hparam is not linear_hparam
    assert len(linear_hparam.choices) > 1
    assert len(ba_hparam.choices) == 1


def test_auto_quantize_rejects_unknown_quant_grouping_scheme():
    model = _QwenStyleHybridAttentionModel()

    with pytest.raises(ValueError, match="quant_grouping_scheme"):
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 6.0},
            quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
            data_loader=[model.get_input() for _ in range(2)],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=2,
            num_score_steps=2,
            quant_grouping_scheme="attention_no_w4",
        )


def test_auto_quantize_score_modules_support_keyword_forward():
    model = _QwenStyleKeywordHybridAttentionModel()

    model, _ = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
    )

    layer = model.layers[0]
    assert set(layer["self_attn"].q_proj.get_hparam("quant_recipe").score_modules) == {
        layer["self_attn"]
    }
    assert set(layer["linear_attn"].in_proj_qkv.get_hparam("quant_recipe").score_modules) == {
        layer["linear_attn"]
    }


def test_auto_quantize_score_component_tracking_records_score_batches():
    model = SimpleLinear()
    data_loader = [
        {"input": model.get_input(), "source_id": "code"},
        {"input": model.get_input(), "source_id": "math"},
    ]

    _, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=data_loader,
        forward_step=lambda model, batch: model(batch["input"]),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        score_component_tracking="batch",
    )

    assert search_history["score_component_tracking"] == "batch"
    assert search_history["score_component_metadata"]["mode"] == "batch"
    component_stats = [
        stats
        for stats in search_history["candidate_stats"].values()
        if stats.get("score_components")
    ]
    assert component_stats

    records = component_stats[0]["score_components"]
    assert {record["batch_index"] for record in records} == {0, 1}
    assert {record["source_id"] for record in records} == {"code", "math"}
    assert all(record["component_id"].startswith("source_id:") for record in records)
    assert all(record["score_module_name"] for record in records)
    assert all(record["format"] != "BF16" for record in records)
    assert all(isinstance(record["score"], float) for record in records)


INT4INT8_AWQ_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": [
                {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
                {"num_bits": 8, "axis": None},
            ],
            "enable": True,
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {"num_bits": 8, "axis": None},
            "enable": True,
        },
    ],
    "algorithm": "awq_lite",
}


@pytest.mark.parametrize("config", [mtq.INT4_AWQ_CFG, mtq.INT8_SMOOTHQUANT_CFG, INT4INT8_AWQ_CFG])
def test_pqs_folding(config):
    model_ref = SimpleLinear()
    state_dict_ref = copy.deepcopy(model_ref.state_dict())
    inputs = model_ref.get_input()
    mtq.quantize(model_ref, config, lambda model: model(inputs))

    model_test = SimpleLinear()
    model_test.load_state_dict(state_dict_ref)
    QuantRecipe.disable_folding_pqs_to_weights()
    mtq.quantize(model_test, config, lambda model: model(inputs))

    assert torch.allclose(model_ref(inputs), model_test(inputs))

    QuantRecipe.fold_pqs_to_weights(model_test)
    assert torch.allclose(model_ref(inputs), model_test(inputs))


def _test_data_parallel_auto_quantize(rank, size):
    model = SimpleLinear()

    model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 11.0},
        quantization_formats=[mtq.INT8_SMOOTHQUANT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
    )

    search_history_rank0 = DistributedProcessGroup.get_dist_syncd_obj(
        search_history if rank == 0 else None,
        DistributedProcessGroup(None),
        lambda a: a[0],
    )

    # quantizer_states contains tensors which can't be compared with ==
    sh = {k: v for k, v in search_history.items() if k != "quantizer_states"}
    sh0 = {k: v for k, v in search_history_rank0.items() if k != "quantizer_states"}

    # Assert that the costs, scores and searched recipes are the same across all ranks
    assert sh == sh0

    assert search_history["best"]["is_satisfied"]


def test_data_parallel_auto_quantize(skip_on_windows):
    spawn_multiprocess_job(4, _test_data_parallel_auto_quantize, backend="gloo")


def test_estimate_quant_compression():
    nvfp4_affine_kv_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_AFFINE_KV_CFG)
    assert estimate_quant_compression(nvfp4_affine_kv_cfg) == 0.25

    nvfp4_awq_clip_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_AWQ_CLIP_CFG)
    assert estimate_quant_compression(nvfp4_awq_clip_cfg) == 0.25

    nvfp4_awq_full_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_AWQ_FULL_CFG)
    assert estimate_quant_compression(nvfp4_awq_full_cfg) == 0.25

    nvfp4_awq_lite_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_AWQ_LITE_CFG)
    assert estimate_quant_compression(nvfp4_awq_lite_cfg) == 0.25

    nvfp4_default_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_DEFAULT_CFG)
    assert estimate_quant_compression(nvfp4_default_cfg) == 0.25

    nvfp4_kv_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_KV_CFG)
    assert estimate_quant_compression(nvfp4_kv_cfg) == 0.25

    nvfp4_kv_rotate_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_KV_ROTATE_CFG)
    assert estimate_quant_compression(nvfp4_kv_rotate_cfg) == 0.25

    nvfp4_svdquant_default_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_SVDQUANT_DEFAULT_CFG)
    assert estimate_quant_compression(nvfp4_svdquant_default_cfg) == 0.25

    int8_default_cfg = mtq.config.QuantizeConfig(**mtq.INT8_DEFAULT_CFG)
    assert estimate_quant_compression(int8_default_cfg) == 0.5

    int8_smoothquant_cfg = mtq.config.QuantizeConfig(**mtq.INT8_SMOOTHQUANT_CFG)
    assert estimate_quant_compression(int8_smoothquant_cfg) == 0.5

    fp8_default_cfg = mtq.config.QuantizeConfig(**mtq.FP8_DEFAULT_CFG)
    assert estimate_quant_compression(fp8_default_cfg) == 0.5

    fp8_per_channel_per_token_cfg = mtq.config.QuantizeConfig(**mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG)
    assert estimate_quant_compression(fp8_per_channel_per_token_cfg) == 0.5

    fp8_2d_blockwise_weight_only_cfg = mtq.config.QuantizeConfig(
        **mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG
    )
    assert estimate_quant_compression(fp8_2d_blockwise_weight_only_cfg) == 0.5

    int4_blockwise_weight_only_cfg = mtq.config.QuantizeConfig(**mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    assert estimate_quant_compression(int4_blockwise_weight_only_cfg) == 0.25

    int4_awq_cfg = mtq.config.QuantizeConfig(**mtq.INT4_AWQ_CFG)
    assert estimate_quant_compression(int4_awq_cfg) == 0.25

    w4a8_awq_beta_cfg = mtq.config.QuantizeConfig(**mtq.W4A8_AWQ_BETA_CFG)
    assert estimate_quant_compression(w4a8_awq_beta_cfg) == 0.25

    mxfp8_default_cfg = mtq.config.QuantizeConfig(**mtq.MXFP8_DEFAULT_CFG)
    assert estimate_quant_compression(mxfp8_default_cfg) == 0.5

    mxfp6_default_cfg = mtq.config.QuantizeConfig(**mtq.MXFP6_DEFAULT_CFG)
    assert estimate_quant_compression(mxfp6_default_cfg) == 0.375

    mxfp4_default_cfg = mtq.config.QuantizeConfig(**mtq.MXFP4_DEFAULT_CFG)
    assert estimate_quant_compression(mxfp4_default_cfg) == 0.25

    mxint8_default_cfg = mtq.config.QuantizeConfig(**mtq.MXINT8_DEFAULT_CFG)
    assert estimate_quant_compression(mxint8_default_cfg) == 0.5

    fp8_kv_cfg = mtq.config.QuantizeConfig(**mtq.FP8_KV_CFG)
    assert estimate_quant_compression(fp8_kv_cfg) == 0.5

    fp8_affine_kv_cfg = mtq.config.QuantizeConfig(**mtq.FP8_AFFINE_KV_CFG)
    assert estimate_quant_compression(fp8_affine_kv_cfg) == 0.5


@pytest.mark.parametrize("method", ["gradient", "kl_div"])
def test_auto_quantize_checkpoint_resume(method, tmp_path, capsys):
    """Test that checkpoint can be used to resume an interrupted search."""
    model = SimpleLinear()
    checkpoint_path = str(tmp_path / "autoquant_resume_checkpoint.pth")

    # First run: save checkpoint
    model_1, state_dict_1 = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
        method=method,
        checkpoint=checkpoint_path,
    )

    # Clear captured output from first run
    capsys.readouterr()

    # Second run: resume with same constraint should produce same results
    model_2 = SimpleLinear()
    model_2, state_dict_2 = mtq.auto_quantize(
        model_2,
        constraints={"effective_bits": 6.0},  # Same constraint
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model_2.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
        method=method,
        checkpoint=checkpoint_path,
    )

    # Verify the restore message was printed on second run
    captured = capsys.readouterr()
    assert "Restored from checkpoint, skipping scoring" in captured.out, (
        "Expected restore message when resuming from checkpoint"
    )

    # Verify method is correctly persisted in checkpoint and state dicts
    saved = safe_load(checkpoint_path)
    assert saved["method"] == method
    assert saved["scoring_signature"] == state_dict_1["scoring_signature"]
    assert saved["scoring_signature"] == state_dict_2["scoring_signature"]
    assert saved["scoring_signature"]["num_score_steps"] == 2
    assert saved["cost_denominator"] is not None
    assert saved["best"]["recipe"] == state_dict_2["best"]["recipe"]
    assert (
        pytest.approx(saved["best"]["constraints"]["effective_bits"])
        == state_dict_2["best"]["constraints"]["effective_bits"]
    )
    assert state_dict_1["method"] == method
    assert state_dict_2["method"] == method

    # Results should be identical when using same constraint
    assert state_dict_1["candidate_stats"] == state_dict_2["candidate_stats"]
    assert state_dict_1["best"]["recipe"] == state_dict_2["best"]["recipe"]
    assert (
        pytest.approx(state_dict_1["best"]["constraints"]["effective_bits"])
        == state_dict_2["best"]["constraints"]["effective_bits"]
    )

    # Verify calibration was also restored from checkpoint
    assert "Restored calibration for" in captured.out

    # Verify quantizer_states is saved in checkpoint
    assert "quantizer_states" in saved
    assert len(saved["quantizer_states"]) > 0
    for recipe_state in saved["quantizer_states"].values():
        assert "metadata" in recipe_state
        assert "state_dict" in recipe_state

    # Verify resumed model produces identical quantizer_states
    assert state_dict_1["quantizer_states"].keys() == state_dict_2["quantizer_states"].keys()
    for recipe in state_dict_1["quantizer_states"]:
        s1 = state_dict_1["quantizer_states"][recipe]
        s2 = state_dict_2["quantizer_states"][recipe]
        # Verify metadata (quantizer properties + tensor shape/dtype info) match per quantizer
        assert s1["metadata"].keys() == s2["metadata"].keys()
        for qname in s1["metadata"]:
            assert s1["metadata"][qname] == s2["metadata"][qname], (
                f"Metadata mismatch for {qname} in {recipe}"
            )
        # Verify actual tensor values match per quantizer
        assert s1["state_dict"].keys() == s2["state_dict"].keys()
        for qname in s1["state_dict"]:
            for buf_name in s1["state_dict"][qname]:
                torch.testing.assert_close(
                    s1["state_dict"][qname][buf_name], s2["state_dict"][qname][buf_name]
                )


def test_auto_quantize_checkpoint_resume_clears_stale_objective_metadata(tmp_path):
    checkpoint_path = str(tmp_path / "autoquant_stale_metadata_checkpoint.pth")
    model_1 = SimpleLinear()
    _, first_state = mtq.auto_quantize(
        model_1,
        constraints={
            "effective_bits": 6.0,
            "response_risk": {"entries": []},
            "candidate_rerank": {"enabled": True, "top_k": 2},
        },
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model_1.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        checkpoint=checkpoint_path,
    )

    assert "candidate_packets" in first_state["best"]
    assert "response_risk_source" in first_state["best"]
    assert "candidate_rerank_source" in first_state["best"]

    model_2 = SimpleLinear()
    _, resumed_state = mtq.auto_quantize(
        model_2,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model_2.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        checkpoint=checkpoint_path,
    )

    assert "candidate_packets" not in resumed_state["best"]
    assert "response_risk_source" not in resumed_state["best"]
    assert "candidate_rerank_source" not in resumed_state["best"]
    assert resumed_state["best"]["constraints"]["score_model"] == "raw"


def test_auto_quantize_checkpoint_rejects_grouping_scheme_mismatch(tmp_path):
    """Test that restored candidate stats are not reused with a different grouping scheme."""
    checkpoint_path = str(tmp_path / "autoquant_grouping_checkpoint.pth")

    model_1 = _QwenStyleHybridAttentionModel()
    mtq.auto_quantize(
        model_1,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model_1.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        checkpoint=checkpoint_path,
        quant_grouping_scheme="runtime_fused",
    )

    saved = safe_load(checkpoint_path)
    assert saved["quant_grouping_scheme"] == "runtime_fused"
    assert saved["candidate_stats"]

    model_2 = _QwenStyleHybridAttentionModel()
    with pytest.raises(ValueError, match="grouping scheme"):
        mtq.auto_quantize(
            model_2,
            constraints={"effective_bits": 6.0},
            quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
            data_loader=[model_2.get_input() for _ in range(2)],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=2,
            num_score_steps=2,
            checkpoint=checkpoint_path,
            quant_grouping_scheme="runtime_fused+linear_attn_layer+self_attn_layer",
        )


def test_auto_quantize_checkpoint_rejects_scoring_signature_mismatch(tmp_path):
    """Test that restored candidate stats are not reused with a changed scoring setup."""
    checkpoint_path = str(tmp_path / "autoquant_scoring_signature_checkpoint.pth")

    model_1 = SimpleLinear()
    mtq.auto_quantize(
        model_1,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model_1.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=1,
        checkpoint=checkpoint_path,
    )

    saved = safe_load(checkpoint_path)
    assert saved["scoring_signature"]["num_score_steps"] == 1
    assert saved["candidate_stats"]

    model_2 = SimpleLinear()
    with pytest.raises(ValueError, match="scoring signature.*num_score_steps"):
        mtq.auto_quantize(
            model_2,
            constraints={"effective_bits": 6.0},
            quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
            data_loader=[model_2.get_input() for _ in range(2)],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=2,
            num_score_steps=2,
            checkpoint=checkpoint_path,
        )


def test_auto_quantize_checkpoint_accepts_legacy_signature_without_score_component_tracking(
    tmp_path,
):
    """Default tracking='none' should remain compatible with pre-tracking signatures."""
    checkpoint_path = str(tmp_path / "autoquant_legacy_signature_checkpoint.pth")

    model_1 = SimpleLinear()
    mtq.auto_quantize(
        model_1,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model_1.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        checkpoint=checkpoint_path,
    )

    saved = safe_load(checkpoint_path)
    saved["scoring_signature"].pop("score_component_tracking", None)
    saved.pop("score_component_tracking", None)
    saved.pop("score_component_metadata", None)
    safe_save(saved, checkpoint_path)

    model_2 = SimpleLinear()
    mtq.auto_quantize(
        model_2,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model_2.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        checkpoint=checkpoint_path,
    )

    model_3 = SimpleLinear()
    with pytest.raises(ValueError, match="scoring signature.*score_component_tracking"):
        mtq.auto_quantize(
            model_3,
            constraints={"effective_bits": 6.0},
            quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
            data_loader=[model_3.get_input() for _ in range(2)],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=2,
            num_score_steps=2,
            checkpoint=checkpoint_path,
            score_component_tracking="batch",
        )


def test_auto_quantize_checkpoint_rejects_data_signature_mismatch(tmp_path):
    """Test that restored candidate stats are not reused with changed data provenance."""
    checkpoint_path = str(tmp_path / "autoquant_data_signature_checkpoint.pth")
    data_signature_1 = {
        "dataset": "codeblend",
        "split": "calib",
        "seq_len": 8192,
        "sample_order_hash": "abc123",
    }
    data_signature_2 = {
        "dataset": "codeblend",
        "split": "calib",
        "seq_len": 32768,
        "sample_order_hash": "def456",
    }

    model_1 = SimpleLinear()
    mtq.auto_quantize(
        model_1,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model_1.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        checkpoint=checkpoint_path,
        data_signature=data_signature_1,
    )

    saved = safe_load(checkpoint_path)
    assert saved["scoring_signature"]["data_signature"] == data_signature_1
    assert saved["candidate_stats"]

    model_2 = SimpleLinear()
    with pytest.raises(ValueError, match="scoring signature.*data_signature"):
        mtq.auto_quantize(
            model_2,
            constraints={"effective_bits": 6.0},
            quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
            data_loader=[model_2.get_input() for _ in range(2)],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=2,
            num_score_steps=2,
            checkpoint=checkpoint_path,
            data_signature=data_signature_2,
        )


@pytest.mark.parametrize("method", ["gradient", "hidden_recon", "kl_div"])
def test_get_auto_quantize_config(method):
    model = TransformerBlock()

    _, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(4)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        method=method,
    )

    # Verify search_state has method and module_names
    assert search_state["method"] == method
    for stats in search_state["candidate_stats"].values():
        assert "module_names" in stats
        assert len(stats["module_names"]) > 0

    # Use stored best recipe
    config = mtq.get_auto_quantize_config(search_state)
    assert "quant_cfg" in config
    assert isinstance(config["quant_cfg"], list)
    assert any(
        entry["quantizer_name"] == "*" and entry.get("enable") is False
        for entry in config["quant_cfg"]
    )
    assert config["algorithm"] == "max"

    # Re-solve with different constraints
    config_resoled = mtq.get_auto_quantize_config(
        search_state, constraints={"effective_bits": 12.0}
    )
    assert "quant_cfg" in config_resoled

    # Apply config to a fresh model
    fresh_model = TransformerBlock()
    fresh_model = mtq.quantize(fresh_model, config, forward_loop=lambda m: m(model.get_input()))
    output = fresh_model(model.get_input())
    assert output is not None


def test_get_auto_quantize_candidate_packets_replays_search_state():
    model = TransformerBlock()

    _, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(4)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
    )

    replay = mtq.get_auto_quantize_candidate_packets(
        search_state, constraints={"effective_bits": 12.0}, top_k=2
    )
    best_recipe = quant_algorithms._resolve_best_recipe(
        search_state, constraints={"effective_bits": 12.0}
    )

    assert replay["is_satisfied"]
    assert replay["constraints"]["candidate_rerank"]["enabled"] is True
    assert replay["constraints"]["candidate_rerank"]["top_k"] == 2
    assert replay["candidate_rerank_source"]["loaded_signal_entries"] == 0
    assert len(replay["candidate_packets"]) == 2
    assert replay["candidate_packets"][0]["recipe"] == best_recipe
    assert replay["candidate_packets"][0]["launch_authority"] == "no"
    assert replay["candidate_packets"][0]["source_provenance"] == "restored_auto_quantize_state"
    assert replay["candidate_packets"][0]["lp_packet_id"] == 0
    assert replay["candidate_packets"][0]["rerank_rank"] == 0
    assert "family_format_counts" in replay["candidate_packets"][0]
    assert "hparam_family_format_counts" in replay["candidate_packets"][0]


def test_get_auto_quantize_candidate_packets_replays_saved_score_model():
    fp8 = QuantRecipe("FP8_DEFAULT_CFG")
    bf16 = QuantRecipe(None)
    first_hparam = "layers.0.linear_attn.quant_recipe"
    second_hparam = "layers.1.linear_attn.quant_recipe"
    search_state = {
        "method": "gradient",
        "candidate_stats": {
            first_hparam: {
                "formats": [fp8, bf16],
                "scores": [2.0, 0.0],
                "costs": [1.0, 2.0],
                "element_costs": [100.0, 200.0],
                "module_names": ["layers.0.linear_attn.out_proj"],
            },
            second_hparam: {
                "formats": [fp8, bf16],
                "scores": [1.0, 0.0],
                "costs": [1.0, 2.0],
                "element_costs": [1.0, 2.0],
                "module_names": ["layers.1.linear_attn.out_proj"],
            },
        },
        "cost_denominator": 4.0,
        "cost_model": "weight",
        "cost": {},
        "best": {
            "constraints": {
                "effective_bits": 12.0,
                "cost_lower_bound": 1.0,
                "score_model": "per_element",
            }
        },
    }

    replay = mtq.get_auto_quantize_candidate_packets(search_state)

    assert replay["constraints"]["score_model"] == "per_element"
    assert replay["candidate_packets"][0]["recipe"][first_hparam] == fp8
    assert replay["candidate_packets"][0]["recipe"][second_hparam] == bf16


def test_get_auto_quantize_candidate_packets_rejects_kl_div_state():
    with pytest.raises(ValueError, match="does not support method='kl_div'"):
        mtq.get_auto_quantize_candidate_packets(
            {
                "method": "kl_div",
                "candidate_stats": {},
                "best": {"constraints": {"effective_bits": 8.0}},
            }
        )


def test_get_auto_quantize_candidate_packets_applies_signature_rerank():
    bf16 = QuantRecipe(None)
    fp8 = QuantRecipe("FP8_DEFAULT_CFG")
    candidate_stats = {
        "model.layers.0.linear_attn/layer.quant_recipe": {
            "formats": [bf16, fp8],
            "scores": [0.0, 1.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.0.linear_attn.out_proj"],
        },
        "model.layers.1.linear_attn/layer.quant_recipe": {
            "formats": [bf16, fp8],
            "scores": [0.0, 2.0],
            "costs": [2.0, 1.0],
            "module_names": ["model.layers.1.linear_attn.out_proj"],
        },
    }
    search_state = {
        "method": "gradient",
        "candidate_stats": candidate_stats,
        "cost_denominator": 4.0,
        "cost_model": "weight",
        "cost": {},
        "best": {"constraints": {"effective_bits": 12.0}},
    }
    baseline = mtq.get_auto_quantize_candidate_packets(
        search_state,
        constraints={"effective_bits": 12.0, "cost_lower_bound": 1.0},
        top_k=2,
    )
    second_packet_signature = baseline["candidate_packets"][1]["signature"]

    reranked = mtq.get_auto_quantize_candidate_packets(
        search_state,
        constraints={
            "effective_bits": 12.0,
            "cost_lower_bound": 1.0,
            "candidate_rerank": {
                "entries": [{"signature": second_packet_signature, "rerank_score": "-2.0"}],
            },
        },
        top_k=2,
    )

    assert reranked["candidate_rerank_source"]["loaded_signal_entries"] == 1
    assert reranked["candidate_packets"][0]["signature"] == second_packet_signature
    assert reranked["candidate_packets"][0]["lp_packet_id"] == 1
    assert reranked["candidate_packets"][0]["rerank_rank"] == 0
    assert reranked["candidate_packets"][0]["rerank_score"] == pytest.approx(-2.0)
    assert reranked["candidate_packets"][0]["rerank_source_match"] is True
    assert reranked["candidate_packets"][0]["launch_authority"] == "no"
    family_counts = reranked["candidate_packets"][0]["family_format_counts"]
    hparam_family_counts = reranked["candidate_packets"][0]["hparam_family_format_counts"]
    assert sum(family_counts["linear_attn"].values()) == 2
    assert sum(hparam_family_counts["linear_attn"].values()) == 2


def test_get_auto_quantize_config_keeps_selected_lm_head_enabled():
    recipe_config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    recipe_config["quant_cfg"].append({"quantizer_name": "*lm_head*", "enable": False})
    recipe = QuantRecipe(recipe_config, name="explicit_lm_head_disable")
    search_state = {
        "best": {"recipe": {"lm_head.quant_recipe": recipe}},
        "candidate_stats": {"lm_head.quant_recipe": {"module_names": ["lm_head"]}},
        "disabled_layers": ["*visual*", "*mtp*"],
    }

    config = mtq.get_auto_quantize_config(search_state)
    quant_cfg = config["quant_cfg"]
    quantizer_names = [entry["quantizer_name"] for entry in quant_cfg]

    default_disable_idx = next(
        idx for idx, entry in enumerate(quant_cfg) if entry["quantizer_name"] == "*lm_head*"
    )
    weight_idx = next(
        idx
        for idx, entry in enumerate(quant_cfg)
        if entry["quantizer_name"] == "lm_head.weight_quantizer"
    )
    weight_entry = quant_cfg[weight_idx]

    assert "*visual*" in quantizer_names
    assert "*mtp*" in quantizer_names
    assert default_disable_idx < weight_idx
    assert weight_entry["enable"] is True
    assert weight_entry["cfg"]["num_bits"] == (4, 3)
