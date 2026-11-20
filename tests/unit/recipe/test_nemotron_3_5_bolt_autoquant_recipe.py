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

"""Resolve the Nemotron 3.5 Bolt AutoQuantize recipe's search space without calibration."""

import pytest
import torch.nn as nn

native = pytest.importorskip("transformers.models.nemotron_h.modeling_nemotron_h")

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.opt.conversion import apply_mode
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.quantization._auto_quantize_cost import get_auto_quantize_cost_model
from modelopt.torch.quantization.algorithms import AutoQuantizeKLDivSearcher, QuantRecipe
from modelopt.torch.quantization.mode import QuantizeModeRegistry


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "per-expert"])
def test_bolt_recipe_searches_experts_and_holds_the_rest(fused):
    """Experts get every candidate and an IQ2_XXS context; the other linears hold one format."""
    config = native.NemotronHConfig(
        vocab_size=32,
        hidden_size=32,
        layers_block_type=["mamba", "attention", "moe"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        mamba_num_heads=4,
        mamba_head_dim=16,
        ssm_state_size=16,
        n_groups=1,
        n_routed_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        moe_latent_size=16,
        use_mamba_kernels=False,
    )
    model = native.NemotronHForCausalLM(config).eval()
    if not fused:  # The checkpoint's own code keeps one module per expert.
        # Quantization registers the block's class as a sequential MoE: keep that off the built-in.
        class PerExpertMoE(native.NemotronHMoE):
            pass

        for module in model.modules():
            if isinstance(module, native.NemotronHMoE):
                module.__class__ = PerExpertMoE
                module.experts = nn.ModuleList(
                    nn.ModuleDict({"up_proj": nn.Linear(16, 32), "down_proj": nn.Linear(32, 16)})
                    for _ in range(2)
                )

    recipe = load_recipe(
        "models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/auto_quantize/"
        "iq1_iq2_gptq_nvfp4_fp8_experts_at_2p0625bits"
    )
    aq = recipe.auto_quantize
    fixed = QuantRecipe(recipe.quantize.model_dump(), name="fixed")
    model = apply_mode(model, mode="auto_quantize", registry=QuantizeModeRegistry)
    mtq.set_quantizer_by_cfg(model, fixed.config.quant_cfg)
    searcher = AutoQuantizeKLDivSearcher()
    searcher.model = model
    searcher.constraints = aq.constraints.model_dump(exclude_none=True)
    searcher.config = {
        "cost": {"excluded_module_name_patterns": aq.cost_excluded_layers},
        **aq.method_options,
    }
    searcher._cost_model = get_auto_quantize_cost_model("weight")
    spaces = searcher._normalize_module_search_spaces(
        [
            {
                "module_name_patterns": space.module_name_patterns,
                "quantization_formats": [
                    (fmt.model_dump(), f"candidate_{index}")
                    for index, fmt in enumerate(space.candidate_formats)
                ],
                "allow_no_quant": space.allow_no_quant,
            }
            for space in aq.module_search_spaces
        ]
    )
    searcher.insert_hparams_after_merge_rules(model, [], aq.disabled_layers, spaces, fixed)
    groups = [hparam for _, hparam in named_hparams(model, unique=True)]
    searcher._verify_resolved_constraint(groups)

    searched, held = set(), {}
    for group in groups:
        names = group.quant_module_names
        bits = [choice.compression * 16 for choice in group.choices]
        if all(".mixer.experts" in name or ".mixer.shared_experts." in name for name in names):
            # IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, NVFP4, FP8, BF16
            assert bits == [1.75, 2.0625, 2.3125, 2.5625, 4.5, 8, 16]
            assert group.cost_weight == 1
            assert searcher._context_recipe(group).compression * 16 == 2.0625
            searched.update(names)
        else:
            assert group.cost_weight == 0
            held.update(dict.fromkeys(names, bits))

    # The budget is every expert weight: both projections of a fused container, too.
    assert sum(group.get_cost(QuantRecipe(quant_cfg=None)) for group in groups) == sum(
        param.numel()
        for name, param in model.named_parameters()
        if param.dim() > 1 and (".mixer.experts" in name or ".mixer.shared_experts." in name)
    )

    experts = [name for name in searched if ".mixer.experts" in name]
    assert experts and all(name.endswith(".mixer.experts") == fused for name in experts)
    assert any(name.endswith(".mixer.shared_experts.up_proj") for name in searched)
    for suffix, bits in [
        (".mixer.in_proj", [4.5]),
        (".mixer.out_proj", [4.5]),
        (".mixer.q_proj", [8.5]),
        (".mixer.o_proj", [8.5]),
        (".mixer.fc1_latent_proj", [8.5]),
        ("lm_head", [8.5]),
    ]:
        assert [held[name] for name in held if name.endswith(suffix)] == [bits]
