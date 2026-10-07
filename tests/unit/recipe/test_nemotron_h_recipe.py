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

"""Resolve the fixed-MTP recipe on native modules without running GPU calibration."""

import pytest
import torch

native = pytest.importorskip("transformers.models.nemotron_h.modeling_nemotron_h")

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.models.nemotron_h.mtp import _NemotronHMTP
from modelopt.torch.opt.conversion import apply_mode
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.quantization._auto_quantize_cost import get_auto_quantize_cost_model
from modelopt.torch.quantization.algorithms import AutoQuantizeGradientSearcher, QuantRecipe
from modelopt.torch.quantization.mode import QuantizeModeRegistry
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.mark.parametrize("wrapped", [False, True], ids=["standalone", "wrapped"])
@pytest.mark.parametrize("decoder_name", ["model", "backbone"])
@pytest.mark.parametrize("with_mtp", [False, True], ids=["no-mtp", "mtp"])
def test_4p9_mse_recipe_resolves_search_and_fixed_mtp(wrapped, decoder_name, with_mtp):
    """Search decoder/head groups at 4.5/8/16 bits, fix MTP experts, and cast KV."""
    config = native.NemotronHConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=32,
        layers_block_type=["attention", "moe"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        n_routed_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        use_mamba_kernels=False,
        attn_implementation="eager",
        mtp_layers_block_type=["attention", "moe"],
    )
    language_model = native.NemotronHForCausalLM(config).eval()
    # Keep real decoder modules while exercising both native and remote-code namespaces.
    current_name = "model" if "model" in language_model._modules else "backbone"
    if current_name != decoder_name:
        language_model.add_module(decoder_name, language_model._modules.pop(current_name))
    if with_mtp:
        language_model.mtp = _NemotronHMTP(config)
    model = language_model
    prefix = ""
    if wrapped:
        model = torch.nn.Module()
        model.language_model = language_model
        model.vision_model = torch.nn.Linear(32, 32)
        prefix = "language_model."

    recipe = load_recipe("model_type/nemotron_h/auto_quantize/nvfp4_mse_fp8_mtp_fixed_at_4p9bits")
    aq = recipe.auto_quantize
    fixed = QuantRecipe(recipe.quantize.model_dump(), name="fixed_mtp")
    assert fixed.config.algorithm == {"method": "mse", "fp8_scale_sweep": True}
    assert aq.auto_quantize_method == "gradient"
    assert aq.constraints.effective_bits == 4.9

    model = apply_mode(model, mode="auto_quantize", registry=QuantizeModeRegistry)
    mtq.set_quantizer_by_cfg(model, fixed.config.quant_cfg)
    searcher = AutoQuantizeGradientSearcher()
    searcher.model = model
    searcher.constraints = aq.constraints.model_dump(exclude_none=True)
    searcher.config = {"cost": {"excluded_module_name_patterns": aq.cost_excluded_layers}}
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

    searched_names, fixed_expert_names = set(), set()
    for group in groups:
        names = group.quant_module_names
        bits = [choice.compression * 16 for choice in group.solver_choices]
        if any(".mixer.router" in name or name.startswith("vision_model") for name in names):
            assert bits == [16]
        elif all(
            name.startswith(prefix + decoder_name + ".layers.") for name in names
        ) or names == [prefix + "lm_head"]:
            assert bits == [4.5, 8, 16]
            assert group.allow_no_quant and not group.is_fixed
            assert group.cost_weight == 1
            assert group.solver_choices[0].config.algorithm == fixed.config.algorithm
            searched_names.update(names)
        elif all("mtp.layers.1.mixer." in name for name in names):
            assert bits == [4.5]
            assert group.is_fixed and not group.allow_no_quant
            assert group.cost_weight == 0
            fixed_expert_names.update(names)
            for module in group.quant_modules:
                quantizers = [
                    (name, q)
                    for name, q in module.named_modules()
                    if isinstance(q, TensorQuantizer) and ("weight" in name or "input" in name)
                ]
                assert quantizers
                for name, quantizer in quantizers:
                    assert quantizer.is_enabled and quantizer.num_bits == (2, 1)
                    assert quantizer.block_sizes[-1] == 16
                    assert quantizer.block_sizes["type"] == (
                        "static" if "weight" in name else "dynamic"
                    )
        else:
            assert all(name.startswith(prefix + "mtp.") for name in names)
            assert bits == [16] and group.is_fixed and group.cost_weight == 0

    assert prefix + "lm_head" in searched_names
    assert any(".layers.0.mixer.q_proj" in name for name in searched_names)
    assert any(".layers.1.mixer.experts" in name for name in searched_names)
    assert bool(fixed_expert_names) == with_mtp
    if with_mtp:
        assert any(".mixer.experts" in name for name in fixed_expert_names)
        assert any(".mixer.shared_experts" in name for name in fixed_expert_names)

    # The baseline already casts MTP KV; the post-search preset also casts base-model KV.
    for stage in ("baseline", "post-search"):
        if stage == "post-search":
            mtq.set_quantizer_by_cfg(model, aq.kv_cache.quant_cfg)
        kv = {
            name: q
            for name, q in model.named_modules()
            if name.endswith(("k_bmm_quantizer", "v_bmm_quantizer"))
        }
        assert len(kv) == 2 + 2 * with_mtp
        for name, quantizer in kv.items():
            enabled = stage == "post-search" or "mtp." in name
            assert quantizer.is_enabled == enabled
            if enabled:
                assert quantizer.num_bits == (4, 3) and quantizer._use_constant_amax
                assert quantizer._get_amax(torch.tensor([0.01, 1000.0])).item() == 448
                assert not hasattr(quantizer, "_amax")
