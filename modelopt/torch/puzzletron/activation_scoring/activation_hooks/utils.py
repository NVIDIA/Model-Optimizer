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
# mypy: ignore-errors

"""Provides a function to register activation hooks for a model.
Activation hooks are used to compute activation scores for pruning.
"""

import re

from modelopt.torch.nas.plugins.megatron_hooks.base_hooks import (
    ForwardHook,
    IndependentChannelContributionHook,
    IndependentKvHeadContributionHook,
    IterativeChannelContributionHook,
    LayerNormContributionHook,
)
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.modeling_decilm import DeciLMForCausalLM


def register_activation_hooks(
    model: DeciLMForCausalLM, activation_hooks_kwargs: dict
) -> tuple[dict[str, ForwardHook], type[ForwardHook]]:
    hook_class_map = {
        "mlp.down_proj": {
            "independent": IndependentChannelContributionHook,
            "iterative": IterativeChannelContributionHook,
        },
        "self_attn.o_proj": {
            "independent_kv_head_contribution": IndependentKvHeadContributionHook,
        },
        r"regex:experts\.\d+\.down_proj$": {  # For MoE
            "independent": IndependentChannelContributionHook,
        },
        # TODO: maybe this is too generic, and we should have it specifically for
        # input_layernorm and post_attention_layernorm; now it might select qk_norms
        "layernorm": {
            "layer_norm_contribution": LayerNormContributionHook,
        },
    }

    activation_hooks = {}
    target_layer = activation_hooks_kwargs.get("target_layer", "mlp.c_proj")

    if target_layer.startswith("regex:"):
        target_layer_regex = target_layer[len("regex:") :]
        pattern = re.compile(target_layer_regex)

        def match_predicate(module_name, module):
            return pattern.search(module_name)
    else:

        def match_predicate(module_name, module):
            return module_name.endswith(target_layer)

    target_layer_hooks_map = hook_class_map.get(target_layer)
    if target_layer_hooks_map is None:
        raise ValueError(f"no hook classes found for: {target_layer}")

    hook_class = target_layer_hooks_map.get(activation_hooks_kwargs["method"])
    if hook_class is None:
        raise ValueError(f"Unknown hook class: {hook_class}")

    if target_layer == "block":
        pattern = re.compile(r"^transformer\.h\.\d+$")

        def match_predicate(module_name, module):
            return pattern.match(module_name)

    activation_hooks_kwargs["model"] = model
    for module_name, module in model.named_modules():
        if match_predicate(module_name, module):
            block_config = None
            if block_idx_match := re.search(r"\.(\d+)\.", module_name):
                block_idx = int(block_idx_match.group(1))
                block_config = model.config.block_configs[block_idx]
            curr_activation_hooks_kwargs = {
                **activation_hooks_kwargs,
                "block_config": block_config,
            }

            hook = hook_class(module, curr_activation_hooks_kwargs)
            module.register_forward_hook(hook)
            activation_hooks[module_name] = hook

    return activation_hooks, hook_class
