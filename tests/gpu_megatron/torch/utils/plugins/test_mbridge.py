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

import pytest
import yaml
from _test_utils.torch.transformers_models import (
    create_tiny_gpt_oss_dir,
    create_tiny_nemotron_h_dir,
    create_tiny_qwen3_5_moe_vl_dir,
    create_tiny_qwen3_moe_dir,
    create_tiny_qwen3vl_dir,
)
from megatron.bridge import AutoBridge
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.utils.instantiate_utils import instantiate
from megatron.bridge.utils.yaml_utils import dump_dataclass_to_yaml

from modelopt.torch.utils.plugins.mbridge import module_mappings_from_bridge, set_moe_expert_layout


def _round_trip(value):
    """Serialize through the writer used for run_config.yaml, then reload."""
    node = yaml.safe_load(dump_dataclass_to_yaml({"spec": value}))["spec"]
    return node["_target_"], instantiate(node)


@pytest.mark.parametrize(
    ("moe_grouped_gemm", "expected_experts", "expected_target"),
    [
        (
            True,
            "TEGroupedMLP",
            "megatron.bridge.models.hybrid.hybrid_provider.transformer_engine_hybrid_stack_spec",
        ),
        (
            False,
            "SequentialMLP",
            "modelopt.torch.utils.plugins.megatron_layer_specs.te_hybrid_stack_spec_sequential_mlp",
        ),
    ],
)
def test_set_moe_expert_layout_survives_run_config_round_trip(
    moe_grouped_gemm, expected_experts, expected_target
):
    """A provider's stack spec must still build real submodules after a run_config round trip.

    A built ``ModuleSpec`` loses its ``MLPSubmodules`` / ``MoESubmodules`` when written to
    ``run_config.yaml``, so ``set_moe_expert_layout`` stores a factory function instead.
    """
    provider = HybridModelProvider(num_layers=2, hidden_size=64, num_attention_heads=4)
    set_moe_expert_layout(provider, moe_grouped_gemm=moe_grouped_gemm)
    assert provider.moe_grouped_gemm == moe_grouped_gemm

    assert callable(provider.hybrid_stack_spec)

    target, factory = _round_trip(provider.hybrid_stack_spec)
    # The target is an on-disk contract: renaming or moving the factory breaks saved checkpoints.
    assert target == expected_target

    provider.hybrid_stack_spec = factory
    spec = provider._resolve_hybrid_stack_spec()

    mlp = spec.submodules.mlp_layer.submodules.mlp.keywords["submodules"]
    assert mlp.linear_fc1 is not None
    assert mlp.linear_fc2 is not None

    moe = spec.submodules.moe_layer.submodules.mlp.keywords["submodules"]
    assert moe.experts is not None
    # Experts are built through a partial for the grouped-GEMM layout.
    assert getattr(moe.experts, "func", moe.experts).__name__ == expected_experts


def _projs(prefix, names):
    return tuple(f"{prefix}.{name}_proj" for name in names)


@pytest.mark.parametrize(
    ("create_model", "expected"),
    [
        pytest.param(
            create_tiny_qwen3_moe_dir,
            {
                "output_layer": ("lm_head",),
                "decoder.layers.*.mlp.router": ("model.layers.*.mlp.gate",),
                "decoder.layers.*.self_attention.linear_proj": ("model.layers.*.self_attn.o_proj",),
                "decoder.layers.*.self_attention.linear_qkv": _projs(
                    "model.layers.*.self_attn", "qkv"
                ),
                "decoder.layers.*.self_attention.core_attention": ("model.layers.*.self_attn",),
                "decoder.layers.*.mlp.experts.linear_fc1": _projs(
                    "model.layers.*.mlp.experts.*", ("gate", "up")
                ),
                "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2": _projs(
                    "model.layers.*.mlp.experts.*", ("down",)
                ),
            },
            id="qwen3_moe",
        ),
        pytest.param(
            create_tiny_qwen3vl_dir,
            {
                "language_model.output_layer": ("lm_head",),
                "language_model.decoder.layers.*.self_attention.linear_qkv": _projs(
                    "model.language_model.layers.*.self_attn", "qkv"
                ),
                "language_model.decoder.layers.*.mlp.linear_fc1": _projs(
                    "model.language_model.layers.*.mlp", ("gate", "up")
                ),
            },
            id="qwen3vl",
        ),
        pytest.param(
            create_tiny_nemotron_h_dir,
            {
                "output_layer": ("lm_head",),
                "decoder.layers.*.mixer.in_proj": ("backbone.layers.*.mixer.in_proj",),
                "decoder.layers.*.mlp.linear_fc1": _projs("backbone.layers.*.mixer", ("up",)),
                "decoder.layers.*.self_attention.linear_qkv": _projs(
                    "backbone.layers.*.mixer", "qkv"
                ),
                "decoder.layers.*.self_attention.core_attention": ("backbone.layers.*.mixer",),
                "decoder.layers.*.mlp.experts.linear_fc1": _projs(
                    "backbone.layers.*.mixer.experts.*", ("up",)
                ),
            },
            id="nemotron_h",
        ),
        # The rows tests/unit/torch/quantization/test_hf_to_megatron.py hand-writes for Qwen3.5-MoE,
        # except the routed experts: Bridge maps those from how the checkpoint stores them.
        pytest.param(
            create_tiny_qwen3_5_moe_vl_dir,
            {
                "language_model.output_layer": ("lm_head",),
                "language_model.decoder.layers.*.self_attention.in_proj": tuple(
                    f"model.language_model.layers.*.linear_attn.in_proj_{p}"
                    for p in ("qkv", "z", "b", "a")
                ),
                "language_model.decoder.layers.*.mlp.shared_experts.linear_fc1": _projs(
                    "model.language_model.layers.*.mlp.shared_expert", ("gate", "up")
                ),
            },
            id="qwen3_5_moe",
        ),
        # Packed experts: both Megatron layouts, grouped and sequential, map from the same parameters.
        pytest.param(
            create_tiny_gpt_oss_dir,
            {
                "decoder.layers.*.mlp.experts.linear_fc1": (
                    "model.layers.*.mlp.experts.gate_up_proj",
                ),
                "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1": (
                    "model.layers.*.mlp.experts.gate_up_proj",
                ),
                "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2": (
                    "model.layers.*.mlp.experts.down_proj",
                ),
            },
            id="gpt_oss",
        ),
    ],
)
def test_module_mappings_from_bridge(tmp_path, create_model, expected):
    """Each architecture's registry maps its Hugging Face module names onto Megatron's.

    The naming is the architecture's own (NemotronH's MLP is ``mixer.up_proj``, a VLM nests its
    language model) and recipe translation rests on it, so a Megatron-Bridge change to these rows
    must fail here rather than leave a recipe's globs matching nothing.
    """
    bridge = AutoBridge.from_hf_pretrained(str(create_model(tmp_path)))
    mappings = {
        mapping.megatron: frozenset(mapping.hf) for mapping in module_mappings_from_bridge(bridge)
    }
    assert {module: frozenset(hf) for module, hf in expected.items()}.items() <= mappings.items()
