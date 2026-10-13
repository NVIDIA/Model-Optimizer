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
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.utils.instantiate_utils import instantiate
from megatron.bridge.utils.yaml_utils import dump_dataclass_to_yaml

from modelopt.torch.utils.plugins.mbridge import set_moe_expert_layout


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
