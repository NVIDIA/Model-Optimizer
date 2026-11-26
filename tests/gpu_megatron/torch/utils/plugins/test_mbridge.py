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
from functools import partial

import pytest
import yaml
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.utils.instantiate_utils import instantiate, register_allowed_target_prefix
from megatron.bridge.utils.yaml_utils import dump_dataclass_to_yaml
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules

from modelopt.torch.utils.plugins.mbridge import set_moe_expert_layout
from modelopt.torch.utils.plugins.megatron_layer_specs import hybrid_stack_spec_sequential_mlp

register_allowed_target_prefix(f"{__name__}.")


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
@pytest.mark.parametrize("stack_spec", [None, hybrid_stack_spec], ids=["default", "generic-spec"])
def test_set_moe_expert_layout_survives_run_config_round_trip(
    moe_grouped_gemm, expected_experts, expected_target, stack_spec
):
    """A provider's stack spec must still build real submodules after a run_config round trip.

    A built ``ModuleSpec`` loses its ``MLPSubmodules`` / ``MoESubmodules`` when written to
    ``run_config.yaml``, so ``set_moe_expert_layout`` stores a factory function instead.
    """
    provider = HybridModelProvider(num_layers=2, hidden_size=64, num_attention_heads=4)
    if stack_spec is not None:
        provider.hybrid_stack_spec = stack_spec
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


class CustomMoELayer(MoELayer):
    """A model-specific MoE implementation that layout selection must preserve."""


def custom_hybrid_stack_spec():
    spec = copy.deepcopy(hybrid_stack_spec)
    spec.submodules.attention_layer.submodules.self_attention.params["attn_mask_type"] = (
        AttnMaskType.padding
    )
    mlp = spec.submodules.moe_layer.submodules.mlp
    spec.submodules.moe_layer.submodules.mlp = partial(CustomMoELayer, **mlp.keywords)
    shared = get_submodules(spec.submodules.moe_layer.submodules.mlp).shared_experts
    shared.keywords["name"] = "custom_shared_experts"
    return spec


def config_custom_hybrid_stack_spec(config):
    assert isinstance(config, HybridModelProvider)
    return custom_hybrid_stack_spec()


def dense_custom_hybrid_stack_spec():
    spec = custom_hybrid_stack_spec()
    spec.submodules.moe_layer = IdentityOp
    return spec


def wrapped_dense_custom_hybrid_stack_spec():
    spec = dense_custom_hybrid_stack_spec()
    spec.submodules.moe_layer = ModuleSpec(module=IdentityOp)
    return spec


def local_custom_hybrid_stack_spec():
    spec = custom_hybrid_stack_spec()
    get_submodules(spec.submodules.moe_layer.submodules.mlp).experts = get_submodules(
        get_moe_module_spec(use_te=False, num_experts=8, moe_grouped_gemm=False)
    ).experts
    return spec


def module_spec_custom_hybrid_stack_spec():
    return _with_module_spec_experts(custom_hybrid_stack_spec())


def module_spec_local_custom_hybrid_stack_spec():
    return _with_module_spec_experts(local_custom_hybrid_stack_spec())


def _with_module_spec_experts(spec):
    moe = get_submodules(spec.submodules.moe_layer.submodules.mlp)
    experts = moe.experts
    params = dict(experts.keywords)
    submodules = params.pop("submodules")
    moe.experts = ModuleSpec(
        module=experts.func,
        params=params,
        submodules=submodules,
        metainfo={"source": "custom_factory"},
    )
    return spec


@pytest.mark.parametrize(
    ("stack_spec_factory", "binding"),
    [
        (custom_hybrid_stack_spec, "plain"),
        (config_custom_hybrid_stack_spec, "plain"),
        (custom_hybrid_stack_spec, "positional"),
        (custom_hybrid_stack_spec, "keyword"),
        (dense_custom_hybrid_stack_spec, "plain"),
        (wrapped_dense_custom_hybrid_stack_spec, "plain"),
        (local_custom_hybrid_stack_spec, "plain"),
        (module_spec_custom_hybrid_stack_spec, "plain"),
        (module_spec_local_custom_hybrid_stack_spec, "plain"),
    ],
)
def test_custom_hybrid_stack_survives_layout_changes_and_run_config_round_trip(
    stack_spec_factory, binding
):
    provider = HybridModelProvider(num_layers=2, hidden_size=64, num_attention_heads=4)
    provider.hybrid_stack_spec = stack_spec_factory
    original = provider._resolve_hybrid_stack_spec()
    sequential_only = stack_spec_factory in (
        local_custom_hybrid_stack_spec,
        module_spec_local_custom_hybrid_stack_spec,
    )
    if binding == "positional":
        provider.hybrid_stack_spec = partial(hybrid_stack_spec_sequential_mlp, stack_spec_factory)
    elif binding == "keyword":
        provider.hybrid_stack_spec = partial(
            hybrid_stack_spec_sequential_mlp, stack_spec_factory=stack_spec_factory
        )

    for grouped in (False, False, True, False):
        set_moe_expert_layout(provider, moe_grouped_gemm=grouped)
        target, factory = _round_trip(provider.hybrid_stack_spec)
        expected_factory = (
            "hybrid_stack_spec_grouped_mlp" if grouped else "hybrid_stack_spec_sequential_mlp"
        )
        assert target == f"modelopt.torch.utils.plugins.megatron_layer_specs.{expected_factory}"
        provider.hybrid_stack_spec = factory
        assert provider.moe_grouped_gemm == grouped
        if grouped and sequential_only:
            with pytest.raises(ValueError):
                provider._resolve_hybrid_stack_spec()
            continue
        spec = provider._resolve_hybrid_stack_spec()
        assert spec.submodules.mamba_layer == original.submodules.mamba_layer
        assert spec.submodules.attention_layer == original.submodules.attention_layer
        if stack_spec_factory in (
            dense_custom_hybrid_stack_spec,
            wrapped_dense_custom_hybrid_stack_spec,
        ):
            assert spec.submodules.moe_layer == original.submodules.moe_layer
            continue

        mlp = spec.submodules.moe_layer.submodules.mlp
        moe = get_submodules(mlp)
        original_moe = get_submodules(original.submodules.moe_layer.submodules.mlp)
        assert mlp.func is CustomMoELayer
        assert moe.shared_experts.func is original_moe.shared_experts.func
        assert moe.shared_experts.keywords == original_moe.shared_experts.keywords
        assert moe.router is original_moe.router
        if isinstance(original_moe.experts, ModuleSpec) and (grouped or sequential_only):
            assert moe.experts == original_moe.experts
        elif sequential_only:
            assert moe.experts.func is original_moe.experts.func
            assert moe.experts.keywords == original_moe.experts.keywords
        else:
            assert getattr(moe.experts, "func", moe.experts).__name__ == (
                "TEGroupedMLP" if grouped else "SequentialMLP"
            )


def test_custom_hybrid_stack_object_requires_a_factory():
    provider = HybridModelProvider(num_layers=2, hidden_size=64, num_attention_heads=4)
    provider.hybrid_stack_spec = custom_hybrid_stack_spec()
    original = provider.hybrid_stack_spec
    original_grouped = provider.moe_grouped_gemm
    with pytest.raises(ValueError):
        set_moe_expert_layout(provider, moe_grouped_gemm=False)
    assert provider.hybrid_stack_spec is original
    assert provider.moe_grouped_gemm == original_grouped
