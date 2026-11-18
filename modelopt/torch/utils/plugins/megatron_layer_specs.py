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

"""Megatron-Core layer specs used to build models for ModelOpt workflows."""

import copy
import inspect
from collections.abc import Callable

from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec as _te_hybrid_stack_spec,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules

__all__ = [
    "hybrid_stack_spec_grouped_mlp",
    "hybrid_stack_spec_sequential_mlp",
    "te_hybrid_stack_spec_sequential_mlp",
]


def te_hybrid_stack_spec_sequential_mlp() -> ModuleSpec:
    """Return the TE Hybrid stack spec with SequentialMLP MoE experts.

    Named and zero-argument so a provider can store this function instead of the ModuleSpec it
    builds; see ``set_moe_expert_layout`` for why a built spec cannot be serialized.

    Its module path and name are written into ``run_config.yaml`` as a ``_target_``, so moving or
    renaming it breaks every SequentialMLP hybrid checkpoint already saved.
    """
    # The upstream TE hybrid stack spec hardcodes TEGroupedMLP for MoE.
    # Replace it with SequentialMLP (TE linear layers, no grouped gemm dependency).
    # num_experts only has to be non-zero to select the MoE branch; the real count comes from the
    # model config at build time.
    te_hybrid_stack_spec = copy.deepcopy(_te_hybrid_stack_spec)
    te_hybrid_stack_spec.submodules.moe_layer.submodules.mlp = get_moe_module_spec(
        use_te=True, num_experts=8, moe_grouped_gemm=False
    )
    return te_hybrid_stack_spec


def hybrid_stack_spec_sequential_mlp(
    stack_spec_factory: Callable[..., ModuleSpec], config=None
) -> ModuleSpec:
    """Replace only the experts in a model-specific hybrid stack.

    Its module path and name are written into ``run_config.yaml`` as a ``_target_``, so moving or
    renaming it breaks saved custom-stack checkpoints.
    """
    return _hybrid_stack_spec_with_expert_layout(stack_spec_factory, config, moe_grouped_gemm=False)


def hybrid_stack_spec_grouped_mlp(
    stack_spec_factory: Callable[..., ModuleSpec], config=None
) -> ModuleSpec:
    """Preserve a model-specific hybrid stack and require TEGroupedMLP experts.

    Its module path and name are checkpoint targets and must remain stable.
    """
    return _hybrid_stack_spec_with_expert_layout(stack_spec_factory, config, moe_grouped_gemm=True)


def _hybrid_stack_spec_with_expert_layout(stack_spec_factory, config, moe_grouped_gemm):
    stack_spec = copy.deepcopy(
        stack_spec_factory(config)
        if inspect.signature(stack_spec_factory).parameters
        else stack_spec_factory()
    )
    moe_layer = stack_spec.submodules.moe_layer
    if moe_layer is None or moe_layer is IdentityOp:
        return stack_spec

    moe = get_submodules(moe_layer.submodules.mlp)
    expert_type = getattr(moe.experts, "func", moe.experts)
    if moe_grouped_gemm:
        if expert_type is not TEGroupedMLP:
            raise ValueError(
                "Grouped GEMM requires a custom hybrid factory that returns TEGroupedMLP experts; "
                "use moe_grouped_gemm=False for SequentialMLP experts"
            )
        return stack_spec
    if expert_type is SequentialMLP:
        return stack_spec
    if expert_type is not TEGroupedMLP:
        raise ValueError("Custom hybrid experts must use TEGroupedMLP or SequentialMLP")
    moe.experts = get_submodules(
        get_moe_module_spec(use_te=True, num_experts=8, moe_grouped_gemm=False)
    ).experts
    return stack_spec
