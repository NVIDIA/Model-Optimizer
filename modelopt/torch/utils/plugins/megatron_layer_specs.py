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

from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec as _te_hybrid_stack_spec,
)
from megatron.core.transformer.spec_utils import ModuleSpec

__all__ = ["te_hybrid_stack_spec_sequential_mlp"]


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
