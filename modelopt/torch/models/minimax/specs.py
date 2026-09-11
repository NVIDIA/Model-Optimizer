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

"""MiniMax specs (HF model type ``minimax``)."""

from ..specs import ModelSpec, MoESpec, register

__all__: list[str] = []

# MiniMax derives from Mixtral and inherits its w1/w2/w3 expert naming
# (MiniMaxBlockSparseTop2MLP). transformers 5 replaces the per-expert ModuleList with a
# fused MiniMaxExperts, which the structural first-projection check resolves instead --
# this naming describes the per-expert form, so it declines there.
#
# No ExportSpec.grouped_expert_export: legacy get_experts_list keyed off
# ``type(root_model).__name__.lower()`` and "minimaxforcausallm" matched none of its
# substrings, so grouped export raised on this model and the spec preserves that.
register(
    ModelSpec(
        model_type="minimax",
        min_transformers_version="4.57",
        moe_spec=MoESpec(
            block_names=("MiniMaxSparseMoeBlock",),
            expert_linear_names=("w1", "w2", "w3"),
            # w1 = gate, w3 = up, w2 = down (Mixtral convention).
            gate_up_pair=("w1", "w3"),
        ),
    )
)
