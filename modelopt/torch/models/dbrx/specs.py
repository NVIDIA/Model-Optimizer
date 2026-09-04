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

"""DBRX specs (HF model type ``dbrx``)."""

from ..registry import register
from ..specs import ModelSpec, MoESpec, MoEVariant

# Expert names refer to the quantized layout: _QuantDbrxExpertGLU rewrites the fused
# w1/v1/w2 parameters into per-expert w1_linear/v1_linear/w2_linear ModuleLists on
# experts.mlp (see modelopt/torch/quantization/plugins/huggingface.py).
register(
    ModelSpec(
        model_type="dbrx",
        min_transformers_version="4.57",
        moe_spec=MoESpec(
            moe_variants=(
                MoEVariant(
                    block_names=("DbrxFFN",),
                    expert_linear_names=("w1_linear", "w2_linear", "v1_linear"),
                ),
            ),
        ),
    )
)
