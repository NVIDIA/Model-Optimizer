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

"""Qwen3 (dense) specs (HF model type ``qwen3``)."""

from .registry import register
from .specs import ModelSpec

register(
    ModelSpec(
        model_type="qwen3",
        # AWQ pre_quant_scale fusion: fold o_proj into v_proj, down_proj into up_proj.
        pqs_fuse_rules=(
            (("Qwen3Attention",), "v_proj", "o_proj"),
            (("Qwen3MLP",), "up_proj", "down_proj"),
        ),
    )
)
