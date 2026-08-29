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
"""The MoE expert layout must be chosen identically by every script that builds the model."""

import pytest

from modelopt.torch.export.plugins.mcore_common import all_mcore_hf_export_mapping


@pytest.mark.parametrize(
    ("arch", "grouped_is_exportable"),
    [
        # These map fused grouped-GEMM experts, so they keep the faster layout.
        ("NemotronHForCausalLM", True),
        ("Qwen3_5MoeForConditionalGeneration", True),
        ("Qwen3MoeForCausalLM", False),
        ("DeepseekV3ForCausalLM", False),
        ("GptOssForCausalLM", False),
        ("Llama4ForConditionalGeneration", False),
    ],
)
def test_grouped_gemm_exportability(arch, grouped_is_exportable):
    assert arch in all_mcore_hf_export_mapping, f"{arch} is no longer an exported architecture"
    has_rule = "experts.linear_fc1" in all_mcore_hf_export_mapping[arch]
    assert has_rule is grouped_is_exportable, (
        f"{arch} grouped-GEMM exportability changed; use_moe_grouped_gemm would now pick the "
        "other expert layout, which silently breaks checkpoints written by the previous default"
    )
