# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GPU tests for the ``torch_a2a`` expert-parallel MoE transport (``a2a_moe``)."""

import copy

import pytest
import torch
import torch.distributed as dist
from packaging.version import Version

import modelopt.torch.utils.moe as moe

# One MoE layer's expert stage, big enough that the order of each token's top-k adds shows in bf16.
TOKENS, HIDDEN, EXPERTS, TOP_K = 4096, 1024, 16, 8


def _grouped_mm_bit_for_bit_job(rank, size):
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    moe.register_torch_a2a()
    group, _ = dist.new_subgroups(1)
    try:
        config = Qwen3MoeConfig(
            hidden_size=HIDDEN,
            moe_intermediate_size=HIDDEN // 2,
            num_experts=EXPERTS,
            num_experts_per_tok=TOP_K,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
        )
        torch.manual_seed(0)
        grouped = Qwen3MoeExperts(config).to("cuda", torch.bfloat16)
        with torch.no_grad():
            grouped.gate_up_proj.normal_(std=0.05)
            grouped.down_proj.normal_(std=0.05)
        a2a = copy.deepcopy(grouped)
        grouped.config._experts_implementation = "grouped_mm"
        a2a.config._experts_implementation = "torch_a2a"
        x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
        weights, index = torch.randn(TOKENS, EXPERTS, device="cuda").softmax(-1).topk(TOP_K, dim=-1)
        with torch.no_grad():
            want = grouped(x, index, weights)
            got = a2a(x, index, weights, group)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    finally:
        dist.destroy_process_group(group)


def test_torch_a2a_gives_grouped_mms_result_bit_for_bit(dist_workers):
    """fp32 routing weights, as Mixtral's and DeepSeek-V3's routers return, make each token's top-k
    sum an fp32 one, whose bf16 result depends on the order of its adds. At EP=1 the torch_a2a
    forward adds them in grouped_mm's fixed order, not in the order of CUDA atomics."""
    transformers = pytest.importorskip("transformers")
    if Version(transformers.__version__) < Version("5.6"):
        pytest.skip("expert parallelism needs transformers >= 5.6")
    dist_workers.run(_grouped_mm_bit_for_bit_job)
