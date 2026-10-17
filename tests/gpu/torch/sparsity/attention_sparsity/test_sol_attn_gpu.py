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

"""GPU tests for the Sol-Attn sparse attention method."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("sol_attn")

import modelopt.torch.sparsity.attention_sparsity as mtsa


class SDPAAttention(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = SDPAAttention()

    def forward(self, q, k, v):
        return self.attn1(q, k, v)


def _qkv(tokens=4096, heads=4, dtype=torch.bfloat16):
    torch.manual_seed(0)
    return [torch.randn(1, heads, tokens, 128, device="cuda", dtype=dtype) for _ in range(3)]


def _sol_cfg(tau):
    return {
        "sparse_cfg": {
            "*attn*": {"method": "sol_attn", "tau": tau, "collect_stats": True},
            "default": {"enable": False},
        }
    }


@pytest.mark.skipif(torch.cuda.get_device_capability() < (8, 0), reason="Sol-Attn needs SM80+")
def test_sparsified_matches_kernel_semantics():
    """tau=-1e4 routes every KV block to exact attention; tau=1 approximates some blocks."""
    q, k, v = _qkv()
    ref = F.scaled_dot_product_attention(q, k, v).float()

    def run(tau):
        model = mtsa.sparsify(Block(), _sol_cfg(tau))
        out = model(q, k, v)
        assert model.attn1.get_stats()["total_calls"] == 1
        return out.float()

    exact, sparse = run(-1e4), run(1.0)
    assert ((exact - ref).norm() / ref.norm()).item() < 1e-2
    assert torch.isfinite(sparse).all()
    assert not torch.allclose(sparse, exact, atol=1e-2)


def test_ineligible_call_falls_back_to_dense():
    q, k, v = _qkv(dtype=torch.float16)
    model = mtsa.sparsify(Block(), _sol_cfg(1.0))
    torch.testing.assert_close(model(q, k, v), F.scaled_dot_product_attention(q, k, v))
    assert model.attn1.get_stats()["total_calls"] == 0
