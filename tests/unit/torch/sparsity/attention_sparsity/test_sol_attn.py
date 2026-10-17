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

"""CPU unit tests for the Sol-Attn sparse attention method (kernel package mocked)."""

import sys
import types
from unittest.mock import patch

import pytest

pytest.importorskip("transformers")

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import ValidationError

import modelopt.torch.opt as mto
import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.config import SolAttnAttributeConfig
from modelopt.torch.sparsity.attention_sparsity.methods import sol_attn as sol_attn_method
from modelopt.torch.sparsity.attention_sparsity.methods.sol_attn import SolAttn
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule


class SDPAAttention(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)


class TwoAttnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = SDPAAttention()  # self-attention
        self.attn2 = SDPAAttention()  # cross-attention

    def forward(self, x, ctx):
        return self.attn2(self.attn1(x, x, x), ctx, ctx)


def _fake_sol_attn_module(calls):
    def sol_attn(q, k, v, **kwargs):
        calls.append((q.shape, kwargs))
        return torch.zeros_like(q)

    return types.SimpleNamespace(sol_attn=sol_attn, get_sol_attn_backend=lambda device: "triton")


@pytest.mark.parametrize(
    "cfg", [{"thresh_type": "full"}, {"kv_splits": 3}, {"method": "vsa"}, {"tau": "high"}]
)
def test_config_rejects_invalid(cfg):
    with pytest.raises(ValidationError):
        SolAttnAttributeConfig(**cfg)


def test_sparsify_routes_eligible_sdpa_to_sol_attn():
    """Self-attention runs Sol-Attn in BTHD layout; cross-attention stays dense."""
    model = mtsa.sparsify(
        TwoAttnModel(),
        {
            "sparse_cfg": {
                "*attn*": {"method": "sol_attn", "tau": 2.0, "thresh_type": "exact"},
                "default": {"enable": False},
            }
        },
    )
    assert all(
        isinstance(m._sparse_method_instance, SolAttn)
        for m in model.modules()
        if isinstance(m, SparseAttentionModule)
    )
    x = torch.randn(1, 2, 16, 128)  # [batch, heads, seq, dim]
    ctx = torch.randn(1, 2, 8, 128)
    calls = []
    eligible = lambda self, q, k, v, **kw: q.shape == k.shape  # noqa: E731
    with (
        patch.object(sol_attn_method, "_import_sol_attn", lambda: _fake_sol_attn_module(calls)),
        patch.object(SolAttn, "can_replace_sdpa", eligible),
    ):
        out = model(x, ctx)

    assert calls == [((1, 16, 2, 128), {"tau": 2.0, "thresh_type": "exact", "kv_splits": 1})]
    torch.testing.assert_close(out, F.scaled_dot_product_attention(torch.zeros_like(x), ctx, ctx))


def test_ineligible_inputs_stay_dense():
    method = SolAttn()
    q = torch.randn(1, 2, 16, 128)
    assert not method.can_replace_sdpa(q, q, q)  # CPU and float32
    assert not method.can_replace_sdpa(q, q, q, is_causal=True)


def test_missing_package_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sol_attn", None)
    model = mtsa.sparsify(TwoAttnModel(), mtsa.SOL_ATTN_DEFAULT)
    q = torch.randn(1, 2, 16, 128)
    sdpa = F.scaled_dot_product_attention
    with (
        patch.object(SolAttn, "can_replace_sdpa", lambda self, *a, **kw: True),
        pytest.raises(ImportError, match=r"requires the .sol_attn. package"),
    ):
        model(q, q)
    assert F.scaled_dot_product_attention is sdpa  # SDPA patch is undone on error


def test_save_restore():
    model = mtsa.sparsify(TwoAttnModel(), mtsa.SOL_ATTN_DEFAULT)
    restored = mto.restore_from_modelopt_state(TwoAttnModel(), mto.modelopt_state(model))
    methods = [m._method for m in restored.modules() if isinstance(m, SparseAttentionModule)]
    assert methods == ["sol_attn", "sol_attn"]
