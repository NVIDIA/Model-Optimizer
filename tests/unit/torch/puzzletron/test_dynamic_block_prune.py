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

"""CPU tests for dynamic single-block pruning masks."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from modelopt.torch.puzzletron.pruning.attention_ffn_surgery import ffn_keep_mask
from modelopt.torch.puzzletron.pruning.dynamic_block_prune import (
    AttnRemovalSpec,
    FFNRemovalSpec,
    apply_prune_hooks,
    build_block_prune_specs,
)


def _attn_specs(target_num_q, target_num_kv, head_dim=4, orig_q=24, orig_kv=4):
    return build_block_prune_specs(
        down_proj_name=None, o_proj_name="o",
        orig_intermediate=None, target_intermediate=None,
        orig_num_q=orig_q, orig_num_kv=orig_kv,
        target_num_q=target_num_q, target_num_kv=target_num_kv, head_dim=head_dim,
    )


def test_build_specs_covers_all_attention_cases():
    head_dim = 4
    # Q-preserving KV merge is not part of the current pruning contract.
    with pytest.raises(AssertionError, match="target_heads_per_group"):
        _attn_specs(24, 2)
    # Removal cases -> AttnRemovalSpec; kept query heads = target_num_q.
    for tq, tkv, kept in [(20, 4, 20), (18, 3, 18), (15, 3, 15), (12, 2, 12)]:
        (spec,) = _attn_specs(tq, tkv)
        assert isinstance(spec, AttnRemovalSpec)
        assert int(spec.keep_mask.sum()) == kept * head_dim, (tq, tkv)


def test_build_specs_ffn():
    specs = build_block_prune_specs(
        down_proj_name="down_proj", o_proj_name=None,
        orig_intermediate=12, target_intermediate=8,
        orig_num_q=None, orig_num_kv=None, target_num_q=None, target_num_kv=None, head_dim=None,
    )
    assert len(specs) == 1 and isinstance(specs[0], FFNRemovalSpec)
    assert int(specs[0].keep_mask.sum()) == 8


class _FFN(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Attn(nn.Module):
    def __init__(self, hidden, num_q, num_kv, head_dim):
        super().__init__()
        self.num_q, self.num_kv, self.head_dim = num_q, num_kv, head_dim
        self.q_proj = nn.Linear(hidden, num_q * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q * head_dim, hidden, bias=False)

    def forward(self, x):
        t = x.shape[0]
        qh = self.q_proj(x).view(t, self.num_q, self.head_dim)
        kh = self.k_proj(x).view(t, self.num_kv, self.head_dim)
        vh = self.v_proj(x).view(t, self.num_kv, self.head_dim)
        n_in = self.num_q // self.num_kv
        out = torch.empty(t, self.num_q, self.head_dim, dtype=x.dtype)
        for h in range(self.num_q):
            p = torch.softmax(qh[:, h] @ kh[:, h // n_in].t() / self.head_dim**0.5, dim=-1)
            out[:, h] = p @ vh[:, h // n_in]
        return self.o_proj(out.reshape(t, self.num_q * self.head_dim))


def test_ffn_removal_mask_hook_equals_slice():
    torch.manual_seed(0)
    hidden, inter, keep = 6, 12, 5
    ffn = _FFN(hidden, inter).double()
    x = torch.randn(4, hidden, dtype=torch.float64)

    mask = ffn_keep_mask(inter, torch.arange(keep))  # sorted-teacher prefix [:5]
    handles = apply_prune_hooks(ffn, [FFNRemovalSpec("down_proj", mask)])
    got = ffn(x)
    for h in handles:
        h.remove()

    # Reference: physically sliced FFN (first `keep` channels).
    ref = (
        F.silu(x @ ffn.gate_proj.weight[:keep].t()) * (x @ ffn.up_proj.weight[:keep].t())
    ) @ ffn.down_proj.weight[:, :keep].t()
    assert torch.allclose(got, ref, atol=1e-10), (got - ref).abs().max()
