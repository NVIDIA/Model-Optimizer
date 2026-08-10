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

"""Small GPU sanity checks for the experimental Inkling attention adapter."""

import torch

from modelopt.torch.kernels.common.attention.triton_fa import attention
from modelopt.torch.kernels.quantization.attention.bmm1_qdq import fake_quant_k_onwrite
from modelopt.torch.sparsity.attention_sparsity.plugins import vllm as vllm_plugin


def _reference(q, k, v, rel_logits, scale, window_left):
    num_heads = q.shape[1]
    repeat = num_heads // k.shape[1]
    q_heads = q.float().permute(1, 0, 2)
    k_heads = k.float().repeat_interleave(repeat, dim=1).permute(1, 0, 2)
    v_heads = v.float().repeat_interleave(repeat, dim=1).permute(1, 0, 2)
    scores = torch.matmul(q_heads, k_heads.transpose(1, 2)) * scale
    seq_len = q.shape[0]
    distance = (
        torch.arange(seq_len, device=q.device)[:, None]
        - torch.arange(seq_len, device=q.device)[None, :]
    )
    rel_extent = rel_logits.shape[2]
    rel_idx = distance.clamp(0, rel_extent - 1)
    bias = torch.gather(
        rel_logits.float().permute(1, 0, 2),
        2,
        rel_idx.expand(num_heads, -1, -1),
    )
    bias.masked_fill_(~((distance >= 0) & (distance < rel_extent)), 0.0)
    valid = (distance >= 0) & (distance <= window_left)
    probs = torch.softmax((scores + bias).masked_fill(~valid, float("-inf")), dim=-1)
    return torch.matmul(probs, v_heads).permute(1, 0, 2)


def main() -> None:
    """Exercise the Inkling-specific ModelOpt kernel contracts on one GPU."""
    import vllm

    assert vllm.__version__ == "0.26.0", vllm.__version__
    assert getattr(vllm_plugin, "InklingAttention", None) is not None
    torch.manual_seed(123)
    device = torch.device("cuda")
    seq_len, num_heads, num_kv_heads, head_dim = 17, 4, 2, 128
    rel_extent, window_left = 16, 5
    scale = 1.0 / head_dim
    q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    rel_logits = torch.randn(seq_len, num_heads, rel_extent, device=device, dtype=torch.bfloat16)
    starts = torch.tensor([0], device=device, dtype=torch.int32)
    lengths = torch.tensor([seq_len], device=device, dtype=torch.int32)

    out = attention(
        q,
        k,
        v,
        starts,
        lengths,
        seq_len,
        softmax_scale=scale,
        rel_logits=rel_logits,
        window_left=window_left,
    )
    ref = _reference(q, k, v, rel_logits, scale, window_left)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)

    page_size = 16
    k_cache = torch.zeros(2, page_size, num_kv_heads, head_dim, device=device, dtype=k.dtype)
    v_cache = torch.zeros_like(k_cache)
    k_cache.view(-1, num_kv_heads, head_dim)[:seq_len].copy_(k)
    v_cache.view(-1, num_kv_heads, head_dim)[:seq_len].copy_(v)
    block_table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    dummy = torch.empty(0, num_kv_heads, head_dim, device=device, dtype=k.dtype)
    paged = attention(
        q,
        dummy,
        dummy,
        starts,
        lengths,
        seq_len,
        softmax_scale=scale,
        b_seq_len_k=lengths,
        max_input_len_k=seq_len,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        page_size=page_size,
        rel_logits=rel_logits,
        window_left=window_left,
    )
    torch.testing.assert_close(paged.float(), ref, rtol=2e-2, atol=2e-2)

    transformed = attention(
        q,
        k,
        v,
        starts,
        lengths,
        seq_len,
        softmax_scale=scale,
        sparsity_n=2,
        sparsity_m=4,
        dense_recent_tokens=0,
        p_qdq="nvfp4",
        v_qdq="nvfp4",
        rel_logits=rel_logits,
        window_left=window_left,
    )
    assert torch.isfinite(transformed).all()

    cache = torch.randn(2, 16, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    before = cache.clone()
    slots = torch.tensor([0, 17, -1], device=device, dtype=torch.int64)
    fake_quant_k_onwrite(cache, slots)
    assert not torch.equal(cache[0, 0], before[0, 0])
    assert not torch.equal(cache[1, 1], before[1, 1])
    assert torch.equal(cache[0, 1], before[0, 1])
    assert torch.isfinite(cache).all()
    print("INKLING_ATTENTION_SANITY_OK", vllm.__version__, torch.cuda.get_device_name())


if __name__ == "__main__":
    main()
