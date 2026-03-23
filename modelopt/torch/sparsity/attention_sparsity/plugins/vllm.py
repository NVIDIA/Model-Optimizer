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

"""ModelOpt sparse attention backend for vLLM.

Registers a custom vLLM attention backend that uses the ModelOpt Triton kernel
with paged KV cache support. Integration approach:

- No module replacement — the Attention module stays intact with all its state
- Only ``impl`` is swapped from FlashAttentionImpl to ModelOptSparseAttentionImpl
- KV cache update is handled by vLLM (inherited ``do_kv_cache_update``)
- Only ``forward()`` is overridden to call our Triton kernel for prefill
- Decode falls back to FlashAttention (sparse attention has no benefit for single-token queries)
"""

import torch

from modelopt.torch.kernels.triton_fa import attention as triton_attention
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)

# Sparse config is set by the worker before model loading
_sparse_config: dict = {}


def set_sparse_config(config: dict):
    """Set the sparse attention config (called by the worker)."""
    global _sparse_config
    _sparse_config = config


class ModelOptSparseAttentionImpl(FlashAttentionImpl):
    """Attention implementation that uses the ModelOpt Triton kernel.

    Inherits from FlashAttentionImpl to reuse:
    - __init__ (all configuration)
    - do_kv_cache_update (KV cache writing)
    Only overrides forward() to replace the attention computation for prefill.
    """

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with ModelOpt Triton sparse attention kernel."""
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run
            return output.fill_(0)

        # Decode: fall back to FlashAttention (no benefit from sparse attention)
        if attn_metadata.max_query_len <= 1:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Unpack paged KV cache: [2, num_blocks, page_size, num_kv_heads, head_dim]
        key_cache, value_cache = kv_cache.unbind(0)
        page_size = key_cache.shape[1]

        # Build sparse kwargs from global config
        sparse_kw = {}
        sparse_cfg = _sparse_config.get("sparse_cfg", {})
        if isinstance(sparse_cfg, str):
            sparse_cfg = {}
        for pattern, layer_cfg in sparse_cfg.items():
            if pattern in ("default", "calibration"):
                continue
            if isinstance(layer_cfg, dict) and layer_cfg.get("enable", True):
                sparsity_n = layer_cfg.get("sparsity_n", 0)
                if sparsity_n > 0:
                    sparse_kw["sparsity_n"] = sparsity_n
                    sparse_kw["sparsity_m"] = layer_cfg.get("sparsity_m", 4)
                    sparse_kw["num_sink_tokens"] = layer_cfg.get("num_sink_tokens", 0)
                    sparse_kw["dense_window_size"] = layer_cfg.get("dense_window_size", 1)
                threshold = layer_cfg.get("skip_softmax_threshold")
                if threshold:
                    sparse_kw["skip_softmax_threshold"] = threshold
                break

        # Prepare metadata for our kernel
        q = query[:num_actual_tokens].contiguous()
        cu_seqlens_q = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        batch = seq_lens.shape[0]

        b_start_loc = cu_seqlens_q[:batch]
        b_seq_len = cu_seqlens_q[1 : batch + 1] - cu_seqlens_q[:batch]
        max_query_len = attn_metadata.max_query_len

        # Dummy K/V for paged mode — must have correct num_kv_heads for GQA ratio
        k_dummy = torch.empty(0, self.num_kv_heads, self.head_size, device=q.device, dtype=q.dtype)
        v_dummy = k_dummy

        # Call ModelOpt Triton kernel with paged KV
        triton_out = triton_attention(
            q,
            k=k_dummy,
            v=v_dummy,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            max_input_len=max_query_len,
            is_causal=True,
            softmax_scale=self.scale,
            b_start_loc_k=None,
            b_seq_len_k=seq_lens,
            max_input_len_k=attn_metadata.max_seq_len,
            k_cache=key_cache,
            v_cache=value_cache,
            block_table=attn_metadata.block_table,
            page_size=page_size,
            **sparse_kw,
        )

        output[:num_actual_tokens] = triton_out
        return output


class ModelOptSparseAttentionBackend(FlashAttentionBackend):
    """Attention backend that uses ModelOpt's sparse Triton kernel.

    Inherits everything from FlashAttentionBackend except get_impl_cls and get_name.
    """

    @staticmethod
    def get_name() -> str:
        """Return backend name."""
        return "MODELOPT_SPARSE"

    @staticmethod
    def get_impl_cls() -> type:
        """Return the attention implementation class."""
        return ModelOptSparseAttentionImpl
